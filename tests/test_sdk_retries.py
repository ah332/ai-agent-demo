"""真实已安装 SDK + MockTransport：验证请求次数，不访问外网或真实密钥。"""
from contextlib import contextmanager
import os
import unittest
from unittest.mock import patch

import httpx
from openai import (OpenAI, APITimeoutError, APIConnectionError,
                    AuthenticationError, PermissionDeniedError,
                    BadRequestError, RateLimitError, InternalServerError)

from config import get_client


@contextmanager
def simulated_client(outcomes):
    calls = []

    def respond(request):
        outcome = outcomes[min(len(calls), len(outcomes) - 1)]
        calls.append(request)
        if outcome == 'timeout':
            raise httpx.ReadTimeout('simulated timeout', request=request)
        if outcome == 'connection':
            raise httpx.ConnectError('simulated connection failure', request=request)
        if outcome != 200:
            return httpx.Response(outcome, json={'error': {
                'message': 'simulated upstream error', 'type': 'test_error', 'code': 'test'}})
        return httpx.Response(200, json={
            'id': 'mock', 'object': 'chat.completion', 'created': 0, 'model': 'mock',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'mock answer'},
                         'finish_reason': 'stop'}]})

    def make_client(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(
            transport=httpx.MockTransport(respond), trust_env=False))

    # 实际使用项目 get_client() 的 timeout/max_retries；仅替换密钥、地址和传输。
    # 跳过退避等待，保留真实 SDK 的重试判断和次数控制。
    with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'sdk-test-only',
                               'DASHSCOPE_BASE_URL': 'https://example.invalid/v1'}), \
         patch('config.OpenAI', side_effect=make_client), \
         patch('openai._base_client.time.sleep'):
        with get_client() as client:
            yield client, calls


def ask(client):
    return client.chat.completions.create(
        model='mock', messages=[{'role': 'user', 'content': 'test'}])


class SDKRetryTests(unittest.TestCase):
    def test_timeout_then_success_makes_two_attempts(self):
        with simulated_client(['timeout', 200]) as (client, calls):
            self.assertEqual(client.max_retries, 1)
            self.assertEqual(client.timeout, 45.0)
            result = ask(client)
            self.assertEqual(result.choices[0].message.content, 'mock answer')
            self.assertEqual(len(calls), 2)

    def check_failure(self, outcomes, exception_class, attempts):
        with simulated_client(outcomes) as (client, calls):
            with self.assertRaises(exception_class):
                ask(client)
            self.assertEqual(len(calls), attempts)

    def test_persistent_timeout_stops_after_two_attempts(self):
        self.check_failure(['timeout'], APITimeoutError, 2)

    def test_connection_failure_stops_after_two_attempts(self):
        self.check_failure(['connection'], APIConnectionError, 2)

    def test_authentication_failure_is_not_retried(self):
        self.check_failure([401], AuthenticationError, 1)

    def test_permission_failure_is_not_retried(self):
        self.check_failure([403], PermissionDeniedError, 1)

    def test_bad_request_is_not_retried(self):
        self.check_failure([400], BadRequestError, 1)

    def test_rate_limit_stops_after_two_attempts(self):
        self.check_failure([429], RateLimitError, 2)

    def test_server_error_stops_after_two_attempts(self):
        self.check_failure([500], InternalServerError, 2)
