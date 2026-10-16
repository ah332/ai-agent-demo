import unittest
import httpx
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from openai import (APIError, APIConnectionError, APITimeoutError,
                    AuthenticationError, PermissionDeniedError,
                    RateLimitError, BadRequestError)
from paper_api import app, operation_error

class ErrorTests(unittest.TestCase):
    def test_timeout(self):
        request = httpx.Request("POST","https://example.invalid")
        error = APITimeoutError(request=request)

        result = operation_error(error)

        self.assertEqual(result.status_code,504)
        self.assertIn("超时",result.detail)

    def check_mapping(self, error, status, message):
        result = operation_error(error)
        self.assertEqual(result.status_code, status)
        self.assertIn(message, result.detail)
        self.assertNotIn('PRIVATE_REQUEST_MARKER', result.detail)

    def status_error(self, error_class, status):
        request = httpx.Request('POST', 'https://example.invalid')
        response = httpx.Response(status, request=request)
        return error_class('PRIVATE_REQUEST_MARKER', response=response, body=None)

    def test_connection(self):
        error = APIConnectionError(message='PRIVATE_REQUEST_MARKER',
                                   request=httpx.Request('POST', 'https://example.invalid'))
        self.check_mapping(error, 502, '无法连接')

    def test_authentication(self):
        self.check_mapping(self.status_error(AuthenticationError, 401), 502, '认证失败')

    def test_permission(self):
        self.check_mapping(self.status_error(PermissionDeniedError, 403), 502, '访问被拒绝')

    def test_rate_limit(self):
        self.check_mapping(self.status_error(RateLimitError, 429), 502, '请求受限')

    def test_upstream_bad_request(self):
        self.check_mapping(self.status_error(BadRequestError, 400), 502, '请求参数')

    def test_other_api_error(self):
        error = APIError('PRIVATE_REQUEST_MARKER',
                         request=httpx.Request('POST', 'https://example.invalid'), body=None)
        self.check_mapping(error, 502, '接口调用失败')

    def test_invalid_input(self):
        self.check_mapping(ValueError('输入无效'), 400, '输入无效')

    def test_unexpected_error_is_logged_without_exposing_details(self):
        with self.assertLogs('paper_api', level='ERROR') as logs:
            self.check_mapping(RuntimeError('PRIVATE_REQUEST_MARKER'), 500, '本地处理失败')
        self.assertIn('RuntimeError', logs.output[0])
        self.assertNotIn('PRIVATE_REQUEST_MARKER', logs.output[0])

    def test_ask_returns_timeout_to_browser(self):
        paper_id = 'a' * 64
        store = SimpleNamespace(papers=lambda: [{'paper_id': paper_id}])
        error = APITimeoutError(request=httpx.Request('POST', 'https://example.invalid'))
        with patch('paper_api.get_store', return_value=store), \
             patch('paper_api.PaperAgent') as agent:
            agent.return_value.ask.side_effect = error
            with TestClient(app) as client:
                response = client.post('/ask', json={'question': '方法？', 'paper_id': paper_id})
        self.assertEqual(response.status_code, 504)
        self.assertIn('超时', response.json()['detail'])
