"""真 PDF + 真 Chroma + 模拟千问：验证流程，不宣称回答准确率。"""
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from paper_store import PaperStore, parse_pdf
from paper_agent import PaperAgent, NO_EVIDENCE
from paper_api import app, get_store


def make_pdf(*texts):
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=595, height=842)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                 NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({
            NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        safe = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        stream.set_data(f'BT /F1 12 Tf 40 780 Td ({safe}) Tj ET'.encode('ascii'))
        page[NameObject('/Contents')] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def completion(content=None, arguments=None, name='search_paper'):
    message = {'role': 'assistant', 'content': content}
    if arguments is not None:
        message['tool_calls'] = [{'id': 'call_test', 'type': 'function',
                                 'function': {'name': name, 'arguments': arguments}}]
    return ChatCompletion.model_validate({'id': 'test', 'created': 0, 'model': 'fake',
        'object': 'chat.completion', 'choices': [{'index': 0, 'message': message,
        'finish_reason': 'tool_calls' if arguments is not None else 'stop'}]})


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        # 深拷贝，以便核对每一轮传给模型的真实消息。
        self.calls.append(json.loads(json.dumps(kwargs)))
        return next(self.responses)


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.embed_calls = []
        def embed(texts):
            self.embed_calls.append(texts)
            return [[1.0, 0.0, 0.0] for _ in texts]
        self.store = PaperStore(path=Path(self.temp.name) / 'db', embed=embed)
        self.pdf = make_pdf('PCB defect detection introduction.',
                            'Our method uses YOLO. Printed page number: 99.')

    def tearDown(self):
        app.dependency_overrides.clear()
        # Windows cannot delete database files while Chroma holds them open.
        self.store.db.close()
        self.temp.cleanup()

    def stored_hit(self, paper_id, page, index):
        chunk_id = f'{paper_id}:p{page}:c{index}'
        records = self.store.collection.get(ids=[chunk_id], include=['documents', 'metadatas'])
        return {'chunk_id': chunk_id, 'text': records['documents'][0],
                **records['metadatas'][0], 'distance': 0.2}

    def test_context_three_neighbors_preserve_repeated_text_and_scope(self):
        text = '0123456789' * 210
        paper = self.store.ingest(make_pdf(text, 'OTHER_PAGE_MARKER'), 'context.pdf')
        self.store.ingest(make_pdf('OTHER_PAPER_MARKER'), 'other.pdf')
        hit = self.stored_hit(paper['paper_id'], 1, 1)
        calls_before, count_before = len(self.embed_calls), self.store.collection.count()
        expanded = self.store.expand_context([hit])[0]
        # 重复文字不能使算法误把相邻的整段原文当成重叠而删除。
        self.assertEqual(expanded['text'], text)
        self.assertEqual(expanded['chunk_id'], hit['chunk_id'])
        self.assertEqual(expanded['distance'], hit['distance'])
        self.assertEqual(expanded['page'], 1)
        self.assertEqual(expanded['context_chunk_ids'],
                         [f"{paper['paper_id']}:p1:c{i}" for i in range(3)])
        self.assertNotIn('OTHER_PAGE_MARKER', expanded['text'])
        self.assertNotIn('OTHER_PAPER_MARKER', expanded['text'])
        self.assertEqual(len(self.embed_calls), calls_before)
        self.assertEqual(self.store.collection.count(), count_before)

    def test_expanded_context_reaches_model_and_displayed_source(self):
        phrase = 'while Faster R-CNN [7] (mAP 75.4%) and SSD [8] (mAP 66.5%)'
        text = 'A' * 650 + phrase + 'B' * 220
        paper = self.store.ingest(make_pdf(text), 'metric.pdf')
        hit = self.stored_hit(paper['paper_id'], 1, 1)
        self.assertNotIn('Faster R-CNN', hit['text'])
        query_result = {'ids': [[hit['chunk_id']]], 'documents': [[hit['text']]],
                        'metadatas': [[{key: hit[key] for key in ('paper_id', 'filename', 'page', 'pages')}]],
                        'distances': [[hit['distance']]]}
        client = FakeClient([completion(arguments='{"query":"75.4%"}'),
                             completion('75.4% 对应 Faster R-CNN。[S1]')])
        with patch.object(self.store.collection, 'query', return_value=query_result):
            result = PaperAgent(self.store, client).ask('指标属于谁？', paper['paper_id'])
        source = result['sources'][0]
        evidence = json.loads(client.calls[1]['messages'][-1]['content'])['evidence'][0]
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(source['quote'], text)
        self.assertEqual(source['quote'].count('75.4%'), 1)
        self.assertEqual(evidence, source)
        self.assertEqual(source['chunk_ids'], [f"{paper['paper_id']}:p1:c0", hit['chunk_id']])

    def test_context_missing_neighbor_and_unreliable_overlap_preserve_anchor(self):
        paper = self.store.ingest(make_pdf('A' * 900), 'context.pdf')
        hit = self.stored_hit(paper['paper_id'], 1, 1)
        previous_id = f"{paper['paper_id']}:p1:c0"
        self.store.collection.update(ids=[previous_id], documents=['B' * 800], embeddings=[[1.0, 0.0, 0.0]])
        expanded = self.store.expand_context([hit])[0]
        self.assertEqual(expanded['text'], hit['text'])
        self.assertEqual(expanded['context_chunk_ids'], [hit['chunk_id']])
        self.store.collection.delete(ids=[previous_id])
        expanded = self.store.expand_context([hit])[0]
        self.assertEqual(expanded['text'], hit['text'])
        self.assertEqual(expanded['context_chunk_ids'], [hit['chunk_id']])

    def test_context_rejects_neighbor_with_wrong_provenance(self):
        paper = self.store.ingest(make_pdf('A' * 900), 'context.pdf')
        hit = self.stored_hit(paper['paper_id'], 1, 1)
        previous_id = f"{paper['paper_id']}:p1:c0"
        for metadata in ({'paper_id': 'f' * 64, 'page': 1, 'filename': 'context.pdf', 'pages': 1},
                         {'paper_id': paper['paper_id'], 'page': 2, 'filename': 'context.pdf', 'pages': 2}):
            self.store.collection.update(ids=[previous_id], metadatas=[metadata])
            expanded = self.store.expand_context([hit])[0]
            self.assertEqual(expanded['text'], hit['text'])
            self.assertEqual(expanded['context_chunk_ids'], [hit['chunk_id']])

    def test_pdf_pages_and_source(self):
        _, chunks, _ = parse_pdf(self.pdf, 'paper.pdf')
        self.assertEqual([c['metadata']['page'] for c in chunks], [1, 2])
        self.assertIn('YOLO', chunks[1]['text'])
        self.assertEqual(chunks[1]['metadata']['filename'], 'paper.pdf')

    def test_multiple_papers_and_duplicate_no_embedding(self):
        first = self.store.ingest(self.pdf, 'a.pdf')
        calls_before = len(self.embed_calls)
        duplicate = self.store.ingest(self.pdf, 'renamed.pdf')
        self.assertTrue(duplicate['duplicate'])
        self.assertEqual(duplicate['filename'], 'a.pdf')
        self.assertEqual(len(self.embed_calls), calls_before)
        second = self.store.ingest(make_pdf('Different paper.'), 'b.pdf')
        self.assertNotEqual(first['paper_id'], second['paper_id'])
        self.assertEqual(self.store.collection.count(), 3)
        hits = self.store.search('method', first['paper_id'])
        self.assertTrue(all(hit['filename'] == 'a.pdf' for hit in hits))
        reloaded = PaperStore(path=Path(self.temp.name) / 'db', embed=lambda texts: [[1,0,0]]*len(texts))
        try:
            self.assertEqual(len(reloaded.papers()), 2)
        finally:
            reloaded.db.close()

    def test_bad_pdf_and_scanned_pdf(self):
        with self.assertRaises(ValueError):
            parse_pdf(b'not a PDF', 'bad.pdf')
        with self.assertRaisesRegex(ValueError, 'OCR'):
            parse_pdf(make_pdf(''), 'scan.pdf')

    def test_embedding_failure_does_not_write_partial_paper(self):
        self.store._embed_override = lambda texts: []
        with self.assertRaisesRegex(ValueError, '数量'):
            self.store.ingest(self.pdf, 'a.pdf')
        self.assertEqual(self.store.collection.count(), 0)

    def test_loop_backfills_evidence_and_returns_real_source(self):
        paper = self.store.ingest(self.pdf, 'a.pdf')
        client = FakeClient([completion(arguments='{"query":"method"}'),
                             completion('论文讨论缺陷检测。[S1]')])
        result = PaperAgent(self.store, client).ask('研究什么？', paper['paper_id'])
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['sources'][0]['filename'], 'a.pdf')
        self.assertIn(result['sources'][0]['page'], [1, 2])
        self.assertEqual(client.calls[1]['messages'][-1]['role'], 'tool')
        self.assertIn('evidence', client.calls[1]['messages'][-1]['content'])

    def test_bad_tool_json_is_backfilled_then_recovered(self):
        paper = self.store.ingest(self.pdf, 'a.pdf')
        client = FakeClient([completion(arguments='not json'),
                             completion(arguments='{"query":"method"}'),
                             completion('使用检测方法。[S1]')])
        result = PaperAgent(self.store, client).ask('方法？', paper['paper_id'])
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['trace'][0]['status'], 'argument_error')

    def test_unknown_and_missing_citations_are_rejected(self):
        paper = self.store.ingest(self.pdf, 'a.pdf')
        for answer in ['有方法。[S99]', '有方法。[ S99 ]', '有方法。']:
            client = FakeClient([completion(arguments='{"query":"method"}'), completion(answer)])
            result = PaperAgent(self.store, client).ask('方法？', paper['paper_id'])
            self.assertEqual(result['status'], 'citation_error')
            self.assertEqual(result['sources'], [])

    def test_whitespace_citations_are_normalized_without_inventing_sources(self):
        paper = self.store.ingest(self.pdf, 'a.pdf')
        client = FakeClient([completion(arguments='{"query":"method"}'),
                             completion('论文讨论缺陷检测。[ S1 ]')])
        result = PaperAgent(self.store, client).ask('研究什么？', paper['paper_id'])
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['reply'], '论文讨论缺陷检测。[S1]')
        self.assertEqual([source['id'] for source in result['sources']], ['S1'])

    def test_no_evidence_answer(self):
        paper = self.store.ingest(self.pdf, 'a.pdf')
        client = FakeClient([completion(arguments='{"query":"unknown"}'), completion(NO_EVIDENCE)])
        result = PaperAgent(self.store, client).ask('未记载信息？', paper['paper_id'])
        self.assertEqual(result['status'], 'no_evidence')

    def test_iteration_budget(self):
        paper = self.store.ingest(self.pdf, 'a.pdf')
        client = FakeClient([completion(arguments='{"query":"method"}') for _ in range(5)])
        result = PaperAgent(self.store, client).ask('方法？', paper['paper_id'])
        self.assertEqual(result['status'], 'step_limit')
        self.assertEqual(len(client.calls), 5)
        self.assertEqual(client.calls[-1]['tool_choice'], 'none')

    def test_api_upload_ask_validation_and_home(self):
        with patch('paper_api.get_store', return_value=self.store):
            client = TestClient(app)
            self.assertEqual(client.get('/health').status_code, 200)
            self.assertEqual(client.get('/').status_code, 200)
            response = client.post('/papers', files={'file': ('a.pdf', self.pdf, 'application/pdf')})
            self.assertEqual(response.status_code, 200, response.text)
            paper = response.json()
            self.assertEqual(len(client.get('/papers').json()['papers']), 1)
            with patch('paper_api.PaperAgent') as agent:
                agent.return_value.ask.return_value = {'reply': '回答。[S1]', 'sources': [], 'trace': [], 'status': 'ok'}
                self.assertEqual(client.post('/ask', json={'question': '方法？', 'paper_id': paper['paper_id']}).status_code, 200)
            self.assertEqual(client.post('/ask', json={'question': '方法？', 'paper_id': 'f'*64}).status_code, 404)
            self.assertEqual(client.post('/ask', json={'question': '', 'paper_id': paper['paper_id']}).status_code, 422)
            self.assertEqual(client.post('/papers', files={'file': ('bad.pdf', b'bad')}).status_code, 400)


if __name__ == '__main__':
    unittest.main()
