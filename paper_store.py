"""PDF -> 按页切片 -> 千问 Embedding -> Chroma；独立于旧的个人知识库。"""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import RLock
import os
import re

import chromadb
from pypdf import PdfReader

from config import get_client
from paper_text import merge_overlapping

ROOT = Path(__file__).resolve().parent
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PAGES = 200
MAX_CONTEXT_CHARS = 2160
MIN_CONTEXT_OVERLAP = 32


def parse_pdf(data: bytes, filename: str) -> tuple[str, list[dict], list[int]]:
    """每个片段只来自一页，page 表示 PDF 阅读器的第几页，从 1 开始。"""
    if not data or len(data) > MAX_PDF_BYTES:
        raise ValueError("请上传非空、大小不超过 20 MB 的 PDF。")
    filename = Path(filename.replace("\\", "/")).name
    if not filename.lower().endswith(".pdf"):
        raise ValueError("第一版只接受 PDF 文件。")
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("请先解密 PDF，再导入。")
        if len(reader.pages) > MAX_PAGES:
            raise ValueError("第一版每篇论文最多 200 页，请先截取需要的部分。")
        paper_id = sha256(data).hexdigest()
        chunks, empty_pages = [], []
        total_chars = 0
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            total_chars += len(text)
            if total_chars > 500_000:
                raise ValueError("提取文本超过 50 万字符，请先拆分文件。")
            if not text:
                empty_pages.append(page_number)
                continue
            for start in range(0, len(text), 680):
                chunk = text[start:start + 800].strip()
                if chunk:
                    chunks.append({
                        "id": f"{paper_id}:p{page_number}:c{start // 680}",
                        "text": chunk,
                        "metadata": {
                            "paper_id": paper_id, "filename": filename,
                            "page": page_number, "pages": len(reader.pages),
                        },
                    })
        if not chunks:
            raise ValueError("没有提取到文字。扫描版 PDF 需要先 OCR；第一版未实现 OCR。")
        if len(chunks) > 1000:
            raise ValueError("论文片段超过 1000 个，请先拆分文件。")
        return paper_id, chunks, empty_pages
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("PDF 读取失败，请确认文件完整且能在阅读器中打开。") from exc


class PaperStore:
    def __init__(self, path=None, embed=None):
        self.db = chromadb.PersistentClient(
            path=str(path or os.getenv("PAPER_DB_DIR") or ROOT / "paper_chroma_db")
        )
        # 模型改变会使用新集合，避免把不同模型的向量混在一起。
        self.embedding_model = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")
        collection_key = sha256(self.embedding_model.encode()).hexdigest()[:12]
        self.collection = self.db.get_or_create_collection(
            name=f"papers_v1_{collection_key}",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        self._embed_override = embed  # 测试可注入假向量，不消耗 API。
        self._client = None
        self._lock = RLock()

    def embed(self, texts):
        if self._embed_override:
            return self._embed_override(texts)
        if self._client is None:
            self._client = get_client()
        response = self._client.embeddings.create(
            model=self.embedding_model, input=texts, dimensions=1024,
        )
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def ingest(self, data: bytes, filename: str) -> dict:
        paper_id, chunks, empty_pages = parse_pdf(data, filename)
        with self._lock:
            existing = self.collection.get(where={"paper_id": paper_id}, include=["metadatas"])
            if existing["ids"]:
                metadata = existing["metadatas"][0]
                return {"paper_id": paper_id, "filename": metadata["filename"],
                        "pages": metadata["pages"], "chunks": len(existing["ids"]),
                        "duplicate": True, "empty_pages": empty_pages}
            vectors = []
            # 所有向量获取成功后才写入，避免 API 中途失败留下半篇论文。
            for start in range(0, len(chunks), 10):
                batch = chunks[start:start + 10]
                result = self.embed([item["text"] for item in batch])
                if len(result) != len(batch):
                    raise ValueError("Embedding 返回数量不匹配，论文尚未入库。")
                vectors.extend(result)
            self.collection.add(
                ids=[item["id"] for item in chunks],
                documents=[item["text"] for item in chunks],
                metadatas=[item["metadata"] for item in chunks],
                embeddings=vectors,
            )
        metadata = chunks[0]["metadata"]
        return {"paper_id": paper_id, "filename": metadata["filename"],
                "pages": metadata["pages"], "chunks": len(chunks),
                "duplicate": False, "empty_pages": empty_pages}

    def papers(self) -> list[dict]:
        records = self.collection.get(include=["metadatas"])
        papers = {}
        for metadata in records["metadatas"] or []:
            paper = papers.setdefault(metadata["paper_id"], {
                "paper_id": metadata["paper_id"], "filename": metadata["filename"],
                "pages": metadata["pages"], "chunks": 0,
            })
            paper["chunks"] += 1
        return sorted(papers.values(), key=lambda paper: paper["filename"])

    def search(self, query: str, paper_id: str, top_k: int = 4) -> list[dict]:
        query = query.strip()
        if not query or len(query) > 2000:
            raise ValueError("检索语句须为 1～2000 个字符。")
        records = self.collection.get(where={"paper_id": paper_id}, include=[])
        if not records["ids"]:
            return []
        result = self.collection.query(
            query_embeddings=self.embed([query]),
            n_results=min(top_k, len(records["ids"])),
            where={"paper_id": paper_id},
            include=["documents", "metadatas", "distances"],
        )
        hits = [
            {"chunk_id": chunk_id, "text": text, **metadata,
             "distance": float(distance)}
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0], result["documents"][0],
                result["metadatas"][0], result["distances"][0],
            )
        ]
        return self.expand_context(hits)

    def expand_context(self, hits: list[dict]) -> list[dict]:
        """读取同论文、同页的前后邻居；只扩大证据，不更新数据库或向量。"""
        plans, neighbor_ids = {}, set()
        for hit in hits:
            match = re.fullmatch(r"([0-9a-f]{64}):p([1-9]\d*):c(\d+)", hit["chunk_id"])
            if not match:
                continue
            paper_id, page, index = match.group(1), int(match.group(2)), int(match.group(3))
            if paper_id != hit["paper_id"] or page != hit["page"] or len(hit["text"]) > 800:
                continue
            prefix = f"{paper_id}:p{page}:c"
            previous = f"{prefix}{index - 1}" if index > 0 else None
            following = f"{prefix}{index + 1}"
            plans[hit["chunk_id"]] = (previous, following)
            neighbor_ids.update(item for item in (previous, following) if item is not None)

        if not neighbor_ids:
            return hits
        records = self.collection.get(
            ids=sorted(neighbor_ids), include=["documents", "metadatas"],
        )
        neighbors = {
            chunk_id: (text, metadata)
            for chunk_id, text, metadata in zip(
                records["ids"], records["documents"], records["metadatas"],
            )
        }

        def join(left, right):
            # 相邻窗口原始重叠最多 120 字符；限制比较范围，避免重复段落误匹配。
            tail, head = left[-120:], right[:120]
            overlap = len(tail) + len(head) - len(merge_overlapping(tail, head))
            # 太短的匹配不足以可靠连接；不猜测缺失文字或直接拼接。
            if overlap < MIN_CONTEXT_OVERLAP:
                return None
            combined = left + right[overlap:]
            return combined if len(combined) <= MAX_CONTEXT_CHARS else None

        expanded = []
        for hit in hits:
            text, used_ids = hit["text"], [hit["chunk_id"]]
            for position, neighbor_id in enumerate(plans.get(hit["chunk_id"], (None, None))):
                neighbor = neighbors.get(neighbor_id)
                if neighbor is None:
                    continue
                neighbor_text, metadata = neighbor
                if (metadata["paper_id"] != hit["paper_id"] or metadata["page"] != hit["page"]
                        or metadata["filename"] != hit["filename"] or len(neighbor_text) > 800):
                    continue
                combined = join(neighbor_text, text) if position == 0 else join(text, neighbor_text)
                if combined is not None:
                    text = combined
                    used_ids.insert(0, neighbor_id) if position == 0 else used_ids.append(neighbor_id)
            expanded.append({**hit, "text": text, "context_chunk_ids": used_ids})
        return expanded
