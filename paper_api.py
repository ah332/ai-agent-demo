"""启动：python -m uvicorn paper_api:app --host 127.0.0.1 --port 8000"""
from functools import lru_cache
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from openai import (APIError, APITimeoutError, APIConnectionError,
                    AuthenticationError, PermissionDeniedError,
                    RateLimitError, BadRequestError)
from pydantic import BaseModel, Field

from paper_store import MAX_PDF_BYTES, PaperStore
from paper_agent import PaperAgent

app = FastAPI(title="ResearchPilot：论文阅读助手", version="0.2.0")
logger = logging.getLogger(__name__)


@lru_cache
def get_store():
    return PaperStore()


def operation_error(exc):
    if isinstance(exc, APITimeoutError):
        return HTTPException(
            504,
            "千问请求超时，请稍后重试。"
        )
    if isinstance(exc, APIConnectionError):
        return HTTPException(
            502,
            "无法连接千问服务，请检查网络和服务地址。"
        )
    if isinstance(exc, AuthenticationError):
        return HTTPException(
            502,
            "千问认证失败，请检查服务端密钥配置。"
        )
    if isinstance(exc, PermissionDeniedError):
        return HTTPException(
            502,
            "千问访问被拒绝，请检查模型权限和服务配置。"
        )
    if isinstance(exc, RateLimitError):
        # 429 可能涉及频率、额度或其他限制，不能只据状态码断言余额不足。
        return HTTPException(502, "千问请求受限，请检查调用频率、账户额度和服务限制。")
    if isinstance(exc, BadRequestError):
        return HTTPException(502, "千问请求参数被拒绝，请检查模型名称和接口配置。")
    if isinstance(exc, APIError):
        # 不把 SDK 原始异常（可能含请求信息）传给浏览器。
        return HTTPException(502, "千问接口调用失败，请检查密钥、地域、模型权限、余额或网络。")
    if isinstance(exc, ValueError):
        return HTTPException(400, str(exc))
    logger.error("Paper operation failed: %s", type(exc).__name__)
    return HTTPException(500, "本地处理失败，请查看终端并记录错误类型。")


@app.get("/")
def home():
    return FileResponse(Path(__file__).resolve().parent / "paper_ui.html")


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0"}


@app.get("/papers")
def list_papers():
    return {"papers": get_store().papers()}


@app.post("/papers")
def upload_paper(file: UploadFile):
    try:
        data = file.file.read(MAX_PDF_BYTES + 1)
        return get_store().ingest(data, file.filename or "paper.pdf")
    except Exception as exc:
        raise operation_error(exc) from exc
    finally:
        file.file.close()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    paper_id: str = Field(pattern=r"^[0-9a-f]{64}$")


@app.post("/ask")
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(400, "问题不能为空。")
    store = get_store()
    if request.paper_id not in {paper["paper_id"] for paper in store.papers()}:
        raise HTTPException(404, "这篇论文尚未导入，请先上传 PDF。")
    try:
        return PaperAgent(store).ask(request.question, request.paper_id)
    except Exception as exc:
        raise operation_error(exc) from exc
