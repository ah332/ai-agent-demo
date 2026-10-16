import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

def get_client():
    """返回初始化好的大模型客户端"""
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key or key == "your_dashscope_api_key_here":
        raise ValueError("请在项目根目录的 .env 中填写 DASHSCOPE_API_KEY。不要把密钥发到聊天里。")
    return OpenAI(
        api_key=key,
        base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        timeout=45.0,
        max_retries=1,
    )
