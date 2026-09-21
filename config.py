import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_client():
    """返回初始化好的大模型客户端"""
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )