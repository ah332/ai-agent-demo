from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from agent import ChatAgent

#1.创建FastAPI实例
app=FastAPI(title="AI Agent Service")

#2.用内存字典维护多用户会话（未来的大厂实习会用到Redis)
# 结构：{"session_id_1": ChatAgent实例1, "session_id_2": ChatAgent实例2}
sessions={}

#3.定义请求和响应的数据结构（Pydantic自动校验）
class ChatRequest(BaseModel):
    session_id:str
    message:str

class ChatResponse(BaseModel):
    reply:str

#4.健康检查接口（用于监控服务是否存活）
@app.get("/health")
def health_check():
    return{"status":"ok"}

#5.核心对话接口
@app.post("/chat",response_model=ChatResponse)
def chat_endpoint(request:ChatRequest):
    #如果该用户没有会话，就初始化一个Agent
    if request.session_id not in sessions:
        sessions[request.session_id]=ChatAgent(
            system_prompt="你是一个专业的AI助手。请严格基于工具返回的检索结果回答问题。如果检索结果中不包含答案，请直接回答'知识库中未找到相关信息'，绝对不要自己编造答案。"
        )

    agent = sessions[request.session_id]

    try:
        #调用Agent的核心逻辑
        reply = agent.chat(request.message)
        return ChatResponse(reply=reply)
    except Exception as e:
        #如果发生致命错误，返回HTTP 500
        raise HTTPException(status_code=500,detail=str(e))