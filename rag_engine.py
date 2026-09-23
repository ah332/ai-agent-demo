import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from config import get_client # 复用你已有的连接

load_dotenv()

#1.获取大模型客户端（用于获取Embedding）
client = get_client()

# 2. 初始化本地Chroma数据库（数据会自动存在当前目录下的chroma_db文件夹）
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 3. 创建一个集合（类似数据库里的表）
collection = chroma_client.get_or_create_collection(name="key_knowledge")

def get_embedding(text):
    """调用API将文本转为向量"""
    #注意：通义千问的embedding模型是text-embedding-v3
    response = client.embeddings.create(
        model="text-embedding-v3",
        input=text,
        dimensions=1024 #指定向量维度
    )
    return response.data[0].embedding

def load_document(file_path):
    """读取文档并切片"""
    with open(file_path,"r",encoding="utf-8") as f:
        content = f.read()

    #简单切片：每200个字一块，重叠50个字（防止句子被切断丢失上下文）
    chunks = []
    chunk_size=200
    overlap=50
    for i in range(0,len(content),chunk_size - overlap):
        chunk = content[i:i+chunk_size]
        if chunk.strip():
            chunks.append(chunk)

    return chunks

def add_to_knowledge_base(file_path):
    """读取文档，向量化后存入Chroma"""
    chunks = load_document(file_path)
    print(f"[系统日志]:文档切分为{len(chunks)}个片段")

    for i,chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        #存入数据库，指定id、文档内容和向量
        collection.add(
            ids=[f"id_{i}"],
            documents=[chunk],
            embeddings=[embedding]
          )
    print("[系统日志]:知识库构建完成！")

def search_knowledge_base(query):
    """检索知识库（加入相似度阈值过滤）"""
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2, #返回最相关的2个片段
        include=["documents","distances"]#明确要求返回距离
     )
    #提取检索到文本
    retrieved_docs = results["documents"][0]
    distances = results["distances"][0]


    # 设定一个距离阈值（Chroma默认使用L2距离，越小越相似。具体阈值需要根据实际测试调整）
    # 这里假设距离大于 1.5 就认为不相关
    valid_docs = []
    for doc,dist in zip(retrieved_docs,distances):
        if dist<1.5: #阈值可以根据实际情况动态调整
            valid_docs.append(doc)
        else:
            print(f"[系统日志]:过滤掉低相关片段，距离={dist:.2f}")

    if not valid_docs:
        return "未在知识库中找到相关信息。" 


    return "\n".join(valid_docs)



 #测试一下
if __name__=="__main__":
    #第一次运行时，建库
    add_to_knowledge_base("knowledge.txt")

    #模拟提问
    question = "董ah的毕业论文研究什么？"
    context = search_knowledge_base(question)
    print(f"\n[检索结果]：\n{context}")


