messages = [
    {"role":"system", "content":"根据证据回答"},
    {"role":"user","content":"论文用了什么数据集？"}
]

tool_result = {"role":"tool","content":"检索的到的论文片段"}
messages.append(tool_result)

for item in messages:
    print(item["role"],item["content"])
