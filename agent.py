from config import get_client
from tools import tools_schema, get_weather, calculate
from schemas import WeatherArgs,CalculateArgs,KnowledgeArgs
from rag_engine import search_knowledge_base
import json


class ChatAgent:
    def __init__(self, system_prompt="你是一个专业的AI助手。"):
        self.client = get_client()
        self.messages = [{"role": "system", "content": system_prompt}]
        print(f"[系统提示]: Agent 初始化完成，角色设定为：{system_prompt}")

    def chat(self, user_input):
        """接收用户输入，返回AI回复，并处理多步工具调用"""
        self.messages.append({"role": "user", "content": user_input})
        
        # 设置最大步数，防止无限循环
        max_steps = 5
        step = 0

        try:
            while step < max_steps:
                step += 1
                print(f"\n--- 第 {step} 步思考 ---")

                # 请求模型
                response = self.client.chat.completions.create(
                    model="qwen-turbo",
                    messages=self.messages,
                    tools=tools_schema,
                    tool_choice="auto"
                )

                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                # 如果模型不需要调用工具，说明思考结束，直接返回回答
                if not tool_calls:
                    ai_reply = response_message.content
                    self.messages.append({"role": "assistant", "content": ai_reply})
                    return ai_reply

                # 如果模型决定调用工具，先把它的请求记录到历史
                self.messages.append(response_message)

                # 遍历模型想调用的所有工具（支持一次调用多个）
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)#把模型返回的工具参数 JSON 字符串，转换成 Python 能直接处理的数据结构
                    
                    print(f"[系统日志]: 模型决定调用工具：{function_name}，参数：{function_args}")

                    # 执行本地函数
                    if function_name == "get_weather":
                        try:
                            # 校验参数：如果格式不对，会触发异常
                            validated_args = WeatherArgs(**function_args)
                            tool_result = get_weather(validated_args.city)
                        except Exception as e:
                            # 把错误信息作为工具结果，让大模型自己去反思
                            tool_result = f"参数格式错误：{e}，请按照正确的格式重新调用工具。"
                    elif function_name == "calculate":
                        try:
                            validated_args = CalculateArgs(**function_args)
                            tool_result = calculate(validated_args.expression)
                        except Exception as e:
                            tool_result = f"参数格式错误：{e},请按照正确的格式重新调用工具。"
                    elif function_name == "search_knowledge_base":
                        try:
                            validated_args = KnowledgeArgs(**function_args) #这里建议自己补充，参照前面的Pydantic模式
                            tool_result =  search_knowledge_base(validated_args.query)
                        except Exception as e:
                            tool_result = f"检索失败：{e}，请按正确的格式重新调用工具"                   
                    else:
                        tool_result = f"未找到工具：{function_name}"
                    
                    print(f"[系统日志]: 工具执行结果：{tool_result}")

                    # 把工具执行结果加入历史消息
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                # 循环会继续，带着工具结果去请求模型进行下一步推理

            # 如果超过最大步数还没结束，强制终止
            return "[警告]: 已达到最大步数限制 (5步)，Agent 停止工作，请检查任务是否过于复杂。"

        except Exception as e:
            self.messages.pop()
            return f"[出错了]: {e}，请检查网络或API余额。"