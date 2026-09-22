from rag_engine import search_knowledge_base

def get_weather(city:str) -> str:
    """模拟获取天气"""
    # 实际项目中，这里会调用真实的天气API
    if city == "北京":
        return "晴天，25度"
    elif city == "上海":
        return "多云转晴，30度"
    else:
        return "未知城市，天气未知"

def calculate(expression:str) -> str:
    """模拟计算器"""
    try:
        # 注意：实际生产环境不要用 eval，这里仅为教学模拟
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误:{e}"

tools_schema = [
    {
        "type":"function",
        "function":{
            "name":"get_weather",
            "description":"获取指定城市的当前天气",
            "parameters":{
                "type":"object",
                "properties":{
                    "city":{"type":"string","description":"城市名称，例如：北京、上海"}
                },
                "required":["city"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"calculate",
            "description":"计算数学表达式",
            "parameters":{
                "type":"object",
                "properties":{
                    "expression":{"type":"string","description":"数学表达式，例如：1+1、100*5"}
                },
                "required":["expression"]
            }
        }
    },

 {
        "type":"function",
        "function":{
            "name":"search_knowledge_base",
            "description":"当需要回答关于董佶雷的个人信息、毕业论文、项目经历等问题时，使用此工具检索知识库。",
            "parameters":{
                "type":"object",
                "properties": {
                    "query":{"type":"string","description":"要检索的查询语句"}
                },
                "required":["query"]
            }
        }
    }
]