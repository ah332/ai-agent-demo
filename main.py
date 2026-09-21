from agent import ChatAgent

def main():
    agent = ChatAgent(system_prompt="你是一个专业的AI助手。")
    print("AI助手已启动，输入 'exit' 退出对话。")
    
    while True:
        user_input = input("\n你: ")
        if user_input.lower() == 'exit':
            print("再见！")
            break
            
        # 调用Agent的chat方法
        ai_reply = agent.chat(user_input)
        
        # 注意：这里的 f 是英文字母 f，引号是英文引号
        print(f"AI: {ai_reply}")

if __name__ == "__main__":
    main()