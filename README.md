# 🤖 My First AI Agent (ResearchPilot V0.1)

一个基于大模型 API 与原生 Python 实现的智能体（Agent）开发练手项目。

## 📖 项目简介
本项目是我从零开始学习 AI Agent 开发的第一周成果。项目没有依赖 LangChain 等高级框架，而是使用原生 `openai` SDK + Python，从底层手写实现了 Agent 的核心运行机制，深入理解了 LLM 应用的工程化流程。

## ✨ 已实现功能 (Day 1 - Day 7)
- [1] **多轮对话记忆**：通过维护 `messages` 列表，实现上下文记忆。
- [2] **工具调用 (Tool Use)**：大模型自主决定调用外部工具（天气查询、数学计算器）。
- [3] **Agent Loop 循环**：支持多步思考和并行工具调用，加入 `max_steps=5` 防止死循环。
- [4] **结构化参数校验**：引入 Pydantic 对工具参数进行校验，实现“自我纠错”（Self-Correction）。
- [5] **工程化改造**：模块化拆分（config/agent/tools/schemas/main），环境变量分离，Git 版本管理。

## 🏗️ 项目结构
```text
ai-agent-demo/
├── .env                # 存放API Key（已加入.gitignore）
├── .gitignore          # Git忽略文件配置
├── config.py           # 初始化大模型客户端
├── tools.py            # 工具函数定义与Schema描述
├── schemas.py          # Pydantic 数据结构校验
├── agent.py            # ChatAgent 核心类（包含 Agent Loop）
├── main.py             # 命令行交互入口
└── README.md           # 项目说明