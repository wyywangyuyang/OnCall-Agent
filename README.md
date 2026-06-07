# OnCall-Agent

> 企业级智能对话和运维助手，支持 RAG 知识库问答和 AIOps 智能诊断

> 智能0nCall系统通过AI Agent解决团队真实痛点，整合知识库、对话、运维三大核心能力，实现问题自动应答和故障智能排查的一体化服务。致力于降低团队OnCall的人力成本，提升团队效率。前端界面基于 Vibe Coding 快速生成并适配对接。 

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2+-orange.svg)](https://www.langchain.com/)

## ✨ 核心特性

- 🤖 **智能对话** - LangChain 多轮对话 + 流式输出
- 📚 **RAG 问答** - 向量检索增强，支持文档上传、自动建立向量索引、自动更新知识库
- 🔧 **AIOps 诊断** - Plan-Execute-Replan 自动故障诊断和根因分析
- 🌐 **Web 界面** - 现代化 UI，支持多种对话模式：快速问答/流式对话
- 🔌 **MCP 集成** - 日志查询和监控数据工具接入

## 🛠️ 技术栈

- **框架**: FastAPI + LangChain + LangGraph
- **LLM**: 阿里云 DashScope (通义千问)
- **向量库**: Milvus
- **工具协议**: MCP (Model Context Protocol)

## 🚀 快速开始

### 环境要求
- Python 3.10+
- 阿里云 DashScope API Key ([获取地址](https://dashscope.aliyun.com/))

### 访问服务
- **Web 界面**: http://localhost:9900
- **API 文档**: http://localhost:9900/docs

## 📡 API 接口

### 核心接口

| 功能 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 普通对话 | POST | `/api/chat` | 一次性返回 |
| 流式对话 | POST | `/api/chat_stream` | SSE 流式输出 |
| AIOps 诊断 | POST | `/api/aiops` | 自动故障诊断（流式） |
| 文件上传 | POST | `/api/upload` | 上传并索引文档 |
| 健康检查 | GET | `/api/health` | 服务状态检查 |

### 使用示例

```bash
# 普通对话
curl -X POST "http://localhost:9900/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-123","Question":"你好"}'

# 流式对话
curl -X POST "http://localhost:9900/api/chat_stream" \
  -H "Content-Type: application/json" \
  -d '{"Id":"session-123","Question":"你好"}' \
  --no-buffer

# AIOps 诊断
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"session-123"}' \
  --no-buffer
```

## 📁 项目结构

```
OnCall-Agent/
├── app/                                    # 应用核心
│   ├── main.py                             # FastAPI 应用入口
│   ├── config.py                           # 配置管理（环境变量、多路召回、Query 改写等）
│   ├── api/                                # API 路由层
│   │   ├── chat.py                         # 对话接口（RAG 聊天）
│   │   ├── aiops.py                        # AIOps 接口（故障诊断）
│   │   ├── file.py                         # 文件管理（文档上传）
│   │   └── health.py                       # 健康检查（服务状态）
│   ├── services/                           # 业务服务层
│   │   ├── rag_agent_service.py            # RAG Agent（LangGraph 状态图）
│   │   ├── aiops_service.py                # AIOps 服务（计划-执行-重规划）
│   │   ├── vector_store_manager.py         # 向量存储管理器
│   │   ├── vector_embedding_service.py     # 向量 Embedding 服务（DashScope）
│   │   ├── vector_index_service.py         # 向量索引服务 + BM25 FTS5 同步写入
│   │   ├── vector_search_service.py        # 向量检索服务（Milvus）
│   │   ├── bm25_index_service.py           # BM25 关键词检索服务（SQLite FTS5）
│   │   ├── multi_recall_service.py         # 多路召回编排（向量+BM25→RRF 融合）
│   │   ├── reranker_service.py             # Rerank 重排序服务（DashScope qwen3-rerank）
│   │   ├── query_rewriter.py               # Query 改写服务（书面语改写+多查询扩展）
│   │   └── document_splitter_service.py    # 文档分割服务
│   ├── agent/                              # Agent 模块
│   │   ├── mcp_client.py                   # MCP 客户端（工具调用）
│   │   └── aiops/                          # AIOps 核心逻辑
│   │       ├── planner.py                  # 计划制定器
│   │       ├── executor.py                 # 步骤执行器
│   │       ├── replanner.py                # 重规划器
│   │       ├── state.py                    # 状态定义
│   │       └── utils.py                    # 工具函数
│   ├── models/                             # 数据模型层
│   │   ├── aiops.py                        # AIOps 模型
│   │   ├── document.py                     # 文档模型
│   │   ├── request.py                      # 请求模型
│   │   └── response.py                     # 响应模型
│   ├── tools/                              # Agent 工具集
│   │   ├── knowledge_tool.py               # 知识库查询工具（改写→并发检索→合并去重）
│   │   └── time_tool.py                    # 时间工具
│   ├── core/                               # 核心组件
│   │   ├── llm_factory.py                  # LLM 工厂（模型管理）
│   │   └── milvus_client.py                # Milvus 客户端
│   └── utils/                              # 工具类
│       └── logger.py                       # 日志配置（Loguru）
├── evals/                                  # RAGAS 评测模块
│   ├── __init__.py                         # 模块入口
│   ├── dataset_builder.py                 # 评测数据集构建器（LLM 自动生成 QA 对）
│   ├── ragas_evaluator.py                 # RAGAS 评测引擎（检索+生成 6 项指标）
│   ├── run_eval.py                         # CLI 统一入口脚本
│   ├── datasets/                           # 评测数据集目录
│   └── reports/                            # 评测报告输出目录
├── db/                                     # 数据持久化
│   └── oncall_conversation_memory.db       # SQLite 对话记忆数据库
│   └── bm25_index.db                       # 存储BM25关键词检索结果
├── docs/                                   # 开发文档
│   ├── multi-recall-implementation.md      # 多路召回实现文档
│   ├── query-rewrite-implementation.md     # Query 改写实现文档
│   ├── ragas-evaluation-guide.md           # RAGAS 评测指南
│   ├── context-compression-guide.md        # 上下文压缩指南
│   ├── multi-file-support-guide.md         # 多文件支持指南
│   └── 对话记忆持久化实现文档.md             # 对话记忆持久化文档
├── static/                                 # Web 前端（纯静态）
│   ├── index.html                          # 主页面
│   ├── app.js                              # 前端逻辑
│   └── styles.css                          # 样式表
├── mcp_servers/                            # MCP 服务器
│   ├── cls_server.py                       # CLS 日志查询服务--本地模拟数据
│   ├── cls_server_true.py                  # CLS 日志查询服务--可对接腾讯云 CLS
│   ├── monitor_server.py                   # 监控数据服务
│   └── README.md                           # MCP 服务说明
├── aiops-docs/                             # 运维知识库（Markdown 文档）
├── logs/                                   # 日志目录（Loguru 自动创建）
│   └── app_YYYY-MM-DD.log                  # 按天轮转的日志文件
├── uploads/                                # 上传文件临时目录
├── volumes/                                # Milvus 数据持久化目录
├── .env                                    # 环境变量配置（需手动创建）
├── vector-database.yml                     # Milvus Docker Compose 配置
├── pyproject.toml                          # 项目配置（依赖、元数据）
├── uv.lock                                 # uv 依赖锁定文件
├── .gitignore                              # 忽略提交 git 文件配置
├── start-windows.bat                       # 启动服务脚本
├── stop-windows.bat                        # 停止服务脚本
└── README.md                               # 项目说明
```

## 🎯 AIOps 智能运维

基于 **Plan-Execute-Replan** 模式实现自动故障诊断。

### 核心特性
- ✅ 自动制定诊断计划（Planner）
- ✅ 智能工具调用（Executor）
- ✅ 动态调整步骤（Replanner）
- ✅ 流式输出诊断过程
- ✅ 生成结构化报告

### 快速测试

```bash
# 服务已通过 make init 自动启动
# 如需重启服务：make restart

# 访问 Web 界面，点击"智能运维与诊断工具"
# 或使用 API
curl -X POST "http://localhost:9900/api/aiops" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test"}' \
  --no-buffer
```

### 诊断流程
```
1. Planner 制定计划 → 生成 4-6 个诊断步骤
2. Executor 执行步骤 → 调用 MCP 工具（日志查询、监控数据）
3. Replanner 评估结果 → 决定继续/调整/生成报告
4. 输出诊断报告 → 根因分析 + 运维建议
```

### 启动和停止项目
```
# 启动所有服务
.\start-windows.bat

# 停止所有服务
.\stop-windows.bat
```