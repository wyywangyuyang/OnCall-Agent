# CLAUDE.md — OnCall-Agent 项目规范

> 本文件用于指导 Claude Code 在本项目中编写和生成代码。所有代码改动必须遵循以下规范。

---

## 1. 项目概述

**项目名称**: OnCall-Agent
**定位**: 企业级智能对话和运维助手

**核心能力**:
- 智能对话: LangChain 多轮对话 + 流式输出
- RAG 问答: 向量检索增强，支持文档上传和自动索引
- AIOps 诊断: Plan-Execute-Replan 自动故障诊断
- MCP 集成: 日志查询和监控数据工具接入
- Web 界面: 纯静态 HTML/JS/CSS 前端

---

## 2. 技术栈

| 类别 | 技术/工具 |
|------|-----------|
| 语言 | Python 3.11+ (目标 3.13) |
| Web 框架 | FastAPI + uvicorn + sse-starlette |
| AI 框架 | LangChain 1.2+ / LangGraph 1.1+ / langchain-qwq |
| LLM 服务 | 阿里云 DashScope (通义千问 / DeepSeek) |
| 向量数据库 | Milvus (Docker 部署) |
| 嵌入模型 | text-embedding-v4 (DashScope) |
| BM25 检索 | SQLite FTS5 |
| 对话记忆 | SQLite (langgraph-checkpoint-sqlite + aiosqlite) |
| MCP | langchain-mcp-adapters + fastmcp |
| 数据校验 | Pydantic v2 + pydantic-settings |
| 日志 | Loguru |
| 包管理 | uv (uv.lock 依赖锁定) |
| 构建 | setuptools |
| 代码质量 | black + ruff + isort + mypy + pyright |
| 测试 | pytest + pytest-asyncio + pytest-cov |

---

## 3. 项目结构

```
OnCall-Agent/
├── app/                          # 主应用代码（setuptools packages.find 指定）
│   ├── main.py                   # FastAPI 入口，生命周期管理
│   ├── config.py                 # Pydantic Settings 配置管理（全局 config 单例）
│   ├── api/                      # API 路由层
│   │   ├── chat.py               # 对话接口（普通 + SSE 流式）
│   │   ├── aiops.py              # AIOps 智能运维接口
│   │   ├── file.py               # 文件上传/管理接口
│   │   └── health.py             # 健康检查接口
│   ├── agent/                    # Agent 编排层
│   │   ├── aiops/                # AIOps Plan-Execute-Replan Agent
│   │   │   ├── state.py          # PlanExecuteState 状态定义
│   │   │   ├── planner.py        # 规划节点
│   │   │   ├── executor.py       # 执行节点
│   │   │   ├── replanner.py      # 重规划节点
│   │   │   └── utils.py          # Agent 工具函数
│   │   └── mcp_client.py         # MCP 客户端管理（全局单例 + 重试）
│   ├── core/                     # 核心基础设施
│   │   ├── llm_factory.py        # LLM 工厂（OpenAI 兼容模式调用 DashScope）
│   │   └── milvus_client.py      # Milvus 连接管理
│   ├── models/                   # 数据模型（Pydantic）
│   │   ├── request.py            # 请求模型
│   │   ├── response.py           # 响应模型
│   │   ├── aiops.py              # AIOps 模型
│   │   └── document.py           # 文档模型
│   ├── services/                 # 业务逻辑层
│   │   ├── rag_agent_service.py  # RAG Agent 服务（全局单例）
│   │   ├── multi_recall_service.py    # 多路召回编排（向量 + BM25 → RRF）
│   │   ├── bm25_index_service.py      # BM25 FTS5 索引服务
│   │   ├── vector_embedding_service.py # 向量嵌入服务
│   │   ├── vector_index_service.py    # 向量索引服务
│   │   ├── vector_search_service.py   # 向量检索服务
│   │   ├── vector_store_manager.py    # 向量存储管理器
│   │   ├── reranker_service.py        # Rerank 重排序服务
│   │   ├── query_rewriter.py          # Query 改写服务
│   │   └── document_splitter_service.py # 文档分块服务
│   ├── tools/                    # LangChain 工具定义
│   │   ├── knowledge_tool.py     # 知识检索工具
│   │   └── time_tool.py          # 时间工具
│   └── utils/
│       └── logger.py             # Loguru 日志配置
├── mcp_servers/                  # MCP 服务端
│   ├── cls_server.py             # 腾讯云 CLS 日志查询服务
│   └── monitor_server.py         # 监控数据查询服务
├── aiops-docs/                   # AIOps 运维知识文档
├── static/                       # 静态前端文件
├── db/                           # SQLite 数据库文件
├── logs/                         # 日志文件（按天轮转）
├── evals/                        # RAGAS 评测
├── pyproject.toml                # 项目配置和依赖
├── vector-database.yml           # Milvus Docker Compose 配置
├── start-windows.bat             # Windows 启动脚本
├── stop-windows.bat              # Windows 停止脚本
└── .env                          # 环境变量（不提交 Git）
```

---

## 4. 编码规范

### 4.1 代码风格

- **格式化工具**: black（line-length=100, target=py313）
- **Linter**: ruff（line-length=100, target=py313）
- **Import 排序**: isort（profile=black, known-first-party=["app"]）
- **类型检查**: mypy + pyright（basic 模式，ignore_missing_imports=true）
- **行宽**: 100 字符
- **缩进**: 4 空格（禁止 Tab）
- **字符串引号**: 双引号 `"`
- **尾随逗号**: 多行结构使用尾随逗号

### 4.2 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 类名 | PascalCase | `RagAgentService`, `MultiRecallService` |
| 函数/方法 | snake_case | `retrieve_knowledge`, `_vector_recall` |
| 变量 | snake_case | `vector_docs`, `bm25_results` |
| 常量 | UPPER_SNAKE | `DASHSCOPE_BASE_URL` |
| 私有方法 | 前缀下划线 | `_build_system_prompt`, `_bm25_recall_safe` |
| 文件名 | snake_case | `rag_agent_service.py`, `multi_recall_service.py` |
| 模块级单例 | snake_case（模块级变量） | `config`, `rag_agent_service`, `multi_recall_service` |

### 4.3 注释规范

- **模块级文档字符串**: 每个文件开头使用三引号文档字符串说明模块职责
- **类文档字符串**: 说明类的用途和职责
- **方法文档字符串**: 包含 Args / Returns / Yields 等结构化说明
- **行内注释**: 使用中文，解释"为什么"而非"是什么"
- **分段注释**: 使用 `# ===` 分隔符标记代码段落（参考 `knowledge_tool.py`、`multi_recall_service.py` 的风格）

示例:
```python
"""
多路召回编排服务

将向量检索与 BM25 关键词检索两路召回结果
通过 RRF 算法融合，再交由 Reranker 进行精排。
"""
```

### 4.4 类型注解

- 公共方法必须添加参数和返回值类型注解
- 私有方法鼓励添加类型注解
- 使用 `typing` 模块的泛型类型: `List`, `Dict`, `Tuple`, `Optional`, `Any`
- LangGraph 状态使用 `TypedDict` 定义，聚合字段使用 `Annotated[T, operator.add]`

### 4.5 异步编程

- FastAPI 路由方法使用 `async def`
- I/O 密集型操作使用 `async/await`（LLM 调用、数据库、MCP 工具）
- 同步检索工具可用 `ThreadPoolExecutor` 并发（参考 `knowledge_tool.py` 多查询检索）
- 异步生成器使用 `AsyncGenerator` 类型注解（参考流式对话 `query_stream`）

---

## 5. 架构规范

### 5.1 分层架构

```
API 路由层 (api/)  →  服务层 (services/)  →  工具/基础设施层 (tools/, core/)
     ↑                      ↑                        ↑
  请求/响应模型          业务逻辑              LLM/Milvus/MCP
  (models/)          (Agent 编排)           (外部服务对接)
```

- **API 层**: 只做请求接收、参数校验、响应格式化，不包含业务逻辑
- **Service 层**: 核心业务逻辑，全局单例模式
- **Tool 层**: LangChain `@tool` 装饰器定义的工具函数
- **Core 层**: 基础设施（LLM 工厂、Milvus 客户端、日志）

### 5.2 全局单例模式

项目使用模块级单例模式管理服务实例:
- `config` — 全局配置（`app/config.py`）
- `rag_agent_service` — RAG Agent 服务（`app/services/rag_agent_service.py`）
- `multi_recall_service` — 多路召回服务
- `vector_store_manager` — 向量存储管理器
- `reranker_service` — 重排序服务
- `bm25_index_service` — BM25 索引服务
- `query_rewriter` — Query 改写服务
- `milvus_manager` — Milvus 连接管理器
- `llm_factory` — LLM 工厂

新增服务应遵循此模式：在模块底部创建全局实例。

### 5.3 延迟初始化

异步资源（MCP 客户端、SQLite 连接）使用延迟初始化:
- 构造函数 `__init__` 中只做同步配置
- 首次调用时通过 `_initialize_xxx()` 异步方法完成资源初始化
- 使用 `_xxx_initialized` 标志位防止重复初始化

### 5.4 配置管理

- 所有配置通过 `app/config.py` 的 `Settings` 类集中管理
- 使用 `pydantic-settings` 从 `.env` 文件加载
- 新增配置项时：在 `Settings` 类中添加带类型注解和默认值的属性
- `.env` 文件不提交 Git，仅在本地使用
- **注意**: `config.py` 中 `PROJECT_ROOT = Path(__file__).parent.parent`，应用需从项目根目录运行

### 5.5 API 响应格式

**普通对话接口**: 统一返回 `{"code": 200, "message": "success", "data": {...}}` 格式

**流式对话接口**: SSE 格式，事件类型:
- `content` — 内容流式块
- `tool_call` — 工具调用状态
- `search_results` — 检索结果
- `debug` — 调试信息
- `done` — 完成信号
- `error` — 错误信息

### 5.6 RAG 检索管线

完整检索流程:
```
Query 改写（书面语改写 + 多查询扩展）
    → 并发检索（多路召回: 向量 + BM25 → RRF 融合）
    → Rerank 精排
    → 合并去重
    → 格式化输出
```

- `bm25_enabled=False` 时回退到纯向量路
- BM25 路检索失败时自动降级，不影响服务可用性
- 向量路是核心路径，失败应向上抛出

### 5.7 AIOps Agent 架构

Plan-Execute-Replan 模式:
```
Planner（规划）→ Executor（执行）→ Replanner（重规划）→ 循环或结束
```
- 状态通过 `PlanExecuteState(TypedDict)` 在节点间传递
- `past_steps` 使用 `Annotated[List[tuple], operator.add]` 实现追加式更新
- 工具包括本地工具（knowledge_tool, time_tool）和 MCP 工具

---

## 6. 日志规范

- 使用 **Loguru**，禁止使用 `print()` 或标准 `logging`
- 导入方式: `from loguru import logger`
- 日志级别: DEBUG（开发）/ INFO（生产）
- 关键操作必须记录日志: 初始化、工具调用、检索结果、错误
- 日志格式: `{time} | {level} | {module}.{function}:{line} | {message}`
- 文件日志按天轮转，保留 7 天，自动压缩
- 会话相关日志带 `[会话 {session_id}]` 前缀
- Windows 环境下 stdout 已做 UTF-8 修复（见 `logger.py`）

---

## 7. 依赖管理

- 包管理器: **uv**（使用 `uv.lock` 锁定依赖版本）
- 依赖索引: 清华源 `https://pypi.tuna.tsinghua.edu.cn/simple`
- 依赖声明: `pyproject.toml` 的 `[project.dependencies]`
- 开发依赖: `[project.optional-dependencies]` 的 `dev` 组
- 添加新依赖: 修改 `pyproject.toml` 后执行 `uv sync`
- **版本兼容约束**: `pymilvus>=2.4.3,<3.0.0`（与 Milvus 版本绑定）

---

## 8. 启动与运行

### 8.1 前置条件

- Python 3.11+ 虚拟环境（`.venv/`）
- Docker Desktop 运行中（Milvus 通过 Docker Compose 部署）
- `.env` 文件已配置（DashScope API Key 等）

### 8.2 启动方式

```powershell
# Windows 一键启动（Milvus + MCP 服务 + FastAPI + 文档上传）
.\start-windows.bat

# 手动启动 FastAPI（需先启动 Milvus 和 MCP 服务）
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 9900
```

### 8.3 服务地址

| 服务 | 地址 |
|------|------|
| Web 界面 | http://localhost:9900 |
| API 文档 | http://localhost:9900/docs |
| Milvus | localhost:19530 |
| Attu (Milvus 管理) | http://localhost:8000 |
| CLS MCP 服务 | http://localhost:8003/mcp |
| Monitor MCP 服务 | http://localhost:8004/mcp |

### 8.4 停止服务

```powershell
.\stop-windows.bat
```

---

## 9. 代码质量工具

### 9.1 格式化与检查命令

```bash
# 格式化
black app/ --line-length 100

# Lint 检查
ruff check app/

# Import 排序
isort app/ --profile black --line-length 100

# 类型检查
mypy app/

# 运行测试
pytest
```

### 9.2 Ruff 规则

启用的规则集:
- `E` — pycodestyle errors
- `W` — pycodestyle warnings
- `F` — pyflakes
- `I` — isort
- `C` — flake8-comprehensions
- `B` — flake8-bugbear
- `UP` — pyupgrade

忽略的规则:
- `E501` — 行过长（由 black 处理）
- `B008` — 默认参数中的函数调用
- `C901` — 过于复杂
- `W191` — Tab 缩进

### 9.3 测试规范

- 测试目录: `tests/`（当前尚未创建）
- 测试文件命名: `test_*.py` 或 `*_test.py`
- 测试类命名: `Test*`
- 测试方法命名: `test_*`
- 异步测试模式: `asyncio_mode = "auto"`
- 覆盖率目标: `app/` 模块

---

## 10. 文档与知识库

- `aiops-docs/` — AIOps 运维知识文档（Markdown 格式，启动时自动上传到向量库）
- `docs/` — 技术实现文档（已在 .gitignore 中排除）
- `evals/` — RAGAS 评测框架

---

## 11. 开发流程规范

1. **方案对齐**: 复杂功能改动需先输出方案与接收方对齐确认，确认后再开始编码
2. **编码实现**: 遵循上述编码规范
3. **代码检查**: 提交前运行 black + ruff + mypy
4. **测试验证**: 确保现有测试通过，新增功能需补充测试
5. **提交规范**: Git commit message 使用简洁描述性中文

---

## 12. 关键注意事项

1. **Windows 环境**: 终端需设置 `chcp 65001`（UTF-8），Python stdout 已在 `logger.py` 中做编码修复
2. **Pydantic Settings**: 配置类属性必须有类型注解，`Config` 类属性末尾禁止多余逗号，需从项目根目录运行
3. **Docker**: 使用 Docker Desktop，Milvus 镜像需配置国内镜像加速器
4. **MCP 工具**: 使用全局单例管理器 + 重试拦截器（`get_mcp_client_with_retry`）
5. **LangChain 版本**: 最低 1.2.17+，注意 API 可能与旧版不兼容
6. **Structured Output**: `deepseek-v4-pro` 对 `tool_choice` 支持不好，优先使用 `json_mode`（参考 `planner.py`）
7. **流式输出**: 使用 `agent.astream(stream_mode="messages")`，通过 `content_blocks` 提取文本
8. **Milvus 连接**: 在 FastAPI lifespan 中连接和关闭，不要在模块加载时连接
9. **会话记忆**: 使用 `AsyncSqliteSaver` + `thread_id` 实现多轮对话持久化
10. **安全**: 密钥和敏感信息只放 `.env`，禁止硬编码
