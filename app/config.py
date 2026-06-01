"""
配置管理模块
使用 Pydantic Settings 实现类型安全的配置管理
"""
from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict

# 获取项目根目录（.env 文件所在目录）
PROJECT_ROOT = Path(__file__).parent.parent

class Settings(BaseSettings):
    """
    应用配置类
    """
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "OnCall-Agent"
    app_version: str = "0.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_model: str = "deepseek-v4-pro"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 3
    rag_model: str = "deepseek-v4-pro"  # 使用快速响应模型，不带扩展思考

    # 上下文自动压缩（Summarization）配置
    # 当对话历史的 token 数量达到模型最大上下文窗口的指定百分比时，
    # 自动使用大模型对前面的内容进行压缩总结，保留最近的消息不变
    summarization_enabled: bool = True       # 是否启用自动压缩
    summarization_trigger_fraction: float = 0.7  # 触发阈值（70% 上下文窗口时触发）
    summarization_keep_messages: int = 20    # 保留最近多少条消息不被压缩
    summarization_max_input_tokens: int = 1_048_576  # 模型最大上下文窗口（deepseek-v4-pro 为 1M tokens）

    # Rerank 重排序配置
    # 开启后，会先多召回一些文档，再用 Rerank 模型精选出最相关的
    rerank_enabled: bool = True          # 是否启用重排序
    rerank_model: str = "qwen3-rerank"  # 重排序模型（gte-rerank-v2 已于 2026-05-30 下线）
    rerank_retrieve_top_k: int = 10     # 召回阶段取多少篇文档参与重排

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    # 腾讯云 CLS 配置
    tencent_cloud_secret_id: str = ""
    tencent_cloud_secret_key: str = ""
    tencent_cloud_region: str = "ap-beijing"

    # 对话记忆持久化配置（SQLite）
    sqlite_db_dir: str = "db"                                # 数据库文件夹名（相对于项目根目录）
    sqlite_db_name: str = "oncall_conversation_memory.db"    # 数据库文件名

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# 全局配置实例
config = Settings()

