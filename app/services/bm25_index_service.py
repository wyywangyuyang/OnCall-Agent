"""
BM25 关键词检索服务 — 基于 SQLite FTS5

在向量检索之外增加关键词检索通路，形成多路召回架构。
数据库文件 bm25_index.db 独立于对话记忆库，存储在 db/ 目录下。

FTS5 表以 chunk_id 为主键，与 Milvus 一一对应，保证文档写入/删除时
两条通路的数据一致性。

中文支持: 默认 unicode61 tokenizer，对 CJK 字符按单字/二元组切分，
提供基础的中文关键词匹配能力。如需更精准的 jieba 分词，可编译扩展。
"""

import os
import sqlite3
from typing import Dict, List

from langchain_core.documents import Document
from loguru import logger

from app.config import config


class Bm25IndexService:
    """BM25 关键词检索引擎 — SQLite FTS5 实现"""

    # FTS5 表名
    TABLE_NAME: str = "bm25_chunks"

    def __init__(self):
        """初始化 BM25 索引服务（延迟建表，首次使用时触发）"""
        self.db_path: str = ""
        self._initialized: bool = False
        logger.info("BM25 索引服务创建完成（FTS5 表将在首次使用时自动创建）")

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def add_chunks(self, documents: List[Document], chunk_ids: List[str]) -> None:
        """
        批量写入文档分片到 FTS5 索引

        调用时机: 文档索引完成后，与 Milvus 同步写入。
        先按 source 删除旧数据再插入新数据（由调用方负责删除逻辑）。

        Args:
            documents: 文档分片列表
            chunk_ids:  对应的分片 UUID 列表（与 Milvus 主键一致）
        """
        if not documents:
            return

        self._ensure_initialized()
        assert len(documents) == len(chunk_ids), (
            f"documents 与 chunk_ids 长度不一致: {len(documents)} vs {len(chunk_ids)}"
        )

        with sqlite3.connect(self.db_path) as conn:
            for doc, cid in zip(documents, chunk_ids):
                source = doc.metadata.get("source", "")
                file_name = doc.metadata.get("file_name", "")
                content = doc.page_content
                conn.execute(
                    f"INSERT INTO {self.TABLE_NAME}(chunk_id, content, source, file_name) "
                    "VALUES(?, ?, ?, ?)",
                    (cid, content, source, file_name),
                )
            conn.commit()

        logger.info(f"BM25 索引写入完成: {len(documents)} 个分片")

    def delete_by_source(self, file_path: str) -> int:
        """
        按来源文件路径删除 FTS5 索引记录

        Args:
            file_path: 文件路径（与 metadata["source"] 一致）

        Returns:
            int: 删除的分片数量
        """
        self._ensure_initialized()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"DELETE FROM {self.TABLE_NAME} WHERE source = ?",
                (file_path,),
            )
            deleted = cursor.rowcount
            conn.commit()

        if deleted > 0:
            logger.info(f"BM25 索引删除: {file_path}, 移除 {deleted} 个分片")
        return deleted

    def search(
        self, query: str, top_k: int = 10
    ) -> List[Dict]:
        """
        BM25 关键词检索

        使用 FTS5 内置 BM25 打分函数，返回按相关性降序排列的结果。
        FTS5 的 bm25() 返回负值（越小越相关），此处仅用作排名依据。

        Args:
            query: 用户查询文本
            top_k: 返回的最大结果数量

        Returns:
            List[Dict]: 每个结果包含:
                - chunk_id:  分片 UUID（与 Milvus 主键对应）
                - content:   分片文本
                - score:     BM25 分数（负值）
                - metadata:  {"source", "file_name"}
        """
        self._ensure_initialized()

        if not query or not query.strip():
            logger.warning("BM25 查询为空，返回空结果")
            return []

        safe_query = self._sanitize_query(query)
        if not safe_query:
            return []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    f"""
                    SELECT chunk_id, content, source, file_name, bm25({self.TABLE_NAME}) AS score
                    FROM {self.TABLE_NAME}
                    WHERE {self.TABLE_NAME} MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (safe_query, top_k),
                )
                rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            # FTS5 语法错误（如用户输入了特殊字符串），降级返回空
            logger.warning(f"BM25 查询语法错误 (已降级): {e}, query='{safe_query[:100]}'")
            return []

        results: List[Dict] = []
        for row in rows:
            results.append({
                "chunk_id": row[0],
                "content": row[1],
                "score": row[4],
                "metadata": {
                    "source": row[2],
                    "file_name": row[3],
                },
            })

        logger.info(
            f"BM25 检索完成: query='{query[:60]}...', "
            f"safe_query='{safe_query[:60]}', 结果数={len(results)}"
        )
        return results

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """确保数据库和 FTS5 表已创建（幂等）"""
        if self._initialized:
            return

        db_dir = config.sqlite_db_dir
        os.makedirs(db_dir, exist_ok=True)
        self.db_path = os.path.join(db_dir, config.bm25_db_name)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.TABLE_NAME} USING fts5(
                    chunk_id,
                    content,
                    source,
                    file_name
                )
            """)
            conn.commit()

        self._initialized = True
        logger.info(f"BM25 FTS5 索引已初始化: {self.db_path}")

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """
        转义 FTS5 查询中的特殊字符，防止语法错误

        FTS5 查询语法中以下字符/词组有特殊含义:
        - 双引号: 短语匹配边界
        - *, AND, OR, NOT, NEAR: 操作符
        移除双引号避免语法错误；其余字符作为普通 token 被 FTS5 自动处理。
        """
        # 移除双引号（最常见的 FTS5 语法错误来源）
        sanitized = query.replace('"', "").replace("'", "")
        return sanitized.strip()


# ------------------------------------------------------------------
# 全局单例
# ------------------------------------------------------------------
bm25_index_service = Bm25IndexService()
