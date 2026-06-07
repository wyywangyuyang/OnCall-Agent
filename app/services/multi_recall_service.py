"""
多路召回编排服务

将向量检索（Milvus）与 BM25 关键词检索（SQLite FTS5）两路召回结果
通过 RRF（Reciprocal Rank Fusion）算法融合、去重，生成统一候选集，
再交由 Reranker 进行精排。

降级策略:
- BM25 路检索失败时，自动回退到纯向量路，不影响服务可用性
- bm25_enabled=False 时，此服务不被调用（由 knowledge_tool 直接走原有路径）

RRF 算法:
    rrf_score(d) = Σ 1 / (k + rank(d, r))
    其中 k 为常数（默认 60），rank(d, r) 为文档 d 在排名列表 r 中的位置（1-indexed）
    同一 chunk_id 出现在多路时，取其 RRF 最高分（不累加），实现去重。

配置:
    - vector_recall_top_k: 向量路召回数量（默认 10）
    - bm25_top_k:         BM25 路召回数量（默认 10）
    - rrf_k:              RRF 融合常数 k（默认 60）
    - bm25_enabled:       BM25 开关（默认 True）
"""

from typing import Dict, List, Tuple

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.services.vector_store_manager import vector_store_manager


class MultiRecallService:
    """多路召回编排器 — 向量 + BM25 → RRF 融合 → 候选集"""

    def __init__(self):
        """初始化多路召回编排器"""
        logger.info(
            f"多路召回服务初始化完成 | "
            f"向量路 top_k={config.vector_recall_top_k}, "
            f"BM25路 top_k={config.bm25_top_k}, "
            f"RRF k={config.rrf_k}, "
            f"BM25={'启用' if config.bm25_enabled else '禁用'}"
        )

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def recall(self, query: str) -> List[Document]:
        """
        执行多路召回并返回融合后的候选文档列表

        Args:
            query: 用户查询文本

        Returns:
            List[Document]: RRF 融合后的文档列表，按相关性降序排列
                           每个 doc.metadata 包含:
                           - chunk_id:   分片 UUID
                           - rrf_score:  RRF 融合分数
                           - bm25_score: BM25 分数（仅 BM25 路结果）
                           - source/file_name: 来源信息

        降级:
            BM25 路任何异常 → 自动回退，仅使用向量路结果
        """
        if not query or not query.strip():
            logger.warning("查询为空，返回空结果")
            return []

        # ---- Step 1: 向量路召回 ----
        vector_docs = self._vector_recall(query)
        logger.info(f"向量召回: {len(vector_docs)} 篇")

        # ---- Step 2: BM25 路召回（带降级）----
        bm25_results: List[Dict] = []
        if config.bm25_enabled:
            bm25_results = self._bm25_recall_safe(query)
            logger.info(f"BM25 召回: {len(bm25_results)} 篇")

        # ---- Step 3: RRF 融合 ----
        if not vector_docs and not bm25_results:
            logger.warning("两路召回均为空")
            return []

        if not bm25_results:
            # BM25 路无结果（或已降级），直接返回向量路结果
            logger.info("BM25 路无结果，使用纯向量路排名")
            for rank, doc in enumerate(vector_docs, 1):
                doc.metadata["rrf_score"] = 1.0 / (config.rrf_k + rank)
            return vector_docs

        if not vector_docs:
            # 向量路无结果（极端情况），仅用 BM25 路
            logger.warning("向量路无结果，仅使用 BM25 路")
            return self._bm25_to_documents(bm25_results)

        # 两路都有结果 → RRF 融合
        fused = self._rrf_fusion(vector_docs, bm25_results)
        logger.info(f"RRF 融合完成: 候选集={len(fused)} 篇")
        return fused

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _vector_recall(self, query: str) -> List[Document]:
        """
        向量路召回

        直接使用 vector_store_manager 的 similarity_search 方法，
        返回的 Document.metadata 中已包含 chunk_id（由 add_documents 写入）。
        """
        try:
            docs = vector_store_manager.similarity_search(
                query, k=config.vector_recall_top_k
            )
            return docs
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            raise  # 向量路是核心路径，失败应向上抛出

    def _bm25_recall_safe(self, query: str) -> List[Dict]:
        """
        BM25 路召回（安全包装，异常自动降级）

        返回空列表时不影响主流程，上游会回退到纯向量路。
        """
        try:
            from app.services.bm25_index_service import bm25_index_service

            return bm25_index_service.search(query, top_k=config.bm25_top_k)
        except Exception as e:
            logger.warning(f"BM25 检索异常，降级为纯向量路: {e}")
            return []

    def _rrf_fusion(
        self,
        vector_docs: List[Document],
        bm25_results: List[Dict],
    ) -> List[Document]:
        """
        RRF（Reciprocal Rank Fusion）排名融合

        算法:
            对于每篇文档，计算其在各路排名中的 RRF 分数:
                rrf_score = 1 / (k + rank)
            同一 chunk_id 在多路中出现时，取最高 RRF 分（去重规则）。

        Args:
            vector_docs:   向量路召回结果
            bm25_results:  BM25 路召回结果

        Returns:
            List[Document]: 按 RRF 分数降序排列的文档列表
        """
        k = config.rrf_k
        # chunk_id → (max_rrf_score, Document)
        fused_map: Dict[str, Tuple[float, Document]] = {}

        # ---- 向量路排名 ----
        for rank, doc in enumerate(vector_docs, 1):
            cid = doc.metadata.get("chunk_id")
            if cid is None:
                # 兼容旧数据（无 chunk_id 的文档），使用 id() 作为 fallback
                cid = f"__vec_{rank}"
                logger.debug(f"文档缺少 chunk_id，使用 fallback: {cid}")
            rrf = 1.0 / (k + rank)
            if cid not in fused_map or rrf > fused_map[cid][0]:
                fused_map[cid] = (rrf, doc)

        # ---- BM25 路排名 ----
        for rank, result in enumerate(bm25_results, 1):
            cid = result["chunk_id"]
            rrf = 1.0 / (k + rank)
            if cid not in fused_map or rrf > fused_map[cid][0]:
                # BM25 结果转换为 Document
                doc = Document(
                    page_content=result["content"],
                    metadata={
                        **result.get("metadata", {}),
                        "chunk_id": cid,
                        "bm25_score": result.get("score"),
                    },
                )
                fused_map[cid] = (rrf, doc)

        # ---- 按 RRF 分数降序排列 ----
        sorted_items = sorted(
            fused_map.values(), key=lambda item: item[0], reverse=True
        )

        # 将 RRF 分数写入 metadata 方便调试
        fused_docs: List[Document] = []
        for rrf_score, doc in sorted_items:
            doc.metadata["rrf_score"] = round(rrf_score, 6)
            fused_docs.append(doc)

        # 统计两路贡献
        vec_ids = set()
        for doc in vector_docs:
            cid = doc.metadata.get("chunk_id")
            if cid:
                vec_ids.add(cid)
        bm25_ids = {r["chunk_id"] for r in bm25_results}
        overlap = vec_ids & bm25_ids

        logger.info(
            f"RRF 融合: 向量={len(vec_ids)} + BM25={len(bm25_ids)} "
            f"→ 融合后={len(fused_docs)} (两路重叠={len(overlap)})"
        )

        return fused_docs

    def _bm25_to_documents(self, bm25_results: List[Dict]) -> List[Document]:
        """BM25 结果转换为 Document 列表（用于纯 BM25 降级场景）"""
        docs = []
        for result in bm25_results:
            doc = Document(
                page_content=result["content"],
                metadata={
                    **result.get("metadata", {}),
                    "chunk_id": result["chunk_id"],
                    "bm25_score": result.get("score"),
                    "rrf_score": None,
                },
            )
            docs.append(doc)
        return docs


# ------------------------------------------------------------------
# 全局单例
# ------------------------------------------------------------------
multi_recall_service = MultiRecallService()
