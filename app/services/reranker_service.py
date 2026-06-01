"""
重排序服务模块 - 基于阿里云 DashScope Rerank API

在向量检索召回文档后，使用专门的 Rerank 模型（qwen3-rerank）
对候选文档重新进行精细的相关性打分，提升检索精度。

核心流程:
1. 向量检索召回 N 篇候选文档（如 10 篇）
2. Rerank 模型对每篇文档与 query 进行相关性打分
3. 按分数降序排列，取前 K 篇（如 3 篇）返回
"""

from typing import List

import dashscope
from http import HTTPStatus
from langchain_core.documents import Document
from loguru import logger

from app.config import config


class RerankerService:
    """
    重排序服务

    封装 DashScope TextReRank API，提供文档重排能力。
    API 调用失败时自动降级为原始顺序，不影响主流程可用性。
    """

    def __init__(self, model: str | None = None):
        """
        初始化重排序服务

        Args:
            model: 重排序模型名称，默认使用 config.rerank_model（qwen3-rerank）
        """
        self.model = model or config.rerank_model
        logger.info(f"重排序服务初始化完成, model={self.model}")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int | None = None,
    ) -> List[Document]:
        """
        对文档列表进行重排序

        Args:
            query: 用户查询文本
            documents: 待重排的文档列表（向量检索召回的候选文档）
            top_n: 重排后返回前 N 篇，默认使用 config.rag_top_k

        Returns:
            List[Document]: 按相关性降序排列的文档列表

        降级策略:
            - API 调用失败时，返回原始文档顺序的前 top_n 篇
            - 文档为空时，直接返回空列表
        """
        if not documents:
            logger.warning("重排序输入文档列表为空，跳过")
            return []

        if top_n is None:
            top_n = config.rag_top_k

        # 如果召回数量 <= 目标数量，无需重排
        if len(documents) <= top_n:
            logger.debug(
                f"召回文档数({len(documents)}) <= 目标数量({top_n})，跳过重排"
            )
            return documents

        try:
            logger.info(
                f"开始重排序: query长度={len(query)}, "
                f"候选文档数={len(documents)}, top_n={top_n}"
            )

            # 提取文档文本列表
            doc_texts = [doc.page_content for doc in documents]

            # 调用 DashScope Rerank API
            # 参考: https://help.aliyun.com/zh/model-studio/rerank
            resp = dashscope.TextReRank.call(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=top_n,
                return_documents=False,  # 我们只需要索引和分数，文档原文已有
            )

            if resp.status_code != HTTPStatus.OK:
                logger.error(
                    f"Rerank API 调用失败: status_code={resp.status_code}, "
                    f"message={resp.message if hasattr(resp, 'message') else 'unknown'}"
                )
                return self._fallback(documents, top_n)

            # 解析重排结果
            # resp.output.results 是 List[dict]，每个 dict 包含:
            #   - index: 原始 documents 列表中的索引
            #   - relevance_score: 相关性分数（越高越相关）
            rerank_results = resp.output.results

            if not rerank_results:
                logger.warning("Rerank API 返回空结果，降级为原始顺序")
                return self._fallback(documents, top_n)

            # 按 API 返回的顺序（已按 relevance_score 降序排列）重新组织文档
            reordered_docs = []
            for result in rerank_results:
                original_index = result.index
                score = result.relevance_score
                doc = documents[original_index]
                # 将 rerank 分数写入文档元数据，方便调试和追溯
                doc.metadata["rerank_score"] = score
                doc.metadata["rerank_index"] = original_index
                reordered_docs.append(doc)

            logger.info(
                f"重排序完成: 返回 {len(reordered_docs)} 篇文档, "
                f"分数范围: [{rerank_results[-1].relevance_score:.4f}, "
                f"{rerank_results[0].relevance_score:.4f}]"
            )

            return reordered_docs

        except Exception as e:
            logger.error(f"重排序异常，降级为原始顺序: {e}")
            return self._fallback(documents, top_n)

    def _fallback(
        self, documents: List[Document], top_n: int
    ) -> List[Document]:
        """
        降级策略：返回原始文档顺序的前 top_n 篇

        当 Rerank API 不可用时，取向量检索返回的前 N 篇作为兜底。
        向量检索结果是按 L2 距离（越小越相似）排序的，有一定参考价值。
        """
        logger.warning(
            f"重排序降级: 取原始召回结果前 {top_n} 篇 "
            f"(共召回 {len(documents)} 篇)"
        )
        # 标记这些文档未经重排
        for doc in documents[:top_n]:
            doc.metadata["rerank_score"] = None
            doc.metadata["rerank_fallback"] = True
        return documents[:top_n]


# 全局单例
reranker_service = RerankerService()
