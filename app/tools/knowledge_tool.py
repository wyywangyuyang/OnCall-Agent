"""
知识检索工具 -> 从向量数据库中检索相关信息
"""
from typing import Tuple, List

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger
from app.config import config
from app.services.vector_store_manager import vector_store_manager


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """
    从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。

    Args:
        query: 用户的问题或查询

    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识库检索工具被调用：query='{query}'")

        # 从向量数据库中检索相关文档
        vector_store = vector_store_manager.get_vector_store()

        if config.rerank_enabled:
            # ========== 重排模式 ==========
            if config.bm25_enabled:
                # ---- 多路召回: 向量 + BM25 → RRF 融合 ----
                logger.info(
                    f"多路召回模式: 向量路K={config.vector_recall_top_k}, "
                    f"BM25路K={config.bm25_top_k}"
                )
                from app.services.multi_recall_service import multi_recall_service
                docs = multi_recall_service.recall(query)

                if not docs:
                    logger.warning("多路召回未找到相关文档")
                    return "未找到相关信息。", []

                logger.info(f"多路召回返回 {len(docs)} 篇候选文档，开始重排序")
            else:
                # ---- 单路向量召回（原有逻辑，不改动）----
                retrieve_k = config.rerank_retrieve_top_k
                logger.info(f"重排模式: 先召回 {retrieve_k} 篇候选文档")

                retriever = vector_store.as_retriever(
                    search_kwargs={"k": retrieve_k}
                )
                docs = retriever.invoke(query)

                if not docs:
                    logger.warning("知识库检索工具未找到相关文档")
                    return "未找到相关信息。", []

                logger.info(f"向量检索召回 {len(docs)} 篇文档，开始重排序")

            # Step 2: Rerank — 用 Rerank 模型精选最相关的文档
            from app.services.reranker_service import reranker_service
            docs = reranker_service.rerank(query, docs, top_n=config.rag_top_k)

            logger.info(f"重排序后返回 {len(docs)} 篇文档")
        else:
            # ========== 原始模式（不重排）==========
            logger.info(f"普通模式: 召回 {config.rag_top_k} 篇文档（不重排）")
            retriever = vector_store.as_retriever(
                search_kwargs={"k": config.rag_top_k}
            )
            docs = retriever.invoke(query)

            if not docs:
                logger.warning("知识库检索工具未找到相关文档")
                return "未找到相关信息。", []

        # 格式化文档为上下文
        context = format_docs(docs)

        logger.info(f"检索完成, 返回 {len(docs)} 个相关文档")
        return context, docs

    except Exception as e:
        logger.error(f"知识库检索工具调用失败：{e}")
        return f"检索知识时发生错误：{str(e)}。", []

def format_docs(docs: List[Document]) -> str:
    """
    格式化文档为上下文文本

    Args:
        docs: 文档列表

    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []

    # enumerate(docs, 1) -> 对 docs 列表进行枚举,从索引 1 开始计数(而不是默认的 0)
    # i - 当前迭代的索引号(从 1 开始)
    # doc - 当前迭代的文档对象
    # docs - 要遍历的文档列表
    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata
        source = metadata.get("file_name", "未知来源")

        # 提取标题信息（如果有）
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])

        header_str = ">".join(headers) if headers else ""

        # 构建格式化文本
        formatted = f"【参考资料】{i}"
        if header_str:
            formatted += f"\n标题：{header_str}"
        formatted += f"\n来源：{source}"
        formatted += f"\n内容：{doc.page_content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)
