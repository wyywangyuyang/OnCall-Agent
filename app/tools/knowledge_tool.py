"""
知识检索工具 -> 从向量数据库 + BM25 索引中检索相关信息

支持 Query 改写（口语→书面语 + 多查询扩展）以提升召回命中率。
改写的多个查询并发执行检索，结果合并去重后返回。

检索管线:
  Query 改写 → 并发检索(多路召回 + Rerank) → 合并去重 → 格式化输出
"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.vector_store_manager import vector_store_manager


# ============================================================================
# 内部辅助函数
# ============================================================================


def _retrieve_single_query(query: str) -> List[Document]:
    """
    对单条查询执行完整检索管线

    封装 rerank_enabled / bm25_enabled 的所有分支路径，
    单查询和多查询场景共用同一逻辑。

    Args:
        query: 单条查询文本

    Returns:
        List[Document]: 检索到的文档列表（可能为空）
    """
    vector_store = vector_store_manager.get_vector_store()
    docs: List[Document] = []

    if config.rerank_enabled:
        # ---- 重排模式 ----
        if config.bm25_enabled:
            # 多路召回: 向量 + BM25 → RRF 融合
            from app.services.multi_recall_service import multi_recall_service

            docs = multi_recall_service.recall(query)
        else:
            # 单路向量召回
            retriever = vector_store.as_retriever(
                search_kwargs={"k": config.rerank_retrieve_top_k}
            )
            docs = retriever.invoke(query)

        # Rerank 精排
        if docs:
            from app.services.reranker_service import reranker_service

            docs = reranker_service.rerank(query, docs, top_n=config.rag_top_k)
    else:
        # ---- 普通模式（不重排）----
        retriever = vector_store.as_retriever(
            search_kwargs={"k": config.rag_top_k}
        )
        docs = retriever.invoke(query)

    return docs if docs else []


def _deduplicate_by_chunk_id(docs: List[Document]) -> List[Document]:
    """
    按 chunk_id 去重，保留最高 rerank_score

    多查询并发检索可能返回相同的文档分片，
    此函数按 chunk_id 去重，优先保留 rerank_score 最高的副本。

    Args:
        docs: 待去重的文档列表

    Returns:
        List[Document]: 去重后的文档列表，按 rerank_score 降序排列
    """
    if not docs:
        return []

    # 按 rerank_score 降序排列（高分在前，优先保留）
    def _sort_key(d: Document) -> float:
        score = d.metadata.get("rerank_score")
        if score is not None:
            return float(score)
        # 无 rerank_score 的文档排在最后
        return -1.0

    sorted_docs = sorted(docs, key=_sort_key, reverse=True)

    seen: set = set()
    result: List[Document] = []
    for doc in sorted_docs:
        cid = doc.metadata.get("chunk_id")
        if cid is None:
            # 无 chunk_id 的文档（旧数据兼容），使用 content hash 作为 fallback
            cid = f"__hash_{hash(doc.page_content)}"
            doc.metadata["chunk_id"] = cid
        if cid not in seen:
            seen.add(cid)
            result.append(doc)

    return result


# ============================================================================
# 知识检索工具
# ============================================================================


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """
    从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。

    检索流程:
    1. Query 改写（如启用）: 书面语改写 + 可选多查询扩展
    2. 并发检索: 每条查询独立走完整检索管线
    3. 合并去重: 多查询结果按 chunk_id 去重
    4. 格式化: 拼接为上下文文本

    Args:
        query: 用户的问题或查询

    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识库检索工具被调用：query='{query}'")

        # ================================================================
        # Step 1: Query 改写
        #   Phase 1 书面语改写 → 始终执行
        #   Phase 2 多查询扩展 → 可选，由 multi_query_expansion_enabled 控制
        # ================================================================
        if config.query_rewrite_enabled:
            from app.services.query_rewriter import query_rewriter

            queries = query_rewriter.rewrite(query)
            logger.info(
                f"Query 改写完成: 原始 → {len(queries)} 条查询 "
                f"(书面语改写{' + 多查询扩展' if config.multi_query_expansion_enabled else ''})"
            )
        else:
            queries = [query]

        # ================================================================
        # Step 2: 并发检索
        #   单查询: 直接同步调用（省去线程开销）
        #   多查询: ThreadPoolExecutor 并发执行
        # ================================================================
        if len(queries) == 1:
            all_docs = _retrieve_single_query(queries[0])
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            all_docs = []
            with ThreadPoolExecutor(max_workers=len(queries)) as executor:
                future_to_query = {
                    executor.submit(_retrieve_single_query, q): q
                    for q in queries
                }
                for future in as_completed(future_to_query):
                    q = future_to_query[future]
                    try:
                        docs = future.result()
                        if docs:
                            all_docs.extend(docs)
                            logger.debug(
                                f"子查询检索完成: '{q[:50]}...' → {len(docs)} 篇"
                            )
                    except Exception as e:
                        # 单个子查询失败不影响其他查询的结果收集
                        logger.warning(
                            f"子查询检索失败 ['{q[:60]}...']: {e}"
                        )

        # ================================================================
        # Step 3: 合并去重
        # ================================================================
        if not all_docs:
            logger.warning("知识库检索工具未找到相关文档")
            return "未找到相关信息。", []

        if len(queries) > 1:
            before = len(all_docs)
            all_docs = _deduplicate_by_chunk_id(all_docs)
            logger.info(f"多查询结果去重: {before} → {len(all_docs)} 篇")

        # ================================================================
        # Step 4: 格式化
        # ================================================================
        context = format_docs(all_docs)
        logger.info(f"检索完成, 返回 {len(all_docs)} 个相关文档")
        return context, all_docs

    except Exception as e:
        logger.error(f"知识库检索工具调用失败：{e}")
        return f"检索知识时发生错误：{str(e)}。", []


# ============================================================================
# 格式化函数
# ============================================================================


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
