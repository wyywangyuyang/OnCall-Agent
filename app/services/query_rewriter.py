"""
Query 改写服务 — 基于 LLM 的多阶段查询优化

两阶段流水线:
  Phase 1: 书面语改写 (rewrite_to_formal)
    口语 → 书面语，术语标准化，指代消解，语义补全
    始终执行（query_rewrite_enabled=True 时）
  Phase 2: 多查询扩展 (expand_queries)
    从正式查询生成多角度变体，扩大召回覆盖面
    可选（multi_query_expansion_enabled=True 时执行）

LLM 调用方式: 复用 DashScope OpenAI 兼容接口（与 vector_embedding_service 一致）

降级策略:
  - Phase 1 失败 → 返回原始查询
  - Phase 2 失败 → 返回空列表，最终仅使用 formal_query
  - JSON 解析失败 → 尝试正则提取，失败返回空列表
"""

import json
import re
from typing import List

from loguru import logger
from openai import OpenAI

from app.config import config


class QueryRewriter:
    """
    LLM 驱动的多阶段查询改写器

    使用方式:
        rewriter = QueryRewriter()
        queries = rewriter.rewrite("它怎么又报错了")
        # → ["CLS 日志服务为何反复出现错误", "CLS 报错原因分析", ...]
    """

    def __init__(self):
        """初始化 Query 改写器"""
        self.model = config.query_rewrite_model
        self._client: OpenAI | None = None

        logger.info(
            f"Query 改写服务初始化完成 | "
            f"model={self.model}, "
            f"书面语改写={'启用' if config.query_rewrite_enabled else '禁用'}, "
            f"多查询扩展={'启用' if config.multi_query_expansion_enabled else '禁用'} "
            f"(扩展数={config.multi_query_expansion_count})"
        )

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def rewrite(self, original_query: str) -> List[str]:
        """
        统一入口: 书面语改写 + 可选多查询扩展

        Args:
            original_query: 用户原始问题

        Returns:
            List[str]: 查询列表，第一条始终为 formal_query。
                      如果 Phase 2 未启用或失败，仅返回 [formal_query]。
                      如果 Phase 1 失败，返回 [original_query]。
        """
        if not original_query or not original_query.strip():
            return [original_query] if original_query else []

        # ---- Phase 1: 书面语改写（始终执行）----
        formal_query = self.rewrite_to_formal(original_query)
        logger.info(
            f"[Phase 1] 书面语改写: '{original_query[:50]}' → '{formal_query[:80]}'"
        )

        # ---- Phase 2: 多查询扩展（可选）----
        if config.multi_query_expansion_enabled:
            expanded = self.expand_queries(
                formal_query, count=config.multi_query_expansion_count
            )
            logger.info(f"[Phase 2] 多查询扩展: 生成 {len(expanded)} 条变体")
            return [formal_query] + expanded

        return [formal_query]

    def rewrite_to_formal(self, original_query: str) -> str:
        """
        Phase 1: 口语 → 书面语改写

        低温度 (0.1) 保证输出一致性和可预测性。
        失败时返回原始查询作为降级。

        Args:
            original_query: 用户原始问题（可能包含口语、代词、省略）

        Returns:
            str: 书面语形式的查询
        """
        prompt = self._build_formal_rewrite_prompt(original_query)

        try:
            response = self._call_llm(
                prompt=prompt,
                temperature=0.1,
                max_tokens=512,
            )
            if response and response.strip():
                # 去除可能的引号和空白
                cleaned = response.strip().strip('"').strip("'").strip()
                if cleaned:
                    return cleaned

            logger.warning("Phase 1 LLM 返回为空，降级为原始查询")
            return original_query

        except Exception as e:
            logger.warning(f"Phase 1 书面语改写失败（降级为原始查询）: {e}")
            return original_query

    def expand_queries(
        self, formal_query: str, count: int = 3
    ) -> List[str]:
        """
        Phase 2: 多角度查询扩展

        高温度 (0.7) 保证生成多样性。
        要求 LLM 输出 JSON 数组格式。
        失败时返回空列表。

        Args:
            formal_query: Phase 1 输出的正式查询
            count: 期望生成的变体数量

        Returns:
            List[str]: 变体查询列表（不含原始 formal_query）
        """
        prompt = self._build_expansion_prompt(formal_query, count)

        try:
            response = self._call_llm(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1024,
            )
            if not response or not response.strip():
                logger.warning("Phase 2 LLM 返回为空")
                return []

            queries = self._parse_json_response(response)
            if queries:
                # 去重并过滤与原查询高度相似的
                unique = self._deduplicate_queries(queries, formal_query)
                return unique[:count]

            return []

        except Exception as e:
            logger.warning(f"Phase 2 多查询扩展失败（降级为空列表）: {e}")
            return []

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_client(self) -> OpenAI:
        """获取 OpenAI 客户端（懒初始化，复用连接）"""
        if self._client is None:
            self._client = OpenAI(
                api_key=config.dashscope_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        return self._client

    def _call_llm(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        调用 LLM 并返回文本响应

        Args:
            prompt: 完整提示词
            temperature: 采样温度
            max_tokens: 最大输出 token 数

        Returns:
            str: LLM 响应的文本内容

        Raises:
            RuntimeError: API 调用失败
        """
        client = self._get_client()

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not response.choices:
            raise RuntimeError("LLM 返回空的 choices 列表")

        content = response.choices[0].message.content or ""
        return content.strip()

    # ------------------------------------------------------------------
    # 提示词模板
    # ------------------------------------------------------------------

    @staticmethod
    def _build_formal_rewrite_prompt(original_query: str) -> str:
        """构建 Phase 1 书面语改写提示词"""
        return f"""你是一个专业的查询改写助手。请将用户的口语化问题改写为书面语，使其更适合用于文档检索。

改写规则:
1. 术语标准化: 将口语/俚语替换为专业运维术语
   （如 "机器挂了"→"服务器宕机"，"内存爆了"→"内存使用率过高OOM"）
2. 指代消解: 将代词替换为具体实体名称
   （如 "它"→具体服务名，"那个"→具体组件名）
3. 语义补全: 补充省略的主语、宾语，使问题语义完整
4. 保持原始意图: 不添加原始问题中不存在的信息或假设
5. 直接输出改写后的单条查询，不要输出任何解释或额外内容，不要加引号

用户问题: {original_query}

改写后的查询:"""

    @staticmethod
    def _build_expansion_prompt(formal_query: str, count: int) -> str:
        """构建 Phase 2 多查询扩展提示词"""
        return f"""你是一个专业的查询扩展助手。请基于以下查询，生成 {count} 个不同角度的检索查询变体，以提升文档检索的召回覆盖。

扩展维度:
1. 同义词替换: 使用不同但等价的术语表达
   （如 "故障"→"异常/错误/失败"，"排查"→"诊断/定位"）
2. 角度切换: 从不同视角提问
   （原因分析、排查方法、解决方案、配置检查、监控告警等）
3. 粒度变化: 将宽泛问题具体化到特定场景，或将具体问题泛化为通用模式

要求:
- 每条变体应简洁明确，适合用于检索
- 变体之间应有明显差异，避免语义重复
- 严格按 JSON 格式输出，只输出一个 JSON 字符串数组，不要输出任何解释

原始查询: {formal_query}

输出示例: ["变体查询1", "变体查询2", "变体查询3"]

JSON 输出:"""

    # ------------------------------------------------------------------
    # 响应解析 & 去重
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(response: str) -> List[str]:
        """
        从 LLM 响应中解析 JSON 字符串数组

        尝试:
        1. 直接 json.loads
        2. 正则提取 [...] 后再 json.loads
        3. 均失败返回 []
        """
        # 尝试 1: 直接解析
        try:
            result = json.loads(response)
            if isinstance(result, list) and all(isinstance(item, str) for item in result):
                return result
        except json.JSONDecodeError:
            pass

        # 尝试 2: 正则提取 JSON 数组
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list) and all(isinstance(item, str) for item in result):
                    return result
            except json.JSONDecodeError:
                pass

        # 尝试 3: 按行解析（有些 LLM 会输出带编号的列表）
        lines = response.strip().split("\n")
        items = []
        for line in lines:
            line = line.strip()
            # 移除前导编号: "1. ", "1) ", "- ", "• "
            for prefix in ["- ", "• ", "* "]:
                if line.startswith(prefix):
                    line = line[len(prefix):]
                    break
            match = re.match(r'^\d+[\.\)、]\s*', line)
            if match:
                line = line[match.end():]
            # 移除引号
            line = line.strip().strip('"').strip("'").strip()
            if line and len(line) > 3:  # 过滤太短的无效行
                items.append(line)

        if items:
            logger.info(f"从文本中按行解析到 {len(items)} 条查询")
            return items

        logger.warning(f"无法解析 LLM 响应为 JSON 数组: {response[:200]}")
        return []

    @staticmethod
    def _deduplicate_queries(
        queries: List[str], reference_query: str
    ) -> List[str]:
        """
        去重并过滤与参考查询高度相似的变体

        使用简单的规范化比较（小写、去空格）进行去重。
        不引入额外的 NLP 依赖。
        """
        seen = {_normalize(reference_query)}
        result = []

        for q in queries:
            norm = _normalize(q)
            if norm and norm not in seen:
                seen.add(norm)
                result.append(q)

        return result


def _normalize(text: str) -> str:
    """简单文本规范化：小写 + 去空格"""
    return text.lower().replace(" ", "").replace("\n", "")


# ------------------------------------------------------------------
# 全局单例
# ------------------------------------------------------------------
query_rewriter = QueryRewriter()
