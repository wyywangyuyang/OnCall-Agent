"""
RAGAS 评测引擎

基于 RAGAS 框架对 RAG 召回效果进行量化评估，覆盖:
  - 检索质量: Context Precision, Context Recall, Context Relevancy
  - 生成质量: Faithfulness, Answer Relevancy, Answer Correctness

使用方式:
    from evals.ragas_evaluator import RagasEvaluator

    evaluator = RagasEvaluator("evals/datasets/aiops_qa.json")
    result = evaluator.evaluate()
    report_path = evaluator.save_report("evals/reports")
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import config


class EvalResult:
    """评测结果容器"""

    def __init__(self):
        self.scores: Dict[str, float] = {}      # 各指标分数
        self.per_question: List[Dict] = []      # 逐题明细
        self.dataset_path: str = ""
        self.dataset_size: int = 0
        self.eval_time: str = ""
        self.rag_config: Dict = {}
        self.errors: List[str] = []

    @property
    def overall_score(self) -> float:
        """综合评分: 所有指标的算术平均"""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def to_dict(self) -> Dict:
        return {
            "scores": self.scores,
            "overall_score": round(self.overall_score, 4),
            "per_question": self.per_question,
            "dataset_path": self.dataset_path,
            "dataset_size": self.dataset_size,
            "eval_time": self.eval_time,
            "rag_config": self.rag_config,
            "errors": self.errors,
        }


class RagasEvaluator:
    """RAGAS 评测引擎"""

    # 全部 6 个评测指标
    METRIC_NAMES = [
        "context_precision",
        "context_recall",
        "context_relevancy",
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
    ]

    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path: 评测数据集 JSON 文件路径
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

        self._judge_llm = None
        self._judge_embeddings = None
        self._metrics = None
        self._generation_model = None

        logger.info(f"RAGAS 评测器初始化: dataset={dataset_path}")

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def evaluate(self) -> EvalResult:
        """
        执行完整评测流程

        1. 加载数据集
        2. 初始化 RAGAS 指标和 judge model
        3. 逐题: 检索 + 生成 → 收集样本
        4. 调用 ragas.evaluate() 计算指标
        5. 构建 EvalResult

        Returns:
            EvalResult: 包含所有分数和逐题明细的评测结果
        """
        result = EvalResult()
        result.eval_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result.dataset_path = str(self.dataset_path)
        result.rag_config = self._capture_rag_config()

        # ---- Step 1: 加载数据集 ----
        dataset = self._load_dataset()
        result.dataset_size = len(dataset)
        logger.info(f"已加载 {len(dataset)} 组 QA 对")

        # ---- Step 2: 检查依赖 ----
        if not self._check_ragas():
            result.errors.append("ragas 库不可用，无法进行评测")
            return result

        # ---- Step 3: 初始化 RAGAS ----
        try:
            self._init_ragas()
        except Exception as e:
            result.errors.append(f"RAGAS 初始化失败: {e}")
            logger.error(f"RAGAS 初始化失败: {e}")
            return result

        # ---- Step 4: 逐题检索 + 生成 → 构建样本 ----
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

        samples: List[SingleTurnSample] = []
        for i, qa in enumerate(dataset):
            question = qa["question"]
            reference = qa["ground_truth_answer"]

            logger.info(f"[{i+1}/{len(dataset)}] 评测: {question[:60]}...")

            try:
                # 检索
                contexts = self._retrieve_contexts(question)
                logger.debug(f"  检索到 {len(contexts)} 个上下文片段")

                # 生成
                response = self._generate_answer(question, contexts)
                logger.debug(f"  生成答案: {response[:80]}...")

                # 收集
                samples.append(SingleTurnSample(
                    user_input=question,
                    retrieved_contexts=contexts,
                    response=response,
                    reference=reference,
                ))

                # 记录逐题信息（此时尚无逐题分数）
                result.per_question.append({
                    "index": i + 1,
                    "question": question,
                    "source_doc": qa.get("source_doc", ""),
                    "difficulty": qa.get("difficulty", "medium"),
                    "num_contexts": len(contexts),
                    "response_preview": response[:150],
                })

            except Exception as e:
                error_msg = f"[{i+1}] 评测失败: {e}"
                logger.warning(error_msg)
                result.errors.append(error_msg)
                result.per_question.append({
                    "index": i + 1,
                    "question": question,
                    "error": str(e),
                })

        if not samples:
            result.errors.append("所有样本评测均失败，无法生成结果")
            return result

        # ---- Step 5: 调用 RAGAS evaluate ----
        eval_dataset = EvaluationDataset(samples=samples)
        try:
            from ragas import evaluate

            ragas_result = evaluate(
                dataset=eval_dataset,
                metrics=self._metrics,
                llm=self._judge_llm,
                embeddings=self._judge_embeddings,
            )
        except Exception as e:
            result.errors.append(f"ragas.evaluate() 调用失败: {e}")
            logger.error(f"ragas.evaluate() 失败: {e}")
            return result

        # ---- Step 6: 提取分数 ----
        result.scores = self._extract_scores(ragas_result)
        result.per_question = self._attach_per_question_scores(
            result.per_question, ragas_result
        )

        logger.info(
            f"评测完成 | 综合得分: {result.overall_score:.4f} | "
            + " | ".join(
                f"{k}: {v:.3f}" for k, v in result.scores.items()
            )
        )

        return result

    def save_report(self, output_dir: str = "evals/reports") -> str:
        """
        生成并保存 Markdown 评测报告

        Args:
            output_dir: 报告输出目录

        Returns:
            str: 报告文件路径
        """
        # 先执行评测
        result = self.evaluate()

        # 生成报告
        report_md = self._build_markdown_report(result)

        # 保存
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = out_path / f"eval_{timestamp}.md"
        report_path.write_text(report_md, encoding="utf-8")

        logger.info(f"评测报告已保存: {report_path}")
        return str(report_path)

    # ------------------------------------------------------------------
    # 检索与生成
    # ------------------------------------------------------------------

    def _retrieve_contexts(self, question: str) -> List[str]:
        """
        调用 RAG 检索管线，返回上下文文本列表

        直接复用 knowledge_tool 的 _retrieve_single_query，
        保证评测结果与线上检索行为完全一致。
        """
        try:
            from app.tools.knowledge_tool import _retrieve_single_query

            docs = _retrieve_single_query(question)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    def _generate_answer(
        self, question: str, contexts: List[str]
    ) -> str:
        """
        基于检索上下文用 LLM 生成答案

        使用与主服务一致的模型调用方式（langchain_qwq.ChatQwen）。
        """
        if self._generation_model is None:
            from langchain_qwq import ChatQwen

            self._generation_model = ChatQwen(
                model=config.dashscope_model,
                api_key=config.dashscope_api_key,
                temperature=0.3,
            )

        if not contexts:
            return "未找到相关参考资料，无法回答。"

        # 构建提示词
        context_text = "\n\n".join(
            f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)
        )
        prompt = (
            "你是一个专业的运维助手。请基于以下参考资料回答问题。\n"
            "只使用参考资料中的信息，不要编造。\n"
            "如果参考资料不足以回答问题，请明确说明。\n\n"
            f"参考资料:\n{context_text}\n\n"
            f"问题: {question}\n\n"
            "回答:"
        )

        try:
            from langchain_core.messages import HumanMessage

            response = self._generation_model.invoke([HumanMessage(content=prompt)])
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return f"(生成失败: {e})"

    # ------------------------------------------------------------------
    # RAGAS 初始化
    # ------------------------------------------------------------------

    @staticmethod
    def _check_ragas() -> bool:
        """检查 ragas 库是否可用"""
        try:
            import ragas  # noqa: F401
            return True
        except ImportError:
            logger.error(
                "ragas 库未安装。请运行: pip install ragas>=0.3.1"
            )
            return False

    def _init_ragas(self) -> None:
        """初始化 RAGAS judge LLM、嵌入和评测指标"""
        # Judge LLM
        from langchain_qwq import ChatQwen
        from ragas.llms import LangchainLLMWrapper

        self._judge_llm = LangchainLLMWrapper(
            ChatQwen(
                model=config.dashscope_model,
                api_key=config.dashscope_api_key,
                temperature=0,  # 评测用低温度保证一致性
            )
        )

        # Judge Embeddings
        from app.services.vector_embedding_service import vector_embedding_service
        from ragas.embeddings import LangchainEmbeddingsWrapper

        self._judge_embeddings = LangchainEmbeddingsWrapper(
            vector_embedding_service
        )

        # 评测指标
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            context_relevancy,
            faithfulness,
        )

        self._metrics = [
            context_precision,
            context_recall,
            context_relevancy,
            faithfulness,
            answer_relevancy,
            answer_correctness,
        ]

        logger.info(
            f"RAGAS 初始化完成 | "
            f"judge_model={config.dashscope_model}, "
            f"metrics={len(self._metrics)}"
        )

    # ------------------------------------------------------------------
    # 结果提取
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_scores(ragas_result) -> Dict[str, float]:
        """从 ragas EvaluationResult 中提取各指标平均分"""
        scores = {}
        try:
            result_df = ragas_result.to_pandas()
            for col in result_df.columns:
                if col in RagasEvaluator.METRIC_NAMES:
                    series = result_df[col].dropna()
                    if len(series) > 0:
                        scores[col] = round(float(series.mean()), 4)
        except Exception as e:
            logger.error(f"提取分数失败: {e}")
        return scores

    @staticmethod
    def _attach_per_question_scores(
        per_question: List[Dict], ragas_result
    ) -> List[Dict]:
        """将 ragas 逐行分数附加到逐题明细"""
        try:
            df = ragas_result.to_pandas()
            for i, row in df.iterrows():
                if i < len(per_question):
                    for col in df.columns:
                        if col in RagasEvaluator.METRIC_NAMES:
                            val = row[col]
                            per_question[i][col] = (
                                round(float(val), 4)
                                if val is not None and not (
                                    isinstance(val, float) and val != val  # NaN check
                                )
                                else None
                            )
        except Exception as e:
            logger.error(f"附加逐题分数失败: {e}")
        return per_question

    # ------------------------------------------------------------------
    # 报告生成
    # ------------------------------------------------------------------

    def _build_markdown_report(self, result: EvalResult) -> str:
        """构建 Markdown 格式的评测报告"""
        lines = [
            "# RAG 评测报告",
            "",
            f"- **评测时间**: {result.eval_time}",
            f"- **数据集**: {Path(result.dataset_path).name} ({result.dataset_size} 条 QA)",
            f"- **Judge Model**: {config.dashscope_model}",
            f"- **Embedding Model**: {config.dashscope_embedding_model}",
            f"- **RAG 配置**: "
            f"bm25_enabled={config.bm25_enabled}, "
            f"rerank_enabled={config.rerank_enabled}, "
            f"query_rewrite_enabled={config.query_rewrite_enabled}, "
            f"multi_query_expansion_enabled={config.multi_query_expansion_enabled}",
            "",
        ]

        # 错误信息
        if result.errors:
            lines.append("## ⚠️ 错误")
            lines.append("")
            for err in result.errors:
                lines.append(f"- {err}")
            lines.append("")

        if not result.scores:
            lines.append("## 无有效结果")
            lines.append("")
            lines.append("评测未能生成有效结果，请检查上述错误。")
            return "\n".join(lines)

        # ---- 综合评分表 ----
        lines.append("## 综合评分")
        lines.append("")
        lines.append("| 维度 | 指标 | 分数 |")
        lines.append("|------|------|------|")

        dimension_map = {
            "context_precision": ("检索质量", "Context Precision"),
            "context_recall": ("检索质量", "Context Recall"),
            "context_relevancy": ("检索质量", "Context Relevancy"),
            "faithfulness": ("生成质量", "Faithfulness"),
            "answer_relevancy": ("生成质量", "Answer Relevancy"),
            "answer_correctness": ("生成质量", "Answer Correctness"),
        }

        for metric_key, (dim, label) in dimension_map.items():
            score = result.scores.get(metric_key)
            if score is not None:
                lines.append(f"| {dim} | {label} | {score:.4f} |")

        lines.append(f"| **总分** | | **{result.overall_score:.4f}** |")
        lines.append("")

        # ---- 各难度得分 ----
        diff_scores: Dict[str, List[float]] = {}
        for pq in result.per_question:
            if "error" not in pq:
                diff = pq.get("difficulty", "medium")
                pq_scores = [
                    pq.get(k) for k in self.METRIC_NAMES
                    if pq.get(k) is not None
                ]
                if pq_scores:
                    diff_scores.setdefault(diff, []).append(
                        sum(pq_scores) / len(pq_scores)
                    )

        if diff_scores:
            lines.append("## 各难度得分")
            lines.append("")
            lines.append("| 难度 | 数量 | 平均分 |")
            lines.append("|------|------|--------|")
            for diff in ["easy", "medium", "hard"]:
                scores_list = diff_scores.get(diff, [])
                if scores_list:
                    avg = sum(scores_list) / len(scores_list)
                    lines.append(
                        f"| {diff} | {len(scores_list)} | {avg:.4f} |"
                    )
            lines.append("")

        # ---- 逐题明细 ----
        lines.append("## 逐题明细")
        lines.append("")
        header_cols = ["#", "问题", "难度", "CP", "CR", "CRel", "Faith", "AR", "AC"]
        lines.append("| " + " | ".join(header_cols) + " |")

        sep = ["---"] * len(header_cols)
        lines.append("| " + " | ".join(sep) + " |")

        metric_keys_short = [
            "context_precision",
            "context_recall",
            "context_relevancy",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
        ]

        for pq in result.per_question:
            if "error" in pq:
                lines.append(
                    f"| {pq['index']} | {pq['question'][:40]}... | ❌ {pq['error'][:50]} "
                    + "| - | - | - | - | - | - |"
                )
            else:
                values = [
                    f"{pq.get(k, 0):.3f}" if pq.get(k) is not None else "-"
                    for k in metric_keys_short
                ]
                row = [
                    str(pq["index"]),
                    pq["question"][:50] + ("..." if len(pq["question"]) > 50 else ""),
                    pq.get("difficulty", "-"),
                    *values,
                ]
                lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        lines.append("---")
        lines.append(f"*报告由 RAGAS 评测系统自动生成于 {result.eval_time}*")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _load_dataset(self) -> List[Dict]:
        """加载 JSON 数据集"""
        with open(self.dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("数据集必须是 JSON 数组格式")
        return data

    @staticmethod
    def _capture_rag_config() -> Dict:
        """捕获当前 RAG 管线配置"""
        return {
            "bm25_enabled": config.bm25_enabled,
            "bm25_top_k": config.bm25_top_k,
            "rerank_enabled": config.rerank_enabled,
            "rerank_model": config.rerank_model,
            "rerank_retrieve_top_k": config.rerank_retrieve_top_k,
            "rag_top_k": config.rag_top_k,
            "vector_recall_top_k": config.vector_recall_top_k,
            "rrf_k": config.rrf_k,
            "query_rewrite_enabled": config.query_rewrite_enabled,
            "multi_query_expansion_enabled": config.multi_query_expansion_enabled,
            "multi_query_expansion_count": config.multi_query_expansion_count,
            "embedding_model": config.dashscope_embedding_model,
        }


# ============================================================================
# CLI 入口
# ============================================================================


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="RAGAS 评测 — RAG 召回效果量化评估"
    )
    parser.add_argument(
        "--dataset",
        default="evals/datasets/aiops_qa.json",
        help="评测数据集 JSON 文件路径",
    )
    parser.add_argument(
        "--report-dir",
        default="evals/reports",
        help="报告输出目录",
    )

    args = parser.parse_args()

    evaluator = RagasEvaluator(dataset_path=args.dataset)
    report_path = evaluator.save_report(output_dir=args.report_dir)

    print(f"\n✓ 评测完成")
    print(f"  报告: {report_path}")


if __name__ == "__main__":
    main()
