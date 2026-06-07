"""
RAG 评测数据集构建器

从 aiops-docs/ 目录下的运维 Markdown 文档自动生成 QA 对，
利用 LLM 提取文档中的关键知识点，生成 question + ground_truth_answer。

输出格式: JSON 数组，持久化到 evals/datasets/ 目录。
支持手动修改 JSON 文件补充或修正 QA 对。

使用方式:
    python -m evals.dataset_builder
    python -m evals.dataset_builder --docs-dir aiops-docs --output evals/datasets/custom.json
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from loguru import logger
from openai import OpenAI

from app.config import config


class DatasetBuilder:
    """从运维文档自动生成 RAG 评测数据集"""

    def __init__(self, docs_dir: str = "aiops-docs", output_dir: str = "evals/datasets"):
        """
        Args:
            docs_dir:   源文档目录路径
            output_dir: 数据集输出目录
        """
        self.docs_dir = Path(docs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = config.query_rewrite_model
        self._client: OpenAI | None = None

        logger.info(
            f"数据集构建器初始化: docs_dir={self.docs_dir}, "
            f"output_dir={self.output_dir}, model={self.model}"
        )

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        output_filename: str = "aiops_qa.json",
        qa_per_doc: int = 5,
    ) -> List[Dict]:
        """
        构建评测数据集

        Args:
            output_filename: 输出 JSON 文件名
            qa_per_doc:      每篇文档生成的 QA 对数量

        Returns:
            List[Dict]: QA 对列表
        """
        # 1. 扫描文档
        md_files = sorted(self.docs_dir.glob("*.md"))
        if not md_files:
            raise FileNotFoundError(
                f"在 {self.docs_dir} 中未找到任何 .md 文档"
            )

        logger.info(f"找到 {len(md_files)} 篇 Markdown 文档")

        # 2. 逐文档生成 QA 对
        all_qa_pairs: List[Dict] = []
        for file_path in md_files:
            try:
                qa_pairs = self._generate_qa_pairs(file_path, num_pairs=qa_per_doc)
                all_qa_pairs.extend(qa_pairs)
                logger.info(
                    f"✓ {file_path.name}: 生成 {len(qa_pairs)} 组 QA"
                )
            except Exception as e:
                logger.error(f"✗ {file_path.name}: 生成失败 — {e}")

        if not all_qa_pairs:
            raise RuntimeError("未能生成任何 QA 对，请检查文档内容和 LLM 可用性")

        # 3. 保存到文件
        output_path = self.output_dir / output_filename
        output_path.write_text(
            json.dumps(all_qa_pairs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            f"数据集已保存: {output_path} "
            f"({len(all_qa_pairs)} 组 QA, {len(md_files)} 篇文档)"
        )

        # 打印统计
        difficulties = {"easy": 0, "medium": 0, "hard": 0}
        for qa in all_qa_pairs:
            d = qa.get("difficulty", "medium")
            difficulties[d] = difficulties.get(d, 0) + 1
        logger.info(
            f"难度分布: easy={difficulties['easy']}, "
            f"medium={difficulties['medium']}, hard={difficulties['hard']}"
        )

        return all_qa_pairs

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _generate_qa_pairs(
        self, file_path: Path, num_pairs: int = 5
    ) -> List[Dict]:
        """
        对单篇文档生成 QA 对

        Args:
            file_path: 文档文件路径
            num_pairs: 生成的 QA 对数量

        Returns:
            List[Dict]: QA 对列表
        """
        # 读取文档
        content = file_path.read_text(encoding="utf-8")
        if not content or not content.strip():
            logger.warning(f"文档内容为空: {file_path}")
            return []

        # 提取标题（第一个 # 开头的行）
        lines = content.strip().split("\n")
        title = file_path.stem
        for line in lines:
            line = line.strip()
            if line.startswith("# ") and not line.startswith("## "):
                title = line[2:].strip()
                break

        # 限制文档长度（避免超 token）
        max_chars = 6000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...(文档过长，已截断)"

        # 构建提示词
        prompt = self._build_qa_prompt(content, title, num_pairs)

        # 调用 LLM
        response = self._call_llm(prompt, temperature=0.5, max_tokens=4096)
        qa_pairs = self._parse_qa_response(response, file_path.name)

        # 验证和清洗
        valid_pairs = []
        for qa in qa_pairs:
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            if len(q) > 5 and len(a) > 20:  # 过滤太短的无效 QA
                valid_pairs.append({
                    "question": q,
                    "ground_truth_answer": a,
                    "source_doc": file_path.name,
                    "difficulty": qa.get("difficulty", "medium"),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                })

        if len(valid_pairs) < num_pairs:
            logger.warning(
                f"仅生成 {len(valid_pairs)}/{num_pairs} 组有效 QA: {file_path.name}"
            )

        return valid_pairs

    def _get_client(self) -> OpenAI:
        """获取 OpenAI 客户端（懒初始化）"""
        if self._client is None:
            self._client = OpenAI(
                api_key=config.dashscope_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        return self._client

    def _call_llm(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """调用 LLM 并返回文本响应"""
        client = self._get_client()

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if not response.choices:
            raise RuntimeError("LLM 返回空的 choices 列表")

        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # 提示词 & 解析
    # ------------------------------------------------------------------

    @staticmethod
    def _build_qa_prompt(content: str, title: str, num_pairs: int) -> str:
        """构建 QA 生成提示词"""
        return f"""你是一个 RAG 评测数据集生成助手。请阅读以下运维文档，生成 {num_pairs} 组高质量的问答对。

文档标题: {title}

文档内容:
{content}

要求:
1. 问题应覆盖文档中的关键知识点，从不同角度提问（原因分析、排查步骤、解决方案、配置参数、工具使用等）
2. 问题应模拟真实的运维人员提问方式（自然语言，可能包含口语或省略表达）
3. 答案应准确、完整，可直接从文档中找到依据
4. 问题难度分布: 简单约40%（文档中直接有答案的事实类问题）、中等约40%（需要综合多处信息的理解类问题）、困难约20%（需要推理或跨段落关联的分析类问题）

请严格按 JSON 数组格式输出，不要输出任何解释:
[
  {{"question": "问题文本", "answer": "答案文本", "difficulty": "easy"}},
  {{"question": "问题文本", "answer": "答案文本", "difficulty": "medium"}},
  ...
]

JSON 输出:"""

    @staticmethod
    def _parse_qa_response(
        response: str, source_doc: str
    ) -> List[Dict]:
        """
        从 LLM 响应中解析 QA 对 JSON 数组

        三级降级: json.loads → 正则提取 → 空列表
        """
        if not response:
            return []

        # 尝试 1: 直接解析
        try:
            result = json.loads(response)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # 尝试 2: 正则提取 JSON 数组
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # 尝试 3: 尝试修复常见 JSON 错误（未转义换行、尾部逗号等）
        try:
            # 移除 markdown 代码块标记
            cleaned = response
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            elif "```" in cleaned:
                parts = cleaned.split("```")
                if len(parts) >= 2:
                    cleaned = parts[1]

            result = json.loads(cleaned.strip())
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, IndexError):
            pass

        logger.warning(
            f"无法解析 LLM 响应为 JSON 数组 ({source_doc}): {response[:200]}"
        )
        return []


# ============================================================================
# CLI 入口
# ============================================================================


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="从运维文档自动生成 RAG 评测数据集"
    )
    parser.add_argument(
        "--docs-dir",
        default="aiops-docs",
        help="源文档目录 (默认: aiops-docs)",
    )
    parser.add_argument(
        "--output",
        default="aiops_qa.json",
        help="输出 JSON 文件名 (默认: aiops_qa.json)",
    )
    parser.add_argument(
        "--qa-per-doc",
        type=int,
        default=5,
        help="每篇文档生成的 QA 对数量 (默认: 5)",
    )

    args = parser.parse_args()

    builder = DatasetBuilder(docs_dir=args.docs_dir)
    qa_pairs = builder.build_dataset(
        output_filename=args.output,
        qa_per_doc=args.qa_per_doc,
    )
    print(f"\n✓ 数据集构建完成: {len(qa_pairs)} 组 QA 对")


if __name__ == "__main__":
    main()
