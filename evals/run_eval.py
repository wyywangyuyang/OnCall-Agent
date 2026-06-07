"""
RAGAS 评测 CLI 统一入口

支持三种运行模式:
    python -m evals.run_eval --build-dataset         # 仅构建数据集
    python -m evals.run_eval --eval                  # 仅运行评测
    python -m evals.run_eval --all                   # 构建数据集 + 运行评测

完整参数列表:
    python -m evals.run_eval --help

示例:
    # 构建数据集（每篇文档 5 组 QA）
    python -m evals.run_eval --build-dataset --qa-per-doc 5

    # 使用自定义数据集运行评测
    python -m evals.run_eval --eval --dataset evals/datasets/custom.json

    # 一键构建 + 评测
    python -m evals.run_eval --all --qa-per-doc 6 --report-dir evals/reports/
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def cmd_build_dataset(args: argparse.Namespace) -> int:
    """构建评测数据集"""
    from evals.dataset_builder import DatasetBuilder

    print(f"🔨 构建评测数据集")
    print(f"   文档目录: {args.docs_dir}")
    print(f"   输出文件: {args.output}")
    print(f"   每篇 QA 数: {args.qa_per_doc}")
    print()

    try:
        builder = DatasetBuilder(docs_dir=args.docs_dir)
        qa_pairs = builder.build_dataset(
            output_filename=args.output,
            qa_per_doc=args.qa_per_doc,
        )
        print(f"\n✅ 数据集构建完成: {len(qa_pairs)} 组 QA 对")
        print(f"   文件: evals/datasets/{args.output}")
        return 0
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 构建失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_run_eval(args: argparse.Namespace) -> int:
    """运行 RAGAS 评测"""
    from evals.ragas_evaluator import RagasEvaluator

    print(f"📊 运行 RAGAS 评测")
    print(f"   数据集: {args.dataset}")
    print(f"   报告目录: {args.report_dir}")
    print()

    # 检查数据集文件
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"\n❌ 数据集文件不存在: {args.dataset}")
        print(f"   请先运行: python -m evals.run_eval --build-dataset")
        return 1

    try:
        evaluator = RagasEvaluator(dataset_path=str(dataset_path))
        result = evaluator.evaluate()

        if result.errors:
            print(f"\n⚠️  评测过程中出现 {len(result.errors)} 个错误:")
            for err in result.errors:
                print(f"   - {err}")

        if result.scores:
            print(f"\n📈 评测结果:")
            print(f"   综合得分: {result.overall_score:.4f}")
            dimension_map = {
                "context_precision": "检索·Context Precision",
                "context_recall": "检索·Context Recall",
                "context_relevancy": "检索·Context Relevancy",
                "faithfulness": "生成·Faithfulness",
                "answer_relevancy": "生成·Answer Relevancy",
                "answer_correctness": "生成·Answer Correctness",
            }
            for metric_key, label in dimension_map.items():
                score = result.scores.get(metric_key)
                if score is not None:
                    print(f"   {label}: {score:.4f}")

        # 保存报告
        out_path = Path(args.report_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = out_path / f"eval_{timestamp}.md"

        # 重新构建报告（复用 evaluator 的内部方法）
        report_md = evaluator._build_markdown_report(result)
        report_path.write_text(report_md, encoding="utf-8")

        print(f"\n✅ 评测完成")
        print(f"   报告: {report_path}")
        return 0

    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 评测失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main() -> int:
    """CLI 主入口"""
    parser = argparse.ArgumentParser(
        description="RAGAS 评测 — RAG 召回效果量化评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m evals.run_eval --build-dataset                    # 构建数据集
  python -m evals.run_eval --eval                             # 运行评测
  python -m evals.run_eval --all                              # 构建 + 评测
  python -m evals.run_eval --eval --dataset custom.json       # 指定数据集
        """,
    )

    # 运行模式（互斥组）
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build-dataset",
        action="store_true",
        help="构建评测数据集（从文档生成 QA 对）",
    )
    mode.add_argument(
        "--eval",
        action="store_true",
        help="运行 RAGAS 评测",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="构建数据集 + 运行评测",
    )

    # 构建数据集参数
    parser.add_argument(
        "--docs-dir",
        default="aiops-docs",
        help="源文档目录 (默认: aiops-docs)",
    )
    parser.add_argument(
        "--output",
        default="aiops_qa.json",
        help="数据集输出文件名 (默认: aiops_qa.json)",
    )
    parser.add_argument(
        "--qa-per-doc",
        type=int,
        default=5,
        help="每篇文档生成的 QA 对数量 (默认: 5)",
    )

    # 评测参数
    parser.add_argument(
        "--dataset",
        default="evals/datasets/aiops_qa.json",
        help="评测数据集 JSON 文件路径 (默认: evals/datasets/aiops_qa.json)",
    )
    parser.add_argument(
        "--report-dir",
        default="evals/reports",
        help="评测报告输出目录 (默认: evals/reports)",
    )

    args = parser.parse_args()

    # 默认行为: 如果没有指定任何模式，打印帮助
    if not args.build_dataset and not args.eval and not args.all:
        parser.print_help()
        return 0

    # 执行
    if args.all:
        # 先构建再评测
        ret = cmd_build_dataset(args)
        if ret != 0:
            return ret
        print("\n" + "=" * 60 + "\n")
        # --all 模式下，构建的输出文件就是评测的输入
        args.dataset = f"evals/datasets/{args.output}"
        return cmd_run_eval(args)
    elif args.build_dataset:
        return cmd_build_dataset(args)
    elif args.eval:
        return cmd_run_eval(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
