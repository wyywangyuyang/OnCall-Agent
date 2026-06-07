"""
evals — RAGAS 评测模块

提供 RAG 召回效果的量化评估能力，覆盖检索质量与生成质量两个维度。

组件:
- dataset_builder: 从运维文档自动生成 QA 评测数据集
- ragas_evaluator: RAGAS 评测引擎，计算检索+生成共 6 项指标
- run_eval: CLI 统一入口脚本
"""
