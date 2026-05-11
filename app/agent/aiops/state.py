"""
通过 Plan-Execute-Replan 状态定义
基于 LangGraph 官方教程实现
"""
from typing import TypedDict, List, Annotated
import operator

class PlanExecuteState(TypedDict):
    """
    Plan-Execute-Replan 状态定义
    """
    input: str  # 用户输入（任务描述）
    plan: List[str] # 执行计划（步骤列表）
    # past_steps使用Annotated[List[tuple], operator.add]声明，LangGraph会将每次节点返回的past_steps自动追加到列表，而不是覆盖，无需手动维护历史
    # 使用 operator.add 实现追加式更新（而非覆盖）
    past_steps: Annotated[List[tuple], operator.add]    # 已使用的步骤历史
    response: str   # 最终响应/报告