"""
AIOPS Agent 通用工具函数
"""
from typing import List


def format_tools_description(tools: List) -> str:
    """
    格式化工具列表为描述文本
    """
    tool_descriptions = []
    for tool in tools:
        # 获取工具名称和描述     hasattr 是 Python 的内置函数，用于检查对象是否具有指定的属性或方法。
        if hasattr(tool, "name") and hasattr(tool, "description"):
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
    return "\n".join(tool_descriptions)