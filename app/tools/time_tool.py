"""
时间工具 - 获取当前时间信息
"""
from datetime import datetime
from zoneinfo import ZoneInfo
from loguru import logger
from langchain_core.tools import tool

@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """
    获取当前时间

    当用户询问"现在几点"、"今天星期几"、"今天日期"等时间相关问题时，使用此工具。

    Args:
        timezone: 时区，默认为 Asia/Shanghai（北京时间）

    Returns:
        str: 格式化的当前时间信息
    """
    try:
        # 获取指定时区的当前时间
        tz= ZoneInfo(timezone)
        now = datetime.now(tz)

        # 返回格式化的时间和日期
        return now.strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        # 如果发生异常，返回错误信息
        logger.error(f"时间查询工具调用失败：{e}")
        return f"获取当前时间时出错：{str(e)}"