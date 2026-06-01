"""
日志配置模块
使用 Loguru 库进行日志记录和管理
"""
import io
import sys
import loguru
from loguru import logger
from app.config import config


# ============================================================================
# 防御性补丁：修复 Loguru 在遇到包含 {0} 等占位符但缺少对应参数时崩溃的问题
# 某些第三方库（LangChain/LangGraph）可能会调用 logger.exception() 或带
# keyword 参数的日志方法，导致 message.format(*args, **kwargs) 抛出 IndexError
# ============================================================================
_original_log = loguru._logger.Logger._log


def _patched_log(self, level, from_decorator, options, message, args, kwargs):
    """包装 _log 方法，捕获 format 异常防止应用崩溃"""
    try:
        _original_log(self, level, from_decorator, options, message, args, kwargs)
    except (IndexError, KeyError, ValueError) as e:
        # 格式化失败时，回退：将 message 和 args 转为安全字符串重新记录
        try:
            safe_message = str(message)
            if args:
                safe_message += " | args: " + str(args)
            if kwargs:
                safe_message += " | kwargs: " + str(kwargs)
            _original_log(
                self, level, from_decorator, options,
                safe_message, (), {}
            )
        except Exception:
            # 最终降级：什么都不做，避免日志系统导致应用崩溃
            pass


loguru._logger.Logger._log = _patched_log


def _ensure_utf8_stdout():
    """
    确保 stdout 使用 UTF-8 编码，避免 emoji 等 Unicode 字符在 Windows GBK 终端报错。
    start-windows.bat 已设置 chcp 65001，但 Python 的 sys.stdout.encoding 可能仍是 GBK。
    """
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )


def setup_logger():
    """
    配置日志系统
    按照 Loguru 最佳实践配置全局 logger：
    1、修复 stdout 编码（Windows GBK → UTF-8）
    2、移除默认处理器
    3、添加控制台输出（带颜色）
    4、添加文件输出（按天轮转，自动压缩，异步写入）
    """

    # 修复 stdout 编码：Windows 下 Python 的 sys.stdout.encoding 可能是 gbk，
    # 导致 emoji（如 🚀）写入时抛出 UnicodeEncodeError
    _ensure_utf8_stdout()

    # 移除默认处理器
    logger.remove()

    # 添加控制台输出（带颜色）
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>.<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        level="DEBUG" if config.debug else "INFO",
        colorize=True,
        backtrace=True,  # 显示完整异常栈信息
        diagnose=config.debug # Debug 模式下显示变量值
    )

    # 添加文件输出（按天轮转，自动压缩，异步写入）
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        rotation="00:00",  # 每天0点自动切割新日志文件
        retention="7 days",  # 仅保留最近7天的日志
        compression="zip",  # 过期日志自动压缩为zip
        encoding="utf-8",  # 解决中文乱码
        enqueue=True,  # 异步写入，提升性能（避免IO阻塞）
        backtrace=True,  # 显示完整异常栈信息
        diagnose=True,  # 显示变量值，便于调试
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}.{function}:{line} | {message}",
    )

setup_logger()