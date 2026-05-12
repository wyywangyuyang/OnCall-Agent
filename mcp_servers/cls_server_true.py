"""
腾讯云 CLS (Cloud Log Service) MCP Server - 真实实现

对接真实的腾讯云 CLS 日志服务，提供日志查询、检索和分析功能。
使用腾讯云官方 SDK 进行 API 调用。
"""

import logging
import functools
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from fastmcp import FastMCP
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.cls.v20201016 import cls_client, models

# 配置日志
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler(os.path.join(log_dir, 'mcp_cls_true.log'), encoding='utf-8'),
    ]
)
logger = logging.getLogger("CLS_MCP_Server_True")

mcp = FastMCP("CLS-True")


def log_tool_call(func):
    """装饰器：记录工具调用的日志，包括方法名、参数和返回状态"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__

        # 记录调用信息
        logger.info(f"=" * 80)
        logger.info(f"调用方法: {method_name}")

        # 记录参数（排除self等）
        if kwargs:
            try:
                params_str = json.dumps(kwargs, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                params_str = str(kwargs)
            logger.info(f"参数信息:\n{params_str}")
        else:
            logger.info("参数信息: 无")

        # 执行方法
        try:
            result = func(*args, **kwargs)

            # 记录返回状态
            logger.info(f"返回状态: SUCCESS")

            # 记录返回结果摘要（避免日志过长）
            if isinstance(result, dict):
                summary = {k: v if not isinstance(v, (list, dict)) else f"<{type(v).__name__} with {len(v)} items>"
                           for k, v in list(result.items())[:5]}
                logger.info(f"返回结果摘要: {json.dumps(summary, ensure_ascii=False)}")
            else:
                logger.info(f"返回结果: {result}")

            logger.info(f"=" * 80)
            return result

        except Exception as e:
            # 记录错误状态
            logger.error(f"返回状态: ERROR")
            logger.error(f"错误信息: {str(e)}")
            logger.error(f"=" * 80)
            raise

    return wrapper


def get_cls_client(region: str = "ap-beijing"):
    """获取腾讯云 CLS 客户端实例。

    Args:
        region: 地区代码

    Returns:
        cls_client.ClsClient: CLS 客户端实例
    """
    from app.config import config

    secret_id = config.tencent_cloud_secret_id
    secret_key = config.tencent_cloud_secret_key

    if not secret_id or not secret_key:
        raise ValueError("腾讯云凭证未配置，请在 .env 文件中设置 TENCENT_CLOUD_SECRET_ID 和 TENCENT_CLOUD_SECRET_KEY")

    cred = credential.Credential(secret_id, secret_key)
    httpProfile = HttpProfile()
    httpProfile.endpoint = "cls.tencentcloudapi.com"

    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile
    client = cls_client.ClsClient(cred, region, clientProfile)

    return client


def parse_time_or_default(time_str: Optional[str], default_offset_hours: int = 0) -> datetime:
    """解析时间字符串或返回默认时间。

    Args:
        time_str: 时间字符串（格式：YYYY-MM-DD HH:MM:SS）
        default_offset_hours: 默认时间偏移（小时）

    Returns:
        datetime: 解析后的时间对象
    """
    if time_str:
        try:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return datetime.now() + timedelta(hours=default_offset_hours)


def generate_time_series(base_time: datetime, minutes_offset: int) -> str:
    """生成基于基准时间的时间字符串。

    Args:
        base_time: 基准时间
        minutes_offset: 分钟偏移量

    Returns:
        str: 格式化的时间字符串
    """
    result_time = base_time + timedelta(minutes=minutes_offset)
    return result_time.strftime("%Y-%m-%d %H:%M:%S")


@mcp.tool()
@log_tool_call
def get_current_timestamp() -> int:
    """获取当前时间戳（以毫秒为单位）。

    此工具用于获取标准的毫秒时间戳，可用于：
    1. 作为 search_log 的 end_time 参数（查询到现在）
    2. 计算历史时间点作为 start_time 参数

    Returns:
        int: 当前时间戳（毫秒），例如: 1708012345000

    使用示例:
        # 获取当前时间
        current = get_current_timestamp()

        # 计算15分钟前的时间
        fifteen_min_ago = current - (15 * 60 * 1000)

        # 计算1小时前的时间
        one_hour_ago = current - (60 * 60 * 1000)

        # 用于搜索最近15分钟的日志
        search_log(
            topic_id="topic-001",
            start_time=fifteen_min_ago,
            end_time=current
        )
    """
    return int(datetime.now().timestamp() * 1000)


@mcp.tool()
@log_tool_call
def get_region_code_by_name(region_name: str) -> Dict[str, Any]:
    """根据地区名称搜索对应的地区参数。

    Args:
        region_name: 地区名称（如：北京、上海、广州等）

    Returns:
        Dict: 包含地区代码和相关信息的字典
            - region_code: 地区代码
            - region_name: 地区名称
            - available: 是否可用
    """
    # 腾讯云 CLS 支持的地区映射表
    region_mapping = {
        "北京": {"region_code": "ap-beijing", "region_name": "北京", "available": True},
        "上海": {"region_code": "ap-shanghai", "region_name": "上海", "available": True},
        "广州": {"region_code": "ap-guangzhou", "region_name": "广州", "available": True},
        "成都": {"region_code": "ap-chengdu", "region_name": "成都", "available": True},
        "重庆": {"region_code": "ap-chongqing", "region_name": "重庆", "available": True},
        "南京": {"region_code": "ap-nanjing", "region_name": "南京", "available": True},
        "深圳金融": {"region_code": "ap-shenzhen-fsdr", "region_name": "深圳金融", "available": True},
        "上海金融": {"region_code": "ap-shanghai-fsi", "region_name": "上海金融", "available": True},
        "香港": {"region_code": "ap-hongkong", "region_name": "香港", "available": True},
        "新加坡": {"region_code": "ap-singapore", "region_name": "新加坡", "available": True},
        "孟买": {"region_code": "ap-mumbai", "region_name": "孟买", "available": True},
        "东京": {"region_code": "ap-tokyo", "region_name": "东京", "available": True},
        "首尔": {"region_code": "ap-seoul", "region_name": "首尔", "available": True},
        "硅谷": {"region_code": "na-siliconvalley", "region_name": "硅谷", "available": True},
        "弗吉尼亚": {"region_code": "na-ashburn", "region_name": "弗吉尼亚", "available": True},
        "法兰克福": {"region_code": "eu-frankfurt", "region_name": "法兰克福", "available": True},
        "莫斯科": {"region_code": "eu-moscow", "region_name": "莫斯科", "available": True},
    }

    result = region_mapping.get(region_name)
    if result:
        return result
    else:
        return {
            "region_code": None,
            "region_name": region_name,
            "available": False,
            "error": f"未找到地区: {region_name}"
        }


@mcp.tool()
@log_tool_call
def get_topic_info_by_name(topic_name: str, region_code: Optional[str] = None) -> Dict[str, Any]:
    """根据主题名称搜索相关的主题信息。

    Args:
        topic_name: 主题名称
        region_code: 地区代码（可选）

    Returns:
        Dict: 包含主题信息的字典
            - topic_id: 主题ID
            - topic_name: 主题名称
            - region_code: 所属地区
            - create_time: 创建时间
            - log_count: 日志数量
    """
    from app.config import config

    region = region_code or config.tencent_cloud_region

    try:
        client = get_cls_client(region)
        req = models.DescribeTopicsRequest()

        params = {
            "TopicNames": [topic_name]
        }
        req.from_json_string(json.dumps(params))

        resp = client.DescribeTopics(req)

        topics = json.loads(resp.to_json_string()).get("Topics", [])

        if topics:
            topic = topics[0]
            return {
                "topic_id": topic.get("TopicId"),
                "topic_name": topic.get("TopicName"),
                "service_name": topic.get("Description", ""),
                "region_code": region,
                "create_time": topic.get("CreateTime", ""),
                "log_count": 0,
                "description": topic.get("Description", "")
            }
        else:
            return {
                "topic_id": None,
                "topic_name": topic_name,
                "region_code": region,
                "error": f"未找到主题: {topic_name}"
            }

    except Exception as e:
        logger.error(f"查询主题信息失败: {str(e)}")
        return {
            "topic_id": None,
            "topic_name": topic_name,
            "region_code": region,
            "error": f"查询主题失败: {str(e)}"
        }


@mcp.tool()
@log_tool_call
def search_topic_by_service_name(
        service_name: str,
        region_code: Optional[str] = None,
        fuzzy: bool = True
) -> Dict[str, Any]:
    """根据服务名称搜索相关的日志主题信息，支持模糊搜索。

    此工具用于根据服务名称查找对应的日志主题（topic），便于后续进行日志查询。

    Args:
        service_name: 服务名称（必填）
            示例: "data-sync-service", "sync", "data-sync"
            说明: 当 fuzzy=True 时，支持部分匹配

        region_code: 地区代码（可选）
            示例: "ap-beijing", "ap-shanghai"
            说明: 如果指定，只返回该地区的主题

        fuzzy: 是否启用模糊搜索（可选，默认 True）
            True: 部分匹配，例如 "sync" 可以匹配 "data-sync-service"
            False: 精确匹配，必须完全一致

    Returns:
        Dict: 搜索结果
            - total: 匹配到的主题数量
            - topics: 主题列表，每个主题包含:
                * topic_id: 主题ID（用于后续日志查询）
                * topic_name: 主题名称
                * service_name: 服务名称
                * region_code: 所属地区
                * create_time: 创建时间
                * log_count: 日志数量
                * description: 主题描述
            - query: 查询条件

    使用示例:
        # 示例1: 模糊搜索（推荐）
        search_topic_by_service_name(service_name="data-sync")
        # 可以匹配: "data-sync-service", "data-sync-worker" 等

        # 示例2: 精确搜索
        search_topic_by_service_name(
            service_name="data-sync-service",
            fuzzy=False
        )

        # 示例3: 指定地区搜索
        search_topic_by_service_name(
            service_name="sync",
            region_code="ap-beijing"
        )

        # 示例4: 查找后进行日志搜索的完整流程
        # 步骤1: 根据服务名查找 topic
        result = search_topic_by_service_name(service_name="data-sync-service")

        # 步骤2: 获取 topic_id
        topic_id = result["topics"][0]["topic_id"]  # "topic-001"

        # 步骤3: 使用 topic_id 查询日志
        current_ts = get_current_timestamp()
        start_ts = current_ts - (15 * 60 * 1000)
        search_log(
            topic_id=topic_id,
            start_time=start_ts,
            end_time=current_ts
        )
    """
    from app.config import config

    region = region_code or config.tencent_cloud_region

    matched_topics = []

    try:
        client = get_cls_client(region)
        req = models.DescribeTopicsRequest()

        params = {}
        req.from_json_string(json.dumps(params))

        resp = client.DescribeTopics(req)
        all_topics = json.loads(resp.to_json_string()).get("Topics", [])

        for topic in all_topics:
            topic_name = topic.get("TopicName", "")
            description = topic.get("Description", "")

            # 检查是否匹配服务名称
            is_match = False
            if fuzzy:
                # 模糊匹配：在主题名称或描述中查找
                if (service_name.lower() in topic_name.lower() or
                        service_name.lower() in description.lower() or
                        topic_name.lower() in service_name.lower()):
                    is_match = True
            else:
                # 精确匹配
                if topic_name == service_name or description == service_name:
                    is_match = True

            if is_match:
                matched_topics.append({
                    "topic_id": topic.get("TopicId"),
                    "topic_name": topic_name,
                    "service_name": description,
                    "region_code": region,
                    "create_time": topic.get("CreateTime", ""),
                    "log_count": 0,
                    "description": description
                })

        return {
            "total": len(matched_topics),
            "topics": matched_topics,
            "query": {
                "service_name": service_name,
                "region_code": region,
                "fuzzy": fuzzy
            },
            "message": f"找到 {len(matched_topics)} 个匹配的日志主题" if matched_topics else f"未找到服务 '{service_name}' 的日志主题"
        }

    except Exception as e:
        logger.error(f"搜索主题失败: {str(e)}")
        return {
            "total": 0,
            "topics": [],
            "query": {
                "service_name": service_name,
                "region_code": region,
                "fuzzy": fuzzy
            },
            "message": f"搜索失败: {str(e)}"
        }


@mcp.tool()
@log_tool_call
def search_log(
        topic_id: str,
        start_time: int,
        end_time: int,
        query: Optional[str] = None,
        limit: int = 100
) -> Dict[str, Any]:
    """基于提供的查询参数搜索日志。

    Args:
        topic_id: 主题ID（必填）
            示例: "topic-001"

        start_time: 开始时间戳，单位为毫秒（必填，int类型）
            重要: 必须传递整数类型的毫秒时间戳
            获取方式:
            1. 使用 get_current_timestamp() 工具获取当前时间戳
            2. 计算历史时间: current_timestamp - (分钟数 * 60 * 1000)
            示例:
            - 当前时间: 1708012345000
            - 15分钟前: 1708012345000 - (15 * 60 * 1000) = 1708011445000
            - 1小时前: 1708012345000 - (60 * 60 * 1000) = 1708008745000

        end_time: 结束时间戳，单位为毫秒（必填，int类型）
            重要: 必须传递整数类型的毫秒时间戳
            通常使用 get_current_timestamp() 工具获取当前时间作为结束时间
            示例: 1708012345000

        query: 查询语句（可选，CLS 查询语法）
            示例: "level:ERROR" 或 "message:异常"

        limit: 返回结果数量限制（默认100，可选）

    Returns:
        Dict: 搜索结果
            - topic_id: 主题ID
            - start_time: 开始时间戳
            - end_time: 结束时间戳
            - query: 查询语句
            - limit: 结果限制
            - total: 实际返回的日志条数
            - logs: 日志列表，每条日志包含:
                * timestamp: 日志时间（格式: YYYY-MM-DD HH:MM:SS）
                * level: 日志级别
                * message: 日志内容
            - took_ms: 查询耗时（毫秒）
            - message: 查询状态消息

    使用示例:
        # 步骤1: 获取当前时间戳
        current_ts = get_current_timestamp()  # 返回: 1708012345000

        # 步骤2: 计算开始时间（15分钟前）
        start_ts = current_ts - (15 * 60 * 1000)  # 1708011445000

        # 步骤3: 搜索日志
        search_log(
            topic_id="topic-001",
            start_time=start_ts,     # int类型: 1708011445000
            end_time=current_ts,     # int类型: 1708012345000
            limit=100
        )
    """
    from app.config import config

    region = config.tencent_cloud_region

    start_time_seconds = int(start_time / 1000)
    end_time_seconds = int(end_time / 1000)

    try:
        client = get_cls_client(region)

        # 使用 GetLogs 接口获取详细日志
        req = models.GetLogsRequest()

        params = {
            "TopicId": topic_id,
            "From": start_time_seconds,
            "To": end_time_seconds,
            "Query": query or "",
            "Limit": limit,
            "Sort": "desc"  # 按时间降序排列
        }
        req.from_json_string(json.dumps(params))

        resp = client.GetLogs(req)
        result = json.loads(resp.to_json_string())

        logs = []
        log_list = result.get("Results", [])

        for log_data in log_list:
            # 解析日志内容
            try:
                # CLS 返回的日志格式可能是 JSON 或其他格式
                if isinstance(log_data, str):
                    try:
                        log_content = json.loads(log_data)
                    except:
                        log_content = {"message": log_data}
                else:
                    log_content = log_data

                # 提取时间戳
                timestamp = log_content.get("__TIMESTAMP__", 0)
                if isinstance(timestamp, (int, float)):
                    log_time = datetime.fromtimestamp(timestamp)
                    time_str = log_time.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    time_str = str(timestamp)

                # 提取日志级别
                level = log_content.get("level", log_content.get("Level", "INFO"))

                # 提取日志消息
                message = log_content.get("message", log_content.get("Message", str(log_content)))

                log_entry = {
                    "timestamp": time_str,
                    "level": level.upper(),
                    "message": message
                }

                logs.append(log_entry)
            except Exception as e:
                logger.warning(f"解析日志条目失败: {str(e)}")
                continue

        return {
            "topic_id": topic_id,
            "start_time": start_time,
            "end_time": end_time,
            "query": query,
            "limit": limit,
            "total": len(logs),
            "logs": logs,
            "took_ms": result.get("ElapsedMillisecond", 0),
            "message": f"成功查询 {len(logs)} 条日志"
        }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"查询日志失败: {error_msg}")

        # 检查是否是 topic 不存在的错误
        if "TopicNotExist" in error_msg or "not found" in error_msg.lower():
            return {
                "topic_id": topic_id,
                "start_time": start_time,
                "end_time": end_time,
                "query": query,
                "limit": limit,
                "total": 0,
                "logs": [],
                "took_ms": 0,
                "error": f"主题不存在: {topic_id}",
                "message": f"错误: 未找到主题 {topic_id}，请检查 topic_id 是否正确"
            }
        else:
            return {
                "topic_id": topic_id,
                "start_time": start_time,
                "end_time": end_time,
                "query": query,
                "limit": limit,
                "total": 0,
                "logs": [],
                "took_ms": 0,
                "error": f"查询失败: {error_msg}",
                "message": f"错误: 查询日志时发生错误 - {error_msg}"
            }


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8003, path="/mcp")
