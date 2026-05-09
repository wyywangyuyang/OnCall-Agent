"""
对话接口
提供基于 RAG Agent 的普通对话和流式对话接口
"""
import json

from fastapi import APIRouter, HTTPException
from loguru import logger
from sse_starlette import EventSourceResponse

from app.models.request import ChatRequest, ClearRequest
from app.models.response import ApiResponse, SessionInfoResponse
from app.services.rag_agent_service import rag_agent_service

router = APIRouter()

@router.post('/chat')
async def chat(request: ChatRequest):
    """
    快速对话接口
        {
            "code": 200,
            "message": "success",
            "data": {
                "success": true,
                "answer": "回答内容",
                "errorMessage": null
            }
        }

        Args:
            request: 对话请求

        Returns:
            统一格式的对话响应
    """
    try:
        logger.info(f"[会话{request.id}],收到快速对话请求: {request.question}")
        answer = await rag_agent_service.query(
            question=request.question,
            session_id=request.id
        )
        logger.info(f"[会话{request.id}],快速对话完成")
        return {
            "code": 200,
            "message": "success",
            "data": {
                "success": True,
                "answer": answer,
                "errorMessage": None
            }
        }
    except Exception as e:
        logger.error(f"[会话{request.id}],快速对话失败: {str(e)}")
        return {
            "code": 500,
            "message": "error",
            "data": {
                "success": False,
                "answer": None,
                "errorMessage": str(e)
            }
        }

@router.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    """
    流式对话接口（基于 RAG Agent，SSE）

        返回 SSE 格式，data 字段为 JSON：

        工具调用事件:
        event: message
        data: {"type":"tool_call","data":{"tool":"工具名","status":"start|end","input":{...}}}

        内容流式事件:
        event: message
        data: {"type":"content","data":"内容块"}

        完成事件:
        event: message
        data: {"type":"done","data":{"answer":"完整答案","tool_calls":[...]}}

        Args:
            request: 对话请求

        Returns:
            SSE 事件流
    """
    logger.info(f"[会话{request.id}],收到流式对话请求: {request.question}")

    async def event_generator():
        try:
            async for chunk in rag_agent_service.query_stream(
                question=request.question,
                session_id=request.id
            ):
                chunk_type = chunk.get("type", "unknown")
                chunk_data = chunk.get("data", None)

                # 处理调用类型消息 (新增)
                if chunk_type == "debug":
                    # 调试信息，可以选择发送或忽略
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "debug",
                                "node": chunk.get("node", "unknown"),
                                "message_type": chunk.get("message_type", "unknown"),
                            }, ensure_ascii=False
                        )
                    }
                elif chunk_type == "tool_call":
                    # 发送工具调用事件（可选，前端可以显示工具调用状态）
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "tool_call",
                                "data": chunk_data
                            }, ensure_ascii=False
                        )
                    }
                elif chunk_type == "search_result":
                    # 发送搜索结果事件（可选，前端可以忽略）
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "search_result",
                                "data": chunk_data
                            }, ensure_ascii=False
                        )
                    }
                elif chunk_type == "content":
                    # 发送内容块事件 -> 关键：data 必须是 JSON 字符串
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "content",
                                "data": chunk_data
                            }, ensure_ascii=False
                        )
                    }
                elif chunk_type == "complete":
                    # 发送完成信号
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "done",
                                "data": chunk_data
                            }, ensure_ascii=False
                        )
                    }
                elif chunk_type == "error":
                    # 发送错误事件
                    yield {
                        "event": "message",
                        "data": json.dumps(
                            {
                                "type": "error",
                                "data": str(chunk_data)
                            }, ensure_ascii=False
                        )
                    }

            logger.info(f"[会话 {request.id}]，流式对话完成")

        except Exception as e:
            logger.error(f"流式对话失败: {str(e)}")
            yield {
                "event": "message",
                "data": json.dumps(
                    {
                        "type": "error",
                        "data": str(e)
                    }, ensure_ascii=False
                )
            }

    return EventSourceResponse(event_generator())

@router.post("/chat/clear", response_model=ApiResponse)
async def clear_session(request: ClearRequest):
    """
    清空会话历史

    Args:
        request: 清空请求

    Returns:
        操作结果
    """
    try:
        success = rag_agent_service.clear_session(session_id=request.session_id)
        logger.info(f"清空会话：{request.session_id}, 结果：{success}")

        return ApiResponse(
            status="success" if success else "error",
            message="会话历史已清空" if success else "会话历史清空失败",
            data = None
        )

    except Exception as e:
        logger.error(f"清空会话失败: {str(e)}")
        return HTTPException(status_code=500, detail=str(e))

@router.post("chat/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str):
    """
    获取会话信息

    Args:
        session_id: 会话 ID

    Returns:
        会话信息
    """
    try:
        history = rag_agent_service.get_session_info(session_id=session_id)

        return SessionInfoResponse(
            session_id=session_id,
            message_count=len(history),
            history=history
        )

    except Exception as e:
        logger.error(f"获取会话信息失败: {str(e)}")
        return HTTPException(status_code=500, detail=str(e))
