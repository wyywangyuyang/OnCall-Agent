"""
响应数据模型
定义 API 响应的 Pydantic 模型
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class ChatResponse(BaseModel):
    """
    聊天响应模型
    """
    answer: str = Field(..., description="AI 回答")
    session_id: str = Field(..., description="会话 ID")

class SessionInfoResponse(BaseModel):
    """
    会话信息响应模型
    """
    session_id: str = Field(..., description="会话 ID")
    message_count: int = Field(..., description="会话消息数量")
    history: List[Dict[str, str]] = Field(..., description="历史消息列表")

class ApiResponse(BaseModel):
    """
    通用 API 响应模型
    """
    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")
    data: Optional[Any] = Field(None, description="数据")

class HealthCheckResponse(BaseModel):
    """
    健康检查响应模型
    """
    status: str = Field(..., description="状态")
    service: str = Field(..., description="服务名称")
    version: str = Field(..., description="版本")