"""
请求数据模型
定义 API 请求的 pydantic 模型
"""
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """
    聊天请求模型
    """
    id: str = Field(..., description="会话 ID", alias="id")
    question: str = Field(..., description="用户问题", alias="Question")

    class Config:
        populate_by_name = True,
        json_schema_extra = {
            "example": {
                "id": "session-123",
                "Question": "什么是向量数据库？"
            }
        }

class ClearRequest(BaseModel):
    """
    清除会话请求模型
    """
    session_id: str = Field(..., description="会话 ID", alias="sessionId")

    class Config:
        populate_by_name = True,