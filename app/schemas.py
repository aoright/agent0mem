from pydantic import BaseModel, Field
from typing import List, Optional, Union, Any


class MessagePart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[dict] = None


class Message(BaseModel):
    role: str
    content: Union[str, List[Union[dict, MessagePart]]]
    timestamp: Optional[int] = None


class AddRequest(BaseModel):
    request_id: str
    messages: List[Message]
    user_id: str
    session_id: str


class AddResponse(BaseModel):
    success: bool = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: Union[str, List[Union[dict, MessagePart]]]
    options: Optional[List[Any]] = None
    user_id: str
    session_id: Optional[str] = None
    top_k: int = 100


class SearchMemoryItem(BaseModel):
    id: str
    content: str
    text: Optional[str] = None
    score: Optional[float] = None
    created_at: Optional[str] = None


class SearchResponse(BaseModel):
    data: List[SearchMemoryItem]


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "agent0mem"
    version: str = "v1.0"
