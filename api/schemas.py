from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ActionType = Literal[
    "list_sessions",
    "list_models",
    "load_model",
    "unload_model",
    "delete_model",
    "fetch_hf",
    "download_model",
    "cancel_download",
    "download_status",
    "count_tokens",
    "get_model_status",
    "list_presets",
    "get_preset",
    "create_preset",
    "update_preset",
    "delete_preset",
    "create_session",
    "rename_session",
    "delete_session",
    "get_chat_history",
]


class SessionCreateRequest(BaseModel):
    title: str = "New Chat"


class ApiActionRequest(BaseModel):
    client_id: str
    action: ActionType
    conversation_id: Optional[str] = None
    title: Optional[str] = None
    model_path: Optional[str] = None
    filename: Optional[str] = None
    repo: Optional[str] = None
    url: Optional[str] = None
    job_id: Optional[str] = None
    text: Optional[str] = None
    preset_id: Optional[str] = None
    name: Optional[str] = None
    preset: Optional[Dict[str, Any]] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class SSEChatRequest(BaseModel):
    client_id: str
    conversation_id: str = "default_conv"
    content: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 6
    hybrid: bool = True
    rerank: bool = True
    score_threshold: float = 0.18


class RAGCitation(BaseModel):
    index: int
    chunk_id: str
    source: str
    score: float


class RAGRetrieveResponse(BaseModel):
    query: str
    context: str
    citations: List[RAGCitation] = Field(default_factory=list)
    should_answer: bool = True
    reason: Optional[str] = None
