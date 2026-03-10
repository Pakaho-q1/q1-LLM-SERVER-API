from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

class SessionCreateRequest(BaseModel):
    title: str = "New Chat"


OpenAIChatRole = Literal["system", "user", "assistant", "tool"]


class OpenAIImageUrl(BaseModel):
    url: str


class OpenAIContentPartText(BaseModel):
    type: Literal["text"]
    text: str


class OpenAIContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: OpenAIImageUrl


OpenAIMessageContent = Union[
    str,
    List[Union[OpenAIContentPartText, OpenAIContentPartImage]],
]


class OpenAIChatMessage(BaseModel):
    role: OpenAIChatRole
    content: OpenAIMessageContent


class OpenAIChatCompletionRequest(BaseModel):
    model: str = "local-model"
    messages: List[OpenAIChatMessage] = Field(default_factory=list)
    stream: bool = False
    client_id: Optional[str] = None
    conversation_id: Optional[str] = None
    request_id: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[Any] = None
    params: Dict[str, Any] = Field(default_factory=dict)


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
