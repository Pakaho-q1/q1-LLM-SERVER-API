import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _extract_params(params: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(params or {})
    explicit = {
        "temperature": normalized.pop("temperature", None),
        "max_tokens": normalized.pop("max_tokens", None),
        "top_p": normalized.pop("top_p", None),
        "stop": normalized.pop("stop", None),
    }
    for key, value in explicit.items():
        if value is not None:
            normalized[key] = value
    return normalized


async def start_openai_sse_chat(
    app_state: Any,
    *,
    client_id: str,
    conversation_id: str,
    user_input: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    params: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    queue = await app_state.get_sse_queue(client_id)
    if not queue:
        raise ValueError("SSE client not connected")

    req_id = request_id or str(uuid.uuid4())
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    normalized_params = _extract_params(params or {})

    async def _runner():
        try:
            async def status_cb(msg: str):
                await app_state.put_sse_message(client_id, {"type": "status", "message": msg})

            async def chunk_cb(chunk: str):
                openai_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": app_state.llm_engine.model_name or "local-model",
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                await app_state.put_sse_message(client_id, {"__openai_chunk": openai_chunk})

            await app_state.chat_orchestrator.process_chat(
                conversation_id,
                user_input,
                normalized_params,
                status_cb,
                chunk_cb,
                messages=messages,
            )

            final_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": app_state.llm_engine.model_name or "local-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            await app_state.put_sse_message(client_id, {"__openai_chunk": final_chunk})
            await app_state.put_sse_message(client_id, {"__openai_done": True})
        except Exception as exc:
            logger.exception("OpenAI SSE bridge failed")
            await app_state.put_sse_message(client_id, {"type": "error", "message": str(exc)})

    asyncio.create_task(_runner())
    return {"chat_id": chat_id, "created": created, "request_id": req_id}
