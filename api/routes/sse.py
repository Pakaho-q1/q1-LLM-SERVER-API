import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.dependencies import verify_api_key
from api.runtime import app_state, limiter
from api.schemas import SSEChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("/sse/stream")
async def sse_stream(request: Request, client_id: str):
    queue = await app_state.register_sse_client(client_id)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await queue.get()
                except asyncio.CancelledError:
                    break

                if "__openai_chunk" in item:
                    data = json.dumps(item["__openai_chunk"])
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0.001)
                    continue

                if "__openai_done" in item:
                    yield "data: [DONE]\n\n"
                    continue

                try:
                    event = item.get("type")
                    data = json.dumps(item)
                except Exception:
                    data = json.dumps({"type": "error", "message": "invalid event payload"})
                    event = "error"

                if event:
                    yield f"event: {event}\n"
                yield f"data: {data}\n\n"
                await asyncio.sleep(0)
        finally:
            await app_state.unregister_sse_client(client_id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/sse/chat")
@limiter.limit("30/minute")
async def sse_chat_endpoint(request: Request, payload: SSEChatRequest):
    client_id = payload.client_id
    conv_id = payload.conversation_id
    messages = payload.messages or []
    user_input = payload.content or (messages[-1].get("content") if messages else "")
    params = payload.params or {}

    if not user_input:
        raise HTTPException(status_code=400, detail="content is required")

    if isinstance(messages, list):
        system_msg = next(
            (
                m.get("content", "")
                for m in messages
                if isinstance(m, dict) and m.get("role") == "system" and m.get("content")
            ),
            "",
        )
        if system_msg and not params.get("system_prompt"):
            params = {**params, "system_prompt": system_msg}

    queue = await app_state.get_sse_queue(client_id)
    if not queue:
        raise HTTPException(status_code=404, detail="SSE client not connected")

    request_id = payload.request_id or str(uuid.uuid4())
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

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
                conv_id,
                user_input,
                params,
                status_cb,
                chunk_cb,
                messages=None,
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
            logger.exception("Orchestrator error in SSE chat")
            await app_state.put_sse_message(client_id, {"type": "error", "message": str(exc)})

    asyncio.create_task(_runner())
    return {"status": "accepted", "conversation_id": conv_id, "request_id": request_id}
