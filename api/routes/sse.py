import asyncio
import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from api.dependencies import verify_api_key
from api.runtime import app_state

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
