import asyncio
import logging
import os
import json
import uuid
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from managers.chat_orchestrator import ChatOrchestrator
from managers.history_manager import HistoryManager
from managers.preset_manager import PresetManager
from managers.llm_core import LLMEngine
from managers.model_manager import ModelManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="worker_")

project_root = Path(__file__).resolve().parent.parent
models_dir = project_root / "models"


llm_engine = LLMEngine()
model_manager = ModelManager(models_dir=str(models_dir))
preset_manager = PresetManager()

history_manager = HistoryManager(
    engine=llm_engine,
    db_path="data/chat.db",
    n_ctx=llm_engine.n_ctx,
    tokenizer_fn=llm_engine.count_tokens,
)

chat_orchestrator = ChatOrchestrator(llm_engine, history_manager, executor)


limiter = Limiter(key_func=get_remote_address)


API_KEY = os.environ.get("LLM_API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str = Depends(api_key_header)):
    if not API_KEY:
        return
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class DownloadJob(BaseModel):
    id: str
    filename: str
    progress: float = Field(default=0, ge=0, le=100)
    speed: float = Field(default=0)
    eta: float = Field(default=0)
    status: str = Field(default="queued")
    error: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    path: str
    size: int
    loaded: bool = False


async def sse_broadcast(message: Dict[str, Any]) -> None:
    async with sse_lock:
        queues = list(sse_queues.items())
    for client_id, q in queues:
        try:
            await q.put(message)
        except Exception:
            logger.exception(f"Failed to broadcast to {client_id}")


async def put_sse_message(client_id: str, message: Dict[str, Any]) -> None:
    async with sse_lock:
        q = sse_queues.get(client_id)
    if q:
        try:
            await q.put(message)
        except Exception:
            logger.exception("Failed to put SSE message")


async def sse_send_to_client(client_id: str, message: Dict[str, Any]) -> None:
    await put_sse_message(client_id, message)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Server starting...")
    yield
    logger.info("🛑 Server shutting down...")
    executor.shutdown(wait=True)


app = FastAPI(
    title="LLM WebUI API",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


sse_queues: Dict[str, asyncio.Queue] = {}
sse_lock = asyncio.Lock()


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Detailed health check — model status, queue depth, active downloads."""
    active_downloads = [
        j
        for j in model_manager.download_manager.get_jobs()
        if j.get("status") == "downloading"
    ]
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "loaded": bool(llm_engine.llm),
            "name": llm_engine.model_name or None,
            "n_ctx": llm_engine.n_ctx,
        },
        "sse_connections": len(sse_queues),
        "active_downloads": len(active_downloads),
    }


@app.get("/sessions", dependencies=[Depends(verify_api_key)])
async def list_sessions():
    loop = asyncio.get_running_loop()
    sessions = await loop.run_in_executor(executor, history_manager.get_all_sessions)
    return {"data": sessions}


@app.post("/sessions", dependencies=[Depends(verify_api_key)])
async def create_session(payload: Dict[str, Any]):
    title = payload.get("title", "New Chat")
    loop = asyncio.get_running_loop()
    new_session = await loop.run_in_executor(
        executor, history_manager.create_session, title
    )
    return {"data": new_session}


@app.get("/history/{conversation_id}", dependencies=[Depends(verify_api_key)])
async def get_history(conversation_id: str):
    loop = asyncio.get_running_loop()
    messages = await loop.run_in_executor(
        executor, history_manager.get_chat_history, conversation_id
    )
    return {"conversation_id": conversation_id, "data": messages}


@app.post("/api/action", dependencies=[Depends(verify_api_key)])
async def api_action(payload: Dict[str, Any]):
    """Generic REST proxy for non-chat actions."""
    client_id = payload.get("client_id")
    action = payload.get("action")
    if not client_id or not action:
        raise HTTPException(status_code=400, detail="client_id and action required")

    try:
        loop = asyncio.get_running_loop()

        if action == "list_sessions":
            sessions = await loop.run_in_executor(
                executor, history_manager.get_all_sessions
            )
            await put_sse_message(
                client_id, {"type": "sessions_list", "data": sessions}
            )
            return {"data": sessions}

        if action == "list_models":
            await handle_list_models(client_id)
            return {"status": "ok"}
        if action == "load_model":
            await handle_load_model(client_id, payload)
            return {"status": "ok"}
        if action == "unload_model":
            await handle_unload_model(client_id)
            return {"status": "ok"}
        if action == "delete_model":
            await handle_delete_model(client_id, payload)
            return {"status": "ok"}
        if action == "fetch_hf":
            await handle_fetch_hf(client_id, payload)
            return {"status": "ok"}
        if action == "download_model":
            await handle_download_model(client_id, payload)
            return {"status": "ok"}
        if action == "cancel_download":
            await handle_cancel_download(client_id, payload)
            return {"status": "ok"}
        if action == "download_status":
            await handle_download_status(client_id)
            return {"status": "ok"}
        if action == "count_tokens":
            await handle_count_tokens(client_id, payload)
            return {"status": "ok"}
        if action == "get_model_status":
            await handle_get_model_status(client_id)
            return {"status": "ok"}

        if action == "list_presets":
            await handle_list_presets(client_id)
            return {"status": "ok"}
        if action == "get_preset":
            await handle_get_preset(client_id, payload)
            return {"status": "ok"}
        if action == "create_preset":
            await handle_create_preset(client_id, payload)
            return {"status": "ok"}
        if action == "update_preset":
            await handle_update_preset(client_id, payload)
            return {"status": "ok"}
        if action == "delete_preset":
            await handle_delete_preset(client_id, payload)
            return {"status": "ok"}

        if action == "create_session":
            title = payload.get("title", "New Chat")
            new_session = await loop.run_in_executor(
                executor, history_manager.create_session, title
            )
            await put_sse_message(
                client_id, {"type": "session_created", "data": new_session}
            )
            return {"data": new_session}

        if action == "rename_session":
            conv_id = payload.get("conversation_id")
            title = payload.get("title")
            await loop.run_in_executor(
                executor, history_manager.rename_session, conv_id, title
            )
            await put_sse_message(
                client_id,
                {"type": "session_renamed", "conversation_id": conv_id, "title": title},
            )
            sessions = await loop.run_in_executor(
                executor, history_manager.get_all_sessions
            )
            await put_sse_message(
                client_id, {"type": "sessions_list", "data": sessions}
            )
            return {"status": "ok"}

        if action == "delete_session":
            conv_id = payload.get("conversation_id")
            await loop.run_in_executor(
                executor, history_manager.delete_session, conv_id
            )
            await put_sse_message(
                client_id, {"type": "session_deleted", "conversation_id": conv_id}
            )
            sessions = await loop.run_in_executor(
                executor, history_manager.get_all_sessions
            )
            await put_sse_message(
                client_id, {"type": "sessions_list", "data": sessions}
            )
            return {"status": "ok"}

        if action == "get_chat_history":
            conv_id = payload.get("conversation_id")
            messages = await loop.run_in_executor(
                executor, history_manager.get_chat_history, conv_id
            )
            await put_sse_message(
                client_id,
                {"type": "chat_history", "conversation_id": conv_id, "data": messages},
            )
            return {"conversation_id": conv_id, "data": messages}

        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("API action error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sse/stream", dependencies=[Depends(verify_api_key)])
async def sse_stream(request: Request, client_id: str):
    q: asyncio.Queue = asyncio.Queue()
    async with sse_lock:
        sse_queues[client_id] = q

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await q.get()
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
                    data = json.dumps(
                        {"type": "error", "message": "invalid event payload"}
                    )
                    event = "error"

                if event:
                    yield f"event: {event}\n"
                yield f"data: {data}\n\n"
                await asyncio.sleep(0)
        finally:
            async with sse_lock:
                sse_queues.pop(client_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/sse/chat", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def sse_chat_endpoint(request: Request, payload: Dict[str, Any]):
    """Receive chat requests and stream results into client's SSE queue."""
    client_id = payload.get("client_id")
    if not client_id:
        raise HTTPException(status_code=400, detail="client_id is required")

    conv_id = payload.get("conversation_id", "default_conv")
    messages = payload.get("messages") or []
    user_input = payload.get("content") or (
        messages[-1].get("content") if messages else ""
    )
    params = payload.get("params", {}) or {}

    if not user_input:
        raise HTTPException(status_code=400, detail="content is required")

    if isinstance(messages, list):
        system_msg = next(
            (
                m.get("content", "")
                for m in messages
                if isinstance(m, dict)
                and m.get("role") == "system"
                and m.get("content")
            ),
            "",
        )
        if system_msg and not params.get("system_prompt"):
            params = {**params, "system_prompt": system_msg}

    async with sse_lock:
        q = sse_queues.get(client_id)
    if not q:
        raise HTTPException(status_code=404, detail="SSE client not connected")

    request_id = payload.get("request_id") or str(uuid.uuid4())
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    async def _runner():
        try:

            async def status_cb(msg: str):
                await put_sse_message(client_id, {"type": "status", "message": msg})

            async def chunk_cb(chunk: str):
                openai_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": llm_engine.model_name or "local-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                await put_sse_message(client_id, {"__openai_chunk": openai_chunk})

            await chat_orchestrator.process_chat(
                conv_id, user_input, params, status_cb, chunk_cb, messages=None
            )

            final_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": llm_engine.model_name or "local-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            await put_sse_message(client_id, {"__openai_chunk": final_chunk})
            await put_sse_message(client_id, {"__openai_done": True})

        except Exception as e:
            logger.exception("Orchestrator error in SSE chat")
            await put_sse_message(client_id, {"type": "error", "message": str(e)})

    asyncio.create_task(_runner())
    return {"status": "accepted", "conversation_id": conv_id, "request_id": request_id}


async def handle_chat(client_id: str, data: Dict[str, Any]) -> None:
    try:
        conv_id = data.get("conversation_id", "default_conv")
        messages = data.get("messages") or []
        user_input = data.get("content") or (
            messages[-1].get("content") if messages else ""
        )
        params = data.get("params", {}) or {}

        if not user_input:
            await sse_send_to_client(
                client_id, {"type": "error", "message": "Content empty"}
            )
            return

        if isinstance(messages, list):
            system_msg = next(
                (
                    m.get("content", "")
                    for m in messages
                    if isinstance(m, dict)
                    and m.get("role") == "system"
                    and m.get("content")
                ),
                "",
            )
            if system_msg and not params.get("system_prompt"):
                params = {**params, "system_prompt": system_msg}

        async def status_cb(msg):
            await sse_send_to_client(client_id, {"type": "status", "message": msg})

        async def chunk_cb(chunk):
            await sse_send_to_client(client_id, {"type": "chunk", "content": chunk})

        await chat_orchestrator.process_chat(
            conv_id, user_input, params, status_cb, chunk_cb, messages=None
        )
        await sse_send_to_client(
            client_id,
            {"type": "done", "conversation_id": conv_id, "message": "Response saved."},
        )
    except Exception as e:
        logger.error(f"Chat handler error: {e}", exc_info=True)
        await sse_send_to_client(client_id, {"type": "error", "message": str(e)})


async def handle_list_models(client_id: str) -> None:
    try:
        loop = asyncio.get_running_loop()
        models = await loop.run_in_executor(executor, model_manager.list_models)
        await sse_send_to_client(
            client_id, {"type": "models_list", "data": models or []}
        )
    except Exception as e:
        logger.error(f"List models error: {e}", exc_info=True)
        await sse_send_to_client(client_id, {"type": "error", "message": str(e)})


async def handle_load_model(client_id: str, data: Dict[str, Any]) -> None:
    try:
        model_name = data.get("model_path", "")
        if not model_name:
            raise ValueError("model_path is required")
        params = data.get("params", {})
        full_model_path = str(Path(model_manager.models_dir) / model_name)

        await sse_send_to_client(
            client_id, {"type": "status", "message": f"Loading: {model_name}..."}
        )

        loop = asyncio.get_running_loop()
        success, msg = await loop.run_in_executor(
            executor, lambda: llm_engine.load_model(full_model_path, **(params or {}))
        )

        if success:
            await sse_broadcast(
                {"type": "success", "message": f"Model loaded: {model_name}"}
            )
            await sse_broadcast(
                {"type": "model_status", "data": {"running": True, "name": model_name}}
            )
            logger.info(f"✅ Model loaded: {full_model_path}")
        else:
            await sse_send_to_client(client_id, {"type": "error", "message": msg})
    except Exception as e:
        logger.error(f"Load model error: {e}", exc_info=True)
        await sse_send_to_client(client_id, {"type": "error", "message": str(e)})


async def handle_unload_model(client_id: str) -> None:
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, llm_engine.unload_model)
        await sse_broadcast({"type": "success", "message": "Model unloaded"})
        await sse_broadcast(
            {"type": "model_status", "data": {"running": False, "name": ""}}
        )
    except Exception as e:
        logger.error(f"Unload error: {e}", exc_info=True)
        await sse_send_to_client(client_id, {"type": "error", "message": str(e)})


async def handle_delete_model(client_id: str, data: Dict[str, Any]) -> None:
    try:
        filename = data.get("filename", "")
        if not filename:
            raise ValueError("filename is required")
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(
            executor, model_manager.delete_model, filename
        )
        if success:
            await sse_broadcast({"type": "success", "message": f"Deleted: {filename}"})
            await handle_list_models(client_id)
        else:
            await sse_send_to_client(
                client_id, {"type": "error", "message": "Failed to delete model"}
            )
    except Exception as e:
        logger.error(f"Delete error: {e}", exc_info=True)
        await sse_send_to_client(client_id, {"type": "error", "message": str(e)})


async def handle_fetch_hf(client_id: str, data: Dict[str, Any]) -> None:
    try:
        repo = data.get("repo", "")
        if not repo:
            raise ValueError("repo is required")
        loop = asyncio.get_running_loop()
        files = await loop.run_in_executor(executor, model_manager.fetch_hf_repo, repo)
        await sse_send_to_client(client_id, {"type": "hf_files", "data": files or []})
    except Exception as e:
        logger.error(f"HF fetch error: {e}", exc_info=True)
        await sse_send_to_client(client_id, {"type": "error", "message": str(e)})


async def handle_download_model(client_id: str, data: Dict[str, Any]) -> None:
    try:
        url = data.get("url", "")
        if not url:
            raise ValueError("url is required")
        loop = asyncio.get_running_loop()
        job_id = await loop.run_in_executor(executor, model_manager.download_async, url)
        await sse_send_to_client(
            client_id,
            {"type": "success", "message": "Download started", "job_id": job_id},
        )
        logger.info(f"📥 Download job created: {job_id}")
    except Exception as e:
        logger.error(f"Download error: {e}", exc_info=True)
        await sse_send_to_client(client_id, {"type": "error", "message": str(e)})


async def handle_cancel_download(client_id: str, data: Dict[str, Any]) -> None:
    try:
        job_id = data.get("job_id", "")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            executor, model_manager.download_manager.cancel, job_id
        )
        await sse_broadcast({"type": "success", "message": "Download cancelled"})
    except Exception as e:
        logger.error(f"Cancel error: {e}", exc_info=True)


async def handle_download_status(client_id: str) -> None:
    try:
        loop = asyncio.get_running_loop()
        jobs = await loop.run_in_executor(
            executor, model_manager.download_manager.get_jobs
        )
        jobs_list = [
            {
                "id": j.get("id"),
                "filename": j.get("filename"),
                "progress": j.get("progress"),
                "speed": j.get("speed"),
                "eta": j.get("eta"),
                "status": j.get("status"),
                "error": j.get("error"),
            }
            for j in jobs
        ]
        await sse_send_to_client(
            client_id, {"type": "download_status", "data": jobs_list}
        )
    except Exception as e:
        logger.error(f"Status error: {e}", exc_info=True)


async def handle_count_tokens(client_id: str, data: Dict[str, Any]) -> None:
    try:
        text = data.get("text", "")
        loop = asyncio.get_running_loop()
        count = await loop.run_in_executor(executor, llm_engine.count_tokens, text)
        await sse_send_to_client(client_id, {"type": "token_count", "data": count})
    except Exception as e:
        logger.error(f"Token count error: {e}", exc_info=True)


async def handle_list_presets(client_id: str) -> None:
    try:
        loop = asyncio.get_running_loop()
        presets = await loop.run_in_executor(executor, preset_manager.list_presets)
        await sse_send_to_client(client_id, {"type": "presets", "data": presets or []})
    except Exception as e:
        logger.error(f"List presets error: {e}", exc_info=True)
        await sse_send_to_client(client_id, {"type": "error", "message": str(e)})


async def handle_get_preset(client_id: str, data: Dict[str, Any]) -> None:
    try:
        preset_id = data.get("preset_id") or data.get("name", "")
        if not preset_id:
            raise ValueError("preset_id or name is required")
        loop = asyncio.get_running_loop()
        preset = await loop.run_in_executor(
            executor, preset_manager.get_preset, preset_id
        )
        await sse_send_to_client(client_id, {"type": "preset_data", "data": preset})
    except Exception as e:
        logger.error(f"Get preset error: {e}", exc_info=True)
        await sse_send_to_client(client_id, {"type": "error", "message": str(e)})


async def handle_create_preset(client_id: str, data: Dict[str, Any]) -> None:
    try:
        preset_data = data.get("preset", {})
        if not preset_data:
            raise ValueError("preset data is required")
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(
            executor,
            preset_manager.create_preset,
            preset_data.get("name", ""),
            preset_data.get("description", ""),
            preset_data.get("system_prompt", ""),
            preset_data.get("parameters", {}),
        )
        if success:
            await sse_broadcast({"type": "success", "message": "Preset created"})
        else:
            await sse_send_to_client(
                client_id, {"type": "error", "message": "Failed to create preset"}
            )
    except Exception as e:
        logger.error(f"Create preset error: {e}", exc_info=True)
        await sse_send_to_client(client_id, {"type": "error", "message": str(e)})


async def handle_update_preset(client_id: str, data: Dict[str, Any]) -> None:
    try:
        preset_id = data.get("preset_id", "")
        preset_data = data.get("preset", {})
        if not preset_id or not preset_data:
            raise ValueError("preset_id and preset data are required")
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(
            executor,
            preset_manager.update_preset,
            preset_id,
            preset_data.get("description"),
            preset_data.get("system_prompt"),
            preset_data.get("parameters"),
        )
        if success:
            await sse_broadcast({"type": "success", "message": "Preset updated"})
        else:
            await sse_send_to_client(
                client_id, {"type": "error", "message": "Failed to update preset"}
            )
    except Exception as e:
        logger.error(f"Update preset error: {e}", exc_info=True)
        await sse_send_to_client(client_id, {"type": "error", "message": str(e)})


async def handle_delete_preset(client_id: str, data: Dict[str, Any]) -> None:
    try:
        preset_id = data.get("preset_id") or data.get("name", "")
        if not preset_id:
            raise ValueError("preset_id or name is required")
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(
            executor, preset_manager.delete_preset, preset_id
        )
        if success:
            await sse_broadcast({"type": "success", "message": "Preset deleted"})
        else:
            await sse_send_to_client(
                client_id, {"type": "error", "message": "Failed to delete preset"}
            )
    except Exception as e:
        logger.error(f"Delete preset error: {e}", exc_info=True)
        await sse_send_to_client(client_id, {"type": "error", "message": str(e)})


async def handle_get_model_status(client_id: str) -> None:
    try:
        running = bool(llm_engine.llm)
        name = llm_engine.model_name or ""
        await sse_send_to_client(
            client_id,
            {"type": "model_status", "data": {"running": running, "name": name}},
        )
    except Exception as e:
        logger.error(f"Get model status error: {e}", exc_info=True)
        await sse_send_to_client(client_id, {"type": "error", "message": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
