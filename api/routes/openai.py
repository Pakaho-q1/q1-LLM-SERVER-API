import asyncio
import json
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from api.dependencies import verify_api_key
from api.openai_stream import start_openai_sse_chat
from api.runtime import app_state
from api.schemas import OpenAIChatCompletionRequest

router = APIRouter(dependencies=[Depends(verify_api_key)])

PROXY_BASE_URL = os.environ.get("OPENAI_PROXY_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LOCAL_OPENAI_FILES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "openai_files"
LOCAL_OPENAI_FILES_DIR.mkdir(parents=True, exist_ok=True)


def openai_error_response(
    *,
    message: str,
    status_code: int,
    error_type: str = "invalid_request_error",
    code: str | None = None,
    param: str | None = None,
):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


def _normalize_params(payload: OpenAIChatCompletionRequest) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        **(payload.params or {}),
    }
    if payload.temperature is not None:
        merged["temperature"] = payload.temperature
    if payload.max_tokens is not None:
        merged["max_tokens"] = payload.max_tokens
    if payload.top_p is not None:
        merged["top_p"] = payload.top_p
    if payload.stop is not None:
        merged["stop"] = payload.stop
    return merged


def _get_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth[7:].strip()
    return token or None


def _proxy_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
    }


def _proxy_chat_payload(payload: OpenAIChatCompletionRequest) -> Dict[str, Any]:
    proxy_payload: Dict[str, Any] = {
        "model": payload.model,
        "messages": [m.model_dump() for m in payload.messages],
        "stream": bool(payload.stream),
    }
    proxy_payload.update(_normalize_params(payload))
    return proxy_payload


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
            elif hasattr(part, "type") and getattr(part, "type", None) == "text":
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    texts.append(text)
        return "\n".join(t for t in texts if t.strip())

    return ""


def _message_has_image(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            return True
        if hasattr(part, "type") and getattr(part, "type", None) == "image_url":
            return True
    return False


def _messages_has_image(messages: List[Dict[str, Any]]) -> bool:
    for msg in messages:
        if _message_has_image(msg.get("content")):
            return True
    return False


def _extract_attachments_from_params(params: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = params.get("_user_message_metadata")
    if not isinstance(raw, dict):
        return []
    attachments = raw.get("attachments")
    if not isinstance(attachments, list):
        return []

    result: List[Dict[str, str]] = []
    for item in attachments:
        if not isinstance(item, dict):
            continue
        file_id = item.get("file_id")
        if not isinstance(file_id, str) or not file_id.strip():
            continue
        result.append(
            {
                "file_id": file_id.strip(),
                "name": str(item.get("name") or ""),
                "type": str(item.get("type") or ""),
            }
        )
    return result


def _resolve_local_file_path(file_id: str) -> Path | None:
    matches = list(LOCAL_OPENAI_FILES_DIR.glob(f"{file_id}*"))
    if not matches:
        return None
    return matches[0]


def _guess_text_file(content_type: str, filename: str) -> bool:
    lower_ct = (content_type or "").lower()
    if lower_ct.startswith("text/"):
        return True
    if lower_ct in {
        "application/json",
        "application/xml",
        "application/javascript",
        "application/x-javascript",
        "application/x-ndjson",
    }:
        return True

    ext = Path(filename).suffix.lower()
    if ext in {
        ".txt",
        ".md",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
        ".csv",
        ".tsv",
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".html",
        ".css",
        ".sql",
        ".log",
        ".ini",
        ".toml",
        ".cfg",
        ".conf",
        ".sh",
        ".bat",
        ".ps1",
        ".go",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".rs",
    }:
        return True
    return False


def _read_text_file_for_prompt(file_path: Path, max_bytes: int = 120_000) -> Tuple[bool, str]:
    raw = file_path.read_bytes()[:max_bytes]
    try:
        return True, raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return True, raw.decode("utf-8", errors="replace")
        except Exception:
            return False, ""


def _build_attachment_context(params: Dict[str, Any]) -> str:
    attachments = _extract_attachments_from_params(params)
    if not attachments:
        return ""

    blocks: List[str] = []
    for a in attachments:
        file_id = a["file_id"]
        path = _resolve_local_file_path(file_id)
        if path is None:
            continue

        content_type = a["type"] or (mimetypes.guess_type(path.name)[0] or "")
        if _guess_text_file(content_type, path.name):
            ok, text = _read_text_file_for_prompt(path)
            if ok:
                blocks.append(
                    f"[File: {a['name'] or path.name} | id: {file_id}]\n{text.strip()}\n"
                )

    if not blocks:
        return ""

    return (
        "The user attached file content. Use it as trusted context.\n"
        "Attached file excerpts:\n\n"
        + "\n---\n".join(blocks)
    )


def _inject_attachment_context(messages: List[Dict[str, Any]], attachment_context: str) -> List[Dict[str, Any]]:
    if not attachment_context.strip():
        return messages

    return [
        {
            "role": "system",
            "content": attachment_context.strip(),
        },
        *messages,
    ]


async def _proxy_json(method: str, path: str, token: str, **kwargs) -> JSONResponse:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.request(method, f"{PROXY_BASE_URL}{path}", headers=_proxy_headers(token), **kwargs)

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        data = response.json()
    else:
        data = {
            "error": {
                "message": response.text or "Proxy error",
                "type": "server_error",
                "param": None,
                "code": None,
            }
        }

    return JSONResponse(status_code=response.status_code, content=data)


@router.get("/v1/models")
async def list_openai_models(request: Request):
    token = _get_bearer_token(request)
    if token:
        return await _proxy_json("GET", "/models", token)

    loop = asyncio.get_running_loop()
    models = await loop.run_in_executor(app_state.executor, app_state.model_manager.list_models)
    now_ts = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model.get("name", ""),
                "object": "model",
                "created": now_ts,
                "owned_by": "local",
                "permission": [],
            }
            for model in (models or [])
        ],
    }


@router.get("/v1/models/{model_id}")
async def get_openai_model(model_id: str, request: Request):
    token = _get_bearer_token(request)
    if token:
        return await _proxy_json("GET", f"/models/{model_id}", token)

    loop = asyncio.get_running_loop()
    models = await loop.run_in_executor(app_state.executor, app_state.model_manager.list_models)
    found = next((m for m in (models or []) if m.get("name") == model_id), None)
    if not found:
        return openai_error_response(
            message=f"The model '{model_id}' does not exist",
            status_code=404,
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
        )

    return {
        "id": found.get("name", model_id),
        "object": "model",
        "created": int(time.time()),
        "owned_by": "local",
        "permission": [],
    }


@router.post("/v1/files")
async def upload_openai_file(
    request: Request,
    file: UploadFile = File(...),
    purpose: str = Form(default="user_data"),
):
    token = _get_bearer_token(request)
    if token:
        data = await file.read()
        files = {
            "file": (
                file.filename or "upload.bin",
                data,
                file.content_type or "application/octet-stream",
            ),
        }
        return await _proxy_json("POST", "/files", token, files=files, data={"purpose": purpose})

    data = await file.read()
    file_id = f"file-{uuid.uuid4().hex}"
    ext = Path(file.filename or "").suffix
    save_path = LOCAL_OPENAI_FILES_DIR / f"{file_id}{ext}"
    save_path.write_bytes(data)

    return {
        "id": file_id,
        "object": "file",
        "bytes": len(data),
        "created_at": int(time.time()),
        "filename": file.filename or save_path.name,
        "purpose": purpose,
    }


@router.post("/v1/audio/transcriptions")
async def openai_audio_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(default="gpt-4o-mini-transcribe"),
):
    token = _get_bearer_token(request)
    if not token:
        return openai_error_response(
            message="Audio transcription requires external OpenAI API key via Authorization: Bearer",
            status_code=501,
            error_type="invalid_request_error",
            code="transcription_not_configured",
        )

    data = await file.read()
    files = {
        "file": (
            file.filename or "audio.webm",
            data,
            file.content_type or "audio/webm",
        ),
    }
    return await _proxy_json("POST", "/audio/transcriptions", token, files=files, data={"model": model})


@router.get("/v1/files/{file_id}/content")
async def openai_file_content(file_id: str, request: Request):
    token = _get_bearer_token(request)
    if token:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                f"{PROXY_BASE_URL}/files/{file_id}/content",
                headers=_proxy_headers(token),
            )
        if response.status_code >= 400:
            return openai_error_response(
                message=response.text or "Failed to fetch file content",
                status_code=response.status_code,
                error_type="invalid_request_error" if response.status_code < 500 else "server_error",
            )
        media_type = response.headers.get("content-type", "application/octet-stream")
        return StreamingResponse(iter([response.content]), media_type=media_type)

    candidates = list(LOCAL_OPENAI_FILES_DIR.glob(f"{file_id}*"))
    if not candidates:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = candidates[0]
    guessed_content_type = "application/octet-stream"
    lower_name = file_path.name.lower()
    if lower_name.endswith((".png",)):
        guessed_content_type = "image/png"
    elif lower_name.endswith((".jpg", ".jpeg")):
        guessed_content_type = "image/jpeg"
    elif lower_name.endswith(".webp"):
        guessed_content_type = "image/webp"
    elif lower_name.endswith(".gif"):
        guessed_content_type = "image/gif"

    return FileResponse(
        path=str(file_path),
        media_type=guessed_content_type,
        filename=file_path.name,
    )


@router.post("/v1/chat/completions")
async def openai_chat_completions(request: Request, payload: OpenAIChatCompletionRequest):
    if not payload.messages:
        return openai_error_response(
            message="'messages' is required",
            status_code=400,
            code="messages_required",
            param="messages",
        )

    token = _get_bearer_token(request)
    if token:
        proxy_payload = _proxy_chat_payload(payload)
        if proxy_payload.get("stream"):

            async def proxy_stream():
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "POST",
                        f"{PROXY_BASE_URL}/chat/completions",
                        headers=_proxy_headers(token),
                        json=proxy_payload,
                    ) as response:
                        if response.status_code >= 400:
                            body = await response.aread()
                            try:
                                data = json.loads(body.decode("utf-8", errors="ignore") or "{}")
                            except Exception:
                                data = {
                                    "error": {
                                        "message": "Proxy stream failed",
                                        "type": "server_error",
                                        "param": None,
                                        "code": None,
                                    }
                                }
                            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        async for chunk in response.aiter_raw():
                            if chunk:
                                yield chunk

            return StreamingResponse(proxy_stream(), media_type="text/event-stream")

        return await _proxy_json("POST", "/chat/completions", token, json=proxy_payload)

    raw_messages: List[Dict[str, Any]] = [m.model_dump() for m in payload.messages]
    messages = raw_messages
    user_messages = [m for m in messages if m.get("role") == "user"]
    last_user_message = user_messages[-1] if user_messages else {}
    user_input = _extract_text_from_content(last_user_message.get("content"))
    has_image_input = _messages_has_image(messages)

    if has_image_input and not app_state.llm_engine.multimodal_enabled:
        return openai_error_response(
            message="Image input requires a vision-capable model with mmproj loaded",
            status_code=400,
            code="vision_not_enabled",
            param="messages",
        )

    merged_params = _normalize_params(payload)
    attachment_context = _build_attachment_context(merged_params)
    model_messages = _inject_attachment_context(messages, attachment_context)
    has_file_input = bool(_extract_attachments_from_params(merged_params))

    if not user_input and has_image_input:
        user_input = "[Image input]"
    if not user_input and has_file_input:
        user_input = "[File input]"
    if not user_input and not has_image_input and not has_file_input:
        return openai_error_response(
            message="The last user message content is required",
            status_code=400,
            code="content_required",
            param="messages",
        )

    conversation_id = payload.conversation_id or "default_conv"

    if payload.stream and payload.client_id:
        try:
            started = await start_openai_sse_chat(
                app_state,
                client_id=payload.client_id,
                conversation_id=conversation_id,
                user_input=str(user_input),
                messages=model_messages,
                params=merged_params,
                request_id=payload.request_id,
            )
        except ValueError as exc:
            return openai_error_response(
                message=str(exc),
                status_code=404,
                error_type="not_found_error",
                code="sse_client_not_connected",
            )

        return {
            "id": started["chat_id"],
            "object": "chat.completion",
            "created": started["created"],
            "model": app_state.llm_engine.model_name or payload.model or "local-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
            "request_id": started["request_id"],
            "conversation_id": conversation_id,
        }

    if payload.stream:
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())
        model_name = app_state.llm_engine.model_name or payload.model or "local-model"
        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def status_cb(_msg: str):
            return None

        async def chunk_cb(chunk: str):
            await queue.put(chunk)

        async def _runner():
            try:
                await app_state.chat_orchestrator.process_chat(
                    conversation_id,
                    str(user_input),
                    merged_params,
                    status_cb,
                    chunk_cb,
                    messages=model_messages,
                )
                await queue.put(None)
            except Exception as exc:
                await queue.put(exc)

        task = asyncio.create_task(_runner())

        async def event_stream():
            first = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    err_obj = {
                        "error": {
                            "message": str(item),
                            "type": "server_error",
                            "param": None,
                            "code": "completion_failed",
                        }
                    }
                    yield f"data: {json.dumps(err_obj, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": str(item)}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            await task

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    chunks: List[str] = []

    async def status_cb(_msg: str):
        return None

    async def chunk_cb(chunk: str):
        chunks.append(chunk)

    try:
        await app_state.chat_orchestrator.process_chat(
            conversation_id,
            str(user_input),
            merged_params,
            status_cb,
            chunk_cb,
            messages=model_messages,
        )
    except Exception as exc:
        return openai_error_response(
            message=str(exc),
            status_code=500,
            error_type="server_error",
            code="completion_failed",
        )

    content = "".join(chunks)
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    model_name = app_state.llm_engine.model_name or payload.model or "local-model"
    prompt_tokens = sum(max(1, len((m.get("content") or "").split())) for m in messages)
    completion_tokens = max(1, len(content.split())) if content else 0

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
