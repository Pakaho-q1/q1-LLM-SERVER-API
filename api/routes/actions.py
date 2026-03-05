import asyncio

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import verify_api_key
from api.runtime import executor, history_manager, put_sse_message
from api.schemas import ApiActionRequest
from api.services import (
    handle_cancel_download,
    handle_count_tokens,
    handle_create_preset,
    handle_delete_model,
    handle_delete_preset,
    handle_download_model,
    handle_download_status,
    handle_fetch_hf,
    handle_get_model_status,
    handle_get_preset,
    handle_list_models,
    handle_list_presets,
    handle_load_model,
    handle_unload_model,
    handle_update_preset,
    safe_error_detail,
)

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/api/action")
async def api_action(payload: ApiActionRequest):
    client_id = payload.client_id
    action = payload.action
    payload_data = payload.model_dump()

    try:
        loop = asyncio.get_running_loop()

        if action == "list_sessions":
            sessions = await loop.run_in_executor(executor, history_manager.get_all_sessions)
            await put_sse_message(client_id, {"type": "sessions_list", "data": sessions})
            return {"data": sessions}

        if action == "list_models":
            await handle_list_models(client_id)
            return {"status": "ok"}
        if action == "load_model":
            await handle_load_model(client_id, payload_data)
            return {"status": "ok"}
        if action == "unload_model":
            await handle_unload_model(client_id)
            return {"status": "ok"}
        if action == "delete_model":
            await handle_delete_model(client_id, payload_data)
            return {"status": "ok"}
        if action == "fetch_hf":
            await handle_fetch_hf(client_id, payload_data)
            return {"status": "ok"}
        if action == "download_model":
            await handle_download_model(client_id, payload_data)
            return {"status": "ok"}
        if action == "cancel_download":
            await handle_cancel_download(client_id, payload_data)
            return {"status": "ok"}
        if action == "download_status":
            await handle_download_status(client_id)
            return {"status": "ok"}
        if action == "count_tokens":
            await handle_count_tokens(client_id, payload_data)
            return {"status": "ok"}
        if action == "get_model_status":
            await handle_get_model_status(client_id)
            return {"status": "ok"}

        if action == "list_presets":
            await handle_list_presets(client_id)
            return {"status": "ok"}
        if action == "get_preset":
            await handle_get_preset(client_id, payload_data)
            return {"status": "ok"}
        if action == "create_preset":
            await handle_create_preset(client_id, payload_data)
            return {"status": "ok"}
        if action == "update_preset":
            await handle_update_preset(client_id, payload_data)
            return {"status": "ok"}
        if action == "delete_preset":
            await handle_delete_preset(client_id, payload_data)
            return {"status": "ok"}

        if action == "create_session":
            title = payload.title or "New Chat"
            new_session = await loop.run_in_executor(executor, history_manager.create_session, title)
            await put_sse_message(client_id, {"type": "session_created", "data": new_session})
            return {"data": new_session}

        if action == "rename_session":
            conv_id = payload.conversation_id
            title = payload.title
            await loop.run_in_executor(executor, history_manager.rename_session, conv_id, title)
            await put_sse_message(client_id, {"type": "session_renamed", "conversation_id": conv_id, "title": title})
            sessions = await loop.run_in_executor(executor, history_manager.get_all_sessions)
            await put_sse_message(client_id, {"type": "sessions_list", "data": sessions})
            return {"status": "ok"}

        if action == "delete_session":
            conv_id = payload.conversation_id
            await loop.run_in_executor(executor, history_manager.delete_session, conv_id)
            await put_sse_message(client_id, {"type": "session_deleted", "conversation_id": conv_id})
            sessions = await loop.run_in_executor(executor, history_manager.get_all_sessions)
            await put_sse_message(client_id, {"type": "sessions_list", "data": sessions})
            return {"status": "ok"}

        if action == "get_chat_history":
            conv_id = payload.conversation_id
            messages = await loop.run_in_executor(executor, history_manager.get_chat_history, conv_id)
            await put_sse_message(client_id, {"type": "chat_history", "conversation_id": conv_id, "data": messages})
            return {"conversation_id": conv_id, "data": messages}

        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail=safe_error_detail())
