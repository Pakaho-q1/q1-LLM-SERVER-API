import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict

from fastapi import HTTPException

from api.schemas import ApiActionRequest
from api.state import AppState

logger = logging.getLogger(__name__)


def safe_error_detail(message: str = "Internal server error") -> str:
    return message


class ActionService:
    """Centralized action dispatch and orchestration logic."""

    def __init__(self, state: AppState):
        self.state = state
        self._handlers: Dict[str, Callable[[str, Dict[str, Any], ApiActionRequest], Awaitable[Dict[str, Any]]]] = {
            "list_sessions": self._list_sessions,
            "list_models": self._list_models,
            "load_model": self._load_model,
            "unload_model": self._unload_model,
            "delete_model": self._delete_model,
            "fetch_hf": self._fetch_hf,
            "download_model": self._download_model,
            "cancel_download": self._cancel_download,
            "download_status": self._download_status,
            "stop_generation": self._stop_generation,
            "count_tokens": self._count_tokens,
            "get_model_status": self._get_model_status,
            "list_presets": self._list_presets,
            "get_preset": self._get_preset,
            "create_preset": self._create_preset,
            "update_preset": self._update_preset,
            "delete_preset": self._delete_preset,
            "create_session": self._create_session,
            "rename_session": self._rename_session,
            "delete_session": self._delete_session,
            "get_chat_history": self._get_chat_history,
        }

    async def dispatch(self, payload: ApiActionRequest) -> Dict[str, Any]:
        client_id = payload.client_id
        payload_data = payload.model_dump()
        handler = self._handlers.get(payload.action)
        if handler is None:
            raise HTTPException(status_code=400, detail=f"Unknown action: {payload.action}")

        try:
            return await handler(client_id, payload_data, payload)
        except HTTPException:
            raise
        except Exception:
            logger.exception("API action error")
            raise HTTPException(status_code=500, detail=safe_error_detail())

    async def _list_sessions(self, client_id: str, _data: Dict[str, Any], _payload: ApiActionRequest):
        loop = asyncio.get_running_loop()
        sessions = await loop.run_in_executor(self.state.executor, self.state.history_manager.get_all_sessions)
        await self.state.put_sse_message(client_id, {"type": "sessions_list", "data": sessions})
        return {"data": sessions}

    async def _list_models(self, client_id: str, _data: Dict[str, Any], _payload: ApiActionRequest):
        loop = asyncio.get_running_loop()
        models = await loop.run_in_executor(self.state.executor, self.state.model_manager.list_models)
        await self.state.put_sse_message(client_id, {"type": "models_list", "data": models or []})
        return {"status": "ok"}

    async def _load_model(self, client_id: str, data: Dict[str, Any], _payload: ApiActionRequest):
        model_name = data.get("model_path", "")
        if not model_name:
            raise ValueError("model_path is required")

        params = data.get("params", {})
        safe_model_path = self.state.model_manager.resolve_model_path(model_name)
        if safe_model_path is None:
            raise ValueError("invalid model_path")

        await self.state.put_sse_message(client_id, {"type": "status", "message": f"Loading: {model_name}..."})

        loop = asyncio.get_running_loop()
        success, message = await loop.run_in_executor(
            self.state.executor,
            lambda: self.state.llm_engine.load_model(str(safe_model_path), **(params or {})),
        )

        if success:
            await self.state.broadcast_sse_message({"type": "success", "message": f"Model loaded: {model_name}"})
            await self.state.broadcast_sse_message({"type": "model_status", "data": {"running": True, "name": model_name}})
        else:
            await self.state.put_sse_message(client_id, {"type": "error", "message": message})

        return {"status": "ok"}

    async def _unload_model(self, _client_id: str, _data: Dict[str, Any], _payload: ApiActionRequest):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.state.executor, self.state.llm_engine.unload_model)
        await self.state.broadcast_sse_message({"type": "success", "message": "Model unloaded"})
        await self.state.broadcast_sse_message({"type": "model_status", "data": {"running": False, "name": ""}})
        return {"status": "ok"}

    async def _delete_model(self, client_id: str, data: Dict[str, Any], _payload: ApiActionRequest):
        filename = data.get("filename", "")
        if not filename:
            raise ValueError("filename is required")

        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(self.state.executor, self.state.model_manager.delete_model, filename)
        if success:
            await self.state.broadcast_sse_message({"type": "success", "message": f"Deleted: {filename}"})
        else:
            await self.state.put_sse_message(client_id, {"type": "error", "message": "Failed to delete model"})

        return {"status": "ok"}

    async def _fetch_hf(self, client_id: str, data: Dict[str, Any], _payload: ApiActionRequest):
        repo = data.get("repo", "")
        if not repo:
            raise ValueError("repo is required")

        loop = asyncio.get_running_loop()
        files = await loop.run_in_executor(self.state.executor, self.state.model_manager.fetch_hf_repo, repo)
        await self.state.put_sse_message(client_id, {"type": "hf_files", "data": files or []})
        return {"status": "ok"}

    async def _download_model(self, client_id: str, data: Dict[str, Any], _payload: ApiActionRequest):
        url = data.get("url", "")
        if not url:
            raise ValueError("url is required")
        loop = asyncio.get_running_loop()
        job_id = await loop.run_in_executor(self.state.executor, self.state.model_manager.download_async, url)
        await self.state.put_sse_message(client_id, {"type": "success", "message": "Download started", "job_id": job_id})
        return {"status": "ok"}

    async def _cancel_download(self, _client_id: str, data: Dict[str, Any], _payload: ApiActionRequest):
        job_id = data.get("job_id", "")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.state.executor, self.state.model_manager.download_manager.cancel, job_id)
        await self.state.broadcast_sse_message({"type": "success", "message": "Download cancelled"})
        return {"status": "ok"}

    async def _download_status(self, client_id: str, _data: Dict[str, Any], _payload: ApiActionRequest):
        loop = asyncio.get_running_loop()
        jobs = await loop.run_in_executor(self.state.executor, self.state.model_manager.download_manager.get_jobs)
        jobs_list = [
            {
                "id": job.get("id"),
                "filename": job.get("filename"),
                "progress": job.get("progress"),
                "speed": job.get("speed"),
                "eta": job.get("eta"),
                "status": job.get("status"),
                "error": job.get("error"),
            }
            for job in jobs
        ]
        await self.state.put_sse_message(client_id, {"type": "download_status", "data": jobs_list})
        return {"status": "ok"}

    async def _stop_generation(self, client_id: str, _data: Dict[str, Any], _payload: ApiActionRequest):
        await self.state.put_sse_message(client_id, {"type": "done", "reason": "stopped_by_user"})
        return {"status": "ok"}

    async def _count_tokens(self, client_id: str, data: Dict[str, Any], _payload: ApiActionRequest):
        text = data.get("text", "")
        loop = asyncio.get_running_loop()
        count = await loop.run_in_executor(self.state.executor, self.state.llm_engine.count_tokens, text)
        await self.state.put_sse_message(client_id, {"type": "token_count", "data": count})
        return {"status": "ok"}

    async def _get_model_status(self, client_id: str, _data: Dict[str, Any], _payload: ApiActionRequest):
        running = bool(self.state.llm_engine.llm)
        name = self.state.llm_engine.model_name or ""
        await self.state.put_sse_message(client_id, {"type": "model_status", "data": {"running": running, "name": name}})
        return {"status": "ok"}

    async def _list_presets(self, client_id: str, _data: Dict[str, Any], _payload: ApiActionRequest):
        loop = asyncio.get_running_loop()
        presets = await loop.run_in_executor(self.state.executor, self.state.preset_manager.list_presets)
        await self.state.put_sse_message(client_id, {"type": "presets", "data": presets or []})
        return {"status": "ok"}

    async def _get_preset(self, client_id: str, data: Dict[str, Any], _payload: ApiActionRequest):
        preset_id = data.get("preset_id") or data.get("name", "")
        if not preset_id:
            raise ValueError("preset_id or name is required")

        loop = asyncio.get_running_loop()
        preset = await loop.run_in_executor(self.state.executor, self.state.preset_manager.get_preset, preset_id)
        await self.state.put_sse_message(client_id, {"type": "preset_data", "data": preset})
        return {"status": "ok"}

    async def _create_preset(self, client_id: str, data: Dict[str, Any], _payload: ApiActionRequest):
        preset_data = data.get("preset", {})
        if not preset_data:
            raise ValueError("preset data is required")

        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(
            self.state.executor,
            self.state.preset_manager.create_preset,
            preset_data.get("name", ""),
            preset_data.get("description", ""),
            preset_data.get("system_prompt", ""),
            preset_data.get("parameters", {}),
        )
        if success:
            await self.state.broadcast_sse_message({"type": "success", "message": "Preset created"})
        else:
            await self.state.put_sse_message(client_id, {"type": "error", "message": "Failed to create preset"})
        return {"status": "ok"}

    async def _update_preset(self, client_id: str, data: Dict[str, Any], _payload: ApiActionRequest):
        preset_id = data.get("preset_id", "")
        preset_data = data.get("preset", {})
        if not preset_id or not preset_data:
            raise ValueError("preset_id and preset data are required")

        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(
            self.state.executor,
            self.state.preset_manager.update_preset,
            preset_id,
            preset_data.get("description"),
            preset_data.get("system_prompt"),
            preset_data.get("parameters"),
        )

        if success:
            await self.state.broadcast_sse_message({"type": "success", "message": "Preset updated"})
        else:
            await self.state.put_sse_message(client_id, {"type": "error", "message": "Failed to update preset"})
        return {"status": "ok"}

    async def _delete_preset(self, client_id: str, data: Dict[str, Any], _payload: ApiActionRequest):
        preset_id = data.get("preset_id") or data.get("name", "")
        if not preset_id:
            raise ValueError("preset_id or name is required")

        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(self.state.executor, self.state.preset_manager.delete_preset, preset_id)
        if success:
            await self.state.broadcast_sse_message({"type": "success", "message": "Preset deleted"})
        else:
            await self.state.put_sse_message(client_id, {"type": "error", "message": "Failed to delete preset"})
        return {"status": "ok"}

    async def _create_session(self, client_id: str, _data: Dict[str, Any], payload: ApiActionRequest):
        title = payload.title or "New Chat"
        loop = asyncio.get_running_loop()
        new_session = await loop.run_in_executor(self.state.executor, self.state.history_manager.create_session, title)
        await self.state.put_sse_message(client_id, {"type": "session_created", "data": new_session})
        return {"data": new_session}

    async def _rename_session(self, client_id: str, _data: Dict[str, Any], payload: ApiActionRequest):
        conv_id = payload.conversation_id
        title = payload.title
        if not conv_id or not title:
            raise ValueError("conversation_id and title are required")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.state.executor, self.state.history_manager.rename_session, conv_id, title)
        await self.state.put_sse_message(
            client_id,
            {
                "type": "session_renamed",
                "conversation_id": conv_id,
                "title": title,
                "data": {"id": conv_id, "title": title},
            },
        )

        sessions = await loop.run_in_executor(self.state.executor, self.state.history_manager.get_all_sessions)
        await self.state.put_sse_message(client_id, {"type": "sessions_list", "data": sessions})
        return {"status": "ok"}

    async def _delete_session(self, client_id: str, _data: Dict[str, Any], payload: ApiActionRequest):
        conv_id = payload.conversation_id
        if not conv_id:
            raise ValueError("conversation_id is required")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.state.executor, self.state.history_manager.delete_session, conv_id)
        await self.state.put_sse_message(
            client_id,
            {"type": "session_deleted", "conversation_id": conv_id, "data": {"id": conv_id}},
        )

        sessions = await loop.run_in_executor(self.state.executor, self.state.history_manager.get_all_sessions)
        await self.state.put_sse_message(client_id, {"type": "sessions_list", "data": sessions})
        return {"status": "ok"}

    async def _get_chat_history(self, client_id: str, _data: Dict[str, Any], payload: ApiActionRequest):
        conv_id = payload.conversation_id
        loop = asyncio.get_running_loop()
        messages = await loop.run_in_executor(self.state.executor, self.state.history_manager.get_chat_history, conv_id)
        await self.state.put_sse_message(client_id, {"type": "chat_history", "conversation_id": conv_id, "data": messages})
        return {"conversation_id": conv_id, "data": messages}
