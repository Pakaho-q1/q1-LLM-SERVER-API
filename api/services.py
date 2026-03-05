import asyncio
import logging
from typing import Any, Dict

from api.runtime import (
    executor,
    llm_engine,
    model_manager,
    preset_manager,
    put_sse_message,
    sse_broadcast,
)

logger = logging.getLogger(__name__)


def safe_error_detail(message: str = "Internal server error") -> str:
    return message


async def handle_list_models(client_id: str) -> None:
    try:
        loop = asyncio.get_running_loop()
        models = await loop.run_in_executor(executor, model_manager.list_models)
        await put_sse_message(client_id, {"type": "models_list", "data": models or []})
    except Exception as e:
        logger.error("List models error: %s", e, exc_info=True)
        await put_sse_message(client_id, {"type": "error", "message": str(e)})


async def handle_load_model(client_id: str, data: Dict[str, Any]) -> None:
    try:
        model_name = data.get("model_path", "")
        if not model_name:
            raise ValueError("model_path is required")

        params = data.get("params", {})
        safe_model_path = model_manager.resolve_model_path(model_name)
        if safe_model_path is None:
            raise ValueError("invalid model_path")

        await put_sse_message(client_id, {"type": "status", "message": f"Loading: {model_name}..."})

        loop = asyncio.get_running_loop()
        success, msg = await loop.run_in_executor(
            executor, lambda: llm_engine.load_model(str(safe_model_path), **(params or {}))
        )

        if success:
            await sse_broadcast({"type": "success", "message": f"Model loaded: {model_name}"})
            await sse_broadcast({"type": "model_status", "data": {"running": True, "name": model_name}})
        else:
            await put_sse_message(client_id, {"type": "error", "message": msg})
    except Exception as e:
        logger.error("Load model error: %s", e, exc_info=True)
        await put_sse_message(client_id, {"type": "error", "message": str(e)})


async def handle_unload_model(client_id: str) -> None:
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, llm_engine.unload_model)
        await sse_broadcast({"type": "success", "message": "Model unloaded"})
        await sse_broadcast({"type": "model_status", "data": {"running": False, "name": ""}})
    except Exception as e:
        logger.error("Unload error: %s", e, exc_info=True)
        await put_sse_message(client_id, {"type": "error", "message": str(e)})


async def handle_delete_model(client_id: str, data: Dict[str, Any]) -> None:
    try:
        filename = data.get("filename", "")
        if not filename:
            raise ValueError("filename is required")
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(executor, model_manager.delete_model, filename)
        if success:
            await sse_broadcast({"type": "success", "message": f"Deleted: {filename}"})
            await handle_list_models(client_id)
        else:
            await put_sse_message(client_id, {"type": "error", "message": "Failed to delete model"})
    except Exception as e:
        logger.error("Delete error: %s", e, exc_info=True)
        await put_sse_message(client_id, {"type": "error", "message": str(e)})


async def handle_fetch_hf(client_id: str, data: Dict[str, Any]) -> None:
    try:
        repo = data.get("repo", "")
        if not repo:
            raise ValueError("repo is required")
        loop = asyncio.get_running_loop()
        files = await loop.run_in_executor(executor, model_manager.fetch_hf_repo, repo)
        await put_sse_message(client_id, {"type": "hf_files", "data": files or []})
    except Exception as e:
        logger.error("HF fetch error: %s", e, exc_info=True)
        await put_sse_message(client_id, {"type": "error", "message": str(e)})


async def handle_download_model(client_id: str, data: Dict[str, Any]) -> None:
    try:
        url = data.get("url", "")
        if not url:
            raise ValueError("url is required")
        loop = asyncio.get_running_loop()
        job_id = await loop.run_in_executor(executor, model_manager.download_async, url)
        await put_sse_message(client_id, {"type": "success", "message": "Download started", "job_id": job_id})
    except Exception as e:
        logger.error("Download error: %s", e, exc_info=True)
        await put_sse_message(client_id, {"type": "error", "message": str(e)})


async def handle_cancel_download(client_id: str, data: Dict[str, Any]) -> None:
    try:
        job_id = data.get("job_id", "")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, model_manager.download_manager.cancel, job_id)
        await sse_broadcast({"type": "success", "message": "Download cancelled"})
    except Exception as e:
        logger.error("Cancel error: %s", e, exc_info=True)


async def handle_download_status(client_id: str) -> None:
    try:
        loop = asyncio.get_running_loop()
        jobs = await loop.run_in_executor(executor, model_manager.download_manager.get_jobs)
        jobs_list = [{
            "id": j.get("id"),
            "filename": j.get("filename"),
            "progress": j.get("progress"),
            "speed": j.get("speed"),
            "eta": j.get("eta"),
            "status": j.get("status"),
            "error": j.get("error"),
        } for j in jobs]
        await put_sse_message(client_id, {"type": "download_status", "data": jobs_list})
    except Exception as e:
        logger.error("Status error: %s", e, exc_info=True)


async def handle_count_tokens(client_id: str, data: Dict[str, Any]) -> None:
    try:
        text = data.get("text", "")
        loop = asyncio.get_running_loop()
        count = await loop.run_in_executor(executor, llm_engine.count_tokens, text)
        await put_sse_message(client_id, {"type": "token_count", "data": count})
    except Exception as e:
        logger.error("Token count error: %s", e, exc_info=True)


async def handle_list_presets(client_id: str) -> None:
    try:
        loop = asyncio.get_running_loop()
        presets = await loop.run_in_executor(executor, preset_manager.list_presets)
        await put_sse_message(client_id, {"type": "presets", "data": presets or []})
    except Exception as e:
        logger.error("List presets error: %s", e, exc_info=True)
        await put_sse_message(client_id, {"type": "error", "message": str(e)})


async def handle_get_preset(client_id: str, data: Dict[str, Any]) -> None:
    try:
        preset_id = data.get("preset_id") or data.get("name", "")
        if not preset_id:
            raise ValueError("preset_id or name is required")
        loop = asyncio.get_running_loop()
        preset = await loop.run_in_executor(executor, preset_manager.get_preset, preset_id)
        await put_sse_message(client_id, {"type": "preset_data", "data": preset})
    except Exception as e:
        logger.error("Get preset error: %s", e, exc_info=True)
        await put_sse_message(client_id, {"type": "error", "message": str(e)})


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
            await put_sse_message(client_id, {"type": "error", "message": "Failed to create preset"})
    except Exception as e:
        logger.error("Create preset error: %s", e, exc_info=True)
        await put_sse_message(client_id, {"type": "error", "message": str(e)})


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
            await put_sse_message(client_id, {"type": "error", "message": "Failed to update preset"})
    except Exception as e:
        logger.error("Update preset error: %s", e, exc_info=True)
        await put_sse_message(client_id, {"type": "error", "message": str(e)})


async def handle_delete_preset(client_id: str, data: Dict[str, Any]) -> None:
    try:
        preset_id = data.get("preset_id") or data.get("name", "")
        if not preset_id:
            raise ValueError("preset_id or name is required")
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(executor, preset_manager.delete_preset, preset_id)
        if success:
            await sse_broadcast({"type": "success", "message": "Preset deleted"})
        else:
            await put_sse_message(client_id, {"type": "error", "message": "Failed to delete preset"})
    except Exception as e:
        logger.error("Delete preset error: %s", e, exc_info=True)
        await put_sse_message(client_id, {"type": "error", "message": str(e)})


async def handle_get_model_status(client_id: str) -> None:
    try:
        running = bool(llm_engine.llm)
        name = llm_engine.model_name or ""
        await put_sse_message(client_id, {"type": "model_status", "data": {"running": running, "name": name}})
    except Exception as e:
        logger.error("Get model status error: %s", e, exc_info=True)
        await put_sse_message(client_id, {"type": "error", "message": str(e)})
