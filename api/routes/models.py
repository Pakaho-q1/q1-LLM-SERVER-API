import asyncio
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import verify_api_key
from api.runtime import app_state

router = APIRouter(dependencies=[Depends(verify_api_key)])


class LoadModelRequest(BaseModel):
    model_path: str
    params: Dict[str, Any] = Field(default_factory=dict)


class DownloadModelRequest(BaseModel):
    url: str


class CancelDownloadRequest(BaseModel):
    job_id: str


@router.get("/api/models")
async def list_models():
    loop = asyncio.get_running_loop()
    models = await loop.run_in_executor(
        app_state.executor,
        app_state.model_manager.list_models,
    )
    return {"data": models or []}


@router.post("/api/models/load")
async def load_model(payload: LoadModelRequest):
    model_name = payload.model_path
    if not model_name:
        raise HTTPException(status_code=400, detail="model_path is required")

    safe_model_path = app_state.model_manager.resolve_model_path(model_name)
    if safe_model_path is None:
        raise HTTPException(status_code=400, detail="invalid model_path")

    load_plan = app_state.model_manager.build_load_plan(model_name, payload.params or {})

    loop = asyncio.get_running_loop()
    success, message = await loop.run_in_executor(
        app_state.executor,
        lambda: app_state.llm_engine.load_model(
            str(safe_model_path), **(load_plan.get("params") or {})
        ),
    )
    if not success:
        raise HTTPException(status_code=500, detail=message)

    return {
        "status": "ok",
        "data": {
            "name": model_name,
            "vision_enabled": bool(load_plan.get("vision_enabled")),
            "mmproj": load_plan.get("mmproj"),
            "mmproj_score": load_plan.get("mmproj_score"),
            "mmproj_reason": load_plan.get("mmproj_reason"),
        },
    }


@router.post("/api/models/unload")
async def unload_model():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(app_state.executor, app_state.llm_engine.unload_model)
    return {"status": "ok"}


@router.delete("/api/models/{filename}")
async def delete_model(filename: str):
    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(
        app_state.executor,
        app_state.model_manager.delete_model,
        filename,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Failed to delete model")
    return {"status": "ok", "data": {"filename": filename}}


@router.get("/api/models/hf")
async def fetch_hf(repo: str):
    if not repo:
        raise HTTPException(status_code=400, detail="repo is required")
    loop = asyncio.get_running_loop()
    files = await loop.run_in_executor(
        app_state.executor,
        app_state.model_manager.fetch_hf_repo,
    )
    return {"data": files or []}


@router.post("/api/models/downloads")
async def download_model(payload: DownloadModelRequest):
    if not payload.url:
        raise HTTPException(status_code=400, detail="url is required")
    loop = asyncio.get_running_loop()
    job_id = await loop.run_in_executor(
        app_state.executor,
        app_state.model_manager.download_async,
        payload.url,
    )
    return {"status": "ok", "data": {"job_id": job_id}}


@router.get("/api/models/downloads")
async def download_status():
    loop = asyncio.get_running_loop()
    jobs = await loop.run_in_executor(
        app_state.executor,
        app_state.model_manager.download_manager.get_jobs,
    )
    jobs_list: List[Dict[str, Any]] = [
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
    return {"data": jobs_list}


@router.post("/api/models/downloads/cancel")
async def cancel_download(payload: CancelDownloadRequest):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        app_state.executor,
        app_state.model_manager.download_manager.cancel,
        payload.job_id,
    )
    return {"status": "ok"}


@router.get("/api/models/status")
async def model_status():
    running = bool(app_state.llm_engine.is_loaded())
    name = app_state.llm_engine.model_name or ""
    return {
        "running": running,
        "loading": False,
        "name": name,
        "model": name,
        "vision_enabled": bool(app_state.llm_engine.multimodal_enabled),
        "mmproj": app_state.llm_engine.mmproj_path or "",
        "chat_format": app_state.llm_engine.chat_format or "",
    }


class CountTokensRequest(BaseModel):
    text: str


@router.post("/api/tokens/count")
async def count_tokens(payload: CountTokensRequest):
    loop = asyncio.get_running_loop()
    count = await loop.run_in_executor(
        app_state.executor,
        app_state.llm_engine.count_tokens,
        payload.text or "",
    )
    return {"data": count}
