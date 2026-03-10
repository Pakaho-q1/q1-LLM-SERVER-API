import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import verify_api_key
from api.runtime import app_state

router = APIRouter(dependencies=[Depends(verify_api_key)])


class PresetBody(BaseModel):
    name: str
    description: str = ""
    system_prompt: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)


class CreatePresetRequest(BaseModel):
    preset: PresetBody


class UpdatePresetBody(BaseModel):
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class UpdatePresetRequest(BaseModel):
    preset: UpdatePresetBody


@router.get("/api/presets")
async def list_presets():
    loop = asyncio.get_running_loop()
    presets = await loop.run_in_executor(
        app_state.executor,
        app_state.preset_manager.list_presets,
    )
    return {"data": presets or []}


@router.get("/api/presets/{name}")
async def get_preset(name: str):
    loop = asyncio.get_running_loop()
    preset = await loop.run_in_executor(
        app_state.executor,
        app_state.preset_manager.get_preset,
        name,
    )
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")
    return {"data": preset}


@router.post("/api/presets")
async def create_preset(payload: CreatePresetRequest):
    p = payload.preset
    if not p.name:
        raise HTTPException(status_code=400, detail="preset.name is required")
    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(
        app_state.executor,
        app_state.preset_manager.create_preset,
        p.name,
        p.description,
        p.system_prompt,
        p.parameters,
    )
    if not success:
        raise HTTPException(status_code=409, detail="Failed to create preset")
    return {"status": "ok", "data": {"name": p.name}}


@router.put("/api/presets/{name}")
async def update_preset(name: str, payload: UpdatePresetRequest):
    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(
        app_state.executor,
        app_state.preset_manager.update_preset,
        name,
        payload.preset.description,
        payload.preset.system_prompt,
        payload.preset.parameters,
    )
    if not success:
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")
    return {"status": "ok", "data": {"name": name}}


@router.delete("/api/presets/{name}")
async def delete_preset(name: str):
    loop = asyncio.get_running_loop()
    success = await loop.run_in_executor(
        app_state.executor,
        app_state.preset_manager.delete_preset,
        name,
    )
    if not success:
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")
    return {"status": "ok", "data": {"name": name}}

