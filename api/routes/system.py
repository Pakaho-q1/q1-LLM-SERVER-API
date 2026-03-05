from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter

from api.dependencies import REQUIRE_API_KEY
from api.runtime import app_state

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    active_downloads = [
        j for j in app_state.model_manager.download_manager.get_jobs() if j.get("status") == "downloading"
    ]
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "loaded": bool(app_state.llm_engine.llm),
            "name": app_state.llm_engine.model_name or None,
            "n_ctx": app_state.llm_engine.n_ctx,
        },
        "sse_connections": len(app_state.sse_queues),
        "active_downloads": len(active_downloads),
    }


@router.get("/readiness")
async def readiness_check() -> Dict[str, Any]:
    return {
        "status": "ready",
        "auth_required": REQUIRE_API_KEY,
        "models_dir": str(app_state.model_manager.models_dir),
    }
