import os

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = os.environ.get("LLM_API_KEY", "")
REQUIRE_API_KEY = os.environ.get("LLM_REQUIRE_API_KEY", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str = Depends(api_key_header)):
    if not API_KEY:
        return
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def ensure_auth_configuration() -> None:
    if REQUIRE_API_KEY and not API_KEY:
        raise RuntimeError(
            "LLM_API_KEY is required when LLM_REQUIRE_API_KEY=true. "
            "Set LLM_API_KEY or explicitly disable with LLM_REQUIRE_API_KEY=false."
        )
