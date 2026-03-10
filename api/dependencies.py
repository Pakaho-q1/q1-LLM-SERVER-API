import os

from fastapi import Depends, HTTPException, Query, Request
from fastapi.security import APIKeyHeader
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

API_KEY = os.environ.get("LLM_API_KEY", "")
REQUIRE_API_KEY = os.environ.get("LLM_REQUIRE_API_KEY", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


def _extract_api_key(
    request: Request,
    key_header: str | None,
    bearer: HTTPAuthorizationCredentials | None,
    query_key: str | None,
) -> str | None:
    if key_header:
        return key_header.strip()

    if bearer and bearer.scheme.lower() == "bearer" and bearer.credentials:
        return bearer.credentials.strip()

    if query_key:
        return query_key.strip()

    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()

    return None


async def verify_api_key(
    request: Request,
    key_header: str = Depends(api_key_header),
    bearer: HTTPAuthorizationCredentials = Depends(bearer_auth),
    query_key: str | None = Query(default=None, alias="api_key"),
):
    if not REQUIRE_API_KEY:
        return

    provided_key = _extract_api_key(request, key_header, bearer, query_key)
    if provided_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def ensure_auth_configuration() -> None:
    if REQUIRE_API_KEY and not API_KEY:
        raise RuntimeError(
            "LLM_API_KEY is required when LLM_REQUIRE_API_KEY=true. "
            "Set LLM_API_KEY or explicitly disable with LLM_REQUIRE_API_KEY=false."
        )
