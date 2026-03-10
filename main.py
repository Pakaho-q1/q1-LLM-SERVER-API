import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import ensure_auth_configuration
from api.routes.models import router as models_router
from api.routes.openai import router as openai_router
from api.routes.presets import router as presets_router
from api.routes.rag import router as rag_router
from api.routes.sessions import router as sessions_router
from api.routes.sse import router as sse_router
from api.routes.system import router as system_router
from api.runtime import app_state

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("uvicorn.access").disabled = True
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting...")
    ensure_auth_configuration()
    yield
    logger.info("Server shutting down...")
    app_state.shutdown()


app = FastAPI(title="LLM WebUI API", version="1.1.0", lifespan=lifespan)

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


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


def _openai_error_payload(message: str, error_type: str, code: str | None = None):
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if request.url.path.startswith("/v1/"):
        msg = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        if exc.status_code == 401:
            err_type = "authentication_error"
            err_code = "invalid_api_key"
        elif exc.status_code >= 500:
            err_type = "server_error"
            err_code = None
        else:
            err_type = "invalid_request_error"
            err_code = None
        return JSONResponse(
            status_code=exc.status_code,
            content=_openai_error_payload(
                msg,
                err_type,
                code=err_code,
            ),
        )

    message = exc.detail if isinstance(exc.detail, str) else "Request failed"
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": message,
            "detail": exc.detail,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/v1/"):
        return JSONResponse(
            status_code=422,
            content=_openai_error_payload("Validation error", "invalid_request_error", code="validation_error"),
        )

    return JSONResponse(
        status_code=422,
        content={
            "message": "Validation error",
            "detail": exc.errors(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error")
    if request.url.path.startswith("/v1/"):
        return JSONResponse(
            status_code=500,
            content=_openai_error_payload("Internal server error", "server_error"),
        )

    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


app.include_router(system_router)
app.include_router(sessions_router)
app.include_router(models_router)
app.include_router(presets_router)
app.include_router(openai_router)
app.include_router(sse_router)
app.include_router(rag_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning", access_log=False)
