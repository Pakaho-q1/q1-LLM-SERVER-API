import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from slowapi import Limiter
from slowapi.util import get_remote_address

from managers.chat_orchestrator import ChatOrchestrator
from managers.history_manager import HistoryManager
from managers.llm_core import LLMEngine
from managers.model_manager import ModelManager
from managers.preset_manager import PresetManager

logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="worker_")

project_root = Path(__file__).resolve().parent.parent
models_dir = project_root / "models"

llm_engine = LLMEngine()
model_manager = ModelManager(models_dir=str(models_dir))
preset_manager = PresetManager()

history_manager = HistoryManager(
    engine=llm_engine,
    db_path="data/chat.db",
    n_ctx=llm_engine.n_ctx,
    tokenizer_fn=llm_engine.count_tokens,
)

chat_orchestrator = ChatOrchestrator(llm_engine, history_manager, executor)

limiter = Limiter(key_func=get_remote_address)

sse_queues: Dict[str, asyncio.Queue] = {}
sse_lock = asyncio.Lock()


async def sse_broadcast(message: Dict[str, Any]) -> None:
    async with sse_lock:
        queues = list(sse_queues.items())
    for client_id, q in queues:
        try:
            await q.put(message)
        except Exception:
            logger.exception("Failed to broadcast to %s", client_id)


async def put_sse_message(client_id: str, message: Dict[str, Any]) -> None:
    async with sse_lock:
        q = sse_queues.get(client_id)
    if q:
        try:
            await q.put(message)
        except Exception:
            logger.exception("Failed to put SSE message")
