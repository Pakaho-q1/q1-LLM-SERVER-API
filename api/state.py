import asyncio
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

from managers.memory.chat_orchestrator import ChatOrchestrator
from managers.memory.history_manager import HistoryManager
from managers.memory.ingestion import IngestionService, LocalFileStore, MinioFileStore, S3FileStore
from managers.memory.retrieval import RetrievalService
from managers.llm_core import LLMEngine
from managers.model_manager import ModelManager
from managers.preset_manager import PresetManager

logger = logging.getLogger(__name__)


class AppState:
    """Centralized state/logic holder for API runtime dependencies."""

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="worker_")

        project_root = Path(__file__).resolve().parent.parent
        self.models_dir = project_root / "models"

        self.llm_engine = LLMEngine()
        self.model_manager = ModelManager(models_dir=str(self.models_dir))
        self.preset_manager = PresetManager()

        self.history_manager = HistoryManager(
            engine=self.llm_engine,
            db_path="data/chat.db",
            n_ctx=self.llm_engine.n_ctx,
            tokenizer_fn=self.llm_engine.count_tokens,
        )

        self.retrieval_service = RetrievalService(db_path="data/rag.db")

        self.chat_orchestrator = ChatOrchestrator(
            self.llm_engine, self.history_manager, self.executor, self.retrieval_service
        )

        store_backend = os.environ.get("FILE_STORE_BACKEND", "local").lower()
        if store_backend == "s3":
            file_store = S3FileStore()
        elif store_backend == "minio":
            file_store = MinioFileStore()
        else:
            file_store = LocalFileStore("data/uploads")

        self.ingestion_service = IngestionService(
            retriever=self.retrieval_service,
            store=file_store,
        )

        self.sse_queues: Dict[str, asyncio.Queue] = {}
        self.sse_lock = asyncio.Lock()

    async def register_sse_client(self, client_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        async with self.sse_lock:
            self.sse_queues[client_id] = queue
        return queue

    async def unregister_sse_client(self, client_id: str) -> None:
        async with self.sse_lock:
            self.sse_queues.pop(client_id, None)

    async def get_sse_queue(self, client_id: str) -> Optional[asyncio.Queue]:
        async with self.sse_lock:
            return self.sse_queues.get(client_id)

    async def put_sse_message(self, client_id: str, message: Dict[str, Any]) -> None:
        queue = await self.get_sse_queue(client_id)
        if queue is None:
            return

        try:
            await queue.put(message)
        except Exception:
            logger.exception("Failed to put SSE message")

    async def broadcast_sse_message(self, message: Dict[str, Any]) -> None:
        async with self.sse_lock:
            queues = list(self.sse_queues.items())

        for client_id, queue in queues:
            try:
                await queue.put(message)
            except Exception:
                logger.exception("Failed to broadcast to %s", client_id)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)
