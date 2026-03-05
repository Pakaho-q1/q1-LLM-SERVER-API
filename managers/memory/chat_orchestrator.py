import logging
import asyncio
import re
from typing import Dict, Any, Callable, List, Optional


logger = logging.getLogger(__name__)


class ChatOrchestrator:
    def __init__(self, engine, history, executor, retrieval_service=None):
        self.engine = engine
        self.history = history
        self.executor = executor

        self.retrieval_service = retrieval_service
        self._cached_engine = None
        self._is_stream = False
        self._current_engine_id = id(engine)

    def _resolve_engine(self):
        """วิเคราะห์และ Cache ความสามารถของ Engine แบบเรียกครั้งเดียว"""
        if hasattr(self.engine, "completion_stream"):
            self._cached_engine = self.engine.completion_stream
            self._is_stream = True
        else:
            found = False
            for attr_name in dir(self.engine):
                try:
                    attr = getattr(self.engine, attr_name)
                    if hasattr(attr, "completion_stream"):
                        self._cached_engine = attr.completion_stream
                        self._is_stream = True
                        found = True
                        break
                except Exception:
                    continue

            if not found:
                self._cached_engine = getattr(self.engine, "completion", None)
                self._is_stream = False

        if not self._cached_engine:
            raise AttributeError(
                "Neither 'completion_stream' nor 'completion' found on engine."
            )

    def unload_model(self):
        """ล้าง Cache เมื่อปิดหรือเปลี่ยนโมเดล"""
        self.retrieval_service = retrieval_service
        self._cached_engine = None
        self._is_stream = False

    def _stabilize_input(self, user_input: str) -> None:
        """ตรวจสอบและจำกัดปริมาณ Token เพื่อป้องกันระบบค้าง (OOM)"""
        if not user_input or not user_input.strip():
            raise ValueError("Input content cannot be empty")

        token_count = 0
        try:
            if hasattr(self.engine, "count_tokens"):
                token_count = self.engine.count_tokens(user_input)
            elif hasattr(self.engine, "llm") and hasattr(self.engine.llm, "tokenize"):
                token_count = len(self.engine.llm.tokenize(user_input.encode("utf-8")))
            else:
                token_count = len(user_input) // 3
        except Exception:
            token_count = len(user_input.split()) * 1.5

        n_ctx = getattr(self.engine, "n_ctx", 4096)
        limit = n_ctx * 0.6

        if token_count > limit:
            raise ValueError(
                f"Input too large: Estimated {int(token_count)} tokens. "
                f"Limit is {int(limit)} to leave room for memory and AI response."
            )

    async def _build_context(
        self,
        conv_id: str,
        user_input: str,
        params: Dict[str, Any],
        status_cb: Callable,
    ) -> List[Dict[str, Any]]:
        """รวบรวมข้อความ ประวัติ และความจำต่างๆ ตามลำดับเวลา"""
        loop = asyncio.get_running_loop()
        system_prompt = params.get("system_prompt", "You are a precise AI assistant.")

        await status_cb("Searching long-term memory...")
        long_term_memories = await loop.run_in_executor(
            self.executor,
            lambda: self.history.search_memories(
                user_input, conv_id, limit=5, score_threshold=0.25
            ),
        )

        await status_cb("Building context...")
        context_history = await loop.run_in_executor(
            self.executor,
            lambda: self.history.finalize_context(
                conv_id=conv_id,
                options=params,
                user_input=user_input,
                system_tokens_estimate=self._estimate_system_tokens(system_prompt),
                response_tokens_estimate=params.get("max_tokens"),
                long_term_memories=long_term_memories,
            ),
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context_history)
        messages.append({"role": "user", "content": user_input})

        return messages

    def _estimate_system_tokens(self, system_prompt: str) -> int:
        base = len(system_prompt) // 4
        return base + 50

    def _build_grounded_messages(self, user_input: str, params: Dict[str, Any]):
        if not self.retrieval_service or not params.get("use_rag"):
            return None
        retrieved = self.retrieval_service.retrieve(
            user_input,
            limit=int(params.get("rag_top_k", 6)),
            use_hybrid=bool(params.get("rag_hybrid", True)),
            use_rerank=bool(params.get("rag_rerank", True)),
            score_threshold=float(params.get("rag_score_threshold", 0.18)),
        )
        grounded = self.retrieval_service.build_grounded_prompt(user_input, retrieved)
        return grounded

    async def process_chat(
        self,
        conv_id: str,
        user_input: str,
        params: Dict[str, Any],
        status_cb: Callable,
        chunk_cb: Callable,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        try:

            if self._cached_engine is None or self._current_engine_id != id(
                self.engine
            ):
                self._resolve_engine()
                self._current_engine_id = id(self.engine)

            self._stabilize_input(user_input)

            if messages is None:
                messages = await self._build_context(
                    conv_id, user_input, params, status_cb
                )

            grounded = self._build_grounded_messages(user_input, params)
            if grounded is not None:
                if not grounded.get("should_answer"):
                    refusal = "I don't have enough evidence in the provided sources."
                    await chunk_cb(refusal)
                    self.history.add_message(conv_id, "user", user_input)
                    self.history.add_message(conv_id, "assistant", refusal)
                    return

                rag_system = {
                    "role": "system",
                    "content": grounded.get("system_prompt", "") + "\n\nEvidence:\n" + grounded.get("context", ""),
                }
                messages = [rag_system] + [m for m in messages if m.get("role") != "system"]

            await status_cb("Generating response...")
            full_text = ""
            parameters = params.get("parameters", params)
            loop = asyncio.get_running_loop()

            if self._is_stream:
                sync_stream = self._cached_engine(messages, parameters)

                def _next():
                    try:
                        return next(sync_stream)
                    except StopIteration:
                        return None

                while True:
                    chunk = await loop.run_in_executor(self.executor, _next)
                    if chunk is None:
                        break
                    if chunk:
                        full_text += chunk
                        await chunk_cb(chunk)
            else:
                full_text = await loop.run_in_executor(
                    self.executor, lambda: self._cached_engine(messages, parameters)
                )
                if full_text:
                    await chunk_cb(full_text)

            clean_text = re.sub(
                r"<think>.*?</think>", "", full_text, flags=re.DOTALL
            ).strip()

            await status_cb("Saving to memory...")

            self.history.add_message(conv_id, "user", user_input)
            self.history.add_message(conv_id, "assistant", clean_text)

            interaction = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": clean_text},
            ]

            if self.history.should_store_long_term(interaction):
                self.executor.submit(self.history.save_to_memory, interaction, conv_id)

            self.executor.submit(self.history.update_rolling_summary, conv_id)

        except Exception as e:

            logger.exception(
                f"ChatOrchestrator encountered an error in conversation '{conv_id}':"
            )
            raise
