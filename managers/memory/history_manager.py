import sqlite3
import json
import re
import uuid
import time
import hashlib
import math
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Set

from mem0 import Memory
from mem0.llms.base import LLMBase

logger = logging.getLogger(__name__)
RECENCY_HALF_LIFE_DAYS = 30
MIN_EPISODIC_LENGTH = 20
MIN_LTM_USER_LENGTH = 30
OPENAI_FILES_DIR = Path(__file__).resolve().parents[2] / "data" / "openai_files"


class MyLocalLLM(LLMBase):
    def __init__(self, engine):
        self.engine = engine

    def _get_core_engine(self):
        if hasattr(self.engine, "generate_json"):
            return self.engine
        if hasattr(self.engine, "engine") and hasattr(
            self.engine.engine, "generate_json"
        ):
            return self.engine.engine
        if hasattr(self.engine, "llm_engine"):
            return self.engine.llm_engine
        return self.engine

    def generate_response(self, messages: list, **kwargs) -> str:
        core = self._get_core_engine()
        if not hasattr(core, "llm") or core.llm is None:
            return ""
        params = {"max_tokens": kwargs.get("max_tokens", 512), "temperature": 0.0}
        return core.generate_json(messages, params)


class HistoryManager:
    def __init__(
        self,
        engine,
        db_path: str = "data/chat.db",
        n_ctx: int = 4096,
        tokenizer_fn: Callable = None,
    ):
        self.engine = engine
        self.db_path = db_path
        self.n_ctx = n_ctx
        self.tokenizer = tokenizer_fn
        self.safety_margin = 350
        self._local = threading.local()
        self._init_db()
        self._init_memory()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                title      TEXT,
                summary    TEXT DEFAULT '',  -- เพิ่มบรรทัดนี้เพื่อเก็บสรุปแชทเก่าๆ
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id          TEXT PRIMARY KEY,
                conv_id     TEXT NOT NULL,
                role        TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                content     TEXT NOT NULL,
                tokens      INTEGER DEFAULT 0,
                request_id  TEXT UNIQUE,
                message_id  TEXT,
                metadata    TEXT DEFAULT '{}',
                is_archived INTEGER DEFAULT 0,
                created_at  REAL NOT NULL,
                FOREIGN KEY (conv_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conv_time
            ON messages (conv_id, created_at ASC)
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodic_facts (
                id               TEXT PRIMARY KEY,
                conv_id          TEXT NOT NULL,
                content          TEXT NOT NULL,
                content_hash     TEXT NOT NULL,
                importance_score REAL DEFAULT 0.5,
                created_at       REAL NOT NULL,
                is_superseded    INTEGER DEFAULT 0,
                superseded_by    TEXT
            )
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_episodic_conv
            ON episodic_facts (conv_id, created_at ASC)
        """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_episodic_hash
            ON episodic_facts (conv_id, content_hash)
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_memories (
                id            TEXT PRIMARY KEY,
                conv_id       TEXT NOT NULL,
                content       TEXT NOT NULL,
                content_hash  TEXT NOT NULL,
                created_at    REAL NOT NULL,
                updated_at    REAL NOT NULL,
                is_superseded INTEGER DEFAULT 0,
                superseded_by TEXT,
                category      TEXT DEFAULT 'general'
            )
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_semantic_conv
            ON semantic_memories (conv_id, created_at ASC)
        """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_semantic_hash
            ON semantic_memories (conv_id, content_hash)
        """
        )
        conn.commit()

    def _init_memory(self):
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "chat_memory",
                    "path": "data/mem0_qdrant",
                    "embedding_model_dims": 384,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            },
            "llm": {
                "provider": "openai",
                "config": {"api_key": "dummy_key", "model": "gpt-3.5-turbo"},
            },
        }
        self.memory = Memory.from_config(config)
        self.memory.llm = MyLocalLLM(self.engine)

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return self.tokenizer(text)
        return max(1, len(text) // 4)

    def update_rolling_summary(self, conv_id: str, threshold_tokens: int = 5000):
        """
        อัปเดตสรุปประวัติแชทเมื่อ Token ดิบเกินกำหนด (เช่น 5000 Token)
        """
        conn = self._get_conn()

        session = conn.execute(
            "SELECT summary FROM sessions WHERE id=?", (conv_id,)
        ).fetchone()
        current_summary = session["summary"] if session and session["summary"] else ""

        messages = conn.execute(
            """
            SELECT id, role, content, tokens FROM messages 
            WHERE conv_id=? AND is_archived=0 
            ORDER BY created_at ASC
            """,
            (conv_id,),
        ).fetchall()

        if not messages:
            return

        total_tokens = sum(
            m["tokens"] or self._count_tokens(m["content"]) for m in messages
        )

        if total_tokens < threshold_tokens:
            return

        keep_tokens_limit = 1500
        kept_tokens = 0
        keep_count = 0

        for m in reversed(messages):
            tok = m["tokens"] or self._count_tokens(m["content"])
            if kept_tokens + tok > keep_tokens_limit:
                break
            kept_tokens += tok
            keep_count += 1

        if keep_count < 2:
            keep_count = 2

        messages_to_summarize = messages[:-keep_count]
        if not messages_to_summarize:
            return

        ids_to_archive = [msg["id"] for msg in messages_to_summarize]

        chat_log = "\n".join(
            [
                f"{m['role'].capitalize()}: {self._clean_text(m['content'])}"
                for m in messages_to_summarize
            ]
        )

        summary_prompt = [
            {
                "role": "system",
                "content": """You compress chat history into a minimal, highly dense factual summary.

                Rules:
                - Keep ONLY core facts, user preferences, and key decisions.
                - COMPLETELY REMOVE assistant's explanations, greetings, and storytelling.
                - Use short, concise statements or keywords.
                - Do NOT repeat information already in the Current Summary.
                - Write the summary in the exact same language predominantly used by the user in the New messages.
                - Target length: As short as possible (< 120 tokens).

                Respond ONLY in valid JSON format:
                {"summary": "..."}""",
            },
            {
                "role": "user",
                "content": f"""Current summary:
                {current_summary}

                New messages:
                {chat_log}

                Generate the updated summary JSON:""",
            },
        ]

        raw_summary = self.engine.generate_json(summary_prompt, {"max_tokens": 500})

        new_summary = ""
        try:
            parsed_json = json.loads(raw_summary)
            new_summary = parsed_json.get("summary", "")
        except json.JSONDecodeError:
            match = re.search(r'"summary"\s*:\s*"(.*)', raw_summary, re.DOTALL)
            if match:
                new_summary = match.group(1).rstrip('"}')
            else:
                new_summary = raw_summary

        if new_summary and isinstance(new_summary, str):
            conn.execute(
                "UPDATE sessions SET summary=?, updated_at=? WHERE id=?",
                (new_summary.strip(), time.time(), conv_id),
            )
            placeholders = ",".join("?" for _ in ids_to_archive)
            conn.execute(
                f"UPDATE messages SET is_archived=1 WHERE id IN ({placeholders})",
                ids_to_archive,
            )
            conn.commit()
            logger.info(
                f"Updated rolling summary for conversation: {conv_id}. Compressed {len(messages_to_summarize)} messages."
            )

    @staticmethod
    def _clean_text(text: str) -> str:
        """ลบบรรทัดว่างที่ติดกันหลายบรรทัด และตัดช่องว่างหัวท้ายทิ้ง"""
        if not text:
            return ""

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]

    @staticmethod
    def _recency_score(created_at: float, now: float = None) -> float:
        """Exponential decay: 1.0 at creation, 0.5 after RECENCY_HALF_LIFE_DAYS."""
        now = now or time.time()
        age_days = (now - created_at) / 86400
        return math.exp(-math.log(2) * age_days / RECENCY_HALF_LIFE_DAYS)

    @staticmethod
    def _relative_time_label(created_at: float) -> str:
        """Human-readable age for prompt injection."""
        age_sec = time.time() - created_at
        if age_sec < 3600:
            return "just now"
        if age_sec < 86400:
            return f"{int(age_sec / 3600)}h ago"
        if age_sec < 86400 * 7:
            return f"{int(age_sec / 86400)}d ago"
        if age_sec < 86400 * 30:
            return f"{int(age_sec / 86400 / 7)}w ago"
        return f"{int(age_sec / 86400 / 30)}mo ago"

    def create_session(self, title: str = "New Chat") -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        now = time.time()
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
        conn.commit()
        return {"id": session_id, "title": title, "created_at": now, "updated_at": now}

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        rows = (
            self._get_conn()
            .execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            )
            .fetchall()
        )
        return [dict(r) for r in rows]

    def rename_session(self, conv_id: str, title: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
            (title, time.time(), conv_id),
        )
        conn.commit()

    @staticmethod
    def _extract_file_ids(metadata: Any) -> Set[str]:
        ids: Set[str] = set()
        if not isinstance(metadata, dict):
            return ids
        attachments = metadata.get("attachments")
        if not isinstance(attachments, list):
            return ids
        for item in attachments:
            if not isinstance(item, dict):
                continue
            file_id = item.get("file_id")
            if isinstance(file_id, str) and file_id.strip():
                ids.add(file_id.strip())
        return ids

    @staticmethod
    def _parse_metadata(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    def _delete_local_openai_files(self, file_ids: Set[str]) -> None:
        if not file_ids:
            return
        OPENAI_FILES_DIR.mkdir(parents=True, exist_ok=True)
        for file_id in file_ids:
            for p in OPENAI_FILES_DIR.glob(f"{file_id}*"):
                try:
                    if p.is_file():
                        p.unlink()
                except Exception:
                    logger.warning("Failed to delete file: %s", p, exc_info=True)

    def delete_session(self, conv_id: str) -> None:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT metadata FROM messages WHERE conv_id=?",
            (conv_id,),
        ).fetchall()
        candidate_file_ids: Set[str] = set()
        for row in rows:
            candidate_file_ids.update(self._extract_file_ids(self._parse_metadata(row["metadata"])))

        conn.execute("DELETE FROM episodic_facts WHERE conv_id=?", (conv_id,))
        conn.execute("DELETE FROM semantic_memories WHERE conv_id=?", (conv_id,))
        conn.execute("DELETE FROM sessions WHERE id=?", (conv_id,))
        conn.commit()

        if candidate_file_ids:
            other_rows = conn.execute(
                "SELECT metadata FROM messages WHERE conv_id<>?",
                (conv_id,),
            ).fetchall()
            still_used: Set[str] = set()
            for row in other_rows:
                still_used.update(self._extract_file_ids(self._parse_metadata(row["metadata"])))

            self._delete_local_openai_files(candidate_file_ids - still_used)

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        request_id: str = None,
        message_id: str = None,
        metadata: Dict = None,
    ) -> Dict[str, Any]:
        msg_id = message_id or str(uuid.uuid4())
        now = time.time()
        tokens = self._count_tokens(content)

        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO messages
                    (id, conv_id, role, content, tokens, request_id,
                     message_id, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    conv_id,
                    role,
                    content,
                    tokens,
                    request_id,
                    msg_id,
                    json.dumps(metadata or {}),
                    now,
                ),
            )
        except sqlite3.IntegrityError:
            pass
        else:
            conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, conv_id))
            conn.commit()

        return {
            "id": msg_id,
            "role": role,
            "content": content,
            "tokens": tokens,
            "created_at": now,
        }

    def get_chat_history(self, conv_id: str) -> List[Dict[str, Any]]:
        rows = (
            self._get_conn()
            .execute(
                """
            SELECT id, role, content, tokens, created_at, metadata
            FROM messages
            WHERE conv_id=? AND is_archived=0
            ORDER BY created_at ASC
            """,
                (conv_id,),
            )
            .fetchall()
        )
        result: List[Dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            item["metadata"] = self._parse_metadata(item.get("metadata"))
            result.append(item)
        return result

    def get_working_memory(self, conv_id: str, max_tokens: int) -> List[Dict[str, Any]]:

        rows = (
            self._get_conn()
            .execute(
                """
            SELECT role, content, tokens
            FROM messages
            WHERE conv_id=? AND is_archived=0
            ORDER BY created_at DESC
            """,
                (conv_id,),
            )
            .fetchall()
        )

        selected: List[Dict[str, Any]] = []
        token_count = 0
        i = 0

        while i < len(rows):
            msg = rows[i]

            if msg["role"] == "assistant":
                if i + 1 < len(rows) and rows[i + 1]["role"] == "user":
                    user_msg = rows[i + 1]

                    t_assistant = msg["tokens"] or self._count_tokens(msg["content"])
                    t_user = user_msg["tokens"] or self._count_tokens(
                        user_msg["content"]
                    )
                    pair_tokens = t_assistant + t_user

                    if token_count + pair_tokens <= max_tokens:

                        selected.append(
                            {
                                "role": msg["role"],
                                "content": self._clean_text(msg["content"]),
                            }
                        )
                        selected.append(
                            {
                                "role": user_msg["role"],
                                "content": self._clean_text(user_msg["content"]),
                            }
                        )
                        token_count += pair_tokens
                        i += 2
                    else:
                        break
                else:
                    i += 1
            elif msg["role"] == "user":
                i += 1
            else:
                i += 1

        selected.reverse()
        return selected

    def add_episodic_fact(
        self,
        conv_id: str,
        content: str,
        importance_score: float = 0.5,
    ) -> Optional[str]:
        """
        Store a fact from the current session.
        Deduplicates by SHA-256 content hash.
        Returns fact id, or None if duplicate.
        """
        content = content.strip()
        if len(content) < MIN_EPISODIC_LENGTH:
            return None

        content_hash = self._content_hash(content)
        fact_id = str(uuid.uuid4())
        now = time.time()

        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO episodic_facts
                    (id, conv_id, content, content_hash, importance_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (fact_id, conv_id, content, content_hash, importance_score, now),
            )
            conn.commit()
            return fact_id
        except sqlite3.IntegrityError:
            return None

    def get_episodic_facts(self, conv_id: str) -> List[Dict[str, Any]]:
        """Non-superseded episodic facts for this session, chronological."""
        rows = (
            self._get_conn()
            .execute(
                """
            SELECT id, content, importance_score, created_at
            FROM episodic_facts
            WHERE conv_id=? AND is_superseded=0
            ORDER BY created_at ASC
            """,
                (conv_id,),
            )
            .fetchall()
        )
        return [dict(r) for r in rows]

    def supersede_episodic_fact(self, old_id: str, new_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE episodic_facts SET is_superseded=1, superseded_by=? WHERE id=?",
            (new_id, old_id),
        )
        conn.commit()

    def search_memories(
        self,
        query: str,
        conv_id: str,
        limit: int = 5,
        score_threshold: float = 0.25,
    ) -> List[Dict]:
        """
        Search Qdrant then post-process:
          1. Filter below score_threshold
          2. Exclude superseded entries
          3. Deduplicate by content
          4. Re-rank: combined_score = vector_score × recency_decay
          5. Sort oldest→newest (so LLM reads in chronological order)
          6. Attach human-readable time label to each entry
        """
        if self.engine is None:
            return []

        try:
            try:
                res = self.memory.search(query, run_id=conv_id, limit=limit * 3)
            except TypeError:
                res = self.memory.search(query, conv_id)

            raw = res.get("results", []) if isinstance(res, dict) else (res or [])

        except Exception as exc:
            logger.warning(f"Qdrant search failed: {exc}")
            return []

        conn = self._get_conn()
        superseded: Set[str] = {
            r["content_hash"]
            for r in conn.execute(
                "SELECT content_hash FROM semantic_memories "
                "WHERE conv_id=? AND is_superseded=1",
                (conv_id,),
            ).fetchall()
        }
        now = time.time()
        ranked: List[Dict] = []
        seen: Set[str] = set()
        for r in raw:
            vector_score = r.get("score", 0.0)
            if vector_score < score_threshold:
                continue

            content = r.get("memory", r.get("text", "")).strip()
            if not content:
                continue

            h = self._content_hash(content)
            if h in seen or h in superseded:
                continue

            seen.add(h)
            meta = conn.execute(
                "SELECT created_at FROM semantic_memories "
                "WHERE content_hash=? AND conv_id=?",
                (h, conv_id),
            ).fetchone()
            created_at = meta["created_at"] if meta else now
            recency = self._recency_score(created_at, now)
            combined = vector_score * recency
            ranked.append(
                {
                    "content": content,
                    "vector_score": round(vector_score, 3),
                    "recency_score": round(recency, 3),
                    "combined_score": round(combined, 3),
                    "created_at": created_at,
                    "time_label": self._relative_time_label(created_at),
                }
            )
        ranked.sort(key=lambda x: x["combined_score"], reverse=True)
        top = ranked[:limit]
        top.sort(key=lambda x: x["created_at"])
        return top

    def should_store_long_term(self, interaction: List[Dict]) -> bool:
        """
        Semantic signal check — avoids keyword false-positives.
        Requires sufficient length AND at least one strong signal.
        """
        user_text = interaction[0]["content"]
        assistant_text = interaction[1]["content"]

        if len(user_text) < MIN_LTM_USER_LENGTH or len(assistant_text) < 20:
            return False

        strong_signals = [
            "I prefer",
            "I always",
            "I never",
            "I don't like",
            "I like",
            "my project",
            "we use",
            "our stack",
            "I work on",
            "I usually",
            "I'm building",
            "I'm using",
            "my team",
            "my company",
            "ผมชอบ",
            "ฉันชอบ",
            "ไม่ชอบ",
            "ปกติใช้",
            "โปรเจกต์ผม",
            "เราใช้",
            "ทีมผม",
            "บริษัทผม",
        ]
        combined = user_text + " " + assistant_text
        return any(s.lower() in combined.lower() for s in strong_signals)

    def save_to_memory(self, messages: List[Dict], conv_id: str) -> None:
        """
        Save to Qdrant via mem0, then mirror metadata to semantic_memories.
        If same content already exists: refresh updated_at (no duplicate).
        """
        if self.engine is None:
            return

        try:
            try:
                self.memory.add(messages, run_id=conv_id)
            except TypeError:
                try:
                    self.memory.add(messages, user_id=conv_id)
                except TypeError:
                    self.memory.add(messages)

            now = time.time()
            conn = self._get_conn()
            for msg in messages:
                if msg.get("role") != "user":
                    continue
                content = msg.get("content", "").strip()
                if len(content) < MIN_LTM_USER_LENGTH:
                    continue

                h = self._content_hash(content)
                existing = conn.execute(
                    "SELECT id FROM semantic_memories WHERE content_hash=? AND conv_id=?",
                    (h, conv_id),
                ).fetchone()

                if existing:
                    conn.execute(
                        "UPDATE semantic_memories SET updated_at=? WHERE id=?",
                        (now, existing["id"]),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO semantic_memories
                            (id, conv_id, content, content_hash, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (str(uuid.uuid4()), conv_id, content, h, now, now),
                    )
            conn.commit()

        except Exception as exc:
            logger.error(f"save_to_memory failed: {exc}", exc_info=True)

    def finalize_context(
        self,
        conv_id: str,
        options: Dict[str, Any],
        user_input: str,
        system_tokens_estimate: int = 150,
        response_tokens_estimate: Optional[int] = None,
        long_term_memories: Optional[List[Dict]] = None,
    ) -> List[Dict[str, Any]]:

        real_n_ctx = getattr(self.engine, "n_ctx", self.n_ctx)

        real_n_ctx = options.get("n_ctx", real_n_ctx)

        default_output = max(512, int(real_n_ctx * 0.15))

        if options.get("max_tokens") is not None:
            target_response_tokens = options["max_tokens"]
        elif response_tokens_estimate is not None:
            target_response_tokens = response_tokens_estimate
        else:
            target_response_tokens = default_output

        max_output = int(real_n_ctx * 0.4)
        target_response_tokens = min(target_response_tokens, max_output)

        user_input_tokens = self._count_tokens(user_input)
        fixed_budget = (
            system_tokens_estimate
            + user_input_tokens
            + target_response_tokens
            + self.safety_margin
        )

        raw_memory_pool = real_n_ctx - fixed_budget
        memory_pool = max(raw_memory_pool, 200)

        max_ltm_tokens = int(memory_pool * 0.4)
        ltm_list = long_term_memories or []
        filtered_ltm = []
        ltm_tokens = 60 if ltm_list else 0

        for m in ltm_list:
            tok = self._count_tokens(m.get("content", ""))
            if ltm_tokens + tok <= max_ltm_tokens:
                filtered_ltm.append(m)
                ltm_tokens += tok

        memory_pool -= ltm_tokens

        max_episodic_tokens = int(memory_pool * 0.2)
        episodic_facts = self.get_episodic_facts(conv_id)
        filtered_episodic = []
        episodic_tokens = 50 if episodic_facts else 0

        for f in episodic_facts:
            tok = self._count_tokens(f["content"])
            if episodic_tokens + tok <= max_episodic_tokens:
                filtered_episodic.append(f)
                episodic_tokens += tok

        memory_pool -= episodic_tokens

        working_budget = memory_pool

        working = self.get_working_memory(conv_id, working_budget)

        conn = self._get_conn()
        session = conn.execute(
            "SELECT summary FROM sessions WHERE id=?", (conv_id,)
        ).fetchone()
        chat_summary = session["summary"] if session and session["summary"] else ""

        context: List[Dict[str, Any]] = []

        if chat_summary:
            context.append(
                {
                    "role": "system",
                    "content": f"(Conversation Summary So Far)\n{chat_summary}",
                }
            )

        if filtered_ltm:
            lines = [f"[{m['time_label']}] {m['content']}" for m in filtered_ltm]
            context.append(
                {
                    "role": "system",
                    "content": "(Ordered oldest→newest. The LAST entry is the most current belief.)\n"
                    + "\n".join(lines),
                }
            )

        if filtered_episodic:
            lines = [f"- {f['content']}" for f in filtered_episodic]
            context.append(
                {"role": "system", "content": "(Session Facts)\n" + "\n".join(lines)}
            )

        context.extend(working)

        return context




