import json
import math
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    content: str
    metadata: Dict[str, Any]
    score: float


class RetrievalService:
    def __init__(self, db_path: str = "data/rag.db", embedding_dim: int = 256):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id TEXT PRIMARY KEY,
                    source_name TEXT,
                    source_type TEXT,
                    metadata TEXT,
                    created_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding TEXT,
                    created_at REAL,
                    FOREIGN KEY (doc_id) REFERENCES rag_documents(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts
                USING fts5(chunk_id UNINDEXED, content)
                """
            )
            conn.commit()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_\u0E00-\u0E7F]+", (text or "").lower())

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.embedding_dim
        tokens = self._tokenize(text)
        if not tokens:
            return vec
        for tok in tokens:
            idx = hash(tok) % self.embedding_dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        return sum(x * y for x, y in zip(a, b))

    def _rewrite_query(self, query: str) -> str:
        cleaned = re.sub(r"\s+", " ", (query or "")).strip()
        return cleaned

    def index_document(
        self,
        source_name: str,
        source_type: str,
        chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        doc_id = str(uuid.uuid4())
        now = time.time()
        metadata = metadata or {}

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO rag_documents (id, source_name, source_type, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
                (doc_id, source_name, source_type, json.dumps(metadata), now),
            )
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                chunk_meta = {**metadata, "source_name": source_name, "source_type": source_type}
                emb = self._embed(chunk)
                conn.execute(
                    "INSERT INTO rag_chunks (id, doc_id, chunk_index, content, metadata, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (chunk_id, doc_id, i, chunk, json.dumps(chunk_meta), json.dumps(emb), now),
                )
                conn.execute(
                    "INSERT INTO rag_chunks_fts (chunk_id, content) VALUES (?, ?)",
                    (chunk_id, chunk),
                )
            conn.commit()

        return doc_id

    def _bm25_candidates(self, query: str, limit: int = 20) -> List[Tuple[str, float]]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        fts_query = " OR ".join(tokens)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id, bm25(rag_chunks_fts) AS score FROM rag_chunks_fts WHERE rag_chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (fts_query, limit),
            ).fetchall()
        # FTS bm25 lower is better => invert to positive relevance
        result = []
        for r in rows:
            raw = float(r["score"])
            result.append((r["chunk_id"], 1.0 / (1.0 + max(raw, 0.0))))
        return result

    def _vector_candidates(self, query: str, limit: int = 40) -> List[Tuple[str, float]]:
        q_emb = self._embed(query)
        with self._conn() as conn:
            rows = conn.execute("SELECT id, embedding FROM rag_chunks").fetchall()
        scored: List[Tuple[str, float]] = []
        for r in rows:
            emb = json.loads(r["embedding"] or "[]")
            scored.append((r["id"], self._cosine(q_emb, emb)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def retrieve(
        self,
        query: str,
        limit: int = 6,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        score_threshold: float = 0.18,
    ) -> List[RetrievedChunk]:
        rewritten = self._rewrite_query(query)

        bm25_map: Dict[str, float] = {}
        vec_map: Dict[str, float] = {}

        if use_hybrid:
            for cid, s in self._bm25_candidates(rewritten, limit=40):
                bm25_map[cid] = max(bm25_map.get(cid, 0.0), s)

        for cid, s in self._vector_candidates(rewritten, limit=60):
            vec_map[cid] = max(vec_map.get(cid, 0.0), s)

        candidate_ids = set(bm25_map.keys()) | set(vec_map.keys())
        if not candidate_ids:
            return []

        with self._conn() as conn:
            placeholders = ",".join("?" for _ in candidate_ids)
            rows = conn.execute(
                f"SELECT id, doc_id, chunk_index, content, metadata FROM rag_chunks WHERE id IN ({placeholders})",
                list(candidate_ids),
            ).fetchall()

        query_terms = set(self._tokenize(rewritten))
        out: List[RetrievedChunk] = []

        for r in rows:
            chunk_id = r["id"]
            vec_score = vec_map.get(chunk_id, 0.0)
            bm25_score = bm25_map.get(chunk_id, 0.0)
            score = 0.65 * vec_score + 0.35 * bm25_score if use_hybrid else vec_score

            if use_rerank:
                terms = set(self._tokenize(r["content"]))
                overlap = len(query_terms & terms) / max(1, len(query_terms))
                score = 0.8 * score + 0.2 * overlap

            if score < score_threshold:
                continue

            out.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=r["doc_id"],
                    chunk_index=r["chunk_index"],
                    content=r["content"],
                    metadata=json.loads(r["metadata"] or "{}"),
                    score=round(score, 4),
                )
            )

        out.sort(key=lambda x: x.score, reverse=True)
        return out[:limit]

    def build_grounded_prompt(
        self,
        user_query: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Dict[str, Any]:
        if not retrieved_chunks:
            return {
                "should_answer": False,
                "reason": "no_evidence",
                "system_prompt": "You are a careful assistant. If there is no trusted evidence, refuse clearly.",
                "context": "",
            }

        lines = []
        citations = []
        for idx, ch in enumerate(retrieved_chunks, start=1):
            source = ch.metadata.get("source_name", "unknown")
            lines.append(f"[{idx}] ({source}) {ch.content}")
            citations.append({"index": idx, "chunk_id": ch.chunk_id, "source": source, "score": ch.score})

        system = (
            "Use ONLY the provided context. Cite evidence as [n]. "
            "If evidence is insufficient, say: 'I don't have enough evidence in the provided sources.'"
        )
        return {
            "should_answer": True,
            "system_prompt": system,
            "context": "\n".join(lines),
            "citations": citations,
            "query": user_query,
        }
