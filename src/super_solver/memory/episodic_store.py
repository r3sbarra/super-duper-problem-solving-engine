"""Episodic Vector Memory Store.

Stores and queries problem-solving experiences, trajectories, and findings
using fast cosine similarity index in-process.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional

import numpy as np

from super_solver.core.embeddings import embedding_service


class EpisodicVectorStore:
    """In-process SQLite + dense vector store for problem trajectories."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self):
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    category TEXT,
                    title TEXT,
                    content TEXT,
                    vector_blob BLOB,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def insert(
        self,
        item_id: str,
        category: str,
        title: str,
        content: str,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if vector is None:
            vector = embedding_service.encode(f"{title}\n{content}")
        vector_bytes = np.asarray(vector, dtype=np.float32).tobytes()
        meta_str = json.dumps(metadata or {})

        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO episodes (id, category, title, content, vector_blob, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (item_id, category, title, content, vector_bytes, meta_str)
            )

    def search_similar(
        self,
        query_vector: np.ndarray,
        category: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.35,
    ) -> List[Dict[str, Any]]:
        cursor = self._conn.cursor()
        query = "SELECT id, category, title, content, vector_blob, metadata_json FROM episodes"
        params = ()
        if category:
            query += " WHERE category = ?"
            params = (category,)

        rows = cursor.execute(query, params).fetchall()
        results = []

        for r_id, r_cat, r_title, r_content, blob, meta_str in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            sim = embedding_service.cosine_similarity(query_vector, vec)
            if sim >= min_similarity:
                results.append({
                    "id": r_id,
                    "category": r_cat,
                    "title": r_title,
                    "content": r_content,
                    "similarity": float(sim),
                    "metadata": json.loads(meta_str) if meta_str else {},
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM episodes")
        return int(cur.fetchone()[0])

    def close(self):
        self._conn.close()
