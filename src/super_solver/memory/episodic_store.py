"""Episodic Vector Memory Store.

Stores and queries problem-solving experiences, trajectories, and findings
using fast cosine similarity index in-process.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from super_solver.core.embeddings import embedding_service


def get_default_db_path() -> str:
    """Resolves the default database path with support for env vars and test isolation."""
    env_path = os.environ.get("SUPER_SOLVER_DB_PATH")
    if env_path:
        return env_path
    if "PYTEST_CURRENT_TEST" in os.environ:
        return ":memory:"
    data_dir = Path.cwd() / ".super_solver_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / "super_solver.db")


class EpisodicVectorStore:
    """In-process SQLite + dense vector store for problem trajectories and persistent state."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path if db_path is not None else get_default_db_path()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
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
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS dead_ends (
                    id TEXT PRIMARY KEY,
                    description TEXT UNIQUE,
                    vector_blob BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS engine_parameters (
                    param_key TEXT PRIMARY KEY,
                    param_value REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS self_improvement_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_json TEXT,
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
                (item_id, category, title, content, vector_bytes, meta_str),
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
                results.append(
                    {
                        "id": r_id,
                        "category": r_cat,
                        "title": r_title,
                        "content": r_content,
                        "similarity": float(sim),
                        "metadata": json.loads(meta_str) if meta_str else {},
                    }
                )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM episodes")
        return int(cur.fetchone()[0])

    def insert_dead_end(
        self, description: str, vector: np.ndarray, item_id: Optional[str] = None
    ) -> str:
        """Inserts or updates a dead-end record in the persistent store."""
        item_id = item_id or str(uuid.uuid4())
        vector_bytes = np.asarray(vector, dtype=np.float32).tobytes()
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO dead_ends (id, description, vector_blob)
                VALUES (?, ?, ?)
                """,
                (item_id, description, vector_bytes),
            )
        return item_id

    def load_dead_ends(self) -> List[Tuple[str, np.ndarray]]:
        """Loads all registered dead ends as (description, vector) pairs."""
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT description, vector_blob FROM dead_ends ORDER BY created_at ASC"
        ).fetchall()
        result = []
        for desc, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            result.append((desc, vec))
        return result

    def sync_dead_ends(self, dead_ends: List[Tuple[str, np.ndarray]]):
        """Synchronizes the dead_ends table with the provided list of (description, vector)."""
        with self._conn:
            self._conn.execute("DELETE FROM dead_ends")
            for desc, vec in dead_ends:
                item_id = str(uuid.uuid4())
                vector_bytes = np.asarray(vec, dtype=np.float32).tobytes()
                self._conn.execute(
                    "INSERT INTO dead_ends (id, description, vector_blob) VALUES (?, ?, ?)",
                    (item_id, desc, vector_bytes),
                )

    def save_parameters(self, params: Dict[str, Any]):
        """Persists engine parameters to SQLite."""
        with self._conn:
            for key, val in params.items():
                if isinstance(val, (int, float)):
                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO engine_parameters (param_key, param_value, updated_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                        """,
                        (key, float(val)),
                    )

    def load_parameters(self) -> Dict[str, float]:
        """Loads all saved engine parameters from SQLite."""
        cur = self._conn.cursor()
        rows = cur.execute("SELECT param_key, param_value FROM engine_parameters").fetchall()
        return {key: float(val) for key, val in rows}

    def save_improvement_record(self, record: Dict[str, Any]):
        """Persists a self-improvement diagnostic/tuning record."""
        record_json = json.dumps(record)
        with self._conn:
            self._conn.execute(
                "INSERT INTO self_improvement_history (record_json) VALUES (?)", (record_json,)
            )

    def load_improvement_history(self) -> List[Dict[str, Any]]:
        """Loads the self-improvement audit trail."""
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT record_json FROM self_improvement_history ORDER BY id ASC"
        ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def close(self):
        self._conn.close()
