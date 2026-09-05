"""Vector index for problems, solutions, and discovery paths.

A SQLite-backed store that persists typed vectors (problem / solution / path)
and supports:

- cosine-similarity search within a type (``search``)
- cross-type analogical transfer (``analogize``): given a new problem vector,
  find the most similar past problem and return its stored solution + path so
  the solver can reuse a proven trajectory.

Vectors are stored as float32 blobs alongside their text payload and metadata.
The store is backend-agnostic: it accepts any ``np.ndarray`` produced by an
embedder backend.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from super_solver.core.embedder import EmbedderBackend, get_backend


def get_vector_db_path() -> str:
    """Resolve the vector-index DB path (env-overridable, test-isolated)."""
    env_path = os.environ.get("SUPER_SOLVER_VECTOR_DB")
    if env_path:
        return env_path
    if "PYTEST_CURRENT_TEST" in os.environ:
        return ":memory:"
    data_dir = Path.cwd() / ".super_solver_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / "vector_index.db")


class VectorIndex:
    """Persistent typed vector store with similarity search + analogical transfer."""

    def __init__(self, db_path: Optional[str] = None, backend: Optional[EmbedderBackend] = None):
        self.db_path = db_path if db_path is not None else get_vector_db_path()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.backend = backend or get_backend()
        self._conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self):
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,          -- problem | solution | path
                    title TEXT,
                    content TEXT,
                    vector_blob BLOB NOT NULL,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vectors_kind ON vectors(kind)"
            )

    # -- write --------------------------------------------------------------

    def add(
        self,
        kind: str,
        content: str,
        vector: np.ndarray,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        item_id: Optional[str] = None,
    ) -> str:
        """Insert (or replace) a typed vector. Returns its id."""
        item_id = item_id or str(uuid.uuid4())
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        meta = json.dumps(metadata or {})
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO vectors (id, kind, title, content, vector_blob, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (item_id, kind, title, content, blob, meta),
            )
        return item_id

    def add_problem(
        self,
        specification: str,
        vector: np.ndarray,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self.add("problem", specification, vector, title=title, metadata=metadata)

    def add_solution(
        self,
        solution_text: str,
        vector: np.ndarray,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self.add("solution", solution_text, vector, title=title, metadata=metadata)

    def add_path(
        self,
        path_summary: str,
        vector: np.ndarray,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self.add("path", path_summary, vector, title=title, metadata=metadata)

    # -- read ---------------------------------------------------------------

    def _rows(self, kind: Optional[str] = None) -> List[tuple]:
        cur = self._conn.cursor()
        if kind:
            return cur.execute(
                "SELECT id, kind, title, content, vector_blob, metadata_json FROM vectors WHERE kind = ?",
                (kind,),
            ).fetchall()
        return cur.execute(
            "SELECT id, kind, title, content, vector_blob, metadata_json FROM vectors"
        ).fetchall()

    def search(
        self,
        query_vector: np.ndarray,
        kind: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Return vectors most similar to ``query_vector`` (optionally within a kind)."""
        results = []
        for r_id, r_kind, r_title, r_content, blob, meta in self._rows(kind):
            vec = np.frombuffer(blob, dtype=np.float32)
            sim = self.backend.cosine_similarity(query_vector, vec)
            if sim >= min_similarity:
                results.append(
                    {
                        "id": r_id,
                        "kind": r_kind,
                        "title": r_title,
                        "content": r_content,
                        "similarity": float(sim),
                        "metadata": json.loads(meta) if meta else {},
                    }
                )
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def analogize(
        self,
        problem_vector: np.ndarray,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Find similar past problems and return their stored solutions + paths.

        This is the analogical-transfer primitive: given a new problem, retrieve
        the most similar previously-solved problem and its proven solution and
        discovery path, so the solver can reuse a successful trajectory.
        """
        similar_problems = self.search(
            problem_vector, kind="problem", top_k=top_k, min_similarity=min_similarity
        )
        out = []
        for p in similar_problems:
            meta = p["metadata"]
            linked = {
                "problem": p,
                "solution": None,
                "path": None,
            }
            # Resolve linked solution/path ids stored in the problem's metadata
            for link_kind, key in (("solution", "solution_id"), ("path", "path_id")):
                link_id = meta.get(key)
                if link_id:
                    row = self._get_by_id(link_id)
                    if row:
                        linked[link_kind] = {
                            "id": row[0],
                            "kind": row[1],
                            "title": row[2],
                            "content": row[3],
                            "metadata": json.loads(row[5]) if row[5] else {},
                        }
            out.append(linked)
        return out

    def _get_by_id(self, item_id: str) -> Optional[tuple]:
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT id, kind, title, content, vector_blob, metadata_json FROM vectors WHERE id = ?",
            (item_id,),
        ).fetchone()
        return row

    def get(self, item_id: str) -> Optional[Dict[str, Any]]:
        row = self._get_by_id(item_id)
        if not row:
            return None
        return {
            "id": row[0],
            "kind": row[1],
            "title": row[2],
            "content": row[3],
            "metadata": json.loads(row[5]) if row[5] else {},
        }

    def count(self, kind: Optional[str] = None) -> int:
        cur = self._conn.cursor()
        if kind:
            cur.execute("SELECT COUNT(*) FROM vectors WHERE kind = ?", (kind,))
        else:
            cur.execute("SELECT COUNT(*) FROM vectors")
        return int(cur.fetchone()[0])

    def close(self):
        self._conn.close()
