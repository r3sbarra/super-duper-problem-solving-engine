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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from super_solver.core.embedder import EmbedderBackend, get_backend


def _running_under_pytest() -> bool:
    """True when the process is a pytest run, whatever the import order.

    ``PYTEST_CURRENT_TEST`` is only set *while a test executes*, so checking it
    alone is order-dependent: a module imported during collection bound its
    DB path to the REAL database, and any test that then wrote to it -- or
    cleared it -- destroyed the developer's index. That is not theoretical; a
    test fixture clearing the session-wide vector-index singleton wiped a
    738-row local index exactly this way. ``pytest`` is in ``sys.modules`` from
    the moment the runner starts, so it closes the window.
    """
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def get_vector_db_path() -> str:
    """Resolve the vector-index DB path (env-overridable, test-isolated).

    The default lives under the PACKAGE ROOT's ``.super_solver_data``, NOT the
    current working directory: a CWD-relative path silently produced an empty
    index whenever the engine was imported and used from another directory
    (measured: primitive_analogize returned 0 hits from any CWD except the repo
    root). ``SUPER_SOLVER_VECTOR_DB`` still overrides for tests/CI.
    """
    env_path = os.environ.get("SUPER_SOLVER_VECTOR_DB")
    if env_path:
        return env_path
    if _running_under_pytest():
        return ":memory:"
    # Package root = .../src/super_solver/vectorize -> up 3 = repo root
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = repo_root / ".super_solver_data"
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
        """Insert (or replace) a typed vector. Returns its id.

        The id is derived DETERMINISTICALLY from (kind, content) so re-storing
        the same solution replaces it instead of accumulating duplicates
        (measured: the previous random-id scheme left 738 rows for 186 distinct
        solutions, so retrieval returned the same analog up to 8 times).
        """
        if item_id is None:
            import hashlib

            digest = hashlib.sha256(f"{kind}\x00{content}".encode("utf-8")).hexdigest()
            item_id = f"{kind}:{digest[:32]}"
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
        dedupe: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return vectors most similar to ``query_vector`` (optionally within a kind).

        Results are deduplicated by ``(kind, content)`` *before* truncating to
        ``top_k``, keeping the best-scoring copy of each distinct content.

        Without this, an index holding the same content under several ids
        returns the *same* analog ``top_k`` times and crowds out everything
        else.  That is not hypothetical: ``add()`` only began deriving its id
        from the content recently, so indexes written earlier -- and any caller
        that passes an explicit ``item_id``, such as the case harvester -- still
        contain near-duplicate rows.  Measured on the shipped 738-row index, a
        top-5 ``search`` returned 5 copies of one Kruskal solution and a top-4
        ``analogize`` returned 4 copies of one problem.

        Pass ``dedupe=False`` to get the raw scored rows.
        """
        results = []
        for r_id, r_kind, r_title, r_content, blob, meta in self._rows(kind):
            vec = np.frombuffer(blob, dtype=np.float32)
            # Guard against mixed-dimension vectors (embedder config changed between
            # writes). Skip mismatched rows instead of crashing the whole search.
            if vec.shape[0] != query_vector.shape[0]:
                continue
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
        if dedupe:
            # Dedupe after sorting so the best-scoring copy of each content wins.
            unique: List[Dict[str, Any]] = []
            seen: set = set()
            for r in results:
                key = (r["kind"], r["content"])
                if key in seen:
                    continue
                seen.add(key)
                unique.append(r)
            results = unique
        return results[:top_k]

    def search_reranked(
        self,
        query_text: str,
        query_vector: Optional[np.ndarray] = None,
        kind: Optional[str] = None,
        candidate_pool_size: int = 50,
        top_k: int = 5,
        idf_weight: float = 0.45,
        atom_weight: float = 0.25,
        semantic_weight: float = 0.20,
        brevity_weight: float = 0.10,
    ) -> List[Dict[str, Any]]:
        """Two-stage retrieval: broad candidate generation + cross-encoder joint reranking.

        Resolves the retrieval bottleneck where single-stage bi-encoder cosine ranking
        collapses on high lexical-overlap mechanism pools.
        1. Generates top candidate pool via dense vector similarity.
        2. Scores (query, candidate) jointly using:
           - Distinctive token overlap weighted by corpus IDF
           - Structural-atom containment (specific domain primitives)
           - Dense semantic similarity
           - Candidate brevity / specificity penalty
        """
        import re
        from collections import Counter

        from super_solver.vectorize.primitives import strip_to_primitives

        if query_vector is None:
            query_vector = self.backend.encode(query_text)

        candidates = self.search(
            query_vector, kind=kind, top_k=max(candidate_pool_size, top_k)
        )
        if not candidates:
            return []

        stop_words = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
            "by", "is", "are", "that", "this", "it", "its", "as", "at", "from",
            "so", "only", "one", "side", "over", "into", "without", "not", "add",
            "ensure", "use", "make", "model", "system"
        }

        def toks(text: str) -> List[str]:
            return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in stop_words and len(w) > 3]

        # 1. Compute corpus IDF over candidates
        corpus_texts = [(c["title"] or "") + " " + (c["content"] or "") for c in candidates]
        df = Counter()
        for t in corpus_texts:
            for w in set(toks(t)):
                df[w] += 1
        n_corpus = len(corpus_texts)
        idf = {w: float(np.log((n_corpus + 1) / (count + 1)) + 1.0) for w, count in df.items()}

        q_toks = set(toks(query_text))

        def spec_atoms(text: str) -> set[str]:
            prims = strip_to_primitives(text)
            return {p for p in prims if not (p.startswith("act_") or p.startswith("attr_"))}

        q_atoms = spec_atoms(query_text)

        reranked = []
        for cand in candidates:
            c_text = (cand["title"] or "") + " " + (cand["content"] or "")
            c_toks = set(toks(c_text))

            # Feature A: IDF-weighted distinctive token overlap
            shared_toks = q_toks & c_toks
            if q_toks:
                idf_shared = sum(idf.get(w, 1.0) for w in shared_toks)
                idf_total = sum(idf.get(w, 1.0) for w in q_toks)
                joint_token_score = idf_shared / idf_total if idf_total > 0 else 0.0
            else:
                joint_token_score = 0.0

            # Feature B: Structural atom containment
            c_atoms = spec_atoms(c_text)
            if q_atoms:
                atom_score = len(q_atoms & c_atoms) / len(q_atoms)
            else:
                atom_score = 0.0

            # Feature C: Semantic cosine similarity
            sim = float(cand.get("similarity", 0.0))
            sem_score = max(0.0, min(1.0, (sim - 0.2) / 0.5))

            # Feature D: Specificity / brevity penalty (downweight vague, long texts)
            brev_score = 1.0 / (1.0 + max(0, len(c_toks) - 10) * 0.06)

            composite_score = (
                idf_weight * joint_token_score
                + atom_weight * atom_score
                + semantic_weight * sem_score
                + brevity_weight * brev_score
            )

            out_cand = dict(cand)
            out_cand["rerank_score"] = round(composite_score, 4)
            out_cand["rerank_breakdown"] = {
                "joint_token": round(joint_token_score, 4),
                "atom_containment": round(atom_score, 4),
                "semantic": round(sem_score, 4),
                "brevity": round(brev_score, 4),
            }
            reranked.append(out_cand)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

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
            linked = self._link(p)
            out.append(linked)
        return out

    def hybrid_analogize(
        self,
        problem_vector: np.ndarray,
        top_k: int = 3,
        min_similarity: float = 0.0,
        solution_weight: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """Analogical transfer with hybrid scoring: problem-vector + solution-vector.

        Standard dense retrieval matches the query only against stored problem
        vectors. This variant ALSO matches the query against each candidate's
        linked solution vector and blends the two scores, so a problem whose
        *solution* is semantically close to the query ranks higher even if its
        problem wording differs. ``solution_weight`` controls the blend
        (0 = pure problem match, 1 = pure solution match).
        """
        # Candidate problems (broad recall)
        candidates = self.search(
            problem_vector, kind="problem", top_k=max(top_k * 3, 10), min_similarity=min_similarity
        )
        scored = []
        for p in candidates:
            linked = self._link(p)
            prob_sim = p["similarity"]
            sol_sim = 0.0
            if linked["solution"] is not None:
                sol_vec = self._vector_of(linked["solution"]["id"])
                if sol_vec is not None and sol_vec.shape[0] == problem_vector.shape[0]:
                    sol_sim = self.backend.cosine_similarity(problem_vector, sol_vec)
            blended = (1.0 - solution_weight) * prob_sim + solution_weight * sol_sim
            scored.append((blended, prob_sim, sol_sim, linked))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for blended, prob_sim, sol_sim, linked in scored[:top_k]:
            linked["problem"]["similarity"] = round(blended, 4)
            linked["problem"]["problem_similarity"] = round(prob_sim, 4)
            linked["problem"]["solution_similarity"] = round(sol_sim, 4)
            out.append(linked)
        return out

    def _link(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a problem row's linked solution + path into a dict."""
        meta = p["metadata"]
        linked = {"problem": p, "solution": None, "path": None}
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
        return linked

    def _vector_of(self, item_id: str) -> Optional[np.ndarray]:
        row = self._get_by_id(item_id)
        if not row:
            return None
        return np.frombuffer(row[4], dtype=np.float32)

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
