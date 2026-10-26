"""Test two-stage retrieval reranking in VectorIndex."""
from __future__ import annotations

import numpy as np
import pytest

from super_solver.vectorize.index import VectorIndex


def test_search_reranked_basic():
    idx = VectorIndex(db_path=":memory:")

    # Insert test solutions with high lexical overlap but distinct mechanisms
    doc1 = "Use an in-memory LRU cache to store query results and reduce database latency"
    doc2 = "Use a database index on primary key columns to make query scans faster"
    doc3 = "Use rate limiting on the API gateway to prevent resource exhaustion from requests"

    v1 = idx.backend.encode(doc1)
    v2 = idx.backend.encode(doc2)
    v3 = idx.backend.encode(doc3)

    idx.add("solution", doc1, v1, title="LRU cache mechanism")
    idx.add("solution", doc2, v2, title="Index scan mechanism")
    idx.add("solution", doc3, v3, title="Rate limiter mechanism")

    # Search with specific query
    query = "How to cache database queries in memory to lower latency"
    results = idx.search_reranked(query, kind="solution", top_k=2)

    assert len(results) == 2
    top_hit = results[0]
    assert "cache" in top_hit["content"].lower()
    assert "rerank_score" in top_hit
    assert "rerank_breakdown" in top_hit
    assert top_hit["rerank_breakdown"]["joint_token"] > 0.0

    idx.close()
