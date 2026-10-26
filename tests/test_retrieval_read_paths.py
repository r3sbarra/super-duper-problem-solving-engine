"""Regression tests for retrieval read-path invariants.

Two defects are locked in here:

1. ``VectorIndex.search`` (and therefore ``analogize``, ``hybrid_analogize``
   and ``search_reranked``) never deduplicated by content, so an index holding
   the same content under several ids -- legacy indexes written before ``add()``
   derived its id from the content, or any caller passing an explicit
   ``item_id`` such as the case harvester -- returned the *same* analog
   ``top_k`` times, crowding out every other result.

2. ``VectorizationService.primitive_analogize`` dropped the ``empty_match``
   flag from the no-structural-signal sentinel row, which made every downstream
   ``if h.get("empty_match"): continue`` guard dead code and reported the
   literal string ``"<no structural primitives extracted>"`` as a retrieved
   analog.
"""

from __future__ import annotations

from super_solver.vectorize import VectorizationService
from super_solver.vectorize.index import VectorIndex

DUP = "Use a low-tack adhesive that was originally a failed super-strong glue"
OTHER = [
    "Use a circuit breaker to stop cascading failures",
    "Add an index to the column so the slow query stops scanning the whole table",
]
QUERY = "How to make a climbing pad that grips smooth vertical walls without glue"


def test_search_deduplicates_identical_content():
    idx = VectorIndex(db_path=":memory:")
    vec = idx.backend.encode(DUP)
    # Simulate a legacy index: identical content stored under four distinct ids.
    for i in range(4):
        idx.add("solution", DUP, vec, title="Post-it note", item_id=f"legacy-{i}")
    for text in OTHER:
        idx.add("solution", text, idx.backend.encode(text), title="other")

    hits = idx.search(idx.backend.encode(QUERY), kind="solution", top_k=5)
    contents = [h["content"] for h in hits]
    assert len(contents) == len(set(contents)), f"duplicate content returned: {contents}"
    assert len(hits) == 3  # 1 distinct duplicate-content row + 2 others

    # The raw, un-deduplicated rows remain reachable for callers that want them.
    raw = idx.search(idx.backend.encode(QUERY), kind="solution", top_k=10, dedupe=False)
    assert len(raw) == 6  # 4 duplicate-content rows + 2 others
    idx.close()


def test_dedupe_keeps_the_best_scoring_copy():
    """Dedup happens after sorting, so the highest-similarity copy survives."""
    idx = VectorIndex(db_path=":memory:")
    idx.add("solution", DUP, idx.backend.encode(DUP), title="good", item_id="a")
    idx.add("solution", DUP, idx.backend.encode(DUP), title="bad", item_id="b")
    hits = idx.search(idx.backend.encode(DUP), kind="solution", top_k=5)
    assert len(hits) == 1
    assert hits[0]["title"] == "good"
    idx.close()


def test_primitive_analogize_surfaces_empty_match_flag():
    svc = VectorizationService()
    hits = svc.primitive_analogize("zzqq xxvv frobnicate", top_k=3)
    # A target that strips to no primitives yields the explicit sentinel row...
    assert len(hits) == 1
    assert hits[0]["empty_match"] is True, hits[0]
    assert "no structural primitives" in hits[0]["content"]


def test_real_analogs_are_not_flagged_empty():
    svc = VectorizationService()
    sol = "Add an index to the column so the slow query stops scanning the whole table"
    svc.index.add("solution", sol, svc.backend.encode(sol), title="indexing")
    hits = svc.primitive_analogize("reduce the latency of the read path", top_k=2)
    assert hits
    assert all(h["empty_match"] is False for h in hits), hits
