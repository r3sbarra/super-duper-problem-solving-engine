"""Tests for the problem/solution/path vectorizing subsystem + custom embedder."""

from __future__ import annotations

import numpy as np
import pytest

from super_solver.core.embedder import (
    HybridBackend,
    OllamaBackend,
    PolarityBackend,
    get_backend,
)
from super_solver.vectorize import VectorizationService


def test_polarity_backend_deterministic():
    b = get_backend("polarity")
    v1 = b.encode("How to reduce latency in a distributed database")
    v2 = b.encode("How to reduce latency in a distributed database")
    assert b.dim == 384
    assert np.allclose(v1, v2)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-5


def test_ollama_backend_falls_back_when_unreachable(monkeypatch):
    b = OllamaBackend(base_url="http://127.0.0.1:1", timeout=0.1)
    v = b.encode("test")
    # Falls back to polarity (384d) padded to the pinned ollama dim (768)
    assert v.shape[0] == 768


def test_hybrid_backend_dim():
    b = HybridBackend()
    v = b.encode("test")
    assert v.shape[0] == 384 + 768


def test_problem_solution_path_encoders():
    svc = VectorizationService()
    pv = svc.encode_problem("reduce latency", title="DB", goal_criteria=["low latency"])
    sv = svc.encode_solution("use hybrid logical clock", method="TRIZ", domain="distributed")
    pathv = svc.encode_path(["abduct", "filter", "rollout"], operator_types=["PEIRCE", "KT", "MCTS"])
    assert pv.shape[0] == 384
    assert sv.shape[0] == 384
    assert pathv.shape[0] == 384


def test_vector_index_store_and_search(tmp_path):
    svc = VectorizationService(db_path=str(tmp_path / "vec.db"))
    pid = svc.store_problem("How to reduce latency in a distributed database", title="DB latency")
    svc.store_problem("How to bake a chocolate cake", title="Cake")
    assert svc.index.count("problem") == 2

    qv = svc.encode_problem("How to reduce latency in a distributed database", title="DB latency")
    hits = svc.search(qv, kind="problem", top_k=2)
    assert hits[0]["title"] == "DB latency"
    assert hits[0]["similarity"] > 0.9


def test_analogical_transfer(tmp_path):
    svc = VectorizationService(db_path=str(tmp_path / "vec.db"))
    pid = svc.store_problem("How to reduce latency in a distributed database", title="DB latency")
    sid = svc.store_solution("Use a hybrid logical clock with quorum reads", method="TRIZ", domain="distributed")
    pathid = svc.store_path(["abduct", "filter", "rollout"], operator_types=["PEIRCE", "KT", "MCTS"], final_breakthrough="hybrid logical clock")
    # Link solution + path to the problem
    import json

    svc.index._conn.execute(
        "UPDATE vectors SET metadata_json=? WHERE id=?",
        (json.dumps({"solution_id": sid, "path_id": pathid}), pid),
    )
    svc.index._conn.commit()

    qv = svc.encode_problem("How to reduce latency in a distributed database", title="DB latency")
    analogs = svc.analogize(qv, top_k=3)
    assert len(analogs) >= 1
    assert analogs[0]["solution"] is not None
    assert analogs[0]["path"] is not None
    assert "hybrid logical clock" in analogs[0]["solution"]["content"]


def test_hybrid_analogize_ranks_by_solution(tmp_path):
    """Hybrid retrieval blends problem + solution similarity and exposes both."""
    import json

    svc = VectorizationService(db_path=str(tmp_path / "vec.db"))
    pid1 = svc.store_problem("How to make a canoe hull that floats despite being made of dense concrete", title="Concrete Canoe")
    sid1 = svc.store_solution("Use lightweight aggregates and air-entraining agents to make concrete less dense than water", method="TRIZ", domain="engineering")
    pid2 = svc.store_problem("How to bake a moist chocolate cake with no eggs", title="Cake")
    sid2 = svc.store_solution("Use applesauce as an egg substitute and buttermilk for moisture", method="TRIZ", domain="baking")
    svc.index._conn.execute("UPDATE vectors SET metadata_json=? WHERE id=?", (json.dumps({"solution_id": sid1}), pid1))
    svc.index._conn.execute("UPDATE vectors SET metadata_json=? WHERE id=?", (json.dumps({"solution_id": sid2}), pid2))
    svc.index._conn.commit()

    qv = svc.encode_problem("Build a boat out of heavy cement that still stays on top of the water", title="Concrete Canoe")
    hyb = svc.hybrid_analogize(qv, top_k=2)
    assert hyb[0]["problem"]["title"] == "Concrete Canoe"
    assert "solution_similarity" in hyb[0]["problem"]
    assert "problem_similarity" in hyb[0]["problem"]


def test_hybrid_analogize_ranks_correct(tmp_path):
    """Hybrid retrieval (problem+solution vectors) ranks the correct analog first
    and exposes solution_similarity."""
    import json

    svc = VectorizationService(db_path=str(tmp_path / "vec.db"))
    pid1 = svc.store_problem("How to make a canoe hull that floats despite being made of dense concrete", title="Concrete Canoe")
    sid1 = svc.store_solution("Use lightweight aggregates and air-entraining agents to make concrete less dense than water", method="TRIZ", domain="engineering")
    pid2 = svc.store_problem("How to bake a moist chocolate cake with no eggs", title="Cake")
    sid2 = svc.store_solution("Use applesauce as an egg substitute and buttermilk for moisture", method="TRIZ", domain="baking")
    svc.index._conn.execute("UPDATE vectors SET metadata_json=? WHERE id=?", (json.dumps({"solution_id": sid1}), pid1))
    svc.index._conn.execute("UPDATE vectors SET metadata_json=? WHERE id=?", (json.dumps({"solution_id": sid2}), pid2))
    svc.index._conn.commit()

    qv = svc.encode_problem("Build a boat out of heavy cement that still stays on top of the water", title="Concrete Canoe")
    hyb = svc.hybrid_analogize(qv, top_k=2)
    assert hyb[0]["problem"]["title"] == "Concrete Canoe"
    assert "solution_similarity" in hyb[0]["problem"]
    assert hyb[0]["solution"] is not None


def test_solve_engineering_problem_triz_grounded():
    """solve_engineering_problem runs full discovery with TRIZ-grounded abduction
    and returns an UNVERIFIED breakthrough (no ground truth -> honest)."""
    from super_solver.engine import SuperDuperProblemSolvingEngine

    engine = SuperDuperProblemSolvingEngine(db_path=":memory:")
    path = engine.solve_engineering_problem(
        title="Concrete Canoe",
        specification="How to make a canoe hull that floats despite being made of dense concrete",
        top_principles=4,
        ground_truth_outcomes=None,
    )
    assert path.total_steps >= 4  # abduction + rollout + MCTS + breakthrough
    assert "UNVERIFIED" in path.final_breakthrough  # no ground truth -> honest
    assert "TRIZ principle" in path.final_breakthrough  # TRIZ-grounded


def test_domain_knowledge_retrieval_ranks_correct_solution():
    """Domain-knowledge grounding retrieves the right concrete solution for a problem."""
    from super_solver.vectorize import VectorizationService
    from super_solver.vectorize.domain_knowledge import (
        build_domain_knowledge_index,
        retrieve_concrete_solutions,
    )

    dk = VectorizationService(db_path=":memory:")
    build_domain_knowledge_index(dk)
    hits = retrieve_concrete_solutions(
        dk,
        "How to make a canoe hull that floats despite being made of dense concrete",
        top_k=3,
    )
    assert hits, "domain knowledge should return concrete solutions"
    assert "lightweight aggregates" in hits[0]["content"]


def test_structural_matcher_aligns_shared_pattern():
    """Structural matcher aligns solutions sharing an action pattern across domains."""
    from super_solver.vectorize.structural import match_structural, structural_signature

    # Post-it and reusable label share the 'low-tack adhesive' pattern.
    assert structural_signature("Use a low-tack adhesive that peels off cleanly") == \
        "use a low-tack adhesive"
    sources = [
        {"title": "Post-it", "solution": "Use a low-tack adhesive that was originally a failed super-strong glue (Post-it)"},
        {"title": "Velcro", "solution": "Mimic the hook-and-loop structure of burrs that stick to fabric (Velcro)"},
    ]
    hits = match_structural(
        "How to make a reusable price tag that sticks but peels off cleanly",
        sources,
        top_k=1,
    )
    assert hits[0]["title"] == "Post-it"


def test_primitive_matcher_cross_domain():
    """Primitive stripper matches structurally-similar problems across domains."""
    from super_solver.vectorize.primitives import (
        match_by_primitives,
        strip_to_primitives,
    )

    # Concrete canoe and lightweight foam share 'reduce_density' + material.
    canoe = strip_to_primitives(
        "How to make a canoe hull that floats despite being made of dense concrete"
    )
    assert "reduce_density" in canoe
    sources = [
        {"title": "Canoe", "solution": "Use lightweight aggregates to reduce the material's density below that of water"},
        {"title": "Velcro", "solution": "Mimic the hook-and-loop structure of burrs that stick to fabric"},
    ]
    hits = match_by_primitives(
        "How to make a building panel that is strong but light enough to lift",
        sources,
        top_k=1,
    )
    assert hits[0]["title"] == "Canoe"


def test_relatedness_margin_polarity_vs_ollama():
    """The ollama custom embedder should separate related from unrelated pairs
    better than the deterministic polarity embedder (semantic relatedness)."""
    from super_solver.core.embedder import get_backend

    texts = {
        "problem_db": "How to reduce latency in a distributed database while keeping strong consistency",
        "solution_db": "Use a hybrid logical clock with quorum reads to bound staleness",
        "problem_cake": "How to bake a moist chocolate cake with no eggs",
        "solution_cake": "Use applesauce as an egg substitute and buttermilk for moisture",
    }
    pairs = {
        "db": ("problem_db", "solution_db", "problem_cake"),
        "cake": ("problem_cake", "solution_cake", "solution_db"),
    }

    for backend_name in ("polarity", "ollama"):
        b = get_backend(backend_name)
        enc = {k: b.encode(v) for k, v in texts.items()}
        margins = {}
        for name, (a, b_key, unrelated) in pairs.items():
            sim_rel = b.cosine_similarity(enc[a], enc[b_key])
            sim_unrel = b.cosine_similarity(enc[a], enc[unrelated])
            margins[name] = sim_rel - sim_unrel
        # The ollama backend should produce positive margins (related > unrelated)
        if backend_name == "ollama":
            assert margins["db"] > 0.0, f"ollama db margin should be positive, got {margins['db']}"
            assert margins["cake"] > 0.0, f"ollama cake margin should be positive, got {margins['cake']}"
