#!/usr/bin/env python3
"""verify_improvements.py — Direct verification of the newly added Super-Solver improvements.

Verifies:
1. P1: Two-stage retrieval reranker (VectorIndex.search_reranked).
2. P2: Solution Mechanism Synthesizer (SolutionMechanismSynthesizer + engine integration).
3. P3: Suuri-chan Milner LCF formal micro-kernel coupling (LakatosProofVerificationEngine + ProofCertificate).
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

results = {}

# ---------------------------------------------------------------------------
# 1. Verify P1: Two-Stage Retrieval Reranker
# ---------------------------------------------------------------------------
print("[1/3] Testing P1: Two-Stage Retrieval Reranker...")
from super_solver.vectorize.index import VectorIndex

idx = VectorIndex(db_path=":memory:")
doc1 = "Use an in-memory LRU cache to store query results and reduce database latency"
doc2 = "Use a database index on primary key columns to make query scans faster"
doc3 = "Use rate limiting on the API gateway to prevent resource exhaustion from requests"

v1 = idx.backend.encode(doc1)
v2 = idx.backend.encode(doc2)
v3 = idx.backend.encode(doc3)

idx.add("solution", doc1, v1, title="LRU cache mechanism")
idx.add("solution", doc2, v2, title="Index scan mechanism")
idx.add("solution", doc3, v3, title="Rate limiter mechanism")

query = "How to cache database queries in memory to lower latency"
hits = idx.search_reranked(query, kind="solution", top_k=2)

assert len(hits) == 2, f"Expected 2 hits, got {len(hits)}"
assert "cache" in hits[0]["content"].lower(), f"Top hit did not contain cache: {hits[0]}"
assert "rerank_score" in hits[0], "Missing rerank_score"
assert hits[0]["rerank_breakdown"]["joint_token"] > 0.0, "joint_token score should be > 0"
idx.close()
print("  [PASS] P1 Two-Stage Retrieval Reranker: correctly ranked cache mechanism at rank 1 with score breakdown.")
results["p1_retrieval_reranker"] = {
    "status": "PASS",
    "top_hit_title": hits[0]["title"],
    "rerank_score": hits[0]["rerank_score"],
    "breakdown": hits[0]["rerank_breakdown"],
}

# ---------------------------------------------------------------------------
# 2. Verify P2: Solution Mechanism Synthesizer
# ---------------------------------------------------------------------------
print("[2/3] Testing P2: Solution Mechanism Synthesizer...")
from super_solver.frameworks.solution_synthesis import SolutionMechanismSynthesizer

sms = SolutionMechanismSynthesizer()
cat = sms.infer_category("Reduce cache invalidation spikes in p99 latency")
assert cat == "latency", f"Expected latency category, got {cat}"

mechs = sms.synthesize_mechanisms_for_subproblem(
    subproblem_text="Reduce cache invalidation spikes in p99 latency",
    problem_context="High-throughput distributed API gateway",
    max_mechanisms=2,
)
assert len(mechs) == 2, f"Expected 2 mechanisms, got {len(mechs)}"
assert mechs[0]["action_primitive"] == "act_cache", f"Expected act_cache, got {mechs[0]['action_primitive']}"

# Test engine integration
from super_solver.engine import SuperDuperProblemSolvingEngine

engine = SuperDuperProblemSolvingEngine()
spec = "Shortest-path routing for a delivery fleet under time-window constraints"
subproblems = [
    {"subproblem": "Formulate routing objective with vehicle capacities"},
    {"subproblem": "Handle non-linear time-window arrival penalties"},
]
engine_mechs = engine.synthesize_solution_mechanisms(spec, subproblems=subproblems)
assert len(engine_mechs) >= 2, f"Expected >= 2 engine mechanisms, got {len(engine_mechs)}"
print(f"  [PASS] P2 Solution Mechanism Synthesizer: synthesized {len(engine_mechs)} mechanisms from subproblems.")
results["p2_solution_synthesis"] = {
    "status": "PASS",
    "inferred_category": cat,
    "synthesized_sample": mechs[0]["description"],
    "engine_synthesis_count": len(engine_mechs),
}

# ---------------------------------------------------------------------------
# 3. Verify P3: Suuri-chan Milner LCF Formal Micro-Kernel Coupling
# ---------------------------------------------------------------------------
print("[3/3] Testing P3: Suuri-chan Milner LCF Micro-Kernel Coupling...")
from super_solver.frameworks.math_verification import LakatosProofVerificationEngine

verifier = LakatosProofVerificationEngine()
lcf_refl = verifier.verify_with_lcf_microkernel("a + b = a + b")
assert lcf_refl is not None and lcf_refl["is_valid"] is True, f"LCF refl check failed: {lcf_refl}"
assert lcf_refl["rule"] == "refl", f"Expected refl rule, got {lcf_refl.get('rule')}"

lcf_sym = verifier.verify_with_lcf_microkernel("1/3 + 5/7")
assert lcf_sym is not None and lcf_sym["is_valid"] is True, f"LCF sym check failed: {lcf_sym}"
assert lcf_sym["value"] == "22/21", f"Expected 22/21, got {lcf_sym.get('value')}"

paper_res = verifier.verify_mathematical_paper(
    paper_id="test.lcf.01",
    paper_title="Exact rational evaluation of 1/3 + 5/7",
    abstract_text="We establish that 1/3 + 5/7 reduces unconditionally to 22/21.",
    target_conjecture="1/3 + 5/7",
)
assert paper_res.lcf_verified is True, "Expected paper_res.lcf_verified to be True"
assert paper_res.formal_certificate is not None, "Expected formal_certificate in result"
assert any("FORMAL LCF CERTIFICATE" in t for t in paper_res.reasoning_trace), "Missing trace entry"
print(f"  [PASS] P3 Suuri-chan LCF Coupling: successfully emitted formal certificate with {paper_res.formal_certificate['steps_count']} steps.")
results["p3_suuri_lcf_coupling"] = {
    "status": "PASS",
    "refl_check_rule": lcf_refl["rule"],
    "symbolic_evaluation": lcf_sym["value"],
    "paper_lcf_verified": paper_res.lcf_verified,
    "formal_certificate_steps": paper_res.formal_certificate["steps_count"],
}

print("\n" + "=" * 60)
print("ALL 3 IMPROVEMENT MODULES VERIFIED CLEANLY & PASSING")
print("=" * 60)
print(json.dumps(results, indent=2))
