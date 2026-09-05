"""Test whether the large engineering corpus improves super-solver GENERATION.

Loads the 13k engineering corpus (TRIZ + AskNature) into a domain-knowledge
VectorizationService, then runs the full discovery generation pipeline on the
classical engineering cases. Compares generation quality with the large
knowledge base vs the small built-in DOMAIN_KNOWLEDGE.

Usage:
  .venv/bin/python scripts/test_generation_large_corpus.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from super_solver.core.embedder import get_backend
from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.vectorize import VectorizationService

from train_problem_solving import CLASSICAL_CASES, RECENT_CASES


def load_corpus_index(corpus_path, backend, max_entries=None):
    """Load a problem->solution corpus into a VectorizationService as domain knowledge."""
    with open(corpus_path) as f:
        corpus = json.load(f)
    if max_entries:
        corpus = corpus[:max_entries]
    svc = VectorizationService(db_path=":memory:", backend=backend)
    for e in corpus:
        prob = e["problem"]
        sol = e["solution"]
        pid = svc.store_problem(specification=prob, title=prob[:60], metadata={"kind": "domain_knowledge"})
        sid = svc.store_solution(
            solution_text=sol, method="domain_knowledge", domain="engineering",
            title=prob[:60], metadata={"problem_context": prob},
        )
        svc.index._conn.execute(
            "UPDATE vectors SET metadata_json=? WHERE id=?",
            (json.dumps({"solution_id": sid, "kind": "domain_knowledge"}), pid),
        )
    svc.index._conn.commit()
    return svc


def cosine(a, b):
    import numpy as np

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="scratch/eng_corpus.json")
    ap.add_argument("--embedder", default="polarity")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--recent", action="store_true")
    args = ap.parse_args()

    cases = list(CLASSICAL_CASES)
    if args.recent:
        cases += list(RECENT_CASES)
    if args.limit:
        cases = cases[: args.limit]

    backend = get_backend(args.embedder)
    dk = load_corpus_index(args.corpus, backend, max_entries=args.limit)
    print(f"Loaded {len(cases)} test cases, corpus index seeded")

    engine = SuperDuperProblemSolvingEngine(db_path=":memory:")
    correct = 0
    sims = []
    for case in cases:
        path = engine.solve_engineering_problem(
            title=case["title"],
            specification=case["problem"],
            goal_criteria=["Find a novel solution to this problem"],
            top_principles=4,
            ground_truth_outcomes=None,
            domain_knowledge=dk,
            concrete_top_k=3,
        )
        breakthrough = path.final_breakthrough
        # Strip UNVERIFIED/CONFIRMED prefix.
        for prefix in ("UNVERIFIED: ", "CONFIRMED: "):
            if breakthrough.startswith(prefix):
                breakthrough = breakthrough[len(prefix):]
        sim = cosine(backend.encode(breakthrough), backend.encode(case["solution"]))
        sims.append(sim)
        ok = sim >= 0.6
        if ok:
            correct += 1
        print(f"{'✓' if ok else '~'} {case['title']:<28} sim={sim:.3f} | {breakthrough[:50]}")

    avg = sum(sims) / len(sims) if sims else 0
    print(f"\nRESULT: {correct}/{len(cases)} correct @ sim>=0.6, avg_sim={avg:.3f}")


if __name__ == "__main__":
    main()
