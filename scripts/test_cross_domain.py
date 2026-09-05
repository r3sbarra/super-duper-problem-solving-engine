#!/usr/bin/env python3
"""Cross-domain transfer test: apply a solution from one domain to a
structurally similar problem in another.

The engine's analogical-transfer machinery should retrieve a solution learned
in domain A and apply it to a problem in domain B that shares the same
underlying structure but has different surface entities.

Each case has:
  - source_problem: seeded into the index (domain A)
  - source_solution: the concrete solution in domain A
  - target_problem: a NEW problem in domain B (NOT seeded) with the same structure
  - target_solution: the correct application of the source solution to domain B

The honest test: query with the target problem, and check whether the engine
retrieves the source solution AND applies it correctly to the target domain.

Usage:
  .venv/bin/python scripts/test_cross_domain.py [--embedder ollama]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from super_solver.core.embedder import get_backend
from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.vectorize import VectorizationService
from super_solver.vectorize.domain_knowledge import build_domain_knowledge_index

# ---------------------------------------------------------------------------
# Cross-domain cases: same structure, different surface entities.
# ---------------------------------------------------------------------------
CROSS_DOMAIN_CASES = [
    {
        "title": "Velcro -> Gecko Climbing",
        "source_problem": "How to create a fastener that can be opened and closed repeatedly without wearing out",
        "source_solution": "Mimic the hook-and-loop structure of burrs that stick to fabric (Velcro)",
        "target_problem": "How to make a climbing pad that grips smooth vertical walls without glue or suction",
        "target_solution": "Mimic the microscopic setae structure of gecko feet that adhere to smooth surfaces",
    },
    {
        "title": "Concrete Canoe -> Lightweight Foam",
        "source_problem": "How to make a canoe hull that floats despite being made of dense concrete",
        "source_solution": "Use lightweight aggregates and air-entraining agents to reduce the material's density below that of water",
        "target_problem": "How to make a building panel that is strong but light enough to lift and install by hand",
        "target_solution": "Use lightweight aggregates and air-entraining agents to reduce the panel's density while keeping strength",
    },
    {
        "title": "Photocatalytic Cement -> Self-Cleaning Glass",
        "source_problem": "How to reduce urban air pollution from building surfaces",
        "source_solution": "Use photocatalytic cement (titanium dioxide) that breaks down air pollutants when exposed to sunlight",
        "target_problem": "How to keep skyscraper windows clean without manual washing",
        "target_solution": "Use a photocatalytic titanium dioxide coating on the glass that breaks down dirt when exposed to sunlight",
    },
    {
        "title": "LPG Liquefaction -> Hydrogen Transport",
        "source_problem": "How to transport a large volume of gas efficiently in a small container",
        "source_solution": "Liquefy the gas under pressure and cold to reduce its volume dramatically for transport",
        "target_problem": "How to ship hydrogen fuel efficiently in a compact tank on a truck",
        "target_solution": "Liquefy the hydrogen under extreme cold to reduce its volume for transport",
    },
    {
        "title": "Hybrid Clock -> Sensor Sync",
        "source_problem": "How to reduce latency in a distributed database while keeping strong consistency",
        "source_solution": "Use a hybrid logical clock with quorum reads to bound staleness while keeping ordering",
        "target_problem": "How to keep a network of IoT sensors time-synchronized without a central clock",
        "target_solution": "Use a hybrid logical clock combining physical time with logical counters across the sensor network",
    },
    {
        "title": "Boeing Camouflage -> Military Decoy",
        "source_problem": "How to hide a large bomber factory from aerial reconnaissance during wartime",
        "source_solution": "Camouflage the structure to resemble its surroundings, e.g. a fake neighborhood on a factory roof",
        "target_problem": "How to protect a military base from satellite surveillance",
        "target_solution": "Camouflage the base to resemble its surroundings so it is invisible from above",
    },
    {
        "title": "Post-it -> Reusable Label",
        "source_problem": "How to make a paper note that sticks temporarily and can be removed without residue",
        "source_solution": "Use a low-tack adhesive that was originally a failed super-strong glue (Post-it)",
        "target_problem": "How to make a reusable price tag that sticks to products but peels off cleanly",
        "target_solution": "Use a low-tack adhesive that holds firmly but peels off without leaving residue",
    },
    {
        "title": "Suspension Bridge -> Cable-Stayed Roof",
        "source_problem": "How to span a wide river without building many support pillars in the water",
        "source_solution": "Hang the deck from cables suspended between tall towers, transferring load to the towers",
        "target_problem": "How to build a stadium roof that covers a huge area without interior columns",
        "target_solution": "Hang the roof from cables suspended between perimeter towers, transferring load to the towers",
    },
]


def cosine(a, b):
    a = np.asarray(a, float).reshape(1, -1)
    b = np.asarray(b, float).reshape(1, -1)
    d = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    d[d == 0] = 1e-9
    return float((a @ b.T)[0, 0] / d[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="ollama", choices=["polarity", "ollama"])
    args = ap.parse_args()
    get_backend(args.embedder)

    # Seed the domain-knowledge index (source solutions live here).
    dk = VectorizationService(db_path=":memory:")
    build_domain_knowledge_index(dk)
    # Also seed each source problem->solution pair explicitly.
    for case in CROSS_DOMAIN_CASES:
        pid = dk.store_problem(specification=case["source_problem"], title=case["title"], metadata={"kind": "source"})
        sid = dk.store_solution(solution_text=case["source_solution"], method="source", domain="engineering", title=case["title"], metadata={"problem_context": case["source_problem"]})
        import json
        dk.index._conn.execute("UPDATE vectors SET metadata_json=? WHERE id=?", (json.dumps({"solution_id": sid, "kind": "source"}), pid))
    dk.index._conn.commit()

    engine = SuperDuperProblemSolvingEngine(db_path=":memory:")

    print(f"Cross-domain transfer test on {len(CROSS_DOMAIN_CASES)} cases, embedder={args.embedder}")
    print("=" * 78)

    correct = 0
    total = 0
    sims = []
    for case in CROSS_DOMAIN_CASES:
        title = case["title"]
        # Query with the TARGET problem (domain B, never seeded).
        path = engine.solve_engineering_problem(
            title=title,
            specification=case["target_problem"],
            goal_criteria=["Find a novel solution to this problem"],
            top_principles=4,
            ground_truth_outcomes=None,
            domain_knowledge=dk,
            concrete_top_k=3,
        )
        gen_text = path.final_breakthrough.split(": ", 1)[-1] if ": " in path.final_breakthrough else path.final_breakthrough
        sim = cosine(get_backend().encode(gen_text), get_backend().encode(case["target_solution"]))
        sims.append(sim)
        total += 1
        ok = sim >= 0.60
        if ok:
            correct += 1
        mark = "✓" if ok else "~"
        print(f"  {mark} {title:<30} sim={sim:.3f}")
        print(f"      generated: {gen_text[:100]}")
        print(f"      expected:  {case['target_solution'][:100]}")

    print("=" * 78)
    avg = sum(sims) / len(sims) if sims else 0.0
    print(f"RESULT: cross-domain transfer {correct}/{total} correct @ sim>=0.60, avg_sim={avg:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
