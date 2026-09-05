#!/usr/bin/env python3
"""End-to-end demo: super-solver problem/solution/path vectorizing with a
custom embedder, verified through falsci and exposed via lab-ass.

Run:  .venv/bin/python scripts/demo_vectorizing.py
"""

from __future__ import annotations

import json
import os
import sys

# Ensure the engine is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from super_solver.core.embedder import get_backend  # noqa: E402
from super_solver.vectorize import VectorizationService  # noqa: E402


def main():
    print("=" * 70)
    print("super-solver vectorizing demo (custom embedder + falsci + lab-ass)")
    print("=" * 70)

    # 1. Custom embedder backends
    print("\n[1] Custom embedder backends")
    for name in ("polarity", "ollama", "hybrid"):
        b = get_backend(name)
        v = b.encode("How to reduce latency in a distributed database")
        print(f"    {name:9s} dim={v.shape[0]:5d}")

    # 2. Vectorize a problem, solution, and path; index them
    print("\n[2] Vectorize + index problem/solution/path")
    svc = VectorizationService()
    pid = svc.store_problem(
        "How to reduce latency in a distributed database while keeping strong consistency",
        title="DB latency",
    )
    sid = svc.store_solution(
        "Use a hybrid logical clock with quorum reads to bound staleness",
        method="TRIZ #1 Segmentation",
        domain="distributed systems",
    )
    pathid = svc.store_path(
        ["abduct hypothesis", "boundary filter", "MCTS rollout", "breakthrough"],
        operator_types=["PEIRCE", "KT", "MCTS", "PLATT"],
        final_breakthrough="hybrid logical clock",
    )
    # Link solution + path to the problem for analogical transfer
    svc.index._conn.execute(
        "UPDATE vectors SET metadata_json=? WHERE id=?",
        (json.dumps({"solution_id": sid, "path_id": pathid}), pid),
    )
    svc.index._conn.commit()
    print(f"    stored problem={pid[:8]} solution={sid[:8]} path={pathid[:8]}")

    # 3. Analogical transfer
    print("\n[3] Analogical transfer (find similar past problem -> reuse its path)")
    qv = svc.encode_problem(
        "How to reduce latency in a distributed database while keeping strong consistency",
        title="DB latency",
    )
    analogs = svc.analogize(qv, top_k=3)
    if analogs:
        a = analogs[0]
        print(f"    similar problem: {a['problem']['content'][:60]}...")
        print(f"    linked solution: {a['solution']['content'] if a['solution'] else '(none)'}")
        print(f"    linked path:     {a['path']['content'] if a['path'] else '(none)'}")

    # 4. falsci verification: polarity vs ollama embedder relatedness
    print("\n[4] falsci verification (semantic relatedness margin)")
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
        for name, (a, rel, unrel) in pairs.items():
            margins[name] = b.cosine_similarity(enc[a], enc[rel]) - b.cosine_similarity(enc[a], enc[unrel])
        verdict = "PASS" if all(m > 0 for m in margins.values()) else "FAIL"
        print(f"    {backend_name:9s} db_margin={margins['db']:+.3f} cake_margin={margins['cake']:+.3f} -> {verdict}")

    # 5. lab-ass exposure (REST)
    print("\n[5] lab-ass REST exposure (vectorize/search/analogize)")
    import httpx

    try:
        r = httpx.post(
            "http://127.0.0.1:3002/api/sessions",
            json={"title": "vectorizer-demo", "topic": "demo"},
            timeout=5,
        )
        sid_session = r.json().get("id", "")
        if sid_session:
            vr = httpx.post(
                f"http://127.0.0.1:3002/api/sessions/{sid_session}/super-solver/vectorize",
                json={"kind": "problem", "specification": "How to reduce latency in a distributed database", "title": "DB latency"},
                timeout=10,
            ).json()
            print(f"    lab-ass vectorize: available={vr.get('available')} dim={vr.get('dim')}")
        else:
            print("    lab-ass: could not create session (is it running?)")
    except Exception as exc:
        print(f"    lab-ass: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
