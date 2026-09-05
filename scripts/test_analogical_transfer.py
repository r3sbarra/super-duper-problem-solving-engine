#!/usr/bin/env python3
"""Test whether super-solver can find/solve real-world problems that were
solved with novel solutions, via analogical transfer.

Method (honest test):
1. Seed the vector index with KNOWN problem->solution->path cases (real
   novel-solution engineering/science cases from the web).
2. Query with a NOVEL, rephrased formulation of each problem — wording the
   solver has NOT seen (different words, same underlying problem).
3. Check whether analogical transfer retrieves the CORRECT solution as the
   top hit. This tests genuine semantic relatedness, not string matching.

Runs with both the polarity and ollama custom embedders so we can compare.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from super_solver.core.embedder import get_backend  # noqa: E402
from super_solver.vectorize import VectorizationService  # noqa: E402

# Real problem -> novel solution pairs, with a NOVEL query formulation
CASES = [
    {
        "title": "Concrete Canoe",
        "problem": "How to make a canoe hull that floats despite being made of dense concrete",
        "query": "Build a boat out of heavy cement that still stays on top of the water",
        "solution": "Use lightweight aggregates and air-entraining agents to make concrete less dense than water",
        "path": ["identify density contradiction", "add air-entraining agents", "reduce aggregate density", "float the hull"],
    },
    {
        "title": "Smog-Eating Church",
        "problem": "How to reduce urban air pollution from building surfaces",
        "query": "Cut down city smog using the walls of buildings themselves",
        "solution": "Use photocatalytic cement that breaks down air pollutants when exposed to sunlight",
        "path": ["identify pollution source", "apply photocatalytic coating", "activate with sunlight", "clean surrounding air"],
    },
    {
        "title": "Boeing Camouflage Factory",
        "problem": "How to hide a large bomber factory from aerial reconnaissance during wartime",
        "query": "Conceal a giant aircraft plant so enemy planes flying overhead cannot spot it",
        "solution": "Transform the factory roof into a fake suburban neighborhood with plywood houses and canvas streets",
        "path": ["identify aerial threat", "build fake town on roof", "add props and mannequins", "deceive reconnaissance"],
    },
    {
        "title": "Hybrid Logical Clock",
        "problem": "How to reduce latency in a distributed database while keeping strong consistency",
        "query": "Speed up a replicated data store without sacrificing that all nodes agree on ordering",
        "solution": "Use a hybrid logical clock with quorum reads to bound staleness",
        "path": ["abduct hypothesis", "boundary filter", "MCTS rollout", "breakthrough"],
    },
    {
        "title": "LPG Parameter Change",
        "problem": "How to transport a large volume of gas efficiently in a small container",
        "query": "Ship a huge amount of fuel gas in a compact tank that fits on a truck",
        "solution": "Change the physical state of the gas to liquid under pressure to reduce transport volume",
        "path": ["identify volume problem", "apply parameter change principle", "liquefy under pressure", "reduce transport volume"],
    },
    {
        "title": "Flat-Pack Furniture",
        "problem": "How to make furniture that assembles quickly without special tools",
        "query": "Design a bookshelf a customer can put together in minutes with no drill or screwdriver",
        "solution": "Pre-drill holes in flat-pack furniture so assembly is immediate",
        "path": ["identify assembly friction", "apply preliminary action principle", "pre-drill holes", "enable immediate assembly"],
    },
    {
        "title": "Stick Vacuum Segmentation",
        "problem": "How to make a vacuum cleaner that works both as a full unit and a portable handheld",
        "query": "One cleaning appliance that is both a full-size floor vacuum and a small grab-and-go unit",
        "solution": "Divide the vacuum into independent parts so the handheld unit detaches",
        "path": ["identify portability need", "apply segmentation principle", "detach handheld unit", "dual-mode operation"],
    },
    {
        "title": "Molasses Flood Safety",
        "problem": "How to prevent catastrophic failure of large liquid storage tanks",
        "query": "Stop giant tanks of thick liquid from bursting and flooding the neighborhood",
        "solution": "Pioneer expert-witness safety analysis and stricter tank inspection standards after the 1919 Boston molasses flood",
        "path": ["identify tank failure", "analyze structural causes", "establish expert witness standards", "prevent recurrence"],
    },
]


def run(backend_name: str, db_path: str) -> dict:
    svc = VectorizationService(backend=get_backend(backend_name), db_path=db_path)
    # Seed the index with the KNOWN (canonical) problem formulations
    for case in CASES:
        pid = svc.store_problem(case["problem"], title=case["title"])
        sid = svc.store_solution(case["solution"], method="TRIZ", domain="engineering", title=case["title"])
        pathid = svc.store_path(case["path"], operator_types=["PEIRCE", "KT", "TRIZ", "PLATT"], final_breakthrough=case["solution"], title=case["title"])
        svc.index._conn.execute(
            "UPDATE vectors SET metadata_json=? WHERE id=?",
            (json.dumps({"solution_id": sid, "path_id": pathid}), pid),
        )
    svc.index._conn.commit()

    # Query with the NOVEL formulation; check if the correct solution is retrieved
    results = []
    for case in CASES:
        qv = svc.encode_problem(case["query"], title=case["title"])
        analogs = svc.analogize(qv, top_k=3)
        if not analogs:
            results.append({"title": case["title"], "found": False, "top_sim": 0.0, "correct": False})
            continue
        top = analogs[0]
        top_sol = top["solution"]["content"] if top["solution"] else ""
        correct = top_sol == case["solution"]
        results.append({
            "title": case["title"],
            "found": True,
            "top_sim": round(top["problem"]["similarity"], 3),
            "correct": correct,
            "retrieved_solution": top_sol[:55],
        })
    return {"backend": backend_name, "results": results}


def main():
    print("=" * 78)
    print("super-solver analogical-transfer test: NOVEL problem formulations")
    print("=" * 78)
    for backend_name in ("polarity", "ollama"):
        db = f"/tmp/vec_novel_{backend_name}.db"
        if os.path.exists(db):
            os.remove(db)
        out = run(backend_name, db)
        results = out["results"]
        n_correct = sum(1 for r in results if r["correct"])
        n_found = sum(1 for r in results if r["found"])
        print(f"\n[{backend_name}] found={n_found}/{len(results)} correct={n_correct}/{len(results)}")
        for r in results:
            mark = "✓" if r["correct"] else ("~" if r["found"] else "✗")
            print(f"  {mark} {r['title']:28s} sim={r['top_sim']:.3f}  {r.get('retrieved_solution','(not found)')}")


if __name__ == "__main__":
    main()
