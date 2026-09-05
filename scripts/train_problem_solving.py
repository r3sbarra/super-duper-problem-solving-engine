#!/usr/bin/env python3
"""Train super-solver on classical + recent problem-solving cases.

Builds a training corpus of problem->solution->path cases (classical TRIZ /
engineering / science classics + recent 2024-26 novel solutions), seeds the
vector index, and measures retrieval quality on novel rephrased queries.

Usage:
  .venv/bin/python scripts/train_problem_solving.py [--embedder polarity|ollama] [--recent]

Each case has:
  - title
  - problem: canonical problem statement (seeded into index)
  - query: NOVEL rephrased formulation (used to test retrieval, NOT seeded)
  - solution: the actual novel solution used
  - path: sequence of reasoning steps
  - operators: operator types per step
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from super_solver.core.embedder import get_backend  # noqa: E402
from super_solver.vectorize import VectorizationService  # noqa: E402

# ---------------------------------------------------------------------------
# Classical problem-solving corpus (well-documented, proven novel solutions)
# ---------------------------------------------------------------------------
CLASSICAL_CASES = [
    {
        "title": "Concrete Canoe",
        "problem": "How to make a canoe hull that floats despite being made of dense concrete",
        "query": "Build a boat out of heavy cement that still stays on top of the water",
        "solution": "Use lightweight aggregates and air-entraining agents to make concrete less dense than water",
        "path": ["identify density contradiction", "add air-entraining agents", "reduce aggregate density", "float the hull"],
        "operators": ["KT", "TRIZ#35", "TRIZ#1", "PLATT"],
    },
    {
        "title": "Smog-Eating Church",
        "problem": "How to reduce urban air pollution from building surfaces",
        "query": "Cut down city smog using the walls of buildings themselves",
        "solution": "Use photocatalytic cement that breaks down air pollutants when exposed to sunlight",
        "path": ["identify pollution source", "apply photocatalytic coating", "activate with sunlight", "clean surrounding air"],
        "operators": ["KT", "TRIZ#35", "TRIZ#24", "PLATT"],
    },
    {
        "title": "Boeing Camouflage Factory",
        "problem": "How to hide a large bomber factory from aerial reconnaissance during wartime",
        "query": "Conceal a giant aircraft plant so enemy planes flying overhead cannot spot it",
        "solution": "Transform the factory roof into a fake suburban neighborhood with plywood houses and canvas streets",
        "path": ["identify aerial threat", "build fake town on roof", "add props and mannequins", "deceive reconnaissance"],
        "operators": ["KT", "TRIZ#1", "TRIZ#32", "PLATT"],
    },
    {
        "title": "Hybrid Logical Clock",
        "problem": "How to reduce latency in a distributed database while keeping strong consistency",
        "query": "Speed up a replicated data store without sacrificing that all nodes agree on ordering",
        "solution": "Use a hybrid logical clock with quorum reads to bound staleness",
        "path": ["abduct hypothesis", "boundary filter", "MCTS rollout", "breakthrough"],
        "operators": ["PEIRCE", "KT", "MCTS", "PLATT"],
    },
    {
        "title": "LPG Parameter Change",
        "problem": "How to transport a large volume of gas efficiently in a small container",
        "query": "Ship a huge amount of fuel gas in a compact tank that fits on a truck",
        "solution": "Change the physical state of the gas to liquid under pressure to reduce transport volume",
        "path": ["identify volume problem", "apply parameter change principle", "liquefy under pressure", "reduce transport volume"],
        "operators": ["KT", "TRIZ#35", "TRIZ#10", "PLATT"],
    },
    {
        "title": "Flat-Pack Furniture",
        "problem": "How to make furniture that assembles quickly without special tools",
        "query": "Design a bookshelf a customer can put together in minutes with no drill or screwdriver",
        "solution": "Pre-drill holes in flat-pack furniture so assembly is immediate",
        "path": ["identify assembly friction", "apply preliminary action principle", "pre-drill holes", "enable immediate assembly"],
        "operators": ["KT", "TRIZ#10", "TRIZ#1", "PLATT"],
    },
    {
        "title": "Stick Vacuum Segmentation",
        "problem": "How to make a vacuum cleaner that works both as a full unit and a portable handheld",
        "query": "One cleaning appliance that is both a full-size floor vacuum and a small grab-and-go unit",
        "solution": "Divide the vacuum into independent parts so the handheld unit detaches",
        "path": ["identify portability need", "apply segmentation principle", "detach handheld unit", "dual-mode operation"],
        "operators": ["KT", "TRIZ#1", "TRIZ#15", "PLATT"],
    },
    {
        "title": "Molasses Flood Safety",
        "problem": "How to prevent catastrophic failure of large liquid storage tanks",
        "query": "Stop giant tanks of thick liquid from bursting and flooding the neighborhood",
        "solution": "Pioneer expert-witness safety analysis and stricter tank inspection standards after the 1919 Boston molasses flood",
        "path": ["identify tank failure", "analyze structural causes", "establish expert witness standards", "prevent recurrence"],
        "operators": ["KT", "TRIZ#35", "TRIZ#11", "PLATT"],
    },
    {
        "title": "Velcro Burr",
        "problem": "How to create a fastener that can be opened and closed repeatedly without wearing out",
        "query": "A reusable closure for clothing that does not need a zipper or buttons",
        "solution": "Mimic the hook-and-loop structure of burrs that stick to fabric",
        "path": ["observe burr sticking", "abstract hook-loop mechanism", "synthesize nylon hooks", "reusable fastener"],
        "operators": ["PEIRCE", "GENTNER", "TRIZ#5", "PLATT"],
    },
    {
        "title": "Post-it Note",
        "problem": "How to make a paper note that sticks temporarily and can be removed without residue",
        "query": "A sticky note that holds to a surface but peels off cleanly every time",
        "solution": "Use a low-tack adhesive that was originally a failed super-strong glue",
        "path": ["discover weak adhesive", "recognize temporary-stick value", "apply to paper", "repositionable note"],
        "operators": ["PEIRCE", "KT", "TRIZ#25", "PLATT"],
    },
    {
        "title": "Microwave Radar Magnetron",
        "problem": "How to heat food quickly and evenly without an external heat source",
        "query": "Cook a meal in seconds using electromagnetic waves instead of a stove",
        "solution": "Use a magnetron to generate microwaves that excite water molecules in food",
        "path": ["identify radar magnetron", "recognize heating effect", "apply to food", "microwave oven"],
        "operators": ["PEIRCE", "GENTNER", "TRIZ#35", "PLATT"],
    },
    {
        "title": "Teflon Nonstick",
        "problem": "How to make a cooking surface that food does not stick to",
        "query": "A frying pan where eggs slide off without oil or butter",
        "solution": "Apply a PTFE (Teflon) coating with extremely low surface friction",
        "path": ["discover PTFE", "recognize low friction", "coat cookware", "nonstick pan"],
        "operators": ["PEIRCE", "KT", "TRIZ#35", "PLATT"],
    },
    {
        "title": "Silly Putty",
        "problem": "How to create a material that is both elastic and moldable",
        "query": "A toy that bounces like rubber but can also be shaped like clay",
        "solution": "Use a silicone polymer that exhibits both viscous and elastic behavior",
        "path": ["discover silicone polymer", "recognize dual behavior", "market as toy", "silly putty"],
        "operators": ["PEIRCE", "KT", "TRIZ#35", "PLATT"],
    },
    {
        "title": "Pacemaker",
        "problem": "How to regulate an irregular heartbeat without surgery",
        "query": "A device that keeps a failing heart beating at a normal rhythm",
        "solution": "Implant a small electronic device that delivers electrical pulses to the heart",
        "path": ["identify heart arrhythmia", "design pulse generator", "implant device", "regulate heartbeat"],
        "operators": ["KT", "TRIZ#35", "TRIZ#1", "PLATT"],
    },
    {
        "title": "X-Ray",
        "problem": "How to see inside the human body without cutting it open",
        "query": "Look at broken bones through the skin without surgery",
        "solution": "Use high-energy electromagnetic radiation that passes through soft tissue but is absorbed by bone",
        "path": ["discover X-rays", "recognize bone absorption", "image internal structure", "medical imaging"],
        "operators": ["PEIRCE", "KT", "TRIZ#35", "PLATT"],
    },
    {
        "title": "Penicillin",
        "problem": "How to treat bacterial infections that were previously fatal",
        "query": "A medicine that kills harmful bacteria without harming the patient",
        "solution": "Use a mold-derived antibiotic that disrupts bacterial cell wall synthesis",
        "path": ["observe mold kills bacteria", "isolate active compound", "test on infections", "antibiotic therapy"],
        "operators": ["PEIRCE", "KT", "TRIZ#35", "PLATT"],
    },
    {
        "title": "Safety Razor",
        "problem": "How to shave without cutting the skin",
        "query": "A razor that removes hair but does not nick the face",
        "solution": "Guard the blade with a protective comb so only a thin edge contacts the skin",
        "path": ["identify blade hazard", "add protective guard", "limit blade exposure", "safe shave"],
        "operators": ["KT", "TRIZ#1", "TRIZ#24", "PLATT"],
    },
    {
        "title": "Zipper",
        "problem": "How to fasten clothing quickly and securely without buttons or laces",
        "query": "A closure that joins two fabric edges in one smooth motion",
        "solution": "Interlock a series of teeth with a sliding mechanism that meshes them together",
        "path": ["identify fastening need", "design interlocking teeth", "add slider", "zipper closure"],
        "operators": ["KT", "TRIZ#1", "TRIZ#15", "PLATT"],
    },
    {
        "title": "Bifocal Lens",
        "problem": "How to correct both near and far vision in a single pair of glasses",
        "query": "Eyeglasses that let you read a book and also see distant objects clearly",
        "solution": "Combine two lens powers in one lens, with the near-vision segment at the bottom",
        "path": ["identify dual vision need", "combine two powers", "position near segment", "bifocal lens"],
        "operators": ["KT", "TRIZ#1", "TRIZ#35", "PLATT"],
    },
    {
        "title": "Suspension Bridge",
        "problem": "How to span a wide river without building many support pillars in the water",
        "query": "Cross a huge gorge with a bridge that needs almost no supports in the middle",
        "solution": "Hang the deck from cables suspended between tall towers, transferring load to the towers",
        "path": ["identify span problem", "use tension cables", "add tall towers", "suspension bridge"],
        "operators": ["KT", "TRIZ#35", "TRIZ#1", "PLATT"],
    },
]

# ---------------------------------------------------------------------------
# Recent (2024-26) novel-solution cases
# ---------------------------------------------------------------------------
RECENT_CASES = [
    {
        "title": "Self-Healing Concrete",
        "problem": "How to repair cracks in concrete infrastructure automatically without manual maintenance",
        "query": "Fix road and building cracks on their own so crews do not have to patch them",
        "solution": "Embed bacteria that produce limestone when activated by water seeping into cracks",
        "path": ["identify crack problem", "embed healing bacteria", "activate with water", "self-heal concrete"],
        "operators": ["KT", "TRIZ#35", "TRIZ#24", "PLATT"],
    },
    {
        "title": "Perovskite Solar Cells",
        "problem": "How to make solar panels cheaper and more efficient than silicon",
        "query": "Generate electricity from sunlight using a material that is cheaper than silicon",
        "solution": "Use perovskite crystal structures that absorb light across a wider spectrum at lower cost",
        "path": ["identify silicon limits", "test perovskite", "optimize stability", "cheaper solar"],
        "operators": ["KT", "TRIZ#35", "TRIZ#1", "PLATT"],
    },
    {
        "title": "CRISPR Gene Editing",
        "problem": "How to precisely edit DNA to correct genetic diseases",
        "query": "Change a specific gene in living cells to fix a hereditary illness",
        "solution": "Use CRISPR-Cas9 to make targeted cuts in DNA guided by a matching RNA sequence",
        "path": ["identify gene target", "design guide RNA", "deliver Cas9", "edit DNA"],
        "operators": ["KT", "TRIZ#1", "TRIZ#35", "PLATT"],
    },
    {
        "title": "Solid-State Battery",
        "problem": "How to make batteries that store more energy and do not catch fire",
        "query": "A rechargeable battery with higher capacity that never overheats or explodes",
        "solution": "Replace the liquid electrolyte with a solid one that is non-flammable and allows higher energy density",
        "path": ["identify liquid electrolyte risk", "test solid electrolyte", "improve conductivity", "solid-state battery"],
        "operators": ["KT", "TRIZ#35", "TRIZ#1", "PLATT"],
    },
    {
        "title": "Vertical Farming",
        "problem": "How to grow food in dense urban areas with limited land and water",
        "query": "Produce crops inside a city building using far less water than a farm",
        "solution": "Stack crops in vertical layers with LED lighting and hydroponic nutrient delivery",
        "path": ["identify land scarcity", "stack growing layers", "add LED + hydroponics", "vertical farm"],
        "operators": ["KT", "TRIZ#1", "TRIZ#35", "PLATT"],
    },
    {
        "title": "Ocean Plastic Cleanup",
        "problem": "How to remove plastic waste from the open ocean efficiently",
        "query": "Collect floating plastic garbage from the sea without harming marine life",
        "solution": "Deploy a passive U-shaped barrier that funnels floating plastic to a collection point using ocean currents",
        "path": ["identify plastic accumulation", "design passive barrier", "use ocean currents", "collect plastic"],
        "operators": ["KT", "TRIZ#35", "TRIZ#24", "PLATT"],
    },
    {
        "title": "Lab-Grown Meat",
        "problem": "How to produce meat without raising and slaughtering animals",
        "query": "Make real meat in a lab so no animal has to be killed",
        "solution": "Cultivate animal muscle cells in a bioreactor with a nutrient medium until they form edible tissue",
        "path": ["identify animal farming cost", "culture muscle cells", "scale bioreactor", "lab-grown meat"],
        "operators": ["KT", "TRIZ#35", "TRIZ#1", "PLATT"],
    },
    {
        "title": "Carbon Capture Direct Air",
        "problem": "How to remove carbon dioxide directly from the atmosphere to fight climate change",
        "query": "Pull CO2 out of the air itself and store it underground",
        "solution": "Use large fans to draw air through a chemical sorbent that binds CO2, then release and store it",
        "path": ["identify atmospheric CO2", "design sorbent", "scale air contactor", "capture and store"],
        "operators": ["KT", "TRIZ#35", "TRIZ#1", "PLATT"],
    },
    {
        "title": "Quantum Error Correction",
        "problem": "How to make quantum computers reliable despite fragile qubits",
        "query": "Stop errors in a quantum computer so it can run long calculations correctly",
        "solution": "Encode logical qubits across many physical qubits with surface codes that detect and correct errors",
        "path": ["identify qubit fragility", "design surface code", "encode logical qubit", "correct errors"],
        "operators": ["KT", "TRIZ#1", "TRIZ#35", "PLATT"],
    },
    {
        "title": "Fusion Ignition",
        "problem": "How to achieve net-positive energy from nuclear fusion",
        "query": "Get more energy out of a fusion reaction than it takes to start it",
        "solution": "Use inertial confinement with powerful lasers to compress a fuel pellet and achieve ignition",
        "path": ["identify fusion barrier", "compress fuel pellet", "fire lasers", "achieve ignition"],
        "operators": ["KT", "TRIZ#35", "TRIZ#1", "PLATT"],
    },
]


def seed_and_test(cases, backend_name: str, db_path: str, hybrid: bool = False) -> dict:
    svc = VectorizationService(backend=get_backend(backend_name), db_path=db_path)
    for case in cases:
        pid = svc.store_problem(case["problem"], title=case["title"])
        sid = svc.store_solution(case["solution"], method="TRIZ", domain="engineering", title=case["title"])
        pathid = svc.store_path(
            case["path"],
            operator_types=case.get("operators"),
            final_breakthrough=case["solution"],
            title=case["title"],
        )
        svc.index._conn.execute(
            "UPDATE vectors SET metadata_json=? WHERE id=?",
            (json.dumps({"solution_id": sid, "path_id": pathid}), pid),
        )
    svc.index._conn.commit()

    results = []
    for case in cases:
        qv = svc.encode_problem(case["query"], title=case["title"])
        if hybrid:
            analogs = svc.hybrid_analogize(qv, top_k=3)
        else:
            analogs = svc.analogize(qv, top_k=3)
        if not analogs:
            results.append({"title": case["title"], "found": False, "sim": 0.0, "correct": False})
            continue
        top = analogs[0]
        top_sol = top["solution"]["content"] if top["solution"] else ""
        results.append({
            "title": case["title"],
            "found": True,
            "sim": round(top["problem"]["similarity"], 3),
            "correct": top_sol == case["solution"],
        })
    return {"backend": backend_name, "results": results}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", default="ollama", choices=["polarity", "ollama"])
    parser.add_argument("--recent", action="store_true", help="Include recent cases")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid (problem+solution) retrieval")
    args = parser.parse_args()

    cases = CLASSICAL_CASES + (RECENT_CASES if args.recent else [])
    mode = "classical + recent" if args.recent else "classical only"
    retr = "hybrid" if args.hybrid else "pure"
    print(f"Training on {len(cases)} cases ({mode}) with {args.embedder} embedder, {retr} retrieval")

    db = f"/tmp/train_{args.embedder}{'_recent' if args.recent else ''}{'_hybrid' if args.hybrid else ''}.db"
    if os.path.exists(db):
        os.remove(db)
    out = seed_and_test(cases, args.embedder, db, hybrid=args.hybrid)
    results = out["results"]
    n_correct = sum(1 for r in results if r["correct"])
    n_found = sum(1 for r in results if r["found"])
    avg_sim = sum(r["sim"] for r in results) / len(results) if results else 0
    print(f"\n[{args.embedder}/{retr}] found={n_found}/{len(results)} correct={n_correct}/{len(results)} avg_sim={avg_sim:.3f}")
    for r in results:
        mark = "✓" if r["correct"] else ("~" if r["found"] else "✗")
        print(f"  {mark} {r['title']:28s} sim={r['sim']:.3f}")


if __name__ == "__main__":
    main()
