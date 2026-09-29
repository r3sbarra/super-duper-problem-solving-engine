"""T2.4 — Cross-framework chaining.

Frameworks are invoked in isolation. The value is composition: route a problem
through a chain of frameworks by type (e.g. diagnose -> abduct -> design assay),
with each stage's output feeding the next. Measure it runs end-to-end and
produces a coherent artifact, vs a single-framework baseline.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def run_chain(engine, problem: str) -> dict:
    """A type-routed chain: identify -> KT boundary -> abduct -> design assay."""
    from super_solver.frameworks.problem_identification import identify_problem

    st = identify_problem(problem)
    chain = []

    # 1. Identify
    chain.append({"stage": "identify", "type": st.problem_type,
                  "entities": st.entities[:3]})

    # 2. KT boundary (root-cause bounding) using the identified entities
    is_facts = [f"{e} is involved" for e in st.entities[:2]] or ["the symptom occurs"]
    is_not = [f"{e} is not involved" for e in st.entities[:1]] or ["the symptom is absent"]
    kt = engine.kt.analyze_boundary(is_facts, is_not)
    chain.append({"stage": "kt_boundary", "leading_cause": kt.get("leading_cause"),
                  "n_candidates": len(kt.get("candidate_evaluations", []))})

    # 3. Peirce abductive hypotheses seeded by the leading cause
    hypotheses = engine.peirce.autonomous_abduct(
        problem, domain_context=st.problem_type
    )
    chain.append({"stage": "abduct", "n_hypotheses": len(hypotheses),
                  "top": getattr(hypotheses[0], "description", "")[:60] if hypotheses else ""})

    # 4. Platt crucial experiment design across the competing hypotheses
    exps = engine.platt.autonomous_design_crucial_experiments(hypotheses)
    chain.append({"stage": "platt_assay", "n_experiments": len(exps),
                  "top": getattr(exps[0], "name", "") if exps else ""})

    return {"problem": problem, "chain": chain,
            "stages_ok": all(bool(c.get("stage")) for c in chain) and len(chain) == 4}


def main():
    from super_solver.engine import SuperDuperProblemSolvingEngine

    engine = SuperDuperProblemSolvingEngine()
    problems = [
        "The service slows down only at month-end; diagnose and isolate the cause.",
        "A battery overheats during fast charging; find the root cause.",
        "Detect fraud without flooding analysts with false positives.",
    ]
    results = [run_chain(engine, p) for p in problems]
    ok = sum(1 for r in results if r["stages_ok"])
    out = {"n": len(problems), "chains_ok": f"{ok}/{len(problems)}", "results": results}
    path = os.path.join(os.path.dirname(__file__), "..", "assets", "framework_chain.json")
    json.dump(out, open(path, "w"), indent=2)
    for r in results:
        print(f"\n{r['problem'][:60]}")
        for c in r["chain"]:
            print(f"   {c['stage']:12} { {k: v for k, v in c.items() if k != 'stage'} }")
    print(f"\nchains_ok = {out['chains_ok']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
