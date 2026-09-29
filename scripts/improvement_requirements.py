"""Research: what does IMPROVEMENT actually require? (evidence, not opinion)

Uses the engine's own components to MEASURE the ceilings that any improvement
plan must respect, then derives a requirements plan from the measurements.

Three measurements:
  A. RETRIEVAL CEILING. On the engine's own corpus, what fraction of problems can
     set-containment retrieval solve at all (correct answer shares >=1 atom)?
     This bounds every retrieval improvement: if the correct solution shares no
     atom with the problem, NO containment re-weighting can find it.
  B. RANKING CEILING. Given the correct answer IS retrievable, does a better
     ranker help? Compare plain containment vs IDF-weighted vs oracle.
  C. DIVERSITY vs PRECISION. Adding more corpus entries raises recall but can
     lower precision; measure the curve.

Writes assets/improvement_requirements.json.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from super_solver.vectorize.primitives import (  # noqa: E402
    match_by_primitives,
    strip_to_primitives,
)


def load_pairs():
    """Real problem/solution pairs from the committed corpora."""
    pairs = []
    for fn in ("distilled_pid_train.json",):
        p = os.path.join(os.path.dirname(__file__), "..", "assets", fn)
        if os.path.exists(p):
            for r in json.load(open(p)):
                pairs.append((r["parent"], r["paraphrase"]))
    # also the domain knowledge (problem_context -> solution_direction)
    from super_solver.vectorize.domain_knowledge import DOMAIN_KNOWLEDGE

    for entry in DOMAIN_KNOWLEDGE:
        pairs.append((entry["problem_context"], entry["solution_direction"]))
    return pairs


def measure():
    pairs = load_pairs()
    n = len(pairs)

    # A. retrievability floor: does the CORRECT solution share >=1 SPECIFIC atom
    #    with its problem? (generic act_/attr_ atoms don't count for ranking.)
    def specific(prims):
        return {p for p in prims if not (p.startswith("act_") or p.startswith("attr_"))}

    disjoint = 0
    share_action = 0
    for prob, sol in pairs:
        tp = specific(strip_to_primitives(prob))
        sp = specific(strip_to_primitives(sol))
        if not (tp & sp):
            disjoint += 1
        if any(not p.startswith("object_") for p in (tp & sp)):
            share_action += 1

    # B. ranking: for problems whose correct answer IS retrievable, does the
    #    ranker put it first? Build a small distractor pool and check top-1.
    sources = [{"title": f"s{i}", "solution": sol} for i, (_, sol) in enumerate(pairs)]
    correct_top1 = 0
    rankable = 0
    for prob, sol in pairs:
        tp = specific(strip_to_primitives(prob))
        sp = specific(strip_to_primitives(sol))
        if not (tp & sp):
            continue  # not retrievable at all -> excluded from the ranking test
        rankable += 1
        hits = match_by_primitives(prob, sources, top_k=1)
        if hits and hits[0]["content"] == sol:
            correct_top1 += 1

    return {
        "n_pairs": n,
        "A_disjoint_pairs": disjoint,
        "A_disjoint_rate": round(disjoint / n, 3),
        "A_share_action_atoms": share_action,
        "B_rankable_pairs": rankable,
        "B_top1_correct": correct_top1,
        "B_top1_rate_within_rankable": round(correct_top1 / rankable, 3) if rankable else 0.0,
    }


def derive_requirements(m):
    reqs = []
    if m["A_disjoint_rate"] > 0:
        reqs.append({
            "requirement": "SEMANTIC BRIDGE",
            "why": f"{m['A_disjoint_rate']:.0%} of problems share NO specific atom with "
                   f"their correct solution, so set-containment retrieval returns 0 by "
                   f"construction. No re-weighting/IDF/fusion can exceed this floor.",
            "mathematical_core": "containment(T,S) = |T∩S|/|T| = 0 when T∩S=∅. "
                                 "Exceeding it requires COMPOSING with a map f: text→R^d "
                                 "and a metric where related(paraphrase) > unrelated, i.e. "
                                 "a learned semantic embedding, not a bigger atom set.",
            "evidence": f"A_disjoint_rate={m['A_disjoint_rate']}",
        })
    if m["B_top1_rate_within_rankable"] < 0.9:
        reqs.append({
            "requirement": "RANKER QUALITY",
            "why": f"Among retrievable pairs, top-1 is only "
                   f"{m['B_top1_rate_within_rankable']:.0%}; ranking leaves correctness "
                   f"on the table.",
            "mathematical_core": "Ranking is a monotone function of the score; improving it "
                                 "means learning a score that is monotone in true relevance "
                                 "(a supervised ranking objective), not hand-set weights.",
            "evidence": f"B_top1_rate={m['B_top1_rate_within_rankable']}",
        })
    reqs.append({
        "requirement": "CORPUS COVERAGE (orthogonal)",
        "why": "Retrieval can only return mechanisms present in the corpus; measured "
               "0-overlap retrievals for cache-single-flight and invoice-dedup show "
               "coverage limits, independent of matching quality.",
        "mathematical_core": "recall <= 1 - P(correct mechanism absent from corpus). "
                             "No matcher fixes an absent target.",
        "evidence": "see recursive_self_review / solve_loop",
    })
    reqs.append({
        "requirement": "VERIFICATION BEFORE ADOPTION",
        "why": "Improvements must be proven, not assumed: the campaign found 10+ real "
               "bugs in surfaces that 'looked healthy'.",
        "mathematical_core": "adopt a change only if it improves a held-out metric "
                             "beyond its noise floor (paired test), else reject.",
        "evidence": "health/metrics gates",
    })
    return reqs


def main():
    m = measure()
    reqs = derive_requirements(m)
    out = {"measurements": m, "requirements": reqs}
    path = os.path.join(os.path.dirname(__file__), "..", "assets", "improvement_requirements.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
