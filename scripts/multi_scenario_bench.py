"""Multi-scenario benchmark for the super-solver problem-identification /
decomposition pipeline.

Runs the FULL pipeline (identify_problem + decompose + relevance_score) against
scenarios grouped into buckets, and reports per-bucket quality metrics:
  - type_id_accuracy   (identify_problem vs a labeled gold type)
  - slot_coverage      (fraction of identified slots addressed by subproblems)
  - anchor_specificity (fraction of subproblems naming a real entity/goal, not a
                        vacuous template)
  - decomposition_ok   (subproblems are non-empty, distinct, and about the
                        problem, not about generic framing words)

Buckets: standard, long, short, multi-goal, ill-posed, negation, cross-domain,
mathematical. This is the harness used to locate residual weaknesses.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from super_solver.frameworks.problem_identification import (
    decompose,
    identify_problem,
    relevance_score,
)

# (bucket, parent, gold_type)
SCENARIOS = [
    # --- standard ---
    ("standard", "Shortest-path routing for a delivery fleet under time-window constraints.", "optimization"),
    ("standard", "Lithium metal battery: rapid charging causes dendrite growth and thermal runaway.", "diagnosis"),
    ("standard", "Design a scheme so a message is processed exactly once despite retries.", "design"),
    ("standard", "Estimate how long a queue takes to drain after a traffic spike.", "estimation"),
    # --- long (multi-sentence) ---
    ("long",
     "A hospital must keep operating through regional power failures and network outages lasting several days. "
     "It must preserve patient records, coordinate staff across wards, and never lose a critical update. "
     "The current system writes to a single database in one region and has no offline mode.",
     "design"),
    ("long",
     "A city wants to reduce traffic congestion during peak hours without building new roads. "
     "Commuters are price-sensitive, public transit is underused, and the road network has several bottlenecks. "
     "Any policy must be fair across income groups.",
     "optimization"),
    # --- short ---
    ("short", "Cache stampede?", "diagnosis"),
    ("short", "Lose weight?", "optimization"),
    ("short", "Prove a limit exists.", "proof"),
    # --- multi-goal ---
    ("multi-goal", "Minimize latency AND cost AND carbon footprint for a global API.", "optimization"),
    ("multi-goal", "Detect fraud, reduce false positives, and keep the model explainable to auditors.", "design"),
    ("multi-goal", "Build a battery that is lighter, charges faster, and lasts longer.", "optimization"),
    ("multi-goal", "Design a drone that flies longer, carries more, and stays quiet.", "design"),
    # --- ill-posed ---
    ("ill-posed", "Make it better.", "explanation"),
    ("ill-posed", "Why is everything broken?", "diagnosis"),
    ("ill-posed", "something about data and stuff maybe improve it", "explanation"),
    # --- negation ---
    ("negation", "Reduce read latency without adding any cache layer.", "optimization"),
    ("negation", "Make the service faster without increasing cost or degrading accuracy.", "optimization"),
    # --- cross-domain ---
    ("cross-domain", "Make a climbing pad that grips smooth vertical walls without glue.", "design"),
    ("cross-domain", "Keep a hospital's oxygen supply from running out during a strike.", "design"),
    ("cross-domain", "Explain how antibiotic resistance spreads in a hospital.", "explanation"),
    # --- mathematical ---
    ("mathematical", "Prove there are infinitely many primes.", "proof"),
    ("mathematical", "Show that a tree with n nodes has exactly n-1 edges.", "proof"),
    ("mathematical", "Find the maximum value of x^2 y subject to x + y = 10.", "optimization"),
]


def _anchor_specificity(struct, subproblems):
    """Fraction of generated subproblems that reference a real entity or goal."""
    ents = [e.lower() for e in struct.entities]
    goal = (struct.goal or "").lower()
    goal_toks = set(t for t in goal.split() if len(t) > 3)
    ok = 0
    total = 0
    for s in subproblems:
        if not s.get("is_new", True):
            continue
        total += 1
        low = s["subproblem"].lower()
        if any(e and e in low for e in ents) or any(t in low for t in goal_toks):
            ok += 1
    return (ok / total) if total else 0.0


def evaluate_scenario(bucket, parent, gold_type):
    struct = identify_problem(parent)
    out = decompose(parent)
    subs = out["subproblems"]
    gen = [s for s in subs if s.get("is_new", True)]
    metrics = out["metrics"]

    type_ok = int(struct.problem_type == gold_type)
    anchor = _anchor_specificity(struct, subs)
    # distinctness
    texts = [s["subproblem"] for s in gen]
    distinct = len(set(texts)) / len(texts) if texts else 0.0
    # not about generic framing words only
    vacuous = sum(
        1 for s in gen
        if any(w in s["subproblem"].lower() for w in (" of rapid ", " of causes ", " of delivery "))
    )
    return {
        "bucket": bucket,
        "parent": parent,
        "pred_type": struct.problem_type,
        "gold_type": gold_type,
        "type_ok": type_ok,
        "slot_coverage": metrics["slot_coverage"],
        "anchor_specificity": round(anchor, 3),
        "distinctness": round(distinct, 3),
        "vacuous": vacuous,
        "n_gen": metrics["n_generated"],
    }


def main():
    rows = [evaluate_scenario(*s) for s in SCENARIOS]
    buckets = {}
    for r in rows:
        buckets.setdefault(r["bucket"], []).append(r)

    print(f"{len(rows)} scenarios across {len(buckets)} buckets\n")
    print(f"{'bucket':12} {'n':>2} {'type_acc':>8} {'coverage':>8} {'anchor':>7} {'distinct':>8} {'vacuous':>7}")
    for b, rs in buckets.items():
        n = len(rs)
        print(
            f"{b:12} {n:>2} "
            f"{sum(r['type_ok'] for r in rs)/n:>8.2f} "
            f"{sum(r['slot_coverage'] for r in rs)/n:>8.2f} "
            f"{sum(r['anchor_specificity'] for r in rs)/n:>7.2f} "
            f"{sum(r['distinctness'] for r in rs)/n:>8.2f} "
            f"{sum(r['vacuous'] for r in rs):>7}"
        )
    n = len(rows)
    print(
        f"\nOVERALL      {n:>2} "
        f"{sum(r['type_ok'] for r in rows)/n:>8.2f} "
        f"{sum(r['slot_coverage'] for r in rows)/n:>8.2f} "
        f"{sum(r['anchor_specificity'] for r in rows)/n:>7.2f} "
        f"{sum(r['distinctness'] for r in rows)/n:>8.2f} "
        f"{sum(r['vacuous'] for r in rows):>7}"
    )

    print("\nType-ID misses:")
    for r in rows:
        if not r["type_ok"]:
            print(f"  [{r['bucket']:11}] gold={r['gold_type']:12} pred={r['pred_type']:12} :: {r['parent'][:46]}")
    print("\nLow-anchor scenarios (<0.5):")
    for r in rows:
        if r["anchor_specificity"] < 0.5:
            print(f"  [{r['bucket']:11}] anchor={r['anchor_specificity']:.2f} cov={r['slot_coverage']:.2f} :: {r['parent'][:46]}")


if __name__ == "__main__":
    main()
