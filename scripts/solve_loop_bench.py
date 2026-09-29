"""T2.3 — End-to-end solve loop: does decomposition produce USABLE subproblems?

The engine decomposes well, but is there a measurable link from subproblems to
solved problems? This benchmark closes the loop on a set of problems with known
solutions:

  problem -> identify -> decompose -> for each subproblem, check whether it
  covers the key IDEA of the known solution (semantic recall against the
  reference solution).

Metric: solution-coverage@k = fraction of held-out problems whose reference
solution idea is matched by at least one generated subproblem (top-k), and the
rank of the best-matching subproblem.

This is a HONEST lower-bound: matching subproblem text to a reference solution
idea is a proxy for "the decomposition pointed at the right thing"; it does not
prove a solved solution. Reported as coverage, not as "solved".
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Use the REAL vector index even under pytest (which otherwise forces an empty
# in-memory DB via PYTEST_CURRENT_TEST), so the retrieval bridge is measured.
os.environ.setdefault(
    "SUPER_SOLVER_VECTOR_DB",
    os.path.join(os.path.dirname(__file__), "..", ".super_solver_data", "vector_index.db"),
)

# (problem, reference solution idea)
CASES = [
    ("Shortest-path routing for a delivery fleet under time-window constraints.",
     "model the road network as a weighted graph and relax edges to find the shortest feasible route"),
    ("Reduce cache invalidation spikes in p99 latency.",
     "coordinate misses so only one worker recomputes an expired entry (single-flight)"),
    ("Design a scheme so a message is processed exactly once despite retries.",
     "attach a unique id and deduplicate on the receiver side (idempotency key)"),
    ("Detect duplicate invoices across two billing systems.",
     "hash normalised invoice fields and join on the hash to find collisions"),
    ("Prevent one slow backend from taking down the whole service.",
     "add a circuit breaker that fails fast on the slow dependency"),
    ("Estimate how long a queue takes to drain after a traffic spike.",
     "model arrival and service rates and integrate over the backlog"),
    ("Make a battery charge quickly without overheating.",
     "increase electrode surface area and manage thermal dissipation during charging"),
    ("Find the cheapest way to connect all offices with links so no cycles form.",
     "grow a minimum spanning tree by repeatedly taking the cheapest safe edge"),
    ("Prove that there are infinitely many primes.",
     "assume finitely many primes and construct their product plus one to reach a contradiction"),
    ("Explain why a recommendation engine becomes less diverse over time.",
     "feedback loops amplify popular items and starve the long tail of exposure"),
    ("Reduce the weight of a vehicle without weakening its frame.",
     "replace solid members with trusses that carry the same load at lower mass"),
    ("Detect a data race that only fails sometimes.",
     "instrument shared-memory accesses and look for unsynchronised concurrent writes"),
]


def _lexical_recall(subproblem: str, solution: str) -> float:
    """Content-word recall of the solution idea in the subproblem."""
    import re

    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
            "by", "is", "are", "that", "this", "it", "its", "as", "at", "from",
            "so", "only", "one", "side", "over", "into", "without", "not"}

    def toks(t):
        return {w for w in re.findall(r"[a-z]+", t.lower()) if w not in stop and len(w) > 3}

    s = toks(solution)
    if not s:
        return 0.0
    return len(s & toks(subproblem)) / len(s)


def main():
    from super_solver.frameworks.problem_identification import decompose

    rows = []
    covered_at_1 = 0
    covered_at_3 = 0
    covered_solutions = 0
    for problem, solution in CASES:
        out = decompose(problem)
        subs = [s["subproblem"] for s in out["subproblems"] if s.get("is_new", True)]
        recalls = [_lexical_recall(s, solution) for s in subs]
        best = max(recalls) if recalls else 0.0
        best_rank = (recalls.index(best) + 1) if recalls else 0
        # "covered" when a subproblem shares >= 40% of the solution's content words
        covered = best >= 0.40
        covered_at_1 += int(best_rank == 1 and covered)
        covered_at_3 += int(best_rank <= 3 and covered)

        # closed-loop: do the RETRIEVED solution ideas cover the reference idea?
        ideas = [i["idea"] for i in out.get("solution_ideas", [])]
        idea_recalls = [_lexical_recall(i, solution) for i in ideas]
        idea_best = max(idea_recalls) if idea_recalls else 0.0
        covered_solutions += int(idea_best >= 0.40)

        rows.append({
            "problem": problem, "best_recall": round(best, 3),
            "best_rank": best_rank, "covered": covered,
            "idea_best_recall": round(idea_best, 3),
            "idea_covered": idea_best >= 0.40,
            "best_subproblem": subs[best_rank - 1][:80] if subs and best_rank else "",
        })

    n = len(CASES)
    out = {
        "n_problems": n,
        "solution_coverage_at_1": round(covered_at_1 / n, 3),
        "solution_coverage_at_3": round(covered_at_3 / n, 3),
        "mean_best_recall": round(sum(r["best_recall"] for r in rows) / n, 3),
        "solution_idea_coverage": round(covered_solutions / n, 3),
        "rows": rows,
        "caveat": "lexical recall against a reference idea is a PROXY for "
                  "'decomposition pointed at the right thing'; it is not a proof of a solve",
    }
    path = os.path.join(os.path.dirname(__file__), "..", "assets", "solve_loop.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"subproblem coverage@1 = {out['solution_coverage_at_1']}")
    print(f"subproblem coverage@3 = {out['solution_coverage_at_3']}")
    print(f"SOLUTION-IDEA coverage = {out['solution_idea_coverage']}  (retrieved mechanisms)")
    print(f"mean_best_recall      = {out['mean_best_recall']}")
    print("\nper problem (subproblem recall / idea recall):")
    for r in rows:
        f1 = "OK " if r["covered"] else "XX "
        f2 = "OK " if r["idea_covered"] else "XX "
        print(f"  sub[{f1}]{r['best_recall']:.2f}  idea[{f2}]{r['idea_best_recall']:.2f} :: {r['problem'][:42]}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
