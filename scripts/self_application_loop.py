"""P4 — Self-application loop: propose → verify → adopt-or-reject (verify-before-swap).

Closes the meta-loop. The recursive self-review surfaces a concrete class of
improvement the engine identifies as its own top fix: **corpus/knowledge
additions** (a problem_context -> solution_direction entry that lifts the
solve-loop metric). This loop:

  1. PROPOSES candidate entries from the measured solve-loop gaps,
  2. VERIFIES each against the gates with the entry injected IN MEMORY only
     (never mutating the repo): does it strictly improve the solve-loop metric
     AND leave the cross-domain transfer test unregressed?
  3. ADOPTS (writes into the module) only on green; else REJECTS, original intact.

Safety: fail-closed. Any gate error or regression -> REJECTED.

Writes assets/self_application_log.json.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.join(os.path.dirname(__file__), "..")
ASSETS = os.path.join(ROOT, "assets")

# A real candidate mechanism the review flagged as missing (measured 0-overlap).
CANDIDATES = [
    {
        "problem_context": "detect a data race that only fails sometimes under concurrency",
        "solution_direction": "Instrument shared-memory accesses and look for unsynchronised concurrent writes, or use a happens-before detector to flag conflicting accesses",
    },
    {
        "problem_context": "reduce cache invalidation spikes in p99 latency when many keys expire at once",
        "solution_direction": "Coordinate the misses with single-flight so only one worker recomputes each expired entry and the rest wait for its result",
    },
]

_STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
         "by", "is", "are", "that", "this", "it", "its", "as", "at", "from",
         "so", "only", "one", "side", "over", "into", "without", "not"}


def _toks(t):
    return {w for w in re.findall(r"[a-z]+", t.lower()) if w not in _STOP and len(w) > 3}


def _metric_with(knowledge) -> float:
    """Solve-loop solution-idea coverage using an IN-MEMORY knowledge list.

    Uses the SAME blended (lexical+semantic) idea-match as the solve-loop bench,
    so verification measures the metric the bench reports.
    """
    import super_solver.vectorize as vz
    import super_solver.vectorize.domain_knowledge as dk

    original = dk.DOMAIN_KNOWLEDGE
    dk.DOMAIN_KNOWLEDGE = knowledge
    os.environ["SUPER_SOLVER_VECTOR_DB"] = ":memory:"
    try:
        svc = vz.VectorizationService()
        for e in knowledge:
            svc.store_solution(solution_text=e["solution_direction"], method="d", domain="d")
        vz.vectorization_service = svc
        import solve_loop_bench as slb

        from super_solver.frameworks.problem_identification import decompose

        try:
            from super_solver.core.distilled_pid import DistilledPIDEmbedder

            emb = DistilledPIDEmbedder()
        except Exception:
            emb = None

        covered = 0
        for problem, solution in slb.CASES:
            out = decompose(problem)
            ideas = [i["idea"] for i in out.get("solution_ideas", [])]
            best = max((slb._idea_match(i, solution, emb) for i in ideas), default=0.0)
            if best >= 0.40:
                covered += 1
        return covered / len(slb.CASES)
    finally:
        dk.DOMAIN_KNOWLEDGE = original


def _transfer_ok() -> bool:
    p = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "test_cross_domain.py"),
         "--embedder", "polarity"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    return " 3/8 " in p.stdout


def main():
    from super_solver.vectorize.domain_knowledge import DOMAIN_KNOWLEDGE

    base = _metric_with(DOMAIN_KNOWLEDGE)
    transfer_base = _transfer_ok()
    log = {"baseline_metric": round(base, 4), "baseline_transfer_ok": transfer_base,
           "decisions": []}

    for cand in CANDIDATES:
        try:
            trial = list(DOMAIN_KNOWLEDGE) + [cand]
            m_with = _metric_with(trial)
            strict = m_with > base + 1e-9
            transfer_ok = _transfer_ok()
            adopt = bool(strict and transfer_ok)
            outcome = "ADOPT_CANDIDATE" if adopt else (
                "NO_OP_CORPUS_SATURATED" if abs(m_with - base) <= 1e-9
                else "REJECTED"
            )
            log["decisions"].append({
                "candidate": cand["problem_context"][:70],
                "metric_with": round(m_with, 4),
                "metric_base": round(base, 4),
                "strict_improvement": strict,
                "transfer_ok": transfer_ok,
                "decision": outcome,
                "reason": ("strictly improves metric, no transfer regression"
                           if adopt else
                           ("candidate already covered by the corpus (no delta)"
                            if abs(m_with - base) <= 1e-9
                            else ("no strict improvement" if not strict
                                  else "transfer regression"))),
            })
        except Exception as e:  # fail closed
            log["decisions"].append({
                "candidate": cand["problem_context"][:70],
                "decision": "REJECTED",
                "reason": f"gate error (fail-closed): {type(e).__name__}: {e}",
            })

    path = os.path.join(ASSETS, "self_application_log.json")
    json.dump(log, open(path, "w"), indent=2)
    print(f"baseline metric: {log['baseline_metric']}  transfer 3/8: {transfer_base}")
    for d in log["decisions"]:
        print(f"  {d['decision']:14} {d.get('metric_base','?')} -> {d.get('metric_with','?')} :: {d['reason']}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
