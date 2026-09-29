"""Run the full evaluation battery and write a versioned metrics artifact.

Produces ``assets/metrics_artifact.json`` recording, for the current engine
build, every headline metric across all committed probe suites plus the
multi-scenario bench. The artifact is meant to be committed and DIFFED over
time, so an improvement (or a regression) is visible without re-running
anything by hand.

Usage:
  python scripts/publish_metrics.py            # write/refresh the artifact
  python scripts/publish_metrics.py --check    # fail if metrics regress floors
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.join(os.path.dirname(__file__), "..")
ASSETS = os.path.join(ROOT, "assets")
ARTIFACT = os.path.join(ASSETS, "metrics_artifact.json")

PROBE_SUITES = {
    "stress_32": "pid_stress_probes.json",
    "holdout_v2_48": "pid_holdout_v2.json",
    "ood_13": "pid_ood_probes.json",
}


def _car(probes, scorer):
    wins = 0
    for r in probes:
        a = scorer(r["parent"], r["relevant"])
        b = scorer(r["parent"], r["irrelevant"])
        wins += int(a > b)
    return round(wins / len(probes), 4), len(probes)


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def collect() -> dict:
    import multi_scenario_bench as bench

    from super_solver.core.distilled_pid import DistilledPIDEmbedder
    from super_solver.core.embedder import get_backend
    from super_solver.frameworks.problem_identification import (
        identify_problem,
        relevance_score,
    )

    art = DistilledPIDEmbedder()
    polarity = get_backend("polarity")

    metrics = {"probe_suites": {}, "scenario_bench": {}}

    for name, fn in PROBE_SUITES.items():
        path = os.path.join(ASSETS, fn)
        if not os.path.exists(path):
            continue
        probes = json.load(open(path))
        # scorer (semantic + negation guard)
        sc_car, n = _car(probes, lambda p, c: relevance_score(p, c)[0])
        # raw distilled cosine
        rd_car, _ = _car(
            probes,
            lambda p, c: art.cosine_similarity(art.encode(p), art.encode(c)),
        )
        # polarity baseline
        pol_car, _ = _car(
            probes,
            lambda p, c: polarity.cosine_similarity(polarity.encode(p), polarity.encode(c)),
        )
        metrics["probe_suites"][name] = {
            "n": n,
            "scorer_car": sc_car,
            "distilled_raw_car": rd_car,
            "polarity_car": pol_car,
        }

    # scenario bench aggregate
    rows = bench.SCENARIOS
    type_ok = 0
    cov = 0.0
    vacuous = 0
    for bucket, parent, gold in rows:
        st = identify_problem(parent)
        out = bench.evaluate_scenario(bucket, parent, gold)
        type_ok += int(st.problem_type == gold)
        cov += out["slot_coverage"]
        vacuous += out["vacuous"]
    metrics["scenario_bench"] = {
        "n": len(rows),
        "type_accuracy": round(type_ok / len(rows), 4),
        "mean_slot_coverage": round(cov / len(rows), 4),
        "vacuous_subproblems": vacuous,
    }

    # artifact hash for provenance
    if art.available:
        from super_solver.core.distilled_pid import _DEFAULT_WEIGHTS

        try:
            h = hashlib.sha256(open(_DEFAULT_WEIGHTS, "rb").read()).hexdigest()[:16]
        except Exception:
            h = "unknown"
        metrics["distilled_artifact_sha256_16"] = h

    return metrics


# Floors below which a regression should fail the --check gate.
FLOORS = {
    ("probe_suites", "stress_32", "scorer_car"): 0.95,
    ("probe_suites", "holdout_v2_48", "scorer_car"): 0.90,
    ("probe_suites", "ood_13", "scorer_car"): 0.85,
    ("scenario_bench", "type_accuracy"): 0.75,
    ("scenario_bench", "mean_slot_coverage"): 0.80,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    metrics = collect()
    payload = {
        "version": "0.1.0",
        "git_rev": _git_rev(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
    }

    if args.check:
        failed = []
        for path, floor in FLOORS.items():
            cur = metrics
            for k in path:
                cur = cur.get(k, {})
            if isinstance(cur, dict):
                cur = cur
            val = cur
            if not isinstance(val, (int, float)) or val < floor:
                failed.append((path, val, floor))
        if failed:
            for p, v, f in failed:
                print(f"REGRESSION: {'.'.join(p)} = {v} < floor {f}")
            sys.exit(1)
        print("metrics OK (all floors met)")
        return

    os.makedirs(ASSETS, exist_ok=True)
    with open(ARTIFACT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {ARTIFACT}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
