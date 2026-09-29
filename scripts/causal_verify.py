"""T2.2 — Causal-VSA verification against synthetic ground truth.

Builds synthetic Structural Causal Models with known linear mechanisms, computes
the expected do-effect by hand (the analytic value), and checks that
VectorizedCausalModel.intervene recovers the correct SIGN and ordering.

Ground truth: for a linear SCM with y = sum_i w_i * x_i, do(X = x0) changes a
descendant by the product of edge weights along the path times (x0 - baseline).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def build_chain():
    """x -> y -> z with known weights; do(x=1) effect on z = wx*wy."""
    from super_solver.core.causal_vsa import VectorizedCausalModel

    m = VectorizedCausalModel()
    for n in ("x", "y", "z"):
        m.add_variable(n, baseline_value=0.5)
    m.add_causal_edge("x", "y", 0.8)
    m.add_causal_edge("y", "z", 0.5)
    return m


def build_confounded():
    """c -> x, c -> y (confounder): do(x) must affect y via x->y only."""
    from super_solver.core.causal_vsa import VectorizedCausalModel

    m = VectorizedCausalModel()
    for n in ("c", "x", "y"):
        m.add_variable(n, baseline_value=0.5)
    m.add_causal_edge("c", "x", 0.7)
    m.add_causal_edge("c", "y", 0.6)
    m.add_causal_edge("x", "y", 0.4)
    return m


def build_fork():
    """x <- c -> y: do(x) must have ZERO effect on y (no directed path)."""
    from super_solver.core.causal_vsa import VectorizedCausalModel

    m = VectorizedCausalModel()
    for n in ("c", "x", "y"):
        m.add_variable(n, baseline_value=0.5)
    m.add_causal_edge("c", "x", 0.7)
    m.add_causal_edge("c", "y", 0.6)
    return m


def main():
    results = {}

    # 1. chain: do(x=1) -> z positive effect, magnitude ~ wx*wy*(1-0.5)=0.2
    m = build_chain()
    r = m.intervene("x", 1.0, consequence_target="z")
    results["chain"] = {
        "ace": r["average_causal_effect"],
        "expected_sign": "positive",
        "got_sign": "positive" if r["average_causal_effect"] > 0 else "negative",
        "sign_ok": r["average_causal_effect"] > 0,
        "expected_ace": 0.8 * 0.5 * 0.5,
        "got_ace": round(r["average_causal_effect"], 4),
    }

    # 2. confounded: do(x=1) -> y positive (via x->y), not via the confounder
    m = build_confounded()
    r = m.intervene("x", 1.0, consequence_target="y")
    results["confounded"] = {
        "ace": r["average_causal_effect"],
        "expected_sign": "positive",
        "sign_ok": r["average_causal_effect"] > 0,
    }

    # 3. fork: do(x=1) -> y must be ~ZERO (x does not cause y)
    m = build_fork()
    r = m.intervene("x", 1.0, consequence_target="y")
    results["fork"] = {
        "ace": r["average_causal_effect"],
        "expected_sign": "zero",
        "sign_ok": abs(r["average_causal_effect"]) < 1e-6,
    }

    # ordering check: chain effect on y (proximal) > effect on z (distal)
    m = build_chain()
    ry = m.intervene("x", 1.0, consequence_target="y")["average_causal_effect"]
    rz = m.intervene("x", 1.0, consequence_target="z")["average_causal_effect"]
    results["ordering"] = {
        "y_effect": ry, "z_effect": rz,
        "proximal_gt_distal": ry > rz > 0,
        "expected_sign": "y > z > 0",
        "sign_ok": ry > rz > 0,
    }

    ok = sum(1 for v in results.values() if v["sign_ok"])
    total = len(results)
    out = {"checks": results, "passed": f"{ok}/{total}"}
    path = os.path.join(os.path.dirname(__file__), "..", "assets", "causal_verification.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
