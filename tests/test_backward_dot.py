"""Tests for the backward-DoT self-consistency check (super-solver self-inspire win).

The forward-only DoT refiner had no drift/hallucination check. self_consistency uses
a forward-prediction residual: a solution is consistent iff forward-diffusing the
actual problem premise lands near it.
"""

from __future__ import annotations

import numpy as np

from super_solver.latent.thought_diffusion import ThoughtDiffusionRefiner


def _unit(rng):
    v = rng.randn(384)
    return v / np.linalg.norm(v)


def test_self_consistency_identifies_consistent_solution():
    r = ThoughtDiffusionRefiner(dim=384)
    rng = np.random.RandomState(1)
    premise = _unit(rng)
    # A solution that forward-diffusing the premise exactly produces -> CONSISTENT.
    consistent = r._forward_predict(premise, [premise], [])
    res = r.self_consistency(consistent, premise, [premise], [])
    assert res["verdict"] == "CONSISTENT", res
    assert res["score"] > 0.98


def test_self_consistency_flags_hallucinated_solution():
    r = ThoughtDiffusionRefiner(dim=384)
    rng = np.random.RandomState(2)
    premise = _unit(rng)
    far = _unit(rng)  # a random, unrelated vector
    res = r.self_consistency(far, premise, [premise], [])
    assert res["verdict"] == "HALLUCINATED", res
    assert res["score"] < 0.45


def test_self_consistency_orders_consistent_gt_drifted_gt_hallucinated():
    r = ThoughtDiffusionRefiner(dim=384)
    rng = np.random.RandomState(3)
    premise = _unit(rng)
    consistent = r._forward_predict(premise, [premise], [])
    near = premise + 0.1 * rng.randn(384); near /= np.linalg.norm(near)
    far = _unit(rng)
    s_consistent = r.self_consistency(consistent, premise, [premise], [])["score"]
    s_near = r.self_consistency(near, premise, [premise], [])["score"]
    s_far = r.self_consistency(far, premise, [premise], [])["score"]
    assert s_consistent > s_near > s_far


def test_reverse_denoise_returns_unit_vector():
    r = ThoughtDiffusionRefiner(dim=384)
    rng = np.random.RandomState(4)
    sol = _unit(rng)
    recon = r.reverse_denoise(sol, [sol], [])
    assert abs(np.linalg.norm(recon) - 1.0) < 1e-4
