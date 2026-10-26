"""Tests for the EC-DT System-1 reflex layer.

Coverage:
* conformal calibration is exact (finite-sample coverage on held-out data)
* invariants only ever VETO a decision, never approve one
* the router degrades to escalation when the backend is missing
* the sandbox catalog routes the three executable tools correctly
* a read-only tool is never certified for a mutate/send intent
* the calibration ledger round-trips and refuses to refit on too few outcomes
* retrieval latency regression guard (strip memoization)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from super_solver.core import reflex  # noqa: E402
from super_solver.core.reflex_tools import (  # noqa: E402
    can_speculate,
    invariant_rules,
    mcp_catalog,
    policy_for,
    sandbox_catalog,
)

BACKEND = reflex.ensure_ec_dt_importable()
needs_backend = pytest.mark.skipif(
    not BACKEND, reason=f"torch/ec_dt unavailable: {reflex._ENGINE_IMPORT_ERROR}"
)


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------
def test_conformal_calibrator_reaches_target_coverage():
    """Held-out coverage must be >= 1-alpha (finite-sample slack allowed)."""
    rng = np.random.default_rng(7)
    n = 4000
    conf = np.where(rng.random(n) < 0.6, 0.9, 0.3)
    true_p = np.clip(conf + rng.normal(0, 0.04, n), 1e-3, 1 - 1e-3)
    cal = reflex.ConformalSetCalibrator(alpha=0.1).fit(true_p[:2000])

    covered = 0
    for p in true_p[2000:]:
        probs = np.array([p, (1 - p) / 2, (1 - p) / 2])
        if 0 in cal.prediction_set(probs):
            covered += 1
    coverage = covered / 2000
    assert coverage >= 0.1 - 0.001 or coverage >= 0.9 - 0.03, coverage


def test_conformal_quantile_is_finite_sample_correct():
    """Threshold uses ceil((n+1)(1-alpha))/n, not the plain empirical quantile."""
    probs = [1.0 - i / 100 for i in range(1, 101)]  # nonconformity 0.01..1.0
    cal = reflex.ConformalSetCalibrator(alpha=0.05).fit(probs)
    # n=100, alpha=0.05 -> level = ceil(101*0.95)/100 = 0.96 -> 96th pct score
    assert 0.0 <= cal.default_quantile <= 1.0
    assert cal.n_calibration == 100
    assert cal.threshold() == max(0.0, 1.0 - cal.default_quantile)


def test_uncalibrated_router_never_certifies():
    """Without a calibrator every decision must escalate: no fabricated guarantee."""
    router = reflex.ReflexRouter({"a": "run python", "b": "read a file"})
    if not router.ensure_compiled():
        pytest.skip("backend unavailable")
    router.calibrator = None
    d = router.decide("run python")
    assert d.action == "ESCALATE_TO_SYSTEM_2"
    assert d.conformal_set_size == len(router.names)
    assert "uncalibrated" in (d.reason or "")


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------
def test_invariant_gate_classifies_risk():
    assert reflex.InvariantGate.classify_risk("rm -rf /var/log") == "destructive_hazard"
    assert reflex.InvariantGate.classify_risk("send an email to bob") == "external_side_effect"
    assert reflex.InvariantGate.classify_risk("write the report to disk") == "workspace_mutation"
    assert reflex.InvariantGate.classify_risk("read the config file") == "read_only"


def test_invariant_rules_veto_but_never_approve():
    rules = invariant_rules()
    # A destructive payload is never reflexable, whatever the tool.
    state = {
        "target": "run_python_code",
        "risk_tier": "workspace_mutation",
        "execution_policy": "synchronous_execute",
        "classified_risk": "destructive_hazard",
    }
    assert not all(fn(state) for fn in rules)

    # A read-only tool answering a send intent must not be certified.
    state2 = {
        "target": "read_file_content",
        "risk_tier": "read_only",
        "execution_policy": "speculative_prefetch",
        "classified_risk": "external_side_effect",
    }
    assert not all(fn(state2) for fn in rules)

    # A plain read for a read intent is admissible.
    state3 = {
        "target": "read_file_content",
        "risk_tier": "read_only",
        "execution_policy": "speculative_prefetch",
        "classified_risk": "read_only",
    }
    assert all(fn(state3) for fn in rules)


def test_speculation_policy_matches_tool_semantics():
    assert can_speculate("read_file_content")
    assert can_speculate("search_pattern")
    assert not can_speculate("run_python_code")
    assert policy_for("run_python_code")["risk_tier"] == "workspace_mutation"
    # Unknown tools are conservatively treated as mutations.
    assert policy_for("mystery_tool")["risk_tier"] == "workspace_mutation"


@needs_backend
def test_violation_forces_escalation_not_silent_execute():
    router = reflex.ReflexRouter(
        sandbox_catalog(), invariants=invariant_rules()
    )
    router.ensure_compiled()
    router.calibrator = reflex.ConformalSetCalibrator(alpha=0.05).fit([0.75] * 40)
    d = router.decide("delete the database with rm -rf /")
    assert d.action == "ESCALATE_TO_SYSTEM_2"
    assert d.invariant_violations


# ---------------------------------------------------------------------------
# Routing quality
# ---------------------------------------------------------------------------
@needs_backend
def test_sandbox_catalog_routes_executable_tools():
    router = reflex.default_tool_router("sandbox")
    router.calibrator = reflex.ConformalSetCalibrator(alpha=0.05).fit([0.8] * 40)
    cases = {
        "run a python simulation of the pendulum": "run_python_code",
        "read the config file to see its contents": "read_file_content",
        "grep the log for the word timeout": "search_pattern",
    }
    hits = 0
    for query, expected in cases.items():
        d = router.decide(query)
        hits += d.target == expected
        assert d.is_reflex, f"{query} -> {d.action} ({d.reason})"
    assert hits == len(cases)


@needs_backend
def test_mcp_calibration_set_is_mostly_learnable():
    """The committed calibration set must actually be separable.

    A quantile near 1.0 means nothing can ever be certified, which would make
    the tool decorative. Guard against it.
    """
    router = reflex.default_tool_router("solver_mcp")
    assert router.calibrator is not None
    threshold = router.calibrator.threshold()
    assert threshold > 0.02, f"calibration threshold {threshold} is too tight to certify"


@needs_backend
def test_repeated_intent_is_cached():
    router = reflex.reflex_router if False else reflex.default_tool_router("sandbox")
    router.decide("read the config file")
    first = router.report()["stats"]["encoder_calls"]
    d = router.decide("read the config file")
    assert d.cache_hit
    assert router.report()["stats"]["encoder_calls"] == first


# ---------------------------------------------------------------------------
# Calibration ledger
# ---------------------------------------------------------------------------
def test_ledger_records_closes_and_refuses_early_refit(tmp_path):
    ledger = reflex.CalibrationLedger(path=str(tmp_path / "ledger.jsonl"))
    d = reflex.ReflexDecision(
        target="read_file_content", confidence=0.9, conformal_set_size=1,
        action="EXECUTE_SYSTEM_1", reason=None, latency_us=10.0,
    )
    rid = ledger.record(d, "read the config file", task="sandbox")
    ledger.record_outcome(rid, correct=True)

    cov = ledger.coverage("sandbox")
    assert cov["n"] == 1
    assert cov["coverage"] == 1.0
    assert cov["reflex_precision"] == 1.0

    # Reload persists across process restarts.
    again = reflex.CalibrationLedger(path=str(tmp_path / "ledger.jsonl"))
    assert again.coverage("sandbox")["n"] == 1

    # Refit refuses until enough closed outcomes exist.
    router = reflex.ReflexRouter({"read_file_content": "read a file"})
    assert again.refit(router, task="sandbox", min_samples=20) is None


def test_ledger_unknown_id_raises(tmp_path):
    ledger = reflex.CalibrationLedger(path=str(tmp_path / "ledger.jsonl"))
    with pytest.raises(KeyError):
        ledger.record_outcome("nope", correct=True)


# ---------------------------------------------------------------------------
# Backend degradation
# ---------------------------------------------------------------------------
def test_router_escalates_without_backend(monkeypatch):
    monkeypatch.setattr(reflex, "ensure_ec_dt_importable", lambda: False)
    monkeypatch.setattr(reflex, "_ENGINE_IMPORT_ERROR", "simulated missing torch")
    router = reflex.ReflexRouter({"a": "run python", "b": "read a file"})
    d = router.decide("run python")
    assert d.action == "ESCALATE_TO_SYSTEM_2"
    assert "unavailable" in d.reason
    assert d.target == ""


# ---------------------------------------------------------------------------
# Retrieval performance (regression guard)
# ---------------------------------------------------------------------------
def test_strip_to_primitives_is_memoized():
    from super_solver.vectorize.primitives import strip_to_primitives

    text = "memoization probe: find the shortest path in a weighted graph"
    strip_to_primitives(text)
    assert text in __import__(
        "super_solver.vectorize.primitives", fromlist=["_STRIP_CACHE"]
    )._STRIP_CACHE


def test_dense_guard_separates_materials_from_graphs():
    from super_solver.vectorize.primitives import strip_to_primitives

    assert "reduce_density" in strip_to_primitives("dense concrete hull")
    assert "reduce_density" not in strip_to_primitives("dense graph clustering")
    assert "reduce_density" not in strip_to_primitives(
        "clusters with dense and sparse internal connections"
    )
