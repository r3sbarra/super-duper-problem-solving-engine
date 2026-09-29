"""Tests for autonomous KT boundary candidate-cause generation.

Regression guard: analyze_boundary must never return an empty analysis when only
IS / IS-NOT facts are supplied — it generates and ranks candidate root causes
from the contrast between the two regimes.
"""
from __future__ import annotations

from super_solver.frameworks.kepner_tregoe import KepnerTregoeEngine


def test_analyze_boundary_generates_candidates_autonomously():
    kt = KepnerTregoeEngine()
    r = kt.analyze_boundary(
        is_facts=["light switch sparks in the bedroom", "happens on Tuesdays"],
        is_not_facts=["does not spark in the kitchen", "never on Saturdays"],
    )
    assert r["causes_generated_autonomously"] is True
    assert len(r["candidate_evaluations"]) > 0
    assert r["leading_cause"]
    # Candidates are ranked: consistent ones first, by margin descending.
    margins = [c["margin"] for c in r["candidate_evaluations"]]
    assert margins == sorted(margins, reverse=True)


def test_analyze_boundary_respects_supplied_causes():
    kt = KepnerTregoeEngine()
    r = kt.analyze_boundary(
        is_facts=["sparks in bedroom"],
        is_not_facts=["does not spark in kitchen"],
        candidate_causes=["a loose wire in the bedroom circuit"],
    )
    assert r["causes_generated_autonomously"] is False
    assert len(r["candidate_evaluations"]) == 1
    assert r["candidate_evaluations"][0]["cause"] == "a loose wire in the bedroom circuit"


def test_generated_causes_exclude_non_factor_verbs():
    kt = KepnerTregoeEngine()
    causes = kt.generate_candidate_causes(
        is_facts=["the switch sparks on Tuesdays"],
        is_not_facts=["the switch is fine on Saturdays"],
    )
    joined = " ".join(causes).lower()
    # 'sparks'/'happens' style process verbs must not become factor names.
    assert "'happens'" not in joined
    assert "'sparks'" not in joined
    assert "'tuesdays'" in joined
