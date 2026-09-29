"""Multi-scenario regression guard for the identification + decomposition pipeline.

Runs the 22-scenario / 8-bucket benchmark committed at assets/pid_scenario_bench.json
and asserts the pipeline stays above quality floors, so the type-classifier and
slot-balancing improvements cannot silently regress.
"""
from __future__ import annotations

import json
import os

import pytest

from super_solver.frameworks.problem_identification import decompose, identify_problem

_BENCH = os.path.join(os.path.dirname(__file__), "..", "assets", "pid_scenario_bench.json")


def _rows():
    if not os.path.exists(_BENCH):
        pytest.skip("scenario bench absent")
    return json.load(open(_BENCH))


def _eval(row):
    struct = identify_problem(row["parent"])
    out = decompose(row["parent"])
    gen = [s for s in out["subproblems"] if s.get("is_new", True)]
    ents = [e.lower() for e in struct.entities]
    anchor = (
        sum(
            1 for s in gen
            if any(e and e in s["subproblem"].lower() for e in ents)
        ) / len(gen)
        if gen else 0.0
    )
    return {
        "type_ok": int(struct.problem_type == row["gold_type"]),
        "coverage": out["metrics"]["slot_coverage"],
        "anchor": anchor,
        "vacuous": sum(
            1 for s in gen
            if any(w in s["subproblem"].lower() for w in (" of rapid ", " of causes "))
        ),
    }


def test_scenario_type_accuracy_floor():
    rows = _rows()
    ok = sum(_eval(r)["type_ok"] for r in rows)
    assert ok / len(rows) >= 0.75, f"type accuracy {ok}/{len(rows)} below floor 0.75"


def test_scenario_slot_coverage_floor():
    rows = _rows()
    cov = sum(_eval(r)["coverage"] for r in rows) / len(rows)
    assert cov >= 0.80, f"mean slot coverage {cov:.3f} below floor 0.80"


def test_no_vacuous_subproblems_anywhere():
    rows = _rows()
    total_vacuous = sum(_eval(r)["vacuous"] for r in rows)
    assert total_vacuous == 0, f"{total_vacuous} vacuous subproblems produced"


def test_entities_are_specific_not_clause_merged():
    struct = identify_problem(
        "A hospital must keep operating through regional power failures. "
        "It must preserve patient records, coordinate staff across wards."
    )
    for e in struct.entities:
        # no entity should be longer than 3 words (clause merge guard)
        assert len(e.split()) <= 3, f"entity too long (clause merge): {e!r}"


def test_multi_goal_problem_is_split_into_objectives():
    """A multi-objective problem yields one decomposition per objective."""
    spec = "Minimize latency AND cost AND carbon footprint for a global API."
    struct = identify_problem(spec)
    assert len(struct.objectives) >= 2, f"objectives not detected: {struct.objectives}"
    out = decompose(spec)
    assert any(
        s.get("target_slot", "").startswith("objective:")
        for s in out["subproblems"]
    ), "no per-objective subproblem produced"


def test_vague_problem_yields_clarification_angles():
    """An ill-posed ask is flagged and yields clarification angles."""
    out = decompose("Make it better.")
    assert out["structure"]["is_vague"] is True
    gen = [s for s in out["subproblems"] if s.get("is_new", True)]
    assert any(s.get("heuristic") == "clarification" for s in gen)
    # Must NOT emit a type-specific angle that merely echoes the vague goal.
    assert not any("Formulate the objective function" in s["subproblem"] for s in gen)


def test_well_posed_problem_not_flagged_vague():
    for spec in [
        "Shortest-path routing for a delivery fleet under time-window constraints.",
        "Prove there are infinitely many primes.",
        "Design a scheme so a message is processed exactly once despite retries.",
    ]:
        assert identify_problem(spec).is_vague is False, f"false vague flag: {spec}"
