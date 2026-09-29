"""Tests for the principled problem-identification & decomposition pipeline.

Regression guard for the fixes to super_solver.frameworks.problem_identification:
- type identification accuracy (>= 5/6 on the labeled set),
- entity extraction surfaces domain nouns (not generic verbs),
- generated subproblems have a non-compressed relevance spread,
- slot coverage >= 0.6,
- caller-subproblem ranking separates overlapping probes (AUC 1.0) and the
  known paraphrase ceiling is documented rather than silently faked.
"""
from __future__ import annotations

import statistics

from super_solver.frameworks.problem_identification import (
    PROBLEM_TYPES,
    decompose,
    identify_problem,
    relevance_score,
)

TYPED = [
    ("Lithium metal battery: rapid charging causes dendrite growth and thermal runaway.", "diagnosis"),
    ("Shortest-path routing for a delivery fleet under time-window constraints.", "optimization"),
    ("Prove that every even integer greater than 2 is the sum of two primes.", "proof"),
    ("Design a lightweight climbing pad that grips smooth vertical walls.", "design"),
    ("Estimate the half-life of the isotope from the decay curve.", "estimation"),
    ("Find the exit in an unknown maze using the fewest probes.", "search"),
]

OVERLAP_PROBES = [
    {
        "problem": "Lithium metal battery: rapid charging causes dendrite growth and thermal runaway.",
        "relevant": ["Characterize dendrite nucleation kinetics at the anode surface"],
        "irrelevant": ["Design a marketing campaign for electric vehicles"],
    },
    {
        "problem": "Prove that every even integer greater than 2 is the sum of two primes.",
        "relevant": ["Search for a counterexample within a large parameter range"],
        "irrelevant": ["Book a conference venue"],
    },
]


def _auc(rel, irr):
    wins = ties = total = 0
    for r in rel:
        for i in irr:
            total += 1
            if r > i:
                wins += 1
            elif r == i:
                ties += 1
    return (wins + 0.5 * ties) / total if total else 0.0


def test_identify_problem_type_accuracy():
    correct = sum(
        int(identify_problem(spec).problem_type == gold)
        for spec, gold in TYPED
    )
    assert correct >= 5, f"type id accuracy too low: {correct}/6"


def test_entities_are_domain_nouns_not_generic_verbs():
    st = identify_problem(
        "Lithium metal battery: rapid charging causes dendrite growth and thermal runaway."
    )
    joined = " ".join(st.entities).lower()
    assert "dendrite" in joined or "thermal runaway" in joined
    # generic framing verbs must not be surfaced as entities
    assert "rapid" not in st.entities
    assert "causes" not in st.entities


def test_generated_subproblems_have_relevance_spread():
    out = decompose("Shortest-path routing for a delivery fleet under time-window constraints.")
    rels = [s["relevance_score"] for s in out["subproblems"] if s["is_new"]]
    assert rels, "no subproblems generated"
    spread = max(rels) - min(rels)
    # Old template path compressed to ~0.04; require a much wider dynamic range.
    assert spread > 0.15, f"relevance spread still compressed: {spread}"


def test_slot_coverage():
    out = decompose("Design a lightweight climbing pad that grips smooth vertical walls.")
    assert out["metrics"]["slot_coverage"] >= 0.6
    assert "goal" in out["metrics"]["slots_covered"]


def test_subproblems_are_not_vacuous_anchor_fills():
    out = decompose("Lithium metal battery: rapid charging causes dendrite growth.")
    for s in out["subproblems"]:
        if not s["is_new"]:
            continue
        low = s["subproblem"].lower()
        # The original bug produced e.g. "Test the asymptotic behavior of rapid..."
        assert " of rapid " not in low
        assert " of causes " not in low


def test_caller_subproblem_ranking_overlap_probes():
    aucs = []
    for p in OVERLAP_PROBES:
        rel = [relevance_score(p["problem"], s)[0] for s in p["relevant"]]
        irr = [relevance_score(p["problem"], s)[0] for s in p["irrelevant"]]
        aucs.append(_auc(rel, irr))
    assert statistics.mean(aucs) >= 0.99, f"overlap ranking AUC too low: {aucs}"


def test_redundancy_suppression_keeps_distinct_angles():
    out = decompose("Estimate the half-life of the isotope from the decay curve.")
    texts = [s["subproblem"] for s in out["subproblems"]]
    assert len(texts) == len(set(texts)), "duplicate subproblems survived"


def test_decompose_is_robust_to_adversarial_inputs():
    """Empty / degenerate / clause-heavy inputs must not crash and must degrade."""
    for spec, title in [
        ("", ""),
        ("   ", ""),
        ("a", "x"),
        ("the the the and and", ""),
    ]:
        out = decompose(spec, title)
        assert "structure" in out and "subproblems" in out
        # Degenerate input must not claim high slot coverage.
        assert out["metrics"]["slot_coverage"] <= 0.6


def test_entities_do_not_absorb_subordinate_clauses():
    st = identify_problem(
        "Optimize the trade-off between battery energy density and charge speed "
        "while staying under 5% degradation."
    )
    for ent in st.entities:
        assert "while" not in ent.split(), f"entity absorbed a clause: {ent}"
        assert not ent.endswith("and"), f"entity has dangling connective: {ent}"


# --- Optional semantic scorer -------------------------------------------------

def test_semantic_scorer_is_off_by_default(monkeypatch):
    """With the flag unset, no semantic term is used (zero-daemon contract)."""
    monkeypatch.delenv("SUPER_SOLVER_SEMANTIC_SCORER", raising=False)
    import super_solver.frameworks.problem_identification as pid

    # Reset the memoized resolution so the env change is observed.
    pid._SEMANTIC_STATE["resolved"] = False
    pid._SEMANTIC_STATE["backend"] = None
    assert pid.semantic_similarity("hello", "hi") is None
    _, breakdown = pid.relevance_score("shortest path routing", "relax edges in a graph")
    assert "semantic" not in breakdown


def test_semantic_scorer_flag_without_daemon_falls_back(monkeypatch):
    """Flag on + unreachable daemon must fall back to polarity, not raise."""
    monkeypatch.setenv("SUPER_SOLVER_SEMANTIC_SCORER", "1")
    # Point at a dead port so the ollama backend cannot reach a daemon.
    monkeypatch.setenv("SUPER_SOLVER_LLM_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
    import importlib

    import super_solver.frameworks.problem_identification as pid

    importlib.reload(pid)
    # semantic may be None (fallback) or a real value only if a daemon happens
    # to be reachable; either way it must not raise and score must be finite.
    score, breakdown = pid.relevance_score("a parent problem", "a child")
    assert score == score  # not NaN
    assert isinstance(breakdown, dict)
    importlib.reload(pid)
