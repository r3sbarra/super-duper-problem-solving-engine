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

import os
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


# --- Offline distilled embedder (artifact-gated) ------------------------------

def _artifact_path():
    import os

    return os.path.join(
        os.path.dirname(__file__), "..", "assets", "distilled_pid_large.npz"
    )


def test_distilled_pid_embedder_breaks_paraphrase_ceiling():
    """The committed offline artifact must rank correct paraphrases above
    distractors (CAR >= 0.875). Skips if the artifact is absent."""
    import pytest

    if not os.path.exists(_artifact_path()):
        pytest.skip("distilled_pid artifact not present")
    from super_solver.core.distilled_pid import DistilledPIDEmbedder

    e = DistilledPIDEmbedder(weights_path=_artifact_path())
    assert e.available
    probes = [
        ("Shortest-path routing for a delivery fleet under time-window constraints.",
         "Model the road network as a weighted graph and relax edges",
         "Paint the delivery van blue"),
        ("Prove that the sum of the first n odd numbers equals n squared.",
         "Use induction, pairing each new odd number with a growing square",
         "Book a flight to Berlin"),
        ("Design a scheme so a message is processed exactly once despite retries.",
         "Attach a unique id and deduplicate on the receiver side",
         "Water the office plants"),
    ]
    wins = 0
    for parent, rel, irr in probes:
        r = e.cosine_similarity(e.encode(parent), e.encode(rel))
        i = e.cosine_similarity(e.encode(parent), e.encode(irr))
        wins += int(r > i)
    assert wins == len(probes), f"distilled embedder CAR {wins}/{len(probes)}"


def test_distilled_pid_is_deterministic():
    import pytest

    if not os.path.exists(_artifact_path()):
        pytest.skip("distilled_pid artifact not present")
    from super_solver.core.distilled_pid import DistilledPIDEmbedder

    a = DistilledPIDEmbedder(weights_path=_artifact_path())
    b = DistilledPIDEmbedder(weights_path=_artifact_path())
    s = "routing a fleet under time windows"
    assert list(a.encode(s)) == list(b.encode(s))


def test_distilled_pid_robust_on_32_diverse_probes():
    """On the committed 32-probe stress set, the distilled embedder must score
    CAR >= 0.9 while the polarity baseline stays well below it."""
    import json

    import pytest

    artifact = _artifact_path()
    probe_file = os.path.join(
        os.path.dirname(__file__), "..", "assets", "pid_stress_probes.json"
    )
    if not (os.path.exists(artifact) and os.path.exists(probe_file)):
        pytest.skip("distilled artifact or stress probes absent")
    from super_solver.core.distilled_pid import DistilledPIDEmbedder
    from super_solver.core.embedder import get_backend

    probes = json.load(open(probe_file))
    distilled = DistilledPIDEmbedder(weights_path=artifact)
    polarity = get_backend("polarity")

    def car(embed):
        wins = 0
        for r in probes:
            a = embed.cosine_similarity(embed.encode(r["parent"]), embed.encode(r["relevant"]))
            b = embed.cosine_similarity(embed.encode(r["parent"]), embed.encode(r["irrelevant"]))
            wins += int(a > b)
        return wins / len(probes)

    dc = car(distilled)
    pc = car(polarity)
    assert dc >= 0.9, f"distilled CAR {dc:.3f} below floor"
    assert dc > pc + 0.3, f"distilled {dc:.3f} not clearly above polarity {pc:.3f}"
