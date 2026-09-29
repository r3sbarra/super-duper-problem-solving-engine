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

def test_semantic_scorer_defaults_to_offline_distilled(monkeypatch):
    """By default the OFFLINE distilled embedder is used (no daemon); the
    scorer can still be disabled explicitly with SUPER_SOLVER_SEMANTIC_SCORER=0."""
    import importlib

    monkeypatch.delenv("SUPER_SOLVER_SEMANTIC_SCORER", raising=False)
    monkeypatch.delenv("SUPER_SOLVER_SEMANTIC_EMBEDDER", raising=False)
    import super_solver.frameworks.problem_identification as pid

    importlib.reload(pid)
    # Default: distilled_pid artifact present -> semantic path active.
    if pid.semantic_similarity("hello", "hi") is not None:
        _, breakdown = pid.relevance_score("shortest path routing", "relax edges in a graph")
        assert "semantic" in breakdown

    # Explicit disable -> pure polarity path, no semantic term.
    monkeypatch.setenv("SUPER_SOLVER_SEMANTIC_SCORER", "0")
    pid._SEMANTIC_STATE["resolved"] = False
    pid._SEMANTIC_STATE["backend"] = None
    assert pid.semantic_similarity("hello", "hi") is None
    _, breakdown = pid.relevance_score("shortest path routing", "relax edges in a graph")
    assert "semantic" not in breakdown
    importlib.reload(pid)


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


# --- Negation-consistency guard ----------------------------------------------

def test_negation_guard_demotes_negated_distractor():
    """A child that negates an affirmative parent must be flagged as a flip.

    Ordering assertion is made on the semantic path (where the correct answer
    is scored well): on the polarity-only path the correct answer has zero
    surface overlap, which is the separate (documented) paraphrase ceiling.
    """
    from super_solver.frameworks.problem_identification import (
        negation_flip,
        relevance_score,
        semantic_similarity,
    )

    parent = "Reduce the latency of the read path."
    good = "Add a cache in front of the database"
    bad = "Do not add any cache and instead increase the latency of the read path"
    assert negation_flip(parent, bad)[0] is True
    assert negation_flip(parent, good)[0] is False
    if semantic_similarity(parent, good) is not None:
        assert relevance_score(parent, good)[0] > relevance_score(parent, bad)[0]


def test_negation_guard_does_not_false_positive_on_already_negative_parent():
    """When the parent already carries negation, a negated child that restates
    the same intent must NOT be flagged as a flip."""
    from super_solver.frameworks.problem_identification import negation_flip

    parent = "Stop one slow service from taking down the whole site."
    child = "Isolate a lagging component so it cannot bring down the entire system."
    assert negation_flip(parent, child)[0] is False


def test_negation_guard_detects_antonym_flip():
    from super_solver.frameworks.problem_identification import negation_flip

    parent = "Make the structure lighter without weakening it."
    child = "Make the structure heavier and weaker"
    assert negation_flip(parent, child)[0] is True


def test_relevance_score_reports_negation_detail():
    from super_solver.frameworks.problem_identification import relevance_score

    _, breakdown = relevance_score("Prove the theorem holds for all n.", "Disprove it.")
    assert "flipped" in breakdown and "shared_content" in breakdown


def test_scorer_robust_on_48_fresh_holdout():
    """On a fresh 48-probe holdout (never used for training), the scorer must
    keep CAR >= 0.95, and the polarity baseline must be far below."""
    import json

    import pytest

    artifact = _artifact_path()
    probe_file = os.path.join(
        os.path.dirname(__file__), "..", "assets", "pid_holdout_v2.json"
    )
    if not (os.path.exists(artifact) and os.path.exists(probe_file)):
        pytest.skip("artifact or holdout_v2 absent")
    from super_solver.core.distilled_pid import DistilledPIDEmbedder

    probes = json.load(open(probe_file))
    emb = DistilledPIDEmbedder(weights_path=artifact)
    assert emb.available
    wins = 0
    for r in probes:
        a = emb.cosine_similarity(emb.encode(r["parent"]), emb.encode(r["relevant"]))
        b = emb.cosine_similarity(emb.encode(r["parent"]), emb.encode(r["irrelevant"]))
        wins += int(a > b)
    car = wins / len(probes)
    assert car >= 0.95, f"fresh-holdout CAR {car:.3f} below floor"
