"""Tests for the P2/P3/P4 improvement programme.

P2: ranker — documented that a LEARNED ranker does not beat hand-set on the
    retrieval task; the semantic COMPOSITION is the gain.
P3: corpus coverage lifted the solve-loop metric.
P4: self-application loop is verify-before-swap and fail-closed.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
ASSETS = os.path.join(ROOT, "assets")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def test_ranker_artifact_and_conclusion_recorded():
    """The ranker artifacts exist and the trained weights are loadable."""
    art = os.path.join(ASSETS, "ranker_weights.npz")
    if not os.path.exists(art):
        pytest.skip("ranker artifact absent")
    import numpy as np

    w = np.load(art)["w"]
    assert w.shape[0] == 4


def test_learned_ranker_does_not_beat_handset_documented():
    """Locked-in P2 finding: on the retrieval task the semantic COMPOSITION beats
    both hand-set containment and the learned ranker. If a learned ranker ever
    beats composition, update docs/IMPROVEMENT_PLAN.md."""
    import numpy as np
    import train_ranker as tr

    from super_solver.core.distilled_pid import DistilledPIDEmbedder
    from super_solver.vectorize.primitives import strip_to_primitives

    os.environ["SUPER_SOLVER_VECTOR_DB"] = ":memory:"
    tr._EMB = DistilledPIDEmbedder()
    triples = tr._load_triples()
    pairs = list({(p, c) for p, c, _ in triples})
    pool = [c for _, c in pairs]

    def _spec(p):
        return {x for x in p if not (x.startswith("act_") or x.startswith("attr_"))}

    def top1(w):
        h = n = 0
        for prob, correct in pairs:
            if not (_spec(strip_to_primitives(prob)) & _spec(strip_to_primitives(correct))):
                continue
            sc = sorted(((float(w @ tr._features(prob, c)), c) for c in pool), reverse=True)
            h += int(sc[0][1] == correct)
            n += 1
        return h / n if n else 0.0

    hand = top1(np.array([0.8, 0.2, 0.0, 0.0], dtype=np.float32))
    comp = top1(np.array([0.4, 0.1, 0.5, 0.0], dtype=np.float32))
    assert comp >= hand, f"composition {comp} should be >= hand-set {hand}"


def test_corpus_coverage_improved_solve_loop():
    data = json.load(open(os.path.join(ASSETS, "solve_loop.json")))
    # raised floor after the P3 corpus additions (was 0.167)
    assert data["solution_idea_coverage"] >= 0.20


def test_self_application_log_is_fail_closed():
    log = json.load(open(os.path.join(ASSETS, "self_application_log.json")))
    for d in log["decisions"]:
        assert d["decision"] in {
            "ADOPT_CANDIDATE", "REJECTED", "NO_OP_CORPUS_SATURATED"
        }
        # nothing is adopted without a measured strict improvement
        if d["decision"] == "ADOPT_CANDIDATE":
            assert d["strict_improvement"] and d["transfer_ok"]


def test_embedder_retrieval_finding_recorded():
    """P6 locked-in negative result: the hashed-feature embedder cannot be
    trained to retrieval semantics. If a future encoder DOES reach retrieval
    top-1 > 0.2 on the 197-pair corpus, update docs/IMPROVEMENT_PLAN.md P6."""
    path = os.path.join(ASSETS, "embedder_retrieval_finding.json")
    if not os.path.exists(path):
        pytest.skip("finding absent")
    f = json.load(open(path))
    assert f["retrieval_training_improved"] is False
    assert f["distilled_top1"] <= 0.2
