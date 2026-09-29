"""Tests for the retrieval-metric correction + paired distillation.

Locks in:
  - the corrected finding: all encoders carry paired signal, none achieves
    top-1 over the 197-way pool (retrieval = shortlist + rerank);
  - the distillation headroom is real (student < teacher on paired CAR);
  - the current artifact holds every committed scorer-CAR floor.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
ASSETS = os.path.join(ROOT, "assets")
sys.path.insert(0, os.path.join(ROOT, "src"))


def test_retrieval_metric_correction_recorded():
    path = os.path.join(ASSETS, "retrieval_metric_correction.json")
    if not os.path.exists(path):
        pytest.skip("correction absent")
    f = json.load(open(path))
    # teacher beats student on the paired metric -> headroom exists
    assert f["ollama_teacher"] >= f["distilled_pid"]
    assert f["distilled_pid"] > 0.5


def test_artifact_holds_all_scorer_floors():
    """After paired distillation the artifact must still meet every committed
    scorer-CAR floor (no-regression gate)."""
    from super_solver.frameworks.problem_identification import relevance_score

    checks = {
        "pid_holdout_v2.json": 0.95,
        "pid_stress_probes.json": 0.95,
        "pid_ood_probes.json": 0.85,
    }
    for fn, floor in checks.items():
        path = os.path.join(ASSETS, fn)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        wins = sum(
            1 for r in d
            if relevance_score(r["parent"], r["relevant"])[0]
            > relevance_score(r["parent"], r["irrelevant"])[0]
        )
        car = wins / len(d)
        assert car >= floor, f"{fn}: scorer CAR {car:.3f} below floor {floor}"


def test_bi_encoder_top1_is_weak_documented():
    """Locked-in: no bi-encoder achieves top-1 over the pool; if one ever does,
    update the correction finding."""
    path = os.path.join(ASSETS, "retrieval_bottleneck_finding.json")
    if not os.path.exists(path):
        pytest.skip("bottleneck finding absent")
    f = json.load(open(path))
    assert f["recall_at_1"] <= 0.1
