"""Metrics artifact regression guard.

Runs the published metrics collector and asserts every headline metric meets its
floor, so an improvement/regression to the engine is caught in CI. This is the
committed-artifact counterpart of scripts/publish_metrics.py --check.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
SCRIPT = os.path.join(ROOT, "scripts", "publish_metrics.py")


def test_metrics_floors_met():
    if not os.path.exists(SCRIPT):
        pytest.skip("publish_metrics.py absent")
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"metrics regression:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    )


def test_metrics_artifact_is_committed():
    art = os.path.join(ROOT, "assets", "metrics_artifact.json")
    assert os.path.exists(art), "metrics_artifact.json not present"
