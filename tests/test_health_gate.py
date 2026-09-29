"""T3.1 — CI gate: every probed subsystem must be OK (no FAIL, no DEGENERATE).

Runs scripts/health_probe.py and asserts the summary is clean, so a subsystem
regressing to a crash or a degenerate (empty/non-discriminating) output fails CI.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
SCRIPT = os.path.join(ROOT, "scripts", "health_probe.py")
REPORT = os.path.join(ROOT, "assets", "health_report.json")


def test_health_probe_all_ok():
    if not os.path.exists(SCRIPT):
        pytest.skip("health_probe.py absent")
    proc = subprocess.run(
        [sys.executable, SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, f"health probe crashed:\n{proc.stderr}"
    report = json.load(open(REPORT))
    summary = report["summary"]
    assert summary["fail"] == 0, (
        "subsystems FAILING: "
        + ", ".join(r["subsystem"] for r in report["results"] if r["status"] == "FAIL")
    )
    assert summary["degenerate"] == 0, (
        "subsystems DEGENERATE: "
        + ", ".join(
            r["subsystem"] for r in report["results"] if r["status"] == "DEGENERATE"
        )
    )
