"""T3.4 — Determinism harness.

Runs key pipelines twice (in separate processes, to catch seed/state leakage) and
asserts the primary outputs are byte-identical. Guards the engine's determinism
claims (distilled embedder, decomposition, pareto).
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")

_SNIPPET = r"""
import sys, hashlib, json
sys.path.insert(0, {src!r})
out = {{}}
from super_solver.frameworks.problem_identification import decompose
d = decompose("Reduce latency without increasing cost while staying under budget.")
out["decompose"] = [s["subproblem"] for s in d["subproblems"]]
from super_solver.core.distilled_pid import DistilledPIDEmbedder
e = DistilledPIDEmbedder()
if e.available:
    out["emb"] = [float(x) for x in e.encode("shortest path routing under time windows")][:16]
from super_solver.frameworks.pareto import Objective, Candidate, resolve
r = resolve([Objective("x", "minimize"), Objective("y", "minimize")],
            [Candidate("a", {{"x": 1, "y": 9}}), Candidate("b", {{"x": 9, "y": 1}})])
out["pareto"] = r.pareto_front
print(hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest())
"""


def _run():
    src = os.path.join(ROOT, "src")
    proc = subprocess.run(
        [sys.executable, "-c", _SNIPPET.format(src=src)],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()[-1]


def test_pipelines_are_deterministic_across_processes():
    h1 = _run()
    h2 = _run()
    assert h1 == h2, f"non-deterministic output: {h1} != {h2}"
