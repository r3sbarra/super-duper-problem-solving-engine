"""T2.4 — Cross-framework chaining runs end-to-end.

A problem is routed through a chain (identify -> KT boundary -> abduct -> Platt
assay), each stage consuming the previous stage's output. Guards that the chain
completes and every stage yields a non-empty artifact.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_framework_chain_completes():
    from super_solver.engine import SuperDuperProblemSolvingEngine

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import importlib

    fc = importlib.import_module("framework_chain")
    engine = SuperDuperProblemSolvingEngine()
    r = fc.run_chain(engine, "The service slows down only at month-end; diagnose the cause.")
    assert r["stages_ok"]
    stages = [c["stage"] for c in r["chain"]]
    assert stages == ["identify", "kt_boundary", "abduct", "platt_assay"]
    # every stage produced something
    for c in r["chain"]:
        assert c.get("stage")
    assert r["chain"][1]["n_candidates"] >= 1, "KT stage produced no candidates"
    assert r["chain"][2]["n_hypotheses"] >= 1, "abduction produced no hypotheses"
    assert r["chain"][3]["n_experiments"] >= 1, "Platt produced no experiments"
