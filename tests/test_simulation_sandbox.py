"""Tests for the In Silico Code Experiment Simulation Sandbox."""

from super_solver.engine import SuperDuperProblemSolvingEngine
from super_solver.frameworks.sandbox import CodeExperimentSandbox


def test_sandbox_executes_python_script_and_parses_json():
    """Validates executing isolated python code and extracting structured JSON observations."""
    sandbox = CodeExperimentSandbox()

    code = """
import json
data = {"status": "success", "observation": "peak_detected", "snr": 14.2}
print("__RESULT_JSON__:" + json.dumps(data))
"""
    res = sandbox.execute_python_code(code)
    assert res.success is True
    assert res.output["status"] == "success"
    assert res.output["observation"] == "peak_detected"
    assert res.observation_category == "peak_detected"
    assert res.execution_time_ms > 0


def test_sandbox_thermal_simulation_demarcation():
    """Validates physical simulation distinguishing impurity artifacts from true superconductivity."""
    sandbox = CodeExperimentSandbox()

    # 1. Simulate multiphase sample with Cu2S impurity at 377 K (104 C)
    res_imp = sandbox.run_thermal_phase_simulation(
        material_name="Pb10-xCux(PO4)6O",
        target_temp_k=377.0,
        has_secondary_phase=True,
    )
    assert res_imp.success is True
    assert res_imp.output["is_superconducting"] is False
    assert res_imp.output["observation"] == "resistivity_drop_without_zero_resistance"
    assert "Cu2S" in res_imp.output["mechanism"]

    # 2. Simulate pure single-crystal sample at 377 K
    res_pure = sandbox.run_thermal_phase_simulation(
        material_name="Pure Pb9Cu(PO4)6O",
        target_temp_k=377.0,
        has_secondary_phase=False,
    )
    assert res_pure.success is True
    assert res_pure.output["is_superconducting"] is False
    assert res_pure.output["observation"] == "transparent_insulator"


def test_sandbox_timeout_protection():
    """Validates that infinite loops or hanging simulations terminate safely without hanging."""
    sandbox = CodeExperimentSandbox(timeout=0.5)

    hanging_code = """
import time
while True:
    time.sleep(0.1)
"""
    res = sandbox.execute_python_code(hanging_code)
    assert res.success is False
    assert "timed out" in (res.error_message or "").lower()
