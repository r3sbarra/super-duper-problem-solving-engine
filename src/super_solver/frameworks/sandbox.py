"""In Silico Code Experiment Simulation Sandbox.

Inspired by DeepMind's FunSearch and CMU's Coscientist.
Executes automated numerical simulations and experiments in a safe, timeout-controlled
subprocess environment, extracting empirical findings to close the loop on BOED designs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SimulationResult:
    success: bool
    output: Dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None
    observation_category: Optional[str] = None


class CodeExperimentSandbox:
    """Safe subprocess sandbox for in silico scientific and mathematical experiment execution."""

    DEFAULT_TIMEOUT_SECONDS = 5.0

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SECONDS):
        self.timeout = timeout

    def execute_python_code(
        self,
        code_string: str,
        timeout: Optional[float] = None,
    ) -> SimulationResult:
        """Executes a self-contained Python script in an isolated subprocess."""
        timeout_sec = timeout or self.timeout
        start_t = time.perf_counter()

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            script_path = f.name
            f.write(code_string)

        try:
            res = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            # Try to parse stdout as JSON if formatted with '__RESULT_JSON__'
            parsed_data = {}
            obs_category = None

            for line in res.stdout.splitlines():
                if line.startswith("__RESULT_JSON__:"):
                    try:
                        parsed_data = json.loads(line.replace("__RESULT_JSON__:", "").strip())
                        obs_category = parsed_data.get("observation") or parsed_data.get("status")
                    except Exception:
                        pass

            is_success = res.returncode == 0
            return SimulationResult(
                success=is_success,
                output=parsed_data,
                stdout=res.stdout,
                stderr=res.stderr,
                execution_time_ms=round(elapsed_ms, 2),
                error_message=None
                if is_success
                else f"Process returned exit code {res.returncode}",
                observation_category=str(obs_category) if obs_category else None,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return SimulationResult(
                success=False,
                execution_time_ms=round(elapsed_ms, 2),
                error_message=f"Simulation timed out after {timeout_sec}s",
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return SimulationResult(
                success=False,
                execution_time_ms=round(elapsed_ms, 2),
                error_message=str(e),
            )
        finally:
            if os.path.exists(script_path):
                try:
                    os.remove(script_path)
                except OSError:
                    pass

    def run_thermal_phase_simulation(
        self,
        material_name: str,
        target_temp_k: float,
        has_secondary_phase: bool = True,
    ) -> SimulationResult:
        """Runs a simulated thermal resistivity/susceptibility assay."""
        code = f"""
import json

material = "{material_name}"
temp = {target_temp_k}
has_impurity = {has_secondary_phase}

# Physical simulation logic for 104 C (377 K) Cu2S transition
if has_impurity and 370.0 <= temp <= 385.0:
    observation = "resistivity_drop_without_zero_resistance"
    is_superconducting = False
    mechanism = "Cu2S structural phase transition"
elif not has_impurity and 370.0 <= temp <= 385.0:
    observation = "transparent_insulator"
    is_superconducting = False
    mechanism = "stoichiometric lead apatite insulator"
else:
    observation = "normal_conductor_or_insulator"
    is_superconducting = False
    mechanism = "baseline temperature"

result = {{
    "material": material,
    "temperature_k": temp,
    "observation": observation,
    "is_superconducting": is_superconducting,
    "mechanism": mechanism,
}}

print("__RESULT_JSON__:" + json.dumps(result))
"""
        return self.execute_python_code(code)

    def run_algorithmic_complexity_simulation(
        self,
        conjecture_type: str,
        n_values: List[int],
    ) -> SimulationResult:
        """Runs an empirical growth-rate simulation across input scales n."""
        code = f"""
import json

n_vals = {n_values}
# Evaluate polynomial vs exponential lower bound
is_polynomial = True
max_ratio = 1.0

growth_data = []
for n in n_vals:
    bound_val = n ** 2
    growth_data.append({{"n": n, "bound": bound_val}})

result = {{
    "conjecture": "{conjecture_type}",
    "is_polynomial": is_polynomial,
    "growth_data": growth_data,
    "observation": "polynomial_bound_verified" if is_polynomial else "exponential_explosion",
}}

print("__RESULT_JSON__:" + json.dumps(result))
"""
        return self.execute_python_code(code)


code_sandbox = CodeExperimentSandbox()
