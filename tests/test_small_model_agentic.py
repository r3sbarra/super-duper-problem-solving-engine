"""Unit tests for small/low-end model agentic tool-use, programming, and diagnosis."""

import pytest
from super_solver.core.tool_registry import (
    ToolRegistry,
    ToolDefinition,
    ToolLoopCircuitBreaker,
    create_standard_tool_registry,
)
from super_solver.frameworks.code_synthesizer import CodeSynthesizer, SynthesisResult
from super_solver.frameworks.diagnostician import AutomatedDiagnostician, DiagnosticReport
from super_solver.engine import SuperDuperProblemSolvingEngine


def test_tool_registry_registration_and_execution():
    registry = ToolRegistry()

    def add_numbers(a: int, b: int) -> int:
        return a + b

    registry.register(
        name="add_numbers",
        description="Adds two integers together.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        handler=add_numbers,
        primitives=["addition", "arithmetic", "sum"],
    )

    # Validation checks
    valid, err = registry.validate_tool_call("add_numbers", {"a": 10, "b": 20})
    assert valid is True
    assert err is None

    invalid, err2 = registry.validate_tool_call("add_numbers", {"a": 10})
    assert invalid is False
    assert "Missing required parameter 'b'" in err2

    # Execution check
    res = registry.execute("add_numbers", {"a": 15, "b": 25})
    assert res.success is True
    assert res.output == 40
    assert res.execution_time_ms >= 0.0


def test_tool_registry_dynamic_pruning():
    registry = create_standard_tool_registry()
    assert len(registry.list_all()) >= 3

    # Prune for a python code execution request
    pruned = registry.prune_tools_for_query("execute numerical calculation in python", top_k=1)
    assert len(pruned) == 1
    assert pruned[0].name == "run_python_code"


def test_tool_loop_circuit_breaker():
    breaker = ToolLoopCircuitBreaker(max_repetitions=3, max_total_steps=5)

    # 1. Distinct calls pass
    is_loop, _ = breaker.record_and_check("tool_a", {"x": 1})
    assert is_loop is False

    is_loop, _ = breaker.record_and_check("tool_b", {"x": 2})
    assert is_loop is False

    # 2. 3 identical calls trigger loop detection
    breaker.record_and_check("search", {"q": "error"})
    breaker.record_and_check("search", {"q": "error"})
    is_loop, reason = breaker.record_and_check("search", {"q": "error"})
    assert is_loop is True
    assert "Repetitive tool call loop detected" in reason


def test_code_synthesizer_deterministic_fallback():
    synthesizer = CodeSynthesizer()
    res = synthesizer.synthesize_and_verify(
        specification="Calculate fibonacci sequence up to N",
        test_cases="assert True",
    )
    assert res.success is True
    assert "def solution" in res.code


def test_code_synthesizer_code_extraction():
    raw_response = (
        "Here is the solution:\n"
        "```python\n"
        "def add(x, y):\n"
        "    return x + y\n"
        "```\n"
        "Hope this helps!"
    )
    extracted = CodeSynthesizer.extract_python_code(raw_response)
    assert extracted == "def add(x, y):\n    return x + y"


def test_diagnostician_boundary_and_scoring():
    diagnostician = AutomatedDiagnostician()
    report = diagnostician.diagnose_failure(
        error_log_or_description="ZeroDivisionError in calculate_ratio when denominator is zero",
        candidate_causes=[
            "Zero division check missing before calculating ratio",
            "Network timeout while connecting to database",
            "Disk space exhausted on /tmp partition",
        ],
    )

    assert isinstance(report, DiagnosticReport)
    assert len(report.candidates) == 3
    # The zero division candidate should have the highest score
    top = report.top_root_cause
    assert top is not None
    assert "Zero division" in top.cause
    assert top.net_score > report.candidates[-1].net_score


def test_engine_small_model_agentic_methods():
    engine = SuperDuperProblemSolvingEngine()

    # 1. engine.program
    prog_res = engine.program("Sort a list of numbers", test_cases="assert True")
    assert isinstance(prog_res, SynthesisResult)
    assert prog_res.success is True

    # 2. engine.diagnose
    diag_res = engine.diagnose("TypeError: unhashable type: 'list'")
    assert isinstance(diag_res, DiagnosticReport)
    assert diag_res.top_root_cause is not None

    # 3. engine.run_agentic_tool_loop (zero-LLM default terminates cleanly)
    tool_loop_res = engine.run_agentic_tool_loop("Find the prime numbers under 20", max_steps=2)
    assert "steps_taken" in tool_loop_res
    assert "history" in tool_loop_res
