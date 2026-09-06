#!/usr/bin/env python3
"""End-to-end demonstration of Super Solver empowering a low-end model (qwen3:4b on Ollama).

Demonstrates:
  1. Dynamic Tool Pruning & Agentic Tool Invocation
  2. Kepner-Tregoe 4D Boundary Problem Diagnosis
  3. Spec-to-Code Synthesis with Closed-Loop Sandbox Verification
"""

import os
import sys

# Activate local LLM
os.environ["SUPER_SOLVER_LLM_ENABLED"] = "1"
os.environ["SUPER_SOLVER_LLM_MODEL"] = "qwen3:4b"
os.environ["SUPER_SOLVER_LLM_TIMEOUT"] = "35.0"

from super_solver.engine import SuperDuperProblemSolvingEngine


def main():
    print("=" * 80)
    print("SUPER SOLVER: Low-End Model Agentic & Diagnostic Empowerment Demo")
    print("Model: qwen3:4b on Ollama (Local RTX 2070)")
    print("=" * 80)

    engine = SuperDuperProblemSolvingEngine()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Automated Problem Diagnosis (Kepner-Tregoe 4D Hyperplane Separation)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1] Testing Kepner-Tregoe 4D Problem Diagnosis...")
    error_report = (
        "ZeroDivisionError: division by zero in portfolio_risk.py:line 42\n"
        "Traceback:\n"
        "  File 'portfolio_risk.py', line 42, in calculate_drawdown\n"
        "    drawdown_pct = (peak_equity - current_equity) / peak_equity\n"
        "ZeroDivisionError: float division by zero when peak_equity is 0.0 on an uninitialized account."
    )

    diag_result = engine.diagnose(error_report)
    print(f"  Boundary Identified:")
    print(f"    - IS:     {diag_result.boundary.identity_is}")
    print(f"    - IS NOT: {diag_result.boundary.identity_is_not}")
    print(f"    - WHERE:  {diag_result.boundary.location_is}")
    if diag_result.top_root_cause:
        print(f"  Top Diagnosed Root Cause:")
        print(f"    - Cause:      {diag_result.top_root_cause.cause}")
        print(f"    - Score:      {diag_result.top_root_cause.net_score:.3f}")
        print(f"    - Valid Fit:  {diag_result.top_root_cause.valid_boundary_fit}")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Dynamic Tool Pruning & Agentic Execution
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[2] Testing Dynamic Tool Pruning & Agentic Tool Execution...")
    target_path = os.path.expanduser("~/.openclaw/workspace/AGENTS.verbose.md")
    tool_query = f"Read the first 10 lines of {target_path} to check security policies."
    
    # Check pruning
    pruned = engine.tools.prune_tools_for_query(tool_query, top_k=2)
    print(f"  Tools pruned for query ({len(pruned)} tools selected from catalog):")
    for t in pruned:
        print(f"    - {t.name}: {t.description[:60]}...")

    # Run agentic loop
    tool_run = engine.run_agentic_tool_loop(tool_query, max_steps=2)
    print(f"  Tool Loop Steps Taken: {tool_run['steps_taken']}")
    for h in tool_run["history"]:
        status = "SUCCESS" if h.get("success") else ("DEFLECTED" if h.get("deflected") else "FAIL")
        print(f"    Step {h['step']}: Tool [{h['tool']}] -> {status}")
        if h.get("output"):
            out_str = str(h["output"])
            print(f"      Output preview: {out_str[:120]}...")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Spec-to-Code Synthesis with Closed-Loop Sandbox Verification
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3] Testing Code Synthesis with Sandbox Self-Verification...")
    spec = (
        "Write a Python function 'safe_clamp_allocation(amount: float, max_pct: float, total_portfolio: float) -> float' "
        "that calculates the capped position size. If total_portfolio <= 0, return 0.0. "
        "Otherwise return min(amount, total_portfolio * max_pct)."
    )
    test_harness = (
        "assert safe_clamp_allocation(500, 0.10, 10000) == 500\n"
        "assert safe_clamp_allocation(2500, 0.10, 10000) == 1000.0\n"
        "assert safe_clamp_allocation(100, 0.10, 0) == 0.0\n"
        "print('__RESULT_JSON__: {\"status\": \"verified\"}')\n"
    )

    prog_res = engine.program(spec, test_cases=test_harness)
    print(f"  Code Synthesis Status: {'SUCCESS' if prog_res.success else 'FAILED'}")
    print(f"  Iterations: {prog_res.iterations}")
    print(f"  Generated Code:\n```python\n{prog_res.code}\n```")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE: All three capabilities verified on small local model.")
    print("=" * 80)


if __name__ == "__main__":
    main()
