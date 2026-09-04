#!/usr/bin/env python3
"""Example: Running autonomous self-audit and optimization against the engine itself."""

from super_solver.engine import SuperDuperProblemSolvingEngine


def main():
    print("=== Super-Duper-Problem-Solving-Engine: Autonomous Self-Audit ===")
    engine = SuperDuperProblemSolvingEngine()

    audit = engine.run_self_audit_and_optimization()

    print(f"\nAudit Status: {audit['self_evaluation_status']}")
    print(f"DPLL Logic Prover Verdict: {audit['dpll_invariant_verification']['verdict']}")
    print(f"Causal ACE Effect: +{audit['causal_intervention_audit']['average_causal_effect']:.2f}")
    print(f"BOED EIG Utility: {audit['boed_tuning_utility']['expected_information_gain']:.3f}")
    print(f"Calibrated Parameters: {audit['tuned_parameters']}")

    print("\nExecuting Self-Improvement Trajectory...")
    discovery = engine.improve_self()
    print(f"Optimization Outcome: {discovery.final_breakthrough}")


if __name__ == "__main__":
    main()
