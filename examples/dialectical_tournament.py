#!/usr/bin/env python3
"""Example: Running a Red Team vs Blue Team dialectical debate tournament."""

from super_solver.engine import SuperDuperProblemSolvingEngine


def main():
    print("=== Super-Duper-Problem-Solving-Engine: Dialectical Tournament ===")
    engine = SuperDuperProblemSolvingEngine()

    hypotheses = [
        "Cu2S impurity phase transition drives the 104 C resistance artifact",
        "Stoichiometric lead apatite is an intrinsic room temperature superconductor",
        "Ferromagnetic iron impurities cause full Meissner levitation",
    ]

    context = (
        "Synthesis of Pb10-x Cux (PO4)6O generates Cu2S secondary phases. "
        "Pure stoichiometric single crystals exhibit high resistivity and no Meissner levitation."
    )

    results = engine.run_tournament(
        candidate_hypotheses=hypotheses,
        context=context,
        top_k=len(hypotheses),
    )

    print(f"\nDebate Results ({len(results)} evaluated):")
    for i, r in enumerate(results, 1):
        status = "SURVIVED" if r.survived else "ELIMINATED"
        print(f"[{i}] [{status}] Score: {r.robustness_score:.2f} | Verdict: {r.verdict.value}")
        print(f"    Hypothesis: {r.hypothesis}")


if __name__ == "__main__":
    main()
