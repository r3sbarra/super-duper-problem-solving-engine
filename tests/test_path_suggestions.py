"""Tests path suggestion and trajectory repair capabilities.

Validates that SuperDuperProblemSolvingEngine can autonomously suggest multiple
novel, ranked paths forward for both:
1. Mathematical challenges blocked by barriers (Lakatos repair strategies).
2. Scientific and engineering trade-offs (TRIZ, Polya, Gentner, and Platt).
"""

from super_solver.engine import SuperDuperProblemSolvingEngine


def test_suggest_paths_for_mathematical_conjecture():
    """Validates suggesting barrier-evasion paths for P vs NP complexity lower bounds."""
    engine = SuperDuperProblemSolvingEngine()

    problem_spec = (
        "Prove circuit size lower bounds for NP-complete problems against general Boolean circuits. "
        "Combinatorial monotone approximations and oracle simulations fail due to known barriers."
    )
    context = "P versus NP"

    suggested = engine.suggest_paths(
        problem_specification=problem_spec,
        context=context,
        top_k=3,
    )

    assert len(suggested) == 3
    for p in suggested:
        assert p.path_id
        assert p.title
        assert p.description
        assert p.feasibility_score > 0.0
        assert p.novelty_score > 0.0
        assert p.dead_end_safety_margin > 0.0
        assert len(p.recommended_next_action) > 10

    # Verify that mathematical barrier evasion paths (e.g. Arithmetization or Fine-Grained SETH) are suggested
    path_titles = [p.title.lower() for p in suggested]
    assert any(
        "arithmetization" in t or "fine-grained" in t or "seth" in t or "inventive" in t
        for t in path_titles
    )


def test_suggest_paths_for_scientific_engineering_tradeoff():
    """Validates suggesting discovery paths for an energy storage bottleneck."""
    engine = SuperDuperProblemSolvingEngine()

    problem_spec = (
        "Lithium metal battery: Rapid charging rate causes lithium dendrite growth and thermal runaway. "
        "Need ultra-fast charging speed without compromising safety and cycle lifespan."
    )

    suggested = engine.suggest_paths(
        problem_specification=problem_spec,
        top_k=3,
    )

    assert len(suggested) == 3
    strategies = [p.strategy_type for p in suggested]

    # Verify diverse multi-paradigm suggestions (TRIZ, Polya, Gentner, Platt)
    assert any("TRIZ" in s or "POLYA" in s or "ANALOGICAL" in s or "PLATT" in s for s in strategies)

    # Check top suggested path structure
    top_path = suggested[0]
    assert top_path.feasibility_score >= 0.20
    assert top_path.novelty_score >= 0.10
    assert top_path.dead_end_safety_margin >= 0.50
    assert len(top_path.rationale) > 15
