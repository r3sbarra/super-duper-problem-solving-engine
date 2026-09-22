"""Unit tests for the lab-ass bridge: DPLL entail + BOED next_experiment."""

from super_solver.lab_ass_bridge.hooks import LabAssSuperSolverBridge


def test_bridge_dpll_entail_proves():
    bridge = LabAssSuperSolverBridge()
    result = bridge.dpll_entail(
        premises=["rain => wet", "wet => slippery", "rain"],
        target_claim="slippery",
    )
    assert result["proved"] is True
    assert result["verdict"] == "LOGICALLY_SOUND_PROOF"
    assert result["clauses_evaluated"] > 0


def test_bridge_dpll_entail_countermodel():
    bridge = LabAssSuperSolverBridge()
    result = bridge.dpll_entail(
        premises=["rain => wet"],
        target_claim="slippery",
    )
    assert result["proved"] is False
    assert result["verdict"] == "COUNTERMODEL_EXISTS"
    assert result["countermodel"] is not None


def test_bridge_decompose_merges_generated_subproblems():
    """decompose_topic_with_polya_and_triz must merge genuinely NEW generated
    subproblems (is_new=True) with the caller-supplied re-ranking, instead of only
    re-ranking the caller's own list. Regression for the finding that the Polya
    generator was never wired into the decompose path."""
    bridge = LabAssSuperSolverBridge()
    known = ["Background & Context", "Current State & Recent Developments"]
    result = bridge.decompose_topic_with_polya_and_triz(
        "Find a new proof that there are infinitely many primes",
        known,
    )
    aux = result["auxiliary_angles"]
    # The caller's subproblems must be present (re-ranked), AND new ones appended.
    titles = {a.get("subproblem", "") for a in aux}
    for k in known:
        assert k in titles
    assert any(a.get("is_new", False) for a in aux)
    # All entries carry the keys the lab-ass plugin consumes.
    for a in aux:
        assert "subproblem" in a and "is_essential" in a and "relevance_score" in a
    # The output is richer than the caller's two inputs.
    assert len(aux) > len(known)


def test_bridge_next_experiment_ranks():
    bridge = LabAssSuperSolverBridge()
    priors = {"H1": 0.5, "H2": 0.5}
    designs = [
        {
            "design_id": "exp_squid",
            "name": "SQUID",
            "description": "direct",
            "parameters": {},
            "cost": 2.0,
        },
        {
            "design_id": "exp_dc",
            "name": "DC probe",
            "description": "confounded",
            "parameters": {},
            "cost": 1.0,
        },
    ]
    likelihood = {
        "exp_squid": {
            "flux": {"H1": 0.95, "H2": 0.02},
            "no_flux": {"H1": 0.05, "H2": 0.98},
        },
        "exp_dc": {
            "drop": {"H1": 0.85, "H2": 0.85},
            "no_drop": {"H1": 0.15, "H2": 0.15},
        },
    }
    ranked = bridge.next_experiment(priors, designs, likelihood)
    assert len(ranked) == 2
    # informative SQUID beats uninformative DC probe
    assert ranked[0]["design_id"] == "exp_squid"
    assert ranked[0]["expected_information_gain"] > 0.40
    assert ranked[1]["design_id"] == "exp_dc"
    assert ranked[1]["expected_information_gain"] < 0.05


def test_score_novelty_is_not_gameable_by_random_noise():
    """score_novelty must rank genuine creative recombinations ABOVE pure random
    token-noise. Regression: the old metric was frontier novelty = 1 - max_cos to
    known, which is maximized by random gibberish (far from everything) -- any
    search maximizing it degenerated to noise. Now it multiplies frontier novelty
    by a lexical-recombination grounding (fraction of the candidate's content
    words that occur in the known domain), so pure noise shares no vocabulary and
    scores 0 while a real novel idea recombines known concepts and scores > 0."""
    import random
    import string

    bridge = LabAssSuperSolverBridge()
    known = [
        "prime numbers are infinite",
        "goldbach conjecture on even numbers",
        "twin primes remain unproven in general",
        "riemann zeta zeros on critical line",
    ]
    random.seed(7)
    noise = [
        "".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(8, 20)))
        for _ in range(8)
    ]
    real = [
        "prove twin primes are infinite by a probabilistic density transfer on the zeta zeros",
        "a sieve invariant bounding prime gaps between consecutive primes via the zeta function",
        "generalize the goldbach argument to all integers using an operator on the critical line",
    ]

    res = bridge.score_novelty(noise + real, known)
    by_cand = {c["candidate"]: c["novelty"] for c in res["candidates"]}

    # All pure-noise strings must score 0 (no lexical grounding).
    for n in noise:
        assert by_cand[n] == 0.0, f"noise {n!r} scored {by_cand[n]}"
    # Every genuine creative recombination must score above noise.
    for r in real:
        assert by_cand[r] > 0.0
    assert min(by_cand[r] for r in real) > max(by_cand[n] for n in noise)
    # The richest recombination reusing both known clusters ranks highest.
    assert by_cand[real[0]] == max(by_cand[r] for r in real)
