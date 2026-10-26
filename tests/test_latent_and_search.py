"""Tests for continuous latent reasoning, thought diffusion, and negative manifold search."""

import numpy as np

from super_solver.core.embeddings import embedding_service
from super_solver.latent.continuous_thought import ContinuousThoughtController
from super_solver.latent.thought_diffusion import ThoughtDiffusionRefiner
from super_solver.search.mcts_prm import LatentMCTSEngine, ProcessRewardModel
from super_solver.search.negative_manifold import NegativeManifoldRepulsor


def test_coconut_continuous_latent_rollout_and_bfs():
    ctrl = ContinuousThoughtController(dim=384)

    s0 = ctrl.initialize_latent_thought("Understand quantum coherence in warm biological systems")
    op_a = embedding_service.encode("Investigate radical pair mechanism in cryptochrome")
    op_b = embedding_service.encode("Examine classical thermal noise dissipation")

    s1 = ctrl.step_continuous_thought(s0, op_a)
    assert s1.shape == (384,)
    assert abs(np.linalg.norm(s1) - 1.0) < 1e-4

    operators = [("RadicalPair", op_a), ("ThermalNoise", op_b)]
    goal = embedding_service.encode("Avian magnetoreception biological sensor")

    trajectories = ctrl.rollout_latent_bfs(
        s0, candidate_operators=operators, depth=2, beam_width=2, goal_vector=goal
    )
    assert len(trajectories) == 2
    assert len(trajectories[0]) == 2


def test_thought_diffusion_refinement():
    refiner = ThoughtDiffusionRefiner(dim=384)

    t0 = np.random.normal(0, 1.0, size=384)
    t0 = t0 / np.linalg.norm(t0)
    t1 = np.random.normal(0, 1.0, size=384)
    t1 = t1 / np.linalg.norm(t1)

    target_attractor = embedding_service.encode("High thermal stability target")
    initial_sim = embedding_service.cosine_similarity(t1, target_attractor)

    refined = refiner.denoise_trajectory(
        trajectory=[t0, t1],
        constraint_attractors=[target_attractor],
        diffusion_steps=5,
        step_size=0.3,
    )

    refined_sim = embedding_service.cosine_similarity(refined[1], target_attractor)
    assert refined_sim > initial_sim


def test_negative_manifold_repulsion():
    repulsor = NegativeManifoldRepulsor(repulsor_weight=0.8)

    dead_end_desc = "Classical harmonic oscillator approximation without electronic polarization"
    repulsor.register_dead_end(dead_end_desc)

    # Candidate step directly reiterating this failed direction
    toxic_candidate = embedding_service.encode(
        "Run classical harmonic oscillator approximation without electronic polarization"
    )
    is_near, sim_dead, desc = repulsor.check_proximity(toxic_candidate)
    assert is_near is True
    assert sim_dead > 0.60

    goal = embedding_service.encode("Ab initio DFT quantum electronic polarization")
    deflected = repulsor.deflect_trajectory(toxic_candidate, goal_vector=goal)

    new_is_near, new_sim, _ = repulsor.check_proximity(deflected)
    assert new_sim < sim_dead


def test_latent_mcts_with_prm():
    repulsor = NegativeManifoldRepulsor()
    repulsor.register_dead_end("Toxic direction of infinite memory recursion")

    prm = ProcessRewardModel(repulsor=repulsor)
    mcts = LatentMCTSEngine(prm=prm)

    s0 = embedding_service.encode("Start root optimization")
    goal = embedding_service.encode("Global minimum energy state")

    good_op = embedding_service.encode("Adaptive gradient step toward global minimum")
    bad_op = embedding_service.encode("Toxic direction of infinite memory recursion")

    candidates = [
        ("GoodStep", good_op, 0.7),
        ("BadStep", bad_op, 0.1),
    ]

    best_path = mcts.search_best_path(s0, candidates, goal_vector=goal, num_simulations=10)
    assert len(best_path) >= 1
    assert best_path[0][0] == "GoodStep"


def test_prm_reward_differentiates_dead_end_states():
    """The PRM step reward must let near-dead-end states score NEGATIVELY so MCTS
    UCB1 can avoid them. Regression: the old clamp to [0.01, 1.0] floor-clamped
    every dead-end-adjacent state to 0.01, flattening the dead-end signal, and the
    dead-end weight was 0.6 in code vs 0.5 in the docstring."""
    repulsor = NegativeManifoldRepulsor()
    repulsor.register_dead_end("prove the riemann hypothesis in one step")
    prm = ProcessRewardModel(repulsor=repulsor)

    goal = embedding_service.encode("solve the hard theorem")
    far = embedding_service.encode("refactor a database query")
    near_de = embedding_service.encode("prove the riemann hypothesis")
    near_op = embedding_service.encode("prove the riemann hypothesis in one step")

    r_far = prm.score_transition(
        embedding_service.encode("cache a list of results"), far, goal
    )
    r_near = prm.score_transition(near_de, near_op, goal)

    # Near-dead-end must score strictly below far-from-dead-end, ideally negative.
    assert r_near < r_far
    assert r_near <= 0.0
    # Reward stays within [-1, 1].
    for r in (r_far, r_near):
        assert -1.0 <= r <= 1.0


def test_mcts_path_is_independent_of_operator_list_order():
    """Regression: MCTS used to expand operators in list order.

    ``expand_one_child`` took the *first untried* operator rather than the
    best-scoring one, and the extracted path broke visit-ties by list position,
    so the PRM reward never decided the trajectory. Measured on a synthetic
    10-operator task (2 goal-directed, 8 noise) with identical rewards, the
    reported good-operator fraction was 1.000 when the good operators were
    listed first and 0.000 when they were listed last -- i.e. the search was
    reporting list position, not search quality. A reward-guided search must
    return the same trajectory whatever order the operators arrive in.
    """
    import numpy as np


    dim = 32
    rng = np.random.default_rng(0)
    goal = rng.standard_normal(dim).astype(np.float32)
    goal /= np.linalg.norm(goal)
    start = -goal.copy()

    good = []
    for i in range(2):
        v = goal + 0.15 * rng.standard_normal(dim).astype(np.float32)
        good.append((f"good_{i}", (v / np.linalg.norm(v)).astype(np.float32), 0.9))
    noise = []
    for i in range(8):
        v = rng.standard_normal(dim).astype(np.float32)
        noise.append((f"noise_{i}", (v / np.linalg.norm(v)).astype(np.float32), 0.3))

    def run(ops):
        mcts = LatentMCTSEngine(prm=ProcessRewardModel(repulsor=NegativeManifoldRepulsor()))
        path = mcts.search_best_path(start, ops, goal, num_simulations=15)
        assert path
        return sum(1 for name, _, _ in path if name.startswith("good")) / len(path)

    first = run(good + noise)
    last = run(noise + good)
    assert first == last, f"search depends on list order: first={first}, last={last}"
    assert first > 0.5, f"search does not select the high-reward operators: {first}"


def test_mcts_prefers_the_highest_reward_operator():
    """The reward is the deciding signal, whatever the input order."""
    import numpy as np

    dim = 16
    rng = np.random.default_rng(1)
    start = rng.standard_normal(dim).astype(np.float32)
    start /= np.linalg.norm(start)
    goal = rng.standard_normal(dim).astype(np.float32)
    goal /= np.linalg.norm(goal)

    def vec():
        v = rng.standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    ops = [("LOW", vec(), 0.0), ("HIGH", vec(), 0.95), ("MID", vec(), 0.5)]

    for order in (ops, list(reversed(ops))):
        mcts = LatentMCTSEngine(prm=ProcessRewardModel(repulsor=NegativeManifoldRepulsor()))
        path = mcts.search_best_path(start, order, goal, num_simulations=6)
        assert path
        assert path[0][0] == "HIGH", f"picked {path[0][0]} for order {[o[0] for o in order]}"
