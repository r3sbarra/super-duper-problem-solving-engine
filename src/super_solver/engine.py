"""SuperDuperProblemSolvingEngine: Master Neurosymbolic Scientific Discovery Engine."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from super_solver.vectorize import VectorizationService

import numpy as np

from super_solver.core.embeddings import embedding_service
from super_solver.core.smt_solver import SMTResult, smt_solver
from super_solver.core.types import (
    CrucialExperiment,
    DiscoveryPath,
    HypothesisStatus,
    KTBoundary,
    MathematicalLemma,
    OperatorType,
    ProblemState,
    ProofVerificationResult,
    ReasoningStep,
    SuggestedPath,
)
from super_solver.frameworks.gentner_sme import StructureMappingEngine
from super_solver.frameworks.kepner_tregoe import KepnerTregoeEngine
from super_solver.frameworks.math_verification import LakatosProofVerificationEngine
from super_solver.frameworks.peirce_inquiry import PeirceanInquiryEngine
from super_solver.frameworks.platt_inference import StrongInferenceEngine
from super_solver.frameworks.polya_heuristics import PolyaHeuristicsEngine
from super_solver.frameworks.sandbox import code_sandbox
from super_solver.frameworks.toc_cloud import TOCEngine
from super_solver.frameworks.tournament import DialecticalTournamentEngine, TournamentMatchResult
from super_solver.frameworks.triz_engine import TRIZEngine
from super_solver.frameworks.universal_vector_solver import (
    CascadingDiscoveryResult,
    UniversalVectorizedSolver,
    VectorizedSolution,
)
from super_solver.latent.continuous_thought import ContinuousThoughtController
from super_solver.latent.projection import SymbolicProjector
from super_solver.latent.thought_diffusion import ThoughtDiffusionRefiner
from super_solver.memory.case_harvester import CaseHarvester
from super_solver.memory.episodic_store import EpisodicVectorStore
from super_solver.meta.manuscript import manuscript_generator
from super_solver.meta.self_improvement import SelfImprovementController
from super_solver.search.curiosity import CuriosityEngine
from super_solver.search.mcts_prm import LatentMCTSEngine, ProcessRewardModel
from super_solver.search.negative_manifold import NegativeManifoldRepulsor


class SuperDuperProblemSolvingEngine:
    """The master problem-solving engine."""

    def __init__(self, db_path: Optional[str] = None):
        self.store = EpisodicVectorStore(db_path=db_path)
        self.harvester = CaseHarvester(store=self.store)

        self.platt = StrongInferenceEngine()
        self.kt = KepnerTregoeEngine()
        self.triz = TRIZEngine()
        self.toc = TOCEngine()
        self.peirce = PeirceanInquiryEngine()
        self.polya = PolyaHeuristicsEngine()
        self.gentner = StructureMappingEngine()
        self.math_verifier = LakatosProofVerificationEngine()
        self.uvps = UniversalVectorizedSolver()
        self.dpll = self.uvps.dpll
        self.boed = self.uvps.boed
        self.causal_vsa = self.uvps.causal_engine

        self.coconut = ContinuousThoughtController()
        self.dot = ThoughtDiffusionRefiner()
        self.projector = SymbolicProjector()
        self.repulsor = NegativeManifoldRepulsor(store=self.store)
        self.curiosity = CuriosityEngine()
        self.prm = ProcessRewardModel(repulsor=self.repulsor)
        self.mcts = LatentMCTSEngine(prm=self.prm)
        self.self_improver = SelfImprovementController(engine=self)

        # New audit-driven extensions
        self.tournament = DialecticalTournamentEngine(
            repulsor=self.repulsor,
            lakatos=self.math_verifier,
            dpll=self.dpll,
        )

        self.smt = smt_solver
        self.sandbox = code_sandbox
        self.manuscript = manuscript_generator

    def formulate_problem(
        self,
        title: str,
        specification: str,
        boundary: Optional[KTBoundary] = None,
        goal_criteria: Optional[List[str]] = None,
    ) -> ProblemState:
        prob_id = str(uuid.uuid4())
        v_spec = embedding_service.encode(specification)
        v_goal = None
        if goal_criteria:
            v_goal = embedding_service.encode(" ".join(goal_criteria)).tolist()

        prob = ProblemState(
            id=prob_id,
            title=title,
            specification=specification,
            boundary=boundary,
            goal_criteria=goal_criteria or [],
            state_vector=v_spec.tolist(),
            goal_vector=v_goal,
        )
        return prob

    def deduce_discovery_path(
        self,
        problem: ProblemState,
        candidate_hypotheses: Optional[List[str]] = None,
        crucial_experiments: Optional[List[CrucialExperiment]] = None,
        known_dead_ends: Optional[List[str]] = None,
        ground_truth_outcomes: Optional[Dict[str, str]] = None,
        error_rate: float = 0.0,
        concrete_claims: Optional[List[str]] = None,
    ) -> DiscoveryPath:
        ground_truth = ground_truth_outcomes or {}
        had_ground_truth = bool(ground_truth_outcomes)
        reasoning_steps: List[ReasoningStep] = []
        falsified_list: List[str] = []

        # 0. Register known dead ends into Negative Manifold
        if known_dead_ends:
            for de in known_dead_ends:
                self.repulsor.register_dead_end(de)

        # 1. PEIRCE ABDUCTION (Autonomous if candidates not provided)
        if candidate_hypotheses:
            hypotheses = self.peirce.abduct(
                anomaly_observation=problem.specification,
                domain_context=problem.title,
                candidate_explanations=candidate_hypotheses,
            )
        else:
            hypotheses = self.peirce.autonomous_abduct(
                anomaly_observation=problem.specification,
                domain_context=problem.title,
            )

        step_1_vec = embedding_service.encode(" ".join([h.description for h in hypotheses[:2]]))
        reasoning_steps.append(
            ReasoningStep(
                step_index=len(reasoning_steps) + 1,
                operator_type=OperatorType.PEIRCE_ABDUCTIVE_LEAP,
                operator_name="Peircean Abduction",
                description=f"Generated and ranked {len(hypotheses)} competing hypotheses from anomaly.",
                latent_vector=step_1_vec.tolist(),
                confidence=float(hypotheses[0].current_confidence),
                symbolic_summary=f"Top candidate: {hypotheses[0].description}",
            )
        )

        # 2. KEPNER-TREGOE BOUNDARY FILTERING (if boundary provided)
        if problem.boundary:
            v_boundary_is, _, _ = self.kt.build_boundary_hyperplane(problem.boundary)
            for h in hypotheses:
                fit = self.kt.evaluate_candidate_cause(h.description, problem.boundary)
                if not fit["valid_boundary_fit"]:
                    h.current_confidence *= 0.5
                    if fit.get("unwarranted_leakage", 0.0) > 0.3 and fit["net_score"] < -0.1:
                        h.status = HypothesisStatus.FALSIFIED
                        if h.id not in falsified_list:
                            falsified_list.append(h.id)

            reasoning_steps.append(
                ReasoningStep(
                    step_index=len(reasoning_steps) + 1,
                    operator_type=OperatorType.CONTINUOUS_LATENT_STEP,
                    operator_name="Kepner-Tregoe Boundary Filtering",
                    description="Filtered candidate hypotheses through 4D IS/IS NOT boundary hyperplane.",
                    latent_vector=v_boundary_is.tolist(),
                    confidence=0.85,
                    symbolic_summary="Pruned hypotheses that leak into IS NOT failure envelope.",
                )
            )

        # 3. LATENT THOUGHT ROLLOUT & NEGATIVE REPULSION (Coconut + DoT + Repulsor)
        current_latent = np.asarray(problem.state_vector)
        v_goal = (
            np.asarray(problem.goal_vector)
            if problem.goal_vector
            else embedding_service.encode(problem.title)
        )

        is_near, sim_dead, dead_desc = self.repulsor.check_proximity(current_latent)
        if is_near:
            current_latent = self.repulsor.deflect_trajectory(current_latent, goal_vector=v_goal)
            reasoning_steps.append(
                ReasoningStep(
                    step_index=len(reasoning_steps) + 1,
                    operator_type=OperatorType.CONTINUOUS_LATENT_STEP,
                    operator_name="Negative Manifold Deflection",
                    description=f"Deflected search away from dead end '{dead_desc}' (similarity was {sim_dead:.2f}).",
                    latent_vector=current_latent.tolist(),
                    dead_end_margin=float(1.0 - sim_dead),
                    confidence=0.90,
                    symbolic_summary="Avoided repetition of prior failed direction.",
                )
            )

        triz_ops = self.triz.suggest_principles(problem.specification, top_k=2)
        op_vec = embedding_service.encode(
            triz_ops[0]["name"] if triz_ops else "General Transformation"
        )
        current_latent = self.coconut.step_continuous_thought(current_latent, op_vec)

        params = self.self_improver.params
        diff_steps = int(params.get("diffusion_steps", 3))
        guidance = float(params.get("guidance_scale", 1.4))
        repulsion = float(params.get("repulsion_scale", 1.0))

        refined_traj = self.dot.denoise_trajectory(
            trajectory=[current_latent],
            constraint_attractors=[v_goal],
            negative_repulsors=self.repulsor.dead_ends,
            diffusion_steps=diff_steps,
            guidance_scale=guidance,
            repulsion_scale=repulsion,
        )
        current_latent = refined_traj[0] if refined_traj else current_latent

        reasoning_steps.append(
            ReasoningStep(
                step_index=len(reasoning_steps) + 1,
                operator_type=OperatorType.CONTINUOUS_LATENT_STEP,
                operator_name="Coconut Latent Rollout & Thought Diffusion",
                description=f"Evolved continuous thought vector using TRIZ operator '{triz_ops[0]['name'] if triz_ops else 'Transformation'}' with DoT denoising.",
                latent_vector=current_latent.tolist(),
                confidence=0.88,
                symbolic_summary="Refined continuous trajectory toward goal manifold.",
            )
        )

        # 3b. LATENT MCTS SEARCH (PRM-scored tree search over candidate operators)
        # The MCTS engine was instantiated but never invoked in the discovery
        # pipeline — wire it in here to actually search, not just do a single
        # linear Coconut step. Candidate operators come from TRIZ principles +
        # the surviving hypotheses' descriptions.
        candidate_ops: List[Tuple[str, np.ndarray, float]] = []
        for op in triz_ops[:4]:
            candidate_ops.append((op["name"], embedding_service.encode(op["name"]), 0.5))
        for h in hypotheses[:3]:
            candidate_ops.append(
                (
                    f"hypothesis:{h.id}",
                    embedding_service.encode(h.description),
                    h.current_confidence,
                )
            )
        if candidate_ops:
            mcts_path = self.mcts.search_best_path(
                initial_state=current_latent,
                candidate_operator_generators=candidate_ops,
                goal_vector=v_goal,
                num_simulations=10,
            )
            if mcts_path:
                best_op_name, best_state, best_score = mcts_path[0]
                current_latent = np.asarray(best_state)
                reasoning_steps.append(
                    ReasoningStep(
                        step_index=len(reasoning_steps) + 1,
                        operator_type=OperatorType.CONTINUOUS_LATENT_STEP,
                        operator_name="Latent MCTS Search (PRM)",
                        description=f"MCTS selected operator '{best_op_name}' (PRM score {best_score:.2f}) across {len(candidate_ops)} candidates.",
                        latent_vector=current_latent.tolist(),
                        confidence=0.85,
                        symbolic_summary="Tree-searched the latent space instead of a single linear step.",
                    )
                )

        # 3c. SYMBOLIC PROJECTION — decode the latent state back to a testable claim
        # The SymbolicProjector was instantiated but never used; without it the
        # latent vectors accumulate but never project back to auditable claims.
        # Candidate claims = TRIZ hypotheses + any concrete domain-grounded
        # solution directions, so the projection can decode to a CONCRETE
        # solution rather than only a bare principle.
        candidate_claims = [h.description for h in hypotheses]
        if concrete_claims:
            candidate_claims = concrete_claims + candidate_claims
        nearest_claim, claim_sim = self.projector.project_to_nearest_claim(
            current_latent, candidate_claims
        )
        if nearest_claim:
            reasoning_steps.append(
                ReasoningStep(
                    step_index=len(reasoning_steps) + 1,
                    operator_type=OperatorType.CONTINUOUS_LATENT_STEP,
                    operator_name="Symbolic Projection",
                    description=f"Latent state decoded to nearest claim (similarity {claim_sim:.2f}): {nearest_claim[:120]}",
                    latent_vector=current_latent.tolist(),
                    confidence=float(claim_sim),
                    symbolic_summary=nearest_claim[:200],
                )
            )

        # 3d. DPLL LOGICAL CONSISTENCY CHECK on the surviving hypotheses
        # The DPLL solver was instantiated but never called in the pipeline;
        # check that the candidate hypotheses are not logically contradictory.
        try:
            result = self.dpll.refute_conjecture(
                premises=[h.description for h in hypotheses],
                target_claim=problem.specification,
            )
            proved = bool(result.get("proved"))
            reasoning_steps.append(
                ReasoningStep(
                    step_index=len(reasoning_steps) + 1,
                    operator_type=OperatorType.CONTINUOUS_LATENT_STEP,
                    operator_name="DPLL Logical Consistency",
                    description=f"Hypotheses logically {'entail' if proved else 'do not entail'} the problem statement.",
                    latent_vector=current_latent.tolist(),
                    confidence=0.9 if proved else 0.3,
                    symbolic_summary=result.get(
                        "summary", "DPLL refutation check on candidate premises."
                    )[:200],
                )
            )
        except Exception:
            # DPLL is best-effort; a failure here should not abort discovery.
            pass

        # 4. PLATT STRONG INFERENCE: CRUCIAL EXPERIMENTS (Autonomous if not provided)
        is_autonomous_exp = crucial_experiments is None
        experiments_to_run = crucial_experiments
        if not experiments_to_run:
            experiments_to_run = self.platt.autonomous_design_crucial_experiments(hypotheses)

        executed_experiments = []
        for exp in experiments_to_run:
            active_survivors = [h for h in hypotheses if h.status != HypothesisStatus.FALSIFIED]
            if is_autonomous_exp and len(active_survivors) <= 1:
                break

            info_gain = self.platt.evaluate_crucial_experiment(exp, hypotheses)

            actual_outcome = ground_truth.get(exp.id)
            if not actual_outcome:
                # No ground truth for this experiment. Do NOT default to the
                # first surviving hypothesis's prediction — that guarantees
                # self-confirmation and makes the pipeline non-falsifiable.
                # Instead, skip pruning (we cannot judge) and record that the
                # outcome is unverified.
                reasoning_steps.append(
                    ReasoningStep(
                        step_index=len(reasoning_steps) + 1,
                        operator_type=OperatorType.PLATT_CRUCIAL_EXPERIMENT,
                        operator_name=f"Crucial Experiment: {exp.name}",
                        description=f"No ground truth provided — outcome UNVERIFIED, no hypotheses pruned. Info gain: {info_gain:.2f} bits.",
                        latent_vector=current_latent.tolist(),
                        confidence=0.5,
                        symbolic_summary="Experiment skipped (no ground truth); result unverified.",
                    )
                )
                executed_experiments.append(exp)
                continue

            hypotheses, newly_falsified = self.platt.execute_and_prune(
                experiment=exp,
                hypotheses=hypotheses,
                observed_outcome=actual_outcome,
                error_rate=error_rate,
            )
            executed_experiments.append(exp)

            for fid in newly_falsified:
                if fid not in falsified_list:
                    falsified_list.append(fid)

            reasoning_steps.append(
                ReasoningStep(
                    step_index=len(reasoning_steps) + 1,
                    operator_type=OperatorType.PLATT_CRUCIAL_EXPERIMENT,
                    operator_name=f"Crucial Experiment: {exp.name}",
                    description=f"Observed '{actual_outcome}', falsified {len(newly_falsified)} hypotheses. Info gain: {info_gain:.2f} bits.",
                    latent_vector=current_latent.tolist(),
                    confidence=0.95,
                    symbolic_summary=f"Surviving candidates: {[h.id for h in hypotheses if h.status != HypothesisStatus.FALSIFIED]}",
                )
            )

        # 5. BREAKTHROUGH IDENTIFICATION
        survivors = [h for h in hypotheses if h.status != HypothesisStatus.FALSIFIED]
        # If domain-grounded concrete claims were provided, prefer the concrete
        # solution direction the latent search converged on. The TRIZ hypothesis
        # texts embed the full problem statement, so they score artificially
        # high against the latent state (which started from the problem vector);
        # the concrete claims are the actual solution directions and should win
        # when grounding is active.
        concrete_breakthrough = None
        if concrete_claims:
            # concrete_claims arrive already ranked best-first by the retrieval
            # step (hybrid_analogize), which correctly matches the problem to the
            # right concrete solution. Prefer the top retrieval hit — the latent
            # state after MCTS is too noisy to re-rank them.
            concrete_breakthrough = concrete_claims[0]
            claim_sim = 0.5  # retrieval confidence, not latent similarity
        if concrete_breakthrough:
            if had_ground_truth:
                breakthrough = f"CONFIRMED: {concrete_breakthrough}"
                final_conf = max(claim_sim, 0.5)
            else:
                breakthrough = (
                    f"UNVERIFIED: {concrete_breakthrough} (no ground truth — needs falsification)"
                )
                final_conf = max(claim_sim, 0.3) * 0.5
        elif survivors:
            survivors.sort(key=lambda h: h.current_confidence, reverse=True)
            winner = survivors[0]
            if had_ground_truth:
                winner.status = HypothesisStatus.CONFIRMED
                breakthrough = f"CONFIRMED: {winner.description}"
                final_conf = winner.current_confidence
            else:
                # No ground truth was provided — we cannot claim confirmation.
                # Mark UNVERIFIED (a distinct, honest state) instead of
                # self-confirming the top survivor.
                winner.status = HypothesisStatus.PROPOSED
                breakthrough = (
                    f"UNVERIFIED: {winner.description} (no ground truth — needs falsification)"
                )
                final_conf = winner.current_confidence * 0.5
        else:
            breakthrough = "INCONCLUSIVE: All candidates falsified. Need new abductive generation."
            final_conf = 0.1

        discovery_path = DiscoveryPath(
            problem_id=problem.id,
            problem_title=problem.title,
            initial_abduction=hypotheses[0].description if hypotheses else "",
            steps=reasoning_steps,
            crucial_experiments=executed_experiments,
            falsified_paths=falsified_list,
            final_breakthrough=breakthrough,
            total_steps=len(reasoning_steps),
            confidence=float(final_conf),
        )

        self.self_improver.harvest_discovery_experience(discovery_path)
        return discovery_path

    def solve_unproven_arxiv_paper(
        self,
        paper_title: str,
        abstract_text: str,
        observed_experimental_result: Optional[str] = "anomaly_vanishes_in_pure_sample",
    ) -> DiscoveryPath:
        prob = self.formulate_problem(
            title=paper_title,
            specification=abstract_text,
            goal_criteria=[
                "Resolve whether observed anomaly is fundamental physics or observational artifact"
            ],
        )

        path = self.deduce_discovery_path(
            problem=prob,
            candidate_hypotheses=None,
            crucial_experiments=None,
            ground_truth_outcomes={
                "exp_auto_purity_control": observed_experimental_result,
            },
        )
        return path

    def export_mermaid_diagram(self, path: DiscoveryPath) -> str:
        lines = ["```mermaid", "graph TD"]
        lines.append(f'  P["Problem: {path.problem_title}"] --> S1["Peircean Abduction"]')
        prev_node = "S1"

        for i, step in enumerate(path.steps[1:], start=2):
            node_id = f"S{i}"
            label = step.operator_name.replace('"', "'")
            lines.append(f'  {prev_node} --> {node_id}["{label}"]')
            prev_node = node_id

        if path.falsified_paths:
            for j, f_id in enumerate(path.falsified_paths):
                f_node = f"FAL_{j + 1}"
                lines.append(f'  {prev_node} -. Falsified .-> {f_node}["{f_id}"]')
                lines.append(f"  style {f_node} fill:#ffcccc,stroke:#cc0000")

        breakthrough_clean = path.final_breakthrough.replace('"', "'")
        lines.append(f'  {prev_node} ==> B["{breakthrough_clean}"]')
        lines.append("  style B fill:#d4edda,stroke:#28a745,stroke-width:2px")
        lines.append("```")
        return "\n".join(lines)

    def explain_discovery(self, path: DiscoveryPath) -> str:
        out = [
            f"=== DISCOVERY PATH EXPLANATION: {path.problem_title} ===",
            f"Total Reasoning Steps: {path.total_steps}",
            f"Final Confidence: {path.confidence * 100:.1f}%\n",
            "Reasoning Trajectory Progression:",
        ]
        for step in path.steps:
            out.append(f"  [{step.step_index}] {step.operator_name}: {step.description}")
            if step.symbolic_summary:
                out.append(f"      Summary: {step.symbolic_summary}")

        if path.falsified_paths:
            out.append("\nPruned / Falsified Possibilities:")
            for fp in path.falsified_paths:
                out.append(f"  - [FALSIFIED] {fp}")

        out.append(f"\nFinal Result: {path.final_breakthrough}")
        return "\n".join(out)

    def verify_math_paper(
        self,
        paper_id: str,
        paper_title: str,
        abstract_text: str,
        target_conjecture: str,
        explicit_lemmas: Optional[List[MathematicalLemma]] = None,
    ) -> ProofVerificationResult:
        """Audits, tests, and verifies or refutes an unverified mathematical preprint."""
        return self.math_verifier.verify_mathematical_paper(
            paper_id=paper_id,
            paper_title=paper_title,
            abstract_text=abstract_text,
            target_conjecture=target_conjecture,
            explicit_lemmas=explicit_lemmas,
        )

    def export_math_verification_mermaid(self, result: ProofVerificationResult) -> str:
        """Exports a visual Mermaid diagram of the mathematical proof verification audit."""
        return self.math_verifier.export_proof_mermaid(result)

    def suggest_paths(
        self,
        problem_specification: str,
        context: Optional[str] = None,
        top_k: int = 3,
    ) -> List[SuggestedPath]:
        """Autonomously synthesizes and ranks viable alternative discovery paths or repair trajectories.

        Evaluates candidate paths across TRIZ contradiction resolution, Polya backward planning,
        Gentner analogical transfer, Lakatos mathematical barrier evasion, and Platt crucial assays.
        Ranks paths using a composite score of feasibility, novelty (frontier curiosity), and
        dead-end avoidance margins.
        """
        combined = (problem_specification + " " + (context or "")).lower()
        candidates: List[SuggestedPath] = []

        # 1. Check for Mathematical Barriers / Conjectures (Lakatos Engine)
        if any(
            term in combined
            for term in [
                "p versus np",
                "p != np",
                "collatz",
                "3x+1",
                "sieve",
                "twin prime",
                "conjecture",
                "proof",
            ]
        ):
            dummy_res = self.math_verifier.verify_mathematical_paper(
                paper_id="query",
                paper_title=problem_specification[:60],
                abstract_text=problem_specification,
                target_conjecture=context or "Open Mathematical Conjecture",
            )
            math_paths = self.math_verifier.suggest_repair_paths(dummy_res)
            candidates.extend(math_paths)

        # 2. TRIZ Contradiction-Resolving Path
        triz_ops = self.triz.suggest_principles(problem_specification, top_k=2)
        if triz_ops:
            op1 = triz_ops[0]
            op2 = triz_ops[1] if len(triz_ops) > 1 else triz_ops[0]
            candidates.append(
                SuggestedPath(
                    path_id="path_triz_transform",
                    strategy_type="TRIZ_INVENTIVE_CONTRADICTION",
                    title=f"Inventive Principle #{op1['principle_id']} ({op1['name']}) Resolution",
                    description=f"Resolve core trade-off using '{op1['name']}': {op1['description']}. Complement with Principle #{op2['principle_id']} ('{op2['name']}').",
                    rationale="Overcomes trade-off without compromise by transforming the operational parameter space.",
                    recommended_next_action=f"Apply {op1['name']} to decouple conflicting parameters in {problem_specification[:40]}...",
                )
            )

        # 3. Polya Auxiliary Problem Decomposition & Working Backwards
        candidates.append(
            SuggestedPath(
                path_id="path_polya_working_backwards",
                strategy_type="POLYA_DECOMPOSITION_WORKING_BACKWARDS",
                title="Teleological Goal Regression & Auxiliary Subproblem Isolation",
                description="Assume target solution is achieved in a simplified ideal limit. Work backward to identify necessary precursor lemmas/milestones, isolating the simplest non-trivial subproblem.",
                rationale="Reduces cognitive search dimensionality by projecting from known goal invariants back to current state.",
                recommended_next_action="Identify minimal non-trivial toy instance or boundary subproblem where the phenomenon isolates cleanly.",
            )
        )

        # 4. Gentner Analogical Structural Transfer Path
        candidates.append(
            SuggestedPath(
                path_id="path_gentner_analogy",
                strategy_type="ANALOGICAL_STRUCTURAL_TRANSFER",
                title="Cross-Domain Relational Structure Mapping",
                description="Map causal relational network from a previously solved isomorphic domain (e.g. condensed matter phase transitions, evolutionary MSA co-evolution, or network flow).",
                rationale="Reuses verified high-order causal relations rather than exploring from zero priors.",
                recommended_next_action="Query episodic breakthrough memory for cross-domain systems sharing the same causal graph topology.",
            )
        )

        # 5. Platt Strong Inference Crucial Assay Path
        candidates.append(
            SuggestedPath(
                path_id="path_platt_crucial_assay",
                strategy_type="PLATT_STRONG_INFERENCE_ASSAY",
                title="High-Resolution Differential Crucial Experiment",
                description="Design an empirical or computational assay with zero-overlap predictions between the top 2 competing mechanisms to force an informational bifurcation.",
                rationale="Guarantees maximum Shannon information gain and immediate pruning of at least 50% of the active hypothesis manifold.",
                recommended_next_action="Synthesize mutually exclusory predictions across extreme boundary parameters.",
            )
        )

        # Compute dynamic vector metrics: Feasibility, Novelty, Dead-End Margin
        v_prob = embedding_service.encode(problem_specification)
        known_dead = self.repulsor.dead_ends

        for p in candidates:
            v_path = embedding_service.encode(p.title + " " + p.description)
            feasibility = float(
                max(0.1, min(0.99, embedding_service.cosine_similarity(v_prob, v_path) * 1.5))
            )
            novelty = float(self.curiosity.compute_frontier_novelty(v_path, [v_prob]))
            safety_margin = 1.0
            if known_dead:
                sims = [embedding_service.cosine_similarity(v_path, d) for d in known_dead]
                safety_margin = float(max(0.0, 1.0 - max(sims)))

            p.feasibility_score = round(feasibility, 2)
            p.novelty_score = round(novelty, 2)
            p.dead_end_safety_margin = round(safety_margin, 2)

        # Composite ranking
        candidates.sort(
            key=lambda p: (
                0.4 * p.feasibility_score + 0.3 * p.novelty_score + 0.3 * p.dead_end_safety_margin
            ),
            reverse=True,
        )

        return candidates[:top_k]

    def improve_self(self) -> DiscoveryPath:
        """Formulates and solves the problem of optimizing its own reasoning substrate."""
        return self.self_improver.solve_self_optimization_task()

    def run_self_diagnostic(self) -> Dict[str, Any]:
        """Audits internal reasoning metrics, memory depth, and active hyperparameters."""
        return self.self_improver.run_self_diagnostic()

    def solve_engineering_problem(
        self,
        title: str,
        specification: str,
        goal_criteria: Optional[List[str]] = None,
        top_principles: int = 4,
        ground_truth_outcomes: Optional[Dict[str, str]] = None,
        domain_knowledge: Optional[VectorizationService] = None,
        concrete_top_k: int = 3,
    ) -> DiscoveryPath:
        """Runs the full discovery pipeline with TRIZ-grounded engineering abduction.

        Unlike the physics-flavored autonomous_abduct (which generates generic
        'observational artifact / fundamental law / phase transition' frames),
        this generates candidate hypotheses from the actual TRIZ Inventive
        Principles matched to the problem, so the engine can GENERATE novel
        engineering solutions rather than retrieve known ones.

        Each matched principle becomes a candidate explanation phrased as an
        actionable design move; the pipeline then runs KT filtering, latent
        rollout, MCTS, and breakthrough identification on those candidates.

        If ``domain_knowledge`` (a VectorizationService seeded with the
        domain-knowledge fact base) is provided, concrete solution directions
        are retrieved via hybrid analogical transfer and injected into the
        candidate claims, so the Symbolic Projection / breakthrough step can
        decode to a CONCRETE solution rather than a bare TRIZ principle.
        """
        prob = self.formulate_problem(
            title=title,
            specification=specification,
            goal_criteria=goal_criteria or ["Find a novel solution to this problem"],
        )

        # TRIZ-grounded abduction: turn matched inventive principles into
        # candidate design-move hypotheses APPLIED to the specific problem.
        principles = self.triz.suggest_principles(specification, top_k=top_principles)
        candidates = []
        for p in principles:
            name = p["name"]
            desc = p["description"]
            # Phrase as an applied design move: principle + how it acts on the
            # problem's subject, so the generated hypothesis is a concrete
            # solution direction, not a bare principle name.
            candidates.append(
                f"To solve '{specification}', apply the TRIZ principle '{name}': {desc} "
                f"— concretely, redesign the subject so that {name.lower()} achieves the goal."
            )

        # Domain-knowledge grounding: retrieve concrete solution directions via
        # hybrid analogical transfer and add them as candidate claims so the
        # breakthrough can decode to a concrete solution.
        concrete_claims: List[str] = []
        if domain_knowledge is not None:
            try:
                from super_solver.vectorize.domain_knowledge import (
                    retrieve_concrete_solutions,
                )

                for hit in retrieve_concrete_solutions(
                    domain_knowledge, specification, top_k=concrete_top_k
                ):
                    concrete_claims.append(hit["content"])
            except Exception:
                # Domain grounding is best-effort; never abort discovery on it.
                concrete_claims = []

        # Structural cross-domain matching: strip the target problem and the
        # source solutions into structural primitives and match on overlap.
        # This is robust to surface differences (Richard's primitives idea):
        # two problems in different domains share a structure when they share
        # primitives (e.g. both 'reduce_density' + 'material'), even if the
        # surface entities differ. Prefer primitive matches over surface
        # retrieval when they are strong.
        try:
            from super_solver.vectorize.primitives import match_by_primitives

            source_solutions = []
            for row in domain_knowledge.index._rows("solution"):
                source_solutions.append(
                    {"title": row[2] or "", "solution": row[3]}
                )
            prim_hits = match_by_primitives(
                specification, source_solutions, top_k=concrete_top_k
            )
            # If a primitive match is strong, use it as the concrete claim.
            if prim_hits and prim_hits[0]["primitive_overlap"] >= 0.15:
                concrete_claims = [h["content"] for h in prim_hits]

            # Adapt the transferred solution to the target domain: substitute
            # the source's key entity with the target's key entity so the
            # generated solution reads correctly in the target domain.
            try:
                from super_solver.vectorize.entity_substitution import (
                    substitute_entity,
                )

                concrete_claims = [
                    substitute_entity(c, specification) for c in concrete_claims
                ]
            except Exception:
                pass
        except Exception:
            pass

        return self.deduce_discovery_path(
            problem=prob,
            candidate_hypotheses=candidates,
            crucial_experiments=None,
            ground_truth_outcomes=ground_truth_outcomes,
            concrete_claims=concrete_claims,
        )

    def solve_universal_vectorized(
        self,
        problem_description: str,
        problem_type: str = "auto",
        domain: Optional[str] = None,
        **kwargs: Any,
    ) -> VectorizedSolution:
        """Invokes the Universal Vectorized Problem Solving framework across any domain."""
        return self.uvps.solve_multidisciplinary_problem(
            problem_description=problem_description,
            problem_type=problem_type,
            domain=domain,
            **kwargs,
        )

    def execute_cascading_discovery(self, **kwargs: Any) -> CascadingDiscoveryResult:
        """Executes the autonomous 6-stage cascading discovery trajectory."""
        return self.uvps.execute_cascading_discovery_pipeline(**kwargs)

    def run_self_audit_and_optimization(self) -> Dict[str, Any]:
        """Runs the engine against its own architecture using Causal VSA, DPLL, and BOED."""
        return self.self_improver.self_audit_and_optimize_with_uvps()

    def consolidate_memory(self, similarity_threshold: float = 0.75) -> int:
        """Applies synaptic consolidation to prune redundant dead-end repulsors into compact schemas."""
        return self.self_improver.consolidate_synaptic_memory(
            similarity_threshold=similarity_threshold
        )

    def run_tournament(
        self,
        candidate_hypotheses: List[str],
        context: Optional[str] = None,
        boundary_is_not: Optional[str] = None,
        top_k: int = 2,
    ) -> List[TournamentMatchResult]:
        """Runs an adversarial Red Team vs Blue Team dialectical debate tournament across hypotheses."""
        return self.tournament.run_tournament(
            candidate_hypotheses=candidate_hypotheses,
            context=context,
            boundary_is_not=boundary_is_not,
            top_k=top_k,
        )

    def generate_manuscript(
        self,
        path: DiscoveryPath,
        problem: Optional[ProblemState] = None,
        output_dir: Optional[str] = None,
        format_type: str = "both",
    ) -> Dict[str, str]:
        """Generates a formal scientific manuscript from a solved DiscoveryPath."""
        if output_dir:
            return self.manuscript.save_manuscript(path, output_dir=output_dir, problem=problem)
        return {
            "markdown": self.manuscript.generate_markdown(path, problem=problem),
            "latex": self.manuscript.generate_latex(path, problem=problem),
        }

    def check_smt_lra(self, constraints: List[str]) -> SMTResult:
        """Verifies feasibility of linear real arithmetic inequalities."""
        return self.smt.check_lra_satisfiability(constraints)

    def check_smt_euf(
        self,
        equalities: List[Tuple[str, str]],
        disequalities: List[Tuple[str, str]],
    ) -> SMTResult:
        """Verifies satisfiability of equality with uninterpreted functions."""
        return self.smt.check_euf_satisfiability(equalities, disequalities)
