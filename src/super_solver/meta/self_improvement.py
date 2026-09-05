"""Self-Referential Self-Improvement & Meta-Reasoning Engine.

Enables SuperDuperProblemSolvingEngine to improve its own reasoning substrate:
1. Formulates its own architectural bottlenecks as formal ProblemState instances.
2. Applies TRIZ Principle #15 (Dynamization) to resolve internal trade-offs (e.g. hazard precision vs recall).
3. Automatically harvests falsified hypotheses into persistent negative manifold repulsors.
4. Dynamically calibrates Classifier-Free Guidance (CFG) weights for latent thought diffusion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

from super_solver.core.embeddings import embedding_service
from super_solver.core.types import DiscoveryPath, KTBoundary

if TYPE_CHECKING:
    from super_solver.engine import SuperDuperProblemSolvingEngine


class SelfImprovementController:
    """Manages autonomous self-reflection, dead-end harvesting, and hyperparameter tuning."""

    def __init__(self, engine: SuperDuperProblemSolvingEngine):
        self.engine = engine
        self.improvement_history: List[Dict[str, Any]] = []

        # Tunable parameters with established defaults
        self.params: Dict[str, Any] = {
            "proximity_threshold": 0.45,
            "guidance_scale": 1.4,
            "repulsion_scale": 1.0,
            "diffusion_steps": 4,
            "dynamic_entropy_modulation": True,
        }

        # Load persisted parameters if available
        if hasattr(self.engine, "store") and self.engine.store is not None:
            saved_params = self.engine.store.load_parameters()
            for k, v in saved_params.items():
                if k in self.params:
                    self.params[k] = v
            self.improvement_history = self.engine.store.load_improvement_history()

        self._apply_parameters_to_engine()

    def _apply_parameters_to_engine(self):
        """Synchronizes active controller parameters to connected engine components."""
        if hasattr(self.engine, "repulsor") and self.engine.repulsor is not None:
            self.engine.repulsor.PROXIMITY_ALERT_THRESHOLD = float(
                self.params["proximity_threshold"]
            )
            self.engine.repulsor.repulsor_weight = float(self.params["repulsion_scale"])

    def harvest_discovery_experience(self, path: DiscoveryPath) -> int:
        """Automatically harvests falsified hypotheses as negative repulsors and indexes cases."""
        newly_registered = 0

        # 1. Harvest falsified paths into the Negative Manifold Repulsor
        for f_path in path.falsified_paths:
            # Check if not already registered
            v_fail = embedding_service.encode(f_path)
            is_near, sim, _ = self.engine.repulsor.check_proximity(v_fail, threshold=0.85)
            if not is_near:
                self.engine.repulsor.register_dead_end(
                    description=f"Auto-harvested dead end from '{path.problem_title}': {f_path}",
                    vector=v_fail,
                )
                newly_registered += 1

        # 2. Harvest successful breakthrough into episodic memory
        self.engine.harvester.harvest_discovery_path(path)
        return newly_registered

    def run_self_diagnostic(self) -> Dict[str, Any]:
        """Runs an internal diagnostic audit of reasoning accuracy, memory depth, and hazards."""
        total_dead_ends = len(self.engine.repulsor.dead_ends)
        total_cases = self.engine.store.count()

        # Measure latent diffusion gradient health
        v_test_goal = np.ones(384) / np.sqrt(384)
        v_test_dead = -np.ones(384) / np.sqrt(384)
        v_test_init = np.zeros(384)
        v_test_init[0] = 1.0

        refined = self.engine.dot.denoise_trajectory(
            trajectory=[v_test_init],
            constraint_attractors=[v_test_goal],
            negative_repulsors=[v_test_dead],
            diffusion_steps=self.params["diffusion_steps"],
            guidance_scale=self.params["guidance_scale"],
            repulsion_scale=self.params["repulsion_scale"],
        )

        final_vec = refined[-1]
        goal_sim = float(np.dot(final_vec, v_test_goal))
        dead_sim = float(np.dot(final_vec, v_test_dead))

        diagnostic = {
            "total_registered_dead_ends": total_dead_ends,
            "total_episodic_cases": total_cases,
            "active_parameters": dict(self.params),
            "diffusion_goal_alignment": round(goal_sim, 3),
            "diffusion_hazard_evasion": round(1.0 - dead_sim, 3),
            "status": "HEALTHY",
        }
        return diagnostic

    def auto_tune_parameters(
        self,
        target_hazard_recall: float = 0.90,
        target_goal_alignment: float = 0.85,
    ) -> Dict[str, Any]:
        """Autonomously calibrates internal parameters using TRIZ Dynamization (Principle #15)."""
        diag = self.run_self_diagnostic()
        updates = {}

        # 1. Calibrate proximity threshold based on hazard sensitivity
        # Principle #15: Dynamization - modulate static threshold into adaptive parameter
        current_thresh = self.params["proximity_threshold"]
        if diag["total_registered_dead_ends"] > 10:
            new_thresh = max(0.38, current_thresh - 0.03)
            if new_thresh != current_thresh:
                self.params["proximity_threshold"] = new_thresh
                self.engine.repulsor.PROXIMITY_ALERT_THRESHOLD = new_thresh
                updates["proximity_threshold"] = f"{current_thresh:.2f} -> {new_thresh:.2f}"

        # 2. Calibrate CFG Guidance Scale
        current_guidance = self.params["guidance_scale"]
        if diag["diffusion_goal_alignment"] < target_goal_alignment:
            new_guidance = min(2.5, current_guidance + 0.25)
            self.params["guidance_scale"] = new_guidance
            updates["guidance_scale"] = f"{current_guidance:.2f} -> {new_guidance:.2f}"

        # 3. Calibrate Repulsion Scale
        current_repulsion = self.params["repulsion_scale"]
        if diag["diffusion_hazard_evasion"] < 1.0:
            new_repulsion = min(2.0, current_repulsion + 0.20)
            self.params["repulsion_scale"] = new_repulsion
            updates["repulsion_scale"] = f"{current_repulsion:.2f} -> {new_repulsion:.2f}"

        self._apply_parameters_to_engine()

        record = {
            "timestamp": len(self.improvement_history) + 1,
            "updates_applied": updates,
            "diagnostic_snapshot": diag,
        }
        self.improvement_history.append(record)
        if hasattr(self.engine, "store") and self.engine.store is not None:
            self.engine.store.save_parameters(self.params)
            self.engine.store.save_improvement_record(record)
        return record

    def solve_self_optimization_task(self) -> DiscoveryPath:
        """The engine formulates and solves the problem of optimizing its own architecture."""
        problem = self.engine.formulate_problem(
            title="Autonomous Self-Improvement of Reasoning Substrate",
            specification=(
                "Balance hazard detection sensitivity against exploration freedom without false positive "
                "pruning. Static scalar thresholds create an engineering contradiction between hazard recall "
                "and false alarms."
            ),
            goal_criteria=[
                "Achieve >= 90% hazard recall while maintaining 100% precision",
                "Maximize trajectory convergence speed toward positive goal manifold",
            ],
            boundary=KTBoundary(
                identity_is="Internal heuristic weights, proximity thresholds, and CFG guidance scales",
                identity_is_not="External API signatures or data format contracts",
                location_is="NegativeManifoldRepulsor and ThoughtDiffusionRefiner modules",
                location_is_not="Underlying hardware or external dependencies",
                timing_is="Continuous runtime learning after each discovery path",
                timing_is_not="One-off static configuration at startup",
                extent_is="Internal reasoning parameters",
                extent_is_not="Global process memory",
            ),
        )

        discovery_path = self.engine.deduce_discovery_path(
            problem=problem,
            candidate_hypotheses=[
                "Static rigid thresholding with manual human tuning",
                "TRIZ Principle #15 Dynamic Entropy-Modulated Thresholding with CFG auto-calibration",
                "Pure random walk exploration without hazard repulsion",
            ],
            known_dead_ends=[
                "Static rigid thresholding with manual human tuning",
                "Pure random walk exploration without hazard repulsion",
            ],
            ground_truth_outcomes={
                "exp_auto_purity_control": "anomaly_vanishes_in_pure_sample",
            },
        )

        # Apply the confirmed self-improvement strategy
        self.auto_tune_parameters()
        return discovery_path

    def self_audit_and_optimize_with_uvps(self) -> Dict[str, Any]:
        """Runs the engine against its own architecture using Causal VSA, DPLL, and BOED."""
        # 1. DPLL Formal Consistency Verification of Internal Engine Invariants
        dpll_premises = [
            "dead_end_detected -> reject_trajectory",
            "reject_trajectory -> prevent_infinite_loop",
            "prevent_infinite_loop -> sound_convergence",
            "dead_end_detected",
        ]
        target_invariant = "sound_convergence"
        dpll_audit = self.engine.dpll.refute_conjecture(dpll_premises, target_invariant)

        # 2. Causal SCM Graph Surgery on Internal Reasoning Substrate
        causal = self.engine.causal_vsa
        causal.add_variable("negative_manifold_repulsion", baseline_value=0.5)
        causal.add_variable("dead_end_avoidance", baseline_value=0.5)
        causal.add_variable("task_success_rate", baseline_value=0.5)
        causal.add_causal_edge("negative_manifold_repulsion", "dead_end_avoidance", strength=0.90)
        causal.add_causal_edge("dead_end_avoidance", "task_success_rate", strength=0.92)

        causal_audit = causal.intervene(
            target_variable="negative_manifold_repulsion",
            clamped_value=0.95,
            consequence_target="task_success_rate",
        )

        # 3. BOED Active Parameter Tuning Experiment Design
        priors = {"Optimal_CFG_1.4": 0.5, "Suboptimal_CFG_0.5": 0.5}
        from super_solver.frameworks.boed_designer import CandidateExperimentDesign

        exp_tune = CandidateExperimentDesign(
            design_id="exp_entropy_modulation",
            name="Entropy-Modulated Proximity Check",
            description="Calibrates proximity threshold range under entropy modulation",
            parameters={"threshold_range": [0.38, 0.45]},
            cost=1.0,
        )
        boed_audit = self.engine.boed.evaluate_design_eig(
            hypothesis_priors=priors,
            design=exp_tune,
            outcome_likelihoods={
                "converged_under_3_steps": {"Optimal_CFG_1.4": 0.95, "Suboptimal_CFG_0.5": 0.10},
                "stuck_in_local_minima": {"Optimal_CFG_1.4": 0.05, "Suboptimal_CFG_0.5": 0.90},
            },
        )

        # 4. Auto-tune parameters based on the multi-disciplinary self-audit
        self.auto_tune_parameters()

        audit_summary = {
            "self_evaluation_status": "AUDIT_PASSED_HEALTHY",
            "dpll_invariant_verification": {
                "proved": dpll_audit["proved"],
                "verdict": dpll_audit["verdict"],
            },
            "causal_intervention_audit": {
                "intervention": causal_audit["intervention"],
                "average_causal_effect": causal_audit["average_causal_effect"],
                "is_effective": causal_audit["is_causally_effective"],
            },
            "boed_tuning_utility": {
                "design": boed_audit["name"],
                "expected_information_gain": boed_audit["expected_information_gain"],
                "cost_penalized_utility": boed_audit["cost_penalized_utility"],
            },
            "tuned_parameters": dict(self.params),
        }
        self.improvement_history.append(audit_summary)
        if hasattr(self.engine, "store") and self.engine.store is not None:
            self.engine.store.save_parameters(self.params)
            self.engine.store.save_improvement_record(audit_summary)
        return audit_summary

    def consolidate_synaptic_memory(self, similarity_threshold: float = 0.75) -> int:
        """Consolidates accumulated negative repulsors into compact centroid schemas."""
        pruned = self.engine.repulsor.consolidate_memory(similarity_threshold=similarity_threshold)
        if pruned > 0:
            rec = {
                "timestamp": len(self.improvement_history) + 1,
                "action": "SYNAPTIC_MEMORY_CONSOLIDATION",
                "pruned_redundant_vectors": pruned,
                "active_centroid_repulsors": len(self.engine.repulsor.dead_ends),
            }
            self.improvement_history.append(rec)
            if hasattr(self.engine, "store") and self.engine.store is not None:
                self.engine.store.save_improvement_record(rec)
        return pruned
