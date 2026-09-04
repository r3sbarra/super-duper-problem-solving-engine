"""John R. Platt's Strong Inference Framework (Science, 1964).

Systematically maintains an ensemble of alternative hypotheses, autonomously designs
crucial discriminating experiments, and prunes possibilities with both strict Popperian
falsification and noise-tolerant Bayesian updating.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from super_solver.core.types import CrucialExperiment, Hypothesis, HypothesisStatus


class StrongInferenceEngine:
    """Platt's Strong Inference Engine: Alternative hypotheses + Exclusory Pruning."""

    def __init__(self):
        pass

    def compute_entropy(self, hypotheses: List[Hypothesis]) -> float:
        """Computes the Shannon entropy of current hypothesis probability distribution."""
        active = [h for h in hypotheses if h.status != HypothesisStatus.FALSIFIED]
        if not active:
            return 0.0
        total_conf = sum(h.current_confidence for h in active)
        if total_conf <= 0:
            return 0.0

        probs = [h.current_confidence / total_conf for h in active]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        return float(entropy)

    def autonomous_design_crucial_experiments(
        self,
        hypotheses: List[Hypothesis],
    ) -> List[CrucialExperiment]:
        """Autonomously designs crucial discriminating experiments tailored to partition
        the active hypotheses without requiring human pre-specification.
        """
        active = [h for h in hypotheses if h.status != HypothesisStatus.FALSIFIED]
        if len(active) <= 1:
            return []

        experiments = []

        # 1. High-Resolution Purity & Selection Filter Test (Partitions Artifacts vs Invariant Laws)
        predictions_exp1 = {}
        for h in active:
            arch = h.metadata.get("archetype", "")
            if "artifact" in arch:
                predictions_exp1[h.id] = "anomaly_vanishes_in_pure_sample"
            else:
                predictions_exp1[h.id] = "anomaly_persists_in_pure_sample"

        # Only add if it creates a non-trivial partition (different outcomes)
        if len(set(predictions_exp1.values())) > 1:
            experiments.append(CrucialExperiment(
                id="exp_auto_purity_control",
                name="High-Resolution Purity & Systematic Filtering Assay",
                description="Apply ultra-strict kinematic or compositional filtering to isolate pure unpolluted sample subset.",
                target_hypotheses=list(predictions_exp1.keys()),
                exclusory_predictions=predictions_exp1,
            ))

        # 2. External Field / Environmental Boundary Modulation Test
        predictions_exp2 = {}
        for h in active:
            arch = h.metadata.get("archetype", "")
            if "environmental" in arch:
                predictions_exp2[h.id] = "modulated_by_external_field"
            elif "fundamental" in arch:
                predictions_exp2[h.id] = "universal_across_all_environments"
            else:
                predictions_exp2[h.id] = "independent_of_field"

        if len(set(predictions_exp2.values())) > 1:
            experiments.append(CrucialExperiment(
                id="exp_auto_field_modulation",
                name="Environmental Boundary & External Field Modulation Assay",
                description="Measure anomaly across varying external field conditions or background environments.",
                target_hypotheses=list(predictions_exp2.keys()),
                exclusory_predictions=predictions_exp2,
            ))

        return experiments

    def evaluate_crucial_experiment(
        self,
        experiment: CrucialExperiment,
        hypotheses: List[Hypothesis],
    ) -> float:
        """Calculates expected information gain / exclusory power of an experiment."""
        active_map = {h.id: h for h in hypotheses if h.status != HypothesisStatus.FALSIFIED}
        if len(active_map) <= 1:
            return 0.0

        current_entropy = self.compute_entropy(hypotheses)
        outcome_groups: Dict[str, List[Hypothesis]] = {}

        for hid, outcome in experiment.exclusory_predictions.items():
            if hid in active_map:
                outcome_groups.setdefault(outcome, []).append(active_map[hid])

        total_weight = sum(h.current_confidence for h in active_map.values())
        if total_weight <= 0:
            return 0.0

        cond_entropy = 0.0
        for group in outcome_groups.values():
            group_weight = sum(h.current_confidence for h in group)
            p_outcome = group_weight / total_weight
            if p_outcome <= 0:
                continue

            inner_probs = [h.current_confidence / group_weight for h in group]
            inner_entropy = -sum(p * math.log2(p) for p in inner_probs if p > 0)
            cond_entropy += p_outcome * inner_entropy

        info_gain = max(0.0, current_entropy - cond_entropy)
        experiment.information_gain = float(info_gain)
        return float(info_gain)

    def execute_and_prune(
        self,
        experiment: CrucialExperiment,
        hypotheses: List[Hypothesis],
        observed_outcome: str,
        error_rate: float = 0.0,
    ) -> Tuple[List[Hypothesis], List[str]]:
        """Executes the crucial experiment with observed outcome."""
        experiment.actual_outcome = observed_outcome
        falsified_ids = []

        for h in hypotheses:
            if h.id in experiment.exclusory_predictions:
                expected = experiment.exclusory_predictions[h.id]
                if expected != observed_outcome:
                    if error_rate == 0.0:
                        h.status = HypothesisStatus.FALSIFIED
                        h.current_confidence = 0.0
                        falsified_ids.append(h.id)
                    else:
                        h.current_confidence *= error_rate
                        if h.current_confidence < 0.05:
                            h.status = HypothesisStatus.FALSIFIED
                            falsified_ids.append(h.id)
                else:
                    h.status = HypothesisStatus.TESTING
                    multiplier = (1.0 - error_rate) if error_rate > 0 else 1.5
                    h.current_confidence = min(0.99, h.current_confidence * multiplier)

        experiment.falsified_hypotheses = falsified_ids

        surviving = [h for h in hypotheses if h.status != HypothesisStatus.FALSIFIED]
        if surviving:
            total = sum(h.current_confidence for h in surviving)
            if total > 0:
                for h in surviving:
                    h.current_confidence = float(h.current_confidence / total)
            if len(surviving) == 1:
                surviving[0].status = HypothesisStatus.CONFIRMED

        return hypotheses, falsified_ids
