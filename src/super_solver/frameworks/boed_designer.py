"""Bayesian Optimal Experimental Design (BOED) Engine.

Selects the single most informative experiment to run next by maximizing Expected Information Gain (EIG):
EIG(d) = H(H) - E_{y ~ p(y|d)} [ H(H | y, d) ]
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


class CandidateExperimentDesign:
    """Specification of an experimental protocol with design parameters and cost."""

    def __init__(
        self,
        design_id: str,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        cost: float = 1.0,
    ):
        self.design_id = design_id
        self.name = name
        self.description = description
        self.parameters = parameters
        self.cost = cost


class BOEDDesigner:
    """Bayesian Optimal Experimental Design optimizer via Shannon Entropy & Mutual Information."""

    def __init__(self):
        pass

    def compute_shannon_entropy(self, probabilities: List[float]) -> float:
        """Computes Shannon entropy H(P) in nats."""
        total = sum(probabilities)
        if total <= 0:
            return 0.0
        norm_p = [p / total for p in probabilities if p > 0]
        return float(-sum(p * math.log(p + 1e-12) for p in norm_p))

    def evaluate_design_eig(
        self,
        hypothesis_priors: Dict[str, float],
        design: CandidateExperimentDesign,
        outcome_likelihoods: Dict[str, Dict[str, float]],
        # outcome_likelihoods: {outcome_name: {hypothesis_id: P(outcome | hyp, design)}}
    ) -> Dict[str, Any]:
        """Calculates the Expected Information Gain (EIG) for a candidate experimental design.

        EIG = H_prior - sum_{y} P(y) * H_posterior(y)
        """
        hyp_names = list(hypothesis_priors.keys())
        priors = [hypothesis_priors[h] for h in hyp_names]
        prior_entropy = self.compute_shannon_entropy(priors)

        # Compute marginal probability of each outcome P(y | d) = sum_h P(y | h, d) * P(h)
        outcome_marginals: Dict[str, float] = {}
        posteriors_per_outcome: Dict[str, List[float]] = {}

        for outcome, hyp_conds in outcome_likelihoods.items():
            p_y = 0.0
            unnorm_posteriors = []
            for h in hyp_names:
                p_h = hypothesis_priors[h]
                p_y_given_h = hyp_conds.get(h, 0.5)
                joint = p_y_given_h * p_h
                p_y += joint
                unnorm_posteriors.append(joint)

            outcome_marginals[outcome] = p_y
            if p_y > 0:
                posteriors_per_outcome[outcome] = [j / p_y for j in unnorm_posteriors]
            else:
                posteriors_per_outcome[outcome] = list(priors)

        # Expected posterior entropy E_y[ H(H | y, d) ]
        total_p_y = sum(outcome_marginals.values())
        if total_p_y > 0:
            normalized_marginals = {k: v / total_p_y for k, v in outcome_marginals.items()}
        else:
            normalized_marginals = {k: 1.0 / len(outcome_marginals) for k in outcome_marginals}

        expected_posterior_entropy = 0.0
        for outcome, p_y in normalized_marginals.items():
            h_post = self.compute_shannon_entropy(posteriors_per_outcome[outcome])
            expected_posterior_entropy += p_y * h_post

        eig = max(0.0, prior_entropy - expected_posterior_entropy)
        # Cost-penalized utility
        cost_penalized_utility = eig / max(0.1, math.sqrt(design.cost))

        return {
            "design_id": design.design_id,
            "name": design.name,
            "prior_entropy": float(prior_entropy),
            "expected_posterior_entropy": float(expected_posterior_entropy),
            "expected_information_gain": float(eig),
            "cost_penalized_utility": float(cost_penalized_utility),
            "outcome_marginals": normalized_marginals,
        }

    def rank_optimal_experiments(
        self,
        hypothesis_priors: Dict[str, float],
        candidate_designs: List[CandidateExperimentDesign],
        likelihood_matrix: Dict[str, Dict[str, Dict[str, float]]],
        # likelihood_matrix: {design_id: {outcome_name: {hyp_id: P(y|h, d)}}}
    ) -> List[Dict[str, Any]]:
        """Ranks candidate experiments by their cost-penalized Expected Information Gain."""
        ranked = []
        for design in candidate_designs:
            outcome_likelihoods = likelihood_matrix.get(design.design_id, {})
            metrics = self.evaluate_design_eig(hypothesis_priors, design, outcome_likelihoods)
            ranked.append(metrics)

        ranked.sort(key=lambda x: x["cost_penalized_utility"], reverse=True)
        return ranked
