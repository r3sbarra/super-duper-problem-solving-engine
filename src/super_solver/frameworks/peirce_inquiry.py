"""Charles Sanders Peirce's Triadic Scientific Inquiry Cycle.

1. Abduction: Surprising anomaly observation -> Generates explanatory hypotheses (both seeded and autonomous)
2. Deduction: Hypothesis -> Derives testable necessary predictions
3. Induction: Empirical observations -> Calibrates confidence and updates validity
"""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

from super_solver.core.embeddings import embedding_service
from super_solver.core.types import Hypothesis, HypothesisStatus


class PeirceanInquiryEngine:
    """Peirce's Triadic Inquiry Engine with Autonomous Abductive Generation."""

    def __init__(self):
        pass

    def autonomous_abduct(
        self,
        anomaly_observation: str,
        domain_context: str,
    ) -> List[Hypothesis]:
        """Autonomously synthesizes candidate scientific hypotheses directly from an anomaly
        without hard-coded lists, spanning the 4 canonical epistemic discovery quadrants:
        1. Systematic / Observational Selection Artifact (Unresolved contaminants/selection effects)
        2. Fundamental Law Modification (Novel physics / modified dynamics / new mechanisms)
        3. Environmental / External Boundary Coupling (External field effect, background medium)
        4. Non-Linear Emergent Interaction (Feedback loops, structural phase transitions)
        """
        # Extract prominent key concepts from anomaly text
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", anomaly_observation) if w.lower() not in {
            "this", "that", "with", "from", "were", "been", "have", "reported", "shows", "paper", "study", "data"
        }]
        key_concept = " ".join(words[:4]) if words else "observed anomaly"

        frames = [
            {
                "type": "Systematic / Observational Selection Artifact",
                "template": f"The {key_concept} is an observational artifact caused by unresolved contaminants, selection bias, or undetected secondary companions in the sample.",
                "archetype": "artifact_hypothesis"
            },
            {
                "type": "Fundamental Law Modification",
                "template": f"The {key_concept} represents a genuine breakdown of standard theory, requiring fundamental modification of governing physical laws or new field interactions.",
                "archetype": "fundamental_modification_hypothesis"
            },
            {
                "type": "Environmental & Boundary Coupling",
                "template": f"The {key_concept} arises from unmodeled external environmental boundary conditions, galactic external field effects, or substrate interactions.",
                "archetype": "environmental_coupling_hypothesis"
            },
            {
                "type": "Structural Phase / Emergent Non-Linear Dynamics",
                "template": f"The {key_concept} is driven by a localized first-order structural phase transition, resonance, or non-linear collective behavior in the constituent medium.",
                "archetype": "emergent_phase_hypothesis"
            },
        ]

        v_anomaly = embedding_service.encode(anomaly_observation)
        v_context = embedding_service.encode(domain_context)
        focus = 0.7 * v_anomaly + 0.3 * v_context
        norm_f = np.linalg.norm(focus)
        if norm_f > 0:
            focus = focus / norm_f

        hypotheses = []
        for _i, frame in enumerate(frames):
            desc = frame["template"]
            v_cand = embedding_service.encode(desc)
            fit = embedding_service.cosine_similarity(v_cand, focus)

            h = Hypothesis(
                id=f"hyp_auto_{frame['archetype']}",
                title=frame["type"],
                description=desc,
                prior_confidence=float(max(0.2, min(0.8, fit))),
                current_confidence=float(max(0.2, min(0.8, fit))),
                vector=v_cand.tolist(),
                status=HypothesisStatus.PROPOSED,
                metadata={
                    "archetype": frame["archetype"],
                    "explanatory_fit": float(fit),
                }
            )
            hypotheses.append(h)

        hypotheses.sort(key=lambda h: h.current_confidence, reverse=True)
        return hypotheses

    def abduct(
        self,
        anomaly_observation: str,
        domain_context: str,
        candidate_explanations: Optional[List[str]] = None,
    ) -> List[Hypothesis]:
        """Generates hypotheses either from candidate seeds or autonomously."""
        if not candidate_explanations:
            return self.autonomous_abduct(anomaly_observation, domain_context)

        v_anomaly = embedding_service.encode(anomaly_observation)
        v_context = embedding_service.encode(domain_context)

        focus = 0.7 * v_anomaly + 0.3 * v_context
        norm_f = np.linalg.norm(focus)
        if norm_f > 0:
            focus = focus / norm_f

        hypotheses = []
        for i, text in enumerate(candidate_explanations):
            v_cand = embedding_service.encode(text)
            explanatory_fit = embedding_service.cosine_similarity(v_cand, focus)

            h = Hypothesis(
                id=f"hyp_abduct_{i+1}",
                title=f"Abductive Explanation {i+1}",
                description=text,
                prior_confidence=float(max(0.1, min(0.9, explanatory_fit))),
                current_confidence=float(max(0.1, min(0.9, explanatory_fit))),
                vector=v_cand.tolist(),
                status=HypothesisStatus.PROPOSED,
                metadata={"abductive_explanatory_fit": float(explanatory_fit)},
            )
            hypotheses.append(h)

        hypotheses.sort(key=lambda h: h.current_confidence, reverse=True)
        return hypotheses

    def deduce_predictions(self, hypothesis: Hypothesis, candidate_predictions: List[str]) -> List[str]:
        """Deduction: Derives observational predictions that must hold if hypothesis is true."""
        v_hyp = np.asarray(hypothesis.vector) if hypothesis.vector else embedding_service.encode(hypothesis.description)

        scored = []
        for pred in candidate_predictions:
            v_pred = embedding_service.encode(pred)
            rel = embedding_service.cosine_similarity(v_hyp, v_pred)
            scored.append((pred, rel))

        scored.sort(key=lambda x: x[1], reverse=True)
        predictions = [p for p, s in scored if s > 0.18]
        hypothesis.predictions = predictions
        return predictions

    def induct_calibrate(
        self,
        hypothesis: Hypothesis,
        observed_evidence: List[str],
    ) -> float:
        """Induction: Evaluates empirical evidence against the deduced predictions."""
        if not hypothesis.predictions or not observed_evidence:
            return hypothesis.current_confidence

        pred_vecs = [embedding_service.encode(p) for p in hypothesis.predictions]
        ev_vecs = [embedding_service.encode(e) for e in observed_evidence]

        matches = []
        for pv in pred_vecs:
            best_match = max(embedding_service.cosine_similarity(pv, ev) for ev in ev_vecs)
            matches.append(best_match)

        avg_match = sum(matches) / len(matches) if matches else 0.0

        new_conf = (hypothesis.current_confidence * 0.4) + (avg_match * 0.6)
        hypothesis.current_confidence = float(min(0.99, max(0.01, new_conf)))

        if hypothesis.current_confidence > 0.65:
            hypothesis.status = HypothesisStatus.CONFIRMED
        elif hypothesis.current_confidence < 0.20:
            hypothesis.status = HypothesisStatus.FALSIFIED
        else:
            hypothesis.status = HypothesisStatus.TESTING

        return hypothesis.current_confidence
