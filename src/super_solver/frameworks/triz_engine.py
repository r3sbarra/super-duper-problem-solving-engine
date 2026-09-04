"""TRIZ Inventive Problem Solving & Contradiction Resolution Engine.

Implements Altshuller's 40 Inventive Principles and Contradiction Matrix
to resolve engineering and scientific trade-offs without compromise.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from super_solver.core.embeddings import embedding_service


TRIZ_PRINCIPLES: Dict[int, Dict[str, str]] = {
    1: {"name": "Segmentation", "desc": "Divide an object/problem into independent parts; make it sectional; increase degree of fragmentation."},
    2: {"name": "Extraction", "desc": "Extract the disturbing part or property from an object, or extract only the necessary part."},
    3: {"name": "Local Quality", "desc": "Change an object's structure from uniform to non-uniform; make each part fulfill a different useful function."},
    4: {"name": "Asymmetry", "desc": "Change the shape of an object from symmetrical to asymmetrical; reinforce asymmetry to achieve goal."},
    5: {"name": "Merging", "desc": "Bring closer together or merge identical or similar objects, operations, or data streams."},
    6: {"name": "Universality", "desc": "Make an object or subsystem perform multiple functions, eliminating the need for other parts."},
    7: {"name": "Nested Doll", "desc": "Place one object inside another; place each object, in turn, inside the other."},
    8: {"name": "Anti-Weight", "desc": "To compensate for the weight/burden of an object, merge it with other objects that provide lift/support."},
    9: {"name": "Preliminary Anti-Action", "desc": "If it will be necessary to do an action with both harmful and useful effects, create opposing stress in advance."},
    10: {"name": "Preliminary Action", "desc": "Perform, before it is needed, the required change of an object (either fully or partially). Pre-arrange objects."},
    11: {"name": "Beforehand Cushioning", "desc": "Prepare emergency means beforehand to compensate for the relatively low reliability of an object."},
    12: {"name": "Equipotentiality", "desc": "In a potential field, limit condition changes (e.g. avoid position shifts in gravity or electric potential)."},
    13: {"name": "The Other Way Round", "desc": "Invert the action used to solve the problem (e.g. instead of cooling the object, heat the exterior)."},
    14: {"name": "Spheroidality - Curvature", "desc": "Instead of using rectilinear parts, surfaces, or forms, use spherical or curvilinear ones."},
    15: {"name": "Dynamics", "desc": "Allow an object or process to automatically adjust for optimal performance at each stage of operation."},
    16: {"name": "Partial or Excessive Actions", "desc": "If 100% of an object or result is hard to achieve using a specified method, use 'slightly less' or 'slightly more'."},
    17: {"name": "Another Dimension", "desc": "Move into an additional dimension (from 1D to 2D, 2D to 3D, or spatial to temporal/latent)."},
    18: {"name": "Mechanical Vibration", "desc": "Cause an object or field to oscillate or vibrate; increase frequency up to ultrasonic."},
    19: {"name": "Periodic Action", "desc": "Instead of continuous action, use periodic or pulsating actions; if already periodic, change frequency."},
    20: {"name": "Continuity of Useful Action", "desc": "Carry on work continuously; make all parts of an object work at full load, all the time."},
    21: {"name": "Skipping / Fast Forward", "desc": "Conduct a process, or certain stages (e.g. destructive, harmful, or high-risk operations) at high speed."},
    22: {"name": "Blessing in Disguise", "desc": "Use harmful factors to achieve a positive effect; eliminate primary harm by adding another harm."},
    23: {"name": "Feedback", "desc": "Introduce feedback (referring back, cross-checking) to improve a process or action."},
    24: {"name": "Intermediary", "desc": "Use an intermediary carrier article or intermediate process; merge one object temporarily with another."},
    25: {"name": "Self-Service", "desc": "Make an object serve itself by performing auxiliary helpful functions; use waste resources."},
    26: {"name": "Copying", "desc": "Instead of an unavailable, expensive, fragile object, use simpler and inexpensive copies or models."},
    27: {"name": "Cheap Short-Living Objects", "desc": "Replace an expensive object with a multiple of inexpensive objects, compromising certain properties."},
    28: {"name": "Mechanics Substitution", "desc": "Replace a mechanical means with sensory, optical, acoustic, or electromagnetic means."},
    29: {"name": "Pneumatics and Hydraulics", "desc": "Use gaseous and liquid parts of an object instead of solid parts."},
    30: {"name": "Flexible Shells and Thin Films", "desc": "Use flexible shells and thin films instead of three-dimensional structures; isolate object from external environment."},
    31: {"name": "Porous Materials", "desc": "Make an object porous or add porous elements (inserts, coatings)."},
    32: {"name": "Color Changes", "desc": "Change the color of an object or its external environment; change transparency."},
    33: {"name": "Homogeneity", "desc": "Make objects interacting with a given object of the same material (or material with identical properties)."},
    34: {"name": "Discarding and Recovering", "desc": "Make portions of an object that have fulfilled their functions go away (discard by dissolving, evaporating) or modify them directly."},
    35: {"name": "Parameter Changes", "desc": "Change an object's physical state (e.g. phase, density, flexibility, temperature, concentration)."},
    36: {"name": "Phase Transitions", "desc": "Use phenomena occurring during phase transitions (e.g. volume changes, loss or absorption of heat)."},
    37: {"name": "Thermal Expansion", "desc": "Use thermal expansion (or contraction) of materials."},
    38: {"name": "Strong Oxidants", "desc": "Replace normal environment with enriched or active field/reaction (e.g. oxygen-enriched, ionized)."},
    39: {"name": "Inert Atmosphere", "desc": "Replace normal environment with an inert one; add neutral parts or inert materials."},
    40: {"name": "Composite Materials", "desc": "Change from uniform to composite (multiple) materials with tailored complementary properties."}
}


# Canonical Altshuller Contradiction Lookup Table for classic parameter conflicts
CONTRADICTION_MATRIX: Dict[Tuple[str, str], List[int]] = {
    ("strength", "weight"): [40, 8, 1, 15],
    ("weight", "strength"): [8, 40, 1, 35],
    ("speed", "accuracy"): [28, 10, 19, 32],
    ("accuracy", "speed"): [28, 10, 19, 13],
    ("capacity", "volume"): [35, 17, 7, 2],
    ("volume", "capacity"): [14, 35, 7, 17],
    ("power", "energy_loss"): [19, 35, 36, 12],
    ("complexity", "reliability"): [2, 27, 1, 25],
    ("reliability", "complexity"): [11, 23, 27, 35],
    ("stability", "temperature"): [35, 39, 19, 21],
    ("information", "noise"): [24, 23, 15, 1],
}


class TRIZEngine:
    """TRIZ Inventive Operator Matcher and Contradiction Resolver."""

    def __init__(self):
        self.principle_vectors: Dict[int, np.ndarray] = {}
        for pid, data in TRIZ_PRINCIPLES.items():
            text = f"{data['name']}: {data['desc']}"
            self.principle_vectors[pid] = embedding_service.encode(text)

    def resolve_contradiction_matrix(
        self,
        improving_parameter: str,
        worsening_parameter: str,
    ) -> List[Dict[str, Any]]:
        """Directly queries Altshuller's Contradiction Matrix for known parameter trade-offs."""
        key = (improving_parameter.strip().lower(), worsening_parameter.strip().lower())
        principle_ids = CONTRADICTION_MATRIX.get(key)
        if not principle_ids:
            # Fallback to semantic matching across both parameters
            return self.suggest_principles(f"Improve {improving_parameter} without worsening {worsening_parameter}")

        results = []
        for pid in principle_ids:
            data = TRIZ_PRINCIPLES[pid]
            results.append({
                "principle_id": pid,
                "name": data["name"],
                "description": data["desc"],
                "source": "altshuller_matrix",
            })
        return results

    def suggest_principles(
        self,
        contradiction_description: str,
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        """Matches a problem contradiction to the most relevant TRIZ Inventive Principles."""
        v_contra = embedding_service.encode(contradiction_description)
        scores = []

        for pid, v_principle in self.principle_vectors.items():
            sim = embedding_service.cosine_similarity(v_contra, v_principle)
            scores.append((pid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for pid, sim in scores[:top_k]:
            data = TRIZ_PRINCIPLES[pid]
            results.append({
                "principle_id": pid,
                "name": data["name"],
                "description": data["desc"],
                "similarity_score": float(sim),
                "source": "vector_similarity",
            })
        return results

    def apply_principle_transformation(
        self,
        state_vector: np.ndarray,
        principle_id: int,
        blend_factor: float = 0.35,
    ) -> np.ndarray:
        """Applies a TRIZ transformation operator to steer the state vector in representation space."""
        if principle_id not in self.principle_vectors:
            return state_vector
        p_vec = self.principle_vectors[principle_id]
        new_state = (1.0 - blend_factor) * state_vector + blend_factor * p_vec
        norm = np.linalg.norm(new_state)
        return new_state / (norm + 1e-12) if norm > 0 else new_state
