"""Universal Vector Operators for Cross-Disciplinary Problem Solving.

Implements high-dimensional continuous operators for:
1. Anti-symmetric Contradiction Tensors (C = v_imp (x) v_agg - v_agg (x) v_imp)
2. Null-Space Projectors for Constraint & Assumption Inversion (P_A^perp)
3. Maximum-Margin Diagnostic Boundary Kernels
4. Hyperdimensional Structural Algebra for Cross-Domain Analogy
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from super_solver.core.embeddings import embedding_service
from super_solver.core.vsa import VSAEngine


class VectorContradictionTensor:
    """Anti-symmetric continuous contradiction tensor representing directed trade-offs."""

    def __init__(self, improving_vector: np.ndarray, worsening_vector: np.ndarray):
        v_imp = improving_vector / (np.linalg.norm(improving_vector) + 1e-12)
        v_worse = worsening_vector / (np.linalg.norm(worsening_vector) + 1e-12)

        self.v_imp = v_imp
        self.v_worse = v_worse
        # Anti-symmetric bi-vector outer product in matrix representation
        # C_ij = v_imp[i] * v_worse[j] - v_worse[i] * v_imp[j]
        self.bivector_matrix = np.outer(v_imp, v_worse) - np.outer(v_worse, v_imp)
        # Vector difference direction
        self.directional_gradient = v_imp - v_worse

    def tensor_inner_product(self, operator_matrix: np.ndarray) -> float:
        """Computes Frobenius inner product <C, M> = Tr(C^T M)."""
        return float(np.sum(self.bivector_matrix * operator_matrix))

    def evaluate_principle_alignment(self, principle_vector: np.ndarray) -> float:
        """Evaluates how effectively an inventive operator resolves the contradiction.
        
        A breakthrough operator aligns positively with the improvement direction
        while acting orthogonal or restorative to the worsening gradient.
        """
        p = principle_vector / (np.linalg.norm(principle_vector) + 1e-12)
        sim_imp = float(np.dot(p, self.v_imp))
        sim_worse = float(np.dot(p, self.v_worse))
        
        # Reward improvement alignment; penalize worsening alignment; reward gradient synthesis
        alignment = (1.2 * sim_imp) - (0.8 * sim_worse) + (0.5 * float(np.dot(p, self.directional_gradient)))
        return float(alignment)


class NullSpaceAssumptionProjector:
    """Projects problem states into the orthogonal complement of unexamined domain assumptions."""

    def __init__(self, assumption_vectors: List[np.ndarray], dim: int = 384):
        self.dim = dim
        if assumption_vectors:
            # Stack into matrix A (dim x k)
            normed = [v / (np.linalg.norm(v) + 1e-12) for v in assumption_vectors if np.linalg.norm(v) > 0]
            if normed:
                A = np.column_stack(normed)  # (dim, k)
                # Compute projection matrix P_A = A (A^T A)^-1 A^T
                # Using SVD for numerical stability
                U, S, Vt = np.linalg.svd(A, full_matrices=False)
                # Keep significant singular values
                rank = int(np.sum(S > 1e-5))
                if rank > 0:
                    Ur = U[:, :rank]
                    self.P_A = Ur @ Ur.T
                else:
                    self.P_A = np.zeros((dim, dim))
            else:
                self.P_A = np.zeros((dim, dim))
        else:
            self.P_A = np.zeros((dim, dim))

        # Orthogonal complement projector P_A^perp = I - P_A
        self.P_perp = np.eye(dim) - self.P_A

    def project_to_null_space(self, vector: np.ndarray) -> np.ndarray:
        """Projects a vector into the null-space of the assumptions (assumption-free subspace)."""
        proj = self.P_perp @ vector
        norm = np.linalg.norm(proj)
        return proj / (norm + 1e-12) if norm > 0 else proj

    def assumption_overlap(self, vector: np.ndarray) -> float:
        """Measures how much a candidate relies on the unexamined assumptions."""
        norm_v = np.linalg.norm(vector)
        if norm_v == 0:
            return 0.0
        unit_v = vector / norm_v
        reconstructed = self.P_A @ unit_v
        return float(np.linalg.norm(reconstructed))


class ContinuousBoundaryDiagnosticKernel:
    """Continuous max-margin boundary kernel for multi-dimensional root cause and failure analysis."""

    def __init__(self, is_manifestations: List[np.ndarray], is_not_manifestations: List[np.ndarray]):
        self.is_vectors = [v / (np.linalg.norm(v) + 1e-12) for v in is_manifestations if np.linalg.norm(v) > 0]
        self.is_not_vectors = [v / (np.linalg.norm(v) + 1e-12) for v in is_not_manifestations if np.linalg.norm(v) > 0]

        # Compute positive and negative centroids
        if self.is_vectors:
            self.c_is = np.mean(self.is_vectors, axis=0)
            self.c_is = self.c_is / (np.linalg.norm(self.c_is) + 1e-12)
        else:
            self.c_is = np.zeros(384)

        if self.is_not_vectors:
            self.c_not = np.mean(self.is_not_vectors, axis=0)
            self.c_not = self.c_not / (np.linalg.norm(self.c_not) + 1e-12)
        else:
            self.c_not = np.zeros(384)

        # Normal separating vector
        w = self.c_is - self.c_not
        norm_w = np.linalg.norm(w)
        self.w = w / (norm_w + 1e-12) if norm_w > 0 else w

    def evaluate_candidate(self, candidate_vector: np.ndarray, leakage_weight: float = 1.4) -> Dict[str, float]:
        """Evaluates a candidate vector against the positive and negative manifestation fields."""
        v = candidate_vector / (np.linalg.norm(candidate_vector) + 1e-12)
        sim_is = float(np.dot(v, self.c_is))
        sim_not = float(np.dot(v, self.c_not))
        margin = float(np.dot(v, self.w))

        # Margin-guided score: strongly rewards separating margin and penalizes negative margin
        margin_term = (2.0 * margin) if margin > 0 else (4.0 * margin)
        leakage = max(0.0, sim_not - 0.18)
        net_diagnostic_score = sim_is + margin_term - (leakage_weight * leakage)

        return {
            "net_score": float(net_diagnostic_score),
            "sim_to_is": float(sim_is),
            "sim_to_is_not": float(sim_not),
            "margin": float(margin),
            "leakage": float(leakage),
            "is_valid": bool(margin > 0.02 and net_diagnostic_score > 0.15),
        }
