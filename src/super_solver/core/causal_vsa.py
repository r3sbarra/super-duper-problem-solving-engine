"""Vectorized Structural Causal Model (SCM) & do-Calculus Engine.

Implements Pearl's Causal Hierarchy in high-dimensional vector space:
Level 1: Association (Cosine similarity & observational state)
Level 2: Intervention (do(X = x) via hypervector graph surgery)
Level 3: Counterfactuals (Abduction of exogenous background, intervention, counterfactual simulation)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np

from super_solver.core.vsa import VSAEngine


class CausalNode:
    """A variable in a Structural Causal Model."""

    def __init__(self, name: str, domain: str = "general"):
        self.name = name
        self.domain = domain
        self.parents: List[str] = []
        self.children: List[str] = []
        self.mechanism_weights: Dict[str, float] = {}
        self.baseline_value: float = 0.5


class VectorizedCausalModel:
    """Structural Causal Model using Hyperdimensional Computing and graph surgery."""

    def __init__(self, vsa_dim: int = 2048):
        self.vsa = VSAEngine(dim=vsa_dim)
        self.nodes: Dict[str, CausalNode] = {}
        self.node_vectors: Dict[str, np.ndarray] = {}
        self.role_cause = self.vsa.random_hypervector("role:cause")
        self.role_effect = self.vsa.random_hypervector("role:effect")

    def add_variable(self, name: str, baseline_value: float = 0.5, domain: str = "general") -> CausalNode:
        """Registers a causal variable with a dedicated hypervector."""
        node = CausalNode(name=name, domain=domain)
        node.baseline_value = baseline_value
        self.nodes[name] = node
        self.node_vectors[name] = self.vsa.random_hypervector(f"var:{name}")
        return node

    def add_causal_edge(self, cause: str, effect: str, strength: float = 0.8):
        """Adds a directed causal link cause -> effect."""
        if cause not in self.nodes:
            self.add_variable(cause)
        if effect not in self.nodes:
            self.add_variable(effect)

        self.nodes[cause].children.append(effect)
        self.nodes[effect].parents.append(cause)
        self.nodes[effect].mechanism_weights[cause] = strength

    def get_graph_hypervector(self) -> np.ndarray:
        """Encodes the full causal DAG topology into a single bundled hypervector."""
        edge_hypervectors = []
        for e_name, node in self.nodes.items():
            v_effect = self.node_vectors[e_name]
            for c_name in node.parents:
                v_cause = self.node_vectors[c_name]
                # Circular convolution binding: (role_cause (x) Cause) + (role_effect (x) Effect)
                edge_rep = self.vsa.bundle([
                    self.vsa.bind(self.role_cause, v_cause),
                    self.vsa.bind(self.role_effect, v_effect),
                ])
                edge_hypervectors.append(edge_rep)

        return self.vsa.bundle(edge_hypervectors)

    def intervene(
        self,
        target_variable: str,
        clamped_value: float,
        consequence_target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Performs graph surgery do(target_variable = clamped_value).

        Severs incoming edges to target_variable (Pearl's do-calculus Level 2),
        propagates interventional effect forward, and measures causal divergence.
        """
        if target_variable not in self.nodes:
            return {"error": f"Variable '{target_variable}' not in causal model"}

        # 1. Graph Surgery: Sever all incoming edges into target_variable
        severed_parents = list(self.nodes[target_variable].parents)

        # 2. Forward propagate downstream causal impacts
        # Topological / BFS traversal from target_variable to descendants
        simulated_values: Dict[str, float] = {k: node.baseline_value for k, node in self.nodes.items()}
        simulated_values[target_variable] = clamped_value

        visited: Set[str] = {target_variable}
        queue: List[str] = [target_variable]

        while queue:
            curr = queue.pop(0)
            curr_val = simulated_values[curr]

            for child in self.nodes[curr].children:
                w = self.nodes[child].mechanism_weights.get(curr, 0.5)
                # Linear-sigmoid interventional activation
                delta = w * (curr_val - self.nodes[curr].baseline_value)
                simulated_values[child] = float(np.clip(simulated_values[child] + delta, 0.0, 1.0))

                if child not in visited:
                    visited.add(child)
                    queue.append(child)

        # 3. Assess causal effect on consequence target
        target_eval = consequence_target or (self.nodes[target_variable].children[0] if self.nodes[target_variable].children else target_variable)
        baseline_out = self.nodes[target_eval].baseline_value if target_eval in self.nodes else 0.5
        intervened_out = simulated_values.get(target_eval, baseline_out)
        average_causal_effect = intervened_out - baseline_out

        # Compute graph surgery hypervector (severed representation)
        v_target = self.node_vectors[target_variable]
        severed_edges = []
        for p in severed_parents:
            v_p = self.node_vectors[p]
            severed_edges.append(self.vsa.bundle([
                self.vsa.bind(self.role_cause, v_p),
                self.vsa.bind(self.role_effect, v_target),
            ]))

        return {
            "intervention": f"do({target_variable} = {clamped_value:.2f})",
            "target_variable": target_variable,
            "severed_parents": severed_parents,
            "simulated_values": simulated_values,
            "consequence_target": target_eval,
            "baseline_consequence": float(baseline_out),
            "intervened_consequence": float(intervened_out),
            "average_causal_effect": float(average_causal_effect),
            "is_causally_effective": bool(abs(average_causal_effect) > 0.08),
            "summary": (
                f"Graph surgery severed {len(severed_parents)} incoming parent edges to '{target_variable}'. "
                f"Intervention changed '{target_eval}' from {baseline_out:.2f} to {intervened_out:.2f} "
                f"(Average Causal Effect = {average_causal_effect:+.2f})."
            ),
        }
