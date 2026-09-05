"""Autonomous hooks and adapters connecting SuperDuperProblemSolvingEngine to lab-ass."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from super_solver.engine import SuperDuperProblemSolvingEngine


class LabAssSuperSolverBridge:
    """Provides advanced problem-solving capabilities directly to lab-ass sessions."""

    def __init__(self, engine: Optional[SuperDuperProblemSolvingEngine] = None):
        self.engine = engine or SuperDuperProblemSolvingEngine()

    def decompose_topic_with_polya_and_triz(
        self,
        topic: str,
        initial_subproblems: List[str],
    ) -> Dict[str, Any]:
        """Decomposes a difficult research topic into auxiliary angles using Polya & TRIZ."""
        aux_angles = self.engine.polya.decompose_auxiliary_problems(
            problem_specification=topic,
            known_subproblems=initial_subproblems,
        )
        principles = self.engine.triz.suggest_principles(topic, top_k=3)

        return {
            "auxiliary_angles": aux_angles,
            "suggested_triz_operators": principles,
        }

    def check_dead_end_deflection(
        self,
        proposed_content: str,
        known_dead_ends: List[str],
    ) -> Dict[str, Any]:
        """Checks if a proposed finding or angle approaches a dead end and computes deflection."""

        from super_solver.core.embeddings import embedding_service

        rep = self.engine.repulsor
        for de in known_dead_ends:
            rep.register_dead_end(de)

        v_prop = embedding_service.encode(proposed_content)
        is_near, max_sim, closest = rep.check_proximity(v_prop)

        if is_near:
            rep.deflect_trajectory(v_prop)

        return {
            "hazard_detected": is_near,
            "max_similarity_to_dead_end": float(max_sim),
            "closest_dead_end": closest,
            "deflected": bool(is_near),
        }

    def suggest_paths(
        self,
        problem_specification: str,
        context: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Synthesize and rank alternative discovery paths (TRIZ/Polya/Gentner/Lakatos/Platt).

        Exposes the engine's suggest_paths() to lab-ass so agents can get
        ranked research directions instead of only a decomposition.
        """
        paths = self.engine.suggest_paths(
            problem_specification=problem_specification,
            context=context,
            top_k=top_k,
        )
        return [
            {
                "title": getattr(p, "title", ""),
                "description": getattr(p, "description", ""),
                "framework": getattr(p, "framework", ""),
                "feasibility": float(getattr(p, "feasibility_score", 0.0) or 0.0),
                "novelty": float(getattr(p, "novelty_score", 0.0) or 0.0),
                "dead_end_margin": float(getattr(p, "dead_end_margin", 0.0) or 0.0),
            }
            for p in paths
        ]

    def run_tournament(
        self,
        candidate_hypotheses: List[str],
        context: Optional[str] = None,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """Run an adversarial dialectical debate across hypotheses; return ranked winners.

        Useful as a pre-filter before adding hypotheses to a lab-ass session.
        """
        results = self.engine.run_tournament(
            candidate_hypotheses=candidate_hypotheses,
            context=context,
            top_k=top_k,
        )
        return [
            {
                "hypothesis": getattr(r, "hypothesis", ""),
                "score": float(getattr(r, "score", 0.0) or 0.0),
                "verdict": getattr(r, "verdict", ""),
            }
            for r in results
        ]

    def verify_math_paper(
        self,
        paper_title: str,
        abstract_text: str,
        target_conjecture: str,
    ) -> Dict[str, Any]:
        """Audit/verify/refute an unverified mathematical claim (Lakatos engine).

        Exposes the engine's math verification to lab-ass so a claim can be
        checked for logical soundness before being promoted to a Claim Card.
        """
        result = self.engine.verify_math_paper(
            paper_id="bridge",
            paper_title=paper_title,
            abstract_text=abstract_text,
            target_conjecture=target_conjecture,
        )
        return {
            "verdict": getattr(result, "verdict", ""),
            "summary": getattr(result, "summary", ""),
            "confidence": float(getattr(result, "confidence", 0.0) or 0.0),
        }

    def deduce_discovery_path(
        self,
        problem_specification: str,
        candidate_hypotheses: List[str],
        ground_truth_outcomes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Run the full discovery pipeline (abduction → MCTS → Platt → breakthrough).

        Returns a JSON-safe summary. Without ground truth, the result is marked
        UNVERIFIED (never self-confirmed).
        """
        from super_solver.core.embeddings import embedding_service
        from super_solver.core.types import ProblemState

        prob = ProblemState(
            id="bridge",
            title=problem_specification[:120],
            specification=problem_specification,
            goal_criteria=["resolve"],
            state_vector=embedding_service.encode(problem_specification).tolist(),
            goal_vector=embedding_service.encode(problem_specification).tolist(),
        )
        path = self.engine.deduce_discovery_path(
            problem=prob,
            candidate_hypotheses=candidate_hypotheses,
            ground_truth_outcomes=ground_truth_outcomes,
        )
        return {
            "final_breakthrough": path.final_breakthrough,
            "confidence": float(path.confidence),
            "total_steps": path.total_steps,
            "falsified_paths": path.falsified_paths,
            "steps": [
                {"operator": s.operator_name, "summary": s.symbolic_summary} for s in path.steps
            ],
        }

    def dpll_entail(
        self,
        premises: List[str],
        target_claim: str,
    ) -> Dict[str, Any]:
        """Check whether the premises logically entail the target claim (DPLL).

        Uses resolution refutation: if (premises AND NOT target) is
        unsatisfiable, the claim is a necessary logical consequence. Returns a
        JSON-safe verdict with the countermodel when entailment fails.
        """
        result = self.engine.dpll.refute_conjecture(
            premises=premises,
            target_claim=target_claim,
        )
        return {
            "proved": bool(result.get("proved")),
            "verdict": result.get("verdict", ""),
            "summary": result.get("summary", ""),
            "countermodel": result.get("countermodel"),
            "clauses_evaluated": int(result.get("clauses_evaluated", 0)),
        }

    def next_experiment(
        self,
        hypothesis_priors: Dict[str, float],
        candidate_designs: List[Dict[str, Any]],
        likelihood_matrix: Dict[str, Dict[str, Dict[str, float]]],
    ) -> List[Dict[str, Any]]:
        """Rank candidate experiments by Expected Information Gain (BOED).

        Selects the single most informative experiment to run next. Each
        candidate design is ``{design_id, name, description, parameters, cost}``
        and the likelihood matrix is ``{design_id: {outcome: {hyp_id: P}}}``.
        Returns designs ranked by cost-penalized EIG (best first).
        """
        from super_solver.frameworks.boed_designer import CandidateExperimentDesign

        designs = [
            CandidateExperimentDesign(
                design_id=d.get("design_id", f"d{i}"),
                name=d.get("name", ""),
                description=d.get("description", ""),
                parameters=d.get("parameters", {}),
                cost=float(d.get("cost", 1.0)),
            )
            for i, d in enumerate(candidate_designs)
        ]
        ranked = self.engine.boed.rank_optimal_experiments(
            hypothesis_priors=hypothesis_priors,
            candidate_designs=designs,
            likelihood_matrix=likelihood_matrix,
        )
        return [
            {
                "design_id": r.get("design_id", ""),
                "name": r.get("name", ""),
                "expected_information_gain": float(r.get("expected_information_gain", 0.0)),
                "cost_penalized_utility": float(r.get("cost_penalized_utility", 0.0)),
                "prior_entropy": float(r.get("prior_entropy", 0.0)),
                "expected_posterior_entropy": float(r.get("expected_posterior_entropy", 0.0)),
            }
            for r in ranked
        ]

    # ------------------------------------------------------------------
    # Problem / Solution / Path vectorizing (custom embedder + vector index)
    # ------------------------------------------------------------------

    def _vector_service(self):
        from super_solver.vectorize import VectorizationService

        if getattr(self, "_vec_svc", None) is None:
            self._vec_svc = VectorizationService()
        return self._vec_svc

    def vectorize_problem(
        self,
        specification: str,
        title: str = "",
        goal_criteria: Optional[List[str]] = None,
        boundary_is: Optional[List[str]] = None,
        boundary_is_not: Optional[List[str]] = None,
        store: bool = True,
    ) -> Dict[str, Any]:
        """Encode a problem into a vector; optionally index it for later retrieval."""
        svc = self._vector_service()
        vec = svc.encode_problem(
            specification, title, goal_criteria, boundary_is, boundary_is_not
        )
        result = {
            "kind": "problem",
            "title": title,
            "specification": specification,
            "dim": int(vec.shape[0]),
            "vector": vec.tolist(),
        }
        if store:
            result["id"] = svc.store_problem(
                specification, title, goal_criteria, boundary_is, boundary_is_not
            )
        return result

    def vectorize_solution(
        self,
        solution_text: str,
        method: str = "",
        domain: str = "",
        operators: Optional[List[str]] = None,
        title: str = "",
        store: bool = True,
    ) -> Dict[str, Any]:
        """Encode a solution into a vector; optionally index it."""
        svc = self._vector_service()
        vec = svc.encode_solution(solution_text, method, domain, operators)
        result = {
            "kind": "solution",
            "title": title,
            "solution": solution_text,
            "dim": int(vec.shape[0]),
            "vector": vec.tolist(),
        }
        if store:
            result["id"] = svc.store_solution(
                solution_text, method, domain, operators, title=title
            )
        return result

    def vectorize_path(
        self,
        steps: List[str],
        operator_types: Optional[List[str]] = None,
        final_breakthrough: str = "",
        title: str = "",
        store: bool = True,
    ) -> Dict[str, Any]:
        """Encode a discovery path (sequence of reasoning steps) into a vector."""
        svc = self._vector_service()
        vec = svc.encode_path(steps, operator_types, final_breakthrough)
        result = {
            "kind": "path",
            "title": title,
            "steps": steps,
            "dim": int(vec.shape[0]),
            "vector": vec.tolist(),
        }
        if store:
            result["id"] = svc.store_path(
                steps, operator_types, final_breakthrough, title=title
            )
        return result

    def search_vectors(
        self,
        query_text: str,
        kind: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        """Search the vector index for entries similar to ``query_text``."""
        svc = self._vector_service()
        vec = svc.backend.encode(query_text)
        hits = svc.search(vec, kind=kind, top_k=top_k, min_similarity=min_similarity)
        return {"query": query_text, "kind": kind, "hits": hits}

    def analogize_problem(
        self,
        specification: str,
        title: str = "",
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        """Find similar past problems and return their linked solutions + paths.

        This is the analogical-transfer primitive: given a new problem, retrieve
        the most similar previously-solved problem and its proven solution and
        discovery path so the solver can reuse a successful trajectory.
        """
        svc = self._vector_service()
        vec = svc.encode_problem(specification, title=title)
        analogs = svc.analogize(vec, top_k=top_k, min_similarity=min_similarity)
        return {"problem": specification, "analogs": analogs}
