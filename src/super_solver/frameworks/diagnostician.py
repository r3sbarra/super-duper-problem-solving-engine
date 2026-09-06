"""Automated Kepner-Tregoe Diagnostician for Small/Low-End Models.

Transforms unstructured, messy error traces and system bug reports into a
rigorous 4-dimensional contrastive boundary (IS vs IS NOT) across Identity,
Location, Timing, and Extent. Projects candidate causes onto the separating
hyperplane to isolate true root causes with high mathematical margin.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from super_solver.core.llm_parser import OptionalLLMClient, llm_client
from super_solver.core.types import KTBoundary
from super_solver.frameworks.kepner_tregoe import KepnerTregoeEngine

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticCandidate:
    cause: str
    net_score: float
    valid_boundary_fit: bool
    hyperplane_projection: float
    violated_dimensions: List[str] = field(default_factory=list)


@dataclass
class DiagnosticReport:
    problem_summary: str
    boundary: KTBoundary
    candidates: List[DiagnosticCandidate]
    top_root_cause: Optional[DiagnosticCandidate] = None
    is_resolved: bool = False


class AutomatedDiagnostician:
    """Automated root-cause diagnosis engine powered by Kepner-Tregoe 4D boundaries."""

    def __init__(
        self,
        client: Optional[OptionalLLMClient] = None,
        kt_engine: Optional[KepnerTregoeEngine] = None,
    ):
        self.client = client or llm_client
        self.kt_engine = kt_engine or KepnerTregoeEngine()

    def diagnose_failure(
        self,
        error_log_or_description: str,
        context: Optional[str] = None,
        candidate_causes: Optional[List[str]] = None,
    ) -> DiagnosticReport:
        """Constructs a 4D KTBoundary and ranks candidate root causes."""
        # 1. Structure the problem into a 4D IS / IS NOT boundary
        boundary = self._build_boundary(error_log_or_description, context)

        # 2. Extract or generate candidate causes
        causes = candidate_causes or self._generate_candidate_causes(error_log_or_description, boundary)

        # 3. Evaluate each candidate cause through the KT hyperplane separator
        evaluations: List[DiagnosticCandidate] = []
        for cause in causes:
            res = self.kt_engine.evaluate_candidate_cause(cause, boundary)
            evaluations.append(
                DiagnosticCandidate(
                    cause=cause,
                    net_score=res["net_score"],
                    valid_boundary_fit=res["valid_boundary_fit"],
                    hyperplane_projection=res["hyperplane_projection"],
                    violated_dimensions=res.get("violated_dimensions", []),
                )
            )

        # Sort by net score descending
        evaluations.sort(key=lambda x: x.net_score, reverse=True)
        top_cause = evaluations[0] if evaluations else None

        return DiagnosticReport(
            problem_summary=error_log_or_description[:200],
            boundary=boundary,
            candidates=evaluations,
            top_root_cause=top_cause,
            is_resolved=bool(top_cause and top_cause.valid_boundary_fit),
        )

    def _build_boundary(self, description: str, context: Optional[str] = None) -> KTBoundary:
        """Uses the small model (or deterministic fallback) to extract the 4D IS/IS-NOT boundary."""
        if not self.client.enabled:
            # Deterministic fallback boundary
            return KTBoundary(
                identity_is=description[:100],
                identity_is_not="expected normal execution without errors",
                location_is="system execution stack",
                location_is_not="unrelated modules",
                timing_is="during invocation",
                timing_is_not="prior to execution",
                extent_is="failure observed",
                extent_is_not="catastrophic complete crash",
            )

        system_instruction = (
            "You are a Kepner-Tregoe root-cause diagnostic specialist. Given a bug report or error log, "
            "decompose it into a 4-dimensional contrastive matrix: IS (what is failing) vs IS NOT (what could be failing but is not). "
            "Output strictly a JSON object with keys: 'identity_is', 'identity_is_not', 'location_is', 'location_is_not', "
            "'timing_is', 'timing_is_not', 'extent_is', 'extent_is_not'."
        )

        prompt = (
            f"Error / Incident Report:\n{description}\n\n"
            f"Context:\n{context or 'No additional context.'}\n\n"
            f"Output strictly valid JSON with the 8 Kepner-Tregoe boundary fields."
        )

        raw = self.client.generate_text(prompt, system=system_instruction, json_format=True)
        if raw:
            try:
                data = json.loads(self.client.clean_json_string(raw))
                return KTBoundary(
                    identity_is=str(data.get("identity_is", description[:100])),
                    identity_is_not=str(data.get("identity_is_not", "normal execution")),
                    location_is=str(data.get("location_is", "current module")),
                    location_is_not=str(data.get("location_is_not", "other modules")),
                    timing_is=str(data.get("timing_is", "at runtime")),
                    timing_is_not=str(data.get("timing_is_not", "at startup")),
                    extent_is=str(data.get("extent_is", "error triggered")),
                    extent_is_not=str(data.get("extent_is_not", "unaffected components")),
                )
            except Exception as e:
                logger.debug(f"Failed to parse KT boundary JSON: {e}")

        # Fallback
        return KTBoundary(
            identity_is=description[:100],
            identity_is_not="normal execution",
            location_is="current stack",
            location_is_not="other systems",
            timing_is="runtime",
            timing_is_not="idle",
            extent_is="reproducible error",
            extent_is_not="unrelated workflows",
        )

    def _generate_candidate_causes(self, description: str, boundary: KTBoundary) -> List[str]:
        """Generates 3-5 candidate root causes using abduction."""
        if not self.client.enabled:
            return [
                f"Configuration mismatch or missing dependency in {boundary.location_is}",
                f"Unexpected input format violating contract in {boundary.identity_is}",
                "Resource exhaustion or timeout during execution",
            ]

        system_instruction = (
            "You are a systems diagnostic engineer. Based on the error report and Kepner-Tregoe boundary, "
            "propose 3 to 5 distinct, falsifiable candidate root causes. Output strictly a JSON array of strings."
        )

        prompt = (
            f"Error:\n{description}\n\n"
            f"Boundary:\n- IS: {boundary.identity_is} at {boundary.location_is} ({boundary.timing_is})\n"
            f"- IS NOT: {boundary.identity_is_not} at {boundary.location_is_not}\n\n"
            f"Output strictly a JSON list of candidate root cause strings."
        )

        raw = self.client.generate_text(prompt, system=system_instruction, json_format=True)
        if raw:
            try:
                data = json.loads(self.client.clean_json_string(raw))
                if isinstance(data, list) and all(isinstance(x, str) for x in data):
                    return data
            except Exception as e:
                logger.debug(f"Failed to parse candidate causes: {e}")

        return [
            f"Fault in {boundary.identity_is}",
            f"Environment issue at {boundary.location_is}",
            "Timing / concurrency race condition",
        ]
