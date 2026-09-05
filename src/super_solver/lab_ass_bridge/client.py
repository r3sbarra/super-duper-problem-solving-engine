"""REST Client for lab-ass integration.

Connects to the lab-ass REST API. Endpoints match lab-ass's actual routes
(verified against lab_ass/api/*). Default base URL is lab-ass's port 3002
(port 3000 is Grafana).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx


class LabAssClient:
    """Connects to lab-ass REST API."""

    def __init__(self, base_url: Optional[str] = None):
        resolved_url = (
            base_url
            or os.getenv("LAB_ASS_URL")
            or os.getenv("LAB_ASS_BASE_URL", "http://127.0.0.1:3002")
        )
        self.base_url = resolved_url.rstrip("/")

    async def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Fetch full session state (angles, findings, dead-ends, agents)."""
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            resp = await client.get(f"/api/sessions/{session_id}")
            resp.raise_for_status()
            return resp.json()

    async def add_finding(
        self,
        session_id: str,
        content: str,
        confidence: float = 0.9,
        agent_id: Optional[str] = None,
        angle_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            payload = {
                "content": content,
                "confidence": confidence,
                "agent_id": agent_id,
                "angle_id": angle_id,
                "auto_dedup": True,
            }
            resp = await client.post(f"/api/sessions/{session_id}/findings", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def add_findings_batch(
        self,
        session_id: str,
        findings: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Bulk-add multiple findings in one request.

        Each finding dict supports: content (required), confidence, agent_id,
        angle_id, source_url, source_id.
        """
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            resp = await client.post(
                f"/api/sessions/{session_id}/findings/batch",
                json={"findings": findings},
            )
            resp.raise_for_status()
            return resp.json()

    async def vote_finding(
        self,
        session_id: str,
        finding_id: str,
        vote: str,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Vote on a finding's quality. vote is one of 'up', 'down', 'neutral'."""
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            resp = await client.post(
                f"/api/sessions/{session_id}/findings/{finding_id}/vote",
                json={"vote": vote, "agent_id": agent_id},
            )
            resp.raise_for_status()
            return resp.json()

    async def deduplicate_findings(self, session_id: str) -> Dict[str, Any]:
        """Trigger deduplication of semantically redundant findings in the session."""
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            resp = await client.post(f"/api/sessions/{session_id}/findings/deduplicate")
            resp.raise_for_status()
            return resp.json()

    async def search_similar_findings(
        self,
        session_id: str,
        content: str,
        threshold: float = 0.8,
    ) -> Dict[str, Any]:
        """Search for semantically similar findings within the session."""
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            resp = await client.post(
                f"/api/sessions/{session_id}/findings/search",
                json={"content": content, "threshold": threshold},
            )
            resp.raise_for_status()
            return resp.json()

    async def register_dead_end(
        self,
        session_id: str,
        reason: str,
        angle_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            payload = {"reason": reason, "angle_id": angle_id}
            resp = await client.post(f"/api/sessions/{session_id}/dead-ends", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def get_shared_blackboard(self, session_id: str) -> Dict[str, Any]:
        """Fetch the shared agent blackboard for a session (cross-agent context)."""
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            resp = await client.get(f"/api/sessions/{session_id}/blackboard")
            resp.raise_for_status()
            return resp.json()

    async def send_message(
        self,
        session_id: str,
        content: str,
        agent_id: Optional[str] = None,
        kind: str = "update",
        recipient_agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send an inter-agent message or broadcast within a session.

        kind: 'update' | 'question' | 'handoff' | 'alert'
        """
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            resp = await client.post(
                f"/api/sessions/{session_id}/messages",
                json={
                    "content": content,
                    "agent_id": agent_id,
                    "kind": kind,
                    "recipient_agent_id": recipient_agent_id,
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def create_claim_card(
        self,
        session_id: str,
        statement: str,
        h0: Optional[str] = None,
        h1: Optional[str] = None,
        falsification_criteria: Optional[str] = None,
        finding_ids: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """Create a Claim Card (the shared falsifiable-claim identity)."""
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            payload = {
                "statement": statement,
                "h0": h0,
                "h1": h1,
                "falsification_criteria": falsification_criteria,
                "finding_ids": finding_ids or [],
            }
            resp = await client.post(f"/api/sessions/{session_id}/claims", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def record_claim_verdict(
        self,
        session_id: str,
        claim_id: str,
        verdict: str,
        verdict_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a verification verdict (verified/falsified/inconclusive)."""
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            payload = {"verdict": verdict, "verdict_ref": verdict_ref}
            resp = await client.post(
                f"/api/sessions/{session_id}/claims/{claim_id}/verdict", json=payload
            )
            resp.raise_for_status()
            return resp.json()

    async def preregister_hypothesis(
        self,
        session_id: str,
        hypothesis_id: str,
        null_hypothesis: str,
        testable_prediction: str,
        falsification_criteria: str,
        agent_id: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Pre-register a hypothesis (OSF standard) to prevent post-hoc bias.

        Links engine-generated hypotheses to lab-ass for full audit trail:
        abduction → preregistration → experiment → confirmation/falsification.
        """
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            payload = {
                "hypothesis_id": hypothesis_id,
                "null_hypothesis": null_hypothesis,
                "testable_prediction": testable_prediction,
                "falsification_criteria": falsification_criteria,
                "agent_id": agent_id,
                "variables": variables or {},
            }
            resp = await client.post(
                f"/api/sessions/{session_id}/hypotheses/{hypothesis_id}/preregister",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    async def design_experiment(
        self,
        session_id: str,
        title: str,
        methodology: str,
        agent_id: Optional[str] = None,
        angle_id: Optional[str] = None,
        hypothesis_id: Optional[str] = None,
        control_group: Optional[str] = None,
        experimental_group: Optional[str] = None,
        target_metric: Optional[str] = None,
        execution_command: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Design an experimental protocol manifest in lab-ass.

        Call this after super_solver next_experiment (BOED) selects the top
        design, writing the manifest back so lab-ass tracks it.
        """
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            payload = {
                "title": title,
                "methodology": methodology,
                "agent_id": agent_id,
                "angle_id": angle_id,
                "hypothesis_id": hypothesis_id,
                "control_group": control_group,
                "experimental_group": experimental_group,
                "target_metric": target_metric,
                "execution_command": execution_command,
                "parameters": parameters or {},
            }
            resp = await client.post(
                f"/api/sessions/{session_id}/experiments",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    async def record_experiment_run(
        self,
        session_id: str,
        experiment_id: str,
        outcome: str,
        metrics: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        status: str = "completed",
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record execution results of an experiment back to lab-ass.

        Closes the feedback loop: super-solver runs an experiment, records the
        outcome, and lab-ass updates hypothesis confidence accordingly.
        """
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            payload = {
                "outcome": outcome,
                "metrics": metrics or {},
                "agent_id": agent_id,
                "status": status,
                "notes": notes,
            }
            resp = await client.post(
                f"/api/sessions/{session_id}/experiments/{experiment_id}/runs",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    async def get_session_digest(self, session_id: str) -> Dict[str, Any]:
        """Fetch a compact digest of session progress (token-efficient alternative to get_state)."""
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            resp = await client.get(f"/api/sessions/{session_id}/digest")
            resp.raise_for_status()
            return resp.json()

    async def get_workload(self, session_id: str) -> Dict[str, Any]:
        """Get the current workload distribution across agents in a session."""
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            resp = await client.get(f"/api/sessions/{session_id}/workload")
            resp.raise_for_status()
            return resp.json()

    async def synthesize_report(self, session_id: str) -> Dict[str, Any]:
        """Trigger lab-ass to synthesize a final research report from all session findings."""
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            resp = await client.post(f"/api/sessions/{session_id}/synthesize")
            resp.raise_for_status()
            return resp.json()

    async def export_session(self, session_id: str) -> Dict[str, Any]:
        """Fetch the reproducible research bundle (RO-Crate-shaped JSON-LD)."""
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            resp = await client.get(f"/api/sessions/{session_id}/export")
            resp.raise_for_status()
            return resp.json()
