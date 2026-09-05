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
        resolved_url = base_url or os.getenv("LAB_ASS_URL") or os.getenv("LAB_ASS_BASE_URL", "http://127.0.0.1:3002")
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

    async def export_session(self, session_id: str) -> Dict[str, Any]:
        """Fetch the reproducible research bundle (RO-Crate-shaped JSON-LD)."""
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            resp = await client.get(f"/api/sessions/{session_id}/export")
            resp.raise_for_status()
            return resp.json()
