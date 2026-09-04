"""REST Client for lab-ass integration."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import httpx


class LabAssClient:
    """Connects to lab-ass REST API."""

    def __init__(self, base_url: Optional[str] = None):
        resolved_url = base_url or os.getenv("LAB_ASS_URL") or os.getenv("LAB_ASS_BASE_URL", "http://127.0.0.1:8000")
        self.base_url = resolved_url.rstrip("/")

    async def get_session_state(self, session_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            resp = await client.get(f"/api/sessions/{session_id}/state")
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
