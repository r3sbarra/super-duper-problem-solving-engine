"""Case Harvester & Cross-Domain Experience Transfer.

Harvests solved discovery trajectories and prepares them for analogical transfer
across different scientific and technical domains.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from super_solver.core.embeddings import embedding_service
from super_solver.core.types import DiscoveryPath
from super_solver.memory.episodic_store import EpisodicVectorStore


class CaseHarvester:
    """Harvests and retrieves solved problem-solving trajectories."""

    def __init__(self, store: Optional[EpisodicVectorStore] = None):
        self.store = store or EpisodicVectorStore()

    def harvest_discovery_path(self, path: DiscoveryPath):
        """Indexes a solved discovery path into episodic memory."""
        summary = f"Problem: {path.problem_title}\nBreakthrough: {path.final_breakthrough}\nSteps: {len(path.steps)}"
        vec = embedding_service.encode(f"{path.problem_title}\n{path.final_breakthrough}")

        self.store.insert(
            item_id=f"case_{path.problem_id}",
            category="discovery_case",
            title=path.problem_title,
            content=summary,
            vector=vec,
            metadata={
                "steps_count": len(path.steps),
                "experiments_count": len(path.crucial_experiments),
                "confidence": path.confidence,
            },
        )

    def retrieve_analogous_cases(
        self, current_problem: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Finds analogous prior cases for inspiration and structural transfer."""
        q_vec = embedding_service.encode(current_problem)
        return self.store.search_similar(q_vec, category="discovery_case", top_k=top_k)
