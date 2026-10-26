"""Solution Mechanism Synthesizer (SMS).

Bridges the structural gap between Polya heuristic decomposition and concrete solution
retrieval. Polya heuristics split goals into auxiliary subproblems, but subproblems
alone do not contain implementation mechanisms.

The Solution Mechanism Synthesizer translates each subproblem into actionable, concrete
computational and engineering mechanisms by composing:
1. Structured problem slots (givens, unknowns, constraints, goal).
2. Specific domain action primitives (e.g. caching, indexing, partitioning, memoization,
   branch-and-bound, relaxation, dynamic programming).
3. Matched TRIZ inventive principles.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from super_solver.vectorize.primitives import strip_to_primitives

# Curated catalog of concrete mechanism templates keyed by target objective patterns
_MECHANISM_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "latency": [
        {
            "name": "in_memory_lru_cache",
            "action": "act_cache",
            "template": "Introduce an in-memory LRU cache with deterministic key hashing and bounded capacity to bypass repetitive computation.",
            "triz_principle": "Prior Action",
        },
        {
            "name": "asynchronous_pipelining",
            "action": "act_pipeline",
            "template": "Decouple synchronous execution paths into an asynchronous worker pipeline with bounded ring buffers.",
            "triz_principle": "Segmentation",
        },
    ],
    "concurrency": [
        {
            "name": "lock_free_ring_buffer",
            "action": "act_partition",
            "template": "Partition state per worker thread using atomic sequence numbers or single-writer lock-free ring buffers.",
            "triz_principle": "Asymmetry",
        },
        {
            "name": "optimistic_concurrency_control",
            "action": "act_verify",
            "template": "Apply optimistic concurrency control with monotonic version stamps and exponential backoff retry.",
            "triz_principle": "Flexible Shells and Thin Films",
        },
    ],
    "optimization": [
        {
            "name": "branch_and_bound_pruning",
            "action": "act_prune",
            "template": "Formulate a Lagrangian relaxation or linear-programming dual bound to prune non-optimal subtrees in branch-and-bound.",
            "triz_principle": "Partial or Excessive Actions",
        },
        {
            "name": "dynamic_programming_memoization",
            "action": "act_memoize",
            "template": "Discretize state space into overlapping optimal substructures with topological memoization.",
            "triz_principle": "Periodic Action",
        },
    ],
    "consistency": [
        {
            "name": "idempotent_event_deduplication",
            "action": "act_filter",
            "template": "Assign monotonic UUIDs or sliding-window Bloom filters at ingestion boundaries for zero-leakage deduplication.",
            "triz_principle": "Feedback",
        },
        {
            "name": "two_phase_commit_with_outbox",
            "action": "act_isolate",
            "template": "Implement a transactional outbox table with change-data-capture guarantees for eventual cross-system consistency.",
            "triz_principle": "Preliminary Anti-Action",
        },
    ],
    "resilience": [
        {
            "name": "circuit_breaker_and_bulkhead",
            "action": "act_isolate",
            "template": "Wrap outbound client calls in a token-bucket rate limiter with fail-fast circuit breaker and concurrency bulkheads.",
            "triz_principle": "Local Quality",
        },
        {
            "name": "graceful_shedding_and_fallback",
            "action": "act_shed",
            "template": "Monitor service queuing latency and shed non-critical requests using priority queues with cached fallbacks.",
            "triz_principle": "Equipotentiality",
        },
    ],
    "generic": [
        {
            "name": "hierarchical_decomposition",
            "action": "act_decompose",
            "template": "Recursively split the state space into independent sub-domains, solve each monotonically, and merge via structural join.",
            "triz_principle": "Segmentation",
        },
        {
            "name": "boundary_invariance_screening",
            "action": "act_screen",
            "template": "Screen candidate states against known negative invariant manifolds before executing full verification.",
            "triz_principle": "Preliminary Anti-Action",
        },
    ],
}


# Category cue vocabulary.
#
# * ``words``  -- matched on word boundaries, so "bound" does not fire on
#                 "boundary" and "race" does not fire on "trace".
# * ``stems``  -- matched on a LEADING word boundary only, so "spike" matches
#                 "spikes"/"spiking" and "invoic" matches "invoice"/"invoices".
# * ``phrases``-- plain substring cues for multi-word signals.
_CATEGORY_CUES: Dict[str, Dict[str, tuple]] = {
    "latency": {
        "words": ("latency", "slow", "fast", "speed", "p99", "delay", "throughput"),
        "stems": ("latenc", "spike"),
        "phrases": (),
    },
    "concurrency": {
        "words": ("race", "races", "deadlock", "lock", "locks", "parallel", "contention"),
        "stems": ("concurren", "thread"),
        "phrases": (),
    },
    "optimization": {
        "words": (
            "shortest",
            "cheapest",
            "routing",
            "route",
            "routes",
            "bound",
            "bounds",
            "cost",
            "costs",
            "schedule",
            "allocation",
        ),
        "stems": ("optimi", "minimi", "maximi", "schedul"),
        "phrases": (),
    },
    "consistency": {
        "words": (
            "duplicate",
            "duplicates",
            "atomic",
            "atomicity",
            "idempotent",
            "transaction",
            "transactions",
            "consistency",
            "consistent",
        ),
        "stems": ("invoic", "consisten", "idempoten", "dedup"),
        "phrases": (),
    },
    "resilience": {
        "words": ("outage", "overheat", "battery", "recover", "recovery", "cluster"),
        "stems": ("resilien", "crash", "fail", "degrad"),
        "phrases": ("slow backend",),
    },
}


class SolutionMechanismSynthesizer:
    """Synthesizes actionable mechanism proposals from auxiliary subproblems."""

    def __init__(self):
        self.templates = _MECHANISM_TEMPLATES

    def infer_category(self, text: str) -> str:
        """Infers the engineering/algorithmic category from subproblem text.

        Matching is WORD-BOUNDARY aware and the category is chosen by best
        score, not by first match.  Both details are bug fixes:

        * The previous test was a raw substring check -- ``"bound" in text``
          fires on the word *"boundary"*, which is how a subproblem reading
          "Characterize the boundary/special cases of ..." was classified as an
          *optimisation* problem and handed a branch-and-bound / Lagrangian
          template.  That produced breakthroughs like "Characterize the
          boundary/special cases of ... to prune non-optimal subtrees in
          branch-and-bound".
        * First-match ordering made ``resilience`` unreachable whenever the text
          also contained "slow" (latency) or "fail" (also resilience): "Prevent
          one slow backend from crashing the whole cluster" was reported as
          ``latency``.

        Returns ``"generic"`` when no cue matches at all.
        """
        t = (text or "").lower()
        best_category = "generic"
        best_score = 0
        for category, cues in _CATEGORY_CUES.items():
            score = 0
            for word in cues["words"]:
                if re.search(rf"\b{re.escape(word)}\b", t):
                    score += 1
            for stem in cues["stems"]:
                if re.search(rf"\b{re.escape(stem)}", t):
                    score += 1
            for phrase in cues["phrases"]:
                if phrase in t:
                    score += 1
            if score > best_score:
                best_score = score
                best_category = category
        return best_category

    def category_from(self, subproblem_text: str, problem_context: str = "") -> str:
        """Category of a subproblem, falling back to the problem context.

        The context is only consulted when the subproblem itself carries no cue,
        so an uninformative subproblem still inherits the problem's domain
        without letting the context overrule what the subproblem is actually
        about.
        """
        category = self.infer_category(subproblem_text)
        if category == "generic":
            category = self.infer_category(problem_context)
        return category

    def synthesize_mechanisms_for_subproblem(
        self,
        subproblem_text: str,
        problem_context: str = "",
        max_mechanisms: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generates concrete candidate mechanisms for an auxiliary subproblem."""
        category = self.category_from(subproblem_text, problem_context)
        available = self.templates.get(category, self.templates["generic"])

        prims = strip_to_primitives(subproblem_text + " " + problem_context)
        results = []

        for item in available[:max_mechanisms]:
            mechanism_desc = (
                f"To address '{subproblem_text}': {item['template']} "
                f"(TRIZ: {item['triz_principle']})"
            )
            results.append(
                {
                    "mechanism_name": item["name"],
                    "category": category,
                    "action_primitive": item["action"],
                    "description": mechanism_desc,
                    "triz_principle": item["triz_principle"],
                    "matched_primitives": list(prims),
                }
            )

        return results

    def synthesize_for_decomposition(
        self,
        problem_specification: str,
        subproblems: List[Dict[str, Any]],
        max_per_subproblem: int = 2,
    ) -> List[Dict[str, Any]]:
        """Augments a list of decomposed subproblems with synthesized mechanisms."""
        synthesized_pool: List[Dict[str, Any]] = []

        for sub in subproblems:
            stext = sub.get("subproblem") or sub.get("title") or str(sub)
            mechs = self.synthesize_mechanisms_for_subproblem(
                stext,
                problem_context=problem_specification,
                max_mechanisms=max_per_subproblem,
            )
            for m in mechs:
                synthesized_pool.append(
                    {
                        "source_subproblem": stext,
                        "mechanism": m["description"],
                        "name": m["mechanism_name"],
                        "category": m["category"],
                        "triz_principle": m["triz_principle"],
                    }
                )

        return synthesized_pool
