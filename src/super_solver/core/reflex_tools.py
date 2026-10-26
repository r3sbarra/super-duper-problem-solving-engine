"""Tool catalogs exposed to the reflex router.

Two catalogs, deliberately separate:

* ``mcp_catalog()``   -- the 20 ``super_solver_*`` MCP tools, read from the
  committed ``assets/mcp_schema.json`` so the router can never drift from the
  server it is routing to. A test asserts the two agree.
* ``sandbox_catalog()`` -- the executable tools in ``core.tool_registry``, which
  is where a reflex decision actually has side effects. Each entry carries the
  risk tier and default execution policy that the invariant gate enforces.

The distinction matters: choosing an MCP tool is a read-only recommendation,
while choosing a sandbox tool starts a subprocess. Only the latter can be
gated on risk, and only the latter is eligible for speculative prefetch.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

MCP_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "assets",
    "mcp_schema.json",
)


def mcp_catalog(path: Optional[str] = None) -> Dict[str, str]:
    """Map MCP tool name -> its description (used as the routing text)."""
    p = path or MCP_SCHEMA_PATH
    with open(p, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return {name: (spec.get("description") or name) for name, spec in schema.items()}


def sandbox_catalog(registry: Optional[Any] = None) -> Dict[str, str]:
    """Map executable tool name -> description from the live registry."""
    from super_solver.core.tool_registry import default_tool_registry

    reg = registry or default_tool_registry
    return {t.name: f"{t.description} {' '.join(t.primitives)}".strip() for t in reg.list_all()}


# Risk tier and policy per executable tool. Anything not listed defaults to
# workspace_mutation/synchronous_execute -- the conservative choice.
TOOL_POLICIES: Dict[str, Dict[str, str]] = {
    "run_python_code": {
        "risk_tier": "workspace_mutation",
        "execution_policy": "synchronous_execute",
        "note": "spawns a subprocess; may write files from inside the script",
    },
    "read_file_content": {
        "risk_tier": "read_only",
        "execution_policy": "speculative_prefetch",
        "note": "idempotent filesystem read",
    },
    "search_pattern": {
        "risk_tier": "read_only",
        "execution_policy": "speculative_prefetch",
        "note": "pure string search over caller-supplied text",
    },
}

DEFAULT_POLICY = {
    "risk_tier": "workspace_mutation",
    "execution_policy": "synchronous_execute",
    "note": "unclassified tool: treated as a mutation",
}


def policy_for(tool_name: str) -> Dict[str, str]:
    return dict(TOOL_POLICIES.get(tool_name, DEFAULT_POLICY))


def can_speculate(tool_name: str) -> bool:
    """Speculative prefetch is allowed only for read-only, idempotent tools."""
    p = policy_for(tool_name)
    return p["risk_tier"] == "read_only" and p["execution_policy"] == "speculative_prefetch"


def invariant_rules() -> List[Any]:
    """Hard invariants enforced on every executable reflex decision.

    Each rule takes the decision state ``{target, risk_tier, execution_policy,
    query}`` and returns True when admissible.
    """

    def read_only_tools_are_read_only(state: Dict[str, Any]) -> bool:
        target = state.get("target", "")
        if target in ("read_file_content", "search_pattern"):
            return state.get("risk_tier") == "read_only"
        return True

    def no_speculation_on_mutation(state: Dict[str, Any]) -> bool:
        if state.get("risk_tier") == "read_only":
            return True
        return state.get("execution_policy") != "speculative_prefetch"

    def hazards_require_confirmation(state: Dict[str, Any]) -> bool:
        if state.get("risk_tier") != "destructive_hazard":
            return True
        return state.get("execution_policy") == "require_confirmation"

    def no_hazard_text_for_reflex(state: Dict[str, Any]) -> bool:
        # A destructive shell string must never be dispatched reflexively, even
        # if the chosen tool looks safe: the payload, not the tool, is the risk.
        return "destructive_hazard" != state.get("classified_risk")

    def unknown_tool_is_not_reflex(state: Dict[str, Any]) -> bool:
        return state.get("target", "") in TOOL_POLICIES

    def mutation_or_external_never_reflexes_a_read(state: Dict[str, Any]) -> bool:
        """A read-only tool must not be reflexively chosen for a task that
        MUTATES or LEAVES the machine.

        Without this, "email the report to the client" routed to
        read_file_content and was certified EXECUTE_SYSTEM_1 -- technically a
        safe tool, but the reflex would have answered the wrong question and
        silently skipped the LLM that should have realised the user wanted a
        send. The tool being harmless does not make the DECISION harmless.
        """
        if state.get("risk_tier") == "read_only":
            return state.get("classified_risk") == "read_only"
        return True

    return [
        read_only_tools_are_read_only,
        no_speculation_on_mutation,
        hazards_require_confirmation,
        no_hazard_text_for_reflex,
        unknown_tool_is_not_reflex,
        mutation_or_external_never_reflexes_a_read,
    ]
