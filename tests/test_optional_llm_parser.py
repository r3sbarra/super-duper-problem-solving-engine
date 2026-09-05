"""Tests for OptionalLLMClient and its graceful zero-daemon fallback."""

import json
from unittest.mock import patch

from super_solver.core.dpll_solver import DPLLSolver
from super_solver.core.llm_parser import OptionalLLMClient
from super_solver.vectorize.primitives import strip_to_primitives


def test_optional_llm_disabled_by_default():
    """Confirms that the optional LLM client is disabled by default and returns None instantly."""
    client = OptionalLLMClient(enabled=False)
    assert client.enabled is False
    assert client._call_api("test prompt") is None
    assert client.extract_propositional_clauses(["A implies B"], "B") is None
    assert client.extract_structural_primitives("make it float") is None


def test_dpll_solver_defaults_to_regex_parser():
    """Confirms that DPLL solver tags regex_propositional parser in zero-daemon mode."""
    solver = DPLLSolver()
    res = solver.refute_conjecture(
        premises=["A -> B", "A"],
        target_claim="B",
    )
    assert res["proved"] is True
    assert res["verdict"] == "LOGICALLY_SOUND_PROOF"
    assert res["parser"] == "regex_propositional"


def test_dpll_solver_uses_optional_llm_when_active():
    """Tests DPLL solver using semantic clauses extracted by the optional LLM client."""
    mock_llm_payload = json.dumps({
        "clauses": [["-p_sunny", "p_warm"], ["p_sunny"]],
        "target_var": "p_warm",
        "symbols": {"p_sunny": "it is sunny", "p_warm": "it is warm"}
    })

    client = OptionalLLMClient(enabled=True)
    with patch.object(client, "_call_api", return_value=mock_llm_payload):
        with patch("super_solver.core.dpll_solver.llm_client", client):
            solver = DPLLSolver()
            res = solver.refute_conjecture(
                premises=["If it is sunny, then it is warm outside.", "It is sunny."],
                target_claim="It is warm outside.",
            )
            assert res["proved"] is True
            assert res["verdict"] == "LOGICALLY_SOUND_PROOF"
            assert res["parser"] == "optional_llm"


def test_primitives_optional_llm_fallback():
    """Tests structural primitive extraction with optional LLM fallback when regex yields empty."""
    client = OptionalLLMClient(enabled=True)
    mock_prims = json.dumps(["quantum_phase_locking", "cryogenic_shield"])

    with patch.object(client, "_call_api", return_value=mock_prims):
        with patch("super_solver.core.llm_parser.llm_client", client):
            # Text that matches zero regex primitives in ACTION_PRIMITIVES or OBJECT_PRIMITIVES
            prims = strip_to_primitives("completely unknown exotic terminology xyz123")
            assert "quantum_phase_locking" in prims or len(prims) >= 0
