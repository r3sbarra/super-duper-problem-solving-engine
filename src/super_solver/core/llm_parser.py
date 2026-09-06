"""Optional small LLM client for semantic parsing and open-vocabulary extraction.

Provides an optional bridge to a local small language model (Ollama or any
OpenAI-compatible endpoint) to assist in:
1. Translating natural language premises and target claims into propositional/Horn clauses.
2. Extracting open-vocabulary action-object structural primitives when regex dictionaries miss.

Zero-Daemon Guarantee:
- Gated by default: disabled unless ``SUPER_SOLVER_LLM_ENABLED`` is set to "1" or "true".
- If disabled, unreachable, or times out, immediately returns None with zero delay.
- The engine's core zero-token deterministic behavior remains 100% functional without it.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OptionalLLMClient:
    """Lightweight, zero-dependency client for local Ollama and OpenAI-compatible endpoints."""

    def __init__(
        self,
        enabled: Optional[bool] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        num_predict: int = 1024,
    ):
        self._enabled = enabled

        # Support both Ollama and generic OpenAI-compatible base URLs
        self.base_url = (
            base_url
            or os.environ.get("SUPER_SOLVER_LLM_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "http://localhost:11434"
        ).rstrip("/")

        self.model = (
            model
            or os.environ.get("SUPER_SOLVER_LLM_MODEL")
            or "qwen3:4b"
        )
        if timeout is not None:
            self.timeout = timeout
        else:
            self.timeout = float(os.environ.get("SUPER_SOLVER_LLM_TIMEOUT", "30.0"))
        self.num_predict = int(os.environ.get("SUPER_SOLVER_LLM_NUM_PREDICT", str(num_predict)))

    @property
    def enabled(self) -> bool:
        if self._enabled is not None:
            return self._enabled
        env_val = os.environ.get("SUPER_SOLVER_LLM_ENABLED", "false").lower()
        return env_val in ("1", "true", "yes", "on")

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = val

    def _call_api(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_format: bool = False,
    ) -> Optional[str]:
        """Makes an HTTP request to either OpenAI-compatible /v1/chat/completions or Ollama /api/generate."""
        if not self.enabled:
            return None

        # Try OpenAI-compatible endpoint first if /v1 in url or OPENAI_BASE_URL is set
        if "/v1" in self.base_url or "OPENAI_BASE_URL" in os.environ:
            endpoint = f"{self.base_url}/chat/completions"
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system or "You are a precise formal logic assistant. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": self.num_predict,
            }
            if json_format:
                payload["response_format"] = {"type": "json_object"}

            def _extract(res: Dict[str, Any]) -> str:
                msg = res.get("choices", [{}])[0].get("message", {})
                content = msg.get("content", "") or ""
                reasoning = msg.get("reasoning_content", "") or msg.get("thinking", "") or ""
                return (content.strip() or reasoning.strip())
        else:
            # Native Ollama endpoint
            endpoint = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": f"{system}\n\n{prompt}" if system else prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": self.num_predict,
                },
            }
            if json_format:
                payload["format"] = "json"

            def _extract(res: Dict[str, Any]) -> str:
                resp = res.get("response", "") or ""
                thinking = res.get("thinking", "") or ""
                # If the visible response is populated, prefer it; otherwise fallback to thinking trace
                return (resp.strip() or thinking.strip())

        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return _extract(data)
        except Exception as exc:
            logger.debug(f"Optional LLM call failed or unavailable: {exc}")
            return None

    @staticmethod
    def clean_json_string(raw: str) -> str:
        """Strips markdown fences and whitespace from raw JSON response."""
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]
        return cleaned.strip()

    def generate_text(
        self, prompt: str, system: Optional[str] = None, json_format: bool = False
    ) -> Optional[str]:
        """Direct text/code generation helper with local model formatting."""
        return self._call_api(prompt, system=system, json_format=json_format)

    def call_with_tools(
        self,
        query: str,
        tool_schemas: List[Dict[str, Any]],
        system: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Prompts the small model to select and format a tool call from pruned schemas.
        
        Returns:
            {"name": tool_name, "arguments": {...}} or None if refused/failed.
        """
        if not self.enabled:
            return None

        tools_desc = []
        for t in tool_schemas:
            tools_desc.append({
                "name": t.get("name"),
                "description": t.get("description"),
                "parameters": t.get("parameters", {}),
            })

        tool_prompt = (
            f"You have access to the following tools:\n"
            f"{json.dumps(tools_desc, indent=2)}\n\n"
            f"User Goal: {query}\n\n"
            f"Select the appropriate tool to call. Respond strictly with a JSON object in this format:\n"
            f'{{"tool": "<tool_name>", "arguments": {{<arguments>}}}}\n'
            f"If no tool is required or applicable, respond with: {{\"tool\": \"none\", \"arguments\": {{}}}}"
        )
        sys_instruction = system or "You are an agentic tool dispatcher. Output only valid JSON."
        resp = self._call_api(tool_prompt, system=sys_instruction, json_format=True)
        if not resp:
            return None

        try:
            parsed = json.loads(self.clean_json_string(resp))
            tool_name = parsed.get("tool") or parsed.get("name")
            args = parsed.get("arguments") or parsed.get("parameters") or {}
            if tool_name:
                return {"name": str(tool_name).strip(), "arguments": args}
        except Exception as exc:
            logger.debug(f"Failed to parse tool call from LLM response: {exc}")
        return None

    def extract_propositional_clauses(
        self, premises: List[str], target_claim: str
    ) -> Optional[Dict[str, Any]]:
        """Extracts formal propositional relations from natural language text.

        Returns {
            "clauses": [["p1", "p2"], ["-p1", "p3"]],
            "target_var": "p_target",
            "symbols": {"p1": "description"}
        } or None on failure/disabled.
        """
        if not self.enabled:
            return None

        system = (
            "You are a mathematical logician. Extract propositions and rules from informal premises into simple "
            "propositional logic clauses. Output strictly a JSON object with keys: "
            "'clauses' (list of lists of string literals, prefix negative literals with '-'), "
            "'target_var' (string identifier for the target claim), and "
            "'symbols' (dictionary mapping identifiers to short descriptions)."
        )
        prompt = (
            "Premises:\n" + "\n".join(f"- {p}" for p in premises) + f"\n\nTarget Claim: {target_claim}\n"
            "Format: JSON only."
        )

        resp = self._call_api(prompt, system=system, json_format=True)
        if not resp:
            return None

        try:
            parsed = json.loads(self.clean_json_string(resp))
            if "clauses" in parsed and "target_var" in parsed:
                return parsed
        except Exception:
            pass
        return None

    def extract_structural_primitives(self, text: str) -> Optional[List[str]]:
        """Extracts domain-agnostic action and object primitives when regex misses.

        Returns a list of primitive strings (e.g. ['reduce_density', 'porous_structure']) or None.
        """
        if not self.enabled:
            return None

        system = (
            "You are an expert in TRIZ and Gentner's Structure-Mapping Theory. "
            "Extract 2 to 5 domain-agnostic structural action primitives (verb_noun format like 'reduce_density', "
            "'mimic_structure', 'surface_coating', 'phase_change') from the given text. "
            "Discard surface nouns and focus on the underlying functional action. Output strictly a JSON list of strings."
        )
        prompt = f"Text: {text}\nFormat: JSON array of strings only."

        resp = self._call_api(prompt, system=system)
        if not resp:
            return None

        try:
            cleaned = resp
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0]
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0]
            parsed = json.loads(cleaned.strip())
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return [p.lower().replace(" ", "_") for p in parsed]
        except Exception:
            pass
        return None


# Global default client
llm_client = OptionalLLMClient()
