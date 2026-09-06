"""Tool Registry and Dynamic Pruning for Small/Low-End Models.

Designed specifically for small language models (e.g. 3B-7B) to prevent:
1. Schema bloat (prunes 20+ tool catalogs to top 1-3 candidates via semantic matching).
2. Malformed arguments (validates parameter schemas and required fields).
3. Tool loops (circuit breaker detects repeated failing/identical calls and deflects).
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from super_solver.core.embeddings import embedding_service

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]
    primitives: List[str] = field(default_factory=list)
    is_safe: bool = True

    def to_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolCallResult:
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class ToolLoopCircuitBreaker:
    """Detects and breaks repetitive tool-calling cycles."""

    def __init__(self, max_repetitions: int = 3, max_total_steps: int = 10):
        self.max_repetitions = max_repetitions
        self.max_total_steps = max_total_steps
        self.history: List[Tuple[str, str]] = []

    def record_and_check(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Returns (is_loop, reason) if loop or step limit exceeded."""
        if len(self.history) >= self.max_total_steps:
            return True, f"Max agentic tool steps ({self.max_total_steps}) reached."

        arg_sig = json.dumps(arguments, sort_keys=True)
        self.history.append((tool_name, arg_sig))

        # Check for repetition of identical (tool, args)
        recent_matches = sum(1 for t, a in self.history[-self.max_repetitions:] if t == tool_name and a == arg_sig)
        if recent_matches >= self.max_repetitions:
            return True, f"Repetitive tool call loop detected for '{tool_name}' ({recent_matches} consecutive identical calls)."

        return False, None

    def reset(self) -> None:
        self.history.clear()


class ToolRegistry:
    """Registry managing tools with dynamic pruning and verified execution."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
        primitives: Optional[List[str]] = None,
        is_safe: bool = True,
    ) -> None:
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            primitives=primitives or [],
            is_safe=is_safe,
        )

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_all(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def prune_tools_for_query(self, query: str, top_k: int = 3) -> List[ToolDefinition]:
        """Prunes the tool registry down to the top_k most relevant tools for the query.
        
        Uses embedding cosine similarity between the user query and tool descriptions/primitives.
        This keeps prompt token count low and prevents small models from hallucinating tools.
        """
        if not self._tools:
            return []

        if len(self._tools) <= top_k:
            return list(self._tools.values())

        query_vec = embedding_service.encode(query)
        scored: List[Tuple[float, ToolDefinition]] = []

        for tool in self._tools.values():
            tool_doc = f"{tool.name}: {tool.description} {' '.join(tool.primitives)}"
            tool_vec = embedding_service.encode(tool_doc)
            sim = float(embedding_service.cosine_similarity(query_vec, tool_vec))
            scored.append((sim, tool))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [tool for _, tool in scored[:top_k]]

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Checks tool existence, required arguments, and types against schema."""
        tool = self.get_tool(tool_name)
        if not tool:
            available = ", ".join(self._tools.keys())
            return False, f"Tool '{tool_name}' does not exist. Available tools: [{available}]"

        schema = tool.parameters
        required = schema.get("required", [])
        for req in required:
            if req not in arguments:
                return False, f"Missing required parameter '{req}' for tool '{tool_name}'"

        return True, None

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCallResult:
        """Executes a tool call with timing, argument validation, and exception safety."""
        valid, err = self.validate_tool_call(tool_name, arguments)
        if not valid:
            return ToolCallResult(tool_name=tool_name, success=False, error=err)

        tool = self._tools[tool_name]
        t0 = time.perf_counter()
        try:
            # Filter kwargs to match handler signature if necessary
            sig = inspect.signature(tool.handler)
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                res = tool.handler(**arguments)
            else:
                filtered_args = {k: v for k, v in arguments.items() if k in sig.parameters}
                res = tool.handler(**filtered_args)

            dur_ms = round((time.perf_counter() - t0) * 1000, 2)
            return ToolCallResult(tool_name=tool_name, success=True, output=res, execution_time_ms=dur_ms)
        except Exception as exc:
            dur_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.debug(f"Tool {tool_name} execution failed: {exc}")
            return ToolCallResult(tool_name=tool_name, success=False, error=str(exc), execution_time_ms=dur_ms)


def create_standard_tool_registry() -> ToolRegistry:
    """Constructs a standard tool registry equipped with sandbox and inspection tools."""
    registry = ToolRegistry()

    # 1. Python Sandbox Execution Tool
    def run_python_code(code: str) -> Dict[str, Any]:
        from super_solver.frameworks.sandbox import CodeExperimentSandbox
        sandbox = CodeExperimentSandbox(timeout=10.0)
        res = sandbox.execute_python_code(code)
        return {
            "success": res.success,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "execution_time_ms": res.execution_time_ms,
            "error": res.error_message,
        }

    registry.register(
        name="run_python_code",
        description="Executes a self-contained Python script in an isolated sandbox and returns stdout/stderr.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The self-contained Python script to execute."}
            },
            "required": ["code"],
        },
        handler=run_python_code,
        primitives=["code_execution", "run_script", "compute", "simulate"],
    )

    # 2. File Reader Tool
    def read_file_content(file_path: str, max_lines: int = 150) -> Dict[str, Any]:
        import os
        if not os.path.exists(file_path):
            return {"success": False, "error": f"File '{file_path}' not found."}
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = [next(f) for _ in range(max_lines)]
            return {"success": True, "lines_read": len(lines), "content": "".join(lines)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    registry.register(
        name="read_file_content",
        description="Reads the first N lines of a file from the filesystem.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute or relative file path to read."},
                "max_lines": {"type": "integer", "description": "Maximum number of lines to read (default 150).", "default": 150}
            },
            "required": ["file_path"],
        },
        handler=read_file_content,
        primitives=["read_file", "inspect_code", "examine_logs"],
    )

    # 3. String & Pattern Search Tool
    def search_pattern(query: str, text: str) -> Dict[str, Any]:
        matches = [line for line in text.splitlines() if query.lower() in line.lower()]
        return {"match_count": len(matches), "matches": matches[:20]}

    registry.register(
        name="search_pattern",
        description="Searches for query matches within a block of text or log output.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term or substring."},
                "text": {"type": "string", "description": "Text body to search within."}
            },
            "required": ["query", "text"],
        },
        handler=search_pattern,
        primitives=["search_log", "filter_trace", "find_error"],
    )

    return registry


# Global default registry
default_tool_registry = create_standard_tool_registry()
