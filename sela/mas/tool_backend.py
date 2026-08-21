"""ToolBackend protocol and ToolResult type."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Protocol, runtime_checkable


@dataclass
class ToolResult:
    """Return value for tool callables that produce visual output."""
    text:   str
    images: List[str] = field(default_factory=list)
    images_svg: List[str] = field(default_factory=list)


@runtime_checkable
class ToolBackend(Protocol):
    def get_schemas(self) -> List[Dict]:
        """Return a list of OpenAI-compatible tool/function schemas."""
        ...

    def get_callables(self) -> Dict[str, Callable[..., Any]]:
        """Return a mapping from tool name to its Python callable."""
        ...


class SimpleToolBackend:
    """Convenience implementation: register tools manually by calling add()."""

    def __init__(self) -> None:
        self._schemas: List[Dict] = []
        self._callables: Dict[str, Callable] = {}

    def add(
        self,
        name: str,
        fn: Callable,
        description: str,
        parameters: Dict,
    ) -> "SimpleToolBackend":
        """Register a tool."""
        self._schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
        self._callables[name] = fn
        return self

    def get_schemas(self) -> List[Dict]:
        return list(self._schemas)

    def get_callables(self) -> Dict[str, Callable]:
        return dict(self._callables)
