"""Tool ABC + ToolRegistry + discovery for the Curiosity Engine.

Design goals:
- Subclassing `Tool` auto-registers the class with the global `registry`.
- Sync execution with a per-tool timeout.
- Two schema flavors emitted: Anthropic (`{"name", "description", "input_schema"}`)
  and OpenAI (`{"type": "function", "function": {"name", "description", "parameters"}}`).
- `discover_tools()` imports all modules under engine/tools/ to trigger registration.
"""

from __future__ import annotations

import importlib
import pkgutil
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional


class ToolError(Exception):
    """Raised when a tool cannot complete its work. The message is returned to the model."""


@dataclass
class ToolResult:
    """Uniform wrapper around a tool call's outcome."""

    content: str
    is_error: bool = False

    def to_text(self) -> str:
        return self.content


class Tool(ABC):
    """Base class for all engine tools.

    Subclasses set class-level attributes and implement `execute(args) -> str`.
    Subclassing auto-registers the class with the module-level `registry`.
    """

    # Class-level attributes — required on every subclass.
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    input_schema: ClassVar[dict] = {"type": "object", "properties": {}, "additionalProperties": False}
    timeout_seconds: ClassVar[float] = 30.0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Skip abstract intermediate classes (those without a concrete name).
        if not cls.name:
            return
        registry.register(cls)

    @abstractmethod
    def execute(self, args: dict) -> str:
        """Run the tool and return a string result. Raise ToolError on failure."""


class ToolRegistry:
    """Keyed map of tool classes. Thread-safe registration; single instance per process."""

    def __init__(self):
        self._by_name: dict[str, type[Tool]] = {}
        self._lock = threading.Lock()

    def register(self, tool_cls: type[Tool]):
        if not tool_cls.name:
            return
        with self._lock:
            if tool_cls.name in self._by_name and self._by_name[tool_cls.name] is not tool_cls:
                # Silently skip duplicate registration; last-import-wins would be surprising.
                return
            self._by_name[tool_cls.name] = tool_cls

    def get(self, name: str) -> Optional[type[Tool]]:
        return self._by_name.get(name)

    def all(self) -> list[type[Tool]]:
        return list(self._by_name.values())

    def names(self) -> list[str]:
        return sorted(self._by_name.keys())

    def anthropic_schemas(self) -> list[dict]:
        """Schemas in the Anthropic tools format."""
        return [
            {
                "name": cls.name,
                "description": cls.description,
                "input_schema": cls.input_schema,
            }
            for cls in self._by_name.values()
        ]

    def openai_schemas(self) -> list[dict]:
        """Schemas in the OpenAI chat-completions tools format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": cls.name,
                    "description": cls.description,
                    "parameters": cls.input_schema,
                },
            }
            for cls in self._by_name.values()
        ]

    def execute(self, name: str, args: dict) -> ToolResult:
        """Run a tool by name. Return ToolResult (is_error=True on failure)."""
        cls = self.get(name)
        if cls is None:
            return ToolResult(content=f"error: unknown tool '{name}'", is_error=True)
        instance = cls()
        try:
            output = instance.execute(args or {})
        except ToolError as e:
            return ToolResult(content=f"tool_error: {e}", is_error=True)
        except Exception as e:  # noqa: BLE001 — surface any failure to the model as a string.
            return ToolResult(content=f"unexpected_error: {type(e).__name__}: {e}", is_error=True)
        if not isinstance(output, str):
            output = str(output)
        return ToolResult(content=output, is_error=False)


registry = ToolRegistry()


def discover_tools(package: str = "engine.tools") -> list[str]:
    """Import every submodule under `package` so Tool subclasses register themselves.

    Returns the list of registered tool names.
    """
    pkg = importlib.import_module(package)
    for _finder, modname, _ispkg in pkgutil.iter_modules(pkg.__path__):
        if modname in ("base", "__init__"):
            continue
        importlib.import_module(f"{package}.{modname}")
    return registry.names()
