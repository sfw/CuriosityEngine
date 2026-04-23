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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional


class ToolError(Exception):
    """Raised when a tool cannot complete its work. The message is returned to the model."""


class RateLimiter:
    """Thread-safe token-bucket rate limiter.

    Shared across threads — use one instance per throttled endpoint (e.g. one
    for arXiv, one for Semantic Scholar, one per host for web_fetch). Every
    tool call `.acquire()`s before hitting the network; concurrent callers
    serialize on this shared state.

    Under parallelism, the limiter is the hard guarantee that we don't burst
    the upstream endpoint no matter how many engine threads are firing tools.
    Serial runs also benefit: the LLM frequently emits multiple tool_use blocks
    per response that previously fired back-to-back without pacing.

    - rate: tokens refilled per second (e.g. 1/3 = "1 request every 3 seconds")
    - burst: max tokens the bucket can hold (1 = strict pacing; 5 = allow small
      clusters, recover later)
    - name: for debug logging when a wait occurs
    """

    def __init__(self, rate: float, burst: int = 1, name: str = ""):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(rate)
        self._capacity = max(1, int(burst))
        self._tokens = float(self._capacity)
        self._last = time.monotonic()
        self._cond = threading.Condition()
        self.name = name or f"limiter-{id(self):x}"

    def acquire(self, tokens: int = 1) -> float:
        """Block until `tokens` are available. Returns seconds waited."""
        waited = 0.0
        with self._cond:
            while True:
                now = time.monotonic()
                elapsed = now - self._last
                if elapsed > 0:
                    self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                    self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                needed = tokens - self._tokens
                wait = needed / self._rate
                # Cap any single wait to avoid pathological long blocks; the
                # while-loop will re-evaluate on wakeup.
                wait = min(wait, 5.0) + 0.01
                waited += wait
                self._cond.wait(timeout=wait)

    def __repr__(self):
        return f"RateLimiter(name={self.name!r}, rate={self._rate}, burst={self._capacity})"


class HostRateLimiter:
    """Per-host variant — lazily instantiates a RateLimiter for each hostname.
    Use for endpoints where throttling is per-host (web_fetch, for example).
    """

    def __init__(self, rate: float, burst: int = 1, name: str = ""):
        self._rate = rate
        self._burst = burst
        self._name = name
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

    def acquire(self, host: str, tokens: int = 1) -> float:
        with self._lock:
            limiter = self._limiters.get(host)
            if limiter is None:
                limiter = RateLimiter(
                    rate=self._rate, burst=self._burst,
                    name=f"{self._name or 'host'}:{host}",
                )
                self._limiters[host] = limiter
        return limiter.acquire(tokens)


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
