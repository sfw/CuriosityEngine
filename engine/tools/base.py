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
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional


class ToolError(Exception):
    """Raised when a tool cannot complete its work. The message is returned to the model."""


# Staged cooldown schedule for back-off-on-429. Consecutive failures escalate
# the wait — first failure is cheap (2s); after the schedule's max (30s) is
# reached, the next failure CYCLES BACK to 2s. Cycling rather than plateauing
# at a long cap is the right choice when the upstream is a shared bucket
# (e.g. Semantic Scholar's unauthenticated tier) that can refill in seconds —
# we want to periodically re-probe rather than commit to a long wait.
# note_success() resets the counter to 0 so the next failure starts at 2s.
_COOLDOWN_SCHEDULE = (2.0, 4.0, 8.0, 16.0, 30.0)


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
    - jitter: max extra seconds of uniform-random delay added AFTER token
      acquisition. Breaks fixed-interval request patterns so we don't look
      like a bot to rate-limited endpoints. 0.0 disables. e.g. jitter=1.0
      with rate=1/3s gives effective pacing of 3.0–4.0s between calls.
    - name: for debug logging when a wait occurs
    """

    def __init__(self, rate: float, burst: int = 1, jitter: float = 0.0, name: str = ""):
        if rate <= 0:
            raise ValueError("rate must be positive")
        if jitter < 0:
            raise ValueError("jitter must be non-negative")
        self._rate = float(rate)
        self._capacity = max(1, int(burst))
        self._jitter = float(jitter)
        self._tokens = float(self._capacity)
        self._last = time.monotonic()
        self._cond = threading.Condition()
        # Cooldown timestamp for back-off-on-429. When the upstream endpoint
        # signals overload (HTTP 429 / explicit rate-limit error), the caller
        # invokes note_failure() which sets _cooldown_until to a future
        # monotonic-clock timestamp. Subsequent acquire() calls block until
        # that timestamp passes BEFORE consulting the token bucket. This is
        # the back-off behaviour that pure token-bucket pacing can't provide.
        #
        # Staged backoff: consecutive failures escalate the cooldown duration
        # via _COOLDOWN_SCHEDULE. Caller is expected to invoke note_success()
        # on a successful response to reset the counter; without that the
        # cooldown would stay at the cap forever.
        self._cooldown_until: float = 0.0
        self._consecutive_failures: int = 0
        self._cooldown_logged_for: float = 0.0
        self.name = name or f"limiter-{id(self):x}"

    def acquire(self, tokens: int = 1) -> float:
        """Block until `tokens` are available, then sleep a random jitter.
        Returns total seconds waited (cooldown + token-wait + jitter).
        """
        waited = 0.0
        # Cooldown phase — if note_failure was recently called, wait it out
        # BEFORE attempting to consume tokens. Logged ONCE per distinct
        # cooldown engagement (deduped on _cooldown_until timestamp) so the
        # log doesn't repeat the same line for every blocked acquire.
        with self._cond:
            now = time.monotonic()
            if self._cooldown_until > now:
                cool_wait = self._cooldown_until - now
                if self._cooldown_logged_for != self._cooldown_until:
                    print(f"  [rate-limit cooldown] {self.name}: backing off "
                          f"{cool_wait:.1f}s before next request "
                          f"(failure #{self._consecutive_failures})")
                    self._cooldown_logged_for = self._cooldown_until
                self._cond.wait(timeout=cool_wait)
                waited += cool_wait
            while True:
                now = time.monotonic()
                elapsed = now - self._last
                if elapsed > 0:
                    self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                    self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    break
                needed = tokens - self._tokens
                wait = needed / self._rate
                # Cap any single wait to avoid pathological long blocks; the
                # while-loop will re-evaluate on wakeup.
                wait = min(wait, 5.0) + 0.01
                waited += wait
                self._cond.wait(timeout=wait)
        # Jitter sleep OUTSIDE the condition lock — don't block other threads
        # from evaluating their own token state while we're fuzzing.
        if self._jitter > 0:
            fuzz = random.uniform(0, self._jitter)
            time.sleep(fuzz)
            waited += fuzz
        return waited

    def note_failure(self, cooldown_seconds: Optional[float] = None) -> float:
        """Record a rate-limit rejection from the upstream endpoint. All
        future acquire() calls will block until the chosen cooldown has
        elapsed before consuming tokens.

        If `cooldown_seconds` is None (the typical case), uses the staged
        backoff schedule based on consecutive failures: 2s, 4s, 8s, 16s,
        30s, then CYCLES back to 2s on the next failure. Cycling rather
        than plateauing at a long cap is right for shared-bucket
        endpoints (e.g. Semantic Scholar's unauthenticated tier) that
        can refill quickly — short waits between cycles give us periodic
        probes. Caller is expected to invoke note_success() on a
        successful response to reset the counter.

        If `cooldown_seconds` is explicitly provided, uses that exact value
        WITHOUT advancing the schedule — for callers that want a fixed
        cooldown regardless of history.

        Returns the cooldown duration that was applied (so the caller can
        log it).

        Idempotent: re-engaging within an existing cooldown window will
        only extend if the new deadline is later than the existing one.
        """
        with self._cond:
            if cooldown_seconds is None:
                # Use staged schedule, cycling back to schedule[0] after the
                # last entry. So failure sequence walks 2, 4, 8, 16, 30, 2,
                # 4, 8, ... rather than plateauing at the longest wait —
                # short waits give us periodic probes against shared-bucket
                # endpoints that can refill quickly.
                idx = self._consecutive_failures % len(_COOLDOWN_SCHEDULE)
                cooldown_seconds = _COOLDOWN_SCHEDULE[idx]
                self._consecutive_failures += 1
            if cooldown_seconds <= 0:
                return 0.0
            new_until = time.monotonic() + float(cooldown_seconds)
            if new_until > self._cooldown_until:
                self._cooldown_until = new_until
                # Reset the log-dedupe so the new cooldown gets one log line.
                self._cooldown_logged_for = 0.0
            self._cond.notify_all()
        return float(cooldown_seconds)

    def note_success(self):
        """Caller signals a successful API response — resets the
        consecutive-failure counter so the next failure (if any) starts
        from the bottom of the staged backoff schedule (2s)."""
        with self._cond:
            self._consecutive_failures = 0

    def __repr__(self):
        return (
            f"RateLimiter(name={self.name!r}, rate={self._rate}, "
            f"burst={self._capacity}, jitter={self._jitter})"
        )


class HostRateLimiter:
    """Per-host variant — lazily instantiates a RateLimiter for each hostname.
    Use for endpoints where throttling is per-host (web_fetch, for example).
    """

    def __init__(self, rate: float, burst: int = 1, jitter: float = 0.0, name: str = ""):
        self._rate = rate
        self._burst = burst
        self._jitter = jitter
        self._name = name
        self._limiters: dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

    def acquire(self, host: str, tokens: int = 1) -> float:
        with self._lock:
            limiter = self._limiters.get(host)
            if limiter is None:
                limiter = RateLimiter(
                    rate=self._rate, burst=self._burst, jitter=self._jitter,
                    name=f"{self._name or 'host'}:{host}",
                )
                self._limiters[host] = limiter
        return limiter.acquire(tokens)

    def note_failure(self, host: str, cooldown_seconds: float = 60.0):
        """Forward a rate-limit failure signal to the per-host limiter."""
        with self._lock:
            limiter = self._limiters.get(host)
        if limiter is not None:
            limiter.note_failure(cooldown_seconds)


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
