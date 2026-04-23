"""Sync retry wrapper for LLM SDK calls (provider-agnostic).

Ported from loom/src/loom/models/retry.py — trimmed to sync, duck-typed so it works
with any SDK that exposes status_code / response.headers on its exceptions (Anthropic,
OpenAI, and their derivatives).
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")

# Class-name substrings that indicate transient provider-side errors.
_RETRYABLE_CLASS_MARKERS = (
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
    "ConnectError",   # httpx
    "ReadTimeout",    # httpx
    "WriteTimeout",   # httpx
)

_BACKPRESSURE_STATUS_CODES = frozenset({429, 503, 529})
_BACKPRESSURE_MARKERS = (
    "rate limit",
    "rate-limited",
    "too many requests",
    "overloaded",
    "overload",
    "throttled",
    "try again later",
)
_RETRY_AFTER_RE = re.compile(r"retry[- ]after[:=)\s]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_BACKPRESSURE_MIN_ATTEMPTS = 8
_BACKPRESSURE_BASE_DELAY = 2.0


@dataclass(frozen=True)
class RetryPolicy:
    # Defaults sized for provider-overload events (Moonshot/Kimi, OpenAI during
    # peak load, Claude during extended thinking). 429 / 503 / 529 trip the
    # backpressure path which uses max_delay_seconds as its ceiling — 90s
    # gives providers time to recover from sustained overload without failing
    # the run. max_attempts=10 × up-to-90s backoff = ~10-15 min patience on
    # the worst stretches; good enough for overnight endurance runs.
    max_attempts: int = 10
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 90.0
    jitter_seconds: float = 0.25


def _status_code(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None) if response is not None else None
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(error: BaseException) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is not None and hasattr(headers, "get"):
        raw = headers.get("retry-after")
        if raw:
            try:
                return max(0.0, float(raw))
            except (TypeError, ValueError):
                pass
    match = _RETRY_AFTER_RE.search(str(error or ""))
    if match is None:
        return None
    try:
        return max(0.0, float(match.group(1)))
    except (TypeError, ValueError):
        return None


def _is_backpressure(error: BaseException) -> bool:
    if _status_code(error) in _BACKPRESSURE_STATUS_CODES:
        return True
    text = str(error or "").lower()
    return any(marker in text for marker in _BACKPRESSURE_MARKERS)


def _is_retryable(error: BaseException) -> bool:
    class_name = type(error).__name__
    if any(marker in class_name for marker in _RETRYABLE_CLASS_MARKERS):
        return True
    status = _status_code(error)
    if status is not None and (status >= 500 or status in _BACKPRESSURE_STATUS_CODES):
        return True
    return _is_backpressure(error)


def _compute_delay(error: BaseException, attempt: int, policy: RetryPolicy) -> float:
    delay = min(policy.max_delay_seconds, policy.base_delay_seconds * (2 ** (attempt - 1)))
    retry_after = _retry_after_seconds(error)
    if retry_after is not None:
        delay = max(delay, retry_after)
    elif _is_backpressure(error):
        delay = max(
            delay,
            min(
                _BACKPRESSURE_BASE_DELAY * (2 ** (attempt - 1)),
                max(policy.max_delay_seconds, _BACKPRESSURE_BASE_DELAY),
            ),
        )
    if policy.jitter_seconds > 0:
        delay += random.uniform(0.0, policy.jitter_seconds)
    return max(0.0, delay)


def call_with_retry(
    invoke: Callable[[], T],
    *,
    policy: RetryPolicy = RetryPolicy(),
    on_retry: Callable[[int, int, BaseException, float], None] | None = None,
) -> T:
    """Call `invoke()` with exponential backoff on retryable provider errors."""
    max_attempts = max(1, policy.max_attempts)
    extended = False
    attempt = 0
    last_error: BaseException | None = None

    while attempt < max_attempts:
        attempt += 1
        try:
            return invoke()
        except Exception as error:
            last_error = error
            if _is_backpressure(error) and not extended:
                max_attempts = max(max_attempts, _BACKPRESSURE_MIN_ATTEMPTS)
                extended = True
            if not _is_retryable(error) or attempt >= max_attempts:
                raise
            delay = _compute_delay(error, attempt, policy)
            if on_retry is not None:
                on_retry(attempt, max_attempts, error, delay)
            if delay > 0:
                time.sleep(delay)

    assert last_error is not None
    raise last_error
