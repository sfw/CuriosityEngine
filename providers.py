"""Model client abstraction: one interface across Anthropic + OpenAI-compat endpoints.

OpenAI-compat covers OpenAI itself plus Gemini (openai-compat endpoint), OpenRouter,
Ollama (openai mode), xAI, Groq, Together, DeepSeek, LM Studio, etc. — anything that
speaks the OpenAI chat-completions protocol.

Each client exposes two completion paths:
- `complete_json(prompt, ...)`: single-call completion, JSON dict return.
- `complete_json_with_tools(prompt, ...)`: multi-turn tool-use loop, JSON dict return.

Both Anthropic server tools (web_search, code_execution — provided as raw dicts) and
registered client tools (from `engine.tools.registry`) can be passed together. The
tool-use loop executes client tools and passes results back to the model. Server tools
are handled Anthropic-side.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from json_utils import parse_json_response
from retry_utils import RetryPolicy, call_with_retry

_DEFAULT_MAX_TOOL_ITERATIONS = 40
_WRAP_UP_MARGIN = 5  # iterations before the hard cap at which we tell the model to wrap up


def _openai_token_param_name(model_name: str) -> str:
    """OpenAI renamed `max_tokens` → `max_completion_tokens` for reasoning models
    (GPT-5 family, o1/o3/o4). Older models still take `max_tokens`. Other OpenAI-compat
    providers (Moonshot/Kimi, Gemini, Groq, etc.) accept `max_tokens` — the new name
    is OpenAI-specific for now."""
    n = (model_name or "").strip().lower()
    if n.startswith(("gpt-5", "o1", "o3", "o4")):
        return "max_completion_tokens"
    return "max_tokens"


def _is_reasoning_model(model_name: str) -> bool:
    """Heuristic for models whose internal chain-of-thought counts against the
    output token budget. These need substantially larger max_tokens floors."""
    n = (model_name or "").strip().lower()
    return n.startswith(("gpt-5", "o1", "o3", "o4", "kimi-k2", "kimi-k3"))


_REASONING_MIN_TOKENS = 16000


def _effective_max_tokens(model_name: str, requested: int) -> int:
    """Clamp to a minimum for reasoning models so the thinking budget doesn't
    consume everything before a visible answer gets emitted."""
    if _is_reasoning_model(model_name):
        return max(requested, _REASONING_MIN_TOKENS)
    return requested


@dataclass(frozen=True)
class ModelProfile:
    provider: str                       # "anthropic" | "openai_compat"
    name: str                           # e.g. "claude-sonnet-4-6", "gpt-5.1", "gemini-2.5-pro"
    api_key: str = ""
    base_url: str = ""                  # OpenAI-compat: override endpoint; anthropic: rarely used
    max_tokens: int = 4096
    investigation_max_tokens: int = 8192
    # 1.0 is the safe default for most modern reasoning-first models (Kimi K2.x,
    # OpenAI o1/o3/GPT-5 thinking). Setting anything else can make these models
    # return empty content. For non-thinking models, drop to 0.3-0.7 for determinism.
    temperature: float = 1.0
    # Per-request HTTP timeout in seconds. Reasoning models (Kimi, o-series,
    # GPT-5 thinking, Claude extended thinking) can spend 60-180s "thinking"
    # before the first token streams — the SDK default of 60s causes
    # APITimeoutError on large cross-ref prompts. 300s is a generous ceiling
    # that covers common reasoning workloads without masking true hangs.
    timeout_seconds: float = 300.0


class ModelClient(ABC):
    """Sync interface for a single model profile. Returns parsed JSON dicts."""

    profile: ModelProfile
    supports_server_web_search: bool = False

    @abstractmethod
    def complete_json(
        self,
        prompt: str,
        *,
        tools: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
        policy: RetryPolicy,
        on_retry=None,
    ) -> dict:
        """Single-turn JSON completion. `tools` here are server-tool dicts (Anthropic only)."""

    def complete_json_with_tools(
        self,
        prompt: str,
        *,
        client_tools: Optional[list[dict]] = None,     # schemas (provider-appropriate)
        server_tools: Optional[list[dict]] = None,     # Anthropic-only server tools
        tool_registry=None,                            # ToolRegistry used to execute client tools
        max_tokens: Optional[int] = None,
        max_iterations: int = _DEFAULT_MAX_TOOL_ITERATIONS,
        policy: RetryPolicy,
        on_retry=None,
    ) -> dict:
        """Multi-turn tool-use loop. Default implementation is single-turn with server tools only."""
        return self.complete_json(
            prompt,
            tools=server_tools,
            max_tokens=max_tokens,
            policy=policy,
            on_retry=on_retry,
        )


class AnthropicClient(ModelClient):
    supports_server_web_search = True

    def __init__(self, profile: ModelProfile):
        import anthropic
        self.profile = profile
        kwargs: dict = {}
        if profile.api_key:
            kwargs["api_key"] = profile.api_key
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        if profile.timeout_seconds > 0:
            kwargs["timeout"] = profile.timeout_seconds
        self._client = anthropic.Anthropic(**kwargs)

    def complete_json(
        self,
        prompt,
        *,
        tools=None,
        max_tokens=None,
        policy,
        on_retry=None,
    ) -> dict:
        def _invoke():
            kwargs = {
                "model": self.profile.name,
                "max_tokens": max_tokens or self.profile.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if tools:
                kwargs["tools"] = tools
            return self._client.messages.create(**kwargs)

        response = call_with_retry(_invoke, policy=policy, on_retry=on_retry)
        text = _anthropic_text_from(response)
        if not text:
            raise ValueError("model response had no text content")
        return parse_json_response(text)

    def complete_json_with_tools(
        self,
        prompt,
        *,
        client_tools=None,
        server_tools=None,
        tool_registry=None,
        max_tokens=None,
        max_iterations=_DEFAULT_MAX_TOOL_ITERATIONS,
        policy,
        on_retry=None,
    ) -> dict:
        """Anthropic tool-use loop. Handles tool_use blocks from client tools; passes
        server tools (web_search, code_execution) to the API directly."""
        all_tools: list[dict] = []
        if server_tools:
            all_tools.extend(server_tools)
        if client_tools:
            all_tools.extend(client_tools)

        messages: list[dict] = [{"role": "user", "content": prompt}]
        client_tool_names = {t.get("name") for t in (client_tools or [])}

        for iteration in range(max_iterations):
            def _invoke():
                kwargs = {
                    "model": self.profile.name,
                    "max_tokens": max_tokens or self.profile.max_tokens,
                    "messages": messages,
                }
                if all_tools:
                    kwargs["tools"] = all_tools
                return self._client.messages.create(**kwargs)

            response = call_with_retry(_invoke, policy=policy, on_retry=on_retry)

            # If the model is done (stop_reason != "tool_use"), extract text and return.
            if getattr(response, "stop_reason", None) != "tool_use":
                text = _anthropic_text_from(response)
                if not text:
                    raise ValueError("model response had no text content")
                return parse_json_response(text)

            # Append the assistant turn verbatim and execute any client tool_use blocks.
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                if block.name not in client_tool_names or tool_registry is None:
                    # Server tool (e.g. web_search); Anthropic already executed it.
                    print(f"    [tool·{iteration+1}] {block.name} (server)")
                    continue
                args_preview = _args_preview(block.input)
                print(f"    [tool·{iteration+1}] {block.name}({args_preview})")
                result = tool_registry.execute(block.name, dict(block.input or {}))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result.content,
                    "is_error": result.is_error,
                })

            remaining = max_iterations - iteration - 1
            if remaining <= _WRAP_UP_MARGIN:
                nudge = (
                    f"NOTE: You have {remaining} tool call(s) left before budget exhaustion. "
                    "Stop calling tools now and produce your final JSON answer using what "
                    "you already have."
                )
                if tool_results:
                    # Attach the nudge as a trailing text item on the user turn.
                    messages.append({
                        "role": "user",
                        "content": tool_results + [{"type": "text", "text": nudge}],
                    })
                else:
                    messages.append({"role": "user", "content": nudge})
            elif not tool_results:
                messages.append({"role": "user", "content": "Continue."})
            else:
                messages.append({"role": "user", "content": tool_results})

        raise ValueError(f"tool-use loop exceeded {max_iterations} iterations without final text")


class OpenAICompatClient(ModelClient):
    """Covers OpenAI + any OpenAI-compatible endpoint via base_url override."""

    supports_server_web_search = False

    def __init__(self, profile: ModelProfile):
        import openai
        self.profile = profile
        kwargs: dict = {}
        if profile.api_key:
            kwargs["api_key"] = profile.api_key
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        if profile.timeout_seconds > 0:
            kwargs["timeout"] = profile.timeout_seconds
        self._client = openai.OpenAI(**kwargs)

    def complete_json(
        self,
        prompt,
        *,
        tools=None,
        max_tokens=None,
        policy,
        on_retry=None,
    ) -> dict:
        # Single-turn path. Anthropic server-tool dicts don't translate; drop silently.
        del tools

        def _invoke():
            # NOTE: intentionally no response_format={"type":"json_object"} — some
            # OpenAI-compat providers (Moonshot/Kimi, Gemini via openai endpoint,
            # various Ollama models) return empty content when it's set. Our prompts
            # all explicitly ask for JSON-only output and parse_json_response tolerates
            # markdown fences + junk preamble.
            effective_max = _effective_max_tokens(
                self.profile.name,
                max_tokens or self.profile.max_tokens,
            )
            kwargs = {
                "model": self.profile.name,
                "temperature": self.profile.temperature,
                "messages": [{"role": "user", "content": prompt}],
                _openai_token_param_name(self.profile.name): effective_max,
            }
            return self._client.chat.completions.create(**kwargs)

        response = call_with_retry(_invoke, policy=policy, on_retry=on_retry)
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            finish = getattr(choice, "finish_reason", "?")
            effective_max = _effective_max_tokens(
                self.profile.name,
                max_tokens or self.profile.max_tokens,
            )
            if finish == "length":
                raise ValueError(
                    f"model ({self.profile.name}) exhausted its token budget "
                    f"(max={effective_max}) before producing visible output. "
                    f"Reasoning/thinking models count internal thinking against the "
                    f"budget — raise max_tokens / investigation_max_tokens in Settings "
                    f"(try 32000 for long prompts)."
                )
            raise ValueError(
                f"model ({self.profile.name}) returned empty content (finish_reason={finish!r}). "
                f"Most common causes: temperature setting unsupported by this model "
                f"(Kimi/GPT-5/o-series require 1.0), content filter, or transient provider issue."
            )
        return parse_json_response(text)

    def complete_json_with_tools(
        self,
        prompt,
        *,
        client_tools=None,
        server_tools=None,
        tool_registry=None,
        max_tokens=None,
        max_iterations=_DEFAULT_MAX_TOOL_ITERATIONS,
        policy,
        on_retry=None,
    ) -> dict:
        """OpenAI-compat function-calling loop. Server tools are not supported here."""
        del server_tools  # Anthropic-specific; ignored on this path.

        messages: list[dict] = [{"role": "user", "content": prompt}]

        for iteration in range(max_iterations):
            def _invoke():
                effective_max = _effective_max_tokens(
                    self.profile.name,
                    max_tokens or self.profile.max_tokens,
                )
                kwargs = {
                    "model": self.profile.name,
                    "temperature": self.profile.temperature,
                    "messages": messages,
                    _openai_token_param_name(self.profile.name): effective_max,
                }
                if client_tools:
                    kwargs["tools"] = client_tools
                    kwargs["tool_choice"] = "auto"
                return self._client.chat.completions.create(**kwargs)

            response = call_with_retry(_invoke, policy=policy, on_retry=on_retry)
            choice = response.choices[0]
            message = choice.message

            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                text = (message.content or "").strip()
                if not text:
                    raise ValueError("model response had no text content")
                return parse_json_response(text)

            # Append the assistant turn (may contain both text and tool_calls).
            assistant_turn: dict = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            # Kimi K2.x "thinking mode" (and other reasoning-enabled OpenAI-compat
            # providers) attach a reasoning_content field to the assistant message.
            # Their API then requires it echoed back on subsequent turns, or rejects
            # the request with 'thinking is enabled but reasoning_content is missing'.
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning is None:
                extras = getattr(message, "model_extra", None) or {}
                reasoning = extras.get("reasoning_content") if isinstance(extras, dict) else None
            if reasoning is not None:
                assistant_turn["reasoning_content"] = reasoning
            messages.append(assistant_turn)

            for call in tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                args_preview = _args_preview(args)
                print(f"    [tool·{iteration+1}] {name}({args_preview})")
                if tool_registry is None:
                    result_content = "error: no tool registry configured"
                else:
                    result = tool_registry.execute(name, args)
                    result_content = result.content
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result_content,
                })

            remaining = max_iterations - iteration - 1
            if remaining <= _WRAP_UP_MARGIN:
                messages.append({
                    "role": "user",
                    "content": (
                        f"NOTE: You have {remaining} tool call(s) left before budget exhaustion. "
                        "Stop calling tools now and produce your final JSON answer using what "
                        "you already have."
                    ),
                })

        raise ValueError(f"tool-use loop exceeded {max_iterations} iterations without final text")


class EmbeddingClient:
    """Thin OpenAI-compat embeddings client. Separate from ModelClient because
    embeddings have a very different request shape and not all providers support them."""

    def __init__(self, *, api_key: str, base_url: str = "", model: str = "text-embedding-3-small"):
        import openai
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        r = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in r.data]


def build_embedding_client(profile: ModelProfile, *, model: str = "text-embedding-3-small") -> EmbeddingClient:
    """Construct an EmbeddingClient from a ModelProfile. Only openai_compat profiles
    are supported for now (Anthropic doesn't offer embeddings)."""
    if (profile.provider or "").lower() not in ("openai", "openai_compat", "openai-compat"):
        raise ValueError(
            f"embeddings require an openai_compat profile; got '{profile.provider}'"
        )
    return EmbeddingClient(
        api_key=profile.api_key,
        base_url=profile.base_url,
        model=model,
    )


def build_client(profile: ModelProfile) -> ModelClient:
    provider = (profile.provider or "").strip().lower()
    if provider == "anthropic":
        return AnthropicClient(profile)
    if provider in ("openai", "openai_compat", "openai-compat"):
        return OpenAICompatClient(profile)
    raise ValueError(
        f"unknown provider '{profile.provider}'. "
        f"Expected 'anthropic' or 'openai_compat'."
    )


def _anthropic_text_from(response) -> str:
    """Join text blocks from an Anthropic response, skipping tool-use blocks."""
    parts: list[str] = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def _args_preview(args) -> str:
    """Short, single-line summary of a tool's input for progress printing."""
    if not isinstance(args, dict):
        return str(args)[:80]
    items = []
    for k, v in args.items():
        s = str(v).replace("\n", " ")
        if len(s) > 60:
            s = s[:60] + "…"
        items.append(f"{k}={s!r}" if isinstance(v, str) else f"{k}={s}")
    joined = ", ".join(items)
    if len(joined) > 140:
        joined = joined[:140] + "…"
    return joined
