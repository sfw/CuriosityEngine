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


@dataclass(frozen=True)
class ModelProfile:
    provider: str                       # "anthropic" | "openai_compat"
    name: str                           # e.g. "claude-sonnet-4-6", "gpt-5.1", "gemini-2.5-pro"
    api_key: str = ""
    base_url: str = ""                  # OpenAI-compat: override endpoint; anthropic: rarely used
    max_tokens: int = 4096
    investigation_max_tokens: int = 8192


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
            return self._client.chat.completions.create(
                model=self.profile.name,
                max_tokens=max_tokens or self.profile.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

        response = call_with_retry(_invoke, policy=policy, on_retry=on_retry)
        text = (response.choices[0].message.content or "").strip()
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
        """OpenAI-compat function-calling loop. Server tools are not supported here."""
        del server_tools  # Anthropic-specific; ignored on this path.

        messages: list[dict] = [{"role": "user", "content": prompt}]

        for iteration in range(max_iterations):
            def _invoke():
                kwargs = {
                    "model": self.profile.name,
                    "max_tokens": max_tokens or self.profile.max_tokens,
                    "messages": messages,
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
            messages.append({
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
            })

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
