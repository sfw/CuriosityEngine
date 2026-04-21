"""Model client abstraction: one interface across Anthropic + OpenAI-compat endpoints.

OpenAI-compat covers OpenAI itself plus Gemini (openai-compat endpoint), OpenRouter,
Ollama (openai mode), xAI, Groq, Together, DeepSeek, LM Studio, etc. — anything that
speaks the OpenAI chat-completions protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from json_utils import parse_json_response
from retry_utils import RetryPolicy, call_with_retry


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
        """Call the model, extract text, parse JSON, return dict."""


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

        text_parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        text = "\n".join(text_parts).strip()
        if not text:
            raise ValueError("model response had no text content")
        return parse_json_response(text)


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
        # OpenAI-compat endpoints don't support Anthropic's server tools; silently skip
        # them and leave the model to answer from its priors. A future phase can add
        # client-side tool execution loops.
        if tools:
            on_retry = on_retry  # no-op; keep shape
        del tools  # intentionally dropped for Phase 1

        def _invoke():
            return self._client.chat.completions.create(
                model=self.profile.name,
                max_tokens=max_tokens or self.profile.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

        response = call_with_retry(_invoke, policy=policy, on_retry=on_retry)
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            raise ValueError("model response had no text content")
        return parse_json_response(text)


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
