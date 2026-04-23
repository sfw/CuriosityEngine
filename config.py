"""Curiosity Engine configuration, persisted at ~/.CuriosityEngine/engine.toml.

Phase 1 schema — supports multiple named model profiles:

    [models.primary]
    provider = "anthropic" | "openai_compat"
    name = "..."
    api_key = "..."
    base_url = "..."          # optional — use for Gemini openai-compat / OpenRouter / Ollama / etc.
    max_tokens = 4096
    investigation_max_tokens = 8192

    [models.verifier]         # optional; falls back to primary if omitted
    ...

    [retry]
    max_attempts = 5
    base_delay_seconds = 0.5
    max_delay_seconds = 8.0
    jitter_seconds = 0.25
"""

from __future__ import annotations

import getpass
import os
import sys
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from providers import ModelProfile
from retry_utils import RetryPolicy


@dataclass
class EngineSettings:
    """Operational knobs for the engine loop. Persisted in the [engine] section
    of engine.toml. Editable from the web UI Settings page."""
    cross_ref_window: int = 20
    questions_per_cycle: int = 3
    investigations_per_cycle: int = 1
    cross_ref_frequency: int = 3
    novelty_threshold: float = 0.7
    register_confidence_floor: float = 0.6
    verify_insights: bool = True
    # Cross-domain analog probe: after high-surprise entries, ask the engine
    # what *distant* domains have structural analogs to the finding, then enqueue
    # those as investigable questions. This is where biology→algorithmics-style
    # jumps come from.
    analog_probe_enabled: bool = True
    analog_probe_surprise_threshold: float = 0.5
    # When the verifier returns `inconclusive` (could not reach the claim, not
    # refuted it), the insight becomes a held register entry pending settlement
    # rather than being silently rejected. Held entries have a separate (usually
    # tighter) confidence floor.
    held_entries_enabled: bool = True
    held_confidence_floor: float = 0.7

CONFIG_DIR = Path.home() / ".CuriosityEngine"
CONFIG_PATH = CONFIG_DIR / "engine.toml"

# Common OpenAI-compat endpoints so the setup wizard can offer them.
OPENAI_COMPAT_PRESETS = [
    ("OpenAI",            "https://api.openai.com/v1",                            "gpt-5.1"),
    ("Google Gemini",     "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-pro"),
    ("OpenRouter",        "https://openrouter.ai/api/v1",                         "openai/gpt-5.1"),
    ("Ollama (local)",    "http://localhost:11434/v1",                            "llama3.3"),
    ("xAI",               "https://api.x.ai/v1",                                  "grok-4"),
    ("Groq",              "https://api.groq.com/openai/v1",                       "llama-3.3-70b-versatile"),
    ("Together",          "https://api.together.xyz/v1",                          "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    ("DeepSeek",          "https://api.deepseek.com/v1",                          "deepseek-chat"),
]

ANTHROPIC_MODEL_CHOICES = [
    ("claude-opus-4-7", "most capable; slower; highest cost"),
    ("claude-sonnet-4-6", "balanced (default)"),
    ("claude-haiku-4-5-20251001", "fastest; lowest cost"),
]


@dataclass
class CuriosityEngineConfig:
    primary: ModelProfile
    verifier: ModelProfile            # falls back to a copy of primary if not configured
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    engine: EngineSettings = field(default_factory=EngineSettings)

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> CuriosityEngineConfig:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            if sys.stdin.isatty():
                toml_content = interactive_setup(path)
            else:
                toml_content = _DEFAULT_TOML_PLACEHOLDER
                print(f"Created default config at {path}")
            path.write_text(toml_content)

        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Auto-migrate the Phase 0 single-profile schema to Phase 1 named profiles.
        if "models" not in data and "model" in data:
            data = _migrate_legacy_schema(data, path)

        models = data.get("models", {})
        if not models:
            raise ValueError(
                f"{path} is missing a [models.primary] section. "
                f"Delete the file and re-run to trigger setup."
            )

        primary_data = models.get("primary")
        if not primary_data:
            raise ValueError(f"{path} is missing [models.primary].")

        primary = _profile_from_dict(primary_data, "primary")
        verifier_data = models.get("verifier")
        if verifier_data:
            verifier = _profile_from_dict(verifier_data, "verifier")
        else:
            verifier = replace(primary)

        retry_section = data.get("retry", {})
        retry = RetryPolicy(
            max_attempts=int(retry_section.get("max_attempts", 5)),
            base_delay_seconds=float(retry_section.get("base_delay_seconds", 0.5)),
            max_delay_seconds=float(retry_section.get("max_delay_seconds", 8.0)),
            jitter_seconds=float(retry_section.get("jitter_seconds", 0.25)),
        )

        eng_section = data.get("engine", {})
        engine = EngineSettings(
            cross_ref_window=int(eng_section.get("cross_ref_window", 20)),
            questions_per_cycle=int(eng_section.get("questions_per_cycle", 3)),
            investigations_per_cycle=int(eng_section.get("investigations_per_cycle", 1)),
            cross_ref_frequency=int(eng_section.get("cross_ref_frequency", 3)),
            novelty_threshold=float(eng_section.get("novelty_threshold", 0.7)),
            register_confidence_floor=float(eng_section.get("register_confidence_floor", 0.6)),
            verify_insights=bool(eng_section.get("verify_insights", True)),
            analog_probe_enabled=bool(eng_section.get("analog_probe_enabled", True)),
            analog_probe_surprise_threshold=float(
                eng_section.get("analog_probe_surprise_threshold", 0.5)
            ),
            held_entries_enabled=bool(eng_section.get("held_entries_enabled", True)),
            held_confidence_floor=float(eng_section.get("held_confidence_floor", 0.7)),
        )

        return cls(primary=primary, verifier=verifier, retry=retry, engine=engine)


def _migrate_legacy_schema(data: dict, path: Path) -> dict:
    """Phase 0 had a single [model] section. Lift it into [models.primary]."""
    legacy = data.get("model", {})
    print(f"Migrating {path} from Phase 0 schema to Phase 1 (multi-profile).")
    primary = ModelProfile(
        provider="anthropic",
        name=str(legacy.get("name", "claude-sonnet-4-6")),
        api_key=str(legacy.get("api_key", "")),
        base_url=str(legacy.get("base_url", "")),
        max_tokens=int(legacy.get("max_tokens", 4096)),
        investigation_max_tokens=int(legacy.get("investigation_max_tokens", 8192)),
    )
    retry_section = data.get("retry", {})
    retry = RetryPolicy(
        max_attempts=int(retry_section.get("max_attempts", 5)),
        base_delay_seconds=float(retry_section.get("base_delay_seconds", 0.5)),
        max_delay_seconds=float(retry_section.get("max_delay_seconds", 8.0)),
        jitter_seconds=float(retry_section.get("jitter_seconds", 0.25)),
    )
    # Write the migrated file and re-read so downstream parsing is uniform.
    path.write_text(_build_toml(primary, verifier=None, retry=retry))
    print(f"  Migrated. Consider re-running setup (delete {path}) to configure a verifier model.")
    with open(path, "rb") as f:
        return tomllib.load(f)


def _profile_from_dict(data: dict, role: str) -> ModelProfile:
    provider = data.get("provider")
    if not provider:
        raise ValueError(f"[models.{role}] is missing 'provider'.")
    name = data.get("name")
    if not name:
        raise ValueError(f"[models.{role}] is missing 'name'.")
    return ModelProfile(
        provider=str(provider),
        name=str(name),
        api_key=str(data.get("api_key", "")),
        base_url=str(data.get("base_url", "")),
        max_tokens=int(data.get("max_tokens", 4096)),
        investigation_max_tokens=int(data.get("investigation_max_tokens", 8192)),
        temperature=float(data.get("temperature", 1.0)),
    )


# ─────────────────────────────────────────────
# First-run interactive setup
# ─────────────────────────────────────────────

_DEFAULT_TOML_PLACEHOLDER = """# Curiosity Engine — config placeholder.
# Delete this file and run the engine from a terminal to launch interactive setup.

[models.primary]
provider = "anthropic"
name = "claude-sonnet-4-6"
max_tokens = 4096
investigation_max_tokens = 8192

[retry]
max_attempts = 5
base_delay_seconds = 0.5
max_delay_seconds = 8.0
jitter_seconds = 0.25
"""


def _prompt(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"{question}{suffix}: ").strip()
    except EOFError:
        return default
    return raw or default


def _prompt_yes_no(question: str, default_yes: bool = False) -> bool:
    default_str = "Y/n" if default_yes else "y/N"
    try:
        raw = input(f"{question} [{default_str}]: ").strip().lower()
    except EOFError:
        return default_yes
    if not raw:
        return default_yes
    return raw in ("y", "yes")


def _prompt_choice(options: list[str], default_index: int = 0) -> int:
    while True:
        for i, label in enumerate(options, start=1):
            print(f"  {i}) {label}")
        raw = _prompt("Choose", default=str(default_index + 1))
        try:
            idx = int(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if 1 <= idx <= len(options):
            return idx - 1
        print(f"  Choice must be between 1 and {len(options)}.")


def _prompt_anthropic_model() -> str:
    print("\nModel:")
    labels = [f"{m:<30} {desc}" for m, desc in ANTHROPIC_MODEL_CHOICES] + ["custom model id"]
    idx = _prompt_choice(labels, default_index=1)
    if idx < len(ANTHROPIC_MODEL_CHOICES):
        return ANTHROPIC_MODEL_CHOICES[idx][0]
    while True:
        custom = _prompt("Enter model id")
        if custom:
            return custom
        print("  Model id cannot be empty.")


def _prompt_openai_compat_endpoint() -> tuple[str, str]:
    """Return (base_url, suggested_model). base_url='' means OpenAI default."""
    print("\nEndpoint:")
    labels = [f"{name:<20} {url}" for name, url, _ in OPENAI_COMPAT_PRESETS] + ["custom endpoint"]
    idx = _prompt_choice(labels, default_index=0)
    if idx < len(OPENAI_COMPAT_PRESETS):
        _, url, suggested = OPENAI_COMPAT_PRESETS[idx]
        return url, suggested
    url = _prompt("Enter base_url (OpenAI-compatible /v1 path)")
    suggested = _prompt("Enter model id")
    return url, suggested


def _prompt_api_key(env_var: str, friendly: str) -> str:
    if os.environ.get(env_var):
        print(f"  {env_var} env var detected — it will be used at runtime. Skipping prompt.")
        return ""
    print(f"\n{friendly} API key (input hidden; leave blank to rely on {env_var} at runtime):")
    try:
        key = getpass.getpass("  Key: ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""
    return key


def _prompt_profile(role: str, default_provider: str = "anthropic") -> ModelProfile:
    print(f"\n─── Configuring '{role}' model ───")

    providers = ["anthropic", "openai_compat"]
    labels = [
        "anthropic           Anthropic (Claude) — supports server-side web_search",
        "openai_compat       OpenAI / Gemini / OpenRouter / Ollama / xAI / Groq / Together / DeepSeek / custom",
    ]
    default_idx = providers.index(default_provider)
    idx = _prompt_choice(labels, default_index=default_idx)
    provider = providers[idx]

    if provider == "anthropic":
        name = _prompt_anthropic_model()
        api_key = _prompt_api_key("ANTHROPIC_API_KEY", "Anthropic")
        base_url = ""
    else:
        base_url, suggested_model = _prompt_openai_compat_endpoint()
        name = _prompt("Model id", default=suggested_model)
        api_key = _prompt_api_key("OPENAI_API_KEY", "OpenAI-compat")

    return ModelProfile(
        provider=provider,
        name=name,
        api_key=api_key,
        base_url=base_url,
    )


def _render_profile_toml(role: str, profile: ModelProfile) -> str:
    api_key_line = (
        f'api_key = "{profile.api_key}"' if profile.api_key
        else '# api_key = "..."       # or rely on the env var at runtime'
    )
    base_url_line = (
        f'base_url = "{profile.base_url}"' if profile.base_url
        else '# base_url = "..."      # only needed for non-default endpoints'
    )
    return f"""[models.{role}]
provider = "{profile.provider}"
name = "{profile.name}"
{api_key_line}
{base_url_line}
max_tokens = {profile.max_tokens}
investigation_max_tokens = {profile.investigation_max_tokens}
temperature = {profile.temperature}
"""


def _build_toml(
    primary: ModelProfile,
    verifier: Optional[ModelProfile],
    retry: RetryPolicy,
    engine: Optional[EngineSettings] = None,
) -> str:
    header = "# Curiosity Engine — model connection + engine settings.\n# Generated by first-run setup. Edit freely.\n\n"
    sections = [_render_profile_toml("primary", primary)]
    if verifier is not None:
        sections.append(_render_profile_toml("verifier", verifier))
    else:
        sections.append(
            "# [models.verifier]      # optional; omitting this falls back to primary for the adversarial verify step.\n"
        )
    sections.append(
        f"""[retry]
max_attempts = {retry.max_attempts}
base_delay_seconds = {retry.base_delay_seconds}
max_delay_seconds = {retry.max_delay_seconds}
jitter_seconds = {retry.jitter_seconds}
"""
    )
    eng = engine or EngineSettings()
    sections.append(
        f"""[engine]
# How the loop runs. Bump cross_ref_window on big-context models (Kimi K2.6 @ 256K
# could comfortably handle 80-120) so cross-reference can surface intersections
# across a wider slice of the journal.
cross_ref_window = {eng.cross_ref_window}
questions_per_cycle = {eng.questions_per_cycle}
investigations_per_cycle = {eng.investigations_per_cycle}
cross_ref_frequency = {eng.cross_ref_frequency}
novelty_threshold = {eng.novelty_threshold}
register_confidence_floor = {eng.register_confidence_floor}
verify_insights = {str(eng.verify_insights).lower()}
# Cross-domain analog probe — when a finding is high-surprise, ask the engine
# what distant-field mechanisms are structurally analogous and investigate those.
# This targets the biology→algorithmics style of novelty (applying knowledge
# from one domain to another).
analog_probe_enabled = {str(eng.analog_probe_enabled).lower()}
analog_probe_surprise_threshold = {eng.analog_probe_surprise_threshold}
# Held-state pipeline — when the verifier returns `inconclusive` (couldn't reach
# the claim, not refuted it), insights become held register entries pending
# settlement rather than being silently rejected. Held entries usually require
# slightly higher confidence than active ones to avoid hedged noise.
held_entries_enabled = {str(eng.held_entries_enabled).lower()}
held_confidence_floor = {eng.held_confidence_floor}
"""
    )
    return header + "\n".join(sections)


def interactive_setup(path: Path) -> str:
    print("=" * 62)
    print("  Curiosity Engine — first-run setup")
    print("=" * 62)
    print(f"\nConfig will be written to: {path}")
    print("(You can re-run setup later by deleting that file.)")
    print()
    print("The engine uses two model roles:")
    print("  • primary   — runs introspection, investigation, synthesis")
    print("  • verifier  — adversarially reviews synthesized insights")
    print("For best results the verifier should be a DIFFERENT model family than primary.")

    try:
        primary = _prompt_profile("primary", default_provider="anthropic")

        configure_verifier = _prompt_yes_no(
            "\nConfigure a separate verifier model? (Highly recommended for cross-model verification)",
            default_yes=True,
        )
        verifier: Optional[ModelProfile] = None
        if configure_verifier:
            default_v = "openai_compat" if primary.provider == "anthropic" else "anthropic"
            verifier = _prompt_profile("verifier", default_provider=default_v)
    except KeyboardInterrupt:
        print("\nSetup cancelled. No config written.")
        sys.exit(1)

    toml = _build_toml(primary, verifier, RetryPolicy())
    print(f"\nSaved config to {path}")
    return toml
