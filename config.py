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
    # How many distant-field analogs the probe turns into enqueued questions.
    # Keep modest — each enqueued question spends a future cycle's budget.
    analog_probe_max_analogs: int = 3
    # Assumption probe: complementary to the analog probe — fires on LOW-surprise
    # CONFIRMED findings (the accepted-wisdom regime where load-bearing
    # assumptions hide). Asks the primary to name implicit premises the field
    # takes for granted, then produces investigable questions that would test
    # each assumption's validity. Opposite trigger condition from analog probe,
    # which fires on HIGH-surprise entries to reach outward.
    assumption_probe_enabled: bool = True
    assumption_probe_surprise_threshold: float = 0.3
    # Parallel of analog_probe_max_analogs: how many named assumptions the probe
    # turns into enqueued negation questions per triggering entry.
    assumption_probe_max_assumptions: int = 3
    # When the verifier returns `inconclusive` (could not reach the claim, not
    # refuted it), the insight becomes a held register entry pending settlement
    # rather than being silently rejected. Held entries have a separate (usually
    # tighter) confidence floor.
    held_entries_enabled: bool = True
    held_confidence_floor: float = 0.7
    # Cross-ref is a one-shot generation over a large context (pattern-matching
    # across entries). Reasoning-mode models (Kimi K2.x, Claude extended thinking,
    # o-series) spend most of their first-token budget on thinking that cross-ref
    # doesn't benefit from. Set this to any configured role name (e.g. "verifier",
    # or a custom profile defined under [models.<name>]) to offload cross-ref to a
    # faster non-reasoning model while keeping reasoning on investigation.
    # Empty / "primary" = use the primary profile (backward-compatible default).
    cross_ref_role: str = ""
    # Per-phase routing for the research-directive export pipeline. Directive
    # synthesis is constrained schema-filling (hypothesis / test plan / agentic
    # fields / verification criteria) — it does NOT benefit from reasoning-mode
    # thinking, which adds 60-180s of latency per call without improving output.
    # Route directive section generation to a fast non-reasoning model while
    # keeping the primary as a reasoning model for investigation. Same
    # resolution rules as cross_ref_role: empty / "primary" = use primary;
    # "verifier" = use verifier; any other name = look up under [models.<name>].
    # directive_primary handles the HEAVY sections of the directive pipeline:
    # test_plan, agentic_prompt, verification_criteria. These translate
    # literature-watch predictions into in-house experiments and enforce
    # strict citation/tool grounding — reasoning helps materially. Empty /
    # "primary" = use primary; "verifier" = use verifier; any other name =
    # look up under [models.<name>].
    directive_primary_role: str = ""
    # directive_primary_fast handles the LIGHT sections: hypothesis, ELI5,
    # research_path. These are restatement / style / strategic-prose tasks
    # where reasoning models over-elaborate without improving quality.
    # Empty = fall back to directive_primary (which itself falls back to
    # primary). Set this to a fast non-reasoning profile to cut directive
    # generation time by ~30-40% with no quality loss on these sections.
    directive_primary_fast_role: str = ""
    # Same idea for the directive grounding-review pass. Empty = use the
    # journal's `verifier` profile (the cross-family verifier already configured).
    directive_verifier_role: str = ""
    # How many verifier passes (initial + retries) the directive pipeline will
    # run trying to land a clean output. The loop stops as soon as the verifier
    # returns clean OR this cap is reached. Each retry regenerates the agentic
    # prompt with the previous flags appended; other sections are kept (the
    # agentic prompt is the riskiest section by far). 1 = no retries, 2 =
    # current behavior (initial + 1 retry), 3 = default — gives one more
    # chance to converge after the LLM addresses the first round of flags.
    directive_max_verification_passes: int = 3
    # Per-phase routing for the negative-space gap scan. Same resolution
    # rules as cross_ref_role / directive_*_role.
    #
    # gap_scan_extract_role — runs matrix extraction (step 1 of the scan):
    # reads all journal entries, identifies methods + problems + which
    # entries cover which (method, problem) cells. Multi-document
    # categorization that benefits from inference — REASONING MODEL is the
    # right fit here. Empty / "primary" = use primary (default).
    gap_scan_extract_role: str = ""
    # investigation_assessor_role — Phase 5 of the self-evolving verifier.
    # The investigation loop runs in three stages:
    #   1. form_hypothesis (HYPOTHESIS_PROMPT, pre-search) — exploration
    #   2. run_investigation (INVESTIGATE_PROMPT + tools) — exploration
    #   3. assess_surprise (SURPRISE_PROMPT, post-search) — EVALUATION
    # Stages 1+2 are open/divergent; stage 3 is closed/evaluative. Routing
    # the assessor to a different model from the primary creates a
    # representational separation that resists the "same token space"
    # collapse the source insight (r-9a35e387) prescribes against:
    # "Current LLMs collapse the generation of ideas and their evaluation
    # into the same token space, which invites self-grading, articulate
    # restatement of prior art, and premature optimization for
    # judge-pleasing." Empty / "primary" = use primary (default,
    # backward-compatible). "verifier" = cross-family separation
    # (recommended). Any other name = look up under [models.<name>].
    investigation_assessor_role: str = ""
    # gap_scan_classify_role — runs cell classification (step 2) and
    # investigable-question generation (step 4). Step 2 has large outputs
    # (5,000-10,000 tokens for ~100 empty cells); on reasoning models in
    # non-streaming mode this reliably triggers timeouts. NON-REASONING
    # MODEL is the right fit here; per-cell judgement is world-knowledge
    # heavy, not chain-of-thought heavy. Empty / "verifier" = use verifier
    # (default; assumes verifier is non-reasoning-tier).
    gap_scan_classify_role: str = ""
    # Negative-space gap scan — structural analysis that builds a (method × problem)
    # matrix from the journal's entries and identifies empty cells (combinations
    # nobody in the field has studied). Gated to require a minimum journal size:
    # below the threshold, most empty cells are empty simply because the journal
    # is young, not because the field ignored them. Triggered on-demand via the
    # Admin tab or `--scan-gaps` — not part of the cycle loop.
    negative_space_min_entries: int = 15
    # During the gap-verification step of scan_gaps, a cell classified as
    # "underexplored" is confirmed empty when total structured hits across its
    # verification queries is below this threshold. 5 is a reasonable default
    # for broad searches across 3 sources (crossref/arxiv/semantic_scholar);
    # raise if you're getting false positives on well-covered topics, lower if
    # nothing is ever confirmed empty.
    gap_verification_hit_threshold: int = 5
    # Confidence penalty applied when an engine-side guard downgrades a
    # verdict (e.g. skeptic-probe flips validated→challenged). The LLM's
    # returned confidence was computed before the guard fired; flat
    # confidence on a verdict change is a hedge pattern. This floor keeps
    # stored confidence honest. Set to 0.0 to disable.
    confidence_drop_on_downgrade: float = 0.10
    # Register admission policy. Two modes:
    #   "scalar" (default — backward compatible): a candidate is admitted
    #       iff verdict is validated AND verified_confidence >=
    #       register_confidence_floor AND premises_supported AND novelty
    #       isn't restatement/unsupported. Single-axis quality bar.
    #   "pareto": ALL of the scalar mode's checks PLUS a Pareto-dominance
    #       check against the existing register. A new candidate is rejected
    #       if any existing active entry dominates it across the 4-axis
    #       Pareto set: verified_confidence × premises_supported_count ×
    #       peer_differentiators_count × inverse_alias_gap. Catches the
    #       failure mode where a new entry is "just like X but slightly
    #       worse on every axis" — an admission that the scalar floor
    #       can't see.
    # Phase 4 of the self-evolving verifier.
    register_admission_mode: str = "scalar"
    # Phase 6 of the self-evolving verifier: tournament-ranked Best-of-N
    # synthesis. When a high-novelty xref is promoted to an Insight, the
    # synthesize step generates this many candidate insights, canonicalizes
    # each, and selects the candidate with the largest alias-gap to the
    # existing register (ties broken by self-reported confidence).
    # Default 3; minimum effective value is 2 (1 means single-candidate, no
    # tournament). Higher values trade more LLM cost for more divergent
    # selection. Cost scales linearly: candidate_count × synthesis call.
    # Borrows from Co-Scientist's Ranking agent + standard agentic
    # best-of-N selection patterns.
    synthesis_candidate_count: int = 3
    # Phase 7 of the self-evolving verifier: persona-conditioned
    # introspection. Instead of one self-interrogation per cycle, run N
    # parallel introspections each through a different lens (skeptic,
    # outsider, historian, contrarian, practitioner). Merges
    # uncertainties across personas with persona-attributed source
    # tagging on downstream questions. Borrows from Stanford STORM's
    # multi-perspective question generation.
    # 1 = disabled (single voice, pre-Phase-7 behavior). 3 = default
    # (skeptic + outsider + contrarian). Up to 5 personas available.
    # Cost scales linearly with persona count. Each introspection is
    # cheap (~500-token output) so absolute cost is small.
    introspection_persona_count: int = 3
    # Phase 9 of the self-evolving verifier: hypothesis variants in the
    # investigation loop. The explorer persona generates N divergent
    # candidate hypotheses (each on a different architectural axis) and
    # the system picks the one most distant from majority-literature
    # consensus to actually investigate. Increases the chance that
    # surprise has somewhere to fire. Borrows branching exploration from
    # Tree of Thoughts.
    # 1 = single hypothesis (Phase-9-disabled, pre-Phase-9 behavior).
    # 3 = default. Cost is small per cycle (hypothesis generation is
    # ~100-token output × variant_count); investigation cost unchanged
    # because only the selected variant drives the search.
    hypothesis_variant_count: int = 3
    # Phase 8 of the self-evolving verifier: idea evolution from
    # downgraded extensions. When verify_insight downgrades a fresh
    # candidate to `extension` (the most common downgrade case),
    # autofire the evolution helper: identify the offending canonical-
    # form slot, mutate it via the verifier's diagnostic outputs,
    # synthesize a new candidate from the mutation, run it through full
    # verification. Closes the verifier→generator feedback loop.
    # Borrows mutation-loop pattern from Sakana AI's "AI Scientist" +
    # Co-Scientist's Evolution agent. Internal alignment with r-3c792e21
    # ("typed supervision from false positives via retrospective
    # unification") — verifier downgrades become typed supervision
    # signal for generator-side mutation.
    # Default: True (autofire on fresh verification). Reverify path
    # stays manual via the --evolve-during-reverify flag.
    idea_evolution_enabled: bool = True
    # Quality filter: don't evolve weak material. Evolution skips
    # candidates whose verified_confidence is below this floor.
    idea_evolution_confidence_floor: float = 0.65
    # Max chain depth. The evolved candidate is itself verified — if it
    # ALSO downgrades to extension, do we evolve it again? max_depth=1
    # means evolution fires once per chain (recommended default).
    # Higher values risk runaway perturbation.
    idea_evolution_max_depth: int = 1
    # Questions below this priority are rejected at enqueue time (except
    # human-sourced questions, which always bypass). Default 0.0 = disabled.
    # An earlier default of 0.70 was found to starve new journals — early
    # cycles produce mostly low-priority entry followups (priority scales
    # with surprise_delta) which got dropped en masse before a journal
    # could build enough context to generate higher-priority questions.
    # Set to a non-zero value (e.g. 0.70) on mature journals where you
    # specifically want to prune low-priority noise.
    question_priority_floor: float = 0.0
    # Parallel fan-out — how many investigations / xref-synth+verify pipelines
    # run concurrently per cycle. Default 1 = fully serial (zero behavior change).
    # Rate limiters are shared across threads so raising these does not burst
    # public APIs. Sensible ceilings are 3–4 investigations and 2–3 xref
    # pipelines; beyond that most time is spent waiting on rate limiters anyway.
    parallel_investigations: int = 1
    parallel_xref_pipeline: int = 1
    # ── Selective verification pipeline (A2 plumbing for Phases B/C/D) ──
    # All defaults are NO-OPS — A2 alone changes no behavior. Each knob
    # turns on when the corresponding phase ships.
    #
    # Phase B (alias-gap routing). Candidates with alias_gap below the reject
    # threshold are short-circuited to the rejection_log without an LLM
    # verification call. Candidates above the fast-track threshold skip
    # deep prior-art search and take the cheap path. The middle band runs
    # the existing full pipeline. 0.0 / 1.0 = no-op (no gating, every
    # candidate runs full pipeline).
    alias_gap_reject_threshold: float = 0.0
    alias_gap_fasttrack_threshold: float = 1.0
    # Phase B audit-back: fraction of below-reject-threshold candidates that
    # still get full LLM verification, tagged in the rejection_log as audit
    # samples. Detects discriminator drift — if audit-back items keep
    # validating, the reject threshold is too aggressive. 0.0 = never
    # sample-back (no-op).
    rejection_audit_back_rate: float = 0.0
    # Phase C (paraphrase-perturbation verifier stability). Run verification
    # against N paraphrase variants of the prompt, compute verdict variance
    # as a paraphrase_inconsistency_score. 1 = single pass (no-op,
    # pre-Phase-C behavior). 3 = recommended once C ships.
    paraphrase_variant_count: int = 1
    # Phase D (committee escalation). When paraphrase_inconsistency_score
    # exceeds this threshold (or committee verdicts disagree), escalate
    # to a second cross-family verifier. 1.0 = never escalate (no-op).
    committee_dissent_threshold: float = 1.0

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
    # Any additional [models.<name>] profiles beyond primary/verifier (e.g. a
    # dedicated cross_ref profile). Resolved by role name via resolve_profile().
    extras: dict[str, ModelProfile] = field(default_factory=dict)
    # Resolved cross-ref profile (None = use primary). Computed at load time from
    # [engine].cross_ref_role or [models.cross_ref].
    cross_ref: "ModelProfile | None" = None
    # Resolved directive-pipeline profiles (None = use primary / verifier).
    # Computed at load time from [engine].directive_primary_role /
    # directive_primary_fast_role / directive_verifier_role or matching
    # [models.<role>] sections.
    #   - directive_primary: heavy sections (test_plan, agentic_prompt,
    #     verification_criteria). Reasoning recommended.
    #   - directive_primary_fast: light sections (hypothesis, eli5,
    #     research_path). Non-reasoning recommended; falls back to
    #     directive_primary if unset.
    #   - directive_verifier: directive grounding-review pass. Reasoning
    #     recommended.
    directive_primary: "ModelProfile | None" = None
    directive_primary_fast: "ModelProfile | None" = None
    directive_verifier: "ModelProfile | None" = None
    # Resolved negative-space gap-scan profiles (None = use primary / verifier).
    # gap_scan_extract handles step 1 (matrix extraction) — reasoning helps;
    # gap_scan_classify handles steps 2 + 4 (classify + question generation) —
    # non-reasoning preferred due to large output sizes.
    gap_scan_extract: "ModelProfile | None" = None
    gap_scan_classify: "ModelProfile | None" = None
    # Phase 5: post-search investigation assessor. None = use primary.
    investigation_assessor: "ModelProfile | None" = None

    def resolve_profile(self, role: str) -> "ModelProfile | None":
        """Look up a configured profile by role name."""
        rn = (role or "").strip().lower()
        if rn == "primary":
            return self.primary
        if rn == "verifier":
            return self.verifier
        return self.extras.get(rn)

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

        # Any [models.<name>] beyond primary/verifier → extras dict
        extras: dict[str, ModelProfile] = {}
        for role_name, profile_data in models.items():
            if role_name in ("primary", "verifier"):
                continue
            extras[role_name] = _profile_from_dict(profile_data, role_name)

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
            analog_probe_max_analogs=int(eng_section.get("analog_probe_max_analogs", 3)),
            assumption_probe_enabled=bool(eng_section.get("assumption_probe_enabled", True)),
            assumption_probe_surprise_threshold=float(
                eng_section.get("assumption_probe_surprise_threshold", 0.3)
            ),
            assumption_probe_max_assumptions=int(
                eng_section.get("assumption_probe_max_assumptions", 3)
            ),
            negative_space_min_entries=int(eng_section.get("negative_space_min_entries", 15)),
            gap_verification_hit_threshold=int(
                eng_section.get("gap_verification_hit_threshold", 5)
            ),
            confidence_drop_on_downgrade=float(
                eng_section.get("confidence_drop_on_downgrade", 0.10)
            ),
            question_priority_floor=float(
                eng_section.get("question_priority_floor", 0.70)
            ),
            register_admission_mode=str(
                eng_section.get("register_admission_mode", "scalar")
            ).strip().lower() or "scalar",
            synthesis_candidate_count=max(
                1, int(eng_section.get("synthesis_candidate_count", 3))
            ),
            introspection_persona_count=max(
                1, min(5, int(eng_section.get("introspection_persona_count", 3))),
            ),
            hypothesis_variant_count=max(
                1, min(5, int(eng_section.get("hypothesis_variant_count", 3))),
            ),
            idea_evolution_enabled=bool(
                eng_section.get("idea_evolution_enabled", True)
            ),
            idea_evolution_confidence_floor=float(
                eng_section.get("idea_evolution_confidence_floor", 0.65)
            ),
            idea_evolution_max_depth=max(
                0, min(3, int(eng_section.get("idea_evolution_max_depth", 1))),
            ),
            held_entries_enabled=bool(eng_section.get("held_entries_enabled", True)),
            held_confidence_floor=float(eng_section.get("held_confidence_floor", 0.7)),
            cross_ref_role=str(eng_section.get("cross_ref_role", "")).strip(),
            directive_primary_role=str(eng_section.get("directive_primary_role", "")).strip(),
            directive_primary_fast_role=str(
                eng_section.get("directive_primary_fast_role", "")
            ).strip(),
            directive_verifier_role=str(eng_section.get("directive_verifier_role", "")).strip(),
            directive_max_verification_passes=int(
                eng_section.get("directive_max_verification_passes", 3)
            ),
            gap_scan_extract_role=str(eng_section.get("gap_scan_extract_role", "")).strip(),
            gap_scan_classify_role=str(eng_section.get("gap_scan_classify_role", "")).strip(),
            investigation_assessor_role=str(eng_section.get("investigation_assessor_role", "")).strip(),
            parallel_investigations=int(eng_section.get("parallel_investigations", 1)),
            parallel_xref_pipeline=int(eng_section.get("parallel_xref_pipeline", 1)),
            alias_gap_reject_threshold=float(
                eng_section.get("alias_gap_reject_threshold", 0.0)
            ),
            alias_gap_fasttrack_threshold=float(
                eng_section.get("alias_gap_fasttrack_threshold", 1.0)
            ),
            rejection_audit_back_rate=float(
                eng_section.get("rejection_audit_back_rate", 0.0)
            ),
            paraphrase_variant_count=max(
                1, int(eng_section.get("paraphrase_variant_count", 1)),
            ),
            committee_dissent_threshold=float(
                eng_section.get("committee_dissent_threshold", 1.0)
            ),
        )

        # Resolve cross_ref profile:
        #   1. [engine].cross_ref_role (explicit role name) — takes precedence
        #   2. [models.cross_ref] auto-pickup — convenient for dedicated profile
        #   3. None (falls back to primary in the engine)
        cross_ref_profile: ModelProfile | None = None
        cr_role = (engine.cross_ref_role or "").strip().lower()
        if cr_role and cr_role != "primary":
            if cr_role == "verifier":
                cross_ref_profile = verifier
            elif cr_role in extras:
                cross_ref_profile = extras[cr_role]
            else:
                raise ValueError(
                    f"[engine].cross_ref_role = {cr_role!r} but no matching profile "
                    f"is configured. Add [models.{cr_role}] or pick an existing role."
                )
        elif "cross_ref" in extras:
            cross_ref_profile = extras["cross_ref"]

        # Resolve directive profiles — same rules as cross_ref. Empty = use
        # primary/verifier; "verifier" = use verifier; any other name = look up
        # in extras. Auto-pickup of [models.directive_primary] /
        # [models.directive_verifier] if those sections exist.
        def _resolve_role(role_name: str, role_label: str) -> "ModelProfile | None":
            r = (role_name or "").strip().lower()
            if not r or r == "primary":
                return None
            if r == "verifier":
                return verifier
            if r in extras:
                return extras[r]
            raise ValueError(
                f"[engine].{role_label} = {r!r} but no matching profile is "
                f"configured. Add [models.{r}] or pick an existing role."
            )

        directive_primary_profile = _resolve_role(
            engine.directive_primary_role, "directive_primary_role",
        )
        if directive_primary_profile is None and "directive_primary" in extras:
            directive_primary_profile = extras["directive_primary"]

        directive_primary_fast_profile = _resolve_role(
            engine.directive_primary_fast_role, "directive_primary_fast_role",
        )
        if directive_primary_fast_profile is None and "directive_primary_fast" in extras:
            directive_primary_fast_profile = extras["directive_primary_fast"]

        directive_verifier_profile = _resolve_role(
            engine.directive_verifier_role, "directive_verifier_role",
        )
        if directive_verifier_profile is None and "directive_verifier" in extras:
            directive_verifier_profile = extras["directive_verifier"]

        gap_scan_extract_profile = _resolve_role(
            engine.gap_scan_extract_role, "gap_scan_extract_role",
        )
        if gap_scan_extract_profile is None and "gap_scan_extract" in extras:
            gap_scan_extract_profile = extras["gap_scan_extract"]

        gap_scan_classify_profile = _resolve_role(
            engine.gap_scan_classify_role, "gap_scan_classify_role",
        )
        if gap_scan_classify_profile is None and "gap_scan_classify" in extras:
            gap_scan_classify_profile = extras["gap_scan_classify"]

        investigation_assessor_profile = _resolve_role(
            engine.investigation_assessor_role, "investigation_assessor_role",
        )
        if investigation_assessor_profile is None and "investigation_assessor" in extras:
            investigation_assessor_profile = extras["investigation_assessor"]

        return cls(
            primary=primary, verifier=verifier,
            retry=retry, engine=engine,
            extras=extras, cross_ref=cross_ref_profile,
            directive_primary=directive_primary_profile,
            directive_primary_fast=directive_primary_fast_profile,
            directive_verifier=directive_verifier_profile,
            gap_scan_extract=gap_scan_extract_profile,
            gap_scan_classify=gap_scan_classify_profile,
            investigation_assessor=investigation_assessor_profile,
        )


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
        max_attempts=int(retry_section.get("max_attempts", 10)),
        base_delay_seconds=float(retry_section.get("base_delay_seconds", 0.5)),
        max_delay_seconds=float(retry_section.get("max_delay_seconds", 90.0)),
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
        timeout_seconds=float(data.get("timeout_seconds", 300.0)),
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
# Per-request HTTP timeout in seconds. Raise for reasoning models on big prompts
# (Kimi K2.x, GPT-5 thinking, o-series, Claude extended thinking can spend
# 60-180s "thinking" before streaming the first token).
timeout_seconds = {profile.timeout_seconds}
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
# How many analogs the probe converts into enqueued questions per triggering entry.
analog_probe_max_analogs = {eng.analog_probe_max_analogs}
# Assumption probe — complement to the analog probe. Fires on LOW-surprise
# confirmed findings (where field consensus most likely hides load-bearing
# assumptions); asks the primary model to name implicit premises and enqueue
# investigable questions that would test each. Opposite trigger from analog.
assumption_probe_enabled = {str(eng.assumption_probe_enabled).lower()}
assumption_probe_surprise_threshold = {eng.assumption_probe_surprise_threshold}
# Parallel to analog_probe_max_analogs: how many assumptions → negation questions.
assumption_probe_max_assumptions = {eng.assumption_probe_max_assumptions}
# Negative-space gap scan — minimum entry count before Scan-for-gaps is allowed.
# Below this threshold most empty matrix cells are artifacts of journal youth,
# not real field-level gaps, so the scan would surface noise.
negative_space_min_entries = {eng.negative_space_min_entries}
# During the gap-verification step of scan_gaps, a cell classified as
# "underexplored" is confirmed empty when structured hit count across its
# verification queries is below this. Raise if well-covered topics are being
# falsely confirmed as gaps; lower if nothing is ever confirmed empty.
gap_verification_hit_threshold = {eng.gap_verification_hit_threshold}
# Confidence penalty when an engine-side guard downgrades the verdict.
# Addresses the hedge pattern where the LLM returns flat confidence despite
# a verdict flip. Set to 0.0 to disable.
confidence_drop_on_downgrade = {eng.confidence_drop_on_downgrade}
# Minimum priority for a question to enter the investigation queue. Non-human
# sources with priority below this floor are dropped at enqueue. Human-sourced
# questions bypass — explicit intent overrides the autoscreen.
# Default 0.0 = disabled. An earlier 0.70 default starved new journals (early
# cycles produce mostly low-priority entry followups; the floor dropped them
# before the journal could build context). Set to a non-zero value only on
# mature journals where you specifically want to prune low-priority noise.
question_priority_floor = {eng.question_priority_floor}
# Register admission mode. "scalar" = single confidence floor + status checks
# (default; backward compatible). "pareto" = ALSO require the new entry to be
# non-dominated by any existing active entry across the 4-axis Pareto set
# (verified_confidence × premises_supported_count × peer_differentiators_count
# × inverse_alias_gap). Pareto rejects "just like X but slightly worse on
# every axis" admissions that the scalar floor can't see.
register_admission_mode = "{eng.register_admission_mode}"
# Phase 6 — Best-of-N synthesis tournament. When a high-novelty xref is
# promoted to an Insight, generate this many candidates and select the one
# with the largest alias-gap to the existing register (ties broken by
# self-reported confidence). 1 = single-candidate (Phase-6-disabled,
# matches pre-Phase-6 behavior). 3 = default. Linear LLM cost scaling.
synthesis_candidate_count = {eng.synthesis_candidate_count}
# Phase 7 — persona-conditioned introspection. Run N parallel
# introspections each through a different persona (skeptic, outsider,
# contrarian, historian, practitioner). 1 = single voice (Phase-7-disabled,
# matches pre-Phase-7 behavior). 3 = default. Range: 1-5.
introspection_persona_count = {eng.introspection_persona_count}
# Phase 9 — hypothesis variants in the investigation loop. Generate N
# divergent candidate hypotheses and pick the one most distant from
# majority-literature consensus to investigate. 1 = single hypothesis
# (Phase-9-disabled). 3 = default. Range 1-5.
hypothesis_variant_count = {eng.hypothesis_variant_count}
# Phase 8 — idea evolution from downgraded extensions. When verify_insight
# downgrades a candidate to "extension", autofire mutation: change the
# offending canonical-form slot, synthesize a new candidate, re-verify.
# Closes the verifier→generator loop. Reverify path stays manual via
# --evolve-during-reverify CLI flag.
idea_evolution_enabled = {str(eng.idea_evolution_enabled).lower()}
# Confidence floor — don't try to evolve weak material.
idea_evolution_confidence_floor = {eng.idea_evolution_confidence_floor}
# Max chain depth — 1 = evolution fires once per chain (recommended).
idea_evolution_max_depth = {eng.idea_evolution_max_depth}
# Held-state pipeline — when the verifier returns `inconclusive` (couldn't reach
# the claim, not refuted it), insights become held register entries pending
# settlement rather than being silently rejected. Held entries usually require
# slightly higher confidence than active ones to avoid hedged noise.
held_entries_enabled = {str(eng.held_entries_enabled).lower()}
held_confidence_floor = {eng.held_confidence_floor}
# cross-ref phase is a one-shot pattern-match over a large context; reasoning
# models spend their latency budget on thinking that cross-ref doesn't benefit
# from. Set to any configured role name ("verifier" or a role matching a
# [models.<name>] section) to offload cross-ref to a faster model while
# keeping reasoning for investigation. Empty / "primary" = use primary.
cross_ref_role = "{eng.cross_ref_role}"
# Per-phase model routing for the research-directive export pipeline.
# directive_primary handles HEAVY sections (test_plan, agentic_prompt,
# verification_criteria) — translating literature-watch predictions into
# in-house experiments and enforcing strict citation/tool grounding.
# REASONING MODEL recommended.
# directive_primary_fast handles LIGHT sections (hypothesis, eli5,
# research_path) — restatement / style / strategic prose. NON-REASONING
# MODEL recommended; falls back to directive_primary if unset.
# directive_verifier — directive grounding-review pass. REASONING
# recommended.
# Same resolution rules as cross_ref_role: empty / "primary" = use
# primary; "verifier" = use verifier; any other name must match a
# [models.<name>] section.
directive_primary_role = "{eng.directive_primary_role}"
directive_primary_fast_role = "{eng.directive_primary_fast_role}"
directive_verifier_role = "{eng.directive_verifier_role}"
# Total verifier passes the directive pipeline will run trying to land a
# clean output (initial + retries). Each retry regenerates the agentic
# prompt with the verifier's flags from the prior pass appended. The loop
# stops as soon as a pass returns clean OR this cap is reached. Output
# beyond the cap ships with a prominent "⚠ FLAGGED ISSUES" block.
directive_max_verification_passes = {eng.directive_max_verification_passes}
# Per-phase routing for the negative-space gap scan. Same resolution rules
# as cross_ref_role / directive_*_role.
# gap_scan_extract_role — matrix extraction (step 1). Reads all journal
# entries, identifies methods + problems + cell coverage. Multi-document
# categorization that benefits from inference: REASONING MODEL recommended.
# Empty / "primary" = use primary.
gap_scan_extract_role = "{eng.gap_scan_extract_role}"
# gap_scan_classify_role — cell classification (step 2) + question
# generation (step 4). Step 2 has large outputs (5K-10K tokens for ~100
# empty cells); on reasoning models in non-streaming mode this reliably
# triggers timeouts. NON-REASONING MODEL recommended; per-cell judgement
# is world-knowledge heavy, not chain-of-thought heavy.
# Empty / "verifier" = use verifier.
gap_scan_classify_role = "{eng.gap_scan_classify_role}"
# Investigation explore/verify split (Phase 5). The investigation loop
# runs hypothesis (pre-search) → investigate (with tools) → assess
# (post-search). Stages 1+2 are exploration; stage 3 is evaluation.
# Routing the assessor to a different model from the primary creates
# representational separation that resists self-grading. Empty /
# "primary" = use primary (default). "verifier" = cross-family
# separation (recommended). Any other name must match a [models.<name>]
# section.
investigation_assessor_role = "{eng.investigation_assessor_role}"
# Parallel fan-out. 1 = fully serial (default, preserves prior behavior).
# Higher values run multiple investigations / xref-synth+verify pipelines
# concurrently within a single cycle. Rate limiters are shared process-wide
# so raising these will NOT burst public APIs — waits are redistributed,
# wall-clock per cycle drops. Sensible ceilings: 3–4 investigations and
# 2–3 xref pipelines before rate-limit waits dominate anyway.
parallel_investigations = {eng.parallel_investigations}
parallel_xref_pipeline = {eng.parallel_xref_pipeline}
# ── Selective verification pipeline (A2 plumbing for Phases B/C/D) ──
# All defaults below are NO-OPS until the corresponding phase ships.
# Phase B — alias-gap routing. Candidates below reject_threshold short-
# circuit to rejection_log without LLM verification; above
# fasttrack_threshold skip deep prior-art search. Middle band runs the
# full pipeline. 0.0 / 1.0 = no gating (current behavior).
alias_gap_reject_threshold = {eng.alias_gap_reject_threshold}
alias_gap_fasttrack_threshold = {eng.alias_gap_fasttrack_threshold}
# Phase B audit-back — fraction of below-reject-threshold candidates that
# still get full LLM verification, tagged as audit samples. Detects
# discriminator drift. 0.0 = no audit-back (current behavior).
rejection_audit_back_rate = {eng.rejection_audit_back_rate}
# Phase C — paraphrase-perturbation. N variants per verification, verdict
# variance scored as paraphrase_inconsistency_score. 1 = single pass.
paraphrase_variant_count = {eng.paraphrase_variant_count}
# Phase D — committee escalation when paraphrase_inconsistency_score
# exceeds threshold (or verdicts disagree). 1.0 = never escalate.
committee_dissent_threshold = {eng.committee_dissent_threshold}
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
