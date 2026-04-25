"""Data models for the Curiosity Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import CuriosityEngineConfig


@dataclass
class UncertaintyItem:
    id: str
    description: str
    uncertainty_type: str  # "contradiction" | "gap" | "shallow" | "unstable"
    domain_tags: list[str]
    estimated_importance: float
    related_items: list[str] = field(default_factory=list)


@dataclass
class ResearchQuestion:
    id: str
    question: str
    source_uncertainties: list[str]
    priority_score: float
    domain_tags: list[str]
    investigability_notes: str
    status: str = "pending"


@dataclass
class JournalEntry:
    id: str
    timestamp: str
    question_id: str
    question: str
    hypothesis: str
    confidence_before: float
    methodology: str
    raw_findings: str
    sources: list[str]
    surprise_delta: float
    confidence_after: float
    key_takeaways: list[str]
    new_questions: list[str]
    domain_tags: list[str]
    connections_to: list[str] = field(default_factory=list)
    cross_reference_notes: list[str] = field(default_factory=list)
    # Added so the UI can show *why* surprise is surprising without scanning raw_findings.
    hypothesis_verdict: str = ""               # confirmed | partially_confirmed | contradicted | unresolved
    surprise_explanation: str = ""             # one-paragraph reason for the surprise


@dataclass
class CrossReference:
    id: str
    timestamp: str
    source_entries: list[str]
    connection_type: str  # "pattern" | "contradiction" | "convergence" | "implication"
    description: str
    novelty_score: float
    implications: list[str]
    suggested_questions: list[str]


@dataclass
class Insight:
    id: str
    timestamp: str
    title: str
    description: str
    supporting_evidence: list[str]
    novelty_assessment: str
    confidence: float
    implications: list[str]
    open_questions: list[str]
    counter_arguments: list[str]
    prior_art_check: str = ""


@dataclass
class RegisterEntry:
    """An insight that survived adversarial verification. The durable artifact of the system."""

    id: str
    timestamp: str
    insight_id: str
    title: str
    description: str

    # Substantiation
    supporting_xref_id: str
    supporting_entry_ids: list[str]
    supporting_entry_summaries: list[dict]  # [{id, question, key_takeaways}, ...]
    supporting_sources: list[str]           # web sources aggregated from supporting entries

    # Motivation
    motivation: str
    implications: list[str]

    # Verification
    verdict: str                             # "validated" (only these reach the register)
    verified_confidence: float
    prior_art_found: bool                    # legacy — kept for back-compat; = synthesis_findable
    prior_art_citations: list[str]           # legacy — kept for back-compat; = synthesis_prior_art
    contradicting_findings: list[str]
    reasoning_flaws_considered: list[str]
    verification_summary: str

    # Novelty decomposition (new axis — "premises supported, synthesis not findable" IS the novelty signature)
    premises_supported: bool = True
    premises_support_citations: list[str] = field(default_factory=list)
    synthesis_findable: bool = False
    synthesis_prior_art: list[str] = field(default_factory=list)
    novelty_type: str = ""                   # new_synthesis | restatement | extension | correction | unsupported

    # Carried from the source insight
    open_questions: list[str] = field(default_factory=list)
    counter_arguments: list[str] = field(default_factory=list)

    # Lifecycle (updated by prediction checks + human review)
    # active                     — verified, in the durable register
    # held                       — verifier returned `inconclusive` (couldn't reach, not refuted); awaiting settlement
    # validated_by_prediction    — promoted to active after attached prediction(s) resolved confirmed
    # challenged_by_prediction   — one or more attached predictions resolved refuted
    status: str = "active"

    # Held-state metadata — populated when the verdict was `inconclusive`.
    held_reason: str = ""
    settlement_method: str = ""               # concrete method for settling this later
    settlement_horizon: str = ""              # ISO YYYY-MM-DD target for the settlement signal
    settlement_triggers: list[str] = field(default_factory=list)  # observable outcomes that would promote or refute
    promoted_at: str = ""                     # timestamp when held → active
    promoted_by: str = ""                     # "human:<reviewer>" | "prediction:<id>"

    # Human review (updated by --review-register)
    human_review_status: str = "unreviewed"   # unreviewed | approved | rejected | deferred
    human_review_notes: str = ""
    human_rejection_reason: str = ""          # required when status == "rejected"
    human_reviewer: str = ""
    human_review_at: str = ""

    # Verifier audit trail — full per-call trace of the tool invocations that
    # produced this register entry. Each entry:
    # {iteration, tool, kind, args, result_length, result_preview, is_error}
    # Used by the UI to surface "what did the verifier actually search for?"
    # and by the admin re-verify pass for diagnosing misses.
    verification_tool_calls: list[dict] = field(default_factory=list)

    # Phase-structured prior-art outputs from the verifier (see prompts.VERIFY_PROMPT).
    # Feeds the extension-vs-new_synthesis verdict guard and the register-detail UI.
    central_architectural_move: str = ""
    central_move_prior_art: list[str] = field(default_factory=list)
    functional_decomposition: list[dict] = field(default_factory=list)
    closest_peer_system: dict = field(default_factory=dict)
    skeptic_probe: dict = field(default_factory=dict)
    # Claim's target application domain as the verifier named it in phase 3b.
    # Used to detect when the verifier fell back to a memorized example
    # rather than deriving the domain from the claim text.
    target_application_domain: str = ""

    # Per-anchor evaluations produced by the verifier against the journal's
    # known_prior_art list (human-curated peers). One entry per injected
    # anchor: {anchor_id, is_peer, overlaps_claim, differentiators, reasoning}.
    # Empty if the journal has no known_prior_art entries for this domain.
    known_prior_art_evaluations: list[dict] = field(default_factory=list)

    # Append-only log of re-verification passes triggered from the admin UI.
    # Each entry: {timestamp, verdict, verified_confidence, novelty_type,
    # synthesis_findable, verification_summary, tool_calls, reason_for_reverify}.
    # Original verdict fields on this RegisterEntry are NEVER overwritten —
    # the re-verify flow only appends here, preserving the audit trail.
    reverification_log: list[dict] = field(default_factory=list)


@dataclass
class Prediction:
    """A falsifiable claim attached to a RegisterEntry, checkable over time."""

    id: str
    register_entry_id: str
    created_at: str
    target_date: str                          # ISO date when this prediction should be checked

    claim: str                                # what is predicted
    falsifiable_condition: str                # what observable outcome would confirm/refute it
    check_method: str                         # how to verify (e.g., "search for papers X/Y", "run experiment Z")

    status: str = "pending"                   # pending | confirmed | refuted | inconclusive | expired | already_fulfilled
    last_checked_at: str = ""
    review_log: list[dict] = field(default_factory=list)  # [{checked_at, verdict, reasoning, sources}, ...]

    # Freshness check performed before the prediction registered. If the
    # falsifiable condition was already observably instantiated in the world
    # at creation time, the prediction isn't falsifiable — it's a description.
    # freshness_check.verdict is "fresh" (condition not yet observed), "already_fulfilled"
    # (condition is already true; prediction skipped or marked already_fulfilled),
    # or "skipped" (check disabled or tool unavailable).
    freshness_check: dict = field(default_factory=dict)


@dataclass
class EngineConfig:
    """Operational settings for a single run. Model connection settings live in
    CuriosityEngineConfig (loaded from ~/.CuriosityEngine/engine.toml) and are
    attached here as `connection`."""

    domain: str = "AI/ML research"
    journal_path: str = "./research_journal.json"
    register_markdown_path: str = "./register.md"
    questions_per_cycle: int = 3
    investigations_per_cycle: int = 1
    cross_ref_frequency: int = 3
    cross_ref_window: int = 20  # max entries sent to cross-ref prompt
    novelty_threshold: float = 0.7
    verify_insights: bool = True
    register_confidence_floor: float = 0.6
    max_cycles: int = 10
    analog_probe_enabled: bool = True
    analog_probe_surprise_threshold: float = 0.5
    analog_probe_max_analogs: int = 3
    assumption_probe_enabled: bool = True
    assumption_probe_surprise_threshold: float = 0.3
    assumption_probe_max_assumptions: int = 3
    negative_space_min_entries: int = 15
    gap_verification_hit_threshold: int = 5
    # When an engine-side guard downgrades the verifier's verdict (e.g. the
    # skeptic-probe guard flips validated→challenged), the LLM's returned
    # confidence was computed BEFORE knowing the guard would fire — flat
    # confidence on a verdict change is the hedge signature. This value is
    # subtracted from verified_confidence whenever a guard downgrades the
    # verdict, so the stored confidence reflects the revised assessment.
    # Set to 0 to disable the penalty.
    confidence_drop_on_downgrade: float = 0.10
    # Minimum priority for questions to enter the queue. Non-human sources with
    # priority below this floor are dropped at enqueue time (with a log line).
    # Human-sourced questions always bypass — explicit intent overrides the
    # autoscreen. Default 0.70. Set to 0 to disable.
    question_priority_floor: float = 0.70
    held_entries_enabled: bool = True
    held_confidence_floor: float = 0.7
    # Parallel fan-out knobs. Default 1 = serial (preserves prior behavior
    # exactly). Rate limiters in engine/tools/_rate_limits.py are shared
    # process-wide, so raising these does NOT burst public APIs — the
    # limiter is the hard guarantee against upstream blocks.
    parallel_investigations: int = 1
    parallel_xref_pipeline: int = 1
    # Total verifier passes the directive pipeline runs trying to land a clean
    # output (initial + retries). Each retry regenerates the agentic prompt
    # with the verifier's flags appended. Loop stops on first clean pass or
    # when this cap is reached. 1 = no retries; 3 = default (gives one extra
    # round of convergence after the first round of fixes).
    directive_max_verification_passes: int = 3
    connection: "CuriosityEngineConfig | None" = None
