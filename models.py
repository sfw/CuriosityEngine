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

    # Lifecycle (updated by prediction checks)
    status: str = "active"                   # active | validated_by_prediction | challenged_by_prediction

    # Human review (updated by --review-register)
    human_review_status: str = "unreviewed"   # unreviewed | approved | rejected | deferred
    human_review_notes: str = ""
    human_rejection_reason: str = ""          # required when status == "rejected"
    human_reviewer: str = ""
    human_review_at: str = ""


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

    status: str = "pending"                   # pending | confirmed | refuted | inconclusive | expired
    last_checked_at: str = ""
    review_log: list[dict] = field(default_factory=list)  # [{checked_at, verdict, reasoning, sources}, ...]


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
    connection: "CuriosityEngineConfig | None" = None
