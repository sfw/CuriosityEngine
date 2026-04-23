"""CuriosityEngine core: init, model plumbing, main run loop. Phases live in siblings."""

from __future__ import annotations

import sys
from typing import Optional

from journal import Journal
from models import EngineConfig
from providers import (
    AnthropicClient,
    EmbeddingClient,
    ModelClient,
    build_client,
    build_embedding_client,
)

from engine.cross_reference import CrossReferenceMixin
from engine.display import DisplayMixin
from engine.introspect import IntrospectionMixin
from engine.investigation import InvestigationMixin
from engine.tools import discover_tools, registry as tool_registry
from engine.verification import VerificationMixin


class CuriosityEngine(
    IntrospectionMixin,
    InvestigationMixin,
    CrossReferenceMixin,
    VerificationMixin,
    DisplayMixin,
):
    """Curiosity loop orchestrator.

    Pipeline:
      introspect → generate_questions → investigate
                                      → (every N cycles) cross_reference → synthesize → verify_insight → register
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        if config.connection is None:
            print("ERROR: engine config has no connection settings. "
                  "Load CuriosityEngineConfig.load() and attach it as config.connection.")
            sys.exit(1)
        self.connection = config.connection

        self.primary: ModelClient = build_client(self.connection.primary)
        self.verifier: ModelClient = build_client(self.connection.verifier)
        # cross_ref client: defaults to primary (backward compatible). When
        # connection.cross_ref is set (via [engine].cross_ref_role or
        # [models.cross_ref]), cross-reference calls go there instead — lets
        # the user offload cross-ref to a fast non-reasoning model while
        # keeping reasoning for investigation / synthesis / analog-probe.
        cross_ref_profile = getattr(self.connection, "cross_ref", None)
        if cross_ref_profile is None or cross_ref_profile is self.connection.primary:
            self.cross_ref_client: ModelClient = self.primary
        else:
            self.cross_ref_client = build_client(cross_ref_profile)

        self.journal = Journal(
            config.journal_path,
            register_markdown_path=config.register_markdown_path,
        )
        self.cycle_count = 0

        # Auto-discover tools the first time any engine spins up; idempotent.
        self.tool_registry = tool_registry
        discover_tools()

        print(f"  primary:  {self.connection.primary.provider} / {self.connection.primary.name}")
        print(f"  verifier: {self.connection.verifier.provider} / {self.connection.verifier.name}")
        if self.cross_ref_client is not self.primary:
            cr_profile = getattr(self.connection, "cross_ref", None)
            if cr_profile is not None:
                print(f"  cross_ref: {cr_profile.provider} / {cr_profile.name}")
        tool_names = self.tool_registry.names()
        if tool_names:
            print(f"  tools:    {', '.join(tool_names)}")

        # Attempt to construct an embedding client from whichever profile supports it.
        self.embedding_client: Optional[EmbeddingClient] = self._best_effort_embedding_client()
        if self.embedding_client is not None:
            print(f"  embed:    {self.embedding_client.model}")

    def _best_effort_embedding_client(self) -> Optional[EmbeddingClient]:
        """Try verifier profile first (usually already OpenAI-compat), then primary.
        Returns None if neither supports embeddings — engine still runs, just without
        semantic features."""
        for profile in (self.connection.verifier, self.connection.primary):
            try:
                return build_embedding_client(profile)
            except Exception:  # noqa: BLE001 — gracefully skip incompatible profiles
                continue
        return None

    # ── Model plumbing ──

    def _on_retry(self, attempt, max_attempts, error, delay):
        print(f"  [retry {attempt}/{max_attempts} in {delay:.1f}s: {type(error).__name__}]")

    def _call_primary(
        self,
        prompt: str,
        *,
        tools: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        return self.primary.complete_json(
            prompt,
            tools=tools if self.primary.supports_server_web_search else None,
            max_tokens=max_tokens,
            policy=self.connection.retry,
            on_retry=self._on_retry,
        )

    def _call_cross_ref(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Route cross-reference generation to the dedicated cross_ref client.
        Falls back to the primary client when no cross_ref profile is set."""
        return self.cross_ref_client.complete_json(
            prompt,
            tools=None,
            max_tokens=max_tokens,
            policy=self.connection.retry,
            on_retry=self._on_retry,
        )

    def _call_verifier(
        self,
        prompt: str,
        *,
        tools: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        return self.verifier.complete_json(
            prompt,
            tools=tools if self.verifier.supports_server_web_search else None,
            max_tokens=max_tokens,
            policy=self.connection.retry,
            on_retry=self._on_retry,
        )

    # Legacy alias — some callers still use _call_model; routes to primary.
    def _call_model(
        self,
        prompt: str,
        *,
        tools: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        return self._call_primary(prompt, tools=tools, max_tokens=max_tokens)

    # Tool-enabled paths (multi-turn loops). Use these from investigation / verify /
    # prediction-check phases where the model needs to issue tool calls.

    def _focus_block(self) -> str:
        """Render the user-set investigation focus as a prompt section, or empty string."""
        focus = (self.journal.focus or "").strip()
        if not focus:
            return ""
        return (
            "USER FOCUS (treat as a hard constraint — the user has directed the engine "
            "to concentrate on this narrow area within the broader domain):\n"
            f"  {focus}\n\n"
        )

    def _tool_list_block(self, for_client: Optional[ModelClient] = None) -> str:
        """Human-readable bullet list of tools available to a specific client."""
        for_client = for_client or self.primary
        is_anthropic = isinstance(for_client, AnthropicClient)
        lines = []
        for cls in self.tool_registry.all():
            if is_anthropic and cls.name in self._ANTHROPIC_RESERVED_TOOL_NAMES:
                continue  # Anthropic's server version of this tool supersedes ours
            first_line = cls.description.split(". ", 1)[0].strip() + "."
            lines.append(f"- `{cls.name}`: {first_line}")
        if is_anthropic:
            lines.append("- `web_search`: Anthropic native web search (live results).")
            lines.append(
                "- `code_execution`: Anthropic native sandboxed Python runtime. Use to test "
                "hypotheses computationally, verify math, simulate small models, plot data."
            )
        return "\n".join(lines)

    # Names that Anthropic provides server-side — our client tools must not shadow them
    # when Anthropic is the active client (duplicate tool names in one request = ambiguity).
    _ANTHROPIC_RESERVED_TOOL_NAMES = frozenset({"web_search", "code_execution"})

    def _client_tool_schemas(self, client) -> list[dict]:
        """Per-provider schemas for the currently-registered client tools."""
        if isinstance(client, AnthropicClient):
            schemas = self.tool_registry.anthropic_schemas()
            return [s for s in schemas if s.get("name") not in self._ANTHROPIC_RESERVED_TOOL_NAMES]
        return self.tool_registry.openai_schemas()

    def _call_primary_with_tools(
        self,
        prompt: str,
        *,
        server_tools: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        return self.primary.complete_json_with_tools(
            prompt,
            client_tools=self._client_tool_schemas(self.primary),
            server_tools=server_tools if self.primary.supports_server_web_search else None,
            tool_registry=self.tool_registry,
            max_tokens=max_tokens,
            policy=self.connection.retry,
            on_retry=self._on_retry,
        )

    def _call_verifier_with_tools(
        self,
        prompt: str,
        *,
        server_tools: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        return self.verifier.complete_json_with_tools(
            prompt,
            client_tools=self._client_tool_schemas(self.verifier),
            server_tools=server_tools if self.verifier.supports_server_web_search else None,
            tool_registry=self.tool_registry,
            max_tokens=max_tokens,
            policy=self.connection.retry,
            on_retry=self._on_retry,
        )

    # ── Main loop ──

    def run_cycle(self) -> dict:
        self.cycle_count += 1
        print(f"\n{'='*60}")
        print(f"  CURIOSITY CYCLE {self.cycle_count}")
        print(f"{'='*60}")

        uncertainties = self.introspect()
        questions = self.generate_questions(uncertainties)

        # Build the investigation pool in priority order:
        #   1. Human-queued questions (always first — explicit human direction).
        #   2. Emergent queued questions (entry followups, xref followups, analog
        #      probes) ranked by the priority_score assigned at enqueue time.
        #   3. Fresh introspection-generated questions when queue is insufficient.
        from uuid import uuid4 as _uuid4
        from models import ResearchQuestion as _Q

        budget = self.config.investigations_per_cycle
        investigation_pool = []

        def _queued_to_rq(item: dict, priority: float) -> _Q:
            return _Q(
                id=f"q-{_uuid4().hex[:8]}",
                question=item.get("question", ""),
                source_uncertainties=[],
                priority_score=priority,
                domain_tags=[],
                investigability_notes=f"queued ({item.get('source','?')})",
            )

        human_queued = self.journal.pop_questions_by_source("human", limit=budget)
        for hq in human_queued:
            investigation_pool.append(_queued_to_rq(hq, 1.0))
            print(f"  [human-directed] {hq.get('question', '')[:100]}")

        remaining = max(0, budget - len(investigation_pool))
        if remaining > 0 and self.journal.question_queue:
            # Pop highest-priority non-human queued questions to fill the budget.
            emergent = self.journal.pop_queued_questions(remaining)
            for eq in emergent:
                pri = float(eq.get("priority", 0.5))
                investigation_pool.append(_queued_to_rq(eq, pri))
                src = eq.get("source", "?")
                print(f"  [queued pri={pri:.2f} src={src}] {eq.get('question', '')[:90]}")

        remaining = max(0, budget - len(investigation_pool))
        if remaining > 0:
            investigation_pool.extend(questions[:remaining])

        entries = []
        for q in investigation_pool[:budget]:
            entry = self.investigate(q)
            entries.append(entry)

        xrefs = []
        insights = []
        registered = []
        if self.cycle_count % self.config.cross_ref_frequency == 0:
            xrefs = self.cross_reference()
            for xref in xrefs:
                if xref.novelty_score >= self.config.novelty_threshold:
                    insight = self.synthesize(xref)
                    if insight:
                        insights.append(insight)
                        if self.config.verify_insights:
                            register_entry = self.verify_insight(insight, xref)
                            if register_entry:
                                registered.append(register_entry)

        return {
            "cycle": self.cycle_count,
            "uncertainties_found": len(uncertainties),
            "questions_generated": len(questions),
            "entries_created": len(entries),
            "cross_references_found": len(xrefs),
            "insights_generated": len(insights),
            "registered": len(registered),
        }

    def run(self, n_cycles: int):
        print("\nStarting Curiosity Engine")
        print(f"Domain: {self.config.domain}")
        print(f"Journal: {self.config.journal_path}")
        print(f"Planned cycles: {n_cycles}")

        results = []
        for _ in range(n_cycles):
            result = self.run_cycle()
            results.append(result)
            print(
                f"\n  Cycle {result['cycle']} complete: "
                f"{result['entries_created']} entries, "
                f"{result['cross_references_found']} cross-refs, "
                f"{result['insights_generated']} insights, "
                f"{result.get('registered', 0)} registered"
            )

        print(f"\n{'='*60}")
        print("  SESSION COMPLETE")
        print(f"{'='*60}")
        print(f"  Total journal entries: {len(self.journal.entries)}")
        print(f"  Total cross-references: {len(self.journal.cross_references)}")
        print(f"  Total insights: {len(self.journal.insights)}")
        print(f"  Register entries (validated): {len(self.journal.register)}")
        print(f"  Predictions on file:          {len(self.journal.predictions)}")

        if self.journal.insights:
            print("\n  INSIGHTS DISCOVERED:")
            for insight in self.journal.insights:
                print(f"\n  [{insight['confidence']:.2f}] {insight['title']}")
                print(f"  {insight['description'][:150]}...")

        return results
