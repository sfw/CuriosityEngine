"""CuriosityEngine core: init, model plumbing, main run loop. Phases live in siblings."""

from __future__ import annotations

import os
import sys
from typing import Optional

from journal import Journal
from models import EngineConfig
from providers import ModelClient, build_client

from engine.cross_reference import CrossReferenceMixin
from engine.display import DisplayMixin
from engine.introspect import IntrospectionMixin
from engine.investigation import InvestigationMixin
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

        self.journal = Journal(
            config.journal_path,
            register_markdown_path=config.register_markdown_path,
        )
        self.cycle_count = 0

        print(f"  primary:  {self.connection.primary.provider} / {self.connection.primary.name}")
        print(f"  verifier: {self.connection.verifier.provider} / {self.connection.verifier.name}")

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

    # ── Main loop ──

    def run_cycle(self) -> dict:
        self.cycle_count += 1
        print(f"\n{'='*60}")
        print(f"  CURIOSITY CYCLE {self.cycle_count}")
        print(f"{'='*60}")

        uncertainties = self.introspect()
        questions = self.generate_questions(uncertainties)

        entries = []
        for q in questions[: self.config.investigations_per_cycle]:
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
