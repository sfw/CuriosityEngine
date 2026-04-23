"""Cross-reference (Phase 4) + synthesize (Phase 5)."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from models import CrossReference, Insight
from prompts import CROSS_REFERENCE_PROMPT, SYNTHESIZE_PROMPT


class CrossReferenceMixin:
    """Find connections across journal entries and promote the best to insights."""

    @staticmethod
    def _slim_entry_for_xref(entry: dict) -> dict:
        """Strip heavy fields that the cross-ref prompt doesn't need."""
        return {
            "id": entry.get("id"),
            "question": entry.get("question"),
            "key_takeaways": entry.get("key_takeaways", []),
            "domain_tags": entry.get("domain_tags", []),
            "surprise_delta": entry.get("surprise_delta", 0.0),
            "hypothesis_verdict": entry.get("hypothesis_verdict"),
        }

    def _select_entries_for_xref(self) -> list[dict]:
        """Pick the entries to cross-reference.

        Uses the knowledge graph to prioritize entries with *unexplored* pairwise
        connections (shared tags / sources that no existing cross-reference has
        captured yet) + recent-baseline + surprise bias. This targets the
        'intersection of knowledge gaps' rather than just the chronological tail.
        """
        from engine.graph import select_entries_for_xref
        pool = select_entries_for_xref(self.journal, window=self.config.cross_ref_window)
        return [self._slim_entry_for_xref(e) for e in pool]

    @staticmethod
    def _slim_xref_for_prompt(xref: dict) -> dict:
        return {
            "id": xref.get("id"),
            "source_entries": xref.get("source_entries", []),
            "connection_type": xref.get("connection_type"),
            "description": xref.get("description", ""),
            "novelty_score": xref.get("novelty_score"),
        }

    def cross_reference(self) -> list[CrossReference]:
        print("\n--- CROSS-REFERENCING ---")

        entries = self.journal.entries
        if len(entries) < 2:
            print("  Not enough entries to cross-reference yet.")
            return []

        slim_entries = self._select_entries_for_xref()
        existing_xrefs = [self._slim_xref_for_prompt(x) for x in self.journal.cross_references]
        print(f"  Analyzing {len(slim_entries)} of {len(entries)} entries (slimmed); "
              f"{len(existing_xrefs)} existing cross-ref(s) shown as priors...")
        entries_json = json.dumps(slim_entries, indent=2)
        existing_xrefs_json = json.dumps(existing_xrefs, indent=2) if existing_xrefs else "[]"
        prompt = CROSS_REFERENCE_PROMPT.format(
            focus_block=self._focus_block(),
            entries_json=entries_json,
            existing_xrefs_json=existing_xrefs_json,
        )

        result = self._call_primary(prompt)
        existing_keys = {
            (tuple(sorted(x.get("source_entries", []))), x.get("connection_type"))
            for x in self.journal.cross_references
        }
        existing_participant_sets = [
            frozenset(x.get("source_entries", []) or [])
            for x in self.journal.cross_references
        ]

        def _participant_overlap(candidate: set[str]) -> float:
            """Max overlap coefficient between candidate and any existing xref.
            Overlap coefficient = |A ∩ B| / min(|A|, |B|) — 1.0 means subset."""
            if not candidate:
                return 0.0
            best = 0.0
            for prior in existing_participant_sets:
                if not prior:
                    continue
                inter = len(candidate & prior)
                denom = min(len(candidate), len(prior))
                if denom == 0:
                    continue
                best = max(best, inter / denom)
            return best

        xrefs = []
        skipped = 0
        skipped_attractor = 0
        for x in result.get("cross_references", []):
            source_ids = x.get("source_entry_ids", [])
            connection_type = x.get("connection_type", "pattern")
            key = (tuple(sorted(source_ids)), connection_type)
            if key in existing_keys:
                skipped += 1
                continue
            # Anti-attractor gate: if the participant set heavily overlaps an existing
            # xref's participants, the insight space is likely to repeat. Require either
            # a very high novelty_score (>= 0.85) to justify reentering, or drop.
            overlap = _participant_overlap(set(source_ids))
            claimed_novelty = float(x.get("novelty_score", 0.5) or 0.0)
            if overlap >= 0.5 and claimed_novelty < 0.85:
                skipped_attractor += 1
                continue
            existing_keys.add(key)

            xref = CrossReference(
                id=f"x-{uuid4().hex[:8]}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                source_entries=source_ids,
                connection_type=connection_type,
                description=x["description"],
                novelty_score=x.get("novelty_score", 0.5),
                implications=x.get("implications", []),
                suggested_questions=x.get("suggested_questions", []),
            )
            xrefs.append(xref)
            self.journal.add_cross_reference(xref)
            for entry_id in xref.source_entries:
                self.journal.annotate_connection(entry_id, xref.id)
            # Priority: xref novelty_score is the direct reward signal — novel connections
            # that raise interesting follow-ups should jump ahead of stale entry-level leftovers.
            xref_priority = max(0.4, min(1.0, 0.4 + 0.6 * float(xref.novelty_score or 0.0)))
            self._enqueue_questions(
                xref.suggested_questions,
                source=f"xref:{xref.id}",
                priority=xref_priority,
            )
            print(f"  [{xref.connection_type}] (novelty={xref.novelty_score:.2f}) {xref.description[:80]}...")

        if skipped:
            print(f"  Skipped {skipped} duplicate cross-reference(s).")
        if skipped_attractor:
            print(f"  Skipped {skipped_attractor} attractor-basin cross-reference(s) "
                  "(participant overlap ≥50%, novelty<0.85).")

        if xrefs:
            self.journal.save()

        return xrefs

    def synthesize(self, xref: CrossReference) -> Optional[Insight]:
        print("\n--- SYNTHESIZING INSIGHT ---")

        supporting = [e for e in self.journal.entries if e["id"] in xref.source_entries]

        prompt = SYNTHESIZE_PROMPT.format(
            focus_block=self._focus_block(),
            xref_json=json.dumps(asdict(xref), indent=2),
            supporting_entries_json=json.dumps(supporting, indent=2),
        )

        result = self._call_primary(prompt)

        insight = Insight(
            id=f"i-{uuid4().hex[:8]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            title=result.get("title", "Untitled Insight"),
            description=result.get("description", ""),
            supporting_evidence=[xref.id] + xref.source_entries,
            novelty_assessment=result.get("novelty_assessment", ""),
            confidence=result.get("confidence", 0.5),
            implications=result.get("implications", []),
            open_questions=result.get("open_questions", []),
            counter_arguments=result.get("counter_arguments", []),
            prior_art_check=result.get("prior_art_check", ""),
        )

        self.journal.add_insight(insight)
        print(f"  INSIGHT: {insight.title}")
        print(f"  Confidence: {insight.confidence:.2f}")
        if insight.prior_art_check:
            print(f"  Prior art: {insight.prior_art_check[:120]}...")
        print(f"  {insight.description[:200]}...")

        return insight
