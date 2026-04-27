"""Cross-reference (Phase 4) + synthesize (Phase 5)."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from models import CrossReference, Insight
from prompts import CROSS_REFERENCE_PROMPT, SYNTHESIZE_PROMPT, SYNTHESIZE_VARIANTS_PROMPT


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

        # Routes to self.primary when no cross_ref profile is configured.
        result = self._call_cross_ref(prompt)
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

    def synthesize_orphaned_xrefs(self) -> dict:
        """Synthesize + verify any cross-reference that doesn't yet have a matching
        Insight — e.g. because a prior run died between cross-ref and synthesis.

        Dedup: an xref is "orphaned" iff its id does not appear in any
        Insight.supporting_evidence. Already-synthesized xrefs are skipped.
        Insights created here flow through the normal verify_insight() gate, so
        the register / held / reject paths apply exactly as in a fresh run.
        """
        from models import CrossReference as _CR

        existing_supports: set[str] = set()
        for i in self.journal.insights:
            for sid in (i.get("supporting_evidence") or []):
                existing_supports.add(sid)

        orphans: list[dict] = []
        for x in self.journal.cross_references:
            xid = x.get("id")
            if not xid or xid in existing_supports:
                continue
            if float(x.get("novelty_score", 0.0) or 0.0) < self.config.novelty_threshold:
                continue  # wasn't going to be synthesized originally; skip
            orphans.append(x)

        print(f"\n--- SYNTHESIZING {len(orphans)} ORPHANED CROSS-REFERENCE(S) ---")
        stats = {"synthesized": 0, "registered": 0, "held": 0, "rejected": 0, "errors": 0}

        for x_dict in orphans:
            xref_fields = set(_CR.__dataclass_fields__)
            xref = _CR(**{k: v for k, v in x_dict.items() if k in xref_fields})
            try:
                insight = self.synthesize(xref)
            except Exception as e:  # noqa: BLE001
                print(f"  [error] {xref.id}: {type(e).__name__}: {e}")
                stats["errors"] += 1
                continue
            if insight is None:
                continue
            stats["synthesized"] += 1

            if not self.config.verify_insights:
                continue
            try:
                register_entry = self.verify_insight(insight, xref)
            except Exception as e:  # noqa: BLE001
                print(f"  [error verify] {insight.id}: {type(e).__name__}: {e}")
                stats["errors"] += 1
                continue
            if register_entry is None:
                stats["rejected"] += 1
            elif register_entry.status == "held":
                stats["held"] += 1
            else:
                stats["registered"] += 1

        print(
            f"\nOrphan-synth complete: synthesized={stats['synthesized']}  "
            f"registered={stats['registered']}  held={stats['held']}  "
            f"rejected={stats['rejected']}  errors={stats['errors']}"
        )
        return stats

    def synthesize(self, xref: CrossReference) -> Optional[Insight]:
        """Synthesize an Insight from a cross-reference.

        Phase 6 of the self-evolving verifier: when configured for
        candidate_count > 1, generate N divergent candidate insights,
        canonicalize each, and select the candidate with the largest
        alias-gap to the existing register. Ties broken by self-reported
        confidence. Borrows from Co-Scientist's Ranking agent + agentic
        best-of-N selection. Falls back to single-candidate synthesis
        when synthesis_candidate_count == 1 (pre-Phase-6 behavior).
        """
        print("\n--- SYNTHESIZING INSIGHT ---")

        supporting = [e for e in self.journal.entries if e["id"] in xref.source_entries]

        candidate_count = max(
            1, int(getattr(self.connection.engine, "synthesis_candidate_count", 3)),
        )

        if candidate_count <= 1:
            # Pre-Phase-6 path: single candidate, no tournament.
            prompt = SYNTHESIZE_PROMPT.format(
                focus_block=self._focus_block(),
                xref_json=json.dumps(asdict(xref), indent=2),
                supporting_entries_json=json.dumps(supporting, indent=2),
            )
            result = self._call_primary(prompt)
            return self._build_and_persist_insight(result, xref, divergence_axis="")

        # Phase 6: tournament-ranked Best-of-N.
        prompt = SYNTHESIZE_VARIANTS_PROMPT.format(
            focus_block=self._focus_block(),
            xref_json=json.dumps(asdict(xref), indent=2),
            supporting_entries_json=json.dumps(supporting, indent=2),
            candidate_count=candidate_count,
        )
        try:
            result = self._call_primary(prompt)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] variant synthesis failed: {type(e).__name__}: {e}")
            return None

        candidates = list(result.get("candidates") or [])
        if not candidates:
            print("  [warn] variant synthesis returned no candidates")
            return None

        print(f"  [tournament] {len(candidates)} candidate(s) generated; scoring by alias-gap…")
        winner = self._select_synthesis_candidate(candidates, xref)
        if winner is None:
            print("  [warn] tournament selected no winner — using highest-confidence candidate as fallback")
            winner = max(
                candidates, key=lambda c: float(c.get("confidence", 0) or 0),
            )

        return self._build_and_persist_insight(
            winner, xref, divergence_axis=(winner.get("divergence_axis") or "").strip(),
        )

    def _select_synthesis_candidate(
        self, candidates: list[dict], xref: CrossReference,
    ) -> Optional[dict]:
        """Tournament selector: canonicalize each candidate, score by
        alias-gap to the existing register, return the most-distant
        candidate. Confidence breaks ties when alias-gaps are within
        ALIAS_GAP_TIE_TOLERANCE of each other.

        Falls back to the most-confident candidate if canonicalization
        produces no usable canonical forms across the cohort.
        """
        # ALIAS_GAP_TIE_TOLERANCE — when two candidates' alias-gaps are
        # within this much of each other, treat them as tied and break
        # the tie by self-reported confidence. Keeps the tournament from
        # being decided by tiny embedding-similarity noise.
        ALIAS_GAP_TIE_TOLERANCE = 0.02

        scored: list[tuple[float, float, int, dict, dict]] = []
        for idx, c in enumerate(candidates):
            title = (c.get("title") or "").strip()
            description = (c.get("description") or "").strip()
            # Try to canonicalize using a synthetic central_move derived
            # from the candidate's own title — _canonicalize_central_move
            # accepts an empty central_move and extracts from description
            # when needed.
            try:
                canonical = self._canonicalize_central_move(
                    title, description, central_architectural_move=title,
                )
            except Exception as e:  # noqa: BLE001 — canonicalization is best-effort
                print(f"    [warn] canonicalize failed on candidate {idx + 1}: {type(e).__name__}: {e}")
                canonical = {}
            # Score: alias-gap if canonical form usable; 0 (worst) otherwise
            if canonical and canonical.get("move_predicate"):
                gap_signal = self._alias_gap(canonical)
                gap = gap_signal.get("gap", 1.0)
            else:
                gap = 0.0
                gap_signal = {"gap": 0.0, "nearest_ids": [], "scored_against": 0}
            conf = float(c.get("confidence") or 0.0)
            scored.append((gap, conf, idx, c, canonical))
            div = (c.get("divergence_axis") or "").strip()[:60]
            print(
                f"    candidate {idx + 1}: gap={gap:.2f} conf={conf:.2f}"
                + (f" · axis={div!r}" if div else "")
            )

        # Sort by gap desc; among entries within ALIAS_GAP_TIE_TOLERANCE of
        # the top, prefer the higher confidence.
        scored.sort(key=lambda s: s[0], reverse=True)
        best_gap = scored[0][0]
        tied = [s for s in scored if abs(s[0] - best_gap) <= ALIAS_GAP_TIE_TOLERANCE]
        if len(tied) > 1:
            tied.sort(key=lambda s: s[1], reverse=True)
            print(
                f"  [tournament] {len(tied)} candidates within "
                f"{ALIAS_GAP_TIE_TOLERANCE:.2f} of best gap; "
                "tiebreaking by confidence."
            )
        winner = tied[0]
        winner_idx = winner[2]
        print(
            f"  [tournament] winner: candidate {winner_idx + 1} "
            f"(gap={winner[0]:.2f}, conf={winner[1]:.2f})"
        )
        return winner[3]

    def _build_and_persist_insight(
        self, result: dict, xref: CrossReference, divergence_axis: str = "",
    ) -> Insight:
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
        if divergence_axis:
            print(f"  Divergence axis (selected): {divergence_axis[:140]}")
        if insight.prior_art_check:
            print(f"  Prior art: {insight.prior_art_check[:120]}...")
        print(f"  {insight.description[:200]}...")
        return insight
