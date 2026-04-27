"""Investigation phase (Phase 3): three-stage hypothesis → web_search → surprise."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from models import JournalEntry, ResearchQuestion
from prompts import (
    ANALOG_PROBE_PROMPT,
    ASSUMPTION_PROBE_PROMPT,
    HYPOTHESIS_PROMPT,
    HYPOTHESIS_VARIANTS_PROMPT,
    INVESTIGATE_PROMPT,
    SURPRISE_PROMPT,
)


def _calibrate_confidence_after(surprise: dict, c0: float) -> None:
    """Enforce the calibration rules stated in SURPRISE_PROMPT in-place.

    LLMs regularly bump confidence upward on partially_confirmed results even
    though the evidence showed the prior was oversimplified. This is a tight
    clamp that only fires when the model's C₁ clearly violates a rule — it never
    invents values beyond what the verdict and delta justify.
    """
    verdict = (surprise.get("hypothesis_verdict") or "").strip().lower()
    delta = float(surprise.get("surprise_delta", 0.0) or 0.0)
    c1 = float(surprise.get("confidence_after", c0) or c0)

    if verdict == "partially_confirmed" and delta < 0.2 and c1 > c0:
        # Oversimplified prior, low surprise: cannot go up.
        surprise["confidence_after"] = c0
    elif verdict == "contradicted" and c1 > c0 - 0.2:
        # Contradicted: must drop meaningfully.
        surprise["confidence_after"] = max(0.0, c0 - 0.2)
    elif verdict == "unresolved" and abs(c1 - c0) < 0.05:
        # Unresolved: pull toward 0.5 by at least half the prior's deviation.
        surprise["confidence_after"] = c0 + 0.5 * (0.5 - c0)


class InvestigationMixin:
    """The hypothesis-first investigation loop. Commits to a prior before searching,
    so surprise is a comparison rather than a self-report."""

    def _form_hypothesis(self, question: ResearchQuestion) -> dict:
        """Phase 9: when configured for variant_count > 1, the explorer
        generates N divergent candidate hypotheses and the system selects
        the one most distant from majority-literature consensus to drive
        the investigation. The unselected variants are stored in the
        returned dict under `considered_variants` for audit. Falls back
        to single-hypothesis (pre-Phase-9 behavior) when count == 1."""
        variant_count = max(1, min(5, int(
            getattr(self.connection.engine, "hypothesis_variant_count", 3),
        )))

        if variant_count <= 1:
            prompt = HYPOTHESIS_PROMPT.format(
                domain=self.config.domain,
                question=question.question,
            )
            return self._call_primary(prompt)

        # Phase 9: generate variants, select most-divergent, return.
        prompt = HYPOTHESIS_VARIANTS_PROMPT.format(
            domain=self.config.domain,
            question=question.question,
            variant_count=variant_count,
        )
        try:
            result = self._call_primary(prompt)
        except Exception as e:  # noqa: BLE001 — fall back to single-hypothesis on failure
            print(f"  [warn] variant hypothesis failed: {type(e).__name__}: {e}; falling back to single-hypothesis path")
            fallback_prompt = HYPOTHESIS_PROMPT.format(
                domain=self.config.domain, question=question.question,
            )
            return self._call_primary(fallback_prompt)

        candidates = list(result.get("candidates") or [])
        if not candidates:
            print("  [warn] variant hypothesis returned no candidates; falling back to single-hypothesis path")
            fallback_prompt = HYPOTHESIS_PROMPT.format(
                domain=self.config.domain, question=question.question,
            )
            return self._call_primary(fallback_prompt)

        # Selection rule (Phase 9 diverge mode): pick the candidate whose
        # divergence_axis is most distinct from the others. Heuristic — when
        # all candidates name a divergence_axis, pick the one whose axis text
        # is least lexically overlapping with the rest. Tie-break by lowest
        # confidence_before (least-confident candidate often == most-distant
        # from majority literature, since the model's confidence is itself a
        # proxy for training-data density on that hypothesis). When variants
        # don't expose divergence_axis cleanly, fall back to lowest-confidence
        # candidate.
        winner = self._select_hypothesis_variant(candidates)
        # Preserve the unselected variants for audit on the JournalEntry.
        winner["considered_variants"] = [
            {
                "hypothesis": c.get("hypothesis", ""),
                "confidence_before": c.get("confidence_before", 0.5),
                "divergence_axis": c.get("divergence_axis", ""),
            }
            for c in candidates if c is not winner
        ]
        axis = (winner.get("divergence_axis") or "").strip()
        print(
            f"  [variants] {len(candidates)} candidate hypotheses generated; "
            f"selected (axis={axis[:80] if axis else '?'!r}, conf={winner.get('confidence_before', 0.5):.2f})"
        )
        return winner

    @staticmethod
    def _select_hypothesis_variant(candidates: list[dict]) -> dict:
        """Phase 9: pick the candidate whose divergence_axis is least
        overlapping with the others (proxy: lowest token-overlap to the
        union of other candidates' axis text). Tie-break by lowest
        confidence_before (least training-data-supported = most likely
        to surface a productive surprise)."""
        if len(candidates) == 1:
            return candidates[0]

        def _tokens(s: str) -> set[str]:
            return {t for t in (s or "").lower().replace(",", " ").split() if len(t) > 2}

        scored: list[tuple[float, float, dict]] = []
        for c in candidates:
            mine = _tokens(c.get("divergence_axis") or "")
            others_union: set[str] = set()
            for other in candidates:
                if other is c:
                    continue
                others_union |= _tokens(other.get("divergence_axis") or "")
            if mine:
                # Distinctness = 1 - jaccard overlap
                inter = len(mine & others_union)
                union = max(1, len(mine | others_union))
                distinctness = 1.0 - (inter / union)
            else:
                # No axis named; treat as moderately distinct (don't penalise
                # too hard — sometimes the LLM produces good hypotheses
                # without explicit axis tagging).
                distinctness = 0.4
            conf = float(c.get("confidence_before") or 0.5)
            # Sort: highest distinctness first, then lowest confidence as tiebreak.
            scored.append((distinctness, -conf, c))

        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return scored[0][2]

    def _run_investigation(self, question: ResearchQuestion) -> dict:
        prompt = INVESTIGATE_PROMPT.format(
            domain=self.config.domain,
            focus_block=self._focus_block(),
            question=question.question,
            investigability_notes=question.investigability_notes,
            tool_list=self._tool_list_block(for_client=self.primary),
        )
        server_tools = [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "code_execution_20250825", "name": "code_execution"},
        ]
        return self._call_primary_with_tools(
            prompt,
            server_tools=server_tools,
            max_tokens=self.connection.primary.investigation_max_tokens,
        )

    def _assess_surprise(
        self,
        question: ResearchQuestion,
        hypothesis: dict,
        findings: dict,
    ) -> dict:
        prompt = SURPRISE_PROMPT.format(
            question=question.question,
            hypothesis_json=json.dumps(hypothesis, indent=2),
            findings_json=json.dumps(findings, indent=2),
        )
        # Phase 5: route to investigation_assessor (defaults to primary).
        # Setting investigation_assessor_role to a different model from the
        # primary creates representational separation between exploration
        # (stages 1+2) and evaluation (this stage), resisting the
        # self-grading collapse that prompted Phase 5.
        return self._call_investigation_assessor(prompt)

    def investigate(self, question: ResearchQuestion) -> JournalEntry:
        """Investigate a research question in three stages: hypothesis, investigation, surprise."""
        print("\n--- INVESTIGATING ---")
        print(f"  Question: {question.question}")

        print("  [1/3] Forming hypothesis...")
        hypothesis = self._form_hypothesis(question)
        print(f"        Hypothesis: {hypothesis.get('hypothesis', '')[:100]}...")

        print("  [2/3] Running investigation (web_search enabled)...")
        findings = self._run_investigation(question)
        print(f"        Sources found: {len(findings.get('sources', []))}")

        print("  [3/3] Assessing surprise against committed hypothesis...")
        surprise = self._assess_surprise(question, hypothesis, findings)
        _calibrate_confidence_after(surprise, float(hypothesis.get("confidence_before", 0.5)))

        entry = JournalEntry(
            id=f"j-{uuid4().hex[:8]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            question_id=question.id,
            question=question.question,
            hypothesis=hypothesis.get("hypothesis", ""),
            confidence_before=hypothesis.get("confidence_before", 0.5),
            methodology=findings.get("methodology", ""),
            raw_findings=findings.get("raw_findings", ""),
            sources=findings.get("sources", []),
            surprise_delta=surprise.get("surprise_delta", 0.0),
            confidence_after=surprise.get("confidence_after", 0.5),
            key_takeaways=findings.get("key_takeaways", []),
            new_questions=surprise.get("new_questions", []),
            domain_tags=question.domain_tags,
            hypothesis_verdict=surprise.get("hypothesis_verdict", ""),
            surprise_explanation=surprise.get("surprise_explanation", ""),
        )

        print(f"  Surprise delta: {entry.surprise_delta:.2f}")
        print(f"  Verdict: {surprise.get('hypothesis_verdict', 'unknown')}")
        print(f"  Confidence: {entry.confidence_before:.2f} -> {entry.confidence_after:.2f}")
        for t in entry.key_takeaways:
            print(f"  Takeaway: {t[:80]}...")

        self.journal.add_entry(entry)
        # Priority: surprise is the best signal we have for "this question will reveal more"
        # — surprising entries suggest adjacent unknowns. Nudge above baseline so surprising
        # follow-ups beat stale low-signal leftovers.
        followup_priority = max(0.3, min(1.0, 0.4 + 0.6 * entry.surprise_delta))
        self._enqueue_questions(
            entry.new_questions,
            source=f"entry:{entry.id}",
            priority=followup_priority,
        )
        # Cross-domain analog probe: on high-surprise entries, ask the engine which
        # DISTANT fields have structural analogs and enqueue those reframed questions.
        if getattr(self.config, "analog_probe_enabled", True):
            self._run_analog_probe(entry)
        # Assumption probe: complementary move. Fires on LOW-surprise CONFIRMED
        # findings — the accepted-wisdom regime where load-bearing assumptions hide.
        if getattr(self.config, "assumption_probe_enabled", True):
            self._run_assumption_probe(entry)
        # Best-effort embed the new entry so semantic features stay current. If the
        # embedding client is unavailable or fails, we just skip — not a cycle-breaking
        # concern.
        if getattr(self, "embedding_client", None) is not None:
            try:
                from engine.embeddings import embed_missing_entries
                n = embed_missing_entries(self.journal, self.embedding_client)
                if n:
                    print(f"  [embed] {n} new entry embedding(s) computed.")
            except Exception as e:  # noqa: BLE001
                print(f"  [embed warn] skipped: {type(e).__name__}: {e}")
        return entry

    def _run_analog_probe(self, entry: JournalEntry) -> int:
        """Ask the primary model which DISTANT domains have structural analogs of
        this finding and enqueue those as high-priority investigable questions.

        Fires only when the entry's surprise_delta crosses the configured
        threshold — a bigger surprise is a stronger signal that an unfamiliar
        domain may have been grazed.
        """
        threshold = float(getattr(self.config, "analog_probe_surprise_threshold", 0.5))
        if entry.surprise_delta < threshold:
            return 0

        recent_tags = self.journal.get_all_domain_tags()
        prompt = ANALOG_PROBE_PROMPT.format(
            entry_question=entry.question,
            entry_surprise=entry.surprise_delta,
            entry_takeaways=json.dumps(entry.key_takeaways, indent=2),
            recent_tags=", ".join(recent_tags) if recent_tags else "(none)",
            engine_domain=getattr(self.config, "domain", "") or "(unspecified)",
        )
        try:
            result = self._call_primary(prompt)
        except Exception as e:  # noqa: BLE001
            print(f"  [analog probe] skipped: {type(e).__name__}: {e}")
            return 0

        analogs = result.get("analogs", []) or []
        if not analogs:
            print("  [analog probe] no strong cross-domain analogs proposed.")
            return 0

        cap = max(1, int(getattr(self.config, "analog_probe_max_analogs", 3)))
        top_analogs = analogs[:cap]
        questions: list[str] = []
        for a in top_analogs:
            q = (a.get("question") or "").strip()
            domain = (a.get("domain") or "").strip()
            if q:
                prefix = f"[analog:{domain}] " if domain else ""
                questions.append(prefix + q)

        if questions:
            self._enqueue_questions(
                questions,
                source=f"analog:{entry.id}",
                priority=0.85,
            )
            print(f"  [analog probe] enqueued {len(questions)} cross-domain question(s) @ pri 0.85:")
            for a in top_analogs:
                print(f"    · {a.get('domain','?')} → {a.get('mechanism','?')[:60]}")
        return len(questions)

    def _run_assumption_probe(self, entry: JournalEntry) -> int:
        """Ask the primary model to name implicit assumptions that a CONFIRMED,
        low-surprise finding depends on, and produce investigable questions that
        would test whether each assumption actually holds.

        The complement to the analog probe: analog reaches outward (distant
        fields with structurally similar mechanisms); assumption reaches inward
        (within-domain premise layer). Firing condition is deliberately inverse
        of analog probe — high field consensus (low surprise + confirmed) is
        precisely where load-bearing assumptions hide unexamined.
        """
        threshold = float(getattr(self.config, "assumption_probe_surprise_threshold", 0.3))
        if entry.surprise_delta > threshold:
            return 0
        if (entry.hypothesis_verdict or "").strip().lower() != "confirmed":
            return 0

        prompt = ASSUMPTION_PROBE_PROMPT.format(
            entry_question=entry.question,
            entry_verdict=entry.hypothesis_verdict or "confirmed",
            entry_surprise=entry.surprise_delta,
            entry_takeaways=json.dumps(entry.key_takeaways, indent=2),
        )
        try:
            result = self._call_primary(prompt)
        except Exception as e:  # noqa: BLE001
            print(f"  [assumption probe] skipped: {type(e).__name__}: {e}")
            return 0

        assumptions = result.get("assumptions", []) or []
        if not assumptions:
            print("  [assumption probe] no substantive field-consensus assumptions proposed.")
            return 0

        cap = max(1, int(getattr(self.config, "assumption_probe_max_assumptions", 3)))
        top_assumptions = assumptions[:cap]
        questions: list[str] = []
        for a in top_assumptions:
            q = (a.get("question") or "").strip()
            premise = (a.get("assumption") or "").strip()
            if q:
                prefix = f"[assumption: {premise[:60]}] " if premise else ""
                questions.append(prefix + q)

        if questions:
            self._enqueue_questions(
                questions,
                source=f"assumption:{entry.id}",
                priority=0.80,
            )
            print(f"  [assumption probe] enqueued {len(questions)} assumption-negation question(s) @ pri 0.80:")
            for a in top_assumptions:
                print(f"    · {a.get('assumption','?')[:75]}")
        return len(questions)
