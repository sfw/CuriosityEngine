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
        prompt = HYPOTHESIS_PROMPT.format(
            domain=self.config.domain,
            question=question.question,
        )
        return self._call_primary(prompt)

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
        return self._call_primary(prompt)

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

        questions: list[str] = []
        for a in analogs[:3]:
            q = (a.get("question") or "").strip()
            domain = (a.get("domain") or "").strip()
            if q:
                prefix = f"[analog:{domain}] " if domain else ""
                questions.append(prefix + q)

        if questions:
            self.journal.enqueue_questions(
                questions,
                source=f"analog:{entry.id}",
                priority=0.85,
            )
            print(f"  [analog probe] enqueued {len(questions)} cross-domain question(s) @ pri 0.85:")
            for a in analogs[:3]:
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

        questions: list[str] = []
        for a in assumptions[:3]:
            q = (a.get("question") or "").strip()
            premise = (a.get("assumption") or "").strip()
            if q:
                prefix = f"[assumption: {premise[:60]}] " if premise else ""
                questions.append(prefix + q)

        if questions:
            self.journal.enqueue_questions(
                questions,
                source=f"assumption:{entry.id}",
                priority=0.80,
            )
            print(f"  [assumption probe] enqueued {len(questions)} assumption-negation question(s) @ pri 0.80:")
            for a in assumptions[:3]:
                print(f"    · {a.get('assumption','?')[:75]}")
        return len(questions)
