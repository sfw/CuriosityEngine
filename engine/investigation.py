"""Investigation phase (Phase 3): three-stage hypothesis → web_search → surprise."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from models import JournalEntry, ResearchQuestion
from prompts import HYPOTHESIS_PROMPT, INVESTIGATE_PROMPT, SURPRISE_PROMPT


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
        )

        print(f"  Surprise delta: {entry.surprise_delta:.2f}")
        print(f"  Verdict: {surprise.get('hypothesis_verdict', 'unknown')}")
        print(f"  Confidence: {entry.confidence_before:.2f} -> {entry.confidence_after:.2f}")
        for t in entry.key_takeaways:
            print(f"  Takeaway: {t[:80]}...")

        self.journal.add_entry(entry)
        self._enqueue_questions(entry.new_questions, source=f"entry:{entry.id}")
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
