"""Introspection + question generation phase (Phases 1 & 2)."""

from __future__ import annotations

import json
from dataclasses import asdict
from uuid import uuid4

from models import ResearchQuestion, UncertaintyItem
from prompts import INTROSPECT_PROMPT, QUESTION_PROMPT


class IntrospectionMixin:
    """Phases 1 & 2: identify uncertainties, turn them into ranked research questions."""

    def _enqueue_questions(self, questions: list[str], source: str):
        self.journal.enqueue_questions(questions, source=source)

    def _build_journal_context(self) -> str:
        recent = self.journal.get_recent_entries(5)
        queued = self.journal.question_queue

        if not recent and not queued:
            return "No previous research journal entries exist yet. This is the first cycle."

        context = ""
        if recent:
            context += "PREVIOUS RESEARCH (most recent entries):\n"
            for entry in recent:
                context += f"\n- Question: {entry['question']}\n"
                context += f"  Key takeaways: {', '.join(entry.get('key_takeaways', []))}\n"
                context += f"  Surprise delta: {entry.get('surprise_delta', 'N/A')}\n"
                context += f"  New questions raised: {', '.join(entry.get('new_questions', []))}\n"

        if queued:
            context += "\nEMERGENT QUESTIONS from prior investigations (still unanswered):\n"
            for q in queued[:10]:
                context += f"  - {q['question']} (from {q['source']})\n"
            context += "Consider uncertainties that would help resolve these emergent questions.\n"

        tags = self.journal.get_all_domain_tags()
        if tags:
            context += f"\nDomains explored so far: {', '.join(tags)}\n"
            context += "IMPORTANT: Prioritize uncertainties in domains NOT yet explored, or at the intersection of explored domains.\n"

        return context

    def introspect(self) -> list[UncertaintyItem]:
        print("\n--- INTROSPECTING ---")

        prompt = INTROSPECT_PROMPT.format(
            domain=self.config.domain,
            focus_block=self._focus_block(),
            journal_context=self._build_journal_context(),
            n_items=self.config.questions_per_cycle + 2,
        )

        result = self._call_primary(prompt)
        items = []
        for u in result.get("uncertainties", []):
            item = UncertaintyItem(
                id=f"u-{uuid4().hex[:8]}",
                description=u["description"],
                uncertainty_type=u["uncertainty_type"],
                domain_tags=u.get("domain_tags", []),
                estimated_importance=u.get("estimated_importance", 0.5),
            )
            items.append(item)
            print(f"  [{item.uncertainty_type}] {item.description[:80]}...")

        return items

    def generate_questions(self, uncertainties: list[UncertaintyItem]) -> list[ResearchQuestion]:
        print("\n--- GENERATING QUESTIONS ---")

        uncertainties_json = json.dumps([asdict(u) for u in uncertainties], indent=2)
        prompt = QUESTION_PROMPT.format(
            domain=self.config.domain,
            focus_block=self._focus_block(),
            uncertainties_json=uncertainties_json,
            n_questions=self.config.questions_per_cycle,
        )

        result = self._call_primary(prompt)
        questions = []
        for q in result.get("questions", []):
            question = ResearchQuestion(
                id=f"q-{uuid4().hex[:8]}",
                question=q["question"],
                source_uncertainties=[
                    uncertainties[i].id
                    for i in q.get("source_uncertainty_indices", [0])
                    if i < len(uncertainties)
                ],
                priority_score=q.get("priority_score", 0.5),
                domain_tags=q.get("domain_tags", []),
                investigability_notes=q.get("investigability_notes", ""),
            )
            questions.append(question)
            print(f"  [priority={question.priority_score:.2f}] {question.question[:80]}...")

        questions.sort(key=lambda q: q.priority_score, reverse=True)
        return questions
