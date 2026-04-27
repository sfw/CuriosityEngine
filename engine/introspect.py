"""Introspection + question generation phase (Phases 1 & 2)."""

from __future__ import annotations

import json
from dataclasses import asdict
from uuid import uuid4

from models import ResearchQuestion, UncertaintyItem
from prompts import INTROSPECT_PROMPT, QUESTION_PROMPT


# Phase 7: persona-conditioned introspection. Each persona is a lens
# applied to the standard INTROSPECT_PROMPT — the framing is prepended
# to the prompt, the rest of the prompt is unchanged. Personas were
# chosen to surface different blind spots:
#   skeptic       — what is the field most likely WRONG about
#   outsider      — what would a researcher from an adjacent field find puzzling
#   contrarian    — what's the strongest argument against the consensus
#   historian     — what did the field believe 10 years ago that turned out wrong
#   practitioner  — what does the field assume can be measured/built that actually can't
# Default 3 personas (skeptic + outsider + contrarian); tunable to 5.
# Borrows from Stanford STORM's multi-perspective question generation.
_INTROSPECTION_PERSONAS = {
    "skeptic": (
        "\n\nPERSONA: You are the SKEPTIC lens. Your specific job is to identify "
        "what this field is most likely WRONG about — load-bearing assumptions "
        "that are widely held but poorly supported. Do not generate generic "
        "uncertainties. Generate skeptical attacks on specific consensus claims. "
        "Resist the urge to soften: your value is in surfacing the discomfort "
        "the other personas would smooth over."
    ),
    "outsider": (
        "\n\nPERSONA: You are the OUTSIDER lens. You are a researcher from an "
        "adjacent field who reads this domain's literature and finds specific "
        "things puzzling — claims that wouldn't survive your field's standards "
        "of evidence, methodologies that look ad hoc by your discipline's "
        "standards, conventions that this field has stopped questioning but "
        "you would not accept. Generate uncertainties grounded in the specific "
        "puzzlement of an adjacent-field perspective."
    ),
    "contrarian": (
        "\n\nPERSONA: You are the CONTRARIAN lens. For each major piece of "
        "consensus in this field, your job is to articulate the strongest "
        "argument AGAINST it. Generate uncertainties as crisp counter-claims. "
        "If you find yourself producing 'maybe X is overstated', sharpen to "
        "'X is wrong because Y'. The output is most useful when it would "
        "make a defender of the consensus uncomfortable."
    ),
    "historian": (
        "\n\nPERSONA: You are the HISTORIAN lens. Your job is pattern recognition "
        "across the field's history of mistaken consensus. What did this field "
        "believe 10 years ago that turned out wrong? What did it abandon, and "
        "what did it claim was settled that wasn't? Generate uncertainties of "
        "the form: 'the current claim X has the same shape as the older "
        "abandoned claim Y' — concrete shape-matches, not generic 'history "
        "shows we're often wrong' platitudes."
    ),
    "practitioner": (
        "\n\nPERSONA: You are the PRACTITIONER lens. You are someone who would "
        "actually have to BUILD or MEASURE the things this field claims. Your "
        "job is to identify uncertainties about what is actually buildable / "
        "measurable / reproducible. Where does the field assume an instrument "
        "exists or a measurement is reliable when it isn't? Where does theory "
        "presume operationalisations that fail on contact with implementation? "
        "Be specific about the gap between claim and constructible artifact."
    ),
}

# Default persona ordering. The first N personas are used when
# introspection_persona_count = N (default 3).
_INTROSPECTION_PERSONA_ORDER = [
    "skeptic", "outsider", "contrarian", "historian", "practitioner",
]


class IntrospectionMixin:
    """Phases 1 & 2: identify uncertainties, turn them into ranked research questions."""

    def _enqueue_questions(self, questions: list[str], source: str, priority: float = 0.5):
        # Pull the priority floor from config so every enqueue path gets the
        # same autoscreen applied. Human-sourced questions bypass — the journal
        # method handles that exemption.
        floor = float(getattr(self.config, "question_priority_floor", 0.0)) or None
        self.journal.enqueue_questions(
            questions, source=source, priority=priority, floor=floor,
        )

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

    def _run_introspection_for_persona(
        self, persona: str, persona_framing: str, n_items: int,
    ) -> list[UncertaintyItem]:
        """Run a single introspection call with a given persona framing.
        Returns uncertainties tagged with the persona name."""
        prompt = INTROSPECT_PROMPT.format(
            domain=self.config.domain,
            focus_block=self._focus_block(),
            journal_context=self._build_journal_context(),
            n_items=n_items,
            persona_framing=persona_framing,
        )
        try:
            result = self._call_primary(prompt)
        except Exception as e:  # noqa: BLE001 — best-effort per persona
            print(f"  [persona={persona}] introspection failed: {type(e).__name__}: {e}")
            return []
        items: list[UncertaintyItem] = []
        for u in result.get("uncertainties", []) or []:
            try:
                item = UncertaintyItem(
                    id=f"u-{uuid4().hex[:8]}",
                    description=u["description"],
                    uncertainty_type=u["uncertainty_type"],
                    domain_tags=u.get("domain_tags", []),
                    estimated_importance=u.get("estimated_importance", 0.5),
                    persona=persona,
                )
                items.append(item)
            except KeyError:
                continue
        return items

    def introspect(self) -> list[UncertaintyItem]:
        """Phase 7: persona-conditioned introspection.

        When `introspection_persona_count >= 2`, runs N parallel
        introspection calls each through a different persona lens
        (skeptic, outsider, contrarian, historian, practitioner) and
        merges the resulting uncertainties — each tagged with the persona
        that surfaced it. When count == 1, single-voice introspection
        (pre-Phase-7 behavior).
        """
        print("\n--- INTROSPECTING ---")

        persona_count = max(1, min(
            len(_INTROSPECTION_PERSONA_ORDER),
            int(getattr(self.connection.engine, "introspection_persona_count", 3)),
        ))

        if persona_count <= 1:
            # Single-voice path: empty persona framing, no persona attribution.
            items = self._run_introspection_for_persona(
                persona="",
                persona_framing="",
                n_items=self.config.questions_per_cycle + 2,
            )
            for item in items:
                print(f"  [{item.uncertainty_type}] {item.description[:80]}...")
            return items

        # Multi-persona path: fan out across personas, merge.
        # Each persona produces fewer items per call than single-voice would,
        # since merging gives us breadth across personas. Cap each persona's
        # call to keep total count comparable to single-voice.
        per_persona_items = max(2, (self.config.questions_per_cycle + 2) // persona_count + 1)
        personas_to_run = _INTROSPECTION_PERSONA_ORDER[:persona_count]
        print(f"  [personas: {', '.join(personas_to_run)} ({persona_count} lenses, ~{per_persona_items} items each)]")

        all_items: list[UncertaintyItem] = []
        for persona in personas_to_run:
            framing = _INTROSPECTION_PERSONAS.get(persona, "")
            items = self._run_introspection_for_persona(
                persona=persona,
                persona_framing=framing,
                n_items=per_persona_items,
            )
            for item in items:
                print(f"    [{persona}/{item.uncertainty_type}] {item.description[:75]}...")
            all_items.extend(items)

        return all_items

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
            source_indices = [
                i for i in q.get("source_uncertainty_indices", [0])
                if i < len(uncertainties)
            ]
            # Phase 7: surface which personas contributed to this question.
            # Look up the persona on each source uncertainty; emit the unique
            # set so the log shows e.g. "personas=skeptic,contrarian".
            source_personas = sorted({
                uncertainties[i].persona for i in source_indices
                if uncertainties[i].persona
            })
            question = ResearchQuestion(
                id=f"q-{uuid4().hex[:8]}",
                question=q["question"],
                source_uncertainties=[uncertainties[i].id for i in source_indices],
                priority_score=q.get("priority_score", 0.5),
                domain_tags=q.get("domain_tags", []),
                investigability_notes=q.get("investigability_notes", ""),
            )
            questions.append(question)
            persona_tag = (
                f" personas={','.join(source_personas)}"
                if source_personas else ""
            )
            print(f"  [priority={question.priority_score:.2f}{persona_tag}] {question.question[:80]}...")

        questions.sort(key=lambda q: q.priority_score, reverse=True)
        return questions
