"""peer_review — deterministic rubric-based review of a research draft.

Scores content against a rubric and returns strengths, issues, and revision
actions. No LLM calls; pure heuristics on sentence structure, citation density,
hedging, and rubric-specific markers.

Lean port of loom's peer_review_simulator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from engine.tools.base import Tool, ToolError

_SENTENCE_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_CITATION_RE = re.compile(r"(\[[^\]]+\]|\([A-Za-z][^\)]*\d{4}[^\)]*\)|https?://\S+|arXiv:\s*\d)")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_HEDGE_RE = re.compile(
    r"\b(suggest|may|might|could|appears|seems|possibly|likely|"
    r"potentially|approximately|roughly)\b",
    re.IGNORECASE,
)
_COUNTER_RE = re.compile(
    r"\b(however|although|on the other hand|counter\s*arg|critics?|"
    r"objection|limitation|caveat|despite|nevertheless)\b",
    re.IGNORECASE,
)
_LIMIT_RE = re.compile(
    r"\b(limitation|caveat|does not generalize|threat to validity|"
    r"assume|assumption|outside the scope|future work)\b",
    re.IGNORECASE,
)
_METHOD_RE = re.compile(
    r"\b(method|approach|protocol|procedure|dataset|benchmark|metric|"
    r"hyperparameter|ablation|reproducib)\b",
    re.IGNORECASE,
)


@dataclass
class Score:
    name: str
    value: float
    rationale: str


_RUBRICS: dict[str, list[tuple[str, float]]] = {
    "general": [
        ("clarity", 0.25),
        ("structure", 0.20),
        ("evidence", 0.25),
        ("limitations", 0.15),
        ("actionability", 0.15),
    ],
    "methodology": [
        ("method_definition", 0.30),
        ("reproducibility", 0.25),
        ("evidence", 0.20),
        ("limitations", 0.15),
        ("clarity", 0.10),
    ],
    "argument_quality": [
        ("thesis", 0.30),
        ("evidence", 0.25),
        ("counterarguments", 0.20),
        ("coherence", 0.15),
        ("clarity", 0.10),
    ],
    "citation_quality": [
        ("citation_density", 0.30),
        ("traceability", 0.25),
        ("source_recency", 0.20),
        ("consistency", 0.15),
        ("distinct_sources", 0.10),
    ],
}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _score_clarity(text: str, sentences: list[str]) -> Score:
    if not sentences:
        return Score("clarity", 0.0, "no sentences detected")
    word_counts = [len(_WORD_RE.findall(s)) for s in sentences]
    avg = sum(word_counts) / len(word_counts)
    # ideal ~15-22 words/sentence; penalize very long or very short averages
    if 12 <= avg <= 24:
        value = 0.9
    elif 8 <= avg <= 30:
        value = 0.7
    else:
        value = 0.4
    return Score("clarity", value, f"avg sentence length {avg:.1f} words")


def _score_structure(text: str) -> Score:
    headings = len(_HEADING_RE.findall(text))
    words = len(_WORD_RE.findall(text))
    if words == 0:
        return Score("structure", 0.0, "empty document")
    density = headings / max(1, words / 250)
    if density >= 1.0:
        return Score("structure", 0.9, f"{headings} heading(s) for ~{words // 250} blocks")
    if density >= 0.4:
        return Score("structure", 0.7, f"{headings} heading(s) — adequate")
    return Score("structure", 0.4, f"only {headings} heading(s) for {words} words")


def _score_evidence(text: str, sentences: list[str]) -> Score:
    if not sentences:
        return Score("evidence", 0.0, "no content")
    with_cite = sum(1 for s in sentences if _CITATION_RE.search(s))
    ratio = with_cite / len(sentences)
    if ratio >= 0.4:
        value = 0.9
    elif ratio >= 0.15:
        value = 0.7
    elif ratio > 0:
        value = 0.5
    else:
        value = 0.2
    return Score("evidence", value, f"{with_cite}/{len(sentences)} sentences carry citations/urls")


def _score_limitations(text: str) -> Score:
    hits = len(_LIMIT_RE.findall(text))
    if hits >= 3:
        return Score("limitations", 0.9, f"{hits} limitations markers")
    if hits >= 1:
        return Score("limitations", 0.6, f"{hits} limitations markers")
    return Score("limitations", 0.2, "no limitations markers detected")


def _score_counterarguments(text: str) -> Score:
    hits = len(_COUNTER_RE.findall(text))
    if hits >= 3:
        return Score("counterarguments", 0.9, f"{hits} counter-argument markers")
    if hits >= 1:
        return Score("counterarguments", 0.6, f"{hits} counter-argument markers")
    return Score("counterarguments", 0.3, "no counter-argument markers detected")


def _score_thesis(text: str) -> Score:
    # First non-heading sentence should look like a thesis statement
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        sentences = _sentences(stripped)
        if sentences:
            first = sentences[0]
            length = len(_WORD_RE.findall(first))
            if 10 <= length <= 40:
                return Score("thesis", 0.85, f"first sentence ({length} words) reads like a thesis")
            return Score("thesis", 0.5, f"first sentence is {length} words — unusual length for a thesis")
    return Score("thesis", 0.2, "no identifiable opening sentence")


def _score_method_definition(text: str) -> Score:
    hits = len(_METHOD_RE.findall(text))
    words = len(_WORD_RE.findall(text)) or 1
    density = hits / max(1, words / 500)
    if density >= 1.0:
        return Score("method_definition", 0.9, f"{hits} method markers across {words} words")
    if density >= 0.3:
        return Score("method_definition", 0.6, f"{hits} method markers")
    return Score("method_definition", 0.3, f"only {hits} method markers")


def _score_reproducibility(text: str) -> Score:
    tokens = {
        "code", "repository", "github", "gitlab", "zenodo", "doi",
        "random seed", "dataset", "benchmark", "version", "hash",
    }
    hits = sum(1 for t in tokens if t in text.lower())
    if hits >= 4:
        return Score("reproducibility", 0.9, f"{hits} reproducibility signals")
    if hits >= 2:
        return Score("reproducibility", 0.6, f"{hits} reproducibility signals")
    return Score("reproducibility", 0.3, "few reproducibility signals")


def _score_actionability(text: str, sentences: list[str]) -> Score:
    if not sentences:
        return Score("actionability", 0.0, "empty")
    imperative_starts = ("use ", "adopt ", "test ", "run ", "apply ", "measure ", "consider ", "investigate ")
    hits = sum(1 for s in sentences if s.lower().startswith(imperative_starts))
    ratio = hits / len(sentences)
    if ratio >= 0.15:
        return Score("actionability", 0.8, f"{hits} actionable sentences")
    if ratio > 0:
        return Score("actionability", 0.5, f"{hits} actionable sentences")
    return Score("actionability", 0.2, "no clearly actionable sentences")


def _score_coherence(text: str, sentences: list[str]) -> Score:
    if len(sentences) < 3:
        return Score("coherence", 0.5, "too few sentences to assess")
    # crude: count transition words
    transitions = ("therefore", "thus", "however", "moreover", "consequently", "in addition", "similarly")
    hits = sum(1 for s in sentences if any(t in s.lower() for t in transitions))
    ratio = hits / len(sentences)
    if ratio >= 0.15:
        return Score("coherence", 0.85, f"{hits} transition markers")
    if ratio >= 0.05:
        return Score("coherence", 0.65, f"{hits} transition markers")
    return Score("coherence", 0.4, "few transition markers")


def _score_citation_density(text: str, sentences: list[str]) -> Score:
    if not sentences:
        return Score("citation_density", 0.0, "no content")
    hits = sum(1 for s in sentences if _CITATION_RE.search(s))
    ratio = hits / len(sentences)
    return Score("citation_density", min(1.0, ratio * 2.5), f"{ratio:.1%} of sentences cite")


def _score_traceability(text: str) -> Score:
    # URL-style or DOI-style references = traceable
    hits = len(re.findall(r"(https?://\S+|doi:\s*\S+|arXiv:\s*\S+)", text, re.IGNORECASE))
    if hits >= 5:
        return Score("traceability", 0.9, f"{hits} direct links/identifiers")
    if hits >= 2:
        return Score("traceability", 0.6, f"{hits} direct links/identifiers")
    return Score("traceability", 0.3, "few direct links/identifiers")


def _score_source_recency(text: str) -> Score:
    years = [int(y) for y in _YEAR_RE.findall(text)]
    if not years:
        return Score("source_recency", 0.3, "no year citations detected")
    recent = sum(1 for y in years if y >= 2020)
    ratio = recent / len(years)
    if ratio >= 0.5:
        return Score("source_recency", 0.9, f"{recent}/{len(years)} citations ≥ 2020")
    if ratio >= 0.2:
        return Score("source_recency", 0.6, f"{recent}/{len(years)} citations ≥ 2020")
    return Score("source_recency", 0.3, f"only {recent}/{len(years)} recent citations")


def _score_consistency(text: str) -> Score:
    # low variance in citation style = more consistent
    styles = 0
    if re.search(r"\[\d+\]", text):
        styles += 1
    if re.search(r"\([A-Za-z]+\s*,?\s*\d{4}\)", text):
        styles += 1
    if re.search(r"arXiv:\s*\d", text, re.IGNORECASE):
        styles += 1
    if styles <= 1:
        return Score("consistency", 0.9, f"{styles} citation style(s) — consistent")
    if styles == 2:
        return Score("consistency", 0.6, "mixed citation styles")
    return Score("consistency", 0.3, "multiple citation styles — inconsistent")


def _score_distinct_sources(text: str) -> Score:
    urls = set(re.findall(r"https?://\S+", text))
    if len(urls) >= 8:
        return Score("distinct_sources", 0.9, f"{len(urls)} distinct URLs")
    if len(urls) >= 3:
        return Score("distinct_sources", 0.6, f"{len(urls)} distinct URLs")
    return Score("distinct_sources", 0.3, f"only {len(urls)} distinct URLs")


_SCORERS = {
    "clarity": lambda t, s: _score_clarity(t, s),
    "structure": lambda t, s: _score_structure(t),
    "evidence": lambda t, s: _score_evidence(t, s),
    "limitations": lambda t, s: _score_limitations(t),
    "actionability": lambda t, s: _score_actionability(t, s),
    "method_definition": lambda t, s: _score_method_definition(t),
    "reproducibility": lambda t, s: _score_reproducibility(t),
    "thesis": lambda t, s: _score_thesis(t),
    "counterarguments": lambda t, s: _score_counterarguments(t),
    "coherence": lambda t, s: _score_coherence(t, s),
    "citation_density": lambda t, s: _score_citation_density(t, s),
    "traceability": lambda t, s: _score_traceability(t),
    "source_recency": lambda t, s: _score_source_recency(t),
    "consistency": lambda t, s: _score_consistency(t),
    "distinct_sources": lambda t, s: _score_distinct_sources(t),
}


class PeerReviewTool(Tool):
    name = "peer_review"
    description = (
        "Deterministically review a research draft against a rubric. Returns scored "
        "criteria (0.0-1.0), strengths, issues, and revision actions. Rubrics: "
        "general, methodology, argument_quality, citation_quality. Pure heuristics "
        "(no LLM calls) — fast, cheap, consistent."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Inline document text to review (preferred).",
            },
            "path": {
                "type": "string",
                "description": "Path to a text file to review (used if content is omitted).",
            },
            "rubric": {
                "type": "string",
                "enum": ["general", "methodology", "argument_quality", "citation_quality"],
                "description": "Which rubric to score against. Default: general.",
            },
        },
    }
    timeout_seconds = 10.0

    def execute(self, args: dict) -> str:
        content = args.get("content") or ""
        path = args.get("path")
        if not content and path:
            p = Path(path).expanduser()
            if not p.exists():
                raise ToolError(f"path does not exist: {p}")
            content = p.read_text()
        if not content.strip():
            raise ToolError("no content to review (provide content or path)")

        rubric_name = (args.get("rubric") or "general").strip()
        rubric = _RUBRICS.get(rubric_name)
        if rubric is None:
            raise ToolError(f"unknown rubric: {rubric_name}")

        sentences = _sentences(content)
        scores: list[tuple[Score, float]] = []
        for criterion, weight in rubric:
            scorer = _SCORERS.get(criterion)
            if scorer is None:
                continue
            scores.append((scorer(content, sentences), weight))

        weighted = sum(s.value * w for s, w in scores)

        strengths = [f"{s.name} ({s.value:.2f}): {s.rationale}" for s, _ in scores if s.value >= 0.7]
        issues = [f"{s.name} ({s.value:.2f}): {s.rationale}" for s, _ in scores if s.value < 0.5]
        actions = []
        for s, _ in scores:
            if s.value < 0.5:
                actions.append(_suggest_action(s.name))

        lines = [f"# peer_review (rubric={rubric_name})", ""]
        lines.append(f"**Overall score**: {weighted:.2f} / 1.00")
        lines.append("")
        lines.append("## Scores")
        for s, w in scores:
            lines.append(f"  - {s.name} (weight {w:.2f}): {s.value:.2f} — {s.rationale}")
        if strengths:
            lines.append("\n## Strengths")
            for x in strengths:
                lines.append(f"  - {x}")
        if issues:
            lines.append("\n## Issues")
            for x in issues:
                lines.append(f"  - {x}")
        if actions:
            lines.append("\n## Revision actions")
            for x in actions:
                lines.append(f"  - {x}")
        return "\n".join(lines)


def _suggest_action(criterion: str) -> str:
    return {
        "clarity": "Shorten or restructure long sentences; aim for 15-22 words average.",
        "structure": "Add headings / sections to guide the reader.",
        "evidence": "Add citations or URLs to support unsupported claims.",
        "limitations": "Explicitly name assumptions, caveats, and threats to validity.",
        "actionability": "Convert descriptive prose into imperative steps or recommendations.",
        "method_definition": "Specify methods, datasets, metrics, and hyperparameters.",
        "reproducibility": "Cite code/repo, random seeds, dataset versions, and artifact hashes.",
        "thesis": "Start with a clear, committal thesis statement.",
        "counterarguments": "Engage the strongest objections; don't sidestep critics.",
        "coherence": "Use explicit transition words between paragraphs and sections.",
        "citation_density": "Cite more frequently — most substantive claims should reference prior work.",
        "traceability": "Prefer direct URLs / DOIs / arXiv ids over vague references.",
        "source_recency": "Include more post-2020 citations, especially in fast-moving subfields.",
        "consistency": "Pick one citation style and stick to it.",
        "distinct_sources": "Draw on more distinct sources; avoid over-relying on a single paper.",
    }.get(criterion, f"Improve {criterion}.")
