"""citation_manager — maintain a local JSON bibliography of cited works.

Supports:
- `add`: append a citation record (title, authors, year, venue, doi, url, etc.).
- `list`: return current citations.
- `format`: render citations in BibTeX or APA.
- `dedupe`: collapse near-duplicate entries by DOI / arxiv id / title similarity.

Citations live in a single JSON file (path is an arg). Zero external deps.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from engine.tools.base import Tool, ToolError

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class CitationRecord:
    title: str
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    doi: str = ""
    arxiv: str = ""
    url: str = ""
    abstract: str = ""
    note: str = ""


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text() or "[]")
    except json.JSONDecodeError as e:
        raise ToolError(f"cannot read bibliography at {path}: {e}") from e
    if not isinstance(data, list):
        raise ToolError(f"bibliography must be a JSON array; found {type(data).__name__}")
    return data


def _save(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False))


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _bibtex_key(record: dict) -> str:
    first_author = ""
    authors = record.get("authors") or []
    if authors:
        last = str(authors[0]).split()[-1] if authors[0] else ""
        first_author = re.sub(r"[^A-Za-z]", "", last).lower()
    year = record.get("year") or "nd"
    title = record.get("title") or ""
    slug = re.sub(r"[^a-z0-9]+", "", title.lower())[:12]
    parts = [p for p in [first_author, str(year), slug] if p]
    return "_".join(parts) or "citation"


def _format_bibtex(records: list[dict]) -> str:
    lines: list[str] = []
    for r in records:
        key = _bibtex_key(r)
        fields = []
        fields.append(f"  title = {{{r.get('title', '')}}}")
        authors = r.get("authors") or []
        if authors:
            fields.append(f"  author = {{{' and '.join(authors)}}}")
        if r.get("year"):
            fields.append(f"  year = {{{r['year']}}}")
        if r.get("venue"):
            fields.append(f"  journal = {{{r['venue']}}}")
        if r.get("doi"):
            fields.append(f"  doi = {{{r['doi']}}}")
        if r.get("arxiv"):
            fields.append(f"  eprint = {{{r['arxiv']}}}")
            fields.append("  archivePrefix = {arXiv}")
        if r.get("url"):
            fields.append(f"  url = {{{r['url']}}}")
        entry_type = "article" if r.get("venue") or r.get("doi") else "misc"
        lines.append(f"@{entry_type}{{{key},\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(lines)


def _format_apa(records: list[dict]) -> str:
    out: list[str] = []
    for r in records:
        authors = r.get("authors") or []
        if len(authors) == 1:
            author_str = authors[0]
        elif len(authors) == 2:
            author_str = f"{authors[0]} & {authors[1]}"
        elif len(authors) > 2:
            author_str = f"{authors[0]} et al."
        else:
            author_str = ""
        year = f"({r.get('year')})" if r.get("year") else ""
        title = r.get("title", "").strip()
        venue = r.get("venue", "").strip()
        doi = r.get("doi", "")
        url = r.get("url", "")
        parts = [p for p in [author_str, year, f"{title}.", venue, f"doi:{doi}" if doi else "", url] if p]
        out.append(" ".join(parts).strip())
    return "\n".join(out)


def _similar(a: dict, b: dict) -> bool:
    """Two records are 'the same' if DOI/arxiv matches, or title Jaccard >= 0.8."""
    for key in ("doi", "arxiv"):
        if a.get(key) and b.get(key) and a[key].lower() == b[key].lower():
            return True
    title_a = (a.get("title") or "").strip().lower()
    title_b = (b.get("title") or "").strip().lower()
    if title_a and title_b and _jaccard(title_a, title_b) >= 0.8:
        return True
    return False


def _dedupe(records: list[dict]) -> tuple[list[dict], int]:
    kept: list[dict] = []
    removed = 0
    for r in records:
        if any(_similar(r, k) for k in kept):
            removed += 1
            continue
        kept.append(r)
    return kept, removed


class CitationManagerTool(Tool):
    name = "citation_manager"
    description = (
        "Manage a local JSON bibliography: add/list citations, format as BibTeX or APA, "
        "and dedupe near-duplicates. Operations: add, list, format, dedupe. A `path` "
        "argument points at the JSON file (created on first add)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "list", "format", "dedupe"],
            },
            "path": {
                "type": "string",
                "description": "Path to the bibliography JSON file.",
            },
            "citation": {
                "type": "object",
                "description": "Citation fields for 'add': title (required), authors, year, venue, doi, arxiv, url, abstract, note.",
            },
            "style": {
                "type": "string",
                "enum": ["bibtex", "apa"],
                "description": "Format for 'format' operation (default bibtex).",
            },
        },
        "required": ["operation", "path"],
    }
    timeout_seconds = 10.0

    def execute(self, args: dict) -> str:
        operation = (args.get("operation") or "").strip()
        path = Path(args.get("path") or "").expanduser()
        if not path:
            raise ToolError("path is required")

        if operation == "add":
            cite = args.get("citation") or {}
            title = (cite.get("title") or "").strip()
            if not title:
                raise ToolError("citation.title is required for add")
            record = asdict(CitationRecord(
                title=title,
                authors=list(cite.get("authors") or []),
                year=cite.get("year"),
                venue=str(cite.get("venue") or ""),
                doi=str(cite.get("doi") or ""),
                arxiv=str(cite.get("arxiv") or ""),
                url=str(cite.get("url") or ""),
                abstract=str(cite.get("abstract") or ""),
                note=str(cite.get("note") or ""),
            ))
            records = _load(path)
            records.append(record)
            _save(path, records)
            return f"added citation ({len(records)} total) to {path}"

        if operation == "list":
            records = _load(path)
            return f"# {len(records)} citation(s) in {path}\n\n" + json.dumps(records, indent=2, ensure_ascii=False)

        if operation == "format":
            records = _load(path)
            style = (args.get("style") or "bibtex").strip().lower()
            if style == "apa":
                return _format_apa(records)
            if style == "bibtex":
                return _format_bibtex(records)
            raise ToolError(f"unknown style: {style}")

        if operation == "dedupe":
            records = _load(path)
            kept, removed = _dedupe(records)
            _save(path, kept)
            return f"deduped: removed {removed}, kept {len(kept)} in {path}"

        raise ToolError(f"unknown operation: {operation}")
