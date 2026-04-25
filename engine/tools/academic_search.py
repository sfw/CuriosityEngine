"""academic_search — keyless search across Crossref, arXiv, and Semantic Scholar.

Returns unified, citation-ready results. No API keys required (Semantic Scholar's
public tier is rate-limited but unauthenticated; use a key if you need higher quotas).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import httpx

from engine.tools.base import Tool, ToolError
from engine.tools._rate_limits import ARXIV, CROSSREF, SEMANTIC_SCHOLAR

_USER_AGENT = "CuriosityEngine/0.1 (research use; contact via repo)"
_TIMEOUT = 25.0


@dataclass
class AcademicResult:
    source: str                       # "crossref" | "arxiv" | "semantic_scholar"
    title: str
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    abstract: str = ""
    url: str = ""
    doi: str = ""
    identifier: str = ""              # source-specific id (arxiv id, SSOrk paperId, etc.)
    citation_count: Optional[int] = None


def _crossref_search(query: str, limit: int) -> list[AcademicResult]:
    """Crossref REST API. https://api.crossref.org/swagger-ui/"""
    CROSSREF.acquire()
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": min(max(1, limit), 50),
        "select": "DOI,title,author,issued,container-title,abstract,URL",
    }
    with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}) as c:
        r = c.get(url, params=params)
        if r.status_code == 429:
            CROSSREF.note_failure(60.0)
            raise ToolError("crossref rate limited (HTTP 429) — 60s cooldown engaged")
        r.raise_for_status()
        data = r.json()

    out: list[AcademicResult] = []
    for item in (data.get("message", {}).get("items") or [])[:limit]:
        title_list = item.get("title") or []
        title = title_list[0] if title_list else ""
        authors = []
        for a in item.get("author") or []:
            name = " ".join(filter(None, [a.get("given"), a.get("family")])).strip()
            if name:
                authors.append(name)
        year = None
        parts = (item.get("issued", {}).get("date-parts") or [[None]])[0]
        if parts and parts[0]:
            try:
                year = int(parts[0])
            except (TypeError, ValueError):
                pass
        venue_list = item.get("container-title") or []
        venue = venue_list[0] if venue_list else ""
        abstract = (item.get("abstract") or "").strip()
        doi = item.get("DOI", "") or ""
        url_out = item.get("URL", "") or (f"https://doi.org/{doi}" if doi else "")
        out.append(AcademicResult(
            source="crossref",
            title=title.strip(),
            authors=authors,
            year=year,
            venue=venue.strip(),
            abstract=abstract,
            url=url_out,
            doi=doi,
            identifier=doi,
        ))
    return out


def _arxiv_search(query: str, limit: int) -> list[AcademicResult]:
    """arXiv Atom feed API. https://info.arxiv.org/help/api/user-manual.html

    On HTTP 429, set a 60s cooldown on the ARXIV limiter so subsequent
    callers (across the whole process) back off before the next request.
    arXiv's documented 1 req/3s pacing is sometimes stricter in practice
    on burst workloads — gap-scan verification with 80+ probes triggers
    it reliably.
    """
    ARXIV.acquire()  # arXiv user manual: 3s between requests
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": min(max(1, limit), 50),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}) as c:
        r = c.get(url, params=params)
        if r.status_code == 429:
            ARXIV.note_failure(60.0)
            raise ToolError("arxiv rate limited (HTTP 429) — 60s cooldown engaged")
        r.raise_for_status()
        body = r.text

    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    try:
        root = ET.fromstring(body)
    except ET.ParseError as e:
        raise ToolError(f"arxiv parse error: {e}") from e

    out: list[AcademicResult] = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
        title = " ".join(title.split())
        abstract = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")
        abstract = " ".join(abstract.split())
        authors = [
            (a.findtext("atom:name", "", ns) or "").strip()
            for a in entry.findall("atom:author", ns)
        ]
        authors = [a for a in authors if a]
        published = entry.findtext("atom:published", "", ns) or ""
        year = None
        if published[:4].isdigit():
            year = int(published[:4])
        link_url = ""
        for link in entry.findall("atom:link", ns):
            if link.get("rel") == "alternate" and link.get("type", "").startswith("text/html"):
                link_url = link.get("href", "")
                break
        arxiv_id = (entry.findtext("atom:id", "", ns) or "").rsplit("/", 1)[-1]
        out.append(AcademicResult(
            source="arxiv",
            title=title,
            authors=authors,
            year=year,
            venue="arXiv",
            abstract=abstract,
            url=link_url or f"https://arxiv.org/abs/{arxiv_id}",
            doi="",
            identifier=arxiv_id,
        ))
    return out


def _semantic_scholar_search(query: str, limit: int) -> list[AcademicResult]:
    """Semantic Scholar Graph API (public, no key required but rate-limited)."""
    SEMANTIC_SCHOLAR.acquire()  # unauth: 100 req / 5min → paced at 1/3s
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": min(max(1, limit), 50),
        "fields": "title,authors.name,year,venue,abstract,externalIds,url,citationCount",
    }
    with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}) as c:
        r = c.get(url, params=params)
        if r.status_code == 429:
            # Set a 60s cooldown so the rest of the scan backs off cleanly
            # rather than hammering the same endpoint at 1/3s pacing.
            SEMANTIC_SCHOLAR.note_failure(60.0)
            raise ToolError("semantic_scholar rate limited (HTTP 429) — 60s cooldown engaged")
        r.raise_for_status()
        data = r.json()

    out: list[AcademicResult] = []
    for item in (data.get("data") or [])[:limit]:
        authors = [a.get("name", "") for a in (item.get("authors") or []) if a.get("name")]
        external = item.get("externalIds") or {}
        doi = external.get("DOI", "") or ""
        arxiv_id = external.get("ArXiv", "") or ""
        identifier = item.get("paperId", "") or doi or arxiv_id
        out.append(AcademicResult(
            source="semantic_scholar",
            title=(item.get("title") or "").strip(),
            authors=authors,
            year=item.get("year"),
            venue=(item.get("venue") or "").strip(),
            abstract=(item.get("abstract") or "").strip(),
            url=item.get("url") or (f"https://doi.org/{doi}" if doi else ""),
            doi=doi,
            identifier=identifier,
            citation_count=item.get("citationCount"),
        ))
    return out


_SEARCHERS = {
    "crossref": _crossref_search,
    "arxiv": _arxiv_search,
    "semantic_scholar": _semantic_scholar_search,
}


def count_results_structured(
    query: str,
    limit_per_source: int = 5,
    sources: Optional[list[str]] = None,
) -> dict:
    """Structured result count across sources — for callers that need an
    authoritative count (not a string-parse of formatted output) and per-source
    error tracking (e.g. gap verification in engine/negative_space.py).

    Returns:
        {
            "total": int,                      # sum of successful-source counts
            "per_source": {src: int | None},   # None = source errored
            "errors": [str, ...],              # one human-readable entry per failure
            "complete": bool,                  # all requested sources returned ok
        }

    `complete=False` is how callers detect "verification unreliable — don't
    treat the count as authoritative." The raw-text count-substring approach
    this replaces had no such signal; a silently-errored search looked the
    same as a genuinely-empty gap.
    """
    query = (query or "").strip()
    if not query:
        return {"total": 0, "per_source": {}, "errors": ["empty query"], "complete": False}
    srcs = list(sources) if sources else list(_SEARCHERS.keys())
    unknown = [s for s in srcs if s not in _SEARCHERS]
    if unknown:
        return {
            "total": 0,
            "per_source": {},
            "errors": [f"unknown source(s): {unknown}"],
            "complete": False,
        }
    limit = max(1, min(50, int(limit_per_source)))

    per_source: dict[str, Optional[int]] = {}
    errors: list[str] = []
    total = 0
    for src in srcs:
        try:
            results = _SEARCHERS[src](query, limit)
            per_source[src] = len(results)
            total += len(results)
        except ToolError as e:
            per_source[src] = None
            errors.append(f"{src}: {e}")
        except httpx.HTTPError as e:
            per_source[src] = None
            errors.append(f"{src}: http error ({e})")
        except Exception as e:  # noqa: BLE001 — surface any API quirk, don't crash caller
            per_source[src] = None
            errors.append(f"{src}: {type(e).__name__}: {e}")
    return {
        "total": total,
        "per_source": per_source,
        "errors": errors,
        "complete": not errors,
    }


def _format_results(results: list[AcademicResult], max_chars: int = 20_000) -> str:
    lines: list[str] = []
    for r in results:
        authors = ", ".join(r.authors[:8]) + (" et al." if len(r.authors) > 8 else "")
        header = f"• [{r.source}] {r.title}"
        if r.year:
            header += f" ({r.year})"
        lines.append(header)
        if authors:
            lines.append(f"    authors: {authors}")
        if r.venue:
            lines.append(f"    venue: {r.venue}")
        if r.citation_count is not None:
            lines.append(f"    citations: {r.citation_count}")
        if r.doi:
            lines.append(f"    doi: {r.doi}")
        if r.identifier and r.identifier != r.doi:
            lines.append(f"    id: {r.identifier}")
        if r.url:
            lines.append(f"    url: {r.url}")
        if r.abstract:
            abstract = r.abstract if len(r.abstract) < 600 else r.abstract[:600] + "…"
            lines.append(f"    abstract: {abstract}")
        lines.append("")
    joined = "\n".join(lines).strip()
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "\n\n[results truncated]"
    return joined or "[no results]"


class AcademicSearchTool(Tool):
    name = "academic_search"
    description = (
        "Search academic literature across Crossref, arXiv, and Semantic Scholar "
        "(keyless). Returns unified results with title, authors, year, venue, abstract, "
        "DOI, url, and citation count where available. Use for discovering primary "
        "sources, checking prior art, and building citation lists. Specify `sources` "
        "to restrict to one or more providers."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (natural language or boolean).",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string", "enum": ["crossref", "arxiv", "semantic_scholar"]},
                "description": "Subset of sources to query. Default: all three.",
            },
            "limit_per_source": {
                "type": "integer",
                "description": "Max results per source (default 10, max 50).",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": ["query"],
    }
    timeout_seconds = 60.0

    def execute(self, args: dict) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            raise ToolError("no query provided")
        sources = args.get("sources") or list(_SEARCHERS.keys())
        if isinstance(sources, str):
            sources = [sources]
        unknown = [s for s in sources if s not in _SEARCHERS]
        if unknown:
            raise ToolError(f"unknown source(s): {unknown}. Valid: {list(_SEARCHERS.keys())}")
        limit = int(args.get("limit_per_source") or 10)
        limit = max(1, min(50, limit))

        results: list[AcademicResult] = []
        errors: list[str] = []
        for src in sources:
            try:
                results.extend(_SEARCHERS[src](query, limit))
            except ToolError as e:
                errors.append(f"{src}: {e}")
            except httpx.HTTPError as e:
                errors.append(f"{src}: http error ({e})")
            except Exception as e:  # noqa: BLE001 — surface any API quirk
                errors.append(f"{src}: {type(e).__name__}: {e}")

        header = f"# academic_search({query!r})\n"
        if errors:
            header += "# errors: " + " | ".join(errors) + "\n"
        header += f"# {len(results)} result(s) across {len(sources)} source(s)\n\n"
        return header + _format_results(results)
