"""archive_access — keyless search across Internet Archive, Wikimedia Commons, Openverse.

For historical research, archived sources, and openly-licensed media.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx

from engine.tools.base import Tool, ToolError
from engine.tools._rate_limits import ARCHIVE_ORG, OPENVERSE, WIKIMEDIA

_USER_AGENT = "CuriosityEngine/0.1 (research use; contact via repo)"
_TIMEOUT = 25.0


@dataclass
class ArchiveResult:
    source: str                       # "internet_archive" | "wikimedia_commons" | "openverse"
    title: str
    url: str
    identifier: str = ""
    description: str = ""
    creator: str = ""
    year: Optional[int] = None
    license: str = ""
    media_type: str = ""              # "text" | "image" | "audio" | "video" | "..."


def _internet_archive_search(query: str, limit: int) -> list[ArchiveResult]:
    """Internet Archive advanced search. https://archive.org/help/aboutsearch.htm"""
    ARCHIVE_ORG.acquire()
    url = "https://archive.org/advancedsearch.php"
    params = {
        "q": query,
        "fl[]": ["identifier", "title", "description", "creator", "year", "mediatype", "licenseurl"],
        "rows": min(max(1, limit), 50),
        "output": "json",
    }
    with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}) as c:
        r = c.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    out: list[ArchiveResult] = []
    for item in (data.get("response", {}).get("docs") or [])[:limit]:
        identifier = item.get("identifier", "") or ""
        year = None
        y = item.get("year")
        if isinstance(y, (int, str)) and str(y)[:4].isdigit():
            year = int(str(y)[:4])
        creator = item.get("creator")
        if isinstance(creator, list):
            creator = ", ".join(str(c) for c in creator if c)
        out.append(ArchiveResult(
            source="internet_archive",
            title=str(item.get("title", "") or "").strip(),
            url=f"https://archive.org/details/{identifier}" if identifier else "",
            identifier=identifier,
            description=str(item.get("description", "") or "").strip(),
            creator=str(creator or ""),
            year=year,
            license=str(item.get("licenseurl", "") or ""),
            media_type=str(item.get("mediatype", "") or ""),
        ))
    return out


def _wikimedia_search(query: str, limit: int) -> list[ArchiveResult]:
    """Wikimedia Commons MediaWiki API."""
    WIKIMEDIA.acquire()
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srnamespace": 6,             # File: namespace
        "srlimit": min(max(1, limit), 50),
    }
    with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}) as c:
        r = c.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    out: list[ArchiveResult] = []
    for item in (data.get("query", {}).get("search") or [])[:limit]:
        pageid = item.get("pageid")
        title = str(item.get("title", "") or "")
        snippet = str(item.get("snippet", "") or "")
        import re
        snippet = re.sub(r"<[^>]+>", "", snippet)
        out.append(ArchiveResult(
            source="wikimedia_commons",
            title=title,
            url=f"https://commons.wikimedia.org/?curid={pageid}" if pageid else "",
            identifier=str(pageid) if pageid else "",
            description=snippet.strip(),
            creator="",
            year=None,
            license="see Commons page for license",
            media_type="file",
        ))
    return out


def _openverse_search(query: str, limit: int) -> list[ArchiveResult]:
    """Openverse — openly-licensed media aggregator. https://api.openverse.org/"""
    OPENVERSE.acquire()
    url = "https://api.openverse.org/v1/images/"
    params = {
        "q": query,
        "page_size": min(max(1, limit), 50),
    }
    with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}) as c:
        r = c.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    out: list[ArchiveResult] = []
    for item in (data.get("results") or [])[:limit]:
        out.append(ArchiveResult(
            source="openverse",
            title=str(item.get("title", "") or "")[:200].strip(),
            url=str(item.get("foreign_landing_url", "") or item.get("url", "") or ""),
            identifier=str(item.get("id", "") or ""),
            description="",
            creator=str(item.get("creator", "") or ""),
            year=None,
            license=str(item.get("license", "") or "") + (
                f" {item.get('license_version') or ''}".rstrip()
            ),
            media_type="image",
        ))
    return out


_SEARCHERS = {
    "internet_archive": _internet_archive_search,
    "wikimedia_commons": _wikimedia_search,
    "openverse": _openverse_search,
}


def _format_results(results: list[ArchiveResult], max_chars: int = 15_000) -> str:
    lines: list[str] = []
    for r in results:
        header = f"• [{r.source}] {r.title}"
        if r.year:
            header += f" ({r.year})"
        lines.append(header)
        if r.creator:
            lines.append(f"    creator: {r.creator}")
        if r.media_type:
            lines.append(f"    type: {r.media_type}")
        if r.license:
            lines.append(f"    license: {r.license}")
        if r.identifier:
            lines.append(f"    id: {r.identifier}")
        if r.url:
            lines.append(f"    url: {r.url}")
        if r.description:
            desc = r.description if len(r.description) < 400 else r.description[:400] + "…"
            lines.append(f"    description: {desc}")
        lines.append("")
    joined = "\n".join(lines).strip()
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "\n\n[results truncated]"
    return joined or "[no results]"


class ArchiveAccessTool(Tool):
    name = "archive_access"
    description = (
        "Search openly-licensed historical and media archives (keyless): Internet "
        "Archive (texts, audio, video, software), Wikimedia Commons (media files), "
        "Openverse (images). Useful for primary sources, historical material, and "
        "openly-licensed figures. Specify `sources` to restrict."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
            },
            "sources": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["internet_archive", "wikimedia_commons", "openverse"],
                },
                "description": "Subset of archives. Default: all three.",
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

        results: list[ArchiveResult] = []
        errors: list[str] = []
        for src in sources:
            try:
                results.extend(_SEARCHERS[src](query, limit))
            except ToolError as e:
                errors.append(f"{src}: {e}")
            except httpx.HTTPError as e:
                errors.append(f"{src}: http error ({e})")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{src}: {type(e).__name__}: {e}")

        header = f"# archive_access({query!r})\n"
        if errors:
            header += "# errors: " + " | ".join(errors) + "\n"
        header += f"# {len(results)} result(s) across {len(sources)} source(s)\n\n"
        return header + _format_results(results)
