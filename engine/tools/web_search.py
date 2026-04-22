"""web_search — keyless web search via DuckDuckGo HTML endpoint.

Intended as a fallback for non-Anthropic primaries that don't have access to the
Anthropic server-side web_search tool. DuckDuckGo's HTML endpoint (the no-JS lite
version) is scrape-friendly and doesn't require a key.

Rate-limited to one request per 2s by default; cooldown doubles on 429/403.

Fallback: if DuckDuckGo fails, try Bing's HTML results. Both are best-effort scrapers;
not as robust as an API.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import quote_plus, urlparse, parse_qs, unquote

import httpx

from engine.tools.base import Tool, ToolError

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_MIN_INTERVAL = 2.0
_COOLDOWN_AFTER_FAIL = 60.0
_TIMEOUT = 20.0


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""


class _PaceGate:
    """Simple per-host throttler that enforces min interval and cooldown on failure."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last: dict[str, float] = {}
        self._cooldown_until: dict[str, float] = {}

    def wait_for(self, host: str):
        now = time.time()
        with self._lock:
            cooldown = self._cooldown_until.get(host, 0.0)
            last = self._last.get(host, 0.0)
        if now < cooldown:
            time.sleep(cooldown - now)
        elif now - last < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - (now - last))
        with self._lock:
            self._last[host] = time.time()

    def on_fail(self, host: str):
        with self._lock:
            self._cooldown_until[host] = time.time() + _COOLDOWN_AFTER_FAIL


_gate = _PaceGate()


class _DDGResultParser(HTMLParser):
    """Minimal parser for DuckDuckGo HTML lite results."""

    def __init__(self):
        super().__init__()
        self.hits: list[SearchHit] = []
        self._in_result = False
        self._in_title = False
        self._in_snippet = False
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []
        self._current_url: str = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if tag == "div" and "result" in classes and "result--no-result" not in classes:
            self._in_result = True
            self._current_title = []
            self._current_snippet = []
            self._current_url = ""
        elif self._in_result and tag == "a" and "result__a" in classes:
            self._in_title = True
            href = a.get("href", "")
            self._current_url = _clean_ddg_href(href)
        elif self._in_result and tag == "a" and "result__snippet" in classes:
            self._in_snippet = True

    def handle_endtag(self, tag):
        if tag == "a":
            if self._in_title:
                self._in_title = False
            elif self._in_snippet:
                self._in_snippet = False
        if tag == "div" and self._in_result:
            title = " ".join(t.strip() for t in self._current_title).strip()
            snippet = " ".join(s.strip() for s in self._current_snippet).strip()
            if title and self._current_url:
                self.hits.append(SearchHit(
                    title=title,
                    url=self._current_url,
                    snippet=snippet,
                ))
            self._in_result = False

    def handle_data(self, data):
        if self._in_title:
            self._current_title.append(data)
        elif self._in_snippet:
            self._current_snippet.append(data)


def _clean_ddg_href(href: str) -> str:
    """DDG wraps result URLs in /l/?uddg=<encoded>."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.path.startswith("/l/") and parsed.query:
        q = parse_qs(parsed.query)
        if "uddg" in q:
            return unquote(q["uddg"][0])
    return href


def _search_ddg(query: str, limit: int) -> list[SearchHit]:
    host = "duckduckgo.com"
    _gate.wait_for(host)
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query, "kl": "us-en"}
    try:
        with httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html"},
            follow_redirects=True,
        ) as c:
            r = c.post(url, data=params)
        if r.status_code in (429, 403):
            _gate.on_fail(host)
            raise ToolError(f"duckduckgo throttled (http {r.status_code})")
        r.raise_for_status()
    except httpx.HTTPError as e:
        _gate.on_fail(host)
        raise ToolError(f"duckduckgo request failed: {e}") from e

    parser = _DDGResultParser()
    parser.feed(r.text)
    return parser.hits[:limit]


class _BingResultParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hits: list[SearchHit] = []
        self._in_h2 = False
        self._in_a = False
        self._current_title: list[str] = []
        self._current_url = ""
        self._in_snippet = False
        self._current_snippet: list[str] = []
        self._hit_in_progress = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "li" and (a.get("class") or "").startswith("b_algo"):
            self._hit_in_progress = True
            self._current_title = []
            self._current_snippet = []
            self._current_url = ""
        elif self._hit_in_progress and tag == "h2":
            self._in_h2 = True
        elif self._in_h2 and tag == "a":
            self._in_a = True
            self._current_url = a.get("href", "")
        elif self._hit_in_progress and tag == "p":
            self._in_snippet = True

    def handle_endtag(self, tag):
        if tag == "a" and self._in_a:
            self._in_a = False
        elif tag == "h2" and self._in_h2:
            self._in_h2 = False
        elif tag == "p" and self._in_snippet:
            self._in_snippet = False
        elif tag == "li" and self._hit_in_progress:
            title = " ".join(self._current_title).strip()
            snippet = " ".join(self._current_snippet).strip()
            if title and self._current_url:
                self.hits.append(SearchHit(
                    title=title,
                    url=self._current_url,
                    snippet=snippet,
                ))
            self._hit_in_progress = False

    def handle_data(self, data):
        if self._in_a:
            self._current_title.append(data)
        elif self._in_snippet:
            self._current_snippet.append(data)


def _search_bing(query: str, limit: int) -> list[SearchHit]:
    host = "bing.com"
    _gate.wait_for(host)
    url = f"https://www.bing.com/search?q={quote_plus(query)}&count={limit}"
    try:
        with httpx.Client(
            timeout=_TIMEOUT,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html"},
            follow_redirects=True,
        ) as c:
            r = c.get(url)
        if r.status_code in (429, 403):
            _gate.on_fail(host)
            raise ToolError(f"bing throttled (http {r.status_code})")
        r.raise_for_status()
    except httpx.HTTPError as e:
        _gate.on_fail(host)
        raise ToolError(f"bing request failed: {e}") from e

    parser = _BingResultParser()
    parser.feed(r.text)
    return parser.hits[:limit]


def _format_hits(hits: list[SearchHit]) -> str:
    lines: list[str] = []
    for i, h in enumerate(hits, start=1):
        lines.append(f"{i}. {h.title}")
        lines.append(f"   {h.url}")
        if h.snippet:
            snip = h.snippet if len(h.snippet) < 300 else h.snippet[:300] + "…"
            lines.append(f"   {snip}")
        lines.append("")
    return "\n".join(lines).strip() or "[no results]"


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Keyless web search via DuckDuckGo (primary) and Bing (fallback) HTML "
        "endpoints. Returns ranked results with title, url, and snippet. Use for "
        "general-purpose web discovery; follow with web_fetch to read a specific "
        "page. Rate-limited; expect cooldowns if providers throttle."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 10, max 25).",
                "default": 10,
                "minimum": 1,
                "maximum": 25,
            },
        },
        "required": ["query"],
    }
    timeout_seconds = 60.0

    def execute(self, args: dict) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            raise ToolError("no query provided")
        limit = max(1, min(25, int(args.get("limit") or 10)))

        errors: list[str] = []
        hits: list[SearchHit] = []
        try:
            hits = _search_ddg(query, limit)
        except ToolError as e:
            errors.append(f"duckduckgo: {e}")

        if not hits:
            try:
                hits = _search_bing(query, limit)
            except ToolError as e:
                errors.append(f"bing: {e}")

        header = f"# web_search({query!r})\n"
        if errors:
            header += "# errors: " + " | ".join(errors) + "\n"
        header += f"# {len(hits)} result(s)\n\n"
        return header + _format_hits(hits)
