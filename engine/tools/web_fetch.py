"""web_fetch — HTTP GET a URL and return extracted plaintext.

Safety:
- Only http(s) schemes.
- Private / loopback / link-local IPs rejected (prevents SSRF to internal services).
- Response size capped.
- Timeout enforced.
- Basic user-agent set.

Content extraction prefers trafilatura's article extraction; falls back to a
conservative BeautifulSoup-style text pull if trafilatura isn't installed.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx

from engine.tools.base import Tool, ToolError

_MAX_BYTES = 2_000_000          # 2 MB
_TIMEOUT_SECONDS = 20.0
_USER_AGENT = "CuriosityEngine/0.1 (+https://github.com/anthropics/claude-code-skills)"
_MAX_OUTPUT_CHARS = 30_000


def _ensure_safe_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolError(f"unsupported scheme: {parsed.scheme}")
    host = parsed.hostname
    if not host:
        raise ToolError("url has no hostname")

    # Resolve to check against private/loopback ranges (SSRF guard).
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except socket.gaierror as e:
        raise ToolError(f"could not resolve host: {e}") from e

    for addr in addresses:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ToolError(f"refusing to fetch private/loopback address ({ip})")


def _extract_text(html: str, url: str) -> str:
    # Prefer trafilatura when available — best article extraction we can easily install.
    try:
        import trafilatura
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        if extracted:
            return extracted.strip()
    except ImportError:
        pass

    # Fallback: minimal tag stripping via html.parser + whitespace normalization.
    from html.parser import HTMLParser

    class _TextPuller(HTMLParser):
        def __init__(self):
            super().__init__()
            self._chunks: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "noscript"):
                self._skip += 1

        def handle_endtag(self, tag):
            if tag in ("script", "style", "noscript") and self._skip:
                self._skip -= 1

        def handle_data(self, data):
            if self._skip:
                return
            self._chunks.append(data)

        def text(self) -> str:
            raw = "".join(self._chunks)
            return "\n".join(line.strip() for line in raw.splitlines() if line.strip())

    parser = _TextPuller()
    parser.feed(html)
    return parser.text()


class WebFetchTool(Tool):
    name = "web_fetch"
    description = (
        "Fetch a web URL and return its plaintext content. Use this after web_search "
        "to read a specific page. Rejects non-http(s) URLs and private IPs. Max 2 MB "
        "response; output truncated to 30k characters. Prefer this to copy-pasting "
        "URLs into the verifier prompt."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute http(s) URL to fetch.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Max characters of plaintext to return (default 30000).",
                "default": _MAX_OUTPUT_CHARS,
                "minimum": 500,
                "maximum": 100_000,
            },
        },
        "required": ["url"],
    }
    timeout_seconds = float(_TIMEOUT_SECONDS + 5.0)

    def execute(self, args: dict) -> str:
        url = (args.get("url") or "").strip()
        if not url:
            raise ToolError("no url provided")
        max_chars = int(args.get("max_chars") or _MAX_OUTPUT_CHARS)
        max_chars = max(500, min(100_000, max_chars))

        _ensure_safe_url(url)

        # Per-host rate limiting so a burst of web_fetch calls against a single
        # paper server (e.g. arxiv.org in a verification sweep) doesn't hammer
        # it — especially important under parallel investigation fan-out.
        from engine.tools._rate_limits import WEB_FETCH
        host = urlparse(url).hostname or ""
        if host:
            WEB_FETCH.acquire(host)

        headers = {"User-Agent": _USER_AGENT, "Accept": "text/html,text/plain,*/*"}
        try:
            with httpx.Client(timeout=_TIMEOUT_SECONDS, follow_redirects=True) as client:
                with client.stream("GET", url, headers=headers) as response:
                    if response.status_code >= 400:
                        raise ToolError(
                            f"http {response.status_code} from {url}"
                        )
                    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                    if content_type and not (
                        content_type.startswith("text/")
                        or content_type.endswith("json")
                        or content_type.endswith("xml")
                        or content_type.endswith("html")
                    ):
                        raise ToolError(f"unsupported content-type: {content_type}")

                    chunks: list[bytes] = []
                    total = 0
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > _MAX_BYTES:
                            raise ToolError(f"response exceeded {_MAX_BYTES} bytes")
                        chunks.append(chunk)
                    body = b"".join(chunks)
        except httpx.RequestError as e:
            raise ToolError(f"request failed: {e}") from e

        text = body.decode("utf-8", errors="replace")
        if content_type.endswith("html") or "<html" in text[:1000].lower():
            text = _extract_text(text, url)

        if not text.strip():
            raise ToolError("no text content extracted")

        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars]
            truncated = True

        header_note = f"# Fetched: {url}\n# Content-Type: {content_type or 'unknown'}\n"
        if truncated:
            header_note += f"# NOTE: content truncated to {max_chars} chars\n"
        return header_note + "\n" + text
