"""JSON extraction helpers for LLM text responses.

Ported from loom/src/loom/engine/semantic_compactor/parse.py.
"""

from __future__ import annotations

import json
import re
from typing import Any


def strip_markdown_fences(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    content = "\n".join(lines[1:])
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        start = match.start()
        try:
            candidate, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a model response expected to contain a JSON object.

    Tolerates markdown fences and junk before/after the object.
    """
    stripped = strip_markdown_fences(str(text or "").strip())
    if not stripped:
        raise ValueError("empty response")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = extract_first_json_object(stripped)
        if parsed is None:
            raise ValueError(f"could not extract JSON from response:\n{stripped[:500]}")
    if not isinstance(parsed, dict):
        raise ValueError("response JSON is not an object")
    return parsed
