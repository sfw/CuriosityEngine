"""Semantic embeddings over journal entries.

Uses an OpenAI-compatible embeddings endpoint (text-embedding-3-small by default).
Gracefully disabled when no suitable provider is configured — the engine will
run, but `--find-similar` and similarity edges in the graph will be unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from providers import EmbeddingClient


def _entry_text(entry: dict) -> str:
    """What we actually embed: question + key_takeaways. Short, semantically dense."""
    parts: list[str] = []
    if entry.get("question"):
        parts.append(entry["question"])
    takeaways = entry.get("key_takeaways") or []
    if takeaways:
        parts.append(". ".join(takeaways))
    return "\n".join(parts)[:8000]  # cap for cheap embedding


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class SimilarityHit:
    entry_id: str
    score: float
    question: str


def embed_missing_entries(journal, client: EmbeddingClient, *, batch_size: int = 50) -> int:
    """Compute embeddings for any journal entry that doesn't have one yet.

    Returns the number of newly-embedded entries.
    """
    missing_ids = journal.missing_embedding_entry_ids()
    if not missing_ids:
        return 0

    index = {e["id"]: e for e in journal.entries}
    embedded = 0
    for i in range(0, len(missing_ids), batch_size):
        chunk_ids = missing_ids[i:i + batch_size]
        texts = [_entry_text(index[eid]) for eid in chunk_ids]
        vectors = client.embed(texts)
        for eid, vec in zip(chunk_ids, vectors):
            journal.set_embedding(eid, vec)
            embedded += 1
    return embedded


def find_similar(
    query: str,
    journal,
    client: EmbeddingClient,
    *,
    top_k: int = 10,
    min_score: float = 0.0,
) -> list[SimilarityHit]:
    """Embed the query and return top-k most similar journal entries."""
    q_vec = client.embed([query])[0]
    hits: list[SimilarityHit] = []
    for entry in journal.entries:
        eid = entry.get("id", "")
        vec = journal.embeddings.get(eid)
        if not vec:
            continue
        score = cosine(q_vec, vec)
        if score >= min_score:
            hits.append(SimilarityHit(
                entry_id=eid,
                score=score,
                question=entry.get("question", ""),
            ))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def similarity_edges(
    journal,
    *,
    threshold: float = 0.65,
    max_per_entry: int = 5,
) -> list[tuple[str, str, float]]:
    """Return pairs of entry_ids whose embeddings exceed `threshold`. Limits the
    number of high-similarity neighbors per entry to keep the graph sparse."""
    entries_with_vec = [
        (e.get("id"), journal.embeddings.get(e.get("id")))
        for e in journal.entries
        if e.get("id") and e.get("id") in journal.embeddings
    ]
    edges: list[tuple[str, str, float]] = []
    per_entry_counts: dict[str, int] = {}

    # Precompute all pairs; sort by score descending; keep until per-entry budget hits.
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(entries_with_vec)):
        id_i, vec_i = entries_with_vec[i]
        for j in range(i + 1, len(entries_with_vec)):
            id_j, vec_j = entries_with_vec[j]
            s = cosine(vec_i, vec_j)
            if s >= threshold:
                pairs.append((id_i, id_j, s))

    pairs.sort(key=lambda t: t[2], reverse=True)
    for id_a, id_b, s in pairs:
        if per_entry_counts.get(id_a, 0) >= max_per_entry:
            continue
        if per_entry_counts.get(id_b, 0) >= max_per_entry:
            continue
        edges.append((id_a, id_b, s))
        per_entry_counts[id_a] = per_entry_counts.get(id_a, 0) + 1
        per_entry_counts[id_b] = per_entry_counts.get(id_b, 0) + 1
    return edges
