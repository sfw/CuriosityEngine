"""Direction insight backpropagation (Arbor / Hypothesis Tree Refinement,
arXiv:2606.11926).

Abstracts clusters of related investigations into direction-level priors and
injects them into introspection as FRONTIER EDGES to push past — the
carry-forward layer CE was missing. Adapted to CE's novelty objective: the
distilled signal is framed as an edge to *exceed*, never a belief to *confirm*,
and abstraction runs on the cross-family verifier model (never the primary) to
avoid the self-flattery loop.

Design: docs/superpowers/specs/2026-06-18-direction-backprop-design.md
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from uuid import uuid4

import networkx as nx

from engine.graph import build_graph
from prompts import ABSTRACT_DIRECTION_PROMPT

# A new cluster supersedes a prior head when their member sets overlap at least
# this much (Jaccard). Below it, the cluster is treated as a fresh direction.
_SUPERSEDE_JACCARD = 0.5


def member_signature(entry_ids: list[str]) -> str:
    """Stable hash of a cluster's sorted member ids. Audit + identical-rerun id."""
    joined = "|".join(sorted(entry_ids))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]


def select_direction_clusters(
    journal, *, min_cluster_size: int, max_count: int,
) -> list[list[dict]]:
    """Connected components of the entry subgraph that are large enough to be a
    "direction". Returns lists of entry dicts, largest component first, capped to
    `max_count`. Reuses the engineered xref/tag/source/semantic edges in
    engine.graph — the curated relatedness structure, CE's analog of an Arbor
    subtree.
    """
    g = build_graph(journal)
    entry_nodes = [n for n, d in g.nodes(data=True) if d.get("kind") == "entry"]
    if not entry_nodes:
        return []
    entry_sub = g.subgraph(entry_nodes)
    index = {f"entry:{e.get('id', '')}": e for e in journal.entries}

    clusters: list[list[dict]] = []
    for comp in nx.connected_components(entry_sub):
        if len(comp) < min_cluster_size:
            continue
        members = [index[n] for n in comp if n in index]
        if len(members) >= min_cluster_size:
            clusters.append(members)

    clusters.sort(key=len, reverse=True)
    return clusters[:max_count]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def build_frontier_edges_block(insights: list[dict], *, max_injected: int) -> str:
    """Render the FRONTIER EDGES injection block from latest direction insights.

    Top `max_injected` by confidence desc, then recency. Framed as edges to push
    PAST — the convergence guard. Returns "" when nothing to inject.
    """
    if max_injected <= 0 or not insights:
        return ""
    ranked = sorted(
        insights,
        key=lambda r: (float(r.get("confidence") or 0.0), r.get("generated_at", "")),
        reverse=True,
    )[:max_injected]
    if not ranked:
        return ""
    lines = [
        "\nFRONTIER EDGES (distilled from prior investigation clusters — do NOT "
        "re-confirm what is settled; your job is to push PAST these open edges):",
    ]
    for rec in ranked:
        label = (rec.get("label") or "direction").strip()
        settled = rec.get("settled") or []
        settled_clause = settled[0].strip() if settled else "—"
        open_edge = (rec.get("open_edge") or "").strip()
        if not open_edge:
            continue
        lines.append(f"  - [{label}] settled: {settled_clause}. OPEN EDGE: {open_edge}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines) + "\n"


def _cluster_payload(journal, cluster: list[dict]) -> str:
    """Compact JSON-ish text of a cluster for the abstraction prompt: each
    member's question + takeaways + surprise, plus any xref descriptions whose
    source entries fall entirely within the cluster."""
    import json

    member_ids = {e.get("id", "") for e in cluster}
    members = [
        {
            "question": e.get("question", ""),
            "key_takeaways": e.get("key_takeaways", []),
            "surprise_delta": e.get("surprise_delta", 0.0),
        }
        for e in cluster
    ]
    xrefs = [
        {
            "connection_type": x.get("connection_type", ""),
            "description": x.get("description", ""),
            "novelty_score": x.get("novelty_score", 0.0),
        }
        for x in getattr(journal, "cross_references", [])
        if set(x.get("source_entries") or []) & member_ids
    ]
    payload = {"investigations": members}
    if xrefs:
        payload["cross_references_within_cluster"] = xrefs[:10]
    return json.dumps(payload, indent=2)


class DirectionBackpropMixin:
    """Phase: direction insight backpropagation. Clusters investigations into
    directions, abstracts each into a frontier-edge prior on the verifier model,
    and persists it (append-only) for injection into introspection."""

    def maybe_abstract_directions(self) -> None:
        """In-loop trigger: abstract directions every N cycles, gated by config.
        Best-effort — a failure here must never kill the cycle."""
        eng = self.connection.engine
        if not getattr(eng, "direction_backprop_enabled", True):
            return
        every = max(1, int(getattr(eng, "direction_abstract_every_n_cycles", 5)))
        if self.cycle_count % every != 0:
            return
        if len(self.journal.entries) < int(getattr(eng, "direction_min_entries", 8)):
            return
        try:
            self.abstract_directions()
        except Exception as e:  # noqa: BLE001 — never kill the cycle
            print(f"  [direction-backprop] skipped: {type(e).__name__}: {str(e)[:160]}")

    def abstract_directions(self) -> dict:
        """Abstract qualifying direction clusters into frontier-edge priors.

        Returns {added, skipped, candidates}. Skips a cluster whose membership is
        unchanged since its last abstraction, or that overlaps a suppressed
        direction (respecting the human's kill)."""
        print("\n--- ABSTRACTING DIRECTIONS ---")
        eng = self.connection.engine
        clusters = select_direction_clusters(
            self.journal,
            min_cluster_size=int(getattr(eng, "direction_min_cluster_size", 4)),
            max_count=int(getattr(eng, "direction_max_count", 5)),
        )
        if not clusters:
            print("  no qualifying direction clusters yet.")
            return {"added": 0, "skipped": 0, "candidates": 0}

        heads = self.journal.direction_heads()
        added = 0
        skipped = 0
        for cluster in clusters:
            entry_ids = sorted(e.get("id", "") for e in cluster if e.get("id"))
            member_set = set(entry_ids)
            sig = member_signature(entry_ids)

            # Best-overlapping prior head (including suppressed ones).
            best = None
            best_j = 0.0
            for h in heads:
                j = _jaccard(member_set, set(h.get("member_entry_ids") or []))
                if j > best_j:
                    best, best_j = h, j

            if best is not None and best_j >= _SUPERSEDE_JACCARD:
                if best.get("suppressed"):
                    skipped += 1  # respect the human's suppression of this lineage
                    continue
                if best.get("member_signature") == sig:
                    skipped += 1  # identical membership — nothing changed
                    continue
                supersedes = best.get("id")
            else:
                supersedes = None

            record = self._abstract_one_direction(cluster, entry_ids, sig, supersedes)
            if record:
                self.journal.add_direction_insight(record)
                # Newly added record becomes a head; drop the one it supersedes.
                heads = [h for h in heads if h.get("id") != supersedes]
                heads.append(record)
                added += 1
                print(f"  [+] {record['label']} → OPEN EDGE: {record['open_edge'][:90]}")

        print(f"  directions: +{added} new, {skipped} unchanged/suppressed "
              f"({len(clusters)} candidates).")
        return {"added": added, "skipped": skipped, "candidates": len(clusters)}

    def _abstract_one_direction(
        self, cluster: list[dict], entry_ids: list[str], sig: str, supersedes,
    ) -> dict | None:
        """One verifier-model abstraction call → a shaped direction record."""
        prompt = ABSTRACT_DIRECTION_PROMPT.format(
            domain=self.config.domain,
            entry_count=len(cluster),
            cluster_json=_cluster_payload(self.journal, cluster),
        )
        try:
            result = self._call_verifier(prompt)
        except Exception as e:  # noqa: BLE001 — best-effort per cluster
            print(f"  [direction] abstraction call failed: {type(e).__name__}: {str(e)[:140]}")
            return None
        open_edge = (result.get("open_edge") or "").strip()
        if not open_edge:
            return None
        settled = result.get("settled") or []
        if isinstance(settled, str):
            settled = [settled]
        return {
            "id": f"dir-{uuid4().hex[:8]}",
            "label": (result.get("label") or "direction").strip(),
            "member_entry_ids": list(entry_ids),
            "member_signature": sig,
            "settled": [str(s).strip() for s in settled if str(s).strip()][:2],
            "open_edge": open_edge,
            "confidence": float(result.get("confidence") or 0.0),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": getattr(self.connection.verifier, "name", ""),
            "journal_size_at_gen": len(self.journal.entries),
            "suppressed": False,
            "supersedes": supersedes,
        }
