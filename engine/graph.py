"""Knowledge graph layer over the journal.

Nodes:
  - entry:<id>       journal entries (investigations)
  - xref:<id>        cross-references
  - insight:<id>     synthesized insights
  - register:<id>    register entries (validated insights)
  - prediction:<id>  falsifiable predictions

Edges (all undirected unless noted; weights optional):
  - shares-tag       two entries share a domain_tag (weight = tag overlap size)
  - cites-source     two entries reference the same URL/DOI (weight = shared count)
  - cross-referenced entry ↔ xref (and xref ↔ its source entries by definition)
  - supports-insight entry / xref → insight
  - registered-as    insight → register
  - predicts         register → prediction

Used to:
  - Select entries for cross-reference that are *graph-distant-but-connected*
    (i.e., two-hop connected through tags but not directly cross-referenced) —
    that's where intersection-of-knowledge-gaps novelty lives.
  - Surface structural summaries (density, clusters, orphans) to the user.
  - Export for external graph tooling.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

try:
    import networkx as nx
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "networkx is required for the graph layer. `pip install networkx>=3.2`."
    ) from e


def _entry_id(e: dict) -> str:
    return f"entry:{e.get('id', '')}"


def _xref_id(x: dict) -> str:
    return f"xref:{x.get('id', '')}"


def _insight_id(i: dict) -> str:
    return f"insight:{i.get('id', '')}"


def _register_id(r: dict) -> str:
    return f"register:{r.get('id', '')}"


def _prediction_id(p: dict) -> str:
    return f"prediction:{p.get('id', '')}"


def build_graph(journal) -> nx.MultiGraph:
    """Build a multigraph from the current journal state. Undirected — we want
    connectivity over semantic direction for the cross-ref use case."""
    g = nx.MultiGraph()

    # Add entry nodes.
    for e in journal.entries:
        g.add_node(
            _entry_id(e),
            kind="entry",
            question=e.get("question", ""),
            domain_tags=list(e.get("domain_tags", []) or []),
            surprise_delta=float(e.get("surprise_delta") or 0.0),
            sources=list(e.get("sources", []) or []),
            key_takeaways=list(e.get("key_takeaways", []) or []),
        )

    # Add xref nodes + edges to source entries.
    for x in journal.cross_references:
        g.add_node(
            _xref_id(x),
            kind="xref",
            connection_type=x.get("connection_type", ""),
            novelty_score=float(x.get("novelty_score") or 0.0),
            description=x.get("description", ""),
        )
        for source_entry_id in (x.get("source_entries") or []):
            eid = f"entry:{source_entry_id}"
            if g.has_node(eid):
                g.add_edge(eid, _xref_id(x), kind="cross-referenced-by")

    # Add insight nodes + link to supporting entries/xrefs.
    for i in journal.insights:
        g.add_node(
            _insight_id(i),
            kind="insight",
            title=i.get("title", ""),
            confidence=float(i.get("confidence") or 0.0),
        )
        for support_id in (i.get("supporting_evidence") or []):
            # supporting_evidence items are a mix of xref ids (x-...) and entry ids (j-...).
            if support_id.startswith("x-"):
                node = f"xref:{support_id}"
            elif support_id.startswith("j-"):
                node = f"entry:{support_id}"
            else:
                continue
            if g.has_node(node):
                g.add_edge(node, _insight_id(i), kind="supports-insight")

    # Add register nodes (insight → register).
    for r in journal.register:
        g.add_node(
            _register_id(r),
            kind="register",
            title=r.get("title", ""),
            verified_confidence=float(r.get("verified_confidence") or 0.0),
            human_review_status=r.get("human_review_status", "unreviewed"),
            status=r.get("status", "active"),
        )
        insight_node = f"insight:{r.get('insight_id', '')}"
        if g.has_node(insight_node):
            g.add_edge(insight_node, _register_id(r), kind="registered-as")

    # Add prediction nodes (register → prediction).
    for p in journal.predictions:
        g.add_node(
            _prediction_id(p),
            kind="prediction",
            claim=p.get("claim", ""),
            status=p.get("status", "pending"),
            target_date=p.get("target_date", ""),
        )
        reg_node = f"register:{p.get('register_entry_id', '')}"
        if g.has_node(reg_node):
            g.add_edge(reg_node, _prediction_id(p), kind="predicts")

    # Semantic-similarity edges (if embeddings available) across entries.
    if getattr(journal, "embeddings", None):
        try:
            from engine.embeddings import similarity_edges
            for id_a, id_b, score in similarity_edges(journal, threshold=0.65, max_per_entry=5):
                node_a = f"entry:{id_a}"
                node_b = f"entry:{id_b}"
                if g.has_node(node_a) and g.has_node(node_b):
                    g.add_edge(node_a, node_b, kind="semantic-similarity", weight=float(score))
        except Exception:
            pass

    # Tag-overlap + source-overlap edges across entries.
    entries = journal.entries
    for i in range(len(entries)):
        e_i = entries[i]
        tags_i = set(e_i.get("domain_tags") or [])
        srcs_i = set(e_i.get("sources") or [])
        for j in range(i + 1, len(entries)):
            e_j = entries[j]
            tags_j = set(e_j.get("domain_tags") or [])
            srcs_j = set(e_j.get("sources") or [])
            tag_overlap = tags_i & tags_j
            src_overlap = srcs_i & srcs_j
            if tag_overlap:
                g.add_edge(
                    _entry_id(e_i),
                    _entry_id(e_j),
                    kind="shares-tag",
                    weight=len(tag_overlap),
                    tags=sorted(tag_overlap),
                )
            if src_overlap:
                g.add_edge(
                    _entry_id(e_i),
                    _entry_id(e_j),
                    kind="cites-source",
                    weight=len(src_overlap),
                )

    return g


# ─────────────────────────────────────────────
# Cross-ref entry selection
# ─────────────────────────────────────────────

@dataclass
class _EntryScore:
    entry_id: str
    score: float
    reason: str


def select_entries_for_xref(
    journal,
    *,
    window: int = 20,
) -> list[dict]:
    """Pick entries for cross-reference using graph structure.

    Heuristic:
      1. Every entry is a candidate.
      2. An entry's 'intersection score' = count of (unconnected-by-existing-xref)
         pairs it participates in where the pair shares tags or sources.
      3. Include all recent entries unconditionally (recency baseline), then fill
         remaining window slots with high-intersection-score older entries.

    This replaces the flat recent-N + high-surprise selection with one that
    specifically targets entries whose connections to each other HAVEN'T yet
    been captured as a cross-reference — the structural definition of
    "knowledge-gap intersection" from the planning doc.
    """
    all_entries = journal.entries
    if len(all_entries) <= window:
        return list(all_entries)

    g = build_graph(journal)

    # Pairs already captured as cross-references. Pair = frozenset of entry nodes.
    existing_pairs: set[frozenset[str]] = set()
    for x in journal.cross_references:
        entry_nodes = [f"entry:{sid}" for sid in (x.get("source_entries") or [])]
        if len(entry_nodes) >= 2:
            for i in range(len(entry_nodes)):
                for j in range(i + 1, len(entry_nodes)):
                    existing_pairs.add(frozenset({entry_nodes[i], entry_nodes[j]}))

    # Score each entry by # of not-yet-cross-referenced connection partners.
    scores: dict[str, float] = {}
    for node, data in g.nodes(data=True):
        if data.get("kind") != "entry":
            continue
        unexplored_partners = 0
        weight = 0.0
        for neighbor in g.neighbors(node):
            if g.nodes[neighbor].get("kind") != "entry":
                continue
            if frozenset({node, neighbor}) in existing_pairs:
                continue
            # Weight by # of shares-tag / cites-source edges in the multigraph.
            edges = g.get_edge_data(node, neighbor) or {}
            for e in edges.values():
                if e.get("kind") in ("shares-tag", "cites-source", "semantic-similarity"):
                    unexplored_partners += 1
                    weight += float(e.get("weight", 1))
        # Surprise bumps the score — we prefer high-surprise entries.
        surprise = float(data.get("surprise_delta") or 0.0)
        scores[node] = weight + 2.0 * unexplored_partners + surprise

    # Baseline: most recent entries (they're likely already in the window anyway).
    recent_ids = [_entry_id(e) for e in all_entries[-max(3, window // 3):]]
    pool_ids = list(dict.fromkeys(recent_ids))  # preserve order, unique

    # Fill remainder with highest-scored unreviewed entries.
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for node, score in ranked:
        if node not in pool_ids:
            pool_ids.append(node)
        if len(pool_ids) >= window:
            break

    index = {_entry_id(e): e for e in all_entries}
    return [index[p] for p in pool_ids if p in index]


# ─────────────────────────────────────────────
# Summary / export helpers
# ─────────────────────────────────────────────


def graph_summary(journal) -> str:
    g = build_graph(journal)
    if g.number_of_nodes() == 0:
        return "Graph: empty (no entries yet)."

    kind_counts = Counter(data.get("kind") for _, data in g.nodes(data=True))
    edge_kind_counts = Counter(data.get("kind") for _, _, data in g.edges(data=True))

    lines: list[str] = []
    lines.append(f"# Knowledge graph over {journal.path}")
    lines.append("")
    lines.append(f"Nodes: {g.number_of_nodes()}")
    for kind, count in kind_counts.most_common():
        lines.append(f"  {kind:<12} {count}")
    lines.append("")
    lines.append(f"Edges: {g.number_of_edges()}")
    for kind, count in edge_kind_counts.most_common():
        lines.append(f"  {kind:<20} {count}")

    # Connected components (entry subgraph).
    entry_sub = g.subgraph(n for n, d in g.nodes(data=True) if d.get("kind") == "entry")
    if entry_sub.number_of_nodes():
        components = list(nx.connected_components(entry_sub))
        lines.append("")
        lines.append(f"Entry subgraph: {entry_sub.number_of_nodes()} nodes, "
                     f"{len(components)} connected component(s)")
        for i, comp in enumerate(sorted(components, key=len, reverse=True)[:5], start=1):
            lines.append(f"  component {i}: {len(comp)} entries")

    # Hubs — entries with the highest degree in the entry subgraph.
    if entry_sub.number_of_edges():
        hubs = sorted(entry_sub.degree(), key=lambda kv: kv[1], reverse=True)[:5]
        lines.append("")
        lines.append("Top 5 hub entries (by degree in entry subgraph):")
        for node, deg in hubs:
            data = g.nodes[node]
            q = (data.get("question") or "")[:100]
            lines.append(f"  deg={deg}  {node.split(':', 1)[1]}  — {q}")

    # Unexplored pairs (shares-tag or cites-source but no cross-ref captures them).
    existing_pairs: set[frozenset[str]] = set()
    for x in journal.cross_references:
        nodes = [f"entry:{sid}" for sid in (x.get("source_entries") or [])]
        if len(nodes) >= 2:
            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    existing_pairs.add(frozenset({nodes[a], nodes[b]}))

    unexplored = []
    for u, v, data in entry_sub.edges(data=True):
        if frozenset({u, v}) in existing_pairs:
            continue
        unexplored.append((u, v, data))
    if unexplored:
        lines.append("")
        lines.append(f"Unexplored connected pairs: {len(unexplored)} — candidate cross-refs "
                     f"the cross_reference phase could surface.")

    return "\n".join(lines)


def export_graph(journal, path: str, fmt: str = "graphml"):
    g = build_graph(journal)
    fmt = fmt.lower()
    if fmt in ("graphml", "xml"):
        nx.write_graphml(g, path)
    elif fmt in ("gexf",):
        nx.write_gexf(g, path)
    elif fmt in ("json", "node_link"):
        import json as _json
        data = nx.node_link_data(g, edges="links")
        with open(path, "w") as f:
            _json.dump(data, f, indent=2)
    else:
        raise ValueError(f"unsupported format: {fmt}. try graphml | gexf | json.")
