"""Tests for direction insight backpropagation.

Standalone + assert-based so they run with plain `python` (no pytest dep), while
remaining pytest-discoverable if pytest is added later. The LLM abstraction call
is never exercised here — only the deterministic clustering, storage, and
injection logic.

Run: .venv/bin/python tests/test_direction_backprop.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.direction_backprop import (  # noqa: E402
    build_frontier_edges_block,
    member_signature,
    select_direction_clusters,
)
from journal import Journal  # noqa: E402


def _entry(eid: str, tags: list[str]) -> dict:
    return {
        "id": eid,
        "question": f"q for {eid}",
        "domain_tags": tags,
        "surprise_delta": 0.5,
        "sources": [],
        "key_takeaways": [f"takeaway {eid}"],
    }


def _stub_journal(entries: list[dict]):
    return SimpleNamespace(
        entries=entries,
        cross_references=[],
        insights=[],
        register=[],
        predictions=[],
        embeddings={},
        path="<stub>",
    )


def test_clustering_respects_min_size_and_cap():
    # alpha: 4 entries (qualifies), beta: 2 entries, gamma: 1 entry.
    entries = (
        [_entry(f"a{i}", ["alpha"]) for i in range(4)]
        + [_entry(f"b{i}", ["beta"]) for i in range(2)]
        + [_entry("g0", ["gamma"])]
    )
    j = _stub_journal(entries)

    clusters = select_direction_clusters(j, min_cluster_size=4, max_count=5)
    assert len(clusters) == 1, f"expected only alpha to qualify, got {len(clusters)}"
    assert len(clusters[0]) == 4

    clusters2 = select_direction_clusters(j, min_cluster_size=2, max_count=5)
    sizes = sorted(len(c) for c in clusters2)
    assert sizes == [2, 4], f"expected alpha(4)+beta(2), got {sizes}"

    # Cap to 1 → only the largest (alpha) component.
    capped = select_direction_clusters(j, min_cluster_size=2, max_count=1)
    assert len(capped) == 1 and len(capped[0]) == 4

    # Largest-first ordering.
    assert len(clusters2[0]) >= len(clusters2[-1])
    print("ok: clustering respects min_size, cap, ordering")


def test_member_signature_is_order_invariant():
    assert member_signature(["x", "y", "z"]) == member_signature(["z", "x", "y"])
    assert member_signature(["a", "b"]) != member_signature(["a", "c"])
    print("ok: member_signature order-invariant + collision-sane")


def test_frontier_block_cap_and_framing():
    insights = [
        {"label": "L1", "settled": ["s1"], "open_edge": "edge one", "confidence": 0.9},
        {"label": "L2", "settled": ["s2"], "open_edge": "edge two", "confidence": 0.5},
        {"label": "L3", "settled": [], "open_edge": "edge three", "confidence": 0.1},
    ]
    block = build_frontier_edges_block(insights, max_injected=2)
    assert "FRONTIER EDGES" in block
    assert "push PAST" in block
    # Cap respected: only the 2 highest-confidence edges.
    assert "edge one" in block and "edge two" in block
    assert "edge three" not in block
    # Confidence ordering: L1 before L2.
    assert block.index("edge one") < block.index("edge two")

    # Empty inputs / zero cap → empty string.
    assert build_frontier_edges_block([], max_injected=4) == ""
    assert build_frontier_edges_block(insights, max_injected=0) == ""

    # An insight with no open_edge contributes no line.
    only_blank = [{"label": "X", "settled": [], "open_edge": "", "confidence": 1.0}]
    assert build_frontier_edges_block(only_blank, max_injected=4) == ""
    print("ok: frontier block cap, framing, ordering, blank-skip")


def test_storage_supersede_suppress_and_persistence():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "journal.json")
        j = Journal(path)

        rec_a = {
            "id": "dir-a", "label": "A", "member_entry_ids": ["e1", "e2", "e3", "e4"],
            "member_signature": member_signature(["e1", "e2", "e3", "e4"]),
            "settled": ["s"], "open_edge": "edge A", "confidence": 0.7,
            "generated_at": "2026-06-18T00:00:00+00:00", "model": "m",
            "journal_size_at_gen": 4, "suppressed": False, "supersedes": None,
        }
        j.add_direction_insight(rec_a)
        assert [r["id"] for r in j.latest_direction_insights()] == ["dir-a"]

        # Grown cluster supersedes A.
        rec_b = dict(rec_a)
        rec_b.update({
            "id": "dir-b", "member_entry_ids": ["e1", "e2", "e3", "e4", "e5"],
            "member_signature": member_signature(["e1", "e2", "e3", "e4", "e5"]),
            "open_edge": "edge B", "generated_at": "2026-06-18T01:00:00+00:00",
            "supersedes": "dir-a",
        })
        j.add_direction_insight(rec_b)
        heads = [r["id"] for r in j.direction_heads()]
        assert heads == ["dir-b"], f"A should be superseded, heads={heads}"
        assert [r["id"] for r in j.latest_direction_insights()] == ["dir-b"]

        # Suppress the head → injection view empty, no fallback to A.
        assert j.suppress_direction_insight("dir-b") is True
        assert j.latest_direction_insights() == []
        assert [r["id"] for r in j.direction_heads()] == ["dir-b"]  # still a head
        assert j.suppress_direction_insight("dir-missing") is False

        # Persistence across reload.
        j2 = Journal(path)
        assert len(j2.direction_insights) == 2
        assert j2.latest_direction_insights() == []  # suppression persisted
    print("ok: storage supersede chain, suppress, persistence")


def main():
    test_clustering_respects_min_size_and_cap()
    test_member_signature_is_order_invariant()
    test_frontier_block_cap_and_framing()
    test_storage_supersede_suppress_and_persistence()
    print("\nALL PASSED")


if __name__ == "__main__":
    main()
