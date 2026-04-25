"""Negative-space mapping: identify (method × problem) combinations that the
journal has NOT investigated and that the wider literature also appears to
have under-studied. Produces investigable questions for verified-empty gaps.

Pipeline:
  1. Hybrid matrix extraction — tags as anchors, LLM enriches from key_takeaways.
  2. Compute empty cells: every (method, problem) pair with zero entries.
  3. LLM classifies each empty cell: underexplored | tried_failed |
     trivially_uninteresting | regulated_boundary | adjacent_but_covered.
  4. For cells classified `underexplored`, run `academic_search` verification
     queries. If the search returns few hits, the gap is confirmed.
  5. LLM generates 1-2 investigable research questions per verified gap.
  6. Enqueue each question at priority 0.85, source=`gap:<short-id>`.
  7. Persist the full scan (matrix + gaps + classifications + enqueued ids) as
     a first-class journal artifact so subsequent scans can be diffed against it.

Triggered on-demand via the Admin tab or `--scan-gaps` — not part of the
cycle loop, because empty-cell density is only meaningful once the journal
has reached a minimum entry count (see [engine].negative_space_min_entries).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from prompts import (
    NEGATIVE_SPACE_CLASSIFY_PROMPT,
    NEGATIVE_SPACE_EXTRACT_PROMPT,
    NEGATIVE_SPACE_QUESTIONS_PROMPT,
)


def _slim_entry_for_matrix(entry: dict) -> dict:
    return {
        "id": entry.get("id"),
        "question": entry.get("question", "")[:200],
        "key_takeaways": entry.get("key_takeaways", []),
        "domain_tags": entry.get("domain_tags", []),
    }


def _cell_key(cell: dict) -> tuple[str, str]:
    return (
        (cell.get("method") or "").strip(),
        (cell.get("problem") or "").strip(),
    )


def diff_coverage_scans(older: dict, newer: dict) -> dict:
    """Compute a structural diff between two coverage scans.

    Returns:
        {
            "older_id": str, "newer_id": str,
            "gaps_filled":      [{method, problem, older_status, ...}],  # were gaps, now have coverage
            "gaps_still_open":  [{method, problem, status}],              # still verified_empty or incomplete
            "gaps_emerged":     [{method, problem, status}],              # weren't gaps before; are now
            "questions_enqueued_from_older_now_covered": [...],
            "summary": str,
        }

    A cell is a "gap" in a scan iff its `verification_status` is
    `verified_empty` OR `verification_incomplete`. Filling = the same
    (method, problem) now has coverage in the newer scan's `cells` (or is no
    longer classified as a gap). Emerging = the reverse.

    Method/problem lists can shift between scans (the LLM extractor is not
    deterministic), so cells only present in one scan's axis are reported as
    `emerged` or `filled` heuristically — if a (method, problem) pair is
    absent from the newer scan's matrix entirely, we treat it as "filled by
    reframing" rather than a persistent gap.
    """
    _OPEN = ("verified_empty", "verification_incomplete")
    older_gaps = {
        _cell_key(g): g
        for g in (older.get("gaps") or [])
        if g.get("verification_status") in _OPEN
    }
    newer_gaps = {
        _cell_key(g): g
        for g in (newer.get("gaps") or [])
        if g.get("verification_status") in _OPEN
    }
    newer_covered = {_cell_key(c) for c in (newer.get("cells") or [])}
    # "Known to exist in newer scan" = covered OR classified (any status, open or not).
    # Used to distinguish "cell was reclassified as non-gap" from "cell's axis
    # dropped out of the newer matrix entirely".
    newer_any_classified = {_cell_key(g) for g in (newer.get("gaps") or [])}
    newer_known = newer_covered | newer_any_classified

    gaps_filled = []
    gaps_still_open = []
    for key, older_gap in older_gaps.items():
        m, p = key
        if key in newer_covered:
            gaps_filled.append({
                "method": m, "problem": p,
                "older_status": older_gap.get("verification_status"),
                "resolution": "covered",
                "entry_count_now": next(
                    (c.get("entry_count", 0) for c in (newer.get("cells") or [])
                     if _cell_key(c) == key),
                    0,
                ),
            })
        elif key in newer_gaps:
            gaps_still_open.append({
                "method": m, "problem": p,
                "older_status": older_gap.get("verification_status"),
                "current_status": newer_gaps[key].get("verification_status"),
            })
        elif key in newer_known:
            # Cell appears in newer matrix classified as non-gap
            # (adjacent_but_covered, tried_failed, etc.) — filled by reclassification.
            gaps_filled.append({
                "method": m, "problem": p,
                "older_status": older_gap.get("verification_status"),
                "resolution": "reclassified",
                "entry_count_now": 0,
            })
        else:
            # Axis shifted between scans — the (method, problem) pair isn't in
            # the newer matrix at all (method or problem dropped from the
            # LLM's extraction). Treat as filled-by-reframing; still notable.
            gaps_filled.append({
                "method": m, "problem": p,
                "older_status": older_gap.get("verification_status"),
                "resolution": "axis_shifted_or_dropped",
                "entry_count_now": 0,
            })

    gaps_emerged = []
    for key, newer_gap in newer_gaps.items():
        if key in older_gaps:
            continue
        m, p = key
        gaps_emerged.append({
            "method": m, "problem": p,
            "status": newer_gap.get("verification_status"),
        })

    summary = (
        f"{len(gaps_filled)} filled · {len(gaps_still_open)} still open · "
        f"{len(gaps_emerged)} newly emerged"
    )
    return {
        "older_id": older.get("id", ""),
        "older_timestamp": older.get("timestamp", ""),
        "newer_id": newer.get("id", ""),
        "newer_timestamp": newer.get("timestamp", ""),
        "gaps_filled": gaps_filled,
        "gaps_still_open": gaps_still_open,
        "gaps_emerged": gaps_emerged,
        "summary": summary,
    }


class NegativeSpaceMixin:
    """Structural absence analysis over the journal's accumulated state.

    Only exposes one public method — `scan_gaps()`. Runs on demand.
    """

    def scan_gaps(self) -> Optional[dict]:
        """Run the full negative-space pipeline. Returns the persisted scan
        dict on success, or None if the journal is too young."""
        threshold = int(getattr(self.config, "negative_space_min_entries", 15))
        n_entries = len(self.journal.entries)
        if n_entries < threshold:
            print(
                f"\n--- NEGATIVE-SPACE SCAN (SKIPPED) ---\n"
                f"  journal has {n_entries} entries; need at least {threshold} for "
                f"empty-cell signal to be meaningful. Add more entries and try again."
            )
            return None

        print("\n--- NEGATIVE-SPACE SCAN ---")
        print(f"  journal size: {n_entries} entries")

        # Step 1: Hybrid matrix extraction.
        print("  [1/4] extracting (method × problem) matrix from journal...")
        entries = [_slim_entry_for_matrix(e) for e in self.journal.entries]
        tag_anchors = sorted(self.journal.get_all_domain_tags())
        extract_prompt = NEGATIVE_SPACE_EXTRACT_PROMPT.format(
            focus_block=self._focus_block(),
            entries_json=json.dumps(entries, indent=2),
            tag_anchors=", ".join(tag_anchors) if tag_anchors else "(none)",
        )
        try:
            extract_result = self._call_gap_extract(extract_prompt)
        except Exception as e:  # noqa: BLE001
            print(f"  [abort] matrix extraction failed: {type(e).__name__}: {e}")
            return None

        methods: list[str] = [m.strip() for m in (extract_result.get("methods") or []) if m.strip()]
        problems: list[str] = [p.strip() for p in (extract_result.get("problems") or []) if p.strip()]
        if not methods or not problems:
            print("  [abort] extractor returned empty methods/problems lists.")
            return None

        # Normalize cells + build covered-cell set.
        covered: dict[tuple[str, str], list[str]] = {}
        for c in extract_result.get("cells") or []:
            m = (c.get("method") or "").strip()
            p = (c.get("problem") or "").strip()
            ids = [i for i in (c.get("entry_ids") or []) if i]
            if m in methods and p in problems:
                covered.setdefault((m, p), []).extend(ids)
        print(f"        {len(methods)} methods × {len(problems)} problems = {len(methods)*len(problems)} cells")
        print(f"        {len(covered)} cells covered; {len(methods)*len(problems) - len(covered)} empty")

        # Step 2: Compute empty cells.
        empty_cells: list[dict] = []
        for m in methods:
            for p in problems:
                if (m, p) in covered:
                    continue
                empty_cells.append({"method": m, "problem": p})

        if not empty_cells:
            print("  [done] matrix is fully covered; no gaps to investigate.")
            scan = self._persist_scan(
                methods=methods, problems=problems, covered=covered,
                classified=[], verified_gaps=[], gap_questions={},
                summary="Matrix fully covered — no empty cells at this journal size.",
            )
            return scan

        # Step 3: Classify empty cells.
        print(f"  [2/4] classifying {len(empty_cells)} empty cell(s)...")
        covered_cells_list = [
            {"method": m, "problem": p, "entry_count": len(ids)}
            for (m, p), ids in covered.items()
        ]
        classify_prompt = NEGATIVE_SPACE_CLASSIFY_PROMPT.format(
            focus_block=self._focus_block(),
            methods_json=json.dumps(methods, indent=2),
            problems_json=json.dumps(problems, indent=2),
            covered_cells_json=json.dumps(covered_cells_list, indent=2),
            empty_cells_json=json.dumps(empty_cells, indent=2),
        )
        try:
            classify_result = self._call_gap_classify(classify_prompt)
        except Exception as e:  # noqa: BLE001
            print(f"  [abort] classification failed: {type(e).__name__}: {e}")
            return None

        classified: list[dict] = classify_result.get("classified_cells") or []
        class_counts = Counter(c.get("classification", "?") for c in classified)
        for k, v in class_counts.most_common():
            print(f"        {k}: {v}")

        # Step 4: Verify `underexplored` classifications via academic_search.
        # Uses `count_results_structured` — authoritative per-source result counts
        # with explicit error tracking. Previously this path string-counted "doi:"
        # occurrences in the formatted-text output AND mistakenly called
        # `tool.execute(...)` on the class (not an instance), causing a silent
        # TypeError that dropped the entire verification step to no-op.
        from engine.tools.academic_search import count_results_structured
        hit_threshold = int(getattr(self.config, "gap_verification_hit_threshold", 5))
        underexplored = [c for c in classified if c.get("classification") == "underexplored"]
        print(f"  [3/4] verifying {len(underexplored)} underexplored gap(s) via academic_search...")
        verified_gaps: list[dict] = []
        incomplete_gaps: list[dict] = []
        for cell in underexplored:
            queries = [q for q in (cell.get("verification_search_queries") or []) if q.strip()][:2]
            per_query: list[dict] = []
            any_errors = False
            total_hits = 0
            for q in queries:
                counted = count_results_structured(q, limit_per_source=5)
                per_query.append({
                    "query": q,
                    "total_hits": counted["total"],
                    "per_source": counted["per_source"],
                    "errors": counted["errors"],
                    "complete": counted["complete"],
                })
                total_hits += counted["total"]
                if not counted["complete"]:
                    any_errors = True

            # Verification status requires ALL queries to have run cleanly. A
            # partial failure (e.g. one source rate-limited) halves the
            # evidence base; rather than silently accept it, mark the cell
            # "verification_incomplete" so the scan artifact preserves the
            # signal and the user can retry. Incomplete cells do NOT enqueue
            # questions and are NOT counted among verified gaps.
            cell_short = f"{cell.get('method','?')[:30]} × {cell.get('problem','?')[:30]}"
            if not queries or any_errors:
                incomplete_gaps.append({
                    **cell,
                    "search_verification": per_query,
                    "total_hits_estimate": total_hits,
                    "verification_status": "incomplete",
                })
                reason = "no queries" if not queries else "query errors"
                print(f"        ⊘ {cell_short}  (verification incomplete: {reason})")
                continue
            if total_hits < hit_threshold:
                verified_gaps.append({
                    **cell,
                    "search_verification": per_query,
                    "total_hits_estimate": total_hits,
                    "verification_status": "verified_empty",
                })
                print(f"        ✓ {cell_short}  (hits={total_hits} < {hit_threshold})")
            else:
                print(f"        ✗ {cell_short}  (hits={total_hits} ≥ {hit_threshold}, likely adjacent_but_covered)")

        # Step 5: Generate investigable questions for verified gaps.
        gap_questions: dict[tuple[str, str], list[str]] = {}
        enqueued_ids: list[str] = []
        if verified_gaps:
            print(f"  [4/4] generating investigable questions for {len(verified_gaps)} verified gap(s)...")
            questions_prompt = NEGATIVE_SPACE_QUESTIONS_PROMPT.format(
                focus_block=self._focus_block(),
                verified_gaps_json=json.dumps(verified_gaps, indent=2),
            )
            try:
                questions_result = self._call_gap_classify(questions_prompt)
            except Exception as e:  # noqa: BLE001
                print(f"        [warn] question generation failed: {type(e).__name__}: {e}")
                questions_result = {"gap_questions": []}
            for gq in (questions_result.get("gap_questions") or []):
                m = (gq.get("method") or "").strip()
                p = (gq.get("problem") or "").strip()
                qs = [q.strip() for q in (gq.get("questions") or []) if q.strip()]
                if not qs:
                    continue
                gap_questions[(m, p)] = qs
                prefix = f"[gap: {m[:30]} × {p[:30]}] "
                prefixed = [prefix + q for q in qs]
                scan_short_id = uuid4().hex[:6]
                self._enqueue_questions(
                    prefixed,
                    source=f"gap:{scan_short_id}",
                    priority=0.85,
                )
                enqueued_ids.append(f"gap:{scan_short_id}")
                for q in qs:
                    print(f"        + {q[:120]}")
        else:
            print("  [done] no verified gaps — no questions enqueued.")

        incomplete_note = (
            f" {len(incomplete_gaps)} verification_incomplete (retry-worthy);"
            if incomplete_gaps else ""
        )
        summary = (
            f"Scan over {n_entries} entries. Matrix: {len(methods)} methods × "
            f"{len(problems)} problems. "
            f"{len(covered)} cells covered, {len(empty_cells)} empty. "
            f"{len(underexplored)} classified underexplored;{incomplete_note} "
            f"{len(verified_gaps)} verified by search; "
            f"{sum(len(qs) for qs in gap_questions.values())} questions enqueued."
        )
        print(f"\n  {summary}")

        scan = self._persist_scan(
            methods=methods, problems=problems, covered=covered,
            classified=classified, verified_gaps=verified_gaps,
            incomplete_gaps=incomplete_gaps,
            gap_questions=gap_questions, summary=summary,
        )
        return scan

    def _persist_scan(
        self,
        *,
        methods: list[str],
        problems: list[str],
        covered: dict[tuple[str, str], list[str]],
        classified: list[dict],
        verified_gaps: list[dict],
        gap_questions: dict[tuple[str, str], list[str]],
        summary: str,
        incomplete_gaps: Optional[list[dict]] = None,
    ) -> dict:
        """Persist the scan into journal.coverage_scans."""
        cells = [
            {"method": m, "problem": p, "entry_ids": list(ids), "entry_count": len(ids)}
            for (m, p), ids in covered.items()
        ]

        gaps: list[dict] = []
        # Index classifications by (method, problem) for quick merge.
        class_by_cell: dict[tuple[str, str], dict] = {
            ((c.get("method") or "").strip(), (c.get("problem") or "").strip()): c
            for c in classified
        }
        verified_by_key: dict[tuple[str, str], dict] = {
            ((g.get("method") or "").strip(), (g.get("problem") or "").strip()): g
            for g in verified_gaps
        }
        incomplete_by_key: dict[tuple[str, str], dict] = {
            ((g.get("method") or "").strip(), (g.get("problem") or "").strip()): g
            for g in (incomplete_gaps or [])
        }
        for cell_key, cell_info in class_by_cell.items():
            m, p = cell_key
            if cell_key in verified_by_key:
                verification_status = "verified_empty"
                verification = verified_by_key[cell_key]
            elif cell_key in incomplete_by_key:
                verification_status = "verification_incomplete"
                verification = incomplete_by_key[cell_key]
            else:
                # Classified but either not "underexplored" or verified to have
                # coverage (adjacent_but_covered). Neither enqueued nor retry-worthy.
                verification_status = cell_info.get("classification", "?")
                verification = None
            gaps.append({
                "method": m,
                "problem": p,
                "classification": cell_info.get("classification", "?"),
                "reasoning": cell_info.get("reasoning", ""),
                "verification_search_queries": cell_info.get("verification_search_queries", []),
                "verified_empty": verification_status == "verified_empty",
                "verification_status": verification_status,
                "verification_hits_estimate": (
                    verification.get("total_hits_estimate") if verification else None
                ),
                "search_verification": (
                    verification.get("search_verification") if verification else []
                ),
                "questions_enqueued": gap_questions.get(cell_key, []),
            })

        scan = {
            "id": f"cs-{uuid4().hex[:8]}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "journal_size_at_scan": len(self.journal.entries),
            "methods": methods,
            "problems": problems,
            "cells": cells,
            "gaps": gaps,
            "summary": summary,
        }
        self.journal.add_coverage_scan(scan)
        print(f"  [saved] coverage scan {scan['id']}")
        return scan
