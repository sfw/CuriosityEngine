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
            extract_result = self._call_primary(extract_prompt)
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
            classify_result = self._call_primary(classify_prompt)
        except Exception as e:  # noqa: BLE001
            print(f"  [abort] classification failed: {type(e).__name__}: {e}")
            return None

        classified: list[dict] = classify_result.get("classified_cells") or []
        class_counts = Counter(c.get("classification", "?") for c in classified)
        for k, v in class_counts.most_common():
            print(f"        {k}: {v}")

        # Step 4: Verify `underexplored` classifications via academic_search.
        underexplored = [c for c in classified if c.get("classification") == "underexplored"]
        print(f"  [3/4] verifying {len(underexplored)} underexplored gap(s) via academic_search...")
        verified_gaps: list[dict] = []
        for cell in underexplored:
            queries = cell.get("verification_search_queries") or []
            hits_per_query: list[int] = []
            search_snippets: list[dict] = []
            for q in queries[:2]:  # at most 2 queries per gap
                if not q.strip():
                    continue
                try:
                    tool = self.tool_registry.get("academic_search")
                    if tool is None:
                        break
                    raw = tool.execute({"query": q, "limit_per_source": 5})
                    hits = raw.count("doi:") + raw.count("arxiv:") + raw.count("http")
                    hits_per_query.append(hits)
                    search_snippets.append({"query": q, "hits_estimate": hits})
                except Exception as e:  # noqa: BLE001
                    print(f"        [warn] search failed on {q!r}: {type(e).__name__}: {e}")
                    continue
            total_hits = sum(hits_per_query)
            # Heuristic: fewer than ~5 total hits across 2 queries → gap confirmed.
            verified = total_hits < 5 if hits_per_query else False
            if verified:
                verified_gaps.append({
                    **cell,
                    "search_verification": search_snippets,
                    "total_hits_estimate": total_hits,
                })
                print(f"        ✓ {cell.get('method','?')[:30]} × {cell.get('problem','?')[:30]}  (hits≈{total_hits})")
            else:
                print(f"        ✗ {cell.get('method','?')[:30]} × {cell.get('problem','?')[:30]}  (hits≈{total_hits}, likely adjacent_but_covered)")

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
                questions_result = self._call_primary(questions_prompt)
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
                self.journal.enqueue_questions(
                    prefixed,
                    source=f"gap:{scan_short_id}",
                    priority=0.85,
                )
                enqueued_ids.append(f"gap:{scan_short_id}")
                for q in qs:
                    print(f"        + {q[:120]}")
        else:
            print("  [done] no verified gaps — no questions enqueued.")

        summary = (
            f"Scan over {n_entries} entries. Matrix: {len(methods)} methods × "
            f"{len(problems)} problems. "
            f"{len(covered)} cells covered, {len(empty_cells)} empty. "
            f"{len(underexplored)} classified underexplored; "
            f"{len(verified_gaps)} verified by search; "
            f"{sum(len(qs) for qs in gap_questions.values())} questions enqueued."
        )
        print(f"\n  {summary}")

        scan = self._persist_scan(
            methods=methods, problems=problems, covered=covered,
            classified=classified, verified_gaps=verified_gaps,
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
        verified_keys = {((g.get("method") or "").strip(), (g.get("problem") or "").strip())
                         for g in verified_gaps}
        for cell_key, cell_info in class_by_cell.items():
            m, p = cell_key
            is_verified = cell_key in verified_keys
            verification = None
            if is_verified:
                verification = next(
                    (v for v in verified_gaps
                     if (v.get("method") or "").strip() == m
                     and (v.get("problem") or "").strip() == p),
                    None,
                )
            gaps.append({
                "method": m,
                "problem": p,
                "classification": cell_info.get("classification", "?"),
                "reasoning": cell_info.get("reasoning", ""),
                "verification_search_queries": cell_info.get("verification_search_queries", []),
                "verified_empty": is_verified,
                "verification_hits_estimate": (
                    verification.get("total_hits_estimate") if verification else None
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
