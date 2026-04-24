"""Persistent research journal stored as JSON."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models import CrossReference, Insight, JournalEntry, Prediction, RegisterEntry
from register import render_markdown


class Journal:
    """Append-mostly research journal backed by a single JSON file.

    Thread-safety: every mutating method calls `self.save()`, which serializes
    through `self._save_lock` and writes atomically via temp-file + rename.
    Multiple threads can safely call `add_entry`, `add_insight`, etc. concurrently
    — writes are ordered by lock acquisition, and the on-disk state is always
    a complete snapshot (never a partial JSON write).
    """

    def __init__(self, path: str, register_markdown_path: Optional[str] = None):
        self.path = path
        self.register_markdown_path = register_markdown_path
        self.entries: list[dict] = []
        self.cross_references: list[dict] = []
        self.insights: list[dict] = []
        self.register: list[dict] = []
        self.predictions: list[dict] = []
        self.question_queue: list[dict] = []
        self.focus: str = ""                       # user-set investigation focus
        self.last_domain: str = ""                 # last domain used on a run against this journal
        self.embeddings: dict[str, list[float]] = {}  # entry_id -> dense vector
        self.coverage_scans: list[dict] = []       # negative-space gap scans over time
        self._save_lock = threading.Lock()
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
                self.entries = data.get("entries", [])
                self.cross_references = data.get("cross_references", [])
                self.insights = data.get("insights", [])
                self.register = data.get("register", [])
                self.predictions = data.get("predictions", [])
                self.question_queue = data.get("question_queue", [])
                self.focus = str(data.get("focus", ""))
                self.last_domain = str(data.get("last_domain", ""))
                self.embeddings = dict(data.get("embeddings", {}))
                self.coverage_scans = list(data.get("coverage_scans", []))

    def save(self):
        """Serialize the full journal state.

        Thread-safe + atomic: the lock serializes concurrent writers, and the
        write goes to a temp file + os.replace so readers never see a partial
        JSON document (critical now that investigation/synth/verify may fan
        out across threads — see Phase 1 parallelism in engine/core.py).
        """
        payload = {
            "entries": self.entries,
            "cross_references": self.cross_references,
            "insights": self.insights,
            "register": self.register,
            "predictions": self.predictions,
            "question_queue": self.question_queue,
            "focus": self.focus,
            "last_domain": self.last_domain,
            "embeddings": self.embeddings,
            "coverage_scans": self.coverage_scans,
            "metadata": {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "total_entries": len(self.entries),
                "total_insights": len(self.insights),
                "total_register_entries": len(self.register),
                "total_predictions": len(self.predictions),
            },
        }
        serialized = json.dumps(payload, indent=2)
        with self._save_lock:
            target = Path(self.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file in the same directory so os.replace is atomic
            # on POSIX (same filesystem). NamedTemporaryFile(delete=False) keeps
            # the file around after close so we can rename it.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(target.parent),
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as tf:
                tf.write(serialized)
                tf.flush()
                os.fsync(tf.fileno())
                tmp_path = tf.name
            os.replace(tmp_path, self.path)

    def add_entry(self, entry: JournalEntry):
        self.entries.append(asdict(entry))
        self.save()

    def add_cross_reference(self, xref: CrossReference):
        self.cross_references.append(asdict(xref))
        self.save()

    def add_insight(self, insight: Insight):
        self.insights.append(asdict(insight))
        self.save()

    def add_register_entry(self, entry: RegisterEntry):
        """Append a validated insight to the register. Also rewrites register.md."""
        self.register.append(asdict(entry))
        self.save()
        self._write_register_markdown()

    def _write_register_markdown(self):
        if not self.register_markdown_path:
            return
        md = render_markdown(self.register, self.predictions)
        path = Path(self.register_markdown_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md)

    def add_prediction(self, prediction: Prediction):
        self.predictions.append(asdict(prediction))
        self.save()
        self._write_register_markdown()

    def update_prediction(self, prediction_id: str, *, status: str, review_entry: dict):
        """Record a check result and update status."""
        for p in self.predictions:
            if p.get("id") == prediction_id:
                p["status"] = status
                p["last_checked_at"] = review_entry.get("checked_at", datetime.now(timezone.utc).isoformat())
                p.setdefault("review_log", []).append(review_entry)
                self.save()
                self._write_register_markdown()
                return

    def update_register_entry_status(self, register_entry_id: str, status: str):
        for e in self.register:
            if e.get("id") == register_entry_id:
                e["status"] = status
                self.save()
                self._write_register_markdown()
                return

    def append_register_reverification(self, register_entry_id: str, log_entry: dict) -> bool:
        """Append a re-verification pass to a register entry's reverification_log.

        The entry's original verdict fields are NEVER overwritten — this is
        strictly an append-only audit trail so we can compare old vs new
        verdicts produced under updated verification rules.
        """
        for e in self.register:
            if e.get("id") == register_entry_id:
                if "reverification_log" not in e or not isinstance(e["reverification_log"], list):
                    e["reverification_log"] = []
                e["reverification_log"].append(dict(log_entry))
                self.save()
                return True
        return False

    def promote_register_entry(self, register_entry_id: str, *, promoted_by: str):
        """Promote a held register entry to `active`. Records who/when did it."""
        for e in self.register:
            if e.get("id") == register_entry_id:
                if e.get("status") != "held":
                    return False
                e["status"] = "active"
                e["promoted_at"] = datetime.now(timezone.utc).isoformat()
                e["promoted_by"] = promoted_by
                self.save()
                self._write_register_markdown()
                return True
        return False

    def held_register_entries(self) -> list[dict]:
        return [e for e in self.register if e.get("status") == "held"]

    def registered_insight_ids(self) -> set[str]:
        """Insight ids that already have a corresponding register entry (any status)."""
        return {e.get("insight_id") for e in self.register if e.get("insight_id")}

    def add_coverage_scan(self, scan: dict):
        """Append a negative-space gap scan. Scan shape:
        {id, timestamp, journal_size_at_scan, methods, problems, cells, gaps, summary}
        """
        self.coverage_scans.append(dict(scan))
        self.save()

    def latest_coverage_scan(self) -> Optional[dict]:
        return self.coverage_scans[-1] if self.coverage_scans else None

    def update_register_entry_review(
        self,
        register_entry_id: str,
        *,
        status: str,
        notes: str = "",
        rejection_reason: str = "",
        reviewer: str = "",
    ):
        """Record a human review outcome on a register entry."""
        for e in self.register:
            if e.get("id") == register_entry_id:
                e["human_review_status"] = status
                e["human_review_notes"] = notes
                e["human_rejection_reason"] = rejection_reason
                e["human_reviewer"] = reviewer
                e["human_review_at"] = datetime.now(timezone.utc).isoformat()
                self.save()
                self._write_register_markdown()
                return

    def unreviewed_register_entries(self) -> list[dict]:
        return [e for e in self.register if (e.get("human_review_status") or "unreviewed") == "unreviewed"]

    def human_rejection_feedback(self, limit: int = 20) -> list[dict]:
        """Latest human rejections; used to inject prior-human-feedback into verify prompts."""
        rejections = [
            {
                "title": e.get("title", ""),
                "rejection_reason": e.get("human_rejection_reason", ""),
                "notes": e.get("human_review_notes", ""),
            }
            for e in self.register
            if e.get("human_review_status") == "rejected"
            and (e.get("human_rejection_reason") or e.get("human_review_notes"))
        ]
        return rejections[-limit:]

    def due_predictions(self, *, include_overdue: bool = True) -> list[dict]:
        """Return pending predictions whose target_date has arrived."""
        now = datetime.now(timezone.utc).date().isoformat()
        out: list[dict] = []
        for p in self.predictions:
            if p.get("status") != "pending":
                continue
            target = p.get("target_date", "")
            if include_overdue and target and target <= now:
                out.append(p)
            elif target == now:
                out.append(p)
        return out

    def predictions_for_entry(self, register_entry_id: str) -> list[dict]:
        return [p for p in self.predictions if p.get("register_entry_id") == register_entry_id]

    def get_recent_entries(self, n: int = 10) -> list[dict]:
        return self.entries[-n:]

    def get_all_domain_tags(self) -> list[str]:
        tags: set[str] = set()
        for entry in self.entries:
            tags.update(entry.get("domain_tags", []))
        return sorted(tags)

    def get_high_surprise_entries(self, threshold: float = 0.6) -> list[dict]:
        return [e for e in self.entries if e.get("surprise_delta", 0) >= threshold]

    def enqueue_questions(
        self,
        questions: list[str],
        source: str,
        priority: float = 0.5,
        *,
        floor: Optional[float] = None,
    ):
        """Push emergent questions onto the queue for future cycles.

        `priority` is the expected-reward signal from the source (e.g. the parent
        entry's surprise_delta, or the parent xref's novelty_score). Higher is
        better. Questions are popped via source-round-robin + in-source priority
        (see `pop_queued_questions`); human-sourced questions still jump the
        queue via `pop_questions_by_source`.

        `floor`: if provided, questions with priority below this are dropped at
        enqueue time. Human-sourced questions bypass the floor — human intent
        overrides the autoscreen. Below-floor drops are logged so the user can
        tell why a would-be queue entry disappeared.
        """
        existing = {q.get("question") for q in self.question_queue}
        dropped = 0
        source_is_human = source.startswith("human") if source else False
        for q in questions:
            q_text = (q or "").strip()
            if not q_text or q_text in existing:
                continue
            p = float(max(0.0, min(1.0, priority)))
            if floor is not None and p < float(floor) and not source_is_human:
                dropped += 1
                continue
            self.question_queue.append({
                "question": q_text,
                "source": source,
                "added_at": datetime.now(timezone.utc).isoformat(),
                "priority": p,
            })
            existing.add(q_text)
        if dropped:
            print(
                f"  [queue floor] dropped {dropped} question(s) from source={source!r} "
                f"with priority < {floor} (human sources bypass the floor)."
            )
        self.save()

    # Source-family mapping for round-robin scheduling. Each queued question's
    # source string starts with one of these prefixes; we bucket by prefix and
    # rotate across non-empty buckets when popping. Keeps high-frequency sources
    # (xref) from starving low-frequency ones (gap, analog, assumption).
    _SOURCE_BUCKETS = ("human", "xref", "gap", "analog", "assumption", "entry", "fresh", "other")

    @classmethod
    def _bucket_for_source(cls, source: str) -> str:
        """Map a source string to its round-robin bucket."""
        if not source:
            return "fresh"
        prefix = source.split(":", 1)[0] if ":" in source else source
        if prefix in cls._SOURCE_BUCKETS:
            return prefix
        return "other"

    def pop_queued_questions(self, n: int) -> list[dict]:
        """Take up to n queued questions via source-round-robin + in-source priority.

        Rotation order: human → xref → gap → analog → assumption → entry → fresh → other.
        Each round pulls ONE highest-priority item from each non-empty bucket, then
        loops. This guarantees sources with lower priority ceilings (e.g. gap/analog
        at 0.85) still get visited when a high-ceiling source (xref at 0.87+) would
        otherwise starve them.

        Equivalent to: partition the queue by bucket; sort each partition by
        priority desc (FIFO tiebreak); pop round-robin until budget exhausted
        or all partitions empty.
        """
        if n <= 0 or not self.question_queue:
            return []
        # Partition into buckets. Preserve insertion index for FIFO tiebreak.
        buckets: dict[str, list[tuple[int, dict]]] = {b: [] for b in self._SOURCE_BUCKETS}
        for idx, q in enumerate(self.question_queue):
            buckets[self._bucket_for_source(q.get("source") or "")].append((idx, q))
        # Sort each bucket: priority desc, FIFO by index for ties.
        for key in buckets:
            buckets[key].sort(key=lambda it: (-float(it[1].get("priority", 0.5)), it[0]))

        popped: list[dict] = []
        popped_indices: set[int] = set()
        # Rotate across non-empty buckets in fixed order; each round pulls one
        # top-priority item per bucket. Stop when budget exhausted or all empty.
        while len(popped) < n:
            any_taken_this_round = False
            for bucket_name in self._SOURCE_BUCKETS:
                if len(popped) >= n:
                    break
                if buckets[bucket_name]:
                    idx, q = buckets[bucket_name].pop(0)
                    popped.append(q)
                    popped_indices.add(idx)
                    any_taken_this_round = True
            if not any_taken_this_round:
                break
        self.question_queue = [
            q for i, q in enumerate(self.question_queue) if i not in popped_indices
        ]
        self.save()
        return popped

    def set_embedding(self, entry_id: str, vector: list[float]):
        self.embeddings[entry_id] = list(vector)
        self.save()

    def missing_embedding_entry_ids(self) -> list[str]:
        return [e["id"] for e in self.entries if e.get("id") and e["id"] not in self.embeddings]

    def set_focus(self, focus: str):
        self.focus = focus.strip()
        self.save()

    def clear_focus(self):
        self.focus = ""
        self.save()

    def set_last_domain(self, domain: str):
        d = (domain or "").strip()
        if d and d != self.last_domain:
            self.last_domain = d
            self.save()

    def questions_by_source(self, source_prefix: str) -> list[dict]:
        return [q for q in self.question_queue if (q.get("source") or "").startswith(source_prefix)]

    def pop_questions_by_source(self, source_prefix: str, *, limit: int) -> list[dict]:
        """Remove and return up to `limit` queued questions whose source starts with prefix."""
        if limit <= 0:
            return []
        matches: list[dict] = []
        remaining: list[dict] = []
        for q in self.question_queue:
            if len(matches) < limit and (q.get("source") or "").startswith(source_prefix):
                matches.append(q)
            else:
                remaining.append(q)
        if matches:
            self.question_queue = remaining
            self.save()
        return matches

    def clear_question_queue(self, *, source_prefix: Optional[str] = None) -> int:
        before = len(self.question_queue)
        if source_prefix:
            self.question_queue = [
                q for q in self.question_queue
                if not (q.get("source") or "").startswith(source_prefix)
            ]
        else:
            self.question_queue = []
        removed = before - len(self.question_queue)
        if removed:
            self.save()
        return removed

    def annotate_connection(self, entry_id: str, xref_id: str):
        """Record that a cross-reference touches an entry."""
        for entry in self.entries:
            if entry.get("id") == entry_id:
                connections = entry.setdefault("connections_to", [])
                if xref_id not in connections:
                    connections.append(xref_id)
                return
