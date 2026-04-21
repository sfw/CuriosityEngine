"""Persistent research journal stored as JSON."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models import CrossReference, Insight, JournalEntry, Prediction, RegisterEntry
from register import render_markdown


class Journal:
    """Append-mostly research journal backed by a single JSON file."""

    def __init__(self, path: str, register_markdown_path: Optional[str] = None):
        self.path = path
        self.register_markdown_path = register_markdown_path
        self.entries: list[dict] = []
        self.cross_references: list[dict] = []
        self.insights: list[dict] = []
        self.register: list[dict] = []
        self.predictions: list[dict] = []
        self.question_queue: list[dict] = []
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

    def save(self):
        with open(self.path, "w") as f:
            json.dump({
                "entries": self.entries,
                "cross_references": self.cross_references,
                "insights": self.insights,
                "register": self.register,
                "predictions": self.predictions,
                "question_queue": self.question_queue,
                "metadata": {
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "total_entries": len(self.entries),
                    "total_insights": len(self.insights),
                    "total_register_entries": len(self.register),
                    "total_predictions": len(self.predictions),
                },
            }, f, indent=2)

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

    def enqueue_questions(self, questions: list[str], source: str):
        """Push emergent questions onto the queue for future cycles."""
        existing = {q.get("question") for q in self.question_queue}
        for q in questions:
            q_text = (q or "").strip()
            if not q_text or q_text in existing:
                continue
            self.question_queue.append({
                "question": q_text,
                "source": source,
                "added_at": datetime.now(timezone.utc).isoformat(),
            })
            existing.add(q_text)
        self.save()

    def pop_queued_questions(self, n: int) -> list[dict]:
        """Take up to n queued questions off the front of the queue."""
        if n <= 0 or not self.question_queue:
            return []
        popped = self.question_queue[:n]
        self.question_queue = self.question_queue[n:]
        self.save()
        return popped

    def annotate_connection(self, entry_id: str, xref_id: str):
        """Record that a cross-reference touches an entry."""
        for entry in self.entries:
            if entry.get("id") == entry_id:
                connections = entry.setdefault("connections_to", [])
                if xref_id not in connections:
                    connections.append(xref_id)
                return
