"""Verification (Phase 6) + prediction checks (Phase 7). Cross-model adversarial review."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from models import CrossReference, Insight, Prediction, RegisterEntry
from prompts import PREDICTION_CHECK_PROMPT, VERIFY_PROMPT


class VerificationMixin:
    """Adversarial review of insights, plus falsifiable-prediction lifecycle."""

    @staticmethod
    def _slim_entry_for_register(entry: dict) -> dict:
        return {
            "id": entry.get("id"),
            "question": entry.get("question"),
            "key_takeaways": entry.get("key_takeaways", []),
        }

    def verify_insight(
        self,
        insight: Insight,
        xref: CrossReference,
    ) -> Optional[RegisterEntry]:
        """Adversarially verify an insight. Return a RegisterEntry only if it passes."""
        print("\n--- VERIFYING INSIGHT ---")
        print(f"  Target: {insight.title}")

        supporting = [e for e in self.journal.entries if e["id"] in xref.source_entries]
        slim_supporting = [self._slim_entry_for_register(e) for e in supporting]

        prompt = VERIFY_PROMPT.format(
            insight_json=json.dumps(asdict(insight), indent=2),
            xref_json=json.dumps(asdict(xref), indent=2),
            supporting_entries_json=json.dumps(slim_supporting, indent=2),
        )
        tools = [{"type": "web_search_20250305", "name": "web_search"}]
        result = self._call_verifier(
            prompt,
            tools=tools,
            max_tokens=self.connection.verifier.investigation_max_tokens,
        )

        verdict = result.get("verdict", "refuted")
        verified_confidence = float(result.get("verified_confidence", 0.0))
        print(f"  Verdict: {verdict}")
        print(f"  Verified confidence: {verified_confidence:.2f}")
        summary = result.get("verification_summary", "")
        if summary:
            print(f"  {summary[:200]}...")

        floor = self.config.register_confidence_floor
        if verdict != "validated" or verified_confidence < floor:
            print(f"  Not registered (verdict={verdict}, floor={floor}).")
            return None

        supporting_sources: list[str] = []
        seen: set[str] = set()
        for e in supporting:
            for src in e.get("sources", []) or []:
                if src not in seen:
                    supporting_sources.append(src)
                    seen.add(src)

        register_entry = RegisterEntry(
            id=f"r-{uuid4().hex[:8]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            insight_id=insight.id,
            title=insight.title,
            description=insight.description,
            supporting_xref_id=xref.id,
            supporting_entry_ids=list(xref.source_entries),
            supporting_entry_summaries=slim_supporting,
            supporting_sources=supporting_sources,
            motivation=result.get("motivation", ""),
            implications=insight.implications,
            verdict=verdict,
            verified_confidence=verified_confidence,
            prior_art_found=bool(result.get("prior_art_found", False)),
            prior_art_citations=result.get("prior_art_citations", []) or [],
            contradicting_findings=result.get("contradicting_findings", []) or [],
            reasoning_flaws_considered=result.get("reasoning_flaws", []) or [],
            verification_summary=summary,
            open_questions=insight.open_questions,
            counter_arguments=insight.counter_arguments,
        )

        self.journal.add_register_entry(register_entry)
        print(f"  REGISTERED: {register_entry.id} -> {self.config.register_markdown_path}")

        self._persist_predictions(result.get("predictions", []) or [], register_entry.id)
        return register_entry

    def _persist_predictions(self, raw_predictions: list[dict], register_entry_id: str):
        kept = 0
        for p in raw_predictions:
            claim = (p.get("claim") or "").strip()
            condition = (p.get("falsifiable_condition") or "").strip()
            method = (p.get("check_method") or "").strip()
            target = (p.get("target_date") or "").strip()
            if not claim or not condition or not target:
                continue
            prediction = Prediction(
                id=f"p-{uuid4().hex[:8]}",
                register_entry_id=register_entry_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                target_date=target,
                claim=claim,
                falsifiable_condition=condition,
                check_method=method,
            )
            self.journal.add_prediction(prediction)
            kept += 1
            print(f"  PREDICTION registered: {prediction.id} (target {prediction.target_date})")
        if kept == 0:
            print("  No falsifiable predictions emitted for this insight.")

    def check_predictions(self, *, all_pending: bool = False) -> list[dict]:
        """Check predictions whose target_date has arrived (or all pending if all_pending=True).

        Uses the verifier client with web_search enabled. Updates prediction status and,
        when a register entry's predictions have resolved, updates the entry status.
        """
        print("\n--- CHECKING PREDICTIONS ---")
        if all_pending:
            pending = [p for p in self.journal.predictions if p.get("status") == "pending"]
        else:
            pending = self.journal.due_predictions(include_overdue=True)

        if not pending:
            print("  No predictions due for review.")
            return []

        entry_by_id = {e.get("id"): e for e in self.journal.register}
        updated: list[dict] = []
        tools = [{"type": "web_search_20250305", "name": "web_search"}]
        today = datetime.now(timezone.utc).date().isoformat()

        for prediction in pending:
            entry = entry_by_id.get(prediction.get("register_entry_id"))
            if entry is None:
                print(f"  Skipping {prediction.get('id')}: parent register entry not found.")
                continue

            print(f"\n  Checking {prediction.get('id')} (target {prediction.get('target_date')}):")
            print(f"    {prediction.get('claim', '')[:110]}")

            prompt = PREDICTION_CHECK_PROMPT.format(
                insight_title=entry.get("title", ""),
                insight_description=entry.get("description", ""),
                claim=prediction.get("claim", ""),
                falsifiable_condition=prediction.get("falsifiable_condition", ""),
                check_method=prediction.get("check_method", ""),
                created_at=prediction.get("created_at", ""),
                target_date=prediction.get("target_date", ""),
                today=today,
            )
            result = self._call_verifier(
                prompt,
                tools=tools,
                max_tokens=self.connection.verifier.investigation_max_tokens,
            )

            verdict = (result.get("verdict") or "inconclusive").strip().lower()
            if verdict not in ("confirmed", "refuted", "inconclusive", "expired"):
                verdict = "inconclusive"

            review_entry = {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "verdict": verdict,
                "reasoning": result.get("reasoning", ""),
                "sources": result.get("sources", []) or [],
            }
            self.journal.update_prediction(prediction["id"], status=verdict, review_entry=review_entry)
            updated.append({"prediction_id": prediction["id"], "verdict": verdict})
            print(f"    Verdict: {verdict}")

            self._reconcile_entry_status(entry.get("id"))

        print(f"\n  {len(updated)} prediction(s) reviewed.")
        return updated

    def _reconcile_entry_status(self, register_entry_id: str):
        """Update a register entry's status based on the current verdicts of its predictions."""
        predictions = self.journal.predictions_for_entry(register_entry_id)
        if not predictions:
            return
        statuses = [p.get("status") for p in predictions]
        if "refuted" in statuses:
            self.journal.update_register_entry_status(register_entry_id, "challenged_by_prediction")
        elif all(s == "confirmed" for s in statuses):
            self.journal.update_register_entry_status(register_entry_id, "validated_by_prediction")
        # Otherwise leave as-is (active) — some confirmed, some pending, some inconclusive.
