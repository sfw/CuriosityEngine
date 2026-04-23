"""Verification (Phase 6) + prediction checks (Phase 7). Cross-model adversarial review."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from models import CrossReference, Insight, Prediction, RegisterEntry  # noqa: F401
from prompts import PREDICTION_CHECK_PROMPT, VERIFY_PROMPT


def _prompt_line(prompt: str) -> str:
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        return ""


def _prompt_choice_with_default(prompt: str, valid: tuple[str, ...], default: str) -> str:
    while True:
        raw = _prompt_line(f"{prompt} [default={default}]: ").strip().lower()
        if not raw:
            return default
        if raw in valid:
            return raw
        print(f"  Please choose one of: {', '.join(valid)}")


class VerificationMixin:
    """Adversarial review of insights, plus falsifiable-prediction lifecycle."""

    @staticmethod
    def _slim_entry_for_register(entry: dict) -> dict:
        return {
            "id": entry.get("id"),
            "question": entry.get("question"),
            "key_takeaways": entry.get("key_takeaways", []),
        }

    def _register_gate(
        self,
        verdict: str,
        confidence: float,
        premises_supported: bool,
        synthesis_findable: bool,
    ) -> tuple[str, str, list[str]]:
        """Apply the register gate.

        Returns (outcome, entry_status, reasons):
          - ('register', 'active', []) — validated + premises + !synthesis + conf>=floor
          - ('hold',     'held',   []) — inconclusive + premises + conf>=held_floor (if enabled)
          - ('reject',   '',       [reasons...]) — anything else
        """
        floor = float(getattr(self.config, "register_confidence_floor", 0.6))
        held_enabled = bool(getattr(self.config, "held_entries_enabled", True))
        held_floor = float(getattr(self.config, "held_confidence_floor", 0.7))

        if (
            verdict == "validated"
            and premises_supported
            and not synthesis_findable
            and confidence >= floor
        ):
            return ("register", "active", [])

        if (
            held_enabled
            and verdict == "inconclusive"
            and premises_supported
            and confidence >= held_floor
        ):
            return ("hold", "held", [])

        reasons: list[str] = []
        if verdict not in ("validated", "inconclusive"):
            reasons.append(f"verdict={verdict}")
        elif verdict == "validated" and confidence < floor:
            reasons.append(f"conf<{floor}")
        elif verdict == "inconclusive":
            if not held_enabled:
                reasons.append("held_entries_disabled")
            elif confidence < held_floor:
                reasons.append(f"held_conf<{held_floor}")
        if not premises_supported:
            reasons.append("premises_unsupported")
        if verdict == "validated" and synthesis_findable:
            reasons.append("synthesis_already_in_literature")
        return ("reject", "", reasons)

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
            tool_list=self._tool_list_block(for_client=self.verifier),
            prior_human_rejections_json=json.dumps(
                self.journal.human_rejection_feedback(), indent=2,
            ),
        )
        server_tools = [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "code_execution_20250825", "name": "code_execution"},
        ]
        result = self._call_verifier_with_tools(
            prompt,
            server_tools=server_tools,
            max_tokens=self.connection.verifier.investigation_max_tokens,
        )

        verdict = (result.get("verdict") or "refuted").strip().lower()
        verified_confidence = float(result.get("verified_confidence", 0.0))
        premises_supported = bool(result.get("premises_supported", True))
        synthesis_findable = bool(result.get("synthesis_findable", False))
        novelty_type = (result.get("novelty_type") or "").strip()
        summary = result.get("verification_summary", "")

        # Inconclusive guardrail: if the verifier hedged `inconclusive` without
        # naming a specific epistemic gap in the summary, downgrade to
        # `challenged`. We look for gap-shaped phrases; absence is the signal.
        if verdict == "inconclusive":
            gap_markers = ("cannot access", "cannot reach", "could not access",
                           "paywalled", "proprietary", "pre-publication", "unpublished",
                           "requires an experiment", "requires experiment",
                           "no public literature", "no meaningful results",
                           "could not find", "unavailable", "behind paywall",
                           "cannot run", "cannot verify via", "not in the public")
            s_lower = (summary or "").lower()
            if not any(m in s_lower for m in gap_markers):
                print("  [guardrail] verdict=inconclusive but summary lacks a named "
                      "epistemic gap — downgrading to challenged.")
                verdict = "challenged"

        # Challenged-hedge guardrail: the verifier frequently returns
        # `challenged` on insights whose decomposition unambiguously says
        # "genuine new synthesis" — novelty_type ∈ {new_synthesis, correction},
        # premises supported, synthesis not findable, confidence clear the
        # register floor. That configuration IS the signature the prompt tells
        # the LLM to mark `validated`. RLHF-trained hedging still pushes models
        # to `challenged` despite the instruction.
        #
        # Rule: if the signature holds, default to UPGRADE to validated. Block
        # the upgrade only when `reasoning_flaws` contains a specific,
        # substantive critique (not a restatement of ingredient-existence,
        # which the prompt explicitly tells the LLM not to count).
        floor = float(getattr(self.config, "register_confidence_floor", 0.6))
        if (
            verdict == "challenged"
            and novelty_type in ("new_synthesis", "correction")
            and premises_supported
            and not synthesis_findable
            and verified_confidence >= floor
        ):
            reasoning_flaws = result.get("reasoning_flaws", []) or []
            # Markers of a REAL synthesis-level flaw — the kind that legitimately
            # justifies `challenged` even on a new-synthesis decomposition.
            # If any flaw matches one of these, respect the verdict.
            #
            # Marker list is pattern-matched case-insensitively against each
            # reasoning_flaw string. Expand when you observe hedge phrasings
            # being wrongly flagged as substantive or vice-versa.
            substantive_flaw_markers = (
                # Inferential leap / claim-doesn't-follow patterns
                "leap", "does not follow", "doesn't follow",
                "does not show", "doesn't show",
                "does not establish", "doesn't establish",
                "does not support", "doesn't support",
                "does not validate", "doesn't validate",
                "not established", "not validated", "not supported",
                "not justified", "without justification",
                "not equivalent",
                # Scope / overclaim / generalization issues
                "generaliz",  # generalize, generalization, overgeneralization
                "overclaim", "overstate", "overstatement",
                "overextend",
                "too strong", "too broad", "too general",
                "beyond the scope", "outside the scope",
                # Transfer / domain-mismatch assumptions
                "assumes transfer", "assumes this transfer",
                "transfers to",  # "assumes this transfers to [new domain]"
                "depends on a",   # "depends on a <setting> that doesn't apply"
                "not shown to transfer",
                # Alternatives undermine "necessary" claims
                "viable alternative", "alternative architecture",
                "alternative mechanism", "alternative approach",
                "other approach",
                # Assumption / evidence issues
                "assumes without", "unsupported assumption",
                "unfounded", "without warrant",
                "insufficient evidence",
                # Conflation / mechanism / causal
                "conflat",  # conflate, conflation
                "no mechanism", "mechanism is missing", "mechanism missing",
                "causal direction", "causal is unclear",
                "reverse causation", "spurious",
                # Sample / scope-of-evidence
                "sample too narrow", "sample size",
                # Contradiction / consistency
                "contradicts", "inconsistent with",
                # Interpretive moves not justified by cited work
                "treated as if", "treated as though",
                "cross-model interpretation",
                "does not prove", "doesn't prove",
            )
            substantive_flaws = [
                fl for fl in reasoning_flaws
                if fl and any(m in fl.lower() for m in substantive_flaw_markers)
            ]
            # Log the decision inputs so the user can spot-check guardrail calls.
            if reasoning_flaws:
                print(f"  [guardrail-check] verifier listed {len(reasoning_flaws)} reasoning_flaw(s):")
                for fl in reasoning_flaws:
                    print(f"    - {str(fl)[:200]}")
            if substantive_flaws:
                print(
                    f"  [guardrail-check] {len(substantive_flaws)} flaw(s) matched substantive markers — "
                    f"respecting `challenged` verdict."
                )
            else:
                print(
                    f"  [guardrail] verdict=challenged but decomposition is unambiguous "
                    f"(novelty={novelty_type}, premises=✓, synthesis_findable=✗) and "
                    f"reasoning_flaws "
                    f"{'is empty' if not reasoning_flaws else 'contains no substantive critique markers'} — "
                    f"upgrading to validated."
                )
                verdict = "validated"

        print(f"  Verdict: {verdict} · novelty={novelty_type or '?'} "
              f"· premises={'✓' if premises_supported else '✗'} "
              f"· synthesis_findable={'✓' if synthesis_findable else '✗'}")
        print(f"  Verified confidence: {verified_confidence:.2f}")
        if summary:
            print(f"  {summary[:200]}...")

        outcome, entry_status, gate_reasons = self._register_gate(
            verdict, verified_confidence, premises_supported, synthesis_findable,
        )
        if outcome == "reject":
            print(f"  Not registered ({', '.join(gate_reasons)}).")
            return None

        supporting_sources: list[str] = []
        seen: set[str] = set()
        for e in supporting:
            for src in e.get("sources", []) or []:
                if src not in seen:
                    supporting_sources.append(src)
                    seen.add(src)

        synthesis_prior_art = result.get("synthesis_prior_art", []) or []
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
            # Legacy fields map to synthesis-level prior art for readers not on the new schema yet.
            prior_art_found=synthesis_findable,
            prior_art_citations=list(synthesis_prior_art),
            contradicting_findings=result.get("contradicting_findings", []) or [],
            reasoning_flaws_considered=result.get("reasoning_flaws", []) or [],
            verification_summary=summary,
            premises_supported=premises_supported,
            premises_support_citations=result.get("premises_support_citations", []) or [],
            synthesis_findable=synthesis_findable,
            synthesis_prior_art=list(synthesis_prior_art),
            novelty_type=novelty_type,
            status=entry_status,
            held_reason=(summary if entry_status == "held" else ""),
            settlement_method=(result.get("settlement_method", "") or "") if entry_status == "held" else "",
            settlement_horizon=(result.get("settlement_horizon", "") or "") if entry_status == "held" else "",
            settlement_triggers=(result.get("settlement_triggers", []) or []) if entry_status == "held" else [],
            open_questions=insight.open_questions,
            counter_arguments=insight.counter_arguments,
        )

        self.journal.add_register_entry(register_entry)
        if entry_status == "held":
            print(f"  HELD: {register_entry.id} (awaiting settlement)")
            if register_entry.settlement_method:
                print(f"    settlement: {register_entry.settlement_method[:120]}")
            if register_entry.settlement_horizon:
                print(f"    horizon:    {register_entry.settlement_horizon}")
        else:
            print(f"  REGISTERED: {register_entry.id} -> {self.config.register_markdown_path}")
            # Predictions only emit for active registration. Held entries carry
            # settlement_triggers instead; those can be turned into Predictions
            # at human-review promotion time.
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
        tools = [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "code_execution_20250825", "name": "code_execution"},
        ]
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
                tool_list=self._tool_list_block(for_client=self.verifier),
            )
            result = self._call_verifier_with_tools(
                prompt,
                server_tools=tools,
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

    def review_register(self):
        """Walk unreviewed register entries and prompt for approve / reject / defer / skip."""
        pending = self.journal.unreviewed_register_entries()
        print(f"\n--- REVIEW REGISTER ({len(pending)} unreviewed) ---")
        if not pending:
            print("  All register entries already reviewed.")
            return

        for i, entry in enumerate(pending, start=1):
            print(f"\n{'='*62}")
            print(f"  Entry {i}/{len(pending)}  id={entry.get('id')}")
            print(f"{'='*62}")
            print(f"Title:       {entry.get('title', '')}")
            print(f"Registered:  {entry.get('timestamp', '')}")
            print(f"Verdict:     {entry.get('verdict', '')}  "
                  f"(conf {entry.get('verified_confidence', 0.0):.2f})")
            is_held = entry.get("status") == "held"
            if entry.get("status") and entry["status"] != "active":
                print(f"Lifecycle:   {entry['status']}")
            desc = entry.get("description", "")
            if desc:
                print(f"\n{desc}")
            motivation = entry.get("motivation", "")
            if motivation:
                print(f"\nMotivation:  {motivation}")
            summary = entry.get("verification_summary", "")
            if summary:
                print(f"\nVerification summary: {summary}")
            if is_held:
                sm = entry.get("settlement_method", "")
                sh = entry.get("settlement_horizon", "")
                st = entry.get("settlement_triggers", []) or []
                if sm:
                    print(f"\nSettlement method:   {sm}")
                if sh:
                    print(f"Settlement horizon:  {sh}")
                if st:
                    print("Settlement triggers:")
                    for t in st:
                        print(f"  - {t}")
            citations = entry.get("prior_art_citations") or []
            if citations:
                print("\nPrior art cited by verifier:")
                for c in citations[:5]:
                    print(f"  - {c}")
            sources = entry.get("supporting_sources") or []
            if sources:
                print(f"\n{len(sources)} supporting source(s); first few:")
                for s in sources[:5]:
                    print(f"  - {s}")

            if is_held:
                action = _prompt_choice_with_default(
                    "\nAction: [a]pprove  [p]romote-to-active  [c]onvert-triggers-to-predictions  "
                    "[r]eject  [d]efer  [s]kip  [q]uit",
                    valid=("a", "p", "c", "r", "d", "s", "q"),
                    default="s",
                )
            else:
                action = _prompt_choice_with_default(
                    "\nAction: [a]pprove  [r]eject  [d]efer  [s]kip  [q]uit",
                    valid=("a", "r", "d", "s", "q"),
                    default="s",
                )
            if action == "q":
                print("\nAborted review; progress saved.")
                return
            if action == "s":
                print("  Skipped.")
                continue

            notes = _prompt_line("  Optional notes: ")
            reviewer = _prompt_line("  Reviewer name (optional): ")

            if action == "a":
                self.journal.update_register_entry_review(
                    entry.get("id", ""),
                    status="approved",
                    notes=notes,
                    reviewer=reviewer,
                )
                print("  Marked approved.")
            elif action == "r":
                rejection_reason = _prompt_line("  Rejection reason (required): ").strip()
                if not rejection_reason:
                    print("  Rejection requires a reason; skipping this entry.")
                    continue
                self.journal.update_register_entry_review(
                    entry.get("id", ""),
                    status="rejected",
                    notes=notes,
                    rejection_reason=rejection_reason,
                    reviewer=reviewer,
                )
                print("  Marked rejected. Reason will inform future verifications.")
            elif action == "d":
                self.journal.update_register_entry_review(
                    entry.get("id", ""),
                    status="deferred",
                    notes=notes,
                    reviewer=reviewer,
                )
                print("  Deferred. You can revisit later.")
            elif action == "p":
                # Promote held → active. Optionally convert settlement_triggers to Predictions.
                promoted = self.journal.promote_register_entry(
                    entry.get("id", ""),
                    promoted_by=f"human:{reviewer or 'unknown'}",
                )
                if not promoted:
                    print("  Could not promote (entry is not in held state).")
                else:
                    print("  Promoted to active.")
                    triggers = entry.get("settlement_triggers", []) or []
                    if triggers:
                        want = _prompt_choice_with_default(
                            f"  Attach {len(triggers)} settlement trigger(s) as Predictions? [y/n]",
                            valid=("y", "n"), default="n",
                        )
                        if want == "y":
                            horizon = (entry.get("settlement_horizon", "") or "").strip()
                            self._persist_triggers_as_predictions(
                                triggers, entry.get("id", ""), default_target_date=horizon,
                            )
                    self.journal.update_register_entry_review(
                        entry.get("id", ""),
                        status="approved",
                        notes=notes or "promoted from held",
                        reviewer=reviewer,
                    )
            elif action == "c":
                triggers = entry.get("settlement_triggers", []) or []
                if not triggers:
                    print("  No settlement triggers to convert.")
                else:
                    horizon = (entry.get("settlement_horizon", "") or "").strip()
                    self._persist_triggers_as_predictions(
                        triggers, entry.get("id", ""), default_target_date=horizon,
                    )
                    print(f"  Converted {len(triggers)} trigger(s) into predictions; entry stays held.")

        print("\nReview complete.")

    def _persist_triggers_as_predictions(
        self,
        triggers: list[str],
        register_entry_id: str,
        *,
        default_target_date: str = "",
    ) -> int:
        """Turn settlement_triggers into Prediction records so --check-predictions
        can auto-settle the held entry later. Each trigger becomes one prediction
        whose falsifiable_condition is the trigger text.
        """
        kept = 0
        for t in triggers:
            trigger = (t or "").strip()
            if not trigger:
                continue
            prediction = Prediction(
                id=f"p-{uuid4().hex[:8]}",
                register_entry_id=register_entry_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                target_date=default_target_date,
                claim=f"Settlement trigger for held entry: {trigger}",
                falsifiable_condition=trigger,
                check_method="Reuse the settlement_method recorded on the held register entry.",
            )
            self.journal.add_prediction(prediction)
            kept += 1
            print(f"    PREDICTION from trigger: {prediction.id} (target {prediction.target_date or 'tbd'})")
        return kept

    def _reconcile_entry_status(self, register_entry_id: str):
        """Update a register entry's status based on the current verdicts of its predictions.

        Held entries get promoted to `active` when all attached predictions are
        confirmed (promotion is automatic from reality, not from a human).
        """
        predictions = self.journal.predictions_for_entry(register_entry_id)
        if not predictions:
            return
        statuses = [p.get("status") for p in predictions]
        entry = next((e for e in self.journal.register if e.get("id") == register_entry_id), None)
        if entry is None:
            return
        is_held = entry.get("status") == "held"

        if "refuted" in statuses:
            self.journal.update_register_entry_status(register_entry_id, "challenged_by_prediction")
        elif all(s == "confirmed" for s in statuses):
            if is_held:
                pred_id = next((p.get("id") for p in predictions if p.get("id")), "?")
                self.journal.promote_register_entry(
                    register_entry_id, promoted_by=f"prediction:{pred_id}",
                )
                # After promotion the status is `active`; mark as validated_by_prediction.
                self.journal.update_register_entry_status(register_entry_id, "validated_by_prediction")
            else:
                self.journal.update_register_entry_status(register_entry_id, "validated_by_prediction")
        # Otherwise leave as-is.

    def reverify_unregistered_insights(
        self,
        *,
        only_ids: Optional[list[str]] = None,
    ) -> dict:
        """Re-run verification on insights that don't already have a register entry.

        Used to elevate previously-rejected insights under newer rules (e.g. the
        premises-vs-synthesis split + `inconclusive`/held pipeline). Each
        verification that passes the gate creates a new RegisterEntry.
        """
        from models import CrossReference as _CR, Insight as _I

        already_registered = self.journal.registered_insight_ids()
        candidates: list[dict] = []
        for i in self.journal.insights:
            iid = i.get("id")
            if not iid or iid in already_registered:
                continue
            if only_ids and iid not in only_ids:
                continue
            candidates.append(i)

        print(f"\n--- REVERIFYING {len(candidates)} unregistered insight(s) ---")
        stats = {"registered": 0, "held": 0, "rejected": 0, "errors": 0}
        xref_by_id = {x.get("id"): x for x in self.journal.cross_references}

        for insight_dict in candidates:
            iid = insight_dict.get("id")
            # Reconstruct the Insight dataclass. Use only fields the dataclass accepts.
            insight_fields = set(_I.__dataclass_fields__)
            insight = _I(**{k: v for k, v in insight_dict.items() if k in insight_fields})

            # The supporting xref is stored in supporting_evidence as an id prefixed `x-`.
            xref_id = next(
                (sid for sid in (insight.supporting_evidence or []) if str(sid).startswith("x-")),
                None,
            )
            if not xref_id or xref_id not in xref_by_id:
                print(f"  [skip] {iid}: no reachable supporting xref")
                stats["errors"] += 1
                continue
            xref_fields = set(_CR.__dataclass_fields__)
            xref = _CR(**{k: v for k, v in xref_by_id[xref_id].items() if k in xref_fields})

            try:
                print(f"\n  --- reverifying {iid}: {insight.title[:90]}")
                register_entry = self.verify_insight(insight, xref)
            except Exception as e:  # noqa: BLE001
                print(f"  [error] {iid}: {type(e).__name__}: {e}")
                stats["errors"] += 1
                continue

            if register_entry is None:
                stats["rejected"] += 1
            elif register_entry.status == "held":
                stats["held"] += 1
            else:
                stats["registered"] += 1

        print(
            f"\nReverify complete: registered={stats['registered']}  held={stats['held']}  "
            f"rejected={stats['rejected']}  errors={stats['errors']}"
        )
        return stats
