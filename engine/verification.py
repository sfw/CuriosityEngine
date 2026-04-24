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
        *,
        novelty_type: str = "",
        peer_has_differentiators: bool = False,
    ) -> tuple[str, str, list[str]]:
        """Apply the register gate.

        Three valid registration paths:
          1. new_synthesis / correction: validated + premises + !synthesis_findable + conf>=floor
          2. extension: validated + premises + peer_has_differentiators + conf>=floor
             (synthesis_findable may be True — central move IS findable, that's the
             definition of extension. What justifies registration is that the peer
             system differs on substantive axes.)
          3. inconclusive → held: held_enabled + premises + conf>=held_floor

        Returns (outcome, entry_status, reasons).
        """
        floor = float(getattr(self.config, "register_confidence_floor", 0.6))
        held_enabled = bool(getattr(self.config, "held_entries_enabled", True))
        held_floor = float(getattr(self.config, "held_confidence_floor", 0.7))

        # Path 1: new_synthesis / correction (composite not in literature).
        if (
            verdict == "validated"
            and premises_supported
            and not synthesis_findable
            and confidence >= floor
            and novelty_type in ("", "new_synthesis", "correction")
        ):
            return ("register", "active", [])

        # Path 2: extension with substantive differentiators. Central move
        # may be in literature (synthesis_findable=True) — what validates the
        # registration is the explicit peer-differentiator set.
        if (
            verdict == "validated"
            and novelty_type == "extension"
            and premises_supported
            and peer_has_differentiators
            and confidence >= floor
        ):
            return ("register", "active", [])

        # Path 3: inconclusive → held.
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
        if verdict == "validated" and synthesis_findable and novelty_type != "extension":
            reasons.append("synthesis_already_in_literature")
        if verdict == "validated" and novelty_type == "extension" and not peer_has_differentiators:
            reasons.append("extension_without_differentiators")
        if novelty_type == "restatement":
            reasons.append("restatement")
        if novelty_type == "unsupported":
            reasons.append("unsupported_premises")
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
        # Capture the full tool-call trace so we can persist it on the register
        # entry — makes after-the-fact audits of verifier misses ("why didn't
        # this search find X?") possible without re-running the whole pass.
        tool_trace: list[dict] = []
        result = self._call_verifier_with_tools(
            prompt,
            server_tools=server_tools,
            max_tokens=self.connection.verifier.investigation_max_tokens,
            trace=tool_trace,
        )

        verdict = (result.get("verdict") or "refuted").strip().lower()
        # Preserve the LLM's original verdict for the confidence-drop audit:
        # when a guard downgrades validated→challenged / inconclusive→challenged,
        # we want to reflect that downgrade in the stored confidence rather than
        # trusting the LLM's hedged-but-still-high number.
        llm_returned_verdict = verdict
        verified_confidence = float(result.get("verified_confidence", 0.0))
        premises_supported = bool(result.get("premises_supported", True))
        synthesis_findable = bool(result.get("synthesis_findable", False))
        novelty_type = (result.get("novelty_type") or "").strip()
        summary = result.get("verification_summary", "")

        # Phase-structured fields (new verifier schema).
        central_architectural_move = (result.get("central_architectural_move") or "").strip()
        central_move_prior_art = result.get("central_move_prior_art", []) or []
        functional_decomposition = result.get("functional_decomposition", []) or []
        closest_peer_system = result.get("closest_peer_system") or {}
        skeptic_probe = result.get("skeptic_probe") or {}

        # Phase-1 guard: if the central architectural move is already published
        # (central_move_prior_art non-empty with substantive entries), downgrade
        # novelty_type from new_synthesis → extension. This is the guard that
        # catches the co-scientist-class failure mode: a headline move matches
        # a known system, but auxiliary refinements let the verifier call it
        # "new_synthesis" because the full composite isn't findable.
        substantive_phase1 = [
            c for c in central_move_prior_art
            if isinstance(c, str) and len(c.strip()) > 20
            and not c.lower().startswith(("no ", "none", "empty", "searched"))
        ]
        if novelty_type == "new_synthesis" and substantive_phase1:
            print(f"  [phase-1 guard] central move has {len(substantive_phase1)} substantive "
                  f"prior-art hit(s) — downgrading novelty_type new_synthesis → extension.")
            novelty_type = "extension"
            # Reflect in the synthesis_findable axis: the central move IS findable
            # even if the full composite is not. This keeps the register_gate honest.
            synthesis_findable = True

        # Skeptic-probe guard: if the final skeptic smell test surfaced
        # disqualifying prior art, respect it. The verifier is instructed not
        # to rationalise past it, but we enforce it mechanically too.
        if skeptic_probe.get("disqualifies"):
            print(f"  [skeptic-probe] verifier's own final smell test disqualified the claim: "
                  f"{str(skeptic_probe.get('query',''))[:120]}")
            if verdict == "validated":
                verdict = "challenged"
            synthesis_findable = True
            if novelty_type == "new_synthesis":
                novelty_type = "extension"

        # Peer-system guard: if a complete peer system was identified with
        # substantive overlap and no compelling differentiators, likewise
        # downgrade. This catches "system X already does this" cases the
        # composite-only search missed.
        peer_name = (closest_peer_system.get("name") or "").strip()
        peer_overlap = (closest_peer_system.get("overlap_summary") or "").strip()
        peer_differentiators = closest_peer_system.get("differentiators") or []
        if (
            peer_name
            and len(peer_overlap) > 30
            and not peer_differentiators
            and novelty_type == "new_synthesis"
        ):
            print(f"  [peer-system guard] closest peer system {peer_name!r} has substantive "
                  f"overlap and no stated differentiators — downgrading to extension.")
            novelty_type = "extension"
            synthesis_findable = True

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
        # Extensions count for the upgrade IF the peer system has substantive
        # differentiators — that's the evidence the extension actually adds
        # something beyond the known peer. Without differentiators an
        # "extension" is effectively a restatement and must stay challenged.
        peer_has_differentiators_for_upgrade = bool(
            (closest_peer_system.get("differentiators") or [])
        )
        extension_eligible_for_upgrade = (
            novelty_type == "extension"
            and peer_has_differentiators_for_upgrade
        )
        new_synthesis_eligible_for_upgrade = (
            novelty_type in ("new_synthesis", "correction")
            and not synthesis_findable
        )
        if (
            verdict == "challenged"
            and (new_synthesis_eligible_for_upgrade or extension_eligible_for_upgrade)
            and premises_supported
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
                if extension_eligible_for_upgrade:
                    print(
                        f"  [guardrail] verdict=challenged but decomposition is a valid extension "
                        f"(novelty=extension, premises=✓, peer has "
                        f"{len(closest_peer_system.get('differentiators') or [])} differentiator(s)) and "
                        f"reasoning_flaws "
                        f"{'is empty' if not reasoning_flaws else 'contains no substantive critique markers'} — "
                        f"upgrading to validated."
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

        # Confidence-drop on any guard-induced downgrade. When a guard flipped
        # a "validated" verdict to "challenged" (skeptic-probe) or an
        # "inconclusive" to "challenged" (inconclusive guardrail), the LLM's
        # originally-returned confidence was computed before knowing the guard
        # would fire. Flat confidence on a verdict change is the classic hedge
        # signature — mechanically penalize it so the stored confidence
        # reflects the revised assessment.
        if llm_returned_verdict != verdict and verdict == "challenged":
            _penalty = float(getattr(
                self.config, "confidence_drop_on_downgrade", 0.10,
            ))
            _pre = verified_confidence
            verified_confidence = max(0.0, verified_confidence - _penalty)
            print(
                f"  [confidence-drop] verdict {llm_returned_verdict}→{verdict} via guard — "
                f"conf {_pre:.2f} → {verified_confidence:.2f} (penalty {_penalty:.2f})"
            )

        print(f"  Verdict: {verdict} · novelty={novelty_type or '?'} "
              f"· premises={'✓' if premises_supported else '✗'} "
              f"· synthesis_findable={'✓' if synthesis_findable else '✗'}")
        print(f"  Verified confidence: {verified_confidence:.2f}")
        if summary:
            print(f"  {summary[:200]}...")

        peer_has_differentiators = bool(
            closest_peer_system.get("differentiators") or []
        )
        outcome, entry_status, gate_reasons = self._register_gate(
            verdict, verified_confidence, premises_supported, synthesis_findable,
            novelty_type=novelty_type,
            peer_has_differentiators=peer_has_differentiators,
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
            verification_tool_calls=list(tool_trace),
            central_architectural_move=central_architectural_move,
            central_move_prior_art=list(central_move_prior_art),
            functional_decomposition=list(functional_decomposition),
            closest_peer_system=dict(closest_peer_system),
            skeptic_probe=dict(skeptic_probe),
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

            # Freshness check: ask the verifier client to evaluate whether the
            # falsifiable_condition is already observably instantiated. A claim
            # that's already true at creation time isn't a prediction; it's a
            # description. Flag it — don't register as pending.
            freshness = self._check_prediction_freshness(claim, condition)

            status = "pending"
            if freshness.get("verdict") == "already_fulfilled":
                status = "already_fulfilled"
                print(f"  PREDICTION already-fulfilled (skipped pending status): {claim[:100]}")
                if freshness.get("evidence"):
                    print(f"    evidence: {str(freshness['evidence'])[:200]}")

            prediction = Prediction(
                id=f"p-{uuid4().hex[:8]}",
                register_entry_id=register_entry_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                target_date=target,
                claim=claim,
                falsifiable_condition=condition,
                check_method=method,
                status=status,
                freshness_check=freshness,
            )
            self.journal.add_prediction(prediction)
            kept += 1
            if status == "pending":
                print(f"  PREDICTION registered: {prediction.id} (target {prediction.target_date})")
            else:
                print(f"  PREDICTION marked already_fulfilled: {prediction.id}")
        if kept == 0:
            print("  No falsifiable predictions emitted for this insight.")

    def _check_prediction_freshness(self, claim: str, condition: str) -> dict:
        """Probe whether a prediction's falsifiable_condition is ALREADY
        instantiated in the world at creation time.

        Strategy: one targeted web_search, then a short LLM call asking the
        verifier to judge whether the top results already satisfy the
        condition. Keeps it cheap (1 tool call + 1 LLM call) — not a full
        re-verification loop.

        Returns a dict: {verdict: "fresh"|"already_fulfilled"|"skipped",
        query: str, top_results_preview: str, evidence: str, reasoning: str}.
        """
        from engine.tools.base import registry as tool_registry
        web_tool = tool_registry.get("web_search")
        if web_tool is None:
            return {"verdict": "skipped", "reason": "web_search tool unavailable"}

        # Build the freshness query from the condition — keep it short, the
        # condition is usually a single sentence describing the observable.
        query = (condition[:220] + "...") if len(condition) > 220 else condition
        try:
            search_result = tool_registry.execute("web_search", {"query": query, "limit": 5})
            top_preview = search_result.content[:2000] if search_result and search_result.content else ""
        except Exception as e:  # noqa: BLE001
            return {
                "verdict": "skipped",
                "reason": f"web_search failed: {type(e).__name__}: {e}",
                "query": query,
            }

        if not top_preview.strip() or search_result.is_error:
            return {
                "verdict": "skipped",
                "reason": "web_search returned no usable content",
                "query": query,
            }

        # LLM judgement — has this condition already been met?
        judge_prompt = (
            "You are judging whether a prediction's falsifiable condition has "
            "ALREADY been observably instantiated in the world AT THIS MOMENT, "
            "making the prediction not actually a prediction but a description.\n\n"
            f"CLAIM: {claim}\n\n"
            f"FALSIFIABLE CONDITION: {condition}\n\n"
            "TOP WEB SEARCH RESULTS for the condition:\n"
            f"{top_preview}\n\n"
            "Decide: do the top results already satisfy the falsifiable condition? "
            "If the condition describes a state that the search results demonstrate "
            "is already true, answer already_fulfilled. Only answer already_fulfilled "
            "if there is concrete, cited evidence the condition holds NOW — a vague "
            "news item mentioning the topic is not enough.\n\n"
            'Respond EXACTLY: {"verdict": "fresh" | "already_fulfilled", '
            '"evidence": "1-2 sentence summary of the specific result(s) that do or '
            'do not satisfy the condition, with URL if present", '
            '"reasoning": "short explanation"}'
        )
        try:
            judgement = self._call_verifier(judge_prompt, max_tokens=800)
        except Exception as e:  # noqa: BLE001
            return {
                "verdict": "skipped",
                "reason": f"freshness judge failed: {type(e).__name__}: {e}",
                "query": query,
                "top_results_preview": top_preview[:500],
            }

        verdict = (judgement.get("verdict") or "").strip().lower()
        if verdict not in ("fresh", "already_fulfilled"):
            verdict = "fresh"  # default to fresh on ambiguous judge output
        return {
            "verdict": verdict,
            "query": query,
            "top_results_preview": top_preview[:500],
            "evidence": (judgement.get("evidence") or "").strip(),
            "reasoning": (judgement.get("reasoning") or "").strip(),
        }

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

    def reverify_register_entries(
        self,
        *,
        only_ids: Optional[list[str]] = None,
        max_confidence: Optional[float] = None,
        novelty_types: Optional[list[str]] = None,
        only_new_synthesis: bool = False,
        reason: str = "admin re-verify under updated rules",
    ) -> dict:
        """Re-run the verifier over existing register entries WITHOUT mutating
        the originals. Each pass is appended to the entry's `reverification_log`
        so the old verdict is preserved alongside the new one.

        Filters (combine as AND):
          - only_ids: explicit subset of register entry ids
          - max_confidence: only re-verify entries with verified_confidence ≤ this
          - novelty_types: only re-verify entries whose novelty_type is in this set
          - only_new_synthesis: convenience shorthand for novelty_types=['new_synthesis']

        Use this after changing verification rules (e.g. the phase-1 guard for
        central-move prior art, the skeptic smell test, or the peer-system
        guard) to audit whether previously-validated entries still hold up.
        """
        from models import CrossReference as _CR, Insight as _I

        if only_new_synthesis and not novelty_types:
            novelty_types = ["new_synthesis"]
        novelty_set = set(novelty_types or [])

        insights_by_id = {i.get("id"): i for i in self.journal.insights}
        xrefs_by_id = {x.get("id"): x for x in self.journal.cross_references}

        candidates: list[dict] = []
        for e in self.journal.register:
            eid = e.get("id")
            if not eid:
                continue
            if only_ids and eid not in only_ids:
                continue
            if novelty_set and (e.get("novelty_type") or "") not in novelty_set:
                continue
            if max_confidence is not None and float(e.get("verified_confidence", 0.0)) > max_confidence:
                continue
            candidates.append(e)

        print(f"\n--- RE-VERIFYING {len(candidates)} register entr(ies) "
              f"(filters: only_ids={bool(only_ids)}, novelty={novelty_set or 'any'}, "
              f"max_conf={max_confidence}) ---")
        stats = {"examined": 0, "verdict_changed": 0, "same_verdict": 0, "errors": 0, "skipped": 0}

        for e in candidates:
            stats["examined"] += 1
            eid = e.get("id")
            insight_id = e.get("insight_id")
            xref_id = e.get("supporting_xref_id")
            insight_dict = insights_by_id.get(insight_id)
            xref_dict = xrefs_by_id.get(xref_id)
            if insight_dict is None or xref_dict is None:
                print(f"  [skip] {eid}: missing source insight ({insight_id}) or xref ({xref_id})")
                stats["skipped"] += 1
                continue

            insight_fields = set(_I.__dataclass_fields__)
            insight = _I(**{k: v for k, v in insight_dict.items() if k in insight_fields})
            xref_fields = set(_CR.__dataclass_fields__)
            xref = _CR(**{k: v for k, v in xref_dict.items() if k in xref_fields})

            # Rebuild the verify prompt and run it — we reuse the same pipeline
            # `verify_insight` uses but WITHOUT calling add_register_entry at the
            # end. We inline the minimum needed to get verdict fields + trace.
            supporting = [x for x in self.journal.entries if x["id"] in xref.source_entries]
            slim_supporting = [self._slim_entry_for_register(x) for x in supporting]
            from prompts import VERIFY_PROMPT
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
            tool_trace: list[dict] = []
            try:
                result = self._call_verifier_with_tools(
                    prompt,
                    server_tools=server_tools,
                    max_tokens=self.connection.verifier.investigation_max_tokens,
                    trace=tool_trace,
                )
            except Exception as ex:  # noqa: BLE001
                print(f"  [error] {eid}: {type(ex).__name__}: {ex}")
                stats["errors"] += 1
                continue

            new_verdict = (result.get("verdict") or "").strip().lower()
            new_novelty = (result.get("novelty_type") or "").strip()
            new_conf = float(result.get("verified_confidence", 0.0))
            old_verdict = (e.get("verdict") or "").strip().lower()
            old_novelty = (e.get("novelty_type") or "").strip()

            changed = (new_verdict != old_verdict) or (new_novelty != old_novelty)
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "old_verdict": old_verdict,
                "old_novelty_type": old_novelty,
                "old_verified_confidence": float(e.get("verified_confidence", 0.0)),
                "new_verdict": new_verdict,
                "new_novelty_type": new_novelty,
                "new_verified_confidence": new_conf,
                "new_premises_supported": bool(result.get("premises_supported", True)),
                "new_synthesis_findable": bool(result.get("synthesis_findable", False)),
                "new_central_architectural_move": (result.get("central_architectural_move") or "").strip(),
                "new_central_move_prior_art": list(result.get("central_move_prior_art", []) or []),
                "new_synthesis_prior_art": list(result.get("synthesis_prior_art", []) or []),
                "new_closest_peer_system": dict(result.get("closest_peer_system") or {}),
                "new_skeptic_probe": dict(result.get("skeptic_probe") or {}),
                "new_functional_decomposition": list(result.get("functional_decomposition") or []),
                "new_contradicting_findings": list(result.get("contradicting_findings") or []),
                "new_reasoning_flaws": list(result.get("reasoning_flaws") or []),
                "new_verification_summary": (result.get("verification_summary") or "").strip(),
                "tool_calls": list(tool_trace),
                "verdict_changed": changed,
            }
            self.journal.append_register_reverification(eid, log_entry)

            if changed:
                stats["verdict_changed"] += 1
                print(f"  ⚠ {eid}: {old_verdict}/{old_novelty} → {new_verdict}/{new_novelty} "
                      f"(conf {float(e.get('verified_confidence',0.0)):.2f} → {new_conf:.2f})")
            else:
                stats["same_verdict"] += 1
                print(f"  ✓ {eid}: verdict unchanged ({new_verdict}/{new_novelty})")

        print(
            f"\nRe-verify complete: examined={stats['examined']} · "
            f"verdict_changed={stats['verdict_changed']} · same={stats['same_verdict']} · "
            f"errors={stats['errors']} · skipped={stats['skipped']}"
        )
        return stats

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
