"""Verification (Phase 6) + prediction checks (Phase 7). Cross-model adversarial review."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from engine.embeddings import cosine
from models import CrossReference, Insight, Prediction, RegisterEntry  # noqa: F401
from prompts import CANONICAL_FORM_PROMPT, PREDICTION_CHECK_PROMPT, VERIFY_PROMPT

# Alias-gap thresholds. The metric is `gap = 1 - similarity` where
#   similarity = 0.3 * slot_match + 0.7 * cosine(canonical_form_text)
# Smaller gap = closer alias.
#
# Weighting: embedding cosine carries more weight than exact slot match
# because canonicalization tends to produce specific mechanism strings
# that rarely string-match across structurally identical claims. Cosine
# captures structural similarity that surface variation hides; slot match
# is a confirmation signal when it does fire.
#
# Thresholds were calibrated on the ideation_on_ideation register
# (39 entries, 21 canonicalized): genuine aliases sit at gap 0.30-0.40
# under this formula; clearly-distinct claims sit above 0.50.
#
#   gap < ALIAS_GAP_STRICT  → near-identical canonical form. Treated as an
#                              articulate restatement: novelty_type is
#                              downgraded and the candidate's verdict is
#                              flagged for register-gate scrutiny.
#   gap < ALIAS_GAP_BAND    → in the disagreement-gating band. Logged as a
#                              soft alias signal; does NOT mechanically
#                              downgrade in Phase 1, but populates
#                              `aliasing_against` on the canonical_form for
#                              downstream review (Phase 2 will use this to
#                              trigger structured-delta scoring).
#   gap >= ALIAS_GAP_BAND   → comfortably distinct. No alias signal.
ALIAS_GAP_STRICT = 0.30
ALIAS_GAP_BAND = 0.45
_ALIAS_SLOT_WEIGHT = 0.3
_ALIAS_COSINE_WEIGHT = 0.7


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

    # ── Canonicalization layer (Phase 1 — feeds alias-gap detection) ──

    @staticmethod
    def _canonical_form_text(c: dict) -> str:
        """Concatenated slot text used as the embedding input. Order-stable
        so the same canonical form embeds to (approximately) the same vector
        across calls."""
        if not c:
            return ""
        parts = [
            c.get("move_predicate") or "",
            c.get("on_substrate") or "",
            c.get("with_mechanism") or "",
            c.get("target_domain") or "",
        ]
        for k in (c.get("key_constraints") or []):
            parts.append(str(k))
        return " | ".join(p.strip() for p in parts if p and str(p).strip())

    def _canonicalize_central_move(
        self,
        title: str,
        description: str,
        central_architectural_move: str = "",
    ) -> dict:
        """Extract the canonical structured form of a claim. Returns an empty
        dict if the model could not produce a clean canonical form (which
        is itself a valid signal: claims with no clean canonical form are
        usually motivational rather than load-bearing).

        `central_architectural_move` is OPTIONAL. When empty, the model
        derives the move from `description` directly — this is the path
        used by Stage 1 of the three-stage verifier (canonicalization
        runs before the heavy verifier extracts central_architectural_move).
        When non-empty, it serves as a pre-extraction hint for the
        canonicalizer to structure.
        """
        if not (description or "").strip():
            # Nothing to extract from. Empty input → empty output (not an error).
            return {}
        prompt = CANONICAL_FORM_PROMPT.format(
            engine_domain=getattr(self.config, "domain", "") or "(unspecified)",
            title=title,
            description=description,
            central_architectural_move=(central_architectural_move or "").strip(),
        )
        try:
            result = self._call_verifier(prompt, max_tokens=600)
        except Exception as e:  # noqa: BLE001 — canonicalization is best-effort
            print(f"  [canonicalize] failed: {type(e).__name__}: {e}")
            return {}
        canonical = {
            "move_predicate": (result.get("move_predicate") or "").strip().lower(),
            "on_substrate": (result.get("on_substrate") or "").strip().lower(),
            "with_mechanism": (result.get("with_mechanism") or "").strip().lower(),
            "target_domain": (result.get("target_domain") or "").strip().lower(),
            "key_constraints": [
                str(c).strip().lower() for c in (result.get("key_constraints") or [])
                if str(c).strip()
            ],
        }
        # An empty move_predicate means the model gave up on structuring —
        # treat as no canonical form rather than a partial one (avoids
        # downstream alias-gap math on a degenerate vector).
        if not canonical["move_predicate"]:
            return {}
        return canonical

    def _alias_gap(self, candidate: dict, *, exclude_id: str = "") -> dict:
        """Compute the alias gap between `candidate` (a canonical form) and
        every register entry that has a populated canonical_form. Returns:
            {"gap": float in [0,1],
             "nearest_ids": list[str],
             "scored_against": int}
        gap=0.0 means perfect alias of nearest entry; gap=1.0 means
        orthogonal (or no comparable entries on file).

        `exclude_id` skips that register entry from the comparison set —
        used by the force-reverse-canonicalization path so an entry doesn't
        match against its own previous canonical form.

        Scoring combines:
          - structured-slot match: fraction of {predicate, substrate,
            mechanism} that match exactly between candidate and existing
            (lowercase string equality after canonicalization, not embedding)
          - embedding cosine on the full canonical-form text, when an
            embedding client is available

        The combined score weights slots heavier (0.6) than embeddings (0.4)
        because exact slot matches are a stronger restatement signal than
        soft text similarity. If no embedding client is configured, falls
        back to slot-only scoring."""
        if not candidate or not candidate.get("move_predicate"):
            return {"gap": 1.0, "nearest_ids": [], "scored_against": 0}

        candidate_text = self._canonical_form_text(candidate)
        candidate_emb: Optional[list[float]] = None
        emb_client = getattr(self, "embedding_client", None)
        if emb_client is not None and candidate_text:
            try:
                candidate_emb = emb_client.embed([candidate_text])[0]
            except Exception as e:  # noqa: BLE001 — embedding is optional
                print(f"  [alias-gap] embedding failed: {type(e).__name__}: {e}")
                candidate_emb = None

        best_score = 0.0
        nearest_ids: list[str] = []
        scored = 0
        for e in self.journal.register:
            if e.get("status") != "active":
                continue
            if exclude_id and e.get("id") == exclude_id:
                continue
            ec = e.get("canonical_form") or {}
            if not ec.get("move_predicate"):
                continue
            scored += 1

            # Structured-slot match
            slot_hits = sum([
                ec.get("move_predicate") == candidate.get("move_predicate"),
                ec.get("on_substrate") == candidate.get("on_substrate"),
                ec.get("with_mechanism") == candidate.get("with_mechanism"),
            ])
            slot_score = slot_hits / 3.0

            # Embedding cosine (best-effort)
            emb_score = 0.0
            if candidate_emb is not None:
                existing_text = self._canonical_form_text(ec)
                if existing_text:
                    try:
                        existing_emb = emb_client.embed([existing_text])[0]
                        emb_score = max(0.0, cosine(candidate_emb, existing_emb))
                    except Exception:  # noqa: BLE001
                        emb_score = 0.0

            combined = (
                (_ALIAS_SLOT_WEIGHT * slot_score) + (_ALIAS_COSINE_WEIGHT * emb_score)
                if candidate_emb is not None
                else slot_score
            )
            if combined > best_score:
                best_score = combined
                nearest_ids = [e.get("id")]
            elif combined == best_score and best_score > 0:
                nearest_ids.append(e.get("id"))

        return {
            "gap": max(0.0, 1.0 - best_score),
            "nearest_ids": nearest_ids[:5],
            "scored_against": scored,
        }

    # ── Pareto admission (Phase 4) ──────────────────────────────────────

    # Default axis set for the Pareto admission gate. Excludes
    # `known_prior_art_score` because it's degenerate (all-zero) on
    # journals that have no human-curated known_prior_art anchors —
    # which is the common case. When a journal does maintain anchors
    # AND verifier evaluations produce differentiator lists, that
    # axis can be re-enabled via config (future work; not exposed
    # in Phase 4).
    _PARETO_AXES = (
        "verified_confidence",
        "premises_supported_count",
        "peer_differentiators_count",
        "inverse_alias_gap",
    )

    @staticmethod
    def _compute_pareto_axes(
        *,
        verified_confidence: float,
        premises_support_citations: list,
        closest_peer_system: dict,
        known_prior_art_evaluations: list,
        alias_gap: float,
    ) -> dict:
        """Single source of truth for an entry's pareto_axes dict.
        Used by verify_insight (new candidates) and the backfill pass
        (existing entries). The full 5-axis dict is stored on the entry,
        even though the default Pareto check uses only 4 axes — keeps
        future axis tuning possible without re-running verification."""
        kpa = known_prior_art_evaluations or []
        return {
            "verified_confidence": float(verified_confidence),
            "premises_supported_count": len(premises_support_citations or []),
            "peer_differentiators_count": len(
                (closest_peer_system or {}).get("differentiators") or []
            ),
            "known_prior_art_score": (
                float(sum(1 for ev in kpa if ev.get("differentiators")))
                / max(1.0, len(kpa))
            ),
            "inverse_alias_gap": float(alias_gap),
        }

    @classmethod
    def _pareto_dominates(cls, existing_axes: dict, candidate_axes: dict) -> bool:
        """True iff `existing` dominates `candidate` (>= on every axis,
        AND > on at least one). Missing axes are treated as 0.0."""
        if not existing_axes or not candidate_axes:
            return False
        strict_better = False
        for axis in cls._PARETO_AXES:
            e = float(existing_axes.get(axis, 0.0))
            c = float(candidate_axes.get(axis, 0.0))
            if e < c:
                return False  # existing loses on this axis → cannot dominate
            if e > c:
                strict_better = True
        return strict_better

    def _check_pareto_admission(self, candidate_axes: dict) -> tuple[bool, list[str]]:
        """Returns (admitted, dominating_entry_ids). admitted=True iff no
        existing register entry dominates the candidate on the configured
        axis set. Entries without `pareto_axes` populated do not
        participate in the comparison (they predate Phase 1)."""
        dominating: list[str] = []
        for e in self.journal.register:
            if e.get("status") != "active":
                continue
            ex = e.get("pareto_axes") or {}
            if not ex:
                continue
            if self._pareto_dominates(ex, candidate_axes):
                dominating.append(e.get("id", ""))
        return (not dominating, dominating)

    def compute_pareto_frontier(self) -> list[dict]:
        """Active register entries on the current Pareto frontier — the
        entries that are not dominated by any other entry on the
        configured axis set. These are the entries actually setting the
        admission bar: a new candidate must beat at least one of them on
        at least one axis to be admitted under pareto mode.

        Returns a list of register-entry dicts (not RegisterEntry
        instances). Entries without `pareto_axes` are excluded.
        """
        candidates = [
            e for e in self.journal.register
            if e.get("status") == "active" and (e.get("pareto_axes") or {})
        ]
        frontier: list[dict] = []
        for cand in candidates:
            cand_axes = cand["pareto_axes"]
            dominated = False
            for other in candidates:
                if other.get("id") == cand.get("id"):
                    continue
                if self._pareto_dominates(other.get("pareto_axes") or {}, cand_axes):
                    dominated = True
                    break
            if not dominated:
                frontier.append(cand)
        return frontier

    def test_pareto_admission(self, axes_csv: str) -> dict:
        """Diagnostic: run the Pareto admission check against the current
        register with caller-supplied synthetic axes. Lets us exercise
        the admission logic directly without finding a real candidate
        that passes the heavy verifier first.

        `axes_csv` is a 4-value comma-separated string in fixed order:
            verified_confidence, premises_supported_count,
            peer_differentiators_count, inverse_alias_gap

        Example: "0.75,8,4,0.30" → a candidate with confidence 0.75,
        8 premise citations, 4 peer differentiators, alias gap 0.30.
        """
        parts = [p.strip() for p in (axes_csv or "").split(",")]
        if len(parts) != 4:
            print(
                f"\n--- PARETO ADMISSION TEST ---\n"
                f"  ERROR: expected 4 comma-separated values "
                f"(verified_confidence, premises_supported_count, "
                f"peer_differentiators_count, inverse_alias_gap); got "
                f"{len(parts)}: {parts!r}"
            )
            return {"error": "wrong arity", "given": parts}

        try:
            candidate = {
                "verified_confidence": float(parts[0]),
                "premises_supported_count": float(parts[1]),
                "peer_differentiators_count": float(parts[2]),
                "inverse_alias_gap": float(parts[3]),
            }
        except ValueError as e:
            print(f"\n--- PARETO ADMISSION TEST ---\n  ERROR: could not parse: {e}")
            return {"error": str(e), "given": parts}

        admission_mode = (
            getattr(getattr(self.connection, "engine", None), "register_admission_mode", "scalar") or "scalar"
        ).strip().lower()
        admitted, dominating = self._check_pareto_admission(candidate)

        print("\n--- PARETO ADMISSION TEST ---")
        print(f"  current admission mode: {admission_mode}")
        print(f"  axes used by check:     {list(self._PARETO_AXES)}")
        print()
        print("  candidate axes:")
        for axis in self._PARETO_AXES:
            print(f"    {axis:<30}: {candidate[axis]}")
        print()
        print(f"  admitted: {admitted}")
        if not admitted:
            print(f"  dominated by {len(dominating)} entr(ies):")
            register_lookup = {e.get("id", ""): e for e in self.journal.register}
            for rid in dominating[:8]:
                e = register_lookup.get(rid, {})
                p = e.get("pareto_axes") or {}
                title = (e.get("title") or "")[:60]
                print(
                    f"    {rid}: "
                    f"conf={p.get('verified_confidence', 0):.2f} "
                    f"prem={int(p.get('premises_supported_count', 0))} "
                    f"peer={int(p.get('peer_differentiators_count', 0))} "
                    f"inv_g={p.get('inverse_alias_gap', 0):.3f}  {title}"
                )
            if len(dominating) > 8:
                print(f"    … (+{len(dominating) - 8} more)")
            if admission_mode != "pareto":
                print(
                    f"\n  note: admission_mode is currently {admission_mode!r} — "
                    f"this candidate would NOT actually be rejected at registration "
                    f"under the current config. Flip register_admission_mode to "
                    f"'pareto' (Settings page or engine.toml) to enable enforcement."
                )
        else:
            print("  → no existing register entry dominates the candidate.")
            if admission_mode == "pareto":
                print(
                    "  → under current config, this candidate would be admitted "
                    "(if it also passes the scalar gate)."
                )
        return {
            "admitted": admitted,
            "dominating_entry_ids": dominating,
            "admission_mode": admission_mode,
            "candidate_axes": candidate,
        }

    def show_pareto_frontier(self) -> None:
        """CLI: print the current Pareto frontier with each entry's
        winning axis (the axis on which it dominates at least one other
        active entry). Useful for understanding what bar new candidates
        face under pareto admission mode."""
        frontier = self.compute_pareto_frontier()
        active_with_axes = sum(
            1 for e in self.journal.register
            if e.get("status") == "active" and (e.get("pareto_axes") or {})
        )
        active_total = sum(
            1 for e in self.journal.register if e.get("status") == "active"
        )
        admission_mode = (
            getattr(getattr(self.connection, "engine", None), "register_admission_mode", "scalar") or "scalar"
        ).strip().lower()

        print("\n--- PARETO FRONTIER ---")
        print(f"  axes:                            {list(self._PARETO_AXES)}")
        print(f"  admission mode:                  {admission_mode}")
        print(f"  active entries (total):          {active_total}")
        print(f"  active entries with pareto_axes: {active_with_axes}")
        print(f"  on frontier:                     {len(frontier)}")
        if active_with_axes < active_total:
            print(
                f"  note: {active_total - active_with_axes} active entr(ies) lack pareto_axes "
                "(predate Phase 1) and do not participate in the Pareto frontier."
            )
        print()
        if not frontier:
            print("  (frontier empty — no entries with pareto_axes populated)")
            return

        print(
            f"  {'id':<14} {'conf':>5} {'prem':>5} {'peer':>5} {'inv_g':>6}  "
            "wins on              title"
        )
        print("  " + "-" * 110)
        for e in sorted(
            frontier, key=lambda x: -x["pareto_axes"].get("verified_confidence", 0),
        ):
            p = e["pareto_axes"]
            wins: list[str] = []
            for axis in self._PARETO_AXES:
                others = [
                    other["pareto_axes"].get(axis, 0)
                    for other in frontier
                    if other.get("id") != e.get("id")
                ]
                if others and float(p.get(axis, 0)) > max(float(o) for o in others):
                    wins.append(axis.replace("_count", "").replace("verified_", ""))
            title = (e.get("title") or "")[:60]
            print(
                f"  {e['id']:<14} "
                f"{p['verified_confidence']:>5.2f} "
                f"{int(p['premises_supported_count']):>5} "
                f"{int(p['peer_differentiators_count']):>5} "
                f"{p['inverse_alias_gap']:>6.3f}  "
                f"{','.join(wins) or '(tied)':<20} {title}"
            )

    # ── Component-resolved novelty (Phase 3) ────────────────────────────

    @staticmethod
    def _slug(s: str) -> str:
        """Slug a free-text dimension name into a stable component-novelty key.
        Lowercase, alnum-only with single underscores."""
        out = "".join((c if c.isalnum() else "_") for c in (s or "").lower())
        while "__" in out:
            out = out.replace("__", "_")
        return out.strip("_") or "unnamed"

    @staticmethod
    def _decompose_novelty(
        *,
        novelty_type: str,
        premises_supported: bool,
        alias_tier: str,
        central_move_prior_art: list,
        functional_decomposition: list,
        closest_peer_system: dict,
    ) -> dict:
        """Compute per-component novelty status from the verifier's outputs.

        Phase 3 stores novelty per architectural component instead of (only)
        as a single entry-level rollup. Reverification can then flip a
        single component's status — e.g. "central move now has prior art
        but the dimension-3 differentiator is still novel" — without
        forcing a binary entry-level verdict change.

        Rules (deterministic, no LLM):
          • central_move:
              - "unsupported"   if premises are not supported
              - "restatement"   if Stage 2 alias-gap fired STRICT
              - "extension"     if central_move_prior_art has substantive entries
              - "correction"    if the LLM verdict named correction explicitly
              - "new_synthesis" otherwise
          • decomposition_<dim_slug>: one per functional_decomposition row
              - "new_synthesis" if no nearest_exemplar named
              - "restatement"   if how_ours_differs is empty / hand-wavy
              - "extension"     otherwise (substantive differentiator named)
          • closest_peer_system:
              - "new_synthesis" if no peer named
              - "restatement"   if peer named but no differentiators
              - "extension"     otherwise
        """
        components: dict[str, str] = {}

        # central_move
        substantive_pa = [
            c for c in (central_move_prior_art or [])
            if isinstance(c, str) and len(c.strip()) > 20
            and not c.lower().startswith(("no ", "none", "empty", "searched"))
        ]
        if not premises_supported:
            components["central_move"] = "unsupported"
        elif alias_tier == "STRICT":
            components["central_move"] = "restatement"
        elif substantive_pa:
            components["central_move"] = "extension"
        elif (novelty_type or "").lower() == "correction":
            components["central_move"] = "correction"
        else:
            components["central_move"] = "new_synthesis"

        # functional_decomposition rows — one component per dimension
        handwave_markers = (
            "figure out", "iterate until", "try various", "various approaches",
            "appropriate", "as needed", "tbd", "to be determined",
        )
        for d in (functional_decomposition or []):
            if not isinstance(d, dict):
                continue
            dim = (d.get("dimension") or "").strip()
            if not dim:
                continue
            key = "decomposition_" + VerificationMixin._slug(dim)
            nearest = (d.get("nearest_exemplar") or "").strip()
            diff = (d.get("how_ours_differs") or "").strip()
            if not nearest or nearest.lower() in ("none", "n/a", "unknown", "—", "-"):
                components[key] = "new_synthesis"
            elif (
                not diff or len(diff) < 15
                or any(m in diff.lower() for m in handwave_markers)
            ):
                components[key] = "restatement"
            else:
                components[key] = "extension"

        # closest_peer_system
        peer = closest_peer_system or {}
        peer_name = (peer.get("name") or "").strip()
        peer_diffs = peer.get("differentiators") or []
        peer_diffs = [d for d in peer_diffs if isinstance(d, str) and len(d.strip()) > 5]
        if not peer_name:
            components["closest_peer_system"] = "new_synthesis"
        elif not peer_diffs:
            components["closest_peer_system"] = "restatement"
        else:
            components["closest_peer_system"] = "extension"

        return components

    @staticmethod
    def _component_novelty_delta(old: dict, new: dict) -> dict:
        """Return only the keys whose status changed between old and new.
        Format: {key: {"from": old_status, "to": new_status}}.
        Used by reverification to surface what flipped, instead of
        overwriting the entry's stored component_novelty wholesale."""
        delta: dict = {}
        for k, v in (new or {}).items():
            if (old or {}).get(k) != v:
                delta[k] = {"from": (old or {}).get(k, "(absent)"), "to": v}
        for k in (old or {}):
            if k not in (new or {}):
                delta[k] = {"from": old[k], "to": "(absent)"}
        return delta

    @staticmethod
    def _build_canonical_form_context(
        canonical_form: dict,
        alias_tier: str,
        alias_signal: dict,
        register_lookup: dict,
    ) -> str:
        """Format the canonical-form + alias context block injected into
        VERIFY_PROMPT (Stage 3). Empty string when there's nothing to say.
        Contains:
          - the pre-extracted canonical form (so the verifier doesn't need
            to re-derive central_architectural_move from scratch — it can
            still override the slot values, but starts from a structured
            anchor instead of free description prose)
          - if Stage 2 flagged BAND, an explicit instruction to look for
            differentiators against the named soft-aliased peer entries"""
        if not canonical_form:
            return "(no canonical form extracted — Stage 1 found no clean structural move)"
        lines: list[str] = []
        lines.append("Stage 1 canonical form (pre-extracted from the insight):")
        lines.append(f'  move_predicate:  "{canonical_form.get("move_predicate", "")}"')
        lines.append(f'  on_substrate:    "{canonical_form.get("on_substrate", "")}"')
        lines.append(f'  with_mechanism:  "{canonical_form.get("with_mechanism", "")}"')
        lines.append(f'  target_domain:   "{canonical_form.get("target_domain", "")}"')
        kc = canonical_form.get("key_constraints") or []
        if kc:
            lines.append(f"  key_constraints: {kc}")
        lines.append(
            "Use this as a structural anchor when extracting "
            "central_architectural_move. You may refine it, but do NOT drop "
            "specificity already captured here."
        )
        if alias_tier == "BAND" and alias_signal.get("nearest_ids"):
            nearest_ids = alias_signal["nearest_ids"]
            lines.append("")
            lines.append(
                f"Stage 2 flagged this candidate as soft-aliased "
                f"(gap={alias_signal['gap']:.2f}) to existing register "
                f"entries: {nearest_ids}."
            )
            for rid in nearest_ids[:3]:
                peer = register_lookup.get(rid, {})
                peer_title = (peer.get("title") or "").strip()
                peer_canonical = peer.get("canonical_form") or {}
                if peer_canonical:
                    pc_text = " ".join(filter(None, [
                        peer_canonical.get("move_predicate"),
                        peer_canonical.get("on_substrate"),
                        peer_canonical.get("with_mechanism"),
                    ]))
                    lines.append(f"  {rid}: '{pc_text}' — {peer_title[:80]}")
            lines.append(
                "In your Phase 3a functional decomposition and Phase 3b "
                "closest_peer_system analysis, EXPLICITLY evaluate whether "
                "this candidate has concrete differentiators against these "
                "soft-aliased peers. If no differentiators surface, the "
                "claim is at most an extension; if the architectural move "
                "is identical, it is a restatement."
            )
        return "\n".join(lines)

    # ── STRICT-alias short-circuit (Phase 2 — Stage 2 reject path) ──

    @staticmethod
    def _build_strict_alias_result(
        insight: Insight,
        canonical_form: dict,
        nearest_ids: list[str],
        gap: float,
        register_lookup: dict,
    ) -> dict:
        """Build a synthetic verifier result that mirrors the schema
        produced by VERIFY_PROMPT, used when Stage 2 detects a STRICT
        alias and we want to skip the heavy phased prior-art search.

        The synthetic result threads through the existing post-stage-3
        guards and register_gate as if the heavy verifier had returned
        it — so all the established machinery (peer-system guard,
        confidence-drop, gate rejection on novelty=restatement) fires
        normally. No code branching needed downstream."""
        nearest = nearest_ids[0] if nearest_ids else ""
        peer = register_lookup.get(nearest, {}) if nearest else {}
        peer_title = (peer.get("title") or "").strip()
        peer_summary = (peer.get("verification_summary") or "").strip()[:280]
        canonical_text = " ".join(filter(None, [
            canonical_form.get("move_predicate"),
            canonical_form.get("on_substrate"),
            canonical_form.get("with_mechanism"),
        ]))
        summary_lines = [
            f"Stage 2 alias detection: this candidate's canonical form "
            f"({canonical_text}) is structurally identical (gap={gap:.2f}) "
            f"to existing register entry {nearest}"
            + (f" ({peer_title})" if peer_title else "")
            + ".",
        ]
        if peer_summary:
            summary_lines.append(f"Existing entry's summary: {peer_summary}")
        summary_lines.append(
            "Heavy phased prior-art search skipped — structural match is "
            "unambiguous from the canonical form alone."
        )
        return {
            "verdict": "challenged",
            "verified_confidence": 0.30,
            "premises_supported": True,
            "premises_support_citations": [],
            "synthesis_findable": True,
            "synthesis_prior_art": [
                f"register:{rid}" for rid in nearest_ids
            ],
            "novelty_type": "restatement",
            "central_architectural_move": canonical_text or insight.title,
            "central_move_prior_art": [
                f"register:{rid}" for rid in nearest_ids
            ],
            "functional_decomposition": [],
            "closest_peer_system": {
                "name": peer_title,
                "url": "",
                "overlap_summary": peer_summary,
                "differentiators": [],
            } if peer_title else {},
            "skeptic_probe": {
                "candidate_queries": [],
                "query": "stage-2 alias detection",
                "top_result_summary": (
                    f"Identical canonical form to register entry {nearest}"
                ) if nearest else "Identical canonical form to existing register entry",
                "followup_query": "",
                "followup_summary": "",
                "disqualifies": True,
            },
            "target_application_domain": canonical_form.get("target_domain", ""),
            "known_prior_art_evaluations": [],
            "contradicting_findings": [],
            "reasoning_flaws": [],
            "verification_summary": " ".join(summary_lines),
            "motivation": (insight.description or "")[:500],
            "predictions": [],
        }

    def verify_insight(
        self,
        insight: Insight,
        xref: CrossReference,
    ) -> Optional[RegisterEntry]:
        """Adversarially verify an insight. Return a RegisterEntry only if it passes.

        Three-stage architecture (Phase 2):
          Stage 1 — canonicalize the central move from the raw insight
                    (one focused LLM call, ~5s).
          Stage 2 — deterministic alias-gap detection vs the register's
                    existing canonical_forms. STRICT alias short-circuits
                    to a synthetic verifier result; BAND alias adds
                    differentiator-seeking context to Stage 3's prompt;
                    CLEAR proceeds normally.
          Stage 3 — existing phased prior-art search (the heavy
                    VERIFY_PROMPT call), augmented with Stage 1's
                    canonical_form as pre-extracted context.
        """
        print("\n--- VERIFYING INSIGHT ---")
        print(f"  Target: {insight.title}")

        supporting = [e for e in self.journal.entries if e["id"] in xref.source_entries]
        slim_supporting = [self._slim_entry_for_register(e) for e in supporting]

        engine_domain = getattr(self.config, "domain", "") or "(unspecified)"

        # ── Stage 1: canonicalize from raw insight ──────────────────────
        print("  [stage 1] canonicalizing central architectural move…")
        canonical_form = self._canonicalize_central_move(
            insight.title, insight.description,
        )
        if canonical_form and canonical_form.get("move_predicate"):
            print(
                f"    canonical: '{canonical_form.get('move_predicate')}' "
                f"'{canonical_form.get('on_substrate')}' via "
                f"'{canonical_form.get('with_mechanism')}'"
            )
        else:
            print("    [warn] no canonical form extracted — claim has no clean structural move")

        # ── Stage 2: alias-gap detection ────────────────────────────────
        alias_signal: dict = {"gap": 1.0, "nearest_ids": [], "scored_against": 0}
        alias_tier: str = "CLEAR"  # CLEAR | BAND | STRICT
        if canonical_form:
            alias_signal = self._alias_gap(canonical_form)
            gap = alias_signal["gap"]
            scored = alias_signal["scored_against"]
            nearest = alias_signal["nearest_ids"]
            if scored == 0:
                print("  [stage 2] no canonicalized register entries on file — skipping comparison")
            elif gap < ALIAS_GAP_STRICT and nearest:
                alias_tier = "STRICT"
                print(
                    f"  [stage 2] STRICT alias — gap={gap:.2f} (<{ALIAS_GAP_STRICT}) "
                    f"vs {nearest}. Skipping heavy verifier; emitting restatement."
                )
            elif gap < ALIAS_GAP_BAND and nearest:
                alias_tier = "BAND"
                print(
                    f"  [stage 2] BAND alias — gap={gap:.2f} (<{ALIAS_GAP_BAND}) "
                    f"vs {nearest}. Stage 3 will be primed for differentiator search."
                )
            else:
                print(
                    f"  [stage 2] CLEAR — gap={gap:.2f} vs nearest of "
                    f"{scored} canonicalized entr{'y' if scored == 1 else 'ies'}"
                )

        # ── STRICT short-circuit: skip Stage 3 entirely ─────────────────
        register_lookup = {e.get("id", ""): e for e in self.journal.register}
        tool_trace: list[dict] = []
        if alias_tier == "STRICT":
            result = self._build_strict_alias_result(
                insight, canonical_form, alias_signal["nearest_ids"],
                alias_signal["gap"], register_lookup,
            )
        else:
            # ── Stage 3: heavy phased prior-art search ──────────────────
            known_prior_art = self.journal.match_known_prior_art([engine_domain])
            canonical_form_context = self._build_canonical_form_context(
                canonical_form, alias_tier, alias_signal, register_lookup,
            )
            prompt = VERIFY_PROMPT.format(
                insight_json=json.dumps(asdict(insight), indent=2),
                xref_json=json.dumps(asdict(xref), indent=2),
                supporting_entries_json=json.dumps(slim_supporting, indent=2),
                tool_list=self._tool_list_block(for_client=self.verifier),
                prior_human_rejections_json=json.dumps(
                    self.journal.human_rejection_feedback(), indent=2,
                ),
                engine_domain=engine_domain,
                known_prior_art_json=json.dumps(known_prior_art, indent=2),
                canonical_form_context=canonical_form_context,
            )
            server_tools = [
                {"type": "web_search_20250305", "name": "web_search"},
                {"type": "code_execution_20250825", "name": "code_execution"},
            ]
            # Capture the full tool-call trace so we can persist it on the register
            # entry — makes after-the-fact audits of verifier misses ("why didn't
            # this search find X?") possible without re-running the whole pass.
            print("  [stage 3] running phased prior-art search with canonical-form context…")
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
        target_application_domain = (result.get("target_application_domain") or "").strip()
        known_prior_art_evaluations = result.get("known_prior_art_evaluations", []) or []

        # Known-prior-art guard: if any injected anchor was evaluated as a
        # peer with substantive claim overlap AND no differentiators, the
        # claim is at most an extension (or refuted). Human has already told
        # the verifier these matter — respect that signal mechanically.
        peer_with_overlap_no_diff = [
            ev for ev in known_prior_art_evaluations
            if ev.get("is_peer") and ev.get("overlaps_claim")
            and not (ev.get("differentiators") or [])
        ]
        if peer_with_overlap_no_diff and novelty_type == "new_synthesis":
            print(
                f"  [known-prior-art guard] {len(peer_with_overlap_no_diff)} anchor(s) "
                f"evaluated as peer with overlap + no differentiators — downgrading "
                f"new_synthesis → extension (synthesis_findable flipped)."
            )
            novelty_type = "extension"
            synthesis_findable = True

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

        # Canonicalization + alias-gap detection happened up front in
        # Stages 1 and 2 (before this Stage-3 result was generated).
        # The canonical_form is already populated; aliasing_against is
        # already attached when applicable. No work to do here.
        if canonical_form and alias_tier == "STRICT":
            # The result we're processing is the synthetic STRICT-alias
            # output; tag the canonical_form with the nearest_ids so the
            # downstream RegisterEntry surfaces them in the UI.
            canonical_form["aliasing_against"] = alias_signal["nearest_ids"]
        elif canonical_form and alias_tier == "BAND":
            canonical_form["aliasing_against"] = alias_signal["nearest_ids"]

        print(f"  Verdict: {verdict} · novelty={novelty_type or '?'} "
              f"· premises={'✓' if premises_supported else '✗'} "
              f"· synthesis_findable={'✓' if synthesis_findable else '✗'}")
        print(f"  Verified confidence: {verified_confidence:.2f}")
        if summary:
            print(f"  {summary[:200]}...")

        peer_has_differentiators = bool(
            closest_peer_system.get("differentiators") or []
        )

        # Phase 4: compute pareto_axes once and persist on every entry
        # (regardless of admission mode). Even in scalar mode the values
        # are stored so that flipping admission_mode → pareto later
        # doesn't require backfilling.
        pareto_axes = self._compute_pareto_axes(
            verified_confidence=verified_confidence,
            premises_support_citations=result.get("premises_support_citations", []) or [],
            closest_peer_system=closest_peer_system,
            known_prior_art_evaluations=known_prior_art_evaluations,
            alias_gap=alias_signal.get("gap", 1.0),
        )

        outcome, entry_status, gate_reasons = self._register_gate(
            verdict, verified_confidence, premises_supported, synthesis_findable,
            novelty_type=novelty_type,
            peer_has_differentiators=peer_has_differentiators,
        )

        # Phase 4: Pareto admission check, applied AFTER the scalar gate
        # approves. New entry is rejected if any existing active entry
        # dominates it on the configured axis set. Scalar mode → no-op.
        admission_mode = (
            getattr(getattr(self.connection, "engine", None), "register_admission_mode", "scalar") or "scalar"
        ).strip().lower()
        if outcome == "register" and admission_mode == "pareto":
            # Defensive: if the candidate's pareto_axes are all zero
            # (or empty), something upstream produced no useful signal.
            # Most likely a regression in _compute_pareto_axes or a
            # broken verifier output. Log a clear warning so this can't
            # silently slip through; skip the admission check to avoid
            # rejecting on degenerate data alone.
            all_zero = not pareto_axes or all(
                float(pareto_axes.get(axis, 0.0)) == 0.0
                for axis in self._PARETO_AXES
            )
            if all_zero:
                print(
                    "  [pareto admission] WARNING: candidate has all-zero pareto_axes — "
                    "indicates a regression in _compute_pareto_axes or empty verifier "
                    "output. Skipping Pareto check; falling back to scalar gate decision."
                )
            else:
                admitted, dominating = self._check_pareto_admission(pareto_axes)
                if not admitted:
                    ids = ", ".join(dominating[:3])
                    more = f" (+{len(dominating) - 3} more)" if len(dominating) > 3 else ""
                    print(
                        f"  [pareto admission] candidate dominated by {ids}{more} "
                        f"on axes {list(self._PARETO_AXES)} — rejecting."
                    )
                    outcome = "reject"
                    gate_reasons = list(gate_reasons) + [f"pareto_dominated_by={dominating[:3]}"]

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
        component_novelty = self._decompose_novelty(
            novelty_type=novelty_type,
            premises_supported=premises_supported,
            alias_tier=alias_tier,
            central_move_prior_art=central_move_prior_art,
            functional_decomposition=functional_decomposition,
            closest_peer_system=closest_peer_system,
        )
        if component_novelty:
            print(
                "  [component-novelty] "
                + " · ".join(f"{k}={v}" for k, v in component_novelty.items())
            )
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
            target_application_domain=target_application_domain,
            known_prior_art_evaluations=list(known_prior_art_evaluations),
            canonical_form=dict(canonical_form),
            component_novelty=dict(component_novelty),
            pareto_axes=pareto_axes,
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
        needs_canonicalization: bool = False,
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
          - needs_canonicalization: only re-verify entries that LACK a populated
            canonical_form (predate the Phase 1 canonicalization layer). Use to
            scope a reverify pass to legacy "dark" entries without re-running
            the heavy verifier on entries already canonicalized.

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
            if needs_canonicalization and (e.get("canonical_form") or {}).get("move_predicate"):
                # Entry already has a populated canonical_form — skip; the
                # caller wants to scope to legacy "dark" entries only.
                continue
            candidates.append(e)

        print(f"\n--- RE-VERIFYING {len(candidates)} register entr(ies) "
              f"(filters: only_ids={bool(only_ids)}, novelty={novelty_set or 'any'}, "
              f"max_conf={max_confidence}, needs_canon={needs_canonicalization}) ---")
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
            reverify_engine_domain = getattr(self.config, "domain", "") or "(unspecified)"
            reverify_known_prior_art = self.journal.match_known_prior_art([reverify_engine_domain])
            # Reverification reuses any pre-computed canonical_form on the
            # register entry as the Stage-1 anchor instead of re-running the
            # canonicalizer (saves an LLM call). If the entry has no
            # canonical_form yet, the context block self-narrates that.
            reverify_canonical_form = e.get("canonical_form") or {}
            reverify_alias_signal = (
                self._alias_gap(reverify_canonical_form, exclude_id=eid)
                if reverify_canonical_form else
                {"gap": 1.0, "nearest_ids": [], "scored_against": 0}
            )
            reverify_alias_tier = "CLEAR"
            if reverify_canonical_form and reverify_alias_signal["scored_against"] > 0:
                gap = reverify_alias_signal["gap"]
                if gap < ALIAS_GAP_STRICT:
                    reverify_alias_tier = "STRICT"
                elif gap < ALIAS_GAP_BAND:
                    reverify_alias_tier = "BAND"
            reverify_register_lookup = {e.get("id", ""): e for e in self.journal.register}
            reverify_canonical_context = self._build_canonical_form_context(
                reverify_canonical_form, reverify_alias_tier, reverify_alias_signal,
                reverify_register_lookup,
            )
            prompt = VERIFY_PROMPT.format(
                insight_json=json.dumps(asdict(insight), indent=2),
                xref_json=json.dumps(asdict(xref), indent=2),
                supporting_entries_json=json.dumps(slim_supporting, indent=2),
                tool_list=self._tool_list_block(for_client=self.verifier),
                prior_human_rejections_json=json.dumps(
                    self.journal.human_rejection_feedback(), indent=2,
                ),
                engine_domain=reverify_engine_domain,
                known_prior_art_json=json.dumps(reverify_known_prior_art, indent=2),
                canonical_form_context=reverify_canonical_context,
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

            # Phase 3: per-component novelty + delta vs the entry's prior
            # component_novelty (if any). Surfaces which architectural
            # components changed status during reverification — finer-grained
            # than just the entry-level verdict flip.
            new_component_novelty = self._decompose_novelty(
                novelty_type=new_novelty,
                premises_supported=bool(result.get("premises_supported", True)),
                alias_tier=reverify_alias_tier,
                central_move_prior_art=list(result.get("central_move_prior_art", []) or []),
                functional_decomposition=list(result.get("functional_decomposition") or []),
                closest_peer_system=dict(result.get("closest_peer_system") or {}),
            )
            component_delta = self._component_novelty_delta(
                e.get("component_novelty") or {}, new_component_novelty,
            )

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
                "new_target_application_domain": (result.get("target_application_domain") or "").strip(),
                "new_known_prior_art_evaluations": list(result.get("known_prior_art_evaluations", []) or []),
                "new_functional_decomposition": list(result.get("functional_decomposition") or []),
                "new_contradicting_findings": list(result.get("contradicting_findings") or []),
                "new_reasoning_flaws": list(result.get("reasoning_flaws") or []),
                "new_verification_summary": (result.get("verification_summary") or "").strip(),
                "new_component_novelty": new_component_novelty,
                "component_novelty_delta": component_delta,
                "tool_calls": list(tool_trace),
                "verdict_changed": changed,
            }
            self.journal.append_register_reverification(eid, log_entry)

            # Additive structural-field population: lift dark legacy entries
            # into Phase 1+ coverage when the reverify produces fresh
            # central_move material. ONLY writes fields that are currently
            # empty — never overwrites a populated structural field. This
            # is consistent with the audit-trail discipline: log preserves
            # old verdicts, and structural fields that didn't exist when
            # the entry was created can be filled in once without violating
            # the no-mutate rule. (To re-canonicalize an already-populated
            # entry, use --backfill-canonical-forms --backfill-force.)
            new_central_move = (result.get("central_architectural_move") or "").strip()
            populated_structural: list[str] = []
            if new_central_move and not (e.get("canonical_form") or {}).get("move_predicate"):
                try:
                    canonical = self._canonicalize_central_move(
                        e.get("title") or "",
                        e.get("description") or "",
                        new_central_move,
                    )
                except Exception as ex:  # noqa: BLE001
                    print(f"  [warn] canonicalization failed during reverify: {type(ex).__name__}: {ex}")
                    canonical = {}
                if canonical:
                    e["canonical_form"] = canonical
                    populated_structural.append("canonical_form")
            if not e.get("component_novelty"):
                e["component_novelty"] = new_component_novelty
                populated_structural.append("component_novelty")
            if not e.get("pareto_axes"):
                # alias_gap for the new canonical_form (if any) computed
                # against the rest of the register, mirroring verify_insight.
                pareto_alias_gap = 1.0
                if e.get("canonical_form", {}).get("move_predicate"):
                    try:
                        ag = self._alias_gap(e["canonical_form"], exclude_id=eid)
                        pareto_alias_gap = ag.get("gap", 1.0)
                    except Exception:  # noqa: BLE001
                        pareto_alias_gap = 1.0
                e["pareto_axes"] = self._compute_pareto_axes(
                    verified_confidence=new_conf,
                    premises_support_citations=result.get("premises_support_citations", []) or [],
                    closest_peer_system=dict(result.get("closest_peer_system") or {}),
                    known_prior_art_evaluations=list(
                        result.get("known_prior_art_evaluations", []) or []
                    ),
                    alias_gap=pareto_alias_gap,
                )
                populated_structural.append("pareto_axes")
            if populated_structural:
                self.journal.save()
                print(f"  [structural] populated: {', '.join(populated_structural)}")

            if component_delta:
                print(
                    "  [component-novelty delta] "
                    + " · ".join(
                        f"{k}: {v['from']}→{v['to']}"
                        for k, v in component_delta.items()
                    )
                )

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

    # ── Three-stage diagnostic (Phase 2 verification harness) ──────────

    def test_three_stage(self, title: str, description: str) -> dict:
        """Run Stages 1+2 of the three-stage verifier on a hypothetical
        claim — without invoking Stage 3 (the heavy phased prior-art
        search) or persisting anything. Used to verify that the
        canonicalization layer extracts cleanly and that alias-gap
        thresholds are tuned correctly against the current register.

        Prints diagnostic output and returns a dict for programmatic
        callers / tests.
        """
        print("\n--- THREE-STAGE TEST (Stages 1+2 only) ---")
        print(f"  Title:       {title}")
        print(f"  Description: {description[:200]}{'…' if len(description) > 200 else ''}")
        print()

        # Stage 1
        print("  [stage 1] canonicalize from raw description…")
        canonical = self._canonicalize_central_move(title, description)
        if not canonical or not canonical.get("move_predicate"):
            print("    [warn] Stage 1 produced an empty canonical form — claim has no clean structural move.")
            return {
                "canonical_form": {},
                "alias_signal": {"gap": 1.0, "nearest_ids": [], "scored_against": 0},
                "tier": "NONE",
            }
        print(f"    move_predicate:  {canonical.get('move_predicate')!r}")
        print(f"    on_substrate:    {canonical.get('on_substrate')!r}")
        print(f"    with_mechanism:  {canonical.get('with_mechanism')!r}")
        print(f"    target_domain:   {canonical.get('target_domain')!r}")
        kc = canonical.get("key_constraints") or []
        if kc:
            print(f"    key_constraints: {kc}")
        print()

        # Stage 2
        print("  [stage 2] alias-gap detection vs canonicalized register…")
        alias_signal = self._alias_gap(canonical)
        gap = alias_signal["gap"]
        scored = alias_signal["scored_against"]
        nearest = alias_signal["nearest_ids"]

        if scored == 0:
            print("    No canonicalized register entries on file — no comparison possible.")
            tier = "CLEAR"
        elif gap < ALIAS_GAP_STRICT and nearest:
            tier = "STRICT"
        elif gap < ALIAS_GAP_BAND and nearest:
            tier = "BAND"
        else:
            tier = "CLEAR"
        print(f"    scored_against: {scored} canonicalized register entr{'y' if scored == 1 else 'ies'}")
        print(f"    gap:            {gap:.3f}")
        print(f"    tier:           {tier}  (STRICT < {ALIAS_GAP_STRICT}, BAND < {ALIAS_GAP_BAND})")
        if nearest:
            register_lookup = {e.get("id", ""): e for e in self.journal.register}
            print(f"    nearest_ids:    {nearest}")
            for rid in nearest[:3]:
                peer = register_lookup.get(rid, {})
                pcanonical = peer.get("canonical_form") or {}
                ptext = " ".join(filter(None, [
                    pcanonical.get("move_predicate"),
                    pcanonical.get("on_substrate"),
                    pcanonical.get("with_mechanism"),
                ]))
                ptitle = (peer.get("title") or "")[:80]
                print(f"      {rid}: '{ptext}' — {ptitle}")

        print()
        if tier == "STRICT":
            print("  → Stage 3 would be SKIPPED. Synthetic restatement result emitted.")
        elif tier == "BAND":
            print("  → Stage 3 would run with differentiator-seeking context primed for the named peers.")
        else:
            print("  → Stage 3 would run normally (no alias context; only the pre-extracted canonical form).")

        return {
            "canonical_form": canonical,
            "alias_signal": alias_signal,
            "tier": tier,
        }

    # ── Canonical-form backfill (Phase 1 maintenance pass) ──────────────

    def backfill_canonical_forms(self, force: bool = False) -> dict:
        """One-shot maintenance pass — populate `canonical_form` on every
        active register entry that lacks one. Safe to interrupt and re-run:
        entries already canonicalized are skipped (unless `force=True`),
        so the operation is idempotent and incremental.

        Also fills `pareto_axes` if the entry is missing those (Phase 0
        wrote them, so older entries lack them; backfilling them here lets
        Phase 4's admission gate flip on cleanly when it ships).

        `force=True` re-canonicalizes EVERY active entry, overwriting
        existing canonical_form values. Use after the canonicalization
        prompt has been revised — the old form was generated under a
        prior prompt and may no longer match the new prompt's slot
        conventions, so re-canonicalizing is the only way to keep
        cross-entry comparisons honest.

        Returns a stats dict for logging / programmatic callers.
        """
        register = self.journal.register
        if force:
            candidates = [e for e in register if e.get("status") == "active"]
        else:
            candidates = [
                e for e in register
                if e.get("status") == "active" and not (e.get("canonical_form") or {})
            ]
        print(
            f"\n--- BACKFILL CANONICAL FORMS ---\n"
            f"  register: {len(register)} total · {len(candidates)} need canonicalization"
        )
        if not candidates:
            print("  nothing to do.")
            return {"scanned": len(register), "canonicalized": 0, "skipped": 0, "errors": 0}

        stats = {"scanned": len(register), "canonicalized": 0, "skipped": 0, "errors": 0}
        for i, entry in enumerate(candidates, 1):
            rid = entry.get("id", "?")
            title = (entry.get("title") or "")[:90]
            move = (entry.get("central_architectural_move") or "").strip()
            print(f"\n  [{i}/{len(candidates)}] {rid}: {title}")
            if not move:
                print("    [skip] no central_architectural_move on this entry")
                stats["skipped"] += 1
                continue
            try:
                canonical = self._canonicalize_central_move(
                    entry.get("title") or "",
                    entry.get("description") or "",
                    move,
                )
            except Exception as e:  # noqa: BLE001
                print(f"    [error] {type(e).__name__}: {e}")
                stats["errors"] += 1
                continue
            if not canonical:
                print("    [skip] canonicalization produced empty form")
                stats["skipped"] += 1
                continue

            # Compute alias signal against entries already canonicalized
            # (i.e. processed earlier in this pass + any populated
            # going-forward by verify_insight).
            try:
                alias_signal = self._alias_gap(canonical, exclude_id=rid)
            except Exception as e:  # noqa: BLE001
                print(f"    [warn] alias_gap failed: {type(e).__name__}: {e}")
                alias_signal = {"gap": 1.0, "nearest_ids": [], "scored_against": 0}

            gap = alias_signal["gap"]
            nearest = alias_signal["nearest_ids"]
            if alias_signal["scored_against"] > 0 and gap < ALIAS_GAP_BAND and nearest:
                canonical["aliasing_against"] = nearest
                tier = "STRICT" if gap < ALIAS_GAP_STRICT else "BAND"
                print(f"    [alias-gap] {tier} gap={gap:.2f} vs {nearest}")

            # Persist directly on the entry dict (Journal.register is a list
            # of dicts post-load). Backfill `pareto_axes` opportunistically
            # too; the values come from already-stored fields, no LLM.
            entry["canonical_form"] = canonical
            # Phase 3: populate component_novelty deterministically from
            # already-stored verifier outputs. No LLM call needed; backfill
            # always overwrites because the rule logic itself may have
            # evolved between commits.
            entry["component_novelty"] = self._decompose_novelty(
                novelty_type=(entry.get("novelty_type") or "").strip(),
                premises_supported=bool(entry.get("premises_supported", True)),
                alias_tier=(
                    "STRICT" if (gap < ALIAS_GAP_STRICT and nearest)
                    else "BAND" if (gap < ALIAS_GAP_BAND and nearest)
                    else "CLEAR"
                ),
                central_move_prior_art=list(
                    entry.get("central_move_prior_art", []) or []
                ),
                functional_decomposition=list(
                    entry.get("functional_decomposition") or []
                ),
                closest_peer_system=dict(entry.get("closest_peer_system") or {}),
            )
            if entry["component_novelty"]:
                print(
                    "    component_novelty: "
                    + " · ".join(
                        f"{k}={v}" for k, v in entry["component_novelty"].items()
                    )[:200]
                )
            if not entry.get("pareto_axes") or force:
                entry["pareto_axes"] = self._compute_pareto_axes(
                    verified_confidence=float(entry.get("verified_confidence", 0.0)),
                    premises_support_citations=entry.get("premises_support_citations", []) or [],
                    closest_peer_system=entry.get("closest_peer_system") or {},
                    known_prior_art_evaluations=entry.get("known_prior_art_evaluations") or [],
                    alias_gap=gap,
                )
            stats["canonicalized"] += 1

        # Persist once at the end — the journal save is atomic so we don't
        # want N writes for N entries.
        self.journal.save()
        print(
            f"\nBackfill complete: canonicalized={stats['canonicalized']}  "
            f"skipped={stats['skipped']}  errors={stats['errors']}"
        )
        return stats
