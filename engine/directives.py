"""Research directives export — translate verified register entries into
concrete, executable research plans.

Architecture: a small multi-call harness. Each section of the directive
is either deterministic (pure restructuring of existing register fields —
Theory, Prior Art Positioning, References) or produced by a FOCUSED LLM
call bounded at ≤1500 output tokens (Hypothesis, Test Plan, Agentic
Prompt, Verification Criteria). Sections are then stitched into a
markdown template deterministically.

Why not one big call: the earlier monolithic design asked the primary
for a huge JSON-wrapped markdown payload (≤8192 tokens out), which
non-streaming models silently hang on for 10+ minutes. Small focused
calls finish in 20–60s each, failures localise to one section, and
progress is visible in the run log.

Grounding discipline preserved: the agentic-prompt call receives the
citation and tool allowlists and is forbidden from inventing either.
A final verifier pass reviews the assembled markdown for any fabrication
the per-section calls smuggled through. If flagged, one retry of only
the agentic-prompt section (the riskiest) is attempted. Further failures
ship an annotated "⚠ FLAGGED ISSUES" block so the human reader sees
concerns before executing.

Domain-neutrality: every prompt is structural — no field-specific
examples, no illustrative domains. Every LLM call threads `{engine_domain}`
so the model anchors on the journal's actual subject, not a memorised
default.

Output destinations:
- Per-record: data/{journal_stem}_directives/r-{id}.md + .json sidecar
- Bundle:     data/{journal_stem}_directives/bundle-{timestamp}.md + .json
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from prompts import (
    DIRECTIVE_AGENTIC_PROMPT_PROMPT,
    DIRECTIVE_HYPOTHESIS_PROMPT,
    DIRECTIVE_TEST_PLAN_PROMPT,
    DIRECTIVE_VERIFICATION_CRITERIA_PROMPT,
    DIRECTIVE_VERIFIER_PROMPT,
)


# Tool names allowlisted for inclusion in the agentic prompt. Domain-neutral
# by design — engine tools are all keyless / subject-agnostic, and the
# orchestrator primitives (Bash, Read, Write, Edit, Glob, Grep) describe
# structure not content. Unknown names in the generated prompt are flagged
# by the verifier as fabrications.
_AGENT_TOOL_ALLOWLIST = [
    "web_search",
    "web_fetch",
    "academic_search",
    "archive_access",
    "citation_manager",
    "peer_review",
    "calculator",
    "code_execution",
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
]

# Per-section max output tokens — tight caps prevent the hang we saw on
# the monolithic 8192-token call. Sections are short; these are generous.
_SECTION_MAX_TOKENS = 1500
_AGENTIC_PROMPT_MAX_TOKENS = 2500


class DirectivesMixin:
    """Mixin attached to CuriosityEngine — research-directive export pipeline."""

    def qualifying_register_entries(self) -> list[dict]:
        """Entries eligible for export: active + effective verdict validated +
        at least one attached prediction still open (not confirmed, refuted,
        already_fulfilled, or expired).
        """
        register = self.journal.register
        preds_by_entry: dict[str, list[dict]] = {}
        for p in self.journal.predictions:
            rid = p.get("register_entry_id")
            if rid:
                preds_by_entry.setdefault(rid, []).append(p)

        qualifying: list[dict] = []
        for r in register:
            if r.get("status") != "active":
                continue
            rv_log = r.get("reverification_log") or []
            eff_verdict = (rv_log[-1].get("new_verdict") if rv_log else r.get("verdict", "")).lower()
            if eff_verdict != "validated":
                continue
            preds = preds_by_entry.get(r.get("id", ""), [])
            open_preds = [
                p for p in preds
                if (p.get("status") or "pending").lower() not in (
                    "confirmed", "refuted", "already_fulfilled", "expired",
                )
            ]
            if not open_preds:
                continue
            qualifying.append({**r, "_attached_predictions": preds, "_open_predictions": open_preds})
        return qualifying

    def _citations_for_entry(self, entry: dict) -> list[str]:
        """Allowlist of citations the primary is allowed to reference in the
        directive. Built from every URL/DOI/identifier the source material
        already contains. No invented strings get through."""
        seen: set[str] = set()
        out: list[str] = []

        def _add(s):
            if not s:
                return
            s = str(s).strip()
            if not s or s in seen:
                return
            seen.add(s)
            out.append(s)

        for k in (
            "premises_support_citations",
            "synthesis_prior_art",
            "prior_art_citations",
            "central_move_prior_art",
            "contradicting_findings",
            "supporting_sources",
        ):
            for v in entry.get(k, []) or []:
                _add(v)
        peer = entry.get("closest_peer_system") or {}
        _add(peer.get("url"))
        for d in (entry.get("functional_decomposition") or []):
            _add(d.get("nearest_exemplar"))
        for rv in (entry.get("reverification_log") or []):
            for v in (rv.get("new_central_move_prior_art") or []):
                _add(v)
            _add((rv.get("new_closest_peer_system") or {}).get("url"))
        return out

    # ── Section generators ──────────────────────────────────────────

    def _section_hypothesis(
        self, entry_for_prompt: dict, engine_domain: str, predictions: list[dict],
    ) -> str:
        prompt = DIRECTIVE_HYPOTHESIS_PROMPT.format(
            engine_domain=engine_domain,
            register_entry_json=json.dumps(entry_for_prompt, indent=2),
            predictions_json=json.dumps(predictions, indent=2),
        )
        result = self._call_primary(prompt, max_tokens=_SECTION_MAX_TOKENS)
        return (result.get("hypothesis") or "").strip()

    def _section_test_plan(
        self, entry_for_prompt: dict, engine_domain: str,
        predictions: list[dict], hypothesis: str,
    ) -> list[dict]:
        prompt = DIRECTIVE_TEST_PLAN_PROMPT.format(
            engine_domain=engine_domain,
            register_entry_json=json.dumps(entry_for_prompt, indent=2),
            predictions_json=json.dumps(predictions, indent=2),
            hypothesis=hypothesis or "(none generated)",
        )
        result = self._call_primary(prompt, max_tokens=_SECTION_MAX_TOKENS)
        return list(result.get("steps") or [])

    def _section_agentic_prompt(
        self, entry_for_prompt: dict, engine_domain: str,
        hypothesis: str, test_plan: list[dict],
        citations: list[str], tool_allowlist: list[str],
    ) -> dict:
        prompt = DIRECTIVE_AGENTIC_PROMPT_PROMPT.format(
            engine_domain=engine_domain,
            register_entry_json=json.dumps(entry_for_prompt, indent=2),
            hypothesis=hypothesis or "(none generated)",
            test_plan_json=json.dumps(test_plan, indent=2),
            citations_json=json.dumps(citations, indent=2),
            tool_allowlist_json=json.dumps(tool_allowlist, indent=2),
        )
        result = self._call_primary(prompt, max_tokens=_AGENTIC_PROMPT_MAX_TOKENS)
        return {
            "agentic_prompt": (result.get("agentic_prompt") or "").strip(),
            "tool_names_used": list(result.get("tool_names_used") or []),
            "citations_used": list(result.get("citations_used") or []),
            "unresolved_dependencies": list(result.get("unresolved_dependencies") or []),
        }

    def _section_verification_criteria(
        self, entry_for_prompt: dict, engine_domain: str,
        predictions: list[dict], hypothesis: str,
    ) -> dict:
        prompt = DIRECTIVE_VERIFICATION_CRITERIA_PROMPT.format(
            engine_domain=engine_domain,
            register_entry_json=json.dumps(entry_for_prompt, indent=2),
            predictions_json=json.dumps(predictions, indent=2),
            hypothesis=hypothesis or "(none generated)",
        )
        result = self._call_primary(prompt, max_tokens=_SECTION_MAX_TOKENS)
        return {
            "confirmed": (result.get("confirmed") or "").strip(),
            "refuted": (result.get("refuted") or "").strip(),
            "inconclusive": (result.get("inconclusive") or "").strip(),
        }

    # ── Deterministic assembly ──────────────────────────────────────

    def _theory_section(self, entry: dict) -> str:
        title = entry.get("title", "").strip()
        desc = entry.get("description", "").strip()
        premises = entry.get("premises_support_citations", []) or []
        lines = [desc] if desc else []
        if premises:
            lines.append("")
            lines.append("**Load-bearing premises** (with supporting citations):")
            for p in premises[:10]:
                lines.append(f"- {p}")
        return "\n".join(lines) if lines else f"(description unavailable for {title})"

    def _positioning_section(self, entry: dict) -> str:
        peer = entry.get("closest_peer_system") or {}
        fd = entry.get("functional_decomposition") or []
        lines: list[str] = []
        name = (peer.get("name") or "").strip()
        url = (peer.get("url") or "").strip()
        overlap = (peer.get("overlap_summary") or "").strip()
        diffs = peer.get("differentiators") or []
        if name:
            header = f"**Closest peer system:** {name}"
            if url:
                header += f" ({url})"
            lines.append(header)
            if overlap:
                lines.append("")
                lines.append(overlap)
            if diffs:
                lines.append("")
                lines.append("**Differentiators** — ways this theory departs from the peer:")
                for d in diffs:
                    lines.append(f"- {d}")
        if fd:
            lines.append("")
            lines.append("**Functional decomposition** — positioning along independent dimensions:")
            lines.append("")
            lines.append("| Dimension | Nearest exemplar | How this theory differs |")
            lines.append("| --- | --- | --- |")
            for d in fd:
                dim = (d.get("dimension") or "?").strip()
                ex = (d.get("nearest_exemplar") or "—").strip().replace("|", "\\|")
                diff = (d.get("how_ours_differs") or "—").strip().replace("|", "\\|")
                lines.append(f"| {dim} | {ex} | {diff} |")
        if not lines:
            lines.append("(no peer system identified by the verifier)")
        return "\n".join(lines)

    def _references_section(self, citations: list[str]) -> str:
        if not citations:
            return "_(no citations provided on this register entry)_"
        return "\n".join(f"- {c}" for c in citations)

    def _assemble_markdown(
        self,
        entry: dict,
        hypothesis: str,
        test_plan: list[dict],
        agentic: dict,
        criteria: dict,
        citations: list[str],
        flags: Optional[list[str]] = None,
    ) -> str:
        rid = entry.get("id", "unknown")
        title = entry.get("title", "").strip() or rid
        verdict = (entry.get("verdict") or "").strip()
        novelty = (entry.get("novelty_type") or "").strip()
        conf = entry.get("verified_confidence", 0.0)

        parts: list[str] = []
        if flags:
            parts.append(
                "> ⚠ **FLAGGED ISSUES** — the verifier found the following concerns. "
                "Read these before executing any step below; some items may be fabrications."
            )
            parts.append("")
            for f in flags:
                parts.append(f"> - {f}")
            parts.append("")
            parts.append("---")
            parts.append("")

        parts.append(f"# {title}")
        parts.append("")
        parts.append(
            f"> **Source register entry**: `{rid}` · "
            f"**Verdict**: {verdict} · **Novelty**: {novelty} · "
            f"**Confidence**: {conf:.2f}"
        )
        parts.append("")

        parts.append("## Theory")
        parts.append(self._theory_section(entry))
        parts.append("")

        parts.append("## Hypothesis")
        parts.append(hypothesis or "_(generator returned no hypothesis)_")
        parts.append("")

        parts.append("## Prior Art Positioning")
        parts.append(self._positioning_section(entry))
        parts.append("")

        parts.append("## Test Plan")
        if test_plan:
            for step in test_plan:
                n = step.get("n", "?")
                inp = (step.get("input") or "").strip()
                act = (step.get("action") or "").strip()
                out = (step.get("output") or "").strip()
                parts.append(f"{n}. **Input**: {inp}  ")
                parts.append(f"   **Action**: {act}  ")
                parts.append(f"   **Output**: {out}")
        else:
            parts.append("_(generator returned no test plan)_")
        parts.append("")

        parts.append("## Agentic Prompt")
        parts.append("")
        parts.append("```")
        parts.append(agentic.get("agentic_prompt") or "(generator returned no agentic prompt)")
        parts.append("```")
        if agentic.get("unresolved_dependencies"):
            parts.append("")
            parts.append("**Unresolved dependencies flagged by the generator:**")
            for u in agentic["unresolved_dependencies"]:
                parts.append(f"- {u}")
        parts.append("")

        parts.append("## Verification Criteria")
        parts.append("")
        parts.append("| Outcome | Observable signal |")
        parts.append("| --- | --- |")
        parts.append(f"| Confirmed | {(criteria.get('confirmed') or '—').replace(chr(124), chr(92) + chr(124))} |")
        parts.append(f"| Refuted | {(criteria.get('refuted') or '—').replace(chr(124), chr(92) + chr(124))} |")
        parts.append(f"| Inconclusive | {(criteria.get('inconclusive') or '—').replace(chr(124), chr(92) + chr(124))} |")
        parts.append("")

        parts.append("## References")
        parts.append(self._references_section(citations))
        parts.append("")

        return "\n".join(parts)

    # ── Pipeline orchestration ──────────────────────────────────────

    def _heartbeat(self, label: str, t0: float):
        elapsed = time.monotonic() - t0
        print(f"  [{elapsed:5.1f}s] {label}")

    def _run_directive_pipeline(self, entry: dict) -> dict:
        """Multi-section synthesis → verifier review → selective retry.

        Returns: {markdown, verdict, verifier_report, flagged_issues, register_entry_id}.
        """
        entry_id = entry.get("id", "unknown")
        t0 = time.monotonic()
        print(f"\n--- DIRECTIVE PIPELINE for {entry_id} ---")

        engine_domain = getattr(self.config, "domain", "") or "(unspecified)"
        citations = self._citations_for_entry(entry)
        tool_allowlist = list(_AGENT_TOOL_ALLOWLIST)
        entry_for_prompt = {
            k: v for k, v in entry.items()
            if k not in ("reverification_log", "verification_tool_calls")
            and not k.startswith("_")
        }
        predictions = entry.get("_attached_predictions", []) or []

        # Section 1: hypothesis
        self._heartbeat("generating hypothesis (1/4)", t0)
        try:
            hypothesis = self._section_hypothesis(entry_for_prompt, engine_domain, predictions)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] hypothesis generation failed: {type(e).__name__}: {e}")
            hypothesis = ""

        # Section 2: test plan (conditioned on hypothesis)
        self._heartbeat("generating test plan (2/4)", t0)
        try:
            test_plan = self._section_test_plan(
                entry_for_prompt, engine_domain, predictions, hypothesis,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [error] test plan generation failed: {type(e).__name__}: {e}")
            test_plan = []

        # Section 3: agentic prompt (conditioned on hypothesis + test plan)
        self._heartbeat("generating agentic prompt (3/4)", t0)
        try:
            agentic = self._section_agentic_prompt(
                entry_for_prompt, engine_domain, hypothesis, test_plan,
                citations, tool_allowlist,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [error] agentic prompt generation failed: {type(e).__name__}: {e}")
            agentic = {
                "agentic_prompt": "",
                "tool_names_used": [],
                "citations_used": [],
                "unresolved_dependencies": [f"generation failed: {type(e).__name__}"],
            }

        # Section 4: verification criteria
        self._heartbeat("generating verification criteria (4/4)", t0)
        try:
            criteria = self._section_verification_criteria(
                entry_for_prompt, engine_domain, predictions, hypothesis,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [error] verification criteria generation failed: {type(e).__name__}: {e}")
            criteria = {"confirmed": "", "refuted": "", "inconclusive": ""}

        # Assemble markdown (deterministic)
        self._heartbeat("assembling markdown", t0)
        markdown = self._assemble_markdown(
            entry, hypothesis, test_plan, agentic, criteria, citations,
        )

        # Verifier review pass
        self._heartbeat("running verifier review", t0)
        footer = {
            "title": (entry.get("title") or "").strip(),
            "tool_names_used": agentic.get("tool_names_used", []),
            "citations_used": agentic.get("citations_used", []),
            "unresolved_dependencies": agentic.get("unresolved_dependencies", []),
        }
        verify_prompt = DIRECTIVE_VERIFIER_PROMPT.format(
            register_entry_json=json.dumps(entry_for_prompt, indent=2),
            citations_json=json.dumps(citations, indent=2),
            tool_allowlist_json=json.dumps(tool_allowlist, indent=2),
            directive_markdown=markdown,
            directive_footer_json=json.dumps(footer, indent=2),
        )
        try:
            report = self._call_verifier(verify_prompt, max_tokens=self.connection.verifier.max_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] verifier review failed: {type(e).__name__}: {e}")
            report = {
                "ok": False, "severity": "fatal",
                "overall_assessment": f"verifier raised {type(e).__name__}: {e}",
                "unlisted_citations": [], "unlisted_tools": [],
                "handwave_steps": [], "non_measurable_criteria": [],
                "self_declaration_mismatches": [],
            }

        sev = (report.get("severity") or "").strip().lower()
        if report.get("ok") or sev == "clean":
            self._heartbeat("verifier: ✓ clean", t0)
            return {
                "markdown": markdown,
                "verdict": "clean",
                "verifier_report": report,
                "flagged_issues": [],
                "register_entry_id": entry_id,
            }

        flags = _collect_flags(report)
        self._heartbeat(f"verifier flagged {len(flags)} issue(s) — retrying agentic prompt only", t0)

        # Selective retry: the agentic prompt is the section most prone to
        # grounding violations; regenerating just it (with the verifier flags
        # appended to its prompt) is cheaper than regenerating everything.
        try:
            retry_prompt = DIRECTIVE_AGENTIC_PROMPT_PROMPT.format(
                engine_domain=engine_domain,
                register_entry_json=json.dumps(entry_for_prompt, indent=2),
                hypothesis=hypothesis,
                test_plan_json=json.dumps(test_plan, indent=2),
                citations_json=json.dumps(citations, indent=2),
                tool_allowlist_json=json.dumps(tool_allowlist, indent=2),
            ) + (
                "\n\nYOUR PRIOR OUTPUT FAILED GROUNDING REVIEW. Flags from the verifier:\n"
                + json.dumps(flags, indent=2)
                + "\n\nRegenerate the agentic prompt addressing EACH flag. In particular:\n"
                "- Remove any non-allowlist citation; replace with an allowlist citation or mark 'UNRESOLVED: ...'.\n"
                "- Replace any non-allowlist tool with an allowlist tool, or say so in unresolved_dependencies.\n"
                "- Rewrite hand-wave steps into concrete executable tool calls."
            )
            retry_result = self._call_primary(retry_prompt, max_tokens=_AGENTIC_PROMPT_MAX_TOKENS)
            agentic_retry = {
                "agentic_prompt": (retry_result.get("agentic_prompt") or "").strip(),
                "tool_names_used": list(retry_result.get("tool_names_used") or []),
                "citations_used": list(retry_result.get("citations_used") or []),
                "unresolved_dependencies": list(retry_result.get("unresolved_dependencies") or []),
            }
        except Exception as e:  # noqa: BLE001
            print(f"  [error] agentic retry failed: {type(e).__name__}: {e}")
            agentic_retry = agentic  # keep the original

        markdown_retry = self._assemble_markdown(
            entry, hypothesis, test_plan, agentic_retry, criteria, citations,
        )

        # Re-verify the retry
        footer_retry = {
            "title": (entry.get("title") or "").strip(),
            "tool_names_used": agentic_retry.get("tool_names_used", []),
            "citations_used": agentic_retry.get("citations_used", []),
            "unresolved_dependencies": agentic_retry.get("unresolved_dependencies", []),
        }
        verify_prompt_retry = DIRECTIVE_VERIFIER_PROMPT.format(
            register_entry_json=json.dumps(entry_for_prompt, indent=2),
            citations_json=json.dumps(citations, indent=2),
            tool_allowlist_json=json.dumps(tool_allowlist, indent=2),
            directive_markdown=markdown_retry,
            directive_footer_json=json.dumps(footer_retry, indent=2),
        )
        try:
            report_retry = self._call_verifier(verify_prompt_retry, max_tokens=self.connection.verifier.max_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] verifier retry failed: {type(e).__name__}: {e}")
            report_retry = {"ok": False, "severity": "fatal"}

        sev_retry = (report_retry.get("severity") or "").strip().lower()
        if report_retry.get("ok") or sev_retry == "clean":
            self._heartbeat("verifier (retry): ✓ clean", t0)
            return {
                "markdown": markdown_retry,
                "verdict": "clean",
                "verifier_report": report_retry,
                "flagged_issues": [],
                "register_entry_id": entry_id,
            }

        flags_retry = _collect_flags(report_retry)
        self._heartbeat(f"verifier (retry) still flagged {len(flags_retry)} — annotating output", t0)
        annotated = self._assemble_markdown(
            entry, hypothesis, test_plan, agentic_retry, criteria, citations,
            flags=flags_retry,
        )
        return {
            "markdown": annotated,
            "verdict": "needs_fixes" if sev_retry != "fatal" else "fatal",
            "verifier_report": report_retry,
            "flagged_issues": flags_retry,
            "register_entry_id": entry_id,
        }

    # ── Export entry points (unchanged from prior shape) ────────────

    def _directives_dir(self) -> Path:
        journal_path = Path(self.config.journal_path)
        stem = journal_path.stem
        d = journal_path.parent / f"{stem}_directives"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def export_directive_for(self, register_entry_id: str) -> dict:
        entry = next(
            (r for r in self.journal.register if r.get("id") == register_entry_id),
            None,
        )
        if entry is None:
            raise ValueError(f"register entry {register_entry_id} not found")
        preds = [p for p in self.journal.predictions if p.get("register_entry_id") == register_entry_id]
        open_preds = [
            p for p in preds
            if (p.get("status") or "pending").lower() not in (
                "confirmed", "refuted", "already_fulfilled", "expired",
            )
        ]
        enriched = {**entry, "_attached_predictions": preds, "_open_predictions": open_preds}
        result = self._run_directive_pipeline(enriched)
        out_dir = self._directives_dir()
        md_path = out_dir / f"{register_entry_id}.md"
        sidecar_path = out_dir / f"{register_entry_id}.verification.json"
        md_path.write_text(result["markdown"] or f"# Directive generation failed for {register_entry_id}\n")
        sidecar_path.write_text(json.dumps({
            "register_entry_id": register_entry_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "verdict": result["verdict"],
            "verifier_report": result.get("verifier_report", {}),
            "flagged_issues": result["flagged_issues"],
        }, indent=2))
        print(
            f"  [saved] {md_path} · verdict={result['verdict']} · "
            f"flags={len(result['flagged_issues'])}"
        )
        return {
            "path": str(md_path),
            "sidecar_path": str(sidecar_path),
            "verdict": result["verdict"],
            "flagged_issues_count": len(result["flagged_issues"]),
            "register_entry_id": register_entry_id,
        }

    def export_directives_bundle(self) -> dict:
        qualifying = self.qualifying_register_entries()
        if not qualifying:
            print("  [bundle] no qualifying register entries.")
            return {"path": "", "entries_included": 0, "entries_flagged": 0}

        print(f"\n--- DIRECTIVES BUNDLE: {len(qualifying)} qualifying entr(ies) ---")
        sections: list[str] = []
        sidecar_entries: list[dict] = []
        flagged_count = 0

        for entry in qualifying:
            result = self._run_directive_pipeline(entry)
            sections.append(result["markdown"] or f"<!-- {entry.get('id')}: generation failed -->")
            sidecar_entries.append({
                "register_entry_id": entry.get("id"),
                "verdict": result["verdict"],
                "flagged_issues": result["flagged_issues"],
            })
            if result["verdict"] != "clean":
                flagged_count += 1

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = self._directives_dir()
        md_path = out_dir / f"bundle-{ts}.md"
        sidecar_path = out_dir / f"bundle-{ts}.verification.json"

        header = (
            f"# Research Directives Bundle · {ts}\n\n"
            f"Domain: **{getattr(self.config, 'domain', '')}**  \n"
            f"Entries included: **{len(qualifying)}**  \n"
            f"Entries flagged by verifier: **{flagged_count}**\n\n"
            "---\n\n"
        )
        md_path.write_text(header + "\n\n---\n\n".join(sections))
        sidecar_path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "domain": getattr(self.config, "domain", ""),
            "entries": sidecar_entries,
            "entries_flagged": flagged_count,
        }, indent=2))
        print(f"  [saved] {md_path} · flagged={flagged_count}/{len(qualifying)}")
        return {
            "path": str(md_path),
            "sidecar_path": str(sidecar_path),
            "entries_included": len(qualifying),
            "entries_flagged": flagged_count,
        }


def _collect_flags(report: dict) -> list[str]:
    flags: list[str] = []
    for key, label in [
        ("unlisted_citations", "Citation not in allowlist"),
        ("unlisted_tools", "Tool not in allowlist"),
        ("handwave_steps", "Hand-wave step"),
        ("non_measurable_criteria", "Non-measurable criterion"),
        ("self_declaration_mismatches", "Footer/markdown mismatch"),
    ]:
        for item in (report.get(key) or []):
            flags.append(f"{label}: {item}")
    return flags
