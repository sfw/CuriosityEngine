"""Research directives export — translate verified register entries into
concrete, executable research plans.

Two modes: per-record (one directive at a time, fast, on-demand via register
card button) and bundle (all qualifying entries in one document, slower,
admin-triggered). Both share the same primary-synthesize → verifier-review
pipeline with one retry loop.

Grounding discipline is enforced at two layers: the primary's prompt pins
it to provided citations + tool allowlists, and the verifier catches any
fabrication that slipped through. A directive that fails verification gets
one retry with the flags appended; if the retry still fails, a prominent
"⚠ FLAGGED ISSUES" block is prepended to the output so a human reader
notices BEFORE executing.

Output destinations:
- Per-record: data/{journal_stem}_directives/r-{id}.md + .json sidecar
- Bundle:     data/{journal_stem}_directives/bundle-{timestamp}.md + .json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from prompts import DIRECTIVE_SYNTHESIS_PROMPT, DIRECTIVE_VERIFIER_PROMPT


# Tool names allowlisted for inclusion in the agentic prompt. Keep in sync
# with the engine's tool registry + any MCP/Claude Code tools you want the
# directive to reference. Unknown tool names in the generated prompt are
# flagged by the verifier as fabrications.
_AGENT_TOOL_ALLOWLIST = [
    # Engine tools (all keyless)
    "web_search",
    "web_fetch",
    "academic_search",
    "archive_access",
    "citation_manager",
    "peer_review",
    "calculator",
    "code_execution",
    # Common agent orchestrators an executor may have access to
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
]


class DirectivesMixin:
    """Mixin attached to CuriosityEngine — research-directive export pipeline."""

    def qualifying_register_entries(self) -> list[dict]:
        """Entries eligible for export:
          - Audited AND last reverification verdict is validated, OR unaudited with verdict=validated
          - At least one attached prediction that is NOT confirmed/refuted/already_fulfilled
            (i.e. at least one open question remains)
          - Status is active (not held)
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
            # Effective verdict: last re-verification's new_verdict if present,
            # else the stored verdict.
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
        already contains — premises citations, synthesis prior art, supporting
        entry sources, peer-system URLs, etc. No invented strings get through."""
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
        # Re-verification artefacts
        for rv in (entry.get("reverification_log") or []):
            for v in (rv.get("new_central_move_prior_art") or []):
                _add(v)
            _add((rv.get("new_closest_peer_system") or {}).get("url"))
        return out

    def _run_directive_pipeline(self, entry: dict) -> dict:
        """Run the primary → verifier → retry loop for ONE register entry.

        Returns: {
          markdown: str,
          verdict: {"clean"|"needs_fixes"|"fatal"},
          verifier_reports: list[dict],   # one per attempt (1 or 2)
          flagged_issues: list[str],      # pulled from the final verifier report
          register_entry_id: str,
        }
        """
        entry_id = entry.get("id", "unknown")
        print(f"\n--- DIRECTIVE PIPELINE for {entry_id} ---")

        engine_domain = getattr(self.config, "domain", "") or "(unspecified)"
        citations = self._citations_for_entry(entry)
        tool_allowlist = list(_AGENT_TOOL_ALLOWLIST)
        # Strip heavy re-verification log before passing to primary — it's
        # mostly audit trail that doesn't help synthesis.
        entry_for_prompt = {
            k: v for k, v in entry.items()
            if k not in ("reverification_log", "verification_tool_calls")
            and not k.startswith("_")
        }

        verifier_reports: list[dict] = []
        feedback_for_retry: Optional[dict] = None
        final_markdown = ""

        for attempt in range(2):
            print(f"  [attempt {attempt+1}/2] primary synthesising directive...")
            prompt = DIRECTIVE_SYNTHESIS_PROMPT.format(
                engine_domain=engine_domain,
                register_entry_json=json.dumps(entry_for_prompt, indent=2),
                predictions_json=json.dumps(entry.get("_attached_predictions", []), indent=2),
                citations_json=json.dumps(citations, indent=2),
                tool_allowlist_json=json.dumps(tool_allowlist, indent=2),
            )
            if feedback_for_retry:
                prompt += (
                    "\n\nYOUR PRIOR ATTEMPT FAILED VERIFICATION. Flags from the verifier:\n"
                    + json.dumps(feedback_for_retry, indent=2)
                    + "\n\nRegenerate the directive addressing EACH flag. In particular:\n"
                    "- Remove any citation not in the allowlist; replace with a legitimate citation or state the gap.\n"
                    "- Replace any non-allowlist tool with an allowlist tool or say in unresolved_dependencies that no suitable tool exists.\n"
                    "- Rewrite hand-wave steps into concrete executable ones.\n"
                    "- Rewrite non-measurable verification criteria with numerical thresholds or specific observable signals."
                )
            try:
                synth_result = self._call_primary(
                    prompt,
                    max_tokens=self.connection.primary.investigation_max_tokens,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [primary error] {type(e).__name__}: {e}")
                return {
                    "markdown": "",
                    "verdict": "fatal",
                    "verifier_reports": verifier_reports,
                    "flagged_issues": [f"primary synthesis failed: {type(e).__name__}: {e}"],
                    "register_entry_id": entry_id,
                }

            markdown = (synth_result.get("markdown") or "").strip()
            footer = {
                "title": synth_result.get("title", ""),
                "tool_names_used": synth_result.get("tool_names_used", []),
                "citations_used": synth_result.get("citations_used", []),
                "unresolved_dependencies": synth_result.get("unresolved_dependencies", []),
            }
            if not markdown:
                print("  [primary] returned empty markdown — bailing.")
                return {
                    "markdown": "",
                    "verdict": "fatal",
                    "verifier_reports": verifier_reports,
                    "flagged_issues": ["primary returned empty markdown"],
                    "register_entry_id": entry_id,
                }

            final_markdown = markdown

            print("  [verifier] reviewing for fabrication and hand-waves...")
            verify_prompt = DIRECTIVE_VERIFIER_PROMPT.format(
                register_entry_json=json.dumps(entry_for_prompt, indent=2),
                citations_json=json.dumps(citations, indent=2),
                tool_allowlist_json=json.dumps(tool_allowlist, indent=2),
                directive_markdown=markdown,
                directive_footer_json=json.dumps(footer, indent=2),
            )
            try:
                report = self._call_verifier(
                    verify_prompt,
                    max_tokens=self.connection.verifier.max_tokens,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [verifier error] {type(e).__name__}: {e}")
                report = {
                    "ok": False,
                    "severity": "fatal",
                    "overall_assessment": f"verifier raised {type(e).__name__}: {e}",
                    "unlisted_citations": [],
                    "unlisted_tools": [],
                    "handwave_steps": [],
                    "non_measurable_criteria": [],
                    "self_declaration_mismatches": [],
                }
            verifier_reports.append(report)

            sev = (report.get("severity") or "").strip().lower()
            if report.get("ok") or sev == "clean":
                print("  [verifier] ✓ clean")
                return {
                    "markdown": final_markdown,
                    "verdict": "clean",
                    "verifier_reports": verifier_reports,
                    "flagged_issues": [],
                    "register_entry_id": entry_id,
                }

            flags = _collect_flags(report)
            print(f"  [verifier] flagged {len(flags)} issue(s) — "
                  f"{'retrying' if attempt == 0 else 'will annotate output'}.")
            feedback_for_retry = report
            if attempt == 1:
                # Second attempt still failed — ship annotated markdown so the
                # user sees the flags BEFORE executing anything.
                annotated = _prepend_flag_block(final_markdown, flags)
                return {
                    "markdown": annotated,
                    "verdict": "needs_fixes" if sev != "fatal" else "fatal",
                    "verifier_reports": verifier_reports,
                    "flagged_issues": flags,
                    "register_entry_id": entry_id,
                }
        # Unreachable — both attempts complete via early return
        return {
            "markdown": final_markdown,
            "verdict": "fatal",
            "verifier_reports": verifier_reports,
            "flagged_issues": ["unexpected pipeline exit"],
            "register_entry_id": entry_id,
        }

    def _directives_dir(self) -> Path:
        journal_path = Path(self.config.journal_path)
        stem = journal_path.stem
        d = journal_path.parent / f"{stem}_directives"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def export_directive_for(self, register_entry_id: str) -> dict:
        """Generate a research directive for one register entry. Returns
        {path, verdict, flagged_issues_count, register_entry_id}."""
        entry = next(
            (r for r in self.journal.register if r.get("id") == register_entry_id),
            None,
        )
        if entry is None:
            raise ValueError(f"register entry {register_entry_id} not found")
        # Attach predictions so the pipeline can reference them
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
            "verifier_reports": result["verifier_reports"],
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
        """Generate a bundle covering every qualifying register entry. Slower
        than per-record; intended as a periodic snapshot. Returns {path,
        entries_included, entries_flagged}."""
        qualifying = self.qualifying_register_entries()
        if not qualifying:
            print("  [bundle] no qualifying register entries (held audit × open predictions).")
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


def _prepend_flag_block(markdown: str, flags: list[str]) -> str:
    if not flags:
        return markdown
    lines = [
        "> ⚠ **FLAGGED ISSUES** — the verifier found the following concerns during review. "
        "Read these before executing the directive; some items may be fabrications.",
        "",
    ]
    for f in flags:
        lines.append(f"> - {f}")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines) + markdown
