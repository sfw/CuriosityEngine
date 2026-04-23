"""Markdown rendering for the verified-insights register."""

from __future__ import annotations


def render_markdown(register_entries: list[dict], predictions: list[dict] | None = None) -> str:
    """Render the register as a human-readable markdown document.

    predictions: optional flat list of Prediction dicts keyed to entries by register_entry_id.
    """
    lines: list[str] = [
        "# Curiosity Engine — Verified Insights Register",
        "",
        "Insights that survived adversarial verification. Each entry carries full",
        "substantiation, motivation, the verifier's report, and any falsifiable",
        "predictions that follow from the insight.",
        "",
    ]

    if not register_entries:
        lines.append("*No validated insights yet.*")
        lines.append("")
        return "\n".join(lines)

    predictions_by_entry: dict[str, list[dict]] = {}
    for p in predictions or []:
        predictions_by_entry.setdefault(p.get("register_entry_id", ""), []).append(p)

    for entry in register_entries:
        entry_predictions = predictions_by_entry.get(entry.get("id", ""), [])
        lines.extend(_render_entry(entry, entry_predictions))
        lines.append("")

    return "\n".join(lines)


def _render_entry(entry: dict, predictions: list[dict]) -> list[str]:
    out: list[str] = []
    out.append("---")
    out.append("")
    out.append(f"## {entry.get('title', 'Untitled')}")
    out.append("")
    out.append(f"- **Entry ID:** `{entry.get('id', '')}`")
    out.append(f"- **Source Insight:** `{entry.get('insight_id', '')}`")
    out.append(f"- **Registered:** {entry.get('timestamp', '')}")
    out.append(f"- **Verdict:** `{entry.get('verdict', 'unknown')}`")
    out.append(f"- **Verified Confidence:** {entry.get('verified_confidence', 0.0):.2f}")
    out.append(f"- **Lifecycle status:** `{entry.get('status', 'active')}`")
    review_status = entry.get("human_review_status", "unreviewed")
    reviewer = entry.get("human_reviewer", "")
    review_suffix = f" by {reviewer}" if reviewer else ""
    out.append(f"- **Human review:** `{review_status}`{review_suffix}")
    if review_status == "rejected":
        reason = entry.get("human_rejection_reason", "")
        if reason:
            out.append(f"  - rejection reason: {reason}")
    notes = entry.get("human_review_notes", "")
    if notes:
        out.append(f"  - notes: {notes}")
    out.append("")

    description = entry.get("description", "").strip()
    if description:
        out.append("### Description")
        out.append("")
        out.append(description)
        out.append("")

    motivation = entry.get("motivation", "").strip()
    if motivation:
        out.append("### Motivation")
        out.append("")
        out.append(motivation)
        out.append("")

    out.append("### Substantiation")
    out.append("")
    xref_id = entry.get("supporting_xref_id", "")
    if xref_id:
        out.append(f"- Source cross-reference: `{xref_id}`")
    summaries = entry.get("supporting_entry_summaries", []) or []
    if summaries:
        out.append("- Supporting journal entries:")
        for s in summaries:
            question = s.get("question", "").strip().replace("\n", " ")
            out.append(f"  - `{s.get('id', '')}` — {question}")
            takeaways = s.get("key_takeaways", []) or []
            for t in takeaways:
                out.append(f"    - {t}")
    sources = entry.get("supporting_sources", []) or []
    if sources:
        out.append("- Web sources cited during investigation:")
        for src in sources:
            out.append(f"  - {src}")
    out.append("")

    out.append("### Verification")
    out.append("")
    summary = entry.get("verification_summary", "").strip()
    if summary:
        out.append(summary)
        out.append("")

    novelty_type = entry.get("novelty_type", "")
    if novelty_type:
        out.append(f"- **Novelty type:** `{novelty_type}`")

    if "premises_supported" in entry or "synthesis_findable" in entry:
        premises_supported = entry.get("premises_supported", True)
        synthesis_findable = entry.get("synthesis_findable", False)
        out.append(f"- **Premises supported in literature:** {'yes' if premises_supported else 'no'}")
        premises_cites = entry.get("premises_support_citations", []) or []
        for c in premises_cites:
            out.append(f"  - {c}")
        out.append(f"- **Synthesis already in literature:** {'yes' if synthesis_findable else 'no'}")
        synth_cites = entry.get("synthesis_prior_art", []) or entry.get("prior_art_citations", []) or []
        for c in synth_cites:
            out.append(f"  - {c}")
    else:
        # Legacy entry written before the premises/synthesis split.
        prior_art_found = entry.get("prior_art_found", False)
        out.append(f"- **Prior art found:** {'yes' if prior_art_found else 'no'}")
        citations = entry.get("prior_art_citations", []) or []
        for c in citations:
            out.append(f"  - {c}")

    contradictions = entry.get("contradicting_findings", []) or []
    if contradictions:
        out.append("- **Contradicting findings considered:**")
        for c in contradictions:
            out.append(f"  - {c}")

    flaws = entry.get("reasoning_flaws_considered", []) or []
    if flaws:
        out.append("- **Reasoning flaws examined:**")
        for f in flaws:
            out.append(f"  - {f}")
    out.append("")

    implications = entry.get("implications", []) or []
    if implications:
        out.append("### Implications")
        out.append("")
        for imp in implications:
            out.append(f"- {imp}")
        out.append("")

    open_questions = entry.get("open_questions", []) or []
    if open_questions:
        out.append("### Open Questions")
        out.append("")
        for q in open_questions:
            out.append(f"- {q}")
        out.append("")

    counter_args = entry.get("counter_arguments", []) or []
    if counter_args:
        out.append("### Counter-Arguments")
        out.append("")
        for c in counter_args:
            out.append(f"- {c}")
        out.append("")

    if predictions:
        out.append("### Falsifiable Predictions")
        out.append("")
        for p in predictions:
            status = p.get("status", "pending")
            out.append(f"- **`{p.get('id')}`** — status: `{status}`, target: {p.get('target_date', '')}")
            out.append(f"    - **Claim:** {p.get('claim', '')}")
            out.append(f"    - **Falsifiable condition:** {p.get('falsifiable_condition', '')}")
            out.append(f"    - **Check method:** {p.get('check_method', '')}")
            log = p.get("review_log") or []
            if log:
                out.append("    - **Review history:**")
                for review in log:
                    verdict = review.get("verdict", "?")
                    checked_at = review.get("checked_at", "")
                    reasoning = review.get("reasoning", "")
                    out.append(f"        - `{verdict}` @ {checked_at} — {reasoning}")
                    for src in review.get("sources", []) or []:
                        out.append(f"            - {src}")
        out.append("")

    return out
