"""
Curiosity Engine - CLI Entry Point
==================================
A system that enables an LLM to generate its own research questions,
investigate them, and cross-reference findings to surface novel insights.

Model connection settings live at ~/.CuriosityEngine/engine.toml.
(Auto-created on first run.)

Usage:
    python curiosity_engine.py --cycles 1            # prompts for domain if not given
    python curiosity_engine.py --cycles 5 --domain "topic"
    python curiosity_engine.py --cross-ref-only
    python curiosity_engine.py --show-insights
    python curiosity_engine.py --show-register
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from config import CuriosityEngineConfig
from engine import CuriosityEngine
from models import EngineConfig


def main():
    connection = CuriosityEngineConfig.load()
    engine_defaults = EngineConfig()

    parser = argparse.ArgumentParser(description="Curiosity Engine - AI/ML Research Explorer")
    parser.add_argument("--cycles", type=int, default=0, help="Number of curiosity cycles to run (default 0 — specify explicitly)")
    parser.add_argument("--cross-ref-only", action="store_true", help="Only run cross-referencing on existing journal")
    parser.add_argument("--show-insights", action="store_true", help="Display all generated insights")
    parser.add_argument("--show-register", action="store_true", help="Display the verified-insights register")
    parser.add_argument("--review-register", action="store_true", help="Interactively review unreviewed register entries")
    parser.add_argument("--show-predictions", action="store_true", help="List stored predictions with status")
    parser.add_argument("--check-predictions", action="store_true", help="Check due predictions (uses verifier + web_search)")
    parser.add_argument("--check-predictions-all", action="store_true", help="Check ALL pending predictions (ignore target_date)")
    parser.add_argument("--show-journal", action="store_true", help="Display journal summary")
    parser.add_argument("--list-tools", action="store_true", help="List registered tools and exit")
    parser.add_argument("--set-focus", type=str, default=None, metavar="TEXT",
                        help="Set the journal's investigation focus (applies to all subsequent prompts)")
    parser.add_argument("--show-focus", action="store_true", help="Show the current investigation focus")
    parser.add_argument("--clear-focus", action="store_true", help="Remove the current investigation focus")
    parser.add_argument("--add-question", type=str, action="append", default=None, metavar="TEXT",
                        help="Queue a user-directed research question (repeatable)")
    parser.add_argument("--list-questions", action="store_true", help="List queued questions (emergent + human)")
    parser.add_argument("--clear-questions", action="store_true", help="Clear the question queue (all sources)")
    parser.add_argument("--graph-summary", action="store_true", help="Print a knowledge-graph summary over the journal")
    parser.add_argument("--graph-export", type=str, default=None, metavar="PATH",
                        help="Export the knowledge graph to PATH. Format inferred from extension (.graphml / .gexf / .json).")
    parser.add_argument("--find-similar", type=str, default=None, metavar="TEXT",
                        help="Free-text semantic search over journal entries (requires embedding-capable provider)")
    parser.add_argument("--embed-backfill", action="store_true",
                        help="Compute embeddings for any journal entries that don't have one yet")
    parser.add_argument("--top-k", type=int, default=10, help="K for --find-similar (default 10)")
    parser.add_argument("--journal", type=str, default=engine_defaults.journal_path, help="Path to journal file")
    parser.add_argument("--domain", type=str, default=None, help="Research domain (prompts if omitted on a TTY)")
    parser.add_argument("--primary-model", type=str, default=None, help="Override primary model name only (keeps primary profile's provider/endpoint; useful for same-endpoint model switching)")
    parser.add_argument("--verifier-model", type=str, default=None, help="Override verifier model name only (keeps verifier profile's provider/endpoint)")
    parser.add_argument("--primary-role", type=str, default=None, metavar="ROLE",
                        help="Copy a configured profile (by role, e.g. 'verifier') INTO the primary slot — swaps the whole profile including provider/base_url/api_key, not just the name.")
    parser.add_argument("--verifier-role", type=str, default=None, metavar="ROLE",
                        help="Copy a configured profile into the verifier slot.")
    parser.add_argument("--cross-ref-role", type=str, default=None, metavar="ROLE",
                        help="Override the profile used for the cross-reference phase (e.g. 'verifier' to offload cross-ref to the verifier's model for this run).")
    parser.add_argument("--cross-ref-freq", type=int, default=engine_defaults.cross_ref_frequency, help="Run cross-ref every N cycles")
    # Per-run engine-knob overrides. default=None means "inherit from engine.toml".
    parser.add_argument("--cross-ref-window", type=int, default=None,
                        help="Override [engine].cross_ref_window for this run (max entries sent to cross-ref prompt)")
    parser.add_argument("--investigations-per-cycle", type=int, default=None,
                        help="Override [engine].investigations_per_cycle for this run")
    parser.add_argument("--novelty-threshold", type=float, default=None,
                        help="Override [engine].novelty_threshold for this run (0.0–1.0)")
    parser.add_argument("--register-confidence-floor", type=float, default=None,
                        help="Override [engine].register_confidence_floor for this run (0.0–1.0)")
    parser.add_argument("--verify-insights", action=argparse.BooleanOptionalAction, default=None,
                        help="Toggle cross-model verification for this run (--verify-insights / --no-verify-insights)")
    parser.add_argument("--analog-probe-enabled", action=argparse.BooleanOptionalAction, default=None,
                        help="Toggle cross-domain analog probe for this run")
    parser.add_argument("--analog-probe-threshold", type=float, default=None,
                        help="Override [engine].analog_probe_surprise_threshold for this run (0.0–1.0)")
    parser.add_argument("--assumption-probe-enabled", action=argparse.BooleanOptionalAction, default=None,
                        help="Toggle within-domain assumption probe (fires on LOW-surprise confirmed findings to surface implicit premises)")
    parser.add_argument("--assumption-probe-threshold", type=float, default=None,
                        help="Override [engine].assumption_probe_surprise_threshold — probe fires when surprise_delta ≤ this AND verdict == confirmed (default 0.3)")
    parser.add_argument("--held-entries-enabled", action=argparse.BooleanOptionalAction, default=None,
                        help="Allow `inconclusive` verdicts to create held register entries (--held-entries-enabled / --no-held-entries-enabled)")
    parser.add_argument("--held-confidence-floor", type=float, default=None,
                        help="Override [engine].held_confidence_floor (minimum confidence for held entries, 0.0–1.0)")
    parser.add_argument("--reverify-insights", action="store_true",
                        help="Re-run verification on every insight that does NOT already have a register entry — elevates previously-rejected insights under current rules.")
    parser.add_argument("--reverify-insight", type=str, default=None, metavar="INSIGHT_ID",
                        help="Re-verify a single insight by id (e.g. i-abc12345). Overrides --reverify-insights scope.")
    parser.add_argument("--reverify-register", action="store_true",
                        help="Re-run the verifier over EXISTING register entries WITHOUT overwriting them — appends a reverification_log to each entry. Use after changing verification rules to audit whether old verdicts still hold.")
    parser.add_argument("--reverify-register-id", type=str, default=None, metavar="REGISTER_ID",
                        help="Re-verify a single register entry by id (e.g. r-abc12345). Overrides --reverify-register scope.")
    parser.add_argument("--reverify-register-max-confidence", type=float, default=None, metavar="CONF",
                        help="Only re-verify register entries with verified_confidence ≤ this (e.g. 0.8 to skip the most confident ones).")
    parser.add_argument("--reverify-register-novelty-types", type=str, default=None, metavar="TYPES",
                        help="Comma-separated novelty_types to re-verify (e.g. 'new_synthesis,correction'). Default: all.")
    parser.add_argument("--synth-orphaned-xrefs", action="store_true",
                        help="Synthesize + verify every cross-reference that doesn't yet have a matching insight (e.g. after a mid-run failure between cross-ref and synthesis).")
    parser.add_argument("--scan-gaps", action="store_true",
                        help="Run a negative-space scan: build (method × problem) matrix from journal entries, classify empty cells, verify underexplored gaps via academic_search, enqueue questions for verified gaps. Gated by [engine].negative_space_min_entries.")
    parser.add_argument("--export-directive", type=str, default=None, metavar="REGISTER_ID",
                        help="Generate a research directive (markdown) for ONE register entry. Runs the primary+verifier pipeline scoped to that entry. Output: data/{journal}_directives/{id}.md")
    parser.add_argument("--export-directives-bundle", action="store_true",
                        help="Generate a research-directives bundle covering every qualifying register entry (validated × open predictions). Slower than per-record; intended as a periodic snapshot.")
    args = parser.parse_args()

    if args.list_tools:
        from engine.tools import discover_tools, registry as tool_registry
        discover_tools()
        tools = tool_registry.all()
        if not tools:
            print("No tools registered.")
            return
        print(f"{len(tools)} tool(s) registered:\n")
        for cls in tools:
            print(f"  {cls.name}")
            print(f"    {cls.description}")
            print()
        return

    # Resolve domain. Explicit --domain wins; otherwise prompt on a TTY; else default.
    read_only = (
        args.show_insights
        or args.show_register
        or args.review_register
        or args.show_predictions
        or args.show_journal
        or args.check_predictions
        or args.check_predictions_all
        or args.set_focus is not None
        or args.show_focus
        or args.clear_focus
        or args.add_question is not None
        or args.list_questions
        or args.clear_questions
        or args.graph_summary
        or args.graph_export is not None
        or args.find_similar is not None
        or args.embed_backfill
        or args.reverify_insights
        or args.reverify_insight is not None
        or args.synth_orphaned_xrefs
        or args.scan_gaps
    )
    if args.domain is None:
        if read_only:
            args.domain = engine_defaults.domain
        elif sys.stdin.isatty():
            args.domain = _prompt_for_domain(engine_defaults.domain, args.journal)
        else:
            args.domain = engine_defaults.domain

    # Role-based swap first (copies the full profile — provider, base_url, api_key, name),
    # then name-only override refines the name within the (possibly swapped) slot.
    def _resolve_role(role_name: str):
        rn = role_name.strip().lower()
        if rn == "primary":
            return connection.primary
        if rn == "verifier":
            return connection.verifier
        extras = getattr(connection, "extras", {}) or {}
        if rn in extras:
            return extras[rn]
        return None

    if args.primary_role:
        src = _resolve_role(args.primary_role)
        if src is None:
            parser.error(f"--primary-role: unknown profile role {args.primary_role!r}")
        connection.primary = replace(src)
    if args.verifier_role:
        src = _resolve_role(args.verifier_role)
        if src is None:
            parser.error(f"--verifier-role: unknown profile role {args.verifier_role!r}")
        connection.verifier = replace(src)

    if args.cross_ref_role:
        src = _resolve_role(args.cross_ref_role)
        if src is None:
            parser.error(f"--cross-ref-role: unknown profile role {args.cross_ref_role!r}")
        # Copy the resolved profile into the cross_ref slot so the engine
        # builds a dedicated client for it (or aliases to primary if same).
        connection.cross_ref = replace(src)

    if args.primary_model:
        connection.primary = replace(connection.primary, name=args.primary_model)
    if args.verifier_model:
        connection.verifier = replace(connection.verifier, name=args.verifier_model)

    # CLI overrides take precedence over engine.toml values. None = inherit.
    def _override(cli_val, toml_val):
        return toml_val if cli_val is None else cli_val

    config = EngineConfig(
        domain=args.domain,
        journal_path=args.journal,
        cross_ref_frequency=args.cross_ref_freq,
        connection=connection,
        # Engine-level behavior pulled from [engine] section of engine.toml, with CLI overrides.
        cross_ref_window=_override(args.cross_ref_window, connection.engine.cross_ref_window),
        questions_per_cycle=connection.engine.questions_per_cycle,
        investigations_per_cycle=_override(args.investigations_per_cycle, connection.engine.investigations_per_cycle),
        novelty_threshold=_override(args.novelty_threshold, connection.engine.novelty_threshold),
        register_confidence_floor=_override(args.register_confidence_floor, connection.engine.register_confidence_floor),
        verify_insights=_override(args.verify_insights, connection.engine.verify_insights),
        analog_probe_enabled=_override(args.analog_probe_enabled, connection.engine.analog_probe_enabled),
        analog_probe_surprise_threshold=_override(args.analog_probe_threshold, connection.engine.analog_probe_surprise_threshold),
        assumption_probe_enabled=_override(args.assumption_probe_enabled, connection.engine.assumption_probe_enabled),
        assumption_probe_surprise_threshold=_override(args.assumption_probe_threshold, connection.engine.assumption_probe_surprise_threshold),
        held_entries_enabled=_override(args.held_entries_enabled, connection.engine.held_entries_enabled),
        held_confidence_floor=_override(args.held_confidence_floor, connection.engine.held_confidence_floor),
        parallel_investigations=connection.engine.parallel_investigations,
        parallel_xref_pipeline=connection.engine.parallel_xref_pipeline,
        analog_probe_max_analogs=connection.engine.analog_probe_max_analogs,
        assumption_probe_max_assumptions=connection.engine.assumption_probe_max_assumptions,
        gap_verification_hit_threshold=connection.engine.gap_verification_hit_threshold,
        confidence_drop_on_downgrade=connection.engine.confidence_drop_on_downgrade,
        question_priority_floor=connection.engine.question_priority_floor,
    )

    engine = CuriosityEngine(config)

    # ── State mutations (applied first so a single invocation can combine update + action) ──
    did_mutation = False
    if args.set_focus is not None:
        engine.journal.set_focus(args.set_focus)
        print(f"Focus set: {engine.journal.focus!r}")
        did_mutation = True
    if args.clear_focus:
        engine.journal.clear_focus()
        print("Focus cleared.")
        did_mutation = True
    if args.add_question:
        engine._enqueue_questions(args.add_question, source="human", priority=1.0)
        print(f"Queued {len(args.add_question)} user-directed question(s).")
        did_mutation = True
    if args.clear_questions:
        removed = engine.journal.clear_question_queue()
        print(f"Cleared {removed} queued question(s).")
        did_mutation = True

    # ── Read-only inspections ──
    if args.show_focus:
        print(f"Focus: {engine.journal.focus or '(none set)'}")
        return
    if args.list_questions:
        queue = engine.journal.question_queue
        if not queue:
            print("Question queue is empty.")
            return
        print(f"{len(queue)} queued question(s):")
        for q in queue:
            src = q.get("source", "?")
            prefix = " *" if src.startswith("human") else "  "
            print(f"{prefix} [{src}] {q.get('question', '')}")
        return

    # ── Action selection ──
    if args.embed_backfill:
        if engine.embedding_client is None:
            print("No embedding-capable provider configured (need an openai_compat profile with embeddings access).")
            return
        from engine.embeddings import embed_missing_entries
        n = embed_missing_entries(engine.journal, engine.embedding_client)
        print(f"Embedded {n} previously-unembedded entry/entries.")
        return

    if args.find_similar:
        if engine.embedding_client is None:
            print("No embedding-capable provider configured; cannot run similarity search.")
            return
        from engine.embeddings import find_similar
        hits = find_similar(args.find_similar, engine.journal, engine.embedding_client, top_k=args.top_k)
        if not hits:
            print("No similar entries found. (Did you run --embed-backfill on legacy entries?)")
            return
        print(f"Top {len(hits)} entries for {args.find_similar!r}:")
        for h in hits:
            print(f"  {h.score:.3f}  {h.entry_id}  — {h.question[:100]}")
        return

    if args.graph_summary:
        from engine.graph import graph_summary
        print(graph_summary(engine.journal))
        return
    if args.graph_export:
        from engine.graph import export_graph
        ext = args.graph_export.rsplit(".", 1)[-1].lower()
        fmt = {"graphml": "graphml", "xml": "graphml", "gexf": "gexf", "json": "json"}.get(ext, "graphml")
        export_graph(engine.journal, args.graph_export, fmt=fmt)
        print(f"Graph exported to {args.graph_export} ({fmt}).")
        return

    if args.show_insights:
        engine.show_insights()
    elif args.show_register:
        engine.show_register()
    elif args.review_register:
        engine.review_register()
    elif args.show_predictions:
        engine.show_predictions()
    elif args.check_predictions or args.check_predictions_all:
        engine.check_predictions(all_pending=args.check_predictions_all)
    elif args.reverify_insight is not None:
        engine.reverify_unregistered_insights(only_ids=[args.reverify_insight])
    elif args.reverify_insights:
        engine.reverify_unregistered_insights()
    elif args.reverify_register_id is not None:
        engine.reverify_register_entries(only_ids=[args.reverify_register_id])
    elif args.reverify_register:
        novelty_types = None
        if args.reverify_register_novelty_types:
            novelty_types = [
                n.strip() for n in args.reverify_register_novelty_types.split(",") if n.strip()
            ]
        engine.reverify_register_entries(
            max_confidence=args.reverify_register_max_confidence,
            novelty_types=novelty_types,
        )
    elif args.synth_orphaned_xrefs:
        engine.synthesize_orphaned_xrefs()
    elif args.scan_gaps:
        engine.scan_gaps()
    elif args.export_directive:
        engine.export_directive_for(args.export_directive.strip())
    elif args.export_directives_bundle:
        engine.export_directives_bundle()
    elif args.show_journal:
        engine.show_journal_summary()
    elif args.cross_ref_only:
        xrefs = engine.cross_reference()
        for xref in xrefs:
            if xref.novelty_score >= config.novelty_threshold:
                insight = engine.synthesize(xref)
                if insight and config.verify_insights:
                    engine.verify_insight(insight, xref)
    elif args.cycles > 0:
        engine.run(args.cycles)
    elif did_mutation:
        # Mutations already applied above; nothing else requested.
        return
    else:
        parser.print_help()


def _prompt_for_domain(default: str, journal_path: str) -> str:
    """Interactively ask the user for the research domain; show journal context if present."""
    print()
    print("=" * 62)
    print("  Research domain")
    print("=" * 62)

    existing_context = _journal_context_hint(journal_path)
    if existing_context:
        print(existing_context)

    print("What should the engine focus on? (Press Enter to accept the default.)")
    try:
        raw = input(f"  domain [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return raw or default


def _journal_context_hint(journal_path: str) -> str:
    """If the target journal already holds entries, summarize what's there."""
    path = Path(journal_path).expanduser()
    if not path.exists():
        return f"Journal: {path} (will be created)\n"
    try:
        data = json.loads(path.read_text() or "{}")
    except (json.JSONDecodeError, OSError):
        return f"Journal: {path} (unreadable — will overwrite)\n"

    entries = data.get("entries") or []
    register = data.get("register") or []
    if not entries:
        return f"Journal: {path} (empty)\n"

    tags = sorted({tag for e in entries for tag in (e.get("domain_tags") or [])})
    lines = [f"Journal: {path}"]
    lines.append(f"  {len(entries)} entries, {len(register)} registered insight(s).")
    if tags:
        head = ", ".join(tags[:12])
        more = "" if len(tags) <= 12 else f" … (+{len(tags) - 12} more)"
        lines.append(f"  Domains so far: {head}{more}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
