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
    parser.add_argument("--primary-model", type=str, default=None, help="Override primary model name (keeps TOML provider/endpoint)")
    parser.add_argument("--verifier-model", type=str, default=None, help="Override verifier model name")
    parser.add_argument("--cross-ref-freq", type=int, default=engine_defaults.cross_ref_frequency, help="Run cross-ref every N cycles")
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
    )
    if args.domain is None:
        if read_only:
            args.domain = engine_defaults.domain
        elif sys.stdin.isatty():
            args.domain = _prompt_for_domain(engine_defaults.domain, args.journal)
        else:
            args.domain = engine_defaults.domain

    if args.primary_model:
        connection.primary = replace(connection.primary, name=args.primary_model)
    if args.verifier_model:
        connection.verifier = replace(connection.verifier, name=args.verifier_model)

    config = EngineConfig(
        domain=args.domain,
        journal_path=args.journal,
        cross_ref_frequency=args.cross_ref_freq,
        connection=connection,
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
        engine._enqueue_questions(args.add_question, source="human")
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
