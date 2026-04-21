"""
Curiosity Engine - CLI Entry Point
==================================
A system that enables an LLM to generate its own research questions,
investigate them, and cross-reference findings to surface novel insights.

Model connection settings live at ~/.CuriosityEngine/engine.toml.
(Auto-created on first run.)

Usage:
    python curiosity_engine.py --cycles 1
    python curiosity_engine.py --cycles 5
    python curiosity_engine.py --cross-ref-only
    python curiosity_engine.py --show-insights
    python curiosity_engine.py --show-register
"""

import argparse

from config import CuriosityEngineConfig
from engine import CuriosityEngine
from models import EngineConfig


def main():
    connection = CuriosityEngineConfig.load()
    engine_defaults = EngineConfig()

    parser = argparse.ArgumentParser(description="Curiosity Engine - AI/ML Research Explorer")
    parser.add_argument("--cycles", type=int, default=1, help="Number of curiosity cycles to run")
    parser.add_argument("--cross-ref-only", action="store_true", help="Only run cross-referencing on existing journal")
    parser.add_argument("--show-insights", action="store_true", help="Display all generated insights")
    parser.add_argument("--show-register", action="store_true", help="Display the verified-insights register")
    parser.add_argument("--show-predictions", action="store_true", help="List stored predictions with status")
    parser.add_argument("--check-predictions", action="store_true", help="Check due predictions (uses verifier + web_search)")
    parser.add_argument("--check-predictions-all", action="store_true", help="Check ALL pending predictions (ignore target_date)")
    parser.add_argument("--show-journal", action="store_true", help="Display journal summary")
    parser.add_argument("--journal", type=str, default=engine_defaults.journal_path, help="Path to journal file")
    parser.add_argument("--domain", type=str, default=engine_defaults.domain, help="Research domain")
    parser.add_argument("--primary-model", type=str, default=None, help="Override primary model name (keeps TOML provider/endpoint)")
    parser.add_argument("--verifier-model", type=str, default=None, help="Override verifier model name")
    parser.add_argument("--cross-ref-freq", type=int, default=engine_defaults.cross_ref_frequency, help="Run cross-ref every N cycles")
    args = parser.parse_args()

    if args.primary_model:
        connection.primary = _replace_name(connection.primary, args.primary_model)
    if args.verifier_model:
        connection.verifier = _replace_name(connection.verifier, args.verifier_model)

    config = EngineConfig(
        domain=args.domain,
        journal_path=args.journal,
        cross_ref_frequency=args.cross_ref_freq,
        connection=connection,
    )

    engine = CuriosityEngine(config)

    if args.show_insights:
        engine.show_insights()
    elif args.show_register:
        engine.show_register()
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
    else:
        engine.run(args.cycles)


def _replace_name(profile, new_name: str):
    from dataclasses import replace
    return replace(profile, name=new_name)


if __name__ == "__main__":
    main()
