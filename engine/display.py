"""Read-only display/inspection commands (no API calls)."""

from __future__ import annotations

from collections import Counter

from register import render_markdown


class DisplayMixin:
    """show_* methods that dump current state to stdout. None of these make API calls."""

    def show_insights(self):
        if not self.journal.insights:
            print("No insights generated yet. Run some cycles first.")
            return

        for i, insight in enumerate(self.journal.insights):
            print(f"\n{'='*60}")
            print(f"INSIGHT {i+1}: {insight['title']}")
            print(f"{'='*60}")
            print(f"Confidence: {insight['confidence']:.2f}")
            print(f"\n{insight['description']}")
            print(f"\nNovelty Assessment: {insight['novelty_assessment']}")
            if insight.get("prior_art_check"):
                print(f"\nPrior Art Check: {insight['prior_art_check']}")
            print("\nImplications:")
            for imp in insight.get("implications", []):
                print(f"  - {imp}")
            print("\nOpen Questions:")
            for q in insight.get("open_questions", []):
                print(f"  - {q}")
            print("\nCounter-Arguments:")
            for c in insight.get("counter_arguments", []):
                print(f"  - {c}")

    def show_register(self):
        if not self.journal.register:
            print("Register is empty. No insights have passed verification yet.")
            return
        print(render_markdown(self.journal.register, self.journal.predictions))

    def show_predictions(self):
        if not self.journal.predictions:
            print("No predictions have been registered yet.")
            return
        statuses = Counter(p.get("status", "pending") for p in self.journal.predictions)
        print(f"\nPredictions: {len(self.journal.predictions)} total  ({dict(statuses)})")
        for p in self.journal.predictions:
            status = p.get("status", "pending")
            print(f"\n  [{status}] {p.get('id')}  target={p.get('target_date')}")
            print(f"    Claim:     {p.get('claim', '')[:120]}")
            print(f"    Condition: {p.get('falsifiable_condition', '')[:120]}")
            print(f"    Method:    {p.get('check_method', '')[:120]}")
            if p.get("last_checked_at"):
                last = p["review_log"][-1] if p.get("review_log") else None
                if last:
                    print(f"    Last check: {last.get('verdict')} @ {p['last_checked_at']}")

    def show_journal_summary(self):
        print(f"\nJournal: {self.config.journal_path}")
        if self.journal.focus:
            print(f"Focus: {self.journal.focus}")
        print(f"Entries: {len(self.journal.entries)}")
        print(f"Cross-references: {len(self.journal.cross_references)}")
        print(f"Insights: {len(self.journal.insights)}")
        print(f"Register entries (validated): {len(self.journal.register)}")
        print(f"Predictions on file: {len(self.journal.predictions)}")

        if self.journal.entries:
            print(f"\nDomains explored: {', '.join(self.journal.get_all_domain_tags())}")

            high_surprise = self.journal.get_high_surprise_entries()
            if high_surprise:
                print(f"\nHigh-surprise findings ({len(high_surprise)}):")
                for e in high_surprise:
                    print(f"  [{e['surprise_delta']:.2f}] {e['question'][:70]}...")
