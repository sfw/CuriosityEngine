# Curiosity Engine

A research loop that **generates its own questions, investigates them, and decides whether the resulting insights are genuinely novel** — using an adversarial verifier whose architecture has itself been *evolved* from the engine's own validated outputs.

You set a topic. The engine forms hypotheses, runs searches across the open web + academic APIs + sandboxed code execution, compares findings to its prior, and routes anything interesting through a multi-stage verifier before deciding whether to add it to a durable register. Validated entries can then be exported as **research directives** — concrete plans (with citations and tool calls) that a human or LLM-driven agent can actually execute toward a publication.

It works for any domain a human studies: AI research, biology, investing, marketing, governance, etc. Every prompt describes structure, not content — there are no field-specific examples or canned biology illustrations leaking through.

> **Status**: working proof of concept. Single-user, runs locally in Docker. The verifier has been hardened five times based on the engine's *own* validated insights about how research-engine novelty verification should work — see [Self-evolving verifier](#self-evolving-verifier) below.

---

## What it actually does

Given a topic, you run cycles. Each cycle does roughly this:

1. **Asks itself what it doesn't know** about the topic (introspection).
2. **Generates investigable questions** ranked by priority and source.
3. **Forms a hypothesis** about each question *before* searching — so the result of the search can be measured as surprise vs prior, not as self-reported confidence.
4. **Investigates** using web search, academic search, archive access, web fetch, code execution, etc. The engine has tools and uses them.
5. **Assesses surprise** by comparing what it found to what it predicted.
6. **Cross-references** the new entry with everything else in the journal, looking for non-obvious connections — especially across domains.
7. **Synthesizes** high-novelty cross-references into candidate insights.
8. **Verifies** each insight adversarially against current literature: is the central architectural move already deployed somewhere? Is the full composite claim findable? Are there peer systems with substantive overlap? Is the reasoning actually justified?
9. **Registers** the survivors. Each registered entry carries falsifiable predictions for later review.

You can interrupt anywhere: set focus, add your own questions, reject claims with reasons, add known peer systems the verifier must consider, run gap scans, export research directives.

---

## Core ideas

These are load-bearing. They show up in every prompt and every config default.

- **Hypothesis before evidence.** The engine commits to a specific falsifiable answer *before* searching. Surprise becomes a structural comparison, not a self-report.
- **Novelty is structural, not vibes.** "The premises exist in the literature, the synthesis does not" is the *signature* of genuine novelty, not a weakness. Naive verifiers reject this pattern; the phased verifier looks for it explicitly.
- **Different model for evaluation than for generation.** The same model grading its own work tends to flatter its priors. The engine routes verification to a cross-family verifier, and (Phase 5) routes the post-search assessor to a separate model from the explorer.
- **Grounding over generation.** Research directives can't invent URLs or tool names. Every citation must come from an allowlist built from the source register entry; every tool call must exact-match an allowlist. A verifier scans the output for fabrication; anything dubious ships with a `⚠ FLAGGED ISSUES` block.
- **Cross-domain is a first-class move.** On surprising findings, an analog probe asks the engine which *distant* fields have structural analogs. On unsurprising-but-confirmed findings, an assumption probe surfaces premises the field takes for granted. A negative-space scan maps which `(method, problem)` combinations *aren't* in the journal yet.
- **Audit trails, not overwrites.** Re-verification appends to a log instead of mutating the original verdict — both old and new are preserved so you can see what changed under updated rules.
- **Domain-agnostic by construction.** Prompts use shape, not content. The same engine works identically for any structured field.

---

## Pipeline overview

```
introspect       →  what is the engine uncertain about
generate         →  ranked investigable questions (source-round-robin queue)
investigate                       (the explorer persona)
  ├─ hypothesize  →  commit to a pre-investigation answer
  ├─ search       →  web_search · academic_search · web_fetch · code_execution
  └─ assess       →  the assessor persona compares findings to hypothesis
                     (separate prompt persona — optionally separate model)
analog probe     →  on high-surprise, find distant-domain analogs
assumption probe →  on low-surprise + confirmed, surface implicit premises
embed            →  OpenAI embeddings on question + takeaways

cross-reference  →  graph-aware, cross-domain biased
synthesize       →  promote high-novelty xrefs to candidate insights
verify (cross-family, three-stage)
  ├─ Stage 1: canonicalize (extract structured form of central move)
  ├─ Stage 2: alias-gap detection vs existing register
  │     ├─ STRICT alias  → restatement, skip the heavy verifier
  │     └─ BAND alias    → prime the verifier for differentiator search
  └─ Stage 3: phased prior-art search
        Phase 0: premises check
        Phase 1: central architectural move — already deployed?
        Phase 2: full composite claim — already published?
        Phase 3a: functional decomposition (one exemplar per dimension)
        Phase 3b: closest complete peer system (must name target domain)
        Phase 4: contradicting evidence
        Phase 5: reasoning audit
        Final:   skeptic smell test (kill query + followup)

  engine-side guards: phase-1 / peer-system / known-prior-art / skeptic-probe /
                      challenged-hedge / confidence-drop / inconclusive guardrail

  register gate (scalar OR Pareto admission, configurable)
    └─ emit 1-3 falsifiable predictions + freshness check

human review     →  approve / reject-with-reason / defer / promote held entries
check predictions → revisit due claims; reconcile entry status

(on-demand admin operations)
  ├─ scan-gaps                  — (method × problem) matrix + verified gaps
  ├─ reverify-insights          — re-run verifier on unregistered insights
  ├─ reverify-register          — re-verify EXISTING register (append-only)
  ├─ backfill-canonical-forms   — populate canonicalization layer on legacy
  ├─ pareto-frontier            — show which entries set the admission bar
  ├─ export-directive <id>      — generate executable research plan
  └─ export-directives-bundle   — same pipeline across all qualifying entries
```

State persists to `research_journal.json` after every cycle. The human-readable artifact is `register.md`. Generated directives land in `data/{journal_stem}_directives/`.

---

## Data model — what flows through the pipeline

Six kinds of records flow through the engine, each gated against the next. Each level represents progressively *more vetted* knowledge — the gates are where the engine refuses to promote material that doesn't survive scrutiny.

```
Question  →  JournalEntry  →  CrossReference  →  Insight  →  RegisterEntry  →  Prediction
                                                                       └→ Directive (export)
```

### Question

A research prompt waiting to be investigated. Tagged by where it came from.

- **Sources**: `human` (you added it) · `entry:<id>` (a prior investigation produced a follow-up) · `xref:<id>` (a cross-reference suggested it) · `gap:<id>` (negative-space scan found an unexplored cell) · `analog:<id>` (cross-domain analog probe) · `assumption:<id>` (within-domain assumption probe).
- **Gate to investigation**: priority must clear `question_priority_floor` (default 0.0 — disabled). Human-sourced questions always bypass. The queue dequeues round-robin across sources so one source can't starve another.
- **Lives in**: `journal.question_queue`.

### JournalEntry

The record of one investigation cycle. Captures everything: the question that drove it, the hypothesis the engine *committed to before searching*, the tools it used, the findings it produced, and how surprised it was relative to its prior.

- **Key fields**: `question`, `hypothesis`, `confidence_before`, `raw_findings`, `sources` (URLs + DOIs cited), `surprise_delta`, `confidence_after`, `key_takeaways`, `domain_tags`, `hypothesis_verdict` (confirmed/partially_confirmed/contradicted/unresolved).
- **Gate to cross-referencing**: none directly. Every entry is eligible. Cross-referencing fires periodically (every `cross_ref_frequency` cycles, default 3) and considers the most recent `cross_ref_window` entries (default 20). The graph selector biases toward cross-domain pairings.
- **Lives in**: `journal.entries`.

### CrossReference

A non-obvious connection found between two or more JournalEntries. Identified by an LLM scanning the journal for patterns/contradictions/convergences/implications.

- **Key fields**: `source_entries` (the entries being connected), `connection_type`, `novelty_score`, `description`, `implications`.
- **Gate to synthesis**: `novelty_score >= novelty_threshold` (default 0.7). **Anti-attractor**: candidates with ≥50% participant overlap with an existing xref are rejected unless their score is ≥0.85, preventing the same entries from being re-cross-referenced into near-duplicate insights.
- **Lives in**: `journal.cross_references`.

### Insight

A candidate synthesis derived from a high-novelty CrossReference. **Pre-verification** — this is the candidate the verifier will adversarially scrutinize.

- **Key fields**: `description`, `novelty_assessment` (the synthesizer's self-rationale), `confidence`, `prior_art_check` (cheap structural pre-check before the expensive verifier), `implications`, `open_questions`, `counter_arguments`.
- **Gate to register**: full **three-stage adversarial verifier** (Stage 1 canonicalize → Stage 2 alias-gap → Stage 3 phased prior-art search) → engine-side guards → register gate. See [How novelty is verified](#how-novelty-is-verified--the-technical-detail) for the full machinery.
- **Lives in**: `journal.insights`.

### RegisterEntry

A verified insight that survived adversarial review. The durable artifact of the system. Every claim in the register can trace its provenance back to the JournalEntries that triggered it, the verifier's tool-call audit trail, and the predictions attached at registration.

- **Key fields**: `verdict` (validated/challenged/refuted/inconclusive), `verified_confidence`, `novelty_type` (new_synthesis/extension/restatement/correction/unsupported), `central_architectural_move`, `central_move_prior_art`, `functional_decomposition`, `closest_peer_system`, `skeptic_probe`, `canonical_form` (Phase 1 structured tuple), `component_novelty` (per-architectural-component status, Phase 3), `pareto_axes` (4-axis admission scores, Phase 4), `verification_tool_calls` (full per-call audit trail), `reverification_log` (append-only history of audit re-runs), attached `predictions`.
- **Lifecycle status**:
  - `active` — validated and in the register.
  - `held` — verifier returned `inconclusive` (couldn't reach a verdict, but didn't refute either). Held with a settlement plan; converts to `active` if a settlement trigger resolves.
  - `validated_by_prediction` — promoted from `held` after an attached prediction confirmed.
  - `challenged_by_prediction` — at least one attached prediction refuted.
- **Gate from insight**: see [Register gate](#register-gate). Two admission modes: `scalar` (single confidence floor + status checks) or `pareto` (4-axis Pareto-dominance check on top of scalar).
- **Gate to directive export**: `status=active` AND effective verdict (latest in `reverification_log`, or original) = `validated` AND at least one open prediction.
- **Lives in**: `journal.register`. Human-readable mirror at `register.md`.

### Prediction

A falsifiable claim attached to a RegisterEntry, with a target date for resolution. Predictions are how the engine commits to *future* checkability — without them, the register would just be a snapshot of what the verifier currently believes.

- **Key fields**: `claim`, `falsifiable_condition` (the exact observation that would confirm or refute), `check_method` (how to verify), `target_date`, `status`, `resolution_notes`, `freshness_check` (web-search probe at creation time).
- **Gate at creation**: a freshness probe runs targeted web search on the falsifiable condition. If results already instantiate the claim, the prediction stores as `already_fulfilled` (not `pending`) — closes the dead-on-arrival case where a "prediction" was already true at creation time.
- **Lifecycle status**: `pending` | `confirmed` | `refuted` | `already_fulfilled` | `expired`.
- **Lives in**: `journal.predictions`.

### Directive (export artifact, not a journal record)

A research plan generated from a qualifying RegisterEntry — six structured sections culminating in an agentic-prompt block ready to paste into an LLM-driven agent (Claude Code, an MCP orchestrator, etc.). See [Research directives](#research-directives--turning-verified-claims-into-executable-plans).

- **Lives in**: `data/{journal_stem}_directives/r-<id>.md` plus `.verification.json` sidecar.

### Where each gate lives in code

For readers diving into the source:

| Gate | File / function |
|---|---|
| Question priority floor + source round-robin | `journal.py:pop_queued_questions` |
| Hypothesis confidence calibration | `engine/investigation.py:_calibrate_confidence_after` |
| Cross-reference novelty + anti-attractor | `engine/cross_reference.py` + `engine/graph.py` selector |
| Three-stage verifier | `engine/verification.py:verify_insight` |
| Register gate (scalar + Pareto) | `engine/verification.py:_register_gate` + `_check_pareto_admission` |
| Freshness check at prediction creation | `engine/verification.py:_check_prediction_freshness` |
| Directive admission | `engine/directives.py:qualifying_register_entries` |

---

## Self-evolving verifier

This is the meta-layer that distinguishes the current architecture. The engine ran a journal on the topic *"how should LLM-driven research loops verify novelty?"*. Five of the highest-confidence validated insights from that journal were then *applied to the verifier itself*. Each of those insights is now a phase of the verification pipeline.

| Phase | Source insight (conf) | What it does |
|---|---|---|
| **1. Canonicalization + alias-gap** | r-c67457 (0.82) | Each candidate's central move is rendered into a structured tuple `(predicate, substrate, mechanism, target_domain, key_constraints)` *before* the heavy verifier runs. A combined slot-match + embedding-cosine check measures the *gap* to the nearest existing register entry. STRICT (gap < 0.30) short-circuits to "restatement" — the heavy verifier is skipped entirely. BAND (gap < 0.45) primes the heavy verifier to look for differentiators against the named peer. CLEAR proceeds normally. |
| **2. Three-stage verifier** | r-fcbba1 (0.81) | What used to be one monolithic LLM call becomes Stage 1 (canonicalize) → Stage 2 (alias-detect) → Stage 3 (structural delta scoring, the existing phased prior-art search). The structural-delta stage receives the canonical form as pre-extracted context so it doesn't redo work. |
| **3. Component-resolved novelty** | r-bedce5b1 (0.79) | Novelty is tracked *per architectural component*: `central_move`, each `decomposition_<dimension>` row, and `closest_peer_system`. Reverification can flip a single component's status without forcing a binary entry-level verdict change. The UI badges each component independently. |
| **4. Pareto admission** | r-46988c97 (0.78) | Optional admission gate: a new candidate is rejected if any existing entry **dominates** it on every Pareto axis (`verified_confidence × premises_supported_count × peer_differentiators_count × inverse_alias_gap`). Catches the "just like X but slightly worse on every axis" failure that a scalar confidence floor can't see. Opt-in via `register_admission_mode = "pareto"`. |
| **5. Explore/verify space split** | r-9a35e387 (0.77) | The investigation loop's hypothesis-generation persona (open, divergent) is now structurally separated from its surprise-assessment persona (closed, evaluative). Different prompt framings always; optional cross-family model separation via `investigation_assessor_role`. Resists the self-grading collapse where the explorer and assessor merge in the same token space. |

### Diagnostics (no LLM cost — just inspect state)

```bash
python curiosity_engine.py --pareto-frontier
# Print the active register entries that set the Pareto admission bar.
# For each: their values on the four axes, and which axes they uniquely lead on.

python curiosity_engine.py --three-stage-test "claim description here"
# Run Stages 1+2 only on a hypothetical claim. Prints the canonical form + which
# tier (CLEAR/BAND/STRICT) Stage 2 would fire. No heavy verifier, no persistence.
# Useful for understanding what the alias detector sees.

python curiosity_engine.py --pareto-admission-test "0.75,8,4,0.30"
# Run the Pareto admission check with synthetic axes against the live register.
# Tells you if a candidate with those axes would be admitted, and if not, which
# existing entries would dominate it.

python curiosity_engine.py --backfill-canonical-forms
# One-shot maintenance pass. Populates canonical_form and component_novelty on
# existing register entries that lack them. Idempotent. Add --backfill-force
# to re-canonicalize entries that already have a canonical_form (e.g. after
# the canonicalization prompt itself has been revised).
```

---

## Inspirations and prior work

Two distinct sources have shaped CE's architecture, and they're worth distinguishing.

### Phases 1-5 came from CE's own validated register

The five phases of self-evolution we shipped are not borrowed from external systems — they were derived from validated insights *in the engine's own ideation_on_ideation journal*. Each phase has a source register entry:

- **Phase 1** (canonicalization + alias-gap) ← `r-c67457` (conf 0.82)
- **Phase 2** (three-stage verifier) ← `r-fcbba1` (conf 0.81)
- **Phase 3** (component-resolved novelty) ← `r-bedce5b1` (conf 0.79)
- **Phase 4** (Pareto admission) ← `r-46988c97` (conf 0.78)
- **Phase 5** (explore/verify space split) ← `r-9a35e387` (conf 0.77)

That is the literal self-evolution claim: the engine identified architectural patterns its own verifier should adopt, and we applied those patterns to the verifier.

### Phases 6+ borrow from comparable public systems

Once the verifier side stabilized, we audited public research-agent systems for ideas the engine could borrow on the *generator* side. Each subsequent phase credits its inspiration:

- **Phase 6** (Best-of-N synthesis with alias-gap ranking) — borrows the **tournament-ranking** pattern from **[Google's AI Co-Scientist](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/)** (Generation / Reflection / Ranking / Evolution agents) plus standard agentic best-of-N selection. Internally aligns with `r-bd1386df` ("cascading structured pairwise tournaments") and `r-c2ee5c80` ("two-player adversarial game over a library") from CE's own register.
- **Phase 7** (persona-conditioned introspection, planned) — borrows multi-perspective question generation from **[Stanford STORM / Co-STORM](https://github.com/stanford-oval/storm)**. Each persona (skeptic / outsider / historian / contrarian / practitioner) surfaces blind spots the single-voice introspection misses.
- **Phase 8** (idea evolution from downgraded extensions, planned) — borrows the mutation loop from **[Sakana AI's "AI Scientist"](https://github.com/SakanaAI/AI-Scientist)** and the Evolution agent from Co-Scientist. Internally aligns with `r-3c792e21` ("typed supervision from false positives via retrospective unification") — we treat verifier downgrades as typed supervision signal for generator-side mutation.
- **Phase 9** (hypothesis variants in investigation, planned) — borrows branching exploration from **[Tree of Thoughts](https://arxiv.org/abs/2305.10601)**. The explorer persona generates N divergent priors; the most-distant-from-majority-literature variant drives the investigation.

### General agentic patterns CE builds on

- **[Claude Code](https://claude.com/claude-code) and the Claude API** — the directive's `Agentic Prompt` block is structured to be pasted directly into Claude Code, MCP orchestrators, or similar LLM-driven agents. The grounding allowlists + tool-call discipline borrow from established agentic patterns.
- **Cross-family adversarial verification** — the principle that a model evaluating its own output produces no signal is broadly understood; using a different-family model as verifier is standard practice in multi-agent systems. CE's contribution is the *append-only audit trail* + *mechanical guards* on top of cross-family verification, not the cross-family idea itself.
- **Retrieval-augmented generation patterns** — `academic_search` + `web_fetch` + `archive_access` are standard RAG plumbing.

### Academic / methodological lineage

- **Hypothesis-first investigation** is broadly Popperian / falsificationist epistemology applied to LLM tool use. The mechanical surprise comparison is standard Bayesian-update structure.
- **Pareto-dominance admission** (Phase 4) is multi-objective optimization theory applied to a register-admission gate.
- **Negative-space mapping** (the `(method × problem)` matrix) is a long-standing literature-review discipline; CE just instruments it.
- **Falsifiable predictions with target dates** is descended from prediction-market and forecasting-literature practice (Tetlock, Good Judgement Project, Metaculus).

### Where CE is genuinely novel

Some specific combinations don't appear (to my knowledge) in any public system:

- **Append-only audit trails** with `reverification_log` — most systems mutate state on re-evaluation.
- **Self-evolving verifier where the engine's own validated insights have been applied to its own architecture** — Phases 1-5 above.
- **Canonical-form alias-gap detection over a research register** — Phase 1's structural similarity check on `(predicate, substrate, mechanism, target_domain, key_constraints)` tuples is not a pattern I've seen in published agentic systems.
- **Pareto admission gate over multi-axis register entries** — Phase 4's tournament between *existing* register entries and incoming candidates.
- **Literature-watch leakage check on directive verification criteria** — preventing the directive from outsourcing its evidence to "by date X a paper appears."
- **Freshness probe at prediction creation time** — closing the dead-on-arrival case where a "prediction" was already true at creation.

If you're aware of a system that does any of those, please open an issue — I'd want to learn from how they handled it.

---

## How novelty is verified — the technical detail

### The problem the verifier is built around

A language model asked "what's novel here?" will confidently narrate something. Most of those narrations are **articulate restatement** — surface variation on something already published. Three failure modes dominate:

- **Self-grading collapse.** If the same model that generated the insight also grades its novelty, there's no independent signal — only confidence theater.
- **Search-novelty paradox.** Surface-text retrieval *penalizes* genuine novelty (no matches yet exist) and *rewards* restatement (good matches imply the claim is already published).
- **Peer-system blind spot.** Even a phase-structured verifier can miss a headline peer system whose existence disqualifies the claim — if the search queries don't explicitly name the target application domain. (We documented missing Google's AI co-scientist on an LLM-research-agents journal once. Adding known-prior-art anchors fixed that class of miss permanently.)

### Phased prior-art search (Stage 3)

```
Phase 0  premises check       — every load-bearing premise must have literature support
Phase 1  central move         — is the headline move already deployed in a peer system?
                                if yes → novelty_type = extension at most
Phase 2  full composite       — is the entire synthesis already published? if yes → restatement
Phase 3a functional decomp    — split into 3-5 functional dimensions; nearest exemplar per dim
Phase 3b closest peer system  — REQUIRED: at least one query must name the target domain
Phase 4  contradicting evidence
Phase 5  reasoning audit      — is the inferential leap from premises to synthesis justified?
Final    skeptic smell test   — 3-5 candidate kill queries; run the most lethal + a followup
```

### Engine-side guards

These fire mechanically without LLM involvement, after the verifier returns. Their job is to refuse the LLM's hedging:

- **Phase-1 guard** — if `central_move_prior_art` has substantive entries, `new_synthesis` downgrades to `extension`.
- **Peer-system guard** — named peer + substantive overlap + no stated differentiators → downgrade.
- **Known-prior-art guard** — any human-curated anchor evaluated as `is_peer + overlaps_claim` with no differentiators → downgrade.
- **Skeptic-probe guard** — `skeptic_probe.disqualifies = true` → verdict downgrades validated→challenged.
- **Challenged-hedge guard** — RLHF-style hedging detected (decomposition says valid synthesis, verdict says challenged, no substantive flaw named) → upgrade to validated.
- **Confidence-drop guard** — any guard-induced verdict downgrade applies a configurable confidence penalty (default 0.10). LLMs hedge verdict but keep confidence flat — that's a tell, we flush it out.
- **Inconclusive guardrail** — verdict=inconclusive without a named epistemic gap → downgrade to challenged.

### Known-prior-art anchors (human-in-the-loop feedback)

When you spot a missed peer, you add it via the Admin tab: `{domain, system_name, url, notes}`. The verifier's prompt then injects matching anchors as *mandatory* evaluation items on every future claim in that domain. A missing evaluation for a listed anchor is a verification failure. The system catches the miss forever; you don't hand-edit the journal.

### Register gate

Two modes, configurable:

| Mode | Rule |
|---|---|
| `scalar` (default) | Verdict must be `validated` AND confidence ≥ floor AND novelty isn't restatement/unsupported. Single quality bar. |
| `pareto` | All scalar checks PLUS Pareto-dominance check. Reject if any existing active entry beats the candidate on every Pareto axis. Catches "slightly worse on every dimension" admissions the scalar floor can't see. |

Held pipeline: verdict=inconclusive + premises_supported + confidence ≥ held_floor → entry held with a settlement plan instead of rejected.

### Falsifiable predictions + freshness check

Every validated entry emits 1-3 predictions with target dates. Before each prediction is stored, a freshness probe runs: a targeted web search on the falsifiable condition + LLM judgment. If results already instantiate the claim, the prediction is stored as `already_fulfilled` rather than `pending`. Closes the "dead-on-arrival" gap where a prediction was already true at creation time.

### Re-verification audit

Verifier rules evolve. `--reverify-register` re-runs the current pipeline over existing register entries *without* mutating them — each pass appends to `reverification_log` with the full new decomposition. The Register tab surfaces audit-changed entries with a "⚠ audit changed" badge showing the delta inline.

---

## Research directives — turning verified claims into executable plans

The register holds verified claims. A research directive translates one of those claims into a concrete plan a human or LLM-driven agent can execute.

Two modes:
- **Per-record** (daily use) — "Export directive →" button on any active validated register card. ~2-3 min per directive.
- **Bundle** — Admin tab button. Runs the directive pipeline across every qualifying entry (validated, with at least one open prediction).

Output: `data/{journal_stem}_directives/r-{id}.md` (or `bundle-{timestamp}.md`) plus a `.verification.json` sidecar.

### How a directive is structured

Six sections, each produced by a focused LLM call (≤1500-1800 tokens):

| Section | Persona | What it produces |
|---|---|---|
| 1. Hypothesis | fast | What the team measures from their own experiments if the theory holds |
| 2. In plain language | fast | 3-5 sentences for a non-specialist |
| 3. Test plan | reasoning | 3-6 executable steps the team runs themselves |
| 4. Agentic prompt | reasoning | Self-contained instruction block for an LLM-driven agent (citation + tool allowlists, no fabrication) |
| 5. Verification criteria | reasoning | Confirmed/Refuted/Inconclusive signals measured from the team's own outputs |
| 6. Research path to publication | fast | Strategic narrative: study design, data, paper structure, target venue class, phases, risks |

Then deterministic assembly + a verifier review pass. If the verifier flags fabrication or hand-wave language, the agentic-prompt section regenerates with the flags appended (up to `directive_max_verification_passes`, default 3). If the loop exhausts, output ships with a `⚠ FLAGGED ISSUES` block prepended.

Two model dials: `directive_primary` (reasoning recommended — sections 3, 4, 5) and `directive_primary_fast` (non-reasoning recommended — sections 1, 2, 6). The verifier is a third dial, `directive_verifier`. Reasoning helps on the translation/grounding sections; fast is better for restatement/style sections where reasoning over-elaborates.

### Grounding rules (non-negotiable)

- Every URL/DOI/arXiv ID must appear verbatim in the citations allowlist (built from the source register entry). The References section is a deterministic dump of that allowlist.
- Every tool name must exact-match the tool allowlist. No "use a search engine" generic phrasings.
- Every step must be concretely executable. Hand-wave patterns (`figure out`, `iterate until`, `try various`) are flagged.
- Verification criteria measure **the team's own experimental outputs**. Literature-watch framings ("by [date], a benchmark paper reports …") are explicitly forbidden by a dedicated `literature_watch_leakage` check.

---

## Setup

### Docker (recommended — fully isolates `code_execution`)

```bash
git clone <this-repo>
cd CuriosityEngine

./curiosity --show-journal       # first run builds the image (numpy/scipy wheels — several min)
./curiosity --cycles 3
./curiosity --review-register
./curiosity web                  # browser UI at http://localhost:8000
```

The wrapper bind-mounts `~/.CuriosityEngine` (config) and `./data` (state). API keys live in `engine.toml` or env: `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `E2B_API_KEY`.

### Local venv (fewer isolation guarantees)

```bash
git clone <this-repo> && cd CuriosityEngine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python curiosity_engine.py --show-journal     # triggers setup wizard
```

The setup wizard asks for primary + verifier model + advanced settings.

**Recommended: different families for primary and verifier** (Anthropic Claude + OpenAI GPT, say). Same-family verification defeats most of the adversarial point.

---

## Browser UI

```bash
./curiosity web      # FastAPI at http://127.0.0.1:8000 · ./curiosity web stop|logs|restart
```

Tabs:

- **Overview** — counts, high-surprise list, recent entries.
- **Entries** — investigations, filterable by domain tag.
- **Insights** — synthesized insights pre-verification.
- **Register** — validated entries. Per-component novelty badges (rose=restatement, amber=extension, emerald=new_synthesis), audit-changed badges, attached predictions, expandable verification detail. Per-entry "Export directive →" button.
- **Predictions** — status tracker with review history.
- **Focus & Queue** — set focus, add user questions, semantic search, source-grouped queue with drag-and-drop priority.
- **Graph** — interactive sigma.js force-directed graph.
- **Runs** — per-journal run history with live SSE log streaming. Stop button on active runs.
- **Coverage** — negative-space gap matrix + diff vs prior scan. Click any cell for detail.
- **Admin** — maintenance operations (re-verify register, scan gaps, export directives bundle, etc.) plus the Pareto frontier viewer and the known-prior-art anchor list.

Bound to `127.0.0.1` only — no auth, single-user.

---

## CLI reference

```bash
# Running cycles
python curiosity_engine.py --cycles 3 --domain "TOPIC"

# Inspecting state (no API calls)
python curiosity_engine.py --show-journal --show-register --show-predictions
python curiosity_engine.py --pareto-frontier         # what sets the admission bar

# Steering
python curiosity_engine.py --set-focus "TEXT"        # also: --show-focus / --clear-focus
python curiosity_engine.py --add-question "?"        # also: --list-questions / --clear-questions
python curiosity_engine.py --review-register         # interactive approve/reject/defer

# Self-evolving verifier diagnostics (no LLM cost)
python curiosity_engine.py --three-stage-test "claim description"      # Stages 1+2
python curiosity_engine.py --pareto-admission-test "0.75,8,4,0.30"     # Pareto check

# Knowledge graph + semantic retrieval
python curiosity_engine.py --graph-summary
python curiosity_engine.py --find-similar "query" --top-k 10
python curiosity_engine.py --embed-backfill            # embed legacy entries

# Predictions
python curiosity_engine.py --check-predictions         # due only
python curiosity_engine.py --check-predictions-all     # every pending

# Re-verification (idempotent, append-only)
python curiosity_engine.py --reverify-insights                    # all unregistered
python curiosity_engine.py --reverify-insight i-abc12345          # single insight
python curiosity_engine.py --reverify-register                    # audit existing register
python curiosity_engine.py --reverify-register-id r-cd730b6d      # single entry
python curiosity_engine.py --reverify-register --reverify-register-max-confidence 0.85

# Maintenance
python curiosity_engine.py --backfill-canonical-forms             # canonicalize register
python curiosity_engine.py --backfill-canonical-forms --backfill-force   # re-canonicalize all
python curiosity_engine.py --synth-orphaned-xrefs                 # recover from mid-run crashes
python curiosity_engine.py --scan-gaps                            # negative-space scan

# Research directives
python curiosity_engine.py --export-directive r-cd730b6d          # per-record
python curiosity_engine.py --export-directives-bundle             # all qualifying

# Per-run model overrides
python curiosity_engine.py --cycles 3 --primary-model gpt-5.1 --verifier-model gpt-5.4
python curiosity_engine.py --cross-ref-only --cross-ref-role verifier
```

---

## Configuration

Runtime settings live at `~/.CuriosityEngine/engine.toml`. The setup wizard creates a working config; the Settings tab edits it via the UI. Key sections:

```toml
[models.primary]
provider = "anthropic"                # or "openai_compat"
name = "claude-sonnet-4-6"
max_tokens = 4096
investigation_max_tokens = 8192
temperature = 1.0                     # 1.0 for reasoning models
timeout_seconds = 300.0

[models.verifier]                     # cross-family recommended
provider = "openai_compat"
name = "gpt-5.1"
base_url = "https://api.openai.com/v1"

# Any additional [models.<name>] profiles are referenced via the role knobs below.

[engine]
# Loop knobs
cross_ref_frequency = 3
novelty_threshold = 0.7
register_confidence_floor = 0.6
verify_insights = true

# Phase 4: register admission policy
register_admission_mode = "scalar"    # "scalar" | "pareto"

# Generative probes
analog_probe_enabled = true           # cross-domain on high-surprise
analog_probe_surprise_threshold = 0.5
assumption_probe_enabled = true       # within-domain on low-surprise + confirmed
assumption_probe_surprise_threshold = 0.3

# Held pipeline
held_entries_enabled = true
held_confidence_floor = 0.7

# Negative-space gap mapping
negative_space_min_entries = 15
gap_verification_hit_threshold = 5

# Verifier guards
confidence_drop_on_downgrade = 0.10

# Per-phase model routing — empty/"primary" = use primary; "verifier" = use verifier;
# any other name must match a [models.<name>] section.
cross_ref_role = ""                   # cross-reference synthesis
directive_primary_role = ""           # heavy directive sections (test plan, agentic, criteria)
directive_primary_fast_role = ""      # light directive sections (hypothesis, eli5, research path)
directive_verifier_role = ""          # directive grounding-review
gap_scan_extract_role = ""            # gap scan matrix extraction (reasoning helps)
gap_scan_classify_role = ""           # gap scan classification (non-reasoning preferred)
investigation_assessor_role = ""      # Phase 5: post-search assess persona
```

### OpenAI-compat endpoint shortcuts

| Provider | base_url |
|---|---|
| OpenAI | `https://api.openai.com/v1` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Ollama (local) | `http://localhost:11434/v1` |
| xAI | `https://api.x.ai/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| Together | `https://api.together.xyz/v1` |
| DeepSeek | `https://api.deepseek.com/v1` |
| Moonshot (Kimi) | `https://api.moonshot.ai/v1` |

---

## Architecture

```
curiosity_engine.py          CLI entry
engine/                      orchestrator (mixin composition over the engine class)
  core.py                      class def, model client plumbing, main run loop
  introspect.py                introspection → questions
  investigation.py             hypothesis → tool-loop → assessor (Phase 5)
  cross_reference.py           xref → synthesize (graph-aware, cross-domain biased)
  verification.py              three-stage verifier + reverify + Pareto admission
                                 + canonicalization + alias-gap + component novelty
                                 + frontier viewer + admission test
  negative_space.py            gap scan + coverage diffing + known-prior-art matching
  directives.py                research directive export pipeline (multi-call harness)
  display.py                   show_* (no API calls)
  graph.py                     NetworkX knowledge graph + cross-ref selector
  embeddings.py                OpenAI embeddings + cosine similarity
  tools/                       pluggable tools (auto-discovered)
    base.py                      Tool ABC + RateLimiter + ToolRegistry
    _rate_limits.py              named limiters (ARXIV, CROSSREF, OPENALEX, …)
    web_fetch.py                 HTTP GET + plaintext extraction (SSRF-safe)
    web_search.py                DuckDuckGo + Bing HTML (keyless)
    academic_search.py           Crossref + arXiv + OpenAlex + (Semantic Scholar opt-in)
    archive_access.py            Internet Archive + Wikimedia + Openverse
    calculator.py                AST-based safe math
    citation_manager.py          Local bibliography + BibTeX/APA
    peer_review.py               Deterministic rubric (no LLM)
    code_execution.py            Local subprocess or E2B sandbox
config.py                    CuriosityEngineConfig + interactive setup
providers.py                 ModelClient ABC + Anthropic / OpenAI-compat
journal.py                   JSON-backed journal + atomic writes + known-prior-art CRUD
register.py                  markdown rendering for the human-readable register
models.py                    dataclasses (RegisterEntry carries canonical_form,
                              component_novelty, pareto_axes, reverification_log, ...)
prompts.py                   all prompts (domain-agnostic — no field-specific examples)
```

### Tools

The investigator AND the verifier see the full tool set:

| Tool | Provider | Keyless | What it does |
|---|---|---|---|
| `web_search` (Anthropic server) | Anthropic | n/a | Native server-side |
| `web_search` (client) | All | ✅ | DuckDuckGo + Bing HTML fallback |
| `web_fetch` | All | ✅ | HTTP GET + trafilatura extraction |
| `academic_search` | All | ✅ | Crossref + arXiv + OpenAlex (+ Semantic Scholar opt-in) |
| `archive_access` | All | ✅ | Internet Archive + Wikimedia + Openverse |
| `calculator` | All | ✅ | AST math; npv/cagr/wacc/pmt |
| `citation_manager` | All | ✅ | Local bibliography → BibTeX/APA |
| `peer_review` | All | ✅ | Deterministic rubric |
| `code_execution` (Anthropic server) | Anthropic | n/a | Native sandboxed Python |
| `code_execution` (client) | All | ✅ | Local subprocess + optional E2B |

**Rate limits** (shared process-wide; fire before every network call):

- arXiv: 1 req/5s + 2.0s jitter (documented limit is 1/3s but burst-detection on sustained workloads triggers 429s well below the published rate; slowed pacing + cooldown-on-429 is belt-and-suspenders).
- Semantic Scholar: 1/5s + 2.0s jitter — opt-in via `SEMANTIC_SCHOLAR_API_KEY`. Without a key the public tier shares a global bucket across all unauthenticated callers and exhaustion is routine, so SS is silently dropped from the default source list.
- Crossref: 5 req/s + 0.2s jitter.
- OpenAlex: 10 req/s + 0.2s jitter. Set `OPENALEX_MAILTO` to enter the polite pool.
- `web_fetch`: per-host 3 req/s + 0.3s jitter.

429s on academic endpoints engage a staged cooldown (2s → 4s → 8s → 16s → 30s, cycling back) so transient throttles don't trigger 60-second overcorrections. Jitter is added *after* token acquisition so we don't look like a fixed-interval bot to upstream throttlers.

**Adding a tool**: subclass `engine.tools.base.Tool` with `name` / `description` / `input_schema` / `execute(args) -> str`. Auto-registered on engine startup.

**Security**: the Docker path fully isolates `code_execution`. The local subprocess path is *not* a sandbox — use E2B for isolation without Docker.

### Knowledge graph

`engine/graph.py` builds a NetworkX multigraph over entries / xrefs / insights / register / predictions / sources / tags. Source normalization canonicalizes arxiv / DOI / title surface forms so `cites-source` edges fire across variants. Cross-domain bias in selection: `shares-tag` weighted at 0.25×, +2.5 bonus when connected entries have zero `domain_tags` overlap.

### Semantic retrieval

`engine/embeddings.py` — OpenAI-compat embeddings (default `text-embedding-3-small`). Auto-embed on entry creation. Gracefully disabled if no embedding-capable profile is configured.

---

## Honest limitations

- **Model overlap remains.** Even cross-family verifier and primary share training-data overlap. The phased verifier + canonicalization layer narrow the failure mode but don't eliminate it.
- **The verifier doesn't know what it doesn't know.** Known-prior-art anchors close blind spots a human has identified, but can't catch the ones nobody has named yet.
- **`code_execution` is not GPU-scale.** Handles analytical checks, small simulations, and re-deriving cited numbers — not model training or large-scale analysis.
- **Style bias toward articulate claims.** Everything is structured JSON. Pre-articulate or paragraph-needing insights are systematically selected against.
- **Attractor basins still happen** under narrow focus, despite anti-attractor gates. The cross-domain analog probe is the primary counterforce.
- **The 18 dark register entries** in old journals (entries that predate canonicalization) are invisible to alias-gap detection until reverified. Phase 1+ coverage is incremental.

What this is **good** for:

- Session-spanning research discipline that enforces hypothesis-first hygiene.
- Surfacing questions worth investigating, even when the "insights" turn out derivative.
- Building a substantiated, citation-linked reasoning trail (`register.md`).
- Accelerating connection-finding across your own accumulated work.
- Producing executable research plans (via directives) that another agent or researcher can act on.

---

## Files generated at runtime

- `~/.CuriosityEngine/engine.toml` — model + engine config.
- `./data/research_journal.json` (or `--journal`) — full journal state, including `register[].canonical_form / component_novelty / pareto_axes`.
- `./data/register.md` — human-readable register artifact, auto-written on register changes.
- `./data/{journal_stem}_directives/` — generated research directives + `.verification.json` sidecars.
- `./data/_runs/{journal_stem}/*.log|.meta.json` — run logs + metadata, streamed to the Runs tab.

All runtime artifacts are `.gitignore`d.

---

## Development

```bash
.venv/bin/python curiosity_engine.py --help
.venv/bin/ruff check --exclude .venv --exclude data .
pre-commit install                                   # ruff + trufflehog secret scanning
```

CI runs ruff + syntax + smoke + Docker build on every push/PR to `main`.

---

## License

**Copyright (c) 2026 sfw. All rights reserved.**

Shared for reference and evaluation only. No part of the code, documentation, or prompts may be copied, redistributed, modified, or used in derivative or commercial works without the copyright holder's express written permission. See [LICENSE.md](./LICENSE.md) for full terms.
