# Curiosity Engine

A domain-agnostic research loop. The engine generates its own questions from self-assessed uncertainty, investigates them with web search + academic APIs + sandboxed code execution, accumulates state across sessions, and routes synthesized insights through an independent cross-family verifier before committing them to a durable register. Every validated entry carries falsifiable predictions attached for later review. Every generated directive can be exported as an executable research plan with an agentic prompt a human can hand to an LLM-driven agent.

The engine is designed to be **driven iteratively**: set a focus, run cycles, inspect the knowledge graph, inject your own questions, reject shallow claims, and continue. It won't produce novel ideas on its own; with steering it will surface candidates that survive real adversarial scrutiny — regardless of whether the topic is AI research, biology, investing, marketing, governance, or anything else a human studies.

> **Status**: proof of concept. Phases 1–4 and 6 of the roadmap are implemented; Phase 5 (multi-agent disagreement) is planned. See [Honest limitations](#honest-limitations) for what this system genuinely *can't* do and [Roadmap](#roadmap) for what's landed.

---

## Core thesis

> Novel insight rarely emerges from a single query. It emerges when four directions of search compose — **reaching outward** to foreign domains, **reaching inward** to premise-level assumptions, **connecting existing** findings into non-obvious cross-references, and **locating missing** combinations in the possibility space. Adversarial verification then separates genuine synthesis (*premises real, composite claim not in literature*) from articulate restatement — using a phased prior-art search that catches peer systems which structural-query searches miss.

## Design principles

The principles are load-bearing — they shape every prompt, every config default, every verdict gate.

- **Domain-agnostic by construction.** Every prompt describes structure, not content. No field-specific examples, no illustrative domains, no canned example observables. The same engine works identically for AI research, bioengineering, investing, marketing, governance, or any other structured field. `{engine_domain}` is threaded through every LLM call so the model anchors on the journal's actual subject rather than a memorized default.
- **Hypothesis before evidence.** The investigator commits to a specific falsifiable answer *before* searching. Surprise is then a comparison, not a self-report.
- **Novelty is structural, not vibes-based.** Premises vs synthesis are scored independently. The register gate requires either (a) `new_synthesis` with `synthesis_findable=false` or (b) `extension` with substantive peer differentiators.
- **Grounding over generation.** Research directives pin every citation and tool reference to an allowlist derived from source material. A verifier flags any fabrication before the output reaches a human; anything it can't verify ships with a prominent `⚠ FLAGGED ISSUES` block.
- **Cross-domain is a first-class move.** Analog probe (outward reach) + assumption probe (inward reach) + negative-space mapping (absent combinations) + known-prior-art anchors (human-curated peers the verifier must consider).
- **Anti-starvation scheduling.** Question queue uses source-aware round-robin with in-source priority — xref questions can't perpetually starve gap or analog questions.
- **Audit trails, not overwrites.** Re-verify operations append to `reverification_log` rather than mutating original verdicts. The old and new are both preserved so you can see what changed under updated rules.
- **Durable state across sessions.** Every cycle persists to JSON. Full provenance reconstructable from any register entry back to the journal entries that triggered it.
- **Human-in-the-loop, not human-out-of-loop.** `--set-focus`, `--add-question`, `--review-register`, known-prior-art anchors — all first-class controls. Human rejections feed back into future verifier prompts.

---

## Pipeline

```
  introspect       →  uncertainties (what the model is uncertain about)
  generate         →  ranked investigable questions
                           ↑
         (source-round-robin dequeue + priority floor + human-first lane)
  investigate                 (three stages — primary model)
    ├─ hypothesize  →  commit to a pre-investigation answer
    ├─ search       →  web_search + web_fetch + academic_search + code_execution
    └─ assess       →  compare findings to hypothesis, compute surprise_delta

  (analog probe)    →  on surprise ≥ threshold, ask for distant-domain analogs
                       (named sub-fields + mechanisms + translated questions)
                       enqueued at priority 0.85
  (assumption probe)→  on surprise ≤ threshold AND verdict=confirmed, surface
                       implicit premises the field takes for granted; produce
                       negation questions → enqueued at priority 0.80
  (embed)           →  OpenAI embeddings on question + takeaways

  cross-reference   →  graph-aware selection biased toward CROSS-DOMAIN neighbors
                       anti-attractor gate skips ≥50% participant-overlap candidates
  synthesize        →  promote high-novelty xrefs to insights
  verify (cross-family)
    ├─ Phase 0: premises check
    ├─ Phase 1: central architectural move — is the headline move already deployed?
    ├─ Phase 2: full composite claim prior art
    ├─ Phase 3a: functional-dimension decomposition (nearest exemplar per dimension)
    ├─ Phase 3b: closest complete peer system (DOMAIN-ANCHORED query required)
    ├─ Phase 4: contradicting evidence
    ├─ Phase 5: reasoning audit
    └─ Final: skeptic smell test — 3-5 candidate kill queries + followup
    └─ Engine-side guards: phase-1, skeptic-probe, peer-system, known-prior-art,
                          challenged-hedge upgrade, confidence-drop on downgrade
    └─ Register gate: validated + (new_synthesis | correction | extension-with-diff)
        └─ Emit 1-3 falsifiable predictions + FRESHNESS CHECK
            → already_fulfilled predictions marked at creation time, not queued

  human review      →  approve / reject-with-reason / defer / promote held entries
  check predictions →  revisit due predictions; reconcile entry status

  (on-demand ADMIN operations)
    ├─ scan-gaps             — (method × problem) matrix + gap verification + enqueue
    ├─ reverify-insights     — re-run verification on unregistered insights
    ├─ reverify-register     — re-verify EXISTING register (append-only audit)
    ├─ export-directive <id> — generate executable research plan for one entry
    └─ export-directives-bundle — same pipeline across all qualifying entries
```

Every step persists to `research_journal.json`. The human-readable artifact is `register.md`, auto-written whenever the register changes. Generated research directives land in `data/{journal_stem}_directives/`.

---

## Technical briefing — how novelty is conceived and verified

### The problem the engine is built around

A language model asked "what's novel here?" will confidently narrate something. The narration often reads as insight but is usually **articulate restatement**. Three failure modes dominate:

1. **Self-grading collapse.** The same model that generates an insight also grades its novelty — no independent signal.
2. **RAG novelty paradox.** Surface-text retrieval penalizes genuine novelty (no matches yet exist) and rewards restatement (good matches imply the claim is already published).
3. **Ingredients-as-weakness mis-framing.** A truly novel synthesis is a composition of established ingredients. Naive verifiers reject this pattern as "the components are already known" — which is the *signature* of novelty, not a weakness.
4. **Peer-system blind spot.** Even a phase-structured verifier can miss a headline peer system whose existence disqualifies the claim as new synthesis — if the search queries don't explicitly name the target application domain.

### How novelty is *conceived*

**1. Hypothesis-first investigation.** The primary commits to a specific falsifiable answer (`HYPOTHESIS_PROMPT`) before any search runs. `SURPRISE_PROMPT` later compares findings-vs-hypothesis into `surprise_delta ∈ [0,1]`. Because the hypothesis is fixed before evidence arrives, surprise is a *structural comparison*, not a self-report.

**Confidence calibration.** Rules in the prompt AND a post-hoc clamp (`_calibrate_confidence_after`) — on `partially_confirmed + surprise_delta < 0.2`, confidence cannot increase; on `contradicted`, drop ≥0.2; on `unresolved`, pull toward 0.5.

**2. Cross-domain analog probe.** Fires when `surprise_delta ≥ analog_probe_surprise_threshold` (default 0.5). The prompt is deliberately domain-neutral — no biology / economics / thermodynamics examples baked in. The LLM picks the distant domain based on the finding's structural fingerprint; the engine leverages LLM analogical-reasoning capability rather than hard-coding a rotation. Output enqueued at priority 0.85 with source `analog:<entry-id>`.

**2b. Within-domain assumption probe.** Complementary — fires on *low-surprise confirmed* findings where field consensus hides load-bearing assumptions. Trigger is inverse: `surprise_delta ≤ assumption_probe_surprise_threshold` AND `verdict=confirmed`. Asks the primary for implicit premises and emits negation questions at priority 0.80.

**3. Source-round-robin question queue.** Each emergent question carries a `source` tag (`human` / `xref:*` / `gap:*` / `analog:*` / `assumption:*` / `entry:*`). `pop_queued_questions` rotates across non-empty sources each cycle, pulling the highest-priority item from each in turn. Hard guarantee: as long as a source has items, it gets visited — xref questions at 0.87+ can't perpetually starve gap or analog questions at 0.85. A `question_priority_floor` (default 0.70) drops non-human questions below a threshold at enqueue time; human-sourced questions always bypass.

**4. Cross-reference with structural bias.** Every `cross_ref_frequency` cycles (default 3), the graph-aware selector weights `shares-tag` edges at 0.25× and applies a +2.5 cross-domain bonus when connected entries have zero `domain_tags` overlap. Anti-attractor gate: candidates with ≥50% participant overlap with an existing xref are skipped unless claimed novelty ≥ 0.85.

**5. Synthesis with self-check.** High-novelty xrefs (`novelty_score ≥ novelty_threshold`, default 0.7) become Insights via `SYNTHESIZE_PROMPT`, which includes a cheap `prior_art_check` before the expensive cross-model verifier.

**5b. Negative-space gap mapping (admin-triggered).** Locates what *isn't* in the journal:
1. Matrix extraction — `(method, problem)` pairs from entries + tags as anchors.
2. Compute empty cells = `Cartesian(methods × problems) − covered`.
3. Classify each: `underexplored` / `tried_failed` / `trivially_uninteresting` / `regulated_boundary` / `adjacent_but_covered`.
4. Verify `underexplored` cells via `academic_search` — STRUCTURED counts via `count_results_structured()` (not substring parsing of formatted text) + per-query error tracking. Cells where verification errored are marked `verification_incomplete` and preserved in the scan artifact but *not* enqueued.
5. Generate 1–2 investigable questions per verified gap → enqueue at 0.85 with source `gap:<short-id>`.

Scans persist to `journal.coverage_scans`. The Coverage tab surfaces the matrix *and* a diff against the prior scan: filled gaps, still-open gaps, newly-emerged gaps.

### How novelty is *verified*

**6. Phase-structured prior-art search.** The verifier runs a 6-phase agentic sequence, not a single composite query (a single compound query is how we missed Google's AI co-scientist on an LLM-research-agents journal):

| Phase | What | Why |
|---|---|---|
| 0 | Premises check | Every load-bearing premise must have literature support |
| 1 | Central architectural move | Is the headline move already deployed in a peer system? If yes → `novelty_type=extension` at most |
| 2 | Full composite claim | Is the entire synthesis already published? If yes → `restatement` |
| 3a | Functional-dimension decomposition | Split claim into 3-5 functional dimensions (e.g. *act-on / mechanism / scale / constraint / substrate*); name the nearest exemplar per dimension |
| 3b | **Closest complete peer system** | Required: at least one query MUST explicitly name the claim's target application domain. This is the query that catches system-level peers |
| 4 | Contradicting evidence | Active disagreement with the composite claim |
| 5 | Reasoning audit | Is the inferential leap from premises to synthesis justified? |
| Final | Skeptic smell test | Enumerate 3–5 candidate *kill queries*; run the most lethal; if `disqualifies=false`, run a followup attacking a different angle |

**Engine-side guards** (mechanical, fire without LLM involvement):
- **Phase-1 guard** — substantive `central_move_prior_art` → `new_synthesis` downgrades to `extension`.
- **Peer-system guard** — named peer + substantive overlap + no stated differentiators → downgrade.
- **Known-prior-art guard** — any anchor evaluated as `is_peer=true` + `overlaps_claim=true` + empty differentiators → downgrade.
- **Skeptic-probe guard** — `skeptic_probe.disqualifies=true` → verdict downgrades validated→challenged.
- **Challenged-hedge guard** — RLHF hedging detected (decomposition says valid new synthesis, verdict says challenged, no substantive reasoning flaw named) → upgrade to validated.
- **Confidence-drop guard** — any guard-induced verdict downgrade triggers a configurable confidence penalty (default 0.10) so stored confidence reflects the revised assessment (LLMs hedge verdict but keep confidence flat — that's a tell, we flush it out).
- **Inconclusive guardrail** — verdict=inconclusive without a named epistemic gap in the summary downgrades to challenged.

**Known prior-art anchors** (human-curated feedback loop). When a human spots a missed peer (e.g. the Google co-scientist case we documented), they add it via the Admin tab: `{domain, system_name, url, notes}`. The verifier's prompt injects anchors whose domain matches the engine domain (token-set match with plural/singular collapse — not strict substring) as *mandatory* evaluation items. A missing evaluation for a listed anchor is a verification failure. The system catches the miss forever; the human doesn't hand-edit the journal.

**7. Register gate.**

| novelty_type | premises | synthesis_findable | peer differentiators | Outcome |
|---|---|---|---|---|
| new_synthesis / correction | ✓ | ✗ | — | Register |
| extension | ✓ | — | ≥1 substantive | Register (marked as extension) |
| extension | ✓ | — | none | Reject (`extension_without_differentiators`) |
| restatement | — | — | — | Reject |
| unsupported | ✗ | — | — | Reject |

Confidence must clear `register_confidence_floor` (default 0.6). Inconclusive + premises_supported + confidence ≥ `held_confidence_floor` (default 0.7) → held with a settlement plan.

**8. Re-verification audit.** Because verifier rules evolve (phased search, skeptic probe, known-prior-art injection, etc.), `--reverify-register` re-runs the current pipeline over existing register entries *without* mutating them. Each pass appends to `reverification_log` with the full new decomposition: new verdict, new novelty_type, new confidence, new central_move_prior_art, new closest_peer_system, new skeptic_probe, new_known_prior_art_evaluations, and full tool-call trace. The Register tab surfaces audit-changed entries with a sky-blue "⚠ audit changed" badge showing the delta inline (e.g. `new_synthesis→extension`).

**9. Prior-human-rejection feedback.** `--review-register` rejection reasons feed into future verifier prompts as "patterns to avoid repeating."

**10. Falsifiable predictions + freshness check.** For every validated register entry, the verifier emits 1–3 predictions. Before each prediction registers, a **freshness probe** runs: a targeted `web_search` on the falsifiable condition + short LLM judgement. If top results already instantiate the claim, the prediction is stored with status `already_fulfilled` — not `pending`. This closes the "dead-on-arrival prediction" gap where a claim was already true at creation time.

### Tool-call audit trail

Every verifier tool invocation is captured on the resulting register entry as `verification_tool_calls`: `{iteration, tool, kind, args, result_length, result_preview, is_error}`. Visible in the Register tab under "Verification detail". After-the-fact audits of verifier misses become 10-second inspections instead of 10-minute reconstruction exercises.

---

## Research directives — turning register entries into executable plans

The register captures verified claims. A research directive translates one of those claims into something a human (or an LLM-driven agent) can execute to actually test it.

Two modes:

- **Per-record** (daily use) — "Export directive →" button on any active validated register card. Runs the full pipeline scoped to ONE entry in ~2-3 min. Output: `data/{journal_stem}_directives/r-{id}.md` + `.verification.json` sidecar.
- **Bundle** (periodic snapshot) — Admin tab button. Runs per-record on every qualifying entry (validated × at least one open prediction). Output: `bundle-{timestamp}.md`.

**Architecture: multi-call harness.** Each section is either deterministic restructuring (Theory, Prior Art Positioning, References — pure template from existing fields) or produced by a focused LLM call bounded at ≤1500-2500 output tokens:

1. **Hypothesis** (LLM) — what would be observable if the theory is true
2. **Test plan** (LLM, conditioned on hypothesis) — 3-6 executable steps
3. **Agentic prompt** (LLM, conditioned on everything prior + allowlists) — self-contained instruction block for an LLM-driven agent
4. **Verification criteria** (LLM) — Confirmed / Refuted / Inconclusive observable signals

Then deterministic assembly + one verifier review pass for grounding. If flagged, *selective retry* of only the agentic-prompt section (the riskiest). If still flagged, output ships with a prominent `⚠ FLAGGED ISSUES` block prepended.

**Grounding discipline** (non-negotiable):

- Every URL/DOI/arXiv ID referenced must appear verbatim in a citations allowlist built from the source register entry's existing citations. No novel URL generation.
- Every tool named in the agentic prompt must exact-string-match a tool allowlist. No generic "use a search engine" phrasing — only `web_search(query=...)`, `academic_search(query=..., sources=[...])`, etc.
- Every step must be concretely executable. Hand-wave patterns (`figure out`, `iterate until`, `try various`, `use an appropriate tool`) are detected by the verifier and flagged.
- Verification criteria must be objectively measurable — numerical threshold, specific output pattern, citation count, dataset match. Vague language is flagged.

**Per-section progress visibility** — heartbeat log lines between sections so you can see exactly where the pipeline is:

```
[  0.0s] generating hypothesis (1/4)
[ 23.1s] generating test plan (2/4)
[ 54.8s] generating agentic prompt (3/4)
[ 92.4s] generating verification criteria (4/4)
[112.1s] assembling markdown
[112.1s] running verifier review
[148.9s] verifier: ✓ clean
```

A hang in any one section is visible within 30-60s, not buried in a 15-minute silence. Failures localise.

---

## Prerequisites

- **Docker** (recommended) — container fully isolates `code_execution` from your host.
- Or **Python 3.11+** for the local venv path (stdlib `tomllib`).
- API key for at least one provider:
  - **Anthropic** (Claude) — supports server-side `web_search` and `code_execution` natively.
  - **OpenAI-compat** — any endpoint speaking the OpenAI chat-completions protocol: OpenAI, Gemini, OpenRouter, Ollama, xAI, Groq, Together, DeepSeek, Moonshot, LM Studio.

Recommended: **different families for primary and verifier** (Anthropic Claude + OpenAI GPT, say). Same-family verification defeats most of the adversarial point.

## Setup

### Docker (recommended)

```bash
git clone <this-repo>
cd CuriosityEngine

./curiosity --show-journal       # first run builds the image (numpy/scipy/sklearn wheels — several min)
./curiosity --cycles 3
./curiosity --review-register
./curiosity web                  # browser UI at http://localhost:8000 · ./curiosity web stop|logs|restart
./curiosity --rebuild --show-journal    # force rebuild after dependency update
```

The wrapper bind-mounts `~/.CuriosityEngine` (config) and `./data` (state). API keys live in `engine.toml` or env: `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `E2B_API_KEY`.

Migrating a pre-Docker journal:

```bash
mkdir -p ./data && cp research_journal.json ./data/ && cp register.md ./data/
```

First-run setup wizard + `--review-register` + interactive prompts all work — the wrapper opens a TTY.

### Local venv (fewer isolation guarantees)

```bash
git clone <this-repo> && cd CuriosityEngine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install numpy scipy pandas scikit-learn matplotlib    # optional — for code_execution
python curiosity_engine.py --show-journal                 # triggers setup wizard
```

Wizard asks: (1) primary model, (2) verifier model, (3) advanced settings. Existing `engine.toml` auto-migrates to multi-profile schema. See `engine.toml.example` for the complete annotated config.

---

## Browser UI

```bash
./curiosity web      # FastAPI at http://127.0.0.1:8000 · ./curiosity web stop|logs|restart
```

Pages:

- **Journals** — cards per journal with counts, focus, and recent domains.
- **Journal view** (`/journals/{name}`) — tabbed interface:
  - *Overview*: counts + high-surprise list + recent entries.
  - *Entries*: filterable by domain tag.
  - *Insights*: synthesized insights (pre-verification).
  - *Register*: card per validated insight. Novelty filter (audit-aware — effective novelty = latest audit's value if re-verified, else original). Audit-changed badges (⚠ sky-blue) on entries whose audit produced a different verdict. Attached predictions inline with per-status counts + links. Expandable "Verification detail" shows full phase-structured output: central move, closest peer system, functional decomposition, skeptic probe query + outcome, known-prior-art evaluations, tool-call trace, re-verification log. Inline approve / reject / defer. **Export directive →** button per validated active entry.
  - *Predictions*: status tracker with review history + freshness-check metadata.
  - *Focus & Queue*: set focus, add/clear user questions, semantic search. Queue shown as source-grouped collapsible sub-lists (human / xref / gap / analog / assumption / entry / fresh). Drag-and-drop priority reordering within each bucket.
  - *Graph*: interactive sigma.js force-directed graph; click any node for full-text detail.
  - *Runs*: per-journal run history with deep-link anchors (`#run-<id>`). Admin buttons jump straight to the specific run's live log. Failed runs collapsed behind a disclosure.
  - *Coverage*: negative-space gap scan as a visual (method × problem) matrix + diff panel showing changes vs prior scan (filled / still-open / emerged).
  - *Admin*: maintenance operations with work-to-do counters. Cards for: re-run cross-reference, synthesize orphaned xrefs, re-verify unregistered insights, **re-verify register entries (audit)** with filters, scan for gaps, check predictions, **export research directives (bundle)** with prior-directives listing. Separate section for **known prior art anchors** — add/list/remove the human-curated peers the verifier must consider.
- **Run** — form to start a cycle, live SSE stream.
- **Settings** — edit `engine.toml` in place. Every `[engine]` knob exposed including the new parallel/floor/confidence-drop/hit-threshold fields. Non-destructive save preserves additional `[models.<name>]` profiles.
- **Top bar** — activity indicator (emerald=idle, amber=running). Click the amber dot to jump to the active Runs tab.

Bound to `127.0.0.1` only — no auth, single-user.

---

## Usage

### Iterative workflow

The engine is designed as a loop: focus → run → review → redirect → continue.

```bash
./curiosity --set-focus "TOPIC — specific lens" --journal data/my.json
./curiosity --cycles 3 --domain "TOPIC" --journal data/my.json
./curiosity --show-register --journal data/my.json
./curiosity --review-register --journal data/my.json   # approve / reject-with-reason / defer
./curiosity --add-question "a directed question" --journal data/my.json
./curiosity --cycles 3 --journal data/my.json
./curiosity --check-predictions --journal data/my.json
```

How steering propagates:
- `--set-focus` persists on the journal; every prompt gets a `USER FOCUS` section.
- `--add-question` pushes to `question_queue` with `source=human`; investigated first every cycle.
- `--review-register` rejection reasons inject into future verifier prompts as "prior human rejections."
- Known-prior-art anchors (Admin tab) force the verifier to evaluate specific peer systems on every future claim in the matching domain.

### Command reference

```bash
# Running cycles
python curiosity_engine.py --cycles 1
python curiosity_engine.py --cycles 3 --domain "TOPIC"
python curiosity_engine.py --cycles 3 --journal "./mlp.json"
python curiosity_engine.py --cross-ref-only        # skip investigation; just cross-ref + synth + verify

# Inspecting state (no API calls)
python curiosity_engine.py --show-journal --show-insights --show-register --show-predictions
python curiosity_engine.py --list-tools

# Steering
python curiosity_engine.py --set-focus "TEXT"   # --show-focus / --clear-focus
python curiosity_engine.py --add-question "?"   # --list-questions / --clear-questions

# Human review
python curiosity_engine.py --review-register

# Knowledge graph + semantic retrieval
python curiosity_engine.py --graph-summary
python curiosity_engine.py --graph-export graph.json   # .graphml / .gexf / .json
python curiosity_engine.py --embed-backfill            # embed legacy entries
python curiosity_engine.py --find-similar "query" --top-k 10

# Predictions
python curiosity_engine.py --check-predictions         # due only
python curiosity_engine.py --check-predictions-all     # every pending, ignore target_date

# Re-verification (safe to re-run — idempotent)
python curiosity_engine.py --reverify-insights                      # unregistered insights
python curiosity_engine.py --reverify-insight i-abc12345            # single insight
python curiosity_engine.py --reverify-register                      # audit existing register (append-only)
python curiosity_engine.py --reverify-register-id r-cd730b6d        # single register entry
python curiosity_engine.py --reverify-register --reverify-register-max-confidence 0.85
python curiosity_engine.py --reverify-register --reverify-register-novelty-types new_synthesis,correction

# Other maintenance
python curiosity_engine.py --synth-orphaned-xrefs                   # recover from mid-run crashes
python curiosity_engine.py --scan-gaps                              # negative-space scan

# Research directives export
python curiosity_engine.py --export-directive r-cd730b6d            # per-record
python curiosity_engine.py --export-directives-bundle               # all qualifying

# Per-run model overrides
python curiosity_engine.py --cycles 3 --primary-model gpt-5.1 --verifier-model gpt-5.4
python curiosity_engine.py --cycles 3 --primary-role verifier       # profile swap
python curiosity_engine.py --cross-ref-only --cross-ref-role verifier

# Per-run engine-knob overrides (all optional; default inherit from engine.toml)
python curiosity_engine.py --cycles 6 \
    --cross-ref-window 10 --investigations-per-cycle 2 \
    --novelty-threshold 0.65 --register-confidence-floor 0.6 \
    --verify-insights --analog-probe-enabled --analog-probe-threshold 0.4 \
    --held-entries-enabled --held-confidence-floor 0.7
```

Cross-reference runs every `cross_ref_frequency` cycles. High-novelty xrefs get synthesized; the register gate requires `verdict=validated` AND either `new_synthesis/correction + !synthesis_findable` OR `extension + peer_has_differentiators`, with `verified_confidence ≥ floor`.

---

## Configuration

Runtime settings live at `~/.CuriosityEngine/engine.toml`. Structure:

```toml
[models.primary]
provider = "anthropic"                # or "openai_compat"
name = "claude-sonnet-4-6"
# api_key = "..."                     # or rely on ANTHROPIC_API_KEY / OPENAI_API_KEY env
# base_url = "..."                    # only for non-default endpoints
max_tokens = 4096
investigation_max_tokens = 8192
temperature = 1.0                     # 1.0 for reasoning models (Kimi K2.x, GPT-5 thinking, o-series)
timeout_seconds = 300.0               # per-request HTTP timeout; raise for reasoning models on big prompts

[models.verifier]                     # optional; defaults to primary
provider = "openai_compat"
name = "gpt-5.1"
base_url = "https://api.openai.com/v1"
timeout_seconds = 300.0

# Any number of additional [models.<name>] profiles for role-based routing.
# [models.cross_ref]                  # referenced via [engine].cross_ref_role

[retry]
max_attempts = 5
base_delay_seconds = 0.5
max_delay_seconds = 8.0
jitter_seconds = 0.25

[engine]
# Loop knobs
cross_ref_window = 20
questions_per_cycle = 3
investigations_per_cycle = 1
cross_ref_frequency = 3
novelty_threshold = 0.7
register_confidence_floor = 0.6
verify_insights = true
cross_ref_role = ""                   # empty=primary · offload cross-ref to faster non-reasoning model

# Generative probes
analog_probe_enabled = true
analog_probe_surprise_threshold = 0.5
analog_probe_max_analogs = 3          # how many analogs convert to enqueued questions
assumption_probe_enabled = true
assumption_probe_surprise_threshold = 0.3
assumption_probe_max_assumptions = 3

# Held pipeline
held_entries_enabled = true
held_confidence_floor = 0.7

# Negative-space gap mapping
negative_space_min_entries = 15
gap_verification_hit_threshold = 5    # structured-hit count below this → gap confirmed empty

# Verifier guards
confidence_drop_on_downgrade = 0.10   # subtracted from confidence when a guard downgrades verdict

# Question queue scheduling
question_priority_floor = 0.70        # non-human enqueues below this floor are dropped

# Parallel fan-out (experimental — rate limiters are shared so this redistributes wait time, not burst upstream)
parallel_investigations = 1           # concurrent investigate() calls per cycle
parallel_xref_pipeline = 1            # concurrent synth→verify→register pipelines per cycle
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
engine/                      orchestrator package (mixin composition)
  core.py                      class def, init, model plumbing, main run loop
  introspect.py                Phase 1-2: uncertainties → questions
  investigation.py             Phase 3: hypothesis → tool-loop → surprise + analog/assumption probes
  cross_reference.py           Phase 4-5: xref → synthesize (graph-aware selection)
  verification.py              Phase 6-7: adversarial verify + reverify-register + prediction lifecycle
  negative_space.py            Gap scan + coverage diffing + known-prior-art matching
  directives.py                Research directive export pipeline (multi-call harness)
  display.py                   show_* (no API calls)
  graph.py                     NetworkX knowledge graph + cross-ref selector
  embeddings.py                OpenAI embeddings + cosine similarity + find_similar
  tools/                       pluggable tools (auto-discovered)
    base.py                      Tool ABC + RateLimiter + HostRateLimiter + ToolRegistry
    _rate_limits.py              Named limiters (ARXIV, CROSSREF, SEMANTIC_SCHOLAR, …)
    web_fetch.py                 HTTP GET + plaintext extraction (SSRF-safe)
    web_search.py                DuckDuckGo + Bing HTML (keyless)
    academic_search.py           Crossref + arXiv + Semantic Scholar (+ count_results_structured)
    archive_access.py            Internet Archive + Wikimedia + Openverse
    calculator.py                AST-based safe math
    citation_manager.py          Local bibliography + BibTeX/APA
    peer_review.py               Deterministic rubric (no LLM)
    code_execution.py            Local subprocess or E2B sandbox
config.py                    CuriosityEngineConfig + interactive setup
providers.py                 ModelClient ABC + Anthropic/OpenAI-compat + tool trace capture
retry_utils.py               provider-agnostic retry with exponential backoff
journal.py                   JSON-backed journal + atomic writes + known_prior_art CRUD + source round-robin
register.py                  markdown rendering for the verified-insights artifact
models.py                    dataclasses (RegisterEntry carries verification_tool_calls + reverification_log + …)
prompts.py                   all prompts (domain-agnostic — no field-specific examples)
json_utils.py                robust JSON extraction
Dockerfile                   python:3.13-slim + scientific stack + non-root user
docker-compose.yml           volume mounts + env passthrough
curiosity                    wrapper (builds on first run)
.github/workflows/ci.yml     lint + syntax + smoke + Docker build
```

### Tool system

Investigator AND verifier see the full tool set:

| Tool | Provider | Keyless | What it does |
|---|---|---|---|
| `web_search` (Anthropic server) | Anthropic | n/a | Native server-side |
| `web_search` (client) | All | ✅ | DuckDuckGo + Bing HTML fallback |
| `web_fetch` | All | ✅ | HTTP GET + trafilatura extraction |
| `academic_search` | All | ✅ | Crossref + arXiv + Semantic Scholar |
| `archive_access` | All | ✅ | Internet Archive + Wikimedia + Openverse |
| `calculator` | All | ✅ | AST math; npv/cagr/wacc/pmt |
| `citation_manager` | All | ✅ | Local bibliography → BibTeX/APA |
| `peer_review` | All | ✅ | Deterministic rubric |
| `code_execution` (Anthropic server) | Anthropic | n/a | Native sandboxed Python |
| `code_execution` (client) | All | ✅ | Local subprocess + optional E2B |

**Rate limiters** (shared process-wide; fire before every network call):

- arXiv: 1 req/3s + 1.0s jitter (hard per user manual)
- Semantic Scholar: 1/3s + 1.0s jitter (unauthenticated quota)
- Crossref: 5 req/s + 0.2s jitter (polite-pool generous)
- `web_fetch`: per-host 3 req/s + 0.3s jitter
- Archive / Wikimedia / Openverse: 2-3 req/s + modest jitter

Jitter is uniform-random added *after* token acquisition so we don't look like a fixed-interval bot to upstream throttlers.

**Security**: Docker path fully isolates `code_execution`. Local subprocess is *not* a sandbox — use E2B for isolation without Docker.

Adding a tool: subclass `engine.tools.base.Tool` with `name` / `description` / `input_schema` / `execute(args) -> str`. Auto-registered.

### Knowledge graph

`engine/graph.py` builds a NetworkX multigraph over entries / xrefs / insights / register / predictions / sources / tags. Edge types: `shares-tag`, `cites-source`, `semantic-similarity`, `cross-referenced-by`, `supports-insight`, `registered-as`, `predicts`, `cites`, `has-tag`.

Source normalization (`_normalize_source`) canonicalizes arxiv / DOI / title surface forms so `cites-source` edges actually fire across variants.

Cross-domain bias in selection: `shares-tag` at 0.25× weight; +2.5 bonus when connected entries have zero `domain_tags` overlap.

### Semantic retrieval

`engine/embeddings.py` — OpenAI-compat embeddings (default `text-embedding-3-small`). Auto-embed on entry creation. `--embed-backfill` for legacy. `--find-similar "query" --top-k N`. Similarity edges feed the graph.

Gracefully disabled if no embedding-capable profile is configured.

---

## Provenance

Any validated register entry carries a full audit trail in JSON:

```
RegisterEntry
├── supporting_xref_id             — cross-ref that triggered synthesis
│   └── source_entries             — journal entries the xref connected
│       ├── question, hypothesis, raw_findings, sources, surprise_delta
├── premises_supported, premises_support_citations
├── synthesis_findable, synthesis_prior_art
├── central_architectural_move, central_move_prior_art
├── functional_decomposition       — phase 3a output (dimension/exemplar/differentiator)
├── closest_peer_system            — phase 3b output (name/url/overlap/differentiators)
├── skeptic_probe                  — kill queries + outcomes
├── known_prior_art_evaluations    — per-anchor is_peer/overlaps/differentiators
├── novelty_type, verdict, verified_confidence, verification_summary
├── verification_tool_calls        — full per-call trace (name/args/result_length/errors)
├── reverification_log             — append-only audit passes under updated rules
├── predictions                    — falsifiable claims with target_date + freshness_check
└── human_review_status            — approved | rejected + reason | deferred | unreviewed
```

If a reader can reconstruct *why* each insight entered the register, the system is doing its job.

---

## Honest limitations

1. **Model overlap remains.** Primary and verifier share training-data overlap even across families; the premises-vs-synthesis decomposition + phase-structured search narrow the failure mode but don't eliminate it.
2. **LLM-dependent verifier blind spots.** Known-prior-art anchors close specific blind spots the human has identified, but can't catch the ones nobody has named yet.
3. **Code execution is not GPU-scale.** `code_execution` handles analytical checks, small simulations, re-deriving numbers — not model training or large-scale analysis.
4. **Surprise grader is the primary model.** A truly adversarial surprise grader would be a different family; that's future work.
5. **Style bias toward articulate claims.** Everything is structured JSON. Pre-articulate or paragraph-needing insights are selected against.
6. **Attractor basins still happen** under narrow focus, despite A+B gates. The cross-domain analog probe is the primary counterforce.
7. **Analog probe quality is LLM-dependent.** Weaker primary → weaker analogs. Prompt rejects shallow analogs but can't guarantee strong ones.

What this **is** good for:

- Session-spanning research discipline that enforces hypothesis-first hygiene.
- Surfacing questions worth investigating, even when the "insights" turn out derivative.
- Building a substantiated, citation-linked reasoning trail (`register.md`).
- Accelerating connection-finding across your own accumulated work.
- Human-directed deep research where the engine does the search+synthesis legwork and you judge via `--review-register`.
- Producing executable research plans (via directives) that another agent or researcher can act on.

---

## Roadmap

| # | Feature | Status |
|---|---|---|
| 1 | Cross-model adversarial verification | **done** |
| 2 | Predictions with time-horizon + freshness check at creation | **done** |
| 3 | Human-in-the-loop (L1 post-hoc review + rejection feedback) | **done**; L2/L3 planned |
| 4 | Real tools (web_fetch, academic_search, archive_access, calculator, …) | **done** |
| 5 | Multi-agent disagreement (investigators with different priors) | planned |
| 6 | Code execution (Anthropic + client subprocess + E2B) | **done** |
| A | Focus + human question injection | **done** |
| B | Knowledge graph for structural cross-ref | **done** |
| C | Semantic retrieval (embeddings) | **done** |
| D | Premises-vs-synthesis verification decomposition + novelty_type classifier | **done** |
| E | Cross-domain analog probe | **done** |
| F | Priority-ordered queue + source-round-robin scheduling + priority floor | **done** |
| G | Anti-attractor gate on cross-reference | **done** |
| H | Cross-domain bias in graph heuristic | **done** |
| I | Surprise confidence calibration + post-hoc clamp | **done** |
| J | Source normalization | **done** |
| K | `inconclusive` verdict + held register pipeline + settlement plans | **done** |
| L | Re-verification of unregistered insights under current rules | **done** |
| M | Admin tab — maintenance operations + work-to-do counters | **done** |
| N | Orphaned-xref recovery | **done** |
| O | Per-phase model routing + role-based profile swap | **done** |
| P | Challenged-hedge guardrail | **done** |
| Q | Configurable per-request HTTP timeout | **done** |
| R | Within-domain assumption probe | **done** |
| S | Negative-space gap mapping + Coverage tab + scan diff | **done** |
| T | Cycle-level failure isolation | **done** |
| U | **Controlled parallelism** — parallel investigation + synth/verify fan-out with shared rate limiters | **done** |
| V | **Rate limiter + per-limiter jitter** — breaks fixed-interval bot fingerprint | **done** |
| W | **Phase-structured prior-art search** — phases 1/2/3a/3b + skeptic smell test | **done** |
| X | **Engine-side guards** — phase-1, skeptic-probe, peer-system, known-prior-art, confidence-drop | **done** |
| Y | **Known-prior-art feedback list** — human-curated peer anchors the verifier must evaluate | **done** |
| Z | **Re-verify register audit** — append-only reverification_log, never overwrites original verdict | **done** |
| AA | **Research directives export** — per-record + bundle, multi-call harness, grounding-checked agentic prompt | **done** |
| AB | **Extension registration path** — `validated + extension + differentiators` reaches register under own novelty_type | **done** |
| AC | **Verification tool-call audit trail** | **done** |
| AD | **Prediction freshness check** — detects already-fulfilled claims at creation time | **done** |
| — | Foreign-lens phase (scheduled cross-domain creativity burst) | backlog |
| — | Insight de-duplication via embeddings | backlog |
| — | GPU-backed experimental executor | backlog |
| — | Mid-cycle interruption / steering (Phase 3 L2-L3) | backlog |
| — | Multi-journal federation / cross-journal retrieval | backlog |
| — | Adversarial surprise grader (different-family model) | backlog |
| — | Streaming directive generation (per-token log feedback) | backlog |

---

## Development

```bash
.venv/bin/python curiosity_engine.py --help
.venv/bin/ruff check --exclude .venv --exclude data .
pre-commit install                                   # ruff + trufflehog secret scanning
trufflehog git file://. --only-verified --fail
./curiosity --rebuild --show-journal                 # rebuild Docker image
```

### Continuous integration

`.github/workflows/ci.yml` on every push + PR to `main`:

- **`lint-and-smoke`**: ruff, syntax, tool-discovery smoke, core-imports smoke, calculator functional — no API keys.
- **`docker-build`**: verifies the Dockerfile stays green.

---

## Files generated at runtime

- `~/.CuriosityEngine/engine.toml` — model + engine config.
- `./data/research_journal.json` (or `--journal`) — full journal state.
- `./data/register.md` — human-readable register artifact, auto-written on register changes.
- `./data/{journal_stem}_directives/` — generated research directives (per-record and bundle) + `.verification.json` sidecars.
- `./data/_runs/{journal_stem}/*.log|.meta.json` — run logs + metadata (streamed to the Runs tab).
- `*_bibliography.json` / `refs.json` — from the `citation_manager` tool.

All runtime artifacts are `.gitignore`d.

---

## License

**Copyright (c) 2026 sfw. All rights reserved.**

This repository is shared for reference and evaluation only. No part of the code, documentation, or prompts may be copied, redistributed, modified, or used in derivative or commercial works without the copyright holder's express written permission.

See [LICENSE.md](./LICENSE.md) for full terms.
