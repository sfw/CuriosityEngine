# Curiosity Engine

A proof-of-concept research loop that generates its own questions from self-assessed uncertainty, investigates them with web search + academic APIs + sandboxed code execution, accumulates state across sessions, and routes synthesized insights through an independent cross-family verifier before committing them to a durable register — each with falsifiable predictions attached for later review.

The engine is designed to be **driven iteratively**: you set a focus, run cycles, inspect the knowledge graph it builds, inject your own questions or reject shallow claims, and continue. It won't produce novel ideas on its own; with steering it will surface candidates that survive real adversarial scrutiny.

> **Status**: proof of concept. Phases 1–4 and 6 of the roadmap are implemented; Phase 5 (multi-agent disagreement) is planned. Knowledge graph + semantic retrieval + human review loop + containerized runtime + cross-domain analog probe + premises/synthesis verification split + inconclusive/held pipeline + per-phase model routing + challenged-hedge guardrail + admin maintenance operations are all in. See [Honest limitations](#honest-limitations) for what this system genuinely *can't* do.

---

## Core thesis

> Novel insight rarely emerges from a single query. It emerges at the **intersection** of knowledge gaps, when an investigation forces a prior hypothesis to collide with fresh evidence, when a reframing through a foreign domain reveals an analogous mechanism, and when an independent reviewer can verify that *the premises are real* but *the synthesis itself is not yet in the literature*.

The engine operationalizes this by:

1. **Committing to hypotheses** before searching, so surprise is a comparison rather than a self-report.
2. **Cross-referencing** accumulated findings across many sessions to surface connections no single prompt would produce — prioritized by a knowledge graph that biases toward *cross-domain* neighbors and away from already-explored attractor basins.
3. **Probing cross-domain analogs** on high-surprise findings — the engine asks which distant fields (immunology, population ecology, statistical mechanics, …) have structurally analogous mechanisms, and enqueues investigable translations of those analogs. Biology → algorithmics-style novelty, made structural.
4. **Adversarially verifying** synthesized insights with a *different model family* using a **premises-vs-synthesis decomposition**: the verifier must separately establish that the ingredients are real (premises_supported=TRUE) and that the composite claim is NOT in the literature (synthesis_findable=FALSE). The signature of genuine novelty is `supported premises + unfindable synthesis` — exactly the pattern the prior verifier schema was rejecting as "challenged."
5. **Attaching falsifiable predictions** to every validated insight, so the test of time separates predictive claims from post-hoc narrative fitting.
6. **Keeping the human in the loop.** Focus, question injection, register review — the engine is a research *partner*, not an oracle.

---

## Pipeline

```
  introspect        →  uncertainties (what the model is uncertain about)
  generate          →  ranked investigable questions
                            ↑
          (priority-ordered dequeue: human > emergent > generated)
  investigate                   (three stages — primary model)
    ├─ hypothesize  →  commit to a pre-investigation answer
    ├─ search       →  web_search + web_fetch + academic_search + code_execution
    └─ assess       →  compare findings to hypothesis, compute surprise
                       (confidence_after clamped by verdict × surprise-delta rules)
  (analog probe)    →  on surprise ≥ threshold, ask for distant-domain analogs
                       (named sub-fields + mechanisms + translated questions)
                       enqueued at priority 0.85
  (embed)           →  OpenAI embeddings on question + takeaways
  cross-reference   →  graph-aware entry selection biased toward CROSS-DOMAIN
                       neighbors; anti-attractor gate skips candidates whose
                       participant overlap ≥50% with an existing xref
  synthesize        →  promote high-novelty connections to insights
  verify (cross-family)
    └─ adversarial review on a different model family, with tools
    └─ separately assesses: premises_supported (good) × synthesis_findable (bad)
    └─ assigns novelty_type: new_synthesis | extension | correction | restatement | unsupported
        └─ verdict=validated AND premises_supported AND NOT synthesis_findable
           AND conf >= floor  →  REGISTER
            └─ emit 1-3 falsifiable predictions with time horizons
  human review      →  --review-register: approve / reject-with-reason / defer
                       rejection reasons feed into future verifier prompts
  check predictions (later, on demand)
    └─ revisit due predictions via verifier+tools
    └─ update register entry lifecycle status
```

Every step persists to `research_journal.json`. The human-readable artifact is `register.md`, auto-written whenever the register changes.

---

## Technical Briefing — how novelty is conceived and verified

This section walks through the full novelty pipeline end-to-end. It's the section to read if you want to understand *exactly* how the engine tries to produce genuinely novel ideas — not just paraphrase training data — and how it tries to filter derivative claims before they reach the register.

### The problem the engine is built around

A language model asked "what's novel here?" will confidently narrate something. The narration often reads as insight but is usually **articulate restatement**: a fluent reformulation of content already in the training data, dressed in new vocabulary. Three failure modes dominate:

1. **Self-grading collapse.** The same model that generates an insight is also asked to grade its novelty. It has no independent signal — its confidence is an artefact of its own generation momentum.
2. **RAG novelty paradox.** Surface-text retrieval systems penalize genuinely novel claims (the matching passages don't exist yet) and reward restatement (good retrieval matches imply the claim is already published).
3. **Ingredients-as-weakness mis-framing.** A truly novel synthesis is, by definition, a composition of established ingredients into a combination that *isn't yet* in the literature. Naive verifiers reject this pattern as "the components are already known" — which is the *signature* of novelty, not a weakness.

The engine's design is a direct response to each of these failure modes.

### Phase-by-phase: how novelty is *conceived*

**1. Hypothesis-first investigation.** Before any search runs, the primary model commits to a specific, falsifiable answer to the research question (`HYPOTHESIS_PROMPT`). This is stored. The investigation then proceeds — web, academic search, code execution, web fetch — and produces findings. A separate prompt (`SURPRISE_PROMPT`) then compares the committed hypothesis against the findings and produces a `surprise_delta` in [0, 1]. Because the hypothesis is fixed before evidence arrives, surprise is *a structural comparison*, not a self-report. A high surprise_delta is a reliable signal that the investigation's findings diverged from the model's prior — which is where new ideas live.

**Confidence calibration.** Because LLMs routinely inflate `confidence_after` on partially-confirmed results, the engine enforces explicit rules: on `partially_confirmed + surprise_delta < 0.2`, confidence cannot increase; on `contradicted`, confidence must drop by ≥0.2; on `unresolved`, confidence is pulled toward 0.5. Rules are stated in the prompt and re-applied as a post-hoc clamp in `_calibrate_confidence_after()` for defense in depth.

**2. Cross-domain analog probe.** When an entry's `surprise_delta ≥ analog_probe_surprise_threshold` (default 0.5), the engine fires a dedicated prompt (`ANALOG_PROBE_PROMPT`) asking: *which distant fields have mechanisms, laws, or principles that are structurally analogous to this finding?* The LLM is explicitly instructed to name the sub-field and the specific named result (not "biology" but "population ecology's competitive exclusion principle"; not "physics" but "statistical mechanics' maximum entropy principle"), to reject weak or topically-related analogs, and to translate each analog into an investigable question. The output is enqueued at priority 0.85 with source `analog:<entry-id>`. This is the mechanism by which the engine reaches from one domain into another — the structural move that produced neural nets, genetic algorithms, ant colony optimization, and most large cross-field jumps in scientific history.

Critically, the engine *does not* decide which analog domain to use ahead of time. The LLM picks it based on the specific finding. Research has shown LLMs are particularly good at this kind of conceptual analogical reasoning; the engine leverages it rather than hard-coding a rotation.

**3. Priority-ordered question queue.** All emergent questions — entry follow-ups, cross-reference follow-ups, analog probes — carry a `priority` score at enqueue time:
- Human-directed questions: 1.0 (always highest).
- Analog probe questions: 0.85 (cross-domain material is the rarest signal).
- Cross-reference follow-ups: `0.4 + 0.6 × novelty_score` (high-novelty xrefs produce high-priority follow-ups).
- Entry follow-ups: `max(0.3, 0.4 + 0.6 × surprise_delta)` (surprising entries produce high-priority follow-ups).

Each cycle's investigation budget is filled human-first, then from the non-human queue in priority-descending order, and only then from fresh introspection-generated questions. This guarantees that the most-rewarding queued questions drive actual investigations.

**4. Cross-reference with structural bias.** Every `cross_ref_frequency` cycles (default 3), the engine selects a subset of entries and asks for non-obvious connections (`CROSS_REFERENCE_PROMPT`). Entry selection is graph-aware (`engine/graph.py:select_entries_for_xref`):
- `cites-source` and `semantic-similarity` edges contribute full weight.
- `shares-tag` edges contribute at only 0.25× weight — same-domain connections are less interesting than cross-domain ones.
- A **cross-domain bonus** of +2.5 fires when two connected entries have zero overlap in `domain_tags`. This is the selector saying "these two entries share a source or are semantically close but come from different domains — that's a novelty frontier."

The result: the cross-reference prompt is fed participant pairs that span domains, matching what its prompt has always asked for but the selector wasn't previously doing.

**5. Anti-attractor gate.** LLMs re-converge on the same entry clusters when asked to find non-obvious connections — the "attractor basin" failure mode. The engine enforces two gates:
- *Prompt-level.* `CROSS_REFERENCE_PROMPT` explicitly says: if a candidate's `source_entry_ids` overlap ≥50% with any existing xref, skip or reduce novelty by ≥0.3 and justify.
- *Code-level.* In `engine/cross_reference.py`, after the LLM returns candidates, the engine computes an overlap coefficient against each existing xref's participant set. Candidates with overlap ≥0.5 and claimed novelty < 0.85 are skipped (printed as `Skipped N attractor-basin cross-reference(s)`). Novel angles on the same participants are allowed, but they have to prove novelty.

**6. Synthesis with self-check.** High-novelty xrefs (`novelty_score ≥ novelty_threshold`, default 0.7) get synthesized into full Insights via `SYNTHESIZE_PROMPT`. The synthesis prompt includes a `prior_art_check` self-interrogation ("if I searched for this claim, would I find it plainly stated?") — this is a weak but cheap filter before the much more expensive cross-model verification.

### Phase-by-phase: how novelty is *verified*

**7. Cross-family adversarial review.** Each candidate Insight is routed to the **verifier model** — a different model family than the one that produced it (Anthropic Claude as primary + OpenAI GPT or Google Gemini as verifier is the canonical setup). The verifier is prompted as an adversarial reviewer, given web_search and code_execution tools, and asked to attempt refutation.

**8. The premises-vs-synthesis decomposition.** This is the core innovation in the verification step. A naive prior-art search conflates two very different questions:
- **A. Are the ingredients real?** Does each premise the insight depends on have literature support?
- **B. Is the composite claim findable?** Is the full synthesis — not just its parts — already published under some name?

A *new synthesis* has premises_supported=TRUE (ingredients real) and synthesis_findable=FALSE (composition not yet in literature). That is the canonical signature of genuine novelty. Earlier, naive verifier schemas conflated A and B into a single `prior_art_found` field, causing the verifier to reject genuine-novelty candidates with reasoning like *"the individual components are documented, so this isn't novel"* — which is inverted logic.

The current verifier (`VERIFY_PROMPT`) separately scores both axes:

```json
{
  "premises_supported": true,
  "premises_support_citations": [...],
  "synthesis_findable": false,
  "synthesis_prior_art": [...],
  "novelty_type": "new_synthesis | restatement | extension | correction | unsupported",
  ...
}
```

and the prompt explicitly warns the verifier against the "ingredients existed separately" mis-framing:

> **Do NOT penalize an insight for being composed of published ingredients. Penalize only if the *full synthesis itself* is published, or if the reasoning from premises to synthesis is broken.**

**9. The register gate.** A candidate reaches the *active* register only if ALL of:
- `verdict = validated`
- `premises_supported = true`
- `synthesis_findable = false`
- `verified_confidence ≥ register_confidence_floor` (default 0.6)

The `novelty_type` classifier distinguishes between the five outcomes a verifier can reach:

| novelty_type | Meaning | Outcome |
|---|---|---|
| `new_synthesis` | Premises real, full synthesis not in literature | Registerable if confidence clears floor |
| `extension` | Modest extension of a published claim | Marginal — usually registerable at high confidence |
| `correction` | Challenges a published claim with new reasoning/evidence | Registerable as a novel critique |
| `restatement` | Full synthesis is already in the literature under some name | Rejected — not novel |
| `unsupported` | Premises themselves are shaky | Rejected regardless of synthesis |

**9b. When the verifier cannot reach — the `inconclusive` / held path.**

Genuine novelty often lives exactly where verification is hardest: behind paywalls, in proprietary datasets, in pre-publication work, in experiments that require resources the verifier's tools can't access. Treating "I couldn't verify this" as equivalent to "I found a decisive flaw" selects against the very claims that are most likely to be novel.

The verifier therefore has a fourth verdict — `inconclusive` — with explicit trigger conditions. It is used ONLY when:

- Searches returned no meaningful results AND the claim cannot be rephrased into something searchable;
- The claim depends on data or methods the verifier cannot access (proprietary, pre-publication, clinical, GPU-scale, paywalled);
- The claim is empirical and resolves only via an experiment the verifier cannot run;
- The claim sits in a field with genuinely thin public literature.

When verdict is `inconclusive` AND `premises_supported = true` AND confidence ≥ `held_confidence_floor` (default 0.7, usually tighter than the active floor), the insight becomes a **held register entry** — preserved with full provenance, distinguishable from both rejected and active entries, and carrying a **settlement plan**:

- `settlement_method`: the concrete method by which reality could eventually resolve it (paper to watch, benchmark release, dataset/code release, industrial observation).
- `settlement_horizon`: ISO date by when a signal might appear.
- `settlement_triggers`: specific observable outcomes, each of which would either promote the held entry to active or refute it.

Held entries can be promoted to active three ways:

1. **Automatic via prediction.** Settlement triggers can be converted to Prediction records (during human review, or automatically by selecting the "attach triggers as predictions" checkbox on promotion). When `--check-predictions` runs later and all attached predictions resolve `confirmed`, the held entry is promoted to active automatically.
2. **Human review.** `--review-register` presents held entries with a `[p]romote-to-active` action. The user can promote based on domain knowledge even before any prediction resolves.
3. **Manual in web UI.** The Register tab's held cards have a *Promote to active* button.

**Guardrail against misuse of `inconclusive`.** LLMs may instinctively use `inconclusive` as a soft `challenged` to avoid committing. Two defenses:

1. The prompt explicitly rules this out and requires that `verification_summary` name the specific epistemic gap when verdict is `inconclusive`.
2. `engine/verification.py` pattern-matches the summary for gap-naming phrases (`cannot access`, `paywalled`, `pre-publication`, `requires an experiment`, etc.). If none match, the engine downgrades `inconclusive` → `challenged` and logs the downgrade.

**9c. The challenged-hedge guardrail.** LLMs (especially RLHF-trained ones) have a strong hedge reflex — they instinctively reach for `challenged` rather than `validated` even when their own decomposition unambiguously signals new synthesis (`novelty_type=new_synthesis`, `premises_supported=TRUE`, `synthesis_findable=FALSE`, confidence ≥ floor). The prompt explicitly tells the verifier not to hedge in that configuration, but the hedge reflex survives prompt instructions.

The engine adds a code-level guardrail: when the decomposition is unambiguous but verdict is `challenged`, it inspects `reasoning_flaws` for markers of a **substantive** synthesis-level critique (`leap`, `does not show`, `overextend`, `not established`, `too strong`, `conflat`, `viable alternative`, `assumes transfer`, `transfers to`, `treated as if`, `cross-model interpretation`, `contradicts`, and ~40 more). If none of the flaws match a substantive marker, the engine **upgrades the verdict to `validated`** and logs `[guardrail] verdict=challenged but decomposition is unambiguous ... — upgrading to validated.` If any flaw matches a substantive marker, the challenged verdict is respected.

The logic inverts the default: assume the verifier is hedging unless it can name a specific substantive critique. This is defensible because (a) the prompt explicitly instructs the verifier to name specific flaws in that field, (b) empty or ingredient-only reasoning_flaws signal a mere hedge, and (c) the decomposition fields (`premises_supported`, `synthesis_findable`, `novelty_type`) are answering narrower questions that the LLM fills out more reliably than the overall verdict. The guardrail's visibility logs let a human inspect when it fires and tune the marker list.

**Re-reviewing previously-unregistered insights.** Because verifier prompts and rules evolve, you can replay verification over every insight that doesn't yet have a register entry:

- CLI: `curiosity_engine.py --reverify-insights` (all unregistered) or `--reverify-insight <id>` (targeted).
- Web: the **Admin** tab has a Re-verify unregistered insights action that spawns the same subprocess; the run appears in the Runs tab with streaming logs.

Each candidate's verification flows through the current gate — so an insight that was rejected under an older verifier schema may now land as `active` (if it's a `new_synthesis`) or `held` (if the verifier genuinely can't reach it). Insights that already have a register entry (any status) are skipped.

The Admin tab also surfaces three other maintenance operations — re-run cross-reference (generates new xrefs from the current journal state), synthesize orphaned xrefs (recovers from runs that died between cross-ref and synthesis), and check due predictions (resolves pending predictions whose target_date has arrived). Every action is idempotent: it skips items that are already processed, so re-running is always safe.

**10. Prior-human-rejection feedback.** Every human rejection (`--review-register` with a required reason) gets fed into future verifier prompts as *"patterns to avoid repeating — reasons a domain expert rejected previous register entries."* The verifier uses this to apply the same skepticism to candidates with similar weaknesses. A user's taste becomes a learned bar over time.

**11. Falsifiable predictions close the loop.** For every validated register entry, the verifier is required to emit 1–3 falsifiable predictions: each with a specific `claim`, a `falsifiable_condition` that can be checked against reality, a `check_method` (concrete search / benchmark / rerun), and a `target_date` (typically 3–24 months out). `--check-predictions` revisits due predictions using the verifier + tools and marks them `confirmed` / `refuted` / `inconclusive` / `expired`. A register entry's status (`active` / `validated_by_prediction` / `challenged_by_prediction`) is reconciled from its predictions' outcomes. This is the ultimate filter: reality rather than any model's judgment.

### Why this should produce something different from naive LLM "research"

The composed pipeline attacks each of the three failure modes structurally, not just by prompt engineering:

| Failure mode | Structural response |
|---|---|
| Self-grading collapse | Surprise is a comparison of hypothesis-vs-findings, not a self-report. Verification uses a different model family. Predictions close the loop with reality. |
| RAG novelty paradox | Verifier explicitly decomposes premises (want support) from synthesis (want non-support). The absence of literature matches on the synthesis is counted as *evidence for novelty*, not against it. |
| Ingredients-as-weakness | `novelty_type` classifier names the valid novelty categories; prompt explicitly rules out "components existed separately" as a rejection reason. |

**What still fails.** None of the above eliminates the fact that both models share overlap in training data; cross-family reduces but doesn't remove this. Code execution doesn't scale to real experiments. The human is still the final arbiter of what's actually interesting via `--review-register`. The goal is not "automated discovery" — it's *a research partner that enforces the epistemic discipline a disciplined researcher would apply if they had infinite patience.*

### Minimal trace of a register entry's provenance

Any validated register entry carries, in its JSON and in `register.md`, a full provenance chain:

```
RegisterEntry
├── supporting_xref_id          — the cross-reference that triggered synthesis
│   └── source_entries          — the journal entries the xref connected
│       ├── question            — the original research question
│       ├── hypothesis          — pre-investigation answer
│       ├── raw_findings        — investigation output
│       ├── sources             — web URLs, DOIs, arXiv ids (normalized)
│       └── surprise_delta      — how much the finding diverged from the hypothesis
├── premises_supported, premises_support_citations   — the verifier's evidence for A
├── synthesis_findable, synthesis_prior_art          — the verifier's evidence for B
├── novelty_type               — the categorical verdict
├── verification_summary       — one-paragraph justification
├── predictions                — 1-3 falsifiable claims with target_date
└── human_review_status        — approved | rejected + reason | deferred | unreviewed
```

If any later reader (you, the human) can reconstruct *why* each insight entered the register, the system is doing its job. If not, treat any register entry as provisional — the substantiation is the artifact, not the one-line title.

---

## Prerequisites

- **Docker** (recommended) — a container fully isolates `code_execution` from your host.
- Or **Python 3.11+** for the local venv path (stdlib `tomllib`).
- API key for at least one provider:
  - **Anthropic** (Claude) — supports server-side `web_search` and `code_execution` natively.
  - **OpenAI-compat** — any endpoint that speaks the OpenAI chat-completions protocol: OpenAI, Gemini (via its OpenAI-compat endpoint), OpenRouter, Ollama (`/v1` mode), xAI, Groq, Together, DeepSeek, Moonshot (Kimi), LM Studio, and anything else that matches the contract.

Recommended setup: **Anthropic Claude as primary, a different-family model (OpenAI GPT, Google Gemini, etc.) as verifier.** Same-family verification defeats most of the adversarial point. Primary + verifier can use different providers, endpoints, and keys.

---

## Setup

Two paths: **Docker** (recommended — isolates `code_execution` from the host) and **local venv**.

### Docker (recommended)

```bash
git clone <this-repo>
cd CuriosityEngine

# First run builds the image (several minutes — numpy/scipy/sklearn wheels).
./curiosity --show-journal

# Normal runs
./curiosity --cycles 3
./curiosity --review-register
./curiosity --list-tools

# Start the browser UI on http://localhost:8000
./curiosity web              # ./curiosity web stop | logs | restart

# Force rebuild after updating dependencies
./curiosity --rebuild --show-journal
```

The wrapper bind-mounts `~/.CuriosityEngine` (engine.toml) and `./data` (journal, register.md, bibliographies) so state survives container restarts. API keys can live in `engine.toml` or in your shell as `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `E2B_API_KEY`.

If you already have a journal from a pre-Docker local run, migrate it once:

```bash
mkdir -p ./data
cp research_journal.json ./data/ 2>/dev/null || true
cp register.md ./data/ 2>/dev/null || true
```

First-run setup wizard, `--review-register`, and all other interactive prompts work because the wrapper opens a TTY.

### Local venv (lighter, fewer isolation guarantees)

```bash
git clone <this-repo>
cd CuriosityEngine

# Homebrew Python on macOS requires a venv (PEP 668)
python3 -m venv .venv
source .venv/bin/activate

# Core deps
pip install -r requirements.txt

# Optional: scientific Python, so code_execution can import numpy/scipy/etc.
pip install numpy scipy pandas scikit-learn matplotlib

# First run — triggers interactive setup wizard, writes ~/.CuriosityEngine/engine.toml
python curiosity_engine.py --show-journal
```

The wizard asks:
1. **Primary model** — provider (anthropic / openai_compat), model id, API key, optional base_url.
2. **Verifier model** — same prompts, recommended to be a different family.
3. **Advanced settings** (optional) — token limits, retry policy.

If you already have an `engine.toml`, `CuriosityEngineConfig.load()` auto-migrates the older single-profile schema to the current multi-profile form.

See `engine.toml.example` for a complete, annotated config.

---

## Browser UI

```bash
./curiosity web          # starts the FastAPI service on http://127.0.0.1:8000
./curiosity web logs     # tail the uvicorn logs
./curiosity web stop
```

Pages:

- **Journals** (`/journals`) — cards per journal with counts, focus, and recent domains.
- **Journal view** (`/journals/{name}`) — tabbed interface:
  - *Overview*: counts + high-surprise list + recent entries.
  - *Entries*: filterable by domain tag; click to expand full findings / sources / takeaways.
  - *Insights*: synthesized insights (pre-verification).
  - *Register*: card per validated insight with `novelty_type` badge, premises/synthesis decomposition chips, inline approve / reject-with-reason / defer form.
  - *Predictions*: status tracker with review history.
  - *Focus & Queue*: set focus, add/clear user-directed questions, semantic search. Queue items display their `priority` chip and are sorted priority-descending — highest-priority questions are investigated first on the next run.
  - *Graph*: interactive force-directed graph (sigma.js) with node-kind coloring and click-to-detail. Clicking any node fetches a full-text HTML fragment with no text cropping — covers all 7 node kinds (entry / xref / insight / register / prediction / source / tag).
  - *Runs*: per-journal run history with status chips (running / complete / failed); click to expand the full log; live SSE reconnect for any still-streaming run.
  - *Admin*: consolidated maintenance operations with work-to-do counters (unregistered insights / orphaned xrefs / due predictions / held entries). Buttons for: re-run cross-reference (with optional `cross_ref_window` + model-role overrides), synthesize orphaned xrefs (recover from mid-run crashes between cross-ref and synthesis), re-verify unregistered insights under current rules, and check due predictions. Each action spawns a subprocess through the standard run-tracking infrastructure — the top-bar indicator lights up, the log streams in the Runs tab.
- **Run** (`/run`) — form to start a cycle, live SSE stream of output. Domain is pre-filled from the journal's `last_domain` after the first run. Cycles input is stepped in multiples of `cross_ref_frequency` so a run never wastes an investigation on a cycle that doesn't reach cross-ref. Collapsed "Run overrides" panel exposes every engine knob for per-invocation override (dirty-detected server-side so unchanged values don't bloat the command line).
- **Settings** (`/settings`) — edit `engine.toml` in-place: model profiles (provider / name / api_key / base_url / max_tokens / temperature / **timeout_seconds**), retry policy, and every engine knob including `cross_ref_role`, analog probe enable + threshold, held pipeline enable + confidence floor. Save is non-destructive — any additional `[models.<name>]` profiles you've added manually are preserved through round-trips. Config is re-read on every request + every subprocess spawn, so saves apply immediately with no restart.
- **Top bar** — activity indicator: dim emerald dot when idle, amber pulsing dot with a soft glow when one or more runs are streaming. Clicking the amber dot drops you onto the active journal's Runs tab.

The web service shares the same `./data/` journal mount and `~/.CuriosityEngine/engine.toml` as the CLI, so both can be used in parallel. Bound to `127.0.0.1` only — no auth, single-user.

## Usage

### The iterative workflow: focus → run → review → redirect → continue

The engine is designed to be driven as a loop where you narrow scope between runs. Example session on "novel idea generation by LLMs":

```bash
# One-time: set the durable focus on this journal
./curiosity --set-focus "novel idea generation by LLMs — the role of latent-space exploration vs template recombination" \
            --journal data/ideation.json

# Run 3 cycles with that focus active in every prompt
./curiosity --cycles 3 --domain "novel idea generation by LLMs" --journal data/ideation.json

# Review what came out
./curiosity --show-journal       --journal data/ideation.json
./curiosity --show-insights      --journal data/ideation.json
./curiosity --show-register      --journal data/ideation.json
./curiosity --graph-summary      --journal data/ideation.json   # structural view
./curiosity --review-register    --journal data/ideation.json   # approve / reject with reasons

# Direct the next cycles by injecting specific questions
./curiosity --add-question "Does high-temperature decoding produce genuinely novel outputs or statistical recombinations?" \
            --add-question "Is creativity measurable as distance in activation space?" \
            --journal data/ideation.json

# Semantic lookup across accumulated research
./curiosity --find-similar "ways to measure novelty of an output" --top-k 5 --journal data/ideation.json

# Resume — human-queued questions consume the investigations budget first
./curiosity --cycles 3 --journal data/ideation.json

# Later, when enough time has passed, revisit the predictions
./curiosity --check-predictions --journal data/ideation.json
```

**How direction propagates**:
- `--set-focus` persists on the journal; every introspect / generate / investigate / cross-ref / synthesize prompt gets a `USER FOCUS` section with that text as a hard constraint.
- `--add-question` pushes onto `question_queue` with `source="human"`; those questions get investigated first each cycle, ahead of model-generated ones.
- `--review-register` rejection reasons get injected into future verifier prompts as "prior human rejections" — the verifier learns what you consider too thin.
- Cross-reference uses the knowledge graph to specifically surface entry pairs that share tags / sources / embeddings but don't yet have a cross-reference — the structural definition of "knowledge-gap intersection."
- `--find-similar` retrieves by semantic meaning, not keyword, across everything the journal has accumulated.

### Command reference

```bash
# Running cycles
python curiosity_engine.py --cycles 1                              # one cycle
python curiosity_engine.py --cycles 3 --domain "TOPIC"             # override topic
python curiosity_engine.py --cycles 3 --journal "./mlp.json"       # topic-specific journal
python curiosity_engine.py --cross-ref-only                        # skip investigation, just cross-ref + synth + verify

# Inspecting state (no API calls)
python curiosity_engine.py --show-journal       # counts + domains + high-surprise + current focus
python curiosity_engine.py --show-insights      # synthesized insights (pre-verification)
python curiosity_engine.py --show-register      # verified-only insights with substantiation
python curiosity_engine.py --show-predictions   # all stored predictions with status
python curiosity_engine.py --list-tools         # registered research / calculation tools

# Steering
python curiosity_engine.py --set-focus "TEXT"   # persistent investigation focus on this journal
python curiosity_engine.py --show-focus
python curiosity_engine.py --clear-focus
python curiosity_engine.py --add-question "?"   # push a user-directed question onto the queue (repeatable)
python curiosity_engine.py --list-questions
python curiosity_engine.py --clear-questions

# Human review (Phase 3 L1)
python curiosity_engine.py --review-register    # interactively approve / reject / defer each unreviewed register entry

# Knowledge graph
python curiosity_engine.py --graph-summary
python curiosity_engine.py --graph-export graph.json   # .graphml / .gexf / .json

# Semantic retrieval
python curiosity_engine.py --embed-backfill            # embed legacy entries (one-time)
python curiosity_engine.py --find-similar "query" --top-k 10

# Predictions
python curiosity_engine.py --check-predictions         # check due predictions (uses verifier + tools)
python curiosity_engine.py --check-predictions-all     # force-review every pending prediction

# Maintenance — safe to re-run (all operations skip already-handled items)
python curiosity_engine.py --reverify-insights         # re-verify every unregistered insight under current verifier rules
python curiosity_engine.py --reverify-insight i-abc    # re-verify a single insight by id
python curiosity_engine.py --synth-orphaned-xrefs      # synth + verify xrefs that lack a matching insight (recovery)

# Model overrides for a single run
# Name-only override (keeps profile's provider/endpoint/key — use when the endpoint supports multiple models):
python curiosity_engine.py --cycles 3 --primary-model gpt-5.1 --verifier-model gpt-5.4

# Role-based swap (copies the WHOLE profile — provider + base_url + key + name — into the slot):
python curiosity_engine.py --cycles 3 --primary-role verifier       # use verifier profile as primary for this run
python curiosity_engine.py --cross-ref-only --cross-ref-role verifier  # offload cross-ref to verifier profile

# Per-run engine-knob overrides (all optional; default=None means "inherit from engine.toml")
python curiosity_engine.py --cycles 6 \
    --cross-ref-window 10 \
    --investigations-per-cycle 2 \
    --novelty-threshold 0.65 \
    --register-confidence-floor 0.6 \
    --verify-insights \
    --analog-probe-enabled \
    --analog-probe-threshold 0.4 \
    --held-entries-enabled \
    --held-confidence-floor 0.7
```

Cross-reference runs every `cross_ref_frequency` cycles (default 3). On those cycles, high-novelty xrefs (`>= novelty_threshold`, default 0.7) get synthesized into insights, which then face the adversarial verifier. Only those with verdict `validated` AND `premises_supported` AND NOT `synthesis_findable` AND `verified_confidence >= register_confidence_floor` (default 0.6) enter the register as `active`. Verdict `inconclusive` with `premises_supported` and `verified_confidence >= held_confidence_floor` enters as `held`.

---

## Configuration

Runtime settings live at `~/.CuriosityEngine/engine.toml`. Structure:

```toml
[models.primary]
provider = "anthropic"                # or "openai_compat"
name = "claude-sonnet-4-6"
# api_key = "..."                     # or rely on ANTHROPIC_API_KEY / OPENAI_API_KEY env var
# base_url = "..."                    # only for non-default endpoints
max_tokens = 4096
investigation_max_tokens = 8192
temperature = 1.0                     # 1.0 for reasoning models (Kimi K2.x, GPT-5 thinking, o-series)
timeout_seconds = 300.0               # per-request HTTP timeout; raise for reasoning models on big prompts

[models.verifier]                     # optional; defaults to primary if omitted
provider = "openai_compat"
name = "gpt-5.1"
base_url = "https://api.openai.com/v1"
# api_key = "..."
timeout_seconds = 300.0

# Optional: any number of additional [models.<name>] sections define extra profiles
# that can be referenced by role-based CLI flags and the [engine].cross_ref_role setting.
# [models.cross_ref]
# provider = "openai_compat"
# name = "gpt-5.1"
# ... (same schema as primary/verifier)

[retry]
max_attempts = 5
base_delay_seconds = 0.5
max_delay_seconds = 8.0
jitter_seconds = 0.25

[engine]
# How the loop runs. Bump cross_ref_window on big-context models so cross-reference
# can surface intersections across a wider slice of the journal.
cross_ref_window = 20
questions_per_cycle = 3
investigations_per_cycle = 1
cross_ref_frequency = 3
novelty_threshold = 0.7
register_confidence_floor = 0.6
verify_insights = true
# Per-phase model routing: which configured profile handles the cross-reference pass.
# Defaults to primary. Set to "verifier" (or any custom role name matching a
# [models.<name>] section) to offload cross-ref to a faster non-reasoning model
# while keeping reasoning on investigation/synthesis. Recommended when primary is
# a reasoning model (Kimi K2.x, o-series) — cross-ref is one-shot pattern matching
# over a large context that doesn't benefit from extended thinking.
cross_ref_role = ""
# Cross-domain analog probe — on high-surprise entries, ask the primary model which
# DISTANT fields have structurally analogous mechanisms; enqueue the translated
# questions at high priority. Set enabled=false or raise the threshold to disable.
analog_probe_enabled = true
analog_probe_surprise_threshold = 0.5
# Held-state pipeline — when the verifier returns `inconclusive` (couldn't reach
# the claim, not refuted it), insights become held register entries pending
# settlement rather than being silently rejected.
held_entries_enabled = true
held_confidence_floor = 0.7
```

**OpenAI-compat endpoint shortcuts** (any value works in `base_url`):

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

### Operational settings

Defaults live on `EngineSettings` (in `config.py`, persisted under `[engine]` in `engine.toml`) and `EngineConfig` (in `models.py`, the per-run instance):

- `questions_per_cycle = 3` — how many ranked questions generated per cycle.
- `investigations_per_cycle = 1` — how many of those actually investigated.
- `cross_ref_frequency = 3` — cross-reference runs every N cycles.
- `cross_ref_window = 20` — max entries sent to cross-ref prompt (graph-aware selection ranks them).
- `novelty_threshold = 0.7` — cross-ref novelty score required to trigger synthesize.
- `register_confidence_floor = 0.6` — verifier confidence required to register (the gate also requires `premises_supported=true` and `synthesis_findable=false`).
- `verify_insights = True` — toggle cross-model verification.
- `analog_probe_enabled = True` — run the cross-domain analog probe on high-surprise entries.
- `analog_probe_surprise_threshold = 0.5` — minimum surprise_delta required to trigger the analog probe (higher = more selective).
- `cross_ref_role = ""` — per-phase model routing for cross-reference. Empty = use `primary`. Set to `"verifier"` or any custom role matching a `[models.<name>]` section to offload cross-ref to a faster non-reasoning model.
- `held_entries_enabled = True` — toggle the inconclusive → held pipeline.
- `held_confidence_floor = 0.7` — minimum verifier confidence for a held register entry (usually tighter than active's floor).

Per-profile setting:

- `timeout_seconds = 300.0` — per-request HTTP timeout in seconds. Raise for reasoning-mode models (Kimi K2.x, o-series, GPT-5 thinking, Claude extended thinking) that spend 60–180s "thinking" before streaming the first token on large prompts. Set per profile: `[models.primary].timeout_seconds`, `[models.verifier].timeout_seconds`, etc.

---

## Architecture

```
curiosity_engine.py          CLI entry
engine/                      orchestrator package (composed via mixins)
  core.py                      class def, init, model plumbing, main run loop
  introspect.py                Phase 1-2: uncertainties → questions
  investigation.py             Phase 3: hypothesis → tool-loop → surprise
  cross_reference.py           Phase 4-5: xref → synthesize (graph-aware selection)
  verification.py              Phase 6-7: adversarial verify + prediction lifecycle + human review
  display.py                   show_* (no API calls)
  graph.py                     NetworkX knowledge graph builder + cross-ref selector
  embeddings.py                OpenAI embeddings + cosine similarity + find_similar
  tools/                       pluggable research / calculation tools (auto-discovered)
    base.py                      Tool ABC + ToolRegistry + discover_tools()
    web_fetch.py                 HTTP GET + plaintext extraction
    web_search.py                DuckDuckGo + Bing HTML (keyless)
    academic_search.py           Crossref + arXiv + Semantic Scholar (keyless)
    archive_access.py            Internet Archive + Wikimedia + Openverse (keyless)
    calculator.py                AST-based safe math + financial formulas
    citation_manager.py          Local JSON bibliography + BibTeX/APA formatting
    peer_review.py               Deterministic rubric scoring (no LLM)
    code_execution.py            Local subprocess or E2B hosted sandbox
config.py                    CuriosityEngineConfig + interactive setup wizard
providers.py                 ModelClient ABC + Anthropic/OpenAI-compat + tool-use loops + EmbeddingClient
retry_utils.py               provider-agnostic retry with exponential backoff
journal.py                   JSON-backed journal (entries/xrefs/insights/register/predictions/focus/queue/embeddings)
register.py                  markdown rendering for the verified-insights artifact
models.py                    dataclasses (UncertaintyItem, ResearchQuestion, JournalEntry,
                             CrossReference, Insight, RegisterEntry, Prediction, EngineConfig)
prompts.py                   prompt templates for every model call
json_utils.py                robust JSON extraction from LLM text (handles fences/junk)
Dockerfile                   python:3.13-slim + scientific stack + non-root user
docker-compose.yml           TTY, volume mounts for config + data, env passthrough
curiosity                    wrapper script (builds image on first run)
.github/workflows/ci.yml     lint + syntax + smoke-test + Docker build
```

### Tool system

The investigator and verifier both see the full tool set on every call. Currently available:

| Tool | Provider | Keyless? | What it does |
|---|---|---|---|
| `web_search` (Anthropic server) | Anthropic only | n/a | Native live search; provided by Anthropic server-side |
| `web_search` (client, DuckDuckGo + Bing) | All providers | ✅ | Keyless fallback for non-Anthropic primaries |
| `web_fetch` | All | ✅ | HTTP GET + plaintext extraction via trafilatura |
| `academic_search` | All | ✅ | Crossref + arXiv + Semantic Scholar |
| `archive_access` | All | ✅ | Internet Archive + Wikimedia Commons + Openverse |
| `calculator` | All | ✅ | AST-based math; supports npv, cagr, wacc, pmt |
| `citation_manager` | All | ✅ | Local JSON bibliography → BibTeX / APA |
| `peer_review` | All | ✅ | Deterministic rubric scoring (no LLM calls) |
| `code_execution` (Anthropic server) | Anthropic only | n/a | Native sandboxed Python via `code_execution_20250825` |
| `code_execution` (client) | All providers | ✅ | Local subprocess w/ timeout + output cap. Optional E2B hosted sandbox when `E2B_API_KEY` is set (`pip install e2b-code-interpreter`). |

**Scientific Python in `code_execution`**: under Docker the image ships with numpy, scipy, pandas, scikit-learn, matplotlib preinstalled. Under the local venv path, `pip install numpy scipy pandas scikit-learn matplotlib` if you want the model to use them.

**Security**: the **Docker path fully isolates** client-side `code_execution` — arbitrary model-generated Python can only touch the container's filesystem. The local subprocess backend outside Docker is *not* a security sandbox; it restricts env/CPU/timeout/output but cannot stop arbitrary file I/O under your user. For real isolation without Docker, set `E2B_API_KEY` (hosted sandbox, pay-per-use).

Tools are auto-discovered on engine init (any module under `engine/tools/` that subclasses `Tool` registers itself). `./curiosity --list-tools` dumps the current set.

Adding a new tool:

```python
# engine/tools/my_tool.py
from engine.tools.base import Tool, ToolError

class MyTool(Tool):
    name = "my_tool"
    description = "One-line hint, then detail."
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def execute(self, args: dict) -> str:
        return f"result for {args['query']}"
```

That's it — subclassing auto-registers. Both Anthropic and OpenAI-compat primaries see the tool on the next run.

### Knowledge graph

`engine/graph.py` builds a NetworkX multigraph over the journal:

- **Nodes**: journal entries, cross-references, insights, register entries, predictions, sources (normalized), tags.
- **Edges**: `shares-tag`, `cites-source`, `semantic-similarity` (when embeddings are available), `cross-referenced-by`, `supports-insight`, `registered-as`, `predicts`, `cites`, `has-tag`.

**Source normalization.** Raw source strings from investigations arrive in many aliases (`arxiv.org/abs/2404.04865`, `arxiv.org/pdf/2404.04865v1`, `arXiv:2404.04865`, `doi.org/10.…`, bare titles). `_normalize_source()` canonicalizes them to `arxiv:<id>`, `doi:<id>`, or a whitespace-collapsed title, so `cites-source` edges actually appear when two entries reference the same paper via different surface forms.

**Cross-domain bias in selection.** `select_entries_for_xref()` biases entry selection toward *cross-domain* neighbors: `shares-tag` edges contribute only 0.25× weight (same-domain is less interesting than cross-domain), and a +2.5 bonus fires when two connected entries have zero `domain_tags` overlap. This aligns the selector with the cross-reference prompt, which has always said *"connections between entries with DISSIMILAR domain tags are where novelty lives."*

Used to:

1. **Pick entries for cross-reference** that are *connected-but-not-yet-cross-referenced* and *span domains* — structural "knowledge-gap intersections" preferring cross-field bridges.
2. **Surface structural summaries** via `--graph-summary`: component counts, hub entries, unexplored-pair counts.
3. **Export** (`--graph-export graph.graphml|gexf|json`) for Gephi, Cytoscape, D3, etc.
4. **Drive the interactive graph UI** — click any node to see full content (question, takeaways, verdict, premises/synthesis, predictions) with no text cropping.

Without the graph, cross-reference would just send recent-N entries to the model and ask for connections. With it, cross-reference targets the pairs the model *hasn't yet* linked, specifically preferring the cross-domain ones.

### Semantic retrieval

`engine/embeddings.py` embeds each journal entry's question + key_takeaways using an OpenAI-compatible embeddings endpoint (default `text-embedding-3-small`). Embeddings are cached in the journal under the `embeddings` key.

- **Entries are auto-embedded** on creation when an embedding-capable profile is configured.
- **`--embed-backfill`** computes embeddings for any legacy entries.
- **`--find-similar "query" --top-k N`** surfaces entries semantically close to a free-text query.
- **Semantic-similarity edges** feed into the knowledge graph, augmenting tag-based structural analysis.

Gracefully disabled if no embedding-capable profile is configured — the engine still runs; `--find-similar` and similarity edges are unavailable.

### Design principles

- **Hypothesis before evidence.** The investigator commits to a specific, falsifiable answer *before* searching. Surprise is then a comparison, not a self-report. Confidence movement post-investigation is clamped by verdict × surprise-delta rules so LLMs can't inflate post-hoc.
- **Novelty is structural, not vibes-based.** Premises vs synthesis are scored independently; the register gate requires `premises_supported=TRUE AND synthesis_findable=FALSE`. A `novelty_type` classifier distinguishes new synthesis from restatement, extension, correction, or unsupported.
- **Cross-domain is a first-class move.** The cross-domain analog probe fires on surprising findings and asks which distant fields have structurally analogous mechanisms. The cross-reference selector biases toward cross-domain neighbors.
- **Anti-attractor gate.** Cross-references that re-convene the same entry clusters get downranked or skipped — both in the prompt and in code — to push the engine off local minima.
- **Priority-ordered queue.** Emergent questions carry a priority score from the reward signal of their source (surprise, xref novelty, analog). Dequeue is human-first, then priority-descending, then fresh-generated.
- **Durable state across sessions.** Every cycle persists to JSON. Journal + cross-references + insights + register + predictions + question queue + embeddings + focus + last_domain all survive restarts.
- **Human-in-the-loop, not human-out-of-loop.** `--set-focus`, `--add-question`, `--review-register` are first-class controls. Rejections feed back into future verifier prompts.
- **Register-as-artifact.** Only verified insights reach `register.md`. An empty register is a correct signal that nothing survived scrutiny — better than a register full of shallow claims.
- **Predictions close the loop.** Every register entry carries falsifiable predictions with a target_date. `--check-predictions` revisits them later; reality is the final filter.
- **Duck-typed retry.** The retry wrapper detects transient errors across SDKs by status_code and error class name, not by importing provider modules. Adding a new provider is a client class, not a retry rewrite.
- **Mixin composition.** Each phase is a mixin on `CuriosityEngine`. Adding a new phase is a new mixin; no central dispatcher to update.

---

## Honest limitations

The goal — "find novel ideas" — is ambitious; here is what the system as built does *not* do well:

1. **Model overlap remains.** Primary and verifier share large overlap in training data even across families; cross-family verification reduces but doesn't eliminate shared blind spots. The premises-vs-synthesis decomposition makes the failure mode narrower (the verifier has to find the specific synthesis, not just ingredients), but a claim both models have seen will still slip through.
2. **Novelty is measured as "divergence from training-data prior, filtered through two models' judgment."** That's a real bar, but narrower than "genuine discovery." The system is more reliable at *rediscovering* connections from the literature than at *generating* truly unprecedented ones. Where it excels is in **cross-domain reframing** — biology→algorithmics moves are structurally supported by the analog probe, and those are often where the biggest jumps actually come from.
3. **Code execution exists, but real experiments are limited.** `code_execution` lets the model run Python with numpy/scipy, which is enough for analytical checks, small simulations, re-deriving cited numbers. Not enough for training models, large-scale data analysis, or anything requiring GPUs. A dedicated GPU-backed executor would close this gap.
4. **Surprise is calibrated by the same model that generated the hypothesis.** Splitting investigate into three calls helps (the hypothesis is committed to before the findings arrive), and explicit confidence-calibration rules + a post-hoc clamp enforce that confidence movement is consistent with verdict × delta. But the surprise grader is still the primary model. A truly adversarial surprise grader would be a different family.
5. **Cross-ref is bounded by prompt size.** We slim, window (default 20), graph-rank with cross-domain bias, and anti-attractor-gate the candidates — but once the journal is rich in a narrow domain, the engine may find nothing new above the novelty floor. Graceful degradation, not failure.
6. **Attractor basins still happen.** A+B gates reduce re-convergence on the same clusters but don't eliminate it — especially under a narrow focus. The cross-domain analog probe is the primary counterforce; if it returns no strong analogs, the engine reverts to within-domain recombination.
7. **Analog probe quality is LLM-dependent.** Dynamically derived analog domains depend on the primary model's conceptual range. A weaker model will produce weaker analogs. The prompt explicitly rejects shallow analogs ("it's like how brains work") but cannot guarantee strong ones.
8. **Style bias toward articulate claims.** Every output is structured JSON. Insights that need paragraphs to explain, or that are pre-articulate, are selected against.

What this **is** good for:

- A disciplined, session-spanning research assistant that enforces hypothesis-first epistemic hygiene.
- Surfacing questions worth investigating — even when the "insights" themselves turn out to be derivative, the questions often aren't.
- Building a substantiated, citation-linked trail of reasoning (`register.md`) on a topic under active study.
- Finding connections across your own accumulated investigations faster than you could unaided — especially with the graph and semantic search.
- Human-directed deep research where the engine does the search + synthesis legwork and you make the judgment calls via `--review-register`.

---

## Roadmap

| # | Feature | Status |
|---|---|---|
| 1 | Cross-model adversarial verification | **done** (Phase 1) |
| 2 | Predictions with time-horizon | **done** (Phase 2) |
| 3 | Human-in-the-loop ground truth | **done (L1)** — post-hoc review + rejection feedback; L2/L3 (mid-cycle steering, sync interruption) planned |
| 4 | Real tools beyond search (web_fetch, arxiv, Semantic Scholar, calculator, archive, etc.) | **done** (Phase 4) |
| 5 | Multi-agent disagreement (investigators with different priors) | planned |
| 6 | Code execution | **done** — Anthropic server tool + client subprocess/E2B fallback |
| A | Focus + human question injection | **done** |
| B | Knowledge graph for structural cross-ref | **done** |
| C | Semantic retrieval (embeddings) | **done** |
| D | Premises-vs-synthesis verification decomposition + novelty_type classifier | **done** |
| E | Cross-domain analog probe (dynamic LLM-derived distant-field reframing) | **done** |
| F | Priority-ordered question queue + priority-based dequeue | **done** |
| G | Anti-attractor gate on cross-reference (prompt + code) | **done** |
| H | Cross-domain bias in entry-selection graph heuristic | **done** |
| I | Surprise confidence calibration (prompt rules + post-hoc clamp) | **done** |
| J | Source normalization (arXiv/DOI/title → canonical id) | **done** |
| K | `inconclusive` verdict + held register pipeline + settlement plans | **done** |
| L | Re-verification of previously-unregistered insights under current rules (CLI + web button) | **done** |
| M | Admin tab — consolidated maintenance operations (cross-ref / synth-orphans / reverify / check-predictions) with work-to-do counters | **done** |
| N | Orphaned-xref recovery (`--synth-orphaned-xrefs`) — salvages state when a run dies between cross-ref and synthesis | **done** |
| O | Per-phase model routing (`[engine].cross_ref_role` + `[models.<name>]` extras) — offload cross-ref to a faster model | **done** |
| P | Role-based profile swap (`--primary-role` / `--verifier-role` / `--cross-ref-role`) — copies whole profile, not just model name | **done** |
| Q | Challenged-hedge guardrail — code-level upgrade from `challenged` to `validated` when decomposition is unambiguous and reasoning_flaws contain no substantive markers | **done** |
| R | Configurable per-request HTTP timeout (`timeout_seconds` on each profile, default 300s) — handles reasoning-mode first-token latency | **done** |
| — | Controlled parallelism — parallel investigation fan-out + synth fan-out + verify fan-out with shared per-tool rate limiters | backlog |
| — | Foreign-lens phase (scheduled cross-domain creativity burst) | backlog |
| — | Insight de-duplication via embeddings (drop near-duplicate syntheses) | backlog |
| — | GPU-backed experimental executor | backlog |
| — | Mid-cycle interruption / steering (Phase 3 L2-L3) | backlog |
| — | Multi-journal federation / cross-journal retrieval | backlog |
| — | Adversarial surprise grader (different-family model for Phase 3 assess) | backlog |

---

## Development

```bash
# Run the CLI out of the venv
.venv/bin/python curiosity_engine.py --help

# Lint
.venv/bin/ruff check --exclude .venv --exclude data .

# Install pre-commit hooks (trufflehog secret scanning)
pre-commit install

# Manually scan for secrets
trufflehog git file://. --only-verified --fail

# Rebuild the Docker image (after requirements.txt change or Dockerfile edit)
./curiosity --rebuild --show-journal
```

### Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and on pull requests. Two jobs:

- **`lint-and-smoke`**: ruff, syntax check, tool-discovery smoke, core-imports smoke, calculator functional test — no API keys required.
- **`docker-build`**: builds the image to verify the Dockerfile remains green.

---

## Files generated at runtime

- `~/.CuriosityEngine/engine.toml` — model connection config (interactive setup writes this).
- `./research_journal.json` (or `./data/research_journal.json` under Docker) — full journal state: entries, cross-references, insights, register, predictions, queued questions, focus, embeddings. Path configurable via `--journal`.
- `./register.md` — human-readable artifact, auto-written when the register changes. Path configurable via `EngineConfig.register_markdown_path`.
- `*_bibliography.json` / `refs.json` — local bibliography files written by the `citation_manager` tool.

All runtime artifacts are `.gitignore`d by default.

---

## License

**Copyright (c) 2026 sfw. All rights reserved.**

This repository is shared for reference and evaluation only. No part of the code, documentation, or prompts may be copied, redistributed, modified, or used in derivative or commercial works without the copyright holder's express written permission.

See [LICENSE.md](./LICENSE.md) for the full terms.
