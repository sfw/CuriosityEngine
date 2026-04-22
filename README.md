# Curiosity Engine

A proof-of-concept research loop that generates its own questions from self-assessed uncertainty, investigates them with real-time web search, cross-references findings across sessions, and routes synthesized insights through an independent adversarial verifier before committing them to a durable register — each with falsifiable predictions attached.

> **Status**: proof of concept. Phases 1 and 2 of 5 are implemented. The goal of "finding novel ideas" is ambitious and the system has calibrated limitations — see [Honest limitations](#honest-limitations).

---

## Core thesis

> Novel insight rarely emerges from a single query. It emerges at the **intersection** of knowledge gaps, when an investigation forces a prior hypothesis to collide with fresh evidence, and when an independent reviewer fails to find prior art for a resulting connection.

The engine operationalizes this by:

1. **Committing to hypotheses** before searching, so surprise is a comparison rather than a self-report.
2. **Cross-referencing** accumulated findings across many sessions to surface connections no single prompt would produce.
3. **Adversarially verifying** synthesized insights with a *different model family* (cross-model verification), not the one that produced them.
4. **Attaching falsifiable predictions** to every validated insight, so the test of time separates predictive claims from post-hoc narrative fitting.

---

## Pipeline

```
  introspect        →  uncertainties (what the model is uncertain about)
  generate          →  ranked investigable questions
  investigate                 (three stages)
    ├─ hypothesize  →  commit to a pre-investigation answer
    ├─ search       →  web_search for fresh evidence
    └─ assess       →  compare findings to hypothesis, compute surprise
  cross-reference   →  find patterns/contradictions/implications across entries
  synthesize        →  promote high-novelty connections to insights
  verify (cross-model)
    └─ adversarial review with different model family
        └─ validated + confidence >= floor  →  REGISTER
            └─ emit 1-3 falsifiable predictions with time horizons
  check predictions (later, on demand)
    └─ revisit due predictions; mark confirmed/refuted/inconclusive/expired
    └─ update register entry lifecycle status
```

Every step persists to `research_journal.json`. The human-readable artifact is `register.md`, auto-written whenever the register changes.

---

## Prerequisites

- **Python 3.11+** (we use stdlib `tomllib`).
- API key for at least one provider. Two providers are supported:
  - **Anthropic** (Claude) — supports server-side `web_search` as a tool.
  - **OpenAI-compat** — any endpoint that speaks the OpenAI chat-completions protocol: OpenAI itself, Gemini (via its OpenAI-compat endpoint), OpenRouter, Ollama (`/v1` mode), xAI, Groq, Together, DeepSeek, LM Studio, and anything else that matches the contract.

Recommended setup: **Anthropic Claude as primary, a different-family model (OpenAI GPT, Google Gemini, etc.) as verifier.** Same-family verification defeats most of the adversarial point.

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

## Usage

### Iterative workflow: focus → run → review → redirect → continue

The engine is designed to be driven as a loop where you narrow scope between runs. Example session on "novel idea generation by LLMs":

```bash
# One-time: set the durable focus on this journal
./curiosity --set-focus "novel idea generation by LLMs — the role of latent-space exploration vs template recombination" --journal data/ideation.json

# Run 3 cycles with that focus active in every prompt
./curiosity --cycles 3 --domain "novel idea generation by LLMs" --journal data/ideation.json

# Review what came out
./curiosity --show-journal       --journal data/ideation.json
./curiosity --show-insights      --journal data/ideation.json
./curiosity --review-register    --journal data/ideation.json   # approve/reject with reasons
./curiosity --graph-summary      --journal data/ideation.json   # see the knowledge graph

# Direct the next cycles by injecting specific questions
./curiosity --add-question "Does high-temperature decoding produce genuinely novel outputs or statistical recombinations?" \
            --add-question "Is creativity measurable as distance in activation space?" \
            --journal data/ideation.json

# Semantic lookup over accumulated research
./curiosity --find-similar "ways to measure novelty of an output" --top-k 5 --journal data/ideation.json

# Resume — human-queued questions consume the investigations budget first
./curiosity --cycles 3 --journal data/ideation.json
```

**How direction propagates**:
- `--set-focus` persists on the journal; every introspect / generate / investigate / cross-ref / synthesize prompt gets a `USER FOCUS` section.
- `--add-question` pushes onto `question_queue` with `source="human"`; those questions get investigated first each cycle, ahead of model-generated ones.
- `--review-register` rejection reasons get injected into future verifier prompts as "prior human rejections" — the verifier learns what you consider too thin.
- Cross-reference uses the knowledge graph to specifically surface entry pairs that share tags/sources/embeddings but don't yet have a cross-reference — the structural definition of "knowledge-gap intersection."

### Individual commands

```bash
# Run one curiosity cycle against the default domain
python curiosity_engine.py --cycles 1

# Narrow the topic and use a topic-specific journal
python curiosity_engine.py --cycles 3 \
    --domain "mechanistic interpretability of MLP layers in transformers" \
    --journal "./mlp_journal.json"

# Run cross-reference + synthesize + verify on the existing journal
python curiosity_engine.py --cross-ref-only

# Inspect state (no API calls)
python curiosity_engine.py --show-journal       # counts + domains + high-surprise entries
python curiosity_engine.py --show-insights      # synthesized insights (pre-verification)
python curiosity_engine.py --show-register      # verified-only insights with substantiation
python curiosity_engine.py --show-predictions   # all stored predictions with status
python curiosity_engine.py --list-tools         # registered research + calculation tools

# Steering
python curiosity_engine.py --set-focus "TEXT"   # persistent investigation focus on this journal
python curiosity_engine.py --show-focus
python curiosity_engine.py --clear-focus
python curiosity_engine.py --add-question "?"   # push a user-directed question onto the queue
python curiosity_engine.py --list-questions
python curiosity_engine.py --clear-questions

# Knowledge graph
python curiosity_engine.py --graph-summary
python curiosity_engine.py --graph-export graph.json   # .graphml / .gexf / .json

# Semantic retrieval
python curiosity_engine.py --embed-backfill            # embed legacy entries (one-time)
python curiosity_engine.py --find-similar "query" --top-k 10

# Revisit predictions whose target_date has arrived (costs an API call per prediction)
python curiosity_engine.py --check-predictions

# Force-review every pending prediction regardless of target_date
python curiosity_engine.py --check-predictions-all

# Override models for a single run (keeps TOML provider/endpoint intact)
python curiosity_engine.py --cycles 3 --primary-model claude-sonnet-4-6 --verifier-model gpt-5.1
```

Cross-reference runs every `cross_ref_frequency` cycles (default 3). On those cycles, high-novelty xrefs (`>= novelty_threshold`, default 0.7) get synthesized into insights, which then face the adversarial verifier. Only those with verdict `validated` and `verified_confidence >= register_confidence_floor` (default 0.6) enter the register.

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

[models.verifier]                     # optional; defaults to primary if omitted
provider = "openai_compat"
name = "gpt-5.1"
base_url = "https://api.openai.com/v1"
# api_key = "..."

[retry]
max_attempts = 5
base_delay_seconds = 0.5
max_delay_seconds = 8.0
jitter_seconds = 0.25
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

### Operational settings

The CLI wraps `EngineConfig` (in `models.py`); defaults are:

- `questions_per_cycle = 3` — how many ranked questions generated per cycle.
- `investigations_per_cycle = 1` — how many of those actually investigated with web_search.
- `cross_ref_frequency = 3` — cross-reference runs every N cycles.
- `cross_ref_window = 20` — max entries sent to cross-ref prompt.
- `novelty_threshold = 0.7` — cross-ref novelty score required to trigger synthesize.
- `register_confidence_floor = 0.6` — verifier confidence required to register.
- `verify_insights = True` — toggle cross-model verification.

---

## Architecture

```
curiosity_engine.py          CLI entry
engine/                      orchestrator package (composed via mixins)
  core.py                      class def, init, plumbing (_call_*), run loop
  introspect.py                Phase 1-2: uncertainties → questions
  investigation.py             Phase 3: hypothesis → web_search → surprise
  cross_reference.py           Phase 4-5: xref → synthesize
  verification.py              Phase 6-7: adversarial verify + prediction lifecycle
  display.py                   show_* (no API calls)
  tools/                       pluggable research / calculation tools
    base.py                      Tool ABC + ToolRegistry + discover_tools()
    web_fetch.py                 HTTP GET with plaintext extraction
    web_search.py                DuckDuckGo + Bing HTML (keyless)
    academic_search.py           Crossref + arXiv + Semantic Scholar (keyless)
    archive_access.py            Internet Archive + Wikimedia + Openverse (keyless)
    calculator.py                AST-based safe math + financial formulas
    citation_manager.py          Local JSON bibliography + BibTeX/APA formatting
    peer_review.py               Deterministic rubric scoring (no LLM)
config.py                    CuriosityEngineConfig + interactive setup wizard
providers.py                 ModelClient ABC + Anthropic/OpenAI-compat impls + tool-use loops
retry_utils.py               provider-agnostic retry with exponential backoff
journal.py                   JSON-backed journal (entries/xrefs/insights/register/predictions)
register.py                  markdown rendering for the verified-insights artifact
models.py                    dataclasses (UncertaintyItem, ResearchQuestion, JournalEntry,
                             CrossReference, Insight, RegisterEntry, Prediction, EngineConfig)
prompts.py                   prompt templates for every model call
json_utils.py                robust JSON extraction from LLM text (handles fences/junk)
```

### Tool system

The investigator and verifier both see a full tool set on every call. Currently available:

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
| `code_execution` (client) | All providers | ✅ | Local subprocess with timeout + output cap. Optional E2B hosted sandbox when `E2B_API_KEY` is set (`pip install e2b-code-interpreter`). |

**Scientific Python in `code_execution`**: the local subprocess backend runs in the project's venv, so `numpy`, `scipy`, `pandas`, `sklearn`, `matplotlib` are only available if installed there. For a research engine you probably want them:

```bash
pip install numpy scipy pandas scikit-learn matplotlib
```

The E2B backend has these pre-installed — recommended if you're paying to run the engine unattended.

**Security caveat on local `code_execution`**: the subprocess runs under your user with restricted env (HOME/TMPDIR redirected to a scratch dir, CPU limit, timeout, 200 KB output cap) but it is *not* a security sandbox. The model can write files, make network calls, and see things under your `$PATH`. Use E2B (`pip install e2b-code-interpreter` + set `E2B_API_KEY`) for real isolation.

Tools are auto-discovered on engine init (any module under `engine/tools/` that subclasses `Tool` registers itself). `python curiosity_engine.py --list-tools` dumps the current set.

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

### Design principles

- **Hypothesis before evidence.** The investigator commits to a specific, falsifiable answer *before* searching. Surprise is then a comparison, not a self-report.
- **Durable state across sessions.** Every cycle persists to JSON. Journal + cross-references + insights + register + predictions + question queue all survive restarts.
- **Emergent-question feedback loop.** New questions surfaced by investigations and cross-references enter a queue that biases the next introspection pass.
- **Register-as-artifact.** Only verified insights reach `register.md`. An empty register is a correct signal that nothing survived scrutiny — better than a register full of shallow claims.
- **Duck-typed retry.** The retry wrapper detects transient errors across SDKs by status_code and error class name, not by importing provider modules. Adding a new provider is a client class, not a retry rewrite.
- **Mixin composition.** Each phase is a mixin on `CuriosityEngine`. Adding a new phase is a new mixin; no central dispatcher to update.

---

## Honest limitations

The goal — "find novel ideas" — is ambitious; here is what the system as built does *not* do:

1. **Every novelty signal is self-reported.** Cross-ref novelty, synthesize confidence, verify verdict — all LLM-generated. The verifier catches different blind spots when it's a different family, but `web_search` is still the only genuinely external signal and its coverage is bounded by what's indexed.
2. **Novelty is measured as "divergence from training-data prior, filtered through model judgment."** That's a real bar, but narrower than "genuine discovery."
3. **Limited experiments.** Phase 4 added web_fetch, academic_search, archive_access, calculator, citation_manager, and peer_review — but the system still has no path to *running code* against datasets or training small models. Adding a code-execution tool (Anthropic's `code_execution_20250825` server tool, or a sandboxed local runner) would close this gap.
4. **Surprise is calibrated by the same model.** Splitting investigate into three calls helps (the hypothesis is committed to before the findings arrive), but the surprise grader is still the primary model. A truly adversarial surprise grader would be a different family.
5. **Cross-ref is bounded by prompt size.** We slim and window entries (default 20), but once the journal is rich in a narrow domain, the model may find nothing new above the novelty floor. Graceful degradation, not failure — but a real ceiling.
6. **Style bias toward articulate claims.** Every output is structured JSON. Insights that need paragraphs to explain, or that are pre-articulate, are selected against.

What this **is** good for:

- A disciplined, session-spanning research assistant that enforces hypothesis-first epistemic hygiene.
- Surfacing questions worth investigating, even when the "insights" themselves are derivative.
- Building a substantiated trail of reasoning (`register.md`) on a topic under active study.
- Finding connections across your own accumulated investigations faster than you could unaided.

---

## Roadmap

Five features described in the architectural review, in priority order:

| # | Feature | Status |
|---|---|---|
| 1 | Cross-model adversarial verification | **done** (Phase 1) |
| 2 | Predictions with time-horizon | **done** (Phase 2) |
| 3 | Human-in-the-loop ground truth | planned |
| 4 | Real tools beyond search (web_fetch, arxiv, Semantic Scholar, calculator, archive, etc.) | **done** (Phase 4) |
| 5 | Multi-agent disagreement (investigators with different priors) | planned |

---

## Development

```bash
# Run the CLI out of the venv
.venv/bin/python curiosity_engine.py --help

# Install pre-commit hooks (trufflehog secret scanning)
pre-commit install

# Manually scan for secrets
trufflehog git file://. --only-verified --fail
```

---

## Files generated at runtime

- `~/.CuriosityEngine/engine.toml` — model connection config (interactive setup writes this).
- `./research_journal.json` — full journal state (entries, cross-references, insights, register, predictions, queued questions). Path configurable via `--journal`.
- `./register.md` — human-readable artifact, auto-written when the register changes. Path configurable via `EngineConfig.register_markdown_path`.

All three are `.gitignore`d by default except the example config.

---

## License

Proof-of-concept code; no license attached. Treat as exploratory material.
