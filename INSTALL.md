# Installation guide

Step-by-step instructions for getting CuriosityEngine running. Aimed at someone who hasn't run a Docker-based research tool before and just wants the engine working on their laptop.

If you've installed similar projects before, the [README's Setup section](./README.md#setup) is a faster ten-line summary.

---

## What you need before starting

1. **A computer with Docker Desktop** (recommended path) or **Python 3.11+** (alternative path).
   - Docker Desktop: free for personal use. Download for [macOS](https://www.docker.com/products/docker-desktop) / [Windows](https://www.docker.com/products/docker-desktop) / [Linux](https://docs.docker.com/desktop/install/linux/). Open it once after install — Docker won't run until the desktop app is open.
   - Python 3.11+: only needed if you don't want to use Docker. Comes with macOS recent versions or via [python.org](https://www.python.org/downloads/) / your package manager.

2. **An API key from at least one LLM provider**. Cheapest options that work:
   - [Anthropic](https://console.anthropic.com/) (Claude) — supports server-side `web_search` and `code_execution` natively. ~$5-10/month for moderate use.
   - [OpenAI](https://platform.openai.com/api-keys) (GPT) — works as a verifier for cross-family adversarial review. ~$5-10/month for moderate use.
   - **Recommendation: get both**. Use one as primary, the other as verifier. Cross-family verification is one of CE's main design choices and only works with two providers.
   - Other supported: Google Gemini, OpenRouter, Ollama (local), xAI, Groq, Together, DeepSeek, Moonshot, LM Studio. Full list and base URLs in the [README's OpenAI-compat shortcuts](./README.md#openai-compat-endpoint-shortcuts).

3. **A terminal**. Mac: open Terminal.app. Windows: PowerShell. Linux: whatever you use.

4. **Git** for cloning the repo. `git --version` should print something. If not, [install it](https://git-scm.com/downloads).

---

## Path A — Docker (recommended)

Docker fully isolates the `code_execution` tool from your host machine. This matters: the engine can run Python it generates, and you don't want that running directly on your laptop without isolation.

### 1. Clone the repo

```bash
cd ~/                          # or wherever you keep code
git clone <this-repo-url> CuriosityEngine
cd CuriosityEngine
```

### 2. First-run setup

```bash
./curiosity --show-journal
```

The first run does three things:
1. Builds the Docker image (~5-10 minutes — pulls Python 3.13, scientific stack, dependencies).
2. Runs the interactive setup wizard inside the container — asks for your primary model + verifier model + API keys.
3. Saves config to `~/.CuriosityEngine/engine.toml` on your host machine (persists across rebuilds).

The wizard prompts:
- **Primary provider** (anthropic | openai_compat). Pick anthropic if you have a Claude key.
- **Primary model name** (e.g. `claude-sonnet-4-6`).
- **API key** — paste it. Stored locally only.
- **Verifier provider/model** — different family from primary. If primary is Claude, pick OpenAI here (e.g. `gpt-5.1`).
- **Advanced settings** — accept defaults; you can tune later via the web UI.

If something goes wrong: delete `~/.CuriosityEngine/engine.toml` and run the command again to re-trigger the wizard.

### 3. Run a single cycle to verify it works

```bash
./curiosity --cycles 1 --domain "your topic of interest"
```

Replace "your topic of interest" with whatever you actually want to research. Examples: "AI alignment", "battery chemistry", "consumer credit risk modelling". The engine introspects, generates questions, runs investigations with web search, and saves state to `./data/research_journal.json`.

A single cycle takes ~5-15 minutes depending on your model speed and how deep the investigation goes.

### 4. Open the web UI

```bash
./curiosity web
```

Browse to **http://localhost:8000** in your web browser. The web UI gives you:
- A list of journals (each `data/*.json` is a journal).
- Per-journal tabs: Overview, Entries, Insights, Register, Predictions, Focus & Queue, Graph, Runs, Coverage, Admin.
- Settings page for editing every config knob.
- A Run page to start cycles via the browser instead of the terminal.

To stop the web server:
```bash
./curiosity web stop
```

To see what the web server is doing:
```bash
./curiosity web logs
```

To restart it (e.g. after editing the engine code):
```bash
./curiosity web restart
```

### 5. (Optional) Rebuild after dependency updates

If you `git pull` and CE's `requirements.txt` changes:
```bash
./curiosity --rebuild --show-journal
```

This rebuilds the Docker image. Takes ~5-10 minutes. Your config and journal data are unaffected (they live outside the container).

---

## Path B — Local Python venv (no Docker)

Less isolation. The `code_execution` tool will run subprocess Python directly on your machine. Use [E2B](https://e2b.dev/) sandbox for isolation if you take this path. Only recommended if you can't use Docker.

```bash
git clone <this-repo-url> CuriosityEngine
cd CuriosityEngine

python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install numpy scipy pandas scikit-learn matplotlib    # optional, for code_execution

python curiosity_engine.py --show-journal     # triggers setup wizard
```

The wizard works the same as the Docker path. From here, all CLI commands are `python curiosity_engine.py ...` instead of `./curiosity ...`. Web UI:
```bash
uvicorn web.main:app --host 127.0.0.1 --port 8000
```

---

## Using the web interface

Once `./curiosity web` is running and you've opened http://localhost:8000:

### Journals page (root)

Lists every journal under `data/*.json`. Click a journal to enter its dashboard. To create a new journal, just run `./curiosity --cycles 1 --domain "..." --journal data/<your-name>.json`. The first cycle creates the file.

### Per-journal dashboard tabs

- **Overview** — counts, recent entries, surprise summary. Start here to see what the journal looks like.
- **Entries** — every investigation cycle's record. Filter by domain tag.
- **Insights** — synthesized cross-references that haven't been verified yet.
- **Register** — the durable artifact: validated insights with full audit trail. Per-component novelty badges (rose = restatement, amber = extension, emerald = new_synthesis), Pareto-axis values, predictions attached, "Export directive →" button per qualifying entry.
- **Predictions** — falsifiable claims with status tracking. Run `./curiosity --check-predictions` to revisit due ones.
- **Focus & Queue** — set the journal's research focus, add your own questions, drag-and-drop priority within source buckets.
- **Graph** — interactive force-directed knowledge graph. Click any node for detail.
- **Runs** — per-journal run history. Live SSE log streaming on active runs. Stop button on running cycles.
- **Coverage** — negative-space gap matrix. Method × problem cells; click any cell for detail. Diff vs prior scan shows what filled / what's still open / what newly emerged.
- **Admin** — maintenance operations. The cards you'll use most:
  - **Re-verify register entries (audit)** — re-runs the verifier over existing register under updated rules. Append-only; never overwrites.
  - **Scan for unexplored gaps** — builds the (method × problem) matrix.
  - **Export research directives bundle** — runs the directive pipeline across every qualifying entry.
  - **Pareto admission frontier** (read-only) — shows which active entries set the admission bar.
  - **Known prior art anchors** — add human-curated peer systems the verifier MUST evaluate.

### Settings page

Edits `~/.CuriosityEngine/engine.toml` in place. Every engine knob is exposed:
- Model profiles (primary, verifier, optional extras)
- Per-phase model routing (cross-ref, directive_primary, directive_primary_fast, directive_verifier, gap-scan extract/classify, investigation_assessor)
- Loop knobs (cycles per cross-ref, novelty threshold, register confidence floor)
- Self-evolving verifier knobs:
  - `register_admission_mode` (scalar | pareto)
  - `synthesis_candidate_count` (Phase 6)
  - `introspection_persona_count` (Phase 7)
  - `idea_evolution_enabled` + floor + max_depth (Phase 8)
  - `hypothesis_variant_count` (Phase 9)
- Held pipeline + gap-scan thresholds + parallelism

Saving doesn't restart anything — the next cycle picks up the new config.

### Run page

Form to start a cycle. Pick the journal, set the domain (or leave it from the journal's last domain), set cycle count. Live SSE log streams while the cycle runs.

---

## Troubleshooting

**"Docker is not running"** → Open Docker Desktop. Wait for the whale icon to stop animating.

**Web UI 502 / 504** → Container probably crashed. `docker compose logs web` to see why. Common causes: stale config (delete `~/.CuriosityEngine/engine.toml` and re-run `./curiosity --show-journal`), API key invalid, model name wrong.

**"unterminated triple-quoted string literal" or other syntax errors after a git pull** → Docker bind-mount staleness on macOS. Restart the web container: `./curiosity web restart`.

**"rate limited" on academic_search** → expected on long runs. Engine has staged cooldowns (2/4/8/16/30s, cycling back). Just wait. If persistent: set `SEMANTIC_SCHOLAR_API_KEY` (free at [semanticscholar.org](https://www.semanticscholar.org/product/api#api-key-form)) for a private bucket, and `OPENALEX_MAILTO` for OpenAlex's polite pool.

**Cycle takes forever** → some reasoning models (Kimi K2.x, GPT-5 thinking, o-series) hang on big prompts in non-streaming mode. Increase `timeout_seconds` in the model profile, or switch to a smaller / non-reasoning model for the directive_primary_fast role.

**API key not found** → engine.toml stores keys, OR you can set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in your shell environment. Both work.

**Want to migrate an old `research_journal.json`** →
```bash
mkdir -p ./data
cp /old/path/research_journal.json ./data/
cp /old/path/register.md ./data/    # if you have it
```

---

## What to do next

1. **Set a focus** — `./curiosity --set-focus "TOPIC — specific lens"` so every prompt knows what you actually care about.
2. **Run cycles** — start with `--cycles 3` and review the register afterwards.
3. **Review the register** — `./curiosity --review-register` lets you approve / reject-with-reason / defer each new entry. Rejection reasons feed into future verifier prompts.
4. **Add known prior art anchors** — Admin tab → Known prior art. If you spot a peer system the verifier missed, add it; the verifier will be required to evaluate it on every future claim in that domain.
5. **Export directives** — once you have validated register entries, the "Export directive →" button generates a publication-shaped research plan from each entry.

For deeper context on what each piece does and why, read the [README's Data model section](./README.md#data-model--what-flows-through-the-pipeline) and [Self-evolving verifier section](./README.md#self-evolving-verifier).
