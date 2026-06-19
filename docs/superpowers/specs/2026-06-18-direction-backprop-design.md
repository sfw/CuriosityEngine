# Direction Insight Backpropagation + Verifier-as-Held-Out-Gate Note

**Date:** 2026-06-18
**Status:** Design approved, pending implementation
**Influence:** Arbor / Hypothesis Tree Refinement (arXiv:2606.11926, Jin et al.)

## Motivation

Arbor's ablation shows that *carrying distilled lessons forward across the
research loop* is the load-bearing component of long-horizon autonomous
research — removing insight propagation hurt more (81.82% → 54.54% Any-Medal)
than removing the tree structure itself (→ 63.64%). "The tree is useful only
when evidence can accumulate over it."

CE today regenerates uncertainties from scratch every cycle. `_build_journal_context()`
(`engine/introspect.py:89`) shows only the raw last-5 takeaways, the question
queue, and the domain-tag list. There is no *compressed, direction-level memory*
of what a line of inquiry has settled or where its open edge is. The engine can
re-ask resolved questions and fail to push past its own frontier.

This spec adds that missing layer — **adapted to CE's objective**, which is
structural novelty, not a scalar metric. The adaptation is the whole point:
Arbor's propagation is a *convergent* force toward a measurable optimum; CE's
value is *divergence* toward out-of-distribution synthesis. So the carried-forward
signal is framed as a **frontier edge to exceed**, never a belief to confirm.

## Scope

**In scope**
- **Item 1** — Direction Insight Backpropagation: abstract clusters of related
  investigations into direction-level priors and inject them into introspection.
- **Item 3** — Documentation note: CE's adversarial prior-art verifier is the
  role-analog of Arbor's held-out merge gate; CE deliberately rejects a scalar
  dev/held-out gate.

**Explicitly out of scope**
- **Item 2** — frontier pruning / question downweighting. For a novelty objective
  there is no clean falsification signal; pruning on a weak proxy (repeated low
  surprise / verifier rejection) antagonizes CE's premise that verifier-failure ≠
  capability-failure. The CE-safe positive half (push toward under-explored cells)
  already exists as the negative-space scan (`engine/negative_space.py`). Not built.

## Item 1 — Direction Insight Backpropagation

### Data flow

```
every N cycles, before introspect:
  build_graph(journal)                          # engine/graph.py (existing)
    → connected components of the entry subgraph
    → keep components with ≥ direction_min_cluster_size entries
    → cap to top direction_max_count by size
  for each candidate direction:
    _call_verifier(ABSTRACT_DIRECTION_PROMPT)   # cross-family model, NOT primary
    → {label, settled, open_edge, confidence}
    → journal.add_direction_insight(...)        # append-only, supersedes prior

introspect (every cycle):
  _build_journal_context()
    → inject top-K latest non-suppressed open_edges as "FRONTIER EDGES"
```

### 1. Clustering — reuse `engine/graph.py`

Connected components of the entry subgraph (already computed in
`graph_summary`, `engine/graph.py:384-391`). Components are formed from the
engineered edges (shares-tag, cites-source, semantic-similarity, cross-referenced)
— CE's curated relatedness structure, the analog of an Arbor subtree.

- A component with ≥ `direction_min_cluster_size` entries (default **4**) is a
  candidate direction. Singletons/tiny components are skipped — nothing to abstract.
- Cap to the top `direction_max_count` components by size (default **5**) to bound
  token cost and storage volume.
- A direction's identity for supersession is its `member_signature`: a stable hash
  of its sorted member entry ids.

### 2. Abstraction call — verifier (cross-family) model

One `_call_verifier` (`engine/core.py:229`) per candidate direction. Routing to
the verifier model — not `_call_primary` — is deliberate: abstracting over the
engine's own outputs is an *evaluative* act, and CE's design principle is
"different model for evaluation than generation." This breaks the self-flattery
loop where the primary model would author a prior and then consume it.

**Input** (members only): for each member entry — `question`, `key_takeaways`,
`surprise_delta`; plus descriptions of any cross-references whose `source_entries`
fall within the cluster.

**Prompt** (`ABSTRACT_DIRECTION_PROMPT`, new in `prompts.py`): domain-agnostic
(shape, not content, per CE convention). It explicitly invokes CE's novelty
signature — "the premises exist in the literature, the synthesis does not" — and
instructs the model to find the cluster's **negative-space edge**, not to
summarize content. Returns JSON:

| field | meaning |
|---|---|
| `label` | short direction name |
| `settled` | ≤2 bullets, deliberately terse — what this cluster has established |
| `open_edge` | **load-bearing**: the unexplored synthesis / contradiction / missing `(method, problem)` combination at the cluster's frontier, framed as an investigable edge |
| `confidence` | model self-rating [0,1] |

### 3. Storage — append-only, auditable

New journal field `direction_insights: list[dict]`. Each record:

```json
{
  "id": "dir-<8hex>",
  "label": "...",
  "member_entry_ids": ["j-...", "..."],
  "member_signature": "<sha1 of sorted member ids, 10 hex>",
  "settled": ["...", "..."],
  "open_edge": "...",
  "confidence": 0.0,
  "generated_at": "<iso8601>",
  "model": "<verifier model id>",
  "journal_size_at_gen": 0,
  "suppressed": false,
  "supersedes": "dir-... | null"
}
```

Re-abstracting a cluster whose membership changed appends a **new** record and
sets `supersedes` to the prior record's id for that `member_signature` lineage.
Original records are never mutated (audit-trail consistent with
`reverification_log`, `known_prior_art`, `rejection_log`).

**Journal methods** (in `journal.py`, persisted in `save()`/`_load()`):
- `add_direction_insight(record: dict)` — append + save.
- `latest_direction_insights() -> list[dict]` — for each `member_signature`
  lineage, the newest record by `generated_at`, **but only if that newest record
  is not suppressed**. If the lineage's newest record is suppressed, the lineage
  is excluded entirely — a suppressed latest never falls back to a stale older
  record. Used by injection.
- `suppress_direction_insight(insight_id: str) -> bool` — set `suppressed=true`
  on the record so its lineage drops out of `latest_direction_insights` (and thus
  out of injection).

### 4. Injection — frontier edges, not beliefs

`_build_journal_context()` (`engine/introspect.py:89`) gains a block appended
after the existing context, gated on `direction_backprop_enabled` and on there
being ≥1 latest non-suppressed insight. Top-K = `direction_max_injected`
(default **4**), ordered by `confidence` desc then recency:

```
FRONTIER EDGES (distilled from prior investigation clusters — do NOT
re-confirm what is settled; your job is to push PAST these open edges):
  - [<label>] settled: <one clause>. OPEN EDGE: <open_edge>
  ...
```

The framing is the convergence guard: edges to *exceed*, never beliefs to
*confirm*. `settled` is collapsed to a single clause so the model knows what to
skip — not what to anchor on. The hard cap ensures priors cannot dominate the
introspection prompt or homogenize the divergent personas.

### 5. Trigger — automated, every N cycles + on-demand inspection

- **In-loop:** before `introspect()`, every `direction_abstract_every_n_cycles`
  cycles (default **5**), gated by `direction_min_entries` (default **8**) so it
  is silent on small journals. Cycle counting reuses the engine's existing cycle
  index. (Approved: automate the trigger.)
- **Admin ops** (mirroring `--scan-gaps` / `--reverify-insights` in
  `curiosity_engine.py`): `--abstract-directions` to force a run, and
  `--show-directions` to print current latest non-suppressed insights for
  inspection / manual suppression.

### 6. Guards (the "be careful" budget)

| Risk | Guard |
|---|---|
| Self-flattery loop (primary authors + consumes its own prior) | Abstraction on cross-family verifier model |
| Convergence — engine hill-climbs its own beliefs, stops exploring OOD | "Frontier edge to exceed" framing; `settled` collapsed; no auto-pruning |
| Prior pollution dominating introspection | `direction_max_injected` hard cap (4) |
| A bad prior persisting across all future cycles | `suppress_direction_insight` + `--show-directions` inspection |
| Unwanted behavior change | `direction_backprop_enabled` global off switch (default True) |
| Importing Item 2's risks | No question downweighting, no pruning — out of scope |

### 7. Config additions (`config.py`, `[engine]` block)

| key | default | meaning |
|---|---|---|
| `direction_backprop_enabled` | `true` | master switch for abstraction + injection |
| `direction_abstract_every_n_cycles` | `5` | in-loop trigger cadence |
| `direction_min_entries` | `8` | journal size floor before abstraction runs |
| `direction_min_cluster_size` | `4` | min component size to qualify as a direction |
| `direction_max_count` | `5` | max directions abstracted per run |
| `direction_max_injected` | `4` | max frontier edges injected into introspection |

## Item 3 — Verifier-as-held-out-gate note (documentation only)

Add a short subsection to `README.md` (near "Core ideas" / the self-evolving
verifier discussion):

- Arbor's anti-overfitting result is its **held-out merge gate** — a candidate is
  admitted only if it beats the current best on a held-out eval, separating
  "looked good on the exploration signal" from "verified progress."
- CE's adversarial prior-art verifier is the **role-analog**: apparent novelty
  *against the journal* (the dev signal) is admitted only after surviving
  adversarial search *against the literature* (the held-out set). Local novelty
  that collapses under prior-art search is CE's "overfit."
- Honest caveat: this is analogy of **role**, not identity of **mechanism** —
  CE's gate is structural prior-art search, not a scalar metric comparison.
- CE deliberately does **not** adopt a numeric dev/held-out gate: a scalar gate
  would push the engine toward optimizing a novelty *proxy*, violating "novelty
  is structural, not vibes."
- Credit arXiv:2606.11926 as the influence for Item 1.

## Testing

Unit tests (LLM stubbed at the `_call_verifier` seam):
- Clustering: components below `direction_min_cluster_size` excluded; cap to
  `direction_max_count` respected; `member_signature` stable across member order.
- Storage: `add_direction_insight` persists across save/load; re-abstraction
  appends + links `supersedes`; `latest_direction_insights` returns newest
  non-suppressed per lineage; `suppress_direction_insight` removes from latest.
- Injection: block present only when enabled + insights exist; respects
  `direction_max_injected` cap; omits suppressed records; framing string present.
- Trigger: gated by `direction_min_entries` and cadence; off when
  `direction_backprop_enabled=false`.

## Files touched

- `journal.py` — new field + 3 methods + save/load wiring.
- `engine/introspect.py` — injection in `_build_journal_context`; trigger hook.
- `engine/direction_backprop.py` *(new)* — clustering + abstraction orchestration.
- `prompts.py` — `ABSTRACT_DIRECTION_PROMPT`.
- `engine/core.py` — cycle-cadence trigger call (or in introspect mixin).
- `curiosity_engine.py` — `--abstract-directions`, `--show-directions` admin ops.
- `config.py` — 6 config keys with defaults.
- `README.md` — Item 3 note + Item 1 feature description.
- tests — new test module.

## Operational note

Implementation edits `.py`, which triggers the uvicorn reload that kills a live
engine run. Implement only when no engine cycle is active.
