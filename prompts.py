"""Prompt templates for each phase of the Curiosity Engine loop."""

INTROSPECT_PROMPT = """You are a research engine performing structured self-interrogation about your own knowledge in the domain of {domain}.

{focus_block}Your task: identify specific areas where your knowledge is uncertain, contradictory, shallow, or unstable. Be brutally honest about what you don't know well.

{journal_context}

For each uncertainty you identify, classify it as one of:
- "contradiction": You hold beliefs that conflict with each other
- "gap": There's something important you simply don't know
- "shallow": You know the surface-level answer but lack deep understanding
- "unstable": Your answer would change depending on how the question is framed

Respond with EXACTLY this JSON structure (no other text):
{{
  "uncertainties": [
    {{
      "description": "specific description of what you're uncertain about",
      "uncertainty_type": "contradiction|gap|shallow|unstable",
      "domain_tags": ["tag1", "tag2"],
      "estimated_importance": 0.0-1.0,
      "reasoning": "why you think this is uncertain and why it matters"
    }}
  ]
}}

Generate {n_items} uncertainty items. Focus on areas where resolving the uncertainty could lead to genuine insight, not trivia."""


QUESTION_PROMPT = """You are a research engine. Given these areas of uncertainty in {domain}, generate focused research questions that could resolve or clarify them.

{focus_block}UNCERTAINTIES:
{uncertainties_json}

PRIORITIZATION RULES:
- Questions that span MULTIPLE uncertainty regions score highest (intersection = novelty potential)
- Questions that are investigable with web search and reasoning score higher than purely philosophical ones
- Questions that challenge conventional wisdom score higher than confirmatory ones
- Questions should be specific enough to investigate, not vague

Respond with EXACTLY this JSON structure (no other text):
{{
  "questions": [
    {{
      "question": "the specific research question",
      "source_uncertainty_indices": [0, 2],
      "priority_score": 0.0-1.0,
      "domain_tags": ["tag1", "tag2"],
      "investigability_notes": "how you would investigate this",
      "intersection_reasoning": "why this question bridges multiple uncertainties"
    }}
  ]
}}

Generate {n_questions} questions, ranked by priority_score."""


HYPOTHESIS_PROMPT = """You are the EXPLORER persona of a research engine in {domain}. Your role is exploration: open, divergent, committal — pick the most specific answer you can defend and stake out a position. The ASSESSOR (a separate downstream stage) will do the evaluation; right now you are NOT evaluating, you are committing.

QUESTION: {question}

This step exists SOLELY so we can measure surprise later. The downstream assessor compares investigation findings against the hypothesis you commit to here. If you hedge, the surprise signal degenerates. So: do not hedge, do not say "it depends", do not pre-anticipate the assessor's review. Pick the most specific answer you can defend from what you currently believe, and commit to it. Even if the answer feels uncertain, name your single best guess and assign your honest pre-evidence confidence to it.

Respond with EXACTLY this JSON structure (no other text):
{{
  "hypothesis": "your specific, committal pre-investigation answer",
  "confidence_before": 0.0-1.0,
  "reasoning": "why you believe this (brief)",
  "what_would_change_your_mind": "specific finding that would falsify this hypothesis"
}}"""


INVESTIGATE_PROMPT = """You are a research engine investigating a question in {domain}.

{focus_block}QUESTION: {question}

INVESTIGATION NOTES: {investigability_notes}

AVAILABLE TOOLS (use them):
{tool_list}

Use multiple tools where appropriate. web_search gives general web results; academic_search hits Crossref/arXiv/Semantic Scholar for papers with DOIs and citation counts; web_fetch reads a specific URL; archive_access finds primary/historical sources; calculator runs exact math. **code_execution (when available) lets you actually RUN python** — use it to test claims, compute quantities cited in papers, simulate mechanisms, and sanity-check numerical arguments. Prefer primary sources over summaries, and prefer runnable evidence over verbal description.

**Tool budget**: aim for roughly 8-15 tool calls for this investigation, then conclude with your final JSON. A thorough investigation is better than an exhaustive one.

Focus on:
- What does current research say?
- What are the competing perspectives?
- What evidence exists on different sides?
- Are there adjacent fields with relevant findings?

After investigating, respond with EXACTLY this JSON structure (no other text):
{{
  "methodology": "how you investigated (which tools, which searches, which sources you pulled)",
  "raw_findings": "detailed findings (be thorough, cite specific sources where possible)",
  "sources": ["urls, paper titles, doi:... / arXiv:... identifiers, researchers, or concepts referenced"],
  "key_takeaways": ["distilled takeaway 1", "distilled takeaway 2"]
}}"""


SURPRISE_PROMPT = """You are the ASSESSOR persona of a research engine. Your role is evaluation: closed, evidence-grounded, third-party. The EXPLORER (a separate prior stage) committed to a hypothesis BEFORE the investigation ran. Your job is to compare what the investigation found against what the explorer predicted, and assign an honest surprise score.

You are NOT the explorer. You did not write the hypothesis. Treat it as an external claim you are reviewing — a prediction made by someone else that you now have evidence for or against. This separation is structural: when the explorer and assessor are the same persona, surprise scores systematically inflate to flatter the prior. Resist that.

QUESTION: {question}

PRE-INVESTIGATION HYPOTHESIS (committed by the explorer, BEFORE evidence arrived):
{hypothesis_json}

INVESTIGATION FINDINGS (evidence the investigation phase actually surfaced):
{findings_json}

Your task: compare the findings to the hypothesis and produce an honest surprise assessment. A high surprise_delta means the findings diverged significantly from what the explorer predicted. A low surprise_delta means the investigation confirmed the prior. Be honest — the value of this system depends on calibrated surprise signals, not flattering ones. If the explorer's hypothesis was substantially right, say so plainly; if it missed in important ways, say that too.

**CONFIDENCE CALIBRATION RULES (follow these — the system uses confidence_after downstream to weight insights):**

Let C₀ = confidence_before, C₁ = confidence_after. Set C₁ according to the joint verdict × surprise_delta:
- verdict="confirmed" (fully): C₁ ≥ C₀; modest bump only. A confirmed low-surprise result may still cap at ~+0.05–+0.10.
- verdict="partially_confirmed":
  - If surprise_delta < 0.2: prior was roughly right but oversimplified — C₁ ≤ C₀. Do NOT increase confidence. Slight decrease (−0.05 to −0.10) is typical.
  - If surprise_delta ≥ 0.2: prior had meaningful gaps — C₁ < C₀ clearly (usually −0.15 to −0.35).
- verdict="contradicted": C₁ must be substantially below C₀ (usually 0.2–0.5 below, floored at 0).
- verdict="unresolved": pull C₁ toward 0.5 (epistemic humility). If C₀ was far from 0.5, move it at least halfway toward 0.5.

Violating these rules produces over-confident register entries downstream, so enforce them strictly.

Respond with EXACTLY this JSON structure (no other text):
{{
  "surprise_delta": 0.0-1.0,
  "confidence_after": 0.0-1.0,
  "surprise_explanation": "what specifically diverged from the hypothesis (or what confirmed it)",
  "hypothesis_verdict": "confirmed|partially_confirmed|contradicted|unresolved",
  "new_questions": ["question that emerged from the investigation"]
}}"""


CROSS_REFERENCE_PROMPT = """You are a research engine performing cross-referential analysis across multiple journal entries. Your goal is to find NON-OBVIOUS connections between findings.

{focus_block}JOURNAL ENTRIES:
{entries_json}

EXISTING CROSS-REFERENCES (already identified in prior cycles — DO NOT rediscover these):
{existing_xrefs_json}

Your task: Look across ALL entries for connections that are NOT already covered by the existing cross-references above. If your candidate connection restates or narrowly extends an existing one, SKIP it. Hunt for genuinely new angles — different domain pairings, different connection types, different implications. Return an empty list if no genuinely new connection meets the novelty bar.

**ANTI-ATTRACTOR RULE — avoid re-convergence on the same entries.**
If a candidate connection's source_entry_ids overlap ≥50% with the source_entry_ids of ANY existing cross-reference above (i.e. same or nearly-same participant set), you must either:
  (a) skip the candidate entirely, OR
  (b) if you must include it, reduce its novelty_score by at least 0.3 and justify in `reasoning` why the new angle is genuinely different despite the participant overlap.
The prior run showed the same 3-4 entries keep being re-cross-referenced into near-duplicate insights. Prefer connections that pull in entries that have NOT been cross-referenced together before.

Look across ALL entries for:
1. PATTERNS: Multiple entries that independently point toward a common underlying principle
2. CONTRADICTIONS: Findings from different investigations that conflict with each other
3. CONVERGENCE: Different questions that led to surprisingly similar answers
4. IMPLICATIONS: Finding X combined with finding Y implies something neither states alone

CRITICAL: The most valuable connections are between entries with DISSIMILAR domain tags. Same-domain connections are expected; cross-domain connections are where novelty lives.

Respond with EXACTLY this JSON structure (no other text):
{{
  "cross_references": [
    {{
      "source_entry_ids": ["id1", "id2"],
      "connection_type": "pattern|contradiction|convergence|implication",
      "description": "what the connection is (be specific)",
      "novelty_score": 0.0-1.0,
      "implications": ["what this connection might mean"],
      "suggested_questions": ["new questions this connection raises"],
      "reasoning": "why this connection is non-obvious and potentially important"
    }}
  ]
}}

Only report connections with novelty_score >= 0.5. Quality over quantity."""


VERIFY_PROMPT = """You are a SKEPTICAL REVIEWER verifying an insight produced by a research engine. Your job is to challenge this insight rigorously using fresh research. The engine will only enter this insight into its permanent register if your verification passes. Be adversarial — a flattering review does no one any favours.

INSIGHT UNDER REVIEW:
{insight_json}

SUPPORTING CROSS-REFERENCE:
{xref_json}

SUPPORTING JOURNAL ENTRIES (slim view):
{supporting_entries_json}

AVAILABLE TOOLS (use them adversarially):
{tool_list}

PRIOR HUMAN REJECTIONS (patterns to avoid repeating — the reasons a domain expert rejected previous register entries):
{prior_human_rejections_json}

If the candidate insight has the same weakness as any of these prior rejections, apply the same skepticism and reflect it in your verdict.

KNOWN PRIOR ART anchors for this journal's domain (human-curated — peers the verifier MUST evaluate explicitly, not hope to surface via search):
{known_prior_art_json}

CANONICALIZATION CONTEXT (Stage 1 of the three-stage verifier already extracted a structured form of the central architectural move; if Stage 2's alias-gap detector flagged a soft-aliased peer in the existing register, it appears here too):
{canonical_form_context}

For EACH known prior art anchor listed above:
1. Determine whether it is a peer to THIS specific claim (the claim's target_application_domain may be narrower than the journal's domain).
2. If it IS a peer, evaluate whether it substantively overlaps with the insight. If it does, the claim is at most an extension (or refuted, if the overlap is total).
3. Record your evaluation in `known_prior_art_evaluations` — one entry per anchor covering `anchor_id`, `is_peer` (bool), `overlaps_claim` (bool), `differentiators` (list), `reasoning` (one sentence).

A missing known_prior_art_evaluations entry for a listed anchor is a verification failure. Do not skip them.

**Tool budget**: aim for roughly 12-20 tool calls, then render your verdict. Efficient refutation beats exhaustive search. Prior searches can silently miss well-known systems when the query shape is wrong — iterate, don't one-shot.

**CRITICAL — novelty framing**
A *genuinely novel synthesis* has a specific fingerprint: its individual premises (ingredients) are each established in the literature, AND the central architectural move at its core is not already deployed in a published system under any name. "New combination of known parts" is ONLY novel when the combination itself, or the central move it embodies, is not already published.

Three disqualifiers, in decreasing severity:
  • *restatement* — full synthesis is already published under some name. Disqualifies.
  • *central move published* — the headline architectural move (tournament ranking, process-reward ensemble, adversarial debate, etc.) already appears in a peer or competitor system, even if surrounding refinements differ. The claim is then an **extension**, not new_synthesis.
  • *covered by a closest peer system* — one deployed system already addresses the same problem with substantial architectural overlap. The claim is at most an **extension**.

Score the two axes INDEPENDENTLY:

A) **premises_supported**: Are the building blocks this insight depends on well-grounded in the literature? (TRUE is GOOD — ingredients are real.)
B) **synthesis_findable**: Is the specific synthesis / composite claim / architecture itself already published, patented, or widely-discussed under any name? (TRUE is BAD — "novel" claim is not novel.)

The ideal register entry has **premises_supported=TRUE and synthesis_findable=FALSE** AND no published peer system implementing the central move. That is the *signature* of genuine novelty, not a weakness.

**code_execution (when available) lets you actually RUN python** — use it to recompute numerical claims cited by the insight, test the reasoning against real data, or simulate the proposed mechanism.

═══════════════════════════════════════════════════════════════════
PRIOR-ART SEARCH — FOUR PHASES + FINAL SKEPTIC PROBE
═══════════════════════════════════════════════════════════════════

Past failure mode: the verifier composed all the claim's concepts into ONE compound query (e.g. "pairwise tournament + structured reasoning traces + worst-case aggregation + debate + novelty"), failed to find a paper matching ALL of them, and declared synthesis_findable=false. Meanwhile a peer system implementing 3 of the 5 concepts (the headline move + an auxiliary) existed in the literature and was trivially findable with a domain-anchored query. Do not repeat this.

Run each phase as an **agentic iteration**: formulate a query, READ the top hits, formulate follow-ups based on what you found, and follow any interesting lead all the way to the originating paper.

─── PHASE 0: PREMISES CHECK ───
For each ingredient the insight depends on, search for evidence the ingredient is real. Record in `premises_support_citations`. Set `premises_supported=true` iff each load-bearing premise has literature support.

─── PHASE 1: CENTRAL ARCHITECTURAL MOVE ───
Decompose the claim into its SINGLE headline move — the one architectural shift the insight proposes (e.g. "replace scalar scoring with tournament ranking in LLM research loops"). Write this down in `central_architectural_move`. Then search for prior work that already makes that move, IGNORING the auxiliary refinements. Record hits in `central_move_prior_art`. If you find a strong match, the novelty_type is AT MOST `extension` — the insight's contribution is the auxiliary refinements, not the central move itself.

─── PHASE 2: FULL COMPOSITE PRIOR ART ───
Now search for prior work matching the whole composite claim — all the refinements combined. Record in `synthesis_prior_art`. Set `synthesis_findable=true` only if substantive prior work makes essentially the same composite claim.

─── PHASE 3a: FUNCTIONAL DIMENSION DECOMPOSITION ───
Break the claim into 3-5 independent functional dimensions — e.g. *what it acts on / through what mechanism / at what scale / under what constraint / on what substrate*. For each dimension, name the NEAREST published exemplar and how the insight differs along that dimension. Record in `functional_decomposition`. If any dimension has an exemplar that's *indistinguishable* from the claim along that dimension, treat it as partial prior art.

─── PHASE 3b: CLOSEST PEER SYSTEM ───
Explicitly search for the closest *complete* peer system — a deployed or published system that addresses the same problem the insight addresses, even if its internals differ. This is the phase that catches system-level competitors that structural-query searches miss.

**Before searching in this phase**: write out the claim's target application domain in one short phrase, derived from the claim text itself — NOT from any prior example. The journal's overall domain context is: "{engine_domain}". Use it as a scoping hint; if the specific claim targets a narrower or orthogonal sub-domain, name that narrower thing instead. Put this phrase in `target_application_domain`.

**At least one query in this phase MUST explicitly include that domain phrase** in the query string. If the domain is "marketing attribution systems", at least one query reads like "marketing attribution systems <concept>". If the domain is "distributed systems consensus", at least one query reads like "distributed systems consensus <concept>". Do NOT default to any fixed example domain from training data — the domain is whatever your target_application_domain phrase says. Record the closest peer in `closest_peer_system` with overlap summary and concrete differentiators.

─── PHASE 4: CONTRADICTING EVIDENCE ───
Search for research that actively disagrees with the composite claim. Record in `contradicting_findings`.

─── PHASE 5: REASONING AUDIT ───
Examine the connection between supporting entries and the claim. Is the inferential leap justified? Record in `reasoning_flaws`.

─── FINAL: SKEPTIC SMELL TEST ───
Before writing the verdict, run an aggressive adversarial probe. You are a skeptical peer reviewer with 10 minutes and web search, whose JOB is to kill the insight. Follow this sequence:

1. **Enumerate**: write out 3-5 candidate "kill queries" — the queries most likely to surface disqualifying prior art, restatement, or a peer system that does essentially the same thing. Be adversarial: choose queries a hostile reviewer would run. Record in `skeptic_probe.candidate_queries`.
2. **Pick the most lethal** — the candidate with the best chance of returning disqualifying evidence, not the one you're most confident will survive. Record in `skeptic_probe.query`.
3. **Run it.** Record `skeptic_probe.top_result_summary`.
4. **Judge honestly**: would the top result convince a hostile reviewer that the claim is a restatement or covered by existing work? Set `skeptic_probe.disqualifies` accordingly.
5. **Survivor check**: if the first probe returns disqualifies=false, run ONE more query from your candidate list chosen to attack a DIFFERENT angle. Record that as `skeptic_probe.followup_query` and `skeptic_probe.followup_summary`. If either probe disqualifies, mark disqualifies=true.

The system will penalise the verdict if the skeptic probe survived with queries narrow enough that a reviewer would call them non-adversarial. Picking queries designed to confirm rather than kill is a hedging failure, not a verification pass.

═══════════════════════════════════════════════════════════════════

Classify the insight's `novelty_type`:
- "new_synthesis": premises established, composite claim not in literature, **central architectural move is also not in literature** → GENUINE novelty candidate
- "restatement": composite claim already in literature under some name → NOT novel
- "extension": central architectural move is published in some peer system, but the insight's refinements/combination add something new → marginal novelty
- "correction": challenges a published claim with new reasoning/evidence → novel critique
- "unsupported": premises themselves are shaky → reject regardless of synthesis

Render a verdict:
- "validated": premises_supported=TRUE, synthesis_findable=FALSE, no decisive contradictions, reasoning holds. A new_synthesis, correction, or strong extension.
- "challenged": one condition holds weakly — e.g. premises partially supported, synthesis partially findable, reasoning has fixable gaps. Use this SPARINGLY and ONLY with a specific, nameable weakness — not because "the ingredients existed separately" (that is the definition of new_synthesis, not a weakness).
- "refuted": premises_supported=FALSE, or synthesis_findable=TRUE, or decisive contradictions, or fatal reasoning flaw.
- "inconclusive": your adversarial search could not reach the claim, but nothing you found refuted it either. Use THIS verdict ONLY when one or more of these conditions hold:
    • Your searches returned no meaningful results for the specific synthesis AND the claim cannot be rephrased into something searchable.
    • The claim requires data or methods you cannot access (proprietary datasets, pre-publication work, clinical evidence, industrial telemetry, GPU-scale experiments, paywalled journals).
    • The claim is empirical and resolves only with an experiment you cannot run inside `code_execution`.
    • The claim sits in a field where public literature is genuinely thin (e.g. active research frontier with no surveys yet, or heterodox area with sparse coverage).
  When you return `inconclusive`, you MUST name the specific epistemic gap in `verification_summary` (e.g. "the claim hinges on unpublished benchmark results for model X, which are not in the public literature") — a summary that just hedges without naming the gap will be downgraded to `challenged` by the engine.

  Do NOT use `inconclusive` as a softer `challenged`. If you found specific weaknesses, those are `challenged`. If you found specific refutations, those are `refuted`. `inconclusive` means "search did not have purchase — this needs a settlement signal outside my reach."

**Guardrails against the "ingredients existed" failure mode:**
- If your justification for `challenged` or `refuted` amounts to "individual components are documented but this specific combination is not," that is evidence of novelty — revisit and likely upgrade to `validated`.
- Do NOT penalize an insight for being composed of published ingredients. Penalize only if the *full synthesis itself* is published, or if the reasoning from premises to synthesis is broken.

**Self-consistency check — read this BEFORE writing the verdict field:**
If you are about to write `"verdict": "challenged"` while ALSO setting:
  • `novelty_type` to `new_synthesis` or `correction`, AND
  • `premises_supported` to `true`, AND
  • `synthesis_findable` to `false`,
then STOP — those four values together ARE the signature of a genuine new synthesis (premises real, composite claim not in literature). A `challenged` verdict in that configuration is self-contradicting unless `reasoning_flaws` names a specific, substantive flaw that is NOT about ingredients existing separately.

If you cannot name such a substantive flaw (in `reasoning_flaws`), the correct verdict is `validated`. The engine will programmatically upgrade the verdict to `validated` if you return `challenged` in this configuration without a substantive flaw, and will log the upgrade — so hedging here costs you nothing and gives up a correct register entry.

Also provide MOTIVATION: if this insight holds up, WHY does it matter? What does it explain or unlock that wasn't clear before?

FINALLY — if and only if your verdict is "validated" — produce 1-3 FALSIFIABLE PREDICTIONS that follow from the insight. Each prediction must:
- Make a specific empirical claim that could be checked against reality.
- State a falsifiable_condition: exactly what observation would confirm or refute it.
- State a check_method: the realistic way to verify (e.g., "search arxiv/Google Scholar for papers testing X between today and target_date", "look for published benchmark results on dataset Y", "rerun analysis Z on released model weights").
- Have a target_date in ISO YYYY-MM-DD format, typically 3-24 months from today — when the prediction should be checkable.

A prediction that cannot be checked, or that is trivially true either way, is worse than no prediction. Be specific or produce none. If your verdict is "challenged" or "refuted", omit predictions (emit an empty list).

IF AND ONLY IF your verdict is "inconclusive" — emit a SETTLEMENT PLAN instead of predictions. The plan describes how reality (or future work) could eventually settle the claim:
- `settlement_method`: a concrete method — a paper to watch for, a benchmark release to check, a dataset/code release that would unlock verification, an industrial telemetry signal, a specific experiment that would answer it.
- `settlement_horizon`: ISO YYYY-MM-DD target by when the settlement signal might be checkable (3–36 months typical).
- `settlement_triggers`: 1–3 specific observable outcomes, each of which would either PROMOTE the held insight to active (confirming settlement) or REFUTE it. State each trigger in the "if X is observed, then Y" form.

If your verdict is "validated", emit predictions and omit settlement fields. If "inconclusive", emit settlement fields and omit predictions. If "challenged" or "refuted", omit both.

Respond with EXACTLY this JSON structure (no other text):
{{
  "premises_supported": true,
  "premises_support_citations": ["for each premise, one or more real citations/URLs establishing it"],
  "central_architectural_move": "one sentence naming the insight's single headline architectural move (what it proposes to replace or add, shorn of refinements)",
  "central_move_prior_art": ["citations/URLs of published work that already makes this headline move — empty list if none; a non-empty list forces novelty_type=extension unless the auxiliary refinements themselves constitute a second novel move"],
  "synthesis_findable": false,
  "synthesis_prior_art": ["candidates for the whole-composite-claim search; empty list if nothing close found"],
  "functional_decomposition": [
    {{
      "dimension": "what it acts on | mechanism | scale | constraint | substrate (or custom)",
      "nearest_exemplar": "name + URL of closest published exemplar for this dimension",
      "how_ours_differs": "concrete differentiator along this dimension"
    }}
  ],
  "target_application_domain": "one short phrase naming the claim's actual target domain — derived from the claim itself, not from memorized examples",
  "closest_peer_system": {{
    "name": "name of the closest complete peer system, or empty if none found",
    "url": "URL / citation",
    "overlap_summary": "1-2 sentence summary of what this peer system does that overlaps with the insight's claim",
    "differentiators": ["concrete ways the insight differs from or adds to this peer system"]
  }},
  "skeptic_probe": {{
    "candidate_queries": ["3-5 candidate kill queries a hostile reviewer would run"],
    "query": "the single most lethal query — the one most likely to surface disqualifying evidence",
    "top_result_summary": "1-2 sentence summary of the top hit",
    "followup_query": "one additional adversarial query attacking a different angle (required if first query returned disqualifies=false)",
    "followup_summary": "1-2 sentence summary of the followup's top hit, or empty if no followup was needed",
    "disqualifies": false
  }},
  "novelty_type": "new_synthesis|restatement|extension|correction|unsupported",
  "known_prior_art_evaluations": [
    {{
      "anchor_id": "id of the known-prior-art anchor you evaluated",
      "is_peer": false,
      "overlaps_claim": false,
      "differentiators": ["concrete ways the claim differs from this anchor — empty if is_peer=false or overlaps_claim=false"],
      "reasoning": "one sentence explaining the evaluation"
    }}
  ],
  "contradicting_findings": ["specific findings or papers that contradict the composite claim"],
  "reasoning_flaws": ["specific flaws in the evidence-to-claim connection"],
  "motivation": "one-paragraph explanation of why this insight matters if validated",
  "verdict": "validated|challenged|refuted|inconclusive",
  "verified_confidence": 0.0-1.0,
  "verification_summary": "one-paragraph justification citing what the adversarial search found or failed to find, and WHY the verdict follows from premises_supported × synthesis_findable × central_move_prior_art × closest_peer_system × skeptic_probe; for `inconclusive`, explicitly name the epistemic gap that prevented verification",
  "predictions": [
    {{
      "claim": "specific empirical prediction that follows from the insight",
      "falsifiable_condition": "exact observable outcome that would confirm or refute it",
      "check_method": "concrete method for verifying the prediction",
      "target_date": "YYYY-MM-DD"
    }}
  ],
  "settlement_method": "only when verdict is inconclusive — concrete method for eventually settling the claim",
  "settlement_horizon": "YYYY-MM-DD",
  "settlement_triggers": ["observable outcomes that would promote or refute the held claim"]
}}"""


ANALOG_PROBE_PROMPT = """You are a research engine looking for CROSS-DOMAIN ANALOGS of a finding you just uncovered. The best novel ideas often come from recognising that *the same structural mechanism* — same dynamics, same constraints, same failure modes — appears in a field far from the one you're working in, under a different vocabulary.

THE FINDING:
Question investigated: {entry_question}
Surprise delta: {entry_surprise:.2f}
Key takeaways:
{entry_takeaways}

CURRENT RESEARCH DOMAIN: {engine_domain}

DOMAINS ALREADY IN PLAY ON THIS JOURNAL (avoid these — they're NOT distant enough):
{recent_tags}

Your task: identify 2-3 DISTANT domains — fields whose established vocabulary, methods, or mechanisms are structurally analogous to the finding but that do NOT appear in the recent-tags list above. Then, for each, produce a specific, investigable question that translates the analog back into the current research domain.

**"Distant" means a field that a domain expert in {engine_domain} would NOT already routinely consult.** Do not default to a particular off-the-shelf repertoire of cross-domain bridges; your candidate pool is every structured field of knowledge humans have developed, and the best analog is whichever one the finding's structural fingerprint actually matches.

**What counts as a strong analog (not a weak one):**
- STRONG: a specific named mechanism/law/formal result from the analog domain whose dynamics-constraints-failure-modes map onto the finding with enough precision that a concrete testable question falls out.
- WEAK: "this is like <broad field>". Vague. Name the sub-field AND the specific mechanism.
- WEAK: an analog drawn from a field adjacent to {engine_domain}. The whole point is reach; the analog should feel non-obvious.

**Rules:**
- Each analog must be STRUCTURALLY analogous (same dynamics, same constraints, same failure modes), not merely topically related.
- Name a specific mechanism, law, method, or formal result — not just the parent field.
- The investigable question must be answerable with web_search / academic_search — not purely philosophical.
- If you cannot find a STRONG cross-domain analog, return an empty list. Weak analogs pollute the queue. Quality over quantity.

Respond with EXACTLY this JSON structure (no other text):
{{
  "analogs": [
    {{
      "domain": "specific sub-field naming a body of literature and vocabulary distinct from the current research domain",
      "mechanism": "the specific mechanism, law, method, or named result in that domain",
      "structural_analogy": "one sentence explaining which dynamics/constraints/failure-modes map between the finding and this analog",
      "question": "an investigable question that translates the analog into the current research domain"
    }}
  ]
}}"""


NEGATIVE_SPACE_EXTRACT_PROMPT = """You are a research engine performing STRUCTURAL ANALYSIS of a journal's coverage. Before we can identify gaps, we need to build a (method × problem) matrix describing what the journal has already studied.

{focus_block}

JOURNAL ENTRIES (slimmed — question, key_takeaways, domain_tags):
{entries_json}

EXISTING TAG ANCHORS (domain_tags observed across entries; use these as the skeleton):
{tag_anchors}

Your task: extract the canonical (method, problem) pairs from these entries.

- "Method" = the technique, approach, mechanism, architecture, or analytical lens the entry relies on.
  Examples: "ensemble disagreement", "sparse autoencoders", "neurosymbolic verification", "RAG novelty-checking", "prover-verifier games".
- "Problem" = the specific question, goal, failure mode, or phenomenon being addressed.
  Examples: "escaping self-grading", "detecting OOD concepts", "verifying pre-formal ideas", "measuring conceptual novelty".

Rules:
- Use tag_anchors where they're clearly method-like or problem-like, but also extract richer phrases from key_takeaways when they're not captured by tags.
- Keep each method and each problem to 2-6 words. Concrete and specific.
- Prefer 5-12 distinct methods and 5-12 distinct problems; aggressive consolidation beats a sprawling matrix.
- Multiple entries may share the same (method, problem) pair — that's fine, list them all in entry_ids.

Respond with EXACTLY this JSON structure (no other text):
{{
  "methods": ["method 1", "method 2", ...],
  "problems": ["problem 1", "problem 2", ...],
  "cells": [
    {{
      "method": "...",
      "problem": "...",
      "entry_ids": ["j-xxx", "j-yyy"]
    }}
  ]
}}"""


NEGATIVE_SPACE_CLASSIFY_PROMPT = """You are a research engine analyzing EMPTY CELLS in a journal's (method × problem) coverage matrix. Each empty cell is a method × problem combination that the journal HAS NOT investigated. Your job is to classify each gap so the engine knows which are worth pursuing.

{focus_block}

MATRIX CONTEXT (what the journal HAS covered):
Methods: {methods_json}
Problems: {problems_json}
Covered cells (method × problem pairs already studied in this journal):
{covered_cells_json}

EMPTY CELLS (combinations with no entries in this journal):
{empty_cells_json}

For each empty cell, classify it into ONE category:

1. **underexplored** — the combination is genuinely interesting and under-studied. Applying this method to this problem is plausible, potentially valuable, and the field hasn't given it serious attention. This is the HIGH-VALUE category — these become new research questions.

2. **tried_failed** — there's known evidence that this combination has been tried and didn't work (or was explicitly ruled out). Has prior art showing why the combination fails.

3. **trivially_uninteresting** — the combination is technically possible but doesn't make sense in the field under analysis. The method has no load-bearing purchase on the problem; pairing them would be a category error or a rephrasing of a solved thing — not a gap.

4. **regulated_boundary** — the combination is bounded by ethics, regulation, capability constraints, or infrastructure limits. The gap isn't intellectual; it's that the combination cannot be safely, legally, or technically attempted in the current environment.

5. **adjacent_but_covered** — the combination LOOKS empty in this journal's matrix but is actually well-studied in the wider literature under a different terminology. The journal just hasn't absorbed that literature yet.

Rules:
- Be honest about what you DON'T know. If you're uncertain between underexplored and adjacent_but_covered, classify as underexplored and let the verification search settle it.
- For underexplored cells, produce a specific reason ("X problem has these 3 known properties Y that make method Z plausibly effective but no paper connects them").
- Bias toward marking cells as underexplored if unsure — the verification step will filter out false positives via academic_search.

Respond with EXACTLY this JSON structure (no other text):
{{
  "classified_cells": [
    {{
      "method": "...",
      "problem": "...",
      "classification": "underexplored|tried_failed|trivially_uninteresting|regulated_boundary|adjacent_but_covered",
      "reasoning": "one-sentence explanation of why this classification fits",
      "verification_search_queries": ["academic-search query 1", "academic-search query 2"]
    }}
  ]
}}"""


NEGATIVE_SPACE_QUESTIONS_PROMPT = """You are a research engine generating INVESTIGABLE QUESTIONS for verified-empty cells in a method × problem coverage matrix.

{focus_block}

VERIFIED GAPS (method × problem combinations that classified as `underexplored` AND whose `academic_search` verification returned few or no relevant results):
{verified_gaps_json}

For each verified gap, produce 1-2 investigable research questions that would actually attempt the combination or probe why it's empty. Each question should:

- Be concrete and specific — name the method and problem, not just gesture at them.
- Be investigable with web_search / academic_search / code_execution — not pure philosophy.
- Target the GAP, not the surrounding territory. If the gap is "applying X to Y", the question should be something like "Does X address Y's specific failure modes A and B?" — not "What does X do in general?"

Respond with EXACTLY this JSON structure (no other text):
{{
  "gap_questions": [
    {{
      "method": "...",
      "problem": "...",
      "questions": ["question 1 probing this gap", "question 2 probing this gap"]
    }}
  ]
}}"""


ASSUMPTION_PROBE_PROMPT = """You are a research engine surfacing IMPLICIT ASSUMPTIONS in an established finding. Every accepted claim rests on unstated premises — things the field takes for granted without re-examining. The most generative novelty move within a domain is to name those assumptions and ask: what if each one were false?

This probe fires on LOW-surprise CONFIRMED findings — those are where field consensus most likely hides load-bearing assumptions. A finding that confirmed a widely-held hypothesis with little surprise is a signal that the premise layer is unexamined.

THE FINDING (confirmed; low surprise — field consensus regime):
Question investigated: {entry_question}
Hypothesis verdict: {entry_verdict}
Surprise delta: {entry_surprise:.2f}
Key takeaways:
{entry_takeaways}

Your task: list 3-5 IMPLICIT ASSUMPTIONS this finding depends on — the things every practitioner in the field treats as obviously true. For each, produce an INVESTIGABLE question that would test whether the assumption actually holds.

**What counts as a STRONG assumption probe:**
- STRONG: *"The finding assumes monotonic scaling — the community-wide practice of extrapolating from small to large models. Question: in the compute regime where phase transitions are documented (Ganguli et al. 2022), does monotonic extrapolation still hold?"*
- STRONG: *"The finding assumes the evaluation set is i.i.d. with the training distribution. In practice benchmarks are often contaminated with training data. Question: how does the result change on strictly held-out benchmarks released AFTER the model was trained?"*
- WEAK: *"This assumes the training data is representative."* (Trivial; every practitioner says this.)
- WEAK: *"This assumes the hardware is deterministic."* (Known caveat; not a novelty move.)

**Rules:**
- Each assumption must be SPECIFIC — a concrete premise, not a generic epistemological caveat.
- The assumption should be one practitioners TAKE FOR GRANTED — not a known open question the field is already arguing about.
- The investigable question must be answerable with web_search / academic_search / code_execution — not pure philosophy.
- Prefer assumptions whose negation would be SURPRISING if true. "What if the activation function is actually not important for generalization" beats "what if the model is overparameterized" (well-studied).
- If no strong, generative assumption-negation is available, return an empty list. Weak probes pollute the queue; quality over quantity.

Respond with EXACTLY this JSON structure (no other text):
{{
  "assumptions": [
    {{
      "assumption": "specific unstated premise the field takes for granted",
      "why_taken_for_granted": "one-sentence on why nobody in the field re-examines this",
      "implication_if_false": "what would materially change in the field if the assumption turned out to be false",
      "question": "investigable research question that would test whether the assumption holds"
    }}
  ]
}}"""


PREDICTION_CHECK_PROMPT = """You are auditing a previously-registered prediction to see whether reality has caught up with it.

REGISTERED INSIGHT:
{insight_title}
{insight_description}

PREDICTION UNDER CHECK:
  Claim:                   {claim}
  Falsifiable condition:   {falsifiable_condition}
  Check method:            {check_method}
  Originally registered:   {created_at}
  Target check date:       {target_date}
  Today's date:            {today}

AVAILABLE TOOLS:
{tool_list}

Your task:

1. Search for evidence that speaks to this prediction's falsifiable_condition. Follow the check_method as a starting point, but expand as needed. Use academic_search for papers, web_search for general web, web_fetch for specific URLs, archive_access for primary sources.
2. Consider both confirming and contradicting evidence. Don't anchor on the claim being true.
3. If evidence is genuinely unclear or the horizon hasn't been reached in practice, say so — "inconclusive" is a legitimate verdict.

Render one of four verdicts:
- "confirmed": clear evidence the falsifiable_condition resolved in the direction the claim predicts.
- "refuted":   clear evidence the falsifiable_condition resolved against the claim.
- "inconclusive": search did not find strong evidence either way (yet).
- "expired":   the horizon has passed and the question is no longer testable in its original form.

Respond with EXACTLY this JSON structure (no other text):
{{
  "verdict": "confirmed|refuted|inconclusive|expired",
  "reasoning": "one-paragraph justification citing what you found",
  "sources": ["urls, papers, or specific evidence referenced"]
}}"""


SYNTHESIZE_PROMPT = """You are a research engine synthesizing a novel insight from cross-referenced findings.

{focus_block}CROSS-REFERENCE:
{xref_json}

SUPPORTING JOURNAL ENTRIES:
{supporting_entries_json}

Your task: Articulate this connection as a fully-formed insight. Be specific, be bold, but also be honest about limitations.

CRITICAL SELF-CHECK (prior_art_check): Before claiming novelty, honestly assess whether this insight is already well-established in your training data. Ask yourself: "If I searched for this claim, would I find it stated plainly in textbooks, surveys, or well-known papers?" A "novel" insight that merely restates common knowledge is worse than no insight — it poisons the signal. If the insight is well-established, say so in prior_art_check and lower confidence accordingly. Genuine novelty usually lives in the UNEXPECTED CONNECTION between two findings, not in either finding alone.

Respond with EXACTLY this JSON structure (no other text):
{{
  "title": "concise statement of the insight (one sentence)",
  "description": "full articulation of the insight (2-3 paragraphs)",
  "novelty_assessment": "why you believe this is genuinely novel, not just restating known information",
  "prior_art_check": "honest assessment: is the core claim already well-established? If yes, where? If no, what makes this connection genuinely novel?",
  "confidence": 0.0-1.0,
  "implications": ["concrete implication 1", "concrete implication 2"],
  "open_questions": ["what would need to be investigated to validate this"],
  "counter_arguments": ["why this might be wrong"]
}}"""


DIRECTIVE_HYPOTHESIS_PROMPT = """You are composing one section of a RESEARCH DIRECTIVE.

A research directive is a plan a research team executes to take a verified concept from idea to a publishable result. The team runs experiments, gathers data, and produces measurements. The directive is NOT a literature-watch list waiting on other researchers to publish.

This call writes the HYPOTHESIS section only — what the team's own experiments will show if the theory holds.

ENGINE DOMAIN: {engine_domain}
REGISTER ENTRY UNDER TRANSLATION:
{register_entry_json}
ATTACHED OPEN PREDICTIONS (claims already tied to this entry):
{predictions_json}

Your job: state what the EXECUTING TEAM would measure or observe FROM THEIR OWN EXPERIMENTS if the theory were correct. Two to three sentences. No preamble, no section header — just the prose.

Rules:
- The observable is something the team produces from work they do — a measurement on data they collect, a score from a model they train, a state transition they trigger, a comparison they run. NOT "by date X a published paper reports Y" or "the field's benchmark releases show Z" — those are predictions about other researchers, not statements about what the team observes.
- If an attached prediction is phrased as a literature-watch ("by 2027 a paper reports …"), translate it into the equivalent thing the team would measure if they ran the experiment themselves.
- Derive the observable from the claim itself. Do not invent domain-specific details the source entry does not support.
- No examples from any specific field. Shape, not content.

Respond with EXACTLY this JSON structure (no other text):
{{
  "hypothesis": "2-3 sentence statement of what the team measures from their own experiments if the theory holds"
}}"""


DIRECTIVE_TEST_PLAN_PROMPT = """You are composing the TEST PLAN section of a RESEARCH DIRECTIVE.

The directive is a plan the research team executes to take a verified concept toward a publishable result. The test plan is the EXPERIMENTAL PROGRAM the team runs themselves — data collection, measurement, analysis, and comparison. It is NOT a list of literatures to watch for other groups' announcements.

ENGINE DOMAIN: {engine_domain}
REGISTER ENTRY UNDER TRANSLATION:
{register_entry_json}
ATTACHED OPEN PREDICTIONS (with their `check_method` fields — translate these into experiments the team runs themselves):
{predictions_json}
HYPOTHESIS FROM PRIOR SECTION (for coherence; do not restate):
{hypothesis}

Your job: write 3–6 numbered steps that together form the team's experimental program. Each step must have:
- A clear INPUT (what the team starts with — existing data they have or can collect, a source they will fetch, a prior step's output)
- A concrete ACTION (what the team does — collect, measure, compute, compare, train, run)
- An OBSERVABLE OUTPUT (what the team ends up with — a dataset, a score, a measurement, a chart, a comparison table)

Rules:
- The team is the actor. Steps describe THE TEAM's experiments, NOT predictions about what other researchers may publish. If a prediction's `check_method` is phrased as a literature-watch ("search for papers on X by 2027"), translate it: write the experiment the TEAM would run that produces the same kind of evidence directly. Only fall back to a literature-monitoring sub-step if running the experiment in-house is genuinely out of reach (e.g. requires hardware the team cannot access) — and even then, the literature-watch is supplementary to the team's own work.
- No hand-wave phrasing. Phrases like "figure out", "iterate until it works", "try various approaches", "check relevant sources", "use an appropriate method" are forbidden.
- Every step must be concretely executable by the team with access to standard research tools and ordinary research-lab capabilities.
- Do NOT drop predictions silently. Each open prediction must have a corresponding step or sub-step.
- Do NOT name specific tools in this section — the Agentic Prompt section handles tool binding. Here, describe actions at the level of "compute X from dataset Y", "fit model Z and report metric W", etc.
- Do NOT reference citations by URL here — the References section aggregates those.
- No examples from any specific field. Describe structure, not content.

Respond with EXACTLY this JSON structure (no other text):
{{
  "steps": [
    {{"n": 1, "input": "...", "action": "...", "output": "..."}},
    {{"n": 2, "input": "...", "action": "...", "output": "..."}}
  ]
}}"""


DIRECTIVE_AGENTIC_PROMPT_PROMPT = """You are composing the AGENTIC PROMPT section of a research directive. The directive is a plan the team executes to take a verified concept toward a publishable result; this section AUTOMATES THE LEGWORK of the test plan — literature scans, data fetches, code execution, citation tracking — so the team can focus on the experimental and analytical parts a human must drive. The output of this call is what a human pastes into an LLM-driven agent (e.g. Claude Code, an MCP orchestrator). Fabrication here sends a researcher chasing ghosts. Grounding is non-negotiable.

**Return STRUCTURED FIELDS — not free-form prose.** The system will render your structured output into the final agentic prompt deterministically. This keeps your response small and checkable.

ENGINE DOMAIN: {engine_domain}
REGISTER ENTRY UNDER TRANSLATION:
{register_entry_json}
HYPOTHESIS (from prior section):
{hypothesis}
TEST PLAN STEPS (from prior section):
{test_plan_json}

CITATIONS ALLOWLIST (every URL / DOI / identifier you reference MUST appear verbatim in this list):
{citations_json}

AGENT TOOL ALLOWLIST (every tool you name MUST exact-string-match one of these):
{tool_allowlist_json}

Rules:
- Every `tool_call` string you emit must START with an exact allowlist name followed by `(...)`. No generic phrasings. No prose references to "search engine" / "an agent tool" / "a web crawler".
- Every URL / DOI in any `input`, `tool_call`, or `expected_output` field must appear verbatim in the CITATIONS ALLOWLIST. If a required input is not in the allowlist, put the string `"UNRESOLVED: <what's needed>"` in its place AND add the same item to `unresolved_dependencies`. Do not invent plausible URLs.
- No hand-wave language in any field. Phrases like "figure out", "iterate until", "try various", "check relevant sources" are forbidden.
- Halt checkpoints are where a human should review before the agent proceeds. Use them sparingly — typically 1-2 per directive.
- Stop conditions must be objectively measurable — numerical threshold, specific output pattern, citation count, dataset match. Derive them from the claim itself.

Respond with EXACTLY this JSON structure (no other text):
{{
  "inputs": ["short descriptions of the starting inputs the agent needs (files, URLs, datasets). Each item in 3-15 words."],
  "setup_preamble": "1-2 sentence context the agent reads before starting work. What it is doing and why.",
  "steps": [
    {{
      "n": 1,
      "action": "one sentence stating what the agent does at this step",
      "tool_call": "exact tool invocation, e.g. `web_search(query='...')` or `academic_search(query='...', sources=['arxiv'])`. Must start with an allowlist tool name.",
      "expected_output": "what the agent should end up with after this step — a file, a count, a state, a specific observable",
      "halt_after": "empty string OR a short reason to halt for human review after this step"
    }}
  ],
  "output_spec": "where the final outputs land and in what format — a file path, a report structure, a data schema",
  "stop_conditions": {{
    "success": "concrete measurable signal that the test confirms the hypothesis",
    "failure": "concrete measurable signal that the test refutes the hypothesis",
    "inconclusive": "concrete condition under which the test did not resolve"
  }},
  "tool_names_used": ["exact allowlist names that appear in any step's tool_call"],
  "citations_used": ["ONLY citations that appear inside the agentic-prompt structured fields above (inputs, setup_preamble, steps[].action/tool_call/expected_output, output_spec, stop_conditions). Other directive sections may freely cite from the allowlist; do NOT enumerate those here."],
  "unresolved_dependencies": ["items marked UNRESOLVED: descriptions of what was needed but was not in the allowlists; empty if fully grounded"]
}}

Keep each `action` under 25 words. Keep each `tool_call` under 200 characters. Keep each `expected_output` under 30 words. These caps are not stylistic — they are budget caps that keep this response within the model's practical non-streaming return window."""


DIRECTIVE_VERIFICATION_CRITERIA_PROMPT = """You are composing the VERIFICATION CRITERIA section of a research directive — a 3-row table mapping test outcomes to concrete observable signals.

The directive is a plan the team executes to take a verified concept toward a publishable result. Verification criteria are signals the team MEASURES FROM THEIR OWN EXPERIMENTAL OUTPUTS — the artifacts produced by running the Test Plan. Criteria are NOT predictions about what other researchers will publish or what benchmarks the field will release.

ENGINE DOMAIN: {engine_domain}
REGISTER ENTRY UNDER TRANSLATION:
{register_entry_json}
ATTACHED OPEN PREDICTIONS (with falsifiable_condition fields — translate these into measurements of the team's own outputs):
{predictions_json}
HYPOTHESIS (from prior section):
{hypothesis}

Write three rows: Confirmed, Refuted, Inconclusive. Each row's signal must be:
- OBJECTIVELY measurable from the team's own experimental outputs — a numerical threshold computed on data the team produces, a specific pattern in the team's measurements, a comparison ratio between two configurations the team runs, a state transition the team triggers. Not "the approach demonstrates utility", not "evidence supports the claim", not "the field has converged on X".
- FORBIDDEN: criteria that depend on external publications, third-party benchmarks, or other research groups' outputs appearing by some date. Phrases like "by [date], at least one publicly accessible benchmark paper or code release reports …", "a published study confirms …", "the literature shows …", "by the target date, a comparative analysis appears …" are LITERATURE-WATCH phrasings — they predict what other researchers do, not what the executing team measures. Translate any such prediction's `falsifiable_condition` into the equivalent measurement of the team's OWN experimental output. Only if the experiment is genuinely beyond the team's reach (specialised hardware unavailable, etc.) may a criterion fall back on external evidence — and even then it should reference a SPECIFIC named source the team will inspect, not a generic future publication.
- Directly derivable from the predictions' falsifiable_conditions (after translation) OR the hypothesis. No invented criteria.
- Field-agnostic in shape — do not use examples from any specific research domain.

Rules:
- Each signal stands alone. A reader should be able to say "that signal either is present in the team's measurements or it isn't" without interpretation.
- Inconclusive describes the state where the team's experiment could not resolve — e.g. "required dataset cannot be obtained", "the team's compute budget did not permit running the full comparison", "the measurement noise floor exceeds the predicted effect size". NOT a weak version of confirmed/refuted ("mixed evidence") and NOT a "no paper appeared" outcome.

Respond with EXACTLY this JSON structure (no other text):
{{
  "confirmed": "measurement of the team's own experiment that confirms the hypothesis",
  "refuted": "measurement of the team's own experiment that refutes the hypothesis",
  "inconclusive": "condition under which the team's experiment did not resolve"
}}"""


DIRECTIVE_VERIFIER_PROMPT = """You are the VERIFIER reviewing a research directive composed by the primary model. The directive is a plan a team executes to take a verified concept toward a publishable result. Your job is to catch fabrication and framing failures before a human researcher acts on a ghost.

ORIGINAL REGISTER ENTRY:
{register_entry_json}

CITATIONS ALLOWLIST (the primary was told it MUST cite only from this set):
{citations_json}

AGENT TOOL ALLOWLIST (the primary was told it MUST reference only from this set):
{tool_allowlist_json}

DIRECTIVE MARKDOWN UNDER REVIEW:
{directive_markdown}

DIRECTIVE FOOTER (primary's self-declaration of what it used IN THE AGENTIC PROMPT BLOCK):
{directive_footer_json}

Check each of these grounding rules. Flag violations; do NOT rationalise past them.

1. **Citation grounding**: Every URL / DOI / arXiv ID in the directive markdown must appear verbatim in the CITATIONS ALLOWLIST. Partial matches don't count — "arxiv.org/abs/2502.18864" and "arxiv.org/abs/2502.18864v1" are DIFFERENT strings. Any citation not in the allowlist is a fabrication.

   Note: the `## References` section is a deterministic dump of the entire allowlist by design — do not flag those entries as "unlisted"; they are the allowlist.

2. **Tool grounding**: Every tool name in the agentic prompt's code block must exact-match the AGENT TOOL ALLOWLIST. Case-sensitive. Generic phrasings like "search engine" / "an agent tool" / "a web crawler" that don't name a specific allowlist tool are fabrications by omission.

3. **Hand-wave detection**: Scan the Test Plan, Agentic Prompt, and Research Path for vague steps — phrases like "figure out the right approach", "try various prompts", "iterate until it works", "check various sources", "use an appropriate tool", "determine the best". These are hand-waves that render the directive non-executable.

4. **Measurable criteria**: The Verification Criteria table must measure THE TEAM'S OWN EXPERIMENTAL OUTPUTS. Each row needs a concrete observable signal — a numerical threshold computed on the team's data, a specific pattern in the team's measurements, a comparison the team runs. Vague language like "the approach demonstrates utility" is not measurable.

5. **Literature-watch leakage**: A directive criterion that depends on EXTERNAL publications, third-party benchmark releases, or future field-wide outputs is a framing failure — it shifts the burden onto other researchers instead of the team executing this directive. Flag any criterion in the Verification Criteria, Test Plan, or Hypothesis that contains phrasing like "by [date], at least one publicly accessible benchmark paper or code release reports …", "a published study confirms …", "the literature shows …", "by the target date a comparative analysis appears …". The exception is a criterion that names a SPECIFIC source the team will inspect (e.g. a specific dataset release the team monitors as supplementary evidence) — those are allowed when the team's own experimental work is genuinely out of reach.

6. **Self-declaration integrity**: The directive footer's `tool_names_used` and `citations_used` lists describe what the primary used INSIDE the Agentic Prompt code block ONLY. Compare the footer against ONLY the content inside the ```...``` fenced block under `## Agentic Prompt`. Citations or tool mentions appearing in Theory, Hypothesis, Prior Art Positioning, Test Plan, Verification Criteria, Research Path to Publication, or References do NOT need to be enumerated in the footer — those sections may freely reference any allowlist citation. A self-declaration mismatch is when the AGENTIC PROMPT BLOCK names a tool or citation that the footer omits, OR when the footer claims something the agentic prompt block does not actually contain.

Respond with EXACTLY this JSON structure (no other text):

{{
  "ok": true,
  "unlisted_citations": ["every URL/DOI/arXiv ID found in the directive markdown that is NOT in the citations allowlist — empty if clean. Excludes the References section's deterministic dump."],
  "unlisted_tools": ["every tool name found in the agentic prompt that is NOT in the tool allowlist — empty if clean"],
  "handwave_steps": ["direct quotes of hand-wave language in the Test Plan, Agentic Prompt, or Research Path — empty if clean"],
  "non_measurable_criteria": ["direct quotes of non-measurable entries in the Verification Criteria table — empty if clean"],
  "literature_watch_leakage": ["direct quotes of criteria/steps that depend on external publications or third-party benchmarks appearing — empty if clean"],
  "self_declaration_mismatches": ["descriptions of discrepancies between the footer and the AGENTIC PROMPT BLOCK ONLY — empty if clean. Discrepancies between the footer and other sections are NOT mismatches; those sections may freely cite from the allowlist."],
  "overall_assessment": "one paragraph stating whether the directive is executable as-is or what must be fixed",
  "severity": "clean|needs_fixes|fatal"
}}

`ok` is true iff all six checks pass (all arrays empty AND severity=clean). The retry loop depends on this boolean."""


DIRECTIVE_ELI5_PROMPT = """You are composing the IN-PLAIN-LANGUAGE section of a research directive. This section explains the verified concept to a curious non-specialist — someone smart and motivated but with no domain training in this area. The reader should walk away with an accurate, honest mental model of what is being claimed and why it matters.

ENGINE DOMAIN: {engine_domain}
REGISTER ENTRY UNDER TRANSLATION:
{register_entry_json}
HYPOTHESIS (from prior section):
{hypothesis}

Your job: write 3–5 sentences that a layperson can understand without losing the substance.

Rules:
- Avoid jargon. If a domain term is unavoidable, define it in-line in five words or fewer.
- Do not water down the claim. The plain-language version should be a faithful translation, not a vague gesture.
- Concrete > abstract. Where the original entry has a measurable quantity or a specific mechanism, name it in everyday terms.
- No metaphors that misrepresent the mechanism. A metaphor is fine if it makes the structure clearer; a metaphor is wrong if a reader could draw incorrect inferences from it.
- No examples from any specific field. Stay grounded in the entry's own subject.
- No citations, no URLs, no jargon-laden references — those belong elsewhere.

Respond with EXACTLY this JSON structure (no other text):
{{
  "eli5": "3-5 sentences explaining the concept and why it matters, accessible to a smart non-specialist"
}}"""


DIRECTIVE_RESEARCH_PATH_PROMPT = """You are composing the RESEARCH PATH TO PUBLICATION section of a research directive. This section is the human-facing strategic narrative — what a research team would do to take this concept from a verified idea to a published result. It is HIGHER LEVEL than the Test Plan or Agentic Prompt; those handle execution. This section answers: "if my team picked this up tomorrow, what study would we run, what would the paper look like, and how would we get there?"

ENGINE DOMAIN: {engine_domain}
REGISTER ENTRY UNDER TRANSLATION:
{register_entry_json}
HYPOTHESIS (from prior section):
{hypothesis}
TEST PLAN STEPS (from prior section, for coherence):
{test_plan_json}

Your job: write a structured research-program sketch. Each field is short — two or three sentences max. Do not duplicate the Test Plan; the Test Plan is the per-step experimental program, this section is the strategic envelope around it (study design, data, paper shape, venue class, phasing).

Rules:
- The team is the actor. Speak about what the team does, not about what the field will produce.
- No literature-watch framing. "By target date X a paper appears" is forbidden — describe what the team's paper itself would contribute.
- The paper's primary contribution should be derivable from the hypothesis and the Test Plan's measurements.
- Phases are short — name the phase, what it focuses on, and what concrete artifact transitions the team to the next phase. 3 to 5 phases.
- Target venue class is a CLASS not a specific journal/conference (e.g. "peer-reviewed venue in [the entry's primary field], emphasising methodological contributions"). The team will pick the specific venue based on what they produce.
- No examples from any specific field. Speak in shape, not content.
- No citations, no URLs.

Respond with EXACTLY this JSON structure (no other text):
{{
  "study_design": "what kind of study is this — observational, experimental, methods-paper, replication, theoretical, etc. — and the high-level shape of the work the team will perform",
  "data_and_instrumentation": "what data the team needs, where they get it, and what instruments / pipelines / models they need to build or assemble",
  "experiments_summary": "one short paragraph summarising the experiments the team runs (the Test Plan elaborates the steps; this is the elevator pitch of the experimental program)",
  "paper_structure": "the shape of the resulting paper — what sections it would have, what its primary contribution is, and what the central figures or tables would show",
  "target_venue_class": "class of venue the work fits, in shape not specific name",
  "phases": [
    {{"phase": "phase name", "focus": "what the team does in this phase", "exit_criterion": "concrete artifact or measurement that signals the team can move to the next phase"}}
  ],
  "risks_to_publication": ["1-3 specific risks that could stall the paper — e.g. data the team cannot obtain, a key measurement whose noise floor is too high, a confound the team must engineer around. Each risk in one short sentence."]
}}"""


DIRECTIVE_CONTRIBUTION_PROMPT = """You are composing the CONTRIBUTION ARTICULATION section of a research directive. The directive guides a verified concept toward publication; this section names the SPECIFIC contributions the resulting paper will claim, each grounded in concrete differentiators against the named peer systems. Reviewers ask "what's new here?" within the first 30 seconds of reading a paper — pre-articulating the contributions in concrete terms drives the paper's framing.

ENGINE DOMAIN: {engine_domain}
REGISTER ENTRY UNDER TRANSLATION:
{register_entry_json}
HYPOTHESIS (from prior section):
{hypothesis}

Your job: enumerate 3-5 specific contributions of this work, each anchored to a peer baseline named in the register entry's `closest_peer_system` or `functional_decomposition`. Each contribution must be checkable by a concrete experiment — preferably one already named in the test plan, or one that could be added to it.

Rules:
- Each contribution is ONE specific named claim, not a vague "advances the field" / "novel approach" framing.
- `peer_baseline` MUST reference an existing entry: a named system from `closest_peer_system`, a row of `functional_decomposition`, or a citation already in the register's prior-art lists. Do not invent peers.
- `magnitude_of_difference` is qualitative or quantitative — "first to apply X to Y", "reduces Z metric by ~N%", "removes the assumption that A". Specific over vague.
- `evidence_required` is the experimental result that would substantiate this contribution. Should map to a Test Plan step or Verification Criteria signal.
- No literature-watch framings ("by [date] a paper appears…"). Contributions are claims about THIS work, not predictions about other researchers.
- No fabrication: do not name peer systems that aren't in the register. Use the register's vocabulary.

Respond with EXACTLY this JSON structure (no other text):
{{
  "contributions": [
    {{
      "contribution": "one-sentence specific named contribution",
      "peer_baseline": "the existing system/method this is being compared against (must come from the register entry's peer_system or functional_decomposition)",
      "magnitude_of_difference": "qualitative or quantitative differentiator",
      "evidence_required": "what experimental result would substantiate this contribution"
    }}
  ]
}}"""


DIRECTIVE_OPEN_DECISIONS_PROMPT = """You are composing the OPEN DESIGN DECISIONS section of a research directive. The directive contains a Test Plan and Verification Criteria — concrete steps and signals — but several load-bearing decisions are deliberately LEFT TO THE TEAM because the directive can't responsibly make them in advance. Your job is to enumerate those forks explicitly, so the team doesn't waste the first week of the project arguing about decisions that should have been surfaced upfront.

ENGINE DOMAIN: {engine_domain}
REGISTER ENTRY UNDER TRANSLATION:
{register_entry_json}
HYPOTHESIS (from prior section):
{hypothesis}
TEST PLAN STEPS (already drafted):
{test_plan_json}
VERIFICATION CRITERIA (already drafted):
{verification_criteria_json}

Your job: surface 3-5 specific design decisions the team must resolve before executing. These are LOAD-BEARING decisions — at least one Test Plan step is blocked until each is settled. Not edge cases, not optional refinements.

Rules:
- Each decision is a real fork: 2-4 options the team is choosing between, each with a real downside.
- `tradeoff_summary` describes what changes between options — what's at stake, in concrete terms.
- `decision_blocker_risk` names which Test Plan step (or paper section) is gated by this decision.
- Forbidden:
  - "Choose appropriate metrics" (this is a vague hand-wave). Specify which metrics if it's a fork; otherwise drop it.
  - "Decide on the dataset" without naming candidates.
  - Decisions whose answer is obvious from the directive itself.
  - Decisions that the directive should have made (if you find one of these, the test plan or verification criteria has a hole — flag it).
- 3-5 entries; quality over quantity. If the directive has fewer than 3 genuine open decisions, return only what's real.

Respond with EXACTLY this JSON structure (no other text):
{{
  "open_decisions": [
    {{
      "decision": "the question the team must resolve",
      "options": ["option A description", "option B description", "..."],
      "tradeoff_summary": "what changes between options — what's at stake",
      "decision_blocker_risk": "which Test Plan step or paper section is gated by this decision"
    }}
  ]
}}"""


DIRECTIVE_REVIEWER_CONCERNS_PROMPT = """You are composing the ANTICIPATED REVIEWER CONCERNS section of a research directive. The verifier already enumerated 3-5 candidate "kill queries" (skeptic_probe) when validating this register entry — those are precisely the questions a hostile reviewer is most likely to ask. Surface them now as anticipated reviewer challenges with pre-emptive responses, so the team can address them in the paper rather than being caught off-guard at submission.

ENGINE DOMAIN: {engine_domain}
REGISTER ENTRY UNDER TRANSLATION:
{register_entry_json}
HYPOTHESIS (from prior section):
{hypothesis}
CONTRIBUTIONS ARTICULATED (from prior section):
{contributions_json}

Sources of anticipated concerns (in priority order):
1. `skeptic_probe.candidate_queries` — the verifier's enumerated kill queries. Each is a 1-sentence question or claim a hostile reviewer is most likely to raise. THESE ARE THE PRIMARY SOURCE.
2. `closest_peer_system` overlap — if a peer system has substantive overlap and weak differentiators, a reviewer will ask about it. Translate the overlap into a reviewer-concern.
3. `contradicting_findings` — published work that disagrees with the composite claim. Reviewers find these.
4. `reasoning_flaws` — flaws the verifier itself surfaced.

Your job: for each concern, draft a defensible 1-2 sentence response that the team can use directly in the paper. Cite a specific contribution, peer system, or Test Plan step where possible.

Rules:
- DO NOT invent reviewer concerns from nothing. Each `concern` must be traceable to one of the four sources above. The `concern_source` field names which one.
- Responses must be defensible from the directive's own contents — no appeals to "future work will show" or "we plan to investigate".
- Keep each `concern` to one sentence in the voice of a hostile reviewer ("This claim is just X under a different name", "This doesn't address the case where Y").
- Keep each `evidence_against` to 1-2 sentences citing a specific contribution, differentiator, peer-system distinction, or Test Plan element.
- 3-7 entries; do not pad. If the register entry has only 2 genuine concerns, return 2.

Respond with EXACTLY this JSON structure (no other text):
{{
  "anticipated_concerns": [
    {{
      "concern": "what a hostile reviewer would say (one sentence, in their voice)",
      "concern_source": "skeptic_probe | peer_overlap | contradicting_finding | reasoning_flaw",
      "evidence_against": "1-2 sentence response citing a specific contribution / peer-system difference / Test Plan element",
      "paper_section": "introduction | related_work | methods | results | discussion | limitations"
    }}
  ]
}}"""


CANONICAL_FORM_PROMPT = """You are extracting the CANONICAL STRUCTURED FORM of a research claim. This runs as Stage 1 of the three-stage verifier — BEFORE the heavy phased prior-art search. Stage 2 will use your output to detect whether the claim is a structural duplicate of an existing register entry; if so, the heavy verifier is skipped entirely.

Goal: render the claim's central architectural move as a stable structured tuple so two claims that are surface-different but structurally identical canonicalize to (approximately) the same tuple. Downstream code uses this canonical form to detect articulate restatements that surface-similarity alone misses.

ENGINE DOMAIN: {engine_domain}
CLAIM TITLE: {title}
CLAIM DESCRIPTION:
{description}

CENTRAL ARCHITECTURAL MOVE (optional pre-extraction — empty string if you should derive it yourself from the description):
{central_architectural_move}

Fill these slots — each is an ATOMIC, SHORT canonical phrase. Two structurally identical claims must produce the same slots, so prefer short common-form phrasings over verbose specific descriptions. Specificity goes in `key_constraints`, not in the main slots.

- `move_predicate`: ONE VERB or two-word verb-phrase. Lowercase. Examples of shape (NOT for copy-paste): "replaces", "adds", "decomposes", "gates", "routes", "aggregates". Pick the single verb that captures what the move actually does to the system.
- `on_substrate`: 2-5 WORDS naming what the move acts on. Strip qualifiers, scopes, and field-specific decorations. Examples of shape: "novelty scoring", "verifier panel", "expansion gate". If the claim says "novelty verification in pre-formal LLM research loops" → `on_substrate` is just "novelty verification". The "pre-formal LLM research loops" framing belongs in `target_domain`.
- `with_mechanism`: 2-5 WORDS naming the technique class, NOT a full method description. Reduce to its canonical category. Examples of shape: "pairwise tournament", "rejection-option classification", "ensemble disagreement", "symbolic unification", "band-pass control". If the claim describes a "selective triage pipeline with canonicalization-stage reject option, explicit alias-gap band, and disagreement-sensitive heterogeneous committee escalation" → `with_mechanism` is "rejection-option classification" (the dominant technique class) and the rest goes in `key_constraints`.
- `target_domain`: 2-6 words naming the domain context. "llm research loops", "iterative search agents", etc.
- `key_constraints`: 2-4 short bullet phrases (≤8 words each) naming the load-bearing qualifiers — conditions or properties without which the claim collapses to a different (usually weaker or pre-existing) position. THIS is where the specific algorithm, threshold, dimensionality, or topology details belong.

Rules:
- Lowercase prose. No fluff. No hedging language ("may", "can", "potentially").
- Length caps are hard. `on_substrate` and `with_mechanism` exceeding 5 words means you're putting too much in those slots — move detail to `key_constraints`.
- Slots must be derivable from the claim text. Do NOT invent constraints the claim does not assert.
- If the claim has no clear central move (e.g. the description is purely motivational or the central move field is empty), return ALL slots empty. An empty canonical form is a valid output — fabrication is not.
- Domain-neutral: do not import examples or assumptions from any specific research field. Stay grounded in the claim's own subject.

Respond with EXACTLY this JSON (no other text):
{{
  "move_predicate": "...",
  "on_substrate": "...",
  "with_mechanism": "...",
  "target_domain": "...",
  "key_constraints": ["...", "..."]
}}"""
