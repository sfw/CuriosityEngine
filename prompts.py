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


HYPOTHESIS_PROMPT = """You are a research engine about to investigate a question in {domain}. Before investigating, commit to a specific hypothesis.

QUESTION: {question}

This step exists SOLELY so we can measure surprise later. Do not hedge. Do not say "it depends". Pick the most specific answer you can defend from what you currently believe, and commit to it.

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


SURPRISE_PROMPT = """You are a research engine assessing how much an investigation surprised you relative to your pre-committed hypothesis.

QUESTION: {question}

YOUR PRE-INVESTIGATION HYPOTHESIS:
{hypothesis_json}

INVESTIGATION FINDINGS:
{findings_json}

Your task: compare the findings to the hypothesis and produce an honest surprise assessment. A high surprise_delta means the findings diverged significantly from what you expected. A low surprise_delta means the investigation confirmed your prior. Be honest — the value of this system depends on calibrated surprise signals, not flattering ones.

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

**Tool budget**: aim for roughly 8-15 tool calls, then render your verdict. Efficient refutation beats exhaustive search.

**CRITICAL — novelty framing**
A *genuinely novel synthesis* has a specific fingerprint: its individual premises (ingredients) are each established in the literature, but the precise synthesis (how they combine into this specific claim/architecture/mechanism) is NOT published anywhere. That is exactly what we want to register — do not reject it for being a "new combination of known parts."

Only *restatement* — where the full synthesis itself (not just its ingredients) is already in the literature under a different name — disqualifies novelty.

Score the two axes INDEPENDENTLY:

A) **premises_supported**: Are the building blocks this insight depends on well-grounded in the literature? (TRUE is GOOD — it means the ingredients are real.)
B) **synthesis_findable**: Is the specific synthesis / composite claim / architecture itself already published, patented, or widely-discussed under any name? (TRUE is BAD — it means the "novel" claim is not novel.)

The ideal register entry has **premises_supported=TRUE and synthesis_findable=FALSE**. That is the *signature* of genuine novelty, not a weakness.

**code_execution (when available) lets you actually RUN python** — use it to recompute numerical claims cited by the insight, test the reasoning against real data, or simulate the proposed mechanism.

Complete four tasks using the tools:

1. PREMISES CHECK
For each ingredient the insight depends on, search for evidence the ingredient is real. Record what you found in `premises_support_citations`. Set `premises_supported=true` iff each load-bearing premise has real literature support.

2. SYNTHESIS PRIOR-ART SEARCH
Search specifically for prior work that asserts THIS SYNTHESIS — the whole composite claim, not just its parts. Try restatements, different naming conventions, patents, blog posts. Set `synthesis_findable=true` only if you find substantive prior work making essentially the same composite claim. Record what you found (or what you searched and failed to find) in `synthesis_prior_art`.

3. CONTRADICTING EVIDENCE
Search for research that actively disagrees with the composite claim. Record in `contradicting_findings`.

4. REASONING AUDIT
Examine the connection between supporting entries and the claim. Is the inferential leap justified by the premises, or does the argument smuggle in unsupported steps? Record in `reasoning_flaws`.

Classify the insight's `novelty_type`:
- "new_synthesis": premises established, composite claim not in literature → GENUINE novelty candidate
- "restatement": composite claim already in literature under some name → NOT novel
- "extension": composite claim is a modest extension of a published claim → marginal novelty
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
  "synthesis_findable": false,
  "synthesis_prior_art": ["candidates for the whole-synthesis claim you searched for; empty list if nothing close found"],
  "novelty_type": "new_synthesis|restatement|extension|correction|unsupported",
  "contradicting_findings": ["specific findings or papers that contradict the composite claim"],
  "reasoning_flaws": ["specific flaws in the evidence-to-claim connection"],
  "motivation": "one-paragraph explanation of why this insight matters if validated",
  "verdict": "validated|challenged|refuted|inconclusive",
  "verified_confidence": 0.0-1.0,
  "verification_summary": "one-paragraph justification citing what the adversarial search found or failed to find, and WHY the verdict follows from premises_supported × synthesis_findable; for `inconclusive`, explicitly name the epistemic gap that prevented verification",
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


ANALOG_PROBE_PROMPT = """You are a research engine looking for CROSS-DOMAIN ANALOGS of a finding you just uncovered. Biology has inspired algorithms (neural nets, genetic search, ant colonies). Economics has inspired mechanism design for distributed systems. Thermodynamics has inspired learning-rate schedules. The best novel ideas often come from recognizing that *the same structural mechanism* appears in a foreign field, under a different vocabulary.

THE FINDING:
Question investigated: {entry_question}
Surprise delta: {entry_surprise:.2f}
Key takeaways:
{entry_takeaways}

DOMAINS ALREADY IN PLAY ON THIS JOURNAL (avoid these — they're NOT distant enough):
{recent_tags}

Your task: identify 2-3 DISTANT domains — fields whose established vocabulary, methods, or mechanisms are structurally analogous to the finding but that do NOT appear in the recent-tags list above. Then, for each, produce a specific, investigable question that translates the analog back into the research area of the original finding.

**What counts as a strong analog (not a weak one):**
- STRONG: "The finding's coupling dynamics mirror predator-prey oscillations studied in population ecology. Question: can Lotka-Volterra stability conditions be applied to <finding's mechanism> to predict when it will collapse?"
- WEAK: "This is like how brains work." (Vague, already in play, no specific mechanism named.)
- WEAK: "Evolutionary algorithms already exist." (Not an analog, it's the same field.)

**Rules:**
- Each analog domain must be STRUCTURALLY analogous (same dynamics, same constraints, same failure modes), not just topically related.
- Name a specific mechanism, method, law, or formal result from the analog domain. Not just "biology" — name the sub-field and the thing ("population ecology's competitive exclusion principle"; "statistical mechanics' maximum entropy principle"; "antibody-epitope lock-and-key binding in immunology").
- The investigable question must be answerable with web_search / academic_search — not purely philosophical.
- If you cannot find a STRONG cross-domain analog, return an empty list. Weak analogs pollute the queue. Quality over quantity.

Respond with EXACTLY this JSON structure (no other text):
{{
  "analogs": [
    {{
      "domain": "specific sub-field (e.g. 'population ecology', 'statistical mechanics', 'immune system biology')",
      "mechanism": "the specific mechanism, law, or named result in that domain",
      "structural_analogy": "one sentence explaining which dynamics/constraints/failure-modes map between the finding and this analog",
      "question": "an investigable question that translates the analog into the original research area"
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
