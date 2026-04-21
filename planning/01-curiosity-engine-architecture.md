# Curiosity Engine: Architecture & Design

## Overview

The Curiosity Engine is a system that enables an LLM to generate its own research questions from self-assessed uncertainty, investigate them using available tools, accumulate findings in a persistent research journal, and cross-reference entries to surface novel insights. This proof of concept focuses on the AI/ML research domain.

## Core Thesis

Novel insight emerges not from single queries but from the *intersection* of knowledge gaps. By systematically mapping uncertainty, investigating it, and cross-referencing findings over time, a system can discover connections that no individual prompt would produce.

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CURIOSITY LOOP                     │
│                                                      │
│  ┌──────────────┐    ┌──────────────────┐           │
│  │  Uncertainty  │───▶│    Question      │           │
│  │  Introspector │    │    Generator     │           │
│  └──────────────┘    └───────┬──────────┘           │
│                              │                       │
│                              ▼                       │
│                     ┌──────────────────┐            │
│                     │  Investigation   │            │
│                     │     Engine       │            │
│                     │  (tools: search, │            │
│                     │   code, analyze) │            │
│                     └───────┬──────────┘            │
│                              │                       │
│                              ▼                       │
│                     ┌──────────────────┐            │
│                     │    Research      │            │
│                     │    Journal       │            │
│                     │  (persistent)    │            │
│                     └───────┬──────────┘            │
│                              │                       │
│                              ▼                       │
│                     ┌──────────────────┐            │
│                     │ Cross-Reference  │──────┐     │
│                     │    Engine        │      │     │
│                     └──────────────────┘      │     │
│                                               │     │
│                     ┌──────────────────┐      │     │
│                     │    Insight       │◀─────┘     │
│                     │   Synthesizer    │            │
│                     └───────┬──────────┘            │
│                              │                       │
│                              ▼                       │
│                     Feeds new questions              │
│                     back to Introspector              │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Components

### 1. Uncertainty Introspector

**Purpose:** Identify regions of the model's knowledge where it is uncertain, contradictory, or shallow.

**Mechanism:** Structured self-interrogation prompts that ask the model to:
- Identify claims it can make about a topic but isn't confident in
- Find tensions between related concepts it holds
- Detect areas where its knowledge feels thin relative to the topic's importance
- Notice where it would give different answers depending on framing

**Input:** A domain seed (e.g., "AI/ML research") + previous journal entries (for continuity)

**Output:** An `UncertaintyReport` containing categorized uncertainty items:

```python
@dataclass
class UncertaintyItem:
    description: str               # What the uncertainty is about
    uncertainty_type: str          # "contradiction" | "gap" | "shallow" | "unstable"
    domain_tags: list[str]        # e.g., ["transformer_architecture", "scaling_laws"]
    estimated_importance: float   # 0-1: how much resolving this might matter
    related_items: list[str]      # IDs of related uncertainties or journal entries
```

### 2. Question Generator

**Purpose:** Transform uncertainty items into actionable, investigable research questions ranked by potential for novel insight.

**Scoring criteria:**
- **Intersection potential:** Questions that span multiple uncertainty regions score higher (this is where novel connections are most likely)
- **Investigability:** Can this question be meaningfully explored with available tools?
- **Novelty distance:** How far is this from well-trodden territory?
- **Resolution impact:** Would answering this question resolve or clarify multiple uncertainties?

**Output:**

```python
@dataclass
class ResearchQuestion:
    id: str
    question: str
    source_uncertainties: list[str]    # IDs of uncertainty items that generated this
    priority_score: float              # Composite of scoring criteria
    domain_tags: list[str]
    investigability_notes: str         # How the model plans to investigate
    status: str                        # "pending" | "investigating" | "completed" | "abandoned"
```

### 3. Investigation Engine

**Purpose:** Pursue research questions using available tools and produce structured findings.

**Available tools:**
- **Web search:** Search for papers, articles, discussions
- **Code execution:** Run experiments, analyze data, test hypotheses
- **Self-interrogation:** Probe the model's own knowledge from multiple angles

**Process per question:**
1. Form an initial hypothesis (what does the model expect to find?)
2. Investigate using tools
3. Compare findings to hypothesis
4. Calculate "surprise delta" — how much did findings diverge from expectations?
5. Log everything to the journal

**Design principle:** The hypothesis step is critical. Without it, there's no way to measure surprise, and surprise is the primary signal for novelty.

### 4. Research Journal

**Purpose:** Persistent, structured storage of all findings, accessible across sessions.

**Storage:** JSON file on disk (simple for PoC; database for production)

**Schema:**

```python
@dataclass
class JournalEntry:
    id: str
    timestamp: str
    question_id: str
    question: str
    
    # Pre-investigation
    hypothesis: str                    # What the model expected to find
    confidence_before: float           # 0-1
    
    # Investigation
    methodology: str                   # How the question was investigated
    raw_findings: str                  # What was actually found
    sources: list[str]                 # URLs, paper titles, etc.
    
    # Post-investigation
    surprise_delta: float              # 0-1: how much findings diverged from hypothesis
    confidence_after: float            # 0-1
    key_takeaways: list[str]          # Distilled insights
    new_questions: list[str]          # Questions that emerged from investigation
    
    # Metadata
    domain_tags: list[str]
    connections_to: list[str]          # IDs of related journal entries (added over time)
    cross_reference_notes: list[str]   # Added by the cross-reference engine
```

### 5. Cross-Reference Engine

**Purpose:** Periodically review the journal to find patterns, connections, and emergent themes across entries.

**This is the most important component for novelty.**

**Process:**
1. Load all journal entries (or recent window for efficiency)
2. Identify entries with overlapping domain tags but different questions
3. Look for recurring patterns: Do multiple investigations point toward a common underlying principle?
4. Find "surprise clusters" — entries with high surprise deltas in related domains
5. Detect contradictions between findings from different investigations
6. Generate connection hypotheses: "Entry X and Entry Y both found Z, which suggests..."

**Output:**

```python
@dataclass
class CrossReference:
    id: str
    timestamp: str
    source_entries: list[str]          # Journal entry IDs that are connected
    connection_type: str               # "pattern" | "contradiction" | "convergence" | "implication"
    description: str                   # What the connection is
    novelty_score: float               # 0-1: how unexpected/non-obvious this connection is
    implications: list[str]            # What this connection might mean
    suggested_questions: list[str]     # New questions to investigate based on this connection
```

### 6. Insight Synthesizer

**Purpose:** Promote the most significant cross-references into fully articulated novel insights.

**Criteria for promotion:**
- High novelty score
- Supported by multiple independent journal entries
- Has non-trivial implications
- Not already well-established in the model's training data (verified by self-check)

**Output:**

```python
@dataclass
class Insight:
    id: str
    timestamp: str
    title: str                         # Concise statement of the insight
    description: str                   # Full articulation
    supporting_evidence: list[str]     # Journal entry and cross-reference IDs
    novelty_assessment: str            # Why this is believed to be novel
    confidence: float                  # 0-1
    implications: list[str]           
    open_questions: list[str]          # What would need to be true for this to matter?
    counter_arguments: list[str]       # Why this might be wrong
```

---

## Loop Execution

### Single Cycle

```
1. INTROSPECT  → Generate uncertainty report for domain
                  (incorporate previous journal entries for continuity)

2. GENERATE    → Produce 3-5 ranked research questions
                  (prioritize intersection of uncertainty regions)

3. INVESTIGATE → Pursue top 1-2 questions with tools
                  (hypothesis → research → surprise delta)

4. LOG         → Write findings to journal

5. CROSS-REF   → Review journal for new connections
                  (run every N cycles, not every cycle)

6. SYNTHESIZE  → Promote significant cross-references to insights
                  (run when cross-ref finds high-scoring connections)

7. FEED BACK   → New questions from steps 3-6 enter the question queue
```

### Cycle Frequency

- Steps 1-4: Every cycle (the core investigation loop)
- Step 5: Every 3-5 cycles (needs accumulated material to work with)
- Step 6: When cross-reference produces novelty_score > 0.7
- Step 7: Continuous (new questions always feed back)

---

## Configuration

```python
@dataclass
class EngineConfig:
    domain: str = "AI/ML research"
    journal_path: str = "./research_journal.json"
    
    # Per cycle
    questions_per_cycle: int = 3          # Questions generated per introspection
    investigations_per_cycle: int = 1     # Questions actually investigated per cycle
    
    # Cross-reference
    cross_ref_frequency: int = 3          # Run cross-ref every N cycles
    novelty_threshold: float = 0.7        # Minimum score to promote to insight
    
    # Limits
    max_cycles: int = 10                  # Safety limit for automated runs
    max_journal_entries: int = 100        # Cap for PoC
    
    # Model
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
```

---

## Key Design Decisions

1. **Hypothesis-first investigation:** Every question gets a pre-investigation hypothesis so we can measure surprise. Without this, we can't distinguish "learned something new" from "confirmed what we already knew."

2. **Surprise delta as primary signal:** High surprise = the model found something it didn't expect. This is the closest proxy we have for genuine novelty without the full epistemic delta map.

3. **Cross-referencing is batch, not continuous:** We accumulate several journal entries before cross-referencing. Premature cross-referencing on too few entries produces shallow connections.

4. **Self-skepticism built in:** Insights include counter_arguments and open_questions. The system is designed to challenge its own findings.

5. **Domain tags enable far-transfer detection:** When the cross-reference engine finds connections between entries with *dissimilar* domain tags, that's a signal of the far-transfer insight we're specifically looking for.

---

## Future Extensions (Phases 1-2 Integration)

- **Epistemic Delta Map:** Replace self-assessed uncertainty with empirically tracked errors
- **Plastic LoRA Layer:** Route high-confidence, high-utility findings to adapter training
- **Consolidation Cycle:** Use the journal + delta map as the scoring function for what gets baked into weights
- **Multi-agent investigation:** Spawn independent agents to investigate different questions in parallel, then cross-reference their findings
