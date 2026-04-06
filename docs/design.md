# Context Resolution Engine — Design Document

## Problem

LLMs produce more accurate responses when given precisely the right context — not more context, but the right context. Current approaches (RAG, full-vault retrieval, manual prompt engineering) either over-disclose (adding noise) or under-disclose (missing critical information).

## Core Principle

**Granular disclosure.** For each query or task, determine exactly what knowledge is needed, retrieve only that, and let the LLM synthesise from a tight, curated window.

## Abstract Functions

| Function | Description | Owner |
|---|---|---|
| Knowledge representation | Store facts, relationships, and rationale with preserved structure | Graph |
| Context selection | Given a query, determine which knowledge is relevant and retrieve it | Graph (native traversal, self-improving) |
| Rule evaluation | Apply deterministic logic to structured data, return computed answers | Logic engine (Prolog candidate) |
| Knowledge gap detection | Recognise when available knowledge is insufficient; identify what's missing | LLM |
| External retrieval | Fetch missing knowledge from outside the system (web, APIs, databases) | LLM (orchestrating) |
| Synthesis | Produce coherent natural language response from curated context and computed answers | LLM |

## Architecture

```
Query (natural language)
  │
  ▼
Context Selection (graph traversal)
  │
  ├──▶ Knowledge fragments (relevant subgraph)
  ├──▶ Logic engine (deterministic rule evaluation)
  │
  ▼
Gap Detection (LLM)
  │
  ├──▶ Sufficient → Synthesis
  └──▶ Insufficient → External Retrieval → Synthesis
         │
         ▼
      Response
         │
         ▼
      Feedback Loop → Graph enrichment
```

## Feedback Loop

The graph improves through use, analogous to reinforcement learning:

1. Query arrives, graph traversal returns a context slice
2. LLM uses slice to respond
3. Signal indicates context quality:
   - **Explicit**: user corrects the response or confirms accuracy
   - **Implicit (LLM)**: LLM had to request additional context or search externally to fill a gap
   - **Implicit (logic engine)**: rule evaluation failed because a required fact wasn't in the context slice
4. Signal feeds back into graph: strengthen useful edges, weaken noisy ones, add missing connections

The graph learns its own disclosure policy over time.

## Design Constraints

- **General-purpose from the start.** No domain-specific assumptions baked into the engine. The workout domain is the proving ground, not the product.
- **Domain-agnostic graph.** Nodes are knowledge atoms. Edges are typed relationships. Schema of node/edge types must be extensible without code changes.
- **Domain-agnostic rules.** Logic predicates are user-defined or system-learned, not hardcoded.
- **Domain-agnostic feedback.** Reinforcement signal and graph enrichment process work the same regardless of domain.

## Data Model

Two primitives. Both carry extensible metadata and optional content.

### Node

A knowledge atom. No fixed types at the schema level — metadata distinguishes what kind of knowledge it represents (fact, rule, observation, concept, rationale, etc.).

```
Node:
  id          — unique identifier
  content     — the knowledge itself (text, structured data)
  metadata    — extensible key-value pairs (type, domain, confidence, source, timestamps)
```

### Edge

A relationship between two nodes. Edges are rich — they carry content (the "why" of the relationship) and metadata (type, weight, confidence). Not just a labelled pointer.

```
Edge:
  id          — unique identifier
  source      — node id
  target      — node id
  content     — optional (explanation of the relationship)
  metadata    — extensible key-value pairs (type, weight, confidence, source)
```

The weight field is what the feedback loop acts on. Edges that lead to useful context get strengthened. Edges that contribute noise get weakened.

### Design Decision: Single Source of Truth

The graph is the only data store. The logic engine reads from the graph at query time — it does not maintain a separate representation. Rules are nodes (tagged `type: rule`) with structured content. Facts are nodes. The logic engine extracts what it needs from graph state.

Consequences:
- No sync problems between graph and logic layer
- Feedback loop only updates one structure
- Graph schema must support everything the logic engine needs to reason over

## Query Classification

Queries are classified as a tuple, not a single category. Each dimension independently influences how the system responds.

### Query Tuple

```
QueryType:
  intent    — what the user wants
  focus     — what kind of knowledge is needed
```

### Intent (controls the pipeline)

Determines which system components participate and how.

| Intent | Description | Pipeline |
|---|---|---|
| **Retrieve** | Get information | Graph traversal → Synthesis |
| **Evaluate** | Assess state against rules | Graph traversal → Logic engine → Synthesis |
| **Compare** | Contrast two or more things | Multiple subgraph traversals → Synthesis |
| **Act** | Determine what to do | State retrieval → Logic engine → Synthesis |

### Focus (controls the traversal)

Determines which edge types to prioritise and what kind of nodes to seek.

| Focus | Description | Prioritised edges |
|---|---|---|
| **Causal** | Why something is the case | `because_of`, `supports`, `contradicts` |
| **Procedural** | How something works or is done | `requires`, `depends_on`, `next_step` |
| **Temporal** | When something applies or triggers | `triggered_by`, `after`, `if_then` |
| **Attributive** | What something is or has | `is_a`, `has_property`, `belongs_to` |
| **Conditional** | Whether criteria are met | `if_then`, `threshold`, `requires` |

### Examples

| Query | Tuple | What happens |
|---|---|---|
| "Why do I avoid squats?" | `(retrieve, causal)` | Graph follows causal edges from "squats" node → Synthesis |
| "Should I increase bench weight?" | `(evaluate, conditional)` | Graph retrieves state + rules → Logic engine evaluates → Synthesis |
| "Sled drags vs reverse lunges?" | `(compare, attributive)` | Two subgraphs retrieved → Synthesis highlights differences |
| "What should I do today?" | `(act, procedural)` | State + schedule + rules retrieved → Logic engine → Synthesis |
| "When should I deload?" | `(retrieve, temporal)` | Graph follows temporal/conditional edges → Synthesis |
| "How do I progress overhead press?" | `(retrieve, procedural)` | Graph follows procedural edges from OHP node → Synthesis |

### Extensibility

New intents or focus types can be added without redesigning the taxonomy. The traversal strategy is a function of the tuple — each dimension maps independently to traversal parameters.

## Traversal Strategy

The traversal strategy is derived from the query tuple. It configures how the graph walks from seed nodes to produce a context slice.

```
TraversalStrategy:
  seeds          — entry nodes (found via keyword/semantic match, explicit reference, or conversation context)
  edge_filters   — which edge types to follow (derived from query focus)
  depth          — max hops from seed
  weight_floor   — minimum edge weight to traverse (feedback loop tunes this over time)
  budget         — max nodes to return
```

### Seed Selection

Entry nodes are found by combining:

- **Keyword/semantic match** — query terms matched against node content
- **Explicit reference** — query directly names an entity that maps to a node
- **Conversation context** — nodes already touched in the current session

### Stopping Conditions

- **Depth limit** — max hops reached
- **Budget exhausted** — enough nodes collected
- **Weight decay** — edge weights below floor, too far from seed to be relevant
- **Redundancy** — new node overlaps significantly with already-collected content

### Seed Selection Strategies

Testing revealed three distinct seed strategies:

1. **Direct reference** — query names a specific entity. Seed is that node. Works for most queries.
2. **Metadata filter** — query asks about a class of things ("which exercises should I avoid?"). Seeds found by filtering node metadata (e.g. `type=exercise, status=avoided`).
3. **Scan** — query requires aggregation across many entities ("which exercises have stalled?"). Returns all matching nodes without traversal. See Architectural Limitations below.

### Premise Check

Before traversal, the system inspects seed node metadata. If the query intent contradicts the seed's status (e.g. asking to evaluate an exercise marked `status: avoided`), the tuple is overridden to `(retrieve, causal)` so the traversal explains why it's unavailable rather than trying to compute an answer.

This is a general-purpose heuristic — works for any domain where nodes carry a status field.

### Traversal Heuristics

Two heuristics emerged from testing:

**Anti-hub.** Hub nodes (e.g. a rule connected to many exercises) flood the budget with same-type neighbours. When traversing from a non-seed node, edges leading back to the same node type as the seed are deprioritised. This ensures the traversal goes deeper (rule → deload rule) rather than wider (rule → more exercises).

**Recency weighting.** For time-series data (observations/logs), recent entries get higher edge weights (0.3 → 0.9). This biases traversal toward current state rather than historical data.

**Multi-seed balance.** For compare queries with multiple seeds, budget is split evenly across seeds to prevent one from consuming the entire context slice.

## Test Results

### Small Scale (55 nodes, 17K chars)

Treatment matched control accuracy on all 8 queries. 85-92% context reduction. Treatment won on focus/conciseness in 4 of 8, tied the rest.

At this scale, the LLM handled full context without confusion. The graph selection won on signal-to-noise ratio but not on accuracy.

### Large Scale (347 nodes, 97K chars)

| Query | Reduction | Control | Treatment | Winner |
|---|---|---|---|---|
| Bench press increase? | 98% | Correct | Correct | Tie |
| OHP stalled? | 98% | Correct (deload to 37.5kg + HRV correlation) | **Wrong** (said "don't drop weight") | **Control** |
| Squat weight? (safety) | 98% | Safe | Safe | Tie |
| Row vs close-grip bench? | 98% | Correct | Correct | Treatment |
| Which exercises stuck? | 93% | **Correct** (all 4 identified) | **Failed** (no observation data) | **Control** |
| Knee rehab progress? | 98% | Correct | Correct | Tie |
| All injury status? | 99% | Excellent (cross-referenced) | Correct (clean) | Control |
| Back to 4-day split? | 99% | Correct (with proof points) | Correct (focused) | Treatment |

**Key findings:**

1. **Focused queries work.** For questions about a specific entity (bench press, knee rehab, programme choice), 98% context reduction with equal or better accuracy.

2. **Rule chain traversal is critical.** Q2 (OHP deload) failed because the deload rule was pushed out of the budget by hub node flooding. The traversal reached the progression rule but budget was consumed by other exercises before following `falls_back_to → deload_rule`. This is the most important fix.

3. **Scan/aggregation queries don't fit traversal.** Q5 (which exercises stalled?) requires comparing data across all exercises — not walking outward from a seed. A separate "scan" intent is needed that runs analytics over the graph rather than BFS from entry nodes.

4. **Full context enables correlation the treatment cannot.** At 97K chars, the control correlated OHP stalls with HRV drops, sleep quality, and protein intake. It cited cross-exercise data to support programme recommendations. These emergent cross-domain insights are a strength of full context that granular disclosure sacrifices by design. The trade-off is intentional — accuracy on the specific question vs serendipitous correlation — but it should be acknowledged.

5. **The feedback loop would fix Q2.** After the first failed OHP query, the system would learn that `progression_rule → deload_rule` is a critical edge for evaluate/conditional queries. The edge weight would increase. Next time, the deload rule would be included.

### Architectural Limitations Identified

**Scan queries.** When the query requires comparing state across many entities, traversal-based context selection doesn't work. The system needs a `scan` intent that:
- Retrieves all entities matching a filter
- Runs a lightweight analysis (e.g. "compare first and last observation weight")
- Returns only the entities where the analysis is interesting

**Rule chain completeness.** When a rule node is included, all rules it chains to (via `falls_back_to` or similar) must also be included. Budget should not sever rule chains. This is a hard constraint, not a weight-based preference.

**Cross-domain correlation.** Granular disclosure by design excludes tangentially related data. The control response to Q2 correlated OHP performance with HRV, sleep, and stress — something the treatment could never do. For queries where cross-domain insight is valuable, the system may need to explicitly widen the context slice.

## Open Questions

- Storage: property graph, triple store, or something simpler?
- How are edge types standardised across domains?
- What does the tuple-to-strategy mapping look like concretely? (lookup table, rules, learned?)
- How should the scan intent work? Pre-computed summaries, or on-demand analysis?
- When should the system choose to widen context for cross-domain correlation?

## Tooling (deferred)

Tool selection follows from the data model. Candidates under consideration but not committed:

- **Graph store**: TBD (property graph, triple store, Obsidian, custom)
- **Logic engine**: Prolog (candidate, not confirmed)
- **LLM**: Claude
- **Human editing interface**: TBD (Obsidian is a candidate frontend, not the graph itself)
