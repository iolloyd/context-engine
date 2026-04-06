# Context Engine

Graph-based context resolution for LLMs. Given a query, retrieve only the knowledge the model needs — no more, no less.

## Why

LLMs produce more accurate responses when given precisely the right context. Current approaches either over-disclose (RAG dumps, full-vault retrieval) or under-disclose (manual prompt engineering). This engine aims for **granular disclosure**: a tight, curated context slice computed per query, with a feedback loop that learns which edges are worth following.

See [`docs/design.md`](docs/design.md) for the full design.

## Architecture

```
Query (natural language)
  │
  ▼
Classifier ── (intent, focus) tuple ──┐
  │                                    │
  ▼                                    ▼
Seed selection              TraversalStrategy
  │                                    │
  ▼                                    ▼
Graph traversal ◀───────────────────────
  │
  ├──▶ Rule chain closure (structural edges)
  ├──▶ Premise check (status metadata)
  │
  ▼
Context slice ─────┐
  │                │
  ▼                ▼
Logic engine   Gap detection (LLM)
  │                │
  └────────┬───────┘
           ▼
       Synthesis (LLM)
           │
           ▼
        Response
           │
           ▼
     Feedback signal → edge weight updates
```

## Core primitives

Two tables. Everything else is metadata.

```python
class Node:
    id: str
    content: str
    metadata: dict      # type, domain, confidence, source, status, timestamps
    embedding: bytes    # for seed selection

class Edge:
    id: str
    source: str         # node id
    target: str         # node id
    content: str | None # the "why" of the relationship
    metadata: dict      # type, weight, confidence
```

Weight is what the feedback loop acts on.

## Query tuple

```
(intent, focus)

intent ∈ {retrieve, evaluate, compare, act, scan}
focus  ∈ {causal, procedural, temporal, attributive, conditional}
```

The tuple maps to a `TraversalStrategy` (edge filters, depth, weight floor, budget). Mapping starts as a lookup table and can be learned.

## Storage

SQLite + `sqlite-vec` for vector similarity on seed selection. Property graph emulated via two tables. One file, no ops.

## Status

Phase 0 — scaffolding. Design stable, implementation in progress.

## License

MIT
