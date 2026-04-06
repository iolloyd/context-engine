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

## Authoring

Knowledge is authored as a **folder tree** — one folder per node, a `readme.md` with YAML frontmatter for content and metadata, and an optional `edges.yaml` for outgoing typed edges. The tree is the source of truth; SQLite is a rebuildable index. Everything is git-diffable.

```
knowledge/
├── exercises/
│   ├── bench-press/
│   │   ├── readme.md       ← frontmatter + body
│   │   └── edges.yaml      ← outgoing edges with type, weight, content
│   └── squat/
│       ├── readme.md
│       └── edges.yaml
└── rules/
    └── progression/
        ├── readme.md       ← JSON rule body
        └── edges.yaml      ← falls_back_to: deload
```

See [`docs/adr/0001-knowledge-source.md`](docs/adr/0001-knowledge-source.md) for the full rationale (why not Obsidian).

```bash
ctx --db graph.db import fixtures/tree       # build
ctx --db graph.db query "Why do I avoid squats?" --seed exercises/squat
ctx --db graph.db export fixtures/tree       # write back (feedback loop uses this)
```

## Status

Phase 0 — scaffolding. Design stable, implementation in progress.

## License

MIT
