# Implementation Plan

Phased, each phase produces a runnable system that passes its own tests.

## Phase 0 — scaffolding ✅

- Repo layout, `pyproject.toml`, `.gitignore`, CI workflow
- Core `types.py`: `Node`, `Edge`, `Query`, `QueryTuple`, `TraversalStrategy`,
  `ContextSlice`, `FeedbackSignal`
- `pytest` green on an empty test suite

## Phase 1 — SQLite graph store ✅

- Two tables (`nodes`, `edges`) + `node_embeddings` + `sqlite-vec` virtual table
- CRUD, filter by metadata, adjacency queries, weight updates
- Fallback to Python cosine when `sqlite-vec` is unavailable
- Tests: round-trip CRUD, metadata filter, adjacency, weight update

## Phase 2 — classifier and seed selection ✅

- `KeywordClassifier` — deterministic regex, returns `(intent, focus)`
- `SeedSelector` — explicit refs → metadata filter → vector similarity →
  conversation context
- Tests: parametrised classifier table

## Phase 3 — traversal engine ✅

- Priority-queue BFS with:
  - edge-type filter derived from focus
  - weight floor
  - depth/budget limits
  - anti-hub priority adjustment
  - recency weighting for observation nodes
  - rule-seeking boost on evaluate/act
- Per-seed budget split for compare intent
- Tests: causal retrieval, budget enforcement

## Phase 4 — rule chain closure + premise check ✅

- Post-traversal closure on `STRUCTURAL_EDGE_TYPES` from any `type=rule` node
- Closure cannot be severed by budget
- Premise check flips `(evaluate, *)` to `(retrieve, causal)` when a seed
  carries `status ∈ {avoided, blocked, deprecated}`
- Tests: Q2 (OHP deload) pattern passes with tight budget

## Phase 5 — scan intent ✅

- `Scanner` runs a metadata filter + per-entity analyser
- Returns only entities where the analyser flags interesting state
- Example: `stalled_analyser` (domain-agnostic via `metric`/`value`/`ordinal`
  metadata keys)
- Tests: Q5 (stalled exercises) passes

## Phase 6 — logic engine ✅

- `LogicEngine` with pluggable predicates
- Built-ins: `threshold`, `status_is`
- `MissingFact` exception surfaces required-but-missing nodes back to the
  orchestrator
- Tests: progression rule passes against bench observations; deload rule
  reached via rule chain

## Phase 7 — orchestration, gap detection, widen ✅

- `ContextEngine.answer()`: classify → seeds → strategy → traverse → logic →
  widen-and-retry on `MissingFact` → synthesise
- `OfflineSynthesiser` for offline tests; `Synthesiser` protocol for LLM
  backends
- `widen()` strategy escape hatch: 2× budget, +1 depth, no edge filter,
  lower weight floor
- Tests: end-to-end causal and evaluate queries

## Phase 8 — feedback loop ✅

- `FeedbackApplier.apply(signal)`:
  - helpful edges: `w ← clip(w + α·scale·δ)`
  - noisy edges: `w ← clip(w − α·scale·δ)`
  - missing nodes: bootstrap `learned` edges at `w=0.35`
- Source reliability scaling: user > logic_engine > llm
- Tests: helpful edges strengthen, learned edges get created

## Next (beyond v0.1)

1. **LLM classifier** — drop-in replacement for `KeywordClassifier`, uses
   Claude to produce `(intent, focus)` and extract explicit references from
   natural language.
2. **LLM synthesiser** — production synthesiser with gap-detection prompt:
   the model is asked whether the slice is sufficient; if not, returns a
   list of missing node ids / topics that drives a `widen` pass.
3. **Embedding backend** — wire `sentence-transformers` or Anthropic
   embeddings to populate node vectors; enables vector-based seed selection
   end-to-end.
4. **Prolog bridge** — replace `LogicEngine` predicates with a SWI-Prolog
   process so users can write real Horn clauses against graph facts.
5. **Learned tuple → strategy mapping** — store per-tuple edge-type weights
   in the graph itself (as nodes) and let the feedback loop tune them.
6. **Scan pre-computation** — materialised views of common scan analysers
   for large graphs where per-entity analysis is too slow.
7. **Widen heuristic** — count consecutive gap-detection flags on the same
   query and progressively relax filters (2×, 4×, full context).
8. **Graph editor** — CLI / REPL for inspecting nodes, edges, and replaying
   traversals to debug weight misconfiguration.
