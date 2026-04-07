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

## Phase 9 — folder-tree knowledge source ✅

- ADR 0001 decides on the folder-tree format over Obsidian
- `FolderTreeSource` imports/exports the SQLite store
- Derived `part_of` edges synthesised from parent folders; never written back
- Fixture tree in `fixtures/tree/` used for round-trip tests
- `ctx import` / `ctx export` CLI commands

## Phase 10 — embedding backend ✅

- `Embedder` protocol in `embedding.py` with a single `embed(text) -> list[float]` method
- `HashEmbedder(dim=384)` — deterministic, offline, blake2b-based; L2-normalised;
  zero-vector for empty input; no deps beyond stdlib
- `AnthropicEmbedder` — Voyage AI HTTP endpoint via `urllib.request` (zero extra deps);
  configurable model via `CONTEXT_ENGINE_EMBED_MODEL`; lazy import
- `GraphStore.has_embedding(node_id) -> bool` helper (one-line SQL)
- `ContextEngine.__init__` gains `embedder: Embedder | None`; when present passes
  `embedder.embed` as `embed_fn` to `SeedSelector`
- `ContextEngine.index_all()` backfills embeddings for nodes without one;
  returns `(indexed, already_had)` counts
- `ctx index` CLI command: backfills embeddings, prints summary;
  `--embedder hash|anthropic` switch
- `ctx query` now wires a `HashEmbedder` by default, enabling free-form queries
  without `--seed`
- Tests: determinism, dimension, L2 norm, empty input, punctuation-agnostic
  tokenisation, different-tokens-different-vectors, `SeedSelector` returns
  squat in top-k for "Why do I avoid squats?", end-to-end free-form engine query

## Phase 11 — Prolog logic engine ✅

- `PrologLogicEngine` in `prolog_logic.py`: subprocess wrapper around SWI-Prolog
- Rule nodes with `metadata.language=prolog` are compiled to a temp `.pl` file and
  evaluated via `swipl -q -f`; JSON-dialect rule nodes are silently skipped so both
  engines can run over the same slice without conflict
- Fact schema auto-generated from `ContextSlice`: `node/3`, `edge/4`, `metadata/3`
- Multi-clause rule bodies supported via `%---` separator: helper predicates emitted
  verbatim, last chunk becomes body of `passed/0`
- `MissingFact` semantics preserved: `existence_error` from Prolog surfaced as
  `LogicResult(missing=[predicate/arity])`
- `PrologUnavailable` exception on missing `swipl` binary
- `is_available()` helper for conditional test skipping
- Drop-in replacement: `ContextEngine.logic` can be swapped to `PrologLogicEngine()`
  without touching orchestration
- Tests: fact serialisation, mocked pass/missing, real swipl threshold rule,
  helper predicate via `%---`, end-to-end `ContextEngine` swap

## Phase 12 — learned tuple → strategy mapping ✅

- `StrategyResolver` in `strategy.py` wraps the graph store and provides:
  - `bootstrap()`: seeds one strategy node per (intent, focus) pair using the
    hardcoded `strategy_for()` defaults; idempotent; returns created count
  - `resolve(tuple_, seeds, budget_override)`: lazy bootstrap on first call,
    looks up `strategy/<intent>/<focus>` node, sorts edge types by learned
    score, falls back to hardcoded table when node absent
  - `record_success/failure(tuple_, helpful, noisy)`: increments counters in
    node metadata; adjusts per-type scores (+0.05/-0.02 on success, -0.02/-0.05
    on failure); clamps to [0, 1]; adds newly learned edge types at weight 0.35
- Node layout: traversal params in `Node.content` (JSON); counters in
  `Node.metadata` (queryable via `filter_nodes` without content parsing)
- `FeedbackSignal` gains optional `query_tuple` field so callers can propagate
  tuple context through to strategy mutation without breaking existing call sites
- `FeedbackApplier` gains optional `strategies: StrategyResolver` parameter;
  calls `record_success/failure` when a signal carries `query_tuple`
- `Traverser` gains optional `strategies` parameter; uses resolver for premise-
  check tuple overrides when present, falls back to `strategy_for` otherwise
- `ContextEngine` wires `StrategyResolver` through all components and calls
  `self.strategies.resolve()` instead of `strategy_for()` in `answer()`
- Legacy `strategy_for()` function kept intact; all existing call sites and tests
  continue to work without a store argument
- Tests: 10 new tests (8 in `test_strategy.py`, 1 in `test_feedback.py`,
  1 in `test_engine.py`); full suite 69 passed + 3 skipped

## Phase 13 — progressive widen heuristic ✅

- `widen()` in `strategy.py` gains a `level` parameter (default 1 for
  backward compatibility):
  - level 0: no-op
  - level 1: 2× budget, +1 depth, weight_floor -= 0.1, edge_types preserved
  - level 2: 4× budget, +2 depth, weight_floor = 0.0, edge_types cleared
  - level ≥ 3: clamped to level 2
  - Widen applied to the original strategy; not compounding
- `StrategyResolver.downgrade(tuple_)`: permanently bumps budget (×1.5, cap 200),
  depth (+1, cap 10), weight_floor (-0.05); increments `metadata.downgrade_count`
- `ContextEngine.answer()` unified widen loop: synthesiser called at each widen
  level; breaks on first gap-free response or on reaching MAX_WIDEN_LEVEL
- `ContextEngine.__init__` gains `downgrade_threshold` parameter (default 3)
- Prometheus-style `engine.stats` dict: `queries_total`, `widens_level_1`,
  `widens_level_2`, `strategy_downgrades`
- Signature-level failure tracking via `_signature_failures`; after
  `downgrade_threshold` max-widen failures on the same (seeds, intent, focus),
  the strategy node is permanently widened via `downgrade()`
- Tests: 6 new strategy tests (widen levels 0/1/2/clamp, downgrade, downgrade
  caps) + 3 new engine tests (three-tier progression, no-widen success,
  downgrade-fires-after-threshold); full suite 78 passed + 3 skipped

## Phase 14 — graph editor REPL ✅

- `TraceEvent` dataclass added to `traversal.py`; `Traverser.__init__` gains an
  optional `observer: Callable[[TraceEvent], None]` parameter
- Events emitted inside `_bfs_from`: `seed`, `pop`, `collect`, `expand`, `done`
  (budget exhaustion); and `rule_chain` inside `_close_rule_chains`
- Observer call is a single `if` check — zero overhead when not set; no
  `TraceEvent` objects constructed unless observer is present
- New `src/context_engine/repl.py`: `Repl` class with `run()` loop and
  `execute(line) -> bool` seam; commands: `help`, `show`, `neighbors`, `query`,
  `set-weight`, `trace`, `stats`, `quit`/`exit`/`q`
- `query` command accepts `"<text>" [seed1 seed2 ...]` and diffs the new slice
  against the previous one using green `+`, red `-`, and dim unchanged lines
- `trace` command runs a fresh `Traverser` with an event collector and prints
  all events as a rich Table; output is deterministic for a fixed graph/seeds
- Pretty output via `rich.console.Console`, `rich.table.Table`,
  `rich.panel.Panel` — only these three rich primitives used
- `ctx repl` subcommand wired into `cli.py`; `Repl` import is lazy so other
  subcommands pay no startup cost
- `rich>=13.0` added to `[project].dependencies` in `pyproject.toml`
- Tests: 8 new tests in `tests/test_repl.py` (help, show, neighbors,
  set-weight, query, trace determinism, quit, unknown command);
  full suite 86 passed + 3 skipped

## Phase 15 — sentence-transformers semantic embedder ✅

- `EmbedderUnavailable` exception added to `embedding.py` for clean failure when
  optional packages are absent
- `SentenceTransformerEmbedder` wraps `sentence-transformers/all-MiniLM-L6-v2`
  (384-dim, CPU-fast, no API key); lazy import of `sentence_transformers` so the
  engine remains importable without the extra package
- `sentence-transformers>=2.7` added as `[project.optional-dependencies].embeddings`;
  install via `pip install 'context-engine[embeddings]'` or
  `pipx inject context-engine sentence-transformers`
- `default_embedder()` module-level helper: tries `SentenceTransformerEmbedder`,
  falls back to `HashEmbedder` with a one-time stderr warning; `_default_warned`
  flag suppresses duplicate warnings within a process
- `ContextEngine.__init__` calls `default_embedder()` when neither `embedder=` nor
  `embed_fn=` is supplied; no change to callers that pass an explicit embedder
- `ctx index` gains `sentence-transformers` as a choice (new default); prints which
  embedder was resolved so the user can verify the happy path
- README updated with optional-install instructions
- Tests: determinism, L2 norm, semantic quality (n1/n3 retrieved, n2 excluded),
  `EmbedderUnavailable` on missing package, `default_embedder()` fallback with
  warning, warning suppressed on second call; engine test confirms non-None embedder
  when constructed with no explicit argument; full suite 93 passed + 3 skipped

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
