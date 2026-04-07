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

## Phase 16 — incremental import with content-hash caching ✅

- `_compute_source_hash(readme_path, edges_path)` hashes the raw bytes of
  `readme.md` and (if present) `edges.yaml` via SHA-256; the digest is
  injected into `metadata['source_hash']` on every upserted node
- `FolderTreeSource.import_into` computes the hash before parsing each node;
  if the stored `source_hash` matches, the node and its explicit edges are
  skipped entirely, saving parsing and DB writes
- `ImportReport` gains three new counters: `nodes_created`, `nodes_updated`,
  `nodes_skipped_unchanged`; the `nodes` field still counts all visited nodes
- `ContextEngine.index_all` gains a second skip condition: if the node already
  has an embedding **and** `metadata['indexed_hash'] == metadata['source_hash']`,
  it is skipped without re-embedding; after embedding, `indexed_hash` is
  written back via `upsert_node`; return tuple renamed to
  `(indexed, already_fresh)` to reflect the new semantics
- `export_from` strips `source_hash` and `indexed_hash` from the on-disk
  frontmatter so they remain internal system fields rather than user-visible
  metadata
- CLI `import` output now prints the three new counters; `index` output prints
  "already fresh" instead of "already had embeddings"
- Tests: reimport skips unchanged nodes; reimport detects one edited file;
  source_hash is a 64-char hex string in every imported node's metadata;
  second `index_all()` returns `(0, N)` when all nodes are fresh;
  full suite 98 passed + 3 skipped

## Phase 17 — unified frontmatter edges + weights sidecar ✅

- `ParsedNode` gains `frontmatter_edges: list[ParsedEdge]` (default empty list)
- `parse_readme` pops the optional `edges:` key from the YAML frontmatter,
  parses it via the shared `_parse_edge_items` helper, and stores the result
  in `ParsedNode.frontmatter_edges`; the `metadata` dict never contains `edges`
- `parse_edges` refactored to delegate inner-loop parsing to `_parse_edge_items`
  (same validation, same error messages)
- `_load_weights_sidecar(root)` loads `<root>/.ctx/weights.yaml` if present;
  key format on disk is `"<source> -> <target>:<type>"`; returns a
  `dict[(source, edge_type, target), weight]`; missing or empty file returns `{}`
- `_merge_edges(frontmatter_edges, external_edges)` deduplicates by
  `(source, target, edge_type)`; frontmatter entry wins on collision
- `FolderTreeSource.import_into` restructured into two passes: (1) node upserts
  (hash-skipped when unchanged), (2) per-node edge merge + sidecar override +
  edge upsert; pass 2 always runs so sidecar changes take effect without
  touching authored files; the source hash intentionally excludes the sidecar
- `FeedbackApplier.__init__` gains optional `tree_root: Path | None`; when set,
  weight updates are persisted to `<tree_root>/.ctx/weights.yaml` in addition to
  SQLite; `_write_sidecar_entry` creates the directory if missing and preserves
  all existing entries via load-modify-write
- `ContextEngine.__init__` gains optional `tree_root: Path | None = None`,
  forwarded to `FeedbackApplier`; the `query` CLI handler auto-detects a
  `knowledge/` sibling of the db file and passes it as `tree_root`
- `ctx migrate` subcommand: walks all `edges.yaml` files in a tree, merges
  their edges into the corresponding `readme.md` frontmatter (frontmatter wins
  on duplicates), prints a diff summary; without `--yes` it is a dry-run;
  with `--yes` it writes the updated readmes and deletes the edges.yaml files
- README updated: documents `knowledge/.ctx/` gitignore entry alongside
  `.ctx.db`; documents `ctx migrate` command
- Tests: 5 new source tests (frontmatter parsed, imported, merge-wins,
  sidecar override, sidecar missing noop); 2 new feedback tests (sidecar
  written, authored files untouched); 2 new CLI tests (dry-run, --yes);
  full suite 107 passed + 3 skipped

## Phase 18 — LLM-assisted edge suggestion ✅

- `EdgeSuggestion` dataclass: `target`, `type`, `weight`, `rationale`
- `EdgeSuggester` class in `src/context_engine/edge_suggester.py`:
  - Lazy Anthropic SDK import; `is_available()` returns `False` when SDK absent or
    no API key; all failures return `[]` (never raise)
  - `suggest(target_node, candidate_nodes, target_embedding, existing_edge_types)`:
    - When `target_embedding` is provided, candidates assumed pre-sorted by knn
      similarity (caller uses `store.knn`); otherwise sorted by `updated_at`
      descending; truncated to `max_candidates`
    - Forced tool-call (`suggest_edges` tool schema) with structured output:
      `target_id`, `edge_type`, `weight`, `rationale` per suggestion
    - Suggestions referencing unknown `target_id` values are silently dropped
    - Any exception prints a warning to stderr and returns `[]`
- `append_frontmatter_edges(tree_root, node_id, edges)` added to `source.py`:
  parses the target node's `readme.md`, deduplicates edges by
  `(source, target, edge_type)` (existing entries win), serialises back to disk
  preserving all existing frontmatter keys and the full body
- `ctx suggest-edges <node_id>` CLI subcommand:
  - Loads target node from store; error if missing
  - Candidate nodes: all nodes excluding target and `type=strategy` nodes
  - If target has a stored embedding, uses `store.knn` to pre-rank candidates
    by vector similarity; otherwise all nodes passed to suggester
  - Non-interactive (default): prints a `rich` table of suggestions
  - `--interactive`: prompts `[y/n/skip]` for each suggestion; accepted ones
    written back to the node's frontmatter via `append_frontmatter_edges`;
    requires `--tree` to know where the knowledge root is
  - `--max-candidates N` (default 50): cap on how many candidates are considered
- Shell extension (`cxn suggest-edges`): out of scope for this issue — users may
  add a shell wrapper alias; see the `ctx suggest-edges` documentation in README
- Tests (6 new):
  - `test_happy_path_returns_two_suggestions`: mocked client, two valid suggestions
  - `test_invalid_target_id_filtered_out`: three raw suggestions, one invalid — two survive
  - `test_network_failure_returns_empty_and_warns`: RuntimeError → `[]` + stderr warning
  - `test_no_sdk_is_available_returns_false_and_suggest_returns_empty`: patched SDK absence
  - `test_integration_smoke`: live API, skipped without `ANTHROPIC_API_KEY`
  - `test_append_frontmatter_edges_preserves_body`: body byte-identical, frontmatter keys
    preserved, new edge present (in `test_source.py`)
  - `test_suggest_edges_non_interactive_mocked`: patched suggester, output table checked
    (in `test_cli.py`)
- Full suite: 113 passed + 4 skipped

## Phase 19 — cx garden graph health diagnostic ✅

- New `src/context_engine/garden.py`: `GardenConfig`, `Finding`, `Gardener`
- Six checks: orphans, duplicates (embedding + content cosine), dead weights,
  strategy drift, uncovered tuples, recent activity
- Added `GraphStore.get_embedding` and `GraphStore.all_edges` helpers to `store.py`
- New `ctx garden` CLI subcommand with `--dead-weight` and `--drift-rate` flags;
  renders a rich table of findings
- Tests: 11 new tests in `tests/test_garden.py`, 1 new test in `tests/test_cli.py`
- Full suite: 125 passed + 4 skipped

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
