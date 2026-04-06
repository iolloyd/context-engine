# ADR 0001 — Knowledge source format

**Status**: accepted
**Date**: 2026-04-07

## Context

The engine needs an authoritative source of truth for nodes and edges that
humans can author, review, and version. The design doc listed Obsidian as a
candidate but left the decision open. Three options were considered:

1. Obsidian vault (markdown + wikilinks + YAML frontmatter)
2. Hierarchical folder tree (one folder per node, `readme.md` for content,
   `edges.yaml` for outgoing edges)
3. SQLite as source of truth, CLI / web UI for editing

## Decision

Adopt **option 2: hierarchical folder tree**. SQLite remains the query-time
index, rebuilt from the tree. The tree is the source of truth.

## Rationale

**Identity is path-based but git-mediated.** Node ids are relative paths
(`exercises/bench-press`). Renames are `git mv` plus reference updates — a
reviewable commit, not a silent file-watcher event.

**Edges are explicit and rich.** `edges.yaml` holds the `type`, `weight`, and
`content` each `Edge` requires. Obsidian's `[[wikilink]]` cannot express
weighted, typed, annotated edges without fighting the tool.

**Git is the versioning story.** History, blame, branches, PR review, rollback
— all free. The feedback loop writes weight updates back into `edges.yaml`
so every tuning step is a reviewable commit.

**No GUI dependency.** Any text editor works. CI can import, mutate, export,
and diff the tree. Obsidian may still sit on top as a read-only vault but
does not own the data.

**Filesystem as implicit structure.** Parent folders become implicit
`part_of` edges derived at import time; named hierarchies become
`instance_of`. The `edges.yaml` file contains only semantic edges, keeping
it small and human-scannable.

## Consequences

- A folder tree ↔ SQLite importer becomes part of the core library.
- The feedback loop gains a second writer target (`edges.yaml`) in addition
  to SQLite. The tree is canonical; SQLite is rebuildable.
- Authoring tooling is BYO — a future VS Code extension or schema-aware
  editor is a nice-to-have, not a blocker.
- Round-trip tests verify that `import(export(tree)) == tree` modulo
  derived structural edges.

## Format

Directory layout::

    knowledge/
      <domain>/
        <node-slug>/
          readme.md          # content + YAML frontmatter metadata
          edges.yaml         # optional — outgoing edges

### `readme.md`

```markdown
---
type: exercise
status: avoided
domain: fitness
tags: [legs, compound]
---

The squat is a compound lower-body lift performed with a loaded barbell
on the upper back.
```

Frontmatter becomes `Node.metadata`. Body becomes `Node.content`.

### `edges.yaml`

```yaml
edges:
  - target: conditions/knee-pain
    type: because_of
    weight: 0.9
    content: Squats aggravate a chronic meniscus issue.
  - target: rules/progression
    type: if_then
    weight: 0.6
```

`target` is a path relative to `knowledge/`. `type` and `weight` are required.
`content` is optional — the "why" of the relationship.

### Rule nodes

A node with `type: rule` in its frontmatter has its body treated as the rule
definition. For the Python logic engine, the body is a fenced JSON block. A
later Prolog bridge will accept Prolog clauses.

```markdown
---
type: rule
name: progression
---

```json
{
  "predicate": "threshold",
  "args": {"metric": "bench_weight", "op": ">", "value": 65},
  "on_pass": "increase weight by 2.5kg",
  "on_fail": "maintain current weight"
}
```
```

### Derived structural edges

Import walks parent folders and synthesises a `part_of` edge from each node
to its parent folder node (if the parent also has a `readme.md`). These
edges are never written back to `edges.yaml`.

## Alternatives rejected

**Obsidian** — path-based identity plus silent renames plus weak edge typing
make it unsuitable as a source of truth. Can still read the tree as a vault.

**SQLite canonical** — no diffable history, no PR review, no rollback. Fine
for ephemeral state but wrong for knowledge that humans author and curate.
