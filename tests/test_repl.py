"""Tests for the interactive REPL (context_engine.repl).

Each test calls ``Repl.execute()`` directly — no stdin simulation needed.
Output is captured by swapping ``repl.console`` for a recording Console.
"""

from __future__ import annotations

from rich.console import Console

from context_engine.engine import ContextEngine
from context_engine.repl import Repl
from context_engine.store import GraphStore

# ── helpers ─────────────────────────────────────────────────────────────────


def _make_repl(store: GraphStore) -> tuple[Repl, Console]:
    """Return a Repl wired to a recording console."""
    engine = ContextEngine(store)
    repl = Repl(store=store, engine=engine)
    rec = Console(record=True, width=120)
    repl.console = rec
    return repl, rec


# ── tests ────────────────────────────────────────────────────────────────────


def test_help_lists_all_commands(store: GraphStore) -> None:
    repl, rec = _make_repl(store)
    repl.execute("help")
    out = rec.export_text()
    for cmd in ("show", "neighbors", "query", "set-weight", "trace", "quit", "stats"):
        assert cmd in out, f"help output missing command: {cmd!r}"


def test_show_prints_node_content(store: GraphStore) -> None:
    repl, rec = _make_repl(store)
    repl.execute("show squat")
    out = rec.export_text()
    assert "squat" in out
    assert "avoided" in out


def test_neighbors_prints_outgoing_edges(store: GraphStore) -> None:
    repl, rec = _make_repl(store)
    repl.execute("neighbors squat")
    out = rec.export_text()
    assert "because_of" in out
    assert "knee" in out


def test_set_weight_persists(store: GraphStore) -> None:
    # squat → knee edge
    edges = store.out_edges("squat")
    assert edges, "fixture must have edges from squat"
    edge = edges[0]
    repl, _ = _make_repl(store)
    result = repl.execute(f"set-weight {edge.id} 0.25")
    assert result is False
    updated = store.get_edge(edge.id)
    assert updated is not None
    assert abs(updated.weight - 0.25) < 1e-6


def test_query_runs_and_stores_last_slice(store: GraphStore) -> None:
    repl, rec = _make_repl(store)
    # Pass squat as an explicit seed so the query finds something.
    result = repl.execute('query "why avoid squats" squat')
    assert result is False
    # After a successful query the slice must be non-empty.
    assert len(repl._last_slice_ids) > 0  # noqa: SLF001


def test_trace_is_deterministic(store: GraphStore) -> None:
    engine = ContextEngine(store)

    rec1 = Console(record=True, width=120)
    repl1 = Repl(store=store, engine=engine)
    repl1.console = rec1
    repl1.execute('trace "why avoid squats" squat')
    out1 = rec1.export_text()

    rec2 = Console(record=True, width=120)
    repl2 = Repl(store=store, engine=engine)
    repl2.console = rec2
    repl2.execute('trace "why avoid squats" squat')
    out2 = rec2.export_text()

    assert out1 == out2


def test_quit_returns_true(store: GraphStore) -> None:
    repl, _ = _make_repl(store)
    assert repl.execute("quit") is True


def test_unknown_command_does_not_raise(store: GraphStore) -> None:
    repl, rec = _make_repl(store)
    result = repl.execute("nonsense")
    assert result is False
    assert "unknown command" in rec.export_text()
