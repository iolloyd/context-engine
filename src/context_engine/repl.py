"""Interactive REPL for inspecting and editing the context graph.

Provides live inspection of nodes and edges, weight mutation, query replay,
and step-by-step traversal tracing — all against a live ``GraphStore``.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from context_engine.store import GraphStore
from context_engine.traversal import TraceEvent, Traverser
from context_engine.types import Query

if TYPE_CHECKING:
    from context_engine.engine import ContextEngine


_COMMANDS = ("help", "show", "neighbors", "query", "set-weight", "trace", "stats", "quit")


class Repl:
    """Interactive graph editor and query replayer.

    Parameters
    ----------
    store:
        The graph store to inspect and mutate.
    engine:
        Optional ``ContextEngine`` for ``query`` and ``trace`` commands.
        Without it those two commands print a warning and no-op.
    """

    def __init__(self, store: GraphStore, engine: ContextEngine | None = None) -> None:
        self.store = store
        self.engine = engine
        self.console = Console()
        self._last_slice_ids: set[str] = set()

    # ── public interface ────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the REPL loop. Blocks until the user quits."""
        self.console.print(Panel("context-engine repl — type 'help' for commands"))
        while True:
            try:
                line = input("ctx> ")
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return
            if not line.strip():
                continue
            if self.execute(line):
                return

    def execute(self, line: str) -> bool:
        """Execute one REPL command.

        Returns ``True`` when the loop should exit, ``False`` otherwise.
        """
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            self.console.print(f"[red]parse error: {exc}[/red]")
            return False

        if not parts:
            return False

        cmd, *args = parts

        if cmd in ("quit", "exit", "q"):
            return True
        if cmd == "help":
            self._cmd_help()
        elif cmd == "show":
            self._cmd_show(args)
        elif cmd == "neighbors":
            self._cmd_neighbors(args)
        elif cmd == "query":
            self._cmd_query(args)
        elif cmd == "set-weight":
            self._cmd_set_weight(args)
        elif cmd == "trace":
            self._cmd_trace(args)
        elif cmd == "stats":
            self._cmd_stats()
        else:
            self.console.print(f"unknown command: {cmd}; type help for list")

        return False

    # ── command implementations ─────────────────────────────────────────────

    def _cmd_help(self) -> None:
        t = Table(show_header=True, header_style="bold")
        t.add_column("command")
        t.add_column("args")
        t.add_column("description")
        t.add_row("show", "<id>", "print node content and metadata")
        t.add_row("neighbors", "<id>", "print outgoing edges")
        t.add_row("query", '"<text>" [seed1 seed2 ...]', "run query, show diff against last slice")
        t.add_row("set-weight", "<edge_id> <value>", "update an edge weight")
        t.add_row("trace", '"<text>" [seed1 seed2 ...]', "show priority-queue expansion steps")
        t.add_row("stats", "", "print engine stats")
        t.add_row("quit", "", "exit the REPL (also: exit, q)")
        self.console.print(t)

    def _cmd_show(self, args: list[str]) -> None:
        if not args:
            self.console.print("[yellow]usage: show <node_id>[/yellow]")
            return
        node = self.store.get_node(args[0])
        if node is None:
            self.console.print(f"[red]node not found: {args[0]}[/red]")
            return
        t = Table(show_header=False)
        t.add_column("key")
        t.add_column("value")
        t.add_row("id", node.id)
        t.add_row("content", node.content)
        t.add_row("type", node.type or "")
        t.add_row("status", node.status or "")
        t.add_row("created_at", str(node.created_at))
        for k, v in node.metadata.items():
            if k not in ("type", "status"):
                t.add_row(k, str(v))
        self.console.print(t)

    def _cmd_neighbors(self, args: list[str]) -> None:
        if not args:
            self.console.print("[yellow]usage: neighbors <node_id>[/yellow]")
            return
        edges = self.store.out_edges(args[0])
        if not edges:
            self.console.print(f"no outgoing edges from {args[0]!r}")
            return
        t = Table(show_header=True, header_style="bold")
        t.add_column("edge_id")
        t.add_column("type")
        t.add_column("target")
        t.add_column("weight")
        t.add_column("content")
        for e in edges:
            t.add_row(e.id, e.type, e.target, f"{e.weight:.3f}", e.content or "")
        self.console.print(t)

    def _cmd_query(self, args: list[str]) -> None:
        if self.engine is None:
            self.console.print("[yellow]no engine configured[/yellow]")
            return
        if not args:
            self.console.print('[yellow]usage: query "<text>" [seed1 seed2 ...][/yellow]')
            return
        text = args[0]
        explicit_refs = args[1:]
        query = Query(text=text, explicit_refs=explicit_refs)
        resp = self.engine.answer(query)
        new_ids = {n.id for n in resp.slice_.nodes}

        # Diff against previous slice
        added = new_ids - self._last_slice_ids
        removed = self._last_slice_ids - new_ids
        kept = new_ids & self._last_slice_ids

        for nid in sorted(kept):
            self.console.print(f"  {nid}", style="dim")
        for nid in sorted(added):
            self.console.print(f"+ {nid}", style="green")
        for nid in sorted(removed):
            self.console.print(f"- {nid}", style="red")

        self._last_slice_ids = new_ids

    def _cmd_set_weight(self, args: list[str]) -> None:
        if len(args) < 2:
            self.console.print("[yellow]usage: set-weight <edge_id> <value>[/yellow]")
            return
        edge_id = args[0]
        try:
            value = float(args[1])
        except ValueError:
            self.console.print(f"[red]invalid weight: {args[1]}[/red]")
            return
        self.store.update_edge_weight(edge_id, value)
        self.console.print(f"updated edge {edge_id!r} weight → {value:.4f}")

    def _cmd_trace(self, args: list[str]) -> None:
        if self.engine is None:
            self.console.print("[yellow]no engine configured[/yellow]")
            return
        if not args:
            self.console.print('[yellow]usage: trace "<text>" [seed1 seed2 ...][/yellow]')
            return
        text = args[0]
        explicit_refs = args[1:]
        query = Query(text=text, explicit_refs=explicit_refs)

        # Reproduce engine.answer() setup, but with an observer-equipped traverser.
        tuple_ = self.engine.classifier.classify(query)
        seeds = self.engine.seeds.select(query)
        strategy = self.engine.strategies.resolve(tuple_, seeds=seeds)

        events: list[TraceEvent] = []
        tracer = Traverser(self.store, strategies=self.engine.strategies, observer=events.append)
        tracer.traverse(strategy, tuple_)

        t = Table(show_header=True, header_style="bold")
        t.add_column("#", justify="right")
        t.add_column("kind")
        t.add_column("depth", justify="right")
        t.add_column("node")
        t.add_column("edge")
        t.add_column("priority")
        t.add_column("reason")
        for i, ev in enumerate(events):
            t.add_row(
                str(i),
                ev.kind,
                str(ev.depth),
                ev.node_id or "",
                ev.edge_id or "",
                f"{ev.priority:.4f}" if ev.priority is not None else "",
                ev.reason or "",
            )
        self.console.print(t)

    def _cmd_stats(self) -> None:
        if self.engine is None:
            self.console.print("[yellow]no engine configured[/yellow]")
            return
        t = Table(show_header=False)
        t.add_column("stat")
        t.add_column("value", justify="right")
        for k, v in self.engine.stats.items():
            t.add_row(k, str(v))
        self.console.print(t)
