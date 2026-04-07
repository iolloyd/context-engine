"""Prolog-backed logic engine.

Drop-in replacement for ``LogicEngine`` that consults SWI-Prolog as a
subprocess. Each rule node whose metadata sets ``language: prolog`` is
evaluated against facts derived from the rest of the slice. The same
``LogicResult`` shape is returned so callers can swap engines without
touching ``ContextEngine``.

Fact schema (auto-generated from the slice)::

    node(NodeId, Type, Status).
    edge(SourceId, TargetId, EdgeType, Weight).
    metadata(NodeId, Key, Value).

Each rule node body is wrapped in a single ``passed`` predicate via::

    passed :-
        <rule body verbatim>.

Multi-clause rule bodies (helper predicates) are supported via a
``%---`` separator line. Everything before the last ``%---`` is emitted
verbatim; the last chunk becomes the body of ``passed/0``. Example::

    heavy(W) :- W > 65.
    %---
    metadata(N, metric, bench_weight), metadata(N, value, V), heavy(V)

The engine emits::

    heavy(W) :- W > 65.
    passed :- metadata(N, metric, bench_weight), metadata(N, value, V), heavy(V).

The engine asks Prolog to ``call(passed)`` for each rule, with a
``catch/3`` around it that distinguishes "rule succeeded", "rule failed
cleanly", and "missing fact" (existence_error).

No third-party deps — pure stdlib subprocess.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
from typing import Any

from context_engine.logic import LogicResult, MissingFact  # noqa: F401 — re-export
from context_engine.types import ContextSlice, Node


class PrologUnavailable(Exception):
    """Raised when the SWI-Prolog binary cannot be found or executed."""


def _pl_quote(s: str) -> str:
    """Escape a string for use as a Prolog quoted atom.

    Single-quote characters inside the atom are doubled: ``O'Brien`` →
    ``'O''Brien'``.  The surrounding single quotes are added by the caller.
    """
    return s.replace("'", "''")


def _pl_atom(s: str) -> str:
    """Return a safely-quoted Prolog atom for an arbitrary string value."""
    return f"'{_pl_quote(s)}'"


def _pl_value(v: Any) -> str | None:
    """Serialise a metadata scalar to a Prolog term.

    Returns ``None`` for list/dict values (skipped in v0.1).
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return _pl_atom(v)
    # lists and dicts are out of scope for v0.1
    return None


def _serialise_facts(slice_: ContextSlice) -> str:
    """Return Prolog source text for all facts derived from *slice_*.

    Emits one ``node/3``, one set of ``metadata/3`` facts per node, and
    one ``edge/4`` per edge.
    """
    lines: list[str] = []

    for node in slice_.nodes:
        nid = _pl_atom(node.id)
        ntype = _pl_atom(node.type) if node.type else _pl_atom("none")
        nstatus = _pl_atom(node.status) if node.status else "none"
        lines.append(f"node({nid}, {ntype}, {nstatus}).")

        for key, val in node.metadata.items():
            if key in ("type", "status"):
                # already emitted as node/3 arguments
                continue
            serialised = _pl_value(val)
            if serialised is None:
                continue
            lines.append(f"metadata({nid}, {_pl_atom(str(key))}, {serialised}).")

    for edge in slice_.edges:
        src = _pl_atom(edge.source)
        tgt = _pl_atom(edge.target)
        etype = _pl_atom(edge.type)
        weight = str(edge.weight)
        lines.append(f"edge({src}, {tgt}, {etype}, {weight}).")

    return "\n".join(lines)


def _prolog_rules(slice_: ContextSlice) -> list[Node]:
    """Return rule nodes in *slice_* that declare ``language: prolog``."""
    return [
        n
        for n in slice_.nodes
        if n.type == "rule" and n.metadata.get("language") == "prolog"
    ]


def _build_pl_source(facts: str, rule_body: str) -> str:
    """Build a complete ``.pl`` file for a single rule evaluation.

    The rule body may contain a ``%---`` separator to introduce helper
    predicates; see module docstring for details.
    """
    SEPARATOR = "%---"
    chunks = [c.strip() for c in rule_body.split(SEPARATOR)]
    # Last chunk is the body of passed/0; earlier chunks are helpers.
    helpers = chunks[:-1]
    main_body = chunks[-1]

    helper_block = "\n".join(helpers) + "\n" if helpers else ""

    goal = (
        ":- catch(\n"
        "    ( passed -> write(pass) ; write(fail) ),\n"
        "    error(existence_error(procedure, Pi), _),\n"
        "    ( write('missing('), write(Pi), write(')') )\n"
        "), nl, halt.\n"
    )

    return "\n".join([
        ":- set_prolog_flag(double_quotes, codes).",
        ":- discontiguous node/3, edge/4, metadata/3.",
        "",
        facts,
        "",
        helper_block + f"passed :-\n    {main_body}.",
        "",
        goal,
    ])


class PrologLogicEngine:
    """Logic engine that evaluates Prolog-dialect rule nodes via SWI-Prolog.

    Only nodes with ``type=rule`` and ``metadata.language=prolog`` are
    evaluated; JSON-dialect rule nodes are silently skipped, allowing both
    ``LogicEngine`` and ``PrologLogicEngine`` to run over the same slice
    without conflict.
    """

    def __init__(
        self,
        swipl_path: str = "swipl",
        timeout: float = 10.0,
    ) -> None:
        self._swipl_path = swipl_path
        self._timeout = timeout

    # ── public API ───────────────────────────────────────────────────────────

    def evaluate(self, slice_: ContextSlice) -> list[LogicResult]:
        """Evaluate every Prolog rule in *slice_* and return one result each."""
        results: list[LogicResult] = []
        facts = _serialise_facts(slice_)
        for rule_node in _prolog_rules(slice_):
            result = self._evaluate_rule(rule_node, facts)
            results.append(result)
        return results

    def is_available(self) -> bool:
        """Return ``True`` if SWI-Prolog can be invoked successfully."""
        try:
            proc = subprocess.run(
                [self._swipl_path, "--version"],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            return proc.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # ── internals ────────────────────────────────────────────────────────────

    def _evaluate_rule(self, rule_node: Node, facts: str) -> LogicResult:
        rule_body = rule_node.content.strip()
        pl_source = _build_pl_source(facts, rule_body)

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".pl", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(pl_source)
                tmp_path = tmp.name

            try:
                proc = subprocess.run(
                    [self._swipl_path, "-q", "-f", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                )
            except FileNotFoundError:
                raise PrologUnavailable(
                    f"SWI-Prolog binary not found at {self._swipl_path!r}"
                ) from None
            except subprocess.TimeoutExpired:
                return LogicResult(
                    rule_id=rule_node.id,
                    passed=False,
                    explanation="prolog timeout",
                    missing=[],
                )

            return self._parse_output(rule_node.id, proc.stdout)

        finally:
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)

    @staticmethod
    def _parse_output(rule_id: str, stdout: str) -> LogicResult:
        token = stdout.strip()
        if token == "pass":
            return LogicResult(
                rule_id=rule_id,
                passed=True,
                explanation="passed",
                missing=[],
            )
        if token == "fail":
            return LogicResult(
                rule_id=rule_id,
                passed=False,
                explanation="failed",
                missing=[],
            )
        if token.startswith("missing(") and token.endswith(")"):
            pi = token[len("missing("):-1]
            return LogicResult(
                rule_id=rule_id,
                passed=False,
                explanation=f"missing fact: {pi}",
                missing=[pi],
            )
        return LogicResult(
            rule_id=rule_id,
            passed=False,
            explanation=f"prolog parse error: {stdout!r}",
            missing=[],
        )
