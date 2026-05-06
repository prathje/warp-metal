# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed IR for Metal codegen — Phase 1 of the AST migration.

Warp emits its forward IR as a list of C++-like statement strings. The
existing regex-based codegen in :mod:`warp._src.codegen_metal` matches and
rewrites those strings line by line. This module parses each line once into
a small typed-node representation so downstream passes can walk a structured
tree instead of repeatedly running a 50+-pattern regex sweep over strings.

This is purely additive — the existing regex pipeline is unchanged. The
parser/emitter pair is exercised via a round-trip test that asserts
``emit(parse(lines)) == lines`` on every kernel reachable through
``mujoco_warp.step()``. Once we trust the parser end-to-end, follow-up
commits will migrate the preprocess passes onto it one at a time and
finally replace the body-emit loop in ``generate_msl_kernel``.

Design constraints:

- **Faithful to the IR.** A node carries enough information to re-emit the
  original line. We do not invent operations; we only categorize what's
  there.
- **Cheap.** One regex match per line, no fixed-point loops. The parser
  must beat the existing 50+-pattern intrinsic sweep handily.
- **Comprehensive.** Any line that appears in step()'s 80+ kernels must
  parse — unrecognized shapes are an error, not a soft fallback. If new
  patterns appear later we add them deliberately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------
# All nodes carry the original ``raw`` line (with original indentation) so
# the round-trip emitter can reproduce input bit-exactly. Subsequent passes
# will replace nodes with rewritten versions and the emitter will format the
# new shape from the typed fields directly.


@dataclass(frozen=True)
class Node:
    """Base node — every parsed statement subclasses this."""

    raw: str
    """The original IR string, including leading indentation and trailing semicolon."""


@dataclass(frozen=True)
class Comment(Node):
    """``// some text`` — emitted by Warp for source-line tracking."""


@dataclass(frozen=True)
class Pragma(Node):
    """``#line 123 "file.py"`` directives."""


@dataclass(frozen=True)
class Empty(Node):
    """Blank line — preserved for round-trip fidelity."""


@dataclass(frozen=True)
class Return(Node):
    """``return;`` or ``return <value>;``.

    ``value`` is the bare expression text when present (single-value
    returning ``@wp.func`` helpers lower to this form), or ``None`` for
    a bare return. Top-level kernels never carry a value.
    """

    value: str | None = None


@dataclass(frozen=True)
class BlockOpen(Node):
    """``if (var_C) {`` — opens a body block.

    ``cond`` is the condition variable label or expression (e.g., ``"var_29"``
    or ``"!var_12"``). The structural pass folds matching ``BlockClose``
    nodes into nested ``If``/``Else`` (Phase 1.x).
    """

    cond: str


@dataclass(frozen=True)
class BlockElse(Node):
    """``} else {`` — between an if-body and its else-body."""


@dataclass(frozen=True)
class BlockClose(Node):
    """``}`` — closes the most recent open block."""


@dataclass(frozen=True)
class Label(Node):
    """``start_for_K:;`` / ``end_for_K:;`` / ``start_while_K:;`` / ``end_while_K:;``."""

    name: str  # e.g., "start_for_0", "end_while_3"


@dataclass(frozen=True)
class Goto(Node):
    """``goto start_for_K;`` etc."""

    target: str  # e.g., "start_for_0"


@dataclass(frozen=True)
class ForIterCmp(Node):
    """``if (iter_cmp(var_X) == 0) goto end_for_K;`` — Warp's loop-end test.

    Belongs to the for-loop preamble; the structural pass eats it.
    """

    iter_var: str  # the range iterator local label, e.g., "13"
    end_label: str  # e.g., "end_for_0"


@dataclass(frozen=True)
class WhileCondTest(Node):
    """``if ((var_C) == false) goto end_while_K;`` — Warp's while-loop test."""

    cond_var: str  # the condition local label
    end_label: str


@dataclass(frozen=True)
class Tid(Node):
    """``builtin_tid1d(var_X);`` / ``builtin_tid2d(var_X, var_Y);`` etc.

    ``arity`` is 1, 2, or 3; ``targets`` are the destination variable labels.
    """

    arity: int
    targets: tuple[str, ...]


@dataclass(frozen=True)
class Assign(Node):
    """``var_X = <rhs>;`` — an assignment.

    The RHS is a typed :class:`Expr`. For Phase 1 the parser categorizes the
    RHS just enough for downstream passes to recognize the operation; the
    expression's textual form is preserved in the node so an unmodified emit
    is straightforward.
    """

    lhs: str  # variable label being assigned to
    expr: Expr


@dataclass(frozen=True)
class VoidCall(Node):
    """A statement that calls a function for side effects (no LHS).

    Covers ``wp::array_store(...)``, ``wp::store(...)``, the 2-arg/3-arg
    in-place forms (``wp::*_inplace(...)``), ``printf(...)``, ``wp::assign(...)``,
    and user-function calls without an assignment.
    """

    op: str  # e.g., "array_store", "store", "assign_inplace", "printf", "user_call"
    args: tuple[str, ...]
    extra: tuple[tuple[str, Any], ...] = ()
    """For ``op == "user_call"``, holds ``(("name", "<func_name>"),)``."""


# ---------------------------------------------------------------------------
# Structured nodes — produced by the fold pass (Phase 1.1)
# ---------------------------------------------------------------------------
# These don't appear after :func:`parse`; they only exist after :func:`fold`
# has paired matching brace tokens and recognised for/while preambles.


@dataclass(frozen=True)
class If(Node):
    """An ``if`` block.

    Warp's IR lowers Python ``if/else`` to two separate ``if`` statements
    (the second on the negated condition), so we don't carry an else-arm.
    The condition is the bare expression text (e.g., ``"var_29"`` or
    ``"!var_12"``). ``raw_open`` and ``raw_close`` are the literal lines
    that bracket the block in the input — preserved verbatim so emission
    keeps the original indent.
    """

    cond: str
    body: tuple[Node, ...]
    raw_open: str
    raw_close: str


@dataclass(frozen=True)
class For(Node):
    """A dynamic-range ``for`` loop.

    Folded from Warp's 4-line goto-based opener
    (``wp::range`` / ``start_for_K:;`` / ``iter_cmp`` / ``iter_next``) plus
    the trailing ``goto start_for_K`` / ``end_for_K:;`` pair. ``range_var``
    is the local label of the ``wp::range_t`` value (its declaration is
    suppressed downstream — the MSL ``for`` declares the induction
    variable inline). ``start`` and ``stop`` are bare textual expressions
    (e.g., ``"0"``, ``"var_N"``, or for the 2-arg form ``"var_S"``,
    ``"var_E"``).
    """

    iter_var: str
    range_var: str
    start: str
    stop: str
    body: tuple[Node, ...]
    # Stride for the induction variable. ``"1"`` for the common 1-/2-arg
    # ``range(stop)`` / ``range(start, stop)`` forms; some kernel like
    # mujoco_warp's ``_solve_LD_sparse_fused`` use 3-arg ``range(start,
    # stop, step)`` where step is a runtime expression (e.g.
    # ``BLOCK_DIM = wp.block_dim()``).
    step: str = "1"


@dataclass(frozen=True)
class While(Node):
    """A ``while`` loop, folded from Warp's goto-based form.

    ``label_k`` is the numeric loop suffix (e.g., ``"3"`` for ``start_while_3``)
    used by the fold pass to rewrite mid-body ``goto start_while_K`` /
    ``goto end_while_K`` into :class:`Continue` / :class:`Break` and the
    canonical condition test into :class:`WhileCondBreak`.
    """

    label_k: str
    body: tuple[Node, ...]


@dataclass(frozen=True)
class Break(Node):
    """``break;`` — folded from a mid-body ``goto end_while_K``."""


@dataclass(frozen=True)
class Continue(Node):
    """``continue;`` — folded from a mid-body ``goto start_while_K``."""


@dataclass(frozen=True)
class WhileCondBreak(Node):
    """Folded form of ``if ((var_C) == false) goto end_while_K;``.

    Emitted as ``if (!var_C) { break; }`` so a structurally-rewritten
    while loop has no remaining ``goto`` references.
    """

    cond_var: str


@dataclass(frozen=True)
class _DoWhileZero(Node):
    """Synthetic ``do { body } while (0);`` wrapper used when an inlined
    function's body contains mid-function ``return`` statements that we
    rewrote to ``break`` — the wrapper makes those breaks exit the entire
    inlined region in one hop without a goto.
    """

    body: tuple[Node, ...]


@dataclass(frozen=True)
class _RawLine(Node):
    """A literal pre-formatted MSL line. Used by the inliner to emit
    variable declarations for the inlined locals (whose mangled names
    aren't in the host kernel's ``adj.variables`` table, so the standard
    declaration loop wouldn't emit them).
    """


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------
# RHS shapes for ``Assign``. We don't fully decompose composite expressions
# (operator-form arithmetic from Warp's IR is preserved as a string in
# ``Raw``); we just identify the kind so passes can dispatch.


@dataclass(frozen=True)
class Expr:
    """Base — every expression carries its raw textual form so emit works
    even if a pass hasn't been migrated to construct from typed fields yet.
    """

    raw: str


@dataclass(frozen=True)
class Var(Expr):
    """``var_X`` — a bare variable reference."""

    label: str


@dataclass(frozen=True)
class Const(Expr):
    """A literal — ``0``, ``1.0f``, ``true``, etc."""

    value: str


@dataclass(frozen=True)
class Builtin(Expr):
    """``wp::name(args)`` — a builtin call. ``name`` is the bare function
    name (e.g., ``"address"``, ``"add"``, ``"vec_t"``). For type-parametric
    builtins (``vec_t``, ``mat_t``, ``quat_t``), the type tokens are part of
    ``raw`` and are re-parsed by downstream passes that need them.
    """

    name: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class UserCall(Expr):
    """A non-builtin function call. The mangled name resolves to a
    ``warp._src.context.Function`` via the kernel's reference table —
    handled by the inliner (Phase 2).
    """

    name: str  # e.g., "_write_scalar_0"
    args: tuple[str, ...]


@dataclass(frozen=True)
class AddrOf(Expr):
    """``&(<inner>)`` — used by Warp for shape and struct-field pointers.

    ``inner_kind`` distinguishes ``"shape"`` (``&(var_arr.shape)``) from
    ``"field_arrow"`` (``&(var_struct->field)``) and ``"field_dot"``
    (``&(var_struct.field)``).
    """

    inner_kind: str
    inner_target: str  # arr/struct local label
    inner_field: str | None  # field name for struct field, else None


@dataclass(frozen=True)
class RawExpr(Expr):
    """An expression we don't decompose further — operator forms like
    ``(var_a == var_b)``, ``(var_a + var_b)``, ``!var_X``, ``var_X && var_Y``,
    or the result of an in-loop assignment alias ``var_X = var_Y``.

    Downstream passes that need to manipulate these can re-parse the raw
    string; the bulk-rewriting passes (intrinsic translation, finalize)
    operate on raw strings anyway.
    """


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Each pattern matches the FULL line (including trailing semicolon and any
# leading indentation). Patterns are checked in order — the first match
# wins. The order is important: more specific patterns must come first.

_RE_EMPTY = re.compile(r"^\s*$")
_RE_COMMENT = re.compile(r"^\s*//")
_RE_PRAGMA = re.compile(r"^\s*#")
_RE_RETURN = re.compile(r"^\s*return\s*;\s*$")
_RE_RETURN_VALUE = re.compile(r"^\s*return\s+(?P<value>.+?)\s*;\s*$")
_RE_BLOCK_OPEN = re.compile(r"^\s*if\s*\(\s*(?P<cond>.+?)\s*\)\s*\{\s*$")
_RE_BLOCK_ELSE = re.compile(r"^\s*\}\s*else\s*\{\s*$")
_RE_BLOCK_CLOSE = re.compile(r"^\s*\}\s*$")
_RE_LABEL = re.compile(r"^\s*(?P<name>(?:start|end)_(?:for|while)_\d+)\s*:\s*;\s*$")
_RE_GOTO = re.compile(r"^\s*goto\s+(?P<target>\w+)\s*;\s*$")
_RE_FOR_ITER_CMP = re.compile(
    r"^\s*if\s*\(\s*iter_cmp\s*\(\s*var_(?P<iter>\w+)\s*\)\s*==\s*0\s*\)\s*"
    r"goto\s+(?P<end>end_for_\d+)\s*;\s*$"
)
_RE_WHILE_COND = re.compile(
    r"^\s*if\s*\(\s*\(\s*var_(?P<cond>\w+)\s*\)\s*==\s*false\s*\)\s*"
    r"goto\s+(?P<end>end_while_\d+)\s*;\s*$"
)
_RE_TID = re.compile(r"^\s*builtin_tid(?P<arity>\d)d?\s*\((?P<args>[^()]*)\)\s*;\s*$")
_RE_ASSIGN = re.compile(r"^(?P<indent>\s*)var_(?P<lhs>\w+)\s*=\s*(?P<rhs>.+?)\s*;\s*$")
_RE_VOID_CALL = re.compile(r"^(?P<indent>\s*)(?P<func>[\w:]+)\s*(?P<tpl><[^()]*>)?\s*\((?P<args>.*)\)\s*;\s*$")

# RHS shape recognizers used inside Assign parsing. Each captures just enough
# to label the operation; details that downstream passes need stay in the
# raw RHS string carried by the Expr.
_RE_RHS_VAR = re.compile(r"^var_(\w+)$")
_RE_RHS_BUILTIN = re.compile(r"^wp::(\w+)\b")
_RE_RHS_USER_CALL = re.compile(r"^(\w+)\s*\(")
_RE_RHS_ADDR_SHAPE = re.compile(r"^&\s*\(\s*var_(\w+)\s*\.\s*shape\s*\)$")
_RE_RHS_ADDR_FIELD_ARROW = re.compile(r"^&\s*\(\s*var_(\w+)\s*->\s*(\w+)\s*\)$")
_RE_RHS_ADDR_FIELD_DOT = re.compile(r"^&\s*\(\s*var_(\w+)\s*\.\s*(\w+)\s*\)$")


# Constant detection: Warp emits typed literals like ``0``, ``0.0f``, ``true``.
# We only mark the obvious shapes; anything richer falls through to RawExpr.
_RE_CONST = re.compile(r"^(?:-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[uflLU]*|true|false)$")


class MetalASTParseError(ValueError):
    """Raised when a line doesn't match any known shape.

    The error carries the offending line so callers can surface it in a
    debuggable form — the right response is to add a recognizer for the new
    shape, never to silently fall back.
    """


def _split_top_level_args(s: str) -> list[str]:
    """Split a comma-separated arg list, respecting parens and angle brackets.

    ``a, b, c`` -> ``["a", "b", "c"]``
    ``foo(a, b), c`` -> ``["foo(a, b)", "c"]``
    ``vec_t<3, T>(x), y`` -> ``["vec_t<3, T>(x)", "y"]``
    """
    out: list[str] = []
    depth = 0
    angle_depth = 0
    last = 0
    for i, ch in enumerate(s):
        if ch == "(" or ch == "[" or ch == "{":
            depth += 1
        elif ch == ")" or ch == "]" or ch == "}":
            depth -= 1
        elif ch == "<":
            # Only count as angle bracket if it's part of a template ref —
            # cheap heuristic: preceded by an identifier or ``::``.
            if i > 0 and (s[i - 1].isalnum() or s[i - 1] == "_" or s[i - 1] == ":"):
                angle_depth += 1
        elif ch == ">":
            if angle_depth > 0:
                angle_depth -= 1
        elif ch == "," and depth == 0 and angle_depth == 0:
            out.append(s[last:i].strip())
            last = i + 1
    if last < len(s):
        tail = s[last:].strip()
        if tail:
            out.append(tail)
    return out


def _parse_rhs(rhs: str) -> Expr:
    """Categorize the RHS of an assignment into one of the Expr subclasses.

    The classification is shallow on purpose: we identify the shape so passes
    can dispatch (Builtin/UserCall/AddrOf/Var/Const/RawExpr), and leave any
    deeper structure encoded in the raw text. Operator-form expressions like
    ``(var_a == var_b)`` remain RawExpr — Warp's IR doesn't generate enough
    of them, and at the granularity we operate, intrinsic translation runs
    on raw strings anyway.
    """
    s = rhs.strip()

    m = _RE_RHS_VAR.match(s)
    if m:
        return Var(raw=s, label=m.group(1))

    if _RE_CONST.match(s):
        return Const(raw=s, value=s)

    m = _RE_RHS_ADDR_SHAPE.match(s)
    if m:
        return AddrOf(raw=s, inner_kind="shape", inner_target=m.group(1), inner_field=None)

    m = _RE_RHS_ADDR_FIELD_ARROW.match(s)
    if m:
        return AddrOf(raw=s, inner_kind="field_arrow", inner_target=m.group(1), inner_field=m.group(2))

    m = _RE_RHS_ADDR_FIELD_DOT.match(s)
    if m:
        return AddrOf(raw=s, inner_kind="field_dot", inner_target=m.group(1), inner_field=m.group(2))

    m = _RE_RHS_BUILTIN.match(s)
    if m:
        # Builtin: ``wp::name(args)`` or ``wp::name<types>(args)``. We capture
        # the bare name; passes that need to discriminate generic forms (vec_t,
        # mat_t, quat_t) re-parse from the raw string.
        name = m.group(1)
        # Find the opening paren of the call (if present); strip the optional
        # template parameter list between the name and ``(``.
        open_paren = s.find("(")
        if open_paren < 0 or s[-1] != ")":
            return RawExpr(raw=s)
        args = _split_top_level_args(s[open_paren + 1 : -1])
        return Builtin(raw=s, name=name, args=tuple(args))

    m = _RE_RHS_USER_CALL.match(s)
    if m and s[-1] == ")":
        # Non-``wp::`` call — user @wp.func, or a struct constructor like
        # ``StructName_<hash>()``. Treat both as UserCall; downstream passes
        # already distinguish struct-ctor (no args, name in struct table)
        # from real user calls.
        name = m.group(1)
        open_paren = s.find("(")
        args = _split_top_level_args(s[open_paren + 1 : -1])
        return UserCall(raw=s, name=name, args=tuple(args))

    return RawExpr(raw=s)


def parse_line(line: str) -> Node:
    """Parse a single Warp-IR line into a typed :class:`Node`.

    Raises :class:`MetalASTParseError` if no shape matches.
    """
    if _RE_EMPTY.match(line):
        return Empty(raw=line)
    if _RE_COMMENT.match(line):
        return Comment(raw=line)
    if _RE_PRAGMA.match(line):
        return Pragma(raw=line)
    if _RE_RETURN.match(line):
        return Return(raw=line, value=None)
    m = _RE_RETURN_VALUE.match(line)
    if m:
        return Return(raw=line, value=m.group("value"))

    m = _RE_LABEL.match(line)
    if m:
        return Label(raw=line, name=m.group("name"))

    m = _RE_FOR_ITER_CMP.match(line)
    if m:
        return ForIterCmp(raw=line, iter_var=m.group("iter"), end_label=m.group("end"))

    m = _RE_WHILE_COND.match(line)
    if m:
        return WhileCondTest(raw=line, cond_var=m.group("cond"), end_label=m.group("end"))

    m = _RE_GOTO.match(line)
    if m:
        return Goto(raw=line, target=m.group("target"))

    m = _RE_TID.match(line)
    if m:
        arity = int(m.group("arity"))
        args = [a.strip() for a in m.group("args").split(",") if a.strip()]
        targets = tuple(a[len("var_") :] if a.startswith("var_") else a for a in args)
        return Tid(raw=line, arity=arity, targets=targets)

    m = _RE_BLOCK_ELSE.match(line)
    if m:
        return BlockElse(raw=line)
    m = _RE_BLOCK_CLOSE.match(line)
    if m:
        return BlockClose(raw=line)

    m = _RE_BLOCK_OPEN.match(line)
    if m:
        return BlockOpen(raw=line, cond=m.group("cond").strip())

    m = _RE_ASSIGN.match(line)
    if m:
        rhs = m.group("rhs")
        return Assign(raw=line, lhs=m.group("lhs"), expr=_parse_rhs(rhs))

    m = _RE_VOID_CALL.match(line)
    if m:
        func_full = m.group("func")
        # Distinguish wp::name from user calls and printf.
        if func_full == "printf":
            op = "printf"
            args_raw = m.group("args")
            args = tuple(_split_top_level_args(args_raw))
            return VoidCall(raw=line, op=op, args=args, extra=())
        if func_full.startswith("wp::"):
            op = func_full[len("wp::") :]
            args_raw = m.group("args")
            args = tuple(_split_top_level_args(args_raw))
            return VoidCall(raw=line, op=op, args=args, extra=())
        # User function call without LHS.
        args = tuple(_split_top_level_args(m.group("args")))
        return VoidCall(raw=line, op="user_call", args=args, extra=(("name", func_full),))

    raise MetalASTParseError(f"unrecognised IR line: {line!r}")


def parse(lines: list[str]) -> list[Node]:
    """Parse a kernel's forward IR into a flat list of typed nodes.

    Phase 1 keeps the structure flat — control-flow tokens (BlockOpen /
    BlockClose / BlockElse / Label / Goto) appear in the list as siblings.
    A later pass folds matched if/else and for/while regions into nested
    :class:`If` / :class:`For` / :class:`While` nodes so visitors can recurse.
    """
    return [parse_line(line) for line in lines]


def emit(nodes: list[Node]) -> list[str]:
    """Re-emit a parsed node list as IR strings.

    Flat nodes emit their raw line; structured nodes emit their open/close
    pair around a recursive walk of their body.
    """
    out: list[str] = []
    _emit_into(nodes, out)
    return out


def _emit_into(nodes, out: list[str]) -> None:
    for n in nodes:
        if isinstance(n, If):
            out.append(n.raw_open)
            _emit_into(list(n.body), out)
            out.append(n.raw_close)
        elif isinstance(n, For):
            # Match the existing ``_preprocess_for_loops`` output, which
            # emits the synthetic ``for (...) {`` and matching ``}`` at
            # column 0 regardless of nesting.
            #
            # The iter_var is declared at function scope (its decl is
            # emitted by the kernel-level decl loop, not skipped) so that
            # later code outside the loop body can still reference it —
            # this matches Warp's flat IR semantics and is required by
            # mujoco_warp's iterative linesearch which mutates the iter
            # var inside an early-break path. We therefore *don't* re-
            # declare with ``int`` here; just assign in the init clause.
            iv = n.iter_var
            if n.step == "1":
                inc = f"++var_{iv}"
            else:
                inc = f"var_{iv} += {n.step}"
            out.append(f"for (var_{iv} = {n.start}; var_{iv} < {n.stop}; {inc}) {{")
            _emit_into(list(n.body), out)
            out.append("}")
        elif isinstance(n, While):
            # Same column-0 convention as ``_preprocess_while_loops``.
            out.append("while (true) {")
            _emit_into(list(n.body), out)
            out.append("}")
        elif isinstance(n, _DoWhileZero):
            # Inlined function bodies that contain mid-function returns get
            # wrapped in ``do { ... } while (0);`` so the rewritten breaks
            # exit the whole region in one hop. MSL accepts ``do/while``.
            out.append("do {")
            _emit_into(list(n.body), out)
            out.append("} while (0);")
        else:
            out.append(n.raw)


# ---------------------------------------------------------------------------
# Structural fold (Phase 1.1)
# ---------------------------------------------------------------------------
# Recognise the goto/label patterns that Warp emits for dynamic ranges and
# while loops and fold them into structured For / While nodes whose body is
# a recursively-folded list. Also pair every BlockOpen with its matching
# BlockClose / BlockElse and produce :class:`If` nodes.
#
# The fold is OUTPUT-EQUIVALENT to running ``_preprocess_for_loops`` then
# ``_preprocess_while_loops`` from :mod:`warp._src.codegen_metal`: emitting
# the folded tree produces the same lines those preprocessors do. This is
# the validation contract for Phase 1.1 — we don't reinvent the rewrite,
# just structure it.


_RANGE_LINE = re.compile(
    r"^\s*var_(\w+)\s*=\s*wp::range\s*\("
    r"\s*var_(\w+)\s*"
    r"(?:,\s*var_(\w+)\s*)?"
    r"(?:,\s*var_(\w+)\s*)?"
    r"\)\s*;\s*$"
)
_INDENT_OF = re.compile(r"^(\s*)")


def _leading_indent(s: str) -> str:
    """Return the leading whitespace of ``s``."""
    m = _INDENT_OF.match(s)
    return m.group(1) if m else ""


def fold(nodes: list[Node]) -> tuple[list[Node], set[str]]:
    """Fold for/while/if patterns into structured nodes.

    Returns ``(folded_nodes, vars_to_skip_decl)`` where the second value
    lists the local labels whose top-level declaration must be suppressed
    (the ``wp::range_t`` iterator object and the for-loop induction
    variable, both of which the synthetic ``for`` line declares inline).
    """
    skip: set[str] = set()
    folded, _ = _fold_range(nodes, 0, len(nodes), end_label=None, skip=skip)
    return folded, skip


def _match_for_opener(nodes: list[Node], i: int) -> tuple[Node, str, str, str, str, str, str] | None:
    """If ``nodes[i:i+4]`` is a Warp for-loop opener, return its parts.

    Returns ``(range_assign, range_var, start_expr, stop_expr, step_expr,
    iter_var, k_suffix)`` or ``None`` if the 4-line shape doesn't match.
    """
    if i + 3 >= len(nodes):
        return None
    n0 = nodes[i]
    if not isinstance(n0, Assign):
        return None
    m = _RANGE_LINE.match(n0.raw)
    if not m:
        return None
    range_var = m.group(1)
    if m.group(3) is None:
        start_expr = "0"
        stop_expr = f"var_{m.group(2)}"
        step_expr = "1"
    elif m.group(4) is None:
        start_expr = f"var_{m.group(2)}"
        stop_expr = f"var_{m.group(3)}"
        step_expr = "1"
    else:
        start_expr = f"var_{m.group(2)}"
        stop_expr = f"var_{m.group(3)}"
        step_expr = f"var_{m.group(4)}"

    n1 = nodes[i + 1]
    if not isinstance(n1, Label) or not n1.name.startswith("start_for_"):
        return None
    k_suffix = n1.name[len("start_for_") :]

    n2 = nodes[i + 2]
    if not isinstance(n2, ForIterCmp):
        return None
    if n2.iter_var != range_var:
        return None
    if n2.end_label != f"end_for_{k_suffix}":
        return None

    n3 = nodes[i + 3]
    if not isinstance(n3, Assign):
        return None
    # The iter_next assignment looks like ``var_Y = wp::iter_next(var_X);``.
    iter_next_match = re.match(r"^\s*var_(\w+)\s*=\s*wp::iter_next\s*\(\s*var_(\w+)\s*\)\s*;\s*$", n3.raw)
    if not iter_next_match or iter_next_match.group(2) != range_var:
        return None
    iter_var = iter_next_match.group(1)
    return n0, range_var, start_expr, stop_expr, step_expr, iter_var, k_suffix


def _fold_range(
    nodes: list[Node],
    start: int,
    end: int,
    end_label: str | None,
    skip: set[str],
) -> tuple[list[Node], int]:
    """Fold ``nodes[start:end]`` into structured form.

    ``end_label`` (when not ``None``) is the matching close label for the
    enclosing structure; encountering it stops the fold and returns the
    consumed range. The single-pass scan handles for-loops, while-loops,
    and if blocks — anything else passes through unchanged.

    Returns ``(folded_nodes, next_index)`` where ``next_index`` is the
    position immediately after the consumed range (one past ``end_label``).
    """
    out: list[Node] = []
    i = start
    while i < end:
        n = nodes[i]

        # For-loop opener (4-line shape: range / start_label / iter_cmp / iter_next).
        opener = _match_for_opener(nodes, i)
        if opener is not None:
            range_node, range_var, start_expr, stop_expr, step_expr, iter_var, k_suffix = opener
            body_nodes, after = _fold_range(nodes, i + 4, end, end_label=f"end_for_{k_suffix}", skip=skip)
            # Drop every ``goto start_for_K`` in the body — both the trailing
            # one (implicit in the for opener) and any mid-body ones (Warp's
            # lowering of ``continue``). This matches the existing
            # ``_preprocess_for_loops`` behaviour, which drops them all.
            # NOTE: silently dropping ``continue`` is a pre-existing bug; we
            # preserve it here for output equivalence and can fix it in a
            # follow-up that emits ``continue;`` instead.
            body_nodes = _drop_goto(body_nodes, f"start_for_{k_suffix}", end_target=f"end_for_{k_suffix}")
            # Suppress the ``wp::range_t`` opaque-iterator local decl —
            # MSL has no equivalent type and the synthetic ``for`` emits
            # its own state. Keep the iter_var's decl so the variable
            # stays in scope for any post-loop reads (see ``For`` emit).
            skip.add(range_var)
            out.append(
                For(
                    raw=range_node.raw,
                    iter_var=iter_var,
                    range_var=range_var,
                    start=start_expr,
                    stop=stop_expr,
                    body=tuple(body_nodes),
                    step=step_expr,
                )
            )
            i = after
            continue

        # While-loop opener (``start_while_K:;``).
        if isinstance(n, Label) and n.name.startswith("start_while_"):
            k_suffix = n.name[len("start_while_") :]
            body_nodes, after = _fold_range(nodes, i + 1, end, end_label=f"end_while_{k_suffix}", skip=skip)
            # Rewrite mid-body break/continue/cond-test into structural forms
            # so the resulting While body has no remaining ``goto`` references
            # to this loop's labels.
            body_nodes = _rewrite_while_body(body_nodes, k_suffix)
            out.append(
                While(
                    raw=n.raw,
                    label_k=k_suffix,
                    body=tuple(body_nodes),
                )
            )
            i = after
            continue

        # Stop fold if we hit the enclosing structure's close label.
        if end_label is not None and isinstance(n, Label) and n.name == end_label:
            return out, i + 1

        # If-block (no else — Warp lowers ``if/else`` to two separate ifs).
        if isinstance(n, BlockOpen):
            body_nodes, after = _fold_if_body(nodes, i + 1, end, skip)
            close_node = nodes[after]
            if not isinstance(close_node, BlockClose):
                raise MetalASTParseError(f"if-body terminated by unexpected node {type(close_node).__name__}")
            out.append(
                If(
                    raw=n.raw,
                    cond=n.cond,
                    body=tuple(body_nodes),
                    raw_open=n.raw,
                    raw_close=close_node.raw,
                )
            )
            i = after + 1
            continue

        out.append(n)
        i += 1

    if end_label is not None:
        raise MetalASTParseError(f"unterminated structure: expected label {end_label!r} before end of input")
    return out, end


def _fold_if_body(nodes: list[Node], start: int, end: int, skip: set[str]) -> tuple[list[Node], int]:
    """Fold an if-body, stopping at the matching BlockClose.

    Recurses into nested for/while/if. Returns the folded body and the
    index of the terminating BlockClose.
    """
    out: list[Node] = []
    i = start
    while i < end:
        n = nodes[i]
        if isinstance(n, BlockClose):
            return out, i

        opener = _match_for_opener(nodes, i)
        if opener is not None:
            range_node, range_var, start_expr, stop_expr, step_expr, iter_var, k_suffix = opener
            body_nodes, after = _fold_range(nodes, i + 4, end, end_label=f"end_for_{k_suffix}", skip=skip)
            # Drop the trailing ``goto start_for_K`` and rewrite mid-body
            # ``goto end_for_K`` to ``break;`` (matches the top-level
            # ``_fold_range`` for-fold). Without this, for-loops folded
            # inside if-bodies would leak raw gotos.
            body_nodes = _drop_goto(body_nodes, f"start_for_{k_suffix}", end_target=f"end_for_{k_suffix}")
            # Suppress the ``wp::range_t`` opaque-iterator local decl —
            # MSL has no equivalent type and the synthetic ``for`` emits
            # its own state. Keep the iter_var's decl so the variable
            # stays in scope for any post-loop reads (see ``For`` emit).
            skip.add(range_var)
            out.append(
                For(
                    raw=range_node.raw,
                    iter_var=iter_var,
                    range_var=range_var,
                    start=start_expr,
                    stop=stop_expr,
                    body=tuple(body_nodes),
                    step=step_expr,
                )
            )
            i = after
            continue

        if isinstance(n, Label) and n.name.startswith("start_while_"):
            k_suffix = n.name[len("start_while_") :]
            body_nodes, after = _fold_range(nodes, i + 1, end, end_label=f"end_while_{k_suffix}", skip=skip)
            body_nodes = _rewrite_while_body(body_nodes, k_suffix)
            out.append(
                While(
                    raw=n.raw,
                    label_k=k_suffix,
                    body=tuple(body_nodes),
                )
            )
            i = after
            continue

        if isinstance(n, BlockOpen):
            inner, after = _fold_if_body(nodes, i + 1, end, skip)
            close_node = nodes[after]
            if not isinstance(close_node, BlockClose):
                raise MetalASTParseError(f"if-body terminated by unexpected node {type(close_node).__name__}")
            out.append(
                If(
                    raw=n.raw,
                    cond=n.cond,
                    body=tuple(inner),
                    raw_open=n.raw,
                    raw_close=close_node.raw,
                )
            )
            i = after + 1
            continue

        out.append(n)
        i += 1

    raise MetalASTParseError("unterminated if body: hit end of input before }")


def _drop_goto(body: tuple[Node, ...] | list[Node], target: str, end_target: str | None = None) -> list[Node]:
    """Recursively drop ``Goto(start_target)`` and rewrite ``Goto(end_target)``
    to ``break;``.

    ``target`` is the loop's start label (``start_for_K``); gotos to it are
    Warp's lowering of ``continue`` plus the implicit trailing one and we
    drop them to match the existing for-loop fold behaviour.
    ``end_target`` (when supplied) is the loop's end label
    (``end_for_K``); gotos to it are Warp's lowering of an early
    ``break``-style exit (e.g. the conditional terminator that mujoco_warp's
    iterative linesearch emits) and we rewrite each one to ``break;`` so
    the iter_var stays in scope and the body re-uses the structured exit.

    Recurses into :class:`If` bodies. Inner :class:`For` and :class:`While`
    nodes are left alone — gotos belonging to them have already been
    resolved by their own fold pass.
    """
    out: list[Node] = []
    for n in body:
        if isinstance(n, Goto) and n.target == target:
            continue
        if isinstance(n, Goto) and end_target is not None and n.target == end_target:
            indent = _leading_indent(n.raw)
            out.append(Break(raw=f"{indent}break;"))
            continue
        if isinstance(n, If):
            out.append(
                If(
                    raw=n.raw,
                    cond=n.cond,
                    body=tuple(_drop_goto(n.body, target, end_target)),
                    raw_open=n.raw_open,
                    raw_close=n.raw_close,
                )
            )
            continue
        out.append(n)
    return out


# ---------------------------------------------------------------------------
# Function inlining (Phase 2)
# ---------------------------------------------------------------------------
# Some kernels (notably ``_limit_pos`` and the other sensor-write kernels in
# mujoco_warp) write their output only by passing it to a ``@wp.func`` user
# helper. The output-array detector in ``generate_msl_kernel`` scans for
# direct write ops (``wp::array_store`` / ``wp::atomic_*`` / our synthetic
# scalar-store), so without inlining those kernels are rejected as having no
# outputs. Inlining splices the helper's body into the call site so the
# downstream pipeline sees the writes directly.
#
# What we do:
#   1. Walk the AST. When we find a user-function call node:
#      a. Resolve the called overload via the kernel's reference table.
#      b. Build the callee's IR (forward only).
#      c. Mangle every callee local label to ``<inline_id>__<orig>`` so it
#         can't collide with caller locals or other inlined calls.
#      d. Substitute callee parameter names with the caller's actual arg
#         expressions (a textual sub on each line).
#      e. Parse the substituted lines, structurally fold them.
#      f. Recurse: inline any nested user calls in the same way.
#      g. If the body contains any ``return``, wrap it in ``do { ... }
#         while (0)`` and rewrite each ``return`` to ``break``. MSL accepts
#         ``do/while``.
#   2. Splice the resulting node list at the call site.
#
# Output equivalence is no longer the test for this pass — there is no
# "existing pipeline" to compare against (function-call kernels are
# rejected today). Correctness is verified end-to-end by the launch tests:
# inlined kernels run on Metal and produce the same answers as on CPU.


_INLINE_ID_COUNTER = [0]


def _next_inline_id() -> int:
    _INLINE_ID_COUNTER[0] += 1
    return _INLINE_ID_COUNTER[0]


def _substitute_var_refs(line: str, subs: dict[str, str]) -> str:
    """Replace every ``var_<X>`` or ``ret_<i>`` token in ``line`` per ``subs``.

    Uses ``\\b`` word boundaries so we substitute whole identifiers only —
    no mid-identifier matches and no recursion into the replacement value.
    """
    if not subs:
        return line

    def repl(m: re.Match[str]) -> str:
        full = m.group(0)
        return subs.get(full, full)

    # Match both forms in one sweep so ``ret_0 = var_x;`` translates
    # correctly. Order doesn't matter because each match is local.
    return re.sub(r"\b(?:var_|ret_)\w+\b", repl, line)


def _build_function_overload_table(adj) -> dict[str, Any]:
    """Map mangled call-site names (e.g. ``"_write_scalar_0"``) back to the
    specialized ``Function`` overload whose adj we can build and inline.

    Each overload exposes its full mangled name as ``native_func`` (set
    when the overload is registered). The IR call-site name comes from
    that field directly, so we use it as the lookup key.

    Pulls candidates from two sources:
      1. ``adj.get_references()[2]`` — Warp's own table of directly-
         referenced user functions.
      2. ``adj.func.__globals__`` — every Function-typed name visible in
         the kernel's module. Some kernels (notably JIT-defined inner
         kernels like ``_primitive_narrowphase__locals__primitive_narrowphase``)
         call functions that aren't in the ``get_references`` table even
         though they're emitted in the IR; we want to inline those too.
    """
    from warp._src.context import Function  # noqa: PLC0415

    out: dict[str, Any] = {}

    refs = adj.get_references()
    fn_table = refs[2]
    for fn in fn_table:
        for overload in fn.user_overloads.values():
            native = getattr(overload, "native_func", None)
            if native is not None:
                out[native] = overload

    func = getattr(adj, "func", None)
    globals_dict = getattr(func, "__globals__", None) if func is not None else None
    if globals_dict is not None:
        for value in globals_dict.values():
            if not isinstance(value, Function):
                continue
            for overload in value.user_overloads.values():
                native = getattr(overload, "native_func", None)
                if native is not None and native not in out:
                    out[native] = overload

    return out


def _has_return(nodes: tuple[Node, ...] | list[Node]) -> bool:
    for n in nodes:
        if isinstance(n, Return):
            return True
        if isinstance(n, If) and _has_return(n.body):
            return True
        if isinstance(n, (For, While)) and _has_return(n.body):
            # Returns inside loops still need wrapping.
            return True
    return False


def _rewrite_returns_to_breaks(
    nodes: tuple[Node, ...] | list[Node],
    return_value_dst: str | None = None,
) -> list[Node]:
    """Replace ``return;`` with ``break;`` (used after wrapping in do-while-0).

    For single-value-returning functions, Warp's IR uses ``return <expr>;``
    to carry the value. ``return_value_dst`` (when set) is the bare
    variable name (without the ``var_`` prefix) where each such value
    should be assigned before the synthetic break — that's how
    ``var_X = foo(...)`` calls splice in.

    Recurses into ``If`` bodies. Does NOT descend into ``For`` / ``While``
    — a return inside an inner loop would need a separate flag-based
    rewrite (``return`` from inside a loop is uncommon enough that we
    leave it as a future-work hazard; if we hit it the resulting ``break``
    would only break the inner loop, not the do-while-0 wrapper).
    """
    out: list[Node] = []
    for n in nodes:
        if isinstance(n, Return):
            indent = _leading_indent(n.raw)
            if n.value is not None:
                if return_value_dst is None:
                    # The function returns a value but no caller LHS — drop
                    # the value (it's unobservable from the caller side)
                    # and emit just the break.
                    out.append(Break(raw=f"{indent}break;"))
                else:
                    # Synthesize ``var_<dst> = <value>;`` then ``break;``.
                    out.append(_RawLine(raw=f"{indent}var_{return_value_dst} = {n.value};"))
                    out.append(Break(raw=f"{indent}break;"))
            else:
                out.append(Break(raw=f"{indent}break;"))
            continue
        if isinstance(n, If):
            out.append(
                If(
                    raw=n.raw,
                    cond=n.cond,
                    body=tuple(_rewrite_returns_to_breaks(n.body, return_value_dst)),
                    raw_open=n.raw_open,
                    raw_close=n.raw_close,
                )
            )
            continue
        out.append(n)
    return out


# NOTE: ``_DoWhileZero`` is defined near the other structured-body node
# types (above) so every fold pass can ``isinstance`` against it without an
# import cycle.


def _inline_one_call(
    fn_overload,
    caller_args: tuple[str, ...],
    fn_map: dict[str, Any],
    depth: int,
    max_depth: int,
    const_ints_out: dict[str, int],
    struct_locals_out: dict[str, Any],
    return_value_dst: str | None = None,
) -> list[Node]:
    """Inline a single user-function call. Returns the spliced node list.

    ``const_ints_out`` is mutated to record every inlined int-typed
    constant local (mangled label → int value) so subsequent passes can
    treat them the same as ``adj.variables`` constants.

    ``struct_locals_out`` is mutated to record every inlined Struct-typed
    local (mangled label → Struct class) so the kernel-level field-
    pointer pass can resolve field accesses on inlined struct instances.

    ``return_value_dst`` (when non-None) is the bare variable name where
    each ``return <value>;`` in the callee body should write before the
    synthetic break — used for single-value-returning calls of the form
    ``var_X = foo(args);``.
    """
    if depth > max_depth:
        from warp._src.codegen_metal import MetalCodegenError  # noqa: PLC0415

        raise MetalCodegenError(
            f"function inlining exceeded max depth {max_depth} — possible recursion in {fn_overload.key!r}"
        )

    # Build the callee's IR. Pass the same default options the top-level
    # kernel build uses (see ``codegen_metal.generate_msl_kernel``) so that
    # tile-builtin value funcs (``wp.tile``, ``wp.tile_cholesky_solve``,
    # etc.) which read ``options["block_dim"]`` / ``options["output_arch"]``
    # don't crash with a KeyError when the callee is a wrapper around tile
    # primitives.
    if not getattr(fn_overload.adj, "blocks", None):
        fn_overload.adj.build(
            builder=None,
            default_builder_options={
                "enable_backward": False,
                "output_arch": None,
                "block_dim": 256,
            },
        )

    fn_lines = fn_overload.adj.blocks[0].body_forward

    # Build the substitution map.
    inline_id = _next_inline_id()
    fn_param_labels = [a.label for a in fn_overload.adj.args]
    n_params = len(fn_param_labels)
    n_caller_args = len(caller_args)

    # A value-returning ``@wp.func`` like
    #   def f(x): return a, b
    # lowers in Warp's IR to a body that writes its return values into
    # special ``ret_0``, ``ret_1``, ... locals before a bare ``return;``.
    # The call site looks like ``f_<id>(input1, ..., ret0_dst, ret1_dst);``
    # — caller args after the regular params are output slots.
    if n_caller_args < n_params:
        from warp._src.codegen_metal import MetalCodegenError  # noqa: PLC0415

        raise MetalCodegenError(f"inlining {fn_overload.key!r}: expected at least {n_params} args, got {n_caller_args}")
    n_returns = n_caller_args - n_params

    subs: dict[str, str] = {}
    for pname, caller_arg in zip(fn_param_labels, caller_args[:n_params], strict=True):
        subs[f"var_{pname}"] = caller_arg
    # Map each ret_<i> to the corresponding extra caller arg.
    for i in range(n_returns):
        subs[f"ret_{i}"] = caller_args[n_params + i]
    # Mangle locals (everything not a parameter).
    fn_param_set = set(fn_param_labels)
    for var in fn_overload.adj.variables:
        if var.label in fn_param_set:
            continue
        subs[f"var_{var.label}"] = f"var_{inline_id}__{var.label}"

    substituted_lines = [_substitute_var_refs(line, subs) for line in fn_lines]

    # Parse + structurally fold the substituted body.
    fn_nodes = parse(substituted_lines)
    fn_folded, fold_skip = fold(fn_nodes)

    # Recursively inline nested user calls.
    #
    # The callee's body may reference *other* ``@wp.func``s that the
    # kernel itself never calls directly (e.g. a passive-dynamics kernel
    # whose only @wp.func reference is a wrapper which then calls
    # ``mul_quat`` and ``quat_to_vel``). Those names won't be in the
    # kernel-level ``fn_map``, so without expansion the recursive walk
    # leaves them as undeclared identifiers in the emitted MSL. Merge in
    # the callee's own overload table — keyed by mangled native-func name
    # — before recursing.
    callee_fn_map = _build_function_overload_table(fn_overload.adj)
    if callee_fn_map:
        fn_map = {**fn_map, **callee_fn_map}
    fn_folded = _inline_walk(fn_folded, fn_map, depth + 1, max_depth, const_ints_out, struct_locals_out)

    # Record any Struct-typed locals so the kernel-level field-pointer
    # pass can resolve field accesses on inlined struct instances.
    from warp._src.codegen import Struct  # noqa: PLC0415

    for var in fn_overload.adj.variables:
        if var.label in fn_param_set:
            continue
        if isinstance(var.type, Struct):
            mangled_label = f"{inline_id}__{var.label}"
            struct_locals_out[mangled_label] = var.type

    # Emit declarations for the inlined locals. They aren't in the host
    # kernel's ``adj.variables`` table, so the standard declaration loop in
    # ``generate_msl_kernel`` doesn't see them. We emit them as raw lines
    # at the splice point.
    from warp._src.codegen_metal import _msl_constant_str, _msl_var_type  # noqa: PLC0415

    decls: list[Node] = []
    for var in fn_overload.adj.variables:
        if var.label in fn_param_set:
            continue
        # Skip locals that the for-loop fold already absorbed (range
        # iterator and induction variables are declared inline by the
        # synthetic ``for (...) {`` line).
        if var.label in fold_skip:
            continue
        # Skip ret_<i> "locals": they get substituted away by the
        # ret-to-output mapping, so no declaration needed.
        if var.label.startswith("ret_") or var.ctype() == "wp::range_t":
            continue
        try:
            msl_type = _msl_var_type(var.ctype())
        except Exception:
            # If a type isn't recognized, skip — the unsupported-intrinsic
            # guard will surface a clear error if the variable is actually
            # referenced.
            continue
        mangled_label = f"{inline_id}__{var.label}"
        mangled = f"var_{mangled_label}"
        if var.constant is None:
            decls.append(_RawLine(raw=f"    {msl_type} {mangled};"))
        else:
            decls.append(_RawLine(raw=f"    const {msl_type} {mangled} = {_msl_constant_str(var.constant)};"))
            # Record int constants so downstream passes (notably
            # fold_views' slice-step check) can recognise inlined
            # const-zeros the same as kernel-level ones.
            if isinstance(var.constant, int) and not isinstance(var.constant, bool):
                const_ints_out[mangled_label] = var.constant

    # If the body contains any return, wrap and convert to break.
    if _has_return(fn_folded):
        body_with_breaks = _rewrite_returns_to_breaks(fn_folded, return_value_dst)
        return [*decls, _DoWhileZero(raw="", body=tuple(body_with_breaks))]

    return [*decls, *fn_folded]


def _inline_walk(
    nodes: tuple[Node, ...] | list[Node],
    fn_map: dict[str, Any],
    depth: int,
    max_depth: int,
    const_ints_out: dict[str, int],
    struct_locals_out: dict[str, Any],
) -> list[Node]:
    out: list[Node] = []
    for n in nodes:
        # Void user-function call.
        if isinstance(n, VoidCall) and n.op == "user_call":
            name = dict(n.extra).get("name")
            if name in fn_map:
                spliced = _inline_one_call(
                    fn_map[name],
                    n.args,
                    fn_map,
                    depth,
                    max_depth,
                    const_ints_out,
                    struct_locals_out,
                )
                out.extend(spliced)
                continue

        # Single-value-returning user call: ``var_X = foo(args)``. The
        # callee's body uses ``return <value>;`` form; the rewriter
        # converts each one into ``var_X = value; break;`` before
        # wrapping the body in ``do { ... } while (0);``.
        if isinstance(n, Assign) and isinstance(n.expr, UserCall) and n.expr.name in fn_map:
            name = n.expr.name
            spliced = _inline_one_call(
                fn_map[name],
                n.expr.args,
                fn_map,
                depth,
                max_depth,
                const_ints_out,
                struct_locals_out,
                return_value_dst=n.lhs,
            )
            out.extend(spliced)
            continue

        # Recurse into structured bodies.
        if isinstance(n, If):
            out.append(
                If(
                    raw=n.raw,
                    cond=n.cond,
                    body=tuple(_inline_walk(n.body, fn_map, depth, max_depth, const_ints_out, struct_locals_out)),
                    raw_open=n.raw_open,
                    raw_close=n.raw_close,
                )
            )
            continue
        if isinstance(n, For):
            out.append(
                For(
                    raw=n.raw,
                    iter_var=n.iter_var,
                    range_var=n.range_var,
                    start=n.start,
                    stop=n.stop,
                    body=tuple(_inline_walk(n.body, fn_map, depth, max_depth, const_ints_out, struct_locals_out)),
                    step=n.step,
                )
            )
            continue
        if isinstance(n, While):
            out.append(
                While(
                    raw=n.raw,
                    label_k=n.label_k,
                    body=tuple(_inline_walk(n.body, fn_map, depth, max_depth, const_ints_out, struct_locals_out)),
                )
            )
            continue
        if isinstance(n, _DoWhileZero):
            out.append(
                _DoWhileZero(
                    raw=n.raw,
                    body=tuple(_inline_walk(n.body, fn_map, depth, max_depth, const_ints_out, struct_locals_out)),
                )
            )
            continue
        out.append(n)
    return out


def inline_user_calls(
    nodes: list[Node], kernel_adj, max_depth: int = 8
) -> tuple[list[Node], dict[str, int], dict[str, Any]]:
    """Splice every user-``@wp.func`` call into the AST tree.

    Returns ``(inlined_nodes, inlined_const_ints, inlined_struct_locals)``:

    - ``inlined_const_ints``: mangled label → int value for each
      inlined const-int local. Lets ``fold_views`` recognise slice-step
      constants from inside inlined bodies.

    - ``inlined_struct_locals``: mangled label → ``Struct`` class for
      each inlined struct local. Lets the kernel-level struct-field-
      pointer pass resolve field accesses on structs constructed inside
      inlined helper bodies (e.g. the ``Geom`` struct built by
      ``geom_collision_pair`` for primitive narrowphase).

    ``kernel_adj`` is the top-level kernel's ``Adjoint`` object — used
    for the references table that resolves call-site mangled names
    back to Function overloads.
    """
    fn_map = _build_function_overload_table(kernel_adj)
    if not fn_map:
        return list(nodes), {}, {}
    inlined_const_ints: dict[str, int] = {}
    inlined_struct_locals: dict[str, Any] = {}
    out = _inline_walk(
        nodes,
        fn_map,
        depth=0,
        max_depth=max_depth,
        const_ints_out=inlined_const_ints,
        struct_locals_out=inlined_struct_locals,
    )
    return out, inlined_const_ints, inlined_struct_locals


_UNSUPPORTED_CTYPE_PREFIXES = ("wp::str", "wp::tuple_t")


def fold_drop_unsupported_locals(nodes: list[Node], adj) -> tuple[list[Node], set[str]]:
    """Drop dead-code statements whose values are typed in something MSL
    can't represent.

    ``wp::str`` constants only feed ``wp.printf`` calls (diagnostic warnings
    — MSL has no usable printf in regular kernels). ``wp::tuple_t`` locals
    are constructed by ``wp.matrix(..., shape=(N,M), dtype=int)`` sugar but
    never read by the kernel body. Both can be elided entirely: drop the
    assignment lines, drop printf calls, and skip the locals' declarations.

    Mirror of ``_preprocess_drop_unsupported_locals`` from the regex
    pipeline.
    """
    drop_locals: set[str] = set()
    for var in adj.variables:
        ct = var.ctype()
        if any(ct.startswith(p) for p in _UNSUPPORTED_CTYPE_PREFIXES):
            drop_locals.add(var.label)
    skip_decls = set(drop_locals)

    if not drop_locals:
        return nodes, skip_decls

    return _apply_drop_unsupported(nodes, drop_locals), skip_decls


def _apply_drop_unsupported(nodes: tuple[Node, ...] | list[Node], drop: set[str]) -> list[Node]:
    out: list[Node] = []
    for n in nodes:
        if isinstance(n, VoidCall) and n.op == "printf":
            continue
        if isinstance(n, Assign) and n.lhs in drop:
            continue
        if isinstance(n, If):
            out.append(
                If(
                    raw=n.raw,
                    cond=n.cond,
                    body=tuple(_apply_drop_unsupported(n.body, drop)),
                    raw_open=n.raw_open,
                    raw_close=n.raw_close,
                )
            )
            continue
        if isinstance(n, For):
            out.append(
                For(
                    raw=n.raw,
                    iter_var=n.iter_var,
                    range_var=n.range_var,
                    start=n.start,
                    stop=n.stop,
                    body=tuple(_apply_drop_unsupported(n.body, drop)),
                    step=n.step,
                )
            )
            continue
        if isinstance(n, While):
            out.append(
                While(
                    raw=n.raw,
                    label_k=n.label_k,
                    body=tuple(_apply_drop_unsupported(n.body, drop)),
                )
            )
            continue
        if isinstance(n, _DoWhileZero):
            out.append(_DoWhileZero(raw=n.raw, body=tuple(_apply_drop_unsupported(n.body, drop))))
            continue
        out.append(n)
    return out


# ---------------------------------------------------------------------------
# View / slice fold (Phase 1.2a)
# ---------------------------------------------------------------------------
# Fold ``slice_t`` + ``view`` IR patterns into direct array ops on the
# underlying argument. Output-equivalent to ``_preprocess_views`` from
# :mod:`warp._src.codegen_metal`, but operates on parsed nodes and recurses
# into structured bodies so view ops nested inside For/While/If get handled
# the same as top-level ones.
#
# Supported pattern (the only one mujoco_warp uses):
#
#     var_S = wp::slice_t(i, i, 0);          // step == 0 == "integer index"
#     var_V = wp::view(arr, var_S, ...);     // arr is a kernel arg
#     ... downstream wp::address / wp::array_store / wp::atomic_* on var_V ...
#
# The downstream ops are rewritten into the same ops on ``arr``, with the
# slice's integer index(es) prepended to the index list. Slice and view
# definitions become declaration-skipped aliases.


def fold_views(nodes: list[Node], adj, extra_const_ints: dict[str, int] | None = None) -> tuple[list[Node], set[str]]:
    """Fold view aliases through the tree.

    ``extra_const_ints`` lets the caller supply additional const-int locals
    that aren't in ``adj.variables`` — used by the inliner to surface
    inlined constants so slice/view recognition works inside inlined
    bodies.

    Returns ``(rewritten_nodes, skip_decls)`` — local labels whose top-
    level declarations should be suppressed (the slice_t and view
    aliases).
    """
    arg_label_set = {a.label for a in adj.args}
    const_int_vars: dict[str, int] = {}
    for var in adj.variables:
        if var.constant is not None and isinstance(var.constant, int):
            const_int_vars[var.label] = var.constant
    if extra_const_ints:
        const_int_vars.update(extra_const_ints)

    slice_aliases: dict[str, str] = {}
    _collect_slice_aliases(nodes, slice_aliases, const_int_vars)

    view_aliases: dict[str, tuple[str, list[str]]] = {}
    _collect_view_aliases(nodes, view_aliases, slice_aliases, arg_label_set)

    skip_decls: set[str] = set(slice_aliases) | set(view_aliases)
    if not view_aliases:
        return nodes, skip_decls

    return _apply_view_rewrites(nodes, slice_aliases, view_aliases), skip_decls


def _strip_var_prefix(s: str) -> str | None:
    """``"var_X"`` -> ``"X"``; anything else -> ``None``."""
    if s.startswith("var_"):
        return s[len("var_") :]
    return None


def _collect_slice_aliases(
    nodes: tuple[Node, ...] | list[Node],
    out: dict[str, str],
    const_int_vars: dict[str, int],
) -> None:
    """Walk the tree, recording every supported ``wp::slice_t(i, i, 0)`` def.

    The supported pattern is integer-index only: start==stop AND step is a
    locally-constant zero.
    """
    for n in nodes:
        if isinstance(n, Assign) and isinstance(n.expr, Builtin) and n.expr.name == "slice_t":
            args = n.expr.args
            if len(args) == 3:
                start_l = _strip_var_prefix(args[0])
                stop_l = _strip_var_prefix(args[1])
                step_l = _strip_var_prefix(args[2])
                if start_l is not None and start_l == stop_l and step_l is not None and const_int_vars.get(step_l) == 0:
                    out[n.lhs] = start_l
        elif isinstance(n, (If, For, While, _DoWhileZero)):
            _collect_slice_aliases(n.body, out, const_int_vars)


def _collect_view_aliases(
    nodes: tuple[Node, ...] | list[Node],
    out: dict[str, tuple[str, list[str]]],
    slice_aliases: dict[str, str],
    arg_label_set: set[str],
) -> None:
    """Walk the tree, recording every supported ``wp::view(arr, slice...)``."""
    for n in nodes:
        if isinstance(n, Assign) and isinstance(n.expr, Builtin) and n.expr.name == "view":
            args = n.expr.args
            if len(args) >= 2:
                arr_l = _strip_var_prefix(args[0])
                slice_labels = [_strip_var_prefix(a) for a in args[1:]]
                if (
                    arr_l is not None
                    and arr_l in arg_label_set
                    and all(s is not None and s in slice_aliases for s in slice_labels)
                ):
                    out[n.lhs] = (arr_l, [slice_aliases[s] for s in slice_labels])
        elif isinstance(n, (If, For, While, _DoWhileZero)):
            _collect_view_aliases(n.body, out, slice_aliases, arg_label_set)


def _apply_view_rewrites(
    nodes: tuple[Node, ...] | list[Node],
    slice_aliases: dict[str, str],
    view_aliases: dict[str, tuple[str, list[str]]],
) -> list[Node]:
    out: list[Node] = []
    for n in nodes:
        # 1. Drop slice_t / view definitions we resolved as aliases.
        if (
            isinstance(n, Assign)
            and n.lhs in slice_aliases
            and isinstance(n.expr, Builtin)
            and n.expr.name == "slice_t"
        ):
            continue
        if isinstance(n, Assign) and n.lhs in view_aliases and isinstance(n.expr, Builtin) and n.expr.name == "view":
            continue

        # 2. Rewrite address / array_store / atomic_* on a view.
        if isinstance(n, Assign) and isinstance(n.expr, Builtin) and n.expr.name == "address":
            view_arr_l = _strip_var_prefix(n.expr.args[0]) if n.expr.args else None
            if view_arr_l is not None and view_arr_l in view_aliases:
                arr_name, lead_idx_labels = view_aliases[view_arr_l]
                tail = list(n.expr.args[1:])
                all_indices = [f"var_{l}" for l in lead_idx_labels] + tail
                indent = _leading_indent(n.raw)
                new_raw = f"{indent}var_{n.lhs} = wp::address(var_{arr_name}, {', '.join(all_indices)});"
                out.append(
                    Assign(
                        raw=new_raw,
                        lhs=n.lhs,
                        expr=Builtin(
                            raw=new_raw.strip().rstrip(";").split("=", 1)[1].strip(),
                            name="address",
                            args=(f"var_{arr_name}", *all_indices),
                        ),
                    )
                )
                continue

        if isinstance(n, VoidCall) and n.op == "array_store":
            view_arr_l = _strip_var_prefix(n.args[0]) if n.args else None
            if view_arr_l is not None and view_arr_l in view_aliases:
                arr_name, lead_idx_labels = view_aliases[view_arr_l]
                tail = list(n.args[1:])
                lead = [f"var_{l}" for l in lead_idx_labels]
                indent = _leading_indent(n.raw)
                new_args = (f"var_{arr_name}", *lead, *tail)
                new_raw = f"{indent}wp::array_store({', '.join(new_args)});"
                out.append(VoidCall(raw=new_raw, op="array_store", args=new_args, extra=n.extra))
                continue

        # ``wp::tile_load<dt, bc, al, R[, C]>(view, off_0[, off_1])`` with a
        # view of a higher-rank kernel arg. Rewrite to substitute the
        # underlying array name and prepend the leading slice indices —
        # downstream the ``wp::tile_load`` intrinsic regex computes the
        # flat ``base`` offset and ``row_stride`` from the array's shape.
        if (
            isinstance(n, Assign)
            and isinstance(n.expr, Builtin)
            and n.expr.name == "tile_load"
        ):
            view_arr_l = _strip_var_prefix(n.expr.args[0]) if n.expr.args else None
            if view_arr_l is not None and view_arr_l in view_aliases:
                arr_name, lead_idx_labels = view_aliases[view_arr_l]
                tail = list(n.expr.args[1:])
                lead = [f"var_{l}" for l in lead_idx_labels]
                new_args = (f"var_{arr_name}", *lead, *tail)
                tpl_match = re.search(r"wp::tile_load\s*(<[^()]*>)", n.expr.raw)
                tpl = tpl_match.group(1) if tpl_match else ""
                indent = _leading_indent(n.raw)
                new_raw = f"{indent}var_{n.lhs} = wp::tile_load{tpl}({', '.join(new_args)});"
                out.append(
                    Assign(
                        raw=new_raw,
                        lhs=n.lhs,
                        expr=Builtin(
                            raw=f"wp::tile_load{tpl}({', '.join(new_args)})",
                            name="tile_load",
                            args=new_args,
                        ),
                    )
                )
                continue

        # ``wp::tile_store<dt, bc, al>(view, off_0[, off_1], tile)`` — same
        # treatment as tile_load above, but the tile arg sits at the end
        # so we prepend leading indices before it.
        if isinstance(n, VoidCall) and n.op == "tile_store":
            view_arr_l = _strip_var_prefix(n.args[0]) if n.args else None
            if view_arr_l is not None and view_arr_l in view_aliases:
                arr_name, lead_idx_labels = view_aliases[view_arr_l]
                tail = list(n.args[1:])
                lead = [f"var_{l}" for l in lead_idx_labels]
                new_args = (f"var_{arr_name}", *lead, *tail)
                tpl_match = re.search(r"wp::tile_store\s*(<[^()]*>)", n.raw)
                tpl = tpl_match.group(1) if tpl_match else ""
                indent = _leading_indent(n.raw)
                new_raw = f"{indent}wp::tile_store{tpl}({', '.join(new_args)});"
                out.append(VoidCall(raw=new_raw, op="tile_store", args=new_args, extra=n.extra))
                continue

        if (
            isinstance(n, Assign)
            and isinstance(n.expr, Builtin)
            and n.expr.name in ("atomic_add", "atomic_sub", "atomic_min", "atomic_max")
        ):
            view_arr_l = _strip_var_prefix(n.expr.args[0]) if n.expr.args else None
            if view_arr_l is not None and view_arr_l in view_aliases:
                arr_name, lead_idx_labels = view_aliases[view_arr_l]
                tail_idx = n.expr.args[1].strip()
                val = n.expr.args[2].strip()
                all_idx = [f"var_{l}" for l in lead_idx_labels] + [tail_idx]
                # Build a flat index without outer parens (the atomic
                # intrinsic regex in codegen_metal disallows them in the
                # index slot).
                if len(all_idx) == 1:
                    flat_idx = all_idx[0]
                else:
                    terms: list[str] = []
                    for k, idx in enumerate(all_idx):
                        if k == len(all_idx) - 1:
                            terms.append(idx)
                        else:
                            stride = " * ".join(f"{arr_name}_shape[{j}]" for j in range(k + 1, len(all_idx)))
                            terms.append(f"{idx} * {stride}")
                    flat_idx = " + ".join(terms)
                indent = _leading_indent(n.raw)
                new_raw = f"{indent}var_{n.lhs} = wp::{n.expr.name}(var_{arr_name}, {flat_idx}, {val});"
                out.append(
                    Assign(
                        raw=new_raw,
                        lhs=n.lhs,
                        expr=Builtin(
                            raw=f"wp::{n.expr.name}(var_{arr_name}, {flat_idx}, {val})",
                            name=n.expr.name,
                            args=(f"var_{arr_name}", flat_idx, val),
                        ),
                    )
                )
                continue

        # 3. Recurse into structured bodies.
        if isinstance(n, If):
            out.append(
                If(
                    raw=n.raw,
                    cond=n.cond,
                    body=tuple(_apply_view_rewrites(n.body, slice_aliases, view_aliases)),
                    raw_open=n.raw_open,
                    raw_close=n.raw_close,
                )
            )
            continue
        if isinstance(n, For):
            out.append(
                For(
                    raw=n.raw,
                    iter_var=n.iter_var,
                    range_var=n.range_var,
                    start=n.start,
                    stop=n.stop,
                    body=tuple(_apply_view_rewrites(n.body, slice_aliases, view_aliases)),
                    step=n.step,
                )
            )
            continue
        if isinstance(n, While):
            out.append(
                While(
                    raw=n.raw,
                    label_k=n.label_k,
                    body=tuple(_apply_view_rewrites(n.body, slice_aliases, view_aliases)),
                )
            )
            continue
        if isinstance(n, _DoWhileZero):
            out.append(
                _DoWhileZero(
                    raw=n.raw,
                    body=tuple(_apply_view_rewrites(n.body, slice_aliases, view_aliases)),
                )
            )
            continue

        out.append(n)
    return out


# ---------------------------------------------------------------------------
# Indexref-write fold (Phase 1.2b)
# ---------------------------------------------------------------------------
# Fold ``address(vec_arr, i, j) + indexref(addr, k) + store(ptr, val)`` into
# a synthetic ``wp::__metal_scalar_store__`` token that the body emitter in
# :mod:`warp._src.codegen_metal` recognises and lowers to a direct flat-
# offset subscript write. Used by kernels that mutate a single vec component
# of an output (``out[i, j][k] = val`` in user code).
#
# Output-equivalent to ``_preprocess_indexref_writes`` from the regex
# pipeline.


_STORE_OPS = ("store", "assign_inplace", "add_inplace", "sub_inplace", "mul_inplace", "div_inplace")
_ATOMIC_OPS = ("atomic_add", "atomic_sub", "atomic_min", "atomic_max")
_ATOMIC_TO_MSL = {
    "atomic_add": "atomic_fetch_add_explicit",
    "atomic_sub": "atomic_fetch_sub_explicit",
    "atomic_min": "atomic_fetch_min_explicit",
    "atomic_max": "atomic_fetch_max_explicit",
}


def fold_multidim_atomics(
    nodes: list[Node],
    adj,
    vec_arr_info: dict[str, tuple[int, str]],
) -> list[Node]:
    """Flatten multi-dim ``wp::atomic_<op>(arr, i, j, ..., val)`` calls.

    Warp's IR for ``wp.atomic_add(arr2d, i, j, val)`` lowers to a 4-arg
    builtin call (one per index dim plus the value). The downstream
    intrinsic regex in ``codegen_metal`` only handles the 3-arg form
    (``arr, idx, val``); the 4+-arg form ends up as a malformed
    ``atomic_fetch_*_explicit`` call with too many args.

    Two cases:

      - **Scalar-dtype arr**: rewrite to 3-arg form
        ``wp::atomic_<op>(arr, flat_idx, val)`` where ``flat_idx`` is
        ``i * shape[1] * ... + j * shape[N-1] + ...``. The downstream
        regex then translates correctly.

      - **Vec-dtype arr**: MSL has no atomic on vector types, so we
        expand into N component-wise atomic ops emitted as raw
        ``atomic_fetch_<op>_explicit(&arr[idx*N + k], val[k], ...)``
        lines via :class:`_RawLine`. The Assign's LHS (which Warp's IR
        sets to the *previous* value before the atomic) is required to
        be unreferenced — we don't materialise it, since vec-atomic
        return values are rarely used and reconstructing the previous
        vec from N scalar swaps would be racy anyway.

    Mat-dtype atomic isn't seen in the kernels we cover; if encountered,
    it falls through and surfaces a clear error in the unsupported-
    intrinsic guard.
    """
    arg_label_set = {a.label for a in adj.args}
    return _apply_multidim_atomics(nodes, vec_arr_info, arg_label_set)


def _apply_multidim_atomics(
    nodes: tuple[Node, ...] | list[Node],
    vec_arr_info: dict[str, tuple[int, str]],
    arg_label_set: set[str],
) -> list[Node]:
    out: list[Node] = []
    for n in nodes:
        # 3-arg atomic on a vec-typed array: ``wp::atomic_<op>(arr, idx, vec_val)``.
        # MSL has no atomic op on vector types, so this must expand to N
        # per-component scalar atomics regardless of how many user-visible
        # dims the original call had. (Multi-dim calls reach this branch via
        # ``_apply_view_rewrites`` which already pre-flattens the index.)
        if (
            isinstance(n, Assign)
            and isinstance(n.expr, Builtin)
            and n.expr.name in _ATOMIC_OPS
            and len(n.expr.args) == 3
        ):
            arr_arg = n.expr.args[0]
            arr_label = _strip_var_prefix(arr_arg)
            if arr_label is not None and arr_label in arg_label_set and arr_label in vec_arr_info:
                flat_idx = n.expr.args[1]
                value = n.expr.args[2]
                indent = _leading_indent(n.raw)
                vec_n, _ = vec_arr_info[arr_label]
                msl_name = _ATOMIC_TO_MSL[n.expr.name]
                base = f"({flat_idx}) * {vec_n}"
                for k in range(vec_n):
                    out.append(
                        _RawLine(
                            raw=(f"{indent}{msl_name}(&{arr_label}[{base} + {k}], {value}[{k}], memory_order_relaxed);")
                        )
                    )
                continue

        if (
            isinstance(n, Assign)
            and isinstance(n.expr, Builtin)
            and n.expr.name in _ATOMIC_OPS
            and len(n.expr.args) > 3
        ):
            args = n.expr.args
            arr_arg = args[0]
            arr_label = _strip_var_prefix(arr_arg)
            if arr_label is None or arr_label not in arg_label_set:
                out.append(n)
                continue
            indices = list(args[1:-1])
            value = args[-1]
            indent = _leading_indent(n.raw)

            # Build a flat index — same shape as ``_flat_index_expr`` in
            # codegen_metal, but this module doesn't import that one to
            # avoid a cycle. The ``arr_shape[k]`` references get rewritten
            # later by the ``__shapes_packed`` packer in
            # ``generate_msl_kernel``.
            n_idx = len(indices)
            if n_idx == 1:
                flat_idx = indices[0]
            else:
                terms: list[str] = []
                for k, idx in enumerate(indices):
                    if k == n_idx - 1:
                        terms.append(idx)
                    else:
                        stride = " * ".join(f"{arr_label}_shape[{j}]" for j in range(k + 1, n_idx))
                        terms.append(f"{idx} * {stride}")
                flat_idx = " + ".join(terms)

            if arr_label in vec_arr_info:
                vec_n, _ = vec_arr_info[arr_label]
                msl_name = _ATOMIC_TO_MSL[n.expr.name]
                # Per-component atomic ops. Use parens around flat_idx
                # since the per-component multiply binds tighter.
                base = f"({flat_idx}) * {vec_n}"
                for k in range(vec_n):
                    out.append(
                        _RawLine(
                            raw=(f"{indent}{msl_name}(&{arr_label}[{base} + {k}], {value}[{k}], memory_order_relaxed);")
                        )
                    )
                continue

            # Scalar-dtype arr: collapse to 3-arg form. Downstream regex
            # turns it into the right ``atomic_fetch_*_explicit`` call.
            new_raw = f"{indent}var_{n.lhs} = wp::{n.expr.name}(var_{arr_label}, {flat_idx}, {value});"
            out.append(
                Assign(
                    raw=new_raw,
                    lhs=n.lhs,
                    expr=Builtin(
                        raw=f"wp::{n.expr.name}(var_{arr_label}, {flat_idx}, {value})",
                        name=n.expr.name,
                        args=(f"var_{arr_label}", flat_idx, value),
                    ),
                )
            )
            continue

        if isinstance(n, If):
            out.append(
                If(
                    raw=n.raw,
                    cond=n.cond,
                    body=tuple(_apply_multidim_atomics(n.body, vec_arr_info, arg_label_set)),
                    raw_open=n.raw_open,
                    raw_close=n.raw_close,
                )
            )
            continue
        if isinstance(n, For):
            out.append(
                For(
                    raw=n.raw,
                    iter_var=n.iter_var,
                    range_var=n.range_var,
                    start=n.start,
                    stop=n.stop,
                    body=tuple(_apply_multidim_atomics(n.body, vec_arr_info, arg_label_set)),
                    step=n.step,
                )
            )
            continue
        if isinstance(n, While):
            out.append(
                While(
                    raw=n.raw,
                    label_k=n.label_k,
                    body=tuple(_apply_multidim_atomics(n.body, vec_arr_info, arg_label_set)),
                )
            )
            continue
        if isinstance(n, _DoWhileZero):
            out.append(
                _DoWhileZero(
                    raw=n.raw,
                    body=tuple(_apply_multidim_atomics(n.body, vec_arr_info, arg_label_set)),
                )
            )
            continue
        out.append(n)
    return out


def fold_indexref_writes(
    nodes: list[Node],
    adj,
    vec_arr_info: dict[str, tuple[int, str]],
) -> tuple[list[Node], set[str]]:
    """Fold the address/indexref/store chain into a synthetic scalar store.

    Returns ``(rewritten_nodes, skip_decls)``.
    """
    arg_label_set = {a.label for a in adj.args}

    # 1. Find ``var_X = wp::address(vec_arr, idx_args...)`` for vec-typed args.
    addr_aliases: dict[str, tuple[str, list[str]]] = {}
    _collect_vec_address_aliases(nodes, addr_aliases, arg_label_set, vec_arr_info)

    # 2. Find ``var_Y = wp::indexref(var_X, var_idx)`` whose source is a
    #    recorded vec-address alias.
    indexref_aliases: dict[str, tuple[str, list[str], str]] = {}
    referenced_addr_locals: set[str] = set()
    _collect_indexref_aliases(nodes, indexref_aliases, referenced_addr_locals, addr_aliases)

    if not indexref_aliases:
        return nodes, set()

    skip_decls = set(referenced_addr_locals) | set(indexref_aliases)
    return (
        _apply_indexref_rewrites(nodes, indexref_aliases, referenced_addr_locals, vec_arr_info),
        skip_decls,
    )


def _collect_vec_address_aliases(
    nodes: tuple[Node, ...] | list[Node],
    out: dict[str, tuple[str, list[str]]],
    arg_label_set: set[str],
    vec_arr_info: dict[str, tuple[int, str]],
) -> None:
    for n in nodes:
        if isinstance(n, Assign) and isinstance(n.expr, Builtin) and n.expr.name == "address":
            args = n.expr.args
            if len(args) >= 2:
                arr_l = _strip_var_prefix(args[0])
                if arr_l is not None and arr_l in arg_label_set and arr_l in vec_arr_info:
                    idx_labels = [_strip_var_prefix(a) for a in args[1:]]
                    if all(s is not None for s in idx_labels):
                        out[n.lhs] = (arr_l, list(idx_labels))  # type: ignore[arg-type]
        elif isinstance(n, (If, For, While, _DoWhileZero)):
            _collect_vec_address_aliases(n.body, out, arg_label_set, vec_arr_info)


def _collect_indexref_aliases(
    nodes: tuple[Node, ...] | list[Node],
    out: dict[str, tuple[str, list[str], str]],
    referenced_addr_locals: set[str],
    addr_aliases: dict[str, tuple[str, list[str]]],
) -> None:
    for n in nodes:
        if isinstance(n, Assign) and isinstance(n.expr, Builtin) and n.expr.name == "indexref":
            args = n.expr.args
            if len(args) == 2:
                addr_l = _strip_var_prefix(args[0])
                comp_l = _strip_var_prefix(args[1])
                if addr_l is not None and addr_l in addr_aliases and comp_l is not None:
                    arr_name, idx_labels = addr_aliases[addr_l]
                    out[n.lhs] = (arr_name, idx_labels, comp_l)
                    referenced_addr_locals.add(addr_l)
        elif isinstance(n, (If, For, While, _DoWhileZero)):
            _collect_indexref_aliases(n.body, out, referenced_addr_locals, addr_aliases)


def _apply_indexref_rewrites(
    nodes: tuple[Node, ...] | list[Node],
    indexref_aliases: dict[str, tuple[str, list[str], str]],
    referenced_addr_locals: set[str],
    vec_arr_info: dict[str, tuple[int, str]],
) -> list[Node]:
    out: list[Node] = []
    for n in nodes:
        # Drop the address def whose only purpose was to feed an indexref.
        if (
            isinstance(n, Assign)
            and n.lhs in referenced_addr_locals
            and isinstance(n.expr, Builtin)
            and n.expr.name == "address"
        ):
            continue
        # Drop the indexref def itself.
        if (
            isinstance(n, Assign)
            and n.lhs in indexref_aliases
            and isinstance(n.expr, Builtin)
            and n.expr.name == "indexref"
        ):
            continue
        # Replace the store-through-the-indexref-pointer with the synthetic
        # token.
        if isinstance(n, VoidCall) and n.op in _STORE_OPS and len(n.args) == 2:
            ref_l = _strip_var_prefix(n.args[0])
            if ref_l is not None and ref_l in indexref_aliases:
                arr_name, idx_labels, comp_label = indexref_aliases[ref_l]
                val = n.args[1]
                vec_n, _ = vec_arr_info[arr_name]
                idx_vars = [f"var_{lbl}" for lbl in idx_labels]
                if len(idx_vars) == 1:
                    elem_idx = idx_vars[0]
                else:
                    terms: list[str] = []
                    for k, idx in enumerate(idx_vars):
                        if k == len(idx_vars) - 1:
                            terms.append(idx)
                        else:
                            stride = " * ".join(f"{arr_name}_shape[{j}]" for j in range(k + 1, len(idx_vars)))
                            terms.append(f"{idx} * {stride}")
                    elem_idx = " + ".join(terms)
                flat_idx = f"({elem_idx}) * {vec_n} + var_{comp_label}"
                # The existing preprocessor emits this token at fixed
                # 4-space indent; mirror that for output equivalence.
                new_raw = f"    wp::__metal_scalar_store__(var_{arr_name}, {n.op}, {flat_idx}, {val});"
                out.append(
                    VoidCall(
                        raw=new_raw,
                        op="__metal_scalar_store__",
                        args=(f"var_{arr_name}", n.op, flat_idx, val),
                        extra=(),
                    )
                )
                continue

        # Recurse.
        if isinstance(n, If):
            out.append(
                If(
                    raw=n.raw,
                    cond=n.cond,
                    body=tuple(
                        _apply_indexref_rewrites(n.body, indexref_aliases, referenced_addr_locals, vec_arr_info)
                    ),
                    raw_open=n.raw_open,
                    raw_close=n.raw_close,
                )
            )
            continue
        if isinstance(n, For):
            out.append(
                For(
                    raw=n.raw,
                    iter_var=n.iter_var,
                    range_var=n.range_var,
                    start=n.start,
                    stop=n.stop,
                    body=tuple(
                        _apply_indexref_rewrites(n.body, indexref_aliases, referenced_addr_locals, vec_arr_info)
                    ),
                    step=n.step,
                )
            )
            continue
        if isinstance(n, While):
            out.append(
                While(
                    raw=n.raw,
                    label_k=n.label_k,
                    body=tuple(
                        _apply_indexref_rewrites(n.body, indexref_aliases, referenced_addr_locals, vec_arr_info)
                    ),
                )
            )
            continue
        if isinstance(n, _DoWhileZero):
            out.append(
                _DoWhileZero(
                    raw=n.raw,
                    body=tuple(
                        _apply_indexref_rewrites(n.body, indexref_aliases, referenced_addr_locals, vec_arr_info)
                    ),
                )
            )
            continue

        out.append(n)
    return out


def _rewrite_while_body(body: list[Node], k_suffix: str) -> list[Node]:
    """Rewrite this loop's break/continue/cond-test into structural forms.

    Recurses into nested :class:`If` bodies. Inner :class:`For` and
    :class:`While` nodes have already had their own labels resolved by
    their own fold pass; their bodies aren't re-traversed here.
    """
    start_target = f"start_while_{k_suffix}"
    end_target = f"end_while_{k_suffix}"
    out: list[Node] = []
    for n in body:
        if isinstance(n, Goto) and n.target == start_target:
            out.append(Continue(raw=f"{_leading_indent(n.raw)}continue;"))
            continue
        if isinstance(n, Goto) and n.target == end_target:
            out.append(Break(raw=f"{_leading_indent(n.raw)}break;"))
            continue
        if isinstance(n, WhileCondTest) and n.end_label == end_target:
            out.append(
                WhileCondBreak(
                    raw=f"{_leading_indent(n.raw)}if (!var_{n.cond_var}) {{ break; }}",
                    cond_var=n.cond_var,
                )
            )
            continue
        if isinstance(n, If):
            out.append(
                If(
                    raw=n.raw,
                    cond=n.cond,
                    body=tuple(_rewrite_while_body(list(n.body), k_suffix)),
                    raw_open=n.raw_open,
                    raw_close=n.raw_close,
                )
            )
            continue
        out.append(n)
    return out
