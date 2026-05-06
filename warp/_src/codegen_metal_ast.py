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
    """Bare ``return;`` (Warp kernels never return a value)."""


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
_RE_VOID_CALL = re.compile(r"^(?P<indent>\s*)(?P<func>[\w:]+)\s*\((?P<args>.*)\)\s*;\s*$")

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
        return Return(raw=line)

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
            # column 0 regardless of nesting. We can revisit indentation
            # cosmetics after Phase 1.3 lands.
            out.append(f"for (int var_{n.iter_var} = {n.start}; var_{n.iter_var} < {n.stop}; ++var_{n.iter_var}) {{")
            _emit_into(list(n.body), out)
            out.append("}")
        elif isinstance(n, While):
            # Same column-0 convention as ``_preprocess_while_loops``.
            out.append("while (true) {")
            _emit_into(list(n.body), out)
            out.append("}")
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


_RANGE_LINE = re.compile(r"^\s*var_(\w+)\s*=\s*wp::range\s*\(\s*var_(\w+)\s*(?:,\s*var_(\w+)\s*)?\)\s*;\s*$")
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


def _match_for_opener(nodes: list[Node], i: int) -> tuple[Node, str, str, str, str, str] | None:
    """If ``nodes[i:i+4]`` is a Warp for-loop opener, return its parts.

    Returns ``(range_assign, range_var, start_expr, stop_expr, iter_var, k_suffix)``
    or ``None`` if the 4-line shape doesn't match.
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
    else:
        start_expr = f"var_{m.group(2)}"
        stop_expr = f"var_{m.group(3)}"

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
    return n0, range_var, start_expr, stop_expr, iter_var, k_suffix


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
            range_node, range_var, start_expr, stop_expr, iter_var, k_suffix = opener
            body_nodes, after = _fold_range(nodes, i + 4, end, end_label=f"end_for_{k_suffix}", skip=skip)
            # Drop every ``goto start_for_K`` in the body — both the trailing
            # one (implicit in the for opener) and any mid-body ones (Warp's
            # lowering of ``continue``). This matches the existing
            # ``_preprocess_for_loops`` behaviour, which drops them all.
            # NOTE: silently dropping ``continue`` is a pre-existing bug; we
            # preserve it here for output equivalence and can fix it in a
            # follow-up that emits ``continue;`` instead.
            body_nodes = _drop_goto(body_nodes, f"start_for_{k_suffix}")
            skip.add(range_var)
            skip.add(iter_var)
            out.append(
                For(
                    raw=range_node.raw,
                    iter_var=iter_var,
                    range_var=range_var,
                    start=start_expr,
                    stop=stop_expr,
                    body=tuple(body_nodes),
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
            range_node, range_var, start_expr, stop_expr, iter_var, k_suffix = opener
            body_nodes, after = _fold_range(nodes, i + 4, end, end_label=f"end_for_{k_suffix}", skip=skip)
            if body_nodes and isinstance(body_nodes[-1], Goto) and body_nodes[-1].target == f"start_for_{k_suffix}":
                body_nodes = body_nodes[:-1]
            skip.add(range_var)
            skip.add(iter_var)
            out.append(
                For(
                    raw=range_node.raw,
                    iter_var=iter_var,
                    range_var=range_var,
                    start=start_expr,
                    stop=stop_expr,
                    body=tuple(body_nodes),
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


def _drop_goto(body: tuple[Node, ...] | list[Node], target: str) -> list[Node]:
    """Recursively drop every ``Goto(target=...)`` in ``body``.

    Recurses into :class:`If` bodies. Inner :class:`For` and :class:`While`
    nodes are left alone — gotos belonging to them have already been
    resolved by their own fold pass.
    """
    out: list[Node] = []
    for n in body:
        if isinstance(n, Goto) and n.target == target:
            continue
        if isinstance(n, If):
            out.append(
                If(
                    raw=n.raw,
                    cond=n.cond,
                    body=tuple(_drop_goto(n.body, target)),
                    raw_open=n.raw_open,
                    raw_close=n.raw_close,
                )
            )
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


def fold_views(nodes: list[Node], adj) -> tuple[list[Node], set[str]]:
    """Fold view aliases through the tree.

    Returns ``(rewritten_nodes, skip_decls)`` — local labels whose top-level
    declarations should be suppressed (the slice_t and view aliases).
    """
    arg_label_set = {a.label for a in adj.args}
    const_int_vars: dict[str, int] = {}
    for var in adj.variables:
        if var.constant is not None and isinstance(var.constant, int):
            const_int_vars[var.label] = var.constant

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
        elif isinstance(n, (If, For, While)):
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
        elif isinstance(n, (If, For, While)):
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
        elif isinstance(n, (If, For, While)):
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
        elif isinstance(n, (If, For, While)):
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
