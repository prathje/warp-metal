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

    For Phase 1 every node carries its original ``raw`` line so this round-
    trips bit-exactly. As subsequent commits replace nodes with typed
    rewritten versions, this function will format from typed fields instead.
    """
    return [n.raw for n in nodes]
