# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental MSL code generator for Warp's Metal backend.

This module is a thin translator that consumes Warp's pre-emitted typed-IR
strings (``adj.blocks[0].body_forward``) and rewrites them as Metal Shading
Language. It does *not* re-walk the Python AST — it post-processes the
CUDA-flavoured C++ statements that ``codegen.py`` already produces.

Currently supported (anything else raises ``MetalCodegenError`` with a
pointer to the offending statement):
- ``wp.array`` / ``wp.array2d`` / ``wp.array3d`` of scalar dtype
  (``float16/32``, ``int8/16/32/64``, ``uint8/16/32/64``, ``bool``)
- ``wp.tid()`` for 1-, 2-, and 3-D launches (``dim`` may be int or tuple)
- N-dimensional array reads/writes via row-major flat indexing
  (``arr[i, j]`` -> ``arr[i * arr_shape[1] + j]``). MLX auto-generates
  ``<inputname>_shape`` for inputs; the launcher appends a synthetic
  ``<outputname>_shape`` argument for each multi-dim output.
- ``arr.shape[k]`` access inside the kernel body. The IR emits a chain
  ``var_X = &(var_arg.shape); var_Y = wp::load(var_X);
  var_Z = wp::extract(var_Y, k);`` with intermediate ``wp::shape_t`` /
  ``wp::shape_t*`` ctypes. The codegen aliases those locals to the
  ``<arg>_shape`` array and skips the bookkeeping lines.
- Both annotation forms: ``wp.array(dtype=...)`` (callable) and
  ``wp.array2d[float]`` (subscript). The latter produces an
  ``_ArrayAnnotation`` instance rather than a ``warp._src.types.array``,
  but both expose ``ndim`` / ``dtype`` so the codegen treats them
  uniformly.
- Scalar arithmetic intrinsics: ``add``, ``sub``, ``mul``, ``div``, ``mod``
- ``if`` / ``else`` blocks and the comparison operators ``<``, ``<=``,
  ``==``, ``!=``, ``>=``, ``>`` (Warp's IR pre-emits these in plain C/MSL
  syntax, so they pass through the regex-based translator unchanged).
- A whitelist of math builtins listed in ``_MATH_BUILTIN_NAMES`` —
  ``sqrt``, ``abs``, ``min``, ``max``, ``floor``, ``ceil``, ``exp``,
  ``log``, ``sin``, ``cos``, ``tanh``, ``clamp``, etc. — translated to
  MSL's ``metal::`` namespace.
- ``wp.where(cond, a, b)`` -> ``((cond) ? (a) : (b))`` (C-style ternary).
- ``for i in range(...)`` loops, both static (Warp unrolls them, so this
  is a no-op) and dynamic (a structural pre-pass rewrites Warp's
  ``goto``-based loop into a real MSL ``for``).
- ``while cond:`` loops with ``break`` and ``continue``. MSL rejects
  ``goto`` and labeled statements outright, so a pre-pass rewrites the
  goto-based IR as ``while (true) { ... if (!cond) break; ... continue; }``.
- ``wp::assign(dst, src)`` (in-place mutation, used by the loop body for
  accumulator updates).
- Atomic ops on output arrays — ``wp.atomic_add``, ``wp.atomic_sub``,
  ``wp.atomic_min``, ``wp.atomic_max`` — translated to MSL's
  ``atomic_fetch_*_explicit`` with relaxed ordering. The launcher sets
  ``mx.fast.metal_kernel(atomic_outputs=True)`` for kernels that use any
  atomic op, which makes *all* outputs of that kernel ``device atomic<T>*``;
  mixed atomic / non-atomic writes to outputs in the same kernel are
  rejected at codegen time.
- Vec types ``wp.vec2``/``wp.vec3``/``wp.vec4`` (and the corresponding
  int/uint variants) for sizes ``N`` in ``{2, 3, 4}`` where MSL has a
  native ``floatN`` / ``intN`` etc. type. The constructor
  (``wp.vec3(x, y, z)``) maps to ``floatN(...)``, ``[i]`` indexing maps
  to MSL subscript, and ``wp.dot``/``wp.cross``/``wp.normalize``/
  ``wp.length`` are added to the math whitelist as ``metal::*``. Reads
  and writes of ``wp.array(dtype=wp.vec3)`` are expanded into per-
  component scalar accesses (``a[i*3+0]``, ``a[i*3+1]``, ``a[i*3+2]``)
  to avoid the address-space mismatches that ``packed_floatN`` pointer
  casts hit in MLX-generated kernel wrappers. Multi-dim arrays of vec
  types are not yet supported.
- Matrix types ``wp.mat22``/``wp.mat33``/``wp.mat44`` (and the
  corresponding int variants) for ``RxC`` with ``R, C`` in ``{2, 3, 4}``
  where MSL has a native ``floatRxC`` type. ``wp.transpose`` and
  ``wp.determinant`` route to ``metal::*``. ``wp.mat33(...)`` row-major
  flat constructor is reordered into ``floatRxC(floatR(...), ...)``
  column form. ``wp::extract(m, i, j)`` becomes ``m[j][i]`` to match
  MSL's column-major indexing. Reads/writes of ``wp.array(dtype=wp.mat33)``
  scatter/gather between row-major user storage and the column-major
  MSL representation. Multi-dim arrays of mat types are not yet
  supported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class MetalCodegenError(NotImplementedError):
    """Raised when MSL codegen encounters an unsupported construct.

    Subclasses ``NotImplementedError`` so that callers can distinguish
    "incomplete backend" from genuine bugs and decide whether to fall back to
    CPU.
    """


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------

# C++ ctype string (as returned by ``Var.ctype()``) -> MSL type string.
# The values intentionally match Apple's MSL spec:
#   https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
_SCALAR_CTYPE_TO_MSL: dict[str, str] = {
    "wp::float16": "half",
    "wp::float32": "float",
    "wp::int32": "int",
    "wp::uint32": "uint",
    "wp::int64": "long",
    "wp::uint64": "ulong",
    "wp::int8": "char",
    "wp::uint8": "uchar",
    "wp::int16": "short",
    "wp::uint16": "ushort",
    # Bool is the one IR ctype without a ``wp::`` prefix — Warp emits plain
    # ``bool`` for boolean locals (e.g. the result of a comparison op).
    "wp::bool": "bool",
    "bool": "bool",
}

# MSL has built-in vector types ``floatN`` / ``intN`` / ``uintN`` etc. for
# N in {2, 3, 4}. Larger sizes (Warp's vec5, vec6, vec8) would need a custom
# struct, which we defer until they actually show up in a target kernel.
_MSL_VEC_SCALAR_PREFIX: dict[str, str] = {
    "wp::float16": "half",
    "wp::float32": "float",
    "wp::int32": "int",
    "wp::uint32": "uint",
    "wp::int16": "short",
    "wp::uint16": "ushort",
    "wp::int8": "char",
    "wp::uint8": "uchar",
}
_MSL_VEC_SUPPORTED_N = (2, 3, 4)

# Pattern that matches ``wp::vec_t<N, wp::TYPE>`` — used both to translate
# variable ctypes (declarations) and constructor expressions inside the body.
_WP_VEC_T_PAT = re.compile(r"wp::vec_t\s*<\s*(\d+)\s*,\s*wp::(\w+)\s*>")
# Pattern that matches ``wp::mat_t<R, C, wp::TYPE>``.
_WP_MAT_T_PAT = re.compile(r"wp::mat_t\s*<\s*(\d+)\s*,\s*(\d+)\s*,\s*wp::(\w+)\s*>")


def _vec_t_to_msl(n: int, scalar_ctype: str) -> str:
    """Translate ``wp::vec_t<N, wp::TYPE>`` to its MSL name (e.g. ``float3``)."""
    if n not in _MSL_VEC_SUPPORTED_N:
        raise MetalCodegenError(f"MSL codegen does not yet support vec{n} (only sizes 2, 3, 4 have native MSL types)")
    full_ctype = f"wp::{scalar_ctype}"
    if full_ctype not in _MSL_VEC_SCALAR_PREFIX:
        raise MetalCodegenError(f"MSL codegen does not yet support vec_t element type {full_ctype!r}")
    return f"{_MSL_VEC_SCALAR_PREFIX[full_ctype]}{n}"


def _mat_t_to_msl(rows: int, cols: int, scalar_ctype: str) -> str:
    """Translate ``wp::mat_t<R, C, wp::TYPE>`` to its MSL name (e.g. ``float3x3``)."""
    if rows not in _MSL_VEC_SUPPORTED_N or cols not in _MSL_VEC_SUPPORTED_N:
        raise MetalCodegenError(
            f"MSL codegen does not yet support mat{rows}x{cols} (only sizes 2, 3, 4 per dim have native MSL types)"
        )
    full_ctype = f"wp::{scalar_ctype}"
    if full_ctype not in _MSL_VEC_SCALAR_PREFIX:
        raise MetalCodegenError(f"MSL codegen does not yet support mat_t element type {full_ctype!r}")
    return f"{_MSL_VEC_SCALAR_PREFIX[full_ctype]}{rows}x{cols}"


def _translate_vec_t_in(text: str) -> str:
    """Replace every ``wp::vec_t<N, ...>`` and ``wp::mat_t<R, C, ...>`` reference
    in ``text`` with the MSL native type name (``float3``, ``float3x3``, etc.).
    Constructor calls are rewritten elsewhere (see ``_rewrite_mat_t_constructor``)
    because the row-major-to-column-major arg reorder differs from the simple
    name substitution this function does.
    """
    text = _WP_VEC_T_PAT.sub(lambda m: _vec_t_to_msl(int(m.group(1)), m.group(2)), text)
    text = _WP_MAT_T_PAT.sub(lambda m: _mat_t_to_msl(int(m.group(1)), int(m.group(2)), m.group(3)), text)
    return text


# ---------------------------------------------------------------------------
# Matrix constructor and storage layout
# ---------------------------------------------------------------------------
#
# MSL matrices are column-major: ``m[col][row]`` (so ``mat[0]`` is the first
# column as a vector). Warp stores matrices row-major in array buffers and
# the IR's ``wp::mat_t<R, C, T>(v00, v01, v02, v10, ...)`` constructor takes
# row-major flat args.
#
# Convention: keep the *logical* matrix the same on both sides. MSL column
# ``k`` corresponds to logical column ``k``. So ``m * v``, ``m1 * m2``, and
# ``transpose(m)`` all work with native MSL operators. The trade-off:
#  - The constructor must reorder row-major args into column form
#    (handled by ``_rewrite_mat_t_constructor``).
#  - ``wp::extract(m, i, j)`` (row i, col j) becomes ``m[j][i]`` in MSL.
#  - Reads from row-major array storage build columns by gathering strided
#    elements; writes scatter back the same way (handled in
#    ``generate_msl_kernel`` alongside vec arrays).


def _rewrite_mat_t_constructor(text: str) -> str:
    """Rewrite ``wp::mat_t<R, C, wp::T>(v00, v01, ..., v(R-1)(C-1))`` (row-major
    flat) into ``floatRxC(floatR(v00, v10, ...), floatR(v01, v11, ...), ...)``.

    Constructors with the wrong arg count or unsupported types are left
    untouched; the unsupported-intrinsic guard catches them downstream.
    """
    pat = re.compile(r"wp::mat_t<\s*(\d+)\s*,\s*(\d+)\s*,\s*wp::(\w+)\s*>\s*\(([^()]*)\)")

    def repl(m: re.Match[str]) -> str:
        rows, cols = int(m.group(1)), int(m.group(2))
        scalar = m.group(3)
        args_str = m.group(4)
        args = [a.strip() for a in args_str.split(",") if a.strip()]
        if len(args) != rows * cols:
            return m.group(0)
        full_ctype = f"wp::{scalar}"
        if full_ctype not in _MSL_VEC_SCALAR_PREFIX or rows not in _MSL_VEC_SUPPORTED_N:
            return m.group(0)
        msl_scalar = _MSL_VEC_SCALAR_PREFIX[full_ctype]
        msl_vec = f"{msl_scalar}{rows}"
        msl_mat = f"{msl_scalar}{rows}x{cols}"
        col_strs: list[str] = []
        for c in range(cols):
            col_components = [args[r * cols + c] for r in range(rows)]
            col_strs.append(f"{msl_vec}({', '.join(col_components)})")
        return f"{msl_mat}({', '.join(col_strs)})"

    return pat.sub(repl, text)


# Pointer ctypes have a ``*`` suffix; address-space qualifier in MSL is
# ``device`` for buffer-resident memory (the only kind we currently allocate).
_POINTER_ADDRESS_SPACE = "device"


def _msl_scalar_type(ctype: str) -> str:
    """Translate a Warp scalar / vector / matrix ctype string to its MSL equivalent.

    Raises ``MetalCodegenError`` for types MSL cannot represent natively
    (notably ``wp::float64`` — Apple Silicon GPUs have no double-precision
    floating point support).
    """
    if ctype == "wp::float64":
        raise MetalCodegenError("MSL has no native float64; double-precision kernels cannot be lowered to Metal")
    stripped = ctype.strip()
    # ``wp::vec_t<N, wp::TYPE>`` -> ``floatN`` / ``intN`` / etc.
    m = _WP_VEC_T_PAT.fullmatch(stripped)
    if m:
        return _vec_t_to_msl(int(m.group(1)), m.group(2))
    # ``wp::mat_t<R, C, wp::TYPE>`` -> ``floatRxC`` / ``intRxC`` / etc.
    m = _WP_MAT_T_PAT.fullmatch(stripped)
    if m:
        return _mat_t_to_msl(int(m.group(1)), int(m.group(2)), m.group(3))
    if ctype not in _SCALAR_CTYPE_TO_MSL:
        raise MetalCodegenError(f"MSL codegen: unsupported scalar ctype {ctype!r}")
    return _SCALAR_CTYPE_TO_MSL[ctype]


def _msl_pointer_type(ctype: str) -> str:
    assert ctype.endswith("*"), ctype
    inner = ctype[:-1].rstrip()
    return f"{_POINTER_ADDRESS_SPACE} {_msl_scalar_type(inner)}*"


def _msl_var_type(ctype: str) -> str:
    """Translate a local variable's ctype to MSL."""
    if ctype.endswith("*"):
        return _msl_pointer_type(ctype)
    return _msl_scalar_type(ctype)


def _msl_constant_str(value) -> str:
    """Format a Python scalar as an MSL literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    # ``isinstance(True, int)`` is True, so ``bool`` must come first.
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # MSL accepts the same syntax as C++; ``f`` suffix marks single
        # precision so the literal stays in fp32 register pressure.
        return f"{value!r}f"
    raise MetalCodegenError(f"MSL codegen does not yet support constant values of type {type(value).__name__}")


# ---------------------------------------------------------------------------
# Intrinsic translation
# ---------------------------------------------------------------------------

# Math builtins that translate by simple namespace rename: ``wp::sqrt(x)`` →
# ``metal::sqrt(x)``. Argument structure is preserved verbatim. Listed
# explicitly (rather than via a wildcard ``wp::(\w+)`` pattern) so that
# unsupported builtins still raise a clear ``MetalCodegenError`` instead of
# silently passing through and erroring at MSL compile time.
_MATH_BUILTIN_NAMES: tuple[str, ...] = (
    # Unary
    "abs",
    "sqrt",
    "rsqrt",
    "floor",
    "ceil",
    "round",
    "rint",
    "trunc",
    "sign",
    "exp",
    "exp2",
    "log",
    "log2",
    "log10",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "isfinite",
    "isnan",
    "isinf",
    # Binary
    "min",
    "max",
    "atan2",
    "pow",
    "copysign",
    # Vector ops (defined on MSL float2/3/4, int2/3/4, etc. — no namespace
    # acrobatics needed, the same names work on both scalars and vectors).
    "dot",
    "cross",
    "normalize",
    "length",
    "distance",
    # Matrix ops (defined on MSL floatRxC etc.).
    "transpose",
    "determinant",
    # Ternary clamp — same name and 3-arg signature in MSL (``metal::clamp(x, lo, hi)``).
    "clamp",
)


# Multi-D versions of ``wp::address``, ``wp::array_store``, and
# ``builtin_tid2d/3d`` are handled inline in ``generate_msl_kernel`` (their
# arity varies with the array's rank, which doesn't fit a single
# ``re.sub``-style rule).

# Patterns are applied in order. Each entry is (regex, replacement). Captures
# can be back-referenced with \1, \2, etc.
_INTRINSIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # wp.tid() -> thread_position_in_grid.x (1-D dispatch only — 2-D/3-D are
    # handled by the structural matcher in ``generate_msl_kernel``).
    (re.compile(r"\bbuiltin_tid1d\s*\(\s*\)"), "(int)thread_position_in_grid.x"),
    # ``wp::address`` and ``wp::array_store`` are now both handled by the
    # structural pre-pass in ``generate_msl_kernel`` (which knows the array's
    # rank and emits row-major flat indexing). If a stray multi-arg call slips
    # past, the unsupported-intrinsics guard will catch it.
    # wp::load(X) -> X
    # The dataflow collapse in ``generate_msl_kernel`` rewrites all our
    # ``wp::load`` operands from raw pointer locals to subscript expressions
    # (``arr[idx]``), so the load is logically a no-op — the value is already
    # there. If a non-collapsed load slips through, it'll fail the
    # ``_check_no_unsupported_intrinsics`` guard downstream.
    (re.compile(r"wp::load\s*\(\s*([^()]+?)\s*\)"), r"\1"),
    # Scalar arithmetic
    (re.compile(r"wp::add\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"), r"(\1 + \2)"),
    (re.compile(r"wp::sub\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"), r"(\1 - \2)"),
    (re.compile(r"wp::mul\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"), r"(\1 * \2)"),
    (re.compile(r"wp::div\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"), r"(\1 / \2)"),
    (re.compile(r"wp::mod\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"), r"(\1 % \2)"),
    # ``wp::assign(target, value)`` — used by Warp to model in-place mutation
    # of a local (e.g. accumulator updates inside a loop). Translate to a
    # plain assignment statement.
    (
        re.compile(r"wp::assign\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)\s*;"),
        r"\1 = \2;",
    ),
    # ``wp::copy(value)`` is an explicit value copy used by Warp to bind a
    # loaded value to a fresh local (e.g. ``n = a[tid]`` produces
    # ``var_n = wp::copy(var_loaded)``). For scalar / pointer types the copy
    # is just plain assignment.
    (re.compile(r"wp::copy\s*\(\s*([^()]+?)\s*\)"), r"\1"),
    # ``wp::extract(mat, i, j)`` (3-arg, matrix form) — must come BEFORE the
    # 2-arg vec form below, otherwise the non-greedy ``[^()]+?`` for the
    # second arg would swallow ``i, j`` together. MSL matrices are
    # column-major (``m[col][row]``), so the row/col arg order is reversed.
    (
        re.compile(r"wp::extract\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)"),
        r"\1[\3][\2]",
    ),
    # ``wp::extract(vec, idx)`` returns the i-th component. MSL vector types
    # support the C-style ``[i]`` subscript directly.
    (re.compile(r"wp::extract\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"), r"\1[\2]"),
    # ``wp::where(cond, a, b)`` is a select. MSL has ``select(b, a, cond)``
    # but the C-style ternary works for both scalar and vector operands and
    # avoids the surprising arg-order swap.
    (
        re.compile(r"wp::where\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)"),
        r"((\1) ? (\2) : (\3))",
    ),
    # Atomic ops on array elements. MLX-allocated outputs flagged with
    # ``atomic_outputs=True`` are typed ``device atomic<T>*``, so the
    # ``&arr[idx]`` we form here is already a valid atomic-pointer operand.
    # MSL's ``memory_order_relaxed`` matches CUDA's default atomic ordering,
    # which is what Warp's IR semantics imply.
    (
        re.compile(r"wp::atomic_add\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"),
        r"atomic_fetch_add_explicit(&\1[\2], \3, memory_order_relaxed)",
    ),
    (
        re.compile(r"wp::atomic_sub\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"),
        r"atomic_fetch_sub_explicit(&\1[\2], \3, memory_order_relaxed)",
    ),
    (
        re.compile(r"wp::atomic_min\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"),
        r"atomic_fetch_min_explicit(&\1[\2], \3, memory_order_relaxed)",
    ),
    (
        re.compile(r"wp::atomic_max\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"),
        r"atomic_fetch_max_explicit(&\1[\2], \3, memory_order_relaxed)",
    ),
    # Strip Warp scalar-type cast wrappers ``wp::T(x)``. Includes the unsuffixed
    # Python-style names ``wp::float``, ``wp::int``, etc. that Warp emits for
    # ``float(x)`` / ``int(x)`` constructor calls in user code.
    (
        re.compile(
            r"wp::(float|float16|float32|int|int8|int16|int32|int64|"
            r"uint|uint8|uint16|uint32|uint64|bool)\s*\(\s*([^()]+?)\s*\)"
        ),
        r"\2",
    ),
]

# Append math-builtin renames after the operator/cast rules. Each generates a
# simple ``wp::name`` -> ``metal::name`` substitution; the argument list is
# left intact for MSL to resolve via overload.
for _name in _MATH_BUILTIN_NAMES:
    _INTRINSIC_PATTERNS.append((re.compile(rf"\bwp::{_name}\b"), f"metal::{_name}"))
del _name


def _translate_intrinsics(line: str) -> str:
    """Apply intrinsic substitutions until convergence."""
    prev = None
    while prev != line:
        prev = line
        for pat, repl in _INTRINSIC_PATTERNS:
            line = pat.sub(repl, line)
    return line


def _check_no_unsupported_intrinsics(line: str) -> None:
    """After translation, any remaining ``wp::`` or ``builtin_`` is unsupported."""
    if "wp::" in line or "builtin_" in line:
        raise MetalCodegenError(
            f"MSL codegen does not yet support this statement; remaining Warp intrinsic in: {line!r}"
        )


# ---------------------------------------------------------------------------
# Kernel artifact
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Multi-D array indexing
# ---------------------------------------------------------------------------
#
# Warp's IR uses ``wp::address(arr, i, j, ...)`` and
# ``wp::array_store(arr, i, j, ..., val)`` with variable arity. The runtime
# computes the flat index from the array's strides; we don't have those at
# codegen time, so we emit a row-major flat index using the array's *shape*
# instead. For inputs MLX auto-generates ``<argname>_shape``; for outputs we
# add a synthetic ``<argname>_shape`` input at launch time.
#
# Apple Silicon MLX outputs are guaranteed row-major contiguous (MLX
# allocates them), so ``stride[k] = product(shape[k+1:])`` holds and a
# shape-based index is equivalent to a strides-based one.

_ADDRESS_MULTI_PAT = re.compile(
    r"^(?P<indent>\s*)var_(?P<local>\w+)\s*=\s*"
    r"wp::address\s*\(\s*var_(?P<arr>\w+)(?P<rest>(?:\s*,\s*var_\w+)+)\s*\)\s*;\s*$"
)
_ARRAY_STORE_MULTI_PAT = re.compile(
    r"^(?P<indent>\s*)wp::array_store\s*\(\s*var_(?P<arr>\w+)(?P<rest>(?:\s*,\s*[^()]+?)+)\s*\)\s*;\s*$"
)
_TID_2D_PAT = re.compile(r"^(?P<indent>\s*)builtin_tid2d\s*\(\s*var_(?P<i>\w+)\s*,\s*var_(?P<j>\w+)\s*\)\s*;\s*$")
_TID_3D_PAT = re.compile(
    r"^(?P<indent>\s*)builtin_tid3d\s*\(\s*var_(?P<i>\w+)\s*,\s*var_(?P<j>\w+)\s*,\s*var_(?P<k>\w+)\s*\)\s*;\s*$"
)


def _flat_index_expr(arr_name: str, index_var_names: list[str]) -> str:
    """Build a row-major flat-index expression for ``arr[i, j, ...]``.

    For 1-D, returns the lone index unchanged. For 2-D, returns
    ``i * arr_shape[1] + j``. For 3-D, ``i * arr_shape[1] * arr_shape[2]
    + j * arr_shape[2] + k``. Index var names should already include the
    ``var_`` prefix.
    """
    n = len(index_var_names)
    if n == 1:
        return index_var_names[0]
    terms: list[str] = []
    for k, idx in enumerate(index_var_names):
        if k == n - 1:
            terms.append(idx)
        else:
            stride = " * ".join(f"{arr_name}_shape[{j}]" for j in range(k + 1, n))
            terms.append(f"({idx} * {stride})")
    return " + ".join(terms)


def _split_array_store_args(rest: str) -> list[str]:
    """Split the trailing argument list of ``wp::array_store(arr, ...)``.

    The leading comma and surrounding whitespace are part of ``rest``. Returns
    the comma-separated args verbatim (with whitespace stripped). For
    ``, var_0, var_1, var_3``, returns ``['var_0', 'var_1', 'var_3']``.
    """
    # Trim the leading ``,`` and any whitespace, then split on top-level commas.
    # The IR doesn't put nested commas inside parens here (loaded values come
    # in as ``var_X``), so a naive split is safe.
    parts = [p.strip() for p in rest.split(",")]
    return [p for p in parts if p]


@dataclass
class MetalKernelArtifact:
    """Everything ``mx.fast.metal_kernel`` needs to compile a kernel.

    The artifact is consumed by the launch path (step 3d). For step 3c the
    ``source`` string is fed directly to ``mx.fast.metal_kernel`` and a
    1-element dispatch is used to confirm it compiles.
    """

    name: str
    source: str
    input_names: list[str]
    output_names: list[str]
    # Parallel to ``input_names`` — Warp ``Var`` for each input arg, so the
    # dispatch path can resolve element type / shape.
    input_args: list = field(default_factory=list)
    output_args: list = field(default_factory=list)
    # ``True`` if any kernel statement uses ``wp::atomic_*`` on an output
    # array. The launcher passes this through to
    # ``mx.fast.metal_kernel(atomic_outputs=...)`` so that outputs are typed
    # ``device atomic<T>*`` in the generated function signature, and the
    # ``init_value=0.0`` fallback is applied at call time.
    atomic_outputs: bool = False
    # Names of output arguments whose shape we pass as a synthetic kernel
    # input (because MLX only auto-generates ``<name>_shape`` for *inputs*).
    # The launcher constructs ``mx.array(value.shape, dtype=int32)`` for each
    # of these and appends them to the MLX inputs list, in order.
    output_shape_inputs: list[str] = field(default_factory=list)


def _strip_comments_and_directives(line: str) -> str | None:
    s = line.strip()
    if not s:
        return None
    if s.startswith("//"):
        return None
    if s.startswith("#line"):
        return None
    return line


# ---- Dynamic for-loop structural translation -------------------------------
#
# Warp's IR lowers ``for i in range(n):`` (with ``n`` not a compile-time
# constant) to a ``goto``-based loop. Static ranges are fully unrolled and
# need no rewrite — they appear as straight-line code. The dynamic pattern is
# always:
#
#     var_X = wp::range(var_N);
#     start_for_K:;
#         if (iter_cmp(var_X) == 0) goto end_for_K;
#         var_Y = wp::iter_next(var_X);
#         ... body ...
#         goto start_for_K;
#     end_for_K:;
#
# We rewrite the opener to ``for (int var_Y = 0; var_Y < var_N; ++var_Y) {``,
# the trailing ``goto`` to nothing, and the end label to ``}``. The MSL
# compiler does support ``goto`` so a more literal lowering would also work,
# but a real ``for`` produces cleaner emitted code that's easy to read in
# debug builds and removes the need to declare a ``range_t`` type.
_RANGE_PAT = re.compile(r"^\s*var_(\w+)\s*=\s*wp::range\s*\(\s*var_(\w+)\s*\)\s*;\s*$")
_START_LABEL_PAT = re.compile(r"^\s*start_for_(\d+)\s*:\s*;\s*$")
_ITER_CMP_PAT = re.compile(r"^\s*if\s*\(\s*iter_cmp\s*\(\s*var_(\w+)\s*\)\s*==\s*0\s*\)\s*goto\s+end_for_(\d+)\s*;\s*$")
_ITER_NEXT_PAT = re.compile(r"^\s*var_(\w+)\s*=\s*wp::iter_next\s*\(\s*var_(\w+)\s*\)\s*;\s*$")
_END_LABEL_PAT = re.compile(r"^\s*end_for_(\d+)\s*:\s*;\s*$")
_GOTO_START_PAT = re.compile(r"^\s*goto\s+start_for_(\d+)\s*;\s*$")


_WHILE_START_PAT = re.compile(r"^\s*start_while_(\d+)\s*:\s*;\s*$")
_WHILE_END_PAT = re.compile(r"^\s*end_while_(\d+)\s*:\s*;\s*$")
_WHILE_COND_TEST_PAT = re.compile(
    r"^(?P<indent>\s*)if\s*\(\s*\(\s*var_(?P<cond>\w+)\s*\)\s*==\s*false\s*\)\s*goto\s+end_while_(\d+)\s*;\s*$"
)
_WHILE_GOTO_START_PAT = re.compile(r"^(?P<indent>\s*)goto\s+start_while_(\d+)\s*;\s*$")
_WHILE_GOTO_END_PAT = re.compile(r"^(?P<indent>\s*)goto\s+end_while_(\d+)\s*;\s*$")


def _preprocess_while_loops(lines: list[str]) -> list[str]:
    """Rewrite Warp's goto-based ``while`` IR into MSL ``while (true) { ... }``.

    Pattern (per loop, ``K`` is a numeric label suffix):

        start_while_K:;
        ... cond computation ...
        if ((var_X) == false) goto end_while_K;
            ... body ...
        goto start_while_K;
        end_while_K:;

    Rewrite:

        while (true) {
            ... cond computation ...
            if (!var_X) { break; }
            ... body ...
            continue;
        }

    A mid-body ``goto end_while_K;`` (Warp's lowering of ``break``) becomes
    ``break;``; a mid-body ``goto start_while_K;`` (``continue``) becomes
    ``continue;``. MSL does not allow ``goto`` or labeled statements at all,
    so this rewrite is mandatory — verified empirically (the literal goto
    form fails at MSL compile time with "labeled statements are not
    supported in Metal").
    """
    out: list[str] = []
    for line in lines:
        if _WHILE_START_PAT.match(line):
            out.append("while (true) {")
            continue
        if _WHILE_END_PAT.match(line):
            out.append("}")
            continue
        m = _WHILE_COND_TEST_PAT.match(line)
        if m:
            out.append(f"{m.group('indent')}if (!var_{m.group('cond')}) {{ break; }}")
            continue
        m = _WHILE_GOTO_START_PAT.match(line)
        if m:
            out.append(f"{m.group('indent')}continue;")
            continue
        m = _WHILE_GOTO_END_PAT.match(line)
        if m:
            out.append(f"{m.group('indent')}break;")
            continue
        out.append(line)
    return out


def _preprocess_for_loops(lines: list[str]) -> tuple[list[str], set[str]]:
    """Rewrite dynamic-range goto-loops as MSL ``for`` loops.

    Returns ``(processed_lines, vars_to_skip_decl)`` — the second value lists
    Warp local labels whose top-level declaration must be suppressed (the
    range-iterator object, which doesn't exist in our MSL output, and the
    induction variable, which is declared by the ``for`` statement instead).
    """
    processed: list[str] = []
    skip: set[str] = set()

    i = 0
    while i < len(lines):
        line = lines[i]

        # Try to match the canonical 4-line for-loop opener.
        if i + 3 < len(lines):
            m_range = _RANGE_PAT.match(line)
            m_start = _START_LABEL_PAT.match(lines[i + 1])
            m_cmp = _ITER_CMP_PAT.match(lines[i + 2])
            m_next = _ITER_NEXT_PAT.match(lines[i + 3])
            if (
                m_range
                and m_start
                and m_cmp
                and m_next
                and m_range.group(1) == m_cmp.group(1) == m_next.group(2)
                and m_start.group(1) == m_cmp.group(2)
            ):
                range_var_label = m_range.group(1)
                range_arg_label = m_range.group(2)
                iter_var_label = m_next.group(1)
                processed.append(
                    f"for (int var_{iter_var_label} = 0; "
                    f"var_{iter_var_label} < var_{range_arg_label}; "
                    f"++var_{iter_var_label}) {{"
                )
                # Range-iterator local doesn't exist in MSL output; iter var
                # is declared inline by the ``for``.
                skip.add(range_var_label)
                skip.add(iter_var_label)
                i += 4
                continue

        if _GOTO_START_PAT.match(line):
            # Implicit in the for loop — drop.
            i += 1
            continue

        if _END_LABEL_PAT.match(line):
            processed.append("}")
            i += 1
            continue

        processed.append(line)
        i += 1

    return processed, skip


def generate_msl_kernel(kernel) -> MetalKernelArtifact:
    """Build an MSL artifact for a Warp ``Kernel`` object.

    The kernel must already have been added to a module so that
    ``kernel.adj.build()`` can resolve overloads. We invoke ``build()``
    here defensively in case it hasn't run yet.
    """
    adj = kernel.adj

    # Build the IR if not already built. Pass enable_backward=False because
    # we only consume the forward pass — the Metal backend does not yet
    # support autodiff.
    if not getattr(adj, "blocks", None):
        adj.build(builder=None, default_builder_options={"enable_backward": False})

    # Preprocess: rewrite dynamic-range goto-loops into ``for`` loops, then
    # rewrite ``while`` goto-loops into ``while (true) { ... }`` (MSL doesn't
    # support ``goto`` or labeled statements at all). Static-range ``for``s
    # are pre-unrolled by Warp and pass through untouched. The for-loop pass
    # also returns a set of local labels whose top-level declaration should
    # be suppressed (the range iter-state object and the induction variable,
    # which the generated ``for`` declares inline).
    forward_lines, vars_to_skip_decl = _preprocess_for_loops(adj.blocks[0].body_forward)
    forward_lines = _preprocess_while_loops(forward_lines)

    # Classify each array arg as input or output by scanning the IR strings.
    # MLX inputs are ``const device T*`` (read-only) — verified empirically —
    # so any array that is written through must become an MLX output. We can't
    # rely on ``arg.is_write`` here because it is only populated when
    # ``verify_autograd_array_access`` is enabled.
    written_arg_names: set[str] = set()
    atomic_arg_names: set[str] = set()
    array_store_pat = re.compile(r"\s*wp::array_store\s*\(\s*var_([A-Za-z_]\w*)")
    atomic_pat = re.compile(r"wp::atomic_(?:add|sub|min|max)\s*\(\s*var_([A-Za-z_]\w*)")
    for raw in forward_lines:
        m = array_store_pat.match(raw)
        if m:
            written_arg_names.add(m.group(1))
        m = atomic_pat.search(raw)
        if m:
            written_arg_names.add(m.group(1))
            atomic_arg_names.add(m.group(1))

    # MLX's ``atomic_outputs`` flag is per-kernel, not per-output: when set,
    # *every* output is typed ``device atomic<T>*``, which means a regular
    # ``arr[idx] = val`` store on a non-atomic output would no longer compile
    # (the LHS isn't an lvalue of the right type). Reject the mixed case
    # rather than silently miscompiling.
    has_atomic = bool(atomic_arg_names)
    if has_atomic:
        non_atomic_outputs = written_arg_names - atomic_arg_names
        if non_atomic_outputs:
            raise MetalCodegenError(
                f"Kernel {adj.fun_name!r} mixes atomic and non-atomic writes to outputs "
                f"({sorted(non_atomic_outputs)} written via ``arr[idx] = ...``, "
                f"{sorted(atomic_arg_names)} written via ``wp.atomic_*``). MLX makes "
                f"all outputs of a kernel atomic together; mixed kernels are not yet "
                f"supported. Split into two launches."
            )

    input_args: list = []
    output_args: list = []
    for arg in adj.args:
        if not _is_array_arg(arg):
            # Scalar arg — always an input.
            input_args.append(arg)
            continue
        if arg.label in written_arg_names:
            output_args.append(arg)
        else:
            input_args.append(arg)

    if not output_args:
        raise MetalCodegenError(
            f"Kernel {adj.fun_name!r} has no output array; MSL kernels must write through at least one output buffer"
        )

    # --- Dataflow simplification ---------------------------------------
    # Warp's IR splits an ``arr[i]`` (or ``arr[i, j, ...]``) read into
    # ``addr = wp::address(arr, i, j, ...);`` followed by
    # ``val = wp::load(addr);``. The intermediate ``addr`` is a typed pointer
    # in CUDA C++ but in MSL it would inherit a specific address space
    # (``const constant`` for inputs vs ``device`` for outputs) that we
    # can't easily express in a separately-declared local. Collapsing the
    # ``address``/``load`` pair into a direct subscript sidesteps the issue
    # entirely and yields cleaner MSL.
    #
    # For multi-dim arrays we synthesize a row-major flat index from the
    # array's shape (see ``_flat_index_expr``). The shape array is either
    # auto-generated by MLX (for inputs) or appended as a synthetic input by
    # the launcher (for outputs).
    #
    # Vec-typed arrays (``wp.array(dtype=wp.vec3)`` etc.) are flattened on the
    # MLX side to ``(*shape, vec_size)`` of the scalar dtype — MLX has no
    # native vec dtype and ``packed_floatN`` pointer casts hit address-space
    # mismatches in MLX-generated wrappers. So we emit per-component reads
    # ``floatN(arr[i*N+0], arr[i*N+1], ...)`` instead of any cast trick.
    arg_label_set = {a.label for a in adj.args}
    arg_by_label = {a.label: a for a in adj.args}
    # Map argname -> (vec_size, msl_scalar) for each vec-typed array arg, and
    # argname -> (rows, cols, msl_scalar) for each mat-typed array arg.
    vec_arr_info: dict[str, tuple[int, str]] = {}
    mat_arr_info: dict[str, tuple[int, int, str]] = {}
    for arg in adj.args:
        v_info = _vec_dtype_info(arg)
        if v_info is not None:
            vec_arr_info[arg.label] = v_info
            continue
        m_info = _mat_dtype_info(arg)
        if m_info is not None:
            mat_arr_info[arg.label] = m_info

    subscript_map: dict[str, str] = {}  # local label -> "arr[flat_idx]" string
    # Locals that resolve to ``<arr>_shape`` (the MLX-supplied shape array).
    # Tracked separately so we can recognise them when emitting / skipping
    # ``wp::load`` and ``wp::extract`` lines that operate on the shape struct.
    shape_aliases: set[str] = set()

    # ``arr.shape[k]`` lowers to ``var_X = &(var_arr.shape); var_Y = wp::load(var_X);
    # var_Z = wp::extract(var_Y, k);``. The intermediate ``var_X`` (``wp::shape_t*``)
    # and ``var_Y`` (``wp::shape_t``) are types our table doesn't know, so we
    # alias both to ``<arr>_shape`` and skip their declarations + the
    # corresponding ``&(...)`` and ``wp::load(...)`` lines. The trailing
    # ``wp::extract(var_Y, k)`` then translates via the existing 2-arg extract
    # rule into ``arr_shape[k]``.
    shape_address_pat = re.compile(r"^\s*var_(\w+)\s*=\s*&\s*\(\s*var_(\w+)\s*\.\s*shape\s*\)\s*;\s*$")
    load_pat = re.compile(r"^\s*var_(\w+)\s*=\s*wp::load\s*\(\s*var_(\w+)\s*\)\s*;\s*$")
    for raw in forward_lines:
        m = shape_address_pat.match(raw)
        if m:
            local_label = m.group(1)
            arr_arg = m.group(2)
            if arr_arg in arg_label_set:
                subscript_map[local_label] = f"{arr_arg}_shape"
                shape_aliases.add(local_label)
    # Propagate aliases through ``var_Y = wp::load(var_X)`` whose source is
    # already a shape alias.
    for raw in forward_lines:
        m = load_pat.match(raw)
        if m:
            target_label = m.group(1)
            source_label = m.group(2)
            if source_label in shape_aliases:
                subscript_map[target_label] = subscript_map[source_label]
                shape_aliases.add(target_label)

    for raw in forward_lines:
        m = _ADDRESS_MULTI_PAT.match(raw)
        if m:
            local_label = m.group("local")
            arr_arg = m.group("arr")
            if arr_arg not in arg_label_set:
                continue
            indices = re.findall(r"var_(\w+)", m.group("rest"))
            index_var_names = [f"var_{i}" for i in indices]
            if arr_arg in vec_arr_info:
                # Vec-typed: only 1-D arrays of vec are currently supported.
                arg_var = arg_by_label[arr_arg]
                arr_ndim = getattr(arg_var.type, "ndim", 1) or 1
                if arr_ndim != 1 or len(index_var_names) != 1:
                    raise MetalCodegenError(
                        f"MSL codegen does not yet support multi-dim arrays of vec types "
                        f"(kernel {adj.fun_name!r} arg {arr_arg!r})"
                    )
                vec_n, msl_scalar = vec_arr_info[arr_arg]
                msl_vec_type = f"{msl_scalar}{vec_n}"
                idx = index_var_names[0]
                comps = [f"{arr_arg}[{idx} * {vec_n} + {k}]" for k in range(vec_n)]
                subscript_map[local_label] = f"{msl_vec_type}({', '.join(comps)})"
            elif arr_arg in mat_arr_info:
                # Mat-typed: only 1-D arrays of mat are currently supported.
                # Read row-major flat (rows*cols floats per element) and build
                # an MSL ``floatRxC`` whose column k = (M[0][k], M[1][k], ...).
                arg_var = arg_by_label[arr_arg]
                arr_ndim = getattr(arg_var.type, "ndim", 1) or 1
                if arr_ndim != 1 or len(index_var_names) != 1:
                    raise MetalCodegenError(
                        f"MSL codegen does not yet support multi-dim arrays of mat types "
                        f"(kernel {adj.fun_name!r} arg {arr_arg!r})"
                    )
                rows, cols, msl_scalar = mat_arr_info[arr_arg]
                stride = rows * cols
                msl_vec_type = f"{msl_scalar}{rows}"
                msl_mat_type = f"{msl_scalar}{rows}x{cols}"
                idx = index_var_names[0]
                col_strs: list[str] = []
                for c in range(cols):
                    col_components = [f"{arr_arg}[{idx} * {stride} + {r * cols + c}]" for r in range(rows)]
                    col_strs.append(f"{msl_vec_type}({', '.join(col_components)})")
                subscript_map[local_label] = f"{msl_mat_type}({', '.join(col_strs)})"
            else:
                subscript_map[local_label] = f"{arr_arg}[{_flat_index_expr(arr_arg, index_var_names)}]"

    # --- Local variable declarations -----------------------------------
    body_lines: list[str] = []
    for var in adj.variables:
        if var.label in subscript_map:
            # This local was a pointer into an array arg; we'll inline its
            # uses below, so it doesn't need a declaration.
            continue
        if var.label in vars_to_skip_decl:
            # Iterator-state or induction-variable for a translated for-loop;
            # declared inline by the generated ``for`` statement.
            continue
        # Skip ranges that would otherwise hit ``_msl_var_type`` (which doesn't
        # know about ``wp::range_t``). The for-loop translation should already
        # have added these to ``vars_to_skip_decl``, but guard defensively.
        if var.ctype() == "wp::range_t":
            continue
        ctype = var.ctype()
        msl_type = _msl_var_type(ctype)
        if var.constant is None:
            body_lines.append(f"    {msl_type} var_{var.label};")
        else:
            body_lines.append(f"    const {msl_type} var_{var.label} = {_msl_constant_str(var.constant)};")

    # --- Forward statements --------------------------------------------
    def _finalize(translated: str) -> str:
        # Inline subscripts that the address-collapse produced.
        for local_label, subscript in subscript_map.items():
            translated = re.sub(rf"\bvar_{re.escape(local_label)}\b", subscript, translated)
        # Re-run intrinsic translation in case ``wp::load(var_1)`` became
        # ``wp::load(arr[idx])``.
        translated = _translate_intrinsics(translated)
        # Rewrite ``wp::mat_t<R, C, ...>(...)`` constructor calls first — the
        # row-major flat args need reordering into column form. After that,
        # any remaining ``wp::vec_t<N, ...>`` and ``wp::mat_t<R, C, ...>``
        # bare type references are renamed (e.g. variable declarations,
        # dangling type names).
        translated = _rewrite_mat_t_constructor(translated)
        translated = _translate_vec_t_in(translated)
        # Rename ``var_<argname>`` -> ``<argname>`` so the body matches the
        # MLX-generated function signature (which uses the names from
        # ``input_names``/``output_names`` directly).
        for arg in adj.args:
            translated = re.sub(rf"\bvar_{re.escape(arg.label)}\b", arg.label, translated)
        _check_no_unsupported_intrinsics(translated)
        return translated

    for raw in forward_lines:
        line = _strip_comments_and_directives(raw)
        if line is None:
            continue
        # Address lines have been folded into ``subscript_map``.
        if _ADDRESS_MULTI_PAT.match(raw):
            continue
        # ``var_X = &(var_arg.shape);`` and the load that follows are also
        # collapsed via ``subscript_map``/``shape_aliases``; skip the raw lines.
        if shape_address_pat.match(raw):
            continue
        m_load = load_pat.match(raw)
        if m_load and m_load.group(2) in shape_aliases:
            continue
        # ``builtin_tid2d(var_X, var_Y);`` / ``builtin_tid3d(...)`` —
        # arity-specific structural rewrite (the patterns table only handles
        # the 1-D form).
        m = _TID_2D_PAT.match(raw)
        if m:
            indent = m.group("indent")
            i_label = m.group("i")
            j_label = m.group("j")
            body_lines.append(_finalize(f"{indent}var_{i_label} = (int)thread_position_in_grid.x;"))
            body_lines.append(_finalize(f"{indent}var_{j_label} = (int)thread_position_in_grid.y;"))
            continue
        m = _TID_3D_PAT.match(raw)
        if m:
            indent = m.group("indent")
            i_label = m.group("i")
            j_label = m.group("j")
            k_label = m.group("k")
            body_lines.append(_finalize(f"{indent}var_{i_label} = (int)thread_position_in_grid.x;"))
            body_lines.append(_finalize(f"{indent}var_{j_label} = (int)thread_position_in_grid.y;"))
            body_lines.append(_finalize(f"{indent}var_{k_label} = (int)thread_position_in_grid.z;"))
            continue
        # ``wp::array_store(arr, idx0, idx1, ..., val);`` — variable arity,
        # rewrite to ``arr[flat_idx] = val;`` (or per-component for vec-typed
        # outputs).
        m = _ARRAY_STORE_MULTI_PAT.match(raw)
        if m:
            indent = m.group("indent")
            arr = m.group("arr")
            parts = _split_array_store_args(m.group("rest"))
            if len(parts) >= 2:
                indices = parts[:-1]
                value = parts[-1]
                if arr in vec_arr_info:
                    arg_var = arg_by_label[arr]
                    arr_ndim = getattr(arg_var.type, "ndim", 1) or 1
                    if arr_ndim != 1 or len(indices) != 1:
                        raise MetalCodegenError(
                            f"MSL codegen does not yet support stores into multi-dim arrays of vec types "
                            f"(kernel {adj.fun_name!r} arg {arr!r})"
                        )
                    vec_n, _ = vec_arr_info[arr]
                    idx = indices[0]
                    for k in range(vec_n):
                        body_lines.append(_finalize(f"{indent}{arr}[{idx} * {vec_n} + {k}] = {value}[{k}];"))
                    continue
                if arr in mat_arr_info:
                    arg_var = arg_by_label[arr]
                    arr_ndim = getattr(arg_var.type, "ndim", 1) or 1
                    if arr_ndim != 1 or len(indices) != 1:
                        raise MetalCodegenError(
                            f"MSL codegen does not yet support stores into multi-dim arrays of mat types "
                            f"(kernel {adj.fun_name!r} arg {arr!r})"
                        )
                    rows, cols, _ = mat_arr_info[arr]
                    stride = rows * cols
                    idx = indices[0]
                    # Scatter to row-major storage: data[i*RC + r*C + c] =
                    # logical M[r][c] = MSL ``value[c][r]`` (column, then row).
                    for r in range(rows):
                        for c in range(cols):
                            body_lines.append(
                                _finalize(f"{indent}{arr}[{idx} * {stride} + {r * cols + c}] = {value}[{c}][{r}];")
                            )
                    continue
                flat_idx = _flat_index_expr(arr, indices)
                body_lines.append(_finalize(f"{indent}{arr}[{flat_idx}] = {value};"))
                continue
        translated = _translate_intrinsics(line.strip())
        body_lines.append(f"    {_finalize(translated)}")

    source = "\n".join(body_lines) + "\n"

    # Synthesise shape-inputs for any multi-dim *output* array. MLX
    # auto-generates ``<name>_shape`` for inputs only; outputs need it
    # supplied as a separate kernel argument. The launcher will append a
    # corresponding ``mx.array(value.shape, dtype=int32)`` for each.
    output_shape_inputs: list[str] = []
    for out_arg in output_args:
        ndim = getattr(out_arg.type, "ndim", 1) or 1
        if ndim > 1:
            output_shape_inputs.append(out_arg.label)

    base_input_names = [a.label for a in input_args]
    extra_input_names = [f"{name}_shape" for name in output_shape_inputs]

    return MetalKernelArtifact(
        name=adj.fun_name,
        source=source,
        input_names=base_input_names + extra_input_names,
        output_names=[a.label for a in output_args],
        input_args=input_args,
        output_args=output_args,
        atomic_outputs=has_atomic,
        output_shape_inputs=output_shape_inputs,
    )


def _is_array_arg(var) -> bool:
    """Return True if a kernel arg's type is a ``wp.array`` family.

    Handles both annotation forms Warp produces:
      - ``wp.array(dtype=wp.float32)`` — a callable that returns an instance
        of ``warp._src.types.array``.
      - ``wp.array2d[float]`` — the subscript form returning a
        ``_ArrayAnnotation`` (this is what ``mujoco_warp`` uses throughout,
        and is missing the ``_wp_generic_type_str_`` marker the older form
        carries).
    """
    # Lazy import to avoid import cycle with warp._src.types.
    from warp._src.types import _ArrayAnnotationBase, array, indexedarray  # noqa: PLC0415

    t = var.type
    if isinstance(t, (array, indexedarray)):
        return True
    if isinstance(t, _ArrayAnnotationBase):
        return True
    # ``wp.array`` annotations live as classes too:
    return getattr(t, "_wp_generic_type_str_", None) in ("array_t", "indexedarray_t")


def _vec_dtype_info(arg) -> tuple[int, str] | None:
    """If ``arg`` is a ``wp.array`` whose element type is a Warp vec_t, return
    ``(N, msl_scalar_type)``. Otherwise return ``None``.

    Used by the codegen to decide whether to expand reads/writes of this
    array into per-component scalar accesses.
    """
    if not _is_array_arg(arg):
        return None
    dtype = getattr(arg.type, "dtype", None)
    if dtype is None:
        return None
    if getattr(dtype, "_wp_generic_type_str_", None) != "vec_t":
        return None
    n = getattr(dtype, "_length_", None)
    scalar_cls = getattr(dtype, "_wp_scalar_type_", None)
    if n is None or scalar_cls is None:
        return None
    if n not in _MSL_VEC_SUPPORTED_N:
        raise MetalCodegenError(
            f"MSL codegen does not yet support arrays of vec{n} (only sizes 2, 3, 4 have native MSL types)"
        )
    scalar_ctype = f"wp::{scalar_cls.__name__}"
    if scalar_ctype not in _MSL_VEC_SCALAR_PREFIX:
        raise MetalCodegenError(f"MSL codegen does not yet support arrays of vec_t with element type {scalar_ctype!r}")
    return n, _MSL_VEC_SCALAR_PREFIX[scalar_ctype]


def _mat_dtype_info(arg) -> tuple[int, int, str] | None:
    """If ``arg`` is a ``wp.array`` whose element type is a Warp mat_t, return
    ``(rows, cols, msl_scalar_type)``. Otherwise return ``None``.
    """
    if not _is_array_arg(arg):
        return None
    dtype = getattr(arg.type, "dtype", None)
    if dtype is None:
        return None
    if getattr(dtype, "_wp_generic_type_str_", None) != "mat_t":
        return None
    shape = getattr(dtype, "_shape_", None)
    scalar_cls = getattr(dtype, "_wp_scalar_type_", None)
    if shape is None or len(shape) != 2 or scalar_cls is None:
        return None
    rows, cols = int(shape[0]), int(shape[1])
    if rows not in _MSL_VEC_SUPPORTED_N or cols not in _MSL_VEC_SUPPORTED_N:
        raise MetalCodegenError(
            f"MSL codegen does not yet support arrays of mat{rows}x{cols} "
            f"(only sizes 2, 3, 4 per dim have native MSL types)"
        )
    scalar_ctype = f"wp::{scalar_cls.__name__}"
    if scalar_ctype not in _MSL_VEC_SCALAR_PREFIX:
        raise MetalCodegenError(f"MSL codegen does not yet support arrays of mat_t with element type {scalar_ctype!r}")
    return rows, cols, _MSL_VEC_SCALAR_PREFIX[scalar_ctype]


# ---------------------------------------------------------------------------
# Launch path (step 3d)
# ---------------------------------------------------------------------------
#
# These imports are deferred to the function bodies so that simply importing
# this module on a system without MLX (e.g. Linux CI) does not error.


def _wp_dtype_to_mx_dtype(wp_dtype):
    """Translate a Warp scalar dtype to its MLX equivalent.

    Returns the MLX dtype object. Raises ``MetalCodegenError`` for types
    MSL/MLX cannot represent.
    """
    import mlx.core as mx  # noqa: PLC0415

    import warp._src.types as wpt  # noqa: PLC0415

    mapping = {
        wpt.float32: mx.float32,
        wpt.float16: mx.float16,
        wpt.int32: mx.int32,
        wpt.uint32: mx.uint32,
        wpt.int8: mx.int8,
        wpt.uint8: mx.uint8,
        wpt.int16: mx.int16,
        wpt.uint16: mx.uint16,
        wpt.int64: mx.int64,
        wpt.uint64: mx.uint64,
    }
    if wp_dtype is wpt.float64:
        raise MetalCodegenError("MSL/MLX has no native float64; use float32 for Metal kernels")
    if wp_dtype not in mapping:
        raise MetalCodegenError(f"No MLX dtype for Warp dtype {wp_dtype!r}")
    return mapping[wp_dtype]


def _array_view_dtype_and_shape(value):
    """Return ``(mx_dtype, mlx_shape)`` for a Metal-resident wp.array.

    For scalar-dtype arrays, the MLX view shape matches ``value.shape``.
    For vec-typed arrays (``wp.array(dtype=wp.vec3)`` etc.), the underlying
    storage is ``value.size * vec_size`` scalars, exposed to MLX as a flat
    ``(*value.shape, vec_size)`` of the inner scalar dtype. The kernel-side
    body computes per-component indices itself.

    For mat-typed arrays (``wp.array(dtype=wp.mat33)`` etc.), the storage is
    ``value.size * rows * cols`` scalars in row-major layout, exposed as
    ``(*value.shape, rows * cols)`` of the inner scalar dtype.
    """
    dtype = value.dtype
    kind = getattr(dtype, "_wp_generic_type_str_", None)
    if kind == "vec_t":
        n = dtype._length_
        scalar_cls = dtype._wp_scalar_type_
        mx_dtype = _wp_dtype_to_mx_dtype(scalar_cls)
        return mx_dtype, (*value.shape, n)
    if kind == "mat_t":
        rows, cols = dtype._shape_
        scalar_cls = dtype._wp_scalar_type_
        mx_dtype = _wp_dtype_to_mx_dtype(scalar_cls)
        return mx_dtype, (*value.shape, rows * cols)
    return _wp_dtype_to_mx_dtype(dtype), value.shape


def _get_or_build_metal_kernel(kernel):
    """Return ``(artifact, mlx_kernel)`` for a Warp kernel, building+caching on first use.

    The cache lives on the Warp ``Kernel`` object via two attributes;
    Warp regenerates the underlying ``Adjoint`` if the source changes, so
    pinning the cache to the kernel instance is safe.
    """
    import mlx.core as mx  # noqa: PLC0415

    artifact = getattr(kernel, "_metal_artifact", None)
    mlx_kernel = getattr(kernel, "_metal_mlx_kernel", None)
    if artifact is None or mlx_kernel is None:
        artifact = generate_msl_kernel(kernel)
        mlx_kernel = mx.fast.metal_kernel(
            name=artifact.name,
            input_names=artifact.input_names,
            output_names=artifact.output_names,
            source=artifact.source,
            atomic_outputs=artifact.atomic_outputs,
        )
        kernel._metal_artifact = artifact
        kernel._metal_mlx_kernel = mlx_kernel
    return artifact, mlx_kernel


def launch_metal_kernel(kernel, dim, inputs, outputs, device):
    """Dispatch a Warp kernel on a Metal device via MLX.

    This is the Metal-specific equivalent of the CUDA/CPU launch path in
    ``warp._src.context.launch``. It:

    1. Resolves (or builds and caches) the ``MetalKernelArtifact`` and
       ``mx.fast.metal_kernel`` for the kernel.
    2. Translates each Warp argument into an MLX argument:
       - ``wp.array`` inputs become typed views of the MLX-managed unified
         buffer that backs the array (zero-copy, via ``mx.array.view``).
       - Scalar inputs become ``mx.array`` literals.
    3. Lets MLX allocate fresh output buffers (its API does not accept
       user-provided outputs), then ``wp_memcpy_h2h``-copies the MLX result
       into the user's existing ``wp.array`` storage. The copy is between
       two unified-memory addresses, so it's an ordinary host memcpy.
    """
    import mlx.core as mx  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from warp._src.context import _metal_get_buffer, runtime  # noqa: PLC0415

    artifact, mlx_kernel = _get_or_build_metal_kernel(kernel)

    fwd_args = list(inputs) + list(outputs)
    if len(fwd_args) != len(kernel.adj.args):
        raise RuntimeError(
            f"Error launching kernel '{kernel.key}', passed {len(fwd_args)} arguments "
            f"but kernel requires {len(kernel.adj.args)}."
        )

    arg_by_name = {a.label: (i, a) for i, a in enumerate(kernel.adj.args)}

    # ---- Build MLX inputs ----
    # Order must match ``artifact.input_names``: real inputs first (from the
    # kernel signature, in declaration order), then synthetic shape inputs
    # for each multi-dim output (one per entry in ``output_shape_inputs``).
    mlx_inputs: list = []
    for arg_var in artifact.input_args:
        input_name = arg_var.label
        idx, _ = arg_by_name[input_name]
        value = fwd_args[idx]
        if _is_array_arg(arg_var):
            if not getattr(value, "device", None) or not value.device.is_metal:
                raise RuntimeError(
                    f"Kernel '{kernel.key}' argument '{input_name}' must be a wp.array on a Metal "
                    f"device; got {getattr(value, 'device', '?')}"
                )
            mx_buf = _metal_get_buffer(value.ptr)
            if mx_buf is None:
                raise RuntimeError(
                    f"Kernel '{kernel.key}' argument '{input_name}' has no registered MLX buffer "
                    f"(ptr={value.ptr}). Was it allocated by Warp's Metal allocator?"
                )
            mx_dtype, view_shape = _array_view_dtype_and_shape(value)
            # ``view`` reinterprets bytes (no copy); ``reshape`` flattens / shapes for MLX.
            typed = mx_buf.view(mx_dtype).reshape(view_shape)
            mlx_inputs.append(typed)
        else:
            # Scalar input — convert to a 0-D mx.array literal.
            mx_dtype = _wp_dtype_to_mx_dtype(arg_var.type)
            mlx_inputs.append(mx.array(value, dtype=mx_dtype))

    # Append synthetic ``<outname>_shape`` inputs for each multi-dim output
    # array. MLX auto-generates ``<inputname>_shape`` for *inputs*, so the
    # codegen body can use the same naming uniformly.
    for out_name in artifact.output_shape_inputs:
        idx, _ = arg_by_name[out_name]
        value = fwd_args[idx]
        shape_arr = mx.array(np.array(value.shape, dtype=np.int32), dtype=mx.int32)
        mlx_inputs.append(shape_arr)

    # ---- Build MLX output specs from user's output wp.arrays ----
    output_shapes: list = []
    output_dtypes: list = []
    output_dest_arrays: list = []
    for output_name in artifact.output_names:
        idx, arg_var = arg_by_name[output_name]
        value = fwd_args[idx]
        if not _is_array_arg(arg_var):
            raise RuntimeError(
                f"Kernel '{kernel.key}' output '{output_name}' is not a wp.array; "
                f"only array outputs are supported on Metal"
            )
        if not getattr(value, "device", None) or not value.device.is_metal:
            raise RuntimeError(
                f"Kernel '{kernel.key}' output '{output_name}' must be a wp.array on a Metal "
                f"device; got {getattr(value, 'device', '?')}"
            )
        out_mx_dtype, out_view_shape = _array_view_dtype_and_shape(value)
        output_shapes.append(out_view_shape)
        output_dtypes.append(out_mx_dtype)
        output_dest_arrays.append(value)

    # ---- Compute grid ----
    # Warp's ``dim`` can be an int (1-D) or a sequence (multi-D). MLX takes
    # a 3-tuple ``grid=(x, y, z)`` where ``thread_position_in_grid.x`` ranges
    # over the *first* element. Warp's ``i, j = wp.tid()`` returns indices in
    # the same order as ``dim``, so element 0 of ``dim`` -> ``.x``.
    if isinstance(dim, int):
        dims = (dim,)
    else:
        dims = tuple(dim)
    if len(dims) == 0:
        return
    if len(dims) > 3:
        raise RuntimeError(
            f"Metal backend supports up to 3-D launches; kernel '{kernel.key}' was launched with dim={dim}"
        )
    if any(d <= 0 for d in dims):
        return
    grid_x = dims[0]
    grid_y = dims[1] if len(dims) >= 2 else 1
    grid_z = dims[2] if len(dims) >= 3 else 1
    grid = (grid_x, grid_y, grid_z)
    # Pick a threadgroup that's at most 256 threads total and never larger
    # than each grid dimension.
    if len(dims) == 1:
        tg = (min(256, grid_x), 1, 1)
    elif len(dims) == 2:
        tg_x = min(16, grid_x)
        tg_y = min(16, grid_y)
        tg = (tg_x, tg_y, 1)
    else:
        tg_x = min(8, grid_x)
        tg_y = min(8, grid_y)
        tg_z = min(4, grid_z)
        tg = (tg_x, tg_y, tg_z)

    # MLX outputs are uninitialized by default. For atomic-output kernels
    # we *must* zero-initialize so the first ``atomic_fetch_add`` accumulates
    # from a defined zero rather than stale buffer contents (verified
    # empirically in the step-2 atomic probe). Note: this means atomic
    # kernels always start their accumulators at zero — pre-existing values
    # in the user's ``wp.array`` are not preserved across the launch. Most
    # atomic-accumulator usage zeroes the buffer beforehand anyway.
    init_value = 0.0 if artifact.atomic_outputs else None

    out_mx_list = mlx_kernel(
        inputs=mlx_inputs,
        grid=grid,
        threadgroup=tg,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        init_value=init_value,
    )
    # Force the dispatch to complete so the unified-memory copy below sees
    # the final results rather than queued operations.
    if isinstance(out_mx_list, mx.array):
        out_mx_list = [out_mx_list]
    for o in out_mx_list:
        mx.eval(o)

    # ---- Copy MLX outputs into the user's wp.array buffers ----
    for o_mx, dest in zip(out_mx_list, output_dest_arrays, strict=True):
        np_view = np.array(o_mx, copy=False)
        src_ptr = int(np_view.__array_interface__["data"][0])
        nbytes = np_view.nbytes
        if not runtime.core.wp_memcpy_h2h(dest.ptr, src_ptr, nbytes):
            raise RuntimeError(f"Failed to copy Metal kernel output back into wp.array (kernel '{kernel.key}')")
