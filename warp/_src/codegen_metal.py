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
- Scalar / vec / mat arithmetic intrinsics: ``add``, ``sub``, ``mul``,
  ``div``, ``mod``, ``neg`` (unary)
- ``if`` / ``else`` blocks and the comparison operators ``<``, ``<=``,
  ``==``, ``!=``, ``>=``, ``>`` (Warp's IR pre-emits these in plain C/MSL
  syntax, so they pass through the regex-based translator unchanged).
- A whitelist of math builtins listed in ``_MATH_BUILTIN_NAMES`` —
  ``sqrt``, ``abs``, ``min``, ``max``, ``floor``, ``ceil``, ``exp``,
  ``log``, ``sin``, ``cos``, ``tanh``, ``clamp``, etc. — translated to
  MSL's ``metal::`` namespace.
- ``wp.where(cond, a, b)`` -> ``((cond) ? (a) : (b))`` (C-style ternary).
- Field reads/writes on ``wp.array(dtype=SomeStruct)`` for POD structs
  whose fields are scalars / vec / mat with 4-byte scalar width (float32 /
  int32 / uint32 / float16 / etc.). No MSL ``struct`` is emitted; instead
  each field read expands to direct flat-buffer accesses at the field's
  offset.
- Local struct construction (``q = SomeStruct()``), per-field assignment
  (``q.pos = ...``), and storing the local into an array
  (``arr[i] = q``). The struct value is represented as one MSL local per
  field (``var_<X>__<field>`` with ``T(0)`` zero-init); field-pointer
  addresses route through the same ``subscript_map`` mechanism, and
  ``wp::store(addr, val)`` becomes ``<lhs_expr> = val``. Storing a struct
  local into a struct array scatters the fields per-component.
- Struct-typed kernel args (``def k(s: SomeStruct, ...)``). The launcher
  serialises the ``StructInstance`` to bytes (via the ``_ctype`` member)
  and passes it as a 1-D float32 mx.array; the kernel body reads each
  field at its compile-time-known scalar offset (same machinery as
  struct-array field reads, with the array index pinned to 0).
- Currently NOT supported: reading a struct *value* out of an array as
  a local (``m = arr[tid]; m.field`` — workaround: use
  ``arr[tid].field`` directly), nested structs, array fields, mixed
  scalar widths within a struct (e.g. int + float together — would need
  per-field ``as_type`` bit-casts on the float-viewed buffer).
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
  atomic op, which makes *all* outputs of that kernel ``device atomic<T>*``.
  Kernels that mix ``wp.atomic_*`` with plain ``arr[i] = val`` writes on
  *different* outputs are supported by translating each regular scalar
  store to ``atomic_store_explicit`` when the kernel is in atomic mode.
- Vec types ``wp.vec2``/``wp.vec3``/``wp.vec4`` (and the corresponding
  int/uint variants) — native ``floatN`` / ``intN`` etc. in MSL.
  Constructor maps to ``floatN(...)``, ``[i]`` indexing maps to MSL
  subscript, and ``wp.dot``/``wp.cross``/``wp.normalize``/``wp.length``
  are added to the math whitelist as ``metal::*``. Reads and writes of
  ``wp.array(dtype=wp.vec3)`` are expanded into per-component scalar
  accesses (``a[i*3+0]``, ``a[i*3+1]``, ``a[i*3+2]``) to avoid
  ``packed_floatN`` cast issues. Multi-dim arrays of vec types
  (``wp.array2d(dtype=wp.vec3)`` etc.) work via the same expansion plus
  ``_flat_index_expr`` for the user-visible dims.
- Larger vec types (``wp.vec5``, ``wp.spatial_vector`` = vec6,
  ``wp.vec8``, ...) — MSL has no native ``floatN`` for N > 4, so the
  codegen emits a custom ``wp_vec<N>_<scalar>`` struct in the kernel
  header with ``+/-/*//`` operator overloads and a
  ``wp_vec<N>_<scalar>_make(v0, v1, ...)`` factory. ``wp.spatial_top``
  and ``wp.spatial_bottom`` translate to header helpers. Arrays and
  struct fields of these types use the same per-component flat-buffer
  expansion as native vec arrays.
- ``wp.quat_t<T>`` (and ``wp.quat`` — alias for ``quat_t<float32>``).
  Normalised to ``wp.vec_t<4, T>`` at codegen entry so all the vec4
  paths cover quat. Quat-specific operations (``quat_inverse``,
  ``quat_rotate``, etc.) would still need their own translations.
- Trivial intrinsics surfaced by the mujoco_warp recon: ``wp.unot``
  (``!``), ``wp.bit_and/or/xor``, ``wp.lshift/rshift``, ``wp.floordiv``
  (C-style truncation toward zero — diverges from Python ``//`` for
  mixed-sign integers), ``wp.length_sq`` (translates to
  ``metal::dot(v, v)``), and the in-place compound forms
  ``wp.add_inplace``/``sub_inplace``/``mul_inplace``/``div_inplace``/
  ``assign_inplace``.
- Matrix types ``wp.mat22``/``wp.mat33``/``wp.mat44`` (and the
  corresponding int variants) for ``RxC`` with ``R, C`` in ``{2, 3, 4}``
  where MSL has a native ``floatRxC`` type. ``wp.transpose`` and
  ``wp.determinant`` route to ``metal::*``. ``wp.mat33(...)`` row-major
  flat constructor is reordered into ``floatRxC(floatR(...), ...)``
  column form. ``wp::extract(m, i, j)`` becomes ``m[j][i]`` to match
  MSL's column-major indexing. Reads/writes of ``wp.array(dtype=wp.mat33)``
  scatter/gather between row-major user storage and the column-major
  MSL representation. Multi-dim arrays of mat types
  (``wp.array2d(dtype=wp.mat33)`` etc.) are also supported via the same
  scatter/gather expanded for each user-visible dim.
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
# Pattern that matches ``wp::quat_t<wp::TYPE>``. We treat quaternions as
# 4-element vec_t with float32 components — Warp lays them out the same
# way and our vec/mat machinery covers all the array, constructor, and
# extract patterns we currently see in mujoco_warp's hot path. Quat-
# specific operations (``wp.quat_inverse``, ``wp.quat_rotate``, etc.) would
# need their own translations later.
_WP_QUAT_T_PAT = re.compile(r"wp::quat_t\s*<\s*wp::(\w+)\s*>")


def _normalize_quat_t(text: str) -> str:
    """Rewrite every ``wp::quat_t<wp::T>`` reference as ``wp::vec_t<4, wp::T>``.

    Run before any other vec/mat handling so the existing 4-element vec
    paths cover quat seamlessly.
    """
    return _WP_QUAT_T_PAT.sub(lambda m: f"wp::vec_t<4, wp::{m.group(1)}>", text)


_MSL_VEC_NATIVE_N = (2, 3, 4)


def _msl_vec_name(n: int, msl_scalar: str) -> str:
    """Return the MSL type name for ``vec_t<N, T>`` given the MSL scalar prefix.

    Native ``floatN`` etc. for N in {2, 3, 4}; ``wp_vecN_<scalar>`` otherwise.
    """
    if n in _MSL_VEC_NATIVE_N:
        return f"{msl_scalar}{n}"
    return f"wp_vec{n}_{msl_scalar}"


def _msl_vec_ctor(n: int, msl_scalar: str) -> str:
    """Return the constructor / factory expression name for ``vec_t<N, T>``.

    For native sizes this matches the type name (``float3(a, b, c)``). For
    big sizes we use a free factory function ``wp_vecN_<scalar>_make`` —
    MSL doesn't let us declare a variadic struct constructor cleanly.
    """
    if n in _MSL_VEC_NATIVE_N:
        return f"{msl_scalar}{n}"
    return f"wp_vec{n}_{msl_scalar}_make"


def _vec_t_to_msl(n: int, scalar_ctype: str) -> str:
    """Translate ``wp::vec_t<N, wp::TYPE>`` to its MSL name.

    For N in {2, 3, 4} this returns the native MSL type (``floatN`` etc.).
    For larger N (vec5, vec6 = ``spatial_vector``, vec8) it returns a name
    like ``wp_vec6_float`` for which the codegen emits a custom struct
    declaration in the kernel header.
    """
    if n < 2:
        raise MetalCodegenError(f"MSL codegen: vec_t<{n}, ...> not supported (need N >= 2)")
    full_ctype = f"wp::{scalar_ctype}"
    if full_ctype not in _MSL_VEC_SCALAR_PREFIX:
        raise MetalCodegenError(f"MSL codegen does not yet support vec_t element type {full_ctype!r}")
    msl_scalar = _MSL_VEC_SCALAR_PREFIX[full_ctype]
    if n in _MSL_VEC_NATIVE_N:
        return f"{msl_scalar}{n}"
    return f"wp_vec{n}_{msl_scalar}"


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


# ---------------------------------------------------------------------------
# Custom big-vec struct emission (vec5, vec6 = spatial_vector, vec8, ...)
# ---------------------------------------------------------------------------
#
# MSL has no native ``floatN`` for N > 4. We emit a small custom struct per
# ``(N, scalar)`` combination used by the kernel, with operator overloads so
# the existing ``wp::add(a, b) -> (a + b)`` translations Just Work for the
# resulting MSL types.
_BIG_VEC_NAME_PAT = re.compile(r"\bwp_vec(\d+)_(\w+)\b")
# Stripped scalar -> MSL scalar prefix lookup. Goes from "float" / "int" etc.
# back to the same name (it's a no-op convenience map for clarity).
_MSL_PREFIX_TO_SAME = {prefix: prefix for prefix in _MSL_VEC_SCALAR_PREFIX.values()}


def _emit_big_vec_struct(name: str, n: int, msl_scalar: str) -> str:
    """Emit the MSL declaration for a custom big-vec struct.

    The struct has a plain N-element scalar array as its only data member,
    plus operator overloads (+, -, * by scalar, /, [] for read+write, ==).
    Each unrolled component-wise operation is generated explicitly so the
    MSL compiler can vectorise without trusting an MSL ``for`` loop with a
    runtime bound.
    """
    body_lines: list[str] = []
    body_lines.append(f"struct {name} {{")
    body_lines.append(f"    {msl_scalar} c[{n}];")
    body_lines.append(f"    inline thread {msl_scalar}& operator[](int i) thread {{ return c[i]; }}")
    body_lines.append(f"    inline {msl_scalar} operator[](int i) const thread {{ return c[i]; }}")
    body_lines.append("};")
    # Variadic-arg constructor: ``wp_vec6_float(v0, v1, ..., v5)`` so the
    # existing IR ``wp::vec_t<6, ...>(...)`` (with the braces stripped) maps
    # directly. We can't define a templated constructor on the struct in a
    # forward-portable way, so emit a free factory function with the same
    # name as the type — MSL allows ``wp_vec6_float(args...)`` to dispatch
    # to a function of that name when no constructor matches.
    args = ", ".join(f"{msl_scalar} v{i}" for i in range(n))
    body_lines.append(f"inline {name} {name}_make({args}) {{")
    body_lines.append(f"    {name} r;")
    for i in range(n):
        body_lines.append(f"    r.c[{i}] = v{i};")
    body_lines.append("    return r;")
    body_lines.append("}")
    # Operator overloads. Component-wise unrolled for clarity and so MSL's
    # auto-vectoriser sees independent statements.
    for op in ("+", "-", "*", "/"):
        body_lines.append(f"inline {name} operator{op}({name} a, {name} b) {{")
        body_lines.append(f"    {name} r;")
        for i in range(n):
            body_lines.append(f"    r.c[{i}] = a.c[{i}] {op} b.c[{i}];")
        body_lines.append("    return r;")
        body_lines.append("}")
    return "\n".join(body_lines)


def _emit_spatial_helpers() -> str:
    """Helpers specific to ``spatial_vector`` (``vec_t<6, float32>``).

    - ``wp_spatial_top`` / ``wp_spatial_bottom`` decompose into the upper /
      lower vec3.
    - A ``(float3, float3)`` overload of ``wp_vec6_float_make`` mirrors
      Warp's ``wp.spatial_vector(top_vec3, bottom_vec3)`` two-arg
      constructor (used wherever a kernel composes a spatial_vector from
      two vec3s rather than 6 scalars).
    """
    return (
        "inline float3 wp_spatial_top(wp_vec6_float v) { "
        "return float3(v.c[0], v.c[1], v.c[2]); }\n"
        "inline float3 wp_spatial_bottom(wp_vec6_float v) { "
        "return float3(v.c[3], v.c[4], v.c[5]); }\n"
        "inline wp_vec6_float wp_vec6_float_make(float3 a, float3 b) { "
        "return wp_vec6_float_make(a[0], a[1], a[2], b[0], b[1], b[2]); }"
    )


def _build_kernel_header(source: str) -> str:
    """Scan ``source`` for ``wp_vecN_<scalar>`` struct names and emit a
    header block defining each unique one (plus spatial helpers if vec6
    structs are present).
    """
    seen: set[tuple[int, str]] = set()
    for m in _BIG_VEC_NAME_PAT.finditer(source):
        n = int(m.group(1))
        scalar = m.group(2)
        if scalar not in _MSL_PREFIX_TO_SAME:
            continue
        if n in _MSL_VEC_NATIVE_N:
            # Shouldn't happen — native sizes use the ``floatN`` form, not
            # the custom name — but guard defensively.
            continue
        seen.add((n, scalar))
    if not seen:
        return ""
    parts: list[str] = []
    for n, scalar in sorted(seen):
        parts.append(_emit_big_vec_struct(f"wp_vec{n}_{scalar}", n, scalar))
    if (6, "float") in seen:
        parts.append(_emit_spatial_helpers())
    return "\n".join(parts) + "\n"


def _rewrite_vec_t_brace_constructor(text: str) -> str:
    """Rewrite ``wp::vec_t<N, wp::T>(args)`` constructor calls (any arg shape)
    to the corresponding MSL form.

    Args may come in either of two shapes:
      - ``({v0, v1, ...})`` — the brace-init form Warp uses for vec5+ when
        constructing from N scalars.
      - ``(args...)`` — a plain comma-separated list, used for vec_t<N,T>
        ``= float3(scalar)`` (single-arg broadcast), ``spatial_vector(vec3,
        vec3)`` (the canonical 2-arg overload), and the standard
        N-arg form for vec2/3/4.

    For native sizes (N in {2,3,4}) the result is ``floatN(args)`` — MSL's
    native vector type provides all the overloads. For big sizes it's
    ``wp_vecN_<scalar>_make(args)`` — the helper functions we emit in the
    header carry the same overload set we need.
    """
    pat = re.compile(r"wp::vec_t<\s*(\d+)\s*,\s*wp::(\w+)\s*>\s*\(([^()]*)\)")

    def repl(m: re.Match[str]) -> str:
        n = int(m.group(1))
        scalar_ctype = m.group(2)
        args_str = m.group(3).strip()
        # Strip the outer ``{}`` if present.
        if args_str.startswith("{") and args_str.endswith("}"):
            args_str = args_str[1:-1].strip()
        full_ctype = f"wp::{scalar_ctype}"
        if full_ctype not in _MSL_VEC_SCALAR_PREFIX or n < 2:
            return m.group(0)
        msl_scalar = _MSL_VEC_SCALAR_PREFIX[full_ctype]
        if n in _MSL_VEC_NATIVE_N:
            return f"{msl_scalar}{n}({args_str})"
        return f"wp_vec{n}_{msl_scalar}_make({args_str})"

    return pat.sub(repl, text)


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
    # ``wp::quat_t<wp::TYPE>`` is laid out as a 4-element vec_t; rewrite
    # so the vec_t/mat_t handling below covers it.
    stripped = _normalize_quat_t(stripped)
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
    # Unary negation: ``wp::neg(X)`` -> ``(-X)``. Works for scalar / vec / mat
    # because MSL's ``operator-`` is defined on all of those.
    (re.compile(r"wp::neg\s*\(\s*([^()]+?)\s*\)"), r"(-\1)"),
    # Boolean not: ``wp::unot(X)`` -> ``(!X)``.
    (re.compile(r"wp::unot\s*\(\s*([^()]+?)\s*\)"), r"(!\1)"),
    # Bitwise ops. Only ``bit_and`` actually shows up in the mujoco_warp
    # inventory but the others are cheap to add as a set.
    (re.compile(r"wp::bit_and\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"), r"(\1 & \2)"),
    (re.compile(r"wp::bit_or\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"), r"(\1 | \2)"),
    (re.compile(r"wp::bit_xor\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"), r"(\1 ^ \2)"),
    (re.compile(r"wp::lshift\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"), r"(\1 << \2)"),
    (re.compile(r"wp::rshift\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"), r"(\1 >> \2)"),
    # Floor division: ``wp::floordiv(X, Y)`` -> ``(X / Y)``. C-style truncation
    # toward zero, NOT Python ``//`` floor toward -inf — diverges for mixed-
    # sign integer operands (e.g. ``-7 / 2`` is ``-3`` here vs ``-4`` in
    # Python). Mujoco_warp uses this for non-negative array indexing in
    # practice so the divergence is unlikely to surface.
    (re.compile(r"wp::floordiv\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"), r"(\1 / \2)"),
    # ``wp::length_sq(V)`` is the squared length of a vector. MSL has no
    # native ``length_squared``, so we translate to the dot of the vector
    # with itself — works for native ``floatN`` types.
    (re.compile(r"wp::length_sq\s*\(\s*([^()]+?)\s*\)"), r"metal::dot(\1, \1)"),
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
    # ``wp::spatial_top(v)`` / ``wp::spatial_bottom(v)`` — extract the upper
    # / lower vec3 of a ``spatial_vector`` (vec_t<6, float32>). The named
    # helpers are emitted in the kernel header alongside the wp_vec6_float
    # struct itself.
    (re.compile(r"\bwp::spatial_top\b"), "wp_spatial_top"),
    (re.compile(r"\bwp::spatial_bottom\b"), "wp_spatial_bottom"),
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
    # MSL declarations to inject before the kernel function body — used for
    # custom big-vec structs (vec5, vec6 = spatial_vector, vec8) that don't
    # have native MSL ``floatN`` equivalents. Empty for kernels that only
    # use native types.
    header: str = ""


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


_SLICE_T_PAT = re.compile(
    r"^\s*var_(\w+)\s*=\s*wp::slice_t\s*\(\s*var_(\w+)\s*,\s*var_(\w+)\s*,\s*var_(\w+)\s*\)\s*;\s*$"
)
_VIEW_PAT = re.compile(r"^\s*var_(\w+)\s*=\s*wp::view\s*\(\s*var_(\w+)((?:\s*,\s*var_\w+)+)\s*\)\s*;\s*$")
_VIEW_ADDRESS_PAT = re.compile(
    r"^(?P<indent>\s*)var_(?P<local>\w+)\s*=\s*"
    r"wp::address\s*\(\s*var_(?P<arr>\w+)(?P<rest>(?:\s*,\s*var_\w+)*)\s*\)\s*;\s*$"
)
_VIEW_ARRAY_STORE_PAT = re.compile(
    r"^(?P<indent>\s*)wp::array_store\s*\(\s*var_(?P<arr>\w+)(?P<rest>(?:\s*,\s*[^()]+?)+)\s*\)\s*;\s*$"
)
_VIEW_ATOMIC_PAT = re.compile(
    r"^(?P<indent>\s*)var_(?P<local>\w+)\s*=\s*"
    r"wp::(?P<op>atomic_(?:add|sub|min|max))\s*\(\s*var_(?P<arr>\w+)\s*,\s*"
    r"(?P<idx>[^,()]+?)\s*,\s*(?P<val>[^()]+?)\s*\)\s*;\s*$"
)


def _preprocess_views(forward_lines: list[str], adj) -> tuple[list[str], set[str]]:
    """Rewrite ``slice_t`` + ``view`` IR patterns into direct array ops.

    Warp lowers ``arr[i]`` (single integer index on a multi-dim array) to:

        var_S = wp::slice_t(i, i, 0);   // step=0 means "integer index"
        var_V = wp::view(arr, var_S);
        ... uses of var_V ...

    We only support this integer-index pattern — full slices (``arr[i:j]``)
    aren't seen in mujoco_warp's hot path. The view's downstream uses
    (``wp::address``, ``wp::array_store``, ``wp::atomic_*``) are translated
    into the underlying-array equivalent with the leading slice index(es)
    prepended; the ``slice_t`` and ``view`` declaration lines drop out.

    Views passed to a user-defined function aren't supported (we'd need to
    inline the call) — caught later by the unsupported-intrinsic guard.

    Returns ``(rewritten_lines, skip_decls)`` — locals whose top-level
    declarations should be suppressed (the slice_t and view labels).
    """
    arg_label_set = {a.label for a in adj.args}

    const_int_vars: dict[str, int] = {}
    for var in adj.variables:
        if var.constant is not None and isinstance(var.constant, int):
            const_int_vars[var.label] = var.constant

    slice_aliases: dict[str, str] = {}  # slice_local -> idx_label
    for raw in forward_lines:
        m = _SLICE_T_PAT.match(raw)
        if not m:
            continue
        local_label, start_l, stop_l, step_l = m.group(1), m.group(2), m.group(3), m.group(4)
        if start_l == stop_l and const_int_vars.get(step_l) == 0:
            slice_aliases[local_label] = start_l

    view_aliases: dict[str, tuple[str, list[str]]] = {}
    for raw in forward_lines:
        m = _VIEW_PAT.match(raw)
        if not m:
            continue
        view_label = m.group(1)
        arr_name = m.group(2)
        slice_labels = re.findall(r"var_(\w+)", m.group(3))
        if arr_name not in arg_label_set:
            continue
        if not all(s in slice_aliases for s in slice_labels):
            continue
        view_aliases[view_label] = (arr_name, [slice_aliases[s] for s in slice_labels])

    skip_decls: set[str] = set()
    skip_decls.update(slice_aliases.keys())
    skip_decls.update(view_aliases.keys())

    if not view_aliases:
        return forward_lines, skip_decls

    out_lines: list[str] = []
    for raw in forward_lines:
        m = _SLICE_T_PAT.match(raw)
        if m and m.group(1) in slice_aliases:
            continue
        m = _VIEW_PAT.match(raw)
        if m and m.group(1) in view_aliases:
            continue

        m = _VIEW_ADDRESS_PAT.match(raw)
        if m and m.group("arr") in view_aliases:
            indent = m.group("indent")
            local = m.group("local")
            arr_name, lead_idx_labels = view_aliases[m.group("arr")]
            tail_indices = re.findall(r"var_\w+", m.group("rest"))
            all_indices = [f"var_{l}" for l in lead_idx_labels] + tail_indices
            out_lines.append(f"{indent}var_{local} = wp::address(var_{arr_name}, {', '.join(all_indices)});")
            continue

        m = _VIEW_ARRAY_STORE_PAT.match(raw)
        if m and m.group("arr") in view_aliases:
            indent = m.group("indent")
            arr_name, lead_idx_labels = view_aliases[m.group("arr")]
            rest_parts = [p.strip() for p in m.group("rest").split(",") if p.strip()]
            lead_strs = [f"var_{l}" for l in lead_idx_labels]
            out_lines.append(f"{indent}wp::array_store(var_{arr_name}, {', '.join(lead_strs + rest_parts)});")
            continue

        m = _VIEW_ATOMIC_PAT.match(raw)
        if m and m.group("arr") in view_aliases:
            indent = m.group("indent")
            local = m.group("local")
            op = m.group("op")
            arr_name, lead_idx_labels = view_aliases[m.group("arr")]
            tail_idx = m.group("idx").strip()
            val = m.group("val").strip()
            all_indices = [f"var_{l}" for l in lead_idx_labels] + [tail_idx]
            # ``wp::atomic_*`` intrinsic regex disallows parens in its index
            # arg — build a flat-index expression without the outer parens
            # that ``_flat_index_expr`` would add.
            n = len(all_indices)
            if n == 1:
                flat_idx = all_indices[0]
            else:
                terms = []
                for k, idx in enumerate(all_indices):
                    if k == n - 1:
                        terms.append(idx)
                    else:
                        stride = " * ".join(f"{arr_name}_shape[{j}]" for j in range(k + 1, n))
                        terms.append(f"{idx} * {stride}")
                flat_idx = " + ".join(terms)
            out_lines.append(f"{indent}var_{local} = wp::{op}(var_{arr_name}, {flat_idx}, {val});")
            continue

        out_lines.append(raw)

    return out_lines, skip_decls


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
    # Slice/view preprocessing: ``arr[i]`` on a multi-dim array becomes a
    # ``slice_t`` + ``view`` pair we fold into direct array ops on the
    # underlying argument. Emits no extra MSL — the slice_t and view locals
    # become declaration-skipped aliases.
    forward_lines, view_skip_decls = _preprocess_views(forward_lines, adj)
    vars_to_skip_decl |= view_skip_decls

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
    # *every* output is typed ``device atomic<T>*``. We can still support
    # kernels that mix ``wp.atomic_*`` and plain ``arr[i] = val`` writes on
    # *different* outputs by translating the regular-store half into
    # ``atomic_store_explicit`` — one atomic op per scalar element written.
    # Reads from atomic-typed outputs are not yet supported (not seen in the
    # mujoco_warp recon for the affected kernels), and writes through a
    # field of an ``atomic<T>*`` element type aren't representable in MSL,
    # so non-scalar output dtypes still go through the same per-component
    # expansion they always did.
    has_atomic = bool(atomic_arg_names)

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
    # Map argname -> (vec_size, msl_scalar) for each vec-typed array arg,
    # argname -> (rows, cols, msl_scalar) for each mat-typed array arg, and
    # argname -> _StructLayout for each struct-typed array arg.
    vec_arr_info: dict[str, tuple[int, str]] = {}
    mat_arr_info: dict[str, tuple[int, int, str]] = {}
    struct_arr_info: dict[str, _StructLayout] = {}
    for arg in adj.args:
        v_info = _vec_dtype_info(arg)
        if v_info is not None:
            vec_arr_info[arg.label] = v_info
            continue
        m_info = _mat_dtype_info(arg)
        if m_info is not None:
            mat_arr_info[arg.label] = m_info
            continue
        s_info = _struct_dtype_info(arg)
        if s_info is not None:
            struct_arr_info[arg.label] = s_info

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

    # Track ``var_X = wp::address(var_struct_arr, var_idx);`` so the struct
    # field-pointer pass below can resolve ``var_X->field`` back to the
    # array+index pair.
    struct_refs: dict[str, tuple[str, str]] = {}  # local_label -> (arr_name, elem_idx_expr)

    for raw in forward_lines:
        m = _ADDRESS_MULTI_PAT.match(raw)
        if m:
            local_label = m.group("local")
            arr_arg = m.group("arr")
            if arr_arg not in arg_label_set:
                continue
            indices = re.findall(r"var_(\w+)", m.group("rest"))
            index_var_names = [f"var_{i}" for i in indices]
            if arr_arg in struct_arr_info:
                # Don't put a scalar subscript in ``subscript_map`` — the
                # struct pointer itself isn't directly used; only its
                # ``->field`` accesses (handled in the next pass).
                if len(index_var_names) == 1:
                    elem_idx_expr = index_var_names[0]
                else:
                    elem_idx_expr = f"({_flat_index_expr(arr_arg, index_var_names)})"
                struct_refs[local_label] = (arr_arg, elem_idx_expr)
                continue
            if arr_arg in vec_arr_info:
                # Vec-typed array: each *element* is ``vec_n`` consecutive
                # scalars. Compute the linear element index from the user-
                # visible ndim-many indices, then expand each component
                # access. Works for any ndim because the MLX view shape and
                # ``_flat_index_expr`` both use the user-visible dims (the
                # vec component is the innermost MLX dim, not part of the
                # element index).
                vec_n, msl_scalar = vec_arr_info[arr_arg]
                msl_vec_ctor = _msl_vec_ctor(vec_n, msl_scalar)
                if len(index_var_names) == 1:
                    elem_idx = index_var_names[0]
                else:
                    elem_idx = f"({_flat_index_expr(arr_arg, index_var_names)})"
                comps = [f"{arr_arg}[{elem_idx} * {vec_n} + {k}]" for k in range(vec_n)]
                subscript_map[local_label] = f"{msl_vec_ctor}({', '.join(comps)})"
            elif arr_arg in mat_arr_info:
                # Mat-typed array: each *element* is ``rows*cols`` consecutive
                # row-major-stored scalars. Build the column-major MSL
                # ``floatRxC`` from those scalars (column k = M[*][k]).
                rows, cols, msl_scalar = mat_arr_info[arr_arg]
                stride = rows * cols
                msl_vec_type = f"{msl_scalar}{rows}"
                msl_mat_type = f"{msl_scalar}{rows}x{cols}"
                if len(index_var_names) == 1:
                    elem_idx = index_var_names[0]
                else:
                    elem_idx = f"({_flat_index_expr(arr_arg, index_var_names)})"
                col_strs: list[str] = []
                for c in range(cols):
                    col_components = [f"{arr_arg}[{elem_idx} * {stride} + {r * cols + c}]" for r in range(rows)]
                    col_strs.append(f"{msl_vec_type}({', '.join(col_components)})")
                subscript_map[local_label] = f"{msl_mat_type}({', '.join(col_strs)})"
            else:
                subscript_map[local_label] = f"{arr_arg}[{_flat_index_expr(arr_arg, index_var_names)}]"

    # ---- Struct LOCAL variables (``q = SomeStruct(); q.field = ...``) ----
    # We don't materialise the struct; instead, each field becomes its own
    # MSL local (scalar / vec / mat). Field reads/writes route through those
    # per-field locals via the same ``subscript_map`` mechanism used for
    # struct-array fields.
    from warp._src.codegen import Struct as _Struct  # noqa: PLC0415

    struct_local_layouts: dict[str, _StructLayout] = {}
    struct_local_is_arg: dict[str, bool] = {}
    for var in adj.variables:
        if isinstance(var.type, _Struct):
            struct_local_layouts[var.label] = _struct_layout_for(var.type)
            struct_local_is_arg[var.label] = False
    # Struct-typed kernel args: the launcher serialises the struct instance
    # into a flat scalar buffer (see ``_array_view_dtype_and_shape`` for
    # arrays; struct args use the same per-field layout). The kernel sees
    # ``device const float* <argname>`` plus our auto-generated
    # ``<argname>_shape``, and field accesses translate to direct subscripts
    # at the field's scalar offset.
    struct_arg_layouts: dict[str, _StructLayout] = {}
    for arg in adj.args:
        if isinstance(arg.type, _Struct):
            struct_arg_layouts[arg.label] = _struct_layout_for(arg.type)

    def _per_field_local(struct_label: str, field_name: str) -> str:
        # Double underscore separates struct label from field name to avoid
        # clashes with raw Warp local labels (which are integers).
        return f"var_{struct_label}__{field_name}"

    # Struct field pointer pass: handle BOTH ``->`` (struct-array refs) and
    # ``.`` (struct locals) field addresses. The result is the same shape:
    # ``subscript_map[field_local]`` gets an expression that's used as the
    # value when ``wp::load`` reads it, and as the LHS when ``wp::store``
    # writes through it.
    struct_field_addr_pat = re.compile(r"^\s*var_(\w+)\s*=\s*&\s*\(\s*var_(\w+)\s*(->|\.)\s*(\w+)\s*\)\s*;\s*$")
    for raw in forward_lines:
        m = struct_field_addr_pat.match(raw)
        if not m:
            continue
        field_local = m.group(1)
        struct_local = m.group(2)
        accessor = m.group(3)
        field_name = m.group(4)

        if accessor == ".":
            # Struct-local field: alias to the per-field MSL local.
            if struct_local in struct_local_layouts:
                layout = struct_local_layouts[struct_local]
                if field_name not in layout.fields:
                    raise MetalCodegenError(
                        f"Kernel {adj.fun_name!r}: struct {layout.name!r} has no field {field_name!r}"
                    )
                subscript_map[field_local] = _per_field_local(struct_local, field_name)
                continue
            # Struct-arg field: the launcher serialises the struct into a
            # flat scalar buffer of length ``scalars_per_elem``; field
            # accesses are direct subscripts at the field's offset.
            if struct_local in struct_arg_layouts:
                layout = struct_arg_layouts[struct_local]
                field_info = layout.fields.get(field_name)
                if field_info is None:
                    raise MetalCodegenError(
                        f"Kernel {adj.fun_name!r}: struct arg {struct_local!r} has no field {field_name!r}"
                    )
                base = str(field_info.offset)
                if field_info.kind == _STRUCT_FIELD_KIND_SCALAR:
                    subscript_map[field_local] = f"{struct_local}[{base}]"
                elif field_info.kind == _STRUCT_FIELD_KIND_VEC:
                    comps = [f"{struct_local}[{base} + {k}]" for k in range(field_info.size)]
                    ctor = (
                        field_info.msl_type if field_info.size in _MSL_VEC_NATIVE_N else f"{field_info.msl_type}_make"
                    )
                    subscript_map[field_local] = f"{ctor}({', '.join(comps)})"
                elif field_info.kind == _STRUCT_FIELD_KIND_MAT:
                    rows, cols = field_info.rows, field_info.cols
                    msl_vec = field_info.msl_type.split("x")[0]
                    col_strs: list[str] = []
                    for c in range(cols):
                        col_components = [f"{struct_local}[{base} + {r * cols + c}]" for r in range(rows)]
                        col_strs.append(f"{msl_vec}({', '.join(col_components)})")
                    subscript_map[field_local] = f"{field_info.msl_type}({', '.join(col_strs)})"
                continue
            continue

        # accessor == "->": struct-array field, build a flat-buffer expression.
        if struct_local not in struct_refs:
            continue
        arr_name, elem_idx_expr = struct_refs[struct_local]
        layout = struct_arr_info[arr_name]
        field_info = layout.fields.get(field_name)
        if field_info is None:
            raise MetalCodegenError(f"Kernel {adj.fun_name!r}: struct {layout.name!r} has no field {field_name!r}")
        base = f"{elem_idx_expr} * {layout.scalars_per_elem} + {field_info.offset}"
        if field_info.kind == _STRUCT_FIELD_KIND_SCALAR:
            subscript_map[field_local] = f"{arr_name}[{base}]"
        elif field_info.kind == _STRUCT_FIELD_KIND_VEC:
            comps = [f"{arr_name}[({base}) + {k}]" for k in range(field_info.size)]
            ctor = field_info.msl_type if field_info.size in _MSL_VEC_NATIVE_N else f"{field_info.msl_type}_make"
            subscript_map[field_local] = f"{ctor}({', '.join(comps)})"
        elif field_info.kind == _STRUCT_FIELD_KIND_MAT:
            rows, cols = field_info.rows, field_info.cols
            msl_vec = field_info.msl_type.split("x")[0]  # e.g. "float3" from "float3x3"
            col_strs: list[str] = []
            for c in range(cols):
                col_components = [f"{arr_name}[({base}) + {r * cols + c}]" for r in range(rows)]
                col_strs.append(f"{msl_vec}({', '.join(col_components)})")
            subscript_map[field_local] = f"{field_info.msl_type}({', '.join(col_strs)})"

    # --- Local variable declarations -----------------------------------
    body_lines: list[str] = []
    for var in adj.variables:
        if var.label in subscript_map:
            # This local was a pointer into an array arg; we'll inline its
            # uses below, so it doesn't need a declaration.
            continue
        if var.label in struct_refs:
            # Struct-array pointer local — its ``->field`` accesses go
            # through ``subscript_map`` and the struct-pointer itself is
            # never used in code we emit.
            continue
        if var.label in struct_local_layouts:
            # Struct *value* local — emit per-field locals instead, each
            # zero-initialised so default-constructed structs behave as on
            # CPU. The original struct local (var_X with ctype like
            # ``Particle_4b7eabdf``) never appears in our emitted code.
            layout = struct_local_layouts[var.label]
            for field_name, field_info in layout.fields.items():
                local_name = _per_field_local(var.label, field_name)
                # MSL ``T()`` zero-constructs scalar / vec / mat values.
                body_lines.append(f"    {field_info.msl_type} {local_name} = {field_info.msl_type}(0);")
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
        # Normalize ``wp::quat_t<wp::T>`` to ``wp::vec_t<4, wp::T>`` so the
        # existing vec_t machinery (constructor rewrite, type translation,
        # array dtype detection) handles quat as a 4-element vec.
        translated = _normalize_quat_t(translated)
        # Rewrite ``wp::mat_t<R, C, ...>(...)`` constructor calls first —
        # the row-major flat args need reordering into column form.
        # Likewise, ``wp::vec_t<N, ...>({v0, v1, ...})`` (the brace form
        # Warp uses for vec5+) needs the braces stripped and the type
        # renamed before the bare-type translator below sees it.
        translated = _rewrite_mat_t_constructor(translated)
        translated = _rewrite_vec_t_brace_constructor(translated)
        # Bare ``wp::vec_t<N, ...>`` and ``wp::mat_t<R, C, ...>`` type
        # references (in declarations etc.) get renamed to the MSL native
        # type for native sizes, or our custom ``wp_vecN_<scalar>`` for big
        # sizes.
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
        # Struct field pointer line — the field expression is in
        # ``subscript_map`` and gets inlined when the field-ptr local is
        # referenced via ``wp::load``.
        if struct_field_addr_pat.match(raw):
            continue
        # Struct constructor line ``var_X = StructName_<hash>();`` — the
        # per-field locals are zero-initialised at declaration so this is
        # a no-op. Match by checking var_X is a known struct local and the
        # call has no args.
        m_ctor = re.match(r"^\s*var_(\w+)\s*=\s*\w+\s*\(\s*\)\s*;\s*$", raw)
        if m_ctor and m_ctor.group(1) in struct_local_layouts:
            continue
        # ``wp::store(addr_var, value);`` — write through a field pointer.
        # We translate by looking up the LHS expression we recorded in
        # ``subscript_map`` for ``addr_var``.
        # ``wp::store`` and the in-place compound forms (``add_inplace``,
        # ``sub_inplace``, ``mul_inplace``, ``assign_inplace``) all write
        # through a field pointer. Translate via the LHS expression
        # recorded in ``subscript_map`` for the address.
        store_op_map = {
            "store": "=",
            "assign_inplace": "=",
            "add_inplace": "+=",
            "sub_inplace": "-=",
            "mul_inplace": "*=",
            "div_inplace": "/=",
        }
        store_pat = re.compile(
            r"^(?P<indent>\s*)wp::(?P<op>store|assign_inplace|add_inplace|sub_inplace|"
            r"mul_inplace|div_inplace)\s*\(\s*var_(?P<addr>\w+)\s*,\s*(?P<val>[^()]+?)\s*\)\s*;\s*$"
        )
        m_store = store_pat.match(raw)
        if m_store:
            addr = m_store.group("addr")
            value = m_store.group("val")
            indent = m_store.group("indent")
            op = store_op_map[m_store.group("op")]
            if addr in subscript_map:
                lhs = subscript_map[addr]
                body_lines.append(_finalize(f"{indent}{lhs} {op} {value};"))
                continue
            # Stray store — pass through; the unsupported-intrinsic
            # guard will catch it.
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
                # When the kernel uses ``wp.atomic_*`` on any output, MLX
                # makes ALL outputs ``device atomic<T>*`` and a plain
                # ``arr[idx] = val`` won't compile — wrap each scalar
                # write in ``atomic_store_explicit`` instead.
                output_arg_names = {a.label for a in output_args}
                use_atomic_store = has_atomic and arr in output_arg_names

                def _emit_scalar_write(
                    idx_expr: str,
                    rhs: str,
                    indent: str = indent,
                    arr: str = arr,
                    use_atomic_store: bool = use_atomic_store,
                ) -> str:
                    if use_atomic_store:
                        return f"{indent}atomic_store_explicit(&{arr}[{idx_expr}], {rhs}, memory_order_relaxed);"
                    return f"{indent}{arr}[{idx_expr}] = {rhs};"

                if arr in vec_arr_info:
                    vec_n, _ = vec_arr_info[arr]
                    if len(indices) == 1:
                        elem_idx = indices[0]
                    else:
                        elem_idx = f"({_flat_index_expr(arr, indices)})"
                    for k in range(vec_n):
                        body_lines.append(_finalize(_emit_scalar_write(f"{elem_idx} * {vec_n} + {k}", f"{value}[{k}]")))
                    continue
                if arr in mat_arr_info:
                    rows, cols, _ = mat_arr_info[arr]
                    stride = rows * cols
                    if len(indices) == 1:
                        elem_idx = indices[0]
                    else:
                        elem_idx = f"({_flat_index_expr(arr, indices)})"
                    # Scatter to row-major storage: data[i*RC + r*C + c] =
                    # logical M[r][c] = MSL ``value[c][r]`` (column, then row).
                    for r in range(rows):
                        for c in range(cols):
                            body_lines.append(
                                _finalize(
                                    _emit_scalar_write(
                                        f"{elem_idx} * {stride} + {r * cols + c}",
                                        f"{value}[{c}][{r}]",
                                    )
                                )
                            )
                    continue
                if arr in struct_arr_info:
                    # ``arr[i] = struct_local`` — scatter each field of the
                    # struct local into its slot in the flat array buffer.
                    layout = struct_arr_info[arr]
                    if len(indices) == 1:
                        elem_idx = indices[0]
                    else:
                        elem_idx = f"({_flat_index_expr(arr, indices)})"
                    # Strip leading "var_" if present so we can reuse the
                    # per-field-local naming convention.
                    val_struct_label = value[len("var_") :] if value.startswith("var_") else value
                    if val_struct_label not in struct_local_layouts:
                        raise MetalCodegenError(
                            f"Kernel {adj.fun_name!r}: ``arr[i] = X`` where X ({value!r}) is not a "
                            "struct local; only stores from struct locals are supported on Metal"
                        )
                    base = f"{elem_idx} * {layout.scalars_per_elem}"
                    for fname, finfo in layout.fields.items():
                        src = _per_field_local(val_struct_label, fname)
                        off = finfo.offset
                        if finfo.kind == _STRUCT_FIELD_KIND_SCALAR:
                            body_lines.append(_finalize(_emit_scalar_write(f"{base} + {off}", src)))
                        elif finfo.kind == _STRUCT_FIELD_KIND_VEC:
                            for k in range(finfo.size):
                                body_lines.append(_finalize(_emit_scalar_write(f"{base} + {off} + {k}", f"{src}[{k}]")))
                        elif finfo.kind == _STRUCT_FIELD_KIND_MAT:
                            for r in range(finfo.rows):
                                for c in range(finfo.cols):
                                    body_lines.append(
                                        _finalize(
                                            _emit_scalar_write(
                                                f"{base} + {off + r * finfo.cols + c}",
                                                f"{src}[{c}][{r}]",
                                            )
                                        )
                                    )
                    continue
                flat_idx = _flat_index_expr(arr, indices)
                body_lines.append(_finalize(_emit_scalar_write(flat_idx, value)))
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

    header = _build_kernel_header(source)

    return MetalKernelArtifact(
        name=adj.fun_name,
        source=source,
        input_names=base_input_names + extra_input_names,
        output_names=[a.label for a in output_args],
        input_args=input_args,
        output_args=output_args,
        atomic_outputs=has_atomic,
        output_shape_inputs=output_shape_inputs,
        header=header,
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
    if getattr(dtype, "_wp_generic_type_str_", None) not in ("vec_t", "quat_t"):
        return None
    n = getattr(dtype, "_length_", None)
    scalar_cls = getattr(dtype, "_wp_scalar_type_", None)
    if n is None or scalar_cls is None:
        return None
    if n < 2:
        raise MetalCodegenError(f"MSL codegen does not support vec{n} (need N >= 2)")
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
# Struct (``@wp.struct``) layout
# ---------------------------------------------------------------------------
#
# We don't emit MSL ``struct`` declarations. Every read of a struct field on
# an element of a ``wp.array(dtype=SomeStruct)`` is expanded into direct
# scalar/vec/mat accesses on the underlying flat buffer. This avoids the
# address-space-cast issues that pointer-cast approaches hit and is
# structurally similar to how vec/mat arrays already work.
#
# Currently supported: read-only field access on ``wp.array(dtype=SomeStruct)``
# for POD structs whose fields are scalars / vec / mat. The supported scalar
# field width is 4 bytes (``float32``, ``int32``, ``uint32``); mixed-width
# fields, nested structs, and array fields would need different handling and
# are rejected with a clear error.
#
# Currently NOT supported: local struct construction (``q = SomeStruct()``),
# struct-typed kernel args (``def k(s: SomeStruct, ...)``), stores of struct
# values into arrays. Those raise ``MetalCodegenError``.

_STRUCT_FIELD_KIND_SCALAR = "scalar"
_STRUCT_FIELD_KIND_VEC = "vec"
_STRUCT_FIELD_KIND_MAT = "mat"


@dataclass
class _StructFieldInfo:
    name: str
    offset: int  # offset within the struct, in scalars (4-byte units)
    size: int  # number of scalars
    kind: str  # one of ``_STRUCT_FIELD_KIND_*``
    msl_type: str  # MSL name (``float`` / ``float3`` / ``float3x3`` etc.)
    rows: int = 0
    cols: int = 0


@dataclass
class _StructLayout:
    name: str  # mangled name from Warp's struct (``Particle_4b7eabdf``)
    fields: dict[str, _StructFieldInfo] = field(default_factory=dict)
    scalars_per_elem: int = 0


def _classify_struct_field(fname: str, ftype) -> tuple[str, int, str, int, int]:
    """Return ``(kind, size_in_scalars, msl_type, rows, cols)`` for a field type."""
    # vec_t (and quat_t — laid out identically to vec4)
    if getattr(ftype, "_wp_generic_type_str_", None) in ("vec_t", "quat_t"):
        n = int(ftype._length_)
        scalar_cls = ftype._wp_scalar_type_
        scalar_ctype = f"wp::{scalar_cls.__name__}"
        if n < 2 or scalar_ctype not in _MSL_VEC_SCALAR_PREFIX:
            raise MetalCodegenError(f"MSL codegen does not support vec field {fname!r} of {ftype!r} in a struct")
        msl_scalar = _MSL_VEC_SCALAR_PREFIX[scalar_ctype]
        return _STRUCT_FIELD_KIND_VEC, n, _msl_vec_name(n, msl_scalar), 0, 0
    # mat_t
    if getattr(ftype, "_wp_generic_type_str_", None) == "mat_t":
        rows, cols = int(ftype._shape_[0]), int(ftype._shape_[1])
        scalar_cls = ftype._wp_scalar_type_
        scalar_ctype = f"wp::{scalar_cls.__name__}"
        if (
            rows not in _MSL_VEC_SUPPORTED_N
            or cols not in _MSL_VEC_SUPPORTED_N
            or scalar_ctype not in _MSL_VEC_SCALAR_PREFIX
        ):
            raise MetalCodegenError(f"MSL codegen does not support mat field {fname!r} of {ftype!r} in a struct")
        msl_scalar = _MSL_VEC_SCALAR_PREFIX[scalar_ctype]
        return _STRUCT_FIELD_KIND_MAT, rows * cols, f"{msl_scalar}{rows}x{cols}", rows, cols
    # Scalar
    name = getattr(ftype, "__name__", None)
    if name is None:
        raise MetalCodegenError(f"MSL codegen: struct field {fname!r} has unsupported type {ftype!r}")
    scalar_ctype = f"wp::{name}"
    if scalar_ctype == "wp::float64":
        raise MetalCodegenError(f"MSL codegen: struct field {fname!r} is float64 — MSL has no fp64")
    if scalar_ctype not in _SCALAR_CTYPE_TO_MSL:
        raise MetalCodegenError(
            f"MSL codegen does not yet support struct field {fname!r} of type {ftype!r} "
            f"(no MSL scalar mapping for {scalar_ctype})"
        )
    return _STRUCT_FIELD_KIND_SCALAR, 1, _SCALAR_CTYPE_TO_MSL[scalar_ctype], 0, 0


def _struct_layout_for(struct_cls) -> _StructLayout:
    """Build a ``_StructLayout`` for a Warp ``Struct`` instance."""
    layout = _StructLayout(name=getattr(struct_cls, "key", "anonymous_struct"))
    offset = 0
    for fname, fvar in getattr(struct_cls, "vars", {}).items():
        kind, size, msl_type, rows, cols = _classify_struct_field(fname, fvar.type)
        layout.fields[fname] = _StructFieldInfo(
            name=fname, offset=offset, size=size, kind=kind, msl_type=msl_type, rows=rows, cols=cols
        )
        offset += size
    layout.scalars_per_elem = offset
    return layout


def _struct_dtype_info(arg) -> _StructLayout | None:
    """If ``arg`` is a ``wp.array`` whose element type is a Warp ``Struct``,
    return a layout describing its field offsets. Otherwise return ``None``.
    """
    from warp._src.codegen import Struct  # noqa: PLC0415

    if not _is_array_arg(arg):
        return None
    dtype = getattr(arg.type, "dtype", None)
    if not isinstance(dtype, Struct):
        return None
    return _struct_layout_for(dtype)


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
        wpt.bool: mx.bool_,
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

    For struct-typed arrays (``wp.array(dtype=SomeStruct)``), the storage is
    ``value.size * scalars_per_elem`` scalars exposed as float32 (matches
    Warp's tight C packing for the supported all-4-byte-field POD structs).
    The kernel-side body computes per-field offsets itself.
    """
    import mlx.core as mx  # noqa: PLC0415

    from warp._src.codegen import Struct  # noqa: PLC0415

    dtype = value.dtype
    kind = getattr(dtype, "_wp_generic_type_str_", None)
    if kind in ("vec_t", "quat_t"):
        n = dtype._length_
        scalar_cls = dtype._wp_scalar_type_
        mx_dtype = _wp_dtype_to_mx_dtype(scalar_cls)
        return mx_dtype, (*value.shape, n)
    if kind == "mat_t":
        rows, cols = dtype._shape_
        scalar_cls = dtype._wp_scalar_type_
        mx_dtype = _wp_dtype_to_mx_dtype(scalar_cls)
        return mx_dtype, (*value.shape, rows * cols)
    if isinstance(dtype, Struct):
        layout = _struct_layout_for(dtype)
        return mx.float32, (*value.shape, layout.scalars_per_elem)
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
            header=artifact.header,
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

    from warp._src.codegen import Struct  # noqa: PLC0415
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
        elif isinstance(arg_var.type, Struct):
            # Struct-typed arg: serialise the user's ``StructInstance`` into
            # a flat scalar buffer the kernel can index. The struct's
            # ``_ctype`` member is a ``ctypes.Structure`` populated by the
            # field setters Warp generates, so ``bytes(...)`` gives us the
            # right tight-packed layout. We view it as ``mx.float32`` —
            # consistent with how struct *arrays* expose their storage.
            layout = _struct_layout_for(arg_var.type)
            ctype_inst = getattr(value, "_ctype", None)
            if ctype_inst is None:
                raise RuntimeError(
                    f"Kernel '{kernel.key}' arg '{input_name}' is a struct but the "
                    f"Python value lacks a ``_ctype`` member; pass a real "
                    f"``StructInstance`` (e.g. one constructed via ``MyStruct()``)"
                )
            raw = bytes(ctype_inst)
            np_buf = np.frombuffer(raw, dtype=np.float32).copy()
            if np_buf.size != layout.scalars_per_elem:
                raise RuntimeError(
                    f"Kernel '{kernel.key}' arg '{input_name}': struct "
                    f"serialisation produced {np_buf.size} float32s but the "
                    f"computed layout expects {layout.scalars_per_elem}"
                )
            mlx_inputs.append(mx.array(np_buf, dtype=mx.float32))
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
