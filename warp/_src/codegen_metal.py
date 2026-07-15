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

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from warp._src.codegen_metal_ast import emit as _ast_emit
from warp._src.codegen_metal_ast import fold as _ast_fold
from warp._src.codegen_metal_ast import fold_drop_unsupported_locals as _ast_fold_drop
from warp._src.codegen_metal_ast import fold_indexref_writes as _ast_fold_indexref
from warp._src.codegen_metal_ast import fold_multidim_atomics as _ast_fold_multidim_atomics
from warp._src.codegen_metal_ast import fold_views as _ast_fold_views
from warp._src.codegen_metal_ast import inline_user_calls as _ast_inline
from warp._src.codegen_metal_ast import parse as _ast_parse

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
    # ``wp::slice_t`` is a 3-int struct (start, stop, step) used by
    # matrix slicing builtins (``mat[:, c]`` / ``mat[r, :]``). The MSL
    # equivalent ``wp_slice_t`` struct is emitted in the kernel header.
    "wp::slice_t": "wp_slice_t",
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
# ``wp::transform_t<wp::TYPE>`` — a rigid transform laid out as 7 scalars
# (px, py, pz, qx, qy, qz, qw). Normalized to ``vec_t<7>`` the same way
# quats become ``vec_t<4>``; the big-vec struct machinery then covers
# storage, constructors, and arithmetic, while the ``wp_transform_*``
# helpers cover the transform-specific operations.
_WP_TRANSFORM_T_PAT = re.compile(r"wp::transform_t\s*<\s*wp::(\w+)\s*>")


def _normalize_quat_t(text: str) -> str:
    """Rewrite quat/transform type references to their vec_t equivalents.

    ``wp::quat_t<wp::T>`` becomes ``wp::vec_t<4, wp::T>`` and
    ``wp::transform_t<wp::T>`` becomes ``wp::vec_t<7, wp::T>``. Run before
    any other vec/mat handling so the existing vec paths cover both
    seamlessly.
    """
    text = _WP_QUAT_T_PAT.sub(lambda m: f"wp::vec_t<4, wp::{m.group(1)}>", text)
    return _WP_TRANSFORM_T_PAT.sub(lambda m: f"wp::vec_t<7, wp::{m.group(1)}>", text)


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


def _msl_mat_name(rows: int, cols: int, msl_scalar: str) -> str:
    """Return the MSL type name for ``mat_t<R, C, T>``.

    Native ``floatRxC`` only for *square* sizes — MSL's row-vs-column
    access semantics (``m[i]`` returns the i-th *MSL column*, which is
    Warp's "row i" only when the matrix is square — otherwise the
    shape doesn't even match). Non-square sizes route through our
    custom ``wp_mat{R}x{C}_<scalar>`` struct (row-major ``c[R*C]``)
    so reads and writes via ``m[i]`` go through the proxy that
    returns a Warp-row-shaped vector. Cartpole's ``mat_t<2, 3>``
    surfaced this — we now emit ``wp_mat2x3_float`` instead of MSL
    native ``float2x3``.
    """
    if rows == cols and rows in _MSL_VEC_NATIVE_N:
        return f"{msl_scalar}{rows}x{cols}"
    return f"wp_mat{rows}x{cols}_{msl_scalar}"


def _mat_t_to_msl(rows: int, cols: int, scalar_ctype: str) -> str:
    """Translate ``wp::mat_t<R, C, wp::TYPE>`` to its MSL name.

    For sizes in {2, 3, 4} per dim this returns the native ``floatRxC``.
    For larger sizes it returns ``wp_matRxC_<scalar>`` for which the
    codegen emits a custom struct in the kernel header (alongside the
    big-vec structs).
    """
    if rows < 2 or cols < 2:
        raise MetalCodegenError(f"MSL codegen: mat_t<{rows}, {cols}, ...> not supported (need >= 2 per dim)")
    full_ctype = f"wp::{scalar_ctype}"
    if full_ctype not in _MSL_VEC_SCALAR_PREFIX:
        raise MetalCodegenError(f"MSL codegen does not yet support mat_t element type {full_ctype!r}")
    return _msl_mat_name(rows, cols, _MSL_VEC_SCALAR_PREFIX[full_ctype])


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
# Match ``wp_vec<N>_<scalar>`` (struct type or factory call). The
# scalar set is enumerated to anchor the match — a lazy ``(\w+)``
# would greedily pick up ``float_make`` from ``wp_vec6_float_make``,
# classifying the scalar as "float_make" and missing the struct
# emission. The optional ``_make`` suffix lets the same regex match
# both the bare struct name and the factory call.
_BIG_VEC_NAME_PAT = re.compile(
    r"\bwp_vec(\d+)_(half|float|double|int|uint|long|ulong|char|uchar|short|ushort|bool)"
    r"(?:_make)?\b"
)
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
    # Single-scalar broadcast overload: Warp's IR emits
    # ``wp_vecN_<scalar>_make(scalar)`` for ``vec_t<N>(s)`` patterns
    # (e.g. ``wp.vec8(0.0)`` to zero-init or fill).
    body_lines.append(f"inline {name} {name}_make({msl_scalar} v) {{")
    body_lines.append(f"    {name} r;")
    for i in range(n):
        body_lines.append(f"    r.c[{i}] = v;")
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
    # Scalar operators. Warp lowers ``vec * scalar`` and ``scalar * vec``
    # (and the equivalents for ``+``, ``-``, ``/``) to plain ``*``/``/``
    # binary expressions, so MSL needs both orderings. Without these,
    # kernels like ``cdof[dofid] * qvel[dofid]`` (spatial-vector ``*`` joint
    # velocity) fail to compile.
    for op in ("+", "-", "*", "/"):
        body_lines.append(f"inline {name} operator{op}({name} a, {msl_scalar} s) {{")
        body_lines.append(f"    {name} r;")
        for i in range(n):
            body_lines.append(f"    r.c[{i}] = a.c[{i}] {op} s;")
        body_lines.append("    return r;")
        body_lines.append("}")
        body_lines.append(f"inline {name} operator{op}({msl_scalar} s, {name} a) {{")
        body_lines.append(f"    {name} r;")
        for i in range(n):
            body_lines.append(f"    r.c[{i}] = s {op} a.c[{i}];")
        body_lines.append("    return r;")
        body_lines.append("}")
    # Unary minus: ``-spatial_vec``. The IR lowers ``wp.neg(v)`` to
    # ``-v`` for native vec types but for our big-vec structs the codegen
    # path still funnels through ``wp_neg`` (see ``_INTRINSIC_PATTERNS``).
    # Provide both forms so either lowering compiles.
    body_lines.append(f"inline {name} operator-({name} a) {{")
    body_lines.append(f"    {name} r;")
    for i in range(n):
        body_lines.append(f"    r.c[{i}] = -a.c[{i}];")
    body_lines.append("    return r;")
    body_lines.append("}")
    # Equality / inequality: ``vec == zero`` shows up in actuator gating
    # paths (e.g. ``_apply_ft`` checks whether a force-torque pair is the
    # zero spatial vector). Component-wise compare; results AND/OR'd.
    body_lines.append(f"inline bool operator==({name} a, {name} b) {{")
    body_lines.append("    return " + " && ".join(f"a.c[{i}] == b.c[{i}]" for i in range(n)) + ";")
    body_lines.append("}")
    body_lines.append(f"inline bool operator!=({name} a, {name} b) {{")
    body_lines.append("    return !(a == b);")
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


_BIG_MAT_NAME_PAT = re.compile(r"\bwp_mat(\d+)x(\d+)_(\w+)\b")


def _emit_big_mat_struct(name: str, rows: int, cols: int, msl_scalar: str) -> str:
    """Emit the MSL declaration for a custom big-mat struct.

    Storage is row-major (matches Warp's IR semantics directly) — element
    ``(r, c)`` lives at ``c[r * cols + c]``. Exposes:
      * Flat row-major ``_make`` factory.
      * ``operator[](int)`` for both read (returns a row vector by value)
        and write (returns a small proxy that overloads ``= vec`` to
        dispatch to per-element stores). Cartpole's narrowphase uses
        ``mat[i] = vec3`` (set row) and ``vec3 v = mat[i]`` (get row);
        the proxy resolves both.
      * ``_extract`` helper for ``wp::extract(m, r, c)``.
    No arithmetic operator overloads are emitted: the kernels we currently
    cover use big mats as struct fields (zero-init, row write, row read),
    not for matrix algebra.
    """
    n = rows * cols
    # Row-vector type. Native ``floatN`` / ``intN`` for cols in 2..4 and
    # supported scalars; otherwise the custom ``wp_vecN_<scalar>`` struct.
    if cols in (2, 3, 4) and msl_scalar in _MSL_PREFIX_TO_SAME:
        row_type = f"{msl_scalar}{cols}"
        row_make = f"{row_type}(" + ", ".join(f"c[i*{cols}+{j}]" for j in range(cols)) + ")"
        proxy_assign = "; ".join(f"p[{j}] = v[{j}]" for j in range(cols))
        proxy_read = f"{row_type}(" + ", ".join(f"p[{j}]" for j in range(cols)) + ")"
    else:
        row_type = f"wp_vec{cols}_{msl_scalar}"
        row_make = f"{row_type}{{" + ", ".join(f"c[i*{cols}+{j}]" for j in range(cols)) + "}"
        proxy_assign = "; ".join(f"p[{j}] = v.c[{j}]" for j in range(cols))
        proxy_read = f"{row_type}{{" + ", ".join(f"p[{j}]" for j in range(cols)) + "}"
    proxy = f"{name}_row_ref"
    body: list[str] = []
    # Forward-declare the proxy so the struct's non-const operator[]
    # signature can name it before the proxy body is emitted.
    body.append(f"struct {proxy};")
    body.append(f"struct {name} {{")
    body.append(f"    {msl_scalar} c[{n}];")
    body.append(f"    inline {row_type} operator[](int i) const thread {{ return {row_make}; }}")
    body.append(f"    inline {proxy} operator[](int i) thread;")
    body.append("};")
    body.append(f"struct {proxy} {{")
    body.append(f"    thread {msl_scalar}* p;")
    body.append(f"    inline operator {row_type}() const thread {{ return {proxy_read}; }}")
    body.append(f"    inline thread {proxy}& operator=({row_type} v) thread {{")
    body.append(f"        {proxy_assign};")
    body.append("        return *this;")
    body.append("    }")
    body.append("};")
    body.append(f"inline {proxy} {name}::operator[](int i) thread {{ return {proxy}{{&c[i*{cols}]}}; }}")
    args = ", ".join(f"{msl_scalar} v{i}" for i in range(n))
    body.append(f"inline {name} {name}_make({args}) {{")
    body.append(f"    {name} r;")
    for i in range(n):
        body.append(f"    r.c[{i}] = v{i};")
    body.append("    return r;")
    body.append("}")
    # Single-scalar broadcast: ``mat23(0.0)`` zero-fills.
    body.append(f"inline {name} {name}_make({msl_scalar} v) {{")
    body.append(f"    {name} r;")
    for i in range(n):
        body.append(f"    r.c[{i}] = v;")
    body.append("    return r;")
    body.append("}")
    body.append(f"inline {msl_scalar} wp_mat_extract({name} m, int row, int col) {{ return m.c[row * {cols} + col]; }}")
    return "\n".join(body)


_NATIVE_MAT_NAME_PAT = re.compile(r"\b(float|int|uint)([234])x([234])\b")


def _emit_slice_t_struct(source: str) -> str:
    """Emit the ``wp_slice_t`` struct definition if any kernel statement
    references it.  ``wp_slice_t`` is the MSL counterpart of
    ``wp::slice_t`` (start/stop/step ints) and is generated by matrix
    slicing builtins. The struct emit is independent of which matrix
    types appear in the kernel; some kernels declare a slice without
    using it on a native matrix (e.g. mujoco_warp's ``tendon_bias``
    builds slices over big-vec storage).
    """
    if "wp::slice_t" not in source and "wp_slice_t" not in source:
        return ""
    return "struct wp_slice_t { int start; int stop; int step; };"


def _emit_native_mat_extract_overloads(source: str) -> str:
    """Emit ``wp_mat_extract`` overloads for each native ``floatRxC`` /
    ``intRxC`` referenced in the kernel source.

    Native MSL matrices are column-major, so ``m[c][r]`` reads logical row
    ``r``, column ``c``. Wrapping that as a free function lets the same
    ``wp_mat_extract`` translation work for both native and big-mat types
    via overload resolution.

    Also emits slice-overloads of ``wp_mat_extract`` for the row/col
    slice cases (``mat[:, c]`` / ``mat[r, :]``).
    """
    seen: set[tuple[str, int, int]] = set()
    for m in _NATIVE_MAT_NAME_PAT.finditer(source):
        seen.add((m.group(1), int(m.group(2)), int(m.group(3))))
    if not seen:
        return ""
    parts: list[str] = []
    needs_slice = "wp::slice_t" in source or "wp_slice_t" in source
    for scalar, rows, cols in sorted(seen):
        parts.append(
            f"inline {scalar} wp_mat_extract({scalar}{rows}x{cols} m, int row, int col) {{ return m[col][row]; }}"
        )
        if needs_slice:
            # Row-slice: extract a vec spanning a row range at a fixed
            # col. The vec length follows the matrix row count (``[:, c]``
            # — full slice). Apple's MSL builds with NDEBUG, so the
            # vec-length parameter ``N`` from ``wp::extract<N>`` need not
            # be threaded through; full-slice vec dim is ``rows``.
            parts.append(
                f"inline {scalar}{rows} wp_mat_extract({scalar}{rows}x{cols} m, wp_slice_t s, int col) {{\n"
                f"    {scalar}{rows} ret = {scalar}{rows}(0);\n"
                f"    int ii = 0;\n"
                f"    for (int i = s.start; (s.step > 0) ? (i < s.stop) : (i > s.stop); i += s.step) {{\n"
                f"        ret[ii++] = m[col][i];\n"
                f"    }}\n"
                f"    return ret;\n"
                f"}}"
            )
            # Col-slice: vec spanning a col range at a fixed row.
            parts.append(
                f"inline {scalar}{cols} wp_mat_extract({scalar}{rows}x{cols} m, int row, wp_slice_t s) {{\n"
                f"    {scalar}{cols} ret = {scalar}{cols}(0);\n"
                f"    int ii = 0;\n"
                f"    for (int i = s.start; (s.step > 0) ? (i < s.stop) : (i > s.stop); i += s.step) {{\n"
                f"        ret[ii++] = m[i][row];\n"
                f"    }}\n"
                f"    return ret;\n"
                f"}}"
            )
    return "\n".join(parts)


_NATIVE_VEC_NAME_PAT = re.compile(r"\b(float|int|uint)([234])\b")


def _emit_wp_dot_overloads(source: str) -> str:
    """Emit ``wp_dot`` overloads for every vector type used in ``source``.

    MSL's ``metal::dot`` only knows about its native ``floatN`` etc.; our
    custom ``wp_vecN_<scalar>`` big-vec structs need a hand-rolled
    sum-of-products. Wrapping the call as ``wp_dot`` lets the codegen emit
    one uniform translation regardless of operand type — the C++ overload
    resolver picks the right body.
    """
    parts: list[str] = []
    seen_native: set[tuple[str, int]] = set()
    for m in _NATIVE_VEC_NAME_PAT.finditer(source):
        seen_native.add((m.group(1), int(m.group(2))))
    for scalar, n in sorted(seen_native):
        # ``metal::dot`` is float/half-only; integer overloads have to be
        # hand-rolled as a sum-of-products.
        if scalar in ("int", "uint"):
            body_terms = " + ".join(f"a[{i}] * b[{i}]" for i in range(n))
            parts.append(f"inline {scalar} wp_dot({scalar}{n} a, {scalar}{n} b) {{ return {body_terms}; }}")
        else:
            parts.append(f"inline {scalar} wp_dot({scalar}{n} a, {scalar}{n} b) {{ return metal::dot(a, b); }}")

    seen_big: set[tuple[int, str]] = set()
    for m in _BIG_VEC_NAME_PAT.finditer(source):
        n = int(m.group(1))
        scalar = m.group(2)
        if n in _MSL_VEC_NATIVE_N or scalar not in _MSL_PREFIX_TO_SAME:
            continue
        seen_big.add((n, scalar))
    for n, scalar in sorted(seen_big):
        body_terms = " + ".join(f"a.c[{i}] * b.c[{i}]" for i in range(n))
        parts.append(f"inline {scalar} wp_dot(wp_vec{n}_{scalar} a, wp_vec{n}_{scalar} b) {{ return {body_terms}; }}")
    return "\n".join(parts)


_DIAG_HELPER_FLOAT3 = (
    "inline float3x3 wp_diag_float3(float3 v) {\n"
    "    return float3x3(float3(v[0], 0.0f, 0.0f), "
    "float3(0.0f, v[1], 0.0f), "
    "float3(0.0f, 0.0f, v[2]));\n"
    "}"
)


# Quaternion helpers. Quats are stored as ``float4`` with (x, y, z, w)
# layout after the ``vec_t<4>`` normalization; the bodies mirror
# ``warp/native/quat.h`` exactly (including the ``l > 0`` normalize guard —
# Warp's ``kEps`` is 0.0f). Row-major ``m.data[r][c]`` accesses from the
# native code become column-major ``m[c][r]`` here.
_QUAT_HELPERS = """\
inline float4 wp_quat_mul(float4 a, float4 b) {
    return float4(
        a.w * b.x + b.w * a.x + a.y * b.z - b.y * a.z,
        a.w * b.y + b.w * a.y + a.z * b.x - b.z * a.x,
        a.w * b.z + b.w * a.z + a.x * b.y - b.x * a.y,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}
inline float4 wp_quat_inverse(float4 q) {
    return float4(-q.x, -q.y, -q.z, q.w);
}
inline float3 wp_quat_rotate(float4 q, float3 x) {
    float c = 2.0f * q.w * q.w - 1.0f;
    float d = 2.0f * (q.x * x.x + q.y * x.y + q.z * x.z);
    return float3(
        x.x * c + q.x * d + (q.y * x.z - q.z * x.y) * q.w * 2.0f,
        x.y * c + q.y * d + (q.z * x.x - q.x * x.z) * q.w * 2.0f,
        x.z * c + q.z * d + (q.x * x.y - q.y * x.x) * q.w * 2.0f);
}
inline float3 wp_quat_rotate_inv(float4 q, float3 x) {
    float c = 2.0f * q.w * q.w - 1.0f;
    float d = 2.0f * (q.x * x.x + q.y * x.y + q.z * x.z);
    return float3(
        x.x * c + q.x * d - (q.y * x.z - q.z * x.y) * q.w * 2.0f,
        x.y * c + q.y * d - (q.z * x.x - q.x * x.z) * q.w * 2.0f,
        x.z * c + q.z * d - (q.x * x.y - q.y * x.x) * q.w * 2.0f);
}
inline float4 wp_quat_from_axis_angle(float3 axis, float angle) {
    float half_angle = angle * 0.5f;
    float s = metal::sin(half_angle);
    return float4(axis.x * s, axis.y * s, axis.z * s, metal::cos(half_angle));
}
inline void wp_quat_to_axis_angle(float4 q, thread float3& axis, thread float& angle) {
    float3 v = float3(q.x, q.y, q.z);
    float l = metal::length(v);
    float3 n = (l > 0.0f) ? (v / l) : float3(0.0f);
    axis = (q.w < 0.0f) ? -n : n;
    angle = 2.0f * metal::atan2(l, metal::abs(q.w));
}
inline float4 wp_quat_slerp(float4 q0, float4 q1, float t) {
    float3 axis;
    float angle;
    wp_quat_to_axis_angle(wp_quat_mul(wp_quat_inverse(q0), q1), axis, angle);
    return wp_quat_mul(q0, wp_quat_from_axis_angle(axis, t * angle));
}
inline float3x3 wp_quat_to_matrix(float4 q) {
    return float3x3(
        wp_quat_rotate(q, float3(1.0f, 0.0f, 0.0f)),
        wp_quat_rotate(q, float3(0.0f, 1.0f, 0.0f)),
        wp_quat_rotate(q, float3(0.0f, 0.0f, 1.0f)));
}
inline float4 wp_quat_from_matrix(float3x3 m) {
    float tr = m[0][0] + m[1][1] + m[2][2];
    float x, y, z, w, h = 0.0f;
    if (tr >= 0.0f) {
        h = metal::sqrt(tr + 1.0f);
        w = 0.5f * h;
        h = 0.5f / h;
        x = (m[1][2] - m[2][1]) * h;
        y = (m[2][0] - m[0][2]) * h;
        z = (m[0][1] - m[1][0]) * h;
    } else {
        int max_diag = 0;
        if (m[1][1] > m[0][0]) {
            max_diag = 1;
        }
        if (m[2][2] > m[max_diag][max_diag]) {
            max_diag = 2;
        }
        if (max_diag == 0) {
            h = metal::sqrt((m[0][0] - (m[1][1] + m[2][2])) + 1.0f);
            x = 0.5f * h;
            h = 0.5f / h;
            y = (m[1][0] + m[0][1]) * h;
            z = (m[0][2] + m[2][0]) * h;
            w = (m[1][2] - m[2][1]) * h;
        } else if (max_diag == 1) {
            h = metal::sqrt((m[1][1] - (m[2][2] + m[0][0])) + 1.0f);
            y = 0.5f * h;
            h = 0.5f / h;
            z = (m[2][1] + m[1][2]) * h;
            x = (m[1][0] + m[0][1]) * h;
            w = (m[2][0] - m[0][2]) * h;
        }
        if (max_diag == 2) {
            h = metal::sqrt((m[2][2] - (m[0][0] + m[1][1])) + 1.0f);
            z = 0.5f * h;
            h = 0.5f / h;
            x = (m[0][2] + m[2][0]) * h;
            y = (m[2][1] + m[1][2]) * h;
            w = (m[0][1] - m[1][0]) * h;
        }
    }
    float4 q = float4(x, y, z, w);
    float l = metal::length(q);
    return (l > 0.0f) ? (q / l) : float4(0.0f, 0.0f, 0.0f, 1.0f);
}
inline float4 wp_quat_rpy(float roll, float pitch, float yaw) {
    float cy = metal::cos(yaw * 0.5f);
    float sy = metal::sin(yaw * 0.5f);
    float cr = metal::cos(roll * 0.5f);
    float sr = metal::sin(roll * 0.5f);
    float cp = metal::cos(pitch * 0.5f);
    float sp = metal::sin(pitch * 0.5f);
    return float4(
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
        cy * cr * cp + sy * sr * sp);
}"""


# Rigid-transform helpers. Transforms are ``wp_vec7_float`` after the
# ``vec_t<7>`` normalization, laid out (px, py, pz, qx, qy, qz, qw). Bodies
# mirror ``warp/native/spatial.h``. Depends on ``_QUAT_HELPERS`` and the
# ``wp_vec7_float`` struct — both emitted whenever this block is (see the
# header assembly in ``_emit_header_helpers``). The two-arg
# ``wp_vec7_float_make(float3, float4)`` overload mirrors Warp's
# ``wp.transform(p, q)`` constructor the same way the spatial helpers
# overload ``wp_vec6_float_make(float3, float3)``.
_TRANSFORM_HELPERS = """\
inline wp_vec7_float wp_vec7_float_make(float3 p, float4 q) {
    return wp_vec7_float_make(p[0], p[1], p[2], q[0], q[1], q[2], q[3]);
}
inline float3 wp_transform_get_translation(wp_vec7_float t) {
    return float3(t.c[0], t.c[1], t.c[2]);
}
inline float4 wp_transform_get_rotation(wp_vec7_float t) {
    return float4(t.c[3], t.c[4], t.c[5], t.c[6]);
}
inline wp_vec7_float wp_transform_identity() {
    return wp_vec7_float_make(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
}
inline float3 wp_transform_point(wp_vec7_float t, float3 x) {
    return wp_transform_get_translation(t) + wp_quat_rotate(wp_transform_get_rotation(t), x);
}
inline float3 wp_transform_vector(wp_vec7_float t, float3 x) {
    return wp_quat_rotate(wp_transform_get_rotation(t), x);
}
inline wp_vec7_float wp_transform_multiply(wp_vec7_float a, wp_vec7_float b) {
    float4 aq = wp_transform_get_rotation(a);
    return wp_vec7_float_make(
        wp_quat_rotate(aq, wp_transform_get_translation(b)) + wp_transform_get_translation(a),
        wp_quat_mul(aq, wp_transform_get_rotation(b)));
}
inline wp_vec7_float wp_transform_inverse(wp_vec7_float t) {
    float4 q_inv = wp_quat_inverse(wp_transform_get_rotation(t));
    return wp_vec7_float_make(-wp_quat_rotate(q_inv, wp_transform_get_translation(t)), q_inv);
}"""


# Random-number helpers — a port of ``warp/native/rand.h`` (PCG hash from
# Jarzynski & Olano). All integer ops are uint32 wraparound arithmetic, so
# the streams are bit-identical to the CPU/CUDA backends. State is passed
# ``thread uint&`` to mirror the native ``uint32&`` mutation semantics.
# ``wp_randn`` binds its two ``randf`` draws to explicit temporaries: the
# native code calls ``randf`` twice inside one expression, whose evaluation
# order is unspecified in C++ — Clang on both arm64 and MSL evaluates
# left-to-right today, and the temporaries pin that order permanently.
_RAND_HELPERS = """\
inline uint wp_rand_pcg(uint state) {
    uint b = state * 747796405u + 2891336453u;
    uint c = ((b >> ((b >> 28u) + 4u)) ^ b) * 277803737u;
    return (c >> 22u) ^ c;
}
inline uint wp_rand_init(int seed) { return wp_rand_pcg(uint(seed)); }
inline uint wp_rand_init(int seed, int offset) {
    return wp_rand_pcg(uint(seed) + wp_rand_pcg(uint(offset)));
}
inline int wp_randi(thread uint& state) {
    state = wp_rand_pcg(state);
    return int(state);
}
inline int wp_randi(thread uint& state, int lo, int hi) {
    state = wp_rand_pcg(state);
    return int(state % uint(hi - lo) + uint(lo));
}
inline uint wp_randu(thread uint& state) {
    state = wp_rand_pcg(state);
    return state;
}
inline uint wp_randu(thread uint& state, uint lo, uint hi) {
    state = wp_rand_pcg(state);
    return state % (hi - lo) + lo;
}
inline float wp_randf(thread uint& state) {
    state = wp_rand_pcg(state);
    return (state >> 8) * (1.0f / 16777216.0f);
}
inline float wp_randf(thread uint& state, float lo, float hi) {
    return (hi - lo) * wp_randf(state) + lo;
}
inline float wp_randn(thread uint& state) {
    float u = wp_randf(state);
    float v = wp_randf(state);
    return metal::sqrt(-2.0f * metal::log(u + 5.96e-8f)) *
           metal::cos(2.0f * 3.14159265358979323846f * v);
}"""


# Interpolation / misc math / small linear-algebra helpers. Each piece is
# emitted only when the translated source references it (see
# ``_emit_misc_math_helpers``). Bodies mirror the Warp native
# implementations, NOT the closest MSL builtin, where the two differ:
# ``wp.frac`` truncates toward zero (``metal::fract`` floors — diverges for
# negative inputs) and ``wp.lerp`` computes ``a*(1-t) + b*t`` (``metal::mix``
# computes ``a + (b-a)*t`` — different rounding).
_MISC_MATH_HELPERS: dict[str, str] = {
    "wp_lerp": (
        "template <typename T, typename S>\ninline T wp_lerp(T a, T b, S t) { return a * (S(1) - t) + b * t; }"
    ),
    "wp_smoothstep": (
        "template <typename T>\n"
        "inline T wp_smoothstep(T edge0, T edge1, T x) {\n"
        "    x = metal::clamp((x - edge0) / (edge1 - edge0), T(0), T(1));\n"
        "    return x * x * (T(3) - T(2) * x);\n"
        "}"
    ),
    "wp_frac": ("template <typename T>\ninline T wp_frac(T x) { return x - metal::trunc(x); }"),
    "wp_degrees": ("template <typename T>\ninline T wp_degrees(T x) { return x * T(57.29577951308232087679); }"),
    "wp_radians": ("template <typename T>\ninline T wp_radians(T x) { return x * T(0.01745329251994329577); }"),
    # Outer product: element [i][j] = a[i] * b[j]. MSL matrices are
    # column-major, so column j is ``a * b[j]``.
    "wp_outer": (
        "inline float2x2 wp_outer(float2 a, float2 b) {\n"
        "    return float2x2(a * b.x, a * b.y);\n"
        "}\n"
        "inline float3x3 wp_outer(float3 a, float3 b) {\n"
        "    return float3x3(a * b.x, a * b.y, a * b.z);\n"
        "}\n"
        "inline float4x4 wp_outer(float4 a, float4 b) {\n"
        "    return float4x4(a * b.x, a * b.y, a * b.z, a * b.w);\n"
        "}"
    ),
    # Skew-symmetric cross-product matrix: rows [[0,-z,y],[z,0,-x],[-y,x,0]],
    # written as columns for MSL.
    "wp_skew": (
        "inline float3x3 wp_skew(float3 v) {\n"
        "    return float3x3(\n"
        "        float3(0.0f, v.z, -v.y),\n"
        "        float3(-v.z, 0.0f, v.x),\n"
        "        float3(v.y, -v.x, 0.0f));\n"
        "}"
    ),
    "wp_trace": (
        "inline float wp_trace(float2x2 m) { return m[0][0] + m[1][1]; }\n"
        "inline float wp_trace(float3x3 m) { return m[0][0] + m[1][1] + m[2][2]; }\n"
        "inline float wp_trace(float4x4 m) { return m[0][0] + m[1][1] + m[2][2] + m[3][3]; }"
    ),
    # ``wp.sign(0) == 1`` — do NOT swap in ``metal::sign`` (returns 0 at 0).
    # The uint instantiation's ``x < 0`` is always false, so it returns 1
    # like the native uint overloads do.
    "wp_sign": ("template <typename T>\ninline T wp_sign(T x) { return x < T(0) ? T(-1) : T(1); }"),
    # Warp's step is 1 for x < 0 — the reverse of GLSL/MSL ``step``.
    "wp_step": ("template <typename T>\ninline T wp_step(T x) { return x < T(0) ? T(1) : T(0); }"),
    "wp_nonzero": ("template <typename T>\ninline T wp_nonzero(T x) { return x == T(0) ? T(0) : T(1); }"),
    # ``%`` is integer-only in MSL; float/half dispatch to ``metal::fmod``
    # (truncated remainder, matching Warp's native ``mod``).
    "wp_mod": (
        "template <typename T>\n"
        "inline T wp_mod(T a, T b) { return a % b; }\n"
        "inline float wp_mod(float a, float b) { return metal::fmod(a, b); }\n"
        "inline half wp_mod(half a, half b) { return metal::fmod(a, b); }\n"
        "inline float2 wp_mod(float2 a, float2 b) { return metal::fmod(a, b); }\n"
        "inline float3 wp_mod(float3 a, float3 b) { return metal::fmod(a, b); }\n"
        "inline float4 wp_mod(float4 a, float4 b) { return metal::fmod(a, b); }"
    ),
    # MSL has no cbrt; copysign+pow keeps the sign for negative inputs
    # like C's ``cbrtf`` (plain ``pow`` of a negative base is NaN).
    "wp_cbrt": ("inline float wp_cbrt(float x) { return metal::copysign(metal::pow(metal::abs(x), 1.0f / 3.0f), x); }"),
    # Component-wise multiply / divide. The generic template covers
    # vectors (MSL floatN ``*`` is already component-wise, and our
    # big-vec structs overload ``*`` element-wise); native matrix
    # operands need explicit overloads because MSL's ``matNxN * matNxN``
    # is a matrix multiply. Matrix columns are vectors, so per-column
    # vector arithmetic gives the element-wise result.
    "wp_cw_mul": (
        "template <typename T>\n"
        "inline T wp_cw_mul(T a, T b) { return a * b; }\n"
        "inline float2x2 wp_cw_mul(float2x2 a, float2x2 b) "
        "{ return float2x2(a[0] * b[0], a[1] * b[1]); }\n"
        "inline float3x3 wp_cw_mul(float3x3 a, float3x3 b) "
        "{ return float3x3(a[0] * b[0], a[1] * b[1], a[2] * b[2]); }\n"
        "inline float4x4 wp_cw_mul(float4x4 a, float4x4 b) "
        "{ return float4x4(a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]); }"
    ),
    "wp_cw_div": (
        "template <typename T>\n"
        "inline T wp_cw_div(T a, T b) { return a / b; }\n"
        "inline float2x2 wp_cw_div(float2x2 a, float2x2 b) "
        "{ return float2x2(a[0] / b[0], a[1] / b[1]); }\n"
        "inline float3x3 wp_cw_div(float3x3 a, float3x3 b) "
        "{ return float3x3(a[0] / b[0], a[1] / b[1], a[2] / b[2]); }\n"
        "inline float4x4 wp_cw_div(float4x4 a, float4x4 b) "
        "{ return float4x4(a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]); }"
    ),
    "wp_get_diag": (
        "inline float2 wp_get_diag(float2x2 m) { return float2(m[0][0], m[1][1]); }\n"
        "inline float3 wp_get_diag(float3x3 m) { return float3(m[0][0], m[1][1], m[2][2]); }\n"
        "inline float4 wp_get_diag(float4x4 m) { return float4(m[0][0], m[1][1], m[2][2], m[3][3]); }"
    ),
    # Matrix inverse — port of warp/native/mat.h ``inverse_impl`` with
    # kEps == 0.0f (a singular matrix returns the zero matrix, matching
    # the CPU backend). Native code is row-major ``m.data[r][c]``; MSL is
    # column-major ``m[c][r]`` — the a{r}{c} locals below re-establish
    # the native orientation so the cofactor bodies transcribe verbatim.
    # The 4x4 native version accumulates in double; Apple GPUs have no
    # fp64, so intermediates stay float here (~1e-6 relative drift on
    # well-conditioned inputs).
    "wp_inverse": (
        "inline float2x2 wp_inverse(float2x2 m) {\n"
        "    float det = metal::determinant(m);\n"
        "    if (det == 0.0f) { return float2x2(0.0f); }\n"
        "    float rcp = 1.0f / det;\n"
        "    return float2x2(float2(m[1][1], -m[0][1]) * rcp, float2(-m[1][0], m[0][0]) * rcp);\n"
        "}\n"
        "inline float3x3 wp_inverse(float3x3 m) {\n"
        "    float a00 = m[0][0], a01 = m[1][0], a02 = m[2][0];\n"
        "    float a10 = m[0][1], a11 = m[1][1], a12 = m[2][1];\n"
        "    float a20 = m[0][2], a21 = m[1][2], a22 = m[2][2];\n"
        "    float b00 = a11 * a22 - a12 * a21;\n"
        "    float b10 = a12 * a20 - a10 * a22;\n"
        "    float b20 = a10 * a21 - a11 * a20;\n"
        "    float b01 = a02 * a21 - a01 * a22;\n"
        "    float b11 = a00 * a22 - a02 * a20;\n"
        "    float b21 = a01 * a20 - a00 * a21;\n"
        "    float b02 = a01 * a12 - a02 * a11;\n"
        "    float b12 = a02 * a10 - a00 * a12;\n"
        "    float b22 = a00 * a11 - a01 * a10;\n"
        "    float det = a00 * b00 + a01 * b10 + a02 * b20;\n"
        "    if (det == 0.0f) { return float3x3(0.0f); }\n"
        "    float rcp = 1.0f / det;\n"
        "    return float3x3(\n"
        "        float3(b00, b10, b20) * rcp,\n"
        "        float3(b01, b11, b21) * rcp,\n"
        "        float3(b02, b12, b22) * rcp);\n"
        "}\n"
        "inline float4x4 wp_inverse(float4x4 m) {\n"
        "    float x00 = m[0][0], x01 = m[1][0], x02 = m[2][0], x03 = m[3][0];\n"
        "    float x10 = m[0][1], x11 = m[1][1], x12 = m[2][1], x13 = m[3][1];\n"
        "    float x20 = m[0][2], x21 = m[1][2], x22 = m[2][2], x23 = m[3][2];\n"
        "    float x30 = m[0][3], x31 = m[1][3], x32 = m[2][3], x33 = m[3][3];\n"
        "    float y01 = x00 * x11 - x10 * x01;\n"
        "    float y02 = x00 * x21 - x20 * x01;\n"
        "    float y03 = x00 * x31 - x30 * x01;\n"
        "    float y12 = x10 * x21 - x20 * x11;\n"
        "    float y13 = x10 * x31 - x30 * x11;\n"
        "    float y23 = x20 * x31 - x30 * x21;\n"
        "    float z33 = x02 * y12 - x12 * y02 + x22 * y01;\n"
        "    float z23 = x12 * y03 - x32 * y01 - x02 * y13;\n"
        "    float z13 = x02 * y23 - x22 * y03 + x32 * y02;\n"
        "    float z03 = x22 * y13 - x32 * y12 - x12 * y23;\n"
        "    float z32 = x13 * y02 - x23 * y01 - x03 * y12;\n"
        "    float z22 = x03 * y13 - x13 * y03 + x33 * y01;\n"
        "    float z12 = x23 * y03 - x33 * y02 - x03 * y23;\n"
        "    float z02 = x13 * y23 - x23 * y13 + x33 * y12;\n"
        "    y01 = x02 * x13 - x12 * x03;\n"
        "    y02 = x02 * x23 - x22 * x03;\n"
        "    y03 = x02 * x33 - x32 * x03;\n"
        "    y12 = x12 * x23 - x22 * x13;\n"
        "    y13 = x12 * x33 - x32 * x13;\n"
        "    y23 = x22 * x33 - x32 * x23;\n"
        "    float z30 = x11 * y02 - x21 * y01 - x01 * y12;\n"
        "    float z20 = x01 * y13 - x11 * y03 + x31 * y01;\n"
        "    float z10 = x21 * y03 - x31 * y02 - x01 * y23;\n"
        "    float z00 = x11 * y23 - x21 * y13 + x31 * y12;\n"
        "    float z31 = x00 * y12 - x10 * y02 + x20 * y01;\n"
        "    float z21 = x10 * y03 - x30 * y01 - x00 * y13;\n"
        "    float z11 = x00 * y23 - x20 * y03 + x30 * y02;\n"
        "    float z01 = x20 * y13 - x30 * y12 - x10 * y23;\n"
        "    float det = x30 * z30 + x20 * z20 + x10 * z10 + x00 * z00;\n"
        "    if (det == 0.0f) { return float4x4(0.0f); }\n"
        "    float rcp = 1.0f / det;\n"
        "    return float4x4(\n"
        "        float4(z00, z01, z02, z03) * rcp,\n"
        "        float4(z10, z11, z12, z13) * rcp,\n"
        "        float4(z20, z21, z22, z23) * rcp,\n"
        "        float4(z30, z31, z32, z33) * rcp);\n"
        "}"
    ),
    # Binary search over a sorted 1-D span — port of
    # warp/native/array.h ``lower_bound``. Templated on the pointer type
    # because MLX kernel inputs land in either ``device`` or ``constant``
    # address space (non-deterministic; see module docstring).
    "wp_lower_bound": (
        "template <typename PtrT, typename T>\n"
        "inline int wp_lower_bound(PtrT arr, int arr_begin, int arr_end, T value) {\n"
        "    int lower = arr_begin;\n"
        "    int upper = arr_end - 1;\n"
        "    while (lower < upper) {\n"
        "        int mid = lower + (upper - lower) / 2;\n"
        "        if (arr[mid] < value) { lower = mid + 1; } else { upper = mid; }\n"
        "    }\n"
        "    return lower;\n"
        "}"
    ),
}


def _emit_misc_math_helpers(source: str) -> str:
    """Emit the misc-math helper definitions the source references."""
    parts = [body for name, body in _MISC_MATH_HELPERS.items() if name in source]
    return "\n".join(parts)


# Port of ``warp/native/svd.h`` (McAdams et al. branch-free 3x3 SVD via
# quaternion Jacobi iteration), float instantiation only. The scalar-level
# helpers transcribe verbatim — only the entry points differ: native code
# is row-major ``m.data[r][c]``, MSL is column-major ``m[c][r]``, and the
# wrappers below re-establish the native orientation with a{r}{c} locals.
# ``JACOBI_ITERATIONS = 4`` and the 1e-6 epsilons match the float config.
# ``det_sign`` in svd2 inlines Warp's ``sign`` semantics (sign(0) == 1).
_SVD_HELPERS = """\
inline float wp_svd_rsqrt(float x) { return 1.0f / metal::sqrt(x); }
inline void wp_svd_cond_swap(bool c, thread float& X, thread float& Y) {
    float Z = X;
    X = c ? Y : X;
    Y = c ? Z : Y;
}
inline void wp_svd_cond_neg_swap(bool c, thread float& X, thread float& Y) {
    float Z = -X;
    X = c ? Y : X;
    Y = c ? Z : Y;
}
inline void wp_svd_mult_ab(
    float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33,
    float b11, float b12, float b13, float b21, float b22, float b23, float b31, float b32, float b33,
    thread float& m11, thread float& m12, thread float& m13,
    thread float& m21, thread float& m22, thread float& m23,
    thread float& m31, thread float& m32, thread float& m33) {
    m11 = a11 * b11 + a12 * b21 + a13 * b31;
    m12 = a11 * b12 + a12 * b22 + a13 * b32;
    m13 = a11 * b13 + a12 * b23 + a13 * b33;
    m21 = a21 * b11 + a22 * b21 + a23 * b31;
    m22 = a21 * b12 + a22 * b22 + a23 * b32;
    m23 = a21 * b13 + a22 * b23 + a23 * b33;
    m31 = a31 * b11 + a32 * b21 + a33 * b31;
    m32 = a31 * b12 + a32 * b22 + a33 * b32;
    m33 = a31 * b13 + a32 * b23 + a33 * b33;
}
inline void wp_svd_mult_atb(
    float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33,
    float b11, float b12, float b13, float b21, float b22, float b23, float b31, float b32, float b33,
    thread float& m11, thread float& m12, thread float& m13,
    thread float& m21, thread float& m22, thread float& m23,
    thread float& m31, thread float& m32, thread float& m33) {
    m11 = a11 * b11 + a21 * b21 + a31 * b31;
    m12 = a11 * b12 + a21 * b22 + a31 * b32;
    m13 = a11 * b13 + a21 * b23 + a31 * b33;
    m21 = a12 * b11 + a22 * b21 + a32 * b31;
    m22 = a12 * b12 + a22 * b22 + a32 * b32;
    m23 = a12 * b13 + a22 * b23 + a32 * b33;
    m31 = a13 * b11 + a23 * b21 + a33 * b31;
    m32 = a13 * b12 + a23 * b22 + a33 * b32;
    m33 = a13 * b13 + a23 * b23 + a33 * b33;
}
inline void wp_svd_quat_to_mat3(
    const thread float* qV,
    thread float& m11, thread float& m12, thread float& m13,
    thread float& m21, thread float& m22, thread float& m23,
    thread float& m31, thread float& m32, thread float& m33) {
    float w = qV[3];
    float x = qV[0];
    float y = qV[1];
    float z = qV[2];
    float qxx = x * x;
    float qyy = y * y;
    float qzz = z * z;
    float qxz = x * z;
    float qxy = x * y;
    float qyz = y * z;
    float qwx = w * x;
    float qwy = w * y;
    float qwz = w * z;
    m11 = 1.0f - 2.0f * (qyy + qzz);
    m12 = 2.0f * (qxy - qwz);
    m13 = 2.0f * (qxz + qwy);
    m21 = 2.0f * (qxy + qwz);
    m22 = 1.0f - 2.0f * (qxx + qzz);
    m23 = 2.0f * (qyz - qwx);
    m31 = 2.0f * (qxz - qwy);
    m32 = 2.0f * (qyz + qwx);
    m33 = 1.0f - 2.0f * (qxx + qyy);
}
inline void wp_svd_approx_givens_quat(float a11, float a12, float a22, thread float& ch, thread float& sh) {
    const float _gamma = 5.82842712474619f;
    const float _cstar = 0.9238795325112867f;
    const float _sstar = 0.3826834323650898f;
    ch = 2.0f * (a11 - a22);
    sh = a12;
    bool b = _gamma * sh * sh < ch * ch;
    float w = wp_svd_rsqrt(ch * ch + sh * sh);
    ch = b ? w * ch : _cstar;
    sh = b ? w * sh : _sstar;
}
inline void wp_svd_jacobi_conjugation(
    const int x, const int y, const int z,
    thread float& s11, thread float& s21, thread float& s22,
    thread float& s31, thread float& s32, thread float& s33,
    thread float* qV) {
    float ch, sh;
    wp_svd_approx_givens_quat(s11, s21, s22, ch, sh);
    float scale = ch * ch + sh * sh;
    float a = (ch * ch - sh * sh) / scale;
    float b = (2.0f * sh * ch) / scale;
    float _s11 = s11;
    float _s21 = s21;
    float _s22 = s22;
    float _s31 = s31;
    float _s32 = s32;
    float _s33 = s33;
    s11 = a * (a * _s11 + b * _s21) + b * (a * _s21 + b * _s22);
    s21 = a * (-b * _s11 + a * _s21) + b * (-b * _s21 + a * _s22);
    s22 = -b * (-b * _s11 + a * _s21) + a * (-b * _s21 + a * _s22);
    s31 = a * _s31 + b * _s32;
    s32 = -b * _s31 + a * _s32;
    s33 = _s33;
    float tmp[3];
    tmp[0] = qV[0] * sh;
    tmp[1] = qV[1] * sh;
    tmp[2] = qV[2] * sh;
    sh *= qV[3];
    qV[0] *= ch;
    qV[1] *= ch;
    qV[2] *= ch;
    qV[3] *= ch;
    qV[z] += sh;
    qV[3] -= tmp[z];
    qV[x] += tmp[y];
    qV[y] -= tmp[x];
    _s11 = s22;
    _s21 = s32;
    _s22 = s33;
    _s31 = s21;
    _s32 = s31;
    _s33 = s11;
    s11 = _s11;
    s21 = _s21;
    s22 = _s22;
    s31 = _s31;
    s32 = _s32;
    s33 = _s33;
}
inline float wp_svd_dist2(float x, float y, float z) { return x * x + y * y + z * z; }
inline void wp_svd_jacobi_eigenanalysis(
    thread float& s11, thread float& s21, thread float& s22,
    thread float& s31, thread float& s32, thread float& s33,
    thread float* qV) {
    qV[3] = 1.0f;
    qV[0] = 0.0f;
    qV[1] = 0.0f;
    qV[2] = 0.0f;
    for (int i = 0; i < 4; i++) {
        wp_svd_jacobi_conjugation(0, 1, 2, s11, s21, s22, s31, s32, s33, qV);
        wp_svd_jacobi_conjugation(1, 2, 0, s11, s21, s22, s31, s32, s33, qV);
        wp_svd_jacobi_conjugation(2, 0, 1, s11, s21, s22, s31, s32, s33, qV);
    }
}
inline void wp_svd_sort_singular_values(
    thread float& b11, thread float& b12, thread float& b13,
    thread float& b21, thread float& b22, thread float& b23,
    thread float& b31, thread float& b32, thread float& b33,
    thread float& v11, thread float& v12, thread float& v13,
    thread float& v21, thread float& v22, thread float& v23,
    thread float& v31, thread float& v32, thread float& v33) {
    float rho1 = wp_svd_dist2(b11, b21, b31);
    float rho2 = wp_svd_dist2(b12, b22, b32);
    float rho3 = wp_svd_dist2(b13, b23, b33);
    bool c;
    c = rho1 < rho2;
    wp_svd_cond_neg_swap(c, b11, b12);
    wp_svd_cond_neg_swap(c, v11, v12);
    wp_svd_cond_neg_swap(c, b21, b22);
    wp_svd_cond_neg_swap(c, v21, v22);
    wp_svd_cond_neg_swap(c, b31, b32);
    wp_svd_cond_neg_swap(c, v31, v32);
    wp_svd_cond_swap(c, rho1, rho2);
    c = rho1 < rho3;
    wp_svd_cond_neg_swap(c, b11, b13);
    wp_svd_cond_neg_swap(c, v11, v13);
    wp_svd_cond_neg_swap(c, b21, b23);
    wp_svd_cond_neg_swap(c, v21, v23);
    wp_svd_cond_neg_swap(c, b31, b33);
    wp_svd_cond_neg_swap(c, v31, v33);
    wp_svd_cond_swap(c, rho1, rho3);
    c = rho2 < rho3;
    wp_svd_cond_neg_swap(c, b12, b13);
    wp_svd_cond_neg_swap(c, v12, v13);
    wp_svd_cond_neg_swap(c, b22, b23);
    wp_svd_cond_neg_swap(c, v22, v23);
    wp_svd_cond_neg_swap(c, b32, b33);
    wp_svd_cond_neg_swap(c, v32, v33);
}
inline void wp_svd_qr_givens_quat(float a1, float a2, thread float& ch, thread float& sh) {
    const float epsilon = 1.0e-6f;
    float rho = metal::sqrt(a1 * a1 + a2 * a2);
    sh = rho > epsilon ? a2 : 0.0f;
    ch = metal::abs(a1) + metal::max(rho, epsilon);
    bool b = a1 < 0.0f;
    wp_svd_cond_swap(b, sh, ch);
    float w = wp_svd_rsqrt(ch * ch + sh * sh);
    ch *= w;
    sh *= w;
}
inline void wp_svd_qr_decomposition(
    float b11, float b12, float b13, float b21, float b22, float b23, float b31, float b32, float b33,
    thread float& q11, thread float& q12, thread float& q13,
    thread float& q21, thread float& q22, thread float& q23,
    thread float& q31, thread float& q32, thread float& q33,
    thread float& r11, thread float& r12, thread float& r13,
    thread float& r21, thread float& r22, thread float& r23,
    thread float& r31, thread float& r32, thread float& r33) {
    float ch1, sh1, ch2, sh2, ch3, sh3;
    float a, b;
    wp_svd_qr_givens_quat(b11, b21, ch1, sh1);
    a = 1.0f - 2.0f * sh1 * sh1;
    b = 2.0f * ch1 * sh1;
    r11 = a * b11 + b * b21;
    r12 = a * b12 + b * b22;
    r13 = a * b13 + b * b23;
    r21 = -b * b11 + a * b21;
    r22 = -b * b12 + a * b22;
    r23 = -b * b13 + a * b23;
    r31 = b31;
    r32 = b32;
    r33 = b33;
    wp_svd_qr_givens_quat(r11, r31, ch2, sh2);
    a = 1.0f - 2.0f * sh2 * sh2;
    b = 2.0f * ch2 * sh2;
    b11 = a * r11 + b * r31;
    b12 = a * r12 + b * r32;
    b13 = a * r13 + b * r33;
    b21 = r21;
    b22 = r22;
    b23 = r23;
    b31 = -b * r11 + a * r31;
    b32 = -b * r12 + a * r32;
    b33 = -b * r13 + a * r33;
    wp_svd_qr_givens_quat(b22, b32, ch3, sh3);
    a = 1.0f - 2.0f * sh3 * sh3;
    b = 2.0f * ch3 * sh3;
    r11 = b11;
    r12 = b12;
    r13 = b13;
    r21 = a * b21 + b * b31;
    r22 = a * b22 + b * b32;
    r23 = a * b23 + b * b33;
    r31 = -b * b21 + a * b31;
    r32 = -b * b22 + a * b32;
    r33 = -b * b23 + a * b33;
    float sh12 = sh1 * sh1;
    float sh22 = sh2 * sh2;
    float sh32 = sh3 * sh3;
    q11 = (-1.0f + 2.0f * sh12) * (-1.0f + 2.0f * sh22);
    q12 = 4.0f * ch2 * ch3 * (-1.0f + 2.0f * sh12) * sh2 * sh3
        + 2.0f * ch1 * sh1 * (-1.0f + 2.0f * sh32);
    q13 = 4.0f * ch1 * ch3 * sh1 * sh3
        - 2.0f * ch2 * (-1.0f + 2.0f * sh12) * sh2 * (-1.0f + 2.0f * sh32);
    q21 = 2.0f * ch1 * sh1 * (1.0f - 2.0f * sh22);
    q22 = -8.0f * ch1 * ch2 * ch3 * sh1 * sh2 * sh3 + (-1.0f + 2.0f * sh12) * (-1.0f + 2.0f * sh32);
    q23 = -2.0f * ch3 * sh3 + 4.0f * sh1 * (ch3 * sh1 * sh3 + ch1 * ch2 * sh2 * (-1.0f + 2.0f * sh32));
    q31 = 2.0f * ch2 * sh2;
    q32 = 2.0f * ch3 * (1.0f - 2.0f * sh22) * sh3;
    q33 = (-1.0f + 2.0f * sh22) * (-1.0f + 2.0f * sh32);
}
inline void wp_svd3_core(
    float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33,
    thread float& u11, thread float& u12, thread float& u13,
    thread float& u21, thread float& u22, thread float& u23,
    thread float& u31, thread float& u32, thread float& u33,
    thread float& s11, thread float& s12, thread float& s13,
    thread float& s21, thread float& s22, thread float& s23,
    thread float& s31, thread float& s32, thread float& s33,
    thread float& v11, thread float& v12, thread float& v13,
    thread float& v21, thread float& v22, thread float& v23,
    thread float& v31, thread float& v32, thread float& v33) {
    float ATA11, ATA12, ATA13;
    float ATA21, ATA22, ATA23;
    float ATA31, ATA32, ATA33;
    wp_svd_mult_atb(
        a11, a12, a13, a21, a22, a23, a31, a32, a33, a11, a12, a13, a21, a22, a23, a31, a32, a33,
        ATA11, ATA12, ATA13, ATA21, ATA22, ATA23, ATA31, ATA32, ATA33);
    float qV[4];
    wp_svd_jacobi_eigenanalysis(ATA11, ATA21, ATA22, ATA31, ATA32, ATA33, qV);
    wp_svd_quat_to_mat3(qV, v11, v12, v13, v21, v22, v23, v31, v32, v33);
    float b11, b12, b13;
    float b21, b22, b23;
    float b31, b32, b33;
    wp_svd_mult_ab(
        a11, a12, a13, a21, a22, a23, a31, a32, a33, v11, v12, v13, v21, v22, v23, v31, v32, v33,
        b11, b12, b13, b21, b22, b23, b31, b32, b33);
    wp_svd_sort_singular_values(
        b11, b12, b13, b21, b22, b23, b31, b32, b33, v11, v12, v13, v21, v22, v23, v31, v32, v33);
    wp_svd_qr_decomposition(
        b11, b12, b13, b21, b22, b23, b31, b32, b33, u11, u12, u13, u21, u22, u23, u31, u32, u33,
        s11, s12, s13, s21, s22, s23, s31, s32, s33);
}
inline void wp_svd3(float3x3 A, thread float3x3& U, thread float3& sigma, thread float3x3& V) {
    float a11 = A[0][0], a12 = A[1][0], a13 = A[2][0];
    float a21 = A[0][1], a22 = A[1][1], a23 = A[2][1];
    float a31 = A[0][2], a32 = A[1][2], a33 = A[2][2];
    float u11, u12, u13, u21, u22, u23, u31, u32, u33;
    float s11, s12, s13, s21, s22, s23, s31, s32, s33;
    float v11, v12, v13, v21, v22, v23, v31, v32, v33;
    wp_svd3_core(
        a11, a12, a13, a21, a22, a23, a31, a32, a33,
        u11, u12, u13, u21, u22, u23, u31, u32, u33,
        s11, s12, s13, s21, s22, s23, s31, s32, s33,
        v11, v12, v13, v21, v22, v23, v31, v32, v33);
    U = float3x3(float3(u11, u21, u31), float3(u12, u22, u32), float3(u13, u23, u33));
    sigma = float3(s11, s22, s33);
    V = float3x3(float3(v11, v21, v31), float3(v12, v22, v32), float3(v13, v23, v33));
}
inline void wp_qr3(float3x3 A, thread float3x3& Q, thread float3x3& R) {
    float q11, q12, q13, q21, q22, q23, q31, q32, q33;
    float r11, r12, r13, r21, r22, r23, r31, r32, r33;
    wp_svd_qr_decomposition(
        A[0][0], A[1][0], A[2][0], A[0][1], A[1][1], A[2][1], A[0][2], A[1][2], A[2][2],
        q11, q12, q13, q21, q22, q23, q31, q32, q33,
        r11, r12, r13, r21, r22, r23, r31, r32, r33);
    Q = float3x3(float3(q11, q21, q31), float3(q12, q22, q32), float3(q13, q23, q33));
    R = float3x3(float3(r11, r21, r31), float3(r12, r22, r32), float3(r13, r23, r33));
}
inline void wp_eig3(float3x3 A, thread float3x3& Q, thread float3& d) {
    float qV[4];
    float s11 = A[0][0];
    float s21 = A[0][1];
    float s22 = A[1][1];
    float s31 = A[0][2];
    float s32 = A[1][2];
    float s33 = A[2][2];
    float q11, q12, q13, q21, q22, q23, q31, q32, q33;
    wp_svd_jacobi_eigenanalysis(s11, s21, s22, s31, s32, s33, qV);
    wp_svd_quat_to_mat3(qV, q11, q12, q13, q21, q22, q23, q31, q32, q33);
    float t11, t12, t13, t21, t22, t23, t31, t32, t33;
    wp_svd_mult_atb(
        q11, q12, q13, q21, q22, q23, q31, q32, q33,
        A[0][0], A[1][0], A[2][0], A[0][1], A[1][1], A[2][1], A[0][2], A[1][2], A[2][2],
        t11, t12, t13, t21, t22, t23, t31, t32, t33);
    float u11, u12, u13, u21, u22, u23, u31, u32, u33;
    wp_svd_mult_ab(
        t11, t12, t13, t21, t22, t23, t31, t32, t33,
        q11, q12, q13, q21, q22, q23, q31, q32, q33,
        u11, u12, u13, u21, u22, u23, u31, u32, u33);
    Q = float3x3(float3(q11, q21, q31), float3(q12, q22, q32), float3(q13, q23, q33));
    d = float3(u11, u22, u33);
}
inline void wp_svd2(float2x2 A, thread float2x2& U, thread float2& sigma, thread float2x2& V) {
    float a11 = A[0][0], a12 = A[1][0];
    float a21 = A[0][1], a22 = A[1][1];
    float u11, u12, u21, u22, s1, s2, v11, v12, v21, v22;
    float ATA11 = a11 * a11 + a21 * a21;
    float ATA12 = a11 * a12 + a21 * a22;
    float ATA22 = a12 * a12 + a22 * a22;
    float trace = ATA11 + ATA22;
    float diff = ATA11 - ATA22;
    float discriminant = diff * diff + 4.0f * ATA12 * ATA12;
    if (discriminant == 0.0f) {
        s1 = s2 = metal::sqrt(0.5f * trace);
        u11 = v11 = 1.0f;
        u12 = v12 = 0.0f;
        u21 = v21 = 0.0f;
        u22 = v22 = 1.0f;
    } else {
        float sqrt_term = metal::sqrt(discriminant);
        float lambda1 = (trace + sqrt_term) * 0.5f;
        float lambda2 = (trace - sqrt_term) * 0.5f;
        float inv_sigma1 = wp_svd_rsqrt(lambda1);
        float sigma1 = 1.0f / inv_sigma1;
        float sigma2 = metal::sqrt(lambda2);
        float v1y = diff - sqrt_term + 2.0f * ATA12, v1x = diff + sqrt_term - 2.0f * ATA12;
        float len1_sq = v1x * v1x + v1y * v1y;
        if (len1_sq == 0.0f) {
            v11 = 0.707106781186547524401f;
            v21 = v11;
        } else {
            float inv_len1 = wp_svd_rsqrt(len1_sq);
            v11 = v1x * inv_len1;
            v21 = v1y * inv_len1;
        }
        v12 = -v21;
        v22 = v11;
        u11 = (a11 * v11 + a12 * v21) * inv_sigma1;
        u21 = (a21 * v11 + a22 * v21) * inv_sigma1;
        float det_sign = (a11 * a22 - a12 * a21) < 0.0f ? -1.0f : 1.0f;
        u12 = -u21 * det_sign;
        u22 = u11 * det_sign;
        s1 = sigma1;
        s2 = sigma2;
    }
    U = float2x2(float2(u11, u21), float2(u12, u22));
    sigma = float2(s1, s2);
    V = float2x2(float2(v11, v21), float2(v12, v22));
}"""


# ---------------------------------------------------------------------------
# Tile primitive emission (Phase 2 — Cholesky path for mujoco_warp)
# ---------------------------------------------------------------------------
# Warp's tile builtins (``wp.tile_load``, ``wp.tile_store``, ``wp.tile_cholesky``,
# ``wp.tile_cholesky_solve``) target CUDA-LTO via cuBLASDx on the CUDA backend.
# Metal has no equivalent runtime, so we hand-roll small-tile MSL helpers per
# (rows, cols, scalar). The freejoint case uses 6×6 tiles (mass matrix factor)
# and 6×1 (RHS); larger tile sizes are emitted on demand if the kernel uses
# them. Single-thread / private memory only — see metal-shader-expert
# guidance: for N ≤ 32 the synchronization cost of cooperative tiles
# dominates the work on Apple Silicon.

# Match struct *type* references only — anchored on a non-word, non-``_``
# follow so we don't pick up the helper-function suffixes ``_load``,
# ``_store``, ``_cholesky``, ``_cholesky_solve_K``.
_TILE_SCALAR_ALT = "|".join(_MSL_PREFIX_TO_SAME)
_TILE_NAME_PAT = re.compile(rf"\bwp_tile_(\d+)x(\d+)_({_TILE_SCALAR_ALT})(?![\w])")


def _emit_tile_struct(rows: int, cols: int, msl_scalar: str) -> str:
    """Emit the MSL declaration for a tile struct + load/store helpers.

    Storage is row-major flat ``c[rows*cols]``. The struct is meant to live
    in *private* (per-thread) memory — for the small N we currently handle
    the compiler keeps it in registers entirely.
    """
    name = f"wp_tile_{rows}x{cols}_{msl_scalar}"
    n = rows * cols
    parts: list[str] = []
    parts.append(f"struct {name} {{")
    parts.append(f"    {msl_scalar} c[{n}];")
    parts.append("};")
    # Load: read an ``rows × cols`` sub-block from a row-major array. MLX
    # selects the address space (``device`` vs ``constant``) per kernel
    # argument based on size — small read-only buffers (e.g. a 6-element
    # RHS vector) land in ``constant``. We template the load helper on
    # the pointer type so the same call site works for either: the MSL
    # compiler instantiates one body per (address-space) flavor and
    # deduplicates at link time.
    parts.append("template <typename T>")
    parts.append(f"inline {name} {name}_load(T arr, int base, int row_stride, int row_off, int col_off) {{")
    parts.append(f"    {name} t;")
    # Runtime loops (rather than fully unrolled writes) keep the helper
    # small. Apple's MSL compiler silently drops writes when a kernel
    # accumulates 3+ inlined fully-unrolled tile load/store/cholesky
    # bodies (observed at N=16 in mujoco_warp's blocked-Cholesky factor:
    # the third diagonal block's stored values came back as zero with
    # no compile error). The compact form fits within whatever
    # threshold the optimizer respects.
    parts.append(f"    for (int i = 0; i < {rows}; ++i) {{")
    parts.append(f"        for (int j = 0; j < {cols}; ++j) {{")
    parts.append(f"            t.c[i * {cols} + j] = arr[base + (row_off + i) * row_stride + (col_off + j)];")
    parts.append("        }")
    parts.append("    }")
    parts.append("    return t;")
    parts.append("}")
    # Store target is always writable, so it stays ``device``.
    parts.append(
        f"inline void {name}_store(device {msl_scalar}* arr, "
        f"int base, int row_stride, int row_off, int col_off, {name} t) {{"
    )
    parts.append(f"    for (int i = 0; i < {rows}; ++i) {{")
    parts.append(f"        for (int j = 0; j < {cols}; ++j) {{")
    parts.append(f"            arr[base + (row_off + i) * row_stride + (col_off + j)] = t.c[i * {cols} + j];")
    parts.append("        }")
    parts.append("    }")
    parts.append("}")
    return "\n".join(parts)


def _emit_tile_struct_vec(rows: int, cols: int, n_elem: int, msl_scalar: str) -> str:
    """Emit ``wp_tile_RxC_vec<N>_<scalar>`` — a tile of ``vec_t<N, scalar>``.

    Storage is flat ``c[rows*cols*n_elem]`` in element-major,
    component-minor layout: ``c[(i*cols + j)*N + k]`` is the k-th
    component of the (i, j) element. Used by the dense-Jacobian path
    where tiles of ``vec3`` / ``spatial_vector`` (``vec6``) flow
    through ``tile_load`` / ``tile_map`` / ``tile_store``.

    Element accessors emit as inline lambdas to keep the call sites
    typed (the compiler folds them at -O2).
    """
    name = f"wp_tile_{rows}x{cols}_vec{n_elem}_{msl_scalar}"
    flat_n = rows * cols * n_elem
    parts: list[str] = []
    parts.append(f"struct {name} {{")
    parts.append(f"    {msl_scalar} c[{flat_n}];")
    parts.append("};")
    # Load: read ``rows × cols`` elements from a device array of
    # ``vec_t<n_elem, scalar>``, exposed by MLX as a flat scalar buffer
    # of shape ``(*array_shape, n_elem)``. ``row_stride`` is the inner-
    # *element* stride (inner scalar count = n_elem * array.shape[-1]
    # already folded by the caller via ``__shapes_packed``).
    parts.append("template <typename T>")
    parts.append(f"inline {name} {name}_load(T arr, int base, int row_stride, int row_off, int col_off) {{")
    parts.append(f"    {name} t;")
    parts.append(f"    for (int i = 0; i < {rows}; ++i) {{")
    parts.append(f"        for (int j = 0; j < {cols}; ++j) {{")
    parts.append(f"            int elem_off = ((row_off + i) * row_stride + (col_off + j)) * {n_elem};")
    parts.append(f"            for (int k = 0; k < {n_elem}; ++k) {{")
    parts.append(f"                t.c[(i * {cols} + j) * {n_elem} + k] = arr[base + elem_off + k];")
    parts.append("            }")
    parts.append("        }")
    parts.append("    }")
    parts.append("    return t;")
    parts.append("}")
    parts.append(
        f"inline void {name}_store(device {msl_scalar}* arr, "
        f"int base, int row_stride, int row_off, int col_off, {name} t) {{"
    )
    parts.append(f"    for (int i = 0; i < {rows}; ++i) {{")
    parts.append(f"        for (int j = 0; j < {cols}; ++j) {{")
    parts.append(f"            int elem_off = ((row_off + i) * row_stride + (col_off + j)) * {n_elem};")
    parts.append(f"            for (int k = 0; k < {n_elem}; ++k) {{")
    parts.append(f"                arr[base + elem_off + k] = t.c[(i * {cols} + j) * {n_elem} + k];")
    parts.append("            }")
    parts.append("        }")
    parts.append("    }")
    parts.append("}")
    return "\n".join(parts)


def _emit_tile_cholesky(n: int, msl_scalar: str) -> str:
    """Emit ``wp_tile_NxN_<scalar>_cholesky`` — in-place lower Cholesky.

    Crout's algorithm, fully unrolled for compile-time ``n``. ``precise::sqrt``
    is used because MSL defaults to ``-ffast-math`` which would otherwise
    reassociate accumulations and erode stability. ``max(diag, 1e-30)``
    guards against denormal-flush turning a tiny pivot into hard zero.
    """
    name = f"wp_tile_{n}x{n}_{msl_scalar}"
    parts: list[str] = []
    parts.append(f"inline {name} {name}_cholesky({name} A) {{")
    parts.append(f"    {name} L = A;")
    # Disable unrolling on the outer ``j`` loop. With N>=8 and the natural
    # full unroll of the triple-nested loops, Apple's MSL compiler
    # silently elides the writes ``L.c[(N-1)*N + 0..N-3]`` — only the
    # last two entries of the bottom row come back correct (e.g. for N=8
    # rows 7,col 0..5 = 0; row 7 col 6, 7 are right). Holding the j loop
    # at runtime keeps the writes aligned with the algorithm.
    parts.append("    #pragma clang loop unroll(disable)")
    parts.append(f"    for (int j = 0; j < {n}; ++j) {{")
    parts.append(f"        {msl_scalar} d = L.c[j*{n} + j];")
    parts.append("        for (int k = 0; k < j; ++k) {")
    parts.append(f"            {msl_scalar} ljk = L.c[j*{n} + k];")
    # Explicit ``fma(-a, a, d)`` instead of ``d -= a * a``: forces the
    # mul+sub to use the fused-multiply-add path with one rounding
    # instead of two. With Apple-default ``-ffast-math`` the compiler
    # is allowed to re-associate the sequence, which on G1's H matrix
    # (freejoint root + tightly-coupled hinge dofs) accumulates enough
    # round-off to drive the diagonal pivot below the
    # ``max(d, 1e-30)`` clamp, then ``inv = 1/sqrt(1e-30) ≈ 1e15``
    # blows up the rest of the factorization to NaN.
    parts.append("            d = metal::fma(-ljk, ljk, d);")
    parts.append("        }")
    parts.append(f"        d = metal::max(d, ({msl_scalar})1e-30);")
    parts.append(f"        {msl_scalar} ljj = metal::precise::sqrt(d);")
    parts.append(f"        L.c[j*{n} + j] = ljj;")
    parts.append(f"        {msl_scalar} inv = ({msl_scalar})1.0 / ljj;")
    parts.append(f"        for (int i = j + 1; i < {n}; ++i) {{")
    parts.append(f"            {msl_scalar} s = L.c[i*{n} + j];")
    parts.append(f"            for (int k = 0; k < j; ++k) s = metal::fma(-L.c[i*{n} + k], L.c[j*{n} + k], s);")
    parts.append(f"            L.c[i*{n} + j] = s * inv;")
    parts.append(f"            L.c[j*{n} + i] = ({msl_scalar})0.0;")
    parts.append("        }")
    parts.append("    }")
    parts.append("    return L;")
    parts.append("}")
    return "\n".join(parts)


# Cooperative-tile threshold. At N below this the SIMD-cooperative
# variant's barrier overhead dominates the single-thread path — the
# right-looking algorithm has a serial pivot dependency that limits
# parallelism. Empirically (M3, ms steady-state):
#   N=8:   0.6  vs 0.3   (2.0x)  — barely worthwhile
#   N=16:  0.9  vs 0.7   (1.3x)  — barely worthwhile
#   N=24:  ~1.2 vs 0.4   (3.0x)
#   N=32:  1.6  vs 0.5   (3.2x)
#   N=64:  7.6  vs 0.8   (9.5x)
# Below the threshold we keep the single-thread emit (smaller TGM
# footprint, simpler launch, identical numerical drift to CPU).
_COOP_CHOL_MIN_N = 24

# Apple Silicon caps threadgroup memory at 32 KB. ``smem[N*N]`` of float
# costs ``4*N^2`` bytes — at N=90 that's 32400 B. Above the cap the
# kernel fails to load. The blocked-Cholesky path covers larger N.
_COOP_CHOL_MAX_N = 88


# --- Cooperative load/store: kept for future use --------------------
# Tried in commit 02bfba9e and 38f74e5d. At the sizes we currently
# ship (N ≤ 88, tile fits in L1 cache), ``_emit_tile_load_coop``'s
# write-smem → barrier → read-back-to-private round-trip cost more
# than the device-memory savings — Apple Silicon's unified memory +
# L1 absorbs 32 redundant identical reads cheaply. The cooperative
# store version did not measurably reduce latency either, because
# Metal's SIMD-group write coalescing collapses 32 redundant
# identical writes into one transaction.
#
# Keeping the emitters as callable functions (not invoked anywhere
# in the current pipeline) so they're easy to re-enable when:
#   - tile sizes grow past L1 (likely once we add cooperative
#     contact-jacobian or larger Cholesky factors above N=88), or
#   - we add multi-SIMD-group cooperation where smem traffic is
#     unavoidable for the algorithm anyway.
# Re-enabling requires routing through ``repl_load`` /
# ``repl_store`` (see commit 02bfba9e for the call-site rewrite).


def _emit_tile_load_coop(rows: int, cols: int, msl_scalar: str) -> str:
    """Emit ``wp_tile_RxC_<scalar>_load_coop`` — 32-lane cooperative
    load through threadgroup memory.

    Each lane reads ``ceil(R*C / 32)`` elements directly from the
    device array into a shared threadgroup buffer; after the barrier
    every lane reads the full ``R*C`` block back into its private
    struct. Bandwidth into device memory drops from ``32 * R * C``
    (single-thread emit) to ``R * C`` per call.

    Currently unused — see header comment at the top of the
    cooperative section.
    """
    name = f"wp_tile_{rows}x{cols}_{msl_scalar}"
    n = rows * cols
    parts: list[str] = []
    parts.append("template <typename T>")
    parts.append(
        f"inline {name} {name}_load_coop(T arr, "
        f"int base, int row_stride, int row_off, int col_off, "
        f"threadgroup {msl_scalar}* smem, uint lane) {{"
    )
    parts.append("    threadgroup_barrier(metal::mem_flags::mem_threadgroup);")
    parts.append(f"    for (uint idx = lane; idx < {n}u; idx += 32u) {{")
    parts.append(f"        int i = (int)(idx / {cols}u);")
    parts.append(f"        int j = (int)(idx % {cols}u);")
    parts.append("        smem[idx] = arr[base + (row_off + i) * row_stride + (col_off + j)];")
    parts.append("    }")
    parts.append("    threadgroup_barrier(metal::mem_flags::mem_threadgroup);")
    parts.append(f"    {name} t;")
    parts.append(f"    for (int idx = 0; idx < {n}; ++idx) t.c[idx] = smem[idx];")
    parts.append("    return t;")
    parts.append("}")
    return "\n".join(parts)


def _emit_tile_store_coop(rows: int, cols: int, msl_scalar: str) -> str:
    """Emit ``wp_tile_RxC_<scalar>_store_coop`` — 32-lane cooperative
    store.

    Each lane writes its strided slice of the tile to device memory.
    All 32 lanes hold an identical private copy of ``t`` (every
    cooperative tile op leaves each lane with the full tile), so
    writing slice-by-slice with no synchronisation produces the same
    result as the single-thread emit at 1/32 the device-memory
    bandwidth — *if* the device's write coalescer can't fold
    redundant identical writes itself.

    Currently unused — see header comment at the top of the
    cooperative section.
    """
    name = f"wp_tile_{rows}x{cols}_{msl_scalar}"
    n = rows * cols
    parts: list[str] = []
    parts.append(
        f"inline void {name}_store_coop(device {msl_scalar}* arr, "
        f"int base, int row_stride, int row_off, int col_off, {name} t, uint lane) {{"
    )
    parts.append(f"    for (uint idx = lane; idx < {n}u; idx += 32u) {{")
    parts.append(f"        int i = (int)(idx / {cols}u);")
    parts.append(f"        int j = (int)(idx % {cols}u);")
    parts.append("        arr[base + (row_off + i) * row_stride + (col_off + j)] = t.c[idx];")
    parts.append("    }")
    parts.append("}")
    return "\n".join(parts)


def _emit_tile_cholesky_coop(n: int, msl_scalar: str) -> str:
    """Emit ``wp_tile_NxN_<scalar>_cholesky_coop`` — SIMD-cooperative
    right-looking Cholesky.

    Called by all 32 threads of the SIMD group. ``smem`` is a
    threadgroup-memory buffer of ``N*N`` scalars provided by the
    kernel scope. ``lane`` is ``thread_position_in_threadgroup.x``.

    On iteration ``k`` lane ``(k mod 32)`` computes the diagonal pivot
    while every other lane idles at the barrier. After the broadcast,
    all lanes update their strided rows in parallel. The trailing
    barrier-then-private-readback step copies the full result into
    each thread's private struct, paying a redundant N² read so the
    rest of the kernel body (still single-thread) sees a complete
    private tile.
    """
    name = f"wp_tile_{n}x{n}_{msl_scalar}"
    parts: list[str] = []
    parts.append(f"inline {name} {name}_cholesky_coop({name} A, threadgroup {msl_scalar}* smem, uint lane) {{")
    # Start barrier: a previous cooperative op may still be reading smem.
    parts.append("    threadgroup_barrier(metal::mem_flags::mem_threadgroup);")
    # 1. Cooperative load A → smem (each lane writes its strided slice).
    parts.append(f"    for (uint idx = lane; idx < {n * n}u; idx += 32u) {{")
    parts.append("        smem[idx] = A.c[idx];")
    parts.append("    }")
    parts.append("    threadgroup_barrier(metal::mem_flags::mem_threadgroup);")
    # 2. Right-looking Cholesky. Outer ``j`` loop pivots; inner ``i``
    #    distributes column-k updates across lanes.
    parts.append(f"    for (int j = 0; j < {n}; ++j) {{")
    parts.append("        if ((int)lane == (j & 31)) {")
    parts.append(f"            {msl_scalar} d = smem[j*{n} + j];")
    parts.append("            for (int k = 0; k < j; ++k) {")
    parts.append(f"                {msl_scalar} ljk = smem[j*{n} + k];")
    # Explicit fma — see ``_emit_tile_cholesky_inplace`` for rationale.
    parts.append("                d = metal::fma(-ljk, ljk, d);")
    parts.append("            }")
    parts.append(f"            d = metal::max(d, ({msl_scalar})1e-30);")
    parts.append(f"            smem[j*{n} + j] = metal::precise::sqrt(d);")
    parts.append("        }")
    parts.append("        threadgroup_barrier(metal::mem_flags::mem_threadgroup);")
    parts.append(f"        {msl_scalar} pivot = smem[j*{n} + j];")
    parts.append(f"        for (int i = (int)lane; i < {n}; i += 32) {{")
    parts.append("            if (i > j) {")
    parts.append(f"                {msl_scalar} s = smem[i*{n} + j];")
    parts.append(f"                for (int k = 0; k < j; ++k) s = metal::fma(-smem[i*{n} + k], smem[j*{n} + k], s);")
    parts.append(f"                smem[i*{n} + j] = s / pivot;")
    parts.append("            }")
    parts.append("        }")
    parts.append("        threadgroup_barrier(metal::mem_flags::mem_threadgroup);")
    parts.append("    }")
    # 3. Each thread reads the full result into its private L. Upper
    #    triangle gets zeroed in this same pass so the rest of the
    #    kernel sees a clean lower-triangular factor.
    parts.append(f"    {name} L;")
    parts.append(f"    for (int idx = 0; idx < {n * n}; ++idx) {{")
    parts.append(f"        int i = idx / {n};")
    parts.append(f"        int j2 = idx - i * {n};")
    parts.append(f"        L.c[idx] = (j2 > i) ? ({msl_scalar})0 : smem[idx];")
    parts.append("    }")
    parts.append("    return L;")
    parts.append("}")
    return "\n".join(parts)


def _emit_tile_cholesky_inplace(n: int, msl_scalar: str) -> str:
    """Emit ``wp_tile_NxN_<scalar>_cholesky_inplace`` — same body as
    :func:`_emit_tile_cholesky` but mutates the input tile in place
    (no ``L = A`` copy, no return value).

    Marked ``__attribute__((noinline))`` to dodge a Metal compiler
    misoptimization: when this helper is inlined 3+ times into the
    same kernel, the FIRST inlined copy's writes to ``L.c[...]``
    silently come back as zero (no compile error, deterministic).
    Reproducer: three sequential ``tile_cholesky_inplace`` calls on
    different tile locals at N=16 — the first tile's diagonal ends
    up ~0.5 instead of ~16.6, captured even by an immediate
    ``tile_store`` after the first call. Forcing the helper out of
    line restores correctness; the cost is one function-call
    boundary per invocation, which is dwarfed by the Cholesky's
    own work.
    """
    name = f"wp_tile_{n}x{n}_{msl_scalar}"
    parts: list[str] = [f"__attribute__((noinline)) void {name}_cholesky_inplace(thread {name}& L) {{"]
    parts.append("    #pragma clang loop unroll(disable)")
    parts.append(f"    for (int j = 0; j < {n}; ++j) {{")
    parts.append(f"        {msl_scalar} d = L.c[j*{n} + j];")
    parts.append("        for (int k = 0; k < j; ++k) {")
    parts.append(f"            {msl_scalar} ljk = L.c[j*{n} + k];")
    # Explicit ``fma(-a, a, d)`` instead of ``d -= a * a``: forces the
    # mul+sub to use the fused-multiply-add path with one rounding
    # instead of two. With Apple-default ``-ffast-math`` the compiler
    # is allowed to re-associate the sequence, which on G1's H matrix
    # (freejoint root + tightly-coupled hinge dofs) accumulates enough
    # round-off to drive the diagonal pivot below the
    # ``max(d, 1e-30)`` clamp, then ``inv = 1/sqrt(1e-30) ≈ 1e15``
    # blows up the rest of the factorization to NaN.
    parts.append("            d = metal::fma(-ljk, ljk, d);")
    parts.append("        }")
    parts.append(f"        d = metal::max(d, ({msl_scalar})1e-30);")
    parts.append(f"        {msl_scalar} ljj = metal::precise::sqrt(d);")
    parts.append(f"        L.c[j*{n} + j] = ljj;")
    parts.append(f"        {msl_scalar} inv = ({msl_scalar})1.0 / ljj;")
    parts.append(f"        for (int i = j + 1; i < {n}; ++i) {{")
    parts.append(f"            {msl_scalar} s = L.c[i*{n} + j];")
    parts.append(f"            for (int k = 0; k < j; ++k) s = metal::fma(-L.c[i*{n} + k], L.c[j*{n} + k], s);")
    parts.append(f"            L.c[i*{n} + j] = s * inv;")
    parts.append(f"            L.c[j*{n} + i] = ({msl_scalar})0.0;")
    parts.append("        }")
    parts.append("    }")
    parts.append("}")
    return "\n".join(parts)


def _emit_tile_lower_solve_inplace(n: int, k: int, msl_scalar: str) -> str:
    """Emit ``wp_tile_lower_solve_<N>x<K>_<scalar>_inplace``: solve
    ``L * X = B`` in place, where ``L`` is NxN lower-triangular and
    ``B`` is NxK. ``B`` is mutated to hold ``X``.
    """
    L_name = f"wp_tile_{n}x{n}_{msl_scalar}"
    B_name = f"wp_tile_{n}x{k}_{msl_scalar}"
    name = f"wp_tile_lower_solve_{n}x{k}_{msl_scalar}_inplace"
    # ``noinline`` for the same reason as ``cholesky_inplace``: 3+
    # inlined copies in one kernel produce silently-zero writes on
    # Metal at N=16. Out-of-lining restores correctness.
    parts: list[str] = [f"__attribute__((noinline)) void {name}({L_name} L, thread {B_name}& B) {{"]
    parts.append("    #pragma clang loop unroll(disable)")
    parts.append(f"    for (int i = 0; i < {n}; ++i) {{")
    parts.append(f"        for (int col = 0; col < {k}; ++col) {{")
    parts.append(f"            {msl_scalar} s = B.c[i*{k} + col];")
    parts.append(f"            for (int kk = 0; kk < i; ++kk) s = metal::fma(-L.c[i*{n} + kk], B.c[kk*{k} + col], s);")
    parts.append(f"            B.c[i*{k} + col] = s / L.c[i*{n} + i];")
    parts.append("        }")
    parts.append("    }")
    parts.append("}")
    return "\n".join(parts)


def _emit_tile_upper_solve_inplace(n: int, k: int, msl_scalar: str) -> str:
    """Emit ``wp_tile_upper_solve_<N>x<K>_<scalar>_inplace``: solve
    ``U * X = B`` where ``U`` is upper-triangular. The blocked Cholesky
    solver passes ``transpose(L)`` as ``U`` so we read above the
    diagonal — i.e. column k > i — and use ``L.c[i*N + i]`` (the
    diagonal stays where it is on transposition).
    """
    L_name = f"wp_tile_{n}x{n}_{msl_scalar}"
    B_name = f"wp_tile_{n}x{k}_{msl_scalar}"
    name = f"wp_tile_upper_solve_{n}x{k}_{msl_scalar}_inplace"
    # ``noinline`` matches ``lower_solve_inplace`` — same Metal compiler
    # inline-bug at 3+ invocations.
    parts: list[str] = [f"__attribute__((noinline)) void {name}({L_name} U, thread {B_name}& B) {{"]
    parts.append("    #pragma clang loop unroll(disable)")
    parts.append(f"    for (int i = {n} - 1; i >= 0; --i) {{")
    parts.append(f"        for (int col = 0; col < {k}; ++col) {{")
    parts.append(f"            {msl_scalar} s = B.c[i*{k} + col];")
    parts.append(
        f"            for (int kk = i + 1; kk < {n}; ++kk) s = metal::fma(-U.c[i*{n} + kk], B.c[kk*{k} + col], s);"
    )
    parts.append(f"            B.c[i*{k} + col] = s / U.c[i*{n} + i];")
    parts.append("        }")
    parts.append("    }")
    parts.append("}")
    return "\n".join(parts)


def _emit_tile_lower_solve_inplace_transposed(n: int, k: int, msl_scalar: str) -> str:
    """Emit ``wp_tile_lower_solve_<N>x<K>_<scalar>_inplace_transposed``: solve
    ``L * X = transpose(A)`` and write the result back to ``A`` such that
    ``A`` ends up holding ``A * L^{-T}``.

    This fuses ``B = transpose(A); lower_solve(L, B); A = transpose(B)``
    into a single in-place pass. The fused form sidesteps a Metal-compiler
    miscompile observed when the same solve+writeback pair is emitted
    twice in one kernel (mujoco_warp's blocked Cholesky i-loop): the
    second iteration's writeback returns a stale struct and silently
    zeros out rows.

    ``A`` has shape ``(K, N)``; ``L`` is ``N×N`` lower-triangular. Each
    row of ``A`` is independently forward-substituted against ``L``.
    """
    L_name = f"wp_tile_{n}x{n}_{msl_scalar}"
    A_name = f"wp_tile_{k}x{n}_{msl_scalar}"
    name = f"wp_tile_lower_solve_{n}x{k}_{msl_scalar}_inplace_transposed"
    parts: list[str] = [f"__attribute__((noinline)) void {name}({L_name} L, thread {A_name}& A) {{"]
    parts.append("    #pragma clang loop unroll(disable)")
    parts.append(f"    for (int row = 0; row < {k}; ++row) {{")
    parts.append(f"        for (int i = 0; i < {n}; ++i) {{")
    parts.append(f"            {msl_scalar} s = A.c[row*{n} + i];")
    parts.append(f"            for (int kk = 0; kk < i; ++kk) s = metal::fma(-L.c[i*{n} + kk], A.c[row*{n} + kk], s);")
    parts.append(f"            A.c[row*{n} + i] = s / L.c[i*{n} + i];")
    parts.append("        }")
    parts.append("    }")
    parts.append("}")
    return "\n".join(parts)


def _emit_tile_upper_solve_inplace_transposed(n: int, k: int, msl_scalar: str) -> str:
    """Emit ``wp_tile_upper_solve_<N>x<K>_<scalar>_inplace_transposed``: solve
    ``U * X = transpose(A)`` and write the result back to ``A`` such that
    ``A`` ends up holding ``A * U^{-T}``. The fused form mirrors the
    lower-triangular variant; see that function for the rationale.
    """
    L_name = f"wp_tile_{n}x{n}_{msl_scalar}"
    A_name = f"wp_tile_{k}x{n}_{msl_scalar}"
    name = f"wp_tile_upper_solve_{n}x{k}_{msl_scalar}_inplace_transposed"
    parts: list[str] = [f"__attribute__((noinline)) void {name}({L_name} U, thread {A_name}& A) {{"]
    parts.append("    #pragma clang loop unroll(disable)")
    parts.append(f"    for (int row = 0; row < {k}; ++row) {{")
    parts.append(f"        for (int i = {n} - 1; i >= 0; --i) {{")
    parts.append(f"            {msl_scalar} s = A.c[row*{n} + i];")
    parts.append(
        f"            for (int kk = i + 1; kk < {n}; ++kk) s = metal::fma(-U.c[i*{n} + kk], A.c[row*{n} + kk], s);"
    )
    parts.append(f"            A.c[row*{n} + i] = s / U.c[i*{n} + i];")
    parts.append("        }")
    parts.append("    }")
    parts.append("}")
    return "\n".join(parts)


def _emit_tile_matmul(r: int, k: int, n: int, msl_scalar: str) -> str:
    """Emit ``wp_tile_matmul_RxKxN_<scalar>`` — ``C = beta*C + alpha*A*B``.

    A is RxK, B is KxN, C is RxN. All three are passed as struct values
    (C by reference so writes propagate). Used as ``tile_matmul_acc``'s
    backing helper at single-thread block_dim=1; cooperative parallelism
    is task #19 follow-up.
    """
    A = f"wp_tile_{r}x{k}_{msl_scalar}"
    B = f"wp_tile_{k}x{n}_{msl_scalar}"
    C = f"wp_tile_{r}x{n}_{msl_scalar}"
    name = f"wp_tile_matmul_{r}x{k}x{n}_{msl_scalar}"
    # ``noinline`` because mujoco_warp's blocked-Cholesky factor calls
    # this 5+ times per outer iteration. Inlining all copies tipped
    # past the Metal compiler's threshold and silently dropped writes
    # to the C accumulator (same family of bug as the
    # ``cholesky_inplace`` / ``*_solve_inplace`` regressions).
    parts: list[str] = [
        f"__attribute__((noinline)) void {name}({A} A, {B} B, thread {C}& C, {msl_scalar} alpha, {msl_scalar} beta) {{"
    ]
    parts.append(f"    for (int i = 0; i < {r}; ++i) {{")
    parts.append(f"        for (int j = 0; j < {n}; ++j) {{")
    parts.append(f"            {msl_scalar} s = ({msl_scalar})0;")
    parts.append(f"            for (int kk = 0; kk < {k}; ++kk) {{")
    # Explicit ``fma`` accumulator: ``-ffast-math`` allows reassociation,
    # and on G1's H structure the rank-1 updates lose enough precision
    # in float32 to corrupt the downstream Cholesky pivot.
    parts.append(f"                s = metal::fma(A.c[i*{k} + kk], B.c[kk*{n} + j], s);")
    parts.append("            }")
    # ``beta * C + alpha * s`` is also a fused-multiply-add candidate.
    parts.append(f"            C.c[i*{n} + j] = metal::fma(alpha, s, beta * C.c[i*{n} + j]);")
    parts.append("        }")
    parts.append("    }")
    parts.append("}")
    return "\n".join(parts)


def _emit_tile_transpose(rows: int, cols: int, msl_scalar: str) -> str:
    """Emit ``wp_tile_RxC_<scalar>_transpose`` returning a ``CxR`` tile.

    Single-thread copy with swapped indices. The blocked-Cholesky
    pattern in mujoco_warp transposes a 16x16 L-block (square — same
    output struct as input). Handle the rectangular case too.
    """
    in_name = f"wp_tile_{rows}x{cols}_{msl_scalar}"
    out_name = f"wp_tile_{cols}x{rows}_{msl_scalar}"
    parts: list[str] = [f"inline {out_name} {in_name}_transpose({in_name} A) {{"]
    parts.append(f"    {out_name} R;")
    parts.append(f"    for (int i = 0; i < {rows}; ++i) {{")
    parts.append(f"        for (int j = 0; j < {cols}; ++j) {{")
    parts.append(f"            R.c[j*{rows} + i] = A.c[i*{cols} + j];")
    parts.append("        }")
    parts.append("    }")
    parts.append("    return R;")
    parts.append("}")
    return "\n".join(parts)


def _emit_tile_cholesky_solve(n: int, k: int, msl_scalar: str) -> str:
    """Emit ``wp_tile_NxN_<scalar>_cholesky_solve_K`` — solve ``L L^T X = B``.

    The factor ``L`` is square ``N×N``; the RHS ``B`` is ``N×K`` (K=1 for
    vector RHS, the common case). Returns the solution as the same-shape
    tile. Forward/back substitute, no pivoting (L is already factored).
    """
    L_name = f"wp_tile_{n}x{n}_{msl_scalar}"
    if k == 1:
        B_name = f"wp_tile_{n}x1_{msl_scalar}"
        parts: list[str] = []
        parts.append(f"inline {B_name} {L_name}_cholesky_solve_1({L_name} L, {B_name} b) {{")
        parts.append(f"    {B_name} x = b;")
        # Forward: L y = b. Same Apple-MSL unroll bug as ``_cholesky``: at
        # N>=8 a fully unrolled triangular solve drops the writes to
        # x.c[N-1] (and sometimes the row before). Pin the outer loop
        # at runtime to keep the writes intact.
        parts.append("    #pragma clang loop unroll(disable)")
        parts.append(f"    for (int i = 0; i < {n}; ++i) {{")
        parts.append(f"        {msl_scalar} s = x.c[i];")
        parts.append(f"        for (int kk = 0; kk < i; ++kk) s = metal::fma(-L.c[i*{n} + kk], x.c[kk], s);")
        parts.append(f"        x.c[i] = s / L.c[i*{n} + i];")
        parts.append("    }")
        # Backward: L^T x = y
        parts.append("    #pragma clang loop unroll(disable)")
        parts.append(f"    for (int i = {n} - 1; i >= 0; --i) {{")
        parts.append(f"        {msl_scalar} s = x.c[i];")
        parts.append(f"        for (int kk = i + 1; kk < {n}; ++kk) s = metal::fma(-L.c[kk*{n} + i], x.c[kk], s);")
        parts.append(f"        x.c[i] = s / L.c[i*{n} + i];")
        parts.append("    }")
        parts.append("    return x;")
        parts.append("}")
        return "\n".join(parts)
    B_name = f"wp_tile_{n}x{k}_{msl_scalar}"
    parts = []
    parts.append(f"inline {B_name} {L_name}_cholesky_solve_{k}({L_name} L, {B_name} b) {{")
    parts.append(f"    {B_name} x = b;")
    parts.append(f"    for (int col = 0; col < {k}; ++col) {{")
    parts.append(f"        for (int i = 0; i < {n}; ++i) {{")
    parts.append(f"            {msl_scalar} s = x.c[i*{k} + col];")
    parts.append(f"            for (int kk = 0; kk < i; ++kk) s = metal::fma(-L.c[i*{n} + kk], x.c[kk*{k} + col], s);")
    parts.append(f"            x.c[i*{k} + col] = s / L.c[i*{n} + i];")
    parts.append("        }")
    parts.append(f"        for (int i = {n} - 1; i >= 0; --i) {{")
    parts.append(f"            {msl_scalar} s = x.c[i*{k} + col];")
    parts.append(
        f"            for (int kk = i + 1; kk < {n}; ++kk) s = metal::fma(-L.c[kk*{n} + i], x.c[kk*{k} + col], s);"
    )
    parts.append(f"            x.c[i*{k} + col] = s / L.c[i*{n} + i];")
    parts.append("        }")
    parts.append("    }")
    parts.append("    return x;")
    parts.append("}")
    return "\n".join(parts)


def _build_kernel_header(source: str) -> str:
    """Scan ``source`` for helpers we need to emit (big-vec structs,
    big-mat structs, spatial helpers, diag helper, ``wp_mat_extract``
    overloads) and return their definitions.
    """
    seen_vec: set[tuple[int, str]] = set()
    for m in _BIG_VEC_NAME_PAT.finditer(source):
        n = int(m.group(1))
        scalar = m.group(2)
        if scalar not in _MSL_PREFIX_TO_SAME:
            continue
        if n in _MSL_VEC_NATIVE_N:
            continue
        seen_vec.add((n, scalar))
    # Transform helpers are built on the ``wp_vec7_float`` struct — a kernel
    # can reference them without any vec7 local of its own (e.g. a bare
    # ``wp.transform_identity()``), so force the struct in.
    if "wp_transform_" in source:
        seen_vec.add((7, "float"))
    seen_mat: set[tuple[int, int, str]] = set()
    for m in _BIG_MAT_NAME_PAT.finditer(source):
        rows = int(m.group(1))
        cols = int(m.group(2))
        scalar = m.group(3)
        if scalar not in _MSL_PREFIX_TO_SAME:
            continue
        # Skip only *square* native sizes — non-square small mats use
        # our custom struct (see ``_msl_mat_name``).
        if rows == cols and rows in _MSL_VEC_NATIVE_N:
            continue
        seen_mat.add((rows, cols, scalar))
    parts: list[str] = []
    for n, scalar in sorted(seen_vec):
        parts.append(_emit_big_vec_struct(f"wp_vec{n}_{scalar}", n, scalar))
    if (6, "float") in seen_vec:
        parts.append(_emit_spatial_helpers())
    for rows, cols, scalar in sorted(seen_mat):
        parts.append(_emit_big_mat_struct(f"wp_mat{rows}x{cols}_{scalar}", rows, cols, scalar))
    slice_struct = _emit_slice_t_struct(source)
    if slice_struct:
        parts.append(slice_struct)
    native_mat_overloads = _emit_native_mat_extract_overloads(source)
    if native_mat_overloads:
        parts.append(native_mat_overloads)
    if "wp_diag_float3" in source:
        parts.append(_DIAG_HELPER_FLOAT3)
    if "wp_quat_" in source or "wp_transform_" in source:
        parts.append(_QUAT_HELPERS)
    if "wp_transform_" in source:
        parts.append(_TRANSFORM_HELPERS)
    if "wp_rand" in source:
        parts.append(_RAND_HELPERS)
    if "wp_svd" in source or "wp_qr3" in source or "wp_eig3" in source:
        parts.append(_SVD_HELPERS)
    misc_math = _emit_misc_math_helpers(source)
    if misc_math:
        parts.append(misc_math)
    if "wp_dot" in source:
        dot_overloads = _emit_wp_dot_overloads(source)
        if dot_overloads:
            parts.append(dot_overloads)
    # Tile primitive helpers. Emit struct + load/store for each tile shape
    # the kernel references; emit Cholesky / Cholesky-solve only if those
    # specific helpers appear in the source (they're invoked through
    # ``<name>_cholesky`` / ``<name>_cholesky_solve_<K>`` suffixes the
    # intrinsic-translation pass produces).
    seen_tile: set[tuple[int, int, str]] = set()
    for m in _TILE_NAME_PAT.finditer(source):
        seen_tile.add((int(m.group(1)), int(m.group(2)), m.group(3)))
    # ``wp_tile_RxC_<scalar>_transpose`` returns a ``CxR`` struct, so we
    # need both struct shapes declared even if the transposed shape
    # never appears in a load/store call site (it lives only in the
    # function-return type).
    transpose_pat = re.compile(r"\bwp_tile_(\d+)x(\d+)_(\w+)_transpose\b")
    transpose_seen: set[tuple[int, int, str]] = set()
    for m in transpose_pat.finditer(source):
        rows, cols = int(m.group(1)), int(m.group(2))
        scalar = m.group(3)
        transpose_seen.add((rows, cols, scalar))
        seen_tile.add((rows, cols, scalar))
        seen_tile.add((cols, rows, scalar))
    for rows, cols, scalar in sorted(seen_tile):
        parts.append(_emit_tile_struct(rows, cols, scalar))
    # Vec-element tiles (``wp_tile_RxC_vec<N>_<scalar>``) — used by the
    # dense-Jacobian path with ``vec3`` / ``spatial_vector`` (vec6)
    # element types.
    vec_tile_pat = re.compile(r"\bwp_tile_(\d+)x(\d+)_vec(\d+)_(\w+)(?![\w])")
    seen_vec_tile: set[tuple[int, int, int, str]] = set()
    for m in vec_tile_pat.finditer(source):
        rows = int(m.group(1))
        cols = int(m.group(2))
        n_elem = int(m.group(3))
        scalar = m.group(4)
        if scalar in _MSL_PREFIX_TO_SAME:
            seen_vec_tile.add((rows, cols, n_elem, scalar))
    for rows, cols, n_elem, scalar in sorted(seen_vec_tile):
        parts.append(_emit_tile_struct_vec(rows, cols, n_elem, scalar))
    # Cooperative load/store variants — scan for ``..._load_coop`` /
    # ``..._store_coop`` references and emit their definitions
    # alongside the single-thread ones.
    coop_load_pat = re.compile(r"\bwp_tile_(\d+)x(\d+)_(\w+)_load_coop\b")
    coop_store_pat = re.compile(r"\bwp_tile_(\d+)x(\d+)_(\w+)_store_coop\b")
    seen_coop_loads: set[tuple[int, int, str]] = set()
    seen_coop_stores: set[tuple[int, int, str]] = set()
    for m in coop_load_pat.finditer(source):
        seen_coop_loads.add((int(m.group(1)), int(m.group(2)), m.group(3)))
    for m in coop_store_pat.finditer(source):
        seen_coop_stores.add((int(m.group(1)), int(m.group(2)), m.group(3)))
    for rows, cols, scalar in sorted(seen_coop_loads):
        parts.append(_emit_tile_load_coop(rows, cols, scalar))
    for rows, cols, scalar in sorted(seen_coop_stores):
        parts.append(_emit_tile_store_coop(rows, cols, scalar))
    for rows, cols, scalar in sorted(transpose_seen):
        parts.append(_emit_tile_transpose(rows, cols, scalar))
    matmul_pat = re.compile(r"\bwp_tile_matmul_(\d+)x(\d+)x(\d+)_(\w+)\b")
    matmul_seen: set[tuple[int, int, int, str]] = set()
    for m in matmul_pat.finditer(source):
        r_, k_, n_ = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sc_ = m.group(4)
        matmul_seen.add((r_, k_, n_, sc_))
    for r_, k_, n_, sc_ in sorted(matmul_seen):
        parts.append(_emit_tile_matmul(r_, k_, n_, sc_))
    chol_inplace_pat = re.compile(r"\bwp_tile_(\d+)x(\d+)_(\w+)_cholesky_inplace\b")
    chol_inplace_seen: set[tuple[int, str]] = set()
    for m in chol_inplace_pat.finditer(source):
        r_, c_ = int(m.group(1)), int(m.group(2))
        if r_ == c_:
            chol_inplace_seen.add((r_, m.group(3)))
    for n_, sc_ in sorted(chol_inplace_seen):
        parts.append(_emit_tile_cholesky_inplace(n_, sc_))
    lsolve_pat = re.compile(r"\bwp_tile_lower_solve_(\d+)x(\d+)_(\w+)_inplace\b(?!_transposed)")
    usolve_pat = re.compile(r"\bwp_tile_upper_solve_(\d+)x(\d+)_(\w+)_inplace\b(?!_transposed)")
    lsolve_t_pat = re.compile(r"\bwp_tile_lower_solve_(\d+)x(\d+)_(\w+)_inplace_transposed\b")
    usolve_t_pat = re.compile(r"\bwp_tile_upper_solve_(\d+)x(\d+)_(\w+)_inplace_transposed\b")
    lsolve_seen: set[tuple[int, int, str]] = set()
    usolve_seen: set[tuple[int, int, str]] = set()
    lsolve_t_seen: set[tuple[int, int, str]] = set()
    usolve_t_seen: set[tuple[int, int, str]] = set()
    for m in lsolve_pat.finditer(source):
        lsolve_seen.add((int(m.group(1)), int(m.group(2)), m.group(3)))
    for m in usolve_pat.finditer(source):
        usolve_seen.add((int(m.group(1)), int(m.group(2)), m.group(3)))
    for m in lsolve_t_pat.finditer(source):
        lsolve_t_seen.add((int(m.group(1)), int(m.group(2)), m.group(3)))
    for m in usolve_t_pat.finditer(source):
        usolve_t_seen.add((int(m.group(1)), int(m.group(2)), m.group(3)))
    for n_, k_, sc_ in sorted(lsolve_seen):
        parts.append(_emit_tile_lower_solve_inplace(n_, k_, sc_))
    for n_, k_, sc_ in sorted(usolve_seen):
        parts.append(_emit_tile_upper_solve_inplace(n_, k_, sc_))
    for n_, k_, sc_ in sorted(lsolve_t_seen):
        parts.append(_emit_tile_lower_solve_inplace_transposed(n_, k_, sc_))
    for n_, k_, sc_ in sorted(usolve_t_seen):
        parts.append(_emit_tile_upper_solve_inplace_transposed(n_, k_, sc_))
    cholesky_pat = re.compile(r"\bwp_tile_(\d+)x(\d+)_(\w+)_cholesky\b(?!_solve|_inplace|_coop)")
    cholesky_solve_pat = re.compile(r"\bwp_tile_(\d+)x(\d+)_(\w+)_cholesky_solve_(\d+)\b")
    coop_cholesky_pat = re.compile(r"\bwp_tile_(\d+)x(\d+)_(\w+)_cholesky_coop\b")
    seen_cholesky: set[tuple[int, str]] = set()
    seen_coop_cholesky: set[tuple[int, str]] = set()
    for m in cholesky_pat.finditer(source):
        rows, cols = int(m.group(1)), int(m.group(2))
        if rows == cols:
            seen_cholesky.add((rows, m.group(3)))
    for m in coop_cholesky_pat.finditer(source):
        rows, cols = int(m.group(1)), int(m.group(2))
        if rows == cols:
            seen_coop_cholesky.add((rows, m.group(3)))
    for n, scalar in sorted(seen_cholesky):
        parts.append(_emit_tile_cholesky(n, scalar))
    for n, scalar in sorted(seen_coop_cholesky):
        parts.append(_emit_tile_cholesky_coop(n, scalar))
    seen_solve: set[tuple[int, int, str]] = set()
    for m in cholesky_solve_pat.finditer(source):
        rows, cols, k = int(m.group(1)), int(m.group(2)), int(m.group(4))
        if rows == cols:
            seen_solve.add((rows, k, m.group(3)))
    for n, k, scalar in sorted(seen_solve):
        parts.append(_emit_tile_cholesky_solve(n, k, scalar))
    # mujoco_warp ``@wp.func`` helpers referenced by ``tile_map``.
    user_func_defs = _emit_referenced_user_funcs(source)
    if user_func_defs:
        parts.append(user_func_defs)
    if not parts:
        return ""
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
        # Zero-arg ``wp::vec_t<N, T>()`` is the default-construction
        # form Warp uses for ``vec_t = vec5()`` style local declarations.
        # The custom big-vec struct's brace value-init (``T{}``) zero-
        # initialises its scalar array, so we route to that instead of
        # the N-arg ``_make`` factory which would need N zero literals.
        if not args_str:
            return f"wp_vec{n}_{msl_scalar}{{}}"
        return f"wp_vec{n}_{msl_scalar}_make({args_str})"

    return pat.sub(repl, text)


def _rewrite_mat_t_constructor(text: str) -> str:
    """Rewrite ``wp::mat_t<R, C, wp::T>(v00, v01, ..., v(R-1)(C-1))`` (row-major
    flat, optionally brace-wrapped) into MSL form.

    For native sizes (R, C in {2, 3, 4}) we emit ``floatRxC`` constructed
    column-by-column from row-major args. For larger sizes we route to
    ``wp_matRxC_<scalar>_make`` which takes the row-major flat args
    directly (the struct stores row-major).

    Constructors with the wrong arg count or unsupported types are left
    untouched; the unsupported-intrinsic guard catches them downstream.
    """
    pat = re.compile(r"wp::mat_t<\s*(\d+)\s*,\s*(\d+)\s*,\s*wp::(\w+)\s*>\s*\(([^()]*)\)")

    def repl(m: re.Match[str]) -> str:
        rows, cols = int(m.group(1)), int(m.group(2))
        scalar = m.group(3)
        args_str = m.group(4).strip()
        if args_str.startswith("{") and args_str.endswith("}"):
            args_str = args_str[1:-1].strip()
        args = [a.strip() for a in args_str.split(",") if a.strip()]
        full_ctype = f"wp::{scalar}"
        if full_ctype not in _MSL_VEC_SCALAR_PREFIX or rows < 2 or cols < 2:
            return m.group(0)
        msl_scalar = _MSL_VEC_SCALAR_PREFIX[full_ctype]
        # Native MSL matrix only for *square* sizes (``_msl_mat_name``).
        # Non-square goes through the custom ``wp_mat{R}x{C}_<scalar>``
        # struct so the row-access proxy works; ``mat_t<2, 3>(args)``
        # routes to ``wp_mat2x3_float_make(args...)``.
        if not args:
            if rows == cols and rows in _MSL_VEC_NATIVE_N:
                return f"{msl_scalar}{rows}x{cols}(0)"
            return f"wp_mat{rows}x{cols}_{msl_scalar}()"
        # Single-scalar broadcast: ``mat_t<R,C,T>(scalar)``.
        if len(args) == 1:
            scalar_arg = args[0]
            if rows == cols and rows in _MSL_VEC_NATIVE_N:
                # Square native mat: ``floatNxN(scalar)`` is the diagonal-
                # broadcast form and does fill-from-scalar in MSL when N==M.
                return f"{msl_scalar}{rows}x{cols}({scalar_arg})"
            # Non-square or big sizes go through the broadcast factory.
            return f"wp_mat{rows}x{cols}_{msl_scalar}_make({scalar_arg})"
        if len(args) != rows * cols:
            return m.group(0)
        if rows == cols and rows in _MSL_VEC_NATIVE_N:
            msl_vec = f"{msl_scalar}{rows}"
            msl_mat = f"{msl_scalar}{rows}x{cols}"
            col_strs: list[str] = []
            for c in range(cols):
                col_components = [args[r * cols + c] for r in range(rows)]
                col_strs.append(f"{msl_vec}({', '.join(col_components)})")
            return f"{msl_mat}({', '.join(col_strs)})"
        return f"wp_mat{rows}x{cols}_{msl_scalar}_make({', '.join(args)})"

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


_TILE_CTYPE_HEAD_PAT = re.compile(r"wp::tile_(shared|register)_t<\s*(.+)$")
_TILE_SHAPE_PAT = re.compile(r"wp::tile_shape_t<\s*(\d+)\s*(?:,\s*(\d+))?\s*>")


_VEC_T_INNER_PAT = re.compile(r"wp::vec_t<\s*(\d+)\s*,\s*([\w:]+)\s*>")


def _vec_dtype_inner(ctype: str) -> tuple[int, str] | None:
    """Parse a ``wp::vec_t<N, wp::scalar>`` ctype to ``(N, inner_ctype)``.

    Returns ``None`` if ``ctype`` is not a vec_t. Strips any trailing
    qualifiers (``const``, ``&``, ``*``) to keep the parser tolerant of
    parameter-position decorations.
    """
    m = _VEC_T_INNER_PAT.search(ctype)
    if m is None:
        return None
    return int(m.group(1)), m.group(2)


def _parse_tile_ctype(ctype: str) -> tuple[str, str, int, int] | None:
    """Parse ``wp::tile_(shared|register)_t<dtype, layout<shape<R[, C]>, ...>, ...>``.

    Returns ``(kind, dtype_ctype, rows, cols)`` or ``None`` if ``ctype`` isn't a
    tile type. ``kind`` is ``"shared"`` or ``"register"``. ``dtype_ctype`` is
    the raw inner type — could be a scalar (``wp::float32``) or a vec_t
    (``wp::vec_t<3, wp::float32>``).
    """
    head = _TILE_CTYPE_HEAD_PAT.search(ctype)
    if head is None:
        return None
    kind = head.group(1)
    rest = head.group(2)
    # Walk ``rest`` finding the comma at depth 0 that separates dtype from
    # layout — the dtype might itself contain ``<...>`` (vec_t case).
    depth = 0
    split_at = None
    for i, ch in enumerate(rest):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            split_at = i
            break
    if split_at is None:
        return None
    dtype_ctype = rest[:split_at].strip()
    shape_match = _TILE_SHAPE_PAT.search(rest[split_at:])
    if shape_match is None:
        return None
    rows = int(shape_match.group(1))
    cols = int(shape_match.group(2)) if shape_match.group(2) else 1
    return kind, dtype_ctype, rows, cols


def _msl_var_type(ctype: str) -> str:
    """Translate a local variable's ctype to MSL."""
    if ctype.endswith("*"):
        return _msl_pointer_type(ctype)
    # Tile types translate two ways depending on element shape:
    #   * shape == (1,) or (1, 1) — a single-thread tile of one element.
    #     With ``block_dim=1`` (the Metal default) every register-tile
    #     produced by ``wp.tile`` / ``wp.tile_reduce`` collapses to this,
    #     so we lower the type to the inner element type itself. The
    #     ``var_X = wp::tile<...>(var_x)`` IR becomes a plain assign and
    #     ``wp::tile_reduce<op>(tile)`` returns the value unchanged.
    #   * larger shapes — the standard ``wp_tile_RxC_<scalar>`` private-
    #     memory struct (see ``_emit_tile_struct``). Currently scalar-
    #     element only; vec/mat elements at >1 size aren't yet supported.
    parsed = _parse_tile_ctype(ctype)
    if parsed is not None:
        kind, dtype_ctype, rows, cols = parsed
        if rows == 1 and cols == 1:
            return _msl_scalar_type(dtype_ctype)
        # Scalar-element tile: ``wp_tile_RxC_<scalar>``.
        if dtype_ctype in _SCALAR_CTYPE_TO_MSL:
            return f"wp_tile_{rows}x{cols}_{_SCALAR_CTYPE_TO_MSL[dtype_ctype]}"
        # Vec-element tile (``wp::vec_t<N, scalar>``): emit a struct
        # holding ``R*C*N`` flat scalars. ``vec3`` / ``spatial_vector``
        # tiles in mujoco_warp's dense-Jacobian path go through here.
        v = _vec_dtype_inner(dtype_ctype)
        if v is not None:
            n_elem, inner_scalar_ctype = v
            inner_scalar = _SCALAR_CTYPE_TO_MSL.get(inner_scalar_ctype)
            if inner_scalar is None:
                raise MetalCodegenError(
                    f"MSL codegen: tile of {dtype_ctype!r} not supported "
                    f"(unsupported inner scalar {inner_scalar_ctype!r})"
                )
            return f"wp_tile_{rows}x{cols}_vec{n_elem}_{inner_scalar}"
        raise MetalCodegenError(
            f"MSL codegen: tile of {dtype_ctype!r} (shape {rows}x{cols}) not yet supported "
            "(only scalar-element and vec-element tiles emit a struct)"
        )
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
    # Warp's ``uint32`` / ``int32`` etc. wrap into typed-int classes
    # whose ``int(...)`` works. mujoco_warp's solver passes these as
    # bitmask constants (``DisableBit.WARMSTART`` = ``uint32(2)``).
    try:
        return str(int(value))
    except (TypeError, ValueError):
        pass
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
    # wp.block_dim() always lowers to literal ``1`` on the Metal backend.
    # This pairs with the ``block_dim=1`` setting in
    # ``generate_msl_kernel`` and the ``tg_y=1`` launcher policy below to
    # run every cooperative-tile kernel serially per threadgroup. Remove
    # this special-case once we implement real threadgroup-cooperative
    # tiles.
    (re.compile(r"\bbuiltin_block_dim\s*\(\s*\)"), "1"),
    # ``WP_TILE_SYNC()`` is the macro used by ``@wp.func_native`` shims like
    # mujoco_warp's ``_syncthreads`` — translate to MSL's threadgroup
    # barrier. With single-thread threadgroups it's a no-op; with bigger
    # ones it's the same primitive CUDA's ``__syncthreads()`` provides.
    (re.compile(r"\bWP_TILE_SYNC\s*\(\s*\)"), "threadgroup_barrier(mem_flags::mem_threadgroup)"),
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
    # ``wp::mod`` — MSL's ``%`` is integer-only; floats need ``metal::fmod``.
    # Routed through a ``wp_mod`` overload set so both resolve correctly.
    (re.compile(r"wp::mod\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"), r"wp_mod(\1, \2)"),
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
    (re.compile(r"wp::length_sq\s*\(\s*([^()]+?)\s*\)"), r"wp_dot(\1, \1)"),
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
    # ``wp::slice_t(start, stop, step)`` — slice constructor for matrix
    # row/col slicing (``mat[:, c]`` lowers to a ``slice_t`` plus a
    # ``wp::extract<N>`` call). MSL's ``wp_slice_t`` is a plain POD struct
    # without a ctor; brace-init writes the three fields in order.
    (
        re.compile(r"wp::slice_t\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)"),
        r"wp_slice_t{\1, \2, \3}",
    ),
    # ``wp::extract<N>(m, ...)`` — the templated form generated by matrix
    # slicing (``mat[:, c]`` / ``mat[r, :]``). ``N`` is the resulting vec
    # length but our MSL helpers infer it from the matrix dim; just strip
    # the template arg and dispatch to ``wp_mat_extract``. C++ overload
    # resolution picks ``(mat, slice, int)`` vs ``(mat, int, slice)``.
    (
        re.compile(r"wp::extract\s*<\s*\d+\s*>\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)"),
        r"wp_mat_extract(\1, \2, \3)",
    ),
    # ``wp::extract(mat, i, j)`` (3-arg, matrix form) — must come BEFORE the
    # 2-arg vec form below, otherwise the non-greedy ``[^()]+?`` for the
    # second arg would swallow ``i, j`` together. We dispatch to
    # ``wp_mat_extract``, which has overloads emitted in the kernel header
    # for each native ``floatRxC`` (column-major: returns ``m[col][row]``)
    # and each big-mat ``wp_matRxC_<scalar>`` (row-major: returns
    # ``m.c[row * cols + col]``).
    (
        re.compile(r"wp::extract\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)"),
        r"wp_mat_extract(\1, \2, \3)",
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
    # ``wp::diag(vec3)`` builds a 3x3 diagonal matrix. MSL has no native
    # diag-from-vector helper, so we route it through ``wp_diag_float3``
    # (emitted in the kernel header when this token appears in the source).
    # Only the vec3 form is supported — it's the only size used in practice.
    (
        re.compile(r"wp::diag\s*\(\s*([^()]+?)\s*\)"),
        r"wp_diag_float3(\1)",
    ),
    # ``wp::identity<N, wp::T>()`` builds an NxN identity matrix. MSL's
    # native floatNxN single-scalar constructor fills the diagonal, so
    # ``floatNxN(1.0f)`` is the identity. (Same for int/uint.)
    (
        re.compile(r"wp::identity\s*<\s*(\d+)\s*,\s*wp::(\w+)\s*>\s*\(\s*\)"),
        lambda m: (
            f"{_MSL_VEC_SCALAR_PREFIX[f'wp::{m.group(2)}']}{m.group(1)}x{m.group(1)}(1.0f)"
            if f"wp::{m.group(2)}" in _MSL_VEC_SCALAR_PREFIX
            else m.group(0)
        ),
    ),
    # Quaternion builtins — dispatched by name to the ``wp_quat_*`` helpers
    # emitted in the kernel header (see ``_QUAT_HELPERS``). The
    # ``quat_rotate_inv`` rename can't be eaten by the ``quat_rotate`` one:
    # ``_`` is a word character, so ``\b`` after "rotate" doesn't match
    # inside "rotate_inv". ``quat_from_matrix`` may carry explicit template
    # args (``<3, 3, wp::float32>``); strip them — the helper is float3x3-only.
    (re.compile(r"\bwp::quat_rotate_inv\b"), "wp_quat_rotate_inv"),
    (re.compile(r"\bwp::quat_rotate\b"), "wp_quat_rotate"),
    (re.compile(r"\bwp::quat_inverse\b"), "wp_quat_inverse"),
    (re.compile(r"\bwp::quat_from_axis_angle\b"), "wp_quat_from_axis_angle"),
    (re.compile(r"\bwp::quat_to_axis_angle\b"), "wp_quat_to_axis_angle"),
    (re.compile(r"\bwp::quat_to_matrix\b"), "wp_quat_to_matrix"),
    (re.compile(r"\bwp::quat_from_matrix\s*(?:<[^<>]*>)?"), "wp_quat_from_matrix"),
    (re.compile(r"\bwp::quat_slerp\b"), "wp_quat_slerp"),
    (re.compile(r"\bwp::quat_rpy\b"), "wp_quat_rpy"),
    # ``wp.quat_identity()`` — a constant; no helper needed. Layout is
    # (x, y, z, w) so identity is w=1.
    (
        re.compile(r"\bwp::quat_identity\s*(?:<[^<>]*>)?\s*\(\s*\)"),
        "float4(0.0f, 0.0f, 0.0f, 1.0f)",
    ),
    # Rigid-transform builtins — transforms are ``wp_vec7_float`` after the
    # ``vec_t<7>`` normalization; the ``wp_transform_*`` helper bodies live
    # in ``_TRANSFORM_HELPERS``. ``transform_multiply`` must be renamed
    # before the generic ``wp::mul`` handling can't touch it (distinct
    # name, but keep it grouped here for clarity).
    (re.compile(r"\bwp::transform_point\b"), "wp_transform_point"),
    (re.compile(r"\bwp::transform_vector\b"), "wp_transform_vector"),
    (re.compile(r"\bwp::transform_multiply\b"), "wp_transform_multiply"),
    (re.compile(r"\bwp::transform_inverse\b"), "wp_transform_inverse"),
    (re.compile(r"\bwp::transform_get_translation\b"), "wp_transform_get_translation"),
    (re.compile(r"\bwp::transform_get_rotation\b"), "wp_transform_get_rotation"),
    (
        re.compile(r"\bwp::transform_identity\s*(?:<[^<>]*>)?\s*\(\s*\)"),
        "wp_transform_identity()",
    ),
    # Random-number builtins — the ``wp_rand*`` helper bodies are a
    # bit-exact port of warp/native/rand.h (see ``_RAND_HELPERS``).
    (re.compile(r"\bwp::rand_init\b"), "wp_rand_init"),
    (re.compile(r"\bwp::randi\b"), "wp_randi"),
    (re.compile(r"\bwp::randu\b"), "wp_randu"),
    (re.compile(r"\bwp::randf\b"), "wp_randf"),
    (re.compile(r"\bwp::randn\b"), "wp_randn"),
    # Interpolation / misc math — routed through ``wp_*`` helpers whose
    # bodies match Warp's native implementations (NOT the closest MSL
    # builtin — see ``_MISC_MATH_HELPERS`` for where they differ).
    (re.compile(r"\bwp::lerp\b"), "wp_lerp"),
    (re.compile(r"\bwp::smoothstep\b"), "wp_smoothstep"),
    (re.compile(r"\bwp::frac\b"), "wp_frac"),
    (re.compile(r"\bwp::degrees\b"), "wp_degrees"),
    (re.compile(r"\bwp::radians\b"), "wp_radians"),
    # Small linear-algebra builtins with no MSL native equivalent.
    (re.compile(r"\bwp::outer\b"), "wp_outer"),
    (re.compile(r"\bwp::skew\b"), "wp_skew"),
    (re.compile(r"\bwp::trace\b"), "wp_trace"),
    # ``wp::cw_mul`` / ``wp::cw_div`` — component-wise multiply / divide.
    # MUST go through the ``wp_cw_*`` overload set, not raw ``*`` / ``/``:
    # MSL's ``matNxN * matNxN`` is a *matrix multiply*, which silently
    # returned wrong values for matrix operands before the helper existed.
    # For vector operands the generic template lowers back to ``a * b``
    # (component-wise on MSL floatN and on our big-vec structs).
    (
        re.compile(r"\bwp::cw_mul\b"),
        "wp_cw_mul",
    ),
    (
        re.compile(r"\bwp::cw_div\b"),
        "wp_cw_div",
    ),
    # Scalar sign/step/nonzero — hand-rolled because Warp's semantics
    # differ from MSL's: ``wp.sign(0) == 1`` (``metal::sign(0) == 0``),
    # ``wp.step`` is 1 for x < 0 (GLSL/MSL ``step(edge, x)`` is the
    # opposite convention and takes two args).
    (re.compile(r"\bwp::sign\b"), "wp_sign"),
    (re.compile(r"\bwp::step\b"), "wp_step"),
    (re.compile(r"\bwp::nonzero\b"), "wp_nonzero"),
    # ``wp::cbrt`` — MSL has no cbrt; emulate with copysign+pow so
    # negative inputs keep their sign like C's ``cbrtf``.
    (re.compile(r"\bwp::cbrt\b"), "wp_cbrt"),
    # Matrix inverse / diagonal extraction — ``wp_inverse`` overloads for
    # float2x2/3x3/4x4 are ports of warp/native/mat.h (kEps == 0.0f, so
    # a singular matrix returns the zero matrix, matching CPU).
    (re.compile(r"\bwp::inverse\b"), "wp_inverse"),
    (re.compile(r"\bwp::get_diag\b"), "wp_get_diag"),
    # Matrix decompositions — output-parameter builtins ported from
    # warp/native/svd.h (see ``_SVD_HELPERS``).
    (re.compile(r"\bwp::svd3\b"), "wp_svd3"),
    (re.compile(r"\bwp::svd2\b"), "wp_svd2"),
    (re.compile(r"\bwp::qr3\b"), "wp_qr3"),
    (re.compile(r"\bwp::eig3\b"), "wp_eig3"),
    # NOTE: ``wp::lower_bound`` is intentionally NOT handled here — its
    # 2-arg form references the array's ``<argname>_shape`` input, which
    # requires the *final* parameter name. It's rewritten inside
    # ``_finalize`` after the ``var_<argname>`` -> ``<argname>`` rename
    # (see ``_LOWER_BOUND_4ARG_PAT`` / ``_LOWER_BOUND_2ARG_PAT``).
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
    # ``dot`` needs custom overloads to support our big-vec structs;
    # route it through a ``wp_dot`` wrapper that dispatches to either
    # ``metal::dot`` (native floatN) or our hand-rolled sum-of-products
    # (big-vec). Other math builtins fall back to the metal namespace.
    if _name == "dot":
        _INTRINSIC_PATTERNS.append((re.compile(rf"\bwp::{_name}\b"), "wp_dot"))
    else:
        _INTRINSIC_PATTERNS.append((re.compile(rf"\bwp::{_name}\b"), f"metal::{_name}"))
del _name


def _wrap_atomic_load_reads(text: str, arr_name: str) -> str:
    """Wrap any read of ``arr_name[<balanced...>]`` in atomic_load.

    A regex won't suffice because the index expression can contain nested
    ``[...]`` (e.g. ``arr[i * shape[k] + j]``). We scan forward, count
    bracket depth, and rewrite the matched span.

    Reads can appear in many syntactic positions:
      - ``var_X = arr[idx];``                         (plain assign)
      - ``var_X = float3(arr[i], arr[j], arr[k]);``   (constructor arg)
      - ``foo(arr[idx], ...)``                        (call arg)
      - ``var_X = expr op arr[idx];``                 (binary expr operand)

    We exclude WRITE forms by skipping when the next non-space token after
    the index closes is ``=`` (assignment) or when ``=`` directly precedes
    the array name (compound ``+=`` / ``-=`` / etc.).
    """
    out_parts: list[str] = []
    i = 0
    n = len(text)
    needle = f"{arr_name}["
    arr_len = len(arr_name)
    while i < n:
        j = text.find(needle, i)
        if j < 0:
            out_parts.append(text[i:])
            break
        # Word-boundary check: the char before must not be a name char.
        prev_ch = text[j - 1] if j > 0 else ""
        if prev_ch.isalnum() or prev_ch == "_":
            out_parts.append(text[i : j + 1])
            i = j + 1
            continue
        # Skip address-of forms ``&arr[idx]`` — those feed into atomic
        # builtins (``atomic_fetch_add_explicit(&arr[i], val, ...)``)
        # which want the pointer, not the loaded value.
        if prev_ch == "&":
            out_parts.append(text[i : j + 1])
            i = j + 1
            continue
        # Find the matching close bracket, accounting for nesting.
        bracket_start = j + len(needle)
        depth = 1
        k = bracket_start
        while k < n and depth > 0:
            if text[k] == "[":
                depth += 1
            elif text[k] == "]":
                depth -= 1
            k += 1
        if depth != 0:
            out_parts.append(text[i:k])
            i = k
            continue
        # Skip if this is a WRITE: ``arr[idx] = ...`` or ``arr[idx] += ...``.
        # Look at the next non-space char after the closing ``]``.
        m = k
        while m < n and text[m] == " ":
            m += 1
        is_write = m < n and text[m] == "=" and (m + 1 >= n or text[m + 1] != "=")
        # Also a WRITE if the op is compound ``+= /= *= -=``.
        if not is_write and m + 1 < n and text[m + 1] == "=" and text[m] in "+-*/":
            is_write = True
        if is_write:
            out_parts.append(text[i:k])
            i = k
            continue
        idx_expr = text[bracket_start : k - 1]
        out_parts.append(text[i:j])
        out_parts.append(f"atomic_load_explicit(&{arr_name}[{idx_expr}], memory_order_relaxed)")
        i = k
    return "".join(out_parts)


# ``wp::lower_bound`` — binary search over a sorted 1-D array, rewritten in
# ``_finalize`` after the arg rename (the 2-arg form needs the final param
# name to reference the ``<argname>_shape`` input). The 4-arg (arr, begin,
# end, value) form must match before the 2-arg one.
_LOWER_BOUND_4ARG_PAT = re.compile(
    r"wp::lower_bound\s*\(\s*([A-Za-z_]\w*)\s*,\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"
)
_LOWER_BOUND_2ARG_PAT = re.compile(r"wp::lower_bound\s*\(\s*([A-Za-z_]\w*)\s*,\s*([^()]+?)\s*\)")


def _translate_intrinsics(line: str) -> str:
    """Apply intrinsic substitutions until convergence."""
    prev = None
    while prev != line:
        prev = line
        for pat, repl in _INTRINSIC_PATTERNS:
            line = pat.sub(repl, line)
    return line


# Tile-primitive translators. Run as part of ``_finalize`` after the
# regular intrinsic patterns. Each takes the post-pattern line and the
# kernel's array-arg ndim map (so we can compute shape-aware base offsets
# and row strides). See ``_emit_tile_struct`` / ``_emit_tile_cholesky``
# for the helper signatures these calls translate to.
#
# IR shapes seen in mujoco_warp.step():
#   2-D tile from a slice of a 3-D arr (mass-matrix factor):
#     ``var_X = wp::tile_load<wp::float32, true, false, 6, 6>(var_arr, var_lead, var_off0, var_off1)``
#   1-D tile from a slice of a 2-D arr (RHS for cholesky_solve):
#     ``var_X = wp::tile_load<wp::float32, true, false, 6>(var_arr, var_lead, var_off)``
#   Cholesky: ``var_X = wp::tile_cholesky<false>(var_a, var_b, var_c, var_in, var_out)``
#     — five func args; first three are leading zeros from the cuBLASDx LTO
#     calling convention (we ignore them); the in/out tiles are the same
#     local in mujoco_warp's pattern.
#   Cholesky-solve: ``var_X = wp::tile_cholesky_solve<false>(var_a, var_b, var_c, var_L, var_b_in, var_x_out)``


# ``wp::tile<dtype>(x)`` — pack a per-thread value into a register tile of
# shape (block_dim,). With block_dim=1 the tile *is* the value, so we
# lower to a plain assign.
_TILE_BUILTIN_TILE_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile\s*<[^()]*>\s*\(\s*var_(\w+)\s*\)")
# ``wp::tile_reduce(op_fn, tile)`` — reduce across threads. The op (``wp::add``,
# user fn, etc.) comes as the first *argument*, not as a template arg.
# With block_dim=1 the tile is a single element; the reduction is a no-op
# and the output is just the input value.
_TILE_REDUCE_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_reduce\s*\(\s*[\w:]+\s*,\s*var_(\w+)\s*\)")
# ``wp::tile_zeros<dtype, ...>()`` / ``wp::tile_ones<dtype, ...>()`` —
# constant-fill register tile. With block_dim=1 it's the corresponding
# scalar 0 / 1 (or vec ``T(0)`` / ``T(1)``).
_TILE_ZEROS_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_zeros\s*<\s*(?:wp::)?(\w+)\s*[^>]*>\s*\(\s*\)")
_TILE_ONES_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_ones\s*<\s*(?:wp::)?(\w+)\s*[^>]*>\s*\(\s*\)")
# ``wp::tile_arange<int>(N)`` with N=1 → ``var_X = 0``.
_TILE_ARANGE_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_arange\s*<[^>]*>\s*\(\s*\d+\s*\)")
# ``wp::tile_arange<dtype, N>(start, stop, step)`` — multi-element form
# used by mujoco_warp's dense-Jacobian path. Lower to a runtime loop
# that fills ``var_X.c[i] = start + i * step``.
_TILE_ARANGE_3ARG_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_arange\s*<[^>]*>\s*\(([^)]*)\)")
# ``wp::tile_map<fn>(args...)`` — element-wise map. With block_dim=1 each
# tile is a scalar; reduces to a direct call to the target function. The
# template arg holds the function name (e.g. ``wp_mul`` or a user fn).
_TILE_MAP_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_map\s*<\s*([^,>]+?)\s*>\s*\(([^)]*)\)")
# ``wp::tile_map(fn, args...)`` — no-template form used by the dense-
# Jacobian path. ``fn`` is a user @wp.func or a wp builtin, and args
# can mix tiles (scalar- or vec-element) with broadcast scalars. We
# lower this by emitting an explicit element loop. Distinct from the
# templated form above.
_TILE_MAP_NOTPL_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_map\s*\(([^)]*)\)")
# ``wp::tile_binary_map(fn, a, b)`` — same as tile_map but explicitly
# binary; mujoco_warp's dense-Jacobian path emits this for ``wp::sub``
# / ``wp::add`` / ``wp::mul`` over two tiles. Lower identically.
_TILE_BINARY_MAP_NOTPL_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_binary_map\s*\(([^)]*)\)")
# ``wp::tile_<op>(args...)`` — explicit-op tile builtins like
# ``tile_mul`` / ``tile_add`` / ``tile_sub`` / ``tile_div``. Treated
# the same as ``tile_binary_map(wp::<op>, args...)``.
_TILE_OP_NOTPL_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_(add|sub|mul|div)\s*\(([^)]*)\)")
# In-place compound forms ``wp::tile_add_inplace(dst, src)`` etc. —
# accumulates ``src`` into ``dst`` element-wise.
_TILE_OP_INPLACE_PAT = re.compile(r"\bwp::tile_(add|sub|mul|div)_inplace\s*\(([^)]*)\)")
# ``wp::tile_sort(keys, values)`` — cooperative key/value sort, in place,
# ascending by key. mujoco_warp's ``segmented_sort`` (broadphase contact
# sort) and ``contact_sensor_sort`` use this. With our default
# ``block_dim=1`` the cooperative version reduces to a single-thread
# insertion sort over the tile elements.
_TILE_SORT_PAT = re.compile(r"\bwp::tile_sort\s*\(\s*var_(\w+)\s*,\s*var_(\w+)\s*\)")
# ``var_X = wp::tile_diag_add(M_tile, diag_vec, out_tile)`` — adds
# ``diag_vec`` to the diagonal of ``M_tile``, returning the result.
# The mujoco_warp dense-Euler kernel uses this for ``qM + dt*damping``.
_TILE_DIAG_ADD_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_diag_add\s*\(([^)]*)\)")
# ``wp::tile_extract(tile, idx)`` — read the i-th element of a tile. For a
# 1-element tile the only valid index is 0 and the result is the scalar
# value itself.
_TILE_EXTRACT_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_extract\s*\(\s*var_(\w+)\s*,\s*[^)]+?\s*\)")
# ``wp::tile_assign(dst, src, offset_tuple)`` — copy ``src`` into ``dst``
# at ``offset``. With single-element tiles ``offset=(0,0)`` and the call
# is just ``dst = src``.
_TILE_ASSIGN_PAT = re.compile(r"\bwp::tile_assign\s*\(([^)]*)\)")
# ``wp::tile_transpose<...>(t)`` — at shape (1,1) the templated form is
# a no-op. The non-templated form ``wp::tile_transpose(t)`` (emitted by
# the blocked-Cholesky factory funcs) needs the input tile's actual
# shape to materialise a transposed copy; it's lowered separately via
# ``repl_transpose_notpl`` using ``tile_var_dims``.
_TILE_TRANSPOSE_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_transpose\s*<[^()]*>\s*\(\s*var_(\w+)\s*\)")
_TILE_TRANSPOSE_NOTPL_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_transpose\s*\(\s*var_(\w+)\s*\)")
# ``wp::tile_matmul_acc(0, 0, 0, A, B, C, alpha, beta)`` — eight args
# from the cuBLASDx-LTO calling convention. Three leading ints are
# placeholder LTO ptr/seg args, then A (RxK), B (KxN), C (RxN), then
# scalar alpha and beta. ``C = beta*C + alpha*A*B``.
_TILE_MATMUL_ACC_PAT = re.compile(r"\bwp::tile_matmul_acc\s*\(([^)]*)\)")
# Value-returning ``wp::tile_matmul`` form (no ``_acc`` suffix)
# emitted by mujoco_warp's dense JTDAJ kernel: ``var_C = wp::tile_matmul(
# 0, 0, 0, var_A, var_B, var_C, var_alpha, var_beta)``. Same 8-arg
# semantics as the in-place form — C is mutated AND returned.
_TILE_MATMUL_ASSIGN_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_matmul\s*\(([^)]*)\)")
# ``wp::tile_cholesky_inplace<upper>(0, var_X)`` — single tile, in-place
# factorization. Two args after the LTO seg pad: the placeholder and the
# tile to factor. ``upper`` template arg selects upper- vs lower-fill.
_TILE_CHOLESKY_INPLACE_PAT = re.compile(r"\bwp::tile_cholesky_inplace\s*<[^()]*>\s*\(([^)]*)\)")
# ``tile_lower_solve_inplace(0, L_tile, B_tile)`` — solve ``L X = B``
# for X, mutating B. The CPU/no-MathDx dispatch returns
# ``(0, L, y)`` with empty templates, so the IR emits this builtin
# *without* the ``wp::`` prefix and without ``<...>`` template args
# (unlike ``tile_cholesky_inplace`` which keeps a ``<upper>`` flag).
# Match the bare form. Same shape for the upper variant.
_TILE_LOWER_SOLVE_INPLACE_PAT = re.compile(r"\btile_lower_solve_inplace\s*\(([^)]*)\)")
_TILE_UPPER_SOLVE_INPLACE_PAT = re.compile(r"\btile_upper_solve_inplace\s*\(([^)]*)\)")
# ``var_X = wp::tile_view<wp::tile_shared_t<dtype, layout<shape<R,C>,
# stride<...>>, ...>>(parent, row_off, col_off)``. The template arg
# carries the view's output shape; the function args are the parent
# tile and offsets. We materialise the view as a copy via tile_load
# at construction time and then emit a writeback after every mutating
# op (tile_matmul_acc target, tile_*_solve_inplace second arg).
_TILE_VIEW_PAT = re.compile(
    # Match ``var_X = wp::tile_view<wp::tile_shared_t<wp::SCALAR, ...
    # wp::tile_shape_t<R, C>, ...>>(parent, row_off, col_off)`` with a
    # lenient body for the nested template noise. Captures: lhs label,
    # scalar, R, C (optional), parent label, row_off, col_off (optional).
    r"\bvar_(\w+)\s*=\s*wp::tile_view\s*<\s*wp::tile_shared_t\s*<\s*"
    r"wp::(\w+)\s*,.*?wp::tile_shape_t\s*<\s*(\d+)\s*"
    r"(?:,\s*(\d+)\s*)?>.*?>\s*>\s*\(\s*var_(\w+)\s*,"
    r"\s*([^,)]+)\s*(?:,\s*([^,)]+)\s*)?\)"
)
# ``wp::tile_broadcast<dtype, ...>(t)`` — broadcasting a single value
# back to a single value is a copy.
_TILE_BROADCAST_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_broadcast\s*<[^()]*>\s*\(\s*var_(\w+)\s*\)")
# ``wp::tile_matmul(a, b, out)`` — at shape (1,1)x(1,1) this is just a
# scalar multiply; not yet emitted because the failing kernels we cover
# don't reach this path.

_TILE_LOAD_PAT = re.compile(
    # Two template-arg shapes Warp emits:
    #   - ``<dtype, layout, transpose, R[, C]>``      (the original 4/5 form)
    #   - ``<dtype, shared, bounds_check, R, C>``     (the blocked-Cholesky
    #     form — ``shared`` selects threadgroup-memory storage,
    #     ``bounds_check`` toggles edge guards). On single-thread Metal we
    #     ignore both flags: the per-thread struct buffer doesn't need
    #     bounds checks (we're loading a fixed (R, C) into a fixed-sized
    #     local), and "shared" storage degenerates to per-thread for
    #     ``block_dim=1``.
    # ``dtype`` can be a scalar (``wp::float32``), a vec-type
    # (``wp::vec_t<N, wp::SCALAR>``), or — after the inliner has
    # already rewritten — a big-vec struct name (``wp_vec6_float``).
    r"\bvar_(\w+)\s*=\s*wp::tile_load\s*<\s*"
    r"(wp::vec_t<\d+,\s*wp::\w+>|wp::\w+|wp_vec\d+_\w+|wp_mat\d+x\d+_\w+)"
    r"\s*,\s*\w+\s*,\s*\w+\s*,\s*(\d+)\s*(?:,\s*(\d+)\s*)?>\s*\(([^)]*)\)"
)
_TILE_STORE_PAT = re.compile(
    r"\bwp::tile_store\s*<\s*"
    r"(wp::vec_t<\d+,\s*wp::\w+>|wp::\w+|wp_vec\d+_\w+|wp_mat\d+x\d+_\w+)"
    r"\s*,\s*\w+\s*,\s*\w+\s*>\s*\(([^)]*)\)"
)
_TILE_CHOLESKY_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_cholesky\s*<[^()]*>\s*\(([^)]*)\)")
_TILE_CHOLESKY_SOLVE_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_cholesky_solve\s*<[^()]*>\s*\(([^)]*)\)")
# ``wp::tile_cholesky_solve_inplace<upper>(0, var_L, var_b);`` — solves
# ``L L^T x = b`` and mutates ``b`` to hold ``x``. Three args after the
# template arg: the LTO seg pad, the factor tile, and the RHS. Lowers
# to a self-assigning call into the existing non-inplace helper —
# ``var_b = helper(var_L, var_b)``.
_TILE_CHOLESKY_SOLVE_INPLACE_PAT = re.compile(r"\bwp::tile_cholesky_solve_inplace\s*<[^()]*>\s*\(([^)]*)\)")


def _build_flat_base_expr(arr_name: str, lead_idx_args: list[str], inner_dims: int) -> str:
    """Compute the flat base offset of ``arr[lead_0, lead_1, ..., 0, 0, ...]``.

    ``inner_dims`` is the rank of the tile (1 for vector RHS, 2 for matrix
    factor); the array's rank is ``len(lead_idx_args) + inner_dims``.
    Returns a textual expression in terms of ``var_*`` and ``<arr>_shape[k]``
    that the downstream ``__shapes_packed`` packer rewrites to the per-arg
    packed slot.
    """
    if not lead_idx_args:
        return "0"
    arr_ndim = len(lead_idx_args) + inner_dims
    # Multiply each lead index by the product of all *trailing* dim sizes.
    terms: list[str] = []
    for i, idx in enumerate(lead_idx_args):
        stride_dims = list(range(i + 1, arr_ndim))
        if not stride_dims:
            terms.append(idx)
        else:
            stride = " * ".join(f"{arr_name}_shape[{j}]" for j in stride_dims)
            terms.append(f"{idx} * {stride}")
    return " + ".join(terms)


def _emit_tile_writeback(
    parent_label: str,
    parent_rows: int,
    parent_cols: int,
    view_rows: int,
    view_cols: int,
    msl_scalar: str,
    row_off_expr: str,
    col_off_expr: str,
    src_var_expr: str,
) -> str:
    """Emit MSL that copies a view's mutated contents back to its parent
    tile struct's ``c[]`` buffer.

    Used after every mutating call on a view (matmul accumulator,
    *_solve_inplace's RHS) so writes propagate.
    """
    parts: list[str] = []
    parts.append("    do {")
    parts.append(f"        for (int _vb_i = 0; _vb_i < {view_rows}; ++_vb_i) {{")
    parts.append(f"            for (int _vb_j = 0; _vb_j < {view_cols}; ++_vb_j) {{")
    parts.append(
        f"                var_{parent_label}.c[(({row_off_expr}) + _vb_i) * {parent_cols} + "
        f"(({col_off_expr}) + _vb_j)] = {src_var_expr}.c[_vb_i * {view_cols} + _vb_j];"
    )
    parts.append("            }")
    parts.append("        }")
    parts.append("    } while (0)")
    return "\n".join(parts)


def _translate_tile_intrinsics(
    line: str,
    tile_var_dims: dict[str, tuple[int, int, str]],
    view_aliases: dict[str, tuple[str, int, int, int, int, str, str, str]] | None = None,
    coop_chol_seen: set[tuple[int, str]] | None = None,
    is_coop_kernel: bool = False,
    transpose_aliases: dict[str, str] | None = None,
    tile_var_vec_n: dict[str, int] | None = None,
    atomic_output_names: set[str] | None = None,
) -> str:
    if tile_var_vec_n is None:
        tile_var_vec_n = {}
    if atomic_output_names is None:
        atomic_output_names = set()
    """Lower ``wp::tile_*`` calls to ``wp_tile_RxC_<scalar>_*`` helper calls.

    ``tile_var_dims`` maps each tile local label to ``(rows, cols, msl_scalar)``
    so we can pick the right helper for cholesky / cholesky_solve (whose
    dimensions are inferred from their result tile's type rather than
    template args).
    """

    def repl_load(m: re.Match[str]) -> str:
        lhs = m.group(1)
        dtype_token = m.group(2)
        rows = int(m.group(3))
        cols = int(m.group(4)) if m.group(4) else 1
        # Resolve the element type. Three cases:
        #  - scalar: ``wp::float32`` -> msl_scalar=``float``
        #  - vec_t (Warp form): ``wp::vec_t<N, wp::SCALAR>``
        #  - big-vec alias: ``wp_vec<N>_<scalar>`` (post-inliner form)
        msl_scalar = None
        vec_n_elem = 0
        m_vec_t = re.match(r"wp::vec_t<(\d+),\s*wp::(\w+)>", dtype_token)
        if m_vec_t is not None:
            vec_n_elem = int(m_vec_t.group(1))
            inner_ctype = f"wp::{m_vec_t.group(2)}"
            msl_scalar = _SCALAR_CTYPE_TO_MSL.get(inner_ctype)
        elif dtype_token.startswith("wp_vec"):
            mvec = re.match(r"wp_vec(\d+)_(\w+)", dtype_token)
            if mvec is not None:
                vec_n_elem = int(mvec.group(1))
                msl_scalar = mvec.group(2)
        elif dtype_token.startswith("wp::"):
            msl_scalar = _SCALAR_CTYPE_TO_MSL.get(dtype_token)
        if msl_scalar is None:
            return m.group(0)
        args = [a.strip() for a in m.group(5).split(",")]
        # First arg is the array; remaining are leading slice indices then
        # tile offsets. For a 2-D tile we have 2 trailing offsets; for a
        # 1-D tile, 1 trailing offset.
        arr = args[0]
        n_off = 2 if cols > 1 else 1
        lead_idx_args = args[1:-n_off] if len(args) > 1 + n_off else []
        offsets = args[-n_off:]
        arr_name = arr[len("var_") :] if arr.startswith("var_") else arr
        # Vec-element arrays expose an extra inner scalar dim in MLX
        # (``(*shape, n_elem)``); add it to ``inner_dims`` so the base
        # expression includes the element-size stride.
        scalar_inner_dims = 2 if cols > 1 else 1
        if vec_n_elem > 0:
            scalar_inner_dims += 1
        base_expr = _build_flat_base_expr(arr_name, lead_idx_args, inner_dims=scalar_inner_dims)
        if cols > 1:
            row_stride = f"{arr_name}_shape[{len(lead_idx_args) + 1}]"
            row_off, col_off = offsets
        else:
            row_stride = "1"
            # Treat the 1-D tile load as a degenerate ``Rx1`` load with
            # ``col_off=0`` so the same helper signature works.
            row_off, col_off = offsets[0], "0"
        # Shape (1,1) collapses to a scalar local (see ``_msl_var_type``);
        # there's no wp_tile_1x1 struct/helper, so emit a direct subscript.
        if rows == 1 and cols == 1:
            return f"var_{lhs} = {arr}[{base_expr} + ({row_off}) * ({row_stride}) + ({col_off})]"
        # Vec-element tile: route to the vec-element helper. The
        # helper folds the ``n_elem`` scaling into its inner index, so
        # ``row_stride`` here is in ELEMENTS (1 for a 1-D tile of
        # ``vec_t<n_elem, scalar>`` over a 1-D backing array).
        if vec_n_elem > 0:
            helper = f"wp_tile_{rows}x{cols}_vec{vec_n_elem}_{msl_scalar}_load"
            return f"var_{lhs} = {helper}({arr}, {base_expr}, {row_stride}, {row_off}, {col_off})"
        # NB: cooperative tile_load was tried but a SIMD-cooperative
        # threadgroup-memory round-trip (write smem → barrier → read
        # smem into each thread's private struct) ran *slower* than
        # the single-thread emit at the sizes we ship. The 32-lane
        # redundant device reads land in L1 (the 16 KB at N=64 fits),
        # so the barriers + smem traffic of cooperation is pure
        # overhead. Keeping single-thread emit even in coop kernels.
        helper = f"wp_tile_{rows}x{cols}_{msl_scalar}_load"
        return f"var_{lhs} = {helper}({arr}, {base_expr}, {row_stride}, {row_off}, {col_off})"

    def repl_store(m: re.Match[str]) -> str:
        # Group 1 is the full dtype string after the regex update —
        # could be ``wp::SCALAR``, ``wp::vec_t<N, wp::SCALAR>``, or a
        # post-inliner alias (``wp_vec<N>_<scalar>``).
        dtype_token = m.group(1)
        msl_scalar = None
        vec_n_elem = 0
        m_vec_t = re.match(r"wp::vec_t<(\d+),\s*wp::(\w+)>", dtype_token)
        if m_vec_t is not None:
            vec_n_elem = int(m_vec_t.group(1))
            inner_ctype = f"wp::{m_vec_t.group(2)}"
            msl_scalar = _SCALAR_CTYPE_TO_MSL.get(inner_ctype)
        elif dtype_token.startswith("wp_vec"):
            mvec = re.match(r"wp_vec(\d+)_(\w+)", dtype_token)
            if mvec is not None:
                vec_n_elem = int(mvec.group(1))
                msl_scalar = mvec.group(2)
        elif dtype_token.startswith("wp::"):
            msl_scalar = _SCALAR_CTYPE_TO_MSL.get(dtype_token)
        if msl_scalar is None:
            return m.group(0)
        args = [a.strip() for a in m.group(2).split(",")]
        # Last arg is the source tile, first is the destination array,
        # everything between is leading-slice indices + tile offsets.
        arr = args[0]
        tile_var = args[-1]
        tile_label = tile_var[len("var_") :] if tile_var.startswith("var_") else tile_var
        dims = tile_var_dims.get(tile_label)
        if dims is None:
            return m.group(0)
        rows, cols, _ = dims
        n_off = 2 if cols > 1 else 1
        middle = args[1:-1]  # leading-idx + offsets
        lead_idx_args = middle[:-n_off] if len(middle) > n_off else []
        offsets = middle[-n_off:]
        arr_name = arr[len("var_") :] if arr.startswith("var_") else arr
        scalar_inner_dims = 2 if cols > 1 else 1
        if vec_n_elem > 0:
            scalar_inner_dims += 1
        base_expr = _build_flat_base_expr(arr_name, lead_idx_args, inner_dims=scalar_inner_dims)
        if cols > 1:
            row_stride = f"{arr_name}_shape[{len(lead_idx_args) + 1}]"
            row_off, col_off = offsets
        else:
            row_stride = "1"
            row_off, col_off = offsets[0], "0"
        # Shape (1,1) tile is a scalar local — store with a direct subscript.
        if rows == 1 and cols == 1:
            return f"{arr}[{base_expr} + ({row_off}) * ({row_stride}) + ({col_off})] = {tile_var}"
        if vec_n_elem > 0:
            helper = f"wp_tile_{rows}x{cols}_vec{vec_n_elem}_{msl_scalar}_store"
            return f"{helper}({arr}, {base_expr}, {row_stride}, {row_off}, {col_off}, {tile_var})"
        # If the target output is atomic-typed (the kernel uses
        # ``wp.atomic_*`` somewhere), the helper signature
        # ``device <scalar>*`` won't accept ``device atomic<scalar>*``.
        # Emit an inline loop with ``atomic_store_explicit`` per
        # element instead.
        arr_name = arr[len("var_") :] if arr.startswith("var_") else arr
        if arr_name in atomic_output_names:
            n = rows * cols
            store = (
                f"{{ for (int _ts_i = 0; _ts_i < {rows}; ++_ts_i) "
                f"for (int _ts_j = 0; _ts_j < {cols}; ++_ts_j) "
                f"atomic_store_explicit("
                f"&{arr}[{base_expr} + (({row_off}) + _ts_i) * ({row_stride}) + (({col_off}) + _ts_j)], "
                f"{tile_var}.c[_ts_i * {cols} + _ts_j], memory_order_relaxed); }}"
            )
            return store
        # NB: cooperative tile_store (each lane writes its strided
        # slice with no smem) was tried but added latency without
        # measurable bandwidth savings — Metal's SIMD-group write
        # coalescing already collapses 32 redundant identical writes
        # to the same address into a single transaction. Keep single-
        # thread emit; revisit when we add larger-tile workloads where
        # write coalescing breaks down.
        helper = f"wp_tile_{rows}x{cols}_{msl_scalar}_store"
        return f"{helper}({arr}, {base_expr}, {row_stride}, {row_off}, {col_off}, {tile_var})"

    def repl_cholesky(m: re.Match[str]) -> str:
        lhs = m.group(1)
        args = [a.strip() for a in m.group(2).split(",")]
        # Drop the cuBLASDx-LTO leading three padding args.
        if len(args) < 5:
            return m.group(0)
        in_arg = args[3]
        in_label = in_arg[len("var_") :] if in_arg.startswith("var_") else in_arg
        dims = tile_var_dims.get(in_label) or tile_var_dims.get(lhs)
        if dims is None:
            return m.group(0)
        rows, cols, msl_scalar = dims
        if rows != cols:
            return m.group(0)
        # Shape (1,1) collapses to a scalar local: the factor is just
        # sqrt(max(a, eps)). The clamp matches the wp_tile_NxN cholesky
        # helper's near-singular guard.
        if rows == 1:
            return f"var_{lhs} = metal::precise::sqrt(metal::max({in_arg}, ({msl_scalar})1e-30))"
        # Above the cooperative threshold (and within Apple Silicon's
        # threadgroup-memory cap), route through the SIMD-cooperative
        # variant: 32 lanes share one ``wp_tile_chol_smem`` scratch
        # in threadgroup memory, factor cooperatively, then read the
        # result back into each thread's private struct.
        if coop_chol_seen is not None and _COOP_CHOL_MIN_N <= rows <= _COOP_CHOL_MAX_N:
            coop_chol_seen.add((rows, msl_scalar))
            helper = f"wp_tile_{rows}x{cols}_{msl_scalar}_cholesky_coop"
            return f"var_{lhs} = {helper}({in_arg}, wp_tile_chol_smem, _coop_lane)"
        helper = f"wp_tile_{rows}x{cols}_{msl_scalar}_cholesky"
        return f"var_{lhs} = {helper}({in_arg})"

    def repl_cholesky_solve(m: re.Match[str]) -> str:
        lhs = m.group(1)
        args = [a.strip() for a in m.group(2).split(",")]
        # Args layout for ``out = solve(L, b)`` (non-inplace): one leading
        # padding zero, then ``L`` (square N×N), ``b`` (N×K), and ``out``
        # (N×K — same local as the LHS).
        if len(args) < 4:
            return m.group(0)
        L_arg = args[1]
        b_arg = args[2]
        L_label = L_arg[len("var_") :] if L_arg.startswith("var_") else L_arg
        b_label = b_arg[len("var_") :] if b_arg.startswith("var_") else b_arg
        L_dims = tile_var_dims.get(L_label)
        b_dims = tile_var_dims.get(b_label) or tile_var_dims.get(lhs)
        if L_dims is None or b_dims is None:
            return m.group(0)
        n = L_dims[0]
        k = b_dims[1]
        msl_scalar = L_dims[2]
        # Shape (1,1) L with K=1 b collapses to scalar division: solving
        # L*L^T*x = b with scalar L is x = b / (L*L).
        if n == 1 and k == 1:
            return f"var_{lhs} = {b_arg} / ({L_arg} * {L_arg})"
        helper = f"wp_tile_{n}x{n}_{msl_scalar}_cholesky_solve_{k}"
        return f"var_{lhs} = {helper}({L_arg}, {b_arg})"

    # ``wp::tuple(a, b, ...)`` — Warp emits tuple constructions when a
    # value-returning builtin "returns" multiple values (e.g. ``range``)
    # or when shapes/offsets are passed as composite args to tile
    # primitives. In the blocked-cholesky pattern the tuples are
    # created but never read — the consumer (``tile_load``,
    # ``tile_view``) takes the scalar args directly. We elide the
    # tuple-construction line by lowering to an empty no-op statement;
    # the var declaration emitted earlier just goes unused. If an
    # actual use survives downstream, it will trip the unsupported-
    # intrinsic guard with the still-prefixed ``wp::`` name.
    line = re.sub(r"\bvar_(\w+)\s*=\s*wp::tuple\s*\([^)]*\)\s*;?", r"// (elided tuple ctor for var_\1)", line)
    # Block-dim=1 register-tile reductions to plain scalar / vec values.
    # ``var_X = wp::tile<dtype>(var_x)``  →  ``var_X = var_x``
    line = _TILE_BUILTIN_TILE_PAT.sub(r"var_\1 = var_\2", line)

    def repl_tile_reduce(m: re.Match[str]) -> str:
        # ``var_X = wp::tile_reduce(op_fn, var_t)``. For a multi-element
        # tile we accumulate across all elements; for a single-element
        # tile the reduce is identity (the existing assign was correct).
        lhs = m.group(1)
        # Recover the op name from the original line text — m.re's
        # groups don't capture it because the pattern uses ``[\w:]+``
        # without parens. Re-parse from the raw match.
        full = m.group(0)
        # Match ``tile_reduce(op_fn, var_t)`` to pull op_fn out.
        op_match = re.search(r"wp::tile_reduce\s*\(\s*([\w:]+)\s*,\s*var_\w+\s*\)", full)
        op_fn = op_match.group(1) if op_match else "wp::add"
        in_label = m.group(2)
        dims = tile_var_dims.get(in_label)
        if dims is None:
            return f"var_{lhs} = var_{in_label}"
        rows, cols, _scalar = dims
        n = rows * cols
        if n == 1:
            return f"var_{lhs} = var_{in_label}"
        # Map known builtin op names to their accumulation expression.
        if op_fn == "wp::add":
            op_expr = "_tr_acc + var_{lhs}_in.c[_tr_i]"
        elif op_fn == "wp::max":
            op_expr = "metal::max(_tr_acc, var_{lhs}_in.c[_tr_i])"
        elif op_fn == "wp::min":
            op_expr = "metal::min(_tr_acc, var_{lhs}_in.c[_tr_i])"
        elif op_fn == "wp::mul":
            op_expr = "_tr_acc * var_{lhs}_in.c[_tr_i]"
        else:
            # Unknown op — leave the original (will trigger
            # unsupported-intrinsic error with the function name visible).
            return m.group(0)
        # Substitute ``var_{lhs}_in`` placeholder with the actual tile
        # var name; we don't actually need a separate name, but keeping
        # the placeholder readable.
        op_expr = op_expr.replace("{lhs}_in", in_label)
        return (
            f"var_{lhs} = ({{ "
            f"auto _tr_acc = var_{in_label}.c[0]; "
            f"for (int _tr_i = 1; _tr_i < {n}; ++_tr_i) _tr_acc = {op_expr}; "
            f"_tr_acc; }})"
        )

    line = _TILE_REDUCE_PAT.sub(repl_tile_reduce, line)

    def repl_zeros(m: re.Match[str]) -> str:
        return _emit_tile_const_fill(m.group(1), m.group(2), "0")

    def repl_ones(m: re.Match[str]) -> str:
        return _emit_tile_const_fill(m.group(1), m.group(2), "1")

    def _emit_tile_const_fill(lhs: str, scalar_token: str, value: str) -> str:
        # ``wp::tile_zeros`` / ``wp::tile_ones`` may target a single-
        # element tile (which collapses to a scalar local) or a multi-
        # element tile. For multi-element, fill all slots.
        scalar_ctype = scalar_token if scalar_token.startswith("wp::") else f"wp::{scalar_token}"
        msl = _SCALAR_CTYPE_TO_MSL.get(scalar_ctype, "float")
        dims = tile_var_dims.get(lhs)
        if dims is None:
            return f"var_{lhs} = ({msl}){value}"
        rows, cols, _ = dims
        n = rows * cols
        if n == 1:
            return f"var_{lhs} = ({msl}){value}"
        return f"{{ for (int _tc_i = 0; _tc_i < {n}; ++_tc_i) var_{lhs}.c[_tc_i] = ({msl}){value}; }}"

    line = _TILE_ZEROS_PAT.sub(repl_zeros, line)
    line = _TILE_ONES_PAT.sub(repl_ones, line)
    line = _TILE_ARANGE_PAT.sub(r"var_\1 = 0", line)

    def repl_tile_arange_3(m: re.Match[str]) -> str:
        # ``var_X = wp::tile_arange<dtype, N>(start, stop, step)``.
        # For multi-element tiles, fill ``var_X.c[i] = start + i*step``.
        lhs = m.group(1)
        args = [a.strip() for a in m.group(2).split(",")]
        if len(args) != 3:
            # 1-arg form already handled above.
            return m.group(0)
        start, _stop, step = args
        dims = tile_var_dims.get(lhs)
        if dims is None:
            return m.group(0)
        rows, cols, _ = dims
        n = rows * cols
        if n == 1:
            return f"var_{lhs} = ({start})"
        return f"{{ for (int _ar_i = 0; _ar_i < {n}; ++_ar_i) var_{lhs}.c[_ar_i] = ({start}) + _ar_i * ({step}); }}"

    line = _TILE_ARANGE_3ARG_PAT.sub(repl_tile_arange_3, line)

    def repl_map(m: re.Match[str]) -> str:
        # ``var_X = wp::tile_map<fn>(args...)`` — single-element tile case.
        # Just call ``fn(args...)``. ``fn`` may be a builtin (``wp_mul``,
        # ``wp_add``) or a user function — both expand to direct MSL.
        lhs = m.group(1)
        fn = m.group(2).strip()
        # Drop ``wp::`` prefix and special-case the common ones to
        # operators (matches what the regular ``wp::add`` / ``wp::mul``
        # patterns elsewhere produce).
        if fn.startswith("wp::"):
            op = fn[len("wp::") :]
            args = m.group(3)
            if op in ("add", "sub", "mul", "div"):
                op_sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
                a, b = (a.strip() for a in args.split(",", 1))
                return f"var_{lhs} = ({a} {op_sym} {b})"
            return f"var_{lhs} = {fn}({args})"
        # User function — call directly. Strips templating wrapper.
        return f"var_{lhs} = {fn}({m.group(3)})"

    line = _TILE_MAP_PAT.sub(repl_map, line)

    def repl_tile_map_notpl(m: re.Match[str]) -> str:
        # ``var_X = wp::tile_map(fn, arg1, arg2, ..., argN)`` — no-template
        # form (mujoco_warp's dense-Jacobian path uses this with
        # ``_compute_jacp`` / ``wp.dot`` / ``wp.add`` / etc.). Args can mix
        # tiles (scalar- or vec-element) with broadcast scalars. We
        # lower to an explicit element loop, extracting the per-element
        # values from each tile arg and broadcasting scalars verbatim.
        lhs = m.group(1)
        inside = m.group(2)
        args = [a.strip() for a in inside.split(",")]
        if len(args) < 2:
            return m.group(0)
        fn = args[0]
        fn_args = args[1:]

        # Determine the tile shape from the first tile-typed arg.
        def _tile_info(a: str) -> tuple[tuple[int, int, str], int] | None:
            label = a[len("var_") :] if a.startswith("var_") else a
            d = tile_var_dims.get(label)
            if d is None:
                return None
            return d, tile_var_vec_n.get(label, 0)

        tile_dims = None
        for a in fn_args:
            info = _tile_info(a)
            if info is not None:
                tile_dims = info[0]
                break
        if tile_dims is None:
            # No tile args at all — degenerate; fall through to a plain call.
            return f"var_{lhs} = {fn}({', '.join(fn_args)})"
        rows, cols, _scalar = tile_dims
        n_elements = rows * cols

        # Special case: single-element tile collapses to a plain call.
        # ``tile_map(fn, t1)`` where t1 is 1x1 acts like ``fn(t1)``.
        if n_elements == 1:
            return f"var_{lhs} = {fn}({', '.join(fn_args)})"

        # Result tile shape/element type.
        lhs_dims = tile_var_dims.get(lhs)
        lhs_vec_n = tile_var_vec_n.get(lhs, 0)

        # Build the per-arg per-element accessor expressions.
        per_arg_access: list[str] = []
        for a in fn_args:
            info = _tile_info(a)
            if info is None:
                # Scalar broadcast — use the value verbatim.
                per_arg_access.append(a)
                continue
            (_r, _c, a_scalar), a_vec_n = info
            if a_vec_n > 0:
                # Vec-element tile: build vec from ``a_vec_n`` consecutive
                # scalars at offset ``i * a_vec_n``.
                comps = [f"{a}.c[_tm_i * {a_vec_n} + {k}]" for k in range(a_vec_n)]
                if a_vec_n in _MSL_VEC_NATIVE_N:
                    per_arg_access.append(f"{a_scalar}{a_vec_n}({', '.join(comps)})")
                else:
                    per_arg_access.append(f"wp_vec{a_vec_n}_{a_scalar}_make({', '.join(comps)})")
            else:
                # Scalar-element tile: read element ``i`` directly.
                per_arg_access.append(f"{a}.c[_tm_i]")

        # Translate common ``wp::`` builtins to MSL form. The
        # intrinsic-translation pass runs once before tile_map
        # lowering, so the ``wp::sub(...)`` etc. we'd emit otherwise
        # would be left as unsupported.
        if fn.startswith("wp::"):
            op = fn[len("wp::") :]
            arith_ops = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
            if op in arith_ops and len(per_arg_access) == 2:
                call = f"({per_arg_access[0]} {arith_ops[op]} {per_arg_access[1]})"
            elif op == "dot" and len(per_arg_access) == 2:
                call = f"metal::dot({per_arg_access[0]}, {per_arg_access[1]})"
            else:
                call = f"{fn}({', '.join(per_arg_access)})"
        else:
            call = f"{fn}({', '.join(per_arg_access)})"

        # Store result: scalar element vs vec element.
        body: list[str] = ["do {", f"for (int _tm_i = 0; _tm_i < {n_elements}; ++_tm_i) {{"]
        if lhs_vec_n > 0:
            # Vec result: store ``lhs_vec_n`` components per element.
            # ``auto`` keeps us agnostic of the function's exact return type.
            body.append(f"auto _tm_r = {call};")
            for k in range(lhs_vec_n):
                body.append(f"var_{lhs}.c[_tm_i * {lhs_vec_n} + {k}] = _tm_r[{k}];")
        else:
            body.append(f"var_{lhs}.c[_tm_i] = {call};")
        body.append("}")
        body.append("} while(0)")
        return f"var_{lhs}; " + " ".join(body)

    line = _TILE_MAP_NOTPL_PAT.sub(repl_tile_map_notpl, line)
    line = _TILE_BINARY_MAP_NOTPL_PAT.sub(repl_tile_map_notpl, line)

    def _repl_tile_op(m: re.Match[str]) -> str:
        # Rewrite ``wp::tile_<op>(args)`` to the binary_map form so
        # ``repl_tile_map_notpl`` produces the element loop.
        synth = f"var_{m.group(1)} = wp::tile_binary_map(wp::{m.group(2)}, {m.group(3)})"
        return _TILE_BINARY_MAP_NOTPL_PAT.sub(repl_tile_map_notpl, synth)

    line = _TILE_OP_NOTPL_PAT.sub(_repl_tile_op, line)

    def _repl_tile_op_inplace(m: re.Match[str]) -> str:
        # ``wp::tile_<op>_inplace(dst, src)`` — accumulate ``src`` into
        # ``dst`` element-wise (or ``dst /= src`` for ``div``).
        op = m.group(1)
        args = [a.strip() for a in m.group(2).split(",")]
        if len(args) != 2:
            return m.group(0)
        dst, src = args
        op_sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
        dst_label = dst[len("var_") :] if dst.startswith("var_") else dst
        src_label = src[len("var_") :] if src.startswith("var_") else src
        dst_dims = tile_var_dims.get(dst_label)
        src_dims = tile_var_dims.get(src_label)
        if dst_dims is None:
            return m.group(0)
        rows, cols, _ = dst_dims
        n = rows * cols
        if n == 1:
            return f"{dst} = ({dst} {op_sym} {src})"
        if src_dims is None:
            # Scalar broadcast.
            return f"{{ for (int _ti_i = 0; _ti_i < {n}; ++_ti_i) {dst}.c[_ti_i] = {dst}.c[_ti_i] {op_sym} ({src}); }}"
        return (
            f"{{ for (int _ti_i = 0; _ti_i < {n}; ++_ti_i) {dst}.c[_ti_i] = {dst}.c[_ti_i] {op_sym} {src}.c[_ti_i]; }}"
        )

    line = _TILE_OP_INPLACE_PAT.sub(_repl_tile_op_inplace, line)

    def _repl_tile_sort(m: re.Match[str]) -> str:
        # ``wp::tile_sort(keys, values)`` — sort ``values`` along with
        # ``keys`` in ascending key order, in-place. Block_dim=1 reduces
        # the cooperative sort to a single-thread insertion sort.
        keys_label = m.group(1)
        vals_label = m.group(2)
        keys_dims = tile_var_dims.get(keys_label)
        vals_dims = tile_var_dims.get(vals_label)
        if keys_dims is None or vals_dims is None:
            return m.group(0)
        rows, cols, _ = keys_dims
        n = rows * cols
        if n <= 1:
            return "/* tile_sort: trivial */ (void)0"
        return (
            f"{{ for (int _ts_i = 1; _ts_i < {n}; ++_ts_i) {{ "
            f"auto _ts_k = var_{keys_label}.c[_ts_i]; "
            f"auto _ts_v = var_{vals_label}.c[_ts_i]; "
            f"int _ts_j = _ts_i - 1; "
            f"while (_ts_j >= 0 && var_{keys_label}.c[_ts_j] > _ts_k) {{ "
            f"var_{keys_label}.c[_ts_j + 1] = var_{keys_label}.c[_ts_j]; "
            f"var_{vals_label}.c[_ts_j + 1] = var_{vals_label}.c[_ts_j]; "
            f"--_ts_j; "
            f"}} "
            f"var_{keys_label}.c[_ts_j + 1] = _ts_k; "
            f"var_{vals_label}.c[_ts_j + 1] = _ts_v; "
            f"}} }}"
        )

    line = _TILE_SORT_PAT.sub(_repl_tile_sort, line)

    def _repl_tile_diag_add(m: re.Match[str]) -> str:
        # ``var_X = wp::tile_diag_add(M_tile, diag_vec[, out_tile])``.
        # Result: copy M_tile into var_X, then add diag_vec to the
        # diagonal entries.
        lhs = m.group(1)
        args = [a.strip() for a in m.group(2).split(",")]
        if len(args) < 2:
            return m.group(0)
        m_arg = args[0]
        v_arg = args[1]
        m_label = m_arg[len("var_") :] if m_arg.startswith("var_") else m_arg
        m_dims = tile_var_dims.get(m_label)
        if m_dims is None:
            return m.group(0)
        rows, cols, _ = m_dims
        if rows != cols:
            return m.group(0)
        n = rows
        if n == 1:
            # 1x1 tile collapsed to scalar — diag_add is just ``M + v``.
            return f"var_{lhs} = ({m_arg} + {v_arg})"
        return (
            f"{{ for (int _da_i = 0; _da_i < {n}; ++_da_i) "
            f"for (int _da_j = 0; _da_j < {n}; ++_da_j) "
            f"var_{lhs}.c[_da_i * {n} + _da_j] = {m_arg}.c[_da_i * {n} + _da_j] "
            f"+ ((_da_i == _da_j) ? {v_arg}.c[_da_i] : 0.0f); }}"
        )

    line = _TILE_DIAG_ADD_PAT.sub(_repl_tile_diag_add, line)

    def repl_assign(m: re.Match[str]) -> str:
        # ``wp::tile_assign(dst, src, offset_tuple)``. With shape (1,1)
        # tiles the offset is always 0 and we drop it. The IR emits the
        # offset as a ``wp::tuple_t`` literal which is a separate token —
        # split on top-level commas and take the first two args.
        args = [a.strip() for a in m.group(1).split(",")]
        if len(args) < 2:
            return m.group(0)
        return f"{args[0]} = {args[1]}"

    line = _TILE_ASSIGN_PAT.sub(repl_assign, line)
    line = _TILE_TRANSPOSE_PAT.sub(r"var_\1 = var_\2", line)

    def repl_transpose_notpl(m: re.Match[str]) -> str:
        # ``var_X = wp::tile_transpose(var_Y)`` (no template arg) is
        # emitted by the blocked-Cholesky factory funcs. Lower it to
        # the helper ``wp_tile_RxC_<scalar>_transpose(var_Y)`` keyed on
        # the *input*'s shape; the helper returns a CxR tile.
        lhs = m.group(1)
        in_label = m.group(2)
        dims = tile_var_dims.get(in_label)
        if dims is None:
            return m.group(0)
        rows, cols, msl_scalar = dims
        # (1,1) collapses to a scalar — no transpose needed; the assign
        # is simply identity.
        if rows == 1 and cols == 1:
            return f"var_{lhs} = var_{in_label}"
        # On the CUDA/cuBLASDx path, ``tile_transpose`` returns a
        # *layout-only view* over the same shared-memory storage —
        # mutating the result also mutates the source (via swapped
        # strides). On Metal we materialize tiles in private structs,
        # so the helper returns a *copy*. To preserve write-through
        # semantics, register ``lhs`` as a transpose-alias of
        # ``in_label`` so mutating ops (``*_solve_inplace``,
        # ``tile_matmul_acc``'s C target) can emit a transpose-back
        # writeback into the source.
        if transpose_aliases is not None:
            transpose_aliases[lhs] = in_label
        helper = f"wp_tile_{rows}x{cols}_{msl_scalar}_transpose"
        return f"var_{lhs} = {helper}(var_{in_label})"

    line = _TILE_TRANSPOSE_NOTPL_PAT.sub(repl_transpose_notpl, line)

    def repl_matmul_acc(m: re.Match[str]) -> str:
        # ``wp::tile_matmul_acc(0, 0, 0, A, B, C, alpha, beta)``
        args = [a.strip() for a in m.group(1).split(",")]
        if len(args) < 8:
            return m.group(0)
        a_arg, b_arg, c_arg, alpha_arg, beta_arg = args[3], args[4], args[5], args[6], args[7]
        a_label = a_arg[len("var_") :] if a_arg.startswith("var_") else a_arg
        b_label = b_arg[len("var_") :] if b_arg.startswith("var_") else b_arg
        c_label = c_arg[len("var_") :] if c_arg.startswith("var_") else c_arg
        a_dims = tile_var_dims.get(a_label)
        b_dims = tile_var_dims.get(b_label)
        c_dims = tile_var_dims.get(c_label)
        if a_dims is None or b_dims is None or c_dims is None:
            return m.group(0)
        rA, kA, scalar = a_dims
        kB, nB, _ = b_dims
        rC, nC, _ = c_dims
        if kA != kB or rA != rC or nB != nC:
            return m.group(0)
        helper = f"wp_tile_matmul_{rA}x{kA}x{nB}_{scalar}"
        call = f"{helper}({a_arg}, {b_arg}, {c_arg}, {alpha_arg}, {beta_arg})"
        # If C is a view, write its mutated contents back to the parent.
        if view_aliases is not None and c_label in view_aliases:
            parent_label, prows, pcols, vrows, vcols, vscalar, row_off, col_off = view_aliases[c_label]
            wb = _emit_tile_writeback(parent_label, prows, pcols, vrows, vcols, vscalar, row_off, col_off, c_arg)
            return f"{call};\n{wb}"
        return call

    line = _TILE_MATMUL_ACC_PAT.sub(repl_matmul_acc, line)

    def _repl_matmul_assign(m: re.Match[str]) -> str:
        # ``var_X = wp::tile_matmul(0, 0, 0, A, B, C, alpha, beta)``.
        # Treat as the in-place ``matmul_acc`` form — C IS var_X (the
        # mutated accumulator). After the call, the value of ``C`` is
        # the accumulated tile; assignment to var_X is implicit since
        # the helper mutates C in-place. We emit the matmul call and
        # let downstream stages treat the result as already in place.
        # Synthesize the matmul_acc form and re-run that translator.
        synth = f"wp::tile_matmul_acc({m.group(2)})"
        return _TILE_MATMUL_ACC_PAT.sub(repl_matmul_acc, synth)

    line = _TILE_MATMUL_ASSIGN_PAT.sub(_repl_matmul_assign, line)

    def repl_cholesky_inplace(m: re.Match[str]) -> str:
        args = [a.strip() for a in m.group(1).split(",")]
        if len(args) < 2:
            return m.group(0)
        tile_arg = args[1]
        tile_label = tile_arg[len("var_") :] if tile_arg.startswith("var_") else tile_arg
        dims = tile_var_dims.get(tile_label)
        if dims is None:
            return m.group(0)
        rows, cols, scalar = dims
        if rows != cols:
            return m.group(0)
        if rows == 1:
            return f"{tile_arg} = metal::precise::sqrt(metal::max({tile_arg}, ({scalar})1e-30))"
        helper = f"wp_tile_{rows}x{cols}_{scalar}_cholesky_inplace"
        return f"{helper}({tile_arg})"

    line = _TILE_CHOLESKY_INPLACE_PAT.sub(repl_cholesky_inplace, line)

    def _repl_solve_inplace(kind: str, m: re.Match[str]) -> str:
        # Args: leading LTO seg, then L tile, then B tile.
        args = [a.strip() for a in m.group(1).split(",")]
        if len(args) < 3:
            return m.group(0)
        L_arg, B_arg = args[1], args[2]
        L_label = L_arg[len("var_") :] if L_arg.startswith("var_") else L_arg
        B_label = B_arg[len("var_") :] if B_arg.startswith("var_") else B_arg
        L_dims = tile_var_dims.get(L_label)
        B_dims = tile_var_dims.get(B_label)
        if L_dims is None or B_dims is None:
            return m.group(0)
        n = L_dims[0]
        k_cols = B_dims[1]
        scalar = L_dims[2]
        helper = f"wp_tile_{kind}_solve_{n}x{k_cols}_{scalar}_inplace"
        call = f"{helper}({L_arg}, {B_arg})"
        if view_aliases is not None and B_label in view_aliases:
            parent_label, prows, pcols, vrows, vcols, vscalar, row_off, col_off = view_aliases[B_label]
            wb = _emit_tile_writeback(parent_label, prows, pcols, vrows, vcols, vscalar, row_off, col_off, B_arg)
            return f"{call};\n{wb}"
        # If B was produced by ``tile_transpose``, the mutation needs to
        # propagate back to the source so its caller can ``tile_store``
        # the updated values. Rewrite the call to a *transposed* helper
        # that operates directly on the source tile — this skips the
        # temporary and the writeback. Emitting ``var_src = transpose(B)``
        # after the solve triggers a Metal-compiler miscompile when the
        # same solve+writeback pattern appears twice in one kernel
        # (mujoco_warp's blocked Cholesky i-loop): the second iteration's
        # writeback returns a stale struct and the rows silently reset to
        # zero.
        if transpose_aliases is not None and B_label in transpose_aliases:
            src_label = transpose_aliases[B_label]
            src_dims = tile_var_dims.get(src_label)
            if src_dims is not None:
                src_rows, src_cols, _src_scalar = src_dims
                helper_t = f"wp_tile_{kind}_solve_{n}x{k_cols}_{scalar}_inplace_transposed"
                return f"{helper_t}({L_arg}, var_{src_label})"
        return call

    line = _TILE_LOWER_SOLVE_INPLACE_PAT.sub(lambda m: _repl_solve_inplace("lower", m), line)
    line = _TILE_UPPER_SOLVE_INPLACE_PAT.sub(lambda m: _repl_solve_inplace("upper", m), line)

    def _repl_tile_broadcast(m: re.Match[str]) -> str:
        # ``var_X = wp::tile_broadcast<...>(var_Y)``. Source ``var_Y``
        # may have a different shape than target ``var_X`` — e.g.
        # broadcasting a (TILE_SIZE,) tile to (nv_pad, TILE_SIZE) by
        # repeating the source along the new outer dim. We emit an
        # element loop that fills each ``var_X`` slot with the
        # corresponding ``var_Y`` element using row-major indexing.
        lhs = m.group(1)
        in_label = m.group(2)
        lhs_dims = tile_var_dims.get(lhs)
        in_dims = tile_var_dims.get(in_label)
        # Same shape (or single-element source) — identity assign.
        if lhs_dims is None or in_dims is None or lhs_dims == in_dims:
            return f"var_{lhs} = var_{in_label}"
        l_rows, l_cols, _ = lhs_dims
        in_rows, in_cols, _ = in_dims
        l_n = l_rows * l_cols
        in_n = in_rows * in_cols
        if in_n == 1:
            # Single-element source broadcast across the target tile.
            return f"{{ for (int _bc_i = 0; _bc_i < {l_n}; ++_bc_i) var_{lhs}.c[_bc_i] = var_{in_label}.c[0]; }}"
        # Cases:
        #   (a) Source is 1-D (in_cols == 1), target is 2-D — repeat
        #       source along target's outer dim ``r``: var_X.c[r*l_cols
        #       + c] = var_Y.c[c].
        #   (b) Source is 2-D, target is 2-D, target rows == 1 — collapse.
        # We support (a); fall back to identity for other shapes.
        if in_cols == 1 and in_rows == l_cols:
            # Source shape (l_cols, 1) tiled along ``r``.
            return (
                f"{{ for (int _bc_r = 0; _bc_r < {l_rows}; ++_bc_r) "
                f"for (int _bc_c = 0; _bc_c < {l_cols}; ++_bc_c) "
                f"var_{lhs}.c[_bc_r * {l_cols} + _bc_c] = var_{in_label}.c[_bc_c]; }}"
            )
        if in_rows == 1 and in_cols == l_cols:
            # Source shape (1, l_cols).
            return (
                f"{{ for (int _bc_r = 0; _bc_r < {l_rows}; ++_bc_r) "
                f"for (int _bc_c = 0; _bc_c < {l_cols}; ++_bc_c) "
                f"var_{lhs}.c[_bc_r * {l_cols} + _bc_c] = var_{in_label}.c[_bc_c]; }}"
            )
        # Unrecognised — leave identity as a safe fallback.
        return f"var_{lhs} = var_{in_label}"

    line = _TILE_BROADCAST_PAT.sub(_repl_tile_broadcast, line)
    # ``var_X = wp::tile_extract(var_t, idx)`` with shape (1,) → ``var_X = var_t``.
    line = _TILE_EXTRACT_PAT.sub(r"var_\1 = var_\2", line)

    def repl_view(m: re.Match[str]) -> str:
        # ``var_X = wp::tile_view<wp::tile_shared_t<dtype, layout<shape<R,C>, ...>>, ...>>(parent, row_off, col_off)``
        # On single-thread Metal we materialise the view as a struct
        # copy via ``tile_load`` from the parent's ``c[]`` buffer at
        # the given offset. The accompanying ``view_aliases`` entry
        # lets matmul / *_solve_inplace translators emit a writeback
        # that copies the (possibly-mutated) view back to the parent
        # after the call. Cooperative parallelism (sharing memory with
        # the parent via threadgroup pointers) is task #19's next step.
        lhs = m.group(1)
        scalar_ctype = f"wp::{m.group(2)}"
        msl_scalar = _SCALAR_CTYPE_TO_MSL.get(scalar_ctype)
        if msl_scalar is None:
            return m.group(0)
        rows = int(m.group(3))
        cols = int(m.group(4)) if m.group(4) else 1
        parent_label = m.group(5)
        row_off = m.group(6).strip()
        col_off = (m.group(7) or "0").strip()
        parent_dims = tile_var_dims.get(parent_label)
        if parent_dims is None:
            return m.group(0)
        prows, pcols, _ = parent_dims
        if view_aliases is not None:
            view_aliases[lhs] = (parent_label, prows, pcols, rows, cols, msl_scalar, row_off, col_off)
        # Materialise as a struct copy using direct indexing into the
        # parent's ``c[]`` buffer. Avoids a per-shape helper since the
        # dimensions are already known here.
        result_lines: list[str] = []
        result_lines.append(f"wp_tile_{rows}x{cols}_{msl_scalar} var_{lhs}")
        result_lines.append(";")
        result_lines.append("    do {")
        result_lines.append(f"        for (int _vl_i = 0; _vl_i < {rows}; ++_vl_i) {{")
        result_lines.append(f"            for (int _vl_j = 0; _vl_j < {cols}; ++_vl_j) {{")
        result_lines.append(
            f"                var_{lhs}.c[_vl_i * {cols} + _vl_j] = "
            f"var_{parent_label}.c[(({row_off}) + _vl_i) * {pcols} + (({col_off}) + _vl_j)];"
        )
        result_lines.append("            }")
        result_lines.append("        }")
        result_lines.append("    } while (0)")
        return "".join(result_lines)

    line = _TILE_VIEW_PAT.sub(repl_view, line)

    line = _TILE_LOAD_PAT.sub(repl_load, line)
    line = _TILE_STORE_PAT.sub(repl_store, line)
    line = _TILE_CHOLESKY_PAT.sub(repl_cholesky, line)
    line = _TILE_CHOLESKY_SOLVE_PAT.sub(repl_cholesky_solve, line)

    def repl_cholesky_solve_inplace(m: re.Match[str]) -> str:
        # Args: leading LTO seg, L tile (NxN), b tile (NxK).
        args = [a.strip() for a in m.group(1).split(",")]
        if len(args) < 3:
            return m.group(0)
        L_arg, b_arg = args[1], args[2]
        L_label = L_arg[len("var_") :] if L_arg.startswith("var_") else L_arg
        b_label = b_arg[len("var_") :] if b_arg.startswith("var_") else b_arg
        L_dims = tile_var_dims.get(L_label)
        b_dims = tile_var_dims.get(b_label)
        if L_dims is None or b_dims is None:
            return m.group(0)
        n = L_dims[0]
        k_cols = b_dims[1]
        scalar = L_dims[2]
        if n == 1 and k_cols == 1:
            return f"{b_arg} = {b_arg} / ({L_arg} * {L_arg})"
        helper = f"wp_tile_{n}x{n}_{scalar}_cholesky_solve_{k_cols}"
        call = f"{b_arg} = {helper}({L_arg}, {b_arg})"
        if view_aliases is not None and b_label in view_aliases:
            parent_label, prows, pcols, vrows, vcols, vscalar, row_off, col_off = view_aliases[b_label]
            wb = _emit_tile_writeback(parent_label, prows, pcols, vrows, vcols, vscalar, row_off, col_off, b_arg)
            return f"{call};\n{wb}"
        return call

    line = _TILE_CHOLESKY_SOLVE_INPLACE_PAT.sub(repl_cholesky_solve_inplace, line)
    return line


# ---------------------------------------------------------------------------
# Hand-written MSL definitions for mujoco_warp ``@wp.func`` helpers
# that are referenced by ``wp::tile_map`` (which passes them by name —
# the AST inliner only walks DIRECT calls, so functions passed by
# reference don't get inlined and the body is left as an unresolved
# call). The proper fix is a full user-function-emit pipeline that
# mirrors ``generate_msl_kernel`` for ``@wp.func`` bodies; these
# hand-emitted entries are a stop-gap to unblock the dense-Jacobian
# path on Metal until that arrives.
# ---------------------------------------------------------------------------
_USER_FUNC_MSL_DEFS: dict[str, str] = {
    # mujoco_warp/_src/support.py:_compute_jacp
    "_compute_jacp_0": (
        "inline float3 _compute_jacp_0(wp_vec6_float cdof_clip, float3 offset, int affect) {\n"
        "    if (affect == 0) return float3(0.0f);\n"
        "    float3 cdof_lin = float3(cdof_clip.c[3], cdof_clip.c[4], cdof_clip.c[5]);\n"
        "    float3 cdof_ang = float3(cdof_clip.c[0], cdof_clip.c[1], cdof_clip.c[2]);\n"
        "    return cdof_lin + metal::cross(cdof_ang, offset);\n"
        "}"
    ),
    # mujoco_warp/_src/support.py:_compute_jacr
    "_compute_jacr_0": (
        "inline float3 _compute_jacr_0(wp_vec6_float cdof_clip, int affect) {\n"
        "    if (affect == 0) return float3(0.0f);\n"
        "    return float3(cdof_clip.c[0], cdof_clip.c[1], cdof_clip.c[2]);\n"
        "}"
    ),
    # mujoco_warp/_src/solver.py:state_check (ConstraintState.QUADRATIC = 1)
    "state_check_0": ("inline float state_check_0(float D, int state) {\n    return state == 1 ? D : 0.0f;\n}"),
    # mujoco_warp/_src/solver.py:active_check
    "active_check_0": (
        "inline float active_check_0(int tid, int threshold) {\n    return tid >= threshold ? 0.0f : 1.0f;\n}"
    ),
}

_USER_FUNC_REF_PAT = re.compile(r"\b([A-Za-z_]\w*_\d+)\s*\(")


def _emit_referenced_user_funcs(source: str) -> str:
    """Scan ``source`` for known mujoco_warp helper-function references
    and return their MSL definitions concatenated. Functions not in the
    lookup table are skipped — they'll surface as unresolved-symbol MSL
    compile errors with a meaningful function name, which is still
    better than the current ``MetalCodegenError`` gate.
    """
    seen: set[str] = set()
    for m in _USER_FUNC_REF_PAT.finditer(source):
        name = m.group(1)
        if name in _USER_FUNC_MSL_DEFS:
            seen.add(name)
    if not seen:
        return ""
    return "\n".join(_USER_FUNC_MSL_DEFS[n] for n in sorted(seen)) + "\n"


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
    # Legacy: names of outputs whose shape we used to pass as a synthetic
    # per-output ``<name>_shape`` input. Now superseded by the packed
    # shape buffer (see ``shape_packed_arrs``); kept on the artifact for
    # one release in case anything else reads it. Always empty in new
    # code.
    output_shape_inputs: list[str] = field(default_factory=list)
    # Names of every kernel arg whose ``arr.shape[k]`` is referenced from
    # the body. The launcher concatenates each one's runtime shape into
    # a single ``__shapes_packed`` int32 buffer, ordered to match this
    # tuple. The codegen emits ``__shapes_packed[i*<slot> + k]`` for the
    # k-th dim of the i-th array in this list. Combining shapes into one
    # buffer keeps the total kernel-arg count under Metal's 31-slot
    # hardware limit.
    shape_packed_arrs: tuple[str, ...] = ()
    # Per-array shape slot size in ``__shapes_packed`` (each array gets
    # ``shape_packed_slot`` int32s, padded with zeros). Warp arrays cap
    # at 4 dims so 4 is plenty.
    shape_packed_slot: int = 4
    # Names of read-only int32-1D input args packed into a single
    # ``__ints_packed`` MLX buffer to stay under Metal's 31-slot kernel-
    # arg limit. Layout:
    #   ``[off_0, ..., off_{K-1}, scalar_0, ..., scalar_{S-1}, data_arr0..., ...]``
    # where ``off_i`` is the start index of array ``i``'s data within
    # ``__ints_packed``, ``scalar_j`` is the value of the j-th packed
    # scalar (int32 or bool, lifted to int32), and the array data
    # follows. Codegen rewrites:
    #   - ``arr_k[expr]``     -> ``__ints_packed[__ints_packed[k] + (expr)]``
    #   - ``int_scalar_j``    -> ``__ints_packed[K + j]``
    #   - ``bool_scalar_j``   -> ``(bool)__ints_packed[K + j]``
    # Empty when the kernel fits without packing.
    ints_packed_arrs: tuple[str, ...] = ()
    # Names of int32/bool scalar inputs packed into ``__ints_packed``.
    # Stored at slots ``K..K+S-1`` of the buffer (immediately after the
    # array offsets). Bool scalars are lifted to int32 in the buffer and
    # cast back at the access site.
    ints_packed_scalars: tuple[str, ...] = ()
    # Names of read-only float-element input args packed into a single
    # ``__floats_packed`` MLX buffer. Includes scalar-float arrays
    # (``wp.array2d[float]``, ``wp.array3d[float]``) and vec/mat-element
    # arrays whose inner scalar is float (``wp.array2d[wp.vec3]``).
    # Layout: ``[data_arr0..., data_arr1..., ...]``. The per-array start
    # offsets live in ``__ints_packed`` after the int-array offsets and
    # int/bool scalars (so a single packing buffer of integer offsets
    # serves both the int and float packers). Codegen rewrites:
    #   - ``arr_k[expr]`` -> ``__floats_packed[__ints_packed[Ki + S + k] + (expr)]``
    floats_packed_arrs: tuple[str, ...] = ()
    # ``True`` if the kernel emits an output-init prologue and therefore
    # needs every body thread that shares a worldid to be in the same
    # threadgroup so the post-prologue ``threadgroup_barrier`` actually
    # synchronises them. The launcher inflates ``threadgroup`` to
    # ``(1, grid_y, grid_z)`` (capped at the device limit) when this is
    # set; with the default per-thread threadgroup the barrier degenerates
    # to a no-op and the prologue races the body.
    needs_init_barrier: bool = False
    # When non-zero, the kernel uses the SIMD-cooperative tile_cholesky
    # helper at this size — the launcher must dispatch with a 32-thread
    # threadgroup and the kernel body declares a ``threadgroup float
    # wp_tile_chol_smem[N*N]`` scratch at top scope. Zero means
    # single-thread (legacy) tile primitives only.
    coop_chol_n: int = 0
    # Init-shadow packing: when an atomic kernel has too many init
    # outputs to fit per-output ``__init`` shadow buffers under
    # Metal's 31-buffer cap, we pack them into one float buffer
    # (``__init_shadows_floats``) plus one int buffer
    # (``__init_shadows_ints``). The two tuples below list the output
    # names whose data goes into each packed buffer (in pack order).
    # ``init_shadow_packed_outputs`` lists ALL packed outputs in the
    # same order as the kernel's ``__init_shadow_offsets`` array so
    # the launcher can store the right offset per output.
    init_shadow_packed_outputs: tuple[str, ...] = ()
    init_shadow_floats: tuple[str, ...] = ()
    init_shadow_ints: tuple[str, ...] = ()
    # MSL declarations to inject before the kernel function body — used for
    # custom big-vec structs (vec5, vec6 = spatial_vector, vec8) that don't
    # have native MSL ``floatN`` equivalents. Empty for kernels that only
    # use native types.
    header: str = ""


# Match the AST-emitted ``for`` lines so we can flip ``<`` to ``>`` when
# the step expression resolves to a negative integer. The inner ``inc``
# capture handles both ``++var_X`` (step 1) and ``var_X += <step>`` forms.
_FOR_LOOP_LINE_PAT = re.compile(
    r"^(?P<indent>\s*)for \(var_(?P<iv>\w+) = (?P<start>[^;]+); "
    r"var_(?P=iv) < (?P<stop>[^;]+); "
    r"(?P<inc>(?:\+\+var_(?P=iv))|(?:var_(?P=iv) \+= (?P<step>[^)]+)))\) \{$"
)
# Matches lines that introduce a compile-time int constant — used to
# resolve a step expression like ``var_2__49`` to the literal ``-16``.
_CONST_INT_DECL_PAT = re.compile(r"^\s*const int var_(\w+) = (-?\d+);\s*$")


def _fix_negative_step_for_loops(lines: list[str]) -> list[str]:
    """Rewrite ``for (i = a; i < b; i += step) {`` to use ``>`` when
    ``step`` resolves to a negative literal.

    Python's ``range(start, stop, step)`` iterates ``i > stop`` for
    negative steps; the AST emitter writes ``<`` unconditionally, which
    silently makes the loop run zero iterations when ``start > stop``.
    This is the codegen bug behind mujoco_warp's blocked-Cholesky
    failure: the backward-substitution loop ``range(matrix_size -
    block_size, -1, -block_size)`` was compiled to an empty C-style
    ``for`` and the upper-triangular solve never ran, leaving the
    forward-substituted ``y`` as the "answer" instead of computing
    ``x``.

    Step expressions are resolved via the kernel's ``const int var_X
    = N;`` declarations (Warp emits one per constant int operand at the
    top of the body). Non-integer or non-constant steps stay positive
    by convention — covers ``range(0, n, BLOCK_DIM)`` etc.
    """
    const_ints: dict[str, int] = {}
    for raw in lines:
        m = _CONST_INT_DECL_PAT.match(raw)
        if m is not None:
            const_ints[m.group(1)] = int(m.group(2))

    def _step_is_negative(step_expr: str) -> bool:
        s = step_expr.strip()
        if s.startswith("var_"):
            v = const_ints.get(s[len("var_") :])
            return v is not None and v < 0
        try:
            return int(s) < 0
        except ValueError:
            return False

    out: list[str] = []
    for raw in lines:
        m = _FOR_LOOP_LINE_PAT.match(raw)
        if m is None:
            out.append(raw)
            continue
        step = m.group("step")  # ``None`` for the ``++var_X`` form
        if step is None or not _step_is_negative(step):
            out.append(raw)
            continue
        indent = m.group("indent")
        iv = m.group("iv")
        start = m.group("start")
        stop = m.group("stop")
        inc = m.group("inc")
        out.append(f"{indent}for (var_{iv} = {start}; var_{iv} > {stop}; {inc}) {{")
    return out


def _strip_comments_and_directives(line: str) -> str | None:
    s = line.strip()
    if not s:
        return None
    if s.startswith("//"):
        return None
    if s.startswith("#line"):
        return None
    return line


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
    #
    # ``output_arch=None`` routes tile-LTO builtins (e.g., ``tile_cholesky``)
    # to their CPU/no-MathDx dispatch so they emit plain function calls
    # rather than CUDA cuBLASDx LTO IR. ``block_dim`` is required by some
    # of those dispatch funcs even on the no-MathDx path.
    if not getattr(adj, "blocks", None):
        # ``block_dim=1`` forces the cooperative-tile builtins (``wp.tile``,
        # ``wp.tile_reduce``, etc.) to compile-time-shape themselves as
        # single-element tiles. The Metal backend currently runs every
        # cooperative-tile kernel serially (1 thread per threadgroup) — see
        # ``launch_metal_kernel`` for the matching launcher policy. Once
        # we implement real threadgroup-cooperative tiles this can climb
        # back to a real value.
        adj.build(
            builder=None,
            default_builder_options={
                "enable_backward": False,
                "output_arch": None,
                "block_dim": 1,
            },
        )
    # When the kernel is part of a registered module (``module="unique"``),
    # ``adj.build`` sets ``adj.builder_options`` to the module's options
    # dict, which our default options dict can't reach. Backfill the keys
    # tile-builtin value-funcs read at codegen time so their lookups don't
    # ``KeyError``. Use the kernel's per-launch ``block_dim`` if it was
    # threaded through; otherwise the safe default mirrors the Warp CPU
    # backend.
    if getattr(adj, "builder_options", None) is not None:
        adj.builder_options.setdefault("output_arch", None)
        adj.builder_options.setdefault("block_dim", 256)
    # Reset the global ``codegen.options`` ref to our merged dict so any
    # value-func evaluated from ``add_call`` sees the same view.
    import warp._src.codegen as _wp_codegen  # noqa: PLC0415

    _wp_codegen.options = adj.builder_options

    # Preprocess via the AST pipeline (see ``warp._src.codegen_metal_ast``):
    #   parse → structural fold (for/while/if) → drop unsupported locals
    #          → view fold → indexref-write fold → emit
    #
    # Output-equivalent to the old chain of ``_preprocess_*`` functions
    # (validated bit-exact against every kernel reachable through
    # ``mujoco_warp.step()``). The codegen-time win shows up in Phase 1.4
    # when the body emit loop below also moves to the AST tree — for now
    # we still emit a flat string list and the existing regex sweep takes
    # over from there.
    _early_vec_arr_info: dict[str, tuple[int, str]] = {}
    for arg in adj.args:
        v_info = _vec_dtype_info(arg)
        if v_info is not None:
            _early_vec_arr_info[arg.label] = v_info

    _ast_nodes = _ast_parse(adj.blocks[0].body_forward)
    _ast_nodes, _struct_skip = _ast_fold(_ast_nodes)
    # Inline ``@wp.func`` user-function calls into the kernel body before
    # the view / indexref / drop folds run, so those folds see the spliced-
    # in writes (otherwise kernels that write outputs only via helper
    # functions would be rejected as having no outputs). The inliner also
    # surfaces inlined int-constants and inlined struct locals so the
    # downstream slice-step recogniser and field-pointer pass can treat
    # them the same as kernel-level ones.
    _ast_nodes, _inlined_const_ints, _inlined_struct_locals, _inlined_var_ctypes = _ast_inline(_ast_nodes, adj)
    _ast_nodes, _drop_skip = _ast_fold_drop(_ast_nodes, adj)
    _ast_nodes, _view_skip = _ast_fold_views(_ast_nodes, adj, extra_const_ints=_inlined_const_ints)
    _ast_nodes, _indexref_skip = _ast_fold_indexref(_ast_nodes, adj, _early_vec_arr_info)
    # Flatten multi-dim ``wp::atomic_<op>(arr, i, j, ..., val)`` into the
    # 3-arg form the intrinsic regex handles, expanding to per-component
    # atomics for vec-typed arrays.
    _ast_nodes = _ast_fold_multidim_atomics(_ast_nodes, adj, _early_vec_arr_info)
    forward_lines = _ast_emit(_ast_nodes)

    # Negative-step Python ``range`` (e.g. ``range(start, -1, -16)`` in
    # mujoco_warp's blocked-Cholesky backward sub) emits as
    # ``for (i = start; i < stop; i += -step)`` from the AST, which
    # never enters its body when ``start > stop``. Python's intent is to
    # iterate *down* while ``i > stop``. Fix the comparison direction
    # here, after we have the flat IR with all ``const int var_X = N;``
    # declarations visible.
    forward_lines = _fix_negative_step_for_loops(forward_lines)

    vars_to_skip_decl: set[str] = _struct_skip | _drop_skip | _view_skip | _indexref_skip

    # Classify each array arg as input or output by scanning the IR strings.
    # MLX inputs are ``const device T*`` (read-only) — verified empirically —
    # so any array that is written through must become an MLX output. We can't
    # rely on ``arg.is_write`` here because it is only populated when
    # ``verify_autograd_array_access`` is enabled.
    written_arg_names: set[str] = set()
    atomic_arg_names: set[str] = set()
    # Which atomic ops touch each array (``add``/``sub``/``min``/``max``) and
    # which arrays are also targets of plain stores. The output-init prologue
    # uses this to pick a race-free seeding op: outputs touched only by
    # ``atomic_add``/``atomic_sub`` can be seeded with ``atomic_fetch_add``
    # on top of MLX's zero-fill, which commutes with the body's adds — no
    # ordering requirement between the seed and body threadgroups.
    atomic_op_kinds: dict[str, set[str]] = {}
    plain_store_arg_names: set[str] = set()
    array_store_pat = re.compile(r"\s*wp::array_store\s*\(\s*var_([A-Za-z_]\w*)")
    atomic_pat = re.compile(r"wp::atomic_(add|sub|min|max)\s*\(\s*var_([A-Za-z_]\w*)")
    scalar_store_pat = re.compile(r"\s*wp::__metal_scalar_store__\s*\(\s*var_([A-Za-z_]\w*)")
    # ``wp::tile_store<...>(arr, tile, off...)`` — the blocked-Cholesky
    # path writes through tile_store directly (no array_store wrapper).
    tile_store_pat = re.compile(r"wp::tile_store\s*<[^()]*>\s*\(\s*var_([A-Za-z_]\w*)")
    # ``wp::view(arr, slice)`` — slicing an array. The slice can be the
    # target of a downstream tile_store, but the IR loses the connection
    # back to the underlying array after the slice is taken. Treat any
    # array referenced by ``wp::view`` as potentially written.
    view_pat = re.compile(r"wp::view\s*\(\s*var_([A-Za-z_]\w*)")
    # Per-component atomics emitted by the multi-dim atomic fold use raw
    # ``atomic_fetch_<op>_explicit(&arr[...], val, ...)`` lines (no
    # ``wp::`` prefix and no ``var_`` on the array name — the var prefix
    # gets stripped because the array is also a kernel arg, which the
    # name-substitution pass folds to its bare name later). Detect those
    # too so the output classification picks up the write.
    raw_atomic_pat = re.compile(r"atomic_fetch_(add|sub|min|max)_explicit\s*\(\s*&\s*(\w+)\[")
    for raw in forward_lines:
        m = array_store_pat.match(raw)
        if m:
            written_arg_names.add(m.group(1))
            plain_store_arg_names.add(m.group(1))
        m = atomic_pat.search(raw)
        if m:
            written_arg_names.add(m.group(2))
            atomic_arg_names.add(m.group(2))
            atomic_op_kinds.setdefault(m.group(2), set()).add(m.group(1))
        m = scalar_store_pat.match(raw)
        if m:
            written_arg_names.add(m.group(1))
            plain_store_arg_names.add(m.group(1))
        m = tile_store_pat.search(raw)
        if m:
            written_arg_names.add(m.group(1))
        m = view_pat.search(raw)
        if m:
            written_arg_names.add(m.group(1))
        m = raw_atomic_pat.search(raw)
        if m:
            written_arg_names.add(m.group(2))
            atomic_arg_names.add(m.group(2))
            atomic_op_kinds.setdefault(m.group(2), set()).add(m.group(1))

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
        # Args whose names end in ``_out`` are conventionally outputs in
        # mujoco_warp (and most Warp kernels). Treat them as outputs even
        # when the body has no detected writes — this happens for kernels
        # whose write paths are all behind conditionals that don't fire
        # for trivial models (e.g. ``primitive_narrowphase`` with no
        # contact pairs). MLX still needs them bound as output buffers.
        if arg.label in written_arg_names or arg.label.endswith("_out"):
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
    # Field pointers that point into a struct's array field. Those fields
    # aren't materialised on Metal (the launcher doesn't pack arrays into
    # the struct's flat buffer), so writes through these pointers are
    # silently dropped and reads raise — handled in the store and load
    # branches below.
    unused_field_ptrs: set[str] = set()
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
    # Struct locals discovered inside inlined helper bodies (e.g. the
    # ``Geom`` struct built by ``geom_collision_pair`` for primitive
    # narrowphase). Their mangled labels live in the kernel body but
    # not in ``adj.variables``, so the inliner returns them on the side.
    for mangled_label, struct_cls in _inlined_struct_locals.items():
        struct_local_layouts[mangled_label] = _struct_layout_for(struct_cls)
        struct_local_is_arg[mangled_label] = False
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
                field_info = layout.fields[field_name]
                if field_info.kind == _STRUCT_FIELD_KIND_ARRAY_UNUSED:
                    # Array fields aren't materialised on Metal. Mark the
                    # field pointer so any store / read through it is
                    # rejected with a clear error (or, for stores, silently
                    # dropped — see the store handling below).
                    unused_field_ptrs.add(field_local)
                    continue
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
                if field_info.kind == _STRUCT_FIELD_KIND_ARRAY_UNUSED:
                    unused_field_ptrs.add(field_local)
                    continue
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
                    if rows in _MSL_VEC_NATIVE_N and cols in _MSL_VEC_NATIVE_N:
                        msl_vec = field_info.msl_type.split("x")[0]
                        col_strs: list[str] = []
                        for c in range(cols):
                            col_components = [f"{struct_local}[{base} + {r * cols + c}]" for r in range(rows)]
                            col_strs.append(f"{msl_vec}({', '.join(col_components)})")
                        subscript_map[field_local] = f"{field_info.msl_type}({', '.join(col_strs)})"
                    else:
                        comps = [f"{struct_local}[{base} + {r * cols + c}]" for r in range(rows) for c in range(cols)]
                        subscript_map[field_local] = f"{field_info.msl_type}_make({', '.join(comps)})"
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
        if field_info.kind == _STRUCT_FIELD_KIND_ARRAY_UNUSED:
            unused_field_ptrs.add(field_local)
            continue
        base = f"{elem_idx_expr} * {layout.scalars_per_elem} + {field_info.offset}"
        if field_info.kind == _STRUCT_FIELD_KIND_SCALAR:
            subscript_map[field_local] = f"{arr_name}[{base}]"
        elif field_info.kind == _STRUCT_FIELD_KIND_VEC:
            comps = [f"{arr_name}[({base}) + {k}]" for k in range(field_info.size)]
            ctor = field_info.msl_type if field_info.size in _MSL_VEC_NATIVE_N else f"{field_info.msl_type}_make"
            subscript_map[field_local] = f"{ctor}({', '.join(comps)})"
        elif field_info.kind == _STRUCT_FIELD_KIND_MAT:
            rows, cols = field_info.rows, field_info.cols
            if rows in _MSL_VEC_NATIVE_N and cols in _MSL_VEC_NATIVE_N:
                msl_vec = field_info.msl_type.split("x")[0]  # e.g. "float3" from "float3x3"
                col_strs: list[str] = []
                for c in range(cols):
                    col_components = [f"{arr_name}[({base}) + {r * cols + c}]" for r in range(rows)]
                    col_strs.append(f"{msl_vec}({', '.join(col_components)})")
                subscript_map[field_local] = f"{field_info.msl_type}({', '.join(col_strs)})"
            else:
                # Big-mat custom struct: row-major flat factory.
                comps = [f"{arr_name}[({base}) + {r * cols + c}]" for r in range(rows) for c in range(cols)]
                subscript_map[field_local] = f"{field_info.msl_type}_make({', '.join(comps)})"

    # --- Local variable declarations -----------------------------------
    body_lines: list[str] = []
    for var in adj.variables:
        if var.label in subscript_map:
            # This local was a pointer into an array arg; we'll inline its
            # uses below, so it doesn't need a declaration.
            continue
        if var.label in unused_field_ptrs:
            # Pointer into a struct's array field — not materialised on
            # Metal. Writes are dropped, so the local is unreferenced.
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
                if field_info.kind == _STRUCT_FIELD_KIND_ARRAY_UNUSED:
                    continue
                local_name = _per_field_local(var.label, field_name)
                # MSL ``T()`` zero-constructs scalar / vec / mat values.
                # Zero-init form differs by type:
                #   - Native MSL types (``float``, ``float3``, ``float3x3``,
                #     ``int2``, etc.) accept ``T(0)`` — broadcasts the int
                #     to fill all components/elements.
                #   - Custom structs (``wp_mat6x3_float``, ``wp_vec6_float``)
                #     have no one-int constructor; aggregate value-init
                #     ``T()`` zeros the trailing ``c[N]`` array.
                _zero = (
                    f"{field_info.msl_type}()"
                    if field_info.msl_type.startswith(("wp_mat", "wp_vec"))
                    else f"{field_info.msl_type}(0)"
                )
                body_lines.append(f"    {field_info.msl_type} {local_name} = {_zero};")
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

    # Emit per-field local declarations for inlined struct locals (their
    # mangled labels are not in ``adj.variables`` so the loop above
    # didn't catch them).
    for mangled_label in _inlined_struct_locals:
        layout = struct_local_layouts[mangled_label]
        for field_name, field_info in layout.fields.items():
            if field_info.kind == _STRUCT_FIELD_KIND_ARRAY_UNUSED:
                continue
            local_name = _per_field_local(mangled_label, field_name)
            _zero = (
                f"{field_info.msl_type}()"
                if field_info.msl_type.startswith(("wp_mat", "wp_vec"))
                else f"{field_info.msl_type}(0)"
            )
            body_lines.append(f"    {field_info.msl_type} {local_name} = {_zero};")

    # Map tile-typed locals to ``(rows, cols, msl_scalar)`` for the tile-
    # intrinsic translator (it needs the result-tile shape for cholesky /
    # cholesky_solve, where dimensions don't appear in the call's template
    # args). Shape (1,1) entries are kept so the translator can short-
    # circuit cholesky / cholesky_solve to scalar ops even though the
    # local itself collapsed to a plain value via ``_msl_var_type``.
    tile_var_dims: dict[str, tuple[int, int, str]] = {}
    # Parallel map: var label → vec element count (0 for scalar tiles).
    # Lets ``tile_map`` / ``tile_reduce`` / ``tile_extract`` lower
    # element-typed tiles correctly without changing every consumer
    # of ``tile_var_dims``.
    tile_var_vec_n: dict[str, int] = {}

    def _record_tile(label: str, dtype_ctype: str, rows: int, cols: int) -> None:
        if dtype_ctype in _SCALAR_CTYPE_TO_MSL:
            tile_var_dims[label] = (rows, cols, _SCALAR_CTYPE_TO_MSL[dtype_ctype])
            tile_var_vec_n[label] = 0
            return
        v = _vec_dtype_inner(dtype_ctype)
        if v is not None:
            n_elem, inner_ctype = v
            inner_msl = _SCALAR_CTYPE_TO_MSL.get(inner_ctype)
            if inner_msl is not None:
                tile_var_dims[label] = (rows, cols, inner_msl)
                tile_var_vec_n[label] = n_elem

    for var in adj.variables:
        parsed = _parse_tile_ctype(var.ctype())
        if parsed is None:
            continue
        _kind, dtype_ctype, rows, cols = parsed
        _record_tile(var.label, dtype_ctype, rows, cols)
    # Inlined-function tile locals — the inliner records every spliced
    # local's ctype keyed by its mangled label (e.g. ``136__17``); add
    # the tile-typed ones here so the regex translators can resolve
    # ``var_136__17``-style references to the right ``RxC`` shape.
    for label, ctype in _inlined_var_ctypes.items():
        if label in tile_var_dims:
            continue
        parsed = _parse_tile_ctype(ctype)
        if parsed is None:
            continue
        _kind, dtype_ctype, rows, cols = parsed
        _record_tile(label, dtype_ctype, rows, cols)

    # When ``atomic_outputs=True`` is passed to ``mx.fast.metal_kernel``,
    # *every* output buffer comes through as ``device atomic<T>*``. Reads
    # via plain subscripting won't compile, so we'll wrap them in
    # ``atomic_load_explicit`` inside ``_finalize``.
    atomic_output_names: set[str] = {a.label for a in output_args} if has_atomic else set()

    # ``tile_view`` materialisation registry: view local label →
    # (parent_label, parent_rows, parent_cols, view_rows, view_cols,
    # scalar, row_off_expr, col_off_expr). Populated by the
    # ``repl_view`` translator and read by mutating-op translators
    # (matmul_acc, *_solve_inplace) so writes to the view propagate
    # back to the parent tile struct via an explicit copy.
    view_aliases: dict[str, tuple[str, int, int, int, int, str, str, str]] = {}
    # Set of (N, scalar) pairs for which we emitted the cooperative
    # variant of ``tile_cholesky``. Drives kernel-scope threadgroup
    # memory + lane declaration and the launcher's threadgroup-size
    # selection.
    coop_chol_seen: set[tuple[int, str]] = set()
    # Map ``transpose_result_label`` → ``source_label``. On the
    # CUDA/cuBLASDx path ``tile_transpose`` returns a layout-only view
    # over the same shared-memory storage — mutating the result
    # mutates the source. On Metal we copy into a private struct, so
    # mutating ops (``*_solve_inplace``) don't propagate back unless
    # we emit an explicit transpose-back writeback. This map drives
    # that emission. Populated by ``repl_transpose_notpl``.
    transpose_aliases: dict[str, str] = {}
    # Pre-scan: a kernel goes "cooperative" when any of its
    # ``tile_cholesky`` / ``tile_cholesky_inplace`` calls applies to a
    # square tile in the cooperative range. Determining this *before*
    # emit lets the load/store/etc. translators emit cooperative
    # variants for every tile op in the kernel — otherwise they'd
    # only see the cholesky call mid-translation when other ops have
    # already been lowered to the single-thread form.
    _chol_pat_pre = re.compile(r"\bwp::tile_cholesky\s*<[^()]*>\s*\(([^)]*)\)")
    _chol_inplace_pat_pre = re.compile(r"\bwp::tile_cholesky_inplace\s*<[^()]*>\s*\(([^)]*)\)")
    is_coop_kernel = False
    pre_coop_chol_n = 0
    for raw in forward_lines:
        for m in _chol_pat_pre.finditer(raw):
            args = [a.strip() for a in m.group(1).split(",")]
            if len(args) >= 5:
                in_arg = args[3]
                in_label = in_arg[len("var_") :] if in_arg.startswith("var_") else in_arg
                dims = tile_var_dims.get(in_label)
                if dims and dims[0] == dims[1] and _COOP_CHOL_MIN_N <= dims[0] <= _COOP_CHOL_MAX_N:
                    is_coop_kernel = True
                    pre_coop_chol_n = max(pre_coop_chol_n, dims[0])
        for m in _chol_inplace_pat_pre.finditer(raw):
            args = [a.strip() for a in m.group(1).split(",")]
            if len(args) >= 2:
                in_arg = args[1]
                in_label = in_arg[len("var_") :] if in_arg.startswith("var_") else in_arg
                dims = tile_var_dims.get(in_label)
                if dims and dims[0] == dims[1] and _COOP_CHOL_MIN_N <= dims[0] <= _COOP_CHOL_MAX_N:
                    is_coop_kernel = True
                    pre_coop_chol_n = max(pre_coop_chol_n, dims[0])

    # ---- Quaternion-typed locals --------------------------------------
    # Quats are stored as ``vec_t<4>``/``float4``, but ``wp::mul`` on two
    # quats is the Hamilton product — MSL's ``float4 * float4`` is
    # component-wise and produces silently wrong results. Collect every
    # quat-typed local so the rewrite in ``_finalize`` can dispatch
    # quat-quat ``wp::mul`` to ``wp_quat_mul``. Kernel-level vars come from
    # the IR's variable list; vars inside inlined ``@wp.func`` bodies are
    # not in ``adj.variables``, so also scan the body text for their
    # ``wp::quat_t<...> var_X`` declarations.
    quat_var_labels: set[str] = set()
    for var in adj.variables:
        if getattr(var.type, "_wp_generic_type_str_", None) == "quat_t":
            quat_var_labels.add(var.label)
    for arg in adj.args:
        if getattr(arg.type, "_wp_generic_type_str_", None) == "quat_t":
            quat_var_labels.add(arg.label)
    _quat_decl_pat = re.compile(r"wp::quat_t\s*<\s*wp::\w+\s*>\s+var_(\w+)")
    for raw in forward_lines:
        for m in _quat_decl_pat.finditer(raw):
            quat_var_labels.add(m.group(1))
    _quat_mul_call_pat = re.compile(r"wp::mul\s*\(\s*var_(\w+)\s*,\s*var_(\w+)\s*\)")

    def _rewrite_quat_mul(text: str) -> str:
        def repl(m: re.Match[str]) -> str:
            if m.group(1) in quat_var_labels and m.group(2) in quat_var_labels:
                return f"wp_quat_mul(var_{m.group(1)}, var_{m.group(2)})"
            return m.group(0)

        return _quat_mul_call_pat.sub(repl, text)

    # --- Forward statements --------------------------------------------
    def _finalize(translated: str) -> str:
        # Quat-quat multiplies must be intercepted before the generic
        # ``wp::mul`` -> ``(a * b)`` pattern erases the call, and before
        # subscript inlining rewrites the ``var_X`` operand names the
        # type lookup keys on.
        translated = _rewrite_quat_mul(translated)
        # Lower tile intrinsics *before* the subscript-substitute pass —
        # the tile pattern matches on the raw ``wp::tile_*<...>`` shape,
        # which contains ``var_X`` operands that the substitute would
        # otherwise rewrite to expressions and break the parse.
        translated = _translate_tile_intrinsics(
            translated,
            tile_var_dims,
            view_aliases,
            coop_chol_seen,
            is_coop_kernel,
            transpose_aliases,
            tile_var_vec_n,
            atomic_output_names,
        )
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

        # Some IR paths emit the CPU-side native ``floatRxC()`` directly
        # (skipping the ``wp::mat_t<R,C,T>()`` form the rewriter above
        # expects). Native MSL matrix types reject the no-arg form;
        # for *square* sizes ``floatRxR(0)`` constructs an identity-
        # scaled diagonal (zero diag → zero mat), but ``floatRxC(0)``
        # for non-square sizes is also a compile error — MSL only
        # supports the diagonal-broadcast form on square matrices.
        # Rewrite empty constructors to an explicit "zero columns"
        # form: ``floatNxM(floatM(0), floatM(0), ...)`` with N copies
        # of a zero column. The MSL convention is ``floatNxM`` has
        # N columns of M rows (per the spec), so column type is
        # ``floatM``.
        def _zero_native_mat(m: re.Match[str]) -> str:
            scalar = m.group(1)
            n_cols = int(m.group(2))
            n_rows = int(m.group(3))
            col_type = f"{scalar}{n_rows}"
            zeros = ", ".join([f"{col_type}(0)"] * n_cols)
            return f"{scalar}{n_cols}x{n_rows}({zeros})"

        translated = re.sub(
            r"\b((?:float|half|int|uint))(\d+)x(\d+)\s*\(\s*\)",
            _zero_native_mat,
            translated,
        )

        # Same for ``floatRxC(0)`` — the diagonal-scalar form. MSL only
        # accepts it on *square* matrices; non-square needs explicit
        # zero columns.
        def _zero_scalar_mat(m: re.Match[str]) -> str:
            scalar = m.group(1)
            n_cols = int(m.group(2))
            n_rows = int(m.group(3))
            if n_cols == n_rows:
                return m.group(0)  # square: ``floatNxN(0)`` works
            col_type = f"{scalar}{n_rows}"
            zeros = ", ".join([f"{col_type}(0)"] * n_cols)
            return f"{scalar}{n_cols}x{n_rows}({zeros})"

        translated = re.sub(
            r"\b((?:float|half|int|uint))(\d+)x(\d+)\s*\(\s*0\s*\)",
            _zero_scalar_mat,
            translated,
        )
        # ``inff`` is Warp's IR float-infinity literal. MSL has
        # ``INFINITY`` (float) but not ``inff``; substitute.
        translated = re.sub(r"\binff\b", "INFINITY", translated)
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
        # ``wp::lower_bound`` — rewritten here (not in the intrinsic
        # table) because the 2-arg form references ``<argname>_shape``,
        # which only exists under the array's final parameter name.
        translated = _LOWER_BOUND_4ARG_PAT.sub(r"wp_lower_bound(\1, \2, \3, \4)", translated)
        translated = _LOWER_BOUND_2ARG_PAT.sub(r"wp_lower_bound(\1, 0, (int)\1_shape[0], \2)", translated)
        # When the kernel uses ``wp.atomic_*`` on any output, MLX makes
        # *every* output ``device atomic<T>*``. Plain reads
        # ``var_X = atomic_arr[idx]`` then fail to compile because MSL
        # forbids implicit conversion from atomic to value. Wrap each such
        # read in ``atomic_load_explicit(&arr[idx], memory_order_relaxed)``.
        # The index expression can contain nested ``[...]`` (shape lookups
        # like ``arr[i * shape[k] + j]``), so we manually scan for the
        # matching close bracket instead of using a single-shot regex with
        # a character-class exclusion.
        if has_atomic:
            for out_name in atomic_output_names:
                translated = _wrap_atomic_load_reads(translated, out_name)
        _check_no_unsupported_intrinsics(translated)
        return translated

    # Match raw declaration lines emitted by the user-function inliner for
    # *intermediate* pointer-typed locals (e.g. ``device int* var_21__9;``).
    # When the kernel-level address-fold pass aliases that local into
    # ``subscript_map``, the bare ``var_X`` reference inside the decl gets
    # corrupted by the substitute pass in ``_finalize`` (which textually
    # replaces ``var_X`` everywhere). The cleanest fix is to elide the decl
    # itself once we know the local is aliased away.
    inlined_decl_pat = re.compile(r"^\s*[\w<>:*\s]+?\s+var_(\w+)\s*;\s*$")

    for raw in forward_lines:
        line = _strip_comments_and_directives(raw)
        if line is None:
            continue
        # Drop inliner-emitted decls for locals that the address-fold pass
        # aliased into ``subscript_map`` — leaving them in place would let
        # ``_finalize`` substitute the alias expression *into* the decl,
        # producing malformed MSL like ``device int* arr[idx];``.
        m_decl = inlined_decl_pat.match(raw)
        if m_decl and m_decl.group(1) in subscript_map:
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
        # Struct-to-struct local copy ``var_X = var_Y;``. Both X and Y
        # have already been split into per-field locals; expand the
        # whole-struct copy into one per-field assignment per non-array
        # field. Inliner-emitted return-value writes for struct-typed
        # ``@wp.func`` calls land here (e.g. ``var_19 = var_37__0;``
        # from cartpole's ``geom_collision_pair`` -> ``Geom`` chain).
        # Match both the bare ``var_X = var_Y;`` form (after the
        # ``wp::copy`` strip in ``_translate_intrinsics``) AND the raw
        # IR form ``var_X = wp::copy(var_Y);``. We have to catch the
        # raw form here because the body-emission loop runs *before*
        # ``_finalize`` strips ``wp::copy`` — without that, the line
        # falls through to the unsupported-intrinsic guard or, worse,
        # gets emitted as ``var_2 = var_1;`` referencing per-field-
        # split locals that don't exist in MSL.
        m_struct_copy = re.match(r"^(?P<indent>\s*)var_(\w+)\s*=\s*var_(\w+)\s*;\s*$", raw)
        if m_struct_copy is None:
            m_struct_copy = re.match(
                r"^(?P<indent>\s*)var_(\w+)\s*=\s*wp::copy\s*\(\s*var_(\w+)\s*\)\s*;\s*$",
                raw,
            )
        if (
            m_struct_copy
            and m_struct_copy.group(2) in struct_local_layouts
            and m_struct_copy.group(3) in struct_local_layouts
        ):
            indent = m_struct_copy.group("indent")
            dst = m_struct_copy.group(2)
            src = m_struct_copy.group(3)
            dst_layout = struct_local_layouts[dst]
            src_layout = struct_local_layouts[src]
            for fname, finfo in dst_layout.fields.items():
                if finfo.kind == _STRUCT_FIELD_KIND_ARRAY_UNUSED:
                    continue
                if fname not in src_layout.fields:
                    continue  # different layouts; nothing sensible to copy
                dst_local = _per_field_local(dst, fname)
                src_local = _per_field_local(src, fname)
                body_lines.append(f"{indent}{dst_local} = {src_local};")
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
        # 3-arg form: ``wp::*_inplace(vec, idx, val)`` — element-wise
        # assignment to a vec local (``vec[idx] op= val``). MSL natively
        # supports indexed write on its built-in vector types, and our
        # ``wp_vecN_<scalar>`` struct exposes a writable ``operator[]``,
        # so the translation is direct. Match before the 2-arg form because
        # ``[^()]+?`` for the value would otherwise eat the comma + index.
        elem_inplace_pat = re.compile(
            r"^(?P<indent>\s*)wp::(?P<op>assign_inplace|add_inplace|sub_inplace|"
            r"mul_inplace|div_inplace)\s*\(\s*var_(?P<vec>\w+)\s*,\s*var_(?P<idx>\w+)\s*,\s*"
            r"(?P<val>[^()]+?)\s*\)\s*;\s*$"
        )
        m_elem = elem_inplace_pat.match(raw)
        if m_elem:
            indent = m_elem.group("indent")
            op = store_op_map[m_elem.group("op")]
            vec = m_elem.group("vec")
            idx = m_elem.group("idx")
            value = m_elem.group("val")
            body_lines.append(_finalize(f"{indent}var_{vec}[var_{idx}] {op} {value};"))
            continue
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
            if addr in unused_field_ptrs:
                # Drop the write — the target struct field is an array
                # field that isn't materialised on Metal.
                continue
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
        # ``wp::__metal_scalar_store__(arr, op, flat_idx, val);`` — synthetic
        # token emitted by ``_preprocess_indexref_writes`` for single-
        # component writes through ``address + indexref + store``. The
        # flat_idx already accounts for the vec stride, so this lowers to
        # a plain scalar subscript write (or atomic store under atomic mode).
        scalar_store_match = re.match(
            r"^(?P<indent>\s*)wp::__metal_scalar_store__\s*\(\s*var_(?P<arr>\w+)\s*,\s*"
            r"(?P<op>store|assign_inplace|add_inplace|sub_inplace|mul_inplace|div_inplace)\s*,\s*"
            r"(?P<idx>[^,]+(?:,[^,]+)*?)\s*,\s*(?P<val>[^()]+?)\s*\)\s*;\s*$",
            raw,
        )
        if scalar_store_match:
            indent = scalar_store_match.group("indent")
            arr = scalar_store_match.group("arr")
            op_name = scalar_store_match.group("op")
            idx = scalar_store_match.group("idx").strip()
            val = scalar_store_match.group("val").strip()
            output_arg_names = {a.label for a in output_args}
            op_sym = {
                "store": "=",
                "assign_inplace": "=",
                "add_inplace": "+=",
                "sub_inplace": "-=",
                "mul_inplace": "*=",
                "div_inplace": "/=",
            }[op_name]
            if has_atomic and arr in output_arg_names and op_sym == "=":
                line_out = f"{indent}atomic_store_explicit(&{arr}[{idx}], {val}, memory_order_relaxed);"
            else:
                line_out = f"{indent}{arr}[{idx}] {op_sym} {val};"
            body_lines.append(_finalize(line_out))
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
                            native = finfo.rows in _MSL_VEC_NATIVE_N and finfo.cols in _MSL_VEC_NATIVE_N
                            for r in range(finfo.rows):
                                for c in range(finfo.cols):
                                    rhs = f"{src}[{c}][{r}]" if native else f"{src}.c[{r * finfo.cols + c}]"
                                    body_lines.append(
                                        _finalize(
                                            _emit_scalar_write(
                                                f"{base} + {off + r * finfo.cols + c}",
                                                rhs,
                                            )
                                        )
                                    )
                    continue
                flat_idx = _flat_index_expr(arr, indices)
                body_lines.append(_finalize(_emit_scalar_write(flat_idx, value)))
                continue
        # Quat-quat ``wp::mul`` must be intercepted before the generic
        # pattern lowers it to the (component-wise, wrong-for-quats)
        # ``(a * b)`` form.
        translated = _translate_intrinsics(_rewrite_quat_mul(line.strip()))
        body_lines.append(f"    {_finalize(translated)}")

    # ---- Output-init prologue ----------------------------------------
    # MLX recycles output buffers from a pool, so any output element a
    # kernel leaves *unwritten* surfaces stale data from a previous
    # tenant — observed as ``_kinematics_branch`` skipping body 0
    # (worldbody, not in ``body_branches``) and the corresponding
    # ``xpos[0]`` slot ending up with the gravity vector that
    # ``_cacc_world`` left behind in the recycled buffer. The fix:
    # seed each affected output from the user's wp.array data before
    # the kernel body runs.
    #
    # We seed in two cases:
    #   1. The kernel uses ``wp.atomic_*`` on the output. The output is
    #      atomic-typed (``atomic_outputs=True``); we need atomic_store
    #      to seed and the user's previous values to accumulate from.
    #   2. The kernel reads the output back during compute (e.g.
    #      kinematics's ``xpos_out[parent]`` chain) — though for
    #      partially-written stateless outputs (``xpos`` skipping
    #      worldbody) the same seeding fixes both. Detected by scanning
    #      the IR for ``wp::address(var_<out>, ...)`` (the read
    #      pattern Warp emits before ``wp::load``) and by looking for
    #      ``wp::view`` of the output (``arr[w]`` slicing).
    #
    # Partition is by ``thread_position_in_grid.x`` (worldid).
    output_label_set = {a.label for a in output_args}
    addr_read_pat = re.compile(r"wp::address\s*\(\s*var_([A-Za-z_]\w*)\s*,")
    view_read_pat = re.compile(r"wp::view\s*\(\s*var_([A-Za-z_]\w*)\s*,")
    read_outputs: set[str] = set()
    for raw in forward_lines:
        for m in addr_read_pat.finditer(raw):
            if m.group(1) in output_label_set:
                read_outputs.add(m.group(1))
        for m in view_read_pat.finditer(raw):
            if m.group(1) in output_label_set:
                read_outputs.add(m.group(1))
    # Detect kernels that don't follow the ``dim=(nworld, ...)`` launch
    # convention. The init prologue uses
    # ``_init_w = thread_position_in_grid.x`` to partition output
    # initialisation per worldid; kernels like ``_primitive_narrowphase``
    # (launched with ``dim=ncollision``) and ``_efc_contact_init``
    # instead read worldid via an array indirection
    # (``worldid = collision_worldid_in[wp.tid()]``), so thread.x is a
    # collision_id, *not* a worldid. Running the prologue with
    # thread.x as the partition would seed wrong output slots and race
    # with body atomic_adds (we observed cartpole's ``nefc`` over-
    # counting 28 vs CPU 12). For these kernels we skip the prologue
    # entirely — mujoco_warp's caller-side ``d.nefc.zero_()`` plus
    # MLX's ``init_value=0.0`` for atomic outputs gives the same
    # net result as a zero-seed prologue would.
    non_standard_launch = any("_worldid_in" in a.label for a in adj.args if _is_array_arg(a))
    init_outputs: list[str] = []
    if has_atomic and not non_standard_launch:
        # Every atomic output gets seeded so atomic_add accumulates from
        # the user's previous value (else MLX zero-init wipes it).
        init_outputs.extend(a.label for a in output_args if a.label in atomic_arg_names)
        # Atomic kernels also typically share their non-atomic outputs
        # with sibling kernels — mujoco_warp's constraint pipeline has
        # ``_equality_connect``, ``_limit_slide_hinge``, ``_limit_ball``,
        # etc. each writing CONDITIONALLY to different rows of ``d.efc.J``
        # / ``d.efc.aref`` / etc. with their own atomic-allocated row
        # offsets. MLX gives each launch a fresh output buffer, so
        # without seeding, each kernel overwrites the *whole* array —
        # losing every other kernel's writes. Seed every non-atomic
        # output too. Below, the init shadows are packed into
        # ``__init_shadows_floats`` / ``__init_shadows_ints`` buffers
        # to keep the kernel under Metal's 31-buffer cap.
        for a in output_args:
            if a.label not in init_outputs:
                init_outputs.append(a.label)
    # Plus any output the kernel reads back from — preserves elements
    # the kernel doesn't write (kinematics_branch / sensor partial
    # writes) and lets read-then-write patterns see real prior values.
    if not non_standard_launch:
        for a in output_args:
            if a.label in read_outputs and a.label not in init_outputs:
                init_outputs.append(a.label)
    # Plus any output written via ``tile_store(arr, tile, offset=...)``
    # at a non-zero offset. The write is partial and previous launches'
    # writes to other slices need to survive. mujoco_warp's
    # ``_tile_euler_dense`` is the canonical case: dense Euler launches
    # one kernel per skeleton dof block, each writing
    # ``qacc[dof_block_offset:dof_block_offset+TILE]``. Without seeding,
    # MLX's fresh output buffer leaves the OTHER blocks' qacc as zeros.
    if not non_standard_launch:
        # Match the helper-call form ``wp_tile_RxC_<scalar>_store(arr,
        # base, stride, row_off, col_off, tile)`` — six args. When
        # ``row_off`` or ``col_off`` is anything other than literal
        # ``0``, the store covers a partial slice.
        store_call_pat = re.compile(
            r"\bwp_tile_\d+x\d+(?:_vec\d+)?_\w+_store\s*\(\s*"
            r"([\w]+)\s*,\s*[^,]+,\s*[^,]+,\s*([^,]+)\s*,\s*([^,]+)\s*,"
        )
        for raw in body_lines:
            for m in store_call_pat.finditer(raw):
                arr_name = m.group(1)
                row_off = m.group(2).strip()
                col_off = m.group(3).strip()
                if row_off == "0" and col_off == "0":
                    continue
                if arr_name in output_label_set and arr_name not in init_outputs:
                    init_outputs.append(arr_name)
    # Detect kernels with an early ``return;`` at function scope —
    # they have a guarded write path where some thread invocations
    # skip writing their output slot entirely. ``_geom_local_to_global``
    # is the canonical case: ``if body_weldid == 0 and not_mocap:
    # return`` skips the write for static (worldbody-attached) geoms.
    # Without this seed, ``geom_xpos[floor]/[rail1]/[rail2]`` ends up
    # MLX-stale on Metal even though ``put_data`` populated them, which
    # wrecks the broadphase OBB filter for any pair with a static geom.
    #
    # We can't tell at codegen which kernels rely on prior user values
    # vs. which ones expect caller-zeroed outputs (``_friction_dof``
    # has the same early-return pattern but its callers do
    # ``d.nf.zero_()`` etc. and depend on MLX's zero-init). To avoid
    # blanket-seeding, we apply the heuristic only when:
    #   1. The kernel has a top-level ``return;``, AND
    #   2. Adding init shadows for *all* non-atomic outputs would still
    #      fit under Metal's 31-buffer limit. This catches the
    #      ``_geom_local_to_global`` shape (4 outputs, well under)
    #      while leaving ``_friction_dof`` (15 outputs) on MLX's
    #      zero-init path that its callers already expect.
    #
    # The signal is a top-level ``return;`` (not nested inside a
    # ``do { ... } while (0);`` block — those are the inliner's
    # break-as-return wrappers, not actual conditional skips).
    if not non_standard_launch:
        depth = 0
        has_early_return = False
        for raw in forward_lines:
            stripped = raw.strip()
            if stripped.startswith("do "):
                depth += 1
            elif stripped.startswith("} while"):
                depth = max(0, depth - 1)
            elif stripped == "return;" and depth == 0:
                has_early_return = True
                break
        if has_early_return:
            # Tentatively add every output not already seeded.
            tentative = list(init_outputs)
            for a in output_args:
                if a.label not in tentative:
                    tentative.append(a.label)
            # Estimate buffer-slot cost. Each new init shadow is one
            # extra MLX input. ``__shapes_packed`` may add one more
            # slot — be conservative and always reserve it.
            tentative_seeds = len(tentative)
            slots_after = (
                len(input_args) + len(output_args) + tentative_seeds + 1  # __shapes_packed (may or may not be present)
            )
            if slots_after <= 30:
                init_outputs = tentative
    # ---- Decide between per-output __init shadows and packed shadows --
    # When the kernel has many init outputs (e.g. mujoco_warp's
    # 15-output ``_limit_slide_hinge`` / ``_equality_connect``), per-
    # output ``<name>__init`` MLX inputs would push the buffer count
    # past Metal's 31-slot cap. Pack init shadows into one float +
    # one int buffer instead. The choice is per-kernel.
    use_packed_init_shadows = False
    if init_outputs:
        # Conservative: assume ``__shapes_packed`` is needed (it's
        # cheap to over-estimate by 1 here; the real check happens
        # at artifact-build time).
        per_output_slots = len(input_args) + len(output_args) + len(init_outputs) + 1
        if per_output_slots > 30:
            use_packed_init_shadows = True
    init_shadow_floats: list[str] = []  # float-typed init outputs (in pack order)
    init_shadow_ints: list[str] = []  # int/bool-typed init outputs (in pack order)
    if use_packed_init_shadows:
        for out_name in init_outputs:
            arg_var = next(a for a in output_args if a.label == out_name)
            inner_msl = _msl_array_inner_msl_type(arg_var)
            if inner_msl in ("int", "uint", "bool"):
                init_shadow_ints.append(out_name)
            else:
                init_shadow_floats.append(out_name)

    if init_outputs:
        # The init prologue must run before any thread executes the body,
        # otherwise its (idempotent) atomic_stores can clobber a sibling
        # thread's body writes when threads race. We gate the prologue on
        # the leading thread-of-threadgroup (y==0, z==0) and follow it
        # with a device-memory threadgroup barrier — the launcher pairs
        # this with a threadgroup that spans every non-x grid dim so
        # every body thread for the same worldid waits behind the barrier.
        prologue: list[str] = [
            "    // -- Output init prologue (seed from user wp.array data) --",
            "    if (thread_position_in_threadgroup.y == 0 && thread_position_in_threadgroup.z == 0) {",
            "        int _init_w = (int)thread_position_in_grid.x;",
        ]
        # Compute per-output offsets within the packed buffers (only
        # used when ``use_packed_init_shadows`` is set). Each output's
        # data is stored as ``world_count * stride`` scalars at the
        # offset; offsets are passed to the kernel via
        # ``__init_shadow_offsets[i]`` (an extension to ``__ints_packed``).
        for out_name in init_outputs:
            arg_var = next(a for a in output_args if a.label == out_name)
            ndim = getattr(arg_var.type, "ndim", 1)
            v_info = _vec_dtype_info(arg_var)
            m_info = _mat_dtype_info(arg_var)
            # MLX collapses the inner element-dim to ONE per element type
            # (see ``_array_view_dtype_and_shape``):
            #   vec3   -> (*shape, 3)
            #   mat33  -> (*shape, 9)        # rows*cols flattened
            #   struct -> (*shape, scalars)
            # So the per-world stride uses one extra dim beyond ``ndim``
            # for any non-scalar element type — *not* two for mats.
            inner_extra = 1 if (v_info is not None or m_info is not None) else 0
            stride_terms = [f"{out_name}_shape[{k}]" for k in range(1, ndim + inner_extra)]
            stride_expr = " * ".join(stride_terms) if stride_terms else "1"
            if use_packed_init_shadows:
                # Pick the packed buffer + per-output index.
                if out_name in init_shadow_floats:
                    pack_buf = "__init_shadows_floats"
                    pack_idx = init_shadow_floats.index(out_name)
                else:
                    pack_buf = "__init_shadows_ints"
                    pack_idx = init_shadow_ints.index(out_name)
                # Offset constant lives in ``__init_shadow_offsets`` —
                # appended to ``__ints_packed`` by the launcher. The
                # offset position depends on whether float/int packing
                # is active for this kernel; simpler to use one combined
                # offsets table indexed by global init-shadow ordinal.
                init_idx = init_outputs.index(out_name)
                src_expr = f"{pack_buf}[__init_shadow_offsets[{init_idx}] + _init_flat]"
            else:
                src_expr = f"{out_name}__init[_init_flat]"
            if has_atomic:
                # Outputs touched only by ``atomic_add``/``atomic_sub`` (and
                # never plain-stored) are seeded with ``atomic_fetch_add`` on
                # top of MLX's zero-fill. Unlike ``atomic_store``, the add
                # commutes with the body's own adds, so a seed landing *after*
                # another threadgroup's accumulation no longer wipes it — the
                # threadgroup barrier below only orders threads within one
                # threadgroup, not across the grid.
                ops = atomic_op_kinds.get(out_name, set())
                if ops and ops <= {"add", "sub"} and out_name not in plain_store_arg_names:
                    store_stmt = (
                        f"            atomic_fetch_add_explicit(&{out_name}[_init_flat], "
                        f"{src_expr}, memory_order_relaxed);"
                    )
                else:
                    store_stmt = (
                        f"            atomic_store_explicit(&{out_name}[_init_flat], {src_expr}, memory_order_relaxed);"
                    )
            else:
                store_stmt = f"            {out_name}[_init_flat] = {src_expr};"
            # Bound the seed by the output's actual leading dim: kernels are
            # often launched with more x-threads than the output has rows
            # (e.g. ``dim=N`` reductions into a 1-element accumulator), and
            # an unguarded ``_init_w`` would write past the end of the
            # buffer.
            prologue.extend(
                [
                    f"        if (_init_w < (int){out_name}_shape[0]) {{",
                    f"        int _init_stride_{out_name} = {stride_expr};",
                    f"        for (int _init_i = 0; _init_i < _init_stride_{out_name}; ++_init_i) {{",
                    f"            int _init_flat = _init_w * _init_stride_{out_name} + _init_i;",
                    store_stmt,
                    "        }",
                    "        }",
                ]
            )
        prologue.append("    }")
        prologue.append("    threadgroup_barrier(metal::mem_flags::mem_device);")
        # Prepend before the body so the shape-pack rewrite below picks up
        # the prologue's ``arr_shape[k]`` references.
        body_lines = prologue + body_lines
    # Track for the launcher: which outputs need an ``<name>__init``
    # shadow input to be bound. Same name list the artifact uses.
    atomic_init_outputs = init_outputs if not use_packed_init_shadows else []

    # ---- Cooperative tile_cholesky prelude ---------------------------
    # If ``_translate_tile_intrinsics`` rewrote any ``tile_cholesky``
    # call to the ``_coop`` variant, we need to allocate a threadgroup-
    # memory scratch buffer at kernel scope and surface the lane id as
    # a local. Choose the largest N seen so the helper for *any* size
    # can use the same buffer.
    coop_chol_n = pre_coop_chol_n if is_coop_kernel else 0
    if is_coop_kernel:
        # ``_coop_lane`` is the 0..31 lane within the 32-thread
        # threadgroup (cooperative kernels dispatch with
        # ``threadgroup=(32, 1, 1)``). The helper uses it both for
        # work distribution and as the pivot-owner selector. The
        # ``wp_tile_chol_smem`` scratch is sized for the largest
        # square tile any cholesky_coop call will need; the
        # cooperative load/store helpers reuse the same buffer
        # since their tiles are always ≤ ``coop_chol_n × coop_chol_n``
        # in any kernel that goes cooperative.
        coop_prelude = [
            f"    threadgroup float wp_tile_chol_smem[{coop_chol_n * coop_chol_n}];",
            "    uint _coop_lane = thread_position_in_threadgroup.x;",
        ]
        body_lines = coop_prelude + body_lines
        # The launcher dispatches cooperative kernels with
        # ``grid=(32 * nworld, ...)`` and ``threadgroup=(32, 1, 1)``,
        # so 32 threads share a worldid. Rewrite worldid lookups
        # (``thread_position_in_grid.x``) to use the threadgroup
        # index — every thread in the threadgroup gets the same
        # worldid and the cooperative helpers stay in lockstep.
        body_lines = [ln.replace("thread_position_in_grid.x", "threadgroup_position_in_grid.x") for ln in body_lines]

    source = "\n".join(body_lines) + "\n"

    # ---- Pack per-array shape arrays into a single buffer ------------
    # Without this, every multi-dim array referenced via ``<name>_shape``
    # in the source would get its own ``int*`` buffer parameter (MLX
    # auto-generates one for each input that uses ``<name>_shape``, plus
    # we used to add one per multi-dim output explicitly). Metal limits a
    # kernel to 31 buffer arguments; mujoco_warp's larger kernels easily
    # exceed that. Combining all shape arrays into a single
    # ``__shapes_packed`` buffer (with each array's shape at a known
    # offset) brings us back well under the limit.
    _SHAPE_SLOT = 4  # max ndim per array — Warp arrays cap at 4D
    _shape_idx_pat = re.compile(r"\b(\w+)_shape\[([^\]]+)\]")
    _shape_arrs_seen: list[str] = []
    arg_label_set_local = {a.label for a in adj.args}
    for line in body_lines:
        for _m in _shape_idx_pat.finditer(line):
            _arr = _m.group(1)
            if _arr in arg_label_set_local and _arr not in _shape_arrs_seen:
                _shape_arrs_seen.append(_arr)
    _shape_offsets: dict[str, int] = {a: i * _SHAPE_SLOT for i, a in enumerate(_shape_arrs_seen)}

    def _replace_shape_ref(m: re.Match[str]) -> str:
        arr = m.group(1)
        if arr not in _shape_offsets:
            return m.group(0)
        return f"__shapes_packed[{_shape_offsets[arr]} + {m.group(2)}]"

    if _shape_arrs_seen:
        source = "\n".join(_shape_idx_pat.sub(_replace_shape_ref, line) for line in body_lines) + "\n"

    # ---- Drop unused kernel args ------------------------------------
    # Kernels can have many declared args that go unreferenced in the
    # emitted MSL — e.g. ``primitive_narrowphase`` with sphere-only
    # collisions never touches the ``mesh_*`` arrays, and a contact-
    # free model never writes to ``contact_*_out``. Each declared arg
    # eats a buffer slot, and Metal HW caps a kernel at 31 buffer
    # parameters. Filter ``input_args`` / ``output_args`` to just the
    # ones the body actually mentions, so we stay under the limit on
    # trivial models. The launcher skips any arg not in the artifact's
    # used-args list.
    _name_word_pat = re.compile(r"\b\w+\b")
    _used_names: set[str] = set()
    for line in source.splitlines():
        for m in _name_word_pat.finditer(line):
            _used_names.add(m.group(0))
    input_args = [a for a in input_args if a.label in _used_names]
    output_args = [a for a in output_args if a.label in _used_names]

    if not output_args:
        # Every declared output arg is unreferenced in the body — for
        # this specialisation the kernel is truly write-free (e.g.
        # ``primitive_narrowphase`` with sphere-only collisions and no
        # mesh data, all of whose write paths are behind compile-time-
        # eliminated branches). Return a no-op artifact; the launcher
        # short-circuits when it sees an empty ``input_names`` /
        # ``output_names``.
        return MetalKernelArtifact(
            name=adj.fun_name,
            source="",
            input_names=[],
            output_names=[],
            input_args=[],
            output_args=[],
            atomic_outputs=False,
            output_shape_inputs=[],
            shape_packed_arrs=(),
            shape_packed_slot=_SHAPE_SLOT,
            header="",
        )

    init_input_names = [f"{n}__init" for n in atomic_init_outputs]

    # ---- Pack int32-1D inputs into one buffer when over the slot cap --
    # Metal kernels are HW-capped at 31 buffer parameters (slots 0..30).
    # When ``input_args + output_args + init_shadows + __shapes_packed``
    # exceeds 30 we'd fail to compile. Pack every read-only 1-D int32
    # input — and every int/bool scalar input — into a single
    # ``__ints_packed`` buffer instead. Layout:
    #   [off_0, ..., off_{K-1}, scalar_0, ..., scalar_{S-1},
    #    data_arr0..., data_arr1..., ...]
    # ``off_k`` points at the start of array k's data in the buffer.
    # Saves ``(K + S) - 1`` slots when activated.
    ints_packed_arrs: list[str] = []
    ints_packed_scalars: list[str] = []
    floats_packed_arrs: list[str] = []
    shapes_buf_count = 1 if _shape_arrs_seen else 0
    total_slots = len(input_args) + len(output_args) + len(init_input_names) + shapes_buf_count
    if total_slots > 30:
        packable_arrs = [a for a in input_args if _is_int32_1d_array_arg(a)]
        packable_scalars = [a for a in input_args if _is_int_or_bool_scalar_arg(a)]
        # Pack int32-1D arrays + int/bool scalars into ``__ints_packed``
        # only when there's a net win: K + S >= 2 saves slots.
        do_int_pack = (len(packable_arrs) + len(packable_scalars)) >= 2
        # Float-element arrays go into ``__floats_packed``. The float
        # data lives in a separate buffer (different element type) but
        # the per-array offsets share ``__ints_packed`` so we only spend
        # one slot for the offset machinery regardless of how many
        # arrays we pack. Activate if packing recovers any slot — F >= 2
        # is the floor, but also needed when do_int_pack already pays
        # the slot for ``__ints_packed`` (any F >= 1 then is free).
        packable_floats = [a for a in input_args if _is_float_packable_array_arg(a)]
        do_float_pack = len(packable_floats) >= (1 if do_int_pack else 2)

        if do_int_pack or do_float_pack:
            ints_packed_arrs = [a.label for a in packable_arrs] if do_int_pack else []
            ints_packed_scalars = [a.label for a in packable_scalars] if do_int_pack else []
            floats_packed_arrs = [a.label for a in packable_floats] if do_float_pack else []
            packed_set = set(ints_packed_arrs) | set(ints_packed_scalars) | set(floats_packed_arrs)
            K = len(ints_packed_arrs)
            S = len(ints_packed_scalars)
            scalar_info: dict[str, tuple[int, bool]] = {}
            for s_idx, sa in enumerate(packable_scalars if do_int_pack else []):
                scalar_info[sa.label] = (K + s_idx, sa.type is bool)
            # Rewrite every access of a packed array, then every read of
            # a packed scalar.
            new_src_lines = []
            for line in source.splitlines(keepends=True):
                if do_int_pack:
                    for k, name in enumerate(ints_packed_arrs):
                        line = _replace_packed_int_array_access(line, name, k)
                    for s_label, (off, is_bool) in scalar_info.items():
                        repl = f"((bool)__ints_packed[{off}])" if is_bool else f"__ints_packed[{off}]"
                        line = re.sub(rf"\b{re.escape(s_label)}\b", repl, line)
                if do_float_pack:
                    for k, name in enumerate(floats_packed_arrs):
                        line = _replace_packed_float_array_access(line, name, K + S + k)
                new_src_lines.append(line)
            source = "".join(new_src_lines)
            input_args = [a for a in input_args if a.label not in packed_set]

    # Rewrite ``__init_shadow_offsets[i]`` (synthetic placeholder used
    # in the prologue) to its concrete address inside ``__ints_packed``.
    # The header layout is ``[ints_off, scalars, floats_off,
    # init_shadow_off, ...data]`` — init shadow offsets sit right after
    # the float-array offsets at slot ``K + S + F + i``.
    if use_packed_init_shadows:
        K_now = len(ints_packed_arrs)
        S_now = len(ints_packed_scalars)
        F_now = len(floats_packed_arrs)
        base_idx = K_now + S_now + F_now
        source = re.sub(
            r"__init_shadow_offsets\[(\d+)\]",
            lambda m: f"__ints_packed[{base_idx + int(m.group(1))}]",
            source,
        )

    base_input_names = [a.label for a in input_args]
    # Synthetic packed-buffer inputs the launcher fills at dispatch time.
    # Order matters — must match the input-build order in the launcher.
    extra_input_names: list[str] = []
    needs_ints_packed = (
        bool(ints_packed_arrs)
        or bool(ints_packed_scalars)
        or bool(floats_packed_arrs)
        or use_packed_init_shadows  # init_shadow_offsets live in __ints_packed
    )
    if needs_ints_packed:
        extra_input_names.append("__ints_packed")
    if floats_packed_arrs:
        extra_input_names.append("__floats_packed")
    if init_shadow_floats:
        extra_input_names.append("__init_shadows_floats")
    if init_shadow_ints:
        extra_input_names.append("__init_shadows_ints")
    if _shape_arrs_seen:
        extra_input_names.append("__shapes_packed")

    header = _build_kernel_header(source)

    return MetalKernelArtifact(
        name=adj.fun_name,
        source=source,
        input_names=base_input_names + init_input_names + extra_input_names,
        output_names=[a.label for a in output_args],
        input_args=input_args,
        output_args=output_args,
        atomic_outputs=has_atomic,
        output_shape_inputs=[],
        shape_packed_arrs=tuple(_shape_arrs_seen),
        shape_packed_slot=_SHAPE_SLOT,
        ints_packed_arrs=tuple(ints_packed_arrs),
        ints_packed_scalars=tuple(ints_packed_scalars),
        floats_packed_arrs=tuple(floats_packed_arrs),
        needs_init_barrier=bool(init_outputs),
        coop_chol_n=coop_chol_n,
        init_shadow_packed_outputs=tuple(init_outputs) if use_packed_init_shadows else (),
        init_shadow_floats=tuple(init_shadow_floats),
        init_shadow_ints=tuple(init_shadow_ints),
        header=header,
    )


def _is_array_arg_type(t) -> bool:
    """Return True if ``t`` is a ``wp.array`` family type annotation."""
    from warp._src.types import _ArrayAnnotationBase, array, indexedarray  # noqa: PLC0415

    if isinstance(t, (array, indexedarray)):
        return True
    if isinstance(t, _ArrayAnnotationBase):
        return True
    return getattr(t, "_wp_generic_type_str_", None) in ("array_t", "indexedarray_t")


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
    return _is_array_arg_type(var.type)


def _replace_packed_float_array_access(src: str, name: str, idx: int) -> str:
    """Rewrite every ``name[expr]`` to
    ``__floats_packed[__ints_packed[idx] + (expr)]``.

    Same bracket-balancing scanner as the int version. The float
    array data lives in ``__floats_packed`` (a flat float32 buffer);
    the per-array start offset shares ``__ints_packed`` (which is
    already int-typed) so we pay only one slot for the offset
    machinery regardless of how many float arrays we pack.
    """
    pat = re.compile(rf"\b{re.escape(name)}\s*\[")
    out: list[str] = []
    i = 0
    while i < len(src):
        m = pat.search(src, i)
        if m is None:
            out.append(src[i:])
            break
        out.append(src[i : m.start()])
        j = m.end()
        depth = 1
        while j < len(src) and depth > 0:
            ch = src[j]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            j += 1
        if depth != 0:
            out.append(src[m.start() :])
            break
        expr = src[m.end() : j - 1]
        out.append(f"__floats_packed[__ints_packed[{idx}] + ({expr})]")
        i = j
    return "".join(out)


def _replace_packed_int_array_access(src: str, name: str, idx: int) -> str:
    """Rewrite every ``name[expr]`` to ``__ints_packed[__ints_packed[idx] + (expr)]``.

    Walks forward balancing brackets so a nested access like
    ``name[other_arr[i]]`` is handled correctly. Doesn't touch
    ``name_shape[...]`` since the leading word boundary requires the
    next non-word char after ``name`` to be the open bracket (with only
    whitespace allowed between).
    """
    pat = re.compile(rf"\b{re.escape(name)}\s*\[")
    out: list[str] = []
    i = 0
    while i < len(src):
        m = pat.search(src, i)
        if m is None:
            out.append(src[i:])
            break
        out.append(src[i : m.start()])
        # Walk forward from just after the ``[`` to find the matching ``]``,
        # tracking depth so nested ``[...]`` doesn't terminate the scan.
        j = m.end()
        depth = 1
        while j < len(src) and depth > 0:
            ch = src[j]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            j += 1
        if depth != 0:
            out.append(src[m.start() :])
            break
        expr = src[m.end() : j - 1]
        out.append(f"__ints_packed[__ints_packed[{idx}] + ({expr})]")
        i = j
    return "".join(out)


def _is_int32_1d_array_arg(arg) -> bool:
    """True if ``arg`` is a ``wp.array(dtype=wp.int32)`` of rank 1.

    Used to pick read-only int arrays for packing into ``__ints_packed``
    when a kernel exceeds Metal's 31-buffer limit. Excludes vec/mat/
    struct-element arrays (they need larger per-element strides) and
    multi-dim arrays (they need separate offset-and-stride machinery).
    """
    if not _is_array_arg(arg):
        return False
    from warp._src.types import int32  # noqa: PLC0415

    dtype = getattr(arg.type, "dtype", None)
    ndim = getattr(arg.type, "ndim", None)
    return dtype is int32 and ndim == 1


def _is_float_packable_array_arg(arg) -> bool:
    """True if ``arg`` is a ``wp.array`` whose MLX-level storage is a
    flat ``float32`` buffer — i.e. eligible for ``__floats_packed``.

    Includes scalar-float arrays of any rank (``wp.array2d[float]``,
    ``wp.array3d[float]``) and vec/mat-element arrays whose inner type
    is float32-based (``vec3``, ``vec5``, ``mat33``, ...). MLX exposes
    these as ``(*shape, *inner_shape)`` flat ``float32`` views, so the
    kernel-side accesses already produce flat indices that the packed
    buffer can serve directly.

    Excludes int/bool element arrays (those route through the int
    packer if 1-D) and any non-array args.
    """
    if not _is_array_arg(arg):
        return False
    from warp._src.types import float32  # noqa: PLC0415

    dtype = getattr(arg.type, "dtype", None)
    if dtype is float32:
        return True
    # Vec / mat / quat element type — check the inner scalar.
    scalar_cls = getattr(dtype, "_wp_scalar_type_", None)
    return scalar_cls is float32


def _is_int_or_bool_scalar_arg(arg) -> bool:
    """True if ``arg`` is a 0-D scalar of ``int32`` or ``bool`` type.

    Each scalar input becomes its own ``constant int& <name>`` MLX
    parameter at one buffer slot apiece. When a kernel exceeds the
    31-slot cap, packing these into ``__ints_packed`` recovers one slot
    per scalar (minus the one slot the packed buffer itself takes).
    """
    if _is_array_arg(arg):
        return False
    from warp._src.types import int32  # noqa: PLC0415

    t = arg.type
    if t is int32 or t is bool:
        return True
    return False


def _msl_array_inner_msl_type(arg) -> str:
    """Return the MSL scalar name (``"float"``, ``"int"``, ``"bool"``,
    ``"uint"``) for the *inner element* of a wp.array.

    For ``wp.array(dtype=wp.float32)`` returns ``"float"``.
    For ``wp.array(dtype=wp.vec3)`` returns ``"float"`` (vec3's
    inner scalar). For ``wp.array(dtype=wp.int32)`` returns ``"int"``.
    For ``wp.array(dtype=SomeStruct)`` returns ``"float"`` (struct
    storage is float32 in our backing layout).

    Used to route init-shadow data into the right packed buffer
    (``__init_shadows_floats`` vs ``__init_shadows_ints``).
    """
    v = _vec_dtype_info(arg)
    if v is not None:
        return v[1]
    m = _mat_dtype_info(arg)
    if m is not None:
        return m[2]
    dtype = getattr(arg.type, "dtype", None)
    if dtype is None:
        return "float"
    # Struct dtypes: backing storage is float32 (see
    # ``_array_view_dtype_and_shape``).
    from warp._src.codegen import Struct  # noqa: PLC0415

    if isinstance(dtype, Struct):
        return "float"
    # Scalar dtypes — map via ``_SCALAR_CTYPE_TO_MSL``.
    ctype = getattr(dtype, "_type_", None)
    if ctype is not None:
        full = f"wp::{dtype.__name__}"
        return _SCALAR_CTYPE_TO_MSL.get(full, "float")
    name = getattr(dtype, "__name__", "")
    full = f"wp::{name}"
    return _SCALAR_CTYPE_TO_MSL.get(full, "float")


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
    if getattr(dtype, "_wp_generic_type_str_", None) not in ("vec_t", "quat_t", "transform_t"):
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
_STRUCT_FIELD_KIND_ARRAY_UNUSED = "array_unused"
"""Sentinel for ``wp.array``-typed struct fields. We allow them to exist in
the layout (so the struct's other fields can still be read/written) but
their slot has size 0: any read or write through such a field raises a
clear MetalCodegenError. Used for kernels that pass a model struct with
mesh array fields where only the primitive (scalar / vec / mat) fields
are actually accessed at runtime."""


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
    if getattr(ftype, "_wp_generic_type_str_", None) in ("vec_t", "quat_t", "transform_t"):
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
        if rows < 2 or cols < 2 or scalar_ctype not in _MSL_VEC_SCALAR_PREFIX:
            raise MetalCodegenError(f"MSL codegen does not support mat field {fname!r} of {ftype!r} in a struct")
        msl_scalar = _MSL_VEC_SCALAR_PREFIX[scalar_ctype]
        # Native ``floatNxN`` for sizes in {2, 3, 4} per dim; otherwise the
        # custom ``wp_matRxC_<scalar>`` struct (header-emitted big-mat).
        return _STRUCT_FIELD_KIND_MAT, rows * cols, _msl_mat_name(rows, cols, msl_scalar), rows, cols
    # array (1D, 2D, ..., any dtype) — tag as unused. Only kernels that
    # never actually read/write the field will codegen successfully; if
    # the field is touched, the field-pointer pass raises a clear error.
    if _is_array_arg_type(ftype):
        return _STRUCT_FIELD_KIND_ARRAY_UNUSED, 0, "<array>", 0, 0
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
    if kind in ("vec_t", "quat_t", "transform_t"):
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


_NATIVE_BUILTIN_PARAMS = (
    "uint3 thread_position_in_grid [[thread_position_in_grid]]",
    "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
    "uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]",
    "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]",
)


def _native_scalar_msl_type(scalar_ctype: str) -> str:
    """Lookup MSL scalar name for a wp scalar ctype (e.g. ``wp::float32`` -> ``float``)."""
    return _SCALAR_CTYPE_TO_MSL[scalar_ctype]


def _native_array_inner_msl(arg_var) -> str:
    """Return the MSL element type for a Warp array argument's storage."""

    from warp._src.codegen import Struct  # noqa: PLC0415

    dtype = arg_var.type.dtype
    kind = getattr(dtype, "_wp_generic_type_str_", None)
    if kind in ("vec_t", "quat_t", "transform_t", "mat_t"):
        scalar_cls = dtype._wp_scalar_type_
        scalar_ctype = f"wp::{scalar_cls.__name__}"
        return _native_scalar_msl_type(scalar_ctype)
    if isinstance(dtype, Struct):
        # Structs are exposed to the kernel as flat float32 storage —
        # the codegen rewrites field accesses to per-scalar indices.
        return "float"
    scalar_ctype = f"wp::{dtype.__name__}"
    return _native_scalar_msl_type(scalar_ctype)


def _native_scalar_param_msl(arg_var) -> str:
    """Return the MSL type for a Warp *scalar* (non-array, non-struct) argument."""
    scalar_ctype = f"wp::{arg_var.type.__name__}"
    return _native_scalar_msl_type(scalar_ctype)


def _wrap_msl_for_native_dispatch(artifact) -> str:
    """Build the full MSL kernel function string for native dispatch.

    Mirrors what ``mx.fast.metal_kernel`` wraps around the kernel body
    (header includes + ``[[kernel]] void custom_kernel_<name>(...)``
    signature with one buffer per input/output and the thread-position
    builtins). Lets the :class:`MetalDispatcher` compile the source
    directly via ``newLibraryWithSource`` without going through MLX.

    The signature uses the same arg names as ``artifact.input_names`` /
    ``output_names`` and the same packed/init-shadow conventions, so
    the body in ``artifact.source`` works unchanged. Argument-slot
    order matches MLX: real inputs, then init-shadow inputs, then
    packed buffers, then ``__shapes_packed``, then outputs — same
    order the launch path builds bindings.
    """
    from warp._src.codegen import Struct  # noqa: PLC0415

    arg_by_name = {a.label: a for a in artifact.input_args}
    out_by_name = {a.label: a for a in artifact.output_args}

    params: list[str] = []
    slot = 0

    def add_buffer(decl: str) -> None:
        nonlocal slot
        params.append(f"  {decl} [[buffer({slot})]]")
        slot += 1

    # Real inputs (kernel-signature order). Array inputs stay in
    # ``const constant`` — same qualifier MLX uses — so Apple's MSL
    # compiler reaches the same FMA / fast-math fusion decisions and
    # the dispatcher's outputs are bit-identical to the MLX path. We
    # tried ``const device`` (commit 7233d9df1's message records the
    # experiment) to dodge the address-space mismatch when an input
    # and an output bind the same MTLBuffer (mujoco_warp's
    # ``_next_position`` passes ``d.qpos`` as both ``qpos_in`` and
    # ``qpos_out``) — the resulting ``slice_view_array_store``
    # regression made the trade-off untenable. The aliasing concern
    # is handled instead by ``fastMath=False`` in the dispatcher's
    # compile options (the real fix for the qpos divergence) and
    # treating actual ``__init`` shadows as ``const device`` below.
    # Scalars / structs that don't share an MTLBuffer with anything
    # stay in ``constant``.
    for name in artifact.input_names:
        # ``__init`` shadow inputs and packed buffers are not in
        # ``arg_by_name`` — they get handled in their own sections below.
        if name in arg_by_name:
            arg_var = arg_by_name[name]
            if _is_array_arg(arg_var):
                inner = _native_array_inner_msl(arg_var)
                add_buffer(f"const constant {inner}* {name}")
            elif isinstance(arg_var.type, Struct):
                # Struct arg is serialised as flat float32 storage.
                add_buffer(f"const constant float* {name}")
            else:
                # Scalar input — match MLX's wrapping (``const constant T&``)
                # so the body's ``name`` references read as the scalar
                # value directly. The bound bytes (via ``setBytes``)
                # carry exactly one T.
                msl = _native_scalar_param_msl(arg_var)
                add_buffer(f"const constant {msl}& {name}")

    # Init-shadow inputs (``<output>__init``) — one buffer per atomic-
    # output's pre-launch contents, used by the kernel's init prologue.
    # IMPORTANT: declare these as ``const device`` (NOT ``const constant``)
    # so Apple's MSL compiler keeps them in the same address space as the
    # corresponding ``device atomic<T>*`` output. Under native dispatch
    # the shadow and the output bind to the *same* MTLBuffer (the
    # user's wp.array storage). If the wrapper put them in different
    # address spaces the compiler would assume they cannot alias and
    # cache reads/writes inconsistently — observed as outputs getting
    # silently zeroed mid-step (the init prologue's self-copy looked
    # like a no-op write that the compiler then elided). The MLX path
    # always allocates a fresh output so its shadow really does live
    # in a different buffer; only native needs the same-address-space
    # declaration.
    for name in artifact.input_names:
        if name.endswith("__init"):
            out_name = name[: -len("__init")]
            arg_var = out_by_name.get(out_name) or arg_by_name.get(out_name)
            if arg_var is None:
                add_buffer(f"const device float* {name}")
                continue
            inner = _native_array_inner_msl(arg_var)
            add_buffer(f"const device {inner}* {name}")

    # Packed buffers (variable, in launch-path order):
    #   __ints_packed                (int32)   if any int/scalar/float/init pack
    #   __floats_packed              (float)   if floats_packed_arrs
    #   __init_shadows_floats        (float)   if init_shadow_floats
    #   __init_shadows_ints          (int32)   if init_shadow_ints
    has_packed_init = bool(artifact.init_shadow_packed_outputs)
    if artifact.ints_packed_arrs or artifact.ints_packed_scalars or artifact.floats_packed_arrs or has_packed_init:
        add_buffer("const constant int* __ints_packed")
    if artifact.floats_packed_arrs:
        add_buffer("const constant float* __floats_packed")
    if artifact.init_shadow_floats:
        add_buffer("const constant float* __init_shadows_floats")
    if artifact.init_shadow_ints:
        add_buffer("const constant int* __init_shadows_ints")

    # ``__shapes_packed`` always last among inputs (when present).
    if artifact.shape_packed_arrs:
        add_buffer("const constant int* __shapes_packed")

    # Outputs.
    for name in artifact.output_names:
        arg_var = out_by_name.get(name)
        if arg_var is None:
            raise MetalCodegenError(f"Native dispatch: output {name!r} has no matching Var in artifact.output_args")
        inner = _native_array_inner_msl(arg_var)
        qual = f"device atomic<{inner}>*" if artifact.atomic_outputs else f"device {inner}*"
        add_buffer(f"{qual} {name}")

    # Builtins — always include the full set so the body's references
    # resolve regardless of whether it actually uses them. Apple's
    # compiler drops the unused ones.
    for builtin in _NATIVE_BUILTIN_PARAMS:
        params.append(f"  {builtin}")

    signature = f"[[kernel]] void custom_kernel_{artifact.name}(\n" + ",\n".join(params) + ")"
    # Strip the MLX-style output-init prologue from the kernel body.
    # Under native dispatch the output buffer *is* the user's wp.array,
    # so the buffer already holds the prior value the prologue would
    # seed. Worse, our dispatch policy is one thread per threadgroup,
    # so the prologue's ``thread_position_in_threadgroup.y == 0`` guard
    # fires in every thread and its ``threadgroup_barrier`` is a no-op
    # — every thread races to write the seed value while sibling
    # threads' atomic_adds are in flight. The observed symptom was
    # ``d.nefc`` resetting to 0 across sibling atomic-output kernels
    # (``_limit_ball`` → ``_limit_slide_hinge`` → ``_limit_tendon``),
    # so step 1's solver entered with no constraints and produced
    # large qpos drift on pendulum joints (the
    # ``test_pendula_multi_step_warmstart_drift`` failure).
    body = _strip_init_prologue(artifact.source)
    return (
        "#include <metal_stdlib>\n"
        "#include <metal_atomic>\n"
        "using namespace metal;\n"
        f"{artifact.header}\n"
        f"{signature} {{\n"
        f"{body}\n"
        "}\n"
    )


_PROLOGUE_START = "    // -- Output init prologue (seed from user wp.array data) --"
_PROLOGUE_END = "    threadgroup_barrier(metal::mem_flags::mem_device);"


def _strip_init_prologue(source: str) -> str:
    """Remove the codegen-emitted output-init prologue.

    The prologue is the contiguous block from the marker comment down
    to the post-prologue ``threadgroup_barrier``. The native dispatch
    path doesn't need it (the bound output buffer already aliases the
    user's wp.array, which carries the prior value) and it actively
    races with sibling atomic_adds under our single-thread threadgroup
    launch policy.
    """
    start = source.find(_PROLOGUE_START)
    if start == -1:
        return source
    end = source.find(_PROLOGUE_END, start)
    if end == -1:
        return source
    end += len(_PROLOGUE_END)
    # Drop the trailing newline if present so we don't leave a blank
    # line where the prologue was.
    if end < len(source) and source[end] == "\n":
        end += 1
    return source[:start] + source[end:]


def _get_or_build_metal_kernel_native(kernel):
    """Return ``(artifact, pso)`` for a Warp kernel under native dispatch.

    Caches the PSO on the kernel object so subsequent launches skip
    compilation entirely. Falls back to ``(artifact, None)`` for no-op
    kernels (all args pruned), matching the MLX path's behaviour.
    """
    from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

    artifact = getattr(kernel, "_metal_artifact", None)
    pso = getattr(kernel, "_metal_native_pso", None)
    if artifact is None:
        artifact = generate_msl_kernel(kernel)
        kernel._metal_artifact = artifact
    if pso is None:
        if not artifact.input_names and not artifact.output_names:
            kernel._metal_native_pso = "noop"
            return artifact, None
        wrapped = _wrap_msl_for_native_dispatch(artifact)
        kernel._metal_native_wrapped_source = wrapped
        pso = get_dispatcher().compile(wrapped, f"custom_kernel_{artifact.name}")
        kernel._metal_native_pso = pso
    elif pso == "noop":
        return artifact, None
    return artifact, pso


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
        # No-op kernel (every arg pruned as unused — see
        # ``generate_msl_kernel``); we don't compile a Metal kernel for
        # it. The launcher short-circuits when it sees the empty
        # artifact.
        if not artifact.input_names and not artifact.output_names:
            mlx_kernel = None
        else:
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


def _native_pack_scalar_arg(arg_var, value) -> tuple[bytes, int]:
    """Pack a Warp scalar arg into ``(bytes, length)`` for ``setBytes``.

    The kernel signature has ``const constant T& name`` so the bound
    bytes must be exactly ``sizeof(T)`` and laid out as a plain T.
    """
    import ctypes  # noqa: PLC0415

    wp_type = arg_var.type
    # ``warp.float32`` etc. expose a ctypes-compatible ``_type_``;
    # use the underlying ctype to pack.
    ct = getattr(wp_type, "_type_", None)
    if ct is None:
        # Fallback for plain ``bool`` / Python primitives.
        if wp_type is bool:
            ct = ctypes.c_bool
        elif wp_type is int:
            ct = ctypes.c_int32
        elif wp_type is float:
            ct = ctypes.c_float
        else:
            raise MetalCodegenError(f"Native dispatch: don't know how to pack scalar arg of type {wp_type!r}")
    packed = ct(value)
    return bytes(packed), ctypes.sizeof(packed)


def launch_metal_kernel_native(kernel, dim, inputs, outputs, device, block_dim: int = 256):
    """CUDA-style native Metal dispatch.

    Bypasses ``mx.fast.metal_kernel`` and dispatches directly through
    Apple's Metal API via :class:`warp._src.metal_dispatch.MetalDispatcher`.
    Kernels write *in-place* into Warp-owned ``MTLBuffer``s — same
    semantics as CUDA kernels writing into a ``cudaMalloc`` allocation.
    No fresh output allocation, no host memcpy, no per-launch
    ``mx.eval`` sync.

    Gated by :data:`warp.config.metal_native_dispatch`. Requires every
    Metal-resident ``wp.array`` involved in the launch to have been
    allocated by the native allocator path (so the registry holds an
    ``MTLBuffer`` for each ptr); :class:`MetalDefaultAllocator` does
    this automatically when the flag is set.

    This function mirrors the MLX path's argument prep — the same
    init-shadow, packed-int/float, and shapes-packed buffers — so the
    MSL source body works unchanged.
    """
    import ctypes  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from warp._src.codegen import Struct  # noqa: PLC0415
    from warp._src.context import _metal_get_buffer  # noqa: PLC0415
    from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

    if isinstance(dim, int):
        dims_check = (dim,)
    else:
        dims_check = tuple(dim)
    if any(d <= 0 for d in dims_check):
        return

    artifact, pso = _get_or_build_metal_kernel_native(kernel)
    if pso is None:
        return

    fwd_args = list(inputs) + list(outputs)
    n_kernel_args = len(kernel.adj.args)
    if len(fwd_args) != n_kernel_args:
        raise RuntimeError(
            f"Error launching kernel '{kernel.key}', passed {len(fwd_args)} arguments "
            f"but kernel requires {n_kernel_args}."
        )
    # Per-kernel metadata dicts are reused across launches — caching
    # them on the kernel object shaves ~10 µs/launch off the hot path.
    arg_by_name = getattr(kernel, "_metal_native_arg_by_name", None)
    if arg_by_name is None:
        arg_by_name = {a.label: (i, a) for i, a in enumerate(kernel.adj.args)}
        kernel._metal_native_arg_by_name = arg_by_name
        kernel._metal_native_arg_var_by_name = {a.label: a for a in artifact.input_args}
        kernel._metal_native_out_var_by_name = {a.label: a for a in artifact.output_args}
        kernel._metal_native_init_shadow_names = frozenset(n for n in artifact.input_names if n.endswith("__init"))
        # Outputs whose prior values are preserved by some init-shadow
        # mechanism — either a per-output ``<name>__init`` input, or a
        # slot in the packed shadow buffers. Atomic-output kernels that
        # accumulate across multiple sibling launches (mujoco_warp's
        # ``_limit_ball`` → ``_limit_slide_hinge`` → ``_limit_tendon``
        # chain all atomic-adding into ``d.nefc``) rely on this set to
        # suppress the dispatcher's pre-launch ``fill_zero``; without
        # the packed outputs being included, every sibling launch
        # zeroed the accumulator and step-1's solver entered with the
        # wrong constraint count (``nefc == 0`` instead of 2 for the
        # pendula limits).
        kernel._metal_native_output_init_shadow_set = frozenset(
            n[: -len("__init")] for n in artifact.input_names if n.endswith("__init")
        ) | frozenset(artifact.init_shadow_packed_outputs)
    arg_var_by_name = kernel._metal_native_arg_var_by_name
    out_var_by_name = kernel._metal_native_out_var_by_name
    init_shadow_names = kernel._metal_native_init_shadow_names
    output_init_shadow_set = kernel._metal_native_output_init_shadow_set

    dispatcher = get_dispatcher()
    bindings: list = []
    # Per-binding access mode -- used by ``MetalDispatcher`` only in
    # ICB recording mode to compute dependency chunks. For each binding
    # we record one of ``None`` (setBytes / scalar), ``"r"``, ``"w"``,
    # or ``"rw"``. Conservative defaults: real inputs and init shadows
    # are read-only; output buffers are read-write (kernels may both
    # read the init value and write, e.g. ``wp.atomic_add``); packed
    # scalar/int/float allocator slabs are read-only.
    binding_modes: list = []
    transient_refs: list = []  # MTLBuffers we allocate per-launch.

    def _resolve_mtl(value, kernel_name: str, input_name: str):
        if value.ptr is None or value.size == 0:
            # Placeholder MTLBuffer — kernel won't actually iterate (dim
            # check above) but Metal still requires a bound buffer at
            # every slot.
            buf, _ = dispatcher.alloc(4)
            transient_refs.append(buf)
            return buf
        mtl_buf = _metal_get_buffer(value.ptr)
        if mtl_buf is None:
            raise RuntimeError(
                f"Kernel '{kernel_name}' argument {input_name!r} has no registered Metal buffer "
                f"(ptr={value.ptr}). Was the array allocated with metal_native_dispatch enabled?"
            )
        # In native dispatch mode the registry holds MTLBuffer instances;
        # in MLX mode it holds mx.array. Mixing modes in one process is
        # unsupported — the dispatcher will fail later with a clearer
        # error than the per-launch ``hasattr`` probes that originally
        # lived here (each ObjC attribute lookup costs ~30 µs which
        # dominated the hot path).
        return mtl_buf

    def _alloc_buffer_with_data(np_array) -> Any:
        nb = int(np_array.nbytes)
        if nb == 0:
            buf, _ = dispatcher.alloc(4)
            transient_refs.append(buf)
            return buf
        buf, addr = dispatcher.alloc(nb)
        # Memcpy bytes into the shared-storage buffer via the CPU ptr.
        ctypes.memmove(addr, np_array.ctypes.data, nb)
        transient_refs.append(buf)
        return buf

    # init_shadow_names is already cached on the kernel above.

    # 1) Real inputs from the kernel signature.
    for name in artifact.input_names:
        if name in arg_var_by_name:
            arg_var = arg_var_by_name[name]
            idx, _ = arg_by_name[name]
            value = fwd_args[idx]
            if _is_array_arg(arg_var):
                if not getattr(value, "device", None) or not value.device.is_metal:
                    raise RuntimeError(
                        f"Kernel '{kernel.key}' argument {name!r} must be a wp.array on a Metal "
                        f"device; got {getattr(value, 'device', '?')}"
                    )
                bindings.append(_resolve_mtl(value, kernel.key, name))
                binding_modes.append("r")
            elif isinstance(arg_var.type, Struct):
                layout = _struct_layout_for(arg_var.type)
                ctype_inst = getattr(value, "_ctype", None)
                if ctype_inst is None:
                    raise RuntimeError(f"Kernel '{kernel.key}' arg {name!r}: struct value lacks ``_ctype``")
                raw = bytes(ctype_inst)
                np_buf = np.frombuffer(raw, dtype=np.float32).copy()
                if np_buf.size != layout.scalars_per_elem:
                    raise RuntimeError(
                        f"Kernel '{kernel.key}' arg {name!r}: struct serialisation produced "
                        f"{np_buf.size} float32s but layout expects {layout.scalars_per_elem}"
                    )
                bindings.append(_alloc_buffer_with_data(np_buf))
                # Struct args are read-only constants per-launch but they
                # live in a transient MTLBuffer that's not shared with
                # any other binding, so the mode is effectively None.
                binding_modes.append(None)
            else:
                # Scalar input — bind via setBytes.
                data, length = _native_pack_scalar_arg(arg_var, value)
                bindings.append((data, length))
                binding_modes.append(None)

    # 2) Init shadows — bind the corresponding output's MTLBuffer.
    for name in artifact.input_names:
        if name in init_shadow_names:
            out_name = name[: -len("__init")]
            idx, _ = arg_by_name[out_name]
            value = fwd_args[idx]
            bindings.append(_resolve_mtl(value, kernel.key, name))
            # The init shadow points at the SAME MTLBuffer as the output
            # below; mark it ``"r"`` so the dependency tracker sees this
            # command both reads (init load) and writes (output) the
            # buffer, which correctly aggregates to read-write.
            binding_modes.append("r")

    # 3) Packed buffers. Reuses the MLX path's layout logic, just writes
    # to MTLBuffers instead of mx.arrays.
    has_packed_init = bool(artifact.init_shadow_packed_outputs)
    needs_ints_packed = bool(
        artifact.ints_packed_arrs or artifact.ints_packed_scalars or artifact.floats_packed_arrs or has_packed_init
    )
    if needs_ints_packed:
        K = len(artifact.ints_packed_arrs)
        S = len(artifact.ints_packed_scalars)
        F = len(artifact.floats_packed_arrs)
        N = len(artifact.init_shadow_packed_outputs)
        # Header: [int_offsets..., scalars..., float_offsets..., init_offsets...]
        header_np = np.zeros(K + S + F + N, dtype=np.int32)
        # Int-array data follows the header.
        running = K + S + F + N
        int_data_parts: list = []
        for i, arr_name in enumerate(artifact.ints_packed_arrs):
            header_np[i] = running
            idx, _ = arg_by_name[arr_name]
            value = fwd_args[idx]
            sz = int(getattr(value, "size", 0) or 0)
            if value.ptr is None or sz == 0:
                continue
            # Pull the raw int32 bytes from the wp.array via its CPU
            # ptr (unified memory).
            src_bytes = (ctypes.c_int32 * sz).from_address(value.ptr)
            int_data_parts.append(np.frombuffer(src_bytes, dtype=np.int32).copy())
            running += sz
        for j, scalar_name in enumerate(artifact.ints_packed_scalars):
            idx, _ = arg_by_name[scalar_name]
            value = fwd_args[idx]
            header_np[K + j] = int(value)
        # Float-array offsets (relative to __floats_packed).
        float_running = 0
        float_data_parts: list = []
        for k, arr_name in enumerate(artifact.floats_packed_arrs):
            header_np[K + S + k] = float_running
            idx, _ = arg_by_name[arr_name]
            value = fwd_args[idx]
            _, view_shape = _array_view_dtype_and_shape(value)
            sz = int(np.prod(view_shape)) if view_shape else 0
            if value.ptr is None or sz == 0:
                continue
            src_bytes = (ctypes.c_float * sz).from_address(value.ptr)
            float_data_parts.append(np.frombuffer(src_bytes, dtype=np.float32).copy())
            float_running += sz
        # Init-shadow offsets (relative to __init_shadows_floats / __init_shadows_ints).
        init_floats_running = 0
        init_ints_running = 0
        init_floats_parts: list = []
        init_ints_parts: list = []
        if has_packed_init:
            for i, out_name in enumerate(artifact.init_shadow_packed_outputs):
                idx, _ = arg_by_name[out_name]
                value = fwd_args[idx]
                mx_dtype, view_shape = _array_view_dtype_and_shape(value)
                sz = int(np.prod(view_shape)) if view_shape else 0
                is_float_pack = out_name in artifact.init_shadow_floats
                if is_float_pack:
                    header_np[K + S + F + i] = init_floats_running
                else:
                    header_np[K + S + F + i] = init_ints_running
                if value.ptr is None or sz == 0:
                    continue
                if is_float_pack:
                    src_bytes = (ctypes.c_float * sz).from_address(value.ptr)
                    init_floats_parts.append(np.frombuffer(src_bytes, dtype=np.float32).copy())
                    init_floats_running += sz
                else:
                    src_bytes = (ctypes.c_int32 * sz).from_address(value.ptr)
                    init_ints_parts.append(np.frombuffer(src_bytes, dtype=np.int32).copy())
                    init_ints_running += sz

        # Build __ints_packed buffer (header + int-array data).
        if int_data_parts:
            combined = np.concatenate([header_np, *int_data_parts])
        else:
            combined = header_np
        bindings.append(_alloc_buffer_with_data(combined))
        binding_modes.append(None)  # transient per-launch slab
        # __floats_packed
        if artifact.floats_packed_arrs:
            if float_data_parts:
                bindings.append(_alloc_buffer_with_data(np.concatenate(float_data_parts)))
            else:
                bindings.append(_alloc_buffer_with_data(np.zeros(1, dtype=np.float32)))
            binding_modes.append(None)
        # __init_shadows_floats
        if artifact.init_shadow_floats:
            if init_floats_parts:
                bindings.append(_alloc_buffer_with_data(np.concatenate(init_floats_parts)))
            else:
                bindings.append(_alloc_buffer_with_data(np.zeros(1, dtype=np.float32)))
            binding_modes.append(None)
        # __init_shadows_ints
        if artifact.init_shadow_ints:
            if init_ints_parts:
                bindings.append(_alloc_buffer_with_data(np.concatenate(init_ints_parts)))
            else:
                bindings.append(_alloc_buffer_with_data(np.zeros(1, dtype=np.int32)))
            binding_modes.append(None)

    # 4) __shapes_packed.
    if artifact.shape_packed_arrs:
        slot = artifact.shape_packed_slot
        packed = np.zeros(len(artifact.shape_packed_arrs) * slot, dtype=np.int32)
        for i, arr_name in enumerate(artifact.shape_packed_arrs):
            idx, _ = arg_by_name[arr_name]
            value = fwd_args[idx]
            _, view_shape = _array_view_dtype_and_shape(value)
            for k, dim_k in enumerate(view_shape[:slot]):
                packed[i * slot + k] = dim_k
        bindings.append(_alloc_buffer_with_data(packed))
        binding_modes.append(None)

    # 5) Outputs (bind each output wp.array's MTLBuffer directly — kernel
    #    writes IN PLACE, no fresh allocation).
    #
    # Atomic-output kernels under MLX got ``init_value=0`` so their
    # output buffers started zero; the optional ``__init`` shadow
    # prologue then seeded them from the user's wp.array. We now
    # bind the user's MTLBuffer directly, so outputs would inherit
    # the wp.array's prior contents — wrong for pure-accumulator
    # kernels (atomic without ``__init`` shadow). Pre-zero those
    # specific outputs to match MLX semantics. Outputs that DO have
    # a ``__init`` shadow read it during the prologue, so prior
    # data is irrelevant; non-atomic outputs are write-every-cell
    # by Warp convention.
    # output_init_shadow_set already cached on kernel above.
    for name in artifact.output_names:
        idx, arg_var = arg_by_name[name]
        value = fwd_args[idx]
        if not _is_array_arg(arg_var):
            raise RuntimeError(f"Kernel '{kernel.key}' output {name!r} is not a wp.array")
        if not getattr(value, "device", None) or not value.device.is_metal:
            raise RuntimeError(
                f"Kernel '{kernel.key}' output {name!r} must be a wp.array on a Metal "
                f"device; got {getattr(value, 'device', '?')}"
            )
        mtl = _resolve_mtl(value, kernel.key, name)
        if artifact.atomic_outputs and name not in output_init_shadow_set and value.ptr is not None and value.size > 0:
            # Pure accumulator output — must start at zero to match
            # MLX's ``init_value=0`` behaviour for atomic kernels.
            from warp._src.types import type_size_in_bytes  # noqa: PLC0415

            nbytes = int(value.size) * type_size_in_bytes(value.dtype)
            dispatcher.fill_zero(mtl, nbytes)
        bindings.append(mtl)
        # Output buffer -- conservative read-write to handle kernels
        # that both read prior values and write (e.g. ``atomic_add``).
        # The dependency tracker aggregates ``rw`` correctly with any
        # other binding's mode on the same MTLBuffer (so this and an
        # init-shadow ``r`` on the same buffer collapse to RW).
        binding_modes.append("rw")

    # ---- Compute grid + threadgroup (mirrors MLX path's logic) ----
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
    if len(dims) > 1 and dims[-1] == block_dim and block_dim > 1:
        dims = dims[:-1]
    grid_x = dims[0]
    grid_y = dims[1] if len(dims) >= 2 else 1
    grid_z = dims[2] if len(dims) >= 3 else 1
    grid = (grid_x, grid_y, grid_z)
    if len(dims) == 1:
        tg = (min(256, grid_x), 1, 1)
    elif len(dims) == 2:
        tg = (min(256, grid_x), 1, 1)
    else:
        tg = (min(64, grid_x), 1, 1)
    if artifact.coop_chol_n > 0:
        grid_x = grid_x * 32
        grid = (grid_x, grid_y, grid_z)
        tg = (32, 1, 1)
    if artifact.needs_init_barrier and grid_y * grid_z > 1:
        if grid_y * grid_z <= 1024:
            tg = (1, grid_y, grid_z)
        # else: keep default tg and accept potential race (matches MLX path's behaviour).

    # ---- Dispatch (fire and forget) ----
    try:
        dispatcher.dispatch(pso, bindings, grid, tg, binding_modes=binding_modes)
    except Exception:
        if os.environ.get("WARP_METAL_DUMP_ON_FAIL"):
            import tempfile

            dump_dir = tempfile.mkdtemp(prefix=f"warp_metal_fail_{kernel.key}_")
            with open(os.path.join(dump_dir, "wrapped_source.metal"), "w") as f:
                f.write(getattr(kernel, "_metal_native_wrapped_source", "") or "")
            print(f"[warp-metal] kernel '{kernel.key}' failed; dumped to {dump_dir}", flush=True)
        raise

    # Diagnostic: check the guard region of every bound buffer for
    # the sentinel pattern. If anything got clobbered, that kernel
    # wrote past one of its arg buffers — first stop on an OOB hunt.
    # Guard mode is opt-in via ``WARP_METAL_CANARY=1`` (see
    # ``MetalDispatcher.alloc``); the dispatcher returns an empty
    # list and short-circuits when off.
    if os.environ.get("WARP_METAL_CANARY"):
        slot_names = list(artifact.input_names) + list(artifact.output_names)
        dispatcher.check_canaries(kernel.key, bindings, slot_names=slot_names)

    # NB: No mx.eval, no memcpy back. Outputs already live in the user's
    # wp.array MTLBuffers; subsequent kernel launches that read them
    # will queue dependent work, and the dispatcher batches everything
    # into one command buffer until a sync point (``wp.synchronize_device``,
    # ``.numpy()``, etc.).


def launch_metal_kernel(kernel, dim, inputs, outputs, device, block_dim: int = 256):
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
    # When ``metal_native_dispatch`` is enabled the native path takes over
    # entirely — kernels write in-place into Warp-owned MTLBuffers and
    # skip the per-launch ``mx.eval`` + memcpy. The MLX path stays for
    # the default (flag-off) configuration so the rollout is reversible.
    import warp.config as _wp_cfg  # noqa: PLC0415

    if _wp_cfg.metal_native_dispatch:
        return launch_metal_kernel_native(kernel, dim, inputs, outputs, device, block_dim=block_dim)

    import mlx.core as mx  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from warp._src.codegen import Struct  # noqa: PLC0415
    from warp._src.context import _metal_get_buffer, runtime  # noqa: PLC0415

    # Empty launches (any dim component is 0) produce no work, so skip
    # codegen entirely — that lets us tolerate kernels we can't yet
    # codegen (e.g. ones that exceed Metal's 31 buffer-arg HW limit) as
    # long as the model never actually runs them with non-zero dim.
    if isinstance(dim, int):
        _dims_check = (dim,)
    else:
        _dims_check = tuple(dim)
    if any(d <= 0 for d in _dims_check):
        return

    artifact, mlx_kernel = _get_or_build_metal_kernel(kernel)
    # No-op kernel (the prune step found every declared output is
    # unreferenced for this specialisation). Nothing to dispatch.
    if mlx_kernel is None:
        return

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
            mx_dtype, view_shape = _array_view_dtype_and_shape(value)
            # Empty arrays have ``ptr=None`` (Warp doesn't allocate a buffer
            # for zero-size data). MLX still needs a typed buffer of the
            # right shape bound to every kernel input — supply a fresh
            # zero-element ``mx.array`` of matching shape and dtype. The
            # kernel won't actually read from it because dim-loops over
            # the array's shape produce no iterations.
            if value.ptr is None or value.size == 0:
                import mlx.core as mx  # noqa: PLC0415

                mlx_inputs.append(mx.zeros(view_shape, dtype=mx_dtype))
                continue
            mx_buf = _metal_get_buffer(value.ptr)
            if mx_buf is None:
                raise RuntimeError(
                    f"Kernel '{kernel.key}' argument '{input_name}' has no registered MLX buffer "
                    f"(ptr={value.ptr}). Was it allocated by Warp's Metal allocator?"
                )
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

    # Append init-shadow buffers for atomic outputs. Each ``<name>__init``
    # input carries the user's current array data so the kernel prologue
    # can seed the (zero-initialised) atomic output before compute runs.
    # See ``generate_msl_kernel`` for the matching prologue emission.
    init_shadow_names = [n for n in artifact.input_names if n.endswith("__init")]
    for init_name in init_shadow_names:
        out_name = init_name[: -len("__init")]
        if out_name not in arg_by_name:
            raise RuntimeError(
                f"Kernel '{kernel.key}' atomic init shadow '{init_name}' references unknown output '{out_name}'"
            )
        idx, _ = arg_by_name[out_name]
        value = fwd_args[idx]
        if not getattr(value, "device", None) or not value.device.is_metal:
            raise RuntimeError(
                f"Kernel '{kernel.key}' atomic init shadow '{init_name}' must back a "
                f"Metal-device wp.array; got {getattr(value, 'device', '?')}"
            )
        mx_dtype, view_shape = _array_view_dtype_and_shape(value)
        if value.ptr is None or value.size == 0:
            mlx_inputs.append(mx.zeros(view_shape, dtype=mx_dtype))
            continue
        mx_buf = _metal_get_buffer(value.ptr)
        if mx_buf is None:
            raise RuntimeError(
                f"Kernel '{kernel.key}' atomic init shadow '{init_name}' has no registered MLX buffer (ptr={value.ptr})"
            )
        typed = mx_buf.view(mx_dtype).reshape(view_shape)
        mlx_inputs.append(typed)

    # Append the packed int buffer if the kernel exceeded the 31-slot
    # limit and the codegen pulled inputs into ``__ints_packed``.
    # Layout:
    #   ``[off_0, ..., off_{K-1}, scalar_0, ..., scalar_{S-1},
    #     data_arr0..., data_arr1..., ...]``
    # — first K entries are the start offset of each packed array's data,
    # next S entries are the int/bool scalar values (bool lifted to int),
    # then the concatenated data.
    # Append the packed-int buffer when any int / scalar / float-array
    # offset entries need to live there. Layout:
    #   ``[ints_off_0, ..., ints_off_{K-1},
    #      scalar_0, ..., scalar_{S-1},
    #      floats_off_0, ..., floats_off_{F-1},
    #      ints_data_arr0..., ints_data_arr1..., ...]``
    # First K entries are the int-array data start offsets within
    # ``__ints_packed``; next S entries are the int/bool scalar values
    # (bool lifted to int); next F entries are the float-array data
    # start offsets within the *separate* ``__floats_packed`` buffer
    # built below; then the concatenated int-array data.
    has_packed_init = bool(artifact.init_shadow_packed_outputs)
    if artifact.ints_packed_arrs or artifact.ints_packed_scalars or artifact.floats_packed_arrs or has_packed_init:
        K = len(artifact.ints_packed_arrs)
        S = len(artifact.ints_packed_scalars)
        F = len(artifact.floats_packed_arrs)
        # Init-shadow offsets sit after the float-arr offsets in the
        # header. ``__init_shadow_offsets[i]`` is the start position
        # (in scalars) of init shadow ``i`` within its packed buffer
        # (``__init_shadows_floats`` for float outputs,
        # ``__init_shadows_ints`` for int/bool outputs). The kernel
        # accesses it as ``__init_shadow_offsets`` in the prologue —
        # remap to ``__ints_packed[K+S+F + i]``.
        N = len(artifact.init_shadow_packed_outputs)
        header_np = np.zeros(K + S + F + N, dtype=np.int32)
        data_parts: list = []
        running = K + S + F + N
        for i, arr_name in enumerate(artifact.ints_packed_arrs):
            header_np[i] = running
            idx, _ = arg_by_name[arr_name]
            value = fwd_args[idx]
            sz = int(getattr(value, "size", 0) or 0)
            if value.ptr is None or sz == 0:
                continue
            mx_buf = _metal_get_buffer(value.ptr)
            if mx_buf is None:
                raise RuntimeError(
                    f"Kernel '{kernel.key}' packed-int array '{arr_name}' has no registered MLX buffer "
                    f"(ptr={value.ptr}). Was it allocated by Warp's Metal allocator?"
                )
            data_parts.append(mx_buf.view(mx.int32).reshape((sz,)))
            running += sz
        for j, scalar_name in enumerate(artifact.ints_packed_scalars):
            idx, _ = arg_by_name[scalar_name]
            value = fwd_args[idx]
            header_np[K + j] = int(value)
        # Compute per-float-array offsets — values populated below.
        float_running = 0
        float_data_parts: list = []
        for k, arr_name in enumerate(artifact.floats_packed_arrs):
            header_np[K + S + k] = float_running
            idx, _ = arg_by_name[arr_name]
            value = fwd_args[idx]
            mx_dtype, view_shape = _array_view_dtype_and_shape(value)
            sz = int(np.prod(view_shape)) if view_shape else 0
            if value.ptr is None or sz == 0:
                continue
            mx_buf = _metal_get_buffer(value.ptr)
            if mx_buf is None:
                raise RuntimeError(
                    f"Kernel '{kernel.key}' packed-float array '{arr_name}' has no registered MLX buffer "
                    f"(ptr={value.ptr}). Was it allocated by Warp's Metal allocator?"
                )
            float_data_parts.append(mx_buf.view(mx_dtype).reshape((sz,)))
            float_running += sz
        # Compute per-init-shadow offsets within the appropriate
        # packed buffer (``__init_shadows_floats`` or
        # ``__init_shadows_ints``).
        init_floats_running = 0
        init_ints_running = 0
        init_floats_parts: list = []
        init_ints_parts: list = []
        if has_packed_init:
            for i, out_name in enumerate(artifact.init_shadow_packed_outputs):
                idx, _ = arg_by_name[out_name]
                value = fwd_args[idx]
                mx_dtype, view_shape = _array_view_dtype_and_shape(value)
                sz = int(np.prod(view_shape)) if view_shape else 0
                is_float_pack = out_name in artifact.init_shadow_floats
                if is_float_pack:
                    header_np[K + S + F + i] = init_floats_running
                else:
                    header_np[K + S + F + i] = init_ints_running
                if value.ptr is None or sz == 0:
                    # No data to seed; the offset stays valid (no-op
                    # writes by the prologue's stride loop, since size=0).
                    continue
                mx_buf = _metal_get_buffer(value.ptr)
                if mx_buf is None:
                    raise RuntimeError(
                        f"Kernel '{kernel.key}' init-shadow output '{out_name}' has no "
                        f"registered MLX buffer (ptr={value.ptr})"
                    )
                if is_float_pack:
                    init_floats_parts.append(mx_buf.view(mx_dtype).reshape((sz,)))
                    init_floats_running += sz
                else:
                    init_ints_parts.append(mx_buf.view(mx_dtype).reshape((sz,)))
                    init_ints_running += sz
        header_mx = mx.array(header_np, dtype=mx.int32)
        if data_parts:
            mlx_inputs.append(mx.concatenate([header_mx, *data_parts], axis=0))
        else:
            mlx_inputs.append(header_mx)
        # Append the packed-float buffer separately. MLX inputs of
        # different dtypes can't share a buffer, so we keep ints in
        # ``__ints_packed`` and floats in ``__floats_packed`` — the
        # offset table in ``__ints_packed`` (slots K+S..K+S+F-1) tells
        # the kernel where each float array begins inside
        # ``__floats_packed``.
        if artifact.floats_packed_arrs:
            if float_data_parts:
                mlx_inputs.append(mx.concatenate(float_data_parts, axis=0))
            else:
                # Empty placeholder so MLX has a buffer to bind.
                mlx_inputs.append(mx.zeros((1,), dtype=mx.float32))
        # Append packed init-shadow buffers if present.
        if artifact.init_shadow_floats:
            if init_floats_parts:
                mlx_inputs.append(mx.concatenate(init_floats_parts, axis=0))
            else:
                mlx_inputs.append(mx.zeros((1,), dtype=mx.float32))
        if artifact.init_shadow_ints:
            if init_ints_parts:
                mlx_inputs.append(mx.concatenate(init_ints_parts, axis=0))
            else:
                mlx_inputs.append(mx.zeros((1,), dtype=mx.int32))

    # Append the packed shape buffer if the kernel needs any ``arr.shape``
    # access. ``__shapes_packed`` is a single flat int32 array containing
    # the runtime shape of every multi-dim array referenced from the
    # kernel body, padded so each array gets exactly ``shape_packed_slot``
    # entries. The codegen emits ``__shapes_packed[i*slot + k]`` for the
    # k-th dim of the i-th array (``i`` is the array's index in
    # ``artifact.shape_packed_arrs``).
    if artifact.shape_packed_arrs:
        slot = artifact.shape_packed_slot
        packed = np.zeros(len(artifact.shape_packed_arrs) * slot, dtype=np.int32)
        for i, arr_name in enumerate(artifact.shape_packed_arrs):
            idx, _ = arg_by_name[arr_name]
            value = fwd_args[idx]
            # Vec/mat/struct dtypes expand the inner dim in the MLX view
            # — match ``_array_view_dtype_and_shape``'s logic.
            _, mlx_view_shape = _array_view_dtype_and_shape(value)
            shape = list(mlx_view_shape)
            for k, dim_k in enumerate(shape[:slot]):
                packed[i * slot + k] = dim_k
        mlx_inputs.append(mx.array(packed, dtype=mx.int32))

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
        # MLX rejects zero-element outputs (Apple's Metal API can't bind
        # a zero-size MTLBuffer). For kernels with conditionally-unused
        # outputs (e.g. sparse-only arrays in dense mode where the
        # ``if (is_sparse)`` branch never fires), we replace any zero-
        # element output shape with a 1-element placeholder so MLX has
        # something to bind. The kernel won't actually write through the
        # placeholder, and we skip the output's memcpy below.
        if out_view_shape and any(d == 0 for d in out_view_shape):
            out_view_shape = tuple(d if d > 0 else 1 for d in out_view_shape)
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
    # ``wp.launch_tiled(dim=D, block_dim=B)`` appends ``B`` as a trailing
    # grid dim (see ``warp._src.context.launch_tiled``) and expects each of
    # the resulting ``D × B`` threads to cooperate within a tile. We lower
    # ``wp.block_dim()`` to literal ``1`` so cooperative-tile kernels run
    # serially per tile — keep only one thread per tile by collapsing the
    # appended trailing dim. Without this collapse, mujoco_warp's
    # ``linesearch_iterative`` (launched with ``block_dim=32``) runs 32
    # threads each striding through ``for dofid in range(tid, nv, 1)`` and
    # multiply-counts every dof < ``block_dim`` — observed as a 3× scaling
    # of qacc on the freejoint sphere (block_dim=32, nv=6 → dof 2 hit by
    # threads y=0,1,2).
    if len(dims) > 1 and dims[-1] == block_dim and block_dim > 1:
        dims = dims[:-1]
    grid_x = dims[0]
    grid_y = dims[1] if len(dims) >= 2 else 1
    grid_z = dims[2] if len(dims) >= 3 else 1
    grid = (grid_x, grid_y, grid_z)
    # Threadgroup policy: 1 thread on the y/z axes for cooperative-tile
    # kernels (which pack their per-block tid on y) so ``wp.block_dim()``
    # — which we lower to literal ``1`` in the IR — agrees with the
    # actual threadgroup width. The kernel's ``for tid in range(0, N, 1)``
    # loop then iterates everything serially. The launch's ``block_dim``
    # parameter is intentionally ignored here; once we implement real
    # cooperative tiles this will switch back to honoring it.
    if len(dims) == 1:
        tg = (min(256, grid_x), 1, 1)
    elif len(dims) == 2:
        tg = (min(256, grid_x), 1, 1)
    else:
        tg = (min(64, grid_x), 1, 1)

    # Cooperative-tile-Cholesky kernels: 32 threads cooperate on each
    # world's tile via threadgroup memory + ``simdgroup_barrier``-like
    # synchronisation. Inflate the x grid by 32 so every world gets a
    # 32-thread threadgroup, and remap ``wp.tid()`` (the worldid) to
    # ``threadgroup_position_in_grid.x`` in the kernel source so all
    # 32 lanes within a threadgroup see the same worldid.
    if artifact.coop_chol_n > 0:
        grid_x = grid_x * 32
        grid = (grid_x, grid_y, grid_z)
        tg = (32, 1, 1)

    # Kernels with an output-init prologue need every thread that shares
    # a worldid (the y/z grid axes) to be in the same threadgroup so the
    # post-prologue ``threadgroup_barrier`` actually waits for the
    # init-store-issuing thread. Use ``(1, grid_y, grid_z)`` per
    # threadgroup, capped at the Apple GPU's 1024 max-threads-per-
    # threadgroup limit. Falls back to the default tg (with a logged
    # warning) when the per-world thread count exceeds the limit.
    if artifact.needs_init_barrier and grid_y * grid_z > 1:
        if grid_y * grid_z <= 1024:
            tg = (1, grid_y, grid_z)
        else:
            from warp._src.utils import warn  # noqa: PLC0415

            warn(
                f"Kernel '{kernel.key}' needs an init barrier but grid_y*grid_z="
                f"{grid_y * grid_z} exceeds Metal's 1024-threads-per-threadgroup "
                "cap; init prologue may race with body. Output may be incorrect.",
                stacklevel=2,
            )

    # MLX outputs come from a buffer pool — successive launches may receive
    # buffers that previously held a different kernel's output, so any
    # output element a kernel leaves *unwritten* surfaces stale data. For
    # atomic-output kernels we seed each atomic output from the user
    # wp.array via the init prologue above. For non-atomic kernels we
    # zero-init via MLX's scalar ``init_value`` (passed below). Stateful
    # non-atomic kernels — those that read an output back to compute a new
    # value (Euler integration's ``qvel = qvel + qacc * dt``) — still need
    # init from user data; that's tracked by extending the per-output
    # init-shadow generation (see ``_init_outputs`` plumbing) once we add
    # it. For now, every output that's *only written* zero-inits, which
    # matches the CUDA backend's "fresh-launch" semantics for
    # write-everything kernels (kinematics, sensor outputs, etc.).
    init_value = 0.0 if artifact.atomic_outputs else None

    out_mx_list = mlx_kernel(
        inputs=mlx_inputs,
        grid=grid,
        threadgroup=tg,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        init_value=init_value,
    )
    if isinstance(out_mx_list, mx.array):
        out_mx_list = [out_mx_list]
    # Kick off MLX's evaluation asynchronously so the host can continue
    # issuing launches while the previous one's GPU work is still in
    # flight. The actual materialisation happens at ``np.array`` time
    # below, where any compile / launch error surfaces (and we dump the
    # failing MSL source if ``WARP_METAL_DUMP_ON_FAIL`` is set). Switching
    # from ``mx.eval`` to ``mx.async_eval`` here means consecutive
    # ``wp.launch`` calls can pipeline through MLX's stream rather than
    # serializing on per-launch GPU sync.
    try:
        mx.async_eval(out_mx_list)
    except Exception:
        if os.environ.get("WARP_METAL_DUMP_ON_FAIL"):
            import tempfile

            dump_dir = tempfile.mkdtemp(prefix=f"warp_metal_fail_{kernel.key}_")
            with open(os.path.join(dump_dir, "header.metal"), "w") as f:
                f.write(artifact.header or "")
            with open(os.path.join(dump_dir, "source.metal"), "w") as f:
                f.write(artifact.source or "")
            print(f"[warp-metal] kernel '{kernel.key}' failed; dumped to {dump_dir}", flush=True)
        raise

    # ---- Copy MLX outputs into the user's wp.array buffers ----
    # ``np.array(o_mx, copy=False)`` materialises through MLX's
    # ``__array_interface__`` — that's where any pending compile/launch
    # error will surface, so wrap the loop in the same dump-on-fail
    # handler.
    try:
        for o_mx, dest in zip(out_mx_list, output_dest_arrays, strict=True):
            # Skip zero-element user buffers — they hit the placeholder path
            # above (we ran the kernel with a 1-element MLX buffer to satisfy
            # Apple's MTLBuffer API, but the user's wp.array is genuinely
            # zero-sized and there's nothing to copy back).
            if dest.ptr is None or dest.size == 0:
                continue
            np_view = np.array(o_mx, copy=False)
            src_ptr = int(np_view.__array_interface__["data"][0])
            nbytes = np_view.nbytes
            if not runtime.core.wp_memcpy_h2h(dest.ptr, src_ptr, nbytes):
                raise RuntimeError(f"Failed to copy Metal kernel output back into wp.array (kernel '{kernel.key}')")
    except Exception:
        if os.environ.get("WARP_METAL_DUMP_ON_FAIL"):
            import tempfile

            dump_dir = tempfile.mkdtemp(prefix=f"warp_metal_fail_{kernel.key}_")
            with open(os.path.join(dump_dir, "header.metal"), "w") as f:
                f.write(artifact.header or "")
            with open(os.path.join(dump_dir, "source.metal"), "w") as f:
                f.write(artifact.source or "")
            print(f"[warp-metal] kernel '{kernel.key}' failed; dumped to {dump_dir}", flush=True)
        raise
