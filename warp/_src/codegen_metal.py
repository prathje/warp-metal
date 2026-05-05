# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental MSL code generator for Warp's Metal backend.

This module is a thin translator that consumes Warp's pre-emitted typed-IR
strings (``adj.blocks[0].body_forward``) and rewrites them as Metal Shading
Language. It does *not* re-walk the Python AST — it post-processes the
CUDA-flavoured C++ statements that ``codegen.py`` already produces.

Currently supported (anything else raises ``MetalCodegenError`` with a
pointer to the offending statement):
- 1-D ``wp.array`` args of scalar dtype (``float16/32``, ``int8/16/32/64``,
  ``uint8/16/32/64``, ``bool``)
- ``wp.tid()`` (1-D dispatch)
- Address-of, load, store on 1-D arrays
- Scalar arithmetic intrinsics: ``add``, ``sub``, ``mul``, ``div``, ``mod``
- ``if`` / ``else`` blocks and the comparison operators ``<``, ``<=``,
  ``==``, ``!=``, ``>=``, ``>`` (Warp's IR pre-emits these in plain C/MSL
  syntax, so they pass through the regex-based translator unchanged).
- A whitelist of math builtins listed in ``_MATH_BUILTIN_NAMES`` —
  ``sqrt``, ``abs``, ``min``, ``max``, ``floor``, ``ceil``, ``exp``,
  ``log``, ``sin``, ``cos``, ``tanh``, etc. — translated to MSL's
  ``metal::`` namespace.
- ``for i in range(...)`` loops, both static (Warp unrolls them, so this
  is a no-op) and dynamic (a structural pre-pass rewrites Warp's
  ``goto``-based loop into a real MSL ``for``).
- ``wp::assign(dst, src)`` (in-place mutation, used by the loop body for
  accumulator updates).
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

# Pointer ctypes have a ``*`` suffix; address-space qualifier in MSL is
# ``device`` for buffer-resident memory (the only kind we currently allocate).
_POINTER_ADDRESS_SPACE = "device"


def _msl_scalar_type(ctype: str) -> str:
    """Translate a Warp scalar ctype string to its MSL equivalent.

    Raises ``MetalCodegenError`` for types MSL cannot represent natively
    (notably ``wp::float64`` — Apple Silicon GPUs have no double-precision
    floating point support).
    """
    if ctype == "wp::float64":
        raise MetalCodegenError("MSL has no native float64; double-precision kernels cannot be lowered to Metal")
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
)


# Patterns are applied in order. Each entry is (regex, replacement). Captures
# can be back-referenced with \1, \2, etc.
_INTRINSIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # wp.tid() -> thread_position_in_grid.x (we only support 1-D dispatch)
    (re.compile(r"\bbuiltin_tid1d\s*\(\s*\)"), "(int)thread_position_in_grid.x"),
    # wp::address(arr, idx) -> &arr[idx]  (used only if the dataflow collapse
    # in ``generate_msl_kernel`` didn't fold it away — e.g. for non-arg
    # array locals, which we don't currently support but emit a recognisable
    # form for.)
    (re.compile(r"wp::address\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"), r"&\1[\2]"),
    # wp::load(X) -> X
    # The dataflow collapse in ``generate_msl_kernel`` rewrites all our
    # ``wp::load`` operands from raw pointer locals to subscript expressions
    # (``arr[idx]``), so the load is logically a no-op — the value is already
    # there. If a non-collapsed load slips through, it'll fail the
    # ``_check_no_unsupported_intrinsics`` guard downstream.
    (re.compile(r"wp::load\s*\(\s*([^()]+?)\s*\)"), r"\1"),
    # wp::array_store(arr, idx, val) -> arr[idx] = val
    (
        re.compile(r"wp::array_store\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)"),
        r"\1[\2] = \3",
    ),
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

    # Preprocess: rewrite dynamic-range goto-loops into ``for`` loops. Static
    # ranges are pre-unrolled by Warp and pass through untouched. Returns a
    # transformed line list plus a set of Warp local labels whose top-level
    # declaration should be suppressed (the iter-state object and the
    # induction variable, which the generated ``for`` declares inline).
    forward_lines, vars_to_skip_decl = _preprocess_for_loops(adj.blocks[0].body_forward)

    # Classify each array arg as input or output by scanning the IR strings.
    # MLX inputs are ``const device T*`` (read-only) — verified empirically —
    # so any array that is written through must become an MLX output. We can't
    # rely on ``arg.is_write`` here because it is only populated when
    # ``verify_autograd_array_access`` is enabled.
    written_arg_names: set[str] = set()
    for raw in forward_lines:
        # Pattern: wp::array_store(var_<argname>, ...)
        m = re.match(r"\s*wp::array_store\s*\(\s*var_([A-Za-z_]\w*)", raw)
        if m:
            written_arg_names.add(m.group(1))

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
    # Warp's IR splits an ``arr[i]`` read into ``addr = wp::address(arr, i);``
    # followed by ``val = wp::load(addr);``. The intermediate ``addr`` is a
    # typed pointer in CUDA C++ but in MSL it would inherit a specific address
    # space (``const constant`` for inputs vs ``device`` for outputs) that we
    # can't easily express in a separately-declared local. Collapsing the
    # ``address``/``load`` pair into a direct subscript sidesteps the issue
    # entirely and yields cleaner MSL.
    subscript_map: dict[str, str] = {}  # var label -> "arr[idx]" string
    # Warp local labels are integers (``var_0``, ``var_1``); arg labels start
    # with a letter (``var_a``). Both are valid ``\w+`` so use that.
    address_pattern = re.compile(r"\s*var_(\w+)\s*=\s*wp::address\s*\(\s*var_(\w+)\s*,\s*var_(\w+)\s*\)\s*;\s*$")
    for raw in forward_lines:
        m = address_pattern.match(raw)
        if m:
            local_label, arr_arg, idx_label = m.group(1), m.group(2), m.group(3)
            # Only collapse if the array reference is one of the kernel args —
            # otherwise the pointer source might be something more complex.
            if any(arr_arg == a.label for a in adj.args):
                subscript_map[local_label] = f"{arr_arg}[var_{idx_label}]"

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
    for raw in forward_lines:
        line = _strip_comments_and_directives(raw)
        if line is None:
            continue
        # Skip the now-redundant address lines.
        if address_pattern.match(raw):
            continue
        translated = _translate_intrinsics(line.strip())
        # Inline subscripts for any var that became part of subscript_map.
        # Replace ``wp::load(var_X)`` patterns with the array subscript first,
        # then any bare ``var_X`` reference.
        for local_label, subscript in subscript_map.items():
            translated = re.sub(rf"\bvar_{re.escape(local_label)}\b", subscript, translated)
        # Re-run intrinsic translation in case wp::load(var_1) became wp::load(arr[idx]).
        translated = _translate_intrinsics(translated)
        # Rename ``var_<argname>`` -> ``<argname>`` so the body matches MLX's
        # generated function signature (which uses the names from
        # ``input_names``/``output_names`` directly).
        for arg in adj.args:
            translated = re.sub(rf"\bvar_{re.escape(arg.label)}\b", arg.label, translated)
        _check_no_unsupported_intrinsics(translated)
        body_lines.append(f"    {translated}")

    source = "\n".join(body_lines) + "\n"

    return MetalKernelArtifact(
        name=adj.fun_name,
        source=source,
        input_names=[a.label for a in input_args],
        output_names=[a.label for a in output_args],
        input_args=input_args,
        output_args=output_args,
    )


def _is_array_arg(var) -> bool:
    """Return True if a kernel arg's type is a ``wp.array`` family."""
    # Lazy import to avoid import cycle with warp._src.types.
    from warp._src.types import array, indexedarray  # noqa: PLC0415

    t = var.type
    if isinstance(t, array):
        return True
    if isinstance(t, indexedarray):
        return True
    # ``wp.array`` annotations live as classes too:
    return getattr(t, "_wp_generic_type_str_", None) in ("array_t", "indexedarray_t")


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

    # ---- Build MLX inputs in artifact.input_names order ----
    mlx_inputs: list = []
    for input_name in artifact.input_names:
        idx, arg_var = arg_by_name[input_name]
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
            mx_dtype = _wp_dtype_to_mx_dtype(value.dtype)
            # ``view`` reinterprets bytes (no copy); ``reshape`` flattens / shapes for MLX.
            typed = mx_buf.view(mx_dtype).reshape(value.shape)
            mlx_inputs.append(typed)
        else:
            # Scalar input — convert to a 0-D mx.array literal.
            mx_dtype = _wp_dtype_to_mx_dtype(arg_var.type)
            mlx_inputs.append(mx.array(value, dtype=mx_dtype))

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
        output_shapes.append(value.shape)
        output_dtypes.append(_wp_dtype_to_mx_dtype(value.dtype))
        output_dest_arrays.append(value)

    # ---- Compute grid ----
    if isinstance(dim, int):
        total = dim
    else:
        total = 1
        for d in dim:
            total *= d
    if total <= 0:
        return
    threadgroup_size = min(256, total)

    out_mx_list = mlx_kernel(
        inputs=mlx_inputs,
        grid=(total, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
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
