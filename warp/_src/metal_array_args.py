# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Indexed and non-contiguous array kernel arguments on Metal.

The MSL translator addresses every array argument as a dense row-major
buffer (flat indices are built from the array's shape), so a strided or
broadcast ``wp.array`` view, or a ``wp.indexedarray``, cannot be bound
directly. :func:`stage_launch_args` makes such launches work by staging:

1. each affected argument is gathered into a contiguous temporary on the
   device (strides and per-dimension index arrays are resolved by a small
   MSL kernel);
2. the kernel runs on the temporaries (kernels declared with
   ``wp.indexedarray`` parameters are re-specialized with plain
   ``wp.array`` parameters of the same dtype and dimensionality);
3. arguments the kernel writes are scattered back through the same
   strides / indices.

Reverse-mode launches stage the paired gradient arrays the same way (gather
before, scatter back after). All steps are ordinary dispatches, so a staged launch is asynchronous and
records into a Metal graph capture like any other launch.

A staged argument whose memory overlaps another array argument, where
either side is written by the kernel, is rejected: the kernel would update
the staged copy and the other argument independently and the scatter-back
would discard one side's writes. Writes through an argument that aliases
itself (duplicate indices or zero strides) keep the value of the last
scattered element instead of racing.
"""

from __future__ import annotations

from copy import copy as shallowcopy

import warp as wp
from warp._src import metal_utils as _mu

_MSL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// Element e of an N-d (N <= 4) view: decompose e over ``shape`` (row-major),
// map each coordinate through its index array when present, and offset by
// ``strides`` (bytes). ``dir`` 0 gathers view -> dense, 1 scatters back.
struct StageParams {
    uint ndim;
    uint idx_mask;
    uint elem_units;  // copy units per element (4-byte words or bytes)
    uint dir;
    uint shape[4];
    long strides[4];
};

inline long wp_stage_offset(constant StageParams& p, uint e,
                            device const int* i0, device const int* i1,
                            device const int* i2, device const int* i3) {
    device const int* idx[4] = {i0, i1, i2, i3};
    long off = 0;
    uint rem = e;
    for (int d = int(p.ndim) - 1; d >= 0; --d) {
        const uint c = rem % p.shape[d];
        rem /= p.shape[d];
        const long k = ((p.idx_mask >> d) & 1u) ? long(idx[d][c]) : long(c);
        off += k * p.strides[d];
    }
    return off;
}

template <typename U>
kernel void wp_stage_copy(device U* view [[buffer(0)]],
                          device U* dense [[buffer(1)]],
                          device const int* i0 [[buffer(2)]],
                          device const int* i1 [[buffer(3)]],
                          device const int* i2 [[buffer(4)]],
                          device const int* i3 [[buffer(5)]],
                          constant StageParams& p [[buffer(6)]],
                          uint tid [[thread_position_in_grid]]) {
    const uint e = tid / p.elem_units;
    const uint u = tid % p.elem_units;
    const long off = wp_stage_offset(p, e, i0, i1, i2, i3) / long(sizeof(U)) + long(u);
    if (p.dir == 0u) {
        dense[tid] = view[off];
    } else {
        view[off] = dense[tid];
    }
}

template [[host_name("wp_stage_copy_4")]] kernel void wp_stage_copy<uint>(
    device uint*, device uint*, device const int*, device const int*, device const int*, device const int*,
    constant StageParams&, uint);
template [[host_name("wp_stage_copy_1")]] kernel void wp_stage_copy<uchar>(
    device uchar*, device uchar*, device const int*, device const int*, device const int*, device const int*,
    constant StageParams&, uint);
"""


def _needs_staging(value) -> bool:
    if isinstance(value, wp._src.types.indexedarray):
        return value.data is not None
    if isinstance(value, wp._src.types.array):
        return not value.is_contiguous and value.size > 0 and value.ptr is not None
    return False


def _view_layout(value):
    """``(base_array, strides, index_arrays)`` describing ``value``'s elements."""
    if isinstance(value, wp._src.types.indexedarray):
        data = value.data
        indices = [value.indices[d] if d < data.ndim else None for d in range(data.ndim)]
        return data, list(data.strides), indices
    return value, list(value.strides), [None] * value.ndim


def _stage_copy(value, dense, direction: int) -> None:
    from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

    base, strides, indices = _view_layout(value)
    ndim = len(value.shape)
    if ndim > 4:
        raise RuntimeError("Metal launches support strided / indexed arrays of at most 4 dimensions")
    elem_bytes = wp._src.types.type_size_in_bytes(value.dtype)
    view_slice = _mu._slice(base)
    word = elem_bytes % 4 == 0 and view_slice.offset % 4 == 0 and all(s % 4 == 0 for s in strides)
    unit = 4 if word else 1
    elem_units = elem_bytes // unit
    idx_mask = 0
    idx_bindings = []
    disp = get_dispatcher()
    dummy = _mu._scratch.get(disp, "stage_dummy_idx", 16)
    for d in range(4):
        idx = indices[d] if d < len(indices) else None
        if idx is not None:
            idx_mask |= 1 << d
            idx_bindings.append(_mu._slice(idx))
        else:
            idx_bindings.append(dummy)
    shape = list(value.shape) + [1] * (4 - ndim)
    full_strides = strides + [0] * (4 - len(strides))
    params = _mu._bytes("<4I4I4q", ndim, idx_mask, elem_units, direction, *shape, *full_strides)
    n = value.size * elem_units
    view_mode, dense_mode = ("r", "w") if direction == 0 else ("w", "r")
    _mu._dispatch(
        f"wp_stage_copy_{unit}",
        [view_slice, _mu._slice(dense), *idx_bindings, params],
        [
            view_mode,
            dense_mode,
            "r" if idx_mask & 1 else None,
            "r" if idx_mask & 2 else None,
            "r" if idx_mask & 4 else None,
            "r" if idx_mask & 8 else None,
            None,
        ],
        (n, 1, 1),
        (min(256, n), 1, 1),
        source=_MSL_SOURCE,
    )


def _is_dense(value) -> bool:
    return isinstance(value, wp._src.types.array) and value.is_contiguous


def metal_device_strided_copy(dest, src) -> bool:
    """``wp.copy`` between two Metal arrays of equal shape where at least one
    is strided or indexed, as device dispatches of the staging kernel.

    Returns ``False`` (caller falls back to the host strided copy) for
    anything other than ``wp.array`` / ``wp.indexedarray`` views of up to
    four dimensions. The dispatches order after pending kernels on the same
    queue and record into an active Metal graph capture, so the copy replays
    with the graph (the host path cannot).
    """
    from warp._src.context import runtime  # noqa: PLC0415

    kinds = (wp._src.types.array, wp._src.types.indexedarray)
    if not isinstance(src, kinds) or not isinstance(dest, kinds):
        return False
    if src.ndim > 4 or dest.ndim > 4 or src.size == 0:
        return src.size == 0
    for v in (src, dest):
        base = v.data if isinstance(v, wp._src.types.indexedarray) else v
        if base is None or base.ptr is None:
            return False
    device = dest.device
    if _is_dense(dest):
        _stage_copy(src, dest, 0)
    elif _is_dense(src):
        _stage_copy(dest, src, 1)
    else:
        tmp = wp.empty(src.shape, dtype=src.dtype, device=device)
        if device.is_capturing:
            runtime._metal_capture_graph._metal_retain.append(tmp)
        _stage_copy(src, tmp, 0)
        _stage_copy(dest, tmp, 1)
    return True


def _array_variant(kernel):
    """``kernel`` re-specialized with ``wp.array`` in place of every
    ``wp.indexedarray`` parameter (cached on ``kernel``)."""
    variant = kernel.__dict__.get("_metal_array_variant")
    if variant is not None:
        return variant
    annotations = {}
    changed = False
    for arg in kernel.adj.args:
        t = arg.type
        if isinstance(t, wp._src.types.indexedarray):
            t = wp._src.types.array(dtype=t.dtype, ndim=t.ndim)
            changed = True
        annotations[arg.label] = t
    if not changed:
        variant = kernel
    else:
        variant = shallowcopy(kernel)
        # Drop per-kernel Metal launch caches inherited from the original.
        for name in [k for k in variant.__dict__ if k.startswith("_metal")]:
            del variant.__dict__[name]
        variant.adj = wp._src.codegen.Adjoint(kernel.func, annotations, source=kernel.adj.source)
        variant.is_generic = False
        variant.overloads = {}
    kernel._metal_array_variant = variant
    return variant


def _byte_range(value):
    """Conservative ``[lo, hi)`` address range an array argument can touch."""
    base = value.data if isinstance(value, wp._src.types.indexedarray) else value
    if base is None or not base.ptr or base.size == 0:
        return None
    lo = hi = base.ptr
    for dim, stride in zip(base.shape, base.strides):
        extent = (dim - 1) * stride
        if extent < 0:
            lo += extent
        else:
            hi += extent
    return lo, hi + wp._src.types.type_size_in_bytes(base.dtype)


def _check_staged_aliasing(kernel, values, staged_indices, written) -> None:
    args = kernel.adj.args
    ranges = []
    for i, v in enumerate(values):
        if isinstance(v, (wp._src.types.array, wp._src.types.indexedarray)):
            r = _byte_range(v)
            if r is not None:
                ranges.append((i, r))
    for i, (lo, hi) in ranges:
        if i not in staged_indices:
            continue
        for j, (lo2, hi2) in ranges:
            if j == i or lo >= hi2 or lo2 >= hi:
                continue
            if args[i].label in written or args[j].label in written:
                raise RuntimeError(
                    f"Kernel '{kernel.key}': strided/indexed argument {args[i].label!r} overlaps the memory of "
                    f"argument {args[j].label!r}, and one of them is written. On Metal such arguments run "
                    "through a contiguous staging copy, so writes through one would not be seen through the "
                    "other (and could be lost on scatter-back). Pass non-overlapping arrays, or make the "
                    "strided argument contiguous first."
                )


def _written_arg_labels(kernel, block_dim) -> frozenset:
    labels = kernel.__dict__.get("_metal_staged_written")
    if labels is None:
        from warp._src.codegen_metal import generate_msl_kernel  # noqa: PLC0415

        artifact = generate_msl_kernel(kernel)
        labels = frozenset(artifact.output_names)
        kernel._metal_staged_written = labels
    return labels


class StagedLaunch:
    """A launch whose strided / indexed arguments were replaced by
    contiguous temporaries; :meth:`run` launches and scatters back."""

    def __init__(self, kernel, inputs, outputs, staged, adj_inputs=(), adj_outputs=(), adj_staged=()):
        self.kernel = kernel
        self.inputs = inputs
        self.outputs = outputs
        self.adj_inputs = adj_inputs
        self.adj_outputs = adj_outputs
        self._staged = staged  # [(arg_index, original, temporary)]
        self._adj_staged = adj_staged  # [(original, temporary)], all scattered back

    def run(self, launch_fn, block_dim=None):
        result = launch_fn(self.kernel, self.inputs, self.outputs, self.adj_inputs, self.adj_outputs)
        written = _written_arg_labels(self.kernel, block_dim)
        args = self.kernel.adj.args
        for i, original, tmp in self._staged:
            if args[i].label in written:
                _stage_copy(original, tmp, 1)
        # Gradient arrays are read-modify-written by the adjoint kernel
        # (accumulated, or consumed and zeroed for stored outputs).
        for original, tmp in self._adj_staged:
            _stage_copy(original, tmp, 1)
        return result


def stage_launch_args(
    kernel, inputs, outputs, device, adjoint: bool = False, adj_inputs=(), adj_outputs=()
) -> StagedLaunch | None:
    """Return a :class:`StagedLaunch` if any argument needs staging, else ``None``.

    In reverse mode the gradient arrays paired with the arguments are staged
    too: gathered before the adjoint launch and scattered back after it.
    """
    values = [*inputs, *outputs]
    adj_values = [*adj_inputs, *adj_outputs] if adjoint else []
    if not any(_needs_staging(v) for v in (*values, *adj_values)):
        return None
    from warp._src.context import runtime  # noqa: PLC0415

    variant = _array_variant(kernel)
    staged_indices = {i for i, v in enumerate(values) if _needs_staging(v)}
    _check_staged_aliasing(variant, values, staged_indices, _written_arg_labels(variant, None))
    graph = runtime._metal_capture_graph if device.is_capturing else None

    def _stage(value):
        tmp = wp.empty(value.shape, dtype=value.dtype, device=device)
        if graph is not None:
            graph._metal_retain.append(tmp)
        _stage_copy(value, tmp, 0)
        return tmp

    staged = []
    new_values = list(values)
    for i, value in enumerate(values):
        if not _needs_staging(value):
            continue
        tmp = _stage(value)
        new_values[i] = tmp
        staged.append((i, value, tmp))
    adj_staged = []
    new_adj = list(adj_values)
    for i, value in enumerate(adj_values):
        if not _needs_staging(value):
            continue
        tmp = _stage(value)
        new_adj[i] = tmp
        adj_staged.append((value, tmp))
    n_in = len(inputs)
    n_adj_in = len(adj_inputs) if adjoint else 0
    return StagedLaunch(
        variant,
        new_values[:n_in],
        new_values[n_in:],
        staged,
        adj_inputs=tuple(new_adj[:n_adj_in]) if adjoint else adj_inputs,
        adj_outputs=tuple(new_adj[n_adj_in:]) if adjoint else adj_outputs,
        adj_staged=adj_staged,
    )
