# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-side implementations of the ``wp.utils`` primitives on Metal.

``array_scan``, ``radix_sort_pairs``, ``segmented_sort_pairs``,
``runlength_encode``, ``array_sum`` and ``array_inner`` run as hand-written
MSL compute kernels dispatched through
:class:`warp._src.metal_dispatch.MetalDispatcher`. Every primitive is a fixed
sequence of dispatches whose sizes depend only on host-known arguments, so it
runs asynchronously (no host synchronization) and records into a Metal graph
capture exactly like a kernel launch.

Scratch storage comes from a grow-only pool of raw ``MTLBuffer``s keyed by
purpose. Reusing a pool buffer across calls is safe because the dispatcher
submits all work to one in-order queue; a buffer replaced by a larger one
stays alive while queued work or a captured graph still references it.

Algorithms:

* scan: reduce-then-scan over 2048-element tiles (256 threads x 8 items),
  recursing on the per-tile partials.
* radix sort: LSD radix sort with 8-bit digits. Per pass, a histogram kernel
  counts digits per 1024-element tile (digit-major layout), an exclusive scan
  turns counts into global offsets, and a scatter kernel ranks elements
  stably with SIMD-group ballots. Keys are 32- or 64-bit (signed and float
  keys are bit-twiddled into unsigned order), values 4 or 8 bytes.
* segmented sort: one radix sort over 64-bit composite keys
  ``(segment_start << 32) | radix(key)`` carrying the element index, then a
  gather. Segments may be listed in any order (they must not overlap). Elements outside every segment keep their own index as the
  composite prefix and therefore stay in place.
* run-length encode: head flags, inclusive scan, scatter of run heads, and a
  length kernel.
* sum / inner product: a grid-strided threadgroup reduction per output
  element and component, with a second pass over per-threadgroup partials for
  long reductions.

Only 32-bit floating-point data is handled on the device (Apple GPUs have no
``double``); ``float64`` inputs fall back to the host implementation outside
a graph capture and raise inside one.
"""

from __future__ import annotations

import struct
from typing import Any

import warp as wp

_MSL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------------
// Scan
// --------------------------------------------------------------------------

#define SCAN_TG 256u
#define SCAN_ITEMS 8u
#define SCAN_TILE 2048u

// Element (i, c) lives at ``c * cstride + i * stride`` (units of T).
struct ScanParams {
    uint n;
    uint nblocks;
    uint inclusive;
    uint op;           // 0 = sum, 1 = max
    uint has_offsets;
    uint pad0;
    ulong in_stride;
    ulong in_cstride;
    ulong out_stride;
    ulong out_cstride;
};

template <typename T> inline T scan_combine(T a, T b, uint op) { return op == 0u ? a + b : (a > b ? a : b); }

template <typename T>
kernel void wp_scan_reduce(device const T* in [[buffer(0)]],
                           device T* partials [[buffer(1)]],
                           constant ScanParams& p [[buffer(2)]],
                           uint2 tgid [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]]) {
    threadgroup T sh[SCAN_TG];
    const uint c = tgid.y;
    const uint b = tgid.x;
    const uint base = b * SCAN_TILE;
    T acc = T(0);
    for (uint k = 0; k < SCAN_ITEMS; ++k) {
        uint i = base + k * SCAN_TG + lid;
        if (i < p.n) {
            acc = scan_combine(acc, in[c * p.in_cstride + ulong(i) * p.in_stride], p.op);
        }
    }
    sh[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = SCAN_TG / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            sh[lid] = scan_combine(sh[lid], sh[lid + s], p.op);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        partials[c * p.nblocks + b] = sh[0];
    }
}

template <typename T>
kernel void wp_scan_tile(device const T* in [[buffer(0)]],
                         device T* out [[buffer(1)]],
                         device const T* offsets [[buffer(2)]],
                         constant ScanParams& p [[buffer(3)]],
                         uint2 tgid [[threadgroup_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]]) {
    threadgroup T tile[SCAN_TILE];
    threadgroup T sh[SCAN_TG];
    const uint c = tgid.y;
    const uint b = tgid.x;
    const uint base = b * SCAN_TILE;
    // Coalesced load into threadgroup memory (also makes in-place scans
    // safe: the whole tile is read before any element is written).
    for (uint k = 0; k < SCAN_ITEMS; ++k) {
        uint j = k * SCAN_TG + lid;
        uint i = base + j;
        tile[j] = i < p.n ? in[c * p.in_cstride + ulong(i) * p.in_stride] : T(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    T vals[SCAN_ITEMS];
    T run = T(0);
    for (uint k = 0; k < SCAN_ITEMS; ++k) {
        vals[k] = tile[lid * SCAN_ITEMS + k];
        run = scan_combine(run, vals[k], p.op);
    }
    sh[lid] = run;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 1u; off < SCAN_TG; off <<= 1u) {
        T t = lid >= off ? sh[lid - off] : T(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid >= off) {
            sh[lid] = scan_combine(t, sh[lid], p.op);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    T prefix = lid > 0u ? sh[lid - 1u] : T(0);
    if (p.has_offsets != 0u) {
        prefix = scan_combine(offsets[c * p.nblocks + b], prefix, p.op);
    }
    T r = prefix;
    for (uint k = 0; k < SCAN_ITEMS; ++k) {
        if (p.inclusive != 0u) {
            r = scan_combine(r, vals[k], p.op);
            tile[lid * SCAN_ITEMS + k] = r;
        } else {
            tile[lid * SCAN_ITEMS + k] = r;
            r = scan_combine(r, vals[k], p.op);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < SCAN_ITEMS; ++k) {
        uint j = k * SCAN_TG + lid;
        uint i = base + j;
        if (i < p.n) {
            out[c * p.out_cstride + ulong(i) * p.out_stride] = tile[j];
        }
    }
}

#define WP_SCAN_INST(T, NAME)                                                                                  \
    template [[host_name("wp_scan_reduce_" NAME)]] kernel void wp_scan_reduce<T>(                              \
        device const T*, device T*, constant ScanParams&, uint2, uint);                                        \
    template [[host_name("wp_scan_tile_" NAME)]] kernel void wp_scan_tile<T>(                                  \
        device const T*, device T*, device const T*, constant ScanParams&, uint2, uint);
WP_SCAN_INST(int, "int")
WP_SCAN_INST(uint, "uint")
WP_SCAN_INST(long, "long")
WP_SCAN_INST(float, "float")

// --------------------------------------------------------------------------
// Radix sort (LSD, 8-bit digits)
// --------------------------------------------------------------------------

#define RADIX_TG 256u
#define RADIX_ROUNDS 4u
#define RADIX_TILE 1024u
#define RADIX_SIMDS 8u   // RADIX_TG / 32

struct RadixParams {
    uint n;
    uint shift;
    uint bits;
    uint nblocks;
    uint kind;   // 0 = unsigned, 1 = signed, 2 = floating point
    uint pad0;
};

inline uint wp_to_radix(uint k, uint kind) {
    if (kind == 1u) return k ^ 0x80000000u;
    if (kind == 2u) return k ^ ((k >> 31) != 0u ? 0xffffffffu : 0x80000000u);
    return k;
}
inline ulong wp_to_radix(ulong k, uint kind) {
    if (kind == 1u) return k ^ 0x8000000000000000ul;
    if (kind == 2u) return k ^ ((k >> 63) != 0ul ? 0xfffffffffffffffful : 0x8000000000000000ul);
    return k;
}

template <typename K>
kernel void wp_radix_hist(device const K* keys [[buffer(0)]],
                          device uint* hist [[buffer(1)]],
                          constant RadixParams& p [[buffer(2)]],
                          uint bid [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint h[256];
    const uint ndig = 1u << p.bits;
    const uint mask = ndig - 1u;
    atomic_store_explicit(&h[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < RADIX_ROUNDS; ++r) {
        uint i = bid * RADIX_TILE + r * RADIX_TG + lid;
        if (i < p.n) {
            uint d = uint((wp_to_radix(keys[i], p.kind) >> p.shift) & K(mask));
            atomic_fetch_add_explicit(&h[d], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < ndig) {
        hist[lid * p.nblocks + bid] = atomic_load_explicit(&h[lid], memory_order_relaxed);
    }
}

inline ulong wp_ballot(bool pred) {
    return ulong(simd_vote::vote_t(simd_ballot(pred)));
}

template <typename K, typename V>
kernel void wp_radix_scatter(device const K* kin [[buffer(0)]],
                             device const V* vin [[buffer(1)]],
                             device K* kout [[buffer(2)]],
                             device V* vout [[buffer(3)]],
                             device const uint* hist [[buffer(4)]],
                             constant RadixParams& p [[buffer(5)]],
                             uint bid [[threadgroup_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sid [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint base[256];
    threadgroup uint counts[RADIX_SIMDS * 256];
    const uint ndig = 1u << p.bits;
    const uint mask = ndig - 1u;
    if (lid < ndig) {
        base[lid] = hist[lid * p.nblocks + bid];
    }
    const ulong lanemask_lt = (1ul << lane) - 1ul;
    for (uint r = 0; r < RADIX_ROUNDS; ++r) {
        for (uint j = lid; j < RADIX_SIMDS * 256u; j += RADIX_TG) {
            counts[j] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint i = bid * RADIX_TILE + r * RADIX_TG + lid;
        const bool valid = i < p.n;
        K k = K(0);
        V v = V(0);
        uint d = 0u;
        if (valid) {
            k = kin[i];
            v = vin[i];
            d = uint((wp_to_radix(k, p.kind) >> p.shift) & K(mask));
        }
        // Lanes of this SIMD group holding the same digit.
        ulong m = wp_ballot(valid);
        for (uint bit = 0; bit < p.bits; ++bit) {
            const bool set = ((d >> bit) & 1u) != 0u;
            const ulong vb = wp_ballot(set);
            m &= set ? vb : ~vb;
        }
        const uint rank = popcount(m & lanemask_lt);
        if (valid && rank == 0u) {
            counts[sid * 256u + d] = popcount(m);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid) {
            uint off = base[d] + rank;
            for (uint s = 0; s < sid; ++s) {
                off += counts[s * 256u + d];
            }
            kout[off] = k;
            vout[off] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < ndig) {
            uint tot = 0u;
            for (uint s = 0; s < RADIX_SIMDS; ++s) {
                tot += counts[s * 256u + lid];
            }
            base[lid] += tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template [[host_name("wp_radix_hist_32")]] kernel void wp_radix_hist<uint>(
    device const uint*, device uint*, constant RadixParams&, uint, uint);
template [[host_name("wp_radix_hist_64")]] kernel void wp_radix_hist<ulong>(
    device const ulong*, device uint*, constant RadixParams&, uint, uint);
#define WP_RADIX_SCATTER_INST(K, V, NAME)                                                                       \
    template [[host_name("wp_radix_scatter_" NAME)]] kernel void wp_radix_scatter<K, V>(                       \
        device const K*, device const V*, device K*, device V*, device const uint*, constant RadixParams&,     \
        uint, uint, uint, uint);
WP_RADIX_SCATTER_INST(uint, uint, "32_32")
WP_RADIX_SCATTER_INST(uint, ulong, "32_64")
WP_RADIX_SCATTER_INST(ulong, uint, "64_32")
WP_RADIX_SCATTER_INST(ulong, ulong, "64_64")

// --------------------------------------------------------------------------
// Segmented sort helpers
// --------------------------------------------------------------------------

struct SegParams {
    uint n;
    uint nseg;
    uint kind;
    uint pad0;
};

// For every non-empty segment: marks[start] = start + 1 and seg_at[start] =
// segment. An inclusive max-scan of ``marks`` then yields, per element, the
// largest non-empty segment start at or before it -- independent of the order
// in which the segments are listed.
kernel void wp_seg_mark(device const int* seg_start [[buffer(0)]],
                        device const int* seg_end [[buffer(1)]],
                        device int* marks [[buffer(2)]],
                        device int* seg_at [[buffer(3)]],
                        constant SegParams& p [[buffer(4)]],
                        uint s [[thread_position_in_grid]]) {
    if (s >= p.nseg) return;
    const int start = seg_start[s];
    const int end = seg_end[s];
    if (start >= 0 && end > start && uint(start) < p.n) {
        marks[start] = start + 1;
        seg_at[start] = int(s);
    }
}

// After the max-scan, build composite sort keys and stash the original keys /
// values in the scratch halves for the gather.
kernel void wp_seg_compose(device uint* keys [[buffer(0)]],
                           device uint* values [[buffer(1)]],
                           device const int* last_start [[buffer(2)]],
                           device const int* seg_at [[buffer(3)]],
                           device const int* seg_end [[buffer(4)]],
                           device ulong* ckeys [[buffer(5)]],
                           device uint* cvals [[buffer(6)]],
                           constant SegParams& p [[buffer(7)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const int start = last_start[i] - 1;
    uint prefix = i;
    if (start >= 0 && int(i) < seg_end[seg_at[start]]) {
        prefix = uint(start);
    }
    const uint k = keys[i];
    ckeys[i] = (ulong(prefix) << 32) | ulong(wp_to_radix(k, p.kind));
    cvals[i] = i;
    keys[p.n + i] = k;
    values[p.n + i] = values[i];
}

kernel void wp_seg_gather(device uint* keys [[buffer(0)]],
                          device uint* values [[buffer(1)]],
                          device const uint* cvals [[buffer(2)]],
                          constant SegParams& p [[buffer(3)]],
                          uint j [[thread_position_in_grid]]) {
    if (j >= p.n) return;
    const uint src = cvals[j];
    keys[j] = keys[p.n + src];
    values[j] = values[p.n + src];
}

// --------------------------------------------------------------------------
// Run-length encoding
// --------------------------------------------------------------------------

struct RleParams {
    uint n;
    uint pad0;
};

kernel void wp_rle_flags(device const int* v [[buffer(0)]],
                         device int* flags [[buffer(1)]],
                         constant RleParams& p [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    flags[i] = (i == 0u || v[i] != v[i - 1u]) ? 1 : 0;
}

kernel void wp_rle_heads(device const int* v [[buffer(0)]],
                         device const int* pos [[buffer(1)]],
                         device int* run_values [[buffer(2)]],
                         device int* starts [[buffer(3)]],
                         device int* run_count [[buffer(4)]],
                         constant RleParams& p [[buffer(5)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    if (i == 0u || v[i] != v[i - 1u]) {
        const int r = pos[i] - 1;
        run_values[r] = v[i];
        starts[r] = int(i);
    }
    if (i == p.n - 1u) {
        run_count[0] = pos[i];
    }
}

kernel void wp_rle_lengths(device const int* starts [[buffer(0)]],
                           device const int* pos [[buffer(1)]],
                           device int* run_lengths [[buffer(2)]],
                           constant RleParams& p [[buffer(3)]],
                           uint r [[thread_position_in_grid]]) {
    const uint total = uint(pos[p.n - 1u]);
    if (r >= total) return;
    const int end = r + 1u < total ? starts[r + 1u] : int(p.n);
    run_lengths[r] = end - starts[r];
}

// --------------------------------------------------------------------------
// Sum / inner product (float32)
// --------------------------------------------------------------------------

#define RED_TG 256u

// Strides are in float elements. Output ``o`` decomposes over ``shape``
// (the output shape, reduced axis = 1) into a multi-index.
struct ReduceParams {
    uint count;
    uint ncomp;
    uint groups;
    uint ndim;
    uint mode;     // 0 = sum (per component), 1 = inner product
    uint pad0;
    uint shape[4];
    long vstride;  // along the reduced axis
    long bstride;
    long vst[4];
    long bst[4];
    long ost[4];
};

inline void wp_red_offsets(constant ReduceParams& p, uint o, thread long& vo, thread long& bo, thread long& oo) {
    vo = 0; bo = 0; oo = 0;
    uint rem = o;
    for (int d = int(p.ndim) - 1; d >= 0; --d) {
        const uint idx = rem % p.shape[d];
        rem /= p.shape[d];
        vo += long(idx) * p.vst[d];
        bo += long(idx) * p.bst[d];
        oo += long(idx) * p.ost[d];
    }
}

kernel void wp_reduce_f32(device const float* v [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          constant ReduceParams& p [[buffer(3)]],
                          uint3 tgid [[threadgroup_position_in_grid]],
                          uint3 tgcount [[threadgroups_per_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sid [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sh[RED_TG / 32u];
    const uint g = tgid.x;
    const uint c = tgid.y;
    const uint o = tgid.z;
    long vo, bo, oo;
    wp_red_offsets(p, o, vo, bo, oo);
    float acc = 0.0f;
    for (uint k = g * RED_TG + lid; k < p.count; k += p.groups * RED_TG) {
        if (p.mode == 0u) {
            acc += v[vo + long(k) * p.vstride + c];
        } else {
            for (uint cc = 0; cc < p.ncomp; ++cc) {
                acc += v[vo + long(k) * p.vstride + cc] * b[bo + long(k) * p.bstride + cc];
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) sh[sid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float total = 0.0f;
        for (uint s = 0; s < RED_TG / 32u; ++s) total += sh[s];
        if (p.groups == 1u) {
            out[oo + c] = total;
        } else {
            out[(ulong(o) * tgcount.y + c) * p.groups + g] = total;
        }
    }
}

kernel void wp_reduce_partials_f32(device const float* partials [[buffer(0)]],
                                   device float* out [[buffer(1)]],
                                   constant ReduceParams& p [[buffer(2)]],
                                   constant uint& ny [[buffer(3)]],
                                   uint2 tid [[thread_position_in_grid]]) {
    const uint c = tid.x;
    const uint o = tid.y;
    long vo, bo, oo;
    wp_red_offsets(p, o, vo, bo, oo);
    float total = 0.0f;
    device const float* src = partials + (ulong(o) * ny + c) * p.groups;
    for (uint g = 0; g < p.groups; ++g) total += src[g];
    out[oo + c] = total;
}
"""

# Must match the MSL constants above.
_SCAN_TILE = 2048
_RADIX_TILE = 1024
_TG = 256

_SCAN_TYPES = {wp.int32: ("int", "i"), wp.uint32: ("uint", "I"), wp.int64: ("long", "q"), wp.float32: ("float", "f")}


class _Scratch:
    """Grow-only pool of raw shared-storage ``MTLBuffer``s keyed by purpose."""

    def __init__(self):
        self._bufs: dict[str, tuple[Any, int]] = {}

    def get(self, dispatcher, name: str, nbytes: int):
        nbytes = max(int(nbytes), 16)
        entry = self._bufs.get(name)
        if entry is None or entry[1] < nbytes:
            # Over-allocate a little so slowly growing sizes don't realloc
            # on every call.
            size = max(nbytes, (entry[1] * 3 // 2) if entry else 0)
            buf, _addr = dispatcher.alloc(size)
            entry = (buf, size)
            self._bufs[name] = entry
        return entry[0]


_scratch = _Scratch()


def _dispatcher():
    from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

    return get_dispatcher()


def _pso(entry: str, source: str | None = None):
    return _dispatcher().compile(_MSL_SOURCE if source is None else source, entry)


def _slice(arr, byte_offset: int = 0):
    """``BufferSlice`` binding for a Warp array's data (plus ``byte_offset``)."""
    from warp._src.context import _metal_find_buffer  # noqa: PLC0415
    from warp._src.metal_dispatch import BufferSlice  # noqa: PLC0415

    buf, off = _metal_find_buffer(arr.ptr)
    if buf is None:
        raise RuntimeError(
            "wp.utils on Metal requires arrays allocated by Warp's Metal allocator (or wrapped Metal buffers)"
        )
    return BufferSlice(buf, off + byte_offset)


def _bytes(fmt: str, *values) -> tuple[bytes, int]:
    data = struct.pack(fmt, *values)
    return (data, len(data))


def _dispatch(entry: str, bindings: list, modes: list, grid: tuple, tg: tuple, source: str | None = None) -> None:
    _dispatcher().dispatch(_pso(entry, source), bindings, grid, tg, binding_modes=modes)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def _scan_params(n, nblocks, inclusive, op, has_offsets, in_stride, in_cstride, out_stride, out_cstride):
    return _bytes(
        "<6I4Q", n, nblocks, int(inclusive), op, int(has_offsets), 0, in_stride, in_cstride, out_stride, out_cstride
    )


def _scan(src, dst, n, ncomp, tname, inclusive, op=0, in_stride=1, in_cstride=1, out_stride=1, out_cstride=1, level=0):
    """Scan ``n`` elements x ``ncomp`` components from binding ``src`` into
    ``dst`` (bindings may alias for an in-place scan)."""
    if n <= 0:
        return
    disp = _dispatcher()
    nblocks = _ceil_div(n, _SCAN_TILE)
    elem = 8 if tname == "long" else 4
    if nblocks == 1:
        dummy = _scratch.get(disp, "scan_dummy", 16)
        params = _scan_params(n, 1, inclusive, op, 0, in_stride, in_cstride, out_stride, out_cstride)
        _dispatch(
            f"wp_scan_tile_{tname}",
            [src, dst, dummy, params],
            ["r", "w", None, None],
            (_TG, ncomp, 1),
            (_TG, 1, 1),
        )
        return
    partials = _scratch.get(disp, f"scan_partials_{level}", nblocks * ncomp * elem)
    params = _scan_params(n, nblocks, inclusive, op, 0, in_stride, in_cstride, out_stride, out_cstride)
    _dispatch(
        f"wp_scan_reduce_{tname}", [src, partials, params], ["r", "w", None], (nblocks * _TG, ncomp, 1), (_TG, 1, 1)
    )
    # Exclusive scan of the per-tile partials (component-major layout).
    _scan(partials, partials, nblocks, ncomp, tname, False, op, 1, nblocks, 1, nblocks, level + 1)
    params = _scan_params(n, nblocks, inclusive, op, 1, in_stride, in_cstride, out_stride, out_cstride)
    _dispatch(
        f"wp_scan_tile_{tname}",
        [src, dst, partials, params],
        ["r", "w", "r", None],
        (nblocks * _TG, ncomp, 1),
        (_TG, 1, 1),
    )


def array_scan(in_array, out_array, inclusive: bool) -> None:
    scalar_type = wp._src.types.type_scalar_type(in_array.dtype)
    tname = _SCAN_TYPES[scalar_type][0]
    ncomp = wp._src.types.type_size(in_array.dtype)
    ssize = wp._src.types.type_size_in_bytes(scalar_type)
    dsize = wp._src.types.type_size_in_bytes(in_array.dtype)
    in_stride = in_array.strides[0] if in_array.ndim == 1 else dsize
    out_stride = out_array.strides[0] if out_array.ndim == 1 else dsize
    if in_stride % ssize or out_stride % ssize or in_stride < 0 or out_stride < 0:
        raise RuntimeError("array_scan() on Metal requires non-negative, scalar-aligned strides")
    _scan(
        _slice(in_array),
        _slice(out_array),
        in_array.size,
        ncomp,
        tname,
        inclusive,
        0,
        in_stride // ssize,
        1,
        out_stride // ssize,
        1,
    )


# ---------------------------------------------------------------------------
# Radix sort
# ---------------------------------------------------------------------------

_KEY_KIND = {
    wp.uint32: (32, 0),
    wp.int32: (32, 1),
    wp.float32: (32, 2),
    wp.uint64: (64, 0),
    wp.int64: (64, 1),
    wp.float64: (64, 2),
}


def _radix_sort(keys, values, n, key_bits, value_bytes, kind, begin_bit, end_bit, scratch_offset=None):
    """Sort bindings ``keys``/``values`` (each holding ``2 * n`` elements,
    the upper half scratch) on key bits ``[begin_bit, end_bit)``. The sorted
    result ends in the lower half."""
    from warp._src.metal_dispatch import BufferSlice  # noqa: PLC0415

    disp = _dispatcher()
    kbytes = key_bits // 8
    nblocks = _ceil_div(n, _RADIX_TILE)
    hist = _scratch.get(disp, "radix_hist", 256 * nblocks * 4)
    key_lo, key_hi = keys, BufferSlice(keys.buffer, keys.offset + n * kbytes)
    val_lo, val_hi = values, BufferSlice(values.buffer, values.offset + n * value_bytes)
    scatter = f"wp_radix_scatter_{key_bits}_{value_bytes * 8}"
    passes = 0
    shift = begin_bit
    src_k, src_v, dst_k, dst_v = key_lo, val_lo, key_hi, val_hi
    while shift < end_bit:
        bits = min(8, end_bit - shift)
        params = _bytes("<6I", n, shift, bits, nblocks, kind, 0)
        _dispatch(
            f"wp_radix_hist_{key_bits}", [src_k, hist, params], ["r", "w", None], (nblocks * _TG, 1, 1), (_TG, 1, 1)
        )
        _scan(hist, hist, (1 << bits) * nblocks, 1, "uint", False)
        _dispatch(
            scatter,
            [src_k, src_v, dst_k, dst_v, hist, params],
            ["r", "r", "w", "w", "r", None],
            (nblocks * _TG, 1, 1),
            (_TG, 1, 1),
        )
        src_k, src_v, dst_k, dst_v = dst_k, dst_v, src_k, src_v
        shift += bits
        passes += 1
    if passes % 2 == 1:
        disp.device_copy(key_lo.buffer, key_hi.buffer, n * kbytes, key_lo.offset, key_hi.offset)
        disp.device_copy(val_lo.buffer, val_hi.buffer, n * value_bytes, val_lo.offset, val_hi.offset)


def radix_sort_pairs(keys, values, count: int, begin_bit: int, end_bit: int) -> None:
    key_bits, kind = _KEY_KIND[keys.dtype]
    value_bytes = wp._src.types.type_size_in_bytes(values.dtype)
    _radix_sort(_slice(keys), _slice(values), count, key_bits, value_bytes, kind, begin_bit, end_bit)


# ---------------------------------------------------------------------------
# Segmented sort
# ---------------------------------------------------------------------------


def segmented_sort_pairs(keys, values, count: int, seg_start, seg_end, num_segments: int) -> None:
    from warp._src.metal_dispatch import BufferSlice  # noqa: PLC0415

    disp = _dispatcher()
    n = count
    kind = 2 if keys.dtype == wp.float32 else 1
    marks = _scratch.get(disp, "seg_marks", n * 4)
    seg_at = _scratch.get(disp, "seg_at", n * 4)
    ckeys = _scratch.get(disp, "seg_ckeys", 2 * n * 8)
    cvals = _scratch.get(disp, "seg_cvals", 2 * n * 4)
    k_slice, v_slice = _slice(keys), _slice(values)
    s_slice, e_slice = _slice(seg_start), _slice(seg_end)
    params = _bytes("<4I", n, num_segments, kind, 0)
    disp.device_fill(marks, 0, n * 4)
    _dispatch(
        "wp_seg_mark",
        [s_slice, e_slice, marks, seg_at, params],
        ["r", "r", "w", "w", None],
        (max(num_segments, 1), 1, 1),
        (min(_TG, max(num_segments, 1)), 1, 1),
    )
    _scan(marks, marks, n, 1, "int", True, op=1)
    _dispatch(
        "wp_seg_compose",
        [k_slice, v_slice, marks, seg_at, e_slice, ckeys, cvals, params],
        ["rw", "rw", "r", "r", "r", "w", "w", None],
        (n, 1, 1),
        (min(_TG, n), 1, 1),
    )
    prefix_bits = max(1, (n - 1).bit_length())
    _radix_sort(BufferSlice(ckeys, 0), BufferSlice(cvals, 0), n, 64, 4, 0, 0, 32 + prefix_bits)
    _dispatch(
        "wp_seg_gather",
        [k_slice, v_slice, cvals, params],
        ["rw", "rw", "r", None],
        (n, 1, 1),
        (min(_TG, n), 1, 1),
    )


# ---------------------------------------------------------------------------
# Run-length encoding
# ---------------------------------------------------------------------------


def runlength_encode(values, run_values, run_lengths, run_count, n: int) -> None:
    disp = _dispatcher()
    pos = _scratch.get(disp, "rle_pos", n * 4)
    starts = _scratch.get(disp, "rle_starts", n * 4)
    v = _slice(values)
    params = _bytes("<2I", n, 0)
    grid, tg = (n, 1, 1), (min(_TG, n), 1, 1)
    _dispatch("wp_rle_flags", [v, pos, params], ["r", "w", None], grid, tg)
    _scan(pos, pos, n, 1, "int", True)
    _dispatch(
        "wp_rle_heads",
        [v, pos, _slice(run_values), starts, _slice(run_count), params],
        ["r", "r", "w", "w", "w", None],
        grid,
        tg,
    )
    _dispatch("wp_rle_lengths", [starts, pos, _slice(run_lengths), params], ["r", "r", "w", None], grid, tg)


# ---------------------------------------------------------------------------
# Sum / inner product
# ---------------------------------------------------------------------------


def _reduce(mode, a, b, out, count, axis):
    """Reduce ``a`` (and ``b`` for the inner product) into ``out``."""
    disp = _dispatcher()
    ssize = 4
    ncomp = wp._src.types.type_size(a.dtype)
    if axis is None:
        ndim = 1
        shape = [1, 1, 1, 1]
        vst = bst = ost = [0, 0, 0, 0]
        vstride = bstride = ncomp
        n_out = 1
    else:
        ndim = a.ndim
        if ndim > 4:
            raise RuntimeError("array_sum()/array_inner() on Metal support at most 4 dimensions")
        out_shape = list(a.shape)
        out_shape[axis] = 1
        shape = out_shape + [1] * (4 - ndim)

        def elems(strides):
            if any(s % ssize for s in strides):
                raise RuntimeError("array_sum()/array_inner() on Metal require scalar-aligned strides")
            return [s // ssize for s in strides] + [0] * (4 - len(strides))

        vst = elems(a.strides)
        bst = elems(b.strides) if b is not None else [0, 0, 0, 0]
        ost = elems(out.strides)
        vstride = vst[axis]
        bstride = bst[axis] if b is not None else 0
        n_out = 1
        for d in out_shape:
            n_out *= d
    ny = ncomp if mode == 0 else 1
    groups = max(1, min(256, _ceil_div(count * (ncomp if mode == 1 else 1), 16 * _TG)))
    # Keep the grid bounded when there are many outputs.
    if n_out * ny * groups > 65536:
        groups = max(1, 65536 // (n_out * ny))
    params = _bytes(
        "<6I4I2q4q4q4q",
        count,
        ncomp,
        groups,
        ndim,
        mode,
        0,
        *shape,
        vstride,
        bstride,
        *vst,
        *bst,
        *ost,
    )
    a_slice = _slice(a)
    b_slice = _slice(b) if b is not None else a_slice
    out_slice = _slice(out)
    if groups == 1:
        _dispatch(
            "wp_reduce_f32", [a_slice, b_slice, out_slice, params], ["r", "r", "w", None], (_TG, ny, n_out), (_TG, 1, 1)
        )
        return
    partials = _scratch.get(disp, "reduce_partials", n_out * ny * groups * 4)
    _dispatch(
        "wp_reduce_f32",
        [a_slice, b_slice, partials, params],
        ["r", "r", "w", None],
        (groups * _TG, ny, n_out),
        (_TG, 1, 1),
    )
    _dispatch(
        "wp_reduce_partials_f32",
        [partials, out_slice, params, _bytes("<I", ny)],
        ["r", "w", None, None],
        (ny, n_out, 1),
        (1, 1, 1),
    )


def array_sum(values, out, count: int, axis) -> None:
    _reduce(0, values, None, out, count, axis)


def array_inner(a, b, out, count: int, axis) -> None:
    _reduce(1, a, b, out, count, axis)


# ---------------------------------------------------------------------------
# Dispatch policy
# ---------------------------------------------------------------------------


# Below this many elements an eager call on an idle queue runs faster on the
# host (a handful of GPU dispatches costs ~0.1-0.5 ms; the host op on unified
# memory costs microseconds). Inside a capture, or with GPU work queued (the
# host op would first have to drain it), the device path is always used.
HOST_FALLBACK_MAX_ELEMENTS = 32768


def _queue_idle(disp) -> bool:
    if disp._cmd_buf is not None:
        return False
    # MTLCommandBufferStatus: 4 = Completed, 5 = Error. Checking the newest
    # commit is enough for this heuristic (one queue, in-order completion)
    # and keeps it O(1) however many commits await the next sync.
    pending = disp._pending_commits
    return not pending or pending[-1].status() >= 4


def use_device_path(device, scalar_type=None, count: int | None = None) -> bool:
    """``True`` if a Metal ``wp.utils`` call should run on the GPU.

    Requires native dispatch (the MLX path cannot dispatch raw MSL). Data
    with a ``float64`` scalar type cannot be processed on Apple GPUs: such
    calls return ``False`` (host fallback) outside a capture and raise
    inside one. Small eager calls (``count`` below
    :data:`HOST_FALLBACK_MAX_ELEMENTS`) on an idle queue also return
    ``False``, since the host op is cheaper than the dispatches then.
    """
    import warp.config  # noqa: PLC0415

    if not warp.config.metal_native_dispatch:
        return False
    capturing = device.is_capturing
    if scalar_type is wp.float64:
        if capturing:
            raise RuntimeError(
                "wp.utils on float64 Metal arrays runs on the host (Apple GPUs have no double precision) "
                "and cannot be recorded into a Metal graph capture. Use float32 data or move the call "
                "outside the capture."
            )
        return False
    if not capturing and count is not None and count < HOST_FALLBACK_MAX_ELEMENTS and _queue_idle(_dispatcher()):
        return False
    return True


__all__ = [
    "array_inner",
    "array_scan",
    "array_sum",
    "radix_sort_pairs",
    "runlength_encode",
    "segmented_sort_pairs",
    "use_device_path",
]
