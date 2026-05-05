"""Step 2 — feature probes for Warp Metal backend bring-up.

Each probe exercises one MSL feature that Warp's codegen will need, and verifies
bit-exactly (or within fp tolerance) against NumPy.

  Probe 1: threadgroup-memory tree reduction
  Probe 2: device-side atomic_fetch_add on float (Metal 3)
  Probe 3: 2D strided access with per-row broadcast scalar

Run with:
    uv run --with mlx --with numpy --no-project experiments/metal_vector_add/probes.py
"""

from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np

# -----------------------------------------------------------------------------
# Probe 1: threadgroup reduction
# -----------------------------------------------------------------------------

_REDUCE_SOURCE = """
    threadgroup float scratch[256];

    uint tid    = thread_position_in_threadgroup.x;
    uint gid    = threadgroup_position_in_grid.x;
    uint stride = threads_per_threadgroup.x;
    uint idx    = gid * stride + tid;

    scratch[tid] = (idx < n) ? input[idx] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            scratch[tid] += scratch[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[gid] = scratch[0];
    }
"""

_THREADGROUP_SIZE = 256


def reduce_sum_metal(x: mx.array) -> float:
    n = x.size
    num_groups = (n + _THREADGROUP_SIZE - 1) // _THREADGROUP_SIZE

    kernel = mx.fast.metal_kernel(
        name="reduce_sum",
        input_names=["input", "n"],
        output_names=["output"],
        source=_REDUCE_SOURCE,
    )
    (partials,) = kernel(
        inputs=[x, mx.array(n, dtype=mx.uint32)],
        grid=(num_groups * _THREADGROUP_SIZE, 1, 1),
        threadgroup=(_THREADGROUP_SIZE, 1, 1),
        output_shapes=[(num_groups,)],
        output_dtypes=[x.dtype],
    )
    mx.eval(partials)
    return float(np.array(partials).sum(dtype=np.float64))


class ThreadgroupReductionTest(unittest.TestCase):
    def _check(self, n):
        rng = np.random.default_rng(seed=n)
        x_np = rng.standard_normal(n).astype(np.float32)
        got = reduce_sum_metal(mx.array(x_np))
        ref = float(x_np.astype(np.float64).sum())
        # tree reduction has different rounding than sequential — tolerate a few ulps
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-3)

    def test_one_threadgroup(self):
        self._check(256)

    def test_partial_threadgroup(self):
        self._check(100)

    def test_multi_threadgroup_aligned(self):
        self._check(256 * 16)

    def test_multi_threadgroup_unaligned(self):
        self._check(256 * 16 + 37)

    def test_large(self):
        self._check(1 << 20)


# -----------------------------------------------------------------------------
# Probe 2: atomic_fetch_add on float (Metal 3)
# -----------------------------------------------------------------------------

_ATOMIC_ADD_ONES_SOURCE = """
    uint tid = thread_position_in_grid.x;
    if (tid >= n) {
        return;
    }
    atomic_fetch_add_explicit(&out[0], 1.0f, memory_order_relaxed);
"""

_ATOMIC_ADD_INDEX_SOURCE = """
    uint tid = thread_position_in_grid.x;
    if (tid >= n) {
        return;
    }
    atomic_fetch_add_explicit(&out[0], float(tid), memory_order_relaxed);
"""


def _atomic_add_kernel(name: str, source: str):
    return mx.fast.metal_kernel(
        name=name,
        input_names=["n"],
        output_names=["out"],
        source=source,
        atomic_outputs=True,
    )


def _run_atomic(kernel, n: int) -> float:
    threads_per_group = min(256, n)
    grid_x = ((n + threads_per_group - 1) // threads_per_group) * threads_per_group
    (out,) = kernel(
        inputs=[mx.array(n, dtype=mx.uint32)],
        grid=(grid_x, 1, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.float32],
        init_value=0.0,  # MLX outputs are uninitialized by default
    )
    mx.eval(out)
    return float(np.array(out)[0])


class AtomicAddTest(unittest.TestCase):
    def setUp(self):
        # Confirm the output of atomic_outputs=True is zero-initialized.
        # If this assumption breaks on some MLX version, we'd need an explicit init kernel.
        self.kernel_ones = _atomic_add_kernel("atomic_ones", _ATOMIC_ADD_ONES_SOURCE)
        self.kernel_index = _atomic_add_kernel("atomic_index", _ATOMIC_ADD_INDEX_SOURCE)

    def test_each_thread_adds_one_bit_exact(self):
        # N additions of 1.0f sum to exactly N in fp32 for N <= 2^24.
        for n in (1, 64, 1024, 65536):
            with self.subTest(n=n):
                got = _run_atomic(self.kernel_ones, n)
                self.assertEqual(got, float(n), f"n={n}: got {got}, want {float(n)}")

    def test_each_thread_adds_index(self):
        # Sum of 0..n-1 == n*(n-1)/2; race-induced ordering changes fp rounding.
        # For n=65536 the partial sums approach 2^31, so fp32 ulp is ~256;
        # 1e-4 relative error is honest for race-ordered fp32 accumulation.
        for n in (1024, 4096, 65536):
            with self.subTest(n=n):
                got = _run_atomic(self.kernel_index, n)
                ref = n * (n - 1) / 2.0
                np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-3)


# -----------------------------------------------------------------------------
# Probe 3: 2D strided access with per-row broadcast
# -----------------------------------------------------------------------------

_SCALE_ROWS_SOURCE = """
    uint x = thread_position_in_grid.x;  // column
    uint y = thread_position_in_grid.y;  // row
    if (x >= W || y >= H) {
        return;
    }
    uint idx = y * W + x;
    out[idx] = inp[idx] * scale[y];
"""


def scale_rows_metal(inp: mx.array, scale: mx.array) -> mx.array:
    H, W = inp.shape
    assert scale.shape == (H,)

    kernel = mx.fast.metal_kernel(
        name="scale_rows",
        input_names=["inp", "scale", "H", "W"],
        output_names=["out"],
        source=_SCALE_ROWS_SOURCE,
    )
    tg_x = min(16, W)
    tg_y = min(16, H)
    grid_x = ((W + tg_x - 1) // tg_x) * tg_x
    grid_y = ((H + tg_y - 1) // tg_y) * tg_y
    (out,) = kernel(
        inputs=[inp, scale, mx.array(H, dtype=mx.uint32), mx.array(W, dtype=mx.uint32)],
        grid=(grid_x, grid_y, 1),
        threadgroup=(tg_x, tg_y, 1),
        output_shapes=[inp.shape],
        output_dtypes=[inp.dtype],
    )
    return out


class Strided2DTest(unittest.TestCase):
    def _check(self, H, W):
        rng = np.random.default_rng(seed=H * 1000 + W)
        inp_np = rng.standard_normal((H, W)).astype(np.float32)
        scale_np = rng.standard_normal(H).astype(np.float32)

        out = scale_rows_metal(mx.array(inp_np), mx.array(scale_np))
        mx.eval(out)
        got = np.array(out)
        ref = inp_np * scale_np[:, None]
        np.testing.assert_allclose(got, ref, rtol=0, atol=0)

    def test_aligned(self):
        self._check(16, 16)

    def test_rectangular(self):
        self._check(8, 64)

    def test_unaligned(self):
        self._check(17, 33)

    def test_large(self):
        self._check(512, 1024)


if __name__ == "__main__":
    print(f"Metal avail : {mx.metal.is_available()}")
    print(f"Default dev : {mx.default_device()}")
    print()
    unittest.main(verbosity=2, exit=True)
