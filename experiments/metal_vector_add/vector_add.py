"""Standalone MLX/Metal vector-add prototype.

Step 1 of the Warp Metal backend bring-up: prove that we can JIT-compile a
hand-written MSL kernel via mlx.core.fast.metal_kernel, dispatch it on the
GPU, read back results, and verify them against a NumPy reference.

Run with:
    uv run --with mlx --with numpy --no-project experiments/metal_vector_add/vector_add.py
"""

from __future__ import annotations

import time
import unittest

import mlx.core as mx
import numpy as np

_VECTOR_ADD_SOURCE = """
    uint tid = thread_position_in_grid.x;
    if (tid >= n) {
        return;
    }
    c[tid] = a[tid] + b[tid];
"""


def _build_kernel():
    return mx.fast.metal_kernel(
        name="vector_add",
        input_names=["a", "b", "n"],
        output_names=["c"],
        source=_VECTOR_ADD_SOURCE,
    )


def vector_add_metal(a: mx.array, b: mx.array) -> mx.array:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch: {a.dtype} vs {b.dtype}")

    n = a.size
    threads_per_group = min(256, n)
    kernel = _build_kernel()
    (out,) = kernel(
        inputs=[a, b, mx.array(n, dtype=mx.uint32)],
        grid=(n, 1, 1),
        threadgroup=(threads_per_group, 1, 1),
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
    )
    return out


class VectorAddTest(unittest.TestCase):
    def _assert_matches_numpy(self, n: int, dtype):
        rng = np.random.default_rng(seed=0xC0FFEE ^ n)
        a_np = rng.standard_normal(n).astype(dtype)
        b_np = rng.standard_normal(n).astype(dtype)

        a_mx = mx.array(a_np)
        b_mx = mx.array(b_np)

        c_mx = vector_add_metal(a_mx, b_mx)
        mx.eval(c_mx)
        c_np_metal = np.array(c_mx)

        c_np_ref = a_np + b_np
        np.testing.assert_allclose(
            c_np_metal,
            c_np_ref,
            rtol=0,
            atol=0,
            err_msg=f"Metal output disagrees with NumPy at n={n}, dtype={dtype}",
        )

    def test_small(self):
        self._assert_matches_numpy(8, np.float32)

    def test_threadgroup_aligned(self):
        self._assert_matches_numpy(256, np.float32)

    def test_unaligned_tail(self):
        self._assert_matches_numpy(257, np.float32)
        self._assert_matches_numpy(1023, np.float32)

    def test_large(self):
        self._assert_matches_numpy(1 << 20, np.float32)

    def test_int32(self):
        rng = np.random.default_rng(seed=42)
        n = 1024
        a_np = rng.integers(-1000, 1000, size=n, dtype=np.int32)
        b_np = rng.integers(-1000, 1000, size=n, dtype=np.int32)

        c_mx = vector_add_metal(mx.array(a_np), mx.array(b_np))
        mx.eval(c_mx)
        np.testing.assert_array_equal(np.array(c_mx), a_np + b_np)


def _benchmark():
    n = 1 << 24  # 16M elements
    rng = np.random.default_rng(seed=1)
    a_np = rng.standard_normal(n).astype(np.float32)
    b_np = rng.standard_normal(n).astype(np.float32)
    a_mx = mx.array(a_np)
    b_mx = mx.array(b_np)
    mx.eval(a_mx, b_mx)

    # warmup
    for _ in range(3):
        c = vector_add_metal(a_mx, b_mx)
        mx.eval(c)

    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        c = vector_add_metal(a_mx, b_mx)
        mx.eval(c)
    t1 = time.perf_counter()
    metal_per = (t1 - t0) / iters * 1e3

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a_np + b_np
    t1 = time.perf_counter()
    numpy_per = (t1 - t0) / iters * 1e3

    bytes_moved = 3 * n * 4  # a + b read, c written, fp32
    bw = bytes_moved / (metal_per * 1e-3) / 1e9
    print(f"vector_add n={n}:")
    print(f"  Metal: {metal_per:7.3f} ms/iter  ({bw:5.1f} GB/s effective)")
    print(f"  NumPy: {numpy_per:7.3f} ms/iter")


if __name__ == "__main__":
    print(f"MLX version : {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    print(f"Metal avail : {mx.metal.is_available()}")
    print(f"Default dev : {mx.default_device()}")
    print()

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromTestCase(VectorAddTest))

    if result.wasSuccessful():
        print()
        _benchmark()
