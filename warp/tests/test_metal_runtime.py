# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal runtime tests: graph capture state, conditional graph nodes
(``capture_if`` / ``capture_while``), device-side ``wp.utils`` primitives,
64-bit / bitwise / half-precision atomics and strided / indexed kernel
arguments.

The tests run in-process when the Metal backend is enabled for this process
(``WARP_ENABLE_METAL=1``). Otherwise :class:`TestMetalRuntimeSubprocess`
re-runs :class:`TestMetalRuntime` in a subprocess with the backend enabled, so
the default suite still covers them on Apple Silicon.
"""

import os
import platform
import subprocess
import sys
import unittest

import numpy as np

import warp as wp


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _has_mlx() -> bool:
    try:
        import mlx.core as mx  # noqa: PLC0415

        return mx.metal.is_available()
    except Exception:
        return False


def _metal_device():
    if not wp.config.enable_metal:
        return None
    try:
        device = wp.get_device("metal:0")
    except Exception:
        return None
    if not wp.config.metal_native_dispatch:
        return None
    return device


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _incr(a: wp.array[wp.float32]):
    i = wp.tid()
    a[i] = a[i] + 1.0


@wp.kernel
def _countdown(c: wp.array[wp.int32], n: wp.array[wp.int32]):
    n[0] = n[0] + 1
    c[0] = c[0] - 1


@wp.kernel
def _clear_flag(f: wp.array[wp.int32]):
    f[0] = 0


@wp.kernel
def _set_flag(f: wp.array[wp.int32], value: wp.int32):
    f[0] = value


class TestMetalRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = _metal_device()
        if cls.device is None:
            raise unittest.SkipTest("Metal backend with native dispatch is not enabled in this process")

    def setUp(self):
        self._scoped = wp.ScopedDevice(self.device)
        self._scoped.__enter__()
        self._saved_max_iters = wp.config.metal_capture_while_max_iters

    def tearDown(self):
        wp.config.metal_capture_while_max_iters = self._saved_max_iters
        self._scoped.__exit__(None, None, None)

    # -- capture state ------------------------------------------------------

    def test_is_capturing(self):
        a = wp.zeros(8, dtype=wp.float32)
        self.assertFalse(self.device.is_capturing)
        with wp.ScopedCapture() as capture:
            self.assertTrue(self.device.is_capturing)
            self.assertFalse(wp.get_device("cpu").is_capturing)
            wp.launch(_incr, dim=8, inputs=[a])
        self.assertFalse(self.device.is_capturing)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(a.numpy(), 1.0)

    def test_host_ops_raise_inside_capture(self):
        a = wp.zeros(8, dtype=wp.float32)
        with self.assertRaises(RuntimeError):
            with wp.ScopedCapture():
                a.numpy()
        with self.assertRaisesRegex(RuntimeError, "graph capture is active"):
            with wp.ScopedCapture():
                wp.synchronize_device()
        self.assertFalse(self.device.is_capturing)

    def test_capture_pause_resume(self):
        a = wp.zeros(4, dtype=wp.float32)
        b = wp.zeros(4, dtype=wp.float32)
        with wp.ScopedCapture() as capture:
            wp.launch(_incr, dim=4, inputs=[a])
            graph = wp._src.context.capture_pause()
            self.assertFalse(self.device.is_capturing)
            # Runs eagerly, not recorded.
            wp.launch(_incr, dim=4, inputs=[b])
            np.testing.assert_array_equal(b.numpy(), 1.0)
            wp._src.context.capture_resume(graph)
            self.assertTrue(self.device.is_capturing)
            wp.launch(_incr, dim=4, inputs=[a])
        np.testing.assert_array_equal(a.numpy(), 0.0)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(a.numpy(), 2.0)
        np.testing.assert_array_equal(b.numpy(), 1.0)

    # -- conditional graph nodes -------------------------------------------

    def _countdown_body(self, c, n, a):
        def body():
            wp.launch(_countdown, dim=1, inputs=[c, n])
            wp.launch(_incr, dim=a.shape[0], inputs=[a])

        return body

    def test_capture_while_eager(self):
        a = wp.zeros(8, dtype=wp.float32)
        c = wp.array([5], dtype=wp.int32)
        n = wp.zeros(1, dtype=wp.int32)
        # The condition is written by queued kernels; the host must see the
        # current value on every iteration.
        wp.capture_while(c, self._countdown_body(c, n, a))
        self.assertEqual(n.numpy()[0], 5)
        np.testing.assert_array_equal(a.numpy(), 5.0)

    def test_capture_while_graph(self):
        a = wp.zeros(8, dtype=wp.float32)
        c = wp.array([7], dtype=wp.int32)
        n = wp.zeros(1, dtype=wp.int32)
        body = self._countdown_body(c, n, a)
        with wp.ScopedCapture() as capture:
            wp.capture_while(c, body)
        # Nothing ran at capture time.
        self.assertEqual(n.numpy()[0], 0)
        wp.capture_launch(capture.graph)
        self.assertEqual(n.numpy()[0], 7)
        np.testing.assert_array_equal(a.numpy(), 7.0)
        # The trip count is decided on the GPU at replay time.
        c.fill_(3)
        wp.capture_launch(capture.graph)
        self.assertEqual(n.numpy()[0], 10)
        c.fill_(0)
        wp.capture_launch(capture.graph)
        self.assertEqual(n.numpy()[0], 10)

    def test_capture_while_iteration_semantics(self):
        # The condition is latched at the start of each iteration (CUDA
        # semantics): clearing it mid-body still runs the rest of the body.
        a = wp.zeros(4, dtype=wp.float32)
        c = wp.array([1], dtype=wp.int32)
        b = wp.zeros(4, dtype=wp.float32)

        def body():
            wp.launch(_clear_flag, dim=1, inputs=[c])
            wp.launch(_incr, dim=4, inputs=[a])
            # A nested conditional splits the body into several gated runs.
            wp.capture_if(c, on_false=lambda: wp.launch(_incr, dim=4, inputs=[b]))
            wp.launch(_incr, dim=4, inputs=[a])

        with wp.ScopedCapture() as capture:
            wp.capture_while(c, body)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(a.numpy(), 2.0)
        np.testing.assert_array_equal(b.numpy(), 1.0)

    def test_capture_while_budget(self):
        wp.config.metal_capture_while_max_iters = 4
        a = wp.zeros(2, dtype=wp.float32)
        c = wp.array([10], dtype=wp.int32)
        n = wp.zeros(1, dtype=wp.int32)
        with wp.ScopedCapture() as capture:
            wp.capture_while(c, self._countdown_body(c, n, a))
        wp.capture_launch(capture.graph)
        self.assertEqual(n.numpy()[0], 4)
        self.assertEqual(c.numpy()[0], 6)

    def test_capture_if_else(self):
        flag = wp.array([1], dtype=wp.int32)
        a = wp.zeros(4, dtype=wp.float32)
        b = wp.zeros(4, dtype=wp.float32)

        def on_true():
            wp.launch(_incr, dim=4, inputs=[a])
            # Rewriting the condition inside the branch must not also run
            # the else branch in the same execution.
            wp.launch(_clear_flag, dim=1, inputs=[flag])

        def on_false():
            wp.launch(_incr, dim=4, inputs=[b])

        with wp.ScopedCapture() as capture:
            wp.capture_if(flag, on_true, on_false)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(a.numpy(), 1.0)
        np.testing.assert_array_equal(b.numpy(), 0.0)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(a.numpy(), 1.0)
        np.testing.assert_array_equal(b.numpy(), 1.0)

        # Eager path reads the condition after queued writes complete.
        wp.launch(_set_flag, dim=1, inputs=[flag, 1])
        wp.capture_if(flag, on_true, on_false)
        np.testing.assert_array_equal(a.numpy(), 2.0)

    def test_nested_while(self):
        outer = wp.array([3], dtype=wp.int32)
        inner = wp.zeros(1, dtype=wp.int32)
        n_outer = wp.zeros(1, dtype=wp.int32)
        n_inner = wp.zeros(1, dtype=wp.int32)

        def inner_body():
            wp.launch(_countdown, dim=1, inputs=[inner, n_inner])

        def outer_body():
            wp.launch(_set_flag, dim=1, inputs=[inner, 2])
            wp.capture_while(inner, inner_body)
            wp.launch(_countdown, dim=1, inputs=[outer, n_outer])

        with wp.ScopedCapture() as capture:
            wp.capture_while(outer, outer_body)
        wp.capture_launch(capture.graph)
        self.assertEqual(n_outer.numpy()[0], 3)
        self.assertEqual(n_inner.numpy()[0], 6)

    def test_capture_if_graph_branch_raises(self):
        flag = wp.array([1], dtype=wp.int32)
        a = wp.zeros(4, dtype=wp.float32)
        with wp.ScopedCapture() as sub:
            wp.launch(_incr, dim=4, inputs=[a])
        with self.assertRaises(NotImplementedError):
            with wp.ScopedCapture():
                wp.capture_if(flag, sub.graph)


    # -- BVH constructor fallback -----------------------------------------

    def test_lbvh_fallback_is_silent(self):
        import warnings  # noqa: PLC0415

        points = wp.array(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32), dtype=wp.vec3)
        indices = wp.array(np.array([0, 1, 2, 0, 1, 3], dtype=np.int32), dtype=wp.int32)
        lowers = wp.array(np.zeros((4, 3), dtype=np.float32), dtype=wp.vec3)
        uppers = wp.array(np.ones((4, 3), dtype=np.float32), dtype=wp.vec3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mesh = wp.Mesh(points=points, indices=indices, bvh_constructor="lbvh")
            bvh = wp.Bvh(lowers, uppers, constructor="lbvh")
            bvh.rebuild("lbvh")
        self.assertNotEqual(mesh.id, 0)
        self.assertNotEqual(bvh.id, 0)
        messages = [str(w.message) for w in caught if "constructor" in str(w.message)]
        self.assertEqual(messages, [])


    # -- device-side wp.utils ------------------------------------------------

    def _force_device_utils(self):
        from warp._src import metal_utils  # noqa: PLC0415

        saved = metal_utils.HOST_FALLBACK_MAX_ELEMENTS
        metal_utils.HOST_FALLBACK_MAX_ELEMENTS = 0
        self.addCleanup(setattr, metal_utils, "HOST_FALLBACK_MAX_ELEMENTS", saved)

    def test_utils_array_scan(self):
        self._force_device_utils()
        rng = np.random.default_rng(1)
        for dtype in (wp.int32, wp.int64, wp.float32):
            for n in (1, 2049, 300001):
                npt = wp.dtype_to_numpy(dtype)
                if dtype is wp.float32:
                    x = rng.random(n).astype(npt)
                else:
                    x = rng.integers(-100, 100, n).astype(npt)
                for inclusive in (True, False):
                    a = wp.array(x, dtype=dtype)
                    out = wp.empty(n, dtype=dtype)
                    wp.utils.array_scan(a, out, inclusive)
                    ref = np.cumsum(x.astype(np.float64 if dtype is wp.float32 else np.int64))
                    if not inclusive:
                        ref = np.concatenate([[0], ref[:-1]])
                    if dtype is wp.float32:
                        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-2 * np.sqrt(n))
                    else:
                        np.testing.assert_array_equal(out.numpy(), ref.astype(npt))
        # In-place vector scan and strided input.
        x = rng.integers(0, 10, (5000, 3)).astype(np.int32)
        a = wp.array(x, dtype=wp.vec3i)
        wp.utils.array_scan(a, a, True)
        np.testing.assert_array_equal(a.numpy(), np.cumsum(x, axis=0))
        y = rng.integers(0, 10, 9000).astype(np.int32)
        strided = wp.array(y, dtype=wp.int32)[::3]
        out = wp.zeros(3000, dtype=wp.int32)
        wp.utils.array_scan(strided, out, False)
        np.testing.assert_array_equal(out.numpy(), np.concatenate([[0], np.cumsum(y[::3])[:-1]]))

    def test_utils_radix_sort(self):
        self._force_device_utils()
        rng = np.random.default_rng(2)
        cases = {
            wp.int32: lambda n: rng.integers(-(2**31), 2**31, n, dtype=np.int64).astype(np.int32),
            wp.uint32: lambda n: rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32),
            wp.float32: lambda n: (rng.standard_normal(n) * 1e3).astype(np.float32),
            wp.int64: lambda n: rng.integers(-(2**62), 2**62, n, dtype=np.int64),
            wp.uint64: lambda n: rng.integers(0, 2**63, n, dtype=np.uint64) * np.uint64(2) + np.uint64(1),
            wp.float64: lambda n: rng.standard_normal(n) * 1e3,
        }
        for key_type, gen in cases.items():
            for value_type in (wp.int32, wp.int64):
                for n in (1, 1025, 150000):
                    k = gen(n)
                    k[: n // 3] = k[n // 3 : 2 * (n // 3)]  # duplicates exercise stability
                    v = np.arange(n, dtype=wp.dtype_to_numpy(value_type))
                    keys = wp.array(np.concatenate([k, k]), dtype=key_type)
                    values = wp.array(np.concatenate([v, v]), dtype=value_type)
                    wp.utils.radix_sort_pairs(keys, values, n)
                    order = np.argsort(k, kind="stable")
                    np.testing.assert_array_equal(keys.numpy()[:n], k[order])
                    np.testing.assert_array_equal(values.numpy()[:n], v[order])
        # Partial bit ranges match the CPU implementation.
        k = rng.integers(-(2**31), 2**31, 50000, dtype=np.int64).astype(np.int32)
        for begin_bit, end_bit in ((4, 12), (0, 7), (8, 32)):
            results = []
            for device in (self.device, "cpu"):
                keys = wp.array(np.concatenate([k, k]), dtype=wp.int32, device=device)
                values = wp.array(np.arange(2 * k.size, dtype=np.int32), device=device)
                wp.utils.radix_sort_pairs(keys, values, k.size, begin_bit, end_bit)
                results.append((keys.numpy()[: k.size], values.numpy()[: k.size]))
            np.testing.assert_array_equal(results[0][0], results[1][0])
            np.testing.assert_array_equal(results[0][1], results[1][1])

    def test_utils_segmented_sort(self):
        self._force_device_utils()
        rng = np.random.default_rng(3)
        for key_type in (wp.int32, wp.float32):
            n = 40000
            if key_type is wp.float32:
                k = (rng.standard_normal(n) * 100).astype(np.float32)
            else:
                k = rng.integers(-1000, 1000, n).astype(np.int32)
            v = np.arange(n, dtype=np.int32)
            bounds = np.sort(rng.choice(np.arange(1, n), size=800, replace=False)).astype(np.int32)
            for explicit in (True, False):
                results = []
                for device in (self.device, "cpu"):
                    keys = wp.array(np.concatenate([k, k]), dtype=key_type, device=device)
                    values = wp.array(np.concatenate([v, v]), dtype=wp.int32, device=device)
                    if explicit:
                        # Gaps between segments must stay untouched.
                        starts = wp.array(bounds[0::2], dtype=wp.int32, device=device)
                        ends = wp.array(bounds[1::2], dtype=wp.int32, device=device)
                        wp.utils.segmented_sort_pairs(keys, values, n, starts, ends)
                    else:
                        offsets = wp.array(np.concatenate([[0], bounds, [n]]).astype(np.int32), device=device)
                        wp.utils.segmented_sort_pairs(keys, values, n, offsets)
                    results.append((keys.numpy()[:n], values.numpy()[:n]))
                np.testing.assert_array_equal(results[0][0], results[1][0])
                np.testing.assert_array_equal(results[0][1], results[1][1])

    def test_utils_runlength_encode(self):
        self._force_device_utils()
        rng = np.random.default_rng(4)
        for n in (1, 10, 200000):
            x = np.sort(rng.integers(0, max(2, n // 5), n)).astype(np.int32)
            values = wp.array(x)
            run_values = wp.zeros(n, dtype=wp.int32)
            run_lengths = wp.zeros(n, dtype=wp.int32)
            run_count = wp.zeros(1, dtype=wp.int32)
            wp.utils.runlength_encode(values, run_values, run_lengths, run_count)
            unique, counts = np.unique(x, return_counts=True)
            count = run_count.numpy()[0]
            self.assertEqual(count, unique.size)
            np.testing.assert_array_equal(run_values.numpy()[:count], unique)
            np.testing.assert_array_equal(run_lengths.numpy()[:count], counts)
            self.assertEqual(wp.utils.runlength_encode(values, run_values, run_lengths), unique.size)

    def test_utils_sum_inner(self):
        self._force_device_utils()
        rng = np.random.default_rng(5)
        for n in (1, 1000, 1000000):
            x = rng.random(n).astype(np.float32)
            y = rng.random(n).astype(np.float32)
            self.assertAlmostEqual(
                wp.utils.array_sum(wp.array(x)) / n, x.astype(np.float64).sum() / n, delta=1e-5
            )
            self.assertAlmostEqual(
                wp.utils.array_inner(wp.array(x), wp.array(y)) / n, np.dot(x.astype(np.float64), y) / n, delta=1e-5
            )
        x = rng.random((60, 7, 3)).astype(np.float32)
        for axis in (0, 1, -1):
            a = wp.array(x)
            np.testing.assert_allclose(wp.utils.array_sum(a, axis=axis).numpy(), x.sum(axis=axis, keepdims=True), rtol=1e-5)
            np.testing.assert_allclose(
                wp.utils.array_inner(a, a, axis=axis).numpy(), (x * x).sum(axis=axis, keepdims=True), rtol=1e-5
            )
        v = rng.random((1000, 3)).astype(np.float32)
        np.testing.assert_allclose(wp.utils.array_sum(wp.array(v, dtype=wp.vec3)), v.sum(axis=0), rtol=1e-5)
        # float64 runs on the host outside a capture ...
        d = rng.random(100)
        self.assertAlmostEqual(wp.utils.array_sum(wp.array(d, dtype=wp.float64)), d.sum(), places=10)
        # ... and cannot be captured.
        out = wp.zeros(1, dtype=wp.float64)
        d_arr = wp.array(d, dtype=wp.float64)
        with self.assertRaisesRegex(RuntimeError, "float64"):
            with wp.ScopedCapture():
                wp.utils.array_sum(d_arr, out=out)

    def test_utils_capture(self):
        rng = np.random.default_rng(6)
        n = 5000  # small: the device path is used inside captures regardless of size
        src = wp.array(rng.integers(0, 2**31, n).astype(np.int32))
        keys = wp.zeros(2 * n, dtype=wp.int32)
        values = wp.zeros(2 * n, dtype=wp.int32)
        scanned = wp.zeros(n, dtype=wp.int32)
        f = wp.array(rng.random(n).astype(np.float32))
        total = wp.zeros(1, dtype=wp.float32)
        run_values = wp.zeros(n, dtype=wp.int32)
        run_lengths = wp.zeros(n, dtype=wp.int32)
        run_count = wp.zeros(1, dtype=wp.int32)
        with wp.ScopedCapture() as capture:
            wp.copy(keys, src, count=n)
            wp.copy(values, src, count=n)
            wp.utils.radix_sort_pairs(keys, values, n)
            wp.utils.array_scan(values[:n], scanned, False)
            wp.utils.array_sum(f, out=total)
            wp.utils.runlength_encode(keys, run_values, run_lengths, run_count, value_count=n)
        for _ in range(2):
            k = rng.integers(0, 50, n).astype(np.int32)
            src.assign(k)
            wp.capture_launch(capture.graph)
            s = np.sort(k)
            np.testing.assert_array_equal(keys.numpy()[:n], s)
            np.testing.assert_array_equal(
                scanned.numpy(), np.concatenate([[0], np.cumsum(s.astype(np.int64))[:-1]]).astype(np.int32)
            )
            self.assertAlmostEqual(total.numpy()[0], f.numpy().astype(np.float64).sum(), delta=1e-2)
            unique, counts = np.unique(s, return_counts=True)
            self.assertEqual(run_count.numpy()[0], unique.size)
            np.testing.assert_array_equal(run_lengths.numpy()[: unique.size], counts)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalRuntimeSubprocess(unittest.TestCase):
    @unittest.skipIf(wp.config.enable_metal, "Metal enabled in-process; TestMetalRuntime runs directly")
    def test_metal_runtime(self):
        env = dict(os.environ, WARP_ENABLE_METAL="1")
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "warp.tests.test_metal_runtime.TestMetalRuntime"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,
        )
        self.assertEqual(result.returncode, 0, f"{result.stdout[-4000:]}\n{result.stderr[-8000:]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
