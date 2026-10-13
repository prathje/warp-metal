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

from typing import Any

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


@wp.kernel
def _i64_counter(counter: wp.array[wp.int64], slots: wp.array[wp.int64]):
    i = wp.tid()
    slots[i] = wp.atomic_add(counter, 0, wp.int64(1))
    wp.atomic_add(counter, 1, wp.int64(3000000000))
    wp.atomic_sub(counter, 2, wp.int64(i))


@wp.kernel
def _u64_minmax(vals: wp.array[wp.uint64], out: wp.array[wp.uint64]):
    i = wp.tid()
    v = vals[i]
    wp.atomic_max(out, 0, v)
    wp.atomic_min(out, 1, v)


@wp.kernel
def _i64_minmax_exch(vals: wp.array[wp.int64], out: wp.array[wp.int64], last: wp.array[wp.int64]):
    i = wp.tid()
    v = vals[i]
    wp.atomic_max(out, 0, v)
    wp.atomic_min(out, 1, v)
    wp.atomic_exch(last, 0, v)


@wp.kernel
def _u64_max_old(vals: wp.array[wp.uint64], out: wp.array[wp.uint64], old: wp.array[wp.uint64]):
    i = wp.tid()
    old[i] = wp.atomic_max(out, 0, vals[i])


EMPTY_KEY = wp.constant(wp.uint64(0xFFFFFFFFFFFFFFFF))


@wp.func
def _hash_insert(keys: wp.array[wp.uint64], key: wp.uint64) -> int:
    capacity = keys.shape[0]
    idx = int(key % wp.uint64(capacity))
    for _attempt in range(capacity):
        stored = keys[idx]
        if stored == key:
            return idx
        if stored == EMPTY_KEY:
            old = wp.atomic_cas(keys, idx, EMPTY_KEY, key)
            if old == EMPTY_KEY or old == key:
                return idx
        idx = (idx + 1) % capacity
    return -1


@wp.kernel
def _hash_insert_kernel(
    src: wp.array[wp.uint64], keys: wp.array[wp.uint64], best: wp.array[wp.uint64], slot_of: wp.array[wp.int32]
):
    i = wp.tid()
    key = src[i]
    slot = _hash_insert(keys, key)
    slot_of[i] = slot
    # Mixed with the compare-and-swap above: max on another 64-bit array.
    wp.atomic_max(best, slot, (key << wp.uint64(20)) | wp.uint64(i))


@wp.kernel
def _bitwise_atomics(
    a32: wp.array[wp.int32], u32: wp.array[wp.uint32], a64: wp.array[wp.int64], u64: wp.array2d[wp.uint64]
):
    i = wp.tid()
    wp.atomic_or(a32, 0, wp.int32(1) << (i % 31))
    wp.atomic_and(a32, 1, ~(wp.int32(1) << (i % 31)))
    wp.atomic_xor(u32, 0, wp.uint32(1) << wp.uint32(i % 32))
    wp.atomic_or(a64, 0, wp.int64(1) << wp.int64(i % 63))
    wp.atomic_xor(u64, i % 2, 0, wp.uint64(1) << wp.uint64(i % 64))


@wp.kernel
def _cas_2d(a: wp.array2d[wp.int32], out: wp.array2d[wp.int32]):
    i, j = wp.tid()
    out[i, j] = wp.atomic_cas(a, i, j % 2, 0, i * 100 + j + 1)


@wp.kernel
def _f16_atomics(a: wp.array[wp.float16]):
    i = wp.tid()
    wp.atomic_add(a, i % 3, wp.float16(1.0))
    wp.atomic_sub(a, 3 + i % 2, wp.float16(0.5))


@wp.kernel
def _read_only(a: wp.array[float]):
    i = wp.tid()
    x = a[i] * 2.0
    if x > 1.0e6:
        x = 0.0


@wp.kernel
def _scale_indexed(a: wp.indexedarray[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel
def _incr_indexed_vec(a: wp.indexedarray[wp.vec3]):
    i = wp.tid()
    a[i] = a[i] + wp.vec3(1.0, 2.0, 3.0)


@wp.kernel
def _incr_2d(a: wp.array2d[float]):
    i, j = wp.tid()
    a[i, j] = a[i, j] + float(i * 10 + j)


@wp.kernel
def _generic_copy(src: Any, dst: Any):
    i = wp.tid()
    dst[i] = src[i]


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

    def test_icb_grows_past_initial_capacity(self):
        from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

        a = wp.zeros(4, dtype=wp.float32)
        flag = wp.array([1], dtype=wp.int32)
        wp.launch(_incr, dim=4, inputs=[a])
        a.zero_()
        disp = get_dispatcher()
        disp.begin_record(max_commands=4)
        for _ in range(10):
            wp.launch(_incr, dim=4, inputs=[a])
        # Conditional structure recorded across a growth step.
        from warp._src.context import _metal_find_buffer  # noqa: PLC0415

        flag_buf, flag_off = _metal_find_buffer(flag.ptr)
        disp.begin_loop(flag_buf, flag_off, max_iters=3)
        wp.launch(_incr, dim=4, inputs=[a])
        wp.launch(_clear_flag, dim=1, inputs=[flag])
        disp.end_loop()
        for _ in range(10):
            wp.launch(_incr, dim=4, inputs=[a])
        graph = disp.end_record()
        self.assertGreaterEqual(graph.count, 22)
        disp.replay(graph)
        np.testing.assert_array_equal(a.numpy(), 21.0)

    def test_pso_archive_tolerates_concurrent_processes(self):
        import tempfile  # noqa: PLC0415
        import warnings  # noqa: PLC0415

        from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

        disp = get_dispatcher()
        saved = (disp._archive, disp._archive_path, disp._archive_dirty, wp.config.kernel_cache_dir)
        saved_env = os.environ.pop("WARP_METAL_PSO_ARCHIVE", None)
        with tempfile.TemporaryDirectory() as cache_dir:
            try:
                wp.config.kernel_cache_dir = cache_dir
                # Another process is mid-serialize into its temp file.
                other_tmp = os.path.join(cache_dir, "metal_pso_archive_0123456789abcdef.tmp.99999")
                with open(other_tmp, "wb"):
                    pass
                disp._init_binary_archive()
                self.assertTrue(os.path.exists(other_tmp))
                disp._archive_dirty = True
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    disp._serialize_archive_if_dirty()
                    # The archive directory vanishing must not warn either.
                    disp._archive_dirty = True
                    disp._archive_path = os.path.join(cache_dir, "gone", "sub", "metal_pso_archive_x")
                    disp._serialize_archive_if_dirty()
                self.assertEqual([str(w.message) for w in caught if "archive" in str(w.message)], [])
            finally:
                disp._archive, disp._archive_path, disp._archive_dirty, wp.config.kernel_cache_dir = saved
                if saved_env is not None:
                    os.environ["WARP_METAL_PSO_ARCHIVE"] = saved_env

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
                for n in (1, 1025, 70000):
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


    # -- atomics -------------------------------------------------------------

    def test_int64_add_sub(self):
        n = 4096
        counter = wp.zeros(3, dtype=wp.int64)
        slots = wp.zeros(n, dtype=wp.int64)
        wp.launch(_i64_counter, dim=n, inputs=[counter, slots])
        c = counter.numpy()
        self.assertEqual(c[0], n)
        self.assertEqual(c[1], n * 3000000000)  # carries into the high word
        self.assertEqual(c[2], -n * (n - 1) // 2)
        # Previous values handed out by the counter are a permutation.
        np.testing.assert_array_equal(np.sort(slots.numpy()), np.arange(n))

    def test_uint64_min_max(self):
        rng = np.random.default_rng(7)
        vals = rng.integers(0, 2**63, 10000, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
        out = wp.array(np.array([0, 2**64 - 1], dtype=np.uint64), dtype=wp.uint64)
        wp.launch(_u64_minmax, dim=vals.size, inputs=[wp.array(vals, dtype=wp.uint64), out])
        np.testing.assert_array_equal(out.numpy(), [vals.max(), vals.min()])

    def test_uint64_max_previous_value(self):
        # The previous value returned by atomic_max must be exact: exactly
        # one thread observes the initial zero.
        rng = np.random.default_rng(10)
        vals = rng.permutation(np.arange(1, 20001, dtype=np.uint64))
        out = wp.zeros(1, dtype=wp.uint64)
        old = wp.zeros(vals.size, dtype=wp.uint64)
        wp.launch(_u64_max_old, dim=vals.size, inputs=[wp.array(vals, dtype=wp.uint64), out, old])
        self.assertEqual(int(out.numpy()[0]), 20000)
        o = old.numpy()
        self.assertEqual(int((o == 0).sum()), 1)
        self.assertTrue(np.all(np.isin(o[o != 0], vals)))

    def test_int64_min_max_exch(self):
        rng = np.random.default_rng(8)
        vals = rng.integers(-(2**62), 2**62, 5000, dtype=np.int64)
        out = wp.array(np.array([-(2**63), 2**63 - 1], dtype=np.int64), dtype=wp.int64)
        last = wp.zeros(1, dtype=wp.int64)
        wp.launch(_i64_minmax_exch, dim=vals.size, inputs=[wp.array(vals, dtype=wp.int64), out, last])
        np.testing.assert_array_equal(out.numpy(), [vals.max(), vals.min()])
        self.assertIn(last.numpy()[0], vals)

    def test_uint64_hashtable_cas(self):
        # Many threads insert keys from a small set: each key must end in
        # exactly one slot (no torn reads / lost compare-and-swaps).
        rng = np.random.default_rng(9)
        distinct = rng.integers(0, 2**40, 700, dtype=np.uint64)
        src = distinct[rng.integers(0, distinct.size, 60000)]
        keys = wp.full(2048, value=0xFFFFFFFFFFFFFFFF, dtype=wp.uint64)
        best = wp.zeros(2048, dtype=wp.uint64)
        slot_of = wp.zeros(src.size, dtype=wp.int32)
        wp.launch(_hash_insert_kernel, dim=src.size, inputs=[wp.array(src, dtype=wp.uint64), keys, best, slot_of])
        k = keys.numpy()
        stored = k[k != np.uint64(0xFFFFFFFFFFFFFFFF)]
        self.assertEqual(stored.size, distinct.size)
        np.testing.assert_array_equal(np.sort(stored), np.sort(distinct))
        slots = slot_of.numpy()
        self.assertTrue(np.all(slots >= 0))
        np.testing.assert_array_equal(k[slots], src)
        # The max per slot is the largest thread index that inserted that key.
        b = best.numpy()
        for key in distinct[:50]:
            idx = np.nonzero(src == key)[0]
            slot = slots[idx[0]]
            self.assertEqual(int(b[slot]), (int(key) << 20) | int(idx.max()))

    def test_bitwise_atomics(self):
        n = 256
        a32 = wp.array(np.array([0, -1], dtype=np.int32))
        u32 = wp.zeros(1, dtype=wp.uint32)
        a64 = wp.zeros(1, dtype=wp.int64)
        u64 = wp.zeros((2, 1), dtype=wp.uint64)
        wp.launch(_bitwise_atomics, dim=n, inputs=[a32, u32, a64, u64])
        ref32 = 0
        for i in range(n):
            ref32 |= 1 << (i % 31)
        self.assertEqual(int(a32.numpy()[0]), ref32)
        self.assertEqual(int(a32.numpy()[1]), ~ref32)
        x = 0
        for i in range(n):
            x ^= 1 << (i % 32)
        self.assertEqual(int(u32.numpy()[0]), x)
        self.assertEqual(int(a64.numpy()[0]), (1 << 63) - 1)
        refs = [0, 0]
        for i in range(n):
            refs[i % 2] ^= 1 << (i % 64)
        np.testing.assert_array_equal(u64.numpy()[:, 0], np.array(refs, dtype=np.uint64))

    def test_multidim_cas(self):
        a = wp.zeros((3, 2), dtype=wp.int32)
        out = wp.zeros((3, 4), dtype=wp.int32)
        wp.launch(_cas_2d, dim=(3, 4), inputs=[a, out])
        winners = a.numpy()
        o = out.numpy()
        for i in range(3):
            for col in range(2):
                self.assertIn(winners[i, col] - 1 - i * 100, (col, col + 2))
            # Exactly one thread per element saw the initial zero.
            self.assertEqual(int((o[i] == 0).sum()), 2)

    def test_float16_atomics(self):
        a = wp.zeros(5, dtype=wp.float16)
        wp.launch(_f16_atomics, dim=300, inputs=[a])
        np.testing.assert_array_equal(a.numpy(), np.array([100, 100, 100, -75, -75], dtype=np.float16))

    def test_read_only_kernel(self):
        a = wp.ones(64, dtype=float)
        wp.launch(_read_only, dim=64, inputs=[a])
        wp.synchronize_device()


    # -- strided / indexed kernel arguments -----------------------------------

    def test_indexed_array_args(self):
        data = wp.array(np.arange(10, dtype=np.float32))
        idx = wp.array([1, 3, 5], dtype=wp.int32)
        out = wp.zeros(3, dtype=float)
        wp.launch(_scale_indexed, dim=3, inputs=[wp.indexedarray(data, idx)], outputs=[out])
        np.testing.assert_array_equal(out.numpy(), [2.0, 6.0, 10.0])
        # Writes through an indexed view land in the source array only.
        v = wp.zeros(6, dtype=wp.vec3)
        wp.launch(_incr_indexed_vec, dim=2, inputs=[wp.indexedarray(v, wp.array([0, 4], dtype=wp.int32))])
        expected = np.zeros((6, 3), dtype=np.float32)
        expected[[0, 4]] = [1.0, 2.0, 3.0]
        np.testing.assert_array_equal(v.numpy(), expected)
        # Generic kernel specialized on indexed arrays.
        src = wp.array(np.arange(12, dtype=np.float32))
        dst = wp.zeros(12, dtype=float)
        wp.launch(
            _generic_copy,
            dim=2,
            inputs=[
                wp.indexedarray(src, wp.array([11, 0], dtype=wp.int32)),
                wp.indexedarray(dst, wp.array([1, 3], dtype=wp.int32)),
            ],
        )
        np.testing.assert_array_equal(dst.numpy()[[1, 3]], [11.0, 0.0])

    def test_strided_array_args(self):
        big = wp.zeros((4, 6), dtype=float)
        view = big[:, 1::2]
        self.assertFalse(view.is_contiguous)
        wp.launch(_incr_2d, dim=view.shape, inputs=[view])
        expected = np.zeros((4, 6), dtype=np.float32)
        expected[:, 1::2] = np.arange(4)[:, None] * 10 + np.arange(3)[None, :]
        np.testing.assert_array_equal(big.numpy(), expected)
        src = wp.array(np.arange(12, dtype=np.float32))
        dst = wp.zeros(12, dtype=float)
        wp.launch(_generic_copy, dim=6, inputs=[src[::2], dst[1::2]])
        expected = np.zeros(12, dtype=np.float32)
        expected[1::2] = np.arange(0, 12, 2)
        np.testing.assert_array_equal(dst.numpy(), expected)

    def test_indexed_args_in_capture(self):
        data = wp.array(np.arange(10, dtype=np.float32))
        idx = wp.array([1, 3, 5], dtype=wp.int32)
        out = wp.zeros(3, dtype=float)
        with wp.ScopedCapture() as capture:
            wp.launch(_scale_indexed, dim=3, inputs=[wp.indexedarray(data, idx)], outputs=[out])
        data.assign(np.arange(10, dtype=np.float32) * 10.0)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(out.numpy(), [20.0, 60.0, 100.0])


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
