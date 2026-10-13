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
