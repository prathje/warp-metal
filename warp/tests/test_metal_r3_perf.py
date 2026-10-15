# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal codegen cost and artifact-cache behaviour seen in Newton test runs.

Newton's large inlined kernels (narrow phase, Kamino) took minutes to
translate. Quadratic translator passes, artifact-cache keys that changed
with kernel build order, uncached translation failures and cross-checkout
cache sweeps made modules time out; a key that ignored inlined callees
served one closure kernel's artifact for another.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_enabled


def _run_metal_script(test_case, code: str, cache_dir: str, env_extra=None, timeout: int = 120) -> str:
    """Run ``code`` in a fresh Metal-enabled interpreter and return its stdout.

    ``cache_dir`` becomes the kernel cache, so consecutive calls share it.
    """
    prefix = "import warp as wp\nwp.config.enable_metal = True\nwp.config.quiet = True\nwp.init()\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="warp_metal_r3_") as f:
        f.write(prefix + textwrap.dedent(code))
        path = f.name
    env = dict(os.environ, WARP_CACHE_PATH=cache_dir, WARP_METAL_PSO_ARCHIVE="0")
    env.update(env_extra or {})
    try:
        result = subprocess.run(
            [sys.executable, path], check=False, capture_output=True, text=True, timeout=timeout, env=env
        )
    finally:
        os.unlink(path)
    test_case.assertEqual(
        result.returncode,
        0,
        f"exit {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
    )
    return result.stdout


_CLOSURE_FACTORY = """
import numpy as np

def make(flag):
    mod = f"r3_closure_{flag}"

    @wp.func(module=mod)
    def pick(x: float) -> float:
        if wp.static(flag):
            return x + 100.0
        return x - 100.0

    @wp.kernel(module=mod)
    def k(out: wp.array[float]):
        i = wp.tid()
        out[i] = pick(float(i))

    return k

def run(k, dev):
    a = wp.zeros(2, dtype=float, device=dev)
    wp.launch(k, dim=2, outputs=[a], device=dev)
    return a.numpy()
"""


class TestMetalForLoopMaterialize(unittest.TestCase):
    """``_materialize_for_loop_vars`` is pure text processing (no GPU)."""

    def test_rewrites_only_loops_read_after_close(self):
        from warp._src.codegen_metal import _materialize_for_loop_vars  # noqa: PLC0415

        lines = [
            "    for (var_1 = var_0; var_1 < var_9; ++var_1) {",
            "        var_2 = var_1;",
            "    }",
            "    for (var_3 = var_0; var_3 < var_9; ++var_3) {",
            "        var_4 = var_3;",
            "    }",
            "    var_5 = var_3 + var_12;",  # var_3 read after its loop; var_1 only as a prefix
            "    var_6 = var_10;",
        ]
        out = _materialize_for_loop_vars(lines)
        self.assertEqual(out[0], lines[0])
        self.assertEqual(out[3], "    for (int wp_for_it0 = var_0; wp_for_it0 < var_9; ++wp_for_it0) {")
        self.assertEqual(out[4], "    var_3 = wp_for_it0;")
        self.assertEqual(out[5:], lines[4:])

    def test_linear_on_many_loops(self):
        # One loop per 3 lines over a 60k-line body: the previous rescan of
        # the remaining kernel per loop took minutes at this size (this test
        # would effectively hang).
        from warp._src.codegen_metal import _materialize_for_loop_vars  # noqa: PLC0415

        lines = []
        for n in range(20000):
            lines += [
                f"    for (var_{n}_i = var_0; var_{n}_i < var_9; ++var_{n}_i) {{",
                f"        var_{n}_x = var_{n}_i;",
                "    }",
            ]
        lines.append("    var_out = var_19999_i;")
        out = _materialize_for_loop_vars(lines)
        self.assertEqual(len(out), len(lines) + 1)
        self.assertIn("wp_for_it0", out[-5])


class TestMetalArtifactSweep(unittest.TestCase):
    def test_recent_sibling_salts_survive(self):
        import warp._src.codegen_metal as cm  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as base:
            for name in ("keep", "fresh", "stale"):
                os.makedirs(os.path.join(base, name))
            old = time.time() - 2 * cm._ARTIFACT_SWEEP_GRACE_S
            os.utime(os.path.join(base, "stale"), (old, old))
            cm._sweep_stale_artifact_dirs(base, "keep")
            self.assertEqual(sorted(os.listdir(base)), ["fresh", "keep"])


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalArtifactCacheKeys(unittest.TestCase):
    def test_closure_callee_difference_is_not_shared(self):
        # Kernel IR is identical for both flags (the difference is inside the
        # inlined ``pick``); a process that only ever built the ``False``
        # variant used to load the ``True`` variant's cached artifact.
        with tempfile.TemporaryDirectory() as cache:
            for flag in ("True", "False"):
                out = _run_metal_script(
                    self,
                    _CLOSURE_FACTORY
                    + f"""
k = make({flag})
np.testing.assert_array_equal(run(k, "metal:0"), run(k, "cpu"))
print("ok")
""",
                    cache,
                )
                self.assertIn("ok", out)

    def test_key_stable_across_factory_rebuilds(self):
        # Re-creating a closure kernel renames its callees (``pick_0`` ->
        # ``pick_1``) and the process-wide synthetic-temp counter advanced
        # between builds; both changed the key, so the rebuilt kernel missed
        # the cache and was translated again.
        code = (
            _CLOSURE_FACTORY
            + """
import warp._src.codegen_metal as cm

@wp.struct
class S:
    a: wp.vec3
    b: float

def make_struct_kernel():
    @wp.kernel(module="r3_struct")
    def ks(out: wp.array[S]):
        s = S()
        s.a = wp.vec3(1.0, 2.0, 3.0)
        s.b = 4.0
        out[wp.tid()] = s
    return ks

keys = []
for _ in range(2):
    ks = make_struct_kernel()
    wp.launch(ks, dim=1, outputs=[wp.zeros(1, dtype=S, device="metal:0")], device="metal:0")
    k = make(True)
    run(k, "metal:0")
    keys.append((cm._artifact_cache_key(ks, ks.adj), cm._artifact_cache_key(k, k.adj)))
assert keys[0] == keys[1], keys
k_false = make(False)
run(k_false, "metal:0")
assert cm._artifact_cache_key(k_false, k_false.adj) != keys[0][1]
print("ok")
"""
        )
        with tempfile.TemporaryDirectory() as cache:
            self.assertIn("ok", _run_metal_script(self, code, cache))

    def test_translation_failure_is_cached(self):
        # ``metal_print = "error"`` makes translation raise deterministically.
        code = """
import warp._src.codegen_metal as cm
wp.config.metal_print = "error"
calls = [0]
orig = cm._generate_msl_kernel_uncached
def counting(*a, **kw):
    calls[0] += 1
    return orig(*a, **kw)
cm._generate_msl_kernel_uncached = counting

@wp.kernel
def kp(out: wp.array[float]):
    wp.printf("%d\\n", wp.tid())
    out[wp.tid()] = 1.0

for _ in range(2):
    try:
        wp.launch(kp, dim=1, outputs=[wp.zeros(1, dtype=float, device="metal:0")], device="metal:0")
    except cm.MetalCodegenError as e:
        msg = str(e)
    else:
        raise AssertionError("expected MetalCodegenError")
print("calls", calls[0], "printf" in msg or "print" in msg)
"""
        with tempfile.TemporaryDirectory() as cache:
            first = _run_metal_script(self, code, cache)
            self.assertIn("calls 1 True", first)
            # A new process reads the failure from disk.
            second = _run_metal_script(self, code, cache)
            self.assertIn("calls 0 True", second)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalTileBroadcast(unittest.TestCase):
    def test_column_and_row_broadcast(self):
        # warp_nn ``Linear`` adds a bias tile ``(out, 1)`` broadcast to
        # ``(out, batch)``; Metal repeated it along the wrong axis, so only
        # batch row 0 received the bias (Newton DriveNeuralMLP tests).
        snippet = textwrap.dedent(
            """
            import numpy as np

            R, C = 16, 16

            @wp.kernel
            def kcol(src: wp.array2d[float], out: wp.array2d[float]):
                i, j = wp.tid()
                t = wp.tile_zeros(shape=(R, C), dtype=float)
                t += wp.tile_broadcast(wp.tile_load(src, shape=(R, 1), offset=(0, 0)), shape=(R, C))
                wp.tile_store(out, t, offset=(0, 0))

            @wp.kernel
            def krow(src: wp.array[float], out: wp.array2d[float]):
                i, j = wp.tid()
                t = wp.tile_broadcast(wp.tile_load(src, shape=(C,), offset=(0,)), shape=(R, C))
                wp.tile_store(out, t, offset=(0, 0))

            col = np.arange(1, R + 1, dtype=np.float32).reshape(R, 1)
            row = np.arange(1, C + 1, dtype=np.float32)
            for dev in ("cpu", "metal:0"):
                out = wp.zeros((R, C), dtype=float, device=dev)
                wp.launch_tiled(kcol, dim=(1, 1), inputs=[wp.array(col, device=dev)], outputs=[out], device=dev, block_dim=64)
                np.testing.assert_array_equal(out.numpy(), np.broadcast_to(col, (R, C)), err_msg=dev)
                out = wp.zeros((R, C), dtype=float, device=dev)
                wp.launch_tiled(krow, dim=(1, 1), inputs=[wp.array(row, device=dev)], outputs=[out], device=dev, block_dim=64)
                np.testing.assert_array_equal(out.numpy(), np.broadcast_to(row, (R, C)), err_msg=dev)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalAsyncFill(unittest.TestCase):
    def test_zero_and_fill_do_not_drain_queue(self):
        # ``zero_()`` / ``fill_()`` behind queued kernels used to sync() and
        # memset on the host — one GPU round trip each (Newton's VBD solver
        # zeroes five arrays per iteration: 70k syncs, ~600 s in one test).
        # They now enqueue a device fill ordered after the queued kernels.
        snippet = textwrap.dedent(
            """
            import numpy as np
            from warp._src.metal_dispatch import get_dispatcher

            @wp.kernel
            def write(a: wp.array[float], v: float):
                a[wp.tid()] = v

            disp = get_dispatcher()
            calls = [0]
            orig = type(disp).sync
            def counting(self):
                calls[0] += 1
                return orig(self)
            type(disp).sync = counting

            n = 1 << 16
            a = wp.empty(n, dtype=float, device="metal:0")
            b = wp.empty(n, dtype=wp.vec3, device="metal:0")
            for _ in range(3):
                wp.launch(write, dim=n, inputs=[a, 1.0], device="metal:0")
                a.zero_()
                wp.launch(write, dim=n // 2, inputs=[a, 2.0], device="metal:0")
                b.fill_(wp.vec3(1.0, 2.0, 3.0))
            assert calls[0] == 0, calls[0]
            expected = np.zeros(n, dtype=np.float32)
            expected[: n // 2] = 2.0
            np.testing.assert_array_equal(a.numpy(), expected)
            np.testing.assert_array_equal(b.numpy(), np.tile([1.0, 2.0, 3.0], (n, 1)).astype(np.float32))
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalAutoflush(unittest.TestCase):
    def test_busy_gpu_batches_dependent_launches(self):
        # While the GPU is busy, launches share a command buffer (separated
        # by buffer barriers) instead of one commit each; dependent launches
        # must still see each other's writes.
        snippet = textwrap.dedent(
            """
            import numpy as np
            from warp._src.metal_dispatch import get_dispatcher

            @wp.kernel
            def step(src: wp.array[float], dst: wp.array[float]):
                i = wp.tid()
                dst[i] = src[i] * 0.5 + float(i)

            n = 4096
            res = {}
            for dev in ("cpu", "metal:0"):
                a = wp.zeros(n, dtype=float, device=dev)
                b = wp.zeros(n, dtype=float, device=dev)
                if dev == "metal:0":
                    disp = get_dispatcher()
                    wp.synchronize_device(dev)
                    disp._autoflush_every = 8
                    disp._gpu_idle = lambda: False
                for k in range(101):
                    src, dst = (a, b) if k % 2 == 0 else (b, a)
                    wp.launch(step, dim=n, inputs=[src, dst], device=dev)
                if dev == "metal:0":
                    assert len(disp._pending_commits) == 101 // 8, len(disp._pending_commits)
                res[dev] = b.numpy()
            np.testing.assert_allclose(res["metal:0"], res["cpu"], rtol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalNestedCaptureWhileReplay(unittest.TestCase):
    def test_nested_loops_with_varying_exits(self):
        # Replay stops encoding a loop's iterations once the GPU has latched
        # its condition false; iterations that did run (and inner loops that
        # restart per outer iteration) must still all execute, replay after
        # replay.
        snippet = textwrap.dedent(
            """
            import numpy as np

            @wp.kernel
            def body(x: wp.array[float], idx: int):
                x[idx] += 1.0

            @wp.kernel
            def count_down(flag: wp.array[int], cnt: wp.array[int]):
                cnt[0] = cnt[0] - 1
                if cnt[0] <= 0:
                    flag[0] = 0

            @wp.kernel
            def arm(flag: wp.array[int], cnt: wp.array[int], k: wp.array[int]):
                flag[0] = 1
                cnt[0] = k[0]
                k[0] = k[0] % 5 + 1  # next execution of this loop runs a different count

            dev = "metal:0"
            x = wp.zeros(2, dtype=float, device=dev)
            flags = [wp.zeros(1, dtype=int, device=dev) for _ in range(2)]
            cnts = [wp.zeros(1, dtype=int, device=dev) for _ in range(2)]
            ks = [wp.array([3], dtype=int, device=dev), wp.array([2], dtype=int, device=dev)]

            def inner():
                wp.launch(body, dim=1, inputs=[x, 1], device=dev)
                wp.launch(count_down, dim=1, inputs=[flags[1], cnts[1]], device=dev)

            def outer():
                wp.launch(body, dim=1, inputs=[x, 0], device=dev)
                wp.launch(arm, dim=1, inputs=[flags[1], cnts[1], ks[1]], device=dev)
                with wp.ScopedMetalCaptureWhileBudget(16):
                    wp.capture_while(flags[1], while_body=inner)
                wp.launch(count_down, dim=1, inputs=[flags[0], cnts[0]], device=dev)

            with wp.ScopedCapture(device=dev) as cap:
                wp.launch(arm, dim=1, inputs=[flags[0], cnts[0], ks[0]], device=dev)
                with wp.ScopedMetalCaptureWhileBudget(300):
                    wp.capture_while(flags[0], while_body=outer)

            # Host model of the same counters.
            k = [3, 2]
            expected = np.zeros(2)
            def model():
                n_outer = k[0]; k[0] = k[0] % 5 + 1
                for _ in range(n_outer):
                    expected[0] += 1
                    n_inner = k[1]; k[1] = k[1] % 5 + 1
                    expected[1] += n_inner
            for _ in range(4):
                wp.capture_launch(cap.graph)
                model()
            np.testing.assert_array_equal(x.numpy(), expected)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)


_ACOS_SNIPPET = """
import numpy as np

@wp.kernel
def k(x: wp.array[float], xd: wp.array[wp.float64], a: wp.array[float], b: wp.array[float],
      h: wp.array[wp.float16], d: wp.array[wp.float64]):
    i = wp.tid()
    a[i] = wp.acos(x[i])
    b[i] = wp.asin(x[i])
    h[i] = wp.acos(wp.float16(x[i]))
    d[i] = wp.acos(xd[i]) + wp.asin(xd[i])

xs = np.array([1.0000001, -1.0000001, 1.5, 0.5, 1.0, -0.25], np.float32)
res = {}
for dev in ("cpu", "metal:0"):
    outs = [wp.zeros(6, dtype=t, device=dev) for t in (float, float, wp.float16, wp.float64)]
    wp.launch(k, dim=6, inputs=[wp.array(xs, device=dev), wp.array(xs.astype(np.float64), dtype=wp.float64, device=dev)],
              outputs=outs, device=dev)
    res[dev] = [o.numpy() for o in outs]
for c, m in zip(res["cpu"], res["metal:0"]):
    assert np.isfinite(m).all(), m
    np.testing.assert_allclose(m.astype(np.float64), c.astype(np.float64), atol=2e-3)
"""


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalAcosAsinDomain(unittest.TestCase):
    # ``wp.acos`` / ``wp.asin`` clamp to [-1, 1] natively; ``metal::acos``
    # returned NaN for a unit-vector dot that rounded to 1.0000001 (Newton's
    # cloth_hanging semi-implicit lift force NaN'd every particle).
    def test_clamped_domain_demote(self):
        with tempfile.TemporaryDirectory() as cache:
            _run_metal_script(self, _ACOS_SNIPPET, cache, env_extra={"WARP_METAL_FP64": "demote"})

    def test_clamped_domain_emulate(self):
        with tempfile.TemporaryDirectory() as cache:
            _run_metal_script(self, _ACOS_SNIPPET, cache, env_extra={"WARP_METAL_FP64": "emulate"})


def _has_torch_mps() -> bool:
    try:
        import torch  # noqa: PLC0415

        return torch.backends.mps.is_available()
    except Exception:
        return False


_TORCH_ORDER_SNIPPET = """
import numpy as np
import torch

dev = "metal:0"
N = 1 << 16

@wp.kernel
def busy(a: wp.array[float], iters: int):
    i = wp.tid()
    x = a[i]
    for _k in range(iters):
        x = wp.sin(x) * 0.5 + 0.25
    a[i] = x

@wp.kernel
def setv(a: wp.array[float], v: float):
    a[wp.tid()] = v

@wp.kernel
def copyk(a: wp.array[float], b: wp.array[float]):
    b[wp.tid()] = a[wp.tid()]

scratch = wp.zeros(N, dtype=float, device=dev)
def make_busy():
    # Keeps the GPU busy so later launches are batched, not committed.
    wp.launch(busy, dim=N, inputs=[scratch, 2000], device=dev)

make_busy(); wp.synchronize_device(dev)

# Warp writes, Torch reads without wp.synchronize (CUDA same-stream habit).
a = wp.zeros(N, dtype=float, device=dev)
t = wp.to_torch(a)
for it in range(10):
    make_busy()
    wp.launch(setv, dim=N, inputs=[a, float(it + 1)], device=dev)
    assert float(t.sum().item()) == N * (it + 1), ("stale torch read", it)

# zero_() / fill_() on the exported array, read by Torch.
for it in range(5):
    make_busy()
    wp.launch(setv, dim=N, inputs=[a, 7.0], device=dev)
    a.zero_()
    assert float(t.abs().sum().item()) == 0.0, ("stale zero_", it)
    make_busy()
    a.fill_(3.0)
    assert float(t.sum().item()) == 3.0 * N, ("stale fill_", it)

# Torch overwrites a buffer that a pending Warp launch reads.
tt = torch.zeros(N, device="mps")
w = wp.from_torch(tt)
out = wp.zeros(N, dtype=float, device=dev)
for it in range(10):
    tt.fill_(float(it + 1))
    torch.mps.synchronize()
    make_busy()
    wp.launch(copyk, dim=N, inputs=[w, out], device=dev)
    tt.fill_(-1.0)
    torch.mps.synchronize()
    np.testing.assert_array_equal(out.numpy(), np.full(N, it + 1, np.float32))
"""


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
@unittest.skipUnless(_has_torch_mps(), "Torch with MPS is not installed")
class TestMetalSharedBufferOrdering(unittest.TestCase):
    # Batched launches stay uncommitted while the GPU is busy; Metal orders
    # work across command queues (Warp's vs Torch's) only once committed, so
    # launches binding a buffer shared with Torch must commit immediately.
    def test_torch_interop_without_sync(self):
        _run_with_metal_enabled(self, _TORCH_ORDER_SNIPPET, timeout=300)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalDispatcherRobustness(unittest.TestCase):
    def test_bad_autoflush_env_falls_back(self):
        import warnings  # noqa: PLC0415

        from warp._src.metal_dispatch import _autoflush_from_env  # noqa: PLC0415

        old = os.environ.get("WARP_METAL_AUTOFLUSH")
        try:
            os.environ["WARP_METAL_AUTOFLUSH"] = "lots"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertEqual(_autoflush_from_env(), 64)
            self.assertTrue(any("WARP_METAL_AUTOFLUSH" in str(w.message) for w in caught))
            os.environ["WARP_METAL_AUTOFLUSH"] = "8"
            self.assertEqual(_autoflush_from_env(), 8)
        finally:
            if old is None:
                os.environ.pop("WARP_METAL_AUTOFLUSH", None)
            else:
                os.environ["WARP_METAL_AUTOFLUSH"] = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
