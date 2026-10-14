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


if __name__ == "__main__":
    unittest.main(verbosity=2)
