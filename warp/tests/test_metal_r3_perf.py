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


if __name__ == "__main__":
    unittest.main(verbosity=2)
