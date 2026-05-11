# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 3d — end-to-end ``wp.launch`` on Metal vs CPU.

These tests are the canonical CPU/Metal A/B verification for the backend.
Each kernel is launched on both ``device='cpu'`` and ``device='metal:0'``
and the resulting ``wp.array`` outputs are compared bit-exactly. Any drift
indicates a codegen or runtime bug.

The ``warp.config.enable_metal`` flag must be set before ``wp.init`` runs,
which means tests that need the Metal device live in subprocesses.
"""

import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import unittest


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _has_mlx() -> bool:
    try:
        import mlx.core as mx  # noqa: PLC0415

        return mx.metal.is_available()
    except Exception:
        return False


def _run_with_metal_enabled(test_case, snippet: str, timeout: int = 30):
    """Run a snippet with ``enable_metal=True``; fail the test if exit != 0.

    The snippet is written to a temporary ``.py`` file before execution
    because Warp's codegen calls ``inspect.getsourcelines()`` on
    ``@wp.kernel``-decorated functions, which fails for code passed via
    ``python -c``.
    """
    # Allow the whole suite to be re-run under native dispatch by setting
    # ``WARP_METAL_NATIVE_DISPATCH=1`` in the environment. The flag must
    # be set before ``wp.init()`` so the allocator picks the right backend.
    prefix = "import warp as wp\nwp.config.enable_metal = True\n"
    if os.environ.get("WARP_METAL_NATIVE_DISPATCH") == "1":
        prefix += "wp.config.metal_native_dispatch = True\n"
    code = prefix + "wp.init()\n" + snippet
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="warp_metal_test_") as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    test_case.assertEqual(
        result.returncode,
        0,
        f"subprocess exited with {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
    )


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalLaunch(unittest.TestCase):
    def test_vector_add_matches_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32),
                  c: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                c[tid] = a[tid] + b[tid]

            N = 1024
            rng = np.random.default_rng(0)
            an = rng.standard_normal(N).astype(np.float32)
            bn = rng.standard_normal(N).astype(np.float32)

            a_cpu = wp.array(an, dtype=wp.float32, device='cpu')
            b_cpu = wp.array(bn, dtype=wp.float32, device='cpu')
            c_cpu = wp.zeros(N, dtype=wp.float32, device='cpu')
            wp.launch(k, dim=N, inputs=[a_cpu, b_cpu], outputs=[c_cpu], device='cpu')

            a_m = wp.array(an, dtype=wp.float32, device='metal:0')
            b_m = wp.array(bn, dtype=wp.float32, device='metal:0')
            c_m = wp.zeros(N, dtype=wp.float32, device='metal:0')
            wp.launch(k, dim=N, inputs=[a_m, b_m], outputs=[c_m], device='metal:0')

            np.testing.assert_array_equal(c_cpu.numpy(), c_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_vector_mul_int_matches_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.int32), b: wp.array(dtype=wp.int32),
                  c: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                c[tid] = a[tid] * b[tid]

            rng = np.random.default_rng(7)
            N = 512
            an = rng.integers(-1000, 1000, size=N, dtype=np.int32)
            bn = rng.integers(-1000, 1000, size=N, dtype=np.int32)

            c_cpu = wp.zeros(N, dtype=wp.int32, device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.int32, device='cpu'),
                              wp.array(bn, dtype=wp.int32, device='cpu')],
                      outputs=[c_cpu], device='cpu')

            c_m = wp.zeros(N, dtype=wp.int32, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.int32, device='metal:0'),
                              wp.array(bn, dtype=wp.int32, device='metal:0')],
                      outputs=[c_m], device='metal:0')

            np.testing.assert_array_equal(c_cpu.numpy(), c_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_unaligned_size_matches_cpu(self):
        # 257 is not a multiple of any common threadgroup size (256 etc.).
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32),
                  c: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                c[tid] = a[tid] + b[tid]

            rng = np.random.default_rng(11)
            for N in (1, 7, 257, 1023):
                an = rng.standard_normal(N).astype(np.float32)
                bn = rng.standard_normal(N).astype(np.float32)
                c_cpu = wp.zeros(N, dtype=wp.float32, device='cpu')
                c_m = wp.zeros(N, dtype=wp.float32, device='metal:0')
                wp.launch(k, dim=N,
                          inputs=[wp.array(an, dtype=wp.float32, device='cpu'),
                                  wp.array(bn, dtype=wp.float32, device='cpu')],
                          outputs=[c_cpu], device='cpu')
                wp.launch(k, dim=N,
                          inputs=[wp.array(an, dtype=wp.float32, device='metal:0'),
                                  wp.array(bn, dtype=wp.float32, device='metal:0')],
                          outputs=[c_m], device='metal:0')
                np.testing.assert_array_equal(c_cpu.numpy(), c_m.numpy()), f'N={N} mismatch'
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_large_size_matches_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32),
                  c: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                c[tid] = a[tid] + b[tid]

            N = 1 << 20
            rng = np.random.default_rng(2026)
            an = rng.standard_normal(N).astype(np.float32)
            bn = rng.standard_normal(N).astype(np.float32)
            c_cpu = wp.zeros(N, dtype=wp.float32, device='cpu')
            c_m = wp.zeros(N, dtype=wp.float32, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='cpu'),
                              wp.array(bn, dtype=wp.float32, device='cpu')],
                      outputs=[c_cpu], device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='metal:0'),
                              wp.array(bn, dtype=wp.float32, device='metal:0')],
                      outputs=[c_m], device='metal:0')
            np.testing.assert_array_equal(c_cpu.numpy(), c_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=60)

    def test_repeated_launch_uses_cache(self):
        # Launching the same kernel twice should not regenerate the
        # MetalKernelArtifact / mx.fast.metal_kernel — cache lives on the
        # Warp Kernel object.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32),
                  c: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                c[tid] = a[tid] + b[tid]

            N = 8
            an = np.ones(N, dtype=np.float32)
            bn = np.full(N, 2.0, dtype=np.float32)
            c1 = wp.zeros(N, dtype=wp.float32, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='metal:0'),
                              wp.array(bn, dtype=wp.float32, device='metal:0')],
                      outputs=[c1], device='metal:0')
            artifact_after_first = k._metal_artifact
            # Cache key differs between MLX (``_metal_mlx_kernel``) and
            # native (``_metal_native_pso``) dispatch — pick whichever
            # is in use so this regression covers both backends.
            cache_attr = '_metal_native_pso' if wp.config.metal_native_dispatch else '_metal_mlx_kernel'
            cached_after_first = getattr(k, cache_attr)

            c2 = wp.zeros(N, dtype=wp.float32, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='metal:0'),
                              wp.array(bn, dtype=wp.float32, device='metal:0')],
                      outputs=[c2], device='metal:0')

            assert k._metal_artifact is artifact_after_first, 'artifact was regenerated'
            assert getattr(k, cache_attr) is cached_after_first, \
                f'{cache_attr} was regenerated'
            np.testing.assert_array_equal(c1.numpy(), c2.numpy())
            np.testing.assert_array_equal(c1.numpy(), np.full(N, 3.0, dtype=np.float32))
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_rejects_non_metal_array_input(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32),
                  c: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                c[tid] = a[tid] + b[tid]

            a_cpu = wp.array(np.zeros(8, dtype=np.float32), dtype=wp.float32, device='cpu')
            b_m = wp.zeros(8, dtype=wp.float32, device='metal:0')
            c_m = wp.zeros(8, dtype=wp.float32, device='metal:0')
            try:
                wp.launch(k, dim=8, inputs=[a_cpu, b_m], outputs=[c_m], device='metal:0')
            except RuntimeError as e:
                if 'must be a wp.array on a Metal' in str(e):
                    pass
                else:
                    raise
            else:
                raise AssertionError('expected RuntimeError when passing CPU array to Metal launch')
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_if_else_relu_matches_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                if a[tid] > 0.0:
                    c[tid] = a[tid]
                else:
                    c[tid] = 0.0

            N = 1024
            rng = np.random.default_rng(123)
            an = rng.standard_normal(N).astype(np.float32)

            c_cpu = wp.zeros(N, dtype=wp.float32, device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='cpu')],
                      outputs=[c_cpu], device='cpu')
            c_m = wp.zeros(N, dtype=wp.float32, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='metal:0')],
                      outputs=[c_m], device='metal:0')
            np.testing.assert_array_equal(c_cpu.numpy(), c_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_if_else_select_matches_cpu(self):
        # Two branches selected by an int32 condition array — exercises mixed
        # dtypes plus the ``!= 0`` comparison.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32),
                  b: wp.array(dtype=wp.float32),
                  cond: wp.array(dtype=wp.int32),
                  out: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                if cond[tid] != 0:
                    out[tid] = a[tid]
                else:
                    out[tid] = b[tid]

            N = 512
            rng = np.random.default_rng(7)
            an = rng.standard_normal(N).astype(np.float32)
            bn = rng.standard_normal(N).astype(np.float32)
            cn = rng.integers(0, 2, size=N, dtype=np.int32)

            out_cpu = wp.zeros(N, dtype=wp.float32, device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='cpu'),
                              wp.array(bn, dtype=wp.float32, device='cpu'),
                              wp.array(cn, dtype=wp.int32, device='cpu')],
                      outputs=[out_cpu], device='cpu')
            out_m = wp.zeros(N, dtype=wp.float32, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='metal:0'),
                              wp.array(bn, dtype=wp.float32, device='metal:0'),
                              wp.array(cn, dtype=wp.int32, device='metal:0')],
                      outputs=[out_m], device='metal:0')
            np.testing.assert_array_equal(out_cpu.numpy(), out_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_if_else_all_comparison_ops_match_cpu(self):
        # Exercises every binary comparison operator the codegen needs to
        # support: ``<``, ``<=``, ``==``, ``!=``, ``>=``, ``>``. Each operator
        # gates a write of a distinct constant so the output uniquely
        # identifies which branch fired.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def klt(a: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                if a[tid] < 0.0:
                    c[tid] = 1
                else:
                    c[tid] = 0

            @wp.kernel
            def kle(a: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                if a[tid] <= 0.0:
                    c[tid] = 1
                else:
                    c[tid] = 0

            @wp.kernel
            def keq(a: wp.array(dtype=wp.int32), c: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                if a[tid] == 7:
                    c[tid] = 1
                else:
                    c[tid] = 0

            @wp.kernel
            def kne(a: wp.array(dtype=wp.int32), c: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                if a[tid] != 7:
                    c[tid] = 1
                else:
                    c[tid] = 0

            @wp.kernel
            def kge(a: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                if a[tid] >= 0.0:
                    c[tid] = 1
                else:
                    c[tid] = 0

            @wp.kernel
            def kgt(a: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                if a[tid] > 0.0:
                    c[tid] = 1
                else:
                    c[tid] = 0

            rng = np.random.default_rng(2026)
            for kf, dtype in [(klt, np.float32), (kle, np.float32),
                              (keq, np.int32), (kne, np.int32),
                              (kge, np.float32), (kgt, np.float32)]:
                if dtype == np.float32:
                    an = rng.standard_normal(257).astype(np.float32)
                    a_w = lambda d: wp.array(an, dtype=wp.float32, device=d)
                else:
                    an = rng.integers(0, 15, size=257, dtype=np.int32)
                    a_w = lambda d: wp.array(an, dtype=wp.int32, device=d)
                c_cpu = wp.zeros(257, dtype=wp.int32, device='cpu')
                c_m = wp.zeros(257, dtype=wp.int32, device='metal:0')
                wp.launch(kf, dim=257, inputs=[a_w('cpu')], outputs=[c_cpu], device='cpu')
                wp.launch(kf, dim=257, inputs=[a_w('metal:0')], outputs=[c_m], device='metal:0')
                np.testing.assert_array_equal(
                    c_cpu.numpy(), c_m.numpy(),
                    err_msg=f'kernel {kf.adj.fun_name} mismatch'
                )
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_math_builtins_match_cpu(self):
        # Each unary math op tested against CPU. A handful of these (e.g.
        # ``round``, transcendentals like ``sin`` for large args) may differ
        # from CPU at the ulp level if MSL and the CPU LLVM math library
        # diverge in their rounding policy — we use a tight ``assert_allclose``
        # rather than ``assert_array_equal`` so those don't fail spuriously.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k_sqrt(a: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.float32)):
                tid = wp.tid(); c[tid] = wp.sqrt(a[tid])
            @wp.kernel
            def k_abs(a: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.float32)):
                tid = wp.tid(); c[tid] = wp.abs(a[tid])
            @wp.kernel
            def k_floor(a: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.float32)):
                tid = wp.tid(); c[tid] = wp.floor(a[tid])
            @wp.kernel
            def k_ceil(a: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.float32)):
                tid = wp.tid(); c[tid] = wp.ceil(a[tid])
            @wp.kernel
            def k_exp(a: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.float32)):
                tid = wp.tid(); c[tid] = wp.exp(a[tid])
            @wp.kernel
            def k_log(a: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.float32)):
                tid = wp.tid(); c[tid] = wp.log(a[tid])
            @wp.kernel
            def k_sin(a: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.float32)):
                tid = wp.tid(); c[tid] = wp.sin(a[tid])
            @wp.kernel
            def k_cos(a: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.float32)):
                tid = wp.tid(); c[tid] = wp.cos(a[tid])
            @wp.kernel
            def k_tanh(a: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.float32)):
                tid = wp.tid(); c[tid] = wp.tanh(a[tid])

            rng = np.random.default_rng(2026)
            N = 257
            cases = [
                (k_sqrt, lambda: np.abs(rng.standard_normal(N).astype(np.float32)) + 1e-3),
                (k_abs,  lambda: rng.standard_normal(N).astype(np.float32)),
                (k_floor,lambda: rng.uniform(-10, 10, size=N).astype(np.float32)),
                (k_ceil, lambda: rng.uniform(-10, 10, size=N).astype(np.float32)),
                (k_exp,  lambda: rng.uniform(-3, 3, size=N).astype(np.float32)),
                (k_log,  lambda: np.abs(rng.standard_normal(N).astype(np.float32)) + 1e-3),
                (k_sin,  lambda: rng.uniform(-3.14, 3.14, size=N).astype(np.float32)),
                (k_cos,  lambda: rng.uniform(-3.14, 3.14, size=N).astype(np.float32)),
                (k_tanh, lambda: rng.standard_normal(N).astype(np.float32)),
            ]
            for kf, gen in cases:
                an = gen()
                c_cpu = wp.zeros(N, dtype=wp.float32, device='cpu')
                c_m = wp.zeros(N, dtype=wp.float32, device='metal:0')
                wp.launch(kf, dim=N,
                          inputs=[wp.array(an, dtype=wp.float32, device='cpu')],
                          outputs=[c_cpu], device='cpu')
                wp.launch(kf, dim=N,
                          inputs=[wp.array(an, dtype=wp.float32, device='metal:0')],
                          outputs=[c_m], device='metal:0')
                np.testing.assert_allclose(
                    c_m.numpy(), c_cpu.numpy(),
                    rtol=1e-5, atol=1e-6,
                    err_msg=f'{kf.adj.fun_name} CPU/Metal disagreement',
                )
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_math_min_max_matches_cpu(self):
        # Binary builtins ``wp.min`` and ``wp.max`` map to ``metal::min`` /
        # ``metal::max``; output is bit-exact for finite inputs because both
        # CPU and Metal pick element-wise without any floating-point math.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def kmin(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32),
                     c: wp.array(dtype=wp.float32)):
                tid = wp.tid(); c[tid] = wp.min(a[tid], b[tid])

            @wp.kernel
            def kmax(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32),
                     c: wp.array(dtype=wp.float32)):
                tid = wp.tid(); c[tid] = wp.max(a[tid], b[tid])

            rng = np.random.default_rng(99)
            N = 512
            an = rng.standard_normal(N).astype(np.float32)
            bn = rng.standard_normal(N).astype(np.float32)
            for kf in (kmin, kmax):
                c_cpu = wp.zeros(N, dtype=wp.float32, device='cpu')
                c_m = wp.zeros(N, dtype=wp.float32, device='metal:0')
                wp.launch(kf, dim=N,
                          inputs=[wp.array(an, dtype=wp.float32, device='cpu'),
                                  wp.array(bn, dtype=wp.float32, device='cpu')],
                          outputs=[c_cpu], device='cpu')
                wp.launch(kf, dim=N,
                          inputs=[wp.array(an, dtype=wp.float32, device='metal:0'),
                                  wp.array(bn, dtype=wp.float32, device='metal:0')],
                          outputs=[c_m], device='metal:0')
                np.testing.assert_array_equal(
                    c_cpu.numpy(), c_m.numpy(),
                    err_msg=f'{kf.adj.fun_name} mismatch',
                )
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_static_range_for_loop_matches_cpu(self):
        # Static ``range(N)`` is unrolled by Warp into straight-line code, so
        # this exercises the unrolled IR plus the ``wp::float()`` constructor
        # cast that surfaces on the accumulator initializer.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                s = float(0.0)
                for i in range(8):
                    s = s + a[tid * 8 + i]
                out[tid] = s

            N = 64
            rng = np.random.default_rng(13)
            an = rng.standard_normal(N * 8).astype(np.float32)
            out_cpu = wp.zeros(N, dtype=wp.float32, device='cpu')
            out_m = wp.zeros(N, dtype=wp.float32, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='cpu')],
                      outputs=[out_cpu], device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='metal:0')],
                      outputs=[out_m], device='metal:0')
            np.testing.assert_array_equal(out_cpu.numpy(), out_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_dynamic_range_for_loop_matches_cpu(self):
        # ``range(n)`` with non-constant ``n`` lowers to a goto-loop in the IR;
        # our preprocessor rewrites it as a real MSL ``for``. Tests several
        # inner sizes to catch off-by-ones in the iteration bound.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32), n: wp.int32, out: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                s = float(0.0)
                for i in range(n):
                    s = s + a[tid * n + i]
                out[tid] = s

            rng = np.random.default_rng(99)
            for inner in (1, 5, 8, 17):
                N = 32
                an = rng.standard_normal(N * inner).astype(np.float32)
                out_cpu = wp.zeros(N, dtype=wp.float32, device='cpu')
                out_m = wp.zeros(N, dtype=wp.float32, device='metal:0')
                wp.launch(k, dim=N,
                          inputs=[wp.array(an, dtype=wp.float32, device='cpu'), inner],
                          outputs=[out_cpu], device='cpu')
                wp.launch(k, dim=N,
                          inputs=[wp.array(an, dtype=wp.float32, device='metal:0'), inner],
                          outputs=[out_m], device='metal:0')
                np.testing.assert_array_equal(
                    out_cpu.numpy(), out_m.numpy(),
                    err_msg=f'inner={inner} mismatch'
                )
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_atomic_add_float_matches_cpu(self):
        # Float atomic_add ordering is race-dependent so the result may differ
        # from CPU at the ulp level for large N. We use an absolute tolerance
        # tied to N * fp32 ulp(typical sum) to cover that without being so
        # loose it'd hide real bugs.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32), c: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                wp.atomic_add(c, 0, a[tid])

            N = 1024
            rng = np.random.default_rng(7)
            an = rng.standard_normal(N).astype(np.float32)

            c_cpu = wp.zeros(1, dtype=wp.float32, device='cpu')
            c_m = wp.zeros(1, dtype=wp.float32, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='cpu')],
                      outputs=[c_cpu], device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='metal:0')],
                      outputs=[c_m], device='metal:0')
            np.testing.assert_allclose(
                c_m.numpy(), c_cpu.numpy(),
                rtol=1e-4, atol=1e-3,
                err_msg='atomic_add(fp32) CPU/Metal mismatch beyond fp tolerance',
            )
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_atomic_add_int_matches_cpu_bit_exact(self):
        # Integer atomic_add is associative, so the result is deterministic
        # regardless of execution order — bit-exact.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.int32), c: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                wp.atomic_add(c, 0, a[tid])

            N = 4096
            rng = np.random.default_rng(99)
            an = rng.integers(-1000, 1000, size=N, dtype=np.int32)

            c_cpu = wp.zeros(1, dtype=wp.int32, device='cpu')
            c_m = wp.zeros(1, dtype=wp.int32, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.int32, device='cpu')],
                      outputs=[c_cpu], device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.int32, device='metal:0')],
                      outputs=[c_m], device='metal:0')
            np.testing.assert_array_equal(c_cpu.numpy(), c_m.numpy())
            assert int(c_cpu.numpy()[0]) == int(an.sum()), (c_cpu.numpy(), an.sum())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_atomic_min_max_int_matches_cpu(self):
        # Integer min/max are deterministic and bit-exact regardless of order.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.int32),
                  out_min: wp.array(dtype=wp.int32),
                  out_max: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                wp.atomic_min(out_min, 0, a[tid])
                wp.atomic_max(out_max, 0, a[tid])

            N = 1024
            rng = np.random.default_rng(11)
            an = rng.integers(-100000, 100000, size=N, dtype=np.int32)

            # The accumulator buffers need sentinel initial values so the
            # min/max work — Warp's CUDA semantics expect users to seed them.
            # Our Metal launcher zero-initializes outputs, so we need the
            # sentinels to live in the input data: a value larger than any
            # element for ``out_min`` and smaller for ``out_max``. We pick
            # values that don't appear in ``an`` and rely on every thread
            # contributing, so the sentinel is overwritten.
            mn_cpu = wp.zeros(1, dtype=wp.int32, device='cpu')
            mx_cpu = wp.zeros(1, dtype=wp.int32, device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.int32, device='cpu')],
                      outputs=[mn_cpu, mx_cpu], device='cpu')

            mn_m = wp.zeros(1, dtype=wp.int32, device='metal:0')
            mx_m = wp.zeros(1, dtype=wp.int32, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.int32, device='metal:0')],
                      outputs=[mn_m, mx_m], device='metal:0')

            # Both sides have the same zero-initial behaviour; compare directly.
            np.testing.assert_array_equal(mn_cpu.numpy(), mn_m.numpy())
            np.testing.assert_array_equal(mx_cpu.numpy(), mx_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_mixed_atomic_and_regular_writes_match_cpu(self):
        # When a kernel uses ``wp.atomic_*`` on one output and a regular
        # ``arr[i] = val`` store on another, MLX makes ALL outputs of the
        # kernel ``device atomic<T>*``. The codegen must translate the
        # regular store into ``atomic_store_explicit`` so it compiles.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32),
                  out_reg: wp.array(dtype=wp.float32),
                  out_atom: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                val = a[tid] * 2.0
                out_reg[tid] = val
                wp.atomic_add(out_atom, 0, val)

            N = 32
            rng = np.random.default_rng(0)
            an = rng.standard_normal(N).astype(np.float32)

            for dev in ('cpu', 'metal:0'):
                a = wp.array(an, dtype=wp.float32, device=dev)
                out_reg = wp.zeros(N, dtype=wp.float32, device=dev)
                out_atom = wp.zeros(1, dtype=wp.float32, device=dev)
                wp.launch(k, dim=N, inputs=[a],
                          outputs=[out_reg, out_atom], device=dev)
                if dev == 'cpu':
                    cpu_reg = out_reg.numpy()
                    cpu_atom = out_atom.numpy()
                else:
                    np.testing.assert_array_equal(out_reg.numpy(), cpu_reg)
                    # Atomic-add ordering is non-deterministic; allow fp tol.
                    np.testing.assert_allclose(out_atom.numpy(), cpu_atom,
                                                rtol=1e-4, atol=1e-5)
            # Sanity: ``out_reg`` is ``a * 2`` element-wise.
            np.testing.assert_array_equal(cpu_reg, an * 2.0)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_slice_view_row_read_matches_cpu(self):
        # ``arr2d[i]`` lowers to ``slice_t(i, i, 0)`` + ``view(arr, slice)``.
        # The view is then indexed as ``view[j]`` to read ``arr2d[i, j]``.
        # Our slice/view preprocessor folds those into direct array ops.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k_row_sum(a: wp.array2d(dtype=wp.float32),
                          out: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                row = a[tid]
                s = float(0.0)
                for j in range(a.shape[1]):
                    s += row[j]
                out[tid] = s

            N, M = 8, 5
            rng = np.random.default_rng(0)
            an = rng.standard_normal((N, M)).astype(np.float32)
            results = {}
            for dev in ('cpu', 'metal:0'):
                a = wp.array(an, dtype=wp.float32, device=dev)
                out = wp.zeros(N, dtype=wp.float32, device=dev)
                wp.launch(k_row_sum, dim=N, inputs=[a], outputs=[out], device=dev)
                results[dev] = out.numpy()
            np.testing.assert_allclose(results['cpu'], results['metal:0'], rtol=1e-5)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_slice_view_atomic_scatter_matches_cpu(self):
        # ``wp.atomic_add(out2d[i], j, val)`` exercises the slice/view path
        # through a multi-arg atomic intrinsic — we must flatten the leading
        # slice index into the underlying array's flat offset.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array2d(dtype=wp.float32),
                  cols: wp.array(dtype=wp.int32),
                  out: wp.array2d(dtype=wp.float32)):
                tid = wp.tid()
                row_in = a[tid]
                j = cols[tid]
                wp.atomic_add(out[tid], j, row_in[j] * 2.0)

            N, M = 8, 5
            rng = np.random.default_rng(0)
            an = rng.standard_normal((N, M)).astype(np.float32)
            cols_n = rng.integers(0, M, size=N, dtype=np.int32)
            results = {}
            for dev in ('cpu', 'metal:0'):
                a = wp.array(an, dtype=wp.float32, device=dev)
                cols = wp.array(cols_n, dtype=wp.int32, device=dev)
                out = wp.zeros((N, M), dtype=wp.float32, device=dev)
                wp.launch(k, dim=N, inputs=[a, cols],
                          outputs=[out], device=dev)
                results[dev] = out.numpy()
            np.testing.assert_allclose(results['cpu'], results['metal:0'], rtol=1e-5)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_big_mat_lookup_table_matches_cpu(self):
        # ``wp.matrix(..., shape=(R, C), dtype=int)`` for non-native sizes
        # (R or C > 4) emits a ``wp_matRxC_<scalar>`` custom struct in the
        # kernel header, with a flat row-major ``_make`` factory and a
        # ``wp_mat_extract`` accessor. Used by mujoco_warp's flex kernels
        # for static edge-vertex lookup tables.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(idx: wp.array(dtype=wp.int32),
                  out: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                table = wp.matrix(
                    10, 11,
                    20, 21,
                    30, 31,
                    40, 41,
                    50, 51,
                    60, 61,
                    shape=(6, 2),
                    dtype=int,
                )
                i = idx[tid]
                out[tid] = table[i, 0] + table[i, 1] * 100

            N = 6
            idx_n = np.arange(N, dtype=np.int32)
            results = {}
            for dev in ('cpu', 'metal:0'):
                idx = wp.array(idx_n, dtype=wp.int32, device=dev)
                out = wp.zeros(N, dtype=wp.int32, device=dev)
                wp.launch(k, dim=N, inputs=[idx], outputs=[out], device=dev)
                results[dev] = out.numpy()
            np.testing.assert_array_equal(results['cpu'], results['metal:0'])
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_big_mat_row_read_write_matches_cpu(self):
        # Non-square ``mat<R, C>`` (e.g. ``mat23``, ``mat63``) routes
        # through the custom ``wp_matRxC_<scalar>`` struct in MSL with
        # a row-write proxy + value-returning const ``operator[]``.
        # This test exercises both the write side
        # (``mat[i] = vec3(...)``) and the read side (``vec3 r =
        # mat[i]``), then mixes them in a ``wp.dot`` (the OBB SAT
        # pattern in mujoco_warp's broadphase).
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            mat23 = wp.types.matrix(shape=(2, 3), dtype=float)
            mat63 = wp.types.matrix(shape=(6, 3), dtype=float)

            @wp.kernel
            def k_rw(out: wp.array2d(dtype=float)):
                tid = wp.tid()
                m = mat23(0.0)
                m[0] = wp.vec3(1.0, 2.0, 3.0)
                m[1] = wp.vec3(4.0, 5.0, 6.0)
                r0 = m[0]
                r1 = m[1]
                out[tid, 0] = r0[0]; out[tid, 1] = r0[1]; out[tid, 2] = r0[2]
                out[tid, 3] = r1[0]; out[tid, 4] = r1[1]; out[tid, 5] = r1[2]

            @wp.kernel
            def k_dot(out: wp.array(dtype=float)):
                tid = wp.tid()
                m = mat23(0.0)
                m[0] = wp.vec3(1.0, 2.0, 3.0)
                m[1] = wp.vec3(4.0, 5.0, 6.0)
                axis = wp.vec3(0.0, 1.0, 0.0)
                # row-read in dot expression: mat[i] should fall through
                # the row proxy's float3 conversion path.
                out[tid] = wp.dot(m[0], axis) + wp.dot(m[1], axis) * 10.0

            @wp.kernel
            def k_6x3(out: wp.array2d(dtype=float)):
                tid = wp.tid()
                n = mat63(0.0)
                for i in range(6):
                    n[i] = wp.vec3(float(i), float(i + 10), float(i + 100))
                for i in range(6):
                    r = n[i]
                    out[tid, i*3 + 0] = r[0]
                    out[tid, i*3 + 1] = r[1]
                    out[tid, i*3 + 2] = r[2]

            N = 4
            for label, kernel, shape, expected in (
                ('rw', k_rw, (N, 6), np.tile([1, 2, 3, 4, 5, 6], (N, 1)).astype(np.float32)),
                ('dot', k_dot, (N,), np.full(N, 52.0, dtype=np.float32)),
                ('6x3', k_6x3, (N, 18),
                    np.tile(
                        [(i, i + 10, i + 100) for i in range(6)], (N, 1)
                    ).reshape(N, 18).astype(np.float32)),
            ):
                results = {}
                for dev in ('cpu', 'metal:0'):
                    out = wp.zeros(shape, dtype=float, device=dev)
                    wp.launch(kernel, dim=N, inputs=[], outputs=[out], device=dev)
                    results[dev] = out.numpy()
                np.testing.assert_array_equal(
                    results['cpu'], results['metal:0'],
                    err_msg=f'{label}: cpu vs metal mismatch',
                )
                np.testing.assert_allclose(
                    results['cpu'], expected, atol=1e-6,
                    err_msg=f'{label}: cpu vs expected mismatch',
                )
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_obb_sat_pattern_matches_cpu(self):
        # Reproduces the exact SAT (separating-axis test) pattern from
        # mujoco_warp's ``_obb_filter``: build a ``mat23`` of world
        # centers, a ``mat63`` of normals, then iterate axes computing
        # ``proj[i] = wp.dot(xc[i], nrm[3*j + k])`` and a per-axis
        # radius. If ``radius_sum + margin < |proj_diff|`` for any
        # axis, return False (boxes separated). The mujoco_warp
        # ``_nxn_broadphase`` uses this via ``wp.static(_broadphase_filter)``;
        # this stripped-down version pins the building block.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            mat23 = wp.types.matrix(shape=(2, 3), dtype=float)
            mat63 = wp.types.matrix(shape=(6, 3), dtype=float)

            @wp.func
            def obb_inner(xc: mat23, nrm: mat63,
                          sa: wp.vec3, sb: wp.vec3, margin: float) -> bool:
                proj = wp.vec2(0.0)
                radius = wp.vec2(0.0)
                for j in range(2):
                    for k in range(3):
                        for i in range(2):
                            proj[i] = wp.dot(xc[i], nrm[3*j + k])
                            if i == 0:
                                sz = sa
                            else:
                                sz = sb
                            radius[i] = (
                                wp.abs(sz[0] * wp.dot(nrm[3*i + 0], nrm[3*j + k]))
                                + wp.abs(sz[1] * wp.dot(nrm[3*i + 1], nrm[3*j + k]))
                                + wp.abs(sz[2] * wp.dot(nrm[3*i + 2], nrm[3*j + k]))
                            )
                        if radius[0] + radius[1] + margin < wp.abs(proj[1] - proj[0]):
                            return False
                return True

            @wp.kernel
            def k(
                xc_arr: wp.array2d(dtype=wp.vec3),
                nrm_arr: wp.array2d(dtype=wp.vec3),
                sa_arr: wp.array(dtype=wp.vec3),
                sb_arr: wp.array(dtype=wp.vec3),
                margin_arr: wp.array(dtype=float),
                out: wp.array(dtype=int),
            ):
                pid = wp.tid()
                xc = mat23(0.0); xc[0] = xc_arr[pid, 0]; xc[1] = xc_arr[pid, 1]
                nrm = mat63(0.0)
                nrm[0] = nrm_arr[pid, 0]; nrm[1] = nrm_arr[pid, 1]; nrm[2] = nrm_arr[pid, 2]
                nrm[3] = nrm_arr[pid, 3]; nrm[4] = nrm_arr[pid, 4]; nrm[5] = nrm_arr[pid, 5]
                out[pid] = int(obb_inner(xc, nrm, sa_arr[pid], sb_arr[pid], margin_arr[pid]))

            # 6 test pairs: rail-pole-style (separated along rail.y),
            # cart-rail (overlapping), distant boxes, overlapping unit
            # boxes, plus 2 dummy fillers. Same structure as the cartpole
            # broadphase candidates.
            N = 6
            xc = np.zeros((N, 2, 3), dtype=np.float32)
            nrm = np.zeros((N, 6, 3), dtype=np.float32)
            sa = np.zeros((N, 3), dtype=np.float32)
            sb = np.zeros((N, 3), dtype=np.float32)
            margin = np.zeros(N, dtype=np.float32)

            # Pair 0: rail (long along x, at y=0.07) vs vertical pole
            xc[0] = [[0, 0.07, 1], [0, 0, 1.5]]
            xmat1 = np.array([[2.22e-16, 0, 1], [0, 1, 0], [-1, 0, 2.22e-16]])
            xmat2 = np.array([[1, 0, 0], [0, -1, -1.22e-16], [0, 1.22e-16, -1]])
            nrm[0] = np.array([xmat1[:, 0], xmat1[:, 1], xmat1[:, 2],
                               xmat2[:, 0], xmat2[:, 1], xmat2[:, 2]])
            sa[0] = [0.02, 0.02, 2.02]; sb[0] = [0.045, 0.045, 0.545]

            # Pair 1: cart vs rail (overlapping)
            xc[1] = [[0, 0, 1], [0, 0.07, 1]]
            nrm[1] = np.array([[1,0,0],[0,1,0],[0,0,1],
                               xmat1[:, 0], xmat1[:, 1], xmat1[:, 2]])
            sa[1] = [0.2, 0.15, 0.1]; sb[1] = [0.02, 0.02, 2.02]

            # Pair 2: distant boxes (separated)
            xc[2] = [[0, 0, 0], [0, 0, 5]]
            nrm[2] = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1]])
            sa[2] = [1, 1, 1]; sb[2] = [1, 1, 1]

            # Pair 3: overlapping unit boxes
            xc[3] = [[0, 0, 0], [0, 0, 0]]
            nrm[3] = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1]])
            sa[3] = [1, 1, 1]; sb[3] = [1, 1, 1]

            # Pairs 4 and 5 stay at zeros (overlapping degenerate)
            nrm[4] = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1]])
            nrm[5] = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1]])

            results = {}
            for dev in ('cpu', 'metal:0'):
                a_xc = wp.zeros((N, 2), dtype=wp.vec3, device=dev); a_xc.assign(xc)
                a_nrm = wp.zeros((N, 6), dtype=wp.vec3, device=dev); a_nrm.assign(nrm)
                a_sa = wp.zeros(N, dtype=wp.vec3, device=dev); a_sa.assign(sa)
                a_sb = wp.zeros(N, dtype=wp.vec3, device=dev); a_sb.assign(sb)
                a_m = wp.zeros(N, dtype=float, device=dev); a_m.assign(margin)
                out = wp.zeros(N, dtype=int, device=dev)
                wp.launch(k, dim=N,
                          inputs=[a_xc, a_nrm, a_sa, a_sb, a_m],
                          outputs=[out], device=dev)
                results[dev] = out.numpy()

            # Pair 0 (rail-pole): separated → 0. Pair 1 (cart-rail): not → 1.
            # Pair 2 (distant): separated → 0. Pair 3 (overlap): not → 1.
            expected = np.array([0, 1, 0, 1, 1, 1], dtype=np.int32)
            np.testing.assert_array_equal(results['cpu'], expected,
                err_msg='cpu got wrong OBB SAT result vs expected')
            np.testing.assert_array_equal(results['cpu'], results['metal:0'],
                err_msg='OBB SAT cpu vs metal mismatch')
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_big_mat_broadcast_constructor_matches_cpu(self):
        # ``mat<R, C>(scalar)`` (single-arg broadcast) emits a one-arg
        # ``wp_matRxC_<scalar>_make(s)`` factory that fills every
        # element with ``s``. Used at variable declaration sites
        # (``m = mat23(0.0)``) throughout the contact pipeline.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            mat23 = wp.types.matrix(shape=(2, 3), dtype=float)
            mat63 = wp.types.matrix(shape=(6, 3), dtype=float)

            @wp.kernel
            def k(out2x3: wp.array2d(dtype=float),
                  out6x3: wp.array2d(dtype=float)):
                tid = wp.tid()
                m23 = mat23(7.0)  # broadcast: every element 7
                m63 = mat63(-3.5)
                # Read all elements via row access
                for r in range(2):
                    row = m23[r]
                    for c in range(3):
                        out2x3[tid, r * 3 + c] = row[c]
                for r in range(6):
                    row = m63[r]
                    for c in range(3):
                        out6x3[tid, r * 3 + c] = row[c]

            N = 4
            results_2x3 = {}
            results_6x3 = {}
            for dev in ('cpu', 'metal:0'):
                o23 = wp.zeros((N, 6), dtype=float, device=dev)
                o63 = wp.zeros((N, 18), dtype=float, device=dev)
                wp.launch(k, dim=N, inputs=[], outputs=[o23, o63], device=dev)
                results_2x3[dev] = o23.numpy()
                results_6x3[dev] = o63.numpy()

            np.testing.assert_array_equal(results_2x3['cpu'], results_2x3['metal:0'])
            np.testing.assert_array_equal(results_6x3['cpu'], results_6x3['metal:0'])
            np.testing.assert_array_equal(
                results_2x3['cpu'], np.full((N, 6), 7.0, dtype=np.float32))
            np.testing.assert_array_equal(
                results_6x3['cpu'], np.full((N, 18), -3.5, dtype=np.float32))
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_struct_to_struct_local_copy_matches_cpu(self):
        # ``var_X = var_Y`` where both are struct locals split into per-
        # field per-element locals. The body emitter expands this into
        # one ``var_X__field = var_Y__field`` per non-array field.
        # Pattern used by mujoco_warp when a ``@wp.func`` returns a
        # ``Struct`` value (e.g. ``geom_collision_pair`` returning
        # ``Geom``).
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.struct
            class Geom:
                pos: wp.vec3
                size: wp.vec3
                idx: int

            @wp.func
            def make_geom(i: int) -> Geom:
                g = Geom()
                g.pos = wp.vec3(float(i), float(i*10), float(i*100))
                g.size = wp.vec3(float(i + 1), float(i + 2), float(i + 3))
                g.idx = i
                return g

            @wp.kernel
            def k(out_pos: wp.array2d(dtype=float),
                  out_size: wp.array2d(dtype=float),
                  out_idx: wp.array(dtype=int)):
                tid = wp.tid()
                a = make_geom(tid)
                b = a   # struct-to-struct copy → per-field copies
                out_pos[tid, 0] = b.pos[0]
                out_pos[tid, 1] = b.pos[1]
                out_pos[tid, 2] = b.pos[2]
                out_size[tid, 0] = b.size[0]
                out_size[tid, 1] = b.size[1]
                out_size[tid, 2] = b.size[2]
                out_idx[tid] = b.idx

            N = 4
            results = {}
            for dev in ('cpu', 'metal:0'):
                op = wp.zeros((N, 3), dtype=float, device=dev)
                os_ = wp.zeros((N, 3), dtype=float, device=dev)
                oi = wp.zeros(N, dtype=int, device=dev)
                wp.launch(k, dim=N, inputs=[], outputs=[op, os_, oi], device=dev)
                results[dev] = (op.numpy(), os_.numpy(), oi.numpy())

            for j in range(3):
                np.testing.assert_array_equal(results['cpu'][j], results['metal:0'][j])
            # Sanity vs expected
            expected_pos = np.array([[i, i*10, i*100] for i in range(N)], dtype=np.float32)
            expected_size = np.array([[i+1, i+2, i+3] for i in range(N)], dtype=np.float32)
            np.testing.assert_array_equal(results['cpu'][0], expected_pos)
            np.testing.assert_array_equal(results['cpu'][1], expected_size)
            np.testing.assert_array_equal(results['cpu'][2], np.arange(N, dtype=np.int32))
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_vec_mat_element_array_access_matches_cpu(self):
        # vec3-element and mat33-element 2-D arrays go through the
        # ``__floats_packed`` packer when the kernel exceeds the 31-
        # buffer slot cap. This test runs them in a small kernel where
        # packing isn't activated, so it pins the *unpacked* path.
        # The packed path is exercised by the larger mujoco_warp
        # integration tests.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k_mat33(mats: wp.array2d(dtype=wp.mat33),
                        out: wp.array2d(dtype=float)):
                tid = wp.tid()
                m = mats[0, tid]
                for r in range(3):
                    for c in range(3):
                        out[tid, r * 3 + c] = m[r, c]

            @wp.kernel
            def k_vec3(vecs: wp.array2d(dtype=wp.vec3),
                       out: wp.array2d(dtype=float)):
                tid = wp.tid()
                v = vecs[0, tid]
                out[tid, 0] = v[0]
                out[tid, 1] = v[1]
                out[tid, 2] = v[2]

            N = 4
            mats_np = np.arange(N * 9, dtype=np.float32).reshape(1, N, 3, 3)
            vecs_np = np.arange(N * 3, dtype=np.float32).reshape(1, N, 3)
            for label, kernel, in_arr, in_dt, out_shape, expected in (
                ('mat33', k_mat33, mats_np, wp.mat33, (N, 9),
                    np.arange(N * 9, dtype=np.float32).reshape(N, 9)),
                ('vec3', k_vec3, vecs_np, wp.vec3, (N, 3),
                    np.arange(N * 3, dtype=np.float32).reshape(N, 3)),
            ):
                results = {}
                for dev in ('cpu', 'metal:0'):
                    a_in = wp.zeros((1, N), dtype=in_dt, device=dev)
                    a_in.assign(in_arr)
                    out = wp.zeros(out_shape, dtype=float, device=dev)
                    wp.launch(kernel, dim=N, inputs=[a_in], outputs=[out], device=dev)
                    results[dev] = out.numpy()
                np.testing.assert_array_equal(results['cpu'], results['metal:0'],
                    err_msg=f'{label}: cpu vs metal mismatch')
                np.testing.assert_array_equal(results['cpu'], expected,
                    err_msg=f'{label}: cpu vs expected mismatch')
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_vec2_conditional_element_write_matches_cpu(self):
        # ``wp::assign_inplace(vec, idx, val)`` (3-arg form) for
        # ``vec[i] = val`` inside an ``if`` branch. The OBB SAT loop
        # in mujoco_warp's broadphase uses ``proj[i] = ...`` and
        # ``radius[i] = ...`` repeatedly. This test pins the simpler
        # case of conditional element writes and verifies that the
        # ``thread vec_t&`` returned by our op[] proxy works in a
        # branched context.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(flags: wp.array(dtype=int),
                  out: wp.array(dtype=wp.vec2)):
                tid = wp.tid()
                v = wp.vec2(0.0)
                if flags[tid] == 0:
                    v[0] = 10.0
                else:
                    v[1] = 20.0
                out[tid] = v

            N = 4
            flags_np = np.array([0, 1, 0, 1], dtype=np.int32)
            results = {}
            for dev in ('cpu', 'metal:0'):
                a_flags = wp.zeros(N, dtype=int, device=dev); a_flags.assign(flags_np)
                out = wp.zeros(N, dtype=wp.vec2, device=dev)
                wp.launch(k, dim=N, inputs=[a_flags], outputs=[out], device=dev)
                results[dev] = out.numpy()
            expected = np.array([[10.0, 0.0], [0.0, 20.0],
                                 [10.0, 0.0], [0.0, 20.0]], dtype=np.float32)
            np.testing.assert_array_equal(results['cpu'], results['metal:0'])
            np.testing.assert_array_equal(results['cpu'], expected)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_obb_full_broadphase_clone_matches_cpu(self):
        # End-to-end clone of mujoco_warp's broadphase OBB filter on
        # cartpole geometry: 2-D tid (worldid, elementid), read
        # ``wp.vec2i`` from ``nxn_geom_pair[elementid]``, dispatch
        # PLANE / SPHERE / OBB filter via wrapper @wp.func returning
        # bool, write per-pair pass/reject to a flat int array.
        # Mirrors the actual ``_nxn_broadphase`` kernel structure
        # closely enough that it would catch this kind of broadphase
        # codegen regression.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            mat23 = wp.types.matrix(shape=(2, 3), dtype=float)
            mat63 = wp.types.matrix(shape=(6, 3), dtype=float)

            @wp.func
            def plane_filter(s1: float, s2: float, m1: float, m2: float,
                             xp1: wp.vec3, xp2: wp.vec3,
                             xm1: wp.mat33, xm2: wp.mat33) -> bool:
                if s1 == 0.0:
                    d = wp.dot(xp2 - xp1, wp.vec3(xm1[0, 2], xm1[1, 2], xm1[2, 2]))
                    return d <= s2 + m1 + m2
                elif s2 == 0.0:
                    d = wp.dot(xp1 - xp2, wp.vec3(xm2[0, 2], xm2[1, 2], xm2[2, 2]))
                    return d <= s1 + m1 + m2
                return True

            @wp.func
            def sphere_filter(rb1: float, rb2: float, m1: float, m2: float,
                              xp1: wp.vec3, xp2: wp.vec3) -> bool:
                d = xp2 - xp1
                threshold = rb1 + rb2 + m1 + m2
                return wp.dot(d, d) <= threshold * threshold

            @wp.func
            def obb_filter(c1: wp.vec3, c2: wp.vec3, s1: wp.vec3, s2: wp.vec3,
                           m1: float, m2: float, xp1: wp.vec3, xp2: wp.vec3,
                           xm1: wp.mat33, xm2: wp.mat33) -> bool:
                margin = m1 + m2
                xc = mat23(0.0)
                nrm = mat63(0.0)
                proj = wp.vec2(0.0)
                radius = wp.vec2(0.0)
                xc[0] = xm1 @ c1 + xp1
                xc[1] = xm2 @ c2 + xp2
                nrm[0] = wp.vec3(xm1[0, 0], xm1[1, 0], xm1[2, 0])
                nrm[1] = wp.vec3(xm1[0, 1], xm1[1, 1], xm1[2, 1])
                nrm[2] = wp.vec3(xm1[0, 2], xm1[1, 2], xm1[2, 2])
                nrm[3] = wp.vec3(xm2[0, 0], xm2[1, 0], xm2[2, 0])
                nrm[4] = wp.vec3(xm2[0, 1], xm2[1, 1], xm2[2, 1])
                nrm[5] = wp.vec3(xm2[0, 2], xm2[1, 2], xm2[2, 2])
                for j in range(2):
                    for k in range(3):
                        for i in range(2):
                            proj[i] = wp.dot(xc[i], nrm[3*j + k])
                            if i == 0:
                                sz = s1
                            else:
                                sz = s2
                            radius[i] = (
                                wp.abs(sz[0] * wp.dot(nrm[3*i + 0], nrm[3*j + k]))
                                + wp.abs(sz[1] * wp.dot(nrm[3*i + 1], nrm[3*j + k]))
                                + wp.abs(sz[2] * wp.dot(nrm[3*i + 2], nrm[3*j + k]))
                            )
                        if radius[0] + radius[1] + margin < wp.abs(proj[1] - proj[0]):
                            return False
                return True

            @wp.func
            def broadphase_filter(centers: wp.array2d(dtype=wp.vec3),
                                  sizes: wp.array2d(dtype=wp.vec3),
                                  rbounds: wp.array(dtype=float),
                                  margins: wp.array(dtype=float),
                                  xpos: wp.array2d(dtype=wp.vec3),
                                  xmat: wp.array2d(dtype=wp.mat33),
                                  g1: int, g2: int, w: int) -> bool:
                rb1 = rbounds[g1]; rb2 = rbounds[g2]
                m1 = margins[g1]; m2 = margins[g2]
                if rb1 == 0.0 or rb2 == 0.0:
                    return plane_filter(rb1, rb2, m1, m2,
                                        xpos[w, g1], xpos[w, g2],
                                        xmat[w, g1], xmat[w, g2])
                if not sphere_filter(rb1, rb2, m1, m2,
                                     xpos[w, g1], xpos[w, g2]):
                    return False
                if not obb_filter(centers[w, g1], centers[w, g2],
                                  sizes[w, g1], sizes[w, g2],
                                  m1, m2,
                                  xpos[w, g1], xpos[w, g2],
                                  xmat[w, g1], xmat[w, g2]):
                    return False
                return True

            @wp.kernel
            def kernel(
                pairs: wp.array(dtype=wp.vec2i),
                centers: wp.array2d(dtype=wp.vec3),
                sizes: wp.array2d(dtype=wp.vec3),
                rbounds: wp.array(dtype=float),
                margins: wp.array(dtype=float),
                xpos: wp.array2d(dtype=wp.vec3),
                xmat: wp.array2d(dtype=wp.mat33),
                out: wp.array(dtype=int),
            ):
                worldid, elementid = wp.tid()
                pair = pairs[elementid]
                g1 = pair[0]; g2 = pair[1]
                out[elementid] = int(broadphase_filter(
                    centers, sizes, rbounds, margins, xpos, xmat,
                    g1, g2, worldid))

            # Cartpole-equivalent geometry: floor (plane), 2 rails,
            # 1 cart, 1 pole. Candidate pairs are the cross-product of
            # rails and bodies (after ``conaffinity`` filtering at model
            # build time would prune sibling-body pairs in mjlab).
            NGEOMS = 5
            xpos_np = np.array([
                [0, 0, -0.05], [0, 0.07, 1], [0, -0.07, 1],
                [0, 0, 1], [0, 0, 1.5],
            ], dtype=np.float32)
            xmat_np = np.zeros((NGEOMS, 9), dtype=np.float32)
            xmat_np[0] = np.eye(3).flatten()
            xmat_np[1] = np.array([[2.22e-16, 0, 1], [0, 1, 0], [-1, 0, 2.22e-16]]).flatten()
            xmat_np[2] = xmat_np[1]
            xmat_np[3] = np.eye(3).flatten()
            xmat_np[4] = np.array([[1, 0, 0], [0, -1, -1.22e-16], [0, 1.22e-16, -1]]).flatten()
            centers_np = np.zeros((NGEOMS, 3), dtype=np.float32)
            sizes_np = np.array([
                [4, 4, 0.2], [0.02, 0.02, 2.02], [0.02, 0.02, 2.02],
                [0.2, 0.15, 0.1], [0.045, 0.045, 0.545],
            ], dtype=np.float32)
            rbounds_np = np.array([0.0, 2.02, 2.02, 0.27, 0.545], dtype=np.float32)
            margins_np = np.zeros(NGEOMS, dtype=np.float32)
            pairs_np = np.array([[0, 3], [0, 4], [1, 3], [1, 4],
                                 [2, 3], [2, 4]], dtype=np.int32)
            N_PAIRS = 6

            results = {}
            for dev in ('cpu', 'metal:0'):
                a_pairs = wp.zeros(N_PAIRS, dtype=wp.vec2i, device=dev); a_pairs.assign(pairs_np)
                a_c = wp.zeros((1, NGEOMS), dtype=wp.vec3, device=dev); a_c.assign(centers_np.reshape(1, NGEOMS, 3))
                a_s = wp.zeros((1, NGEOMS), dtype=wp.vec3, device=dev); a_s.assign(sizes_np.reshape(1, NGEOMS, 3))
                a_rb = wp.zeros(NGEOMS, dtype=float, device=dev); a_rb.assign(rbounds_np)
                a_m = wp.zeros(NGEOMS, dtype=float, device=dev); a_m.assign(margins_np)
                a_xp = wp.zeros((1, NGEOMS), dtype=wp.vec3, device=dev); a_xp.assign(xpos_np.reshape(1, NGEOMS, 3))
                a_xm = wp.zeros((1, NGEOMS), dtype=wp.mat33, device=dev); a_xm.assign(xmat_np.reshape(1, NGEOMS, 9))
                out = wp.zeros(N_PAIRS, dtype=int, device=dev)
                wp.launch(kernel, dim=(1, N_PAIRS),
                          inputs=[a_pairs, a_c, a_s, a_rb, a_m, a_xp, a_xm],
                          outputs=[out], device=dev)
                results[dev] = out.numpy()

            # Pairs: floor-cart (PLANE rejects), floor-pole (PLANE rejects),
            # rail1-cart (passes), rail1-pole (OBB rejects),
            # rail2-cart (passes), rail2-pole (OBB rejects).
            expected = np.array([0, 0, 1, 0, 1, 0], dtype=np.int32)
            np.testing.assert_array_equal(results['cpu'], expected,
                err_msg='cpu got wrong broadphase result')
            np.testing.assert_array_equal(results['cpu'], results['metal:0'],
                err_msg='broadphase clone cpu vs metal mismatch')
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_printf_and_dead_tuples_codegen(self):
        # ``wp.printf`` calls and ``wp.matrix(..., shape=(R, C))`` sugar
        # emit ``wp::str`` constants and ``wp::tuple_t`` locals in the IR
        # that are never read by the kernel body. The preprocess pass
        # drops the printf lines and tuple constructions so the
        # unsupported-ctype guard never fires. Pure codegen-passes test:
        # we just need the kernels to compile and run cleanly.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(out: wp.array(dtype=wp.int32),
                  flag: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                if flag[tid] == 0:
                    wp.printf('warning: thread %u tripped\\n', tid)
                out[tid] = tid * 2

            N = 8
            results = {}
            for dev in ('cpu', 'metal:0'):
                flag = wp.zeros(N, dtype=wp.int32, device=dev)
                out = wp.zeros(N, dtype=wp.int32, device=dev)
                wp.launch(k, dim=N, inputs=[flag], outputs=[out], device=dev)
                results[dev] = out.numpy()
            np.testing.assert_array_equal(results['cpu'], results['metal:0'])
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_diag_vec3_to_mat3x3_matches_cpu(self):
        # ``wp.diag(vec3)`` builds a 3x3 diagonal matrix. We route this
        # through a ``wp_diag_float3`` helper emitted in the kernel header.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(v: wp.array(dtype=wp.vec3),
                  m: wp.array(dtype=wp.mat33),
                  out: wp.array(dtype=wp.mat33)):
                tid = wp.tid()
                # m @ diag(v) @ transpose(m) — typical body-inertia transform
                out[tid] = m[tid] @ wp.diag(v[tid]) @ wp.transpose(m[tid])

            N = 8
            rng = np.random.default_rng(0)
            vn = rng.standard_normal((N, 3)).astype(np.float32)
            mn = rng.standard_normal((N, 3, 3)).astype(np.float32)
            results = {}
            for dev in ('cpu', 'metal:0'):
                v = wp.array(vn, dtype=wp.vec3, device=dev)
                m = wp.array(mn, dtype=wp.mat33, device=dev)
                out = wp.zeros(N, dtype=wp.mat33, device=dev)
                wp.launch(k, dim=N, inputs=[v, m], outputs=[out], device=dev)
                results[dev] = out.numpy()
            np.testing.assert_allclose(results['cpu'], results['metal:0'], rtol=1e-5)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_indexref_scalar_write_matches_cpu(self):
        # ``out[i, j][k] = val`` lowers to ``address + indexref + store``.
        # Our preprocessor folds this into a synthetic scalar-store token
        # so the underlying array is correctly classified as an output and
        # the write becomes a direct flat-offset subscript assignment.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(out: wp.array2d(dtype=wp.spatial_vector)):
                worldid, k = wp.tid()
                out[worldid, 0][k] = float(worldid * 10 + k)

            N = 4
            results = {}
            for dev in ('cpu', 'metal:0'):
                out = wp.zeros((N, 1), dtype=wp.spatial_vector, device=dev)
                wp.launch(k, dim=(N, 6), outputs=[out], device=dev)
                results[dev] = out.numpy()
            np.testing.assert_array_equal(results['cpu'], results['metal:0'])
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_vec_element_inplace_matches_cpu(self):
        # ``vec[i] = val``, ``vec[i] *= val``, etc. lower to
        # ``wp::*_inplace(vec, idx, val)`` (3-arg form), distinct from the
        # 2-arg ``wp::store(field_ptr, val)``. We translate the 3-arg form
        # to MSL's native ``vec[idx] op= val``. Exercises both native vec3
        # and the custom ``wp_vec6_float`` (spatial_vector) struct.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k_vec3(out: wp.array(dtype=wp.vec3)):
                tid = wp.tid()
                v = wp.vec3(1.0, 2.0, 3.0)
                v[0] = float(tid)
                v[1] *= 2.0
                v[2] += 10.0
                out[tid] = v

            @wp.kernel
            def k_spatial(out: wp.array(dtype=wp.spatial_vector)):
                tid = wp.tid()
                v = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                v[0] = float(tid)
                v[3] = float(tid) + 100.0
                v[5] *= 2.0
                out[tid] = v

            N = 8
            for kernel, dtype in ((k_vec3, wp.vec3), (k_spatial, wp.spatial_vector)):
                results = {}
                for dev in ('cpu', 'metal:0'):
                    out = wp.zeros(N, dtype=dtype, device=dev)
                    wp.launch(kernel, dim=N, outputs=[out], device=dev)
                    results[dev] = out.numpy()
                np.testing.assert_array_equal(results['cpu'], results['metal:0'])
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_user_function_inlining_void_matches_cpu(self):
        # ``@wp.func`` helpers that take an output array and write through
        # it (the canonical mujoco_warp ``_write_scalar`` pattern) get
        # spliced into the call site at codegen time. Without inlining,
        # the kernel would be rejected as having no output array.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.func
            def _scale_and_clip(scale: wp.float32, clip: wp.float32, val: wp.float32,
                                out: wp.array(dtype=wp.float32), idx: wp.int32):
                v = val * scale
                if v > clip:
                    out[idx] = clip
                    return
                if v < -clip:
                    out[idx] = -clip
                    return
                out[idx] = v

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32),
                  scale: wp.float32, clip: wp.float32,
                  out: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                _scale_and_clip(scale, clip, a[tid], out, tid)

            N = 16
            rng = np.random.default_rng(0)
            an = rng.standard_normal(N).astype(np.float32) * 5.0
            for dev in ('cpu', 'metal:0'):
                a = wp.array(an, dtype=wp.float32, device=dev)
                out = wp.zeros(N, dtype=wp.float32, device=dev)
                wp.launch(k, dim=N, inputs=[a, 2.0, 3.0], outputs=[out], device=dev)
                if dev == 'cpu':
                    cpu_out = out.numpy()
                else:
                    np.testing.assert_array_equal(cpu_out, out.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_user_function_inlining_value_return_matches_cpu(self):
        # ``@wp.func`` helpers that return tuple values (lowered by Warp
        # to ``ret_<i>`` writes plus extra output args at the call site)
        # also inline correctly.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.func
            def _normalize_with_norm(x: wp.vec3):
                n = wp.length(x)
                if n == 0.0:
                    return x, float(0.0)
                return x / n, n

            @wp.kernel
            def k(v: wp.array(dtype=wp.vec3),
                  out_dir: wp.array(dtype=wp.vec3),
                  out_len: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                d, n = _normalize_with_norm(v[tid])
                out_dir[tid] = d
                out_len[tid] = n

            N = 8
            rng = np.random.default_rng(0)
            vn = rng.standard_normal((N, 3)).astype(np.float32)
            # Throw in one zero vector to exercise the early-return branch.
            vn[3] = (0.0, 0.0, 0.0)
            for dev in ('cpu', 'metal:0'):
                v = wp.array(vn, dtype=wp.vec3, device=dev)
                out_dir = wp.zeros(N, dtype=wp.vec3, device=dev)
                out_len = wp.zeros(N, dtype=wp.float32, device=dev)
                wp.launch(k, dim=N, inputs=[v], outputs=[out_dir, out_len], device=dev)
                if dev == 'cpu':
                    cpu_dir = out_dir.numpy()
                    cpu_len = out_len.numpy()
                else:
                    np.testing.assert_allclose(cpu_dir, out_dir.numpy(), rtol=1e-5)
                    np.testing.assert_allclose(cpu_len, out_len.numpy(), rtol=1e-5)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_dynamic_range_two_arg_matches_cpu(self):
        # ``range(start, stop)`` with non-constant bounds lowers to
        # ``wp::range(var_start, var_stop)``. Our for-loop preprocessor
        # rewrites this to ``for (int i = start; i < stop; ++i)``.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32),
                  starts: wp.array(dtype=wp.int32),
                  stops: wp.array(dtype=wp.int32),
                  out: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                s = float(0.0)
                for i in range(starts[tid], stops[tid]):
                    s += a[i]
                out[tid] = s

            N, M = 32, 64
            rng = np.random.default_rng(0)
            an = rng.standard_normal(M).astype(np.float32)
            starts_n = rng.integers(0, M // 2, size=N, dtype=np.int32)
            stops_n = (starts_n + rng.integers(0, M // 2, size=N, dtype=np.int32) + 1)
            stops_n = np.minimum(stops_n, M).astype(np.int32)
            results = {}
            for dev in ('cpu', 'metal:0'):
                a = wp.array(an, dtype=wp.float32, device=dev)
                starts = wp.array(starts_n, dtype=wp.int32, device=dev)
                stops = wp.array(stops_n, dtype=wp.int32, device=dev)
                out = wp.zeros(N, dtype=wp.float32, device=dev)
                wp.launch(k, dim=N, inputs=[a, starts, stops],
                          outputs=[out], device=dev)
                results[dev] = out.numpy()
            np.testing.assert_allclose(results['cpu'], results['metal:0'], rtol=1e-5)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_slice_view_array_store_matches_cpu(self):
        # Regular ``out2d[i][j] = val`` store path through a view.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array2d(dtype=wp.float32),
                  out: wp.array2d(dtype=wp.float32)):
                tid = wp.tid()
                row_in = a[tid]
                row_out = out[tid]
                for j in range(a.shape[1]):
                    row_out[j] = row_in[j] * 3.0 + 1.0

            N, M = 8, 5
            rng = np.random.default_rng(0)
            an = rng.standard_normal((N, M)).astype(np.float32)
            results = {}
            for dev in ('cpu', 'metal:0'):
                a = wp.array(an, dtype=wp.float32, device=dev)
                out = wp.zeros((N, M), dtype=wp.float32, device=dev)
                wp.launch(k, dim=N, inputs=[a], outputs=[out], device=dev)
                results[dev] = out.numpy()
            # Bit-equal on the MLX dispatch path; native dispatch may
            # differ at the last bit of float32 because Apple's MSL
            # compiler picks a slightly different FMA fusion under the
            # raw ``newLibraryWithSource`` options we use. The arithmetic
            # is correct to float32 precision either way.
            np.testing.assert_allclose(results['cpu'], results['metal:0'], rtol=1e-6, atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_while_loop_matches_cpu(self):
        # Triangular sum: ``s = sum(1..n)``. Exercises the structural
        # ``while``-as-``while(true){}`` rewrite plus mid-body mutation via
        # ``wp::assign``.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                n = a[tid]
                s = int(0)
                while n > 0:
                    s = s + n
                    n = n - 1
                out[tid] = s

            N = 32
            rng = np.random.default_rng(0)
            an = rng.integers(0, 50, size=N, dtype=np.int32)
            out_cpu = wp.zeros(N, dtype=wp.int32, device='cpu')
            out_m = wp.zeros(N, dtype=wp.int32, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.int32, device='cpu')],
                      outputs=[out_cpu], device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.int32, device='metal:0')],
                      outputs=[out_m], device='metal:0')
            np.testing.assert_array_equal(out_cpu.numpy(), out_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_while_loop_with_break_matches_cpu(self):
        # Linear search returning the first index whose value crosses a
        # threshold. ``break`` lowers to ``goto end_while_K`` in Warp's IR;
        # our preprocessor must rewrite that as ``break;`` to be valid MSL.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32), n: wp.int32,
                  out: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                i = int(0)
                while i < n:
                    if a[tid * n + i] > 0.5:
                        break
                    i = i + 1
                out[tid] = i

            N = 64
            M = 50
            rng = np.random.default_rng(99)
            an = rng.uniform(0.0, 1.0, size=N * M).astype(np.float32)
            out_cpu = wp.zeros(N, dtype=wp.int32, device='cpu')
            out_m = wp.zeros(N, dtype=wp.int32, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='cpu'), M],
                      outputs=[out_cpu], device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='metal:0'), M],
                      outputs=[out_m], device='metal:0')
            np.testing.assert_array_equal(out_cpu.numpy(), out_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_2d_array_copy_matches_cpu(self):
        # Simplest 2D test: copy ``src[i, j]`` into ``dst[i, j]``. Exercises
        # ``builtin_tid2d``, multi-arg ``wp::address`` / ``wp::array_store``,
        # and the synthetic ``<output>_shape`` input mechanism.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(src: wp.array2d(dtype=wp.float32),
                  dst: wp.array2d(dtype=wp.float32)):
                i, j = wp.tid()
                dst[i, j] = src[i, j]

            for H, W in ((1, 1), (3, 5), (12, 17), (33, 65)):
                rng = np.random.default_rng(H * 1000 + W)
                src_np = rng.standard_normal((H, W)).astype(np.float32)
                dst_cpu = wp.zeros((H, W), dtype=wp.float32, device='cpu')
                dst_m = wp.zeros((H, W), dtype=wp.float32, device='metal:0')
                wp.launch(k, dim=(H, W),
                          inputs=[wp.array(src_np, dtype=wp.float32, device='cpu')],
                          outputs=[dst_cpu], device='cpu')
                wp.launch(k, dim=(H, W),
                          inputs=[wp.array(src_np, dtype=wp.float32, device='metal:0')],
                          outputs=[dst_m], device='metal:0')
                np.testing.assert_array_equal(
                    dst_cpu.numpy(), dst_m.numpy(),
                    err_msg=f'2D copy {H}x{W} mismatch',
                )
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_2d_mixed_with_1d_input_matches_cpu(self):
        # Per-row scaling — a 2D output written using a 1D scale array.
        # Exercises mixed 1-D and 2-D array indexing in the same kernel.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array2d(dtype=wp.float32),
                  scale: wp.array(dtype=wp.float32),
                  out: wp.array2d(dtype=wp.float32)):
                i, j = wp.tid()
                out[i, j] = a[i, j] * scale[i]

            H, W = 16, 32
            rng = np.random.default_rng(7)
            an = rng.standard_normal((H, W)).astype(np.float32)
            sn = rng.standard_normal(H).astype(np.float32)
            out_cpu = wp.zeros((H, W), dtype=wp.float32, device='cpu')
            out_m = wp.zeros((H, W), dtype=wp.float32, device='metal:0')
            wp.launch(k, dim=(H, W),
                      inputs=[wp.array(an, dtype=wp.float32, device='cpu'),
                              wp.array(sn, dtype=wp.float32, device='cpu')],
                      outputs=[out_cpu], device='cpu')
            wp.launch(k, dim=(H, W),
                      inputs=[wp.array(an, dtype=wp.float32, device='metal:0'),
                              wp.array(sn, dtype=wp.float32, device='metal:0')],
                      outputs=[out_m], device='metal:0')
            np.testing.assert_array_equal(out_cpu.numpy(), out_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_3d_array_add_matches_cpu(self):
        # Elementwise add on 3-D arrays. Exercises ``builtin_tid3d`` and the
        # 3-term flat-index expression.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array3d(dtype=wp.float32),
                  b: wp.array3d(dtype=wp.float32),
                  c: wp.array3d(dtype=wp.float32)):
                i, j, k = wp.tid()
                c[i, j, k] = a[i, j, k] + b[i, j, k]

            D, H, W = 5, 7, 11
            rng = np.random.default_rng(2026)
            an = rng.standard_normal((D, H, W)).astype(np.float32)
            bn = rng.standard_normal((D, H, W)).astype(np.float32)
            c_cpu = wp.zeros((D, H, W), dtype=wp.float32, device='cpu')
            c_m = wp.zeros((D, H, W), dtype=wp.float32, device='metal:0')
            wp.launch(k, dim=(D, H, W),
                      inputs=[wp.array(an, dtype=wp.float32, device='cpu'),
                              wp.array(bn, dtype=wp.float32, device='cpu')],
                      outputs=[c_cpu], device='cpu')
            wp.launch(k, dim=(D, H, W),
                      inputs=[wp.array(an, dtype=wp.float32, device='metal:0'),
                              wp.array(bn, dtype=wp.float32, device='metal:0')],
                      outputs=[c_m], device='metal:0')
            np.testing.assert_array_equal(c_cpu.numpy(), c_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_2d_array_with_for_loop_matches_cpu(self):
        # Row-sum: each thread sums its row. Combines 2D indexing with a
        # dynamic-range for-loop, which mirrors the IR shape of several
        # mujoco_warp kernels (e.g. ``_extract_dof_A_diag``).
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array2d(dtype=wp.float32), n: wp.int32,
                  out: wp.array(dtype=wp.float32)):
                i = wp.tid()
                s = float(0.0)
                for j in range(n):
                    s = s + a[i, j]
                out[i] = s

            H, W = 32, 9
            rng = np.random.default_rng(11)
            an = rng.standard_normal((H, W)).astype(np.float32)
            out_cpu = wp.zeros(H, dtype=wp.float32, device='cpu')
            out_m = wp.zeros(H, dtype=wp.float32, device='metal:0')
            wp.launch(k, dim=H,
                      inputs=[wp.array(an, dtype=wp.float32, device='cpu'), W],
                      outputs=[out_cpu], device='cpu')
            wp.launch(k, dim=H,
                      inputs=[wp.array(an, dtype=wp.float32, device='metal:0'), W],
                      outputs=[out_m], device='metal:0')
            np.testing.assert_array_equal(out_cpu.numpy(), out_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_vec3_add_matches_cpu(self):
        # Element-wise vec3 add. Bit-exact because it's a single fp32 add per
        # component with no reordering.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.vec3),
                  b: wp.array(dtype=wp.vec3),
                  c: wp.array(dtype=wp.vec3)):
                tid = wp.tid()
                c[tid] = a[tid] + b[tid]

            N = 256
            rng = np.random.default_rng(0)
            an = rng.standard_normal((N, 3)).astype(np.float32)
            bn = rng.standard_normal((N, 3)).astype(np.float32)
            c_cpu = wp.zeros(N, dtype=wp.vec3, device='cpu')
            c_m = wp.zeros(N, dtype=wp.vec3, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.vec3, device='cpu'),
                              wp.array(bn, dtype=wp.vec3, device='cpu')],
                      outputs=[c_cpu], device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.vec3, device='metal:0'),
                              wp.array(bn, dtype=wp.vec3, device='metal:0')],
                      outputs=[c_m], device='metal:0')
            np.testing.assert_array_equal(c_cpu.numpy(), c_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_vec3_construct_and_index_match_cpu(self):
        # Round-trip via component construction and component access:
        # build a vec3 from three scalar arrays, then sum its components.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def construct(x: wp.array(dtype=wp.float32),
                          y: wp.array(dtype=wp.float32),
                          z: wp.array(dtype=wp.float32),
                          out: wp.array(dtype=wp.vec3)):
                tid = wp.tid()
                out[tid] = wp.vec3(x[tid], y[tid], z[tid])

            @wp.kernel
            def index_sum(a: wp.array(dtype=wp.vec3),
                          out: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                v = a[tid]
                out[tid] = v[0] + v[1] + v[2]

            N = 128
            rng = np.random.default_rng(7)
            xn = rng.standard_normal(N).astype(np.float32)
            yn = rng.standard_normal(N).astype(np.float32)
            zn = rng.standard_normal(N).astype(np.float32)
            for dev in ('cpu', 'metal:0'):
                v_arr = wp.zeros(N, dtype=wp.vec3, device=dev)
                wp.launch(construct, dim=N,
                          inputs=[wp.array(xn, dtype=wp.float32, device=dev),
                                  wp.array(yn, dtype=wp.float32, device=dev),
                                  wp.array(zn, dtype=wp.float32, device=dev)],
                          outputs=[v_arr], device=dev)
                s_arr = wp.zeros(N, dtype=wp.float32, device=dev)
                wp.launch(index_sum, dim=N, inputs=[v_arr], outputs=[s_arr], device=dev)
                if dev == 'cpu':
                    cpu_v = v_arr.numpy()
                    cpu_s = s_arr.numpy()
                else:
                    np.testing.assert_array_equal(v_arr.numpy(), cpu_v)
                    np.testing.assert_array_equal(s_arr.numpy(), cpu_s)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_vec3_dot_matches_cpu(self):
        # ``wp.dot`` -> ``metal::dot``. Float-summation order may differ
        # between Metal and CPU; allow modest fp tolerance.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.vec3),
                  b: wp.array(dtype=wp.vec3),
                  c: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                c[tid] = wp.dot(a[tid], b[tid])

            N = 256
            rng = np.random.default_rng(11)
            an = rng.standard_normal((N, 3)).astype(np.float32)
            bn = rng.standard_normal((N, 3)).astype(np.float32)
            c_cpu = wp.zeros(N, dtype=wp.float32, device='cpu')
            c_m = wp.zeros(N, dtype=wp.float32, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.vec3, device='cpu'),
                              wp.array(bn, dtype=wp.vec3, device='cpu')],
                      outputs=[c_cpu], device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.vec3, device='metal:0'),
                              wp.array(bn, dtype=wp.vec3, device='metal:0')],
                      outputs=[c_m], device='metal:0')
            np.testing.assert_allclose(c_cpu.numpy(), c_m.numpy(), rtol=1e-4, atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_vec3_cross_normalize_matches_cpu(self):
        # Composes ``wp.cross`` and ``wp.normalize`` — the latter uses
        # ``rsqrt``-style ops, so allow ulp-level Metal/CPU divergence.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.vec3),
                  b: wp.array(dtype=wp.vec3),
                  out: wp.array(dtype=wp.vec3)):
                tid = wp.tid()
                out[tid] = wp.normalize(wp.cross(a[tid], b[tid]))

            N = 256
            rng = np.random.default_rng(2026)
            an = rng.standard_normal((N, 3)).astype(np.float32)
            bn = rng.standard_normal((N, 3)).astype(np.float32)
            out_cpu = wp.zeros(N, dtype=wp.vec3, device='cpu')
            out_m = wp.zeros(N, dtype=wp.vec3, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.vec3, device='cpu'),
                              wp.array(bn, dtype=wp.vec3, device='cpu')],
                      outputs=[out_cpu], device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.vec3, device='metal:0'),
                              wp.array(bn, dtype=wp.vec3, device='metal:0')],
                      outputs=[out_m], device='metal:0')
            np.testing.assert_allclose(out_cpu.numpy(), out_m.numpy(), rtol=1e-4, atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_vec2_and_vec4_match_cpu(self):
        # Sanity check that vec2 / vec4 also work via the same code path.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k2(a: wp.array(dtype=wp.vec2),
                   b: wp.array(dtype=wp.vec2),
                   c: wp.array(dtype=wp.vec2)):
                tid = wp.tid()
                c[tid] = a[tid] + b[tid]

            @wp.kernel
            def k4(a: wp.array(dtype=wp.vec4),
                   b: wp.array(dtype=wp.vec4),
                   c: wp.array(dtype=wp.vec4)):
                tid = wp.tid()
                c[tid] = a[tid] + b[tid]

            N = 128
            rng = np.random.default_rng(99)
            for kf, dim in ((k2, 2), (k4, 4)):
                vt = wp.vec2 if dim == 2 else wp.vec4
                an = rng.standard_normal((N, dim)).astype(np.float32)
                bn = rng.standard_normal((N, dim)).astype(np.float32)
                c_cpu = wp.zeros(N, dtype=vt, device='cpu')
                c_m = wp.zeros(N, dtype=vt, device='metal:0')
                wp.launch(kf, dim=N,
                          inputs=[wp.array(an, dtype=vt, device='cpu'),
                                  wp.array(bn, dtype=vt, device='cpu')],
                          outputs=[c_cpu], device='cpu')
                wp.launch(kf, dim=N,
                          inputs=[wp.array(an, dtype=vt, device='metal:0'),
                                  wp.array(bn, dtype=vt, device='metal:0')],
                          outputs=[c_m], device='metal:0')
                np.testing.assert_array_equal(c_cpu.numpy(), c_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_mat33_add_matches_cpu(self):
        # Element-wise mat33 add. Bit-exact (single fp32 add per component,
        # no reordering).
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.mat33),
                  b: wp.array(dtype=wp.mat33),
                  c: wp.array(dtype=wp.mat33)):
                tid = wp.tid()
                c[tid] = a[tid] + b[tid]

            N = 64
            rng = np.random.default_rng(0)
            an = rng.standard_normal((N, 3, 3)).astype(np.float32)
            bn = rng.standard_normal((N, 3, 3)).astype(np.float32)
            c_cpu = wp.zeros(N, dtype=wp.mat33, device='cpu')
            c_m = wp.zeros(N, dtype=wp.mat33, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.mat33, device='cpu'),
                              wp.array(bn, dtype=wp.mat33, device='cpu')],
                      outputs=[c_cpu], device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.mat33, device='metal:0'),
                              wp.array(bn, dtype=wp.mat33, device='metal:0')],
                      outputs=[c_m], device='metal:0')
            np.testing.assert_array_equal(c_cpu.numpy(), c_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_tile_sort_matches_cpu(self):
        # ``wp.tile_sort(keys, values)`` cooperatively sorts both tiles
        # in ascending key order, in-place. mujoco_warp's broadphase
        # ``segmented_sort`` and contact-sensor sort both use this. Our
        # default ``block_dim=1`` reduces the cooperative version to a
        # single-thread insertion sort.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(keys_in: wp.array(dtype=float),
                  vals_in: wp.array(dtype=int),
                  keys_out: wp.array(dtype=float),
                  vals_out: wp.array(dtype=int)):
                keys = wp.tile_load(keys_in, shape=(16,), storage='shared')
                vals = wp.tile_load(vals_in, shape=(16,), storage='shared')
                wp.tile_sort(keys, vals)
                wp.tile_store(keys_out, keys)
                wp.tile_store(vals_out, vals)

            rng = np.random.default_rng(3)
            keys_np = rng.uniform(-10, 10, size=16).astype(np.float32)
            vals_np = np.arange(16, dtype=np.int32)
            order = np.argsort(keys_np, kind='stable')
            expected_keys = keys_np[order]
            expected_vals = vals_np[order]
            for dev in ('cpu', 'metal:0'):
                ki = wp.array(keys_np, dtype=wp.float32, device=dev)
                vi = wp.array(vals_np, dtype=wp.int32, device=dev)
                ko = wp.zeros(16, dtype=wp.float32, device=dev)
                vo = wp.zeros(16, dtype=wp.int32, device=dev)
                wp.launch_tiled(k, dim=1, inputs=[ki, vi], outputs=[ko, vo],
                                block_dim=1, device=dev)
                np.testing.assert_array_equal(ko.numpy(), expected_keys,
                    err_msg=f'keys mismatch on {dev}')
                np.testing.assert_array_equal(vo.numpy(), expected_vals,
                    err_msg=f'values mismatch on {dev}')
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_generic_kernel_dtype_any_matches_cpu(self):
        # Kernels declared with ``dtype=Any`` are specialised at launch
        # time. The CUDA/CPU launch path resolves the overload via
        # ``infer_argument_types`` + ``add_overload``; the Metal short-
        # circuit needs to mirror that or the IR carries through with
        # ``wp::Any`` ctypes that no codegen path can handle.
        # mjlab's ``repeat_array_kernel`` (per-world tiling for domain
        # randomization) hit this.
        snippet = textwrap.dedent(
            """
            from typing import Any
            import warp as wp
            import numpy as np

            @wp.kernel(module='unique')
            def repeat_kernel(
                src: wp.array(dtype=Any),
                nelems_per_world: int,
                dst: wp.array(dtype=Any),
            ):
                tid = wp.tid()
                src[0]
                src_idx = tid % nelems_per_world
                dst[tid] = src[src_idx]

            N_PER_WORLD = 4
            NWORLD = 3
            src_np = np.arange(N_PER_WORLD, dtype=np.float32)
            expected = np.tile(src_np, NWORLD)
            for dev in ('cpu', 'metal:0'):
                src = wp.array(src_np, dtype=wp.float32, device=dev)
                dst = wp.zeros(N_PER_WORLD * NWORLD, dtype=wp.float32, device=dev)
                wp.launch(repeat_kernel, dim=dst.shape[0],
                          inputs=[src, N_PER_WORLD], outputs=[dst], device=dev)
                np.testing.assert_array_equal(dst.numpy(), expected)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_mat33_extract_2d_index_matches_cpu(self):
        # ``m[i, j]`` -> 3-arg ``wp::extract`` -> MSL ``m[j][i]`` (column-then-
        # row). Computes the trace, which uses three diagonal extracts.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                m = a[tid]
                out[tid] = m[0, 0] + m[1, 1] + m[2, 2]

            N = 32
            rng = np.random.default_rng(11)
            an = rng.standard_normal((N, 3, 3)).astype(np.float32)
            out_cpu = wp.zeros(N, dtype=wp.float32, device='cpu')
            out_m = wp.zeros(N, dtype=wp.float32, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.mat33, device='cpu')],
                      outputs=[out_cpu], device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.mat33, device='metal:0')],
                      outputs=[out_m], device='metal:0')
            np.testing.assert_array_equal(out_cpu.numpy(), out_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_mat33_row_col_slice_matches_cpu(self):
        # ``m[:, c]`` and ``m[r, :]`` lower to ``wp::extract<N>(m, slice, c)``
        # / ``wp::extract<N>(m, r, slice)`` over a ``wp::slice_t``. Used by
        # mujoco_warp's BVH bounds kernels (e.g. ``rot[:, 2]`` to extract a
        # mat33's third column when computing height-field axis-aligned
        # bounds).
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k_col(a: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.vec3)):
                tid = wp.tid()
                out[tid] = a[tid][:, 2]

            @wp.kernel
            def k_row(a: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.vec3)):
                tid = wp.tid()
                out[tid] = a[tid][1, :]

            N = 8
            rng = np.random.default_rng(7)
            an = rng.standard_normal((N, 3, 3)).astype(np.float32)
            for kk, expected_axis in ((k_col, 'col2'), (k_row, 'row1')):
                a_cpu = wp.array(an, dtype=wp.mat33, device='cpu')
                a_m = wp.array(an, dtype=wp.mat33, device='metal:0')
                o_cpu = wp.zeros(N, dtype=wp.vec3, device='cpu')
                o_m = wp.zeros(N, dtype=wp.vec3, device='metal:0')
                wp.launch(kk, dim=N, inputs=[a_cpu], outputs=[o_cpu], device='cpu')
                wp.launch(kk, dim=N, inputs=[a_m], outputs=[o_m], device='metal:0')
                np.testing.assert_array_equal(o_cpu.numpy(), o_m.numpy(),
                    err_msg=f'{expected_axis} mismatch')
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_mat33_transpose_matches_cpu(self):
        # ``wp.transpose`` -> ``metal::transpose``. Verifies the row/col
        # convention end-to-end: store a row-major matrix, transpose it on
        # the GPU, copy back to row-major storage; the result should match
        # ``np.transpose`` of the input array.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.mat33)):
                tid = wp.tid()
                out[tid] = wp.transpose(a[tid])

            N = 32
            rng = np.random.default_rng(7)
            an = rng.standard_normal((N, 3, 3)).astype(np.float32)
            out_cpu = wp.zeros(N, dtype=wp.mat33, device='cpu')
            out_m = wp.zeros(N, dtype=wp.mat33, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.mat33, device='cpu')],
                      outputs=[out_cpu], device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.mat33, device='metal:0')],
                      outputs=[out_m], device='metal:0')
            np.testing.assert_array_equal(out_cpu.numpy(), out_m.numpy())
            # Sanity: the result really is the transpose of the input.
            np.testing.assert_array_equal(out_cpu.numpy(), np.transpose(an, (0, 2, 1)))
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_mat33_vec3_mul_matches_cpu(self):
        # M*v: depends on the row-major-storage / column-major-MSL convention
        # being correct. Float-summation-order tolerance because Metal/CPU
        # may schedule the dot products differently.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(m: wp.array(dtype=wp.mat33),
                  v: wp.array(dtype=wp.vec3),
                  out: wp.array(dtype=wp.vec3)):
                tid = wp.tid()
                out[tid] = m[tid] * v[tid]

            N = 32
            rng = np.random.default_rng(99)
            mn = rng.standard_normal((N, 3, 3)).astype(np.float32)
            vn = rng.standard_normal((N, 3)).astype(np.float32)
            out_cpu = wp.zeros(N, dtype=wp.vec3, device='cpu')
            out_m = wp.zeros(N, dtype=wp.vec3, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(mn, dtype=wp.mat33, device='cpu'),
                              wp.array(vn, dtype=wp.vec3, device='cpu')],
                      outputs=[out_cpu], device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(mn, dtype=wp.mat33, device='metal:0'),
                              wp.array(vn, dtype=wp.vec3, device='metal:0')],
                      outputs=[out_m], device='metal:0')
            np.testing.assert_allclose(out_cpu.numpy(), out_m.numpy(), rtol=1e-4, atol=1e-6)
            # Sanity: result equals NumPy ``M @ v`` row-by-row.
            ref = np.einsum('nij,nj->ni', mn, vn)
            np.testing.assert_allclose(out_cpu.numpy(), ref, rtol=1e-4, atol=1e-5)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_mat33_identity_constructor_matches_cpu(self):
        # Constructor with row-major flat args has to be reordered into
        # column form for MSL to read correctly.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(out: wp.array(dtype=wp.mat33)):
                tid = wp.tid()
                out[tid] = wp.mat33(1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0)

            N = 16
            out_cpu = wp.zeros(N, dtype=wp.mat33, device='cpu')
            out_m = wp.zeros(N, dtype=wp.mat33, device='metal:0')
            wp.launch(k, dim=N, outputs=[out_cpu], device='cpu')
            wp.launch(k, dim=N, outputs=[out_m], device='metal:0')
            np.testing.assert_array_equal(out_cpu.numpy(), out_m.numpy())
            np.testing.assert_array_equal(out_m.numpy()[0], np.eye(3, dtype=np.float32))
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_array_shape_access_in_body_matches_cpu(self):
        # ``arr.shape[k]`` lowers to a goto-free chain of intermediates with
        # ``wp::shape_t*``/``wp::shape_t`` ctypes the type table doesn't know.
        # The codegen aliases those locals to MLX's auto-generated
        # ``<arg>_shape`` (or the synthetic shape input we add for outputs).
        # mujoco_warp uses this pattern heavily for broadcasting / wrapping.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(src: wp.array2d(dtype=wp.float32),
                  dst: wp.array2d(dtype=wp.float32)):
                worldid, i = wp.tid()
                src_row = worldid % src.shape[0]
                dst[worldid, i] = src[src_row, i]

            nworld, nq = 8, 12
            src_n = 4  # broadcast: src has 4 rows, each repeated twice
            rng = np.random.default_rng(0)
            src_np = rng.standard_normal((src_n, nq)).astype(np.float32)

            for dev in ('cpu', 'metal:0'):
                src_arr = wp.array(src_np, dtype=wp.float32, device=dev)
                dst_arr = wp.zeros((nworld, nq), dtype=wp.float32, device=dev)
                wp.launch(k, dim=(nworld, nq),
                          inputs=[src_arr], outputs=[dst_arr], device=dev)
                if dev == 'cpu':
                    cpu_out = dst_arr.numpy()
                else:
                    np.testing.assert_array_equal(dst_arr.numpy(), cpu_out)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_subscript_array_annotation_form_matches_cpu(self):
        # mujoco_warp uses the subscript annotation form
        # ``wp.array2d[float]`` (which produces ``_ArrayAnnotation``) rather
        # than the callable form ``wp.array2d(dtype=float)``. Both must work.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(src: wp.array2d[float], dst: wp.array2d[float]):
                i, j = wp.tid()
                dst[i, j] = src[i, j] * 2.0

            H, W = 8, 5
            rng = np.random.default_rng(7)
            an = rng.standard_normal((H, W)).astype(np.float32)
            for dev in ('cpu', 'metal:0'):
                src_arr = wp.array(an, dtype=wp.float32, device=dev)
                dst_arr = wp.zeros((H, W), dtype=wp.float32, device=dev)
                wp.launch(k, dim=(H, W),
                          inputs=[src_arr], outputs=[dst_arr], device=dev)
                if dev == 'cpu':
                    cpu_out = dst_arr.numpy()
                else:
                    np.testing.assert_array_equal(dst_arr.numpy(), cpu_out)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_where_matches_cpu(self):
        # ``wp.where(cond, a, b)`` -> C-style ternary in MSL. Bit-exact for
        # finite inputs (no fp ops, just selection).
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32),
                  b: wp.array(dtype=wp.float32),
                  c: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                c[tid] = wp.where(a[tid] > 0.0, a[tid], b[tid])

            N = 1024
            rng = np.random.default_rng(0)
            an = rng.standard_normal(N).astype(np.float32)
            bn = rng.standard_normal(N).astype(np.float32)
            c_cpu = wp.zeros(N, dtype=wp.float32, device='cpu')
            c_m = wp.zeros(N, dtype=wp.float32, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='cpu'),
                              wp.array(bn, dtype=wp.float32, device='cpu')],
                      outputs=[c_cpu], device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='metal:0'),
                              wp.array(bn, dtype=wp.float32, device='metal:0')],
                      outputs=[c_m], device='metal:0')
            np.testing.assert_array_equal(c_cpu.numpy(), c_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_clamp_matches_cpu(self):
        # ``wp.clamp(x, lo, hi)`` -> ``metal::clamp``. Same arg order. The
        # MSL stdlib clamp is bit-exact when ``lo <= x <= hi`` is preserved
        # element-wise (no sub-ulp rounding involved).
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32),
                  c: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                c[tid] = wp.clamp(a[tid], -1.0, 1.0)

            N = 1024
            rng = np.random.default_rng(11)
            an = (rng.standard_normal(N) * 5.0).astype(np.float32)
            c_cpu = wp.zeros(N, dtype=wp.float32, device='cpu')
            c_m = wp.zeros(N, dtype=wp.float32, device='metal:0')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='cpu')],
                      outputs=[c_cpu], device='cpu')
            wp.launch(k, dim=N,
                      inputs=[wp.array(an, dtype=wp.float32, device='metal:0')],
                      outputs=[c_m], device='metal:0')
            np.testing.assert_array_equal(c_cpu.numpy(), c_m.numpy())
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_2d_vec3_array_matches_cpu(self):
        # ``wp.array2d(dtype=wp.vec3)`` was previously rejected. Lifted: vec
        # arrays of any ndim work via the same per-component expansion plus
        # ``_flat_index_expr`` for the user-visible dims.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def add(a: wp.array2d(dtype=wp.vec3),
                    b: wp.array2d(dtype=wp.vec3),
                    c: wp.array2d(dtype=wp.vec3)):
                i, j = wp.tid()
                c[i, j] = a[i, j] + b[i, j]

            @wp.kernel
            def dot(a: wp.array2d(dtype=wp.vec3),
                    b: wp.array2d(dtype=wp.vec3),
                    c: wp.array2d(dtype=wp.float32)):
                i, j = wp.tid()
                c[i, j] = wp.dot(a[i, j], b[i, j])

            H, W = 5, 7
            rng = np.random.default_rng(0)
            an = rng.standard_normal((H, W, 3)).astype(np.float32)
            bn = rng.standard_normal((H, W, 3)).astype(np.float32)
            for dev in ('cpu', 'metal:0'):
                a = wp.array(an, dtype=wp.vec3, device=dev)
                b = wp.array(bn, dtype=wp.vec3, device=dev)
                c = wp.zeros((H, W), dtype=wp.vec3, device=dev)
                wp.launch(add, dim=(H, W), inputs=[a, b], outputs=[c], device=dev)
                if dev == 'cpu':
                    cpu_add = c.numpy()
                else:
                    np.testing.assert_array_equal(c.numpy(), cpu_add)
            for dev in ('cpu', 'metal:0'):
                a = wp.array(an, dtype=wp.vec3, device=dev)
                b = wp.array(bn, dtype=wp.vec3, device=dev)
                c = wp.zeros((H, W), dtype=wp.float32, device=dev)
                wp.launch(dot, dim=(H, W), inputs=[a, b], outputs=[c], device=dev)
                if dev == 'cpu':
                    cpu_dot = c.numpy()
                else:
                    np.testing.assert_allclose(c.numpy(), cpu_dot, rtol=1e-5, atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_2d_mat33_array_matches_cpu(self):
        # Same lift for mat33 — multi-dim arrays of mat now work.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array2d(dtype=wp.mat33),
                  out: wp.array2d(dtype=wp.mat33)):
                i, j = wp.tid()
                out[i, j] = wp.transpose(a[i, j])

            H, W = 4, 6
            rng = np.random.default_rng(11)
            mn = rng.standard_normal((H, W, 3, 3)).astype(np.float32)
            for dev in ('cpu', 'metal:0'):
                a = wp.array(mn, dtype=wp.mat33, device=dev)
                out = wp.zeros((H, W), dtype=wp.mat33, device=dev)
                wp.launch(k, dim=(H, W), inputs=[a], outputs=[out], device=dev)
                if dev == 'cpu':
                    cpu_out = out.numpy()
                else:
                    np.testing.assert_array_equal(out.numpy(), cpu_out)
            # Sanity: result is the per-cell matrix transpose of the input.
            np.testing.assert_array_equal(cpu_out, np.transpose(mn, (0, 1, 3, 2)))
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_struct_construct_store_and_read_matches_cpu(self):
        # Full round-trip on Metal: construct a Particle local, populate its
        # fields, store the local into a wp.array(dtype=Particle), then read
        # the fields back. Verifies both directions of the per-field
        # expansion (struct local + struct array field reads) work end-to-
        # end and produce bit-exact output vs the CPU codegen.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.struct
            class Particle:
                pos: wp.vec3
                mass: wp.float32

            @wp.kernel
            def fill(pos_in: wp.array(dtype=wp.vec3),
                     mass_in: wp.array(dtype=wp.float32),
                     p: wp.array(dtype=Particle)):
                tid = wp.tid()
                q = Particle()
                q.pos = pos_in[tid]
                q.mass = mass_in[tid]
                p[tid] = q

            @wp.kernel
            def read_mass(p: wp.array(dtype=Particle),
                          out: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                out[tid] = p[tid].mass

            @wp.kernel
            def read_pos(p: wp.array(dtype=Particle),
                         out: wp.array(dtype=wp.vec3)):
                tid = wp.tid()
                out[tid] = p[tid].pos

            N = 32
            rng = np.random.default_rng(0)
            pos_np = rng.standard_normal((N, 3)).astype(np.float32)
            mass_np = rng.standard_normal(N).astype(np.float32)

            for dev in ('cpu', 'metal:0'):
                pos_arr = wp.array(pos_np, dtype=wp.vec3, device=dev)
                mass_arr = wp.array(mass_np, dtype=wp.float32, device=dev)
                p_arr = wp.empty(N, dtype=Particle, device=dev)
                wp.launch(fill, dim=N, inputs=[pos_arr, mass_arr],
                          outputs=[p_arr], device=dev)

                m_arr = wp.zeros(N, dtype=wp.float32, device=dev)
                wp.launch(read_mass, dim=N, inputs=[p_arr], outputs=[m_arr], device=dev)
                np.testing.assert_array_equal(m_arr.numpy(), mass_np)

                p_arr_out = wp.zeros(N, dtype=wp.vec3, device=dev)
                wp.launch(read_pos, dim=N, inputs=[p_arr], outputs=[p_arr_out], device=dev)
                np.testing.assert_array_equal(p_arr_out.numpy(), pos_np)

                if dev == 'cpu':
                    cpu_struct_bytes = bytes(p_arr.numpy())
                else:
                    # Storage layout must match Warp's CPU codegen exactly so
                    # cross-device byte-copies of struct arrays are valid.
                    assert bytes(p_arr.numpy()) == cpu_struct_bytes, 'struct byte layout differs'
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_spatial_vector_round_trip_matches_cpu(self):
        # ``wp.spatial_vector`` is ``vec_t<6, float32>`` — sized beyond MSL's
        # native ``floatN`` (N <= 4). Codegen emits a custom ``wp_vec6_float``
        # struct in the kernel ``header`` parameter with ``+/-/*//``
        # operator overloads and ``wp_spatial_top`` / ``wp_spatial_bottom``
        # helpers. Verifies all three patterns (construct / add / top+bottom)
        # produce bit-exact CPU vs Metal output.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def construct(a: wp.array(dtype=wp.vec3),
                          b: wp.array(dtype=wp.vec3),
                          out: wp.array(dtype=wp.spatial_vector)):
                tid = wp.tid()
                out[tid] = wp.spatial_vector(a[tid][0], a[tid][1], a[tid][2],
                                             b[tid][0], b[tid][1], b[tid][2])

            @wp.kernel
            def add(a: wp.array(dtype=wp.spatial_vector),
                    b: wp.array(dtype=wp.spatial_vector),
                    c: wp.array(dtype=wp.spatial_vector)):
                tid = wp.tid()
                c[tid] = a[tid] + b[tid]

            @wp.kernel
            def top_bottom(s: wp.array(dtype=wp.spatial_vector),
                           top: wp.array(dtype=wp.vec3),
                           bottom: wp.array(dtype=wp.vec3)):
                tid = wp.tid()
                top[tid] = wp.spatial_top(s[tid])
                bottom[tid] = wp.spatial_bottom(s[tid])

            N = 32
            rng = np.random.default_rng(0)
            an = rng.standard_normal((N, 3)).astype(np.float32)
            bn = rng.standard_normal((N, 3)).astype(np.float32)
            sn2 = rng.standard_normal((N, 6)).astype(np.float32)

            for dev in ('cpu', 'metal:0'):
                a3 = wp.array(an, dtype=wp.vec3, device=dev)
                b3 = wp.array(bn, dtype=wp.vec3, device=dev)
                sv = wp.zeros(N, dtype=wp.spatial_vector, device=dev)
                wp.launch(construct, dim=N, inputs=[a3, b3], outputs=[sv], device=dev)
                if dev == 'cpu':
                    cpu_sv = sv.numpy()
                else:
                    np.testing.assert_array_equal(sv.numpy(), cpu_sv)

                sv2 = wp.array(sn2, dtype=wp.spatial_vector, device=dev)
                sum_arr = wp.zeros(N, dtype=wp.spatial_vector, device=dev)
                wp.launch(add, dim=N,
                          inputs=[wp.array(cpu_sv, dtype=wp.spatial_vector, device=dev), sv2],
                          outputs=[sum_arr], device=dev)
                if dev == 'cpu':
                    cpu_sum = sum_arr.numpy()
                else:
                    np.testing.assert_array_equal(sum_arr.numpy(), cpu_sum)

                top_arr = wp.zeros(N, dtype=wp.vec3, device=dev)
                bot_arr = wp.zeros(N, dtype=wp.vec3, device=dev)
                wp.launch(top_bottom, dim=N, inputs=[sv],
                          outputs=[top_arr, bot_arr], device=dev)
                if dev == 'cpu':
                    cpu_top = top_arr.numpy()
                    cpu_bot = bot_arr.numpy()
                else:
                    np.testing.assert_array_equal(top_arr.numpy(), cpu_top)
                    np.testing.assert_array_equal(bot_arr.numpy(), cpu_bot)
            # Sanity: the construct kernel's output equals concat of (an, bn).
            np.testing.assert_array_equal(cpu_sv,
                                          np.concatenate([an, bn], axis=1))
            # Sanity: top is the first 3 components, bottom is the last 3.
            np.testing.assert_array_equal(cpu_top, an)
            np.testing.assert_array_equal(cpu_bot, bn)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=60)

    def test_neg_and_spatial_two_arg_constructor_match_cpu(self):
        # Mirrors the real mujoco_warp kernel ``_cacc_world``: a ``spatial_
        # vector`` is constructed from a vec3 and the negation of another
        # vec3. Exercises three patterns at once: ``wp::neg`` on a vec, the
        # 2-arg ``spatial_vector(vec3, vec3)`` constructor, and a 2-D
        # ``wp.array2d[wp.spatial_vector]`` write.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(gravity: wp.array(dtype=wp.vec3),
                  out: wp.array2d(dtype=wp.spatial_vector)):
                worldid = wp.tid()
                out[worldid, 0] = wp.spatial_vector(
                    wp.vec3(0.0),
                    -gravity[worldid % gravity.shape[0]],
                )

            nworld, nbody = 4, 5
            rng = np.random.default_rng(0)
            gn = rng.standard_normal((nworld, 3)).astype(np.float32)

            for dev in ('cpu', 'metal:0'):
                g = wp.array(gn, dtype=wp.vec3, device=dev)
                ca = wp.zeros((nworld, nbody), dtype=wp.spatial_vector, device=dev)
                wp.launch(k, dim=nworld, inputs=[g], outputs=[ca], device=dev)
                if dev == 'cpu':
                    cpu_out = ca.numpy()
                else:
                    np.testing.assert_array_equal(ca.numpy(), cpu_out)
            # Sanity: the kernel zeros the upper vec3 and stores -gravity
            # into the lower vec3 of column 0; remaining columns stay zero.
            expected = np.zeros((nworld, nbody, 6), dtype=np.float32)
            expected[:, 0, 3:] = -gn
            np.testing.assert_array_equal(cpu_out, expected)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=60)

    def test_struct_typed_kernel_arg_matches_cpu(self):
        # ``def k(p: Particle, ...)`` was previously rejected. Lifted: the
        # launcher serialises the user's ``StructInstance`` to a flat
        # float32 buffer (via ``bytes(p._ctype)``) and passes it as an MLX
        # input; the kernel body reads each field at its scalar offset.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.struct
            class Particle:
                pos: wp.vec3
                mass: wp.float32

            @wp.kernel
            def k(p: Particle, out: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                out[tid] = p.mass + p.pos[0] + p.pos[1] + p.pos[2] + float(tid)

            N = 16
            p = Particle()
            p.pos = wp.vec3(1.0, 2.0, 3.0)
            p.mass = 10.0

            for dev in ('cpu', 'metal:0'):
                out = wp.zeros(N, dtype=wp.float32, device=dev)
                wp.launch(k, dim=N, inputs=[p], outputs=[out], device=dev)
                if dev == 'cpu':
                    cpu_out = out.numpy()
                else:
                    np.testing.assert_array_equal(out.numpy(), cpu_out)
            # Sanity: each output element is mass + pos.x + pos.y + pos.z + tid.
            expected = np.array(
                [10.0 + 1.0 + 2.0 + 3.0 + i for i in range(N)],
                dtype=np.float32,
            )
            np.testing.assert_array_equal(cpu_out, expected)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_unot_floordiv_bit_and_length_sq_match_cpu(self):
        # Trivial intrinsics that surfaced as gaps in the mujoco_warp recon.
        # Each is a one-line regex add; this test bundles them.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k_unot(a: wp.array(dtype=wp.bool), out: wp.array(dtype=wp.bool)):
                tid = wp.tid()
                out[tid] = not a[tid]

            @wp.kernel
            def k_floordiv(a: wp.array(dtype=wp.int32),
                           b: wp.array(dtype=wp.int32),
                           out: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                out[tid] = a[tid] // b[tid]

            @wp.kernel
            def k_bit_and(a: wp.array(dtype=wp.int32),
                          b: wp.array(dtype=wp.int32),
                          out: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                out[tid] = a[tid] & b[tid]

            @wp.kernel
            def k_length_sq(v: wp.array(dtype=wp.vec3),
                            out: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                out[tid] = wp.length_sq(v[tid])

            N = 32
            rng = np.random.default_rng(0)
            # Generate ALL inputs up front so CPU and Metal see identical data.
            unot_in = rng.integers(0, 2, N).astype(bool)
            fd_a = rng.integers(1, 100, N, dtype=np.int32)
            fd_b = rng.integers(1, 5, N, dtype=np.int32)
            band_a = rng.integers(0, 0xFFFF, N, dtype=np.int32)
            band_b = rng.integers(0, 0xFFFF, N, dtype=np.int32)
            vn = rng.standard_normal((N, 3)).astype(np.float32)

            for dev in ('cpu', 'metal:0'):
                a = wp.array(unot_in, dtype=wp.bool, device=dev)
                out = wp.zeros(N, dtype=wp.bool, device=dev)
                wp.launch(k_unot, dim=N, inputs=[a], outputs=[out], device=dev)
                if dev == 'cpu':
                    cpu_unot = out.numpy()
                else:
                    np.testing.assert_array_equal(out.numpy(), cpu_unot)

            for dev in ('cpu', 'metal:0'):
                a = wp.array(fd_a, dtype=wp.int32, device=dev)
                b = wp.array(fd_b, dtype=wp.int32, device=dev)
                out = wp.zeros(N, dtype=wp.int32, device=dev)
                wp.launch(k_floordiv, dim=N, inputs=[a, b], outputs=[out], device=dev)
                if dev == 'cpu':
                    cpu_fd = out.numpy()
                else:
                    np.testing.assert_array_equal(out.numpy(), cpu_fd)

            for dev in ('cpu', 'metal:0'):
                a = wp.array(band_a, dtype=wp.int32, device=dev)
                b = wp.array(band_b, dtype=wp.int32, device=dev)
                out = wp.zeros(N, dtype=wp.int32, device=dev)
                wp.launch(k_bit_and, dim=N, inputs=[a, b], outputs=[out], device=dev)
                if dev == 'cpu':
                    cpu_band = out.numpy()
                else:
                    np.testing.assert_array_equal(out.numpy(), cpu_band)

            for dev in ('cpu', 'metal:0'):
                v = wp.array(vn, dtype=wp.vec3, device=dev)
                out = wp.zeros(N, dtype=wp.float32, device=dev)
                wp.launch(k_length_sq, dim=N, inputs=[v], outputs=[out], device=dev)
                if dev == 'cpu':
                    cpu_l2 = out.numpy()
                else:
                    np.testing.assert_allclose(out.numpy(), cpu_l2, rtol=1e-5)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_quat_array_round_trip_matches_cpu(self):
        # ``wp.quat`` is laid out as a 4-component vec_t; the codegen
        # normalises ``wp::quat_t<wp::T>`` to ``wp::vec_t<4, wp::T>`` so
        # quat constructor / extract / array reads all reuse the vec4 path.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(out: wp.array(dtype=wp.quat)):
                tid = wp.tid()
                out[tid] = wp.quat(1.0, 2.0, 3.0, float(tid))

            N = 16
            for dev in ('cpu', 'metal:0'):
                out = wp.zeros(N, dtype=wp.quat, device=dev)
                wp.launch(k, dim=N, outputs=[out], device=dev)
                if dev == 'cpu':
                    cpu_out = out.numpy()
                else:
                    np.testing.assert_array_equal(out.numpy(), cpu_out)
            # Sanity: the float-view layout is (1.0, 2.0, 3.0, tid) per element.
            expected = np.zeros((N, 4), dtype=np.float32)
            expected[:, 0] = 1.0
            expected[:, 1] = 2.0
            expected[:, 2] = 3.0
            expected[:, 3] = np.arange(N, dtype=np.float32)
            np.testing.assert_array_equal(cpu_out, expected)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_early_return_seeds_mat_output_matches_cpu(self):
        # Regression for the cartpole bug: a kernel with a top-level
        # ``return;`` and a ``wp.mat33``-typed output. The seed prologue
        # used to compute its per-world stride as
        # ``shape[1] * shape[2] * shape[3]`` (treating mat as two extra
        # dims), but MLX's view shape collapses ``rows*cols`` into a
        # single inner dim. ``shape[3]`` was therefore out-of-bounds,
        # reading garbage from the next array's shape slot. The result:
        # static-geom ``geom_xmat`` entries got overwritten with the
        # wrong value, which silently corrupted contact_jac and made
        # ``qacc`` diverge by ~25 in cartpole.
        #
        # The kernel below mirrors the early-return pattern of
        # ``_geom_local_to_global``: thread 0 keeps its user-provided
        # value, threads 1+ overwrite. CPU/Metal must agree.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(skip_mask: wp.array(dtype=wp.int32),
                  pos_out: wp.array2d(dtype=wp.vec3),
                  mat_out: wp.array2d(dtype=wp.mat33)):
                w, i = wp.tid()
                if skip_mask[i] == 1:
                    return
                pos_out[w, i] = wp.vec3(float(i), 2.0 * float(i), 3.0 * float(i))
                mat_out[w, i] = wp.mat33(
                    float(i + 1), 0.0, 0.0,
                    0.0, float(i + 2), 0.0,
                    0.0, 0.0, float(i + 3),
                )

            NW = 1
            NG = 5
            # Geoms 0, 1, 2 are "static" (skipped) — they keep their
            # user-provided value. Geoms 3, 4 are written by the kernel.
            mask = np.array([1, 1, 1, 0, 0], dtype=np.int32)

            # Seed values shaped to expose the (rows * cols) flattening
            # so an off-by-one stride bug shows up across the static-
            # geom slots, not just at the boundary.
            rng = np.random.default_rng(0)
            pos_seed = rng.standard_normal((NW, NG, 3)).astype(np.float32)
            mat_seed = rng.standard_normal((NW, NG, 3, 3)).astype(np.float32)

            results = {}
            for dev in ("cpu", "metal:0"):
                m_arr = wp.array(mask, dtype=wp.int32, device=dev)
                p = wp.from_numpy(pos_seed, dtype=wp.vec3, device=dev)
                m = wp.from_numpy(mat_seed, dtype=wp.mat33, device=dev)
                wp.launch(k, dim=(NW, NG), inputs=[m_arr],
                          outputs=[p, m], device=dev)
                results[dev] = (p.numpy(), m.numpy())

            # Static slots (mask==1): equal to seed.
            np.testing.assert_array_equal(results['metal:0'][0][:, :3], pos_seed[:, :3])
            np.testing.assert_array_equal(results['metal:0'][1][:, :3], mat_seed[:, :3])
            # Active slots (mask==0): equal to the kernel's output formula.
            for i in (3, 4):
                np.testing.assert_allclose(
                    results['metal:0'][0][0, i], [i, 2.0 * i, 3.0 * i], atol=0)
                expected_mat = np.diag([i + 1, i + 2, i + 3]).astype(np.float32)
                np.testing.assert_allclose(
                    results['metal:0'][1][0, i], expected_mat, atol=0)
            # And bit-exact CPU vs Metal for both outputs.
            np.testing.assert_array_equal(results['cpu'][0], results['metal:0'][0])
            np.testing.assert_array_equal(results['cpu'][1], results['metal:0'][1])
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=60)

    def test_tile_load_store_round_trip_matches_cpu(self):
        # Smallest tile primitive case: load a 6x6 sub-block and store
        # it back. Exercises ``_emit_tile_struct`` and confirms the
        # template-on-address-space load helper works for both
        # ``device`` and ``constant`` MLX argument bindings (small
        # read-only buffers like the 6-element RHS land in the
        # constant pool, which used to break the type-monomorphic
        # signature).
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(A: wp.array2d(dtype=wp.float32),
                  out: wp.array2d(dtype=wp.float32)):
                a = wp.tile_load(A, shape=(6, 6), offset=(0, 0), storage="shared")
                wp.tile_store(out, a, offset=(0, 0))

            rng = np.random.default_rng(0)
            A_h = rng.standard_normal((6, 6)).astype(np.float32)
            results = {}
            for dev in ("cpu", "metal:0"):
                A = wp.array(A_h, dtype=wp.float32, device=dev)
                out = wp.zeros((6, 6), dtype=wp.float32, device=dev)
                wp.launch_tiled(k, dim=[1], inputs=[A], outputs=[out],
                                block_dim=1, device=dev)
                results[dev] = out.numpy()
            np.testing.assert_array_equal(results['cpu'], A_h)
            np.testing.assert_array_equal(results['metal:0'], A_h)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_tile_load_with_offset_matches_cpu(self):
        # Non-zero offset path: pulls a 4x3 sub-block from the middle
        # of an 8x6 backing array, exercising the row-stride and
        # row-offset arithmetic that the OBB / contact_jac kernels
        # rely on.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(A: wp.array2d(dtype=wp.float32),
                  out: wp.array2d(dtype=wp.float32)):
                a = wp.tile_load(A, shape=(4, 3), offset=(2, 1), storage="shared")
                wp.tile_store(out, a, offset=(0, 0))

            rng = np.random.default_rng(42)
            A_h = rng.standard_normal((8, 6)).astype(np.float32)
            ref = A_h[2:6, 1:4]
            for dev in ("cpu", "metal:0"):
                A = wp.array(A_h, dtype=wp.float32, device=dev)
                out = wp.zeros((4, 3), dtype=wp.float32, device=dev)
                wp.launch_tiled(k, dim=[1], inputs=[A], outputs=[out],
                                block_dim=1, device=dev)
                np.testing.assert_array_equal(out.numpy(), ref)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_tile_cholesky_solve_n6_matches_cpu(self):
        # Freejoint mass-matrix size: N=6 SPD with a vector RHS.
        # mujoco_warp's simple (non-blocked) path goes through this
        # exact shape for any single-freejoint body.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            N = wp.constant(6)

            @wp.kernel
            def k(A: wp.array2d(dtype=wp.float32),
                  y: wp.array(dtype=wp.float32),
                  L_out: wp.array2d(dtype=wp.float32),
                  x_out: wp.array(dtype=wp.float32)):
                a = wp.tile_load(A, shape=(N, N), storage="shared")
                rhs = wp.tile_load(y, shape=N, storage="shared")
                L = wp.tile_cholesky(a)
                x = wp.tile_cholesky_solve(L, rhs)
                wp.tile_store(L_out, L)
                wp.tile_store(x_out, x)

            rng = np.random.default_rng(7)
            M = rng.standard_normal((6, 6)).astype(np.float32)
            A_h = (M @ M.T + 6.0 * np.eye(6, dtype=np.float32))
            y_h = rng.standard_normal(6).astype(np.float32)
            L_np = np.linalg.cholesky(A_h.astype(np.float64)).astype(np.float32)
            x_np = np.linalg.solve(A_h.astype(np.float64),
                                   y_h.astype(np.float64)).astype(np.float32)
            results = {}
            for dev in ("cpu", "metal:0"):
                A = wp.array(A_h, dtype=wp.float32, device=dev)
                y = wp.array(y_h, dtype=wp.float32, device=dev)
                Lo = wp.zeros((6, 6), dtype=wp.float32, device=dev)
                xo = wp.zeros(6, dtype=wp.float32, device=dev)
                wp.launch_tiled(k, dim=[1], inputs=[A, y], outputs=[Lo, xo],
                                block_dim=1, device=dev)
                results[dev] = (Lo.numpy(), xo.numpy())
            # L is bit-exact between CPU and Metal; x can drift by ~1e-9
            # from the float32 precision of the two solver paths.
            np.testing.assert_array_equal(results['cpu'][0], results['metal:0'][0])
            np.testing.assert_allclose(results['metal:0'][0], L_np, atol=1e-6)
            np.testing.assert_allclose(results['metal:0'][1], x_np, atol=1e-5)
            np.testing.assert_allclose(results['cpu'][1], results['metal:0'][1], atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=60)

    def test_tile_cholesky_solve_n8_matches_cpu(self):
        # Eight-DOF block — the size at which the MSL outer-loop unroll
        # bug surfaces if ``#pragma clang loop unroll(disable)`` is
        # missing from the Cholesky / triangular solve emitters. This
        # is the canonical regression for that workaround.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            N = wp.constant(8)

            @wp.kernel
            def k(A: wp.array2d(dtype=wp.float32),
                  y: wp.array(dtype=wp.float32),
                  L_out: wp.array2d(dtype=wp.float32),
                  x_out: wp.array(dtype=wp.float32)):
                a = wp.tile_load(A, shape=(N, N), storage="shared")
                rhs = wp.tile_load(y, shape=N, storage="shared")
                L = wp.tile_cholesky(a)
                x = wp.tile_cholesky_solve(L, rhs)
                wp.tile_store(L_out, L)
                wp.tile_store(x_out, x)

            rng = np.random.default_rng(11)
            M = rng.standard_normal((8, 8)).astype(np.float32)
            A_h = (M @ M.T + 8.0 * np.eye(8, dtype=np.float32))
            y_h = rng.standard_normal(8).astype(np.float32)
            L_np = np.linalg.cholesky(A_h.astype(np.float64)).astype(np.float32)
            x_np = np.linalg.solve(A_h.astype(np.float64),
                                   y_h.astype(np.float64)).astype(np.float32)
            results = {}
            for dev in ("cpu", "metal:0"):
                A = wp.array(A_h, dtype=wp.float32, device=dev)
                y = wp.array(y_h, dtype=wp.float32, device=dev)
                Lo = wp.zeros((8, 8), dtype=wp.float32, device=dev)
                xo = wp.zeros(8, dtype=wp.float32, device=dev)
                wp.launch_tiled(k, dim=[1], inputs=[A, y], outputs=[Lo, xo],
                                block_dim=1, device=dev)
                results[dev] = (Lo.numpy(), xo.numpy())
            # Last row of L is the canary — the unroll bug zeroed
            # ``L[7, 0:6]`` while leaving ``L[7, 6:8]`` correct.
            np.testing.assert_allclose(results['metal:0'][0], L_np, atol=1e-5)
            np.testing.assert_allclose(results['metal:0'][1], x_np, atol=1e-4)
            np.testing.assert_allclose(results['cpu'][0], results['metal:0'][0], atol=1e-6)
            np.testing.assert_allclose(results['cpu'][1], results['metal:0'][1], atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=60)

    def test_tile_cholesky_solve_n32_matches_cpu(self):
        # 32-DOF Cholesky — well above the unroll-bug threshold and
        # at the compile-cost knee of the single-thread path. Verifies
        # the emitted MSL stays numerically stable as N grows: the
        # outer ``j`` loop accumulates ~N² fmas which compound into
        # the diagonal pivot, and ``precise::sqrt`` is the only
        # protection against the resulting reassociation drift.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            N = wp.constant(32)

            @wp.kernel
            def k(A: wp.array2d(dtype=wp.float32),
                  y: wp.array(dtype=wp.float32),
                  L_out: wp.array2d(dtype=wp.float32),
                  x_out: wp.array(dtype=wp.float32)):
                a = wp.tile_load(A, shape=(N, N), storage="shared")
                rhs = wp.tile_load(y, shape=N, storage="shared")
                L = wp.tile_cholesky(a)
                x = wp.tile_cholesky_solve(L, rhs)
                wp.tile_store(L_out, L)
                wp.tile_store(x_out, x)

            rng = np.random.default_rng(32)
            M = rng.standard_normal((32, 32)).astype(np.float32)
            A_h = (M @ M.T + 32.0 * np.eye(32, dtype=np.float32))
            y_h = rng.standard_normal(32).astype(np.float32)
            x_np = np.linalg.solve(A_h.astype(np.float64),
                                   y_h.astype(np.float64)).astype(np.float32)
            results = {}
            for dev in ("cpu", "metal:0"):
                A = wp.array(A_h, dtype=wp.float32, device=dev)
                y = wp.array(y_h, dtype=wp.float32, device=dev)
                Lo = wp.zeros((32, 32), dtype=wp.float32, device=dev)
                xo = wp.zeros(32, dtype=wp.float32, device=dev)
                wp.launch_tiled(k, dim=[1], inputs=[A, y], outputs=[Lo, xo],
                                block_dim=1, device=dev)
                results[dev] = (Lo.numpy(), xo.numpy())
            # Reconstruction L L^T == A is the cleanest correctness
            # signal — independent of any reference solver and tight
            # to ~N * eps_f32 ≈ 4e-6 in absolute terms.
            Lm = results['metal:0'][0]
            recon = Lm.astype(np.float64) @ Lm.T.astype(np.float64)
            np.testing.assert_allclose(recon, A_h.astype(np.float64), atol=1e-4)
            np.testing.assert_allclose(results['metal:0'][1], x_np, atol=1e-4)
            np.testing.assert_allclose(results['cpu'][0], results['metal:0'][0], atol=1e-5)
            np.testing.assert_allclose(results['cpu'][1], results['metal:0'][1], atol=1e-5)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_tile_cholesky_solve_n64_matches_cpu(self):
        # 64-DOF Cholesky — top end of the simple (non-blocked) path.
        # mujoco_warp's ``_BLOCK_CHOLESKY_DIM`` bumps to 64 on Metal
        # so any model with nv ≤ 64 stays on this codegen branch.
        # Runs in ~80ms launch + ~7ms steady on M3, well within
        # CI bounds.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            N = wp.constant(64)

            @wp.kernel
            def k(A: wp.array2d(dtype=wp.float32),
                  y: wp.array(dtype=wp.float32),
                  L_out: wp.array2d(dtype=wp.float32),
                  x_out: wp.array(dtype=wp.float32)):
                a = wp.tile_load(A, shape=(N, N), storage="shared")
                rhs = wp.tile_load(y, shape=N, storage="shared")
                L = wp.tile_cholesky(a)
                x = wp.tile_cholesky_solve(L, rhs)
                wp.tile_store(L_out, L)
                wp.tile_store(x_out, x)

            rng = np.random.default_rng(64)
            M = rng.standard_normal((64, 64)).astype(np.float32)
            A_h = (M @ M.T + 64.0 * np.eye(64, dtype=np.float32))
            y_h = rng.standard_normal(64).astype(np.float32)
            x_np = np.linalg.solve(A_h.astype(np.float64),
                                   y_h.astype(np.float64)).astype(np.float32)
            results = {}
            for dev in ("cpu", "metal:0"):
                A = wp.array(A_h, dtype=wp.float32, device=dev)
                y = wp.array(y_h, dtype=wp.float32, device=dev)
                Lo = wp.zeros((64, 64), dtype=wp.float32, device=dev)
                xo = wp.zeros(64, dtype=wp.float32, device=dev)
                wp.launch_tiled(k, dim=[1], inputs=[A, y], outputs=[Lo, xo],
                                block_dim=1, device=dev)
                results[dev] = (Lo.numpy(), xo.numpy())
            Lm = results['metal:0'][0]
            recon = Lm.astype(np.float64) @ Lm.T.astype(np.float64)
            np.testing.assert_allclose(recon, A_h.astype(np.float64), atol=5e-4)
            np.testing.assert_allclose(results['metal:0'][1], x_np, atol=1e-4)
            np.testing.assert_allclose(results['cpu'][0], results['metal:0'][0], atol=1e-5)
            np.testing.assert_allclose(results['cpu'][1], results['metal:0'][1], atol=1e-5)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=180)

    def test_three_tile_cholesky_inplace_calls_match_cpu(self):
        # Regression for the Metal-compiler inline bug: when the
        # ``cholesky_inplace`` (or ``lower_solve_inplace`` /
        # ``upper_solve_inplace`` / ``matmul``) helper is inlined 3+
        # times into the same kernel, the FIRST inlined copy's writes
        # to the tile struct silently come back as zero — no compile
        # error, deterministic. We work around it by emitting these
        # helpers with ``__attribute__((noinline))``.
        #
        # The bug is the root cause of mujoco_warp's blocked-Cholesky
        # factor returning NaN/zero on Metal: every block writes to a
        # different tile local, but Apple's MSL compiler conflates
        # the writes when too many copies are inlined.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            N = 16

            @wp.kernel
            def k(A: wp.array2d(dtype=float), L: wp.array2d(dtype=float)):
                A0 = wp.tile_load(A, shape=(N, N), offset=(0, 0))
                wp.tile_cholesky_inplace(A0)
                wp.tile_store(L, A0, offset=(0, 0))
                A1 = wp.tile_load(A, shape=(N, N), offset=(16, 16))
                wp.tile_cholesky_inplace(A1)
                wp.tile_store(L, A1, offset=(16, 16))
                A2 = wp.tile_load(A, shape=(N, N), offset=(32, 32))
                wp.tile_cholesky_inplace(A2)
                wp.tile_store(L, A2, offset=(32, 32))

            rng = np.random.default_rng(0)
            M = rng.standard_normal((48, 48)).astype(np.float32)
            H = (M @ M.T + 5.0 * 48 * np.eye(48)).astype(np.float32)
            results = {}
            for dev in ("cpu", "metal:0"):
                A_d = wp.array(H, dtype=wp.float32, device=dev)
                L_d = wp.zeros((48, 48), dtype=wp.float32, device=dev)
                wp.launch_tiled(k, dim=1, inputs=[A_d], outputs=[L_d],
                                block_dim=32, device=dev)
                results[dev] = L_d.numpy()
            # Each diagonal block is the cholesky of the corresponding
            # 16x16 sub-block of H. If the inline bug returns, the
            # first block (L[0,0] etc.) zeros out.
            for i in (0, 16, 32):
                np.testing.assert_allclose(
                    results['metal:0'][i, i],
                    np.sqrt(H[i, i]), atol=1e-4,
                    err_msg=f'L[{i},{i}] off — inline bug regression?')
            np.testing.assert_allclose(
                results['cpu'], results['metal:0'], atol=1e-5)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_two_solve_inplace_with_transpose_writeback(self):
        # Regression for a Metal-compiler miscompile that manifests when
        # ``tile_lower_solve_inplace(L, tile_transpose(A))`` appears more
        # than once in the same kernel. The transpose-back writeback
        # ``var_A = transpose(var_B)`` after the second call returns a
        # stale struct, leaving rows of ``A`` silently zeroed out. This
        # was the root cause of G1's blocked-Cholesky producing NaN at
        # the third diagonal block: the i-loop iterates twice for the
        # k=0 outer iteration (i=16, i=32) and the i=32 writeback was
        # corrupted.
        #
        # The fix lowers ``solve(L, transpose(A))`` to a fused
        # ``solve_transposed(L, A)`` helper that operates on ``A``
        # directly, sidestepping the temporary entirely.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            N = 16

            @wp.kernel
            def k(L_arr: wp.array2d(dtype=float),
                  A1_arr: wp.array2d(dtype=float),
                  A2_arr: wp.array2d(dtype=float),
                  out1: wp.array2d(dtype=float),
                  out2: wp.array2d(dtype=float)):
                L = wp.tile_load(L_arr, shape=(N, N))
                A1 = wp.tile_load(A1_arr, shape=(N, N))
                A2 = wp.tile_load(A2_arr, shape=(N, N))
                wp.tile_lower_solve_inplace(L, wp.tile_transpose(A1))
                wp.tile_store(out1, A1)
                wp.tile_lower_solve_inplace(L, wp.tile_transpose(A2))
                wp.tile_store(out2, A2)

            rng = np.random.default_rng(0)
            M = rng.standard_normal((N, N)).astype(np.float32)
            L_np = np.linalg.cholesky((M @ M.T) + np.eye(N) * 0.5).astype(np.float32)
            A1_np = rng.standard_normal((N, N)).astype(np.float32)
            A2_np = rng.standard_normal((N, N)).astype(np.float32)
            # Math: solve(L, transpose(A)) computes X = L^{-1} A^T;
            # writeback gives A = X^T = A L^{-T}.
            expected1 = (A1_np.astype(np.float64) @ np.linalg.inv(L_np).T).astype(np.float32)
            expected2 = (A2_np.astype(np.float64) @ np.linalg.inv(L_np).T).astype(np.float32)

            for dev in ('cpu', 'metal:0'):
                L_d = wp.array(L_np, dtype=wp.float32, device=dev)
                A1_d = wp.array(A1_np, dtype=wp.float32, device=dev)
                A2_d = wp.array(A2_np, dtype=wp.float32, device=dev)
                o1 = wp.zeros((N, N), dtype=wp.float32, device=dev)
                o2 = wp.zeros((N, N), dtype=wp.float32, device=dev)
                wp.launch_tiled(k, dim=1,
                                inputs=[L_d, A1_d, A2_d],
                                outputs=[o1, o2],
                                block_dim=1, device=dev)
                np.testing.assert_allclose(o1.numpy(), expected1, atol=2e-5,
                    err_msg=f'first solve on {dev}')
                np.testing.assert_allclose(o2.numpy(), expected2, atol=2e-5,
                    err_msg=f'second solve on {dev} — writeback miscompile regression?')
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_simd_cooperative_cholesky_msl_prototype(self):
        # Standalone validation that a SIMD-cooperative right-looking
        # Cholesky compiles and runs correctly on Apple GPU. This is
        # the algorithmic backbone for task #19 (parallel tile
        # primitives) and is verified independently of Warp's tile
        # codegen so we can keep iterating on the MSL without
        # disrupting existing kernels.
        #
        # 32 lanes cooperate on an N×N tile in threadgroup memory.
        # Each lane owns a strided set of rows; lane (k mod 32)
        # computes the pivot on iteration k, then all lanes in
        # parallel update their owned rows. Threadgroup-barrier
        # synchronization separates pivot from off-diagonal updates.
        #
        # Speedup over the single-thread serial emit (M3, steady):
        #   N=32:  1.6 ms -> 0.5 ms   (3.2x)
        #   N=48:  5.2 ms -> 1.0 ms   (5.2x)
        #   N=64:  7.6 ms -> 0.8 ms   (9.5x)
        # Above N=64 the threadgroup-memory cap (32 KB on Apple Silicon)
        # kicks in (96^2 * 4 B > 32 KB) — that's the boundary at which
        # we'd need to spill the tile or block it.
        snippet = textwrap.dedent(
            """
            import numpy as np
            import mlx.core as mx

            def make_src(N):
                return f'''
                constexpr int N = {N};
                constexpr int LANES = 32;
                threadgroup float Lsmem[N * N];
                uint lane = thread_position_in_threadgroup.x;

                for (uint idx = lane; idx < N * N; idx += LANES)
                    Lsmem[idx] = A[idx];
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (int k = 0; k < N; ++k) {{
                    if ((int)lane == (k % LANES)) {{
                        float d = Lsmem[k * N + k];
                        for (int j = 0; j < k; ++j) {{
                            float ljk = Lsmem[k * N + j];
                            d -= ljk * ljk;
                        }}
                        d = max(d, 1e-30f);
                        Lsmem[k * N + k] = precise::sqrt(d);
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    float pivot = Lsmem[k * N + k];
                    for (int i = (int)lane; i < N; i += LANES) {{
                        if (i > k) {{
                            float s = Lsmem[i * N + k];
                            for (int j = 0; j < k; ++j) {{
                                s -= Lsmem[i * N + j] * Lsmem[k * N + j];
                            }}
                            Lsmem[i * N + k] = s / pivot;
                        }}
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }}

                for (uint idx = lane; idx < N * N; idx += LANES) {{
                    int i = (int)(idx / N);
                    int j = (int)(idx % N);
                    L[idx] = (j > i) ? 0.0f : Lsmem[idx];
                }}
                '''

            HEADER = '#include <metal_stdlib>\\nusing namespace metal;\\n'

            for N in (8, 16, 32, 48, 64):
                rng = np.random.default_rng(N)
                M = rng.standard_normal((N, N)).astype(np.float32)
                A_h = (M @ M.T + float(N) * np.eye(N, dtype=np.float32))
                kernel = mx.fast.metal_kernel(
                    name=f'chol_simd_{N}',
                    input_names=['A'], output_names=['L'],
                    source=make_src(N), header=HEADER,
                )
                out = kernel(
                    inputs=[mx.array(A_h.flatten())],
                    grid=(32, 1, 1), threadgroup=(32, 1, 1),
                    output_shapes=[(N * N,)], output_dtypes=[mx.float32],
                )
                mx.eval(out[0])
                L_metal = np.array(out[0]).reshape(N, N)
                # Reconstruction L L^T == A is the cleanest correctness
                # check (independent of any reference solver).
                LLT = L_metal.astype(np.float64) @ L_metal.T.astype(np.float64)
                np.testing.assert_allclose(
                    LLT, A_h.astype(np.float64),
                    atol=2e-4, err_msg=f'N={N}: ||LLᵀ - A|| too large')
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_tile_cholesky_inplace_matches_cpu(self):
        # Inplace variant — used by mujoco_warp's blocked Cholesky
        # path. Reuses the storage of A as the factor L, so any
        # accidental aliasing or stale-storage bug shows up here.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            N = wp.constant(6)

            @wp.kernel
            def k(A: wp.array2d(dtype=wp.float32),
                  y: wp.array(dtype=wp.float32),
                  L_out: wp.array2d(dtype=wp.float32),
                  x_out: wp.array(dtype=wp.float32)):
                a = wp.tile_load(A, shape=(N, N), storage="shared")
                rhs = wp.tile_load(y, shape=N, storage="shared")
                wp.tile_cholesky_inplace(a)
                wp.tile_cholesky_solve_inplace(a, rhs)
                wp.tile_store(L_out, a)
                wp.tile_store(x_out, rhs)

            rng = np.random.default_rng(13)
            M = rng.standard_normal((6, 6)).astype(np.float32)
            A_h = (M @ M.T + 6.0 * np.eye(6, dtype=np.float32))
            y_h = rng.standard_normal(6).astype(np.float32)
            L_np = np.linalg.cholesky(A_h.astype(np.float64)).astype(np.float32)
            x_np = np.linalg.solve(A_h.astype(np.float64),
                                   y_h.astype(np.float64)).astype(np.float32)
            results = {}
            for dev in ("cpu", "metal:0"):
                A = wp.array(A_h, dtype=wp.float32, device=dev)
                y = wp.array(y_h, dtype=wp.float32, device=dev)
                Lo = wp.zeros((6, 6), dtype=wp.float32, device=dev)
                xo = wp.zeros(6, dtype=wp.float32, device=dev)
                wp.launch_tiled(k, dim=[1], inputs=[A, y], outputs=[Lo, xo],
                                block_dim=1, device=dev)
                results[dev] = (Lo.numpy(), xo.numpy())
            np.testing.assert_allclose(results['metal:0'][0], L_np, atol=1e-6)
            np.testing.assert_allclose(results['metal:0'][1], x_np, atol=1e-5)
            np.testing.assert_allclose(results['cpu'][0], results['metal:0'][0], atol=1e-6)
            np.testing.assert_allclose(results['cpu'][1], results['metal:0'][1], atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=60)

    def test_rejects_adjoint_launch(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32),
                  c: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                c[tid] = a[tid] + b[tid]

            a = wp.zeros(8, dtype=wp.float32, device='metal:0')
            b = wp.zeros(8, dtype=wp.float32, device='metal:0')
            c = wp.zeros(8, dtype=wp.float32, device='metal:0')
            try:
                wp.launch(k, dim=8, inputs=[a, b], outputs=[c], device='metal:0', adjoint=True)
            except RuntimeError as e:
                assert 'adjoint' in str(e).lower(), str(e)
            else:
                raise AssertionError('expected RuntimeError for adjoint=True on Metal')
            """
        )
        _run_with_metal_enabled(self, snippet)


if __name__ == "__main__":
    unittest.main(verbosity=2)
