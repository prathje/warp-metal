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

    def test_atomic_add_int_scalar_accumulator_stress(self):
        # Regression test for the output-init prologue race: the prologue's
        # per-threadgroup seed used ``atomic_store``, which could land AFTER
        # another threadgroup's ``atomic_add``s and wipe them (observed as
        # a flaky short-count in ``test_atomic_add_int_matches_cpu_bit_exact``,
        # ~1 in 5 runs at dim=1<<20). Add/sub-only outputs are now seeded
        # with ``atomic_fetch_add``, which commutes with the body's adds.
        # The large dim keeps many threadgroups in flight so a reintroduced
        # ordering bug fails reliably rather than once in a blue moon.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.int32), c: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                wp.atomic_add(c, 0, a[tid])

            N = 1 << 20
            an = np.ones(N, dtype=np.int32)
            a_m = wp.array(an, dtype=wp.int32, device='metal:0')
            for trial in range(5):
                c_m = wp.zeros(1, dtype=wp.int32, device='metal:0')
                wp.launch(k, dim=N, inputs=[a_m], outputs=[c_m], device='metal:0')
                got = int(c_m.numpy()[0])
                assert got == N, f'trial {trial}: lost {N - got} atomic adds'
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_atomic_add_seed_accumulates_from_prior_value(self):
        # The fetch_add seed must accumulate from the wp.array's prior
        # contents (CUDA semantics: users seed accumulators themselves),
        # and repeated launches must keep accumulating.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.int32), c: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                wp.atomic_add(c, 0, a[tid])

            N = 256
            an = np.ones(N, dtype=np.int32)
            a_m = wp.array(an, dtype=wp.int32, device='metal:0')
            c_m = wp.array(np.array([1000], dtype=np.int32), dtype=wp.int32,
                           device='metal:0')
            wp.launch(k, dim=N, inputs=[a_m], outputs=[c_m], device='metal:0')
            wp.launch(k, dim=N, inputs=[a_m], outputs=[c_m], device='metal:0')
            got = int(c_m.numpy()[0])
            assert got == 1000 + 2 * N, f'expected {1000 + 2 * N}, got {got}'
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


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalQuatBuiltins(unittest.TestCase):
    """A/B tests for the quaternion builtin translations.

    Quats are stored as ``float4`` on Metal, which makes ``q1 * q2`` a
    silent-wrongness trap: MSL's ``float4 * float4`` is component-wise, but
    Warp's quat multiply is the Hamilton product. These tests pin the
    ``wp_quat_*`` helper translations against the CPU backend.
    """

    def test_quat_mul_is_hamilton_product(self):
        # Regression: this used to lower to component-wise float4 multiply.
        # Covers both the direct kernel statement and the @wp.func-inlined
        # path (inlined vars are found via textual quat_t declarations, not
        # the kernel IR's variable list).
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.func
            def helper_mul(a: wp.quat, b: wp.quat) -> wp.quat:
                return a * b

            @wp.kernel
            def k(a: wp.array(dtype=wp.quat), b: wp.array(dtype=wp.quat),
                  direct: wp.array(dtype=wp.quat), inlined: wp.array(dtype=wp.quat)):
                tid = wp.tid()
                direct[tid] = a[tid] * b[tid]
                inlined[tid] = helper_mul(a[tid], b[tid])

            N = 64
            rng = np.random.default_rng(0)
            an = rng.standard_normal((N, 4)).astype(np.float32)
            bn = rng.standard_normal((N, 4)).astype(np.float32)
            an /= np.linalg.norm(an, axis=1, keepdims=True)
            bn /= np.linalg.norm(bn, axis=1, keepdims=True)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                direct = wp.zeros(N, dtype=wp.quat, device=dev)
                inlined = wp.zeros(N, dtype=wp.quat, device=dev)
                wp.launch(k, dim=N,
                          inputs=[wp.array(an, dtype=wp.quat, device=dev),
                                  wp.array(bn, dtype=wp.quat, device=dev)],
                          outputs=[direct, inlined], device=dev)
                outs[dev] = (direct.numpy(), inlined.numpy())

            np.testing.assert_allclose(outs['cpu'][0], outs['metal:0'][0], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(outs['cpu'][1], outs['metal:0'][1], rtol=1e-5, atol=1e-6)
            # Guard against the component-wise regression specifically: the
            # Hamilton product of two unit quats differs from the
            # component-wise product for generic inputs.
            assert not np.allclose(outs['metal:0'][0], an * bn, atol=1e-3), \\
                'Metal quat mul looks component-wise again'
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_quat_rotate_inverse_identity_match_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(q: wp.array(dtype=wp.quat), v: wp.array(dtype=wp.vec3),
                  rot: wp.array(dtype=wp.vec3), rot_inv: wp.array(dtype=wp.vec3),
                  inv: wp.array(dtype=wp.quat)):
                tid = wp.tid()
                rot[tid] = wp.quat_rotate(q[tid], v[tid])
                rot_inv[tid] = wp.quat_rotate_inv(q[tid], v[tid])
                inv[tid] = wp.quat_inverse(q[tid]) * wp.quat_identity()

            N = 64
            rng = np.random.default_rng(1)
            qn = rng.standard_normal((N, 4)).astype(np.float32)
            qn /= np.linalg.norm(qn, axis=1, keepdims=True)
            vn = rng.standard_normal((N, 3)).astype(np.float32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                rot = wp.zeros(N, dtype=wp.vec3, device=dev)
                rot_inv = wp.zeros(N, dtype=wp.vec3, device=dev)
                inv = wp.zeros(N, dtype=wp.quat, device=dev)
                wp.launch(k, dim=N,
                          inputs=[wp.array(qn, dtype=wp.quat, device=dev),
                                  wp.array(vn, dtype=wp.vec3, device=dev)],
                          outputs=[rot, rot_inv, inv], device=dev)
                outs[dev] = (rot.numpy(), rot_inv.numpy(), inv.numpy())

            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_allclose(c, m, rtol=1e-5, atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_quat_conversions_match_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(q: wp.array(dtype=wp.quat), v: wp.array(dtype=wp.vec3),
                  s: wp.array(dtype=wp.float32),
                  mat: wp.array(dtype=wp.mat33), round_trip: wp.array(dtype=wp.quat),
                  from_axis: wp.array(dtype=wp.quat), rpy: wp.array(dtype=wp.quat)):
                tid = wp.tid()
                mat[tid] = wp.quat_to_matrix(q[tid])
                round_trip[tid] = wp.quat_from_matrix(wp.quat_to_matrix(q[tid]))
                from_axis[tid] = wp.quat_from_axis_angle(wp.normalize(v[tid]), s[tid])
                rpy[tid] = wp.quat_rpy(v[tid][0], v[tid][1], v[tid][2])

            N = 64
            rng = np.random.default_rng(2)
            qn = rng.standard_normal((N, 4)).astype(np.float32)
            qn /= np.linalg.norm(qn, axis=1, keepdims=True)
            vn = rng.standard_normal((N, 3)).astype(np.float32)
            sn = rng.standard_normal(N).astype(np.float32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                mat = wp.zeros(N, dtype=wp.mat33, device=dev)
                round_trip = wp.zeros(N, dtype=wp.quat, device=dev)
                from_axis = wp.zeros(N, dtype=wp.quat, device=dev)
                rpy = wp.zeros(N, dtype=wp.quat, device=dev)
                wp.launch(k, dim=N,
                          inputs=[wp.array(qn, dtype=wp.quat, device=dev),
                                  wp.array(vn, dtype=wp.vec3, device=dev),
                                  wp.array(sn, dtype=wp.float32, device=dev)],
                          outputs=[mat, round_trip, from_axis, rpy], device=dev)
                outs[dev] = (mat.numpy(), round_trip.numpy(), from_axis.numpy(), rpy.numpy())

            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_allclose(c, m, rtol=1e-5, atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_quat_slerp_matches_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.quat), b: wp.array(dtype=wp.quat),
                  t: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.quat)):
                tid = wp.tid()
                out[tid] = wp.quat_slerp(a[tid], b[tid], t[tid])

            N = 64
            rng = np.random.default_rng(4)
            an = rng.standard_normal((N, 4)).astype(np.float32)
            bn = rng.standard_normal((N, 4)).astype(np.float32)
            an /= np.linalg.norm(an, axis=1, keepdims=True)
            bn /= np.linalg.norm(bn, axis=1, keepdims=True)
            tn = rng.uniform(0.0, 1.0, N).astype(np.float32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                out = wp.zeros(N, dtype=wp.quat, device=dev)
                wp.launch(k, dim=N,
                          inputs=[wp.array(an, dtype=wp.quat, device=dev),
                                  wp.array(bn, dtype=wp.quat, device=dev),
                                  wp.array(tn, dtype=wp.float32, device=dev)],
                          outputs=[out], device=dev)
                outs[dev] = out.numpy()

            np.testing.assert_allclose(outs['cpu'], outs['metal:0'], rtol=1e-4, atol=1e-5)
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalTransformBuiltins(unittest.TestCase):
    """A/B tests for ``wp.transform`` support.

    Transforms lower to ``wp_vec7_float`` (px, py, pz, qx, qy, qz, qw) via
    the same normalization quats use, with ``wp_transform_*`` helpers for
    the transform-specific operations.
    """

    def test_transform_builtins_match_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(t: wp.array(dtype=wp.transform), p: wp.array(dtype=wp.vec3),
                  out_pt: wp.array(dtype=wp.vec3), out_vec: wp.array(dtype=wp.vec3),
                  out_rt: wp.array(dtype=wp.transform),
                  out_ctor: wp.array(dtype=wp.transform),
                  out_id: wp.array(dtype=wp.transform)):
                tid = wp.tid()
                a = t[tid]
                out_pt[tid] = wp.transform_point(a, p[tid])
                out_vec[tid] = wp.transform_vector(a, p[tid])
                out_rt[tid] = wp.transform_multiply(a, wp.transform_inverse(a))
                out_ctor[tid] = wp.transform(wp.transform_get_translation(a) * 2.0,
                                             wp.transform_get_rotation(a))
                out_id[tid] = wp.transform_multiply(a, wp.transform_identity())

            N = 32
            rng = np.random.default_rng(7)
            tn = rng.standard_normal((N, 7)).astype(np.float32)
            tn[:, 3:] /= np.linalg.norm(tn[:, 3:], axis=1, keepdims=True)
            on = rng.standard_normal((N, 3)).astype(np.float32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                out_pt = wp.zeros(N, dtype=wp.vec3, device=dev)
                out_vec = wp.zeros(N, dtype=wp.vec3, device=dev)
                out_rt = wp.zeros(N, dtype=wp.transform, device=dev)
                out_ctor = wp.zeros(N, dtype=wp.transform, device=dev)
                out_id = wp.zeros(N, dtype=wp.transform, device=dev)
                wp.launch(k, dim=N,
                          inputs=[wp.array(tn, dtype=wp.transform, device=dev),
                                  wp.array(on, dtype=wp.vec3, device=dev)],
                          outputs=[out_pt, out_vec, out_rt, out_ctor, out_id],
                          device=dev)
                outs[dev] = [o.numpy() for o in (out_pt, out_vec, out_rt, out_ctor, out_id)]

            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_allclose(c, m, rtol=1e-5, atol=1e-5)
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalMathBuiltins(unittest.TestCase):
    """A/B tests for interpolation / misc-math / small linear-algebra builtins."""

    def test_interp_builtins_match_cpu(self):
        # frac deliberately covers negative inputs: Warp truncates toward
        # zero while metal::fract floors, so a fract-based translation
        # would diverge exactly there.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32),
                  va: wp.array(dtype=wp.vec3), vb: wp.array(dtype=wp.vec3),
                  lerp_s: wp.array(dtype=wp.float32), lerp_v: wp.array(dtype=wp.vec3),
                  smooth: wp.array(dtype=wp.float32), fr: wp.array(dtype=wp.float32),
                  ang: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                lerp_s[tid] = wp.lerp(a[tid], b[tid], 0.3)
                lerp_v[tid] = wp.lerp(va[tid], vb[tid], 0.7)
                smooth[tid] = wp.smoothstep(-1.0, 1.0, a[tid])
                fr[tid] = wp.frac(a[tid] * 3.7)
                ang[tid] = wp.degrees(a[tid]) + wp.radians(b[tid])

            N = 128
            rng = np.random.default_rng(5)
            an = rng.standard_normal(N).astype(np.float32)
            bn = rng.standard_normal(N).astype(np.float32)
            van = rng.standard_normal((N, 3)).astype(np.float32)
            vbn = rng.standard_normal((N, 3)).astype(np.float32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                lerp_s = wp.zeros(N, dtype=wp.float32, device=dev)
                lerp_v = wp.zeros(N, dtype=wp.vec3, device=dev)
                smooth = wp.zeros(N, dtype=wp.float32, device=dev)
                fr = wp.zeros(N, dtype=wp.float32, device=dev)
                ang = wp.zeros(N, dtype=wp.float32, device=dev)
                wp.launch(k, dim=N,
                          inputs=[wp.array(an, dtype=wp.float32, device=dev),
                                  wp.array(bn, dtype=wp.float32, device=dev),
                                  wp.array(van, dtype=wp.vec3, device=dev),
                                  wp.array(vbn, dtype=wp.vec3, device=dev)],
                          outputs=[lerp_s, lerp_v, smooth, fr, ang], device=dev)
                outs[dev] = (lerp_s.numpy(), lerp_v.numpy(), smooth.numpy(),
                             fr.numpy(), ang.numpy())

            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_allclose(c, m, rtol=1e-5, atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_random_builtins_match_cpu(self):
        # The PCG streams (rand_init / randf / randi / randu) are pure
        # uint32 arithmetic and must be BIT-exact against the CPU backend.
        # Range-scaled randf and randn go through float scaling and
        # transcendentals, where FMA contraction / libm ulp differences are
        # expected — those get a small tolerance instead.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(seed: wp.array(dtype=wp.int32),
                  out_f: wp.array(dtype=wp.float32), out_fr: wp.array(dtype=wp.float32),
                  out_i: wp.array(dtype=wp.int32), out_ir: wp.array(dtype=wp.int32),
                  out_u: wp.array(dtype=wp.uint32), out_n: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                state = wp.rand_init(seed[tid], tid)
                out_f[tid] = wp.randf(state)
                out_fr[tid] = wp.randf(state, -2.0, 3.0)
                out_i[tid] = wp.randi(state)
                out_ir[tid] = wp.randi(state, -10, 50)
                out_u[tid] = wp.randu(state)
                out_n[tid] = wp.randn(state)

            N = 4096
            seeds = np.full(N, 42, dtype=np.int32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                arrs = [wp.zeros(N, dtype=dt, device=dev)
                        for dt in (wp.float32, wp.float32, wp.int32,
                                   wp.int32, wp.uint32, wp.float32)]
                wp.launch(k, dim=N,
                          inputs=[wp.array(seeds, dtype=wp.int32, device=dev)],
                          outputs=arrs, device=dev)
                outs[dev] = [a.numpy() for a in arrs]

            c, m = outs['cpu'], outs['metal:0']
            np.testing.assert_array_equal(c[0], m[0])  # randf: bit-exact
            np.testing.assert_array_equal(c[2], m[2])  # randi: bit-exact
            np.testing.assert_array_equal(c[3], m[3])  # randi(lo, hi): bit-exact
            np.testing.assert_array_equal(c[4], m[4])  # randu: bit-exact
            np.testing.assert_allclose(c[1], m[1], rtol=1e-6, atol=1e-6)  # randf(lo, hi)
            np.testing.assert_allclose(c[5], m[5], rtol=1e-5, atol=1e-6)  # randn
            # Sanity: the stream is actually random, not zeros.
            assert np.std(c[0]) > 0.2, 'suspiciously uniform randf output'
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_outer_skew_trace_match_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3),
                  outer: wp.array(dtype=wp.mat33), skew: wp.array(dtype=wp.mat33),
                  tr: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                outer[tid] = wp.outer(a[tid], b[tid])
                skew[tid] = wp.skew(a[tid])
                tr[tid] = wp.trace(wp.outer(a[tid], b[tid]))

            N = 64
            rng = np.random.default_rng(6)
            an = rng.standard_normal((N, 3)).astype(np.float32)
            bn = rng.standard_normal((N, 3)).astype(np.float32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                outer = wp.zeros(N, dtype=wp.mat33, device=dev)
                skew = wp.zeros(N, dtype=wp.mat33, device=dev)
                tr = wp.zeros(N, dtype=wp.float32, device=dev)
                wp.launch(k, dim=N,
                          inputs=[wp.array(an, dtype=wp.vec3, device=dev),
                                  wp.array(bn, dtype=wp.vec3, device=dev)],
                          outputs=[outer, skew, tr], device=dev)
                outs[dev] = (outer.numpy(), skew.numpy(), tr.numpy())

            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_allclose(c, m, rtol=1e-5, atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalScalarBuiltinFixes(unittest.TestCase):
    """A/B tests for scalar builtins whose naive MSL mapping is wrong.

    ``metal::sign(0) == 0`` but ``wp.sign(0) == 1``; MSL ``%`` rejects
    float operands (``wp.mod`` needs ``metal::fmod``); ``metal::sign``
    on int is ambiguous; ``wp.step`` is the reverse of MSL's ``step``;
    MSL has no ``cbrt``.
    """

    def test_sign_step_nonzero_match_cpu(self):
        # Inputs deliberately include +0.0 and -0.0 — the exact values
        # where metal::sign diverges from Warp's semantics.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(x: wp.array(dtype=wp.float32), xi: wp.array(dtype=wp.int32),
                  s: wp.array(dtype=wp.float32), st: wp.array(dtype=wp.float32),
                  nz: wp.array(dtype=wp.float32), si: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                s[tid] = wp.sign(x[tid])
                st[tid] = wp.step(x[tid])
                nz[tid] = wp.nonzero(x[tid])
                si[tid] = wp.sign(xi[tid])

            xn = np.array([0.0, -0.0, 1.5, -1.5, 1e-30, -1e-30, 100.0, -100.0], dtype=np.float32)
            xin = np.array([0, 1, -1, 50, -50, 2147483647, -2147483648, 3], dtype=np.int32)
            N = len(xn)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                s = wp.zeros(N, dtype=wp.float32, device=dev)
                st = wp.zeros(N, dtype=wp.float32, device=dev)
                nz = wp.zeros(N, dtype=wp.float32, device=dev)
                si = wp.zeros(N, dtype=wp.int32, device=dev)
                wp.launch(k, dim=N,
                          inputs=[wp.array(xn, dtype=wp.float32, device=dev),
                                  wp.array(xin, dtype=wp.int32, device=dev)],
                          outputs=[s, st, nz, si], device=dev)
                outs[dev] = (s.numpy(), st.numpy(), nz.numpy(), si.numpy())

            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_array_equal(c, m)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_mod_cbrt_int_minmax_match_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32),
                  xi: wp.array(dtype=wp.int32),
                  fm: wp.array(dtype=wp.float32), im: wp.array(dtype=wp.int32),
                  cb: wp.array(dtype=wp.float32), mm: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                fm[tid] = wp.mod(x[tid], wp.abs(y[tid]) + 0.5)
                im[tid] = wp.mod(xi[tid], 7)
                cb[tid] = wp.cbrt(x[tid])
                mm[tid] = wp.min(xi[tid], 3) + wp.max(xi[tid], -3) + wp.clamp(xi[tid], -50, 50)

            N = 128
            rng = np.random.default_rng(11)
            xn = (rng.standard_normal(N) * 10.0).astype(np.float32)
            yn = rng.standard_normal(N).astype(np.float32)
            xin = rng.integers(-100, 100, N).astype(np.int32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                fm = wp.zeros(N, dtype=wp.float32, device=dev)
                im = wp.zeros(N, dtype=wp.int32, device=dev)
                cb = wp.zeros(N, dtype=wp.float32, device=dev)
                mm = wp.zeros(N, dtype=wp.int32, device=dev)
                wp.launch(k, dim=N,
                          inputs=[wp.array(xn, dtype=wp.float32, device=dev),
                                  wp.array(yn, dtype=wp.float32, device=dev),
                                  wp.array(xin, dtype=wp.int32, device=dev)],
                          outputs=[fm, im, cb, mm], device=dev)
                outs[dev] = (fm.numpy(), im.numpy(), cb.numpy(), mm.numpy())

            np.testing.assert_allclose(outs['cpu'][0], outs['metal:0'][0], rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(outs['cpu'][1], outs['metal:0'][1])
            np.testing.assert_allclose(outs['cpu'][2], outs['metal:0'][2], rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(outs['cpu'][3], outs['metal:0'][3])
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalMatrixBuiltins(unittest.TestCase):
    """A/B tests for matrix builtins: inverse, get_diag, component-wise ops."""

    def test_matrix_inverse_match_cpu(self):
        # The last matrix of each batch is exactly singular — Warp's
        # native inverse returns the zero matrix there (kEps == 0), and
        # the Metal port must reproduce that, not inf/NaN.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k2(a: wp.array(dtype=wp.mat22), out: wp.array(dtype=wp.mat22)):
                tid = wp.tid()
                out[tid] = wp.inverse(a[tid])

            @wp.kernel
            def k3(a: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.mat33)):
                tid = wp.tid()
                out[tid] = wp.inverse(a[tid])

            @wp.kernel
            def k4(a: wp.array(dtype=wp.mat44), out: wp.array(dtype=wp.mat44)):
                tid = wp.tid()
                out[tid] = wp.inverse(a[tid])

            N = 64
            rng = np.random.default_rng(3)
            for n, kern, mtype in ((2, k2, wp.mat22), (3, k3, wp.mat33), (4, k4, wp.mat44)):
                a = rng.standard_normal((N, n, n)).astype(np.float32)
                a += (n + 1.0) * np.eye(n, dtype=np.float32)
                a[-1] = 0.0  # exactly singular
                outs = {}
                for dev in ('cpu', 'metal:0'):
                    out = wp.zeros(N, dtype=mtype, device=dev)
                    wp.launch(kern, dim=N,
                              inputs=[wp.array(a, dtype=mtype, device=dev)],
                              outputs=[out], device=dev)
                    outs[dev] = out.numpy()
                # 4x4 native accumulates in double; the Metal port stays in
                # float32, so allow a slightly looser relative tolerance.
                rtol = 1e-5 if n < 4 else 1e-4
                np.testing.assert_allclose(outs['cpu'], outs['metal:0'], rtol=rtol, atol=1e-6)
                assert np.all(outs['metal:0'][-1] == 0.0), "singular input must invert to the zero matrix"
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_matrix_cw_ops_and_get_diag_match_cpu(self):
        # Regression: cw_mul/cw_div on matrices used to lower to MSL's
        # ``*`` / ``/``, which is a matrix multiply — silently wrong values.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.mat33), b: wp.array(dtype=wp.mat33),
                  cm: wp.array(dtype=wp.mat33), cd: wp.array(dtype=wp.mat33),
                  dg: wp.array(dtype=wp.vec3)):
                tid = wp.tid()
                cm[tid] = wp.cw_mul(a[tid], b[tid])
                cd[tid] = wp.cw_div(a[tid], b[tid])
                dg[tid] = wp.get_diag(a[tid])

            N = 64
            rng = np.random.default_rng(7)
            an = rng.standard_normal((N, 3, 3)).astype(np.float32)
            bn = (np.abs(rng.standard_normal((N, 3, 3))) + 0.5).astype(np.float32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                cm = wp.zeros(N, dtype=wp.mat33, device=dev)
                cd = wp.zeros(N, dtype=wp.mat33, device=dev)
                dg = wp.zeros(N, dtype=wp.vec3, device=dev)
                wp.launch(k, dim=N,
                          inputs=[wp.array(an, dtype=wp.mat33, device=dev),
                                  wp.array(bn, dtype=wp.mat33, device=dev)],
                          outputs=[cm, cd, dg], device=dev)
                outs[dev] = (cm.numpy(), cd.numpy(), dg.numpy())

            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_allclose(c, m, rtol=1e-5, atol=1e-6)
            # Belt and braces: the element-wise product must differ from
            # the matrix product (this is what the old codegen emitted).
            matmul = np.einsum('nij,njk->nik', an, bn)
            assert not np.allclose(outs['metal:0'][0], matmul, atol=1e-3)
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalDecompositions(unittest.TestCase):
    """A/B tests for the svd.h family: svd3, svd2, qr3, eig3."""

    def test_svd3_qr3_svd2_match_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def ksvd3(a: wp.array(dtype=wp.mat33), u: wp.array(dtype=wp.mat33),
                      s: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.mat33)):
                tid = wp.tid()
                U = wp.mat33()
                S = wp.vec3()
                V = wp.mat33()
                wp.svd3(a[tid], U, S, V)
                u[tid] = U
                s[tid] = S
                v[tid] = V

            @wp.kernel
            def kqr3(a: wp.array(dtype=wp.mat33), q: wp.array(dtype=wp.mat33),
                     r: wp.array(dtype=wp.mat33)):
                tid = wp.tid()
                Q = wp.mat33()
                R = wp.mat33()
                wp.qr3(a[tid], Q, R)
                q[tid] = Q
                r[tid] = R

            @wp.kernel
            def ksvd2(a: wp.array(dtype=wp.mat22), u: wp.array(dtype=wp.mat22),
                      s: wp.array(dtype=wp.vec2), v: wp.array(dtype=wp.mat22)):
                tid = wp.tid()
                U = wp.mat22()
                S = wp.vec2()
                V = wp.mat22()
                wp.svd2(a[tid], U, S, V)
                u[tid] = U
                s[tid] = S
                v[tid] = V

            N = 64
            rng = np.random.default_rng(13)
            a3 = rng.standard_normal((N, 3, 3)).astype(np.float32)
            a2 = rng.standard_normal((N, 2, 2)).astype(np.float32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                u = wp.zeros(N, dtype=wp.mat33, device=dev)
                s = wp.zeros(N, dtype=wp.vec3, device=dev)
                v = wp.zeros(N, dtype=wp.mat33, device=dev)
                wp.launch(ksvd3, dim=N, inputs=[wp.array(a3, dtype=wp.mat33, device=dev)],
                          outputs=[u, s, v], device=dev)
                outs[dev] = (u.numpy(), s.numpy(), v.numpy())
            # The Jacobi/QR iteration accumulates float drift; both
            # backends run the identical algorithm, so 1e-4 is ample.
            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_allclose(c, m, rtol=1e-4, atol=1e-4)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                q = wp.zeros(N, dtype=wp.mat33, device=dev)
                r = wp.zeros(N, dtype=wp.mat33, device=dev)
                wp.launch(kqr3, dim=N, inputs=[wp.array(a3, dtype=wp.mat33, device=dev)],
                          outputs=[q, r], device=dev)
                outs[dev] = (q.numpy(), r.numpy())
            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_allclose(c, m, rtol=1e-4, atol=1e-4)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                u = wp.zeros(N, dtype=wp.mat22, device=dev)
                s = wp.zeros(N, dtype=wp.vec2, device=dev)
                v = wp.zeros(N, dtype=wp.mat22, device=dev)
                wp.launch(ksvd2, dim=N, inputs=[wp.array(a2, dtype=wp.mat22, device=dev)],
                          outputs=[u, s, v], device=dev)
                outs[dev] = (u.numpy(), s.numpy(), v.numpy())
            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_allclose(c, m, rtol=1e-4, atol=1e-4)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=60)

    def test_eig3_matches_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.mat33), q: wp.array(dtype=wp.mat33),
                  d: wp.array(dtype=wp.vec3)):
                tid = wp.tid()
                Q = wp.mat33()
                dd = wp.vec3()
                wp.eig3(a[tid] + wp.transpose(a[tid]), Q, dd)
                q[tid] = Q
                d[tid] = dd

            N = 64
            rng = np.random.default_rng(17)
            an = rng.standard_normal((N, 3, 3)).astype(np.float32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                q = wp.zeros(N, dtype=wp.mat33, device=dev)
                d = wp.zeros(N, dtype=wp.vec3, device=dev)
                wp.launch(k, dim=N, inputs=[wp.array(an, dtype=wp.mat33, device=dev)],
                          outputs=[q, d], device=dev)
                outs[dev] = (q.numpy(), d.numpy())

            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_allclose(c, m, rtol=1e-4, atol=1e-4)
            # Validate the Metal result is a real eigendecomposition, not
            # merely bit-similar to the CPU: A @ Q == Q @ diag(d). Warp's
            # fixed 4-sweep Jacobi leaves residuals up to ~1.2e-3 on this
            # data (identical on the CPU backend), so gate at 1e-2.
            qm, dm = outs['metal:0']
            for i in range(N):
                As = (an[i] + an[i].T).astype(np.float64)
                resid = As @ qm[i] - qm[i] @ np.diag(dm[i])
                assert np.abs(resid).max() < 1e-2, f"eig residual too large at {i}"
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=60)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalArrayBuiltins(unittest.TestCase):
    """A/B tests for array-level builtins (lower_bound) and out-param quat helpers."""

    def test_lower_bound_matches_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(arr: wp.array(dtype=wp.float32), vals: wp.array(dtype=wp.float32),
                  out2: wp.array(dtype=wp.int32), out4: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                out2[tid] = wp.lower_bound(arr, vals[tid])
                out4[tid] = wp.lower_bound(arr, 8, 100, vals[tid])

            N = 128
            rng = np.random.default_rng(23)
            arr_n = np.sort(rng.standard_normal(N).astype(np.float32))
            vals_n = rng.standard_normal(N).astype(np.float32)
            # Include exact-match probes: lower_bound's boundary behaviour
            # (first index not less than value) must agree bit-exactly.
            vals_n[:16] = arr_n[rng.integers(0, N, 16)]

            outs = {}
            for dev in ('cpu', 'metal:0'):
                out2 = wp.zeros(N, dtype=wp.int32, device=dev)
                out4 = wp.zeros(N, dtype=wp.int32, device=dev)
                wp.launch(k, dim=N,
                          inputs=[wp.array(arr_n, dtype=wp.float32, device=dev),
                                  wp.array(vals_n, dtype=wp.float32, device=dev)],
                          outputs=[out2, out4], device=dev)
                outs[dev] = (out2.numpy(), out4.numpy())

            np.testing.assert_array_equal(outs['cpu'][0], outs['metal:0'][0])
            np.testing.assert_array_equal(outs['cpu'][1], outs['metal:0'][1])
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_quat_to_axis_angle_matches_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(q: wp.array(dtype=wp.quat), axis: wp.array(dtype=wp.vec3),
                  angle: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                qq = wp.normalize(q[tid])
                ax = wp.vec3()
                ang = float(0.0)
                wp.quat_to_axis_angle(qq, ax, ang)
                axis[tid] = ax
                angle[tid] = ang

            N = 128
            rng = np.random.default_rng(29)
            qn = rng.standard_normal((N, 4)).astype(np.float32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                axis = wp.zeros(N, dtype=wp.vec3, device=dev)
                angle = wp.zeros(N, dtype=wp.float32, device=dev)
                wp.launch(k, dim=N, inputs=[wp.array(qn, dtype=wp.quat, device=dev)],
                          outputs=[axis, angle], device=dev)
                outs[dev] = (axis.numpy(), angle.numpy())

            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_allclose(c, m, rtol=1e-5, atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalSpecialValueBuiltins(unittest.TestCase):
    """isfinite/isnan/isinf must reduce composite types to a scalar bool.

    ``metal::isfinite(float3)`` returns ``bool3`` — assigning that to a
    Warp bool is a compile error (and would be silently wrong if it ever
    type-coerced). The wp_is* helpers reduce with all()/any() to match
    warp/native/vec.h semantics: all-finite / any-nan / any-inf.
    """

    def test_special_values_match_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(xs: wp.array(dtype=wp.float32), vs: wp.array(dtype=wp.vec3),
                  ms: wp.array(dtype=wp.mat33), qs: wp.array(dtype=wp.quat),
                  out: wp.array(dtype=wp.int32)):
                tid = wp.tid()
                r = int(0)
                if wp.isfinite(xs[tid]):
                    r += 1
                if wp.isnan(xs[tid]):
                    r += 2
                if wp.isinf(xs[tid]):
                    r += 4
                if wp.isfinite(vs[tid]):
                    r += 8
                if wp.isnan(vs[tid]):
                    r += 16
                if wp.isinf(vs[tid]):
                    r += 32
                if wp.isfinite(ms[tid]):
                    r += 64
                if wp.isnan(ms[tid]):
                    r += 128
                if wp.isinf(ms[tid]):
                    r += 256
                if wp.isfinite(qs[tid]):
                    r += 512
                if wp.isnan(qs[tid]):
                    r += 1024
                out[tid] = r

            specials = np.array([1.0, -2.5, np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
            N = len(specials)
            vecs = np.tile(specials[:, None], (1, 3)).astype(np.float32)
            # Mixed vectors: one bad component among finite ones must flip
            # the whole-vector verdict (any-nan / any-inf / not-all-finite).
            vecs[0, 1] = np.nan
            vecs[1, 2] = np.inf
            mats = np.tile(specials[:, None, None], (1, 3, 3)).astype(np.float32)
            mats[0, 2, 1] = np.nan
            quats = np.tile(specials[:, None], (1, 4)).astype(np.float32)
            quats[1, 3] = np.nan

            outs = {}
            for dev in ('cpu', 'metal:0'):
                out = wp.zeros(N, dtype=wp.int32, device=dev)
                wp.launch(k, dim=N,
                          inputs=[wp.array(specials, device=dev),
                                  wp.array(vecs, dtype=wp.vec3, device=dev),
                                  wp.array(mats, dtype=wp.mat33, device=dev),
                                  wp.array(quats, dtype=wp.quat, device=dev)],
                          outputs=[out], device=dev)
                outs[dev] = out.numpy()

            np.testing.assert_array_equal(outs['cpu'], outs['metal:0'])
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalComposeAndSpatialBuiltins(unittest.TestCase):
    """matrix_from_cols/rows, spatial algebra, and bitwise invert."""

    def test_matrix_from_cols_rows_match_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(v2: wp.array(dtype=wp.vec2), v3: wp.array(dtype=wp.vec3),
                  v4: wp.array(dtype=wp.vec4),
                  oc2: wp.array(dtype=wp.mat22), or2: wp.array(dtype=wp.mat22),
                  oc3: wp.array(dtype=wp.mat33), or3: wp.array(dtype=wp.mat33),
                  oc4: wp.array(dtype=wp.mat44), or4: wp.array(dtype=wp.mat44)):
                tid = wp.tid()
                a2 = v2[tid]
                a3 = v3[tid]
                a4 = v4[tid]
                oc2[tid] = wp.matrix_from_cols(a2, a2 * 2.0)
                or2[tid] = wp.matrix_from_rows(a2, a2 * 2.0)
                oc3[tid] = wp.matrix_from_cols(a3, a3 * 2.0, a3 - wp.vec3(1.0))
                or3[tid] = wp.matrix_from_rows(a3, a3 * 2.0, a3 - wp.vec3(1.0))
                oc4[tid] = wp.matrix_from_cols(a4, a4 * 2.0, a4 * 3.0, a4 - wp.vec4(1.0))
                or4[tid] = wp.matrix_from_rows(a4, a4 * 2.0, a4 * 3.0, a4 - wp.vec4(1.0))

            N = 64
            rng = np.random.default_rng(31)
            v2n = rng.standard_normal((N, 2)).astype(np.float32)
            v3n = rng.standard_normal((N, 3)).astype(np.float32)
            v4n = rng.standard_normal((N, 4)).astype(np.float32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                os_ = [wp.zeros(N, dtype=dt, device=dev)
                       for dt in (wp.mat22, wp.mat22, wp.mat33, wp.mat33, wp.mat44, wp.mat44)]
                wp.launch(k, dim=N,
                          inputs=[wp.array(v2n, dtype=wp.vec2, device=dev),
                                  wp.array(v3n, dtype=wp.vec3, device=dev),
                                  wp.array(v4n, dtype=wp.vec4, device=dev)],
                          outputs=os_, device=dev)
                outs[dev] = [o.numpy() for o in os_]

            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_array_equal(c, m)
            # from_cols(v0..) must place v0 as column 0 (guards against a
            # silent cols/rows swap that A/B alone wouldn't catch if both
            # backends made the same mistake).
            np.testing.assert_array_equal(outs['cpu'][2][:, :, 0], v3n)
            np.testing.assert_array_equal(outs['cpu'][3][:, 0, :], v3n)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_spatial_algebra_matches_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.spatial_vector), b: wp.array(dtype=wp.spatial_vector),
                  oc: wp.array(dtype=wp.spatial_vector), od: wp.array(dtype=wp.spatial_vector),
                  odot: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                oc[tid] = wp.spatial_cross(a[tid], b[tid])
                od[tid] = wp.spatial_cross_dual(a[tid], b[tid])
                odot[tid] = wp.spatial_dot(a[tid], b[tid])

            N = 128
            rng = np.random.default_rng(37)
            an = rng.standard_normal((N, 6)).astype(np.float32)
            bn = rng.standard_normal((N, 6)).astype(np.float32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                oc = wp.zeros(N, dtype=wp.spatial_vector, device=dev)
                od = wp.zeros(N, dtype=wp.spatial_vector, device=dev)
                odot = wp.zeros(N, dtype=wp.float32, device=dev)
                wp.launch(k, dim=N,
                          inputs=[wp.array(an, dtype=wp.spatial_vector, device=dev),
                                  wp.array(bn, dtype=wp.spatial_vector, device=dev)],
                          outputs=[oc, od, odot], device=dev)
                outs[dev] = (oc.numpy(), od.numpy(), odot.numpy())

            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_allclose(m, c, rtol=1e-5, atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_invert_matches_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(x: wp.array(dtype=wp.int32), u: wp.array(dtype=wp.uint32),
                  ox: wp.array(dtype=wp.int32), ou: wp.array(dtype=wp.uint32)):
                tid = wp.tid()
                ox[tid] = wp.invert(x[tid])
                ou[tid] = wp.invert(u[tid])

            N = 64
            rng = np.random.default_rng(41)
            xn = rng.integers(-(2**31), 2**31 - 1, N).astype(np.int32)
            un = rng.integers(0, 2**32 - 1, N).astype(np.uint32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                ox = wp.zeros(N, dtype=wp.int32, device=dev)
                ou = wp.zeros(N, dtype=wp.uint32, device=dev)
                wp.launch(k, dim=N,
                          inputs=[wp.array(xn, device=dev), wp.array(un, device=dev)],
                          outputs=[ox, ou], device=dev)
                outs[dev] = (ox.numpy(), ou.numpy())

            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_array_equal(c, m)
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalAtomicMinMax(unittest.TestCase):
    """Float atomic_min/max have no MSL fetch op — they run through a
    compare-exchange loop on the reinterpreted bit pattern (wp_atomic_min /
    wp_atomic_max helpers). Both emission paths are covered: the intrinsic
    regex (scalar arrays) and the multi-dim AST fold's raw per-component
    lines (vec-typed arrays)."""

    def test_atomic_minmax_float_contended(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(x: wp.array(dtype=wp.float32), lo: wp.array(dtype=wp.float32),
                  hi: wp.array(dtype=wp.float32)):
                tid = wp.tid()
                # All threads contend on slot 0; per-thread slots alongside.
                wp.atomic_min(lo, 0, x[tid])
                wp.atomic_max(hi, 0, x[tid])
                wp.atomic_min(lo, tid + 1, x[tid] * 0.5)
                wp.atomic_max(hi, tid + 1, x[tid] * 0.5)

            N = 1024
            rng = np.random.default_rng(43)
            # Mix of signs and magnitudes — the CAS loop compares as float,
            # so negative values must order correctly (a raw uint-bits
            # comparison would get them backwards).
            xn = np.concatenate([rng.standard_normal(N - 2) * 100.0,
                                 [-1e30, 1e30]]).astype(np.float32)
            rng.shuffle(xn)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                lo = wp.array(np.full(N + 1, 1e38, dtype=np.float32), device=dev)
                hi = wp.array(np.full(N + 1, -1e38, dtype=np.float32), device=dev)
                wp.launch(k, dim=N, inputs=[wp.array(xn, device=dev), lo, hi], device=dev)
                outs[dev] = (lo.numpy(), hi.numpy())

            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_array_equal(c, m)
            # Independent ground truth for the contended slot.
            np.testing.assert_allclose(outs['metal:0'][0][0], xn.min(), rtol=0)
            np.testing.assert_allclose(outs['metal:0'][1][0], xn.max(), rtol=0)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_atomic_minmax_int_and_2d(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(x: wp.array(dtype=wp.int32), lo: wp.array(dtype=wp.int32),
                  hi: wp.array(dtype=wp.int32), grid: wp.array2d(dtype=wp.float32)):
                tid = wp.tid()
                wp.atomic_min(lo, 0, x[tid])
                wp.atomic_max(hi, 0, x[tid])
                # 4-arg multi-dim form — flattened by the AST fold, then the
                # intrinsic regex routes it through the float helper.
                wp.atomic_min(grid, tid % 4, tid % 8, wp.float32(x[tid]))

            N = 512
            rng = np.random.default_rng(47)
            xn = rng.integers(-10**9, 10**9, N).astype(np.int32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                lo = wp.array(np.array([2**31 - 1], dtype=np.int32), device=dev)
                hi = wp.array(np.array([-(2**31)], dtype=np.int32), device=dev)
                grid = wp.array(np.full((4, 8), 1e38, dtype=np.float32), device=dev)
                wp.launch(k, dim=N, inputs=[wp.array(xn, device=dev), lo, hi, grid], device=dev)
                outs[dev] = (lo.numpy(), hi.numpy(), grid.numpy())

            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_array_equal(c, m)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_atomic_minmax_vec3_components(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(x: wp.array(dtype=wp.vec3), lo: wp.array(dtype=wp.vec3),
                  hi: wp.array(dtype=wp.vec3)):
                tid = wp.tid()
                wp.atomic_min(lo, 0, x[tid])
                wp.atomic_max(hi, 0, x[tid])

            N = 512
            rng = np.random.default_rng(53)
            xn = (rng.standard_normal((N, 3)) * 50.0).astype(np.float32)

            outs = {}
            for dev in ('cpu', 'metal:0'):
                lo = wp.array(np.full((1, 3), 1e38, dtype=np.float32), dtype=wp.vec3, device=dev)
                hi = wp.array(np.full((1, 3), -1e38, dtype=np.float32), dtype=wp.vec3, device=dev)
                wp.launch(k, dim=N, inputs=[wp.array(xn, dtype=wp.vec3, device=dev), lo, hi], device=dev)
                outs[dev] = (lo.numpy(), hi.numpy())

            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_array_equal(c, m)
            np.testing.assert_allclose(outs['metal:0'][0][0], xn.min(axis=0), rtol=0)
            np.testing.assert_allclose(outs['metal:0'][1][0], xn.max(axis=0), rtol=0)
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalComponentAssign(unittest.TestCase):
    """Indexed element assignment on matrix/vector locals.

    ``m[r, c] = v`` lowers to the 4-arg ``wp::assign_inplace(m, r, c, v)``.
    A value-group regex that admits commas turns that into the comma
    expression ``m[r] = c, v`` — which compiles (scalar broadcast onto an
    MSL matrix column) and silently discards the value. The translation
    must route through ``wp_mat_elem_store`` (MSL forbids references to
    vector elements, so it cannot be a reference-returning accessor).
    """

    def test_component_assign_matches_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(out: wp.array(dtype=wp.mat33), outv: wp.array(dtype=wp.vec3),
                  out22: wp.array(dtype=wp.mat22)):
                i = wp.tid()
                m = wp.mat33(0.0)
                m[0, 0] = float(i)
                m[1, 2] = 5.0
                m[2, 1] = -3.0
                m[0, 2] += 2.5   # compound op through read-modify-write
                m[1, 2] *= 2.0
                out[i] = m
                v = wp.vec3()
                v[0] = float(i) * 2.0
                v[2] = 7.0
                outv[i] = v
                m2 = wp.mat22(1.0)
                m2[1, 0] = -4.0
                m2[0, 1] -= float(i)
                out22[i] = m2

            N = 32
            outs = {}
            for dev in ('cpu', 'metal:0'):
                args = [wp.zeros(N, dtype=wp.mat33, device=dev),
                        wp.zeros(N, dtype=wp.vec3, device=dev),
                        wp.zeros(N, dtype=wp.mat22, device=dev)]
                wp.launch(k, dim=N, inputs=args, device=dev)
                outs[dev] = [a.numpy() for a in args]

            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_array_equal(m, c)
            # Independent ground truth: the write must land at [row, col],
            # not transposed and not broadcast over a column.
            np.testing.assert_array_equal(outs['metal:0'][0][:, 1, 2], 10.0)
            np.testing.assert_array_equal(outs['metal:0'][0][:, 2, 1], -3.0)
            np.testing.assert_array_equal(outs['metal:0'][0][:, 0, 2], 2.5)
            np.testing.assert_array_equal(outs['metal:0'][0][:, 0, 0], np.arange(N, dtype=np.float32))
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalStructSupport(unittest.TestCase):
    """Struct args, arrays of structs, and nested structs.

    Struct storage on Metal is a flat float32 *bitcast* of the ctypes
    bytes, so int/uint components must round-trip through ``as_type<>``
    casts — reading an int field as a float value-converts a denormal
    bit pattern to 0 (the original struct-param bug: an int ``count``
    field silently read as 0, zeroing every term it multiplied).
    """

    def test_struct_param_with_int_field(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.struct
            class Params:
                scale: float
                offset: wp.vec3
                count: int

            @wp.kernel
            def k(p: Params, x: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
                i = wp.tid()
                out[i] = x[i] * p.scale + p.offset * float(p.count)

            N = 32
            xn = np.random.default_rng(3).standard_normal((N, 3)).astype(np.float32)
            outs = {}
            for dev in ('cpu', 'metal:0'):
                p = Params()
                p.scale = 2.5
                p.offset = wp.vec3(1.0, -2.0, 3.0)
                p.count = 3
                x = wp.array(xn, dtype=wp.vec3, device=dev)
                out = wp.zeros(N, dtype=wp.vec3, device=dev)
                wp.launch(k, dim=N, inputs=[p, x, out], device=dev)
                outs[dev] = out.numpy()
            np.testing.assert_array_equal(outs['metal:0'], outs['cpu'])
            # The int field must contribute: term is offset * 3, not 0.
            np.testing.assert_allclose(outs['metal:0'][0], xn[0] * 2.5 + np.array([3.0, -6.0, 9.0]), rtol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_struct_array_roundtrip(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.struct
            class Item:
                a: float
                v: wp.vec3
                n: int

            @wp.kernel
            def k(items: wp.array(dtype=Item), out: wp.array(dtype=float)):
                i = wp.tid()
                s = Item()
                s.a = float(i) * 0.5
                s.v = wp.vec3(float(i), 1.0, -1.0)
                s.n = i * 1000 + 7
                items[i] = s
                t = items[i]
                out[i] = t.a + t.v[0] * 2.0 + float(t.n)

            N = 32
            outs = {}
            for dev in ('cpu', 'metal:0'):
                items = wp.zeros(N, dtype=Item, device=dev)
                out = wp.zeros(N, dtype=float, device=dev)
                wp.launch(k, dim=N, inputs=[items, out], device=dev)
                outs[dev] = (items.numpy(), out.numpy())
            for f in outs['cpu'][0].dtype.names:
                np.testing.assert_array_equal(outs['metal:0'][0][f], outs['cpu'][0][f])
            np.testing.assert_array_equal(outs['metal:0'][1], outs['cpu'][1])
            # Bitcast ground truth: the int field must hold i*1000+7 exactly
            # (a value round-trip through float32 would corrupt large ints).
            np.testing.assert_array_equal(outs['metal:0'][0]['n'], np.arange(N) * 1000 + 7)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_nested_struct(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.struct
            class Inner:
                w: wp.vec2
                k: float

            @wp.struct
            class Outer:
                inner: Inner
                bias: float

            @wp.kernel
            def kern(o: Outer, out: wp.array(dtype=float)):
                i = wp.tid()
                t = o.inner  # whole nested-struct load into a local
                out[i] = o.inner.w[0] * o.inner.k + o.inner.w[1] + o.bias * float(i) + t.k

            N = 32
            outs = {}
            for dev in ('cpu', 'metal:0'):
                o = Outer()
                inner = Inner()
                inner.w = wp.vec2(3.0, -4.0)
                inner.k = 2.0
                o.inner = inner
                o.bias = 0.25
                out = wp.zeros(N, dtype=float, device=dev)
                wp.launch(kern, dim=N, inputs=[o, out], device=dev)
                outs[dev] = out.numpy()
            np.testing.assert_array_equal(outs['metal:0'], outs['cpu'])
            np.testing.assert_allclose(outs['metal:0'][4], 3.0 * 2.0 - 4.0 + 0.25 * 4.0 + 2.0, rtol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalNonSquareMatAndDdot(unittest.TestCase):
    """Non-square matrices route through row-major ``wp_matRxC`` structs.

    The array load/store paths must NOT use the native column-major
    ``value[c][r]`` convention for them — that transposes the payload.
    """

    def test_mat23_and_vec5_roundtrip(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            vec5 = wp.types.vector(length=5, dtype=float)
            mat23 = wp.types.matrix(shape=(2, 3), dtype=float)

            @wp.kernel
            def k(out5: wp.array(dtype=vec5), out23: wp.array(dtype=mat23), outf: wp.array(dtype=float)):
                i = wp.tid()
                v = vec5(1.0, 2.0, 3.0, 4.0, float(i))
                out5[i] = v * 2.0
                m = mat23(1.0, 2.0, 3.0, 4.0, 5.0, float(i))
                m[0, 1] = 20.0
                out23[i] = m
                back = out23[i]  # exercise the big-mat array *load* path
                outf[i] = wp.length_sq(v) + back[1, 2] + back[0, 1]

            N = 16
            outs = {}
            for dev in ('cpu', 'metal:0'):
                args = [wp.zeros(N, dtype=vec5, device=dev),
                        wp.zeros(N, dtype=mat23, device=dev),
                        wp.zeros(N, dtype=float, device=dev)]
                wp.launch(k, dim=N, inputs=args, device=dev)
                outs[dev] = [a.numpy() for a in args]
            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_array_equal(m, c)
            # Row-major ground truth: element (1, 0) is 4.0, NOT the
            # transposed 2.0.
            np.testing.assert_array_equal(outs['metal:0'][1][:, 1, 0], 4.0)
            np.testing.assert_array_equal(outs['metal:0'][1][:, 0, 1], 20.0)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_ddot_matches_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
                i = wp.tid()
                m1 = wp.outer(a[i], b[i])
                out[i] = wp.ddot(m1, wp.transpose(m1)) + wp.trace(m1)

            N = 64
            rng = np.random.default_rng(11)
            an = rng.standard_normal((N, 3)).astype(np.float32)
            bn = rng.standard_normal((N, 3)).astype(np.float32)
            outs = {}
            for dev in ('cpu', 'metal:0'):
                out = wp.zeros(N, dtype=float, device=dev)
                wp.launch(k, dim=N, inputs=[wp.array(an, dtype=wp.vec3, device=dev),
                                            wp.array(bn, dtype=wp.vec3, device=dev), out], device=dev)
                outs[dev] = out.numpy()
            np.testing.assert_allclose(outs['metal:0'], outs['cpu'], rtol=1e-5, atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalNoiseAndSampling(unittest.TestCase):
    """Perlin/curl noise and geometric sampling — ports of noise.h/rand.h.

    The RNG streams are bit-identical to CPU, so outputs must match to
    float rounding (the noise lattice interpolation may reassociate).
    """

    def test_sampling_matches_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(seed: int, outs: wp.array(dtype=wp.vec3), outd: wp.array(dtype=wp.vec2),
                  outt: wp.array(dtype=wp.vec2), outh: wp.array(dtype=wp.vec3)):
                i = wp.tid()
                state = wp.rand_init(seed, i)
                outs[i] = wp.sample_unit_sphere(state)
                outd[i] = wp.sample_unit_disk(state)
                outt[i] = wp.sample_triangle(state)
                outh[i] = wp.sample_unit_hemisphere(state)

            N = 256
            res = {}
            for dev in ('cpu', 'metal:0'):
                args = [77,
                        wp.zeros(N, dtype=wp.vec3, device=dev),
                        wp.zeros(N, dtype=wp.vec2, device=dev),
                        wp.zeros(N, dtype=wp.vec2, device=dev),
                        wp.zeros(N, dtype=wp.vec3, device=dev)]
                wp.launch(k, dim=N, inputs=args, device=dev)
                res[dev] = [a.numpy() for a in args if isinstance(a, wp.array)]
            for c, m in zip(res['cpu'], res['metal:0']):
                np.testing.assert_allclose(m, c, rtol=1e-5, atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_noise_family_matches_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(seed: int, xs: wp.array(dtype=wp.vec2), zs: wp.array(dtype=wp.vec3),
                  o1: wp.array(dtype=float), o2: wp.array(dtype=float),
                  o3: wp.array(dtype=wp.vec2), o4: wp.array(dtype=float)):
                i = wp.tid()
                state = wp.rand_init(seed)
                p = xs[i]
                o1[i] = wp.noise(state, p)
                o2[i] = wp.pnoise(state, p, 4, 4)
                o3[i] = wp.curlnoise(state, p)
                o4[i] = wp.noise(state, zs[i])

            N = 256
            rng = np.random.default_rng(5)
            xn = rng.uniform(-3, 3, (N, 2)).astype(np.float32)
            zn = rng.uniform(-3, 3, (N, 3)).astype(np.float32)
            res = {}
            for dev in ('cpu', 'metal:0'):
                args = [5,
                        wp.array(xn, dtype=wp.vec2, device=dev),
                        wp.array(zn, dtype=wp.vec3, device=dev),
                        wp.zeros(N, dtype=float, device=dev),
                        wp.zeros(N, dtype=float, device=dev),
                        wp.zeros(N, dtype=wp.vec2, device=dev),
                        wp.zeros(N, dtype=float, device=dev)]
                wp.launch(k, dim=N, inputs=args, device=dev)
                res[dev] = [a.numpy() for a in args if isinstance(a, wp.array)]
            for c, m in zip(res['cpu'][2:], res['metal:0'][2:]):
                np.testing.assert_allclose(m, c, rtol=1e-4, atol=1e-5)
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalSpatialMatrixArray(unittest.TestCase):
    """Arrays of big square matrices (``wp.spatial_matrix`` = mat66) route
    through the row-major ``wp_matRxC`` struct path, which also needs
    arithmetic operators (mat*mat, mat*vec, +, scalar scale)."""

    def test_spatial_matrix_array_ops(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(ms: wp.array(dtype=wp.spatial_matrix),
                  vs: wp.array(dtype=wp.spatial_vector),
                  out_v: wp.array(dtype=wp.spatial_vector),
                  out_m: wp.array(dtype=wp.spatial_matrix)):
                i = wp.tid()
                m = ms[i]
                out_v[i] = m * vs[i]
                mm = m * m + m * 0.5 - m
                mm[0, 5] = 42.0
                out_m[i] = mm

            N = 32
            rng = np.random.default_rng(7)
            ms_np = rng.standard_normal((N, 6, 6)).astype(np.float32)
            vs_np = rng.standard_normal((N, 6)).astype(np.float32)
            outs = {}
            for dev in ('cpu', 'metal:0'):
                out_v = wp.zeros(N, dtype=wp.spatial_vector, device=dev)
                out_m = wp.zeros(N, dtype=wp.spatial_matrix, device=dev)
                wp.launch(k, dim=N, inputs=[wp.array(ms_np, dtype=wp.spatial_matrix, device=dev),
                                            wp.array(vs_np, dtype=wp.spatial_vector, device=dev),
                                            out_v, out_m], device=dev)
                outs[dev] = (out_v.numpy(), out_m.numpy())
            np.testing.assert_allclose(outs['metal:0'][0], outs['cpu'][0], rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(outs['metal:0'][1], outs['cpu'][1], rtol=1e-4, atol=1e-4)
            assert np.all(outs['metal:0'][1][:, 0, 5] == 42.0)
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalErfAndSampleCdf(unittest.TestCase):
    """erf/erfc/erfinv/erfcinv are MSL ports (single-precision
    approximations, so float32-level agreement, not bit equality);
    sample_cdf shares the CPU's exact PCG stream + binary search."""

    def test_erf_family_matches_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(x: wp.array(dtype=float), y: wp.array(dtype=float),
                  o1: wp.array(dtype=float), o2: wp.array(dtype=float),
                  o3: wp.array(dtype=float), o4: wp.array(dtype=float)):
                i = wp.tid()
                o1[i] = wp.erf(x[i])
                o2[i] = wp.erfc(x[i])
                o3[i] = wp.erfinv(y[i])
                o4[i] = wp.erfcinv(y[i] + 1.0)

            N = 256
            rng = np.random.default_rng(11)
            xn = (rng.standard_normal(N) * 1.5).astype(np.float32)
            yn = rng.uniform(-0.98, 0.98, N).astype(np.float32)
            outs = {}
            for dev in ('cpu', 'metal:0'):
                args = [wp.array(xn, dtype=float, device=dev), wp.array(yn, dtype=float, device=dev)]
                args += [wp.zeros(N, dtype=float, device=dev) for _ in range(4)]
                wp.launch(k, dim=N, inputs=args, device=dev)
                outs[dev] = [a.numpy() for a in args[2:]]
            for c, m in zip(outs['cpu'], outs['metal:0']):
                np.testing.assert_allclose(m, c, rtol=1e-4, atol=1e-5)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_sample_cdf_matches_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(cdf: wp.array(dtype=float), out: wp.array(dtype=int)):
                i = wp.tid()
                state = wp.rand_init(123, i)
                out[i] = wp.sample_cdf(state, cdf)

            rng = np.random.default_rng(3)
            w = np.abs(rng.standard_normal(16)).astype(np.float32) + 0.01
            cdf_np = np.cumsum(w / w.sum()).astype(np.float32)
            N = 256
            outs = {}
            for dev in ('cpu', 'metal:0'):
                out = wp.zeros(N, dtype=int, device=dev)
                wp.launch(k, dim=N, inputs=[wp.array(cdf_np, dtype=float, device=dev), out], device=dev)
                outs[dev] = out.numpy()
            np.testing.assert_array_equal(outs['metal:0'], outs['cpu'])
            assert len(np.unique(outs['cpu'])) > 3
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalIntersect(unittest.TestCase):
    """closest_point_edge_edge / intersect_tri_tri — ports of
    warp/native/intersect.h with macro-order dot/cross arithmetic."""

    def test_edge_edge_and_tri_tri_match_cpu(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(pts: wp.array(dtype=wp.vec3), out_c: wp.array(dtype=wp.vec3), out_i: wp.array(dtype=int)):
                i = wp.tid()
                p1 = pts[i * 6 + 0]
                q1 = pts[i * 6 + 1]
                p2 = pts[i * 6 + 2]
                q2 = pts[i * 6 + 3]
                r1 = pts[i * 6 + 4]
                r2 = pts[i * 6 + 5]
                out_c[i] = wp.closest_point_edge_edge(p1, q1, p2, q2, 1.0e-6)
                out_i[i] = wp.intersect_tri_tri(p1, q1, r1, p2, q2, r2)

            M = 128
            rng = np.random.default_rng(5)
            pts_np = rng.standard_normal((M * 6, 3)).astype(np.float32)
            outs = {}
            for dev in ('cpu', 'metal:0'):
                out_c = wp.zeros(M, dtype=wp.vec3, device=dev)
                out_i = wp.zeros(M, dtype=int, device=dev)
                wp.launch(k, dim=M, inputs=[wp.array(pts_np, dtype=wp.vec3, device=dev), out_c, out_i],
                          device=dev)
                outs[dev] = (out_c.numpy(), out_i.numpy())
            np.testing.assert_allclose(outs['metal:0'][0], outs['cpu'][0], rtol=1e-4, atol=1e-5)
            np.testing.assert_array_equal(outs['metal:0'][1], outs['cpu'][1])
            n_hit = int(outs['cpu'][1].sum())
            assert 0 < n_hit < M  # both outcomes exercised
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalEqualityAndTid4(unittest.TestCase):
    """Warp reduces vector/matrix ``==`` to ONE bool (all components
    equal); MSL yields boolN for native vectors and has no operator== on
    native matrices. 4-D wp.tid folds launch dims 2/3 into grid z."""

    def test_vec_mat_equality_reduces(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3),
                  ms: wp.array(dtype=wp.mat33), sv: wp.array(dtype=wp.spatial_vector),
                  out: wp.array(dtype=float)):
                i = wp.tid()
                r = float(0.0)
                if a[i] == b[i]:
                    r += 1.0
                if not (a[i] == b[i]):
                    r += 10.0
                m = ms[i]
                m2 = ms[i]
                if m == m2:
                    r += 100.0
                m2[1, 1] += 1.0
                if not (m == m2):
                    r += 1000.0
                if sv[i] == sv[i]:
                    r += 10000.0
                if float(i) == 3.0:  # scalar == must stay identical
                    r += 100000.0
                out[i] = r

            N = 64
            rng = np.random.default_rng(9)
            an = rng.standard_normal((N, 3)).astype(np.float32)
            bn = an.copy()
            bn[::3] += 1.0
            ms_np = an.repeat(3, axis=1).reshape(N, 3, 3).astype(np.float32)
            sv_np = np.concatenate([an, bn], axis=1).astype(np.float32)
            outs = {}
            for dev in ('cpu', 'metal:0'):
                out = wp.zeros(N, dtype=float, device=dev)
                wp.launch(k, dim=N, inputs=[wp.array(an, dtype=wp.vec3, device=dev),
                                            wp.array(bn, dtype=wp.vec3, device=dev),
                                            wp.array(ms_np, dtype=wp.mat33, device=dev),
                                            wp.array(sv_np, dtype=wp.spatial_vector, device=dev),
                                            out], device=dev)
                outs[dev] = out.numpy()
            np.testing.assert_array_equal(outs['metal:0'], outs['cpu'])
            expect = np.where(np.all(an == bn, axis=1), 1.0, 10.0) + 11100.0
            expect[3] += 100000.0
            np.testing.assert_array_equal(outs['cpu'], expect.astype(np.float32))
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_tid_4d(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(out: wp.array(dtype=float, ndim=4)):
                i, j, k_, l = wp.tid()
                out[i, j, k_, l] = float(i * 1000 + j * 100 + k_ * 10 + l)

            DIMS = (2, 3, 4, 5)
            outs = {}
            for dev in ('cpu', 'metal:0'):
                out = wp.zeros(DIMS, dtype=float, device=dev)
                wp.launch(k, dim=DIMS, inputs=[out], device=dev)
                outs[dev] = out.numpy()
            np.testing.assert_array_equal(outs['metal:0'], outs['cpu'])
            i, j, k_, l = np.meshgrid(*[np.arange(d) for d in DIMS], indexing='ij')
            np.testing.assert_array_equal(outs['cpu'], (i * 1000 + j * 100 + k_ * 10 + l).astype(np.float32))
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalArrayViews(unittest.TestCase):
    """In-kernel views (``row = a[i]``) fold onto the base array; the
    view's ``.shape[k]`` maps to the base shape shifted by the folded
    leading dims."""

    def test_row_view_shape_sum(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=float, ndim=2), a3: wp.array(dtype=float, ndim=3),
                  out: wp.array(dtype=float)):
                i = wp.tid()
                row = a[i]
                s = float(0.0)
                for j in range(row.shape[0]):
                    s += row[j]
                sub = a3[i]
                for j in range(sub.shape[0]):
                    for k_ in range(sub.shape[1]):
                        s += sub[j, k_] * 0.5
                out[i] = s

            N = 32
            rng = np.random.default_rng(13)
            an = rng.standard_normal((N, 7)).astype(np.float32)
            a3n = rng.standard_normal((N, 3, 5)).astype(np.float32)
            outs = {}
            for dev in ('cpu', 'metal:0'):
                out = wp.zeros(N, dtype=float, device=dev)
                wp.launch(k, dim=N, inputs=[wp.array(an, dtype=float, ndim=2, device=dev),
                                            wp.array(a3n, dtype=float, ndim=3, device=dev), out], device=dev)
                outs[dev] = out.numpy()
            np.testing.assert_allclose(outs['metal:0'], outs['cpu'], rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(outs['cpu'], an.sum(axis=1) + 0.5 * a3n.sum(axis=(1, 2)), rtol=1e-4)
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalUtilsOps(unittest.TestCase):
    """``wp.utils`` reductions/sorts on Metal route through the ``_host``
    native implementations (unified memory + device sync). Before that fix
    they either raised ``UnboundLocalError`` (sum/inner) or silently
    returned stale output data (scan/sort/RLE)."""

    def test_reductions_match_numpy(self):
        snippet = textwrap.dedent(
            """
            import numpy as np
            import warp as wp
            import warp.utils

            rng = np.random.default_rng(42)
            dev = 'metal:0'

            x = rng.standard_normal(1000).astype(np.float32)
            y = rng.standard_normal(1000).astype(np.float32)
            ax = wp.array(x, device=dev)
            ay = wp.array(y, device=dev)
            np.testing.assert_allclose(wp.utils.array_sum(ax), x.sum(), rtol=1e-4)
            np.testing.assert_allclose(wp.utils.array_inner(ax, ay), np.dot(x, y), rtol=1e-4)

            m = rng.standard_normal((6, 5)).astype(np.float32)
            am = wp.array(m, device=dev)
            np.testing.assert_allclose(
                wp.utils.array_sum(am, axis=1).numpy().squeeze(), m.sum(axis=1), rtol=1e-5)

            # A device kernel writes the input first — the host op must see
            # the post-kernel data, which exercises the pre-op sync.
            @wp.kernel
            def scale(a: wp.array(dtype=wp.float32)):
                i = wp.tid()
                a[i] = a[i] * 3.0

            wp.launch(scale, dim=1000, inputs=[ax], device=dev)
            np.testing.assert_allclose(wp.utils.array_sum(ax), 3.0 * x.sum(), rtol=1e-4)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_scan_sort_rle_match_numpy(self):
        snippet = textwrap.dedent(
            """
            import numpy as np
            import warp as wp
            import warp.utils

            rng = np.random.default_rng(7)
            dev = 'metal:0'

            # array_scan (inclusive + exclusive)
            src = rng.integers(0, 10, 256).astype(np.int32)
            a = wp.array(src, device=dev)
            out = wp.zeros(256, dtype=wp.int32, device=dev)
            wp.utils.array_scan(a, out, inclusive=True)
            np.testing.assert_array_equal(out.numpy(), np.cumsum(src).astype(np.int32))
            wp.utils.array_scan(a, out, inclusive=False)
            np.testing.assert_array_equal(
                out.numpy(), (np.cumsum(src) - src).astype(np.int32))

            # radix_sort_pairs (stable)
            n = 128
            keys_np = rng.integers(0, 1 << 20, n).astype(np.int32)
            vals_np = np.arange(n, dtype=np.int32)
            keys = wp.zeros(n * 2, dtype=wp.int32, device=dev)
            vals = wp.zeros(n * 2, dtype=wp.int32, device=dev)
            keys.assign(np.concatenate([keys_np, np.zeros(n, np.int32)]))
            vals.assign(np.concatenate([vals_np, np.zeros(n, np.int32)]))
            wp.utils.radix_sort_pairs(keys, vals, n)
            order = np.argsort(keys_np, kind='stable')
            np.testing.assert_array_equal(keys.numpy()[:n], keys_np[order])
            np.testing.assert_array_equal(vals.numpy()[:n], vals_np[order])

            # segmented_sort_pairs
            seg_starts = np.array([0, 20, 45, n], dtype=np.int32)
            keys.assign(np.concatenate([keys_np, np.zeros(n, np.int32)]))
            vals.assign(np.concatenate([vals_np, np.zeros(n, np.int32)]))
            starts = wp.array(seg_starts, device=dev)
            wp.utils.segmented_sort_pairs(keys, vals, n, starts)
            expect_k = keys_np.copy()
            expect_v = vals_np.copy()
            for s, e in zip(seg_starts[:-1], seg_starts[1:]):
                seg_order = np.argsort(keys_np[s:e], kind='stable')
                expect_k[s:e] = keys_np[s:e][seg_order]
                expect_v[s:e] = vals_np[s:e][seg_order]
            np.testing.assert_array_equal(keys.numpy()[:n], expect_k)
            np.testing.assert_array_equal(vals.numpy()[:n], expect_v)

            # runlength_encode
            rle_src = np.sort(rng.integers(0, 8, 100)).astype(np.int32)
            arr = wp.array(rle_src, device=dev)
            run_values = wp.zeros(100, dtype=wp.int32, device=dev)
            run_lengths = wp.zeros(100, dtype=wp.int32, device=dev)
            n_runs = wp.utils.runlength_encode(arr, run_values, run_lengths)
            uniq, counts = np.unique(rle_src, return_counts=True)
            assert n_runs == len(uniq), (n_runs, len(uniq))
            np.testing.assert_array_equal(run_values.numpy()[:n_runs], uniq)
            np.testing.assert_array_equal(run_lengths.numpy()[:n_runs], counts.astype(np.int32))
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalUnsupportedFeaturesRaise(unittest.TestCase):
    """Unsupported subsystems must raise clear errors, never return silently
    wrong results (zero gradients, flat-indexed indexedarray reads)."""

    def test_indexedarray_launch_raises(self):
        snippet = textwrap.dedent(
            """
            import numpy as np
            import warp as wp

            @wp.kernel
            def k_idx(src: wp.indexedarray(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
                i = wp.tid()
                out[i] = src[i] * 2.0

            dev = 'metal:0'
            a = wp.array(np.arange(20, dtype=np.float32), device=dev)
            indices = wp.array(np.array([3, 7, 11, 19], np.int32), device=dev)
            ia = wp.indexedarray(a, [indices])
            out = wp.zeros(4, dtype=wp.float32, device=dev)
            try:
                wp.launch(k_idx, dim=4, inputs=[ia], outputs=[out], device=dev)
            except Exception as e:
                assert 'indexedarray' in str(e), str(e)
            else:
                raise AssertionError('indexedarray launch should have raised')
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_tape_backward_raises(self):
        snippet = textwrap.dedent(
            """
            import numpy as np
            import warp as wp

            @wp.kernel
            def k_sq(x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32)):
                i = wp.tid()
                y[i] = x[i] * x[i]

            dev = 'metal:0'
            x = wp.array(np.ones(4, np.float32), device=dev, requires_grad=True)
            y = wp.zeros(4, dtype=wp.float32, device=dev, requires_grad=True)
            tape = wp.Tape()
            with tape:
                wp.launch(k_sq, dim=4, inputs=[x], outputs=[y], device=dev)
            # The forward launch must be recorded — a silently empty tape
            # would "succeed" and leave every gradient at zero.
            assert len(tape.launches) == 1, len(tape.launches)
            try:
                tape.backward(grads={y: wp.array(np.ones(4, np.float32), device=dev)})
            except RuntimeError as e:
                assert 'adjoint' in str(e), str(e)
            else:
                raise AssertionError('tape.backward should have raised')
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_mesh_query_kernel_raises(self):
        snippet = textwrap.dedent(
            """
            import numpy as np
            import warp as wp

            @wp.kernel
            def k_mesh(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), d: wp.array(dtype=wp.float32)):
                i = wp.tid()
                q = wp.mesh_query_point(mesh, pts[i], 10.0)
                if q.result:
                    d[i] = 1.0

            dev = 'metal:0'
            mesh_pts = wp.array(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
                                dtype=wp.vec3, device=dev)
            tris = wp.array(np.array([0, 1, 2], np.int32), device=dev)
            mesh = wp.Mesh(points=mesh_pts, indices=tris)
            pts = wp.array(np.zeros((4, 3), np.float32), dtype=wp.vec3, device=dev)
            d = wp.zeros(4, dtype=wp.float32, device=dev)
            try:
                wp.launch(k_mesh, dim=4, inputs=[mesh.id, pts], outputs=[d], device=dev)
            except Exception as e:
                assert 'mesh_query_point' in str(e), str(e)
            else:
                raise AssertionError('mesh query launch should have raised')
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalConstBufferCache(unittest.TestCase):
    """The native launcher caches constant per-launch buffers (packed
    shapes, struct args) keyed on their contents. Repeated launches must
    reuse them; launches with different shapes/struct values must not."""

    def test_repeat_and_varied_launches_stay_correct(self):
        snippet = textwrap.dedent(
            """
            import numpy as np
            import warp as wp

            @wp.kernel
            def k(a: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
                i, j = wp.tid()
                out[i, j] = a[i, j] * 2.0

            dev = 'metal:0'
            rng = np.random.default_rng(3)
            # Two different shapes through the SAME kernel, interleaved, so a
            # wrongly keyed shapes cache would bind the wrong __shapes_packed.
            an = rng.standard_normal((16, 8)).astype(np.float32)
            bn = rng.standard_normal((5, 31)).astype(np.float32)
            a = wp.array(an, device=dev)
            b = wp.array(bn, device=dev)
            out_a = wp.zeros((16, 8), dtype=wp.float32, device=dev)
            out_b = wp.zeros((5, 31), dtype=wp.float32, device=dev)
            for _ in range(3):
                wp.launch(k, dim=(16, 8), inputs=[a], outputs=[out_a], device=dev)
                wp.launch(k, dim=(5, 31), inputs=[b], outputs=[out_b], device=dev)
            np.testing.assert_allclose(out_a.numpy(), an * 2.0)
            np.testing.assert_allclose(out_b.numpy(), bn * 2.0)

            @wp.struct
            class Params:
                scale: wp.float32
                offset: wp.float32

            @wp.kernel
            def k_struct(p: Params, x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32)):
                i = wp.tid()
                y[i] = x[i] * p.scale + p.offset

            xn = rng.standard_normal(64).astype(np.float32)
            x = wp.array(xn, device=dev)
            y = wp.zeros(64, dtype=wp.float32, device=dev)
            # Different struct VALUES per launch — the bytes-keyed cache must
            # not serve launch 1's params to launch 2.
            for scale, offset in ((2.0, 1.0), (3.0, -0.5), (2.0, 1.0)):
                p = Params()
                p.scale = scale
                p.offset = offset
                wp.launch(k_struct, dim=64, inputs=[p, x], outputs=[y], device=dev)
                np.testing.assert_allclose(y.numpy(), xn * scale + offset, rtol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)


def _has_torch_mps() -> bool:
    try:
        import torch  # noqa: PLC0415

        return torch.backends.mps.is_available()
    except Exception:
        return False


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
@unittest.skipUnless(_has_torch_mps(), "Torch with MPS support is not installed")
class TestMetalTorchInterop(unittest.TestCase):
    """Torch interop on Metal. Native dispatch: ``wp.to_torch`` is a
    zero-copy MPS view of the Warp array's MTLBuffer (kDLMetal DLPack) and
    Torch ops write back in place; ``wp.from_torch`` re-wraps such views.
    MLX dispatch: ``to_torch`` copies, ``from_torch`` raises with guidance."""

    def test_to_torch_and_writeback(self):
        snippet = textwrap.dedent(
            """
            import numpy as np
            import torch

            dev = 'metal:0'
            assert wp.device_from_torch(torch.device('mps')).is_metal
            assert wp.device_to_torch(dev) == 'mps'

            @wp.kernel
            def scale2(a: wp.array(dtype=wp.float32)):
                i = wp.tid()
                a[i] = a[i] * 2.0

            src = np.arange(64, dtype=np.float32)
            a = wp.array(src, device=dev)
            t = wp.to_torch(a)
            assert t.device.type == 'mps', t.device
            np.testing.assert_allclose(t.cpu().numpy(), src)

            if wp.config.metal_native_dispatch:
                # zero-copy: Warp kernel writes are visible through the view...
                wp.launch(scale2, dim=64, inputs=[a], device=dev)
                wp.synchronize_device(dev)
                np.testing.assert_allclose(t.cpu().numpy(), src * 2)
                # ...and Torch writes land in the Warp array (action writeback).
                t.copy_(torch.arange(64, dtype=torch.float32, device='mps') + 100.0)
                torch.mps.synchronize()
                np.testing.assert_allclose(a.numpy(), src + 100.0)

                # zero-stride broadcast dims survive (mjlab TorchArray heuristics)
                base = wp.array(np.arange(5, dtype=np.float32), device=dev)
                b = wp.array(ptr=base.ptr, dtype=wp.float32, shape=(4, 5), strides=(0, 4), device=dev, copy=False)
                tb = wp.to_torch(b)
                assert tb.stride(0) == 0, tb.stride()
                np.testing.assert_allclose(tb[3].cpu().numpy(), np.arange(5))
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_from_torch_roundtrip_and_errors(self):
        snippet = textwrap.dedent(
            """
            import numpy as np
            import torch

            dev = 'metal:0'

            @wp.kernel
            def scale2(a: wp.array(dtype=wp.float32)):
                i = wp.tid()
                a[i] = a[i] * 2.0

            src = np.arange(32, dtype=np.float32)
            a = wp.array(src, device=dev)
            t = wp.to_torch(a)

            if wp.config.metal_native_dispatch:
                # a Torch view of Warp memory wraps back zero-copy...
                back = wp.from_torch(t)
                assert back.ptr == a.ptr
                wp.launch(scale2, dim=32, inputs=[back], device=dev)
                wp.synchronize_device(dev)
                np.testing.assert_allclose(a.numpy(), src * 2)
                # ...and releasing the wrapper must not unregister the
                # Warp-owned buffer (launches on ``a`` must keep working).
                import gc
                import warp._src.context as ctx
                del back
                gc.collect()
                assert ctx._metal_get_buffer(a.ptr) is not None
                wp.launch(scale2, dim=32, inputs=[a], device=dev)
                wp.synchronize_device(dev)
                np.testing.assert_allclose(a.numpy(), src * 4)

                # Torch-owned MPS tensors use private storage -> clear error
                try:
                    wp.from_torch(torch.zeros(8, device='mps'))
                    raise AssertionError('expected RuntimeError')
                except RuntimeError as e:
                    assert 'private storage' in str(e), e

                # torch consumes Warp's kDLMetal capsule directly too
                t2 = torch.from_dlpack(wp.to_dlpack(a))
                assert t2.device.type == 'mps'
                np.testing.assert_allclose(t2.cpu().numpy(), a.numpy())
            else:
                # MLX path: from_torch(mps) raises with guidance
                try:
                    wp.from_torch(torch.zeros(8, device='mps'))
                    raise AssertionError('expected RuntimeError')
                except RuntimeError as e:
                    assert 'metal_native_dispatch' in str(e), e
                # ...and the CPU-capsule export gives numpy a zero-copy view
                n = np.from_dlpack(a)
                np.testing.assert_allclose(n, src)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalGraphCapture(unittest.TestCase):
    """``wp.capture_begin/end/launch`` on Metal map onto the dispatcher's
    ICB record/replay (native dispatch only). Launches are recorded, NOT
    executed; host-side memory ops raise instead of silently executing
    once and diverging from CUDA-graph replay semantics."""

    def test_capture_replay_semantics(self):
        snippet = textwrap.dedent(
            """
            import numpy as np

            dev = 'metal:0'

            @wp.kernel
            def add_one(a: wp.array(dtype=wp.float32)):
                i = wp.tid()
                a[i] = a[i] + 1.0

            @wp.kernel
            def double_into(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
                i = wp.tid()
                dst[i] = src[i] * 2.0

            if not wp.config.metal_native_dispatch:
                try:
                    with wp.ScopedDevice(dev):
                        wp.capture_begin()
                    raise AssertionError('expected RuntimeError')
                except RuntimeError as e:
                    assert 'metal_native_dispatch' in str(e), e
                raise SystemExit(0)

            a = wp.zeros(64, dtype=wp.float32, device=dev)
            b = wp.zeros(64, dtype=wp.float32, device=dev)
            # warm modules outside the capture
            wp.launch(add_one, dim=64, inputs=[a], device=dev)
            wp.launch(double_into, dim=64, inputs=[a, b], device=dev)
            wp.synchronize_device(dev)
            a.zero_()
            b.zero_()

            with wp.ScopedDevice(dev):
                wp.capture_begin()
                try:
                    for _ in range(3):
                        wp.launch(add_one, dim=64, inputs=[a])
                    wp.launch(double_into, dim=64, inputs=[a, b])
                finally:
                    g = wp.capture_end()

            # capture must not have executed
            np.testing.assert_allclose(a.numpy(), 0.0)
            np.testing.assert_allclose(b.numpy(), 0.0)

            for _ in range(5):
                wp.capture_launch(g)
            wp.synchronize_device(dev)
            np.testing.assert_allclose(a.numpy(), 15.0)  # 5 replays x 3 increments
            np.testing.assert_allclose(b.numpy(), 30.0)  # dependent kernel saw final a

            # direct launches interleave with replays
            wp.launch(add_one, dim=64, inputs=[a], device=dev)
            wp.capture_launch(g)
            wp.synchronize_device(dev)
            np.testing.assert_allclose(a.numpy(), 19.0)
            np.testing.assert_allclose(b.numpy(), 38.0)

            # ScopedCapture sugar
            with wp.ScopedDevice(dev):
                with wp.ScopedCapture() as cap:
                    wp.launch(add_one, dim=64, inputs=[a])
            wp.capture_launch(cap.graph)
            wp.synchronize_device(dev)
            np.testing.assert_allclose(a.numpy(), 20.0)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_host_ops_raise_during_capture(self):
        snippet = textwrap.dedent(
            """
            import numpy as np

            dev = 'metal:0'
            if not wp.config.metal_native_dispatch:
                raise SystemExit(0)

            a = wp.zeros(8, dtype=wp.float32, device=dev)
            b = wp.zeros(8, dtype=wp.float32, device=dev)

            with wp.ScopedDevice(dev):
                wp.capture_begin()
                try:
                    for fn in (
                        lambda: a.zero_(),
                        lambda: wp.copy(b, a),
                        lambda: wp.array(np.arange(4, dtype=np.float32), device=dev),
                    ):
                        try:
                            fn()
                            raise AssertionError('expected RuntimeError')
                        except RuntimeError as e:
                            assert 'capture' in str(e), e
                finally:
                    g = wp.capture_end()

            # nothing was recorded; replay is a harmless no-op
            wp.capture_launch(g)
            wp.synchronize_device(dev)
            np.testing.assert_allclose(a.numpy(), 0.0)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=60)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalHostOpOrdering(unittest.TestCase):
    """Host-side ops on Metal arrays must drain queued GPU work first, and
    ``np.asarray`` must convert instead of hitting NumPy's fallback
    iteration error."""

    def test_fill_after_unsynced_kernel(self):
        # Before the memset guard, ``zero_()`` ran immediately on the host
        # while the kernel's write was still queued on the GPU — the kernel
        # then overwrote the zeros (silent wrong values on native dispatch).
        snippet = textwrap.dedent(
            """
            import numpy as np

            dev = 'metal:0'

            @wp.kernel
            def fill7(a: wp.array(dtype=wp.float32)):
                i = wp.tid()
                a[i] = 7.0

            a = wp.zeros(1 << 20, dtype=wp.float32, device=dev)
            wp.launch(fill7, dim=1 << 20, inputs=[a], device=dev)
            a.zero_()  # no explicit sync: the guard must order this after the kernel
            np.testing.assert_allclose(a.numpy(), 0.0)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=60)

    def test_np_asarray(self):
        snippet = textwrap.dedent(
            """
            import numpy as np

            src = np.arange(16, dtype=np.float32)
            a = wp.array(src, device='metal:0')
            np.testing.assert_allclose(np.asarray(a), src)
            np.testing.assert_allclose(np.asarray(a, dtype=np.float64), src)
            """
        )
        _run_with_metal_enabled(self, snippet)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalArtifactCache(unittest.TestCase):
    """The on-disk MSL artifact cache must round-trip artifacts exactly and
    degrade to regeneration on any corruption."""

    def test_closure_kernels_with_different_constants_do_not_collide(self):
        # Kernel factories (mujoco_warp's ``@wp.kernel``-in-a-function
        # pattern) produce instantiations that share kernel.key, the arg
        # signature, AND the forward IR statements — the captured Python
        # scalar only shows up as a baked ``const`` declaration sourced
        # from ``Var.constant``. The cache key must include those values,
        # otherwise the second instantiation silently loads the first
        # one's artifact (this bit mujoco_warp's ``_solve_LD_sparse_fused``:
        # an nv=1 solver cached from one model was served to an nv=6
        # model, freezing the dynamics).
        snippet = textwrap.dedent(
            """
            import numpy as np
            import warp as wp
            from warp._src import codegen_metal as cm

            def make(scale):
                @wp.kernel
                def k(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
                    tid = wp.tid()
                    out[tid] = x[tid] * scale
                return k

            k1 = make(1.0)
            k6 = make(6.0)

            key1 = cm._artifact_cache_key(k1, cm._ensure_adj_built(k1))
            key6 = cm._artifact_cache_key(k6, cm._ensure_adj_built(k6))
            assert key1 != key6, 'closure constants must be part of the cache key'

            xn = np.arange(1.0, 9.0, dtype=np.float32)
            x = wp.array(xn, device='metal:0')
            o1 = wp.zeros(8, dtype=wp.float32, device='metal:0')
            o6 = wp.zeros(8, dtype=wp.float32, device='metal:0')
            wp.launch(k1, dim=8, inputs=[x], outputs=[o1], device='metal:0')
            wp.launch(k6, dim=8, inputs=[x], outputs=[o6], device='metal:0')
            np.testing.assert_allclose(o1.numpy(), xn)
            np.testing.assert_allclose(o6.numpy(), xn * 6.0)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_artifact_cache_roundtrip(self):
        snippet = textwrap.dedent(
            """
            import os
            import re

            import numpy as np
            import warp as wp
            from warp._src import codegen_metal as cm

            @wp.func
            def helper(q: wp.quat, v: wp.vec3) -> wp.vec3:
                return wp.quat_rotate(q, v) + wp.quat_rotate_inv(q, v)

            @wp.kernel
            def k(q: wp.array(dtype=wp.quat), x: wp.array(dtype=wp.vec3),
                  out: wp.array(dtype=wp.vec3)):
                tid = wp.tid()
                out[tid] = helper(wp.normalize(q[tid]), x[tid])

            adj = cm._ensure_adj_built(k)
            key = cm._artifact_cache_key(k, adj)
            assert key is not None, 'cache key must be computable'
            path = cm._artifact_cache_path(key)
            if os.path.exists(path):
                os.unlink(path)

            a1 = cm.generate_msl_kernel(k)  # cold: generate + store
            assert os.path.exists(path), 'artifact must be stored after cold generate'
            a2 = cm.generate_msl_kernel(k)  # warm: disk load

            f1 = dict(a1.__dict__)
            f2 = dict(a2.__dict__)
            ia1, ia2 = f1.pop('input_args'), f2.pop('input_args')
            oa1, oa2 = f1.pop('output_args'), f2.pop('output_args')
            assert f1 == f2, 'cached artifact fields must round-trip exactly'
            assert all(x is y for x, y in zip(ia1 + oa1, ia2 + oa2)), \\
                'arg Vars must be reconstructed as the same adj.args objects'

            # Corrupt entry -> silent regenerate. The inliner's var_N__k
            # suffixes are not stable across repeat generations on the same
            # adj, so compare with those normalized.
            with open(path, 'wb') as f:
                f.write(b'garbage')
            a3 = cm.generate_msl_kernel(k)
            norm = lambda s: re.sub(r'var_\\d+__', 'var_X__', s)
            assert norm(a3.source) == norm(a1.source)

            # Kill switch disables the cache entirely.
            os.environ['WARP_METAL_DISABLE_ARTIFACT_CACHE'] = '1'
            assert cm._artifact_cache_path(key) is None
            del os.environ['WARP_METAL_DISABLE_ARTIFACT_CACHE']

            # End-to-end: a launch that consumes the cache-loaded artifact
            # must match the CPU backend.
            N = 32
            rng = np.random.default_rng(31)
            qn = rng.standard_normal((N, 4)).astype(np.float32)
            xn = rng.standard_normal((N, 3)).astype(np.float32)
            outs = {}
            for dev in ('cpu', 'metal:0'):
                out = wp.zeros(N, dtype=wp.vec3, device=dev)
                wp.launch(k, dim=N,
                          inputs=[wp.array(qn, dtype=wp.quat, device=dev),
                                  wp.array(xn, dtype=wp.vec3, device=dev)],
                          outputs=[out], device=dev)
                outs[dev] = out.numpy()
            np.testing.assert_allclose(outs['cpu'], outs['metal:0'], rtol=1e-5, atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)


if __name__ == "__main__":
    unittest.main(verbosity=2)
