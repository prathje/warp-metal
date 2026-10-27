# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal parity with the CPU for behaviour Warp 1.19 changed.

Each test runs the same kernel on ``cpu`` and ``metal:0`` in a subprocess and
compares the results, or pins that Metal rejects a construct loudly
(``MetalCodegenError``) instead of computing something different from the CPU.
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_native


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalUpstream119(unittest.TestCase):
    def test_tile_matmul_beta_zero_does_not_read_out(self):
        # GH-2079: with beta == 0 the product must not read ``out``, so NaN or
        # Inf already in ``out`` cannot reach the result through 0 * out.
        snippet = textwrap.dedent(
            """
            import numpy as np

            MM = 8

            @wp.kernel(module="unique")
            def k_lit(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float), c: wp.array2d(dtype=float)):
                ta = wp.tile_load(a, shape=(MM, MM))
                tb = wp.tile_load(b, shape=(MM, MM))
                tc = wp.tile_load(c, shape=(MM, MM))
                wp.tile_matmul(ta, tb, tc, alpha=2.0, beta=0.0)
                wp.tile_store(c, tc)

            @wp.kernel(module="unique")
            def k_dyn(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float), c: wp.array2d(dtype=float),
                      beta: float):
                ta = wp.tile_load(a, shape=(MM, MM))
                tb = wp.tile_load(b, shape=(MM, MM))
                tc = wp.tile_load(c, shape=(MM, MM))
                wp.tile_matmul(ta, tb, tc, alpha=1.0, beta=beta)
                wp.tile_store(c, tc)

            rng = np.random.default_rng(3)
            a_np = rng.standard_normal((MM, MM)).astype(np.float32)
            b_np = rng.standard_normal((MM, MM)).astype(np.float32)
            c_np = np.ones((MM, MM), np.float32)
            c_np[0, 0] = np.nan
            c_np[3, 4] = np.inf
            c_np[5, 1] = -np.inf
            cases = [
                (k_lit, [], 2.0 * (a_np @ b_np)),
                (k_dyn, [0.0], a_np @ b_np),
                (k_dyn, [0.5], a_np @ b_np + 0.5 * c_np),
            ]
            for k, extra, expected in cases:
                outs = {}
                for dev in ("cpu", "metal:0"):
                    a = wp.array(a_np, device=dev)
                    b = wp.array(b_np, device=dev)
                    c = wp.array(c_np, device=dev)
                    for bd in (1, 64):
                        c.assign(c_np)
                        wp.launch_tiled(k, dim=1, inputs=[a, b, c, *extra], block_dim=bd, device=dev)
                        outs[(dev, bd)] = c.numpy().copy()
                for key, got in outs.items():
                    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5, err_msg=f"{k.key} {extra} {key}")
                for bd in (1, 64):
                    np.testing.assert_array_equal(np.isfinite(outs[("cpu", bd)]), np.isfinite(outs[("metal:0", bd)]))
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_tile_matmul_tf32_warns_and_falls_back(self):
        # GH-937: precision="tf32" is honored only by cuBLASDx. Like the CPU,
        # Metal warns and computes at the operand precision.
        snippet = textwrap.dedent(
            """
            import os

            import numpy as np
            import warp._src.builtins as wpb

            os.environ["WARP_METAL_DISABLE_ARTIFACT_CACHE"] = "1"
            MM = 8

            @wp.kernel(module="unique")
            def k(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float), c: wp.array2d(dtype=float)):
                ta = wp.tile_load(a, shape=(MM, MM))
                tb = wp.tile_load(b, shape=(MM, MM))
                wp.tile_store(c, wp.tile_matmul(ta, tb, alpha=1.5, precision="tf32"))

            rng = np.random.default_rng(4)
            a_np = rng.standard_normal((MM, MM)).astype(np.float32)
            b_np = rng.standard_normal((MM, MM)).astype(np.float32)
            a = wp.array(a_np, device="metal:0")
            b = wp.array(b_np, device="metal:0")
            c = wp.zeros((MM, MM), dtype=float, device="metal:0")
            caught = []
            wpb.log_warning = lambda msg, *args, **kwargs: caught.append(msg)
            wp.launch_tiled(k, dim=1, inputs=[a, b, c], block_dim=64, device="metal:0")
            wp.synchronize_device("metal:0")
            assert any("precision='tf32'" in m for m in caught), caught
            np.testing.assert_allclose(c.numpy(), 1.5 * (a_np @ b_np), rtol=1e-5, atol=1e-5)
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_tile_map_adjoint_with_forward_result(self):
        # GH-2031: tile_map over builtins whose adjoint takes the forward
        # result (exp, sqrt, tanh, pow, div) — gradients match the CPU.
        snippet = textwrap.dedent(
            """
            import numpy as np

            M, N = 4, 8

            @wp.kernel(module="unique")
            def k_unary(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
                t = wp.tile_load(x, shape=(M, N))
                e = wp.tile_map(wp.exp, t)
                s = wp.tile_map(wp.sqrt, t)
                h = wp.tile_map(wp.tanh, t)
                wp.tile_store(y, wp.tile_map(wp.add, wp.tile_map(wp.add, e, s), h))

            @wp.kernel(module="unique")
            def k_binary(x: wp.array2d(dtype=float), z: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
                t = wp.tile_load(x, shape=(M, N))
                u = wp.tile_load(z, shape=(M, N))
                wp.tile_store(y, wp.tile_map(wp.add, wp.tile_map(wp.pow, t, u), wp.tile_map(wp.div, t, u)))

            rng = np.random.default_rng(11)
            x_np = rng.uniform(0.5, 2.0, (M, N)).astype(np.float32)
            z_np = rng.uniform(1.0, 3.0, (M, N)).astype(np.float32)
            g_np = rng.uniform(-1.0, 1.0, (M, N)).astype(np.float32)
            for name, k, ins in (("unary", k_unary, [x_np]), ("binary", k_binary, [x_np, z_np])):
                res = {}
                for dev in ("cpu", "metal:0"):
                    arrs = [wp.array(v, device=dev, requires_grad=True) for v in ins]
                    y = wp.zeros((M, N), dtype=float, device=dev, requires_grad=True)
                    tape = wp.Tape()
                    with tape:
                        wp.launch_tiled(k, dim=1, inputs=[*arrs, y], block_dim=64, device=dev)
                    y.grad = wp.array(g_np, device=dev)
                    tape.backward()
                    res[dev] = [y.numpy()] + [a.grad.numpy() for a in arrs]
                for i, (c, m) in enumerate(zip(res["cpu"], res["metal:0"])):
                    np.testing.assert_allclose(m, c, rtol=2e-5, atol=1e-5, err_msg=f"{name} output {i}")
                assert np.abs(res["cpu"][1]).sum() > 0
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_unported_tile_reductions_raise(self):
        # Metal has no lowering for these yet. 1.19 changed their CPU/CUDA
        # results (GH-2000 tile_sum axis gradients, GH-2003/GH-1781 scan
        # min/max seeds and uint32, GH-2001 broadcast tile_assign gradients,
        # GH-2010 eigenvalue-only eig3 gradients); Metal must keep refusing
        # them instead of computing something else.
        snippet = textwrap.dedent(
            """
            import numpy as np
            from warp._src.codegen_metal import MetalCodegenError

            M, N, L = 4, 8, 16

            @wp.kernel(module="unique")
            def k_sum_axis(a: wp.array2d(dtype=float), r: wp.array(dtype=float)):
                t = wp.tile_load(a, shape=(M, N))
                wp.tile_store(r, wp.tile_sum(t, axis=0))

            @wp.kernel(module="unique")
            def k_reduce_axis(a: wp.array2d(dtype=float), r: wp.array(dtype=float)):
                t = wp.tile_load(a, shape=(M, N))
                wp.tile_store(r, wp.tile_reduce(wp.max, t, axis=1))

            @wp.kernel(module="unique")
            def k_scan_max(a: wp.array(dtype=float), r: wp.array(dtype=float)):
                wp.tile_store(r, wp.tile_scan_max_inclusive(wp.tile_load(a, shape=L)))

            @wp.kernel(module="unique")
            def k_scan_min_u32(a: wp.array(dtype=wp.uint32), r: wp.array(dtype=wp.uint32)):
                wp.tile_store(r, wp.tile_scan_min_inclusive(wp.tile_load(a, shape=L)))

            @wp.kernel(module="unique")
            def k_assign_bcast(x: wp.array(dtype=float), out: wp.array2d(dtype=float)):
                v = wp.tile_load(x, shape=N)
                dst = wp.tile_zeros(shape=(M, N), dtype=float)
                wp.tile_assign(dst, wp.tile_broadcast(v, shape=(M, N)), offset=(0, 0))
                wp.tile_store(out, dst)

            @wp.kernel(module="unique")
            def k_eig(A: wp.array(dtype=wp.mat33), out: wp.array(dtype=float)):
                i = wp.tid()
                Q = wp.mat33()
                d = wp.vec3()
                wp.eig3(A[i], Q, d)
                wp.atomic_add(out, 0, d[0] + d[1] + d[2])

            dev = "metal:0"
            fwd = [
                (k_sum_axis, lambda: [wp.zeros((M, N), dtype=float, device=dev), wp.zeros(N, dtype=float, device=dev)]),
                (k_reduce_axis, lambda: [wp.zeros((M, N), dtype=float, device=dev), wp.zeros(M, dtype=float, device=dev)]),
                (k_scan_max, lambda: [wp.zeros(L, dtype=float, device=dev), wp.zeros(L, dtype=float, device=dev)]),
                (k_scan_min_u32, lambda: [wp.zeros(L, dtype=wp.uint32, device=dev), wp.zeros(L, dtype=wp.uint32, device=dev)]),
            ]
            for k, mk in fwd:
                try:
                    wp.launch_tiled(k, dim=1, inputs=mk(), block_dim=64, device=dev)
                    wp.synchronize_device(dev)
                except MetalCodegenError:
                    continue
                raise AssertionError(f"{k.key} ran on Metal; it needs a CPU-parity test now")

            def bwd_assign():
                x = wp.zeros(N, dtype=float, device=dev, requires_grad=True)
                out = wp.zeros((M, N), dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch_tiled(k_assign_bcast, dim=1, inputs=[x, out], block_dim=64, device=dev)
                out.grad = wp.ones((M, N), dtype=float, device=dev)
                tape.backward()

            def bwd_eig():
                A = wp.array([wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0)], dtype=wp.mat33, device=dev,
                             requires_grad=True)
                out = wp.zeros(1, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(k_eig, dim=1, inputs=[A, out], device=dev)
                tape.backward(loss=out)

            for name, fn in (("adj tile_assign(broadcast)", bwd_assign), ("adj eig3", bwd_eig)):
                try:
                    fn()
                    wp.synchronize_device(dev)
                except MetalCodegenError:
                    continue
                raise AssertionError(f"{name} ran on Metal; it needs a CPU-parity test now")
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)


if __name__ == "__main__":
    unittest.main(verbosity=2)
