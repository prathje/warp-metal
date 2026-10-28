# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reversed ``wp.tid()`` axes for multi-dimensional native Metal launches.

Eligible 2-D / 3-D kernels are built with the tid axes reversed and launched on
the reversed grid, so the last tid index runs along the threadgroup's x axis
(CUDA's order; see ``warp.config.metal_tid_order``). Each test compares Metal
with CPU and checks which kernels took the reversed layout.
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_native


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR10TidOrder(unittest.TestCase):
    def test_reversed_tid_kernels_match_cpu(self):
        snippet = textwrap.dedent(
            """
            import numpy as np
            from warp._src.codegen_metal import _pso_tid_reversed

            @wp.kernel
            def k2(a: wp.array2d[float], b: wp.array2d[float], ids: wp.array2d[int]):
                i, j = wp.tid()
                b[i, j] = a[i, j] * 2.0 + float(j)
                ids[i, j] = i * 1000 + j

            @wp.kernel
            def k3(a: wp.array3d[float], b: wp.array3d[float]):
                i, j, k = wp.tid()
                b[i, j, k] = a[i, j, k] + float(100 * i + 10 * j + k)

            @wp.kernel
            def k2_atomic(a: wp.array2d[float], rowsum: wp.array[float], count: wp.array[int]):
                i, j = wp.tid()
                wp.atomic_add(rowsum, i, a[i, j])
                wp.atomic_add(count, 0, 1)

            @wp.kernel
            def k2_gather(rowadr: wp.array[int], col: wp.array[int], M: wp.array2d[float],
                          v: wp.array2d[float], r: wp.array2d[float]):
                w, j = wp.tid()
                acc = float(0.0)
                for k in range(rowadr[j], rowadr[j + 1]):
                    acc += M[w, k] * v[w, col[k]]
                r[w, j] = acc

            rng = np.random.default_rng(3)

            def run(dev):
                out = {}
                with wp.ScopedDevice(dev):
                    for shape in ((1, 1), (3, 35), (37, 5), (300, 7), (2, 300), (1000, 1)):
                        a = wp.array(rng.standard_normal(shape).astype(np.float32) if dev == "cpu" else out_in[shape])
                        b = wp.zeros(shape, dtype=float)
                        ids = wp.zeros(shape, dtype=int)
                        wp.launch(k2, dim=shape, inputs=[a, b, ids])
                        out[("k2", shape)] = (b.numpy(), ids.numpy())
                        rs = wp.zeros(shape[0], dtype=float)
                        cnt = wp.zeros(1, dtype=int)
                        wp.launch(k2_atomic, dim=shape, inputs=[a, rs, cnt])
                        out[("atomic", shape)] = (rs.numpy(), cnt.numpy())
                        if dev == "cpu":
                            out_in[shape] = a.numpy()
                    for shape in ((2, 3, 4), (5, 1, 33), (1, 40, 9), (64, 3, 3)):
                        a3 = wp.array(out_in.setdefault(shape, rng.standard_normal(shape).astype(np.float32)))
                        b3 = wp.zeros(shape, dtype=float)
                        wp.launch(k3, dim=shape, inputs=[a3, b3])
                        out[("k3", shape)] = b3.numpy()
                    n, nw = 13, 257
                    rowadr = np.cumsum([0] + [1 + (j % 5) for j in range(n)]).astype(np.int32)
                    col = np.array([(j + t) % n for j in range(n) for t in range(1 + (j % 5))], np.int32)
                    M = out_in.setdefault("M", rng.standard_normal((nw, int(rowadr[-1]))).astype(np.float32))
                    v = out_in.setdefault("v", rng.standard_normal((nw, n)).astype(np.float32))
                    r = wp.zeros((nw, n), dtype=float)
                    ins = [wp.array(rowadr), wp.array(col), wp.array(M), wp.array(v), r]
                    wp.launch(k2_gather, dim=(nw, n), inputs=ins)
                    out["gather"] = r.numpy()
                return out

            out_in = {}
            ref = run("cpu")
            got = run("metal:0")
            for key, val in ref.items():
                if key[0] == "atomic":
                    np.testing.assert_allclose(got[key][0], val[0], rtol=1e-5, atol=1e-5, err_msg=str(key))
                    np.testing.assert_array_equal(got[key][1], val[1], err_msg=str(key))
                elif isinstance(val, tuple):
                    for x, y in zip(got[key], val):
                        np.testing.assert_array_equal(x, y, err_msg=str(key))
                else:
                    np.testing.assert_allclose(got[key], val, rtol=1e-6, atol=1e-6, err_msg=str(key))
            assert _pso_tid_reversed(k2._metal_native_pso) == 2
            assert _pso_tid_reversed(k3._metal_native_pso) == 3
            assert _pso_tid_reversed(k2_gather._metal_native_pso) == 2
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_threadgroup_dependent_kernels_keep_their_layout(self):
        # A kernel whose body depends on its threadgroup (a cooperative
        # tile kernel) or that is built with the switch off keeps the
        # original axes.
        snippet = textwrap.dedent(
            """
            import numpy as np
            from warp._src.codegen_metal import _pso_tid_reversed

            @wp.kernel(module="unique")
            def k_off(a: wp.array2d[float], b: wp.array2d[float]):
                i, j = wp.tid()
                b[i, j] = a[i, j] + float(i)

            wp.config.metal_tid_order = False
            with wp.ScopedDevice("metal:0"):
                a = wp.array(np.arange(12, dtype=np.float32).reshape(3, 4))
                b = wp.zeros((3, 4), dtype=float)
                wp.launch(k_off, dim=(3, 4), inputs=[a, b])
                np.testing.assert_array_equal(b.numpy(), np.arange(12).reshape(3, 4) + np.arange(3)[:, None])
            assert _pso_tid_reversed(k_off._metal_native_pso) == 0
            wp.config.metal_tid_order = True

            N = 35

            @wp.kernel(module="unique", enable_backward=False)
            def k_tile(A: wp.array3d[float], L_out: wp.array3d[float]):
                w, blk = wp.tid()
                a = wp.tile_load(A[w], shape=(N, N), storage="shared")
                L = wp.tile_cholesky(a)
                wp.tile_store(L_out[w], L)

            S = np.stack([np.eye(N, dtype=np.float32) * (2.0 + w) for w in range(3)])
            with wp.ScopedDevice("metal:0"):
                Lo = wp.zeros((3, N, N), dtype=float)
                wp.launch_tiled(k_tile, dim=(3, 1), inputs=[wp.array(S)], outputs=[Lo], block_dim=32)
                np.testing.assert_allclose(Lo.numpy(), np.sqrt(S), rtol=1e-6)
            assert _pso_tid_reversed(k_tile._metal_native_pso) == 0
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)


if __name__ == "__main__":
    unittest.main(verbosity=2)
