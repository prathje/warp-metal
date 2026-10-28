# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-10 integration decisions for the Metal backend.

* Tile Cholesky pivots are not clamped. Native ``scalar_cholesky_impl``
  (``tile_cholesky.h``) takes a plain ``sqrt`` of the pivot, so a negative
  pivot gives NaN and a zero pivot inf/NaN on CPU and CUDA; every Metal
  forward (serial, by-value cooperative, threadgroup-resident, in-place, 1x1)
  must give the same NaN / inf pattern, in values and in gradients.
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_native

_COMMON = textwrap.dedent(
    """
    import re
    import numpy as np

    def cases(n):
        # SPD, indefinite pivot, all zero, one exact zero pivot, NaN pivot, NaN off-diagonal.
        rng = np.random.default_rng(n)
        g = rng.standard_normal((n, n))
        spd = (g @ g.T / n + np.eye(n)).astype(np.float32)
        out = [spd.copy()]
        a = spd.copy(); a[n // 2, n // 2] = -1.0; out.append(a)
        out.append(np.zeros((n, n), np.float32))
        a = np.eye(n, dtype=np.float32); a[n // 2, n // 2] = 0.0; out.append(a)
        a = spd.copy(); a[n // 2, n // 2] = np.nan; out.append(a)
        a = spd.copy(); a[n - 1, 0] = np.nan; out.append(a)
        return np.stack(out)

    def same_pattern(c, m, what, atol=1e-5):
        for i in range(c.shape[0]):
            np.testing.assert_array_equal(np.isnan(m[i]), np.isnan(c[i]), err_msg=f"{what} case {i}: NaN pattern")
            np.testing.assert_array_equal(np.isposinf(m[i]), np.isposinf(c[i]), err_msg=f"{what} case {i}: +inf")
            np.testing.assert_array_equal(np.isneginf(m[i]), np.isneginf(c[i]), err_msg=f"{what} case {i}: -inf")
            fin = np.isfinite(c[i])
            scale = max(1.0, float(np.abs(c[i][fin]).max())) if fin.any() else 1.0
            np.testing.assert_allclose(m[i][fin], c[i][fin], rtol=1e-4, atol=atol * scale, err_msg=f"{what} case {i}")

    def helpers(k):
        a = k._metal_artifact
        return set(re.findall(r"_cholesky[a-z_]*(?=\\()", a.source + (a.header or "")))
    """
)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR10CholeskyPivot(unittest.TestCase):
    def test_non_spd_cholesky_forward_matches_cpu_nan_pattern(self):
        snippet = _COMMON + textwrap.dedent(
            """
            def make(N, form, tag):
                NN = N * N
                if form == "oop":
                    @wp.kernel(module=f"pv_{form}_{N}_{tag}", enable_backward=False)
                    def k(A: wp.array3d[float], L: wp.array3d[float]):
                        w = wp.tid()
                        a = wp.tile_load(A[w], shape=(N, N), storage="shared")
                        wp.tile_store(L[w], wp.tile_cholesky(a))
                elif form == "inplace":
                    @wp.kernel(module=f"pv_{form}_{N}_{tag}", enable_backward=False)
                    def k(A: wp.array3d[float], L: wp.array3d[float]):
                        w = wp.tid()
                        a = wp.tile_load(A[w], shape=(N, N), storage="shared")
                        wp.tile_cholesky_inplace(a)
                        wp.tile_store(L[w], a)
                else:  # in-place through a reshape view: threadgroup-resident when N >= 24
                    @wp.kernel(module=f"pv_{form}_{N}_{tag}", enable_backward=False)
                    def k(A: wp.array2d[float], L: wp.array2d[float]):
                        w = wp.tid()
                        flat = wp.tile_load(A[w], shape=(NN,), storage="shared")
                        a = wp.tile_reshape(flat, (N, N))
                        wp.tile_cholesky_inplace(a)
                        wp.tile_store(L[w], wp.tile_reshape(a, (NN,)))
                return k

            seen = set()
            for form in ("oop", "inplace", "resident"):
                for N in (1, 5, 28):
                    if form == "resident" and N == 1:
                        continue
                    A = cases(N)
                    if form == "resident":
                        A = A.reshape(A.shape[0], N * N)
                    for tg in (True, False):
                        res = {}
                        for dev in ("cpu", "metal:0"):
                            wp.config.metal_threadgroup_tiles = tg
                            k = make(N, form, f"{dev[:3]}{int(tg)}")
                            L = wp.zeros(A.shape, dtype=float, device=dev)
                            wp.launch_tiled(k, dim=(A.shape[0],), inputs=[wp.array(A, device=dev)], outputs=[L],
                                            block_dim=32, device=dev)
                            res[dev] = L.numpy()
                            if dev != "cpu":
                                seen |= helpers(k)
                        wp.config.metal_threadgroup_tiles = True
                        same_pattern(res["cpu"], res["metal:0"], f"{form} N={N} tgtiles={tg}")
            # Every Metal forward flavour was exercised.
            for h in ("_cholesky", "_cholesky_coop", "_cholesky_smem", "_cholesky_inplace",
                      "_cholesky_inplace_smem"):
                assert h in seen, (h, seen)
            """
        )
        _run_with_metal_native(self, snippet, timeout=600)

    def test_non_spd_cholesky_gradient_matches_cpu_nan_pattern(self):
        # Before: the clamped Metal forward fed the adjoint a finite factor
        # (pivot 1e-15), giving finite huge gradients where CPU gives NaN.
        snippet = _COMMON + textwrap.dedent(
            """
            def make(N, tag):
                @wp.kernel(module=f"pg_{N}_{tag}")
                def k(A: wp.array3d[float], L: wp.array3d[float]):
                    w = wp.tid()
                    a = wp.tile_load(A[w], shape=(N, N))
                    wp.tile_store(L[w], wp.tile_cholesky(a))
                return k

            for N in (1, 2, 5, 24):
                A = cases(N)
                if N == 2:
                    A = np.stack([np.zeros((2, 2), np.float32), np.array([[1.0, 2.0], [2.0, 1.0]], np.float32)])
                res = {}
                for dev in ("cpu", "metal:0"):
                    k = make(N, dev[:3])
                    a = wp.array(A, device=dev, requires_grad=True)
                    L = wp.zeros(A.shape, dtype=float, device=dev, requires_grad=True)
                    tape = wp.Tape()
                    with tape:
                        wp.launch_tiled(k, dim=(A.shape[0],), inputs=[a], outputs=[L], block_dim=32, device=dev)
                    L.grad = wp.array(np.tril(np.ones(A.shape, np.float32)), device=dev)
                    tape.backward()
                    res[dev] = (L.numpy(), a.grad.numpy())
                same_pattern(res["cpu"][0], res["metal:0"][0], f"L N={N}")
                same_pattern(res["cpu"][1], res["metal:0"][1], f"adj_A N={N}", atol=1e-4)
            """
        )
        _run_with_metal_native(self, snippet, timeout=600)


if __name__ == "__main__":
    unittest.main(verbosity=2)
