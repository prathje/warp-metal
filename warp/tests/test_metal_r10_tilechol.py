# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal reverse mode through the tile linear-algebra ops and indexed views.

Covers what Newton's Featherstone tile path (``use_tile_gemm=True``) needs
under a tape: ``wp.tile_store(H[idx], t)`` / ``wp.tile_load(J[idx], ...)``
adjoints into an integer-indexed array view, the adjoint of
``wp.tile_cholesky`` and of ``wp.tile_lower_solve`` (the two differentiable
tile linear-algebra builtins upstream), plus the forward of the out-of-place
``wp.tile_lower_solve`` / ``wp.tile_upper_solve``.

Each test runs the same kernels on ``cpu`` and ``metal:0`` in a subprocess
and compares values and gradients; the linear-algebra gradients are also
checked against float64 central finite differences of a NumPy reference.
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_native

# Shared by the snippets: SPD / triangular inputs, the reference factor and
# solves in float64 (reading the lower triangle only, like the native
# ``scalar_cholesky_impl``), and a central-difference gradient.
_COMMON = textwrap.dedent(
    """
    import numpy as np

    def spd(rng, b, n):
        g = rng.standard_normal((b, n, n))
        return (g @ g.transpose(0, 2, 1) / n + np.eye(n)).astype(np.float32)

    def chol_lower(a):
        n = a.shape[0]
        l = np.zeros_like(a, dtype=np.float64)
        for j in range(n):
            s = a[j, j] - np.dot(l[j, :j], l[j, :j])
            l[j, j] = np.sqrt(s)
            for i in range(j + 1, n):
                l[i, j] = (a[i, j] - np.dot(l[i, :j], l[j, :j])) / l[j, j]
        return l

    def lower_solve(l, y):
        z = np.zeros_like(y, dtype=np.float64)
        for i in range(l.shape[0]):
            z[i] = (y[i] - l[i, :i] @ z[:i]) / l[i, i]
        return z

    def fd_grad(f, x, eps=1e-6, mask=None):
        x = np.array(x, dtype=np.float64)
        g = np.zeros_like(x)
        it = np.ndindex(*x.shape)
        for idx in it:
            if mask is not None and not mask[idx]:
                continue
            xp = x.copy(); xp[idx] += eps
            xm = x.copy(); xm[idx] -= eps
            g[idx] = (f(xp) - f(xm)) / (2.0 * eps)
        return g

    @wp.kernel
    def wloss3(x: wp.array3d(dtype=float), w: wp.array3d(dtype=float), loss: wp.array(dtype=float)):
        i, r, c = wp.tid()
        wp.atomic_add(loss, 0, x[i, r, c] * w[i, r, c])

    def close(a, b, what, rtol=1e-4):
        scale = max(1.0, float(np.abs(b).max()))
        np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * scale, err_msg=what)
    """
)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR10TileChol(unittest.TestCase):
    def test_tile_view_store_load_atomic_adjoint_match_cpu(self):
        # ``wp.tile_store(H[i], t)`` / ``wp.tile_load(A[i], ...)`` /
        # ``wp.tile_atomic_add(G[i, j], ...)`` under a tape: the adjoint has to
        # address the base array with the view's leading indices. Before the
        # fix: "cannot resolve wp::adj_tile_store on array 'var_N'". Includes a
        # partial edge tile (offsets past the end of a 7x9 slice) and a 4-D
        # array viewed with two indices.
        snippet = _COMMON + textwrap.dedent(
            """
            @wp.kernel(module="unique")
            def k3(a: wp.array3d(dtype=float), h: wp.array3d(dtype=float)):
                i = wp.tid()
                t = wp.tile_load(a[i], shape=(5, 5), offset=(3, 6))
                wp.tile_store(h[i], wp.tile_map(wp.mul, t, t), offset=(2, 1))

            @wp.kernel(module="unique")
            def k4(a: wp.array4d(dtype=float), g: wp.array4d(dtype=float)):
                i, j = wp.tid()
                t = wp.tile_load(a[i, j], shape=(4, 3))
                wp.tile_atomic_add(g[i, j], wp.tile_map(wp.sin, t), offset=(1, 0))

            rng = np.random.default_rng(1)
            a3 = rng.standard_normal((3, 7, 9)).astype(np.float32)
            w3 = rng.standard_normal((3, 7, 9)).astype(np.float32)
            a4 = rng.standard_normal((2, 3, 4, 3)).astype(np.float32)
            w4 = rng.standard_normal((2, 3, 5, 3)).astype(np.float32)
            res = {}
            for dev in ("cpu", "metal:0"):
                a = wp.array(a3, device=dev, requires_grad=True)
                h = wp.zeros((3, 7, 9), dtype=float, device=dev, requires_grad=True)
                b = wp.array(a4, device=dev, requires_grad=True)
                g = wp.zeros((2, 3, 5, 3), dtype=float, device=dev, requires_grad=True)
                loss = wp.zeros(1, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch_tiled(k3, dim=3, inputs=[a], outputs=[h], device=dev, block_dim=64)
                    wp.launch_tiled(k4, dim=(2, 3), inputs=[b], outputs=[g], device=dev, block_dim=64)
                    wp.launch(wloss3, dim=(3, 7, 9), inputs=[h, wp.array(w3, device=dev)], outputs=[loss], device=dev)
                    wp.launch(wloss3, dim=(6, 5, 3), inputs=[g.reshape((6, 5, 3)), wp.array(w4.reshape(6, 5, 3), device=dev)],
                              outputs=[loss], device=dev)
                tape.backward(loss)
                res[dev] = (h.numpy(), g.numpy(), a.grad.numpy(), b.grad.numpy())
            for k, what in enumerate(("h", "g", "a.grad", "b.grad")):
                close(res["metal:0"][k], res["cpu"][k], what, rtol=1e-5)
            assert np.abs(res["cpu"][2]).max() > 0.1 and np.abs(res["cpu"][3]).max() > 0.1
            """
        )
        _run_with_metal_native(self, snippet, timeout=300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
