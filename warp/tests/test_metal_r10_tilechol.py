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

    def test_large_tile_reverse_scatter_matches_cpu(self):
        # Reverse tile_load / tile_store / tile_atomic_add on scalar tiles of
        # >= 256 elements scatter in a loop instead of one statement per
        # element (a 48x48 cholesky backward took ~2 min to compile). Partial
        # edge tiles (offsets past the array end) and a 1-D tile.
        snippet = _COMMON + textwrap.dedent(
            """
            @wp.kernel(module="unique")
            def k(a: wp.array3d(dtype=float), v: wp.array2d(dtype=float), h: wp.array3d(dtype=float),
                  g: wp.array3d(dtype=float), u: wp.array2d(dtype=float)):
                i = wp.tid()
                t = wp.tile_load(a[i], shape=(24, 20), offset=(5, 3))
                wp.tile_store(h[i], wp.tile_map(wp.mul, t, t), offset=(4, 4))
                wp.tile_atomic_add(g[i], wp.tile_map(wp.sin, t), offset=(1, 1))
                s = wp.tile_load(v[i], shape=(300,), offset=(7,))
                wp.tile_store(u[i], wp.tile_map(wp.mul, s, s), offset=(2,))

            rng = np.random.default_rng(7)
            a0 = rng.standard_normal((2, 26, 21)).astype(np.float32)
            v0 = rng.standard_normal((2, 290)).astype(np.float32)
            wh = rng.standard_normal((2, 26, 21)).astype(np.float32)
            wu = rng.standard_normal((2, 290, 1)).astype(np.float32)
            res = {}
            for dev in ("cpu", "metal:0"):
                a = wp.array(a0, device=dev, requires_grad=True)
                v = wp.array(v0, device=dev, requires_grad=True)
                h = wp.zeros((2, 26, 21), dtype=float, device=dev, requires_grad=True)
                g = wp.zeros((2, 26, 21), dtype=float, device=dev, requires_grad=True)
                u = wp.zeros((2, 290), dtype=float, device=dev, requires_grad=True)
                loss = wp.zeros(1, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch_tiled(k, dim=2, inputs=[a, v], outputs=[h, g, u], device=dev, block_dim=64)
                    wp.launch(wloss3, dim=(2, 26, 21), inputs=[h, wp.array(wh, device=dev)], outputs=[loss], device=dev)
                    wp.launch(wloss3, dim=(2, 26, 21), inputs=[g, wp.array(-wh, device=dev)], outputs=[loss], device=dev)
                    wp.launch(wloss3, dim=(2, 290, 1), inputs=[u.reshape((2, 290, 1)), wp.array(wu, device=dev)],
                              outputs=[loss], device=dev)
                tape.backward(loss)
                res[dev] = (h.numpy(), g.numpy(), u.numpy(), a.grad.numpy(), v.grad.numpy())
            for q, what in enumerate(("h", "g", "u", "a.grad", "v.grad")):
                close(res["metal:0"][q], res["cpu"][q], what, rtol=1e-5)
            assert np.abs(res["cpu"][3]).max() > 0.1 and np.abs(res["cpu"][4]).max() > 0.1
            """
        )
        _run_with_metal_native(self, snippet, timeout=300)

    def test_tile_cholesky_adjoint_matches_cpu_and_finite_differences(self):
        # ``L = tile_cholesky(A)`` reverse: Murray (2016), lower-triangle
        # parameterization with the diagonal halved, as the native
        # ``cooperative_scalar_cholesky_adj``. Before the fix: "remaining Warp
        # intrinsic ... wp::adj_tile_cholesky". Sizes cover 1x1 (scalar tile),
        # odd sizes, and 24 / 33, whose forward kernels run the cooperative
        # factorization (the backward kernel stays serial).
        snippet = _COMMON + textwrap.dedent(
            """
            def make(n):
                @wp.kernel(module="unique")
                def k(a: wp.array3d(dtype=float), l: wp.array3d(dtype=float)):
                    i = wp.tid()
                    wp.tile_store(l[i], wp.tile_cholesky(wp.tile_load(a[i], shape=(n, n))))
                return k

            rng = np.random.default_rng(2)
            for n in (1, 2, 5, 10, 17, 24, 33):
                k = make(n)
                a0 = spd(rng, 3, n)
                w0 = rng.standard_normal((3, n, n)).astype(np.float32)
                res = {}
                for dev in ("cpu", "metal:0"):
                    a = wp.array(a0, device=dev, requires_grad=True)
                    l = wp.zeros((3, n, n), dtype=float, device=dev, requires_grad=True)
                    loss = wp.zeros(1, dtype=float, device=dev, requires_grad=True)
                    tape = wp.Tape()
                    with tape:
                        wp.launch_tiled(k, dim=3, inputs=[a], outputs=[l], device=dev, block_dim=64)
                        wp.launch(wloss3, dim=(3, n, n), inputs=[l, wp.array(w0, device=dev)], outputs=[loss], device=dev)
                    tape.backward(loss)
                    res[dev] = (l.numpy(), a.grad.numpy())
                close(res["metal:0"][0], res["cpu"][0], f"L n={n}")
                close(res["metal:0"][1], res["cpu"][1], f"dA n={n}", rtol=2e-4)
                assert np.abs(res["cpu"][1]).max() > 1e-2, n
                # Strict upper triangle gets no gradient (lower parameterization).
                assert np.all(np.triu(res["metal:0"][1][0], 1) == 0.0), n
                if n <= 10:
                    f = lambda x: float(np.sum(chol_lower(x) * w0[0]))
                    ref = fd_grad(f, a0[0].astype(np.float64), mask=np.tril(np.ones((n, n), bool)))
                    close(res["metal:0"][1][0], ref, f"dA vs FD n={n}", rtol=2e-3)
            """
        )
        _run_with_metal_native(self, snippet, timeout=600)

    def test_tile_lower_solve_forward_and_adjoint_match_cpu(self):
        # ``z = tile_lower_solve(L, y)``: the out-of-place forward used to be
        # an MSL compile error ("undeclared identifier 'tile_lower_solve'"),
        # and its reverse (``W = L^-T adj_z``, ``adj_y += W``,
        # ``adj_L -= tril(W z^T)``) had no lowering. Matrix and vector RHS.
        snippet = _COMMON + textwrap.dedent(
            """
            def make(n, m):
                @wp.kernel(module="unique")
                def k2(a: wp.array3d(dtype=float), y: wp.array3d(dtype=float), z: wp.array3d(dtype=float)):
                    i = wp.tid()
                    l = wp.tile_load(a[i], shape=(n, n))
                    wp.tile_store(z[i], wp.tile_lower_solve(l, wp.tile_load(y[i], shape=(n, m))))

                @wp.kernel(module="unique")
                def k1(a: wp.array3d(dtype=float), y: wp.array2d(dtype=float), z: wp.array2d(dtype=float)):
                    i = wp.tid()
                    l = wp.tile_load(a[i], shape=(n, n))
                    wp.tile_store(z[i], wp.tile_lower_solve(l, wp.tile_load(y[i], shape=(n,))))
                return k2 if m else k1

            rng = np.random.default_rng(3)
            for n, m in ((1, 1), (5, 3), (9, 0), (20, 7), (6, 1)):
                k = make(n, m)
                l0 = np.stack([np.linalg.cholesky(s) for s in spd(rng, 2, n).astype(np.float64)]).astype(np.float32)
                shp = (2, n, m) if m else (2, n)
                y0 = rng.standard_normal(shp).astype(np.float32)
                w0 = rng.standard_normal(shp).astype(np.float32)
                res = {}
                for dev in ("cpu", "metal:0"):
                    a = wp.array(l0, device=dev, requires_grad=True)
                    y = wp.array(y0, device=dev, requires_grad=True)
                    z = wp.zeros(shp, dtype=float, device=dev, requires_grad=True)
                    loss = wp.zeros(1, dtype=float, device=dev, requires_grad=True)
                    s3 = (2, n, m if m else 1)
                    tape = wp.Tape()
                    with tape:
                        wp.launch_tiled(k, dim=2, inputs=[a, y], outputs=[z], device=dev, block_dim=64)
                        wp.launch(wloss3, dim=s3, inputs=[z.reshape(s3), wp.array(w0.reshape(s3), device=dev)],
                                  outputs=[loss], device=dev)
                    tape.backward(loss)
                    res[dev] = (z.numpy(), a.grad.numpy(), y.grad.numpy())
                for j, what in enumerate(("z", "dL", "dy")):
                    close(res["metal:0"][j], res["cpu"][j], f"{what} n={n} m={m}", rtol=2e-4)
                assert np.abs(res["cpu"][1]).max() > 1e-2 and np.abs(res["cpu"][2]).max() > 1e-2
                ww = w0[0].reshape(n, -1)
                yy = y0[0].reshape(n, -1).astype(np.float64)
                ll = l0[0].astype(np.float64)
                ref_l = fd_grad(lambda x: float(np.sum(lower_solve(x, yy) * ww)), ll, mask=np.tril(np.ones((n, n), bool)))
                ref_y = fd_grad(lambda x: float(np.sum(lower_solve(ll, x) * ww)), yy)
                close(res["metal:0"][1][0], ref_l, f"dL vs FD n={n}", rtol=2e-3)
                close(res["metal:0"][2][0].reshape(n, -1), ref_y, f"dy vs FD n={n}", rtol=2e-3)
            """
        )
        _run_with_metal_native(self, snippet, timeout=600)

    def test_featherstone_fused_gemm_cholesky_adjoint_matches_cpu(self):
        # The shape of Newton's ``eval_dense_gemm_and_cholesky_tile``: H = sum
        # J_b^T M_b J_b accumulated with ``tile_matmul``, stored into ``H[idx]``,
        # then ``tile_cholesky(tile_diag_add(H, R))`` stored into ``L[idx]``.
        # Gradients reach J and M through the H store and the L store (the
        # latter stops at ``tile_diag_add``, which is not differentiable
        # upstream, on CPU and Metal alike).
        snippet = _COMMON + textwrap.dedent(
            """
            def make(nb, nd):
                @wp.func
                def inertia(j: wp.array2d(dtype=float), m: wp.array2d(dtype=float)):
                    h = wp.tile_zeros(shape=(nd, nd), dtype=float, storage="shared")
                    for b in range(nb):
                        mb = wp.tile_load(m, shape=(6, 6), offset=(b * 6, b * 6))
                        jb = wp.tile_load(j, shape=(6, nd), offset=(b * 6, 0))
                        wp.tile_matmul(wp.tile_transpose(jb), wp.tile_matmul(mb, jb), h)
                    return h

                @wp.kernel(module="unique")
                def k(j: wp.array3d(dtype=float), m: wp.array3d(dtype=float), arm: wp.array(dtype=float),
                      h: wp.array3d(dtype=float), l: wp.array3d(dtype=float)):
                    idx = wp.tid()
                    ht = inertia(j[idx], m[idx])
                    wp.tile_store(h[idx], ht)
                    r = wp.tile_load(arm, shape=nd, offset=0, storage="shared")
                    wp.tile_store(l[idx], wp.tile_cholesky(wp.tile_diag_add(ht, r)))
                return k

            rng = np.random.default_rng(4)
            for nb, nd in ((2, 2), (3, 10)):
                k = make(nb, nd)
                b = 3
                j0 = rng.standard_normal((b, 6 * nb, nd)).astype(np.float32)
                m0 = np.zeros((b, 6 * nb, 6 * nb), np.float32)
                for q in range(nb):
                    m0[:, q * 6:(q + 1) * 6, q * 6:(q + 1) * 6] = spd(rng, b, 6)
                arm0 = np.full(nd, 0.1, np.float32)
                wh = rng.standard_normal((b, nd, nd)).astype(np.float32)
                wl = rng.standard_normal((b, nd, nd)).astype(np.float32)
                res = {}
                for dev in ("cpu", "metal:0"):
                    jj = wp.array(j0, device=dev, requires_grad=True)
                    mm = wp.array(m0, device=dev, requires_grad=True)
                    arm = wp.array(arm0, device=dev, requires_grad=True)
                    h = wp.zeros((b, nd, nd), dtype=float, device=dev, requires_grad=True)
                    l = wp.zeros((b, nd, nd), dtype=float, device=dev, requires_grad=True)
                    loss = wp.zeros(1, dtype=float, device=dev, requires_grad=True)
                    tape = wp.Tape()
                    with tape:
                        wp.launch_tiled(k, dim=b, inputs=[jj, mm, arm], outputs=[h, l], device=dev, block_dim=64)
                        wp.launch(wloss3, dim=(b, nd, nd), inputs=[h, wp.array(wh, device=dev)], outputs=[loss], device=dev)
                        wp.launch(wloss3, dim=(b, nd, nd), inputs=[l, wp.array(wl, device=dev)], outputs=[loss], device=dev)
                    tape.backward(loss)
                    res[dev] = (h.numpy(), l.numpy(), jj.grad.numpy(), mm.grad.numpy())
                for q, what in enumerate(("H", "L", "dJ", "dM")):
                    close(res["metal:0"][q], res["cpu"][q], f"{what} nb={nb} nd={nd}", rtol=2e-4)
                assert np.abs(res["cpu"][2]).max() > 1e-2
            """
        )
        _run_with_metal_native(self, snippet, timeout=600)

    def test_tile_upper_and_cholesky_solve_match_cpu(self):
        # ``tile_upper_solve`` (out of place) forward was an MSL compile error.
        # ``tile_upper_solve`` and ``tile_cholesky_solve`` are not
        # differentiable upstream (``is_differentiable=False``): no reverse
        # statement is emitted, so their inputs get no gradient on CPU, CUDA
        # and Metal alike. Pinned here so a future Metal adjoint would be a
        # deliberate divergence.
        snippet = _COMMON + textwrap.dedent(
            """
            def make(n, m):
                @wp.kernel(module="unique")
                def ku(a: wp.array3d(dtype=float), y: wp.array3d(dtype=float), z: wp.array3d(dtype=float)):
                    i = wp.tid()
                    u = wp.tile_transpose(wp.tile_load(a[i], shape=(n, n)))
                    wp.tile_store(z[i], wp.tile_upper_solve(u, wp.tile_load(y[i], shape=(n, m))))

                @wp.kernel(module="unique")
                def kc(a: wp.array3d(dtype=float), y: wp.array3d(dtype=float), z: wp.array3d(dtype=float)):
                    i = wp.tid()
                    l = wp.tile_load(a[i], shape=(n, n))
                    wp.tile_store(z[i], wp.tile_cholesky_solve(l, wp.tile_load(y[i], shape=(n, m))))
                return ku, kc

            rng = np.random.default_rng(5)
            for n, m in ((1, 1), (6, 2), (9, 1)):
                l0 = np.stack([np.linalg.cholesky(s) for s in spd(rng, 2, n).astype(np.float64)]).astype(np.float32)
                y0 = rng.standard_normal((2, n, m)).astype(np.float32)
                w0 = rng.standard_normal((2, n, m)).astype(np.float32)
                for k in make(n, m):
                    res = {}
                    for dev in ("cpu", "metal:0"):
                        a = wp.array(l0, device=dev, requires_grad=True)
                        y = wp.array(y0, device=dev, requires_grad=True)
                        z = wp.zeros((2, n, m), dtype=float, device=dev, requires_grad=True)
                        loss = wp.zeros(1, dtype=float, device=dev, requires_grad=True)
                        tape = wp.Tape()
                        with tape:
                            wp.launch_tiled(k, dim=2, inputs=[a, y], outputs=[z], device=dev, block_dim=64)
                            wp.launch(wloss3, dim=(2, n, m), inputs=[z, wp.array(w0, device=dev)], outputs=[loss], device=dev)
                        tape.backward(loss)
                        res[dev] = (z.numpy(), a.grad.numpy(), y.grad.numpy())
                    for q, what in enumerate(("z", "dA", "dy")):
                        close(res["metal:0"][q], res["cpu"][q], f"{what} {k.key} n={n}", rtol=2e-4)
                    assert not np.any(res["cpu"][1]) and not np.any(res["cpu"][2])
            """
        )
        _run_with_metal_native(self, snippet, timeout=600)

    def test_tile_cholesky_upper_reverse_is_loud(self):
        # ``fill_mode="upper"`` is not lowered on Metal (forward or reverse);
        # a backward launch must raise, never return a wrong gradient.
        snippet = _COMMON + textwrap.dedent(
            """
            from warp._src.codegen_metal import MetalCodegenError

            @wp.kernel(module="unique")
            def k(a: wp.array3d(dtype=float), l: wp.array3d(dtype=float)):
                i = wp.tid()
                wp.tile_store(l[i], wp.tile_cholesky(wp.tile_load(a[i], shape=(4, 4)), fill_mode="upper"))

            dev = "metal:0"
            a = wp.array(spd(np.random.default_rng(6), 2, 4), device=dev, requires_grad=True)
            l = wp.zeros((2, 4, 4), dtype=float, device=dev, requires_grad=True)
            try:
                tape = wp.Tape()
                with tape:
                    wp.launch_tiled(k, dim=2, inputs=[a], outputs=[l], device=dev, block_dim=64)
                l.grad.fill_(1.0)
                tape.backward()
                wp.synchronize()
            except MetalCodegenError as e:
                assert "upper" in str(e), e
            else:
                raise AssertionError("fill_mode='upper' tile_cholesky ran on Metal")
            """
        )
        _run_with_metal_native(self, snippet, timeout=300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
