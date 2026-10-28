# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Threadgroup-resident tiles for the reshape / indexed-load / in-place Cholesky idiom.

mujoco_warp (v3.15) factorizes each dense block of the mass matrix with
``tile_load_indexed`` -> ``tile_reshape`` -> ``tile_cholesky_inplace(fill_mode="upper")``
-> ``tile_store(tile_reshape(...))`` -> ``tile_cholesky_solve(fill_mode="upper")``.
Without promotion every one of the 32 cooperating lanes held private copies of the
N*N tile (and of the N*N index tile) and ran the factorization redundantly; at
N=35 that kernel took ~2.3 ms per world-batch at one world and was dispatched
outside the ICB because of its private memory.

Each test compares against CPU, checks the result is bitwise equal to the private
path (``warp.config.metal_threadgroup_tiles = False``), and inspects the MSL so a
silent fallback to the private path fails the test.
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_native

_COMMON = textwrap.dedent(
    """
    import numpy as np
    from warp._src import metal_dispatch as md

    def spd_blocks(nw, n, seed):
        rng = np.random.default_rng(seed)
        out = []
        for w in range(nw):
            b = rng.standard_normal((n, n)).astype(np.float32) * 0.3
            out.append(b @ b.T + n * np.eye(n, dtype=np.float32) * (1.0 + 0.01 * w))
        return np.stack(out)

    def msl(k):
        a = k._metal_artifact
        return a.source + (a.header or "")

    def run_both(make, launch):
        # One module per path: identical kernels in one module would share a
        # kernel object (and its artifact, built on first launch).
        res = {}
        for name, dev, flag in (("cpu", "cpu", True), ("on", "metal:0", True), ("off", "metal:0", False)):
            wp.config.metal_threadgroup_tiles = flag
            k = make(name)
            res[name] = launch(k, dev)
            if dev != "cpu":
                res[name + "_src"] = msl(k)
                res[name + "_direct"] = len(md.get_dispatcher()._direct_only)
        wp.config.metal_threadgroup_tiles = True
        return res
    """
)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR10ResidentTiles(unittest.TestCase):
    def test_indexed_reshape_upper_factor_solve_matches_cpu_and_private_path(self):
        # The mujoco_warp v3.15 ``_tile_cholesky_factorize_solve_block`` kernel
        # (N=35, upper fill, CSR gather through an index tile).
        snippet = _COMMON + textwrap.dedent(
            """
            N = 35
            A = N * N

            def make(tag):
                @wp.kernel(module=f"r10_make_{tag}", enable_backward=False)
                def k(M_in: wp.array2d[float], elemid: wp.array[int], y: wp.array2d[float],
                      x: wp.array2d[float], L_out: wp.array2d[float]):
                    w, blk = wp.tid()
                    idx = wp.tile_load(elemid, shape=(A,), offset=(blk * A,), storage="shared")
                    block = wp.tile_load_indexed(M_in[w], idx, shape=(A,), storage="shared")
                    L = wp.tile_reshape(block, (N, N))
                    wp.tile_cholesky_inplace(L, fill_mode="upper")
                    wp.tile_store(L_out[w], wp.tile_reshape(L, (A,)), offset=(0,))
                    rhs = wp.tile_load(y[w], shape=(N,), offset=(0,))
                    sol = wp.tile_cholesky_solve(L, rhs, fill_mode="upper")
                    wp.tile_store(x[w], sol, offset=(0,))
                return k

            nw = 5
            S = spd_blocks(nw, N, 3)
            tri = [(i, j) for i in range(N) for j in range(i, N)]
            pos = {t: p for p, t in enumerate(tri)}
            Mp = np.stack([[S[w][i, j] for (i, j) in tri] for w in range(nw)]).astype(np.float32)
            # Upper triangle from the packed storage, lower slots out of bounds (read as 0).
            eid = np.array([pos[(i, j)] if j >= i else 10**6 for i in range(N) for j in range(N)], np.int32)
            ynp = np.random.default_rng(4).standard_normal((nw, N)).astype(np.float32)

            def launch(k, dev):
                x = wp.zeros((nw, N), dtype=float, device=dev)
                L = wp.zeros((nw, A), dtype=float, device=dev)
                wp.launch_tiled(k, dim=(nw, 1), inputs=[wp.array(Mp, device=dev), wp.array(eid, device=dev),
                                wp.array(ynp, device=dev)], outputs=[x, L], block_dim=32, device=dev)
                return x.numpy(), L.numpy()

            r = run_both(make, launch)
            for i in range(2):
                np.testing.assert_allclose(r["on"][i], r["cpu"][i], atol=5e-6, rtol=1e-5)
                np.testing.assert_array_equal(r["on"][i], r["off"][i])
            Lu = r["on"][1].reshape(nw, N, N)
            assert np.all(np.tril(Lu, -1) == 0.0)
            for w in range(nw):
                np.testing.assert_allclose(Lu[w].T.astype(np.float64) @ Lu[w], S[w], rtol=1e-4, atol=1e-4)
            src = r["on_src"]
            assert "wp_tile_smem_arena" in src and "_cholesky_inplace_smem_upper" in src, "not resident"
            assert "_cholesky_solve_smem_upper_1" in src, "solve did not read the resident factor"
            assert "c[1225]" not in src.split("[[kernel]]")[-1], "a 1225-element private tile is still declared"
            assert "(threadgroup int*)" in src, "index tile not resident"
            assert "wp_tile_smem_arena" not in r["off_src"]
            assert r["on_direct"] == 0, "resident kernel still dispatched outside the ICB"
            """
        )
        _run_with_metal_native(self, snippet, timeout=300)

    def test_lower_inplace_factor_and_reshaped_solve_match_cpu_and_private_path(self):
        # Lower fill, plain 2-D load, the in-place factor stored and solved
        # through a reshape view, one kernel; plus a solve-only kernel that
        # loads a stored factor as a flat tile and reshapes it.
        snippet = _COMMON + textwrap.dedent(
            """
            N = 28
            A = N * N

            def make_factor(tag):
                @wp.kernel(module=f"r10_make_factor_{tag}", enable_backward=False)
                def kf(Ain: wp.array3d[float], y: wp.array2d[float], x: wp.array2d[float], L_out: wp.array2d[float]):
                    w = wp.tid()
                    a = wp.tile_load(Ain[w], shape=(N, N), storage="shared")
                    wp.tile_cholesky_inplace(a)
                    flat = wp.tile_reshape(a, (A,))
                    wp.tile_store(L_out[w], flat)
                    rhs = wp.tile_load(y[w], shape=(N,))
                    sol = wp.tile_cholesky_solve(a, rhs)
                    wp.tile_store(x[w], sol)
                return kf

            def make_solve(tag):
                @wp.kernel(module=f"r10_make_solve_{tag}", enable_backward=False)
                def ks(L_in: wp.array2d[float], y: wp.array2d[float], x: wp.array2d[float]):
                    w = wp.tid()
                    L = wp.tile_reshape(wp.tile_load(L_in[w], shape=(A,), storage="shared"), (N, N))
                    rhs = wp.tile_load(y[w], shape=(N,))
                    wp.tile_store(x[w], wp.tile_cholesky_solve(L, rhs))
                return ks

            nw = 4
            S = spd_blocks(nw, N, 9)
            ynp = np.random.default_rng(10).standard_normal((nw, N)).astype(np.float32)

            def launch_f(k, dev):
                x = wp.zeros((nw, N), dtype=float, device=dev)
                L = wp.zeros((nw, A), dtype=float, device=dev)
                wp.launch_tiled(k, dim=(nw,), inputs=[wp.array(S, device=dev), wp.array(ynp, device=dev)],
                                outputs=[x, L], block_dim=32, device=dev)
                return x.numpy(), L.numpy()

            rf = run_both(make_factor, launch_f)
            for i in range(2):
                np.testing.assert_allclose(rf["on"][i], rf["cpu"][i], atol=5e-6, rtol=1e-5)
                np.testing.assert_array_equal(rf["on"][i], rf["off"][i])
            assert "_cholesky_inplace_smem(" in rf["on_src"] and "_cholesky_solve_smem_1" in rf["on_src"]

            Lst = rf["cpu"][1]

            def launch_s(k, dev):
                x = wp.zeros((nw, N), dtype=float, device=dev)
                wp.launch_tiled(k, dim=(nw,), inputs=[wp.array(Lst, device=dev), wp.array(ynp, device=dev)],
                                outputs=[x], block_dim=32, device=dev)
                return x.numpy()

            rs = run_both(make_solve, launch_s)
            np.testing.assert_allclose(rs["on"], rs["cpu"], atol=5e-6, rtol=1e-5)
            np.testing.assert_array_equal(rs["on"], rs["off"])
            np.testing.assert_allclose(rs["on"], rf["cpu"][0], atol=5e-6, rtol=1e-5)
            assert "wp_tile_smem_arena" in rs["on_src"] and "_cholesky_solve_smem_1" in rs["on_src"]
            """
        )
        _run_with_metal_native(self, snippet, timeout=300)

    def test_unsupported_uses_stay_private_and_correct(self):
        # (a) the index tile is read again after the indexed load -> it must
        # not share the slot (the group may still be resident); (b) a member
        # is read by tile_extract -> the whole group stays private; (c) the
        # reshape source is used after the in-place factorization through
        # the view -> fine on CUDA (alias), and must match here too.
        snippet = _COMMON + textwrap.dedent(
            """
            N = 30
            A = N * N

            def make_a(tag):
                @wp.kernel(module=f"r10_make_a_{tag}", enable_backward=False)
                def ka(M_in: wp.array2d[float], elemid: wp.array[int], L_out: wp.array2d[float],
                       idx_out: wp.array2d[int]):
                    w = wp.tid()
                    idx = wp.tile_load(elemid, shape=(A,), storage="shared")
                    block = wp.tile_load_indexed(M_in[w], idx, shape=(A,), storage="shared")
                    L = wp.tile_reshape(block, (N, N))
                    wp.tile_cholesky_inplace(L)
                    wp.tile_store(L_out[w], block)
                    wp.tile_store(idx_out[w], idx)
                return ka

            def make_b(tag):
                @wp.kernel(module=f"r10_make_b_{tag}", enable_backward=False)
                def kb(Ain: wp.array3d[float], L_out: wp.array2d[float], d_out: wp.array[float]):
                    w = wp.tid()
                    a = wp.tile_load(Ain[w], shape=(N, N), storage="shared")
                    wp.tile_cholesky_inplace(a)
                    wp.tile_store(L_out[w], wp.tile_reshape(a, (A,)))
                    d_out[w] = wp.tile_extract(a, 3, 3)
                return kb

            nw = 3
            S = spd_blocks(nw, N, 21)
            Mfull = S.reshape(nw, A).astype(np.float32)
            eid = np.arange(A, dtype=np.int32)

            def launch_a(k, dev):
                L = wp.zeros((nw, A), dtype=float, device=dev)
                io = wp.zeros((nw, A), dtype=int, device=dev)
                wp.launch_tiled(k, dim=(nw,), inputs=[wp.array(Mfull, device=dev), wp.array(eid, device=dev)],
                                outputs=[L, io], block_dim=32, device=dev)
                return L.numpy(), io.numpy()

            ra = run_both(make_a, launch_a)
            np.testing.assert_allclose(ra["on"][0], ra["cpu"][0], atol=5e-6, rtol=1e-5)
            np.testing.assert_array_equal(ra["on"][1], ra["cpu"][1])
            np.testing.assert_array_equal(ra["on"][0], ra["off"][0])
            assert "(threadgroup int*)" not in ra["on_src"], "a live index tile shared the slot"

            def launch_b(k, dev):
                L = wp.zeros((nw, A), dtype=float, device=dev)
                d = wp.zeros(nw, dtype=float, device=dev)
                wp.launch_tiled(k, dim=(nw,), inputs=[wp.array(S, device=dev)], outputs=[L, d],
                                block_dim=32, device=dev)
                return L.numpy(), d.numpy()

            rb = run_both(make_b, launch_b)
            for i in range(2):
                np.testing.assert_allclose(rb["on"][i], rb["cpu"][i], atol=5e-6, rtol=1e-5)
                np.testing.assert_array_equal(rb["on"][i], rb["off"][i])
            assert "_cholesky_inplace_smem" not in rb["on_src"], "a tile read by tile_extract was promoted"
            """
        )
        _run_with_metal_native(self, snippet, timeout=300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
