# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metal-vs-CPU checks for gaps found running Newton's Kamino solver on Metal.

Every case runs on ``cpu`` and ``metal:0`` and the results must agree. The
Metal device must be enabled before ``wp.init()``, so each case runs in a
subprocess (see ``test_metal_launch``).
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_native

_PRELUDE = textwrap.dedent(
    """
    import numpy as np

    def both(fn):
        return {dev: fn(dev) for dev in ("cpu", "metal:0")}

    def assert_same(res, rtol=1e-5, atol=1e-5):
        a, b = res["cpu"], res["metal:0"]
        if not isinstance(a, (tuple, list)):
            a, b = [a], [b]
        for x, y in zip(a, b):
            np.testing.assert_allclose(np.asarray(y), np.asarray(x), rtol=rtol, atol=atol)
    """
)


def _run(test_case, body: str, timeout: int = 600):
    _run_with_metal_native(test_case, _PRELUDE + textwrap.dedent(body), timeout=timeout)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR4Kamino(unittest.TestCase):
    def test_blocked_cholesky_64_transposes(self):
        # Kamino's blocked LLT at block_size=64: transposed matmul operands and
        # ``transpose(solve(transpose(A)))``. Materialising every transpose
        # overflowed the per-thread stack and crashed the GPU compiler.
        _run(
            self,
            """
            B = 64

            @wp.kernel
            def factor(A: wp.array2d[wp.float32], L: wp.array2d[wp.float32], n: int):
                _b, _lane = wp.tid()
                for k in range(0, n, B):
                    akk = wp.tile_load(A, shape=(B, B), offset=(k, k), storage="shared")
                    for j in range(0, k, B):
                        lb = wp.tile_load(L, shape=(B, B), offset=(k, j))
                        lbt = wp.tile_transpose(lb)
                        wp.tile_matmul(lb, lbt, akk, alpha=-1.0)
                    wp.tile_cholesky_inplace(akk)
                    wp.tile_store(L, akk, offset=(k, k))
                    for i in range(k + B, n, B):
                        aik = wp.tile_load(A, shape=(B, B), offset=(i, k), storage="shared")
                        for j in range(0, k, B):
                            l1 = wp.tile_load(L, shape=(B, B), offset=(i, j))
                            l2 = wp.tile_load(L, shape=(B, B), offset=(k, j))
                            l2t = wp.tile_transpose(l2)
                            wp.tile_matmul(l1, l2t, aik, alpha=-1.0)
                        t = wp.tile_transpose(aik)
                        wp.tile_lower_solve_inplace(akk, t)
                        sol = wp.tile_transpose(t)
                        wp.tile_store(L, sol, offset=(i, k))

            n = 2 * B
            rng = np.random.default_rng(3)
            d = rng.standard_normal((n, n)).astype(np.float32)
            a_np = d @ d.T + n * np.eye(n, dtype=np.float32)

            def run(dev):
                a = wp.array(a_np, device=dev)
                l = wp.zeros_like(a)
                wp.launch_tiled(factor, dim=1, inputs=[a, l, n], block_dim=1, device=dev)
                return np.tril(l.numpy())

            res = both(run)
            assert_same(res, rtol=1e-4, atol=1e-4)
            lm = res["metal:0"]
            np.testing.assert_allclose(lm @ lm.T, a_np, rtol=1e-4, atol=1e-2)
            """,
        )

    def test_blocked_cholesky_64_upper_factor(self):
        # Kamino's RCM variant: an upper-mode factor read through its
        # transpose as the solve's L and as a tile_store source, plus a
        # zero tile that is only stored.
        _run(
            self,
            """
            B = 64

            @wp.kernel
            def factor(A: wp.array2d[wp.float32], L: wp.array2d[wp.float32], n: int, skip: int):
                _b, _lane = wp.tid()
                for k in range(0, n, B):
                    akk = wp.tile_load(A, shape=(B, B), offset=(k, k), storage="shared")
                    for j in range(0, k, B):
                        lb = wp.tile_load(L, shape=(B, B), offset=(k, j))
                        wp.tile_matmul(lb, wp.tile_transpose(lb), akk, alpha=-1.0)
                    wp.tile_cholesky_inplace(akk, fill_mode="upper")
                    lkk = wp.tile_transpose(akk)
                    wp.tile_store(L, lkk, offset=(k, k))
                    for i in range(k + B, n, B):
                        if i // B == skip:
                            zeros = wp.tile_zeros(shape=(B, B), dtype=wp.float32)
                            wp.tile_store(L, zeros, offset=(i, k))
                            continue
                        aik = wp.tile_load(A, shape=(B, B), offset=(i, k), storage="shared")
                        for j in range(0, k, B):
                            l1 = wp.tile_load(L, shape=(B, B), offset=(i, j))
                            l2 = wp.tile_load(L, shape=(B, B), offset=(k, j))
                            wp.tile_matmul(l1, wp.tile_transpose(l2), aik, alpha=-1.0)
                        t = wp.tile_transpose(aik)
                        wp.tile_lower_solve_inplace(lkk, t)
                        wp.tile_store(L, wp.tile_transpose(t), offset=(i, k))

            n = 2 * B
            rng = np.random.default_rng(5)
            d = rng.standard_normal((n, n)).astype(np.float32)
            a_np = d @ d.T + n * np.eye(n, dtype=np.float32)

            def run(dev, skip):
                a = wp.array(a_np, device=dev)
                l = wp.full((n, n), 7.0, dtype=wp.float32, device=dev)
                wp.launch_tiled(factor, dim=1, inputs=[a, l, n, skip], block_dim=1, device=dev)
                return np.tril(l.numpy())

            res = both(lambda dev: run(dev, -1))
            assert_same(res, rtol=1e-4, atol=1e-4)
            lm = res["metal:0"]
            np.testing.assert_allclose(lm @ lm.T, a_np, rtol=1e-4, atol=1e-2)
            res = both(lambda dev: run(dev, 1))
            assert_same(res, rtol=1e-4, atol=1e-4)
            np.testing.assert_array_equal(res["metal:0"][B:, :B], 0.0)
            """,
        )

    def test_transposed_matmul_operands(self):
        # ``tile_matmul(transpose(a), transpose(b), c)`` with rectangular tiles,
        # and a transpose read by ``tile_store`` after a double transpose.
        _run(
            self,
            """
            @wp.kernel
            def k(a: wp.array2d[wp.float32], b: wp.array2d[wp.float32], c: wp.array2d[wp.float32],
                  d: wp.array2d[wp.float32]):
                _i, _lane = wp.tid()
                ta = wp.tile_load(a, shape=(3, 5))
                tb = wp.tile_load(b, shape=(4, 3))
                tc = wp.tile_zeros(shape=(5, 4), dtype=wp.float32)
                wp.tile_matmul(wp.tile_transpose(ta), wp.tile_transpose(tb), tc)
                wp.tile_store(c, tc)
                tt = wp.tile_transpose(ta)
                wp.tile_store(d, wp.tile_transpose(tt))

            rng = np.random.default_rng(0)
            a_np = rng.standard_normal((3, 5)).astype(np.float32)
            b_np = rng.standard_normal((4, 3)).astype(np.float32)

            def run(dev):
                a = wp.array(a_np, device=dev)
                b = wp.array(b_np, device=dev)
                c = wp.zeros((5, 4), dtype=wp.float32, device=dev)
                d = wp.zeros((3, 5), dtype=wp.float32, device=dev)
                wp.launch_tiled(k, dim=1, inputs=[a, b, c, d], block_dim=1, device=dev)
                return c.numpy(), d.numpy()

            res = both(run)
            assert_same(res)
            np.testing.assert_allclose(res["metal:0"][0], a_np.T @ b_np.T, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(res["metal:0"][1], a_np)
            """,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
