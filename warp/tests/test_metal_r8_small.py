# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal-vs-CPU checks for round-8 fixes.

* Reverse-mode ``tile_load`` / ``tile_store`` on a partial edge tile: the
  gradient scatter/gather must skip the elements outside the array, as the
  forward pass does.

Each case runs in a subprocess (see ``test_metal_launch``).
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_enabled

_TILE_ADJ_PRELUDE = textwrap.dedent(
    """
    import numpy as np

    T = 16

    @wp.kernel
    def k_transpose(x: wp.array2d[float], y: wp.array2d[float]):
        t = wp.tile_load(x, shape=(T, T), offset=(0, 0))
        wp.tile_store(y, wp.tile_transpose(t), offset=(0, 0))

    @wp.kernel
    def k_add_bcast(x: wp.array2d[float], b: wp.array2d[float], y: wp.array2d[float]):
        t = wp.tile_load(x, shape=(T, T), offset=(0, 0))
        t += wp.tile_broadcast(wp.tile_load(b, shape=(T, 1), offset=(0, 0)), shape=(T, T))
        wp.tile_store(y, t, offset=(0, 0))

    @wp.kernel
    def k_matmul_tr(a: wp.array2d[float], x: wp.array2d[float], y: wp.array2d[float]):
        A = wp.tile_load(a, shape=(T, T), offset=(0, 0))
        X = wp.tile_load(x, shape=(T, T), offset=(0, 0))
        C = wp.tile_zeros(shape=(T, T), dtype=float)
        wp.tile_matmul(A, wp.tile_transpose(X), C)
        wp.tile_store(y, C, offset=(0, 0))

    @wp.kernel
    def k_scale_1d(x: wp.array[float], y: wp.array[float]):
        i = wp.tid()
        t = wp.tile_load(x, shape=(T,), offset=(i * T,))
        wp.tile_store(y, t * 3.0, offset=(i * T,))

    # warp_nn's Linear layer: Y^T = W X^T + b over partial (T, T) tiles.
    @wp.kernel
    def k_linear(x: wp.array2d[float], w: wp.array2d[float], b: wp.array2d[float], y: wp.array2d[float]):
        i, j = wp.tid()
        C = wp.tile_zeros(shape=(T, T), dtype=float)
        a = wp.tile_load(w, shape=(T, T), offset=(j * T, 0))
        xt = wp.tile_load(x, shape=(T, T), offset=(i * T, 0))
        wp.tile_matmul(a, wp.tile_transpose(xt), C)
        C += wp.tile_broadcast(wp.tile_load(b, shape=(T, 1), offset=(j * T, 0)), shape=(T, T))
        wp.tile_store(y, wp.tile_transpose(C), offset=(i * T, j * T))

    def grads(dev, kern, in_shapes, out_shape, dim, block_dim):
        rng = np.random.default_rng(7)
        ins = [wp.array(rng.standard_normal(s).astype(np.float32), device=dev, requires_grad=True) for s in in_shapes]
        y = wp.zeros(out_shape, device=dev, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch_tiled(kern, dim=dim, inputs=ins + [y], device=dev, block_dim=block_dim)
        seed = np.arange(1, int(np.prod(out_shape)) + 1, dtype=np.float32).reshape(out_shape)
        y.grad = wp.array(seed, device=dev)
        tape.backward()
        return [y.numpy()] + [a.grad.numpy() for a in ins] + [y.grad.numpy()]

    def check(kern, in_shapes, out_shape, dim=1, block_dims=(32, 64, 256)):
        ref = grads("cpu", kern, in_shapes, out_shape, dim, 64)
        for bd in block_dims:
            got = grads("metal:0", kern, in_shapes, out_shape, dim, bd)
            for k, (r, g) in enumerate(zip(ref, got)):
                np.testing.assert_allclose(g, r, rtol=1e-5, atol=1e-5, err_msg=f"{kern.key} bd={bd} output {k}")
    """
)


def _run(test_case, body, timeout=600):
    _run_with_metal_enabled(test_case, _TILE_ADJ_PRELUDE + textwrap.dedent(body), timeout=timeout)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR8TileAdjointEdge(unittest.TestCase):
    def test_partial_tile_transpose_grad(self):
        _run(self, "check(k_transpose, [(3, 2)], (2, 3))\ncheck(k_transpose, [(T, T)], (T, T))\n")

    def test_partial_tile_broadcast_grad(self):
        _run(self, "check(k_add_bcast, [(3, 2), (3, 1)], (3, 2))\n")

    def test_partial_tile_matmul_grad(self):
        _run(self, "check(k_matmul_tr, [(3, 2), (4, 2)], (3, 4))\n")

    def test_partial_tile_1d_grad(self):
        _run(self, "check(k_scale_1d, [(21,)], (21,), dim=2)\n")

    def test_partial_tile_linear_layer_grad(self):
        _run(
            self,
            "check(k_linear, [(20, 7), (3, 7), (3, 1)], (20, 3), dim=(2, 1))\ncheck(k_linear, [(2, 2), (1, 2), (1, 1)], (2, 1))\n",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
