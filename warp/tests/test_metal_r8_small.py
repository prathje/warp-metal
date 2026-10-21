# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal-vs-CPU checks for round-8 fixes.

* Reverse-mode ``tile_load`` / ``tile_store`` on a partial edge tile: the
  gradient scatter/gather must skip the elements outside the array, as the
  forward pass does.
* A ``tile_transpose`` copy read element-wise by a user ``tile_map`` in the
  reverse pass (warp_nn's LSTM cell) is a read, not a write that would make
  the copy stale.
* ``bsr_set_transpose`` (compact) builds its topology from device dispatches
  on Metal, so it matches the CPU transpose and records into a graph capture.
* Metal's Python-side hash-grid build, BVH/mesh refit and winding-number
  precompute stand in for native CUDA calls and must not be recorded on an
  active ``wp.Tape``.
* ``metal_fp64 = "emulate"`` evaluates float64 transcendentals in float32:
  Warp warns once per kernel that does so (also on a warm artifact-cache load).

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

    @wp.func
    def cell_fn(c: float, a: float, b: float):
        return wp.tanh(a) * c + b * b

    # warp_nn LSTMCell shape: a transposed load feeds a user tile_map, the
    # result is stored transposed.
    @wp.kernel
    def k_transpose_tile_map(x: wp.array2d[float], a: wp.array2d[float], b: wp.array2d[float], y: wp.array2d[float]):
        tc = wp.tile_transpose(wp.tile_load(x, shape=(T, T), offset=(0, 0)))
        ta = wp.tile_load(a, shape=(T, T), offset=(0, 0))
        tb = wp.tile_load(b, shape=(T, T), offset=(0, 0))
        c = wp.tile_map(cell_fn, tc, ta, tb)
        wp.tile_store(y, wp.tile_transpose(c), offset=(0, 0))

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

    def test_transpose_copy_read_by_tile_map_grad(self):
        _run(
            self,
            "check(k_transpose_tile_map, [(T, T), (T, T), (T, T)], (T, T))\n"
            "check(k_transpose_tile_map, [(5, 3), (3, 5), (3, 5)], (5, 3))\n",
        )

    def test_partial_tile_linear_layer_grad(self):
        _run(
            self,
            "check(k_linear, [(20, 7), (3, 7), (3, 1)], (20, 3), dim=(2, 1))\ncheck(k_linear, [(2, 2), (1, 2), (1, 1)], (2, 1))\n",
        )


_TILE_EDGE_BODY = textwrap.dedent(
    """
    import numpy as np

    TR, TC = 8, 4

    @wp.kernel(module="unique")
    def k_atomic(x: wp.array2d[float], y: wp.array2d[float], prev: wp.array2d[float]):
        i, j = wp.tid()
        t = wp.tile_load(x, shape=(TR, TC), offset=(i * TR, j * TC))
        p = wp.tile_atomic_add(y, t + wp.tile_ones(shape=(TR, TC), dtype=float), offset=(i * TR, j * TC))
        wp.tile_store(prev, p, offset=(i * TR, j * TC))

    @wp.kernel(module="unique")
    def k_load11(x: wp.array2d[float], y: wp.array2d[float]):
        i, j = wp.tid()
        s = wp.tile_sum(wp.tile_load(x, shape=(1, 1), offset=(i, j)))
        wp.tile_atomic_add(y, wp.tile_ones(shape=(1, 1), dtype=float) * s[0], offset=(0, 0))

    @wp.kernel(module="unique")
    def k_lead_sum(x: wp.array3d[float], y: wp.array[float]):
        b = wp.tid()
        wp.tile_atomic_add(y, wp.tile_sum(wp.tile_load(x, shape=(1, TR, TC), offset=(b, 0, 0))), offset=(0,))

    @wp.kernel(module="unique")
    def k_vec_sum(x: wp.array[wp.vec3], y: wp.array[float]):
        i = wp.tid()
        s = wp.tile_reduce(wp.add, wp.tile_load(x, shape=(TR,), offset=(i * TR,)))
        wp.tile_atomic_add(y, wp.tile_ones(shape=(1,), dtype=float) * wp.dot(s[0], wp.vec3(1.0)), offset=(0,))

    @wp.kernel(module="unique")
    def k_lead_sq(x: wp.array3d[float], y: wp.array3d[float]):
        b, i = wp.tid()
        t = wp.tile_load(x, shape=(1, TR, TC), offset=(b, i * TR, 0))
        wp.tile_store(y, t * t, offset=(b, i * TR, 0))

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 9)).astype(np.float32)
    X3 = rng.standard_normal((6, 13, TC)).astype(np.float32)
    V = rng.standard_normal((40, 3)).astype(np.float32)

    # Every array is a view into a larger buffer, so an element written or
    # read past the view (aliasing the next row or slab) is observable.
    def atomic(dev):
        x = wp.array(X, device=dev)[:13, :7]
        yb, pb = wp.zeros((20, 9), device=dev), wp.zeros((20, 9), device=dev)
        yb[:13, :7].fill_(0.5)
        wp.launch_tiled(k_atomic, dim=(2, 2), inputs=[x, yb[:13, :7], pb[:13, :7]], device=dev, block_dim=32)
        return yb.numpy(), pb.numpy()

    def load11(dev):
        y = wp.zeros((1, 1), device=dev)
        wp.launch_tiled(k_load11, dim=(4, 3), inputs=[wp.array(X, device=dev)[:3, :2], y], device=dev, block_dim=32)
        return (y.numpy(),)

    def lead_sum(dev):
        y = wp.zeros(1, device=dev)
        wp.launch_tiled(k_lead_sum, dim=5, inputs=[wp.array(X3[:, :TR], device=dev)[:3], y], device=dev, block_dim=32)
        return (y.numpy(),)

    def vec_sum(dev):
        y = wp.zeros(1, device=dev)
        x = wp.array(V, dtype=wp.vec3, device=dev)[:21]
        wp.launch_tiled(k_vec_sum, dim=3, inputs=[x, y], device=dev, block_dim=32)
        return (y.numpy(),)

    def lead_grad(dev):
        xb = wp.array(X3, device=dev, requires_grad=True)
        yb = wp.zeros(X3.shape, device=dev, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch_tiled(k_lead_sq, dim=(5, 2), inputs=[xb[:3], yb[:3]], device=dev, block_dim=32)
        yb.grad.fill_(1.0)
        tape.backward()
        return yb.numpy(), xb.grad.numpy(), yb.grad.numpy()

    for fn in (atomic, load11, lead_sum, vec_sum, lead_grad):
        for r, g in zip(fn("cpu"), fn("metal:0")):
            np.testing.assert_allclose(g, r, rtol=1e-5, atol=1e-5, err_msg=fn.__name__)
    """
)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR8TileEdgeBounds(unittest.TestCase):
    def test_partial_tile_atomic_load_and_leading_index_bounds(self):
        _run_with_metal_enabled(self, _TILE_EDGE_BODY, timeout=600)


_TAPE_BODY = textwrap.dedent(
    """
    import io
    import contextlib
    import numpy as np

    @wp.kernel
    def square(x: wp.array[wp.vec3], y: wp.array[float]):
        i = wp.tid()
        y[i] = wp.dot(x[i], x[i])

    dev = "metal:0"
    rng = np.random.default_rng(0)
    pts = wp.array(rng.uniform(0.0, 2.0, (64, 3)).astype(np.float32), dtype=wp.vec3, device=dev, requires_grad=True)
    y = wp.zeros(64, device=dev, requires_grad=True)
    grid = wp.HashGrid(8, 8, 8, device=dev)
    tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32).reshape(-1)
    quad = wp.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=wp.vec3, device=dev)
    mesh = wp.Mesh(points=quad, indices=wp.array(tri, dtype=int, device=dev), support_winding_number=True)

    tape = wp.Tape()
    with tape:
        grid.build(pts, 0.25)
        mesh.refit()
        wp.launch(square, dim=64, inputs=[pts], outputs=[y], device=dev)
    keys = [launch[0].key for launch in tape.launches if not callable(launch)]
    assert keys == [square.key], keys

    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(err):
        tape.backward(grads={y: wp.ones(64, device=dev)})
    assert "incorrect gradients" not in err.getvalue(), err.getvalue()
    np.testing.assert_allclose(pts.grad.numpy(), 2.0 * pts.numpy(), rtol=1e-6)
    """
)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR8InternalLaunchesOffTape(unittest.TestCase):
    def test_hash_grid_build_and_mesh_refit_not_recorded(self):
        _run_with_metal_enabled(self, _TAPE_BODY, timeout=600)


_BSR_TRANSPOSE_BODY = 'import numpy as np\nimport warp.sparse as sp\nM = "metal:0"\n\ndef build(dev, nrow, ncol, block, n, seed, topology=None, empty_row=None):\n    rng = np.random.default_rng(seed)\n    # Unique (row, col) pairs: duplicate triplets are summed in a device-dependent order.\n    cells = np.array([(r, c) for r in range(nrow) for c in range(ncol) if r != empty_row])\n    pick = cells[rng.choice(len(cells), size=min(n, len(cells)), replace=False)]\n    n = len(pick)\n    rows, cols = pick[:, 0].astype(np.int32), pick[:, 1].astype(np.int32)\n    btype = float if block == (1, 1) else wp.types.matrix(shape=block, dtype=float)\n    vals = rng.standard_normal((n, *block)).astype(np.float32).reshape((n,) if block == (1, 1) else (n, *block))\n    if topology == "padded":\n        A = sp.bsr_zeros(nrow, ncol, block_type=btype, device=dev, row_capacity=ncol)\n        sp.bsr_set_from_triplets(A, wp.array(rows, device=dev), wp.array(cols, device=dev),\n                                 wp.array(vals, dtype=btype, device=dev), topology="padded")\n        return A\n    return sp.bsr_from_triplets(nrow, ncol, wp.array(rows, device=dev), wp.array(cols, device=dev),\n                                wp.array(vals, dtype=btype, device=dev))\n\ndef topo(A):\n    A.nnz_sync() if not A.device.is_capturing else None\n    n = int(A.offsets.numpy()[A.nrow])\n    rc = None if A.row_counts is None else A.row_counts.numpy()\n    return A.offsets.numpy()[: A.nrow + 1], A.columns.numpy()[:n], A.values.numpy()[:n], rc\n\ndef check(a, b, tag):\n    for x, y, name in zip(a, b, ["offsets", "columns", "values", "row_counts"]):\n        if x is None and y is None:\n            continue\n        np.testing.assert_array_equal(np.asarray(y), np.asarray(x), err_msg=f"{tag} {name}")\n\ncases = [(7, 5, (1, 1), 30, 1, None), (12, 9, (3, 2), 60, 2, 4), (3, 40, (2, 2), 90, 3, None), (50, 1, (1, 1), 20, 4, None)]\nfor nrow, ncol, block, n, seed, er in cases:\n    for scale in (1.0, -2.5):\n        out = []\n        for dev in ("cpu", M):\n            A = build(dev, nrow, ncol, block, n, seed, empty_row=er)\n            B = sp.bsr_zeros(ncol, nrow, block_type=A.values.dtype if block == (1, 1) else wp.types.matrix(shape=block[::-1], dtype=float), device=dev)\n            sp.bsr_set_transpose(B, A * scale if scale != 1.0 else A)\n            out.append(topo(B))\n        check(out[0], out[1], f"eager {nrow}x{ncol} {block} scale={scale}")\nprint("eager OK")\n\n# padded source (row_counts present)\nout = []\nfor dev in ("cpu", M):\n    A = build(dev, 9, 6, (2, 3), 40, 5, topology="padded")\n    assert A.row_counts is not None\n    B = sp.bsr_zeros(6, 9, block_type=wp.types.matrix(shape=(3, 2), dtype=float), device=dev)\n    sp.bsr_set_transpose(B, A)\n    out.append(topo(B))\ncheck(out[0], out[1], "padded src")\nprint("padded-source OK")\n\n# capture: topology built and transposed in the graph, replayed with new triplets\nnrow, ncol, n = 10, 8, 50\ndef triplets(seed):\n    rng = np.random.default_rng(seed)\n    flat = rng.choice(nrow * ncol, size=n, replace=False)\n    r, c = (flat // ncol).astype(np.int32), (flat % ncol).astype(np.int32)\n    r[: seed % 7] = -1  # a few invalid triplets are dropped\n    return r, c, rng.standard_normal(n).astype(np.float32)\nr0, c0, v0 = triplets(10)\nrows = wp.array(r0, device=M); cols = wp.array(c0, device=M); vals = wp.array(v0, device=M)\nA = sp.bsr_zeros(nrow, ncol, block_type=float, device=M)\nB = sp.bsr_zeros(ncol, nrow, block_type=float, device=M)\nsp.bsr_set_from_triplets(A, rows, cols, vals)  # warm up / size\nsp.bsr_set_transpose(B, A)\nwith wp.ScopedCapture(device=M) as cap:\n    sp.bsr_set_from_triplets(A, rows, cols, vals)\n    sp.bsr_set_transpose(B, A)\nfor seed in (11, 12, 13):\n    r, c, v = triplets(seed)\n    rows.assign(r); cols.assign(c); vals.assign(v)\n    wp.capture_launch(cap.graph)\n    Ac = sp.bsr_from_triplets(nrow, ncol, wp.array(r, device="cpu"), wp.array(c, device="cpu"), wp.array(v, device="cpu"))\n    Bc = sp.bsr_zeros(ncol, nrow, block_type=float, device="cpu")\n    sp.bsr_set_transpose(Bc, Ac)\n    check(topo(Bc), topo(B), f"capture replay seed={seed}")\nprint("capture replay OK")\n'


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR8BsrTranspose(unittest.TestCase):
    def test_bsr_set_transpose_matches_cpu_and_captures(self):
        _run_with_metal_enabled(self, _BSR_TRANSPOSE_BODY, timeout=600)


_F64_FP32_WARN_BODY = textwrap.dedent(
    """
    import contextlib
    import io
    import numpy as np

    @wp.func
    def soft_exp(x: wp.float64):
        return wp.exp(x) + wp.sqrt(x)

    @wp.kernel(module="unique")
    def k_trans(a: wp.array[wp.float64], b: wp.array[wp.float64]):
        i = wp.tid()
        b[i] = soft_exp(a[i]) + wp.pow(a[i], wp.float64(1.5))

    @wp.kernel(module="unique")
    def k_exact(a: wp.array[wp.float64], b: wp.array[wp.float64]):
        i = wp.tid()
        # sqrt is correctly rounded softfloat; a float32 transcendental is not
        # an fp64 one (``* 0`` keeps the result bit-comparable with CPU).
        b[i] = wp.sqrt(a[i]) * a[i] / wp.float64(3.0) + wp.float64(wp.sin(wp.float32(a[i]))) * wp.float64(0.0)

    DEAD = wp.constant(0)

    @wp.kernel(module="unique")
    def k_dead(a: wp.array[wp.float64], b: wp.array[wp.float64]):
        i = wp.tid()
        # Statically dead (constant guard): dropped before MSL, must not warn.
        if DEAD == 1:
            b[i] = wp.exp(a[i])
        else:
            b[i] = a[i] * wp.float64(2.0)

    for k in (k_trans, k_exact, k_dead):
        wp.set_module_options({"metal_fp64": "emulate"}, module=k.module)

    x = np.linspace(0.1, 2.0, 37)
    outs = {}
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        for dev in ("metal:0", "cpu"):
            a = wp.array(x, dtype=wp.float64, device=dev)
            for k in (k_trans, k_exact, k_dead):
                b = wp.zeros_like(a)
                for _ in range(3):
                    wp.launch(k, dim=a.shape[0], inputs=[a, b], device=dev)
                outs[(dev, k.key)] = b.numpy()
    log = err.getvalue()
    hits = [ln for ln in log.splitlines() if "evaluated in float32" in ln]
    assert len(hits) == 1, log
    assert "'k_trans'" in hits[0] and "wp.exp()" in hits[0] and "wp.pow()" in hits[0], hits
    assert "sqrt" not in hits[0].split(" on float64")[0], hits
    # Exact parts stay bit-exact; the fp32 transcendentals carry fp32 error only.
    np.testing.assert_array_equal(outs[("metal:0", k_exact.key)], outs[("cpu", k_exact.key)])
    np.testing.assert_array_equal(outs[("metal:0", k_dead.key)], outs[("cpu", k_dead.key)])
    np.testing.assert_allclose(outs[("metal:0", k_trans.key)], outs[("cpu", k_trans.key)], rtol=1e-6)

    # A new process would load the artifact from the on-disk cache without
    # re-running codegen; that path must warn as well.
    import warp._src.codegen_metal as cm

    cm._f64_fp32_warned.clear()
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        with cm._kernel_fp64_mode(k_trans):
            art = cm._generate_msl_kernel_cached(k_trans)
    assert art.fp32_transcendentals == ("exp", "pow"), art.fp32_transcendentals
    assert "evaluated in float32" in err.getvalue(), err.getvalue()
    print("fp64 transcendental warning OK")
    """
)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR8Fp64EmulateWarning(unittest.TestCase):
    def test_fp32_transcendental_warns_once_per_kernel(self):
        _run_with_metal_enabled(self, _F64_FP32_WARN_BODY, timeout=600)


if __name__ == "__main__":
    unittest.main(verbosity=2)
