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

"""Metal-vs-CPU checks for gaps found running Newton on Metal.

Covers native-snippet array params, ``wp.array(ptr=...)`` locals, 4-D tiled
launches, matrix-element tiles, builtin tile reducers, rebuildable volumes
and strided transfers.

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

    def assert_same(res, atol=1e-5):
        a, b = res["cpu"], res["metal:0"]
        if not isinstance(a, (tuple, list)):
            a, b = [a], [b]
        for x, y in zip(a, b):
            np.testing.assert_allclose(np.asarray(y), np.asarray(x), rtol=1e-5, atol=atol)
    """
)


def _run(test_case, body: str, timeout: int = 300):
    _run_with_metal_native(test_case, _PRELUDE + textwrap.dedent(body), timeout=timeout)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR3Tiles(unittest.TestCase):
    def test_func_native_array_params(self):
        # ``arr.data`` / ``(T*)arr.data`` / ``arr.shape[i]`` in a snippet.
        _run(
            self,
            """
            @wp.func_native("return (uint64_t)arr.data + (uint64_t)(off * sizeof(float));")
            def ptr_of(arr: wp.array[wp.float32], off: int) -> wp.uint64: ...

            @wp.func_native("float* p = (float*)arr.data; p[i] = p[i] * 2.0f + (float)arr.shape[0];")
            def scale(arr: wp.array[wp.float32], i: int): ...

            @wp.kernel
            def k(a: wp.array[wp.float32], out: wp.array[wp.uint64]):
                i = wp.tid()
                out[i] = ptr_of(a, i) - ptr_of(a, 0)
                scale(a, i)

            def run(dev):
                a = wp.array(np.arange(5, dtype=np.float32), device=dev)
                o = wp.zeros(5, dtype=wp.uint64, device=dev)
                wp.launch(k, dim=5, inputs=[a], outputs=[o], device=dev)
                return o.numpy().astype(np.float64), a.numpy()

            assert_same(both(run))
            """,
        )

    def test_pointer_array_views(self):
        # Kamino's per-problem views: ``wp.array(ptr=base + offset, shape=(n, n))``
        # with tile loads/stores that straddle the view's edge (zero-filled
        # loads, dropped stores), element reads/writes and ``.shape``.
        _run(
            self,
            """
            from ctypes import sizeof

            def make(dtype):
                @wp.func_native("return (uint64_t)arr.data;")
                def ptr(arr: wp.array[dtype]) -> wp.uint64: ...

                @wp.func
                def offset_ptr(arr: wp.array[dtype], start: int) -> wp.uint64:
                    return ptr(arr) + wp.uint64(start) * wp.uint64(wp.static(sizeof(dtype._type_)))

                return offset_ptr

            f32_ptr = make(wp.float32)
            i32_ptr = make(wp.int32)

            @wp.kernel
            def k(a: wp.array[float], idx: wp.array[wp.int32], off: int, n: int, out: wp.array[float]):
                i, lane = wp.tid()
                src = wp.array(ptr=f32_ptr(a, off), shape=(n, n), dtype=float)
                dst = wp.array(ptr=f32_ptr(out, off), shape=(n, n), dtype=float)
                ids = wp.array(ptr=i32_ptr(idx, 1), shape=(2,), dtype=wp.int32)
                t = wp.tile_load(src, shape=(2, 2), offset=(n - 1, n - 1))
                wp.tile_store(dst, t + wp.tile_ones(shape=(2, 2), dtype=float), offset=(n - 1, n - 1))
                dst[0, ids[1]] = src[1, 0] + float(dst.shape[1])

            def run(dev):
                a = wp.array(np.arange(20, dtype=np.float32), device=dev)
                idx = wp.array([7, 0, 1], dtype=wp.int32, device=dev)
                o = wp.zeros(20, dtype=float, device=dev)
                wp.launch_tiled(k, dim=1, inputs=[a, idx, 2, 3, o], block_dim=32, device=dev)
                return o.numpy()

            assert_same(both(run))
            """,
        )

    def test_array_slice_views(self):
        # ``arr[a:b]`` in a kernel (Kamino's per-joint nonzero-block ids):
        # reads, ``.shape``, omitted / negative bounds, rows of a 2-D array,
        # and writes through the view.
        _run(
            self,
            """
            @wp.kernel
            def k(ids: wp.array[wp.int32], off: wp.array[wp.int32], m: wp.array2d[float],
                  out: wp.array[wp.int32], mo: wp.array[float]):
                i = wp.tid()
                sub = ids[off[i]:off[i + 1]]
                acc = int(0)
                for j in range(sub.shape[0]):
                    acc += sub[j] * (j + 1)
                out[i] = acc + sub[0] + ids[-2:][0] + ids[:3].shape[0]
                rows = m[1:3]
                mo[i] = rows[1, i] + float(rows.shape[0])
                if i == 0:
                    w = ids[9:]
                    w[0] = 99

            def run(dev):
                o = wp.zeros(3, dtype=wp.int32, device=dev)
                mo = wp.zeros(3, dtype=float, device=dev)
                ids = wp.array(np.arange(10, dtype=np.int32) * 3, device=dev)
                m = wp.array(np.arange(12, dtype=np.float32).reshape(4, 3), device=dev)
                off = wp.array([0, 2, 5, 9], dtype=wp.int32, device=dev)
                wp.launch(k, dim=3, inputs=[ids, off, m, o, mo], device=dev)
                return o.numpy(), mo.numpy(), ids.numpy()

            assert_same(both(run))
            """,
        )

    def test_array_slice_views_handed_to_tiles_and_snippets(self):
        # Argument slices that reach tile_load / tile_store or a native
        # snippet: unit-step leading-dim slices become pointer arrays, the
        # rest must raise (never an undeclared identifier or a silent no-op).
        _run(
            self,
            """
            from warp._src.codegen_metal import MetalCodegenError

            @wp.func_native("((float*)arr.data)[i] += 1.0f;")
            def inc(arr: wp.array[wp.float32], i: int): ...

            @wp.kernel
            def k1(a: wp.array[float], o: wp.array[float]):
                i, lane = wp.tid()
                s = a[3:9]
                t = wp.tile_load(s, shape=8, offset=0)
                wp.tile_store(o, t + wp.tile_ones(shape=8, dtype=float))
                wp.tile_store(a[1:4], wp.tile_ones(shape=4, dtype=float) * float(s.shape[0]))

            @wp.kernel
            def k2(a: wp.array2d[float], o: wp.array2d[float]):
                i, lane = wp.tid()
                s = a[1:3]
                t = wp.tile_load(s, shape=(2, 4), offset=(0, 1))
                wp.tile_store(o, t)

            @wp.kernel
            def k3(a: wp.array[float], o: wp.array[float]):
                i = wp.tid()
                s = a[2:6]
                inc(s, i)
                o[i] = s[i]

            def run(dev):
                a = wp.array(np.arange(10, dtype=np.float32) + 1.0, device=dev)
                o = wp.zeros(8, dtype=float, device=dev)
                wp.launch_tiled(k1, dim=1, inputs=[a, o], block_dim=32, device=dev)
                m = wp.array(np.arange(24, dtype=np.float32).reshape(4, 6), device=dev)
                mo = wp.zeros((2, 4), dtype=float, device=dev)
                wp.launch_tiled(k2, dim=1, inputs=[m, mo], block_dim=32, device=dev)
                b = wp.array(np.arange(10, dtype=np.float32), device=dev)
                bo = wp.zeros(4, dtype=float, device=dev)
                wp.launch(k3, dim=4, inputs=[b, bo], device=dev)
                return o.numpy(), a.numpy(), mo.numpy(), b.numpy(), bo.numpy()

            assert_same(both(run))

            @wp.kernel
            def strided(a: wp.array[float], o: wp.array[float]):
                i, lane = wp.tid()
                wp.tile_store(o, wp.tile_load(a[::2], shape=4, offset=0))

            @wp.kernel
            def row_cols(a: wp.array2d[float], o: wp.array[float]):
                i, lane = wp.tid()
                wp.tile_store(o, wp.tile_load(a[1, 2:6], shape=4, offset=0))

            d = "metal:0"
            for kern, arr in (
                (strided, wp.zeros(8, dtype=float, device=d)),
                (row_cols, wp.zeros((4, 6), dtype=float, device=d)),
            ):
                try:
                    wp.launch_tiled(kern, dim=1, inputs=[arr, wp.zeros(4, dtype=float, device=d)], block_dim=32,
                                    device=d)
                except MetalCodegenError as e:
                    assert "unit-step" in str(e), str(e)
                else:
                    raise AssertionError(f"{kern.key}: expected MetalCodegenError")
            """,
        )

    def test_tile_cholesky_inplace_upper(self):
        # ``fill_mode="upper"``: A = U^T U, U upper-triangular with the lower
        # part zeroed, read from A's upper triangle (A deliberately non-symmetric).
        _run(
            self,
            """
            BS = 16

            @wp.kernel(module="unique")
            def k(a: wp.array2d[float], u: wp.array2d[float], l: wp.array2d[float]):
                i, lane = wp.tid()
                t = wp.tile_load(a, shape=(BS, BS), storage="shared")
                wp.tile_cholesky_inplace(t, fill_mode="upper")
                wp.tile_store(u, t)
                s = wp.tile_load(a, shape=(BS, BS), storage="shared")
                wp.tile_cholesky_inplace(s)
                wp.tile_store(l, s)

            rng = np.random.default_rng(0)
            m = rng.standard_normal((BS, BS)).astype(np.float32)
            a = m @ m.T + BS * np.eye(BS, dtype=np.float32)
            a_up = np.triu(a) + np.tril(rng.standard_normal((BS, BS)).astype(np.float32), -1)

            def run(dev):
                u = wp.zeros((BS, BS), dtype=float, device=dev)
                l = wp.zeros((BS, BS), dtype=float, device=dev)
                wp.launch_tiled(k, dim=1, inputs=[wp.array(a_up, device=dev), u, l], block_dim=32, device=dev)
                return u.numpy(), l.numpy()

            res = both(run)
            assert_same(res, atol=1e-4)
            u = res["metal:0"][0]
            np.testing.assert_allclose(u.T @ u, a, rtol=1e-4, atol=1e-3)
            """,
        )

    def test_tile_cholesky_solve_upper(self):
        # mujoco_warp's blocked LDL: factor in upper mode from a gathered
        # shared tile, then ``tile_cholesky_solve(U, b, fill_mode="upper")``
        # (A = U^T U), both right after the factor and from a stored factor.
        _run(
            self,
            """
            BS = wp.constant(6)
            BA = wp.constant(36)

            @wp.kernel(module="unique")
            def factor_solve(a: wp.array[float], idx: wp.array[int], y: wp.array[float], u_out: wp.array[float],
                             x: wp.array[float], x2: wp.array2d[float]):
                i, lane = wp.tid()
                ids = wp.tile_load(idx, shape=(BA,), storage="shared")
                blk = wp.tile_load_indexed(a, ids, shape=(BA,), storage="shared")
                U = wp.tile_reshape(blk, (BS, BS))
                wp.tile_cholesky_inplace(U, fill_mode="upper")
                wp.tile_store(u_out, wp.tile_reshape(U, (BA,)))
                rhs = wp.tile_load(y, shape=(BS,))
                wp.tile_store(x, wp.tile_cholesky_solve(U, rhs, fill_mode="upper"))
                b2 = wp.tile_load(x2, shape=(BS, 2))
                wp.tile_cholesky_solve_inplace(U, b2, fill_mode="upper")
                wp.tile_store(x2, b2)

            @wp.kernel(module="unique")
            def solve_stored(u_in: wp.array[float], y: wp.array[float], x: wp.array[float]):
                i, lane = wp.tid()
                U = wp.tile_reshape(wp.tile_load(u_in, shape=(BA,)), (BS, BS))
                rhs = wp.tile_load(y, shape=(BS,))
                wp.tile_store(x, wp.tile_cholesky_solve(U, rhs, fill_mode="upper"))

            rng = np.random.default_rng(1)
            m = rng.standard_normal((BS, BS)).astype(np.float32)
            a = (m @ m.T + BS * np.eye(BS, dtype=np.float32)).astype(np.float32)
            perm = rng.permutation(BS * BS)
            a_perm = np.empty(BS * BS, dtype=np.float32)
            a_perm[perm] = a.ravel()
            yv = rng.standard_normal(BS).astype(np.float32)
            b2 = rng.standard_normal((BS, 2)).astype(np.float32)

            def run(dev):
                u = wp.zeros(BS * BS, dtype=float, device=dev)
                x = wp.zeros(BS, dtype=float, device=dev)
                x2 = wp.array(b2, device=dev)
                xs = wp.zeros(BS, dtype=float, device=dev)
                wp.launch_tiled(factor_solve, dim=1, inputs=[wp.array(a_perm, device=dev),
                                wp.array(perm.astype(np.int32), dtype=int, device=dev), wp.array(yv, device=dev),
                                u, x, x2], block_dim=32, device=dev)
                wp.launch_tiled(solve_stored, dim=1, inputs=[u, wp.array(yv, device=dev), xs], block_dim=32,
                                device=dev)
                return u.numpy(), x.numpy(), x2.numpy(), xs.numpy()

            res = both(run)
            assert_same(res, atol=1e-4)
            _u, x, x2, xs = res["metal:0"]
            np.testing.assert_allclose(a @ x, yv, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(a @ xs, yv, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(a @ x2, b2, rtol=1e-3, atol=1e-3)
            """,
        )

    def test_texture_array_args_and_fields(self):
        # ``wp.array[wp.Texture2D]`` kernel args (mujoco_warp's renderer) and
        # ``t.width`` / ``t.height`` reads on texture params, array elements
        # and struct fields.
        _run(
            self,
            """
            @wp.kernel
            def k(textures: wp.array[wp.Texture2D], ids: wp.array[int], uv: wp.array[wp.vec2],
                  out: wp.array[wp.vec4]):
                i = wp.tid()
                t = textures[ids[i]]
                out[i] = wp.texture_sample(t, uv[i], dtype=wp.vec4) + wp.vec4(float(t.width), float(t.height), 0.0, 0.0)

            @wp.struct
            class S:
                t: wp.Texture2D

            @wp.func
            def elem_dims(ts: wp.array[wp.Texture2D], j: int) -> int:
                return ts[j].width * 1000 + ts[j].height

            @wp.kernel
            def elem(ts: wp.array[wp.Texture2D], out: wp.array[int]):
                i = wp.tid()
                out[i] = ts[i].width + 100000 * elem_dims(ts, (i + 1) % 3)

            @wp.kernel
            def dims(t: wp.Texture2D, s: S, out: wp.array[int]):
                tt = s.t
                out[0] = t.width + 10 * t.height + 100 * tt.width + 1000 * s.t.height

            def run(dev):
                rng = np.random.default_rng(0)
                with wp.ScopedDevice(dev):
                    texs = [wp.Texture2D(rng.random((4 + j, 3 + j, 4)).astype(np.float32),
                                         filter_mode=wp.TextureFilterMode.LINEAR) for j in range(3)]
                    arr = wp.array(texs, dtype=wp.Texture2D)
                    ids = wp.array([2, 0, 1, 2], dtype=int)
                    uv = wp.array(rng.random((4, 2)).astype(np.float32), dtype=wp.vec2)
                    out = wp.zeros(4, dtype=wp.vec4)
                    wp.launch(k, dim=4, inputs=[arr, ids, uv, out])
                    s = S()
                    s.t = texs[1]
                    o2 = wp.zeros(1, dtype=int)
                    wp.launch(dims, dim=1, inputs=[texs[2], s, o2])
                    o3 = wp.zeros(3, dtype=int)
                    wp.launch(elem, dim=3, inputs=[arr, o3])
                    return out.numpy(), o2.numpy(), o3.numpy()

            assert_same(both(run))
            """,
        )

    def test_aliased_array_arguments(self):
        # One allocation bound to several array params: Metal treats kernel
        # buffers as non-aliasing, so a read through one param could miss
        # the kernel's own store through another. CPU order must hold.
        _run(
            self,
            """
            @wp.kernel
            def chain(a: wp.array[float], b: wp.array[float]):
                i = wp.tid()
                if i == 0:
                    for j in range(4):
                        b[j + 1] = a[j] * 2.0

            @wp.kernel
            def two_out(b: wp.array[float], c: wp.array[float]):
                i = wp.tid()
                if i == 0:
                    for j in range(4):
                        b[j + 1] = c[j] + 1.0

            @wp.func_native("float v = ((float*)src.data)[i]; ((float*)dst.data)[i + 1] = v * 2.0f;")
            def cp(src: wp.array[wp.float32], dst: wp.array[wp.float32], i: int): ...

            @wp.kernel
            def native_chain(a: wp.array[wp.float32], b: wp.array[wp.float32]):
                i = wp.tid()
                if i == 0:
                    for j in range(4):
                        cp(a, b, j)

            def run(dev):
                with wp.ScopedDevice(dev):
                    x = wp.array(np.arange(1, 7, dtype=np.float32))
                    wp.launch(chain, dim=1, inputs=[x, x])
                    y = wp.array(np.arange(1, 8, dtype=np.float32))
                    wp.launch(chain, dim=1, inputs=[y[1:6], y[2:7]])
                    z = wp.array(np.arange(1, 7, dtype=np.float32))
                    wp.launch(two_out, dim=1, inputs=[z, z])
                    w = wp.array(np.arange(1, 6, dtype=np.float32))
                    wp.launch(native_chain, dim=1, inputs=[w, w])
                    u = wp.array(np.arange(1, 6, dtype=np.float32))
                    v = wp.array(np.arange(1, 6, dtype=np.float32))
                    wp.launch(chain, dim=1, inputs=[u, v])
                    return x.numpy(), y.numpy(), z.numpy(), w.numpy(), v.numpy()

            assert_same(both(run))
            """,
        )

    def test_tile_assign_column_into_2d(self):
        # mujoco_warp's augmented blocked Cholesky writes an (R, 1) column
        # view into a 2-D tile with tile_assign(dst, col, offset=(0, c)); it
        # was lowered along the diagonal, which NaN'd Newton's MuJoCo solver.
        _run(
            self,
            """
            TS = wp.constant(8)
            BS = wp.constant(7)

            @wp.kernel(module="unique", enable_backward=False)
            def k(A: wp.array2d[float], b: wp.array2d[float], out: wp.array2d[float], out2: wp.array2d[float]):
                i = wp.tid()
                col = wp.tile_load(b, shape=(TS, 1), storage="shared")
                t = wp.tile_load(A, shape=(TS, TS), storage="shared")
                wp.tile_assign(t, wp.tile_view(col, shape=(BS, 1), offset=(0, 0)), offset=(0, BS))
                wp.tile_store(out, t)
                r = wp.tile_zeros(shape=(TS, 2), dtype=float, storage="shared")
                wp.tile_assign(r, wp.tile_view(t, shape=(BS, 1), offset=(1, 2)), offset=(1, 1))
                wp.tile_store(out2, r)

            def run(dev):
                rng = np.random.default_rng(0)
                with wp.ScopedDevice(dev):
                    A = wp.array(rng.random((8, 8)).astype(np.float32))
                    b = wp.array(rng.random((8, 1)).astype(np.float32))
                    o = wp.zeros((8, 8), dtype=float)
                    o2 = wp.zeros((8, 2), dtype=float)
                    wp.launch_tiled(k, dim=1, inputs=[A, b, o, o2], block_dim=32)
                    return o.numpy(), o2.numpy()

            assert_same(both(run))
            """,
        )

    def test_newton_codegen_gaps_r3integ(self):
        # wp.transformation(..., dtype=float) (``transform_t<float>``),
        # wp.abs on spatial vectors, and 1x1 tiles in tile_matmul /
        # tile_broadcast (mujoco_warp's dense JTDAJ with one constraint row).
        _run(
            self,
            """
            @wp.kernel
            def xf(r: float, t0: wp.array[wp.transform], t1: wp.array[wp.transform], out: wp.array[wp.transform],
                   qd: wp.array[wp.spatial_vector], m: wp.array[float]):
                i = wp.tid()
                a = t0[i]
                b = t1[i]
                p = wp.lerp(wp.transform_get_translation(a), wp.transform_get_translation(b), r)
                q = wp.quat_slerp(wp.transform_get_rotation(a), wp.transform_get_rotation(b), r)
                out[i] = wp.transformation(p, q, dtype=float)
                m[i] = wp.max(wp.abs(qd[i])) + 10.0 * wp.min(wp.abs(qd[i]))

            @wp.kernel(module="unique")
            def one(D: wp.array[float], J: wp.array2d[float], A: wp.array2d[float], out: wp.array2d[float],
                    out2: wp.array2d[float]):
                i, lane = wp.tid()
                d = wp.tile_load(D, shape=1)
                j = wp.tile_load(J, shape=(1, 4))
                ji = wp.tile_map(wp.mul, wp.tile_transpose(j), wp.tile_broadcast(d, shape=(4, 1)))
                wp.tile_store(out, wp.tile_matmul(ji, j))
                a = wp.tile_load(A, shape=(1, 1))
                wp.tile_store(out2, wp.tile_matmul(a, a) + wp.tile_matmul(j, wp.tile_transpose(j)))

            def run(dev):
                rng = np.random.default_rng(0)
                x0 = rng.random((4, 7)).astype(np.float32)
                x1 = rng.random((4, 7)).astype(np.float32)
                for x in (x0, x1):
                    x[:, 3:] /= np.linalg.norm(x[:, 3:], axis=1, keepdims=True)
                with wp.ScopedDevice(dev):
                    o = wp.zeros(4, dtype=wp.transform)
                    m = wp.zeros(4, dtype=float)
                    qd = wp.array(rng.random((4, 6)).astype(np.float32) - 0.5, dtype=wp.spatial_vector)
                    wp.launch(xf, 4, [0.3, wp.array(x0, dtype=wp.transform), wp.array(x1, dtype=wp.transform), o, qd, m])
                    h = wp.zeros((4, 4), dtype=float)
                    h2 = wp.zeros((1, 1), dtype=float)
                    wp.launch_tiled(one, dim=1, inputs=[wp.array([0.7], dtype=float),
                                    wp.array(rng.random((1, 4)).astype(np.float32)),
                                    wp.array([[1.5]], dtype=float), h, h2], block_dim=32)
                    return o.numpy(), m.numpy(), h.numpy(), h2.numpy()

            assert_same(both(run))
            """,
        )

    def test_tile_reverse_broadcast_and_matmul_acc(self):
        # Serial reverse of tile_broadcast (row, column, scalar sources) and
        # of the accumulating tile_matmul(a, b, c, alpha, beta).
        _run(
            self,
            """
            R, C, K = 4, 3, 2

            @wp.kernel(module="unique")
            def k(row: wp.array[float], col: wp.array2d[float], s: wp.array[float], A: wp.array2d[float],
                  B: wp.array2d[float], out: wp.array2d[float]):
                i, lane = wp.tid()
                a = wp.tile_broadcast(wp.tile_load(row, shape=C), shape=(R, C))
                b = wp.tile_broadcast(wp.tile_load(col, shape=(R, 1)), shape=(R, C))
                c = wp.tile_broadcast(wp.tile_load(s, shape=1), shape=(R, C))
                acc = a + b * b + c
                wp.tile_matmul(wp.tile_load(A, shape=(R, K)), wp.tile_load(B, shape=(K, C)), acc, alpha=0.5, beta=2.0)
                wp.tile_store(out, wp.tile_map(wp.mul, acc, acc))

            def run(dev):
                rng = np.random.default_rng(0)
                vals = [rng.random(sh).astype(np.float32) for sh in ((C,), (R, 1), (1,), (R, K), (K, C))]
                with wp.ScopedDevice(dev):
                    ins = [wp.array(v, requires_grad=True) for v in vals]
                    out = wp.zeros((R, C), dtype=float, requires_grad=True)
                    tape = wp.Tape()
                    with tape:
                        wp.launch_tiled(k, dim=1, inputs=[*ins, out], block_dim=32)
                    out.grad = wp.ones((R, C), dtype=float)
                    tape.backward()
                    return [out.numpy()] + [x.grad.numpy() for x in ins]

            assert_same(both(run), atol=1e-4)
            """,
        )

    def test_aliased_array_groups(self):
        # Overlap is transitive: an input spanning two outputs, a chain of
        # outputs, mixed element types, spilled (>31 buffer) params, and one
        # PSO variant for every offset of a sliding window.
        _run(
            self,
            """
            import importlib.util, os, tempfile
            import warp._src.codegen_metal as cm

            @wp.kernel
            def span(a: wp.array[float], b: wp.array[float], c: wp.array[float]):
                i = wp.tid()
                if i == 0:
                    for j in range(4):
                        b[j] = float(j) + 10.0
                        c[j] = a[j] + a[j + 4] + 100.0
                        c[j] = c[j] + a[j + 4]

            @wp.kernel
            def three(o1: wp.array[float], o2: wp.array[float], o3: wp.array[float]):
                i = wp.tid()
                if i == 0:
                    for j in range(4):
                        o3[j] = float(j + 1)
                        o2[4 + j] = o2[4 + j] + 5.0
                        o3[j] = o3[j] * 3.0
                        o1[0] = o1[0] + o2[0]

            @wp.kernel
            def pun(a: wp.array[wp.int32], b: wp.array[float]):
                i = wp.tid()
                if i == 0:
                    b[0] = 1.0
                    b[1] = float(a[0])

            @wp.kernel
            def chain(a: wp.array[float], b: wp.array[float]):
                i = wp.tid()
                if i == 0:
                    for j in range(4):
                        b[j + 1] = a[j] * 2.0

            n = 33
            NL = chr(10)
            src = "import warp as wp" + NL + "@wp.kernel" + NL + "def many(" + ", ".join(f"a{k}: wp.array[float]" for k in range(n))
            src += ", x: wp.array[float], y: wp.array[float]):" + NL + "    i = wp.tid()" + NL + "    s = float(0.0)" + NL
            src += "".join(f"    s += a{k}[i]" + NL for k in range(n))
            src += "    if i == 0:" + NL + "        for j in range(4):" + NL + "            y[j + 1] = x[j] * 2.0 + s" + NL
            path = os.path.join(tempfile.mkdtemp(), "_alias_many.py")
            open(path, "w").write(src)
            spec = importlib.util.spec_from_file_location("_alias_many", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            def run(dev):
                with wp.ScopedDevice(dev):
                    y = wp.zeros(8, dtype=float)
                    wp.launch(span, dim=1, inputs=[y, y[0:4], y[4:8]])
                    z = wp.zeros(14, dtype=float)
                    wp.launch(three, dim=1, inputs=[z[0:6], z[4:12], z[8:14]])
                    f = wp.zeros(4, dtype=float)
                    wp.launch(pun, dim=1, inputs=[wp.array(ptr=f.ptr, dtype=wp.int32, shape=(4,), device=dev), f])
                    w = wp.array(np.arange(1, 30, dtype=np.float32))
                    for off in range(8):
                        wp.launch(chain, dim=1, inputs=[w[off + 1 : off + 6], w[off + 2 : off + 7]])
                    m = wp.array(np.arange(1, 9, dtype=np.float32))
                    ones = [wp.ones(1, dtype=float) for _ in range(n)]
                    wp.launch(mod.many, dim=1, inputs=[*ones, m, m])
                    return y.numpy(), z.numpy(), f.numpy(), w.numpy(), m.numpy()

            assert_same(both(run))
            variants = [k for k in chain._metal_alias_psos]
            assert len(variants) == 1, variants
            """,
        )

    def test_tile_cholesky_solve_1x1_factor(self):
        # A 1x1 factor is a scalar local; a (1, K) right-hand side must divide
        # element-wise (both fill modes, in- and out-of-place).
        _run(
            self,
            """
            @wp.kernel(module="unique")
            def kl(A: wp.array2d[float], B: wp.array2d[float], X: wp.array2d[float], Y: wp.array2d[float]):
                i, lane = wp.tid()
                U = wp.tile_load(A, shape=(1, 1))
                wp.tile_cholesky_inplace(U, fill_mode="lower")
                b = wp.tile_load(B, shape=(1, 3))
                wp.tile_store(X, wp.tile_cholesky_solve(U, b, fill_mode="lower"))
                wp.tile_cholesky_solve_inplace(U, b, fill_mode="lower")
                wp.tile_store(Y, b)

            @wp.kernel(module="unique")
            def ku(A: wp.array2d[float], B: wp.array2d[float], X: wp.array2d[float], Y: wp.array2d[float]):
                i, lane = wp.tid()
                U = wp.tile_load(A, shape=(1, 1))
                wp.tile_cholesky_inplace(U, fill_mode="upper")
                b = wp.tile_load(B, shape=(1, 3))
                wp.tile_store(X, wp.tile_cholesky_solve(U, b, fill_mode="upper"))
                wp.tile_cholesky_solve_inplace(U, b, fill_mode="upper")
                wp.tile_store(Y, b)

            def run(dev):
                res = []
                with wp.ScopedDevice(dev):
                    for k in (kl, ku):
                        X = wp.zeros((1, 3), dtype=float)
                        Y = wp.zeros((1, 3), dtype=float)
                        wp.launch_tiled(k, dim=1, inputs=[wp.array([[4.0]], dtype=float),
                                        wp.array([[1.0, 2.0, 3.0]], dtype=float), X, Y], block_dim=32)
                        res += [X.numpy(), Y.numpy()]
                return res

            assert_same(both(run))
            """,
        )

    def test_tile_store_bounds_on_arguments(self):
        # A partial trailing tile must not write past the array's rows.
        _run(
            self,
            """
            @wp.kernel
            def k(o: wp.array2d[float]):
                i, lane = wp.tid()
                wp.tile_store(o, wp.tile_ones(shape=(2, 2), dtype=float), offset=(2, 2))

            def run(dev):
                base = wp.zeros(20, dtype=float, device=dev)
                wp.launch_tiled(k, dim=1, inputs=[base[:9].reshape((3, 3))], block_dim=32, device=dev)
                return base.numpy()

            assert_same(both(run))
            """,
        )

    def test_launch_tiled_3d_grid(self):
        # ``launch_tiled(dim=(a, b, c))`` launches 4-D; ``wp.tid()`` reads the
        # first three coordinates.
        _run(
            self,
            """
            TS = 8

            @wp.func
            def clip_to_one(x: wp.float32):
                return wp.min(x, 1.0)

            @wp.kernel
            def k(sp: wp.array3d[wp.float32], out: wp.array3d[wp.float32]):
                w, i, j = wp.tid()
                acc = wp.tile_zeros(shape=(TS, TS), dtype=wp.float32)
                for kk in range((sp.shape[1] + TS - 1) // TS):
                    a = wp.tile_reshape(wp.tile_load(sp, shape=(1, TS, TS), offset=(w, kk * TS, i * TS)), (TS, TS))
                    b = wp.tile_reshape(wp.tile_load(sp, shape=(1, TS, TS), offset=(w, kk * TS, j * TS)), (TS, TS))
                    wp.tile_matmul(wp.tile_transpose(a), b, acc)
                o3 = wp.tile_map(clip_to_one, wp.tile_reshape(acc, (1, TS, TS)))
                wp.tile_store(out, o3, offset=(w, i * TS, j * TS))

            x = (np.random.default_rng(0).random((2, 13, 16)) > 0.7).astype(np.float32)

            def run(dev):
                o = wp.zeros((2, 16, 16), dtype=float, device=dev)
                wp.launch_tiled(k, dim=(2, 2, 2), inputs=[wp.array(x, device=dev), o], block_dim=32, device=dev)
                return o.numpy()

            assert_same(both(run))
            """,
        )

    def test_tile_map_closure_from_repeated_factory(self):
        # The second factory call registers a same-named closure under the
        # module key; the first kernel must still find its own function.
        _run(
            self,
            """
            def make(scale):
                @wp.func
                def f(x: wp.float32) -> wp.int32:
                    return wp.int32(wp.isnan(x)) + wp.int32(scale)

                @wp.kernel(module="unique")
                def k(r: wp.array2d[wp.float32], out: wp.array[wp.int32]):
                    w, i, lane = wp.tid()
                    seg = wp.tile_load(r, shape=(1, 8), offset=(w, i * 8))
                    if lane == 0:
                        wp.atomic_max(out, w, wp.tile_max(wp.tile_map(f, seg))[0])

                return k

            k1 = make(10)
            k2 = make(20)
            x = np.random.default_rng(1).standard_normal((2, 16)).astype(np.float32)
            x[1, 3] = np.nan

            def run(dev):
                outs = []
                for k in (k1, k2):
                    o = wp.zeros(2, dtype=wp.int32, device=dev)
                    wp.launch_tiled(k, dim=(2, 2), inputs=[wp.array(x, device=dev), o], block_dim=32, device=dev)
                    outs.append(o.numpy())
                return outs

            assert_same(both(run))
            """,
        )

    def test_matrix_element_tiles(self):
        _run(
            self,
            """
            @wp.kernel
            def k(v: wp.array[wp.vec3], out: wp.array[wp.mat33], ones: wp.array[wp.mat33],
                  xf: wp.array[wp.transform]):
                i, lane = wp.tid()
                a = v[i]
                m = wp.mat33(a[0], a[1], a[2], 2.0 * a[0], 2.0 * a[1], 2.0 * a[2], 1.0, 2.0, 3.0)
                t = wp.tile_zeros(shape=(1,), dtype=wp.mat33, storage="shared")
                wp.tile_scatter_masked(t, 0, m, lane == 0)
                o = wp.tile_ones(shape=(1,), dtype=wp.mat33)
                tt = wp.tile_zeros(shape=4, dtype=wp.transform, storage="shared")
                tt[1] = wp.transform(a, wp.quat_identity())
                if lane == 0:
                    out[i] = wp.tile_extract(t, 0)
                    ones[i] = wp.tile_extract(o, 0)
                    xf[i] = tt[1]

            x = np.random.default_rng(0).standard_normal((3, 3)).astype(np.float32)

            def run(dev):
                o = wp.zeros(3, dtype=wp.mat33, device=dev)
                o1 = wp.zeros(3, dtype=wp.mat33, device=dev)
                o2 = wp.zeros(3, dtype=wp.transform, device=dev)
                wp.launch_tiled(k, dim=3, inputs=[wp.array(x, dtype=wp.vec3, device=dev), o, o1, o2],
                                block_dim=32, device=dev)
                return o.numpy(), o1.numpy(), o2.numpy()

            assert_same(both(run))
            """,
        )

    def test_tile_reduce_bitwise(self):
        _run(
            self,
            """
            BD = 32

            @wp.kernel(module="unique")
            def k(vals: wp.array[int], out: wp.array2d[int]):
                block_id, lane = wp.tid()
                bits = wp.tile_zeros(shape=BD, dtype=int, storage="shared")
                for e in range(8):
                    wp.tile_scatter_masked(bits, e, vals[block_id * 8 + e], lane == 0)
                loaded = wp.tile_load(vals, shape=8, offset=block_id * 8)
                if lane == 0:
                    out[block_id, 0] = wp.tile_reduce(wp.bit_or, bits)[0]
                    out[block_id, 1] = wp.tile_reduce(wp.bit_or, loaded)[0]
                    out[block_id, 2] = wp.tile_reduce(wp.bit_and, loaded)[0]
                    out[block_id, 3] = wp.tile_reduce(wp.bit_xor, loaded)[0]

            x = np.random.default_rng(0).integers(0, 1 << 10, 32).astype(np.int32)

            def run(dev):
                o = wp.zeros((4, 4), dtype=int, device=dev)
                wp.launch_tiled(k, dim=4, inputs=[wp.array(x, device=dev), o], block_dim=BD, device=dev)
                return o.numpy()

            res = both(run)
            assert_same(res)
            np.testing.assert_array_equal(res["metal:0"][:, 1], np.bitwise_or.reduce(x.reshape(4, 8), axis=1))
            """,
        )

    def test_lane_scatter_reduce_distributed(self):
        # Every lane scatters its own element, then a bitwise block reduce
        # (CUDA lane semantics; CPU runs one lane per block, so compare with
        # a NumPy reference).
        _run(
            self,
            """
            BD = 32

            @wp.kernel(module="unique")
            def k(vals: wp.array[int], out: wp.array[int], nblocks: int):
                block_id, lane = wp.tid()
                bits = wp.tile_zeros(shape=BD, dtype=int, storage="shared")
                for e in range(block_id, 6, nblocks):
                    b = int(0)
                    if lane < 8:
                        b = vals[e * 8 + lane]
                    wp.tile_scatter_masked(bits, lane, b, True)
                    m = wp.tile_reduce(wp.bit_or, bits)[0]
                    if lane < 10 and m & (1 << lane) != 0:
                        wp.atomic_add(out, e, 1 << lane)

            x = np.random.default_rng(3).integers(0, 1 << 10, 48).astype(np.int32)
            o = wp.zeros(6, dtype=int, device="metal:0")
            wp.launch_tiled(k, dim=2, inputs=[wp.array(x, device="metal:0"), o, 2], block_dim=BD, device="metal:0")
            ref = np.bitwise_or.reduce(x.reshape(6, 8), axis=1) & 1023
            np.testing.assert_array_equal(o.numpy(), ref)
            """,
        )

    def test_rebuildable_volumes(self):
        _run(
            self,
            """
            @wp.kernel
            def lookup(vid: wp.uint64, pts: wp.array[wp.vec3i], out: wp.array[int]):
                i = wp.tid()
                p = pts[i]
                out[i] = wp.volume_lookup_index(vid, p[0], p[1], p[2])

            @wp.kernel
            def sample(vid: wp.uint64, pts: wp.array[wp.vec3i], out: wp.array[float]):
                i = wp.tid()
                p = pts[i]
                out[i] = wp.volume_lookup_f(vid, p[0], p[1], p[2])

            rng = np.random.default_rng(0)
            p1 = rng.integers(-20, 20, (50, 3)).astype(np.int32)
            p2 = rng.integers(-40, 40, (80, 3)).astype(np.int32)
            big = rng.integers(-500, 500, (400, 3)).astype(np.int32)
            q = np.concatenate([p1, p2, rng.integers(-40, 40, (30, 3)).astype(np.int32)])

            def run(dev):
                res = []
                st = wp.zeros(1, dtype=wp.uint32, device=dev)
                qa = wp.array(q, dtype=wp.vec3i, device=dev)
                out = wp.zeros(len(q), dtype=int, device=dev)
                v = wp.Volume.allocate_by_voxels(
                    wp.array(p1, dtype=wp.vec3i, device=dev), voxel_size=0.5, device=dev,
                    max_active_voxels=200, status=st,
                )
                vid = v.id
                wp.launch(lookup, dim=len(q), inputs=[v.id, qa, out], device=dev)
                res += [out.numpy().copy(), st.numpy().copy(), list(v.get_active_stats()), list(v.get_rebuild_info()[1:])]
                v.rebuild(wp.array(p2, dtype=wp.vec3i, device=dev), status=st)
                assert v.id == vid
                wp.launch(lookup, dim=len(q), inputs=[v.id, qa, out], device=dev)
                res += [out.numpy().copy(), st.numpy().copy(), [v.get_voxel_count(), v.get_tile_count()]]
                v.rebuild(wp.array(big, dtype=wp.vec3i, device=dev), status=st)
                res += [st.numpy().copy()]
                # Empty reservation, and an exact volume with every point masked out.
                zero = wp.zeros(len(p1), dtype=wp.int32, device=dev)
                e = wp.Volume.allocate_by_voxels(wp.array(p1, dtype=wp.vec3i, device=dev), voxel_size=1.0,
                                                 device=dev, max_active_voxels=4, point_mask=zero)
                e2 = wp.Volume.allocate_by_voxels(wp.array(p1, dtype=wp.vec3i, device=dev), voxel_size=1.0,
                                                  device=dev, point_mask=zero)
                res += [list(e.get_active_stats()), list(e.get_rebuild_info()[1:]), list(e2.get_active_stats())]
                # Rebuildable tile volume with a background value.
                t = wp.Volume.allocate_by_tiles(wp.array(p1[:4] * 8, dtype=wp.vec3i, device=dev), voxel_size=1.0,
                                                bg_value=2.5, device=dev, max_tiles=16)
                t.rebuild(wp.array(p2[:6] * 8, dtype=wp.vec3i, device=dev))
                fo = wp.zeros(len(q), dtype=float, device=dev)
                wp.launch(sample, dim=len(q), inputs=[t.id, qa, fo], device=dev)
                res += [fo.numpy(), list(t.get_active_stats())]
                return [np.asarray(r, dtype=np.float64) for r in res]

            assert_same(both(run))
            """,
        )

    def test_reverse_mode_strided_args(self):
        _run(
            self,
            """
            @wp.kernel
            def gather(src: wp.array3d[float], idx: wp.array[int], dst: wp.array3d[float]):
                i, j, k = wp.tid()
                dst[i, j, k] = src[i, j, idx[k]] * 2.0

            @wp.kernel
            def weighted_sum(a: wp.array3d[float], loss: wp.array[float]):
                i, j, k = wp.tid()
                wp.atomic_add(loss, 0, a[i, j, k] * float(k + 1))

            def run(dev):
                base = wp.array(np.arange(30, dtype=np.float32).reshape(2, 3, 5), device=dev, requires_grad=True)
                src = base[:, ::2, 1:]
                idx = wp.array([3, 0, 2], dtype=int, device=dev)
                dst = wp.zeros((2, 2, 3), dtype=float, device=dev, requires_grad=True)
                loss = wp.zeros(1, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(gather, dim=dst.shape, inputs=[src, idx], outputs=[dst], device=dev)
                    wp.launch(weighted_sum, dim=dst.shape, inputs=[dst], outputs=[loss], device=dev)
                tape.backward(loss)
                return dst.numpy(), base.grad.numpy()

            assert_same(both(run))
            """,
        )

    def test_bsr_from_triplets_device(self):
        # Built on the GPU (sort / scan / scatter), so it also records into a
        # Metal graph capture: out-of-range, duplicate and zero blocks, a device
        # ``count``, and the padded topology.
        _run(
            self,
            """
            import warp.sparse as sp

            rng = np.random.default_rng(0)
            n = 300
            rows = rng.integers(-2, 40, n).astype(np.int32)
            cols = rng.integers(0, 33, n).astype(np.int32)
            vals = rng.standard_normal((n, 2, 2)).astype(np.float32)
            vals[::7] = 0.0

            def run(dev):
                r, c = wp.array(rows, device=dev), wp.array(cols, device=dev)
                v = wp.array(vals, dtype=wp.mat22, device=dev)
                a = sp.bsr_zeros(40, 32, wp.mat22, device=dev)
                sp.bsr_set_from_triplets(a, r, c, v, count=wp.array([250], dtype=int, device=dev))
                a.nnz_sync()
                b = sp.bsr_zeros(40, 32, wp.mat22, device=dev)
                sp.bsr_set_from_triplets(b, r, c, v)
                if dev != "cpu":
                    with wp.ScopedCapture(device=dev) as cap:
                        sp.bsr_set_from_triplets(b, r, c, v)
                    b.offsets.zero_()
                    b.values.zero_()
                    wp.capture_launch(cap.graph)
                b.nnz_sync()
                p = sp.bsr_zeros(40, 32, wp.mat22, device=dev, row_capacity=16)
                sp.bsr_set_from_triplets(p, r, c, v, topology="padded")
                po, pc, cols_p, vals_p = p.offsets.numpy(), p.row_counts.numpy(), p.columns.numpy(), p.values.numpy()
                act = np.concatenate([np.arange(po[i], po[i] + pc[i]) for i in range(40)]).astype(int)
                return [
                    pc, cols_p[act], vals_p[act],
                    a.offsets.numpy(), a.columns.numpy()[: a.nnz], a.values.numpy()[: a.nnz],
                    b.offsets.numpy(), b.columns.numpy()[: b.nnz], b.values.numpy()[: b.nnz],
                ]

            assert_same(both(run))
            """,
        )

    def test_bvh_rebuild_in_capture(self):
        # A captured rebuild replays as a refit: queries on moved items match CPU.
        _run(
            self,
            """
            @wp.kernel
            def move(lo: wp.array[wp.vec3], up: wp.array[wp.vec3], t: wp.array[float]):
                i = wp.tid()
                d = wp.vec3(t[0] * float(i % 5), 0.0, 0.0)
                lo[i] = lo[i] + d
                up[i] = up[i] + d

            @wp.kernel
            def query(bvh: wp.uint64, qlo: wp.array[wp.vec3], qup: wp.array[wp.vec3], hits: wp.array[int]):
                i = wp.tid()
                q = wp.bvh_query_aabb(bvh, qlo[i], qup[i])
                idx = int(0)
                n = int(0)
                while wp.bvh_query_next(q, idx):
                    n += idx + 1
                hits[i] = n

            rng = np.random.default_rng(0)
            c = rng.random((64, 3)).astype(np.float32) * 4.0
            ql = rng.random((32, 3)).astype(np.float32) * 6.0
            qu = ql + 0.7

            def run(dev):
                lo = wp.array(c - 0.1, dtype=wp.vec3, device=dev)
                up = wp.array(c + 0.1, dtype=wp.vec3, device=dev)
                t = wp.array([0.3], dtype=float, device=dev)
                bvh = wp.Bvh(lo, up)
                hits = wp.zeros(32, dtype=int, device=dev)
                qa, qb = wp.array(ql, dtype=wp.vec3, device=dev), wp.array(qu, dtype=wp.vec3, device=dev)

                def step():
                    wp.launch(move, dim=64, inputs=[lo, up, t], device=dev)
                    bvh.rebuild()
                    wp.launch(query, dim=32, inputs=[bvh.id, qa, qb, hits], device=dev)

                if dev == "cpu":
                    step()
                    step()
                else:
                    with wp.ScopedCapture(device=dev) as cap:
                        step()
                    wp.capture_launch(cap.graph)
                    wp.capture_launch(cap.graph)
                return hits.numpy()

            assert_same(both(run))

            # Overlapping strides: each aliased gradient element keeps every share.
            @wp.kernel
            def weigh(src: wp.array2d[float], loss: wp.array[float]):
                i, j = wp.tid()
                wp.atomic_add(loss, 0, src[i, j] * float(i + 2 * j + 1))

            def run_alias(dev):
                base = wp.array(np.arange(4, dtype=np.float32), device=dev, requires_grad=True)
                v = wp.array(ptr=base.ptr, shape=(2, 3), strides=(4, 4), dtype=float, device=dev)
                v.grad = wp.array(ptr=base.grad.ptr, shape=(2, 3), strides=(4, 4), dtype=float, device=dev)
                v.requires_grad = True
                loss = wp.zeros(1, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(weigh, dim=(2, 3), inputs=[v], outputs=[loss], device=dev)
                tape.backward(loss)
                return loss.numpy(), base.grad.numpy()

            assert_same(both(run_alias))
            """,
        )

    def test_strided_copy_in_capture(self):
        # Strided Metal<->Metal copies are device kernels, so they replay.
        _run(
            self,
            """
            d = "metal:0"
            base = wp.array(np.arange(40, dtype=np.float32).reshape(5, 8), device=d)
            dst = wp.zeros((5, 3), dtype=float, device=d)
            dst2 = wp.zeros((5, 8), dtype=float, device=d)
            with wp.ScopedCapture(device=d) as cap:
                wp.copy(dst, base[:, 2:5])
                wp.copy(dst2[:, ::3], base[:, 1:4])
                wp.copy(dst2[:, 1:2], dst[:, 2:3])
            base.assign(base.numpy() * 10.0)
            wp.capture_launch(cap.graph)
            h = base.numpy()
            expect2 = np.zeros((5, 8), np.float32)
            expect2[:, ::3] = h[:, 1:4]
            expect2[:, 1:2] = h[:, 4:5]
            np.testing.assert_array_equal(dst.numpy(), h[:, 2:5])
            np.testing.assert_array_equal(dst2.numpy(), expect2)
            """,
        )

    def test_review_rejects_silently_wrong_cases(self):
        # Cases that used to compile or launch and give wrong results on Metal
        # must now raise.
        _run(
            self,
            """
            from warp._src.codegen_metal import MetalCodegenError

            def raises(fn, exc, text):
                try:
                    fn()
                except exc as e:
                    assert text in str(e), str(e)
                    return
                raise AssertionError(f"expected {exc.__name__} containing {text!r}")

            d = "metal:0"

            # Host ``wp.array.ptr`` handed in as a uint64 is not a device address.
            @wp.kernel
            def from_host_ptr(p: wp.uint64, out: wp.array[float]):
                i = wp.tid()
                out[i] = wp.array(ptr=p, shape=(4,), dtype=float)[i]

            buf = wp.array(np.arange(8, dtype=np.float32), device=d)
            raises(lambda: wp.launch(from_host_ptr, dim=4, inputs=[buf.ptr, wp.zeros(4, dtype=float, device=d)],
                                     device=d), MetalCodegenError, "device address")

            # A struct-field array handed to a native snippet.
            @wp.func_native("((float*)arr.data)[i] += 1.0f;")
            def inc(arr: wp.array[wp.float32], i: int): ...

            @wp.struct
            class S:
                arr: wp.array[wp.float32]

            @wp.kernel
            def via_struct(s: S):
                inc(s.arr, wp.tid())

            st = S()
            st.arr = wp.zeros(4, dtype=float, device=d)
            raises(lambda: wp.launch(via_struct, dim=4, inputs=[st], device=d), MetalCodegenError, "kernel array")

            # A staged (strided) argument handed to a native snippet.
            @wp.kernel
            def via_arg(a: wp.array[wp.float32]):
                inc(a, wp.tid())

            raises(lambda: wp.launch(via_arg, dim=4, inputs=[wp.zeros(8, dtype=float, device=d)[::2]], device=d),
                   RuntimeError, "func_native")

            # Lane-indexed scatter at a block_dim without real lanes.
            @wp.kernel(module="unique")
            def lane_scatter(out: wp.array[int]):
                b, lane = wp.tid()
                t = wp.tile_zeros(shape=64, dtype=int, storage="shared")
                wp.tile_scatter_masked(t, lane, lane, True)
                if lane == 0:
                    out[b] = wp.tile_reduce(wp.add, t)[0]

            raises(lambda: wp.launch_tiled(lane_scatter, dim=1, inputs=[wp.zeros(1, dtype=int, device=d)],
                                           block_dim=64, device=d), RuntimeError, "block_dim=32")

            # Non-float32 gradients of an aliasing view cannot be accumulated.
            @wp.kernel
            def half_sum(src: wp.array2d[wp.float16], loss: wp.array[wp.float16]):
                i, j = wp.tid()
                wp.atomic_add(loss, 0, src[i, j])

            base = wp.zeros(4, dtype=wp.float16, device=d, requires_grad=True)
            v = wp.array(ptr=base.ptr, shape=(2, 3), strides=(2, 2), dtype=wp.float16, device=d)
            v.grad = wp.array(ptr=base.grad.ptr, shape=(2, 3), strides=(2, 2), dtype=wp.float16, device=d)
            v.requires_grad = True
            loss = wp.zeros(1, dtype=wp.float16, device=d, requires_grad=True)
            tape = wp.Tape()
            with tape:
                wp.launch(half_sum, dim=(2, 3), inputs=[v], outputs=[loss], device=d)
            raises(lambda: tape.backward(loss), RuntimeError, "float32")

            # A grouped BVH cannot be rebuilt inside a capture (a refit cannot
            # follow new group ids).
            lo = wp.array(np.zeros((8, 3), np.float32), dtype=wp.vec3, device=d)
            up = wp.array(np.ones((8, 3), np.float32), dtype=wp.vec3, device=d)
            grouped = wp.Bvh(lo, up, groups=wp.array(np.arange(8) % 2, dtype=int, device=d))

            def capture_rebuild():
                with wp.ScopedCapture(device=d):
                    grouped.rebuild()

            raises(capture_rebuild, RuntimeError, "grouped")
            """,
        )

    def test_review_pointer_views_and_gradients(self):
        _run(
            self,
            """
            @wp.func_native("return (uint64_t)arr.data;")
            def ptr(arr: wp.array[wp.float32]) -> wp.uint64: ...

            # Negative indices on wp.array(ptr=...) views and slices wrap.
            @wp.kernel
            def neg(a: wp.array[float], out: wp.array[float]):
                i = wp.tid()
                v = wp.array(ptr=ptr(a) + wp.uint64(8), shape=(2, 3), dtype=float)
                j = -1 - i
                s = a[3:7]
                out[i] = v[-1, j] + v[0, -3] + s[-1]
                v[-2, j] = 100.0 + float(i)

            def run_neg(dev):
                a = wp.array(np.arange(10, dtype=np.float32), device=dev)
                o = wp.zeros(3, dtype=float, device=dev)
                wp.launch(neg, dim=3, inputs=[a, o], device=dev)
                return o.numpy(), a.numpy()

            assert_same(both(run_neg))

            # A non-finite gradient already in a staged strided view survives.
            @wp.kernel
            def dbl(x: wp.array[float], y: wp.array[float]):
                i = wp.tid()
                y[i] = x[i] * 2.0

            def run_inf(dev):
                base = wp.array(np.arange(12, dtype=np.float32), device=dev, requires_grad=True)
                g = np.zeros(12, dtype=np.float32)
                g[2] = np.inf
                g[4] = 1e8
                base.grad.assign(g)
                y = wp.zeros(6, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(dbl, dim=6, inputs=[base[::2]], outputs=[y], device=dev)
                y.grad = wp.ones(6, dtype=float, device=dev)
                tape.backward()
                return base.grad.numpy()

            assert_same(both(run_inf))
            """,
        )

    def test_review_tile_store_bounds(self):
        # Every tile_store lowering drops out-of-range elements: leading
        # (3-D) indices, vec elements, (1, 1) tiles, atomic-typed and
        # threadgroup-resident sources.
        _run(
            self,
            """
            @wp.kernel(module="unique")
            def k(a: wp.array2d[float], o: wp.array2d[float], o3: wp.array3d[float], ov: wp.array2d[wp.vec3],
                  oa: wp.array2d[float], r: int):
                i, lane = wp.tid()
                wp.tile_store(o, wp.tile_load(a, shape=(4, 4), storage="shared"), offset=(1, 1))
                wp.tile_store(o3, wp.tile_ones(shape=(1, 2, 2), dtype=float), offset=(r, 1, 1))
                wp.tile_store(o3, wp.tile_ones(shape=(1, 2, 2), dtype=float), offset=(0, 2, 2))
                wp.tile_store(ov, wp.tile_ones(shape=(2, 2), dtype=wp.vec3), offset=(2, 2))
                wp.tile_store(o, wp.tile_ones(shape=(1, 1), dtype=float), offset=(r, 0))
                wp.tile_store(oa, wp.tile_ones(shape=(2, 2), dtype=float), offset=(2, 2))
                wp.atomic_add(oa, 0, 0, 1.0)

            def run(dev):
                b1 = wp.zeros(30, dtype=float, device=dev)
                b3 = wp.zeros(40, dtype=float, device=dev)
                bv = wp.zeros(20, dtype=wp.vec3, device=dev)
                ba = wp.zeros(20, dtype=float, device=dev)
                wp.launch_tiled(k, dim=1, inputs=[wp.ones((4, 4), dtype=float, device=dev), b1[:9].reshape((3, 3)),
                                b3[:18].reshape((2, 3, 3)), bv[:9].reshape((3, 3)), ba[:9].reshape((3, 3)), 4],
                                block_dim=32, device=dev)
                return b1.numpy(), b3.numpy(), bv.numpy(), ba.numpy()

            assert_same(both(run))
            """,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
