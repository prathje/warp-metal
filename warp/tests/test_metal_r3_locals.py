# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal codegen for local arrays, element addresses and indexing.

Each test launches the same kernel on ``cpu`` and ``metal:0`` and compares:
thread-local ``wp.zeros(shape, dtype)`` arrays, range slices of array
arguments, component stores inside inlined functions, nested struct stores
and ``wp.static`` factory functions called more than once.
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_enabled

_COMPARE = textwrap.dedent(
    """
    import numpy as np

    def compare(run):
        res = {dev: run(dev) for dev in ("cpu", "metal:0")}
        for a, b in zip(res["cpu"], res["metal:0"]):
            np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-5, atol=1e-5)
        return res
    """
)


def _snippet(body: str) -> str:
    return _COMPARE + textwrap.dedent(body)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR3Locals(unittest.TestCase):
    def test_fixed_array_scalar_and_vec(self):
        # ``wp.zeros(shape, dtype)`` inside a kernel: writes from an inlined
        # function, element read-modify-write, ``shape``/``len``, a vec
        # element type and a component store into a vec element.
        snippet = _snippet(
            """
            @wp.func
            def fill(a: wp.array[float], n: int, s: float):
                for i in range(n):
                    a[i] = s * float(i)

            @wp.func
            def sumf(a: wp.array[float]) -> float:
                t = float(0.0)
                for i in range(a.shape[0]):
                    t += a[i]
                return t

            @wp.kernel
            def k(out: wp.array[float], outv: wp.array[wp.vec2]):
                tid = wp.tid()
                a = wp.zeros(shape=8, dtype=float)
                fill(a, 8, float(tid))
                a[3] = a[3] + 1.0
                v = wp.zeros(shape=4, dtype=wp.vec2)
                for i in range(4):
                    v[i] = wp.vec2(float(i), float(tid))
                v[1][0] = 7.0
                a[-1] = a[-1] + 0.5
                out[tid] = sumf(a) + float(a.shape[0]) + float(len(v))
                outv[tid] = v[1] + v[3]

            def run(dev):
                o = wp.zeros(4, dtype=float, device=dev)
                ov = wp.zeros(4, dtype=wp.vec2, device=dev)
                wp.launch(k, 4, [o, ov], device=dev)
                return o.numpy(), ov.numpy()

            res = compare(run)
            np.testing.assert_allclose(res["metal:0"][0], [13.5, 41.5, 69.5, 97.5])
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_fixed_array_2d_mat_views_and_copies(self):
        # 2-D fixed arrays, row views (``a[1]`` aliases the storage), a
        # copy (``b = a`` copies, as ``fixedarray_t::operator=`` does),
        # ``wp.atomic_add`` and matrix elements with an element store.
        snippet = _snippet(
            """
            @wp.func
            def bump(a: wp.array2d[float], i: int):
                a[i, 1] += 2.0

            @wp.kernel
            def k(out: wp.array[float], outm: wp.array[wp.mat22]):
                tid = wp.tid()
                a = wp.zeros(shape=(3, 4), dtype=float)
                for i in range(3):
                    for j in range(4):
                        a[i, j] = float(i * 4 + j + tid)
                bump(a, 2)
                row = a[1]
                row[0] = 100.0
                s = float(0.0)
                for i in range(a.shape[0]):
                    for j in range(a.shape[1]):
                        s += a[i, j]
                b = a
                b[0, 0] = -1.0
                wp.atomic_add(a, 0, 3, 5.0)
                m = wp.zeros(shape=2, dtype=wp.mat22)
                m[0] = wp.mat22(1.0, 2.0, 3.0, float(tid))
                m[1][0, 1] = 9.0
                m[1] = m[1] + m[0]
                out[tid] = s + a[0, 0] + a[0, 3] + b[0, 0] + float(len(m))
                outm[tid] = m[1]

            def run(dev):
                o = wp.zeros(4, dtype=float, device=dev)
                om = wp.zeros(4, dtype=wp.mat22, device=dev)
                wp.launch(k, 4, [o, om], device=dev)
                return o.numpy(), om.numpy()

            compare(run)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_fixed_array_pointer_aliased_view(self):
        # Newton's contact-manifold builder carves two buffers out of one
        # thread-local allocation with ``wp.array(ptr=buf.ptr + offset)``.
        snippet = _snippet(
            """
            @wp.func
            def add_unique(buf: wp.array[wp.vec2], count: int, p: wp.vec2, eps: float):
                for i in range(count):
                    if wp.length(buf[i] - p) < eps:
                        return count, False
                buf[count] = p
                return count + 1, True

            @wp.func
            def area(buf: wp.array[wp.vec2], n: int) -> float:
                a = float(0.0)
                for i in range(n):
                    j = (i + 1) % n
                    a += buf[i][0] * buf[j][1] - buf[j][0] * buf[i][1]
                return 0.5 * a

            @wp.kernel
            def k(pts: wp.array[wp.vec2], out: wp.array[float], cnt: wp.array[int]):
                tid = wp.tid()
                b_buffer = wp.zeros(shape=(10,), dtype=wp.vec2f)
                a_buffer = wp.array(ptr=b_buffer.ptr + wp.uint64(5 * 8), shape=(5,), dtype=wp.vec2f)
                a_count = int(0)
                b_count = int(0)
                for e in range(5):
                    p = pts[(tid + e) % pts.shape[0]]
                    a_count, added = add_unique(a_buffer, a_count, p, 1.0e-3)
                    b_count, added2 = add_unique(b_buffer, b_count, p * 2.0, 1.0e-3)
                out[tid] = area(a_buffer, a_count) + 10.0 * area(b_buffer, b_count) + b_buffer[5][0]
                cnt[tid] = a_count * 10 + b_count

            rng = np.random.default_rng(1)
            P = rng.random((7, 2)).astype(np.float32)
            P[3] = P[1]

            def run(dev):
                pts = wp.array(P, dtype=wp.vec2, device=dev)
                o = wp.zeros(7, dtype=float, device=dev)
                c = wp.zeros(7, dtype=int, device=dev)
                wp.launch(k, 7, [pts, o, c], device=dev)
                return o.numpy(), c.numpy()

            compare(run)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_fixed_array_adjoint(self):
        # The dual of a fixed array is a second fixed array that the
        # element adjoints accumulate into, as on CPU.
        snippet = _snippet(
            """
            @wp.func
            def accum(buf: wp.array[float], x: float):
                for i in range(buf.shape[0]):
                    buf[i] = x * float(i + 1)

            @wp.kernel
            def k(x: wp.array[float], out: wp.array[float]):
                t = wp.tid()
                a = wp.zeros(shape=4, dtype=float)
                accum(a, x[t])
                s = float(0.0)
                for i in range(4):
                    s += a[i] * a[i]
                out[t] = s

            X = np.array([0.5, 1.0, -2.0], dtype=np.float32)

            def run(dev):
                x = wp.array(X, device=dev, requires_grad=True)
                o = wp.zeros(3, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(k, 3, [x, o], device=dev)
                tape.backward(grads={o: wp.ones(3, dtype=float, device=dev)})
                return o.numpy(), x.grad.numpy()

            res = compare(run)
            np.testing.assert_allclose(res["metal:0"][1], 60.0 * X, rtol=1e-5)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_slice_views_of_array_args(self):
        # Range slices keep a dimension with an offset and a stride; Kamino
        # takes ``ids[start:end]`` of an argument and indexes the view.
        snippet = _snippet(
            """
            @wp.func
            def total(a: wp.array[float]) -> float:
                s = float(0.0)
                for i in range(a.shape[0]):
                    s += a[i]
                return s

            @wp.kernel
            def k(x: wp.array2d[float], y: wp.array[float], ids: wp.array[wp.int32],
                  out: wp.array2d[float], cnt: wp.array[float], outv: wp.array[wp.vec4]):
                t = wp.tid()
                row = x[t, 1:7:2]
                col = x[1:, t]
                tail = y[-3:]
                rev = y[::-1]
                sub2 = rev[1:4]
                s = total(row) + 10.0 * total(col) + 100.0 * total(tail) + 1000.0 * sub2[0] + rev[-1]
                o = out[t:t + 1, :]
                o[0, 2] = s
                wp.atomic_add(cnt[1:], t % 2, 1.0)
                out[t, 0] = float(row.shape[0] * 100 + col.shape[0] * 10 + len(tail))
                o2 = out[t]
                o2[1] = float(sub2.shape[0])
                sub = ids[2 * t:2 * t + 2]
                for i in range(2):
                    nz = sub[i]
                    for j in range(4):
                        outv[nz][j] = float(i * 4 + j)

            X = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
            Y = np.arange(6, dtype=np.float32) * 0.5

            def run(dev):
                x = wp.array(X, device=dev)
                y = wp.array(Y, device=dev)
                ids = wp.array(np.arange(8, dtype=np.int32)[::-1].copy(), device=dev)
                o = wp.zeros((4, 3), dtype=float, device=dev)
                c = wp.zeros(3, dtype=float, device=dev)
                ov = wp.zeros(8, dtype=wp.vec4, device=dev)
                wp.launch(k, 4, [x, y, ids, o, c, ov], device=dev)
                return o.numpy(), c.numpy(), ov.numpy()

            compare(run)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_component_store_through_views(self):
        # ``row = a[t]; row[j][1] = x`` (and through a slice) writes a
        # component of an element of a view of an argument.
        snippet = _snippet(
            """
            @wp.func
            def setc(row: wp.array[wp.vec3], j: int, x: float):
                row[j][1] = x
                row[j][2] += x

            @wp.kernel
            def k(a: wp.array2d[wp.vec3]):
                t = wp.tid()
                row = a[t]
                row[0][0] = float(t)
                setc(a[t], 1, 5.0)
                sub = a[t, 1:3]
                sub[1][2] = 7.0

            def run(dev):
                a = wp.zeros((2, 3), dtype=wp.vec3, device=dev)
                wp.launch(k, 2, [a], device=dev)
                return (a.numpy(),)

            compare(run)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_component_store_in_inlined_func(self):
        # ``nzb_values[i][j, 0] = x`` / ``v[i][1] = x`` inside a ``@wp.func``
        # on an array the function was passed (Kamino's Jacobian assembly):
        # the slot IR only appears once the body is inlined.
        snippet = _snippet(
            """
            mat61f = wp.types.matrix(shape=(6, 1), dtype=wp.float32)
            mat66f = wp.types.matrix(shape=(6, 6), dtype=wp.float32)

            @wp.func
            def store_block(nzb_id: int, block: mat66f, nzb_values: wp.array[mat61f], vv: wp.array[wp.vec3]):
                for i in range(6):
                    for j in range(6):
                        nzb_values[nzb_id + i][j, 0] = block[j, i]
                vv[nzb_id][1] = float(nzb_id)
                vv[nzb_id][2] += 2.0

            @wp.kernel
            def k(vals: wp.array[mat61f], vv: wp.array[wp.vec3]):
                t = wp.tid()
                b = mat66f(0.0)
                for i in range(6):
                    for j in range(6):
                        b[i, j] = float(i * 6 + j + t)
                store_block(t * 6, b, vals, vv)

            def run(dev):
                v = wp.zeros(12, dtype=mat61f, device=dev)
                vv = wp.zeros(12, dtype=wp.vec3, device=dev)
                wp.launch(k, 2, [v, vv], device=dev)
                return v.numpy(), vv.numpy()

            compare(run)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_nested_struct_field_store(self):
        # ``p.geom_a = g`` stores a whole struct into a nested-struct field
        # of a struct local (Newton's speculative contact query).
        snippet = _snippet(
            """
            @wp.struct
            class Geo:
                scale: wp.vec3
                kind: int

            @wp.struct
            class Pair:
                a: int
                geom_a: Geo
                geom_b: Geo

            @wp.func
            def make_pair(s: float, flip: bool) -> Pair:
                g = Geo()
                g.scale = wp.vec3(s, 2.0 * s, 3.0)
                g.kind = 4
                h = Geo()
                h.scale = wp.vec3(-s)
                h.kind = 7
                p = Pair()
                p.a = 1
                p.geom_a = g
                p.geom_b = h
                if flip:
                    p.geom_a = h
                return p

            @wp.kernel
            def k(out: wp.array[wp.vec3], ok: wp.array[int]):
                t = wp.tid()
                p = make_pair(float(t), t == 1)
                out[t] = p.geom_a.scale + p.geom_b.scale
                ok[t] = p.geom_a.kind * 10 + p.geom_b.kind

            def run(dev):
                o = wp.zeros(2, dtype=wp.vec3, device=dev)
                c = wp.zeros(2, dtype=int, device=dev)
                wp.launch(k, 2, [o, c], device=dev)
                return o.numpy(), c.numpy()

            res = compare(run)
            np.testing.assert_array_equal(res["metal:0"][1], [47, 77])
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_static_factory_func_called_twice(self):
        # Two ``wp.static(factory(...))(...)`` calls in one kernel resolve to
        # two distinct functions; both must be inlined (warp_nn's GRU cell
        # left the first one as an undeclared call).
        snippet = _snippet(
            """
            def make(shape):
                (d0, d1) = shape

                @wp.func
                def function(A: wp.array2d[float], B: wp.array2d[float], index: tuple[int, int]):
                    i, j = index[0], index[1]
                    C = wp.tile_zeros(shape=(d1, d0), dtype=A.dtype)
                    a = wp.tile_load(A, shape=(d1, d0), offset=(j * d1, 0))
                    b = wp.tile_load(B, shape=(d0, d0), offset=(i * d0, 0))
                    wp.tile_matmul(a, wp.tile_transpose(b), C)
                    return C

                return function

            def mk_kernel(shape):
                @wp.kernel
                def kernel(A: wp.array2d[float], B: wp.array2d[float], out: wp.array2d[float]):
                    i, j = wp.tid()
                    g = wp.static(make(shape))(A, B, index=(i, j))
                    h = wp.static(make(shape))(B, A, index=(i, j))
                    wp.tile_store(out, wp.tile_transpose(g + h), offset=(i * wp.static(shape[0]), j * wp.static(shape[1])))

                return kernel

            k = mk_kernel((2, 2))
            rng = np.random.default_rng(1)
            A = rng.random((4, 2)).astype(np.float32)
            B = rng.random((4, 2)).astype(np.float32)

            def run(dev):
                a = wp.array(A, device=dev)
                b = wp.array(B, device=dev)
                o = wp.zeros((4, 4), dtype=float, device=dev)
                wp.launch_tiled(k, dim=(2, 2), inputs=[a, b, o], block_dim=1, device=dev)
                return (o.numpy(),)

            compare(run)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_runtime_rebound_array_locals(self):
        # mujoco_warp's polygon clipping swaps two scratch arrays inside a
        # dynamic loop (``tmp = a; a = b; b = tmp``): rows of 2-D arguments
        # in one kernel, whole 1-D arguments in the other.
        snippet = _snippet(
            """
            @wp.func
            def clip(n: int, polygon_out: wp.array[wp.vec3], clipped_out: wp.array[wp.vec3]) -> int:
                npoly = n
                for e in range(n):
                    nclip = int(0)
                    for i in range(npoly):
                        P = polygon_out[i]
                        if P[0] > float(e):
                            clipped_out[nclip] = P * 0.5
                            nclip += 1
                        clipped_out[nclip] = P + wp.vec3(1.0)
                        nclip += 1
                        if nclip >= clipped_out.shape[0]:
                            break
                    tmp = polygon_out
                    polygon_out = clipped_out
                    clipped_out = tmp
                    npoly = nclip
                s = int(0)
                for i in range(npoly):
                    s += int(polygon_out[i][0])
                return s

            @wp.kernel
            def k2d(poly: wp.array2d[wp.vec3], clipped: wp.array2d[wp.vec3], out: wp.array[int]):
                t = wp.tid()
                out[t] = clip(3, poly[t], clipped[t])

            @wp.kernel
            def k1d(poly: wp.array[wp.vec3], clipped: wp.array[wp.vec3], out: wp.array[int]):
                out[0] = clip(4, poly, clipped) + int(poly[0][0] * 100.0)

            rng = np.random.default_rng(0)
            P2 = (rng.random((2, 16, 3)) * 4).astype(np.float32)
            P1 = (rng.random((40, 3)) * 4).astype(np.float32)

            def run(dev):
                p = wp.array(P2, dtype=wp.vec3, device=dev)
                c = wp.zeros((2, 16), dtype=wp.vec3, device=dev)
                o = wp.zeros(2, dtype=int, device=dev)
                wp.launch(k2d, 2, [p, c, o], device=dev)
                p1 = wp.array(P1, dtype=wp.vec3, device=dev)
                c1 = wp.zeros(40, dtype=wp.vec3, device=dev)
                o1 = wp.zeros(1, dtype=int, device=dev)
                wp.launch(k1d, 1, [p1, c1, o1], device=dev)
                return o.numpy(), p.numpy(), c.numpy(), o1.numpy(), p1.numpy(), c1.numpy()

            compare(run)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_struct_array_element_array_fields(self):
        # Newton's VBD kernels read ``collision_info_array[0].<array field>``:
        # the element's arrays are bound as kernel inputs at launch.
        snippet = _snippet(
            """
            @wp.struct
            class Info:
                counts: wp.array[wp.int32]
                offsets: wp.array[wp.int32]
                pairs: wp.array[wp.int32]
                dist: wp.array[float]
                scale: float

            @wp.func
            def get_pair(info: Info, v: int, k: int):
                off = info.offsets[v]
                return info.pairs[2 * (off + k) + 1]

            @wp.kernel
            def k(infos: wp.array[Info], out: wp.array[float]):
                t = wp.tid()
                info = infos[0]
                s = float(0.0)
                if t < info.counts.shape[0]:
                    for j in range(info.counts[t]):
                        s += float(get_pair(info, t, j)) * info.dist[t]
                out[t] = s * info.scale + float(infos[0].offsets[t])

            def run(dev):
                info = Info()
                info.counts = wp.array([1, 2, 0], dtype=wp.int32, device=dev)
                info.offsets = wp.array([0, 1, 3], dtype=wp.int32, device=dev)
                info.pairs = wp.array(np.arange(8, dtype=np.int32) * 3, device=dev)
                info.dist = wp.array([0.5, 2.0, 1.0], dtype=float, device=dev)
                info.scale = 2.0
                infos = wp.array([info], dtype=Info, device=dev)
                o = wp.zeros(3, dtype=float, device=dev)
                wp.launch(k, 3, [infos, o], device=dev)
                return (o.numpy(),)

            res = compare(run)
            np.testing.assert_allclose(res["metal:0"][0], [3.0, 97.0, 3.0])
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_tile_broadcast_column(self):
        # A 2-D ``(R, 1)`` column broadcast to a square ``(R, R)`` tile
        # repeats along the columns; it was transposed like a 1-D ``(R,)``
        # tile (warp_nn's RNN/GRU bias add, reached once the cells inline).
        snippet = _snippet(
            """
            @wp.kernel
            def k(b: wp.array2d[float], v: wp.array[float], out: wp.array2d[float], out2: wp.array2d[float]):
                i, j = wp.tid()
                t = wp.tile_zeros(shape=(4, 4), dtype=float)
                t += wp.tile_broadcast(wp.tile_load(b, shape=(4, 1), offset=(0, 0)), shape=(4, 4))
                wp.tile_store(out, t, offset=(0, 0))
                u = wp.tile_broadcast(wp.tile_load(v, shape=4, offset=0), shape=(4, 4))
                wp.tile_store(out2, u, offset=(0, 0))

            def run(dev):
                b = wp.array(np.arange(4, dtype=np.float32).reshape(4, 1) + 1, device=dev)
                v = wp.array(np.arange(4, dtype=np.float32) + 10, device=dev)
                o = wp.zeros((4, 4), dtype=float, device=dev)
                o2 = wp.zeros((4, 4), dtype=float, device=dev)
                for bd in (1, 64):
                    wp.launch_tiled(k, dim=(1, 1), inputs=[b, v, o, o2], block_dim=bd, device=dev)
                return o.numpy(), o2.numpy()

            res = compare(run)
            np.testing.assert_array_equal(res["metal:0"][0][:, 3], [1.0, 2.0, 3.0, 4.0])
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_tile_map_inside_inlined_func(self):
        # ``wp.tile_map(user_fn, ...)`` inside a ``@wp.func`` reached through
        # ``wp.static`` (mujoco_warp's Newton decrement) only appears once
        # the caller is inlined; the mapped function must still be inlined.
        snippet = _snippet(
            """
            @wp.func
            def sums(grad: float, solution: float):
                return wp.vec2(solution * solution, grad * solution)

            def make(n: int):
                @wp.func
                def dec(sol: wp.tile[float, n, 1], b: wp.array2d[float]):
                    g = wp.tile_load(b, shape=(n, 1), offset=(0, 0))
                    return wp.tile_reduce(wp.add, wp.tile_map(sums, g, sol))[0]

                return dec

            @wp.kernel
            def k(a: wp.array2d[float], b: wp.array2d[float], out: wp.array[wp.vec2]):
                sol = wp.tile_load(a, shape=(6, 1), offset=(0, 0))
                out[0] = wp.static(make(6))(sol, b)

            rng = np.random.default_rng(0)
            A = rng.random((6, 1)).astype(np.float32)
            B = rng.random((6, 1)).astype(np.float32)

            def run(dev):
                a = wp.array(A, device=dev)
                b = wp.array(B, device=dev)
                o = wp.zeros(1, dtype=wp.vec2, device=dev)
                wp.launch_tiled(k, dim=1, inputs=[a, b, o], block_dim=1, device=dev)
                return (o.numpy(),)

            compare(run)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_tile_view_column_in_inlined_func(self):
        # ``wp.tile_view(sol, shape=(m, 1))`` of an ``(n, 1)`` shared tile in
        # a ``wp.static`` factory function (mujoco_warp's Newton decrement):
        # the column coordinate of a unit dim was the row index, so the view
        # read the parent's diagonal.
        snippet = _snippet(
            """
            @wp.func
            def sums(grad: float, solution: float):
                return wp.vec2(solution * solution, grad * solution)

            def make(n: int, m: int):
                @wp.func
                def dec(sol: wp.tile[float, n, 1], b: wp.array2d[float], out: wp.array2d[float]):
                    g = wp.tile_load(b, shape=(m, 1), offset=(0, 0), bounds_check=False)
                    act = wp.tile_view(sol, shape=(m, 1), offset=(0, 0))
                    wp.tile_store(out, wp.tile_map(wp.mul, act, -1.0), bounds_check=False)
                    return wp.tile_reduce(wp.add, wp.tile_map(sums, g, act))[0]

                return dec

            @wp.kernel
            def k(a: wp.array2d[float], b: wp.array2d[float], out: wp.array2d[float], res: wp.array[wp.vec2]):
                sol = wp.tile_load(a, shape=(8, 1), offset=(0, 0), storage="shared")
                res[0] = wp.static(make(8, 6))(sol, b, out)

            rng = np.random.default_rng(0)
            A = rng.random((8, 1)).astype(np.float32)
            B = rng.random((6, 1)).astype(np.float32)

            def run(dev):
                a = wp.array(A, device=dev)
                b = wp.array(B, device=dev)
                o = wp.zeros((6, 1), dtype=float, device=dev)
                r = wp.zeros(1, dtype=wp.vec2, device=dev)
                wp.launch_tiled(k, dim=1, inputs=[a, b, o, r], block_dim=1, device=dev)
                return r.numpy(), o.numpy()

            compare(run)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_default_struct_array_fields_are_empty(self):
        # ``p = Provider()`` leaves its array fields empty; Newton's support
        # functions guard each read with ``idx < p.data.shape[0]``.
        snippet = _snippet(
            """
            @wp.struct
            class Provider:
                data: wp.array[wp.vec4i]
                lut: wp.array[int]

            @wp.func
            def has_hull(p: Provider, idx: int) -> bool:
                if idx < 0 or idx >= p.data.shape[0]:
                    return False
                packed = p.data[idx][3]
                return packed > 0

            @wp.func
            def pick(p: Provider, idx: int, fallback: float) -> float:
                if has_hull(p, idx):
                    return float(p.lut[idx])
                return fallback

            @wp.kernel
            def k(x: wp.array[float], provided: wp.array[wp.vec4i], out: wp.array[float]):
                t = wp.tid()
                p = Provider()
                q = p
                full = Provider()
                full.data = provided
                out[t] = pick(q, t, x[t]) + float(full.data.shape[0]) + float(len(p.lut))

            def run(dev):
                x = wp.array([1.0, 2.0], dtype=float, device=dev)
                pv = wp.zeros(3, dtype=wp.vec4i, device=dev)
                o = wp.zeros(2, dtype=float, device=dev)
                wp.launch(k, 2, [x, pv, o], device=dev)
                return (o.numpy(),)

            res = compare(run)
            np.testing.assert_allclose(res["metal:0"][0], [4.0, 5.0])
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_array_handle_copies(self):
        # ``on = plane_normal`` copies an array handle (mujoco_warp's polygon
        # clipping); writes through the copy reach the argument.
        snippet = _snippet(
            """
            @wp.func
            def fill(plane_normal: wp.array[wp.vec3], plane_dist: wp.array[float], n: int):
                on = plane_normal
                pd = plane_dist
                for i in range(n):
                    pd[i] = float(i) * 2.0
                    on[i] = wp.vec3(float(i))
                return pd[n - 1] + on[0][1] + float(on.shape[0])

            @wp.kernel
            def k(pnorm: wp.array2d[wp.vec3], pdist: wp.array2d[float], out: wp.array[float]):
                t = wp.tid()
                out[t] = fill(pnorm[t], pdist[t], 3)

            def run(dev):
                a = wp.zeros((2, 4), dtype=wp.vec3, device=dev)
                b = wp.zeros((2, 4), dtype=float, device=dev)
                o = wp.zeros(2, dtype=float, device=dev)
                wp.launch(k, 2, [a, b, o], device=dev)
                return o.numpy(), a.numpy(), b.numpy()

            compare(run)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_struct_array_dynamic_element_array_fields(self):
        # ``data[gid].field[i]`` with a runtime element index (Newton's Gaussian
        # splat bounds): each field reads through a per-element pointer table.
        snippet = _snippet(
            """
            @wp.struct
            class GData:
                num_points: wp.int32
                transforms: wp.array[wp.transformf]
                scales: wp.array[wp.vec3f]
                opacities: wp.array[wp.float32]
                grid: wp.array2d[wp.float32]
                rot: wp.array[wp.mat33]
                min_response: wp.float32

            @wp.func
            def bounds(g: GData, i: int):
                t = g.transforms[i]
                s = g.scales[i] * g.opacities[i]
                return wp.transform_get_translation(t) + s + wp.vec3(g.min_response) + g.rot[0] * wp.vec3(1.0, 2.0, 3.0)

            @wp.kernel
            def k(ids: wp.array[wp.uint64], data: wp.array[GData], out: wp.array[wp.vec3], cnt: wp.array[float]):
                tid = wp.tid()
                gid = ids[tid]
                acc = wp.vec3(0.0)
                for i in range(data[gid].num_points):
                    acc += bounds(data[gid], i)
                out[tid] = acc
                d = data[gid]
                cnt[tid] = float(d.scales.shape[0] * 10 + len(d.opacities)) + d.grid[1, 2] + float(d.grid.shape[1])

            def run(dev):
                rng = np.random.default_rng(0)
                items = []
                for g in range(3):
                    n = g + 2
                    gd = GData()
                    gd.num_points = n
                    gd.transforms = wp.array(rng.random((n, 7)).astype(np.float32), dtype=wp.transformf, device=dev)
                    gd.scales = wp.array(rng.random((n, 3)).astype(np.float32), dtype=wp.vec3f, device=dev)
                    gd.opacities = wp.array(rng.random(n).astype(np.float32), device=dev)
                    gd.grid = wp.array(rng.random((2, 4)).astype(np.float32), device=dev)
                    gd.rot = wp.array(rng.random((1, 3, 3)).astype(np.float32), dtype=wp.mat33, device=dev)
                    gd.min_response = 0.1 * g
                    items.append(gd)
                data = wp.array(items, dtype=GData, device=dev)
                ids = wp.array([2, 0, 1, 2], dtype=wp.uint64, device=dev)
                o = wp.zeros(4, dtype=wp.vec3, device=dev)
                c = wp.zeros(4, dtype=float, device=dev)
                wp.launch(k, 4, [ids, data, o, c], device=dev)
                # Rebinding an element's arrays on the host is seen by the next launch.
                items[0].scales = wp.zeros(2, dtype=wp.vec3f, device=dev)
                data.assign(wp.array(items, dtype=GData, device=dev))
                o2 = wp.zeros(4, dtype=wp.vec3, device=dev)
                wp.launch(k, 4, [ids, data, o2, c], device=dev)
                return o.numpy(), c.numpy(), o2.numpy()

            compare(run)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_fixed_array_ptr_of_view_and_misaligned_offset(self):
        # ``.ptr`` of a row view is the row's byte address (the element size,
        # not the element count, scales the offset); a constant ``ptr``
        # offset that is not a multiple of the element size raises instead
        # of being truncated.
        snippet = _snippet(
            """
            @wp.kernel
            def k(out: wp.array2d[float]):
                tid = wp.tid()
                m = wp.zeros(shape=(3, 4), dtype=float)
                for i in range(3):
                    for j in range(4):
                        m[i, j] = float(i * 4 + j + tid)
                r = m[1]
                sub = wp.array(ptr=r.ptr + wp.uint64(4), shape=(2,), dtype=float)
                out[tid, 0] = sub[0]
                out[tid, 1] = sub[1]

            def run(dev):
                o = wp.zeros((3, 2), dtype=float, device=dev)
                wp.launch(k, 3, [o], device=dev)
                return (o.numpy(),)

            res = compare(run)
            np.testing.assert_allclose(res["metal:0"][0][0], [5.0, 6.0])

            @wp.kernel
            def k_misaligned(out: wp.array[float]):
                buf = wp.zeros(shape=(4,), dtype=wp.int32)
                sub = wp.array(ptr=buf.ptr + wp.uint64(2), shape=(2,), dtype=wp.int32)
                out[0] = float(sub[0])

            from warp._src.codegen_metal import MetalCodegenError
            try:
                wp.launch(k_misaligned, 1, [wp.zeros(1, dtype=float, device="metal:0")], device="metal:0")
            except MetalCodegenError as e:
                assert "multiple of" in str(e), e
            else:
                raise AssertionError("misaligned ptr offset was not rejected")
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_struct_array_4d_field_runtime_index(self):
        # Pointer-table rows carry all four extents; a 4-D field indexed its
        # last dim with a garbage extent.
        snippet = _snippet(
            """
            @wp.struct
            class G4:
                m: wp.array4d[float]

            @wp.kernel
            def k(ids: wp.array[int], data: wp.array[G4], out: wp.array2d[float]):
                t = wp.tid()
                g = ids[t]
                out[t, 0] = data[g].m[1, 0, 1, 2]
                out[t, 1] = float(data[g].m.shape[3])
                out[t, 2] = data[g].m[-1, -1, -1, -1]

            def run(dev):
                items = []
                for gi in range(2):
                    x = G4()
                    x.m = wp.array(
                        (np.arange(2 * 2 * 2 * (3 + gi), dtype=np.float32) + 100 * gi).reshape(2, 2, 2, 3 + gi),
                        device=dev,
                    )
                    items.append(x)
                data = wp.array(items, dtype=G4, device=dev)
                o = wp.zeros((2, 3), dtype=float, device=dev)
                wp.launch(k, 2, [wp.array([1, 0], dtype=int, device=dev), data, o], device=dev)
                return (o.numpy(),)

            compare(run)
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)

    def test_struct_array_element_arrays_reject_capture(self):
        # The element records are resolved to buffers on the host at launch;
        # a graph replay would keep reading the arrays bound at capture time,
        # so capturing such a kernel raises instead of going stale.
        snippet = _snippet(
            """
            @wp.struct
            class G:
                vals: wp.array[float]

            @wp.kernel
            def kc(data: wp.array[G], out: wp.array[float]):
                out[wp.tid()] = data[0].vals[wp.tid()]

            @wp.kernel
            def kd(ids: wp.array[int], data: wp.array[G], out: wp.array[float]):
                out[wp.tid()] = data[ids[0]].vals[wp.tid()]

            dev = "metal:0"
            g = G()
            g.vals = wp.array([1.0, 2.0], dtype=float, device=dev)
            data = wp.array([g], dtype=G, device=dev)
            out = wp.zeros(2, dtype=float, device=dev)
            ids = wp.zeros(1, dtype=int, device=dev)
            wp.launch(kc, 2, [data, out], device=dev)
            wp.launch(kd, 2, [ids, data, out], device=dev)
            np.testing.assert_allclose(out.numpy(), [1.0, 2.0])
            for kern, args in ((kc, [data, out]), (kd, [ids, data, out])):
                try:
                    with wp.ScopedCapture(device=dev):
                        wp.launch(kern, 2, args, device=dev)
                except RuntimeError as e:
                    assert "cannot be captured" in str(e), e
                else:
                    raise AssertionError("capturing a struct-array array-field read was not rejected")
            # Rebinding a record to a new array is seen by the next launch.
            g.vals = wp.array([5.0, 6.0], dtype=float, device=dev)
            data.assign(wp.array([g], dtype=G, device=dev))
            wp.launch(kd, 2, [ids, data, out], device=dev)
            np.testing.assert_allclose(out.numpy(), [5.0, 6.0])
            """
        )
        _run_with_metal_enabled(self, snippet, timeout=120)


if __name__ == "__main__":
    unittest.main(verbosity=2)
