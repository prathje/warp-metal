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

"""Metal reverse mode for tile_matmul and the mesh closest-point queries.

Gradients from ``metal:0`` are compared with ``cpu`` (``wp.Tape``) and, where
the loss is smooth, with central finite differences. Each case runs in a
subprocess because the Metal device must be enabled before ``wp.init()``
(see ``test_metal_launch``).
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


def _run(test_case, body: str, timeout: int = 300):
    _run_with_metal_native(test_case, _PRELUDE + textwrap.dedent(body), timeout=timeout)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalTileMatmulAdjoint(unittest.TestCase):
    def test_return_form_serial_matches_cpu_and_fd(self):
        # ``out = wp.tile_matmul(w, p)`` followed by a tile_map, the
        # example_diffsim_bear network, at the block widths Newton uses
        # (64, which is serial on Metal, and 32 with only loaded tiles,
        # which also stays serial).
        _run(
            self,
            """
            TT, PC, NB = 8, 8, 3

            @wp.func
            def act(x: float):
                return wp.tanh(x) * 0.7

            @wp.kernel
            def network(phases: wp.array2d[float], weights: wp.array2d[float], out_a: wp.array2d[float]):
                i = wp.tid()
                p = wp.tile_load(phases, shape=(PC, 1))
                w = wp.tile_load(weights, shape=(TT, PC), offset=(i * TT, 0))
                out = wp.tile_matmul(w, p)
                wp.tile_store(out_a, wp.tile_map(act, out), offset=(i * TT, 0))

            @wp.kernel
            def lossk(a: wp.array2d[float], c: wp.array2d[float], loss: wp.array[float]):
                i = wp.tid()
                wp.atomic_add(loss, 0, a[i, 0] * c[i, 0])

            rng = np.random.default_rng(0)
            ph0 = rng.standard_normal((PC, 1)).astype(np.float32)
            W0 = (rng.standard_normal((NB * TT, PC)) * 0.3).astype(np.float32)
            C0 = rng.standard_normal((NB * TT, 1)).astype(np.float32)

            def run(dev, bd, ph=ph0):
                p = wp.array(ph, device=dev, requires_grad=True)
                w = wp.array(W0, device=dev, requires_grad=True)
                a = wp.zeros((NB * TT, 1), dtype=float, device=dev, requires_grad=True)
                loss = wp.zeros(1, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch_tiled(network, dim=NB, inputs=[p, w], outputs=[a], block_dim=bd, device=dev)
                    wp.launch(lossk, dim=NB * TT, inputs=[a, wp.array(C0, device=dev), loss], device=dev)
                tape.backward(loss)
                return loss.numpy()[0], p.grad.numpy(), w.grad.numpy()

            for bd in (64, 32):
                res = both(lambda dev: run(dev, bd))
                assert np.abs(res["cpu"][2]).max() > 0.1
                assert_same(res, atol=1e-5)
                eps = 1e-2
                for k in range(PC):
                    d = np.zeros_like(ph0)
                    d[k] = eps
                    fd = (run("metal:0", bd, ph0 + d)[0] - run("metal:0", bd, ph0 - d)[0]) / (2 * eps)
                    assert abs(fd - res["metal:0"][1][k, 0]) < 1e-3 * max(1.0, abs(fd)), (bd, k, fd)
            """,
        )

    def test_return_form_shapes_alpha_transpose_and_aliasing(self):
        # Rectangular operands, ``alpha``, a transposed operand, the same tile
        # as both operands (adj_A and adj_B are one dual), a chained product,
        # and the accumulating form next to it.
        _run(
            self,
            """
            R, K, N = 5, 3, 4

            @wp.kernel
            def k1(A: wp.array2d[float], B: wp.array2d[float], S: wp.array2d[float], C: wp.array2d[float],
                   out: wp.array2d[float], out2: wp.array2d[float], out3: wp.array2d[float]):
                i = wp.tid()
                a = wp.tile_load(A, shape=(R, K))
                b = wp.tile_load(B, shape=(K, N))
                s = wp.tile_load(S, shape=(R, R))
                c = wp.tile_load(C, shape=(N, R))
                ab = wp.tile_matmul(a, b, alpha=0.5)
                wp.tile_store(out, wp.tile_map(wp.sin, ab))
                sq = wp.tile_matmul(s, s)
                wp.tile_store(out2, wp.tile_map(wp.mul, sq, sq))
                abc = wp.tile_matmul(wp.tile_matmul(a, b), c)
                acc = wp.tile_map(wp.cos, s)
                wp.tile_matmul(abc, s, acc, alpha=-0.25, beta=0.5)
                wp.tile_store(out3, acc)

            @wp.kernel
            def k2(A: wp.array2d[float], B: wp.array2d[float], S: wp.array2d[float], C: wp.array2d[float],
                   out: wp.array2d[float], out2: wp.array2d[float], out3: wp.array2d[float]):
                i = wp.tid()
                s = wp.tile_load(S, shape=(R, R))
                c = wp.tile_load(C, shape=(N, R))
                st = wp.tile_matmul(s, wp.tile_transpose(c), alpha=2.0)
                wp.tile_store(out, wp.tile_map(wp.sin, st))

            @wp.kernel
            def k3(A: wp.array2d[float], B: wp.array2d[float], S: wp.array2d[float], C: wp.array2d[float],
                   out: wp.array2d[float], out2: wp.array2d[float], out3: wp.array2d[float]):
                # A tile times its own transpose (the value-returning form
                # names S and its transpose view in one statement).
                i = wp.tid()
                s = wp.tile_load(S, shape=(R, R))
                wp.tile_store(out2, wp.tile_matmul(s, wp.tile_transpose(s)))

            def run(dev, k):
                rng = np.random.default_rng(1)
                vals = [rng.standard_normal(sh).astype(np.float32) for sh in ((R, K), (K, N), (R, R), (N, R))]
                ws = [rng.standard_normal(sh).astype(np.float32) for sh in ((R, N), (R, R), (R, R))]
                with wp.ScopedDevice(dev):
                    ins = [wp.array(v, requires_grad=True) for v in vals]
                    outs = [wp.zeros(w.shape, dtype=float, requires_grad=True) for w in ws]
                    tape = wp.Tape()
                    with tape:
                        wp.launch_tiled(k, dim=1, inputs=[*ins], outputs=outs, block_dim=64)
                    tape.backward(grads={o: wp.array(w) for o, w in zip(outs, ws)})
                    return [o.numpy() for o in outs] + [x.grad.numpy() for x in ins]

            res = both(lambda dev: run(dev, k1))
            for g in res["cpu"][3:]:
                assert np.abs(g).max() > 0.01
            assert_same(res, rtol=1e-4, atol=1e-4)
            res = both(lambda dev: run(dev, k2))
            for g in res["cpu"][5:]:
                assert np.abs(g).max() > 0.01
            assert_same(res, rtol=1e-4, atol=1e-4)
            res = both(lambda dev: run(dev, k3))
            assert np.abs(res["cpu"][5]).max() > 0.01
            assert_same(res, rtol=1e-4, atol=1e-4)
            """,
        )

    def test_tile_view_dual_reaches_parent(self):
        # The dual of a ``tile_view`` aliases a window of the parent's dual on
        # CPU/CUDA. On Metal the view is a private copy, and its dual was never
        # added back: every gradient through a view came back zero. Covers a
        # view read by tile_map, by the value-returning and the accumulating
        # tile_matmul, a view of a view, and the parent also used directly.
        # A view written in place raises.
        _run(
            self,
            """
            @wp.kernel
            def k(A: wp.array2d[float], B: wp.array2d[float], O: wp.array2d[float], O2: wp.array2d[float],
                  O3: wp.array2d[float]):
                big = wp.tile_map(wp.mul, wp.tile_load(A, shape=(4, 6)), wp.tile_load(A, shape=(4, 6)))
                b = wp.tile_load(B, shape=(3, 3))
                v = wp.tile_view(big, offset=(0, 2), shape=(4, 3))
                wp.tile_store(O, wp.tile_map(wp.add, wp.tile_map(wp.sin, v), wp.tile_matmul(v, b)))
                acc = wp.tile_zeros(shape=(4, 3), dtype=float)
                wp.tile_matmul(wp.tile_view(big, offset=(0, 1), shape=(4, 3)), b, acc)
                wp.tile_store(O2, acc)
                vv = wp.tile_view(wp.tile_view(big, offset=(1, 1), shape=(3, 4)), offset=(1, 0), shape=(2, 3))
                wp.tile_store(O3, wp.tile_map(wp.add, wp.tile_map(wp.cos, vv),
                                              wp.tile_view(big, offset=(0, 0), shape=(2, 3))))

            @wp.kernel
            def k_bad(A: wp.array2d[float], B: wp.array2d[float], O: wp.array2d[float], O2: wp.array2d[float],
                      O3: wp.array2d[float]):
                big = wp.tile_map(wp.mul, wp.tile_load(A, shape=(4, 6)), wp.tile_load(A, shape=(4, 6)))
                b = wp.tile_load(B, shape=(3, 3))
                v = wp.tile_view(big, offset=(0, 2), shape=(4, 3))
                wp.tile_matmul(v, b, v)
                wp.tile_store(O, wp.tile_view(big, offset=(0, 0), shape=(4, 3)))

            rng = np.random.default_rng(5)
            A0 = rng.standard_normal((4, 6)).astype(np.float32)
            B0 = rng.standard_normal((3, 3)).astype(np.float32)
            Ws = [rng.standard_normal(sh).astype(np.float32) for sh in ((4, 3), (4, 3), (2, 3))]

            def run(dev, kern=k):
                with wp.ScopedDevice(dev):
                    a = wp.array(A0, requires_grad=True)
                    b = wp.array(B0, requires_grad=True)
                    outs = [wp.zeros(w.shape, dtype=float, requires_grad=True) for w in Ws]
                    tape = wp.Tape()
                    with tape:
                        wp.launch_tiled(kern, dim=1, inputs=[a, b], outputs=outs, block_dim=64)
                    tape.backward(grads={o: wp.array(w) for o, w in zip(outs, Ws)})
                    return [o.numpy() for o in outs] + [a.grad.numpy(), b.grad.numpy()]

            res = both(run)
            assert np.abs(res["cpu"][3]).max() > 0.5 and np.abs(res["cpu"][4]).max() > 0.5
            assert_same(res, rtol=1e-4, atol=1e-4)
            try:
                run("metal:0", k_bad)
            except Exception as e:
                assert "written in place" in str(e), e
            else:
                raise AssertionError("reverse through a view written in place did not raise")
            """,
        )


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalMeshQueryAdjoint(unittest.TestCase):
    _MESH = """
        # A closed, irregular five-vertex polyhedron; query points all around
        # it land in the vertex, edge and face regions of the closest-point
        # adjoint.
        pts_np = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.9, 0.8, 0.7]], dtype=np.float32)
        idx_np = np.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 4, 2, 3, 4, 3, 1, 4], dtype=np.int32)
        rng = np.random.default_rng(3)
        P0 = rng.uniform(-0.6, 1.6, size=(192, 3)).astype(np.float32)

        def run(kern, dev, P=P0, aux=False):
            mesh = wp.Mesh(points=wp.array(pts_np, dtype=wp.vec3, device=dev),
                           indices=wp.array(idx_np, dtype=wp.int32, device=dev), support_winding_number=True)
            x = wp.array(P, dtype=wp.vec3, device=dev, requires_grad=True)
            out = wp.zeros(len(P), dtype=float, device=dev, requires_grad=True)
            face = wp.zeros(len(P), dtype=int, device=dev)
            tape = wp.Tape()
            with tape:
                wp.launch(kern, dim=len(P), inputs=[mesh.id, x, out, face], device=dev)
            tape.backward(grads={out: wp.ones(len(P), dtype=float, device=dev)})
            return out.numpy().copy(), x.grad.numpy().copy(), face.numpy().copy()
    """

    def test_struct_queries_match_cpu_and_fd(self):
        # Every struct-returning closest-point query, with a loss that reads
        # the face, the barycentrics (through mesh_eval_position and
        # directly) and the sign. Points whose closest feature is shared by
        # several faces may pick a different face on each device (BVH
        # traversal order), so gradients are compared where the faces agree.
        body = (
            self._MESH
            + """
        @wp.func
        def contrib(mesh: wp.uint64, p: wp.vec3, q: wp.MeshQueryPoint):
            d = p - wp.mesh_eval_position(mesh, q.face, q.u, q.v)
            return q.sign * wp.dot(d, d) + 0.3 * q.u - 0.7 * q.v

        @wp.func
        def query_normal(mesh: wp.uint64, p: wp.vec3):
            # returned from a @wp.func: the reverse pass accumulates the
            # query struct's dual
            return wp.mesh_query_point_sign_normal(mesh, p, 10.0)

        def make(n):
            @wp.kernel(module="unique")
            def k(mesh: wp.uint64, pts: wp.array[wp.vec3], out: wp.array[float], face: wp.array[int]):
                i = wp.tid()
                p = pts[i]
                if wp.static(n == 0):
                    q = wp.mesh_query_point_sign_parity(mesh, p, 10.0)
                elif wp.static(n == 1):
                    q = query_normal(mesh, p)
                elif wp.static(n == 2):
                    q = wp.mesh_query_point(mesh, p, 10.0)
                elif wp.static(n == 3):
                    q = wp.mesh_query_point_no_sign(mesh, p, 10.0)
                elif wp.static(n == 4):
                    q = wp.mesh_query_point_sign_winding_number(mesh, p, 10.0)
                else:
                    q = wp.mesh_query_furthest_point_no_sign(mesh, p, 0.0)
                if q.result:
                    out[i] = contrib(mesh, p, q)
                    face[i] = q.face

            return k

        for n in range(6):
            k = make(n)
            oc, gc, fc = run(k, "cpu")
            om, gm, fm = run(k, "metal:0")
            same = fc == fm
            assert same.mean() > 0.85, (n, same.mean())
            assert np.abs(gc[same]).max() > 0.5, n
            np.testing.assert_allclose(om[same], oc[same], rtol=1e-5, atol=1e-5, err_msg=str(n))
            np.testing.assert_allclose(gm[same], gc[same], rtol=1e-4, atol=1e-5, err_msg=str(n))
        """
        )
        _run(self, body, timeout=600)

    def test_distance_gradient_matches_cpu_and_fd(self):
        # Squared distance to the mesh is face-independent, so CPU and Metal
        # must agree on every point and match finite differences.
        body = (
            self._MESH
            + """
        @wp.kernel
        def kd(mesh: wp.uint64, pts: wp.array[wp.vec3], out: wp.array[float], face: wp.array[int]):
            i = wp.tid()
            p = pts[i]
            q = wp.mesh_query_point_sign_parity(mesh, p, 10.0)
            if q.result:
                out[i] = wp.length_sq(p - wp.mesh_eval_position(mesh, q.face, q.u, q.v))

        oc, gc, _ = run(kd, "cpu")
        om, gm, _ = run(kd, "metal:0")
        np.testing.assert_allclose(om, oc, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(gm, gc, rtol=1e-4, atol=1e-5)
        eps = 1e-3
        for i in range(0, len(P0), 23):
            for c in range(3):
                d = np.zeros_like(P0)
                d[i, c] = eps
                fd = (run(kd, "metal:0", P0 + d)[0][i] - run(kd, "metal:0", P0 - d)[0][i]) / (2 * eps)
                assert abs(fd - gm[i, c]) < 2e-3 * max(1.0, abs(fd)), (i, c, fd, gm[i, c])
        """
        )
        _run(self, body, timeout=600)

    def test_out_parameter_forms_match_cpu(self):
        body = (
            self._MESH
            + """
        def make(n):
            @wp.kernel(module="unique")
            def k(mesh: wp.uint64, pts: wp.array[wp.vec3], out: wp.array[float], fo: wp.array[int]):
                i = wp.tid()
                p = pts[i]
                inside = float(1.0)
                face = int(0)
                u = float(0.0)
                v = float(0.0)
                hit = bool(False)
                if wp.static(n == 0):
                    hit = wp.mesh_query_point_sign_parity(mesh, p, 10.0, inside, face, u, v)
                elif wp.static(n == 1):
                    hit = wp.mesh_query_point_sign_normal(mesh, p, 10.0, inside, face, u, v)
                elif wp.static(n == 2):
                    hit = wp.mesh_query_point(mesh, p, 10.0, inside, face, u, v)
                elif wp.static(n == 3):
                    hit = wp.mesh_query_point_no_sign(mesh, p, 10.0, face, u, v)
                elif wp.static(n == 4):
                    hit = wp.mesh_query_point_sign_winding_number(mesh, p, 10.0, inside, face, u, v)
                else:
                    hit = wp.mesh_query_furthest_point_no_sign(mesh, p, 0.0, face, u, v)
                if hit:
                    out[i] = inside * wp.length_sq(p - wp.mesh_eval_position(mesh, face, u, v)) + 0.3 * u - 0.7 * v
                    fo[i] = face

            return k

        for n in range(6):
            k = make(n)
            oc, gc, fc = run(k, "cpu")
            om, gm, fm = run(k, "metal:0")
            same = fc == fm
            assert same.mean() > 0.85, (n, same.mean())
            np.testing.assert_allclose(om[same], oc[same], rtol=1e-5, atol=1e-5, err_msg=str(n))
            np.testing.assert_allclose(gm[same], gc[same], rtol=1e-4, atol=1e-5, err_msg=str(n))
        """
        )
        _run(self, body, timeout=600)


if __name__ == "__main__":
    unittest.main(verbosity=2)
