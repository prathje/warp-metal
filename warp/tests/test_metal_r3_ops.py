# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal codegen regressions for value ops, matrices and adjoints.

Each test launches a kernel on ``cpu`` and ``metal:0`` in a subprocess (the
Metal device must be enabled before ``wp.init``) and compares the outputs.
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_enabled

_COMMON = textwrap.dedent(
    """
    import numpy as np
    from typing import Any

    def both(launch_fn):
        return launch_fn("cpu"), launch_fn("metal:0")

    def assert_same(cpu, metal, **tol):
        if not isinstance(cpu, (tuple, list)):
            cpu, metal = (cpu,), (metal,)
        for c, m in zip(cpu, metal):
            if tol:
                np.testing.assert_allclose(m, c, **tol)
            else:
                np.testing.assert_array_equal(m, c)
    """
)


def _run(test_case, body: str, timeout: int = 180):
    _run_with_metal_enabled(test_case, _COMMON + textwrap.dedent(body), timeout=timeout)


@unittest.skipUnless(_is_apple_silicon() and _has_mlx(), "Metal requires Apple Silicon")
class TestMetalR3Ops(unittest.TestCase):
    def test_matrix_slices(self):
        # Newton's implicit MPM slices a mat66 into a mat55 (``wp::extract<5, 5>``);
        # Kamino slices rows/columns of a mat66 (big-mat struct operand).
        _run(
            self,
            """
            mat55 = wp.types.matrix(shape=(5, 5), dtype=wp.float32)
            mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
            vec5 = wp.types.vector(5, wp.float32)

            @wp.kernel
            def k(a: wp.array[mat66], o1: wp.array[mat55], o2: wp.array[vec5], o3: wp.array[vec5],
                  o4: wp.array[wp.mat33], o5: wp.array[wp.vec2], o6: wp.array[wp.vec3]):
                i = wp.tid()
                m = a[i]
                o1[i] = m[1:6, 1:6]
                o2[i] = m[1:6, 2]
                o3[i] = m[3, 0:5]
                o4[i] = m[0:3, 3:6]
                m3 = o4[i]
                o5[i] = m3[1:3, 0] + m3[2, 0:2]
                o6[i] = m[0:6:2, 5]

            def launch(dev):
                a = wp.array(np.arange(72, dtype=np.float32).reshape(2, 6, 6) * 0.1, dtype=mat66, device=dev)
                outs = [wp.zeros(2, dtype=t, device=dev) for t in (mat55, vec5, vec5, wp.mat33, wp.vec2, wp.vec3)]
                wp.launch(k, 2, [a], outs, device=dev)
                return [o.numpy() for o in outs]

            cpu, metal = both(launch)
            assert_same(cpu, metal)
            """,
        )

    def test_matrix_slice_adjoints(self):
        _run(
            self,
            """
            mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)

            @wp.kernel
            def k(a: wp.array[mat66], b: wp.array[wp.mat33], out: wp.array[float]):
                i = wp.tid()
                m = a[i]
                s = m[1:6, 1:6]
                v = m[1:4, 2]
                n = b[i]
                w = n[1:3, 0]
                out[i] = wp.ddot(s, s) + wp.dot(v, v) * 2.0 + wp.ddot(n[0:2, 1:3], n[0:2, 1:3]) + w[1]

            def launch(dev):
                a = wp.array(np.arange(72, dtype=np.float32).reshape(2, 6, 6) * 0.1, dtype=mat66, device=dev,
                             requires_grad=True)
                b = wp.array(np.arange(18, dtype=np.float32).reshape(2, 3, 3) * 0.1, dtype=wp.mat33, device=dev,
                             requires_grad=True)
                out = wp.zeros(2, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(k, 2, [a, b], [out], device=dev)
                tape.backward(grads={out: wp.ones(2, dtype=float, device=dev)})
                return out.numpy(), a.grad.numpy(), b.grad.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal, rtol=1e-5)
            """,
        )

    def test_one_arg_matrix_constructor(self):
        # warp.fem's interpolate_at_nodes builds ``mat33(value)`` from a
        # value that already is a mat33 (copy), not only from a scalar.
        _run(
            self,
            """
            @wp.func
            def as_mat(x: Any):
                return wp.mat33(x)

            @wp.kernel
            def k(a: wp.array[wp.mat33], f: wp.array[float], o: wp.array[wp.mat33], o2: wp.array[wp.mat22]):
                i = wp.tid()
                o[i] = as_mat(a[i]) + as_mat(f[i]) + wp.mat33(a[i])
                o2[i] = wp.mat22(f[i])

            def launch(dev):
                a = wp.array(np.arange(18, dtype=np.float32).reshape(2, 3, 3), dtype=wp.mat33, device=dev)
                f = wp.array([1.5, -2.0], dtype=float, device=dev)
                o = wp.zeros(2, dtype=wp.mat33, device=dev)
                o2 = wp.zeros(2, dtype=wp.mat22, device=dev)
                wp.launch(k, 2, [a, f], [o, o2], device=dev)
                return o.numpy(), o2.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal)
            """,
        )

    def test_struct_field_composite_atomics(self):
        # warp.fem accumulates mat33 / vec3 field values atomically through
        # arrays held by a struct argument.
        _run(
            self,
            """
            @wp.struct
            class F:
                values: wp.array[wp.mat33]
                v3: wp.array[wp.vec3]

            @wp.kernel
            def k(f: F, a: wp.array[wp.mat33], b: wp.array[wp.mat33]):
                i = wp.tid()
                wp.atomic_add(f.values, i % 2, a[i])
                f.v3[i % 2] += a[i][0]
                wp.atomic_sub(b, i % 2, a[i])

            def launch(dev):
                a = wp.array(np.arange(36, dtype=np.float32).reshape(4, 3, 3), dtype=wp.mat33, device=dev)
                f = F()
                f.values = wp.zeros(2, dtype=wp.mat33, device=dev)
                f.v3 = wp.zeros(2, dtype=wp.vec3, device=dev)
                b = wp.zeros(2, dtype=wp.mat33, device=dev)
                wp.launch(k, 4, [f, a, b], device=dev)
                return f.values.numpy(), f.v3.numpy(), b.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal)
            """,
        )

    def test_zero_length_vectors(self):
        # Newton's tiled camera sizes per-ray hit buffers by a static count
        # that is zero when a scene has no Gaussians.
        _run(
            self,
            """
            N = 0

            @wp.kernel
            def k(x: wp.array[int], out: wp.array[int], fout: wp.array[float]):
                i = wp.tid()
                hits = wp.vector(length=wp.static(N), dtype=wp.uint32)
                f = wp.vector(1.0, length=wp.static(N), dtype=wp.float32)
                g = f + f * 2.0
                n = int(0)
                if x[i] < 0 and n < wp.static(N):
                    hits[n] = wp.uint32(x[i])
                    n += 1
                for j in range(wp.static(N)):
                    out[i] += int(hits[j]) + int(g[j])
                out[i] += n + wp.static(N) + 7
                fout[i] = wp.dot(g, g) + wp.length(g) + float(g == f)

            def launch(dev):
                x = wp.array([1, -2], dtype=int, device=dev)
                out = wp.zeros(2, dtype=int, device=dev)
                fout = wp.zeros(2, dtype=float, device=dev)
                wp.launch(k, 2, [x], [out, fout], device=dev)
                return out.numpy(), fout.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal)
            """,
        )

    def test_big_mat_constructor_and_extract_adjoints(self):
        # Newton's softbody tet energy differentiates a ``mat43`` built from
        # 12 scalars and read by element and by row.
        _run(
            self,
            """
            mat43 = wp.types.matrix(shape=(4, 3), dtype=wp.float32)

            @wp.kernel
            def k(x: wp.array[float], v: wp.array[wp.vec3], out: wp.array[float]):
                i = wp.tid()
                a = x[i]
                m = mat43(a, 2.0 * a, 3.0, a * a, 5.0, 6.0, -a, 8.0, 9.0, 10.0, a, 12.0)
                w = 2.0 * m[1] + m[2] * 3.0
                out[i] = m[3, 1] * m[0, 0] + wp.dot(w, v[i]) + wp.ddot(m, m)

            def launch(dev):
                x = wp.array([0.5, -1.5], dtype=float, device=dev, requires_grad=True)
                v = wp.array(np.arange(6, dtype=np.float32).reshape(2, 3), dtype=wp.vec3, device=dev,
                             requires_grad=True)
                out = wp.zeros(2, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(k, 2, [x, v], [out], device=dev)
                tape.backward(grads={out: wp.ones(2, dtype=float, device=dev)})
                return out.numpy(), x.grad.numpy(), v.grad.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal, rtol=1e-5)
            """,
        )

    def test_tile_dual_reset_in_dynamic_loop(self):
        # Owning tile duals are reset each reverse loop iteration with
        # ``adj_t.grad_zero()`` (warp_nn's tiled GEMM, Newton's neural MLP drive).
        _run(
            self,
            """
            @wp.kernel
            def k(x: wp.array[float], n: int, out: wp.array[float]):
                i = wp.tid()
                for kk in range(n):
                    a = wp.tile_load(x, shape=4, offset=kk * 4)
                    b = a * a
                    wp.tile_store(out, b, offset=kk * 4)

            def launch(dev):
                x = wp.array(np.arange(8, dtype=np.float32) * 0.5, dtype=float, device=dev, requires_grad=True)
                out = wp.zeros(8, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch_tiled(k, dim=1, inputs=[x, 2], outputs=[out], block_dim=1 if dev == "cpu" else 32,
                                    device=dev)
                tape.backward(grads={out: wp.array(np.arange(8, dtype=np.float32) + 1, dtype=float, device=dev)})
                return out.numpy(), x.grad.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal, rtol=1e-6)
            """,
        )

    def test_array_condition_and_atomic_sub_adjoints(self):
        # ``if arr:`` (XPBD's optional constraint weights) lowers to a
        # ``where`` whose condition is an array with no dual; Newton's
        # differentiable contacts use ``wp.atomic_sub`` in reverse mode.
        _run(
            self,
            """
            @wp.struct
            class S:
                f: wp.array[wp.spatial_vector]

            @wp.kernel
            def k(deltas: wp.array[wp.spatial_vector], w: wp.array[float], idx: wp.array[int],
                  qd_out: wp.array[wp.spatial_vector]):
                tid = wp.tid()
                weight = 1.0
                if w:
                    inv_weight = w[tid]
                    if inv_weight > 0.0:
                        weight = 1.0 / inv_weight
                j = tid
                if idx:
                    j = idx[tid]
                wp.atomic_add(qd_out, j, deltas[tid] * weight)

            @wp.kernel
            def ks(x: wp.array[float], v: wp.array[wp.vec3], s: S, out: wp.array[float], o3: wp.array2d[wp.vec3]):
                i = wp.tid()
                wp.atomic_sub(out, i % 2, x[i] * x[i])
                wp.atomic_sub(o3, i % 2, 1, v[i] * 2.0)
                wp.atomic_sub(s.f, i % 2, wp.spatial_vector(x[i], 0.0, 1.0, 2.0, x[i] * 3.0, 0.0))
                out[2] -= x[i]

            def launch(dev, use_w):
                d = wp.array(np.arange(12, dtype=np.float32).reshape(2, 6), dtype=wp.spatial_vector, device=dev,
                             requires_grad=True)
                w = wp.array([2.0, 4.0], dtype=float, device=dev, requires_grad=True) if use_w else None
                idx = wp.array([1, 0], dtype=int, device=dev) if use_w else None
                q = wp.zeros(2, dtype=wp.spatial_vector, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(k, 2, [d, w, idx], [q], device=dev)
                tape.backward(grads={q: wp.array(np.ones((2, 6), dtype=np.float32), dtype=wp.spatial_vector,
                                                 device=dev)})
                return q.numpy(), d.grad.numpy(), (w.grad.numpy() if use_w else np.zeros(1))

            def launch_sub(dev):
                x = wp.array(np.arange(4, dtype=np.float32) + 1, dtype=float, device=dev, requires_grad=True)
                v = wp.array(np.arange(12, dtype=np.float32).reshape(4, 3), dtype=wp.vec3, device=dev,
                             requires_grad=True)
                out = wp.zeros(3, dtype=float, device=dev, requires_grad=True)
                o3 = wp.zeros((2, 2), dtype=wp.vec3, device=dev, requires_grad=True)
                s = S()
                s.f = wp.zeros(2, dtype=wp.spatial_vector, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(ks, 4, [x, v, s], [out, o3], device=dev)
                tape.backward(grads={
                    out: wp.array([1.0, 2.0, 3.0], dtype=float, device=dev),
                    o3: wp.array(np.ones((2, 2, 3), dtype=np.float32), dtype=wp.vec3, device=dev),
                })
                return out.numpy(), o3.numpy(), s.f.numpy(), x.grad.numpy(), v.grad.numpy()

            for use_w in (True, False):
                cpu, metal = both(lambda dev: launch(dev, use_w))
                assert_same(cpu, metal, rtol=1e-6)
            cpu, metal = both(launch_sub)
            assert_same(cpu, metal, rtol=1e-6)
            """,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
