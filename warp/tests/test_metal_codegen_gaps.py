# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal codegen regressions for constructs Newton relies on.

Each test launches a kernel on ``cpu`` and ``metal:0`` in a subprocess (the
Metal device must be enabled before ``wp.init``) and compares the outputs.
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_enabled

_COMMON = textwrap.dedent(
    """
    import numpy as np

    def both(launch_fn):
        res = {}
        for dev in ("cpu", "metal:0"):
            res[dev] = launch_fn(dev)
        return res["cpu"], res["metal:0"]

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


def _run(test_case, body: str, timeout: int = 120):
    _run_with_metal_enabled(test_case, _COMMON + textwrap.dedent(body), timeout=timeout)


@unittest.skipUnless(_is_apple_silicon() and _has_mlx(), "Metal requires Apple Silicon")
class TestMetalCodegenGaps(unittest.TestCase):
    def test_mesh_get_fields(self):
        _run(
            self,
            """
            @wp.func
            def extent(mesh_id: wp.uint64, d: wp.vec3):
                mesh = wp.mesh_get(mesh_id)
                best = float(-1.0e10)
                for i in range(mesh.points.shape[0]):
                    best = wp.max(best, wp.dot(mesh.points[i], d))
                return best

            @wp.kernel
            def k(ids: wp.array[wp.uint64], out: wp.array[float], tri: wp.array[wp.vec3i],
                  counts: wp.array[int], vel: wp.array[wp.vec3]):
                tid = wp.tid()
                m = wp.mesh_get(ids[tid])
                out[tid] = extent(ids[tid], wp.vec3(1.0, 0.5, 0.0))
                tri[tid] = wp.vec3i(m.indices[3 * tid], m.indices[3 * tid + 1], m.indices[3 * tid + 2])
                counts[0] = m.points.shape[0]
                counts[1] = m.indices.shape[0] // 3
                vel[tid] = m.velocities[tid + 1]

            pts = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3]], dtype=np.float32)
            idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)

            def launch(dev):
                m = wp.Mesh(points=wp.array(pts, dtype=wp.vec3, device=dev),
                            indices=wp.array(idx, dtype=int, device=dev),
                            velocities=wp.array(pts * 7.0, dtype=wp.vec3, device=dev))
                ids = wp.array([m.id, m.id], dtype=wp.uint64, device=dev)
                out = wp.zeros(2, dtype=float, device=dev)
                tri = wp.zeros(2, dtype=wp.vec3i, device=dev)
                counts = wp.zeros(2, dtype=int, device=dev)
                vel = wp.zeros(2, dtype=wp.vec3, device=dev)
                wp.launch(k, 2, [ids], [out, tri, counts, vel], device=dev)
                return out.numpy(), tri.numpy(), counts.numpy(), vel.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal)
            assert metal[2].tolist() == [4, 2]
            """,
        )

    def test_mesh_eval_adjoints(self):
        _run(
            self,
            """
            @wp.kernel
            def k(mid: wp.uint64, uv: wp.array[wp.vec2], out: wp.array[float]):
                i = wp.tid()
                p = wp.mesh_eval_position(mid, i, uv[i][0], uv[i][1])
                v = wp.mesh_eval_velocity(mid, i, uv[i][0], uv[i][1])
                out[i] = wp.dot(p, wp.vec3(1.0, 2.0, 3.0)) + wp.length(v)

            pts = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3]], dtype=np.float32)
            idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)

            def launch(dev):
                m = wp.Mesh(points=wp.array(pts, dtype=wp.vec3, device=dev),
                            indices=wp.array(idx, dtype=int, device=dev),
                            velocities=wp.array(pts * 2.0 + 1.0, dtype=wp.vec3, device=dev))
                uv = wp.array([[0.2, 0.3], [0.5, 0.1]], dtype=wp.vec2, device=dev, requires_grad=True)
                o = wp.zeros(2, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(k, 2, [m.id, uv], [o], device=dev)
                tape.backward(grads={o: wp.ones(2, dtype=float, device=dev)})
                return o.numpy(), uv.grad.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal, rtol=1e-6, atol=1e-6)
            """,
        )

    def test_reused_loop_var_in_unrolled_loop(self):
        # mujoco_warp's small Cholesky: ``kk`` is reused by two dynamic loops
        # inside an unrolled ``range(wp.static(N))``; Warp emits the loop-
        # carried copy between ``wp::range`` and the loop label.
        _run(
            self,
            """
            N = 4

            @wp.kernel
            def k(x: wp.array2d[float], y: wp.array2d[float]):
                w = wp.tid()
                size = wp.static(N)
                for i in range(wp.static(N)):
                    d = x[w, i]
                    for kk in range(i):
                        d -= x[w, kk] * 0.5
                    y[w, i] = d
                    for j in range(i + 1, size):
                        v = x[w, j]
                        for kk in range(i):
                            v -= x[w, kk] * 0.25
                        y[w, j] += v

            xn = np.arange(8, dtype=np.float32).reshape(2, N) + 1.0

            def launch(dev):
                x = wp.array(xn, dtype=float, device=dev, requires_grad=True)
                y = wp.zeros((2, N), dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(k, 2, [x], [y], device=dev)
                tape.backward(grads={y: wp.ones((2, N), dtype=float, device=dev)})
                return y.numpy(), x.grad.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal, rtol=1e-6, atol=1e-6)
            """,
        )

    def test_misc_value_ops(self):
        _run(
            self,
            """
            vec2ul = wp.types.vector(2, wp.uint64)

            @wp.kernel
            def k(a: wp.array[wp.mat33], f: wp.array[float], u: wp.array[vec2ul], g: wp.array2d[float],
                  out: wp.array[wp.mat33], fo: wp.array[float], uo: wp.array[wp.uint64], n: wp.array[int]):
                i = wp.tid()
                out[i] = -a[i] + (-(a[i] * 2.0))
                fo[i] = +f[i]
                v = u[i]
                w = vec2ul(v[1], v[0])
                uo[i] = w[0] + w[1] * wp.uint64(2)
                n[0] = len(f)
                n[1] = len(g)

            def launch(dev):
                a = wp.array(np.arange(18, dtype=np.float32).reshape(2, 3, 3), dtype=wp.mat33, device=dev)
                f = wp.array([1.0, -2.0], dtype=float, device=dev)
                u = wp.array(np.array([[1, 2], [3, 4]], dtype=np.uint64), dtype=vec2ul, device=dev)
                g = wp.zeros((3, 4), dtype=float, device=dev)
                out = wp.zeros(2, dtype=wp.mat33, device=dev)
                fo = wp.zeros(2, dtype=float, device=dev)
                uo = wp.zeros(2, dtype=wp.uint64, device=dev)
                n = wp.zeros(2, dtype=int, device=dev)
                wp.launch(k, 2, [a, f, u, g], [out, fo, uo, n], device=dev)
                return out.numpy(), fo.numpy(), uo.numpy(), n.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal)
            """,
        )

    def test_shape_suffixed_array_args(self):
        _run(
            self,
            """
            @wp.kernel
            def k(idx: wp.array[wp.int32], out_shape: wp.array2d[wp.int32]):
                w, g = wp.tid()
                out_shape[w, g] = idx[g] + w * 10 + out_shape.shape[1]

            @wp.kernel
            def k_collide(a: wp.array2d[float], a_shape: wp.array[float]):
                i, j = wp.tid()
                a[i, j] = a_shape[j]

            def launch(dev):
                idx = wp.array([1, -1, 3], dtype=wp.int32, device=dev)
                out = wp.zeros((2, 3), dtype=wp.int32, device=dev)
                wp.launch(k, (2, 3), [idx], [out], device=dev)
                return out.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal)

            from warp._src.codegen_metal import MetalCodegenError
            try:
                wp.launch(k_collide, (2, 3), [], [wp.zeros((2, 3), dtype=float, device="metal:0"),
                                                  wp.zeros(3, dtype=float, device="metal:0")], device="metal:0")
            except MetalCodegenError as e:
                assert "a_shape" in str(e)
            else:
                raise AssertionError("expected a shape-identifier collision error")
            """,
        )

    def test_vector_reductions_and_ctors(self):
        _run(
            self,
            """
            vec6 = wp.types.vector(6, float)
            vec7 = wp.types.vector(7, float)
            vec9 = wp.types.vector(9, float)

            @wp.kernel
            def k(a3: wp.array[wp.vec3], a6: wp.array[vec6], f: wp.array[float], i: wp.array[int],
                  xf: wp.array[wp.transform], v9: wp.array[vec9]):
                t = wp.tid()
                f[3 * t + 0] = wp.max(a3[t]) - wp.min(a3[t])
                f[3 * t + 1] = wp.max(a6[t]) - wp.min(a6[t])
                i[4 * t + 0] = int(wp.argmax(a3[t]))
                i[4 * t + 1] = int(wp.argmin(a3[t]))
                i[4 * t + 2] = int(wp.argmax(a6[t]))
                i[4 * t + 3] = int(wp.argmin(a6[t]))
                xf[t] = wp.transform(a3[t], wp.quat_identity())
                v = vec7(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
                f[3 * t + 2] = v[6]
                v9[t] = vec9(float(t)) + vec9(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

            a3 = np.array([[1, 5, -2], [np.nan, 3, 3], [-1, -1, -1]], dtype=np.float32)
            a6 = np.array([[1, 5, 2, -3, 4, 0], [np.nan, 2, 9, 9, -4, -4], [0, 0, 0, 0, 0, 0]], dtype=np.float32)

            def launch(dev):
                f = wp.zeros(9, dtype=float, device=dev)
                i = wp.zeros(12, dtype=int, device=dev)
                xf = wp.zeros(3, dtype=wp.transform, device=dev)
                v9 = wp.zeros(3, dtype=vec9, device=dev)
                wp.launch(k, 3, [wp.array(a3, dtype=wp.vec3, device=dev), wp.array(a6, dtype=vec6, device=dev)],
                          [f, i, xf, v9], device=dev)
                return f.numpy(), i.numpy(), xf.numpy(), v9.numpy()

            cpu, metal = both(launch)
            # Index results and constructors are exact; skip NaN-propagating reductions.
            assert_same(cpu[1:], metal[1:])
            np.testing.assert_array_equal(np.isnan(cpu[0]), np.isnan(metal[0]))
            assert metal[1][:4].tolist() == [1, 2, 1, 3]
            """,
        )

    def test_tuple_args_and_static_closure_calls(self):
        _run(
            self,
            """
            @wp.func
            def f_tuple(index: tuple[int, int]):
                return float(index[0] * 10 + index[1])

            def make_scaler(s: float):
                @wp.func
                def inner(x: float):
                    return x * s
                return inner

            scale2 = make_scaler(2.0)
            scale3 = make_scaler(3.0)  # registered later under the same key

            @wp.kernel
            def k(out: wp.array[float]):
                i = wp.tid()
                out[2 * i] = f_tuple((i, 2))
                out[2 * i + 1] = wp.static(scale2)(float(i))

            def launch(dev):
                out = wp.zeros(6, dtype=float, device=dev)
                wp.launch(k, 3, [], [out], device=dev)
                return out.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal)
            assert metal.tolist() == [2.0, 0.0, 12.0, 2.0, 22.0, 4.0]
            """,
        )

    def test_by_value_composite_args(self):
        _run(
            self,
            """
            vec6 = wp.types.vector(6, float)
            mat66 = wp.types.matrix((6, 6), float)
            mat23 = wp.types.matrix((2, 3), float)

            @wp.kernel
            def k(t: wp.transform, q: wp.quat, v: vec6, m: mat66, n: mat23, m3: wp.mat33, out: wp.array[float]):
                out[0] = wp.transform_point(t, wp.vec3(1.0, 0.0, 0.0))[0]
                out[1] = q[3] + v[5] + wp.dot(v, v)
                out[2] = m[1, 2] + n[1, 0] + m3[2, 1]
                out[3] = wp.transform_get_translation(t)[2]

            def launch(dev):
                out = wp.zeros(4, dtype=float, device=dev)
                wp.launch(k, 1, [wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_rpy(0.1, 0.2, 0.3)),
                                 wp.quat(0.0, 0.0, 0.0, 1.0), vec6(*range(6)), mat66(*range(36)),
                                 mat23(*range(6)), wp.mat33(*range(9))], [out], device=dev)
                return out.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal, rtol=1e-6, atol=1e-6)
            """,
        )

    def test_spatial_and_reduction_adjoints(self):
        _run(
            self,
            """
            @wp.kernel
            def k(a: wp.array[wp.spatial_vector], b: wp.array[wp.spatial_vector], c: wp.array[wp.vec3],
                  out: wp.array[float]):
                t = wp.tid()
                x = a[t]
                y = b[t]
                cr = wp.spatial_cross(x, y)
                cd = wp.spatial_cross_dual(x, y)
                s = wp.spatial_vector(wp.spatial_top(x) * 2.0, wp.spatial_bottom(y))
                out[t] = (wp.spatial_dot(cr, s) + wp.dot(cd, wp.spatial_vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
                          + wp.max(x) + wp.min(y) + wp.max(c[t]) - wp.min(c[t]))
                d = wp.abs(y[0]) + 1.0
                m = wp.matrix_from_rows(c[t], c[t] * 2.0, wp.vec3(1.0, 2.0, 3.0))
                out[t] += (wp.length(c[t] / d) + wp.length(d / (c[t] + wp.vec3(5.0))) + wp.trace(m / d)
                           + wp.length(x / d))

            rng = np.random.default_rng(0)
            an = rng.standard_normal((4, 6)).astype(np.float32)
            bn = rng.standard_normal((4, 6)).astype(np.float32)
            cn = rng.standard_normal((4, 3)).astype(np.float32)

            def launch(dev):
                a = wp.array(an, dtype=wp.spatial_vector, device=dev, requires_grad=True)
                b = wp.array(bn, dtype=wp.spatial_vector, device=dev, requires_grad=True)
                c = wp.array(cn, dtype=wp.vec3, device=dev, requires_grad=True)
                o = wp.zeros(4, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(k, 4, [a, b, c], [o], device=dev)
                tape.backward(grads={o: wp.ones(4, dtype=float, device=dev)})
                return o.numpy(), a.grad.numpy(), b.grad.numpy(), c.grad.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal, rtol=1e-5, atol=1e-5)
            """,
        )

    def test_print_is_compiled_out(self):
        _run(
            self,
            """
            import warnings

            @wp.func
            def helper(x: float):
                wp.printf("helper %f\\n", x)
                return x * 2.0

            @wp.kernel
            def k(a: wp.array[float], out: wp.array[float]):
                i = wp.tid()
                v = a[i]
                wp.print(v)
                wp.printf("i=%d v=%f\\n", i, v)
                out[i] = helper(v)

            a = wp.array([1.0, 2.0], dtype=float, device="metal:0")
            out = wp.zeros(2, dtype=float, device="metal:0")
            wp.launch(k, 2, [a], [out], device="metal:0")
            assert out.numpy().tolist() == [2.0, 4.0]

            @wp.kernel
            def k_err(out: wp.array[float]):
                wp.printf("boom\\n")
                out[0] = 1.0

            from warp._src.codegen_metal import MetalCodegenError
            wp.config.metal_print = "error"
            try:
                wp.launch(k_err, 1, [], [wp.zeros(1, dtype=float, device="metal:0")], device="metal:0")
            except MetalCodegenError as e:
                assert "metal_print" in str(e)
            else:
                raise AssertionError("expected wp.printf to raise with metal_print='error'")
            """,
        )

    def test_func_native_snippet(self):
        _run(
            self,
            """
            @wp.func_native(\"\"\"
            #if defined(__CUDA_ARCH__)
            return __frsqrt_rn(value);
            #else
            return 1.0f / sqrtf(value);
            #endif
            \"\"\")
            def rsqrt_rn(value: float) -> float: ...

            @wp.kernel
            def k(a: wp.array[float], out: wp.array[float]):
                i = wp.tid()
                out[i] = rsqrt_rn(a[i]) * 2.0

            def launch(dev):
                out = wp.zeros(3, dtype=float, device=dev)
                wp.launch(k, 3, [wp.array([1.0, 4.0, 0.25], dtype=float, device=dev)], [out], device=dev)
                return out.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal)
            """,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
