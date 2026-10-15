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

    def test_expect_checks_are_compiled_out(self):
        # On CPU ``wp.expect_*`` only prints on mismatch; Metal has no printf,
        # so it drops them like ``wp.print`` and the kernel still runs.
        _run(
            self,
            """
            wp.config.metal_print = "ignore"

            @wp.func
            def chk(a: int, b: int):
                wp.expect_eq(a, b)
                wp.expect_neq(a, b + 1)

            @wp.kernel
            def k(x: wp.array[int], v: wp.array[wp.vec3], out: wp.array[int]):
                i = wp.tid()
                chk(x[i], i)
                wp.expect_eq(v[i], v[i])
                wp.expect_near(v[i], v[i] + wp.vec3(0.01), 0.1)
                wp.expect_near(1.0, 1.05, 0.1)
                out[i] = x[i] * 2

            def launch(dev):
                x = wp.array([0, 5], dtype=int, device=dev)
                v = wp.zeros(2, dtype=wp.vec3, device=dev)
                out = wp.zeros(2, dtype=int, device=dev)
                wp.launch(k, 2, [x, v], [out], device=dev)
                return out.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal)

            wp.config.metal_print = "error"

            @wp.kernel
            def k2(x: wp.array[int]):
                wp.expect_eq(x[0], 1)

            try:
                wp.launch(k2, 1, [wp.zeros(1, dtype=int, device="metal:0")], device="metal:0")
            except Exception as e:
                assert "expect" in str(e), str(e)
            else:
                raise AssertionError("metal_print='error' should reject wp.expect_eq")
            """,
        )

    def _fp64_mat_arrays(self, mode: str, tol: str):
        _run(
            self,
            f"""
            wp.config.metal_fp64 = "{mode}"

            @wp.struct
            class Out:
                m: wp.array[wp.mat44d]

            @wp.kernel
            def k(a: wp.array[wp.mat44d], v: wp.array[wp.vec3d], s: wp.array[wp.float64], o: Out,
                  o2: wp.array[wp.mat33d], of: wp.array[float]):
                i = wp.tid()
                m = a[i]
                o.m[i] = wp.transpose(m) * wp.float64(2.0) + m @ m
                r = wp.mat33d(m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2], m[2, 0], m[2, 1], m[2, 2])
                o2[i] = r * s[i]
                of[i] = float(wp.dot(r @ v[i], v[i]) + m[3, 3])

            def launch(dev):
                rng = np.random.default_rng(1)
                a = wp.array(rng.standard_normal((2, 4, 4)), dtype=wp.mat44d, device=dev)
                v = wp.array(rng.standard_normal((2, 3)), dtype=wp.vec3d, device=dev)
                s = wp.array([1.5, -0.25], dtype=wp.float64, device=dev)
                o = Out()
                o.m = wp.zeros(2, dtype=wp.mat44d, device=dev)
                o2 = wp.zeros(2, dtype=wp.mat33d, device=dev)
                of = wp.zeros(2, dtype=float, device=dev)
                wp.launch(k, 2, [a, v, s, o], [o2, of], device=dev)
                return o.m.numpy(), o2.numpy(), of.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal, {tol})
            """,
        )

    def test_fp64_mat_arrays_emulate(self):
        self._fp64_mat_arrays("emulate", "rtol=1e-12, atol=1e-12")

    def test_fp64_mat_arrays_demote(self):
        self._fp64_mat_arrays("demote", "rtol=1e-5, atol=1e-5")

    def test_fp64_mat_arrays_error_names_the_setting(self):
        _run(
            self,
            """
            wp.config.metal_fp64 = "error"

            @wp.kernel
            def k(a: wp.array[wp.mat33d]):
                a[0] = wp.mat33d()

            try:
                wp.launch(k, 1, [wp.zeros(1, dtype=wp.mat33d, device="metal:0")], device="metal:0")
            except Exception as e:
                assert "metal_fp64" in str(e), str(e)
            else:
                raise AssertionError("float64 mat arrays should be rejected under metal_fp64='error'")
            """,
        )

    def _fp64_mat_conversions(self, mode: str):
        # Newton's RTX viewer writes ``wp.mat44d(wp.mat44(...))`` into a
        # ``wp.array[wp.mat44d]``.
        _run(
            self,
            f"""
            wp.config.metal_fp64 = "{mode}"

            @wp.kernel
            def k(a: wp.array[wp.mat44], b: wp.array[wp.mat33d], o: wp.array[wp.mat44d], o2: wp.array[wp.mat33],
                  o3: wp.array[wp.mat22d], o4: wp.array[wp.vec3d]):
                i = wp.tid()
                o[i] = wp.mat44d(a[i])
                o2[i] = wp.mat33(b[i])
                o3[i] = wp.mat22d(wp.mat22(1.0, 2.0, 3.0, 4.0))
                o4[i] = wp.vec3d(wp.vec3(a[i][0, 0], 2.0, 3.0))

            def launch(dev):
                a = wp.array(np.arange(32, dtype=np.float32).reshape(2, 4, 4) * 0.1, dtype=wp.mat44, device=dev)
                b = wp.array(np.arange(18, dtype=np.float64).reshape(2, 3, 3) * 0.1, dtype=wp.mat33d, device=dev)
                outs = [wp.zeros(2, dtype=t, device=dev) for t in (wp.mat44d, wp.mat33, wp.mat22d, wp.vec3d)]
                wp.launch(k, 2, [a, b], outs, device=dev)
                return [x.numpy() for x in outs]

            cpu, metal = both(launch)
            assert_same(cpu, metal)
            """,
        )

    def test_fp64_mat_conversions_emulate(self):
        self._fp64_mat_conversions("emulate")

    def test_fp64_mat_conversions_demote(self):
        self._fp64_mat_conversions("demote")

    def test_fp64_demote_rounds_to_nearest(self):
        # Demoted float64 reads must round like a hardware double->float
        # conversion (ties to even), not truncate the mantissa.
        _run(
            self,
            """
            wp.config.metal_fp64 = "demote"

            @wp.kernel
            def k(a: wp.array[wp.float64], o: wp.array[float]):
                i = wp.tid()
                o[i] = float(a[i])

            rng = np.random.default_rng(0)
            f = rng.standard_normal(1000).astype(np.float32)
            x = np.concatenate([
                rng.standard_normal(20000) * 10.0 ** rng.integers(-30, 30, 20000),
                f.astype(np.float64) + np.spacing(f).astype(np.float64) / 2,  # exact ties
                [3.4028235677973366e38, 1.0e39, -0.0, np.inf, -np.inf],
                # just below FLT_MIN: rounds up to FLT_MIN, not to 0
                [1.1754943508222875e-38 * (1 - 2**-25), -1.1754943508222875e-38 * (1 - 2**-24)],
            ])
            a = wp.array(x, dtype=wp.float64, device="metal:0")
            o = wp.zeros(len(x), dtype=float, device="metal:0")
            wp.launch(k, len(x), [a], [o], device="metal:0")
            ref = x.astype(np.float32)
            np.testing.assert_array_equal(o.numpy(), ref)
            """,
        )

    def test_tile_inplace_add_sub_adjoints(self):
        _run(
            self,
            """
            @wp.kernel
            def k(a: wp.array2d[float], b: wp.array2d[float], out: wp.array2d[float]):
                i, j = wp.tid()
                t = wp.tile_load(a, shape=(2, 4), offset=(i * 2, 0))
                u = wp.tile_load(b, shape=(2, 4), offset=(i * 2, 0))
                t += u * u
                t -= u
                wp.tile_store(out, t * t, offset=(i * 2, 0))

            def launch(dev):
                a = wp.array(np.arange(8, dtype=np.float32).reshape(2, 4) * 0.5, dtype=float, device=dev,
                             requires_grad=True)
                b = wp.array(np.arange(8, dtype=np.float32).reshape(2, 4) * -0.25, dtype=float, device=dev,
                             requires_grad=True)
                out = wp.zeros((2, 4), dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch_tiled(k, dim=(1, 1), inputs=[a, b], outputs=[out],
                                    block_dim=1 if dev == "cpu" else 32, device=dev)
                tape.backward(grads={out: wp.ones((2, 4), dtype=float, device=dev)})
                return out.numpy(), a.grad.numpy(), b.grad.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal, rtol=1e-6)
            """,
        )

    def test_optional_array_condition_adjoint(self):
        # ``if arr:`` / ``wp.where(arr, ...)`` with a ``None`` array: the
        # reverse select must take the same branch as the forward one.
        _run(
            self,
            """
            @wp.kernel
            def k_if(x: wp.array[float], w: wp.array[float], out: wp.array[float]):
                i = wp.tid()
                v = x[i] * 2.0
                if w:
                    v = x[i] * 7.0 * w[i]
                out[i] = v

            @wp.kernel
            def k_where(x: wp.array[float], y: wp.array[float], arr: wp.array[float], out: wp.array[float]):
                i = wp.tid()
                out[i] = wp.where(arr, x[i] * 7.0, y[i] * 3.0)

            def launch(dev, use_arr):
                x = wp.array([1.0, 2.0], dtype=float, device=dev, requires_grad=True)
                y = wp.array([3.0, 4.0], dtype=float, device=dev, requires_grad=True)
                w = wp.array([0.5, 2.0], dtype=float, device=dev) if use_arr else None
                o1 = wp.zeros(2, dtype=float, device=dev, requires_grad=True)
                o2 = wp.zeros(2, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(k_if, 2, [x, w], [o1], device=dev)
                    wp.launch(k_where, 2, [x, y, w], [o2], device=dev)
                tape.backward(grads={o1: wp.ones(2, dtype=float, device=dev),
                                     o2: wp.array([1.0, 2.0], dtype=float, device=dev)})
                return o1.numpy(), o2.numpy(), x.grad.numpy(), y.grad.numpy()

            for use_arr in (False, True):
                cpu, metal = both(lambda dev: launch(dev, use_arr))
                assert_same(cpu, metal)
            """,
        )

    def test_negative_matrix_and_vector_indices(self):
        # Warp wraps negative row/col/component indices (``mat.h``/``vec.h``);
        # constant and runtime ones, reads, writes, slices and adjoints.
        _run(
            self,
            """
            M66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
            M43 = wp.types.matrix(shape=(4, 3), dtype=wp.float32)

            def make(M, O):
                @wp.kernel
                def k(a: wp.array[M], v: wp.array[wp.vec3], r: int, c: int, o: wp.array[O],
                      loss: wp.array[float]):
                    i = wp.tid()
                    m = a[i]
                    q = v[i]
                    row = m[r]
                    m[-1, 0] = q[-1]
                    q[-2] = 4.0
                    o[i] = row
                    rs = m[r, :]
                    cs = m[:, c]
                    loss[i] = (m[r, c] * 3.0 + row[1] * 2.0 + m[-1, -2] + m[-2][0] * 5.0 + m[-1, 0]
                               + q[-2] + wp.dot(q, q) + rs[0] + cs[1] * 2.0)
                return k

            cases = [(M66, wp.types.vector(6, wp.float32), 6, 6), (M43, wp.vec3, 4, 3),
                     (wp.mat33, wp.vec3, 3, 3)]
            for M, O, R, C in cases:
                k = make(M, O)
                for r, c in ((-1, -2), (1, 2), (-R, 0)):
                    def launch(dev):
                        a = wp.array(np.arange(2 * R * C, dtype=np.float32).reshape(2, R, C), dtype=M,
                                     device=dev, requires_grad=True)
                        v = wp.array(np.arange(6, dtype=np.float32).reshape(2, 3), dtype=wp.vec3, device=dev,
                                     requires_grad=True)
                        o = wp.zeros(2, dtype=O, device=dev, requires_grad=True)
                        loss = wp.zeros(2, dtype=float, device=dev, requires_grad=True)
                        tape = wp.Tape()
                        with tape:
                            wp.launch(k, 2, [a, v, r, c], [o, loss], device=dev)
                        tape.backward(grads={loss: wp.ones(2, dtype=float, device=dev)})
                        return o.numpy(), loss.numpy(), a.grad.numpy(), v.grad.numpy()

                    cpu, metal = both(launch)
                    assert_same(cpu, metal)
            """,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
