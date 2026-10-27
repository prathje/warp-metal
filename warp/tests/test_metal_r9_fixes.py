# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal parity fixes for silent CPU/Metal mismatches found in the r9 review.

Each test runs the same kernel on ``cpu`` and ``metal:0`` in a subprocess and
compares the results, or pins that Metal rejects a construct loudly
(``MetalCodegenError`` / ``RuntimeError``) instead of computing something
different from the CPU.
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_native


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR9Fixes(unittest.TestCase):
    def test_range_bounds_reassigned_in_body_match_cpu(self):
        # ``wp::range`` captures start / stop / step once; the loop body
        # reassigning the variables that held them must not change the
        # iteration count (Metal's ``for`` header used to re-read them).
        snippet = textwrap.dedent(
            """
            import numpy as np

            @wp.kernel(module="unique")
            def k_mut(step: int, stop: int, out: wp.array(dtype=int)):
                s = step
                n = stop
                acc = int(0)
                cnt = int(0)
                for i in range(0, n, s):
                    acc += i
                    cnt += 1
                    s = s + 1
                    n = n - 1
                    if cnt > 50:
                        break
                out[0] = acc
                out[1] = cnt

            @wp.kernel(module="unique")
            def k_stop_only(stop: int, out: wp.array(dtype=int)):
                n = stop
                cnt = int(0)
                for i in range(n):
                    cnt += 1
                    n = n - 2
                out[0] = cnt
                out[1] = n

            @wp.kernel(module="unique")
            def k_nested(step: int, out: wp.array(dtype=int)):
                tid = wp.tid()
                acc = int(0)
                m = tid + 6
                for i in range(m):
                    k = i + 4
                    st = step
                    for j in range(k, 0, st):
                        acc += i * 100 + j
                        k = k - 1
                        st = st - 1
                    m = m - 1
                out[tid] = acc

            cases = [
                (k_mut, [2, 20], 2, 1),
                (k_mut, [-2, -20], 2, 1),
                (k_mut, [3, 31], 2, 1),
                (k_stop_only, [9], 2, 1),
                (k_nested, [-1], 4, 4),
            ]
            for k, args, n, dim in cases:
                res = {}
                for dev in ("cpu", "metal:0"):
                    out = wp.zeros(n, dtype=int, device=dev)
                    wp.launch(k, dim=dim, inputs=[*args, out], device=dev)
                    res[dev] = out.numpy().tolist()
                assert res["cpu"] == res["metal:0"], (k.key, args, res)
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_range_bounds_reassigned_in_body_adjoint_matches_cpu(self):
        # The reverse loop recomputes the forward body, so its bounds must be
        # frozen there too (rev_stop reads ``start``).
        snippet = textwrap.dedent(
            """
            import numpy as np

            @wp.kernel(module="unique")
            def k_stop(x: wp.array(dtype=float), stop: int, out: wp.array(dtype=float)):
                n = stop
                acc = float(0.0)
                for i in range(0, n):
                    acc += x[i] * x[i]
                    n = n - 1
                out[0] = acc

            @wp.kernel(module="unique")
            def k_start(x: wp.array(dtype=float), start: int, stop: int, out: wp.array(dtype=float)):
                a = start
                acc = float(0.0)
                for i in range(a, stop, 2):
                    acc += x[i] * float(i + 1)
                    a = a + 3
                out[0] = acc + float(a)

            @wp.kernel(module="unique")
            def k_nest(x: wp.array(dtype=float), stop: int, out: wp.array(dtype=float)):
                tid = wp.tid()
                acc = float(0.0)
                for i in range(stop):
                    m = i + 3
                    for j in range(0, m):
                        acc += x[j] * float(i + tid)
                        m = m - 2
                out[tid] = acc

            for k, extra, nout in [(k_stop, [12], 1), (k_start, [1, 14], 1), (k_nest, [5], 2)]:
                res = {}
                for dev in ("cpu", "metal:0"):
                    x = wp.array(np.arange(16, dtype=np.float32) * 0.25 + 1.0, dtype=float, device=dev,
                                 requires_grad=True)
                    out = wp.zeros(nout, dtype=float, device=dev, requires_grad=True)
                    tape = wp.Tape()
                    with tape:
                        wp.launch(k, dim=nout, inputs=[x, *extra, out], device=dev)
                    tape.backward(grads={out: wp.ones(nout, dtype=float, device=dev)})
                    res[dev] = (out.numpy().copy(), x.grad.numpy().copy())
                np.testing.assert_allclose(res["metal:0"][0], res["cpu"][0], rtol=1e-6, err_msg=k.key)
                np.testing.assert_allclose(res["metal:0"][1], res["cpu"][1], rtol=1e-6, err_msg=k.key)
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_store_adjoint_grad_zeroing_matches_cpu(self):
        # warp/native/array.h adj_array_store zeroes the gradient it consumes
        # only when it is the forward array's own .grad and retain_grad is
        # off; without an adjoint buffer it falls back to the array's .grad.
        # Metal zeroed every buffer (retain_grad, caller-owned adj_inputs)
        # and dropped the fallback.
        snippet = textwrap.dedent(
            """
            import numpy as np

            @wp.kernel(module="unique")
            def mul2(x: wp.array(dtype=float), y: wp.array(dtype=float)):
                i = wp.tid()
                y[i] = 2.0 * x[i]

            @wp.kernel(module="unique")
            def mul3(x: wp.array(dtype=float), y: wp.array(dtype=float)):
                i = wp.tid()
                y[i] = 3.0 * x[i]

            @wp.kernel(module="unique")
            def store_vec(x: wp.array(dtype=float), y: wp.array(dtype=wp.vec3)):
                i = wp.tid()
                y[i] = wp.vec3(x[i], 2.0 * x[i], 3.0 * x[i])

            @wp.kernel(module="unique")
            def store_comp(x: wp.array(dtype=float), y: wp.array(dtype=wp.vec3)):
                i = wp.tid()
                y[i][1] = 5.0 * x[i]

            @wp.struct
            class S:
                a: wp.array(dtype=float)

            @wp.kernel(module="unique")
            def store_struct(x: wp.array(dtype=float), s: S):
                i = wp.tid()
                s.a[i] = 4.0 * x[i]

            def run(dev):
                N = 6
                r = {}
                # retain_grad: two writers
                xa = wp.ones(N, dtype=float, requires_grad=True, device=dev)
                xb = wp.ones(N, dtype=float, requires_grad=True, device=dev)
                y = wp.zeros(N, dtype=float, requires_grad=True, retain_grad=True, device=dev)
                with wp.Tape() as tape:
                    wp.launch(mul2, dim=N, inputs=[xa, y], device=dev)
                    wp.launch(mul3, dim=N, inputs=[xb, y], device=dev)
                tape.backward(grads={y: wp.ones_like(y)})
                r["retain"] = [xa.grad.numpy(), xb.grad.numpy(), y.grad.numpy()]
                # vec / component stores
                for k in (store_vec, store_comp):
                    for retain in (False, True):
                        x = wp.ones(N, dtype=float, requires_grad=True, device=dev)
                        yv = wp.zeros(N, dtype=wp.vec3, requires_grad=True, retain_grad=retain, device=dev)
                        with wp.Tape() as tape:
                            wp.launch(k, dim=N, inputs=[x, yv], device=dev)
                        yv.grad.fill_(wp.vec3(1.0, 1.0, 1.0))
                        tape.backward()
                        r[f"{k.key}{retain}"] = [x.grad.numpy(), yv.grad.numpy()]
                # struct field
                for retain in (False, True):
                    x = wp.ones(N, dtype=float, requires_grad=True, device=dev)
                    s = S()
                    s.a = wp.zeros(N, dtype=float, requires_grad=True, retain_grad=retain, device=dev)
                    with wp.Tape() as tape:
                        wp.launch(store_struct, dim=N, inputs=[x, s], device=dev)
                    s.a.grad.fill_(1.0)
                    tape.backward()
                    r[f"struct{retain}"] = [x.grad.numpy(), s.a.grad.numpy()]
                # external adjoint buffer: caller-owned, not zeroed
                x = wp.ones(N, dtype=float, requires_grad=True, device=dev)
                y = wp.zeros(N, dtype=float, requires_grad=True, device=dev)
                adj_x = wp.zeros(N, dtype=float, device=dev)
                adj_y = wp.full(N, 1.0, dtype=float, device=dev)
                wp.launch(mul2, dim=N, inputs=[x, y], adj_inputs=[adj_x, adj_y], adjoint=True, device=dev)
                r["external"] = [adj_x.numpy(), adj_y.numpy()]
                # no adjoint buffers: fall back to .grad
                x = wp.ones(N, dtype=float, requires_grad=True, device=dev)
                y = wp.zeros(N, dtype=float, requires_grad=True, device=dev)
                y.grad.fill_(1.0)
                wp.launch(mul2, dim=N, inputs=[x, y], adj_inputs=[None, None], adjoint=True, device=dev)
                r["fallback"] = [x.grad.numpy(), y.grad.numpy()]
                return r

            c = run("cpu"); m = run("metal:0")
            for k in c:
                for a, b in zip(c[k], m[k]):
                    np.testing.assert_allclose(b, a, err_msg=k)
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_strided_and_reversed_tile_views_match_cpu(self):
        # tile_view types carry the slice step in their strides; Metal's
        # materialized copy used unit steps and right-aligned dropped dims,
        # so t[::2, ::2], t[::-1, :], t[:, 7] and chained slices read the
        # wrong elements (forward and gradient).
        snippet = textwrap.dedent(
            """
            import numpy as np

            M, N = 16, 32

            @wp.kernel(module="unique")
            def k_strided(src: wp.array2d[float], dst: wp.array2d[float]):
                t = wp.tile_load(src, shape=(M, N))
                wp.tile_store(dst, t[::2, ::2])

            @wp.kernel(module="unique")
            def k_reverse(src: wp.array2d[float], dst: wp.array2d[float]):
                t = wp.tile_load(src, shape=(M, N))
                wp.tile_store(dst, t[::-1, 3:9:2])

            @wp.kernel(module="unique")
            def k_neg_stop(src: wp.array2d[float], dst: wp.array2d[float]):
                t = wp.tile_load(src, shape=(M, N))
                wp.tile_store(dst, t[7:-100:-1, 100::-3])

            @wp.kernel(module="unique")
            def k_column(src: wp.array2d[float], dst: wp.array1d[float]):
                t = wp.tile_load(src, shape=(M, N))
                wp.tile_store(dst, t[:, 7])

            @wp.kernel(module="unique")
            def k_chain(src: wp.array2d[float], dst: wp.array1d[float]):
                t = wp.tile_load(src, shape=(M, N))
                a = t[::-1, :]
                b = a[1:, 5]
                wp.tile_store(dst, b[::2])

            src_np = np.random.default_rng(7).random((M, N), dtype=np.float32)
            cases = [
                (k_strided, (M // 2, N // 2)),
                (k_reverse, (M, 3)),
                (k_neg_stop, (8, 11)),
                (k_column, (M,)),
                (k_chain, (8,)),
            ]
            for k, shape in cases:
                res = {}
                for dev in ("cpu", "metal:0"):
                    src = wp.array(src_np, dtype=float, requires_grad=True, device=dev)
                    dst = wp.zeros(shape, dtype=float, requires_grad=True, device=dev)
                    with wp.Tape() as tape:
                        wp.launch_tiled(k, dim=[1], inputs=[src, dst], block_dim=32, device=dev)
                    adj = np.arange(1, int(np.prod(shape)) + 1, dtype=np.float32).reshape(shape)
                    dst.grad = wp.array(adj, dtype=float, device=dev)
                    tape.backward()
                    res[dev] = (dst.numpy().copy(), src.grad.numpy().copy())
                np.testing.assert_array_equal(res["metal:0"][0], res["cpu"][0], err_msg=k.key)
                np.testing.assert_array_equal(res["metal:0"][1], res["cpu"][1], err_msg=k.key + " grad")
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    _FP64_KERNELS = textwrap.dedent(
        """
        import numpy as np

        @wp.kernel(module="unique")
        def k_vec(p: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
            a = wp.vec3d(wp.float64(p[0][0]), wp.float64(p[0][1]), wp.float64(p[0][2]))
            b = wp.vec3d(wp.float64(p[1][0]), wp.float64(p[1][1]), wp.float64(p[1][2]))
            out[0] = float(wp.length_sq(wp.cross(a, b)))

        @wp.kernel(module="unique")
        def k_scalar(x: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
            big = wp.float64(x[0][0]) * wp.float64(1.0e8)
            out[0] = float((big + wp.float64(x[0][1])) - big)

        @wp.kernel(module="unique")
        def k_param(x: wp.array(dtype=wp.vec3), s: wp.float64, out: wp.array(dtype=float)):
            out[0] = x[0][0] * float(s)

        @wp.kernel(module="unique")
        def k_f32(x: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
            out[0] = wp.length(x[0])

        p_np = np.array([[1.0e4 + 0.3, 2.0e4 + 0.7, 3.1], [1.0e4 + 0.2, 2.0e4 + 0.9, 3.3]], dtype=np.float32)

        def run(k, dev, extra=(), grad=False):
            p = wp.array(p_np, dtype=wp.vec3, device=dev, requires_grad=grad)
            out = wp.zeros(1, dtype=float, device=dev, requires_grad=grad)
            tape = wp.Tape()
            with tape:
                wp.launch(k, dim=1, inputs=[p, *extra, out], device=dev)
            if grad:
                tape.backward(grads={out: wp.ones(1, dtype=float, device=dev)})
                return out.numpy()[0], p.grad.numpy()
            return out.numpy()[0]
        """
    )

    def test_fp64_values_error_mode_raises(self):
        # metal_fp64 = "error" (default): float64 locals, casts and scalar
        # parameters used to compute silently in float32 (vec3d cross: 3.6064e7
        # vs CPU 3.6030e7); they now raise like float64 arrays do.
        snippet = self._FP64_KERNELS + textwrap.dedent(
            """
            from warp._src.codegen_metal import MetalCodegenError
            assert wp.config.metal_fp64 == "error"
            for k, extra in [(k_vec, ()), (k_scalar, ()), (k_param, (wp.float64(2.0),))]:
                run(k, "cpu", extra)
                try:
                    run(k, "metal:0", extra)
                except MetalCodegenError as e:
                    assert "float64" in str(e) and "metal_fp64" in str(e), str(e)
                else:
                    raise AssertionError(f"{k.key}: float64 values on Metal did not raise in 'error' mode")
            # float32-only kernels are unaffected
            assert np.isclose(run(k_f32, "metal:0"), run(k_f32, "cpu"))

            # A float64 field that is only loaded and cast down does no fp64
            # arithmetic; the conversion rounds exactly as on the CPU.
            @wp.struct
            class P:
                nu: wp.float64

            @wp.kernel(module="unique")
            def k_field(p: P, out: wp.array(dtype=float)):
                out[0] = wp.float32(p.nu) * 3.0

            @wp.kernel(module="unique")
            def k_field_math(p: P, out: wp.array(dtype=float)):
                out[0] = wp.float32(p.nu * p.nu)

            pv = P()
            pv.nu = wp.float64(1.0 / 3.0)
            res = {}
            for dev in ("cpu", "metal:0"):
                o = wp.zeros(1, dtype=float, device=dev)
                wp.launch(k_field, dim=1, inputs=[pv, o], device=dev)
                res[dev] = o.numpy()[0]
            assert res["cpu"] == res["metal:0"], res
            try:
                wp.launch(k_field_math, dim=1, inputs=[pv, wp.zeros(1, dtype=float, device="metal:0")],
                          device="metal:0")
            except MetalCodegenError:
                pass
            else:
                raise AssertionError("float64 field arithmetic on Metal did not raise in 'error' mode")

            # Only float32 / float16 casts are exact on the value Metal reads:
            # an integer cast or a predicate on it is not (int32(2.9999999999)
            # was 3 on Metal, 2 on the CPU; isfinite(1e300) was false).
            @wp.kernel(module="unique")
            def k_field_int(p: P, out: wp.array(dtype=int)):
                out[0] = wp.int32(p.nu)

            @wp.kernel(module="unique")
            def k_field_isfinite(p: P, out: wp.array(dtype=int)):
                out[0] = wp.where(wp.isfinite(p.nu), 1, 0)

            @wp.kernel(module="unique")
            def k_field_float(p: P, out: wp.array(dtype=float)):
                out[0] = float(p.nu)

            pv.nu = wp.float64(2.9999999999)
            for k in (k_field_int, k_field_isfinite):
                try:
                    wp.launch(k, dim=1, inputs=[pv, wp.zeros(1, dtype=int, device="metal:0")], device="metal:0")
                except MetalCodegenError:
                    pass
                else:
                    raise AssertionError(f"{k.key}: non-exact use of a float64 field did not raise")
            res = {}
            for dev in ("cpu", "metal:0"):
                o = wp.zeros(1, dtype=float, device=dev)
                wp.launch(k_field_float, dim=1, inputs=[pv, o], device=dev)
                res[dev] = o.numpy()[0]
            assert res["cpu"] == res["metal:0"], res
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_fp64_values_demote_mode_computes_float32(self):
        # "demote" is the explicit opt-in to float32 arithmetic for float64
        # values: same result as the float32 math, not the CPU's double.
        snippet = self._FP64_KERNELS + textwrap.dedent(
            """
            wp.config.metal_fp64 = "demote"
            @wp.kernel(module="unique")
            def k_vec32(p: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
                out[0] = wp.length_sq(wp.cross(p[0], p[1]))

            got = run(k_vec, "metal:0")
            expected = run(k_vec32, "metal:0")
            assert np.isclose(got, expected, rtol=1e-6), (got, expected)
            assert np.isclose(run(k_param, "metal:0", (wp.float64(2.0),)), run(k_param, "cpu", (wp.float64(2.0),)))
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_fp64_values_emulate_mode_matches_cpu(self):
        # "emulate" computes float64 exactly; vec3d cross (and its adjoint)
        # had no wp_f64 overload and failed to compile.
        snippet = self._FP64_KERNELS + textwrap.dedent(
            """
            wp.config.metal_fp64 = "emulate"
            for k, extra in [(k_vec, ()), (k_scalar, ()), (k_param, (wp.float64(2.0),))]:
                c = run(k, "cpu", extra)
                m = run(k, "metal:0", extra)
                assert c == m, (k.key, c, m)
            c_out, c_grad = run(k_vec, "cpu", grad=True)
            m_out, m_grad = run(k_vec, "metal:0", grad=True)
            assert c_out == m_out, (c_out, m_out)
            np.testing.assert_allclose(m_grad, c_grad, rtol=1e-6)
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_static_struct_constant_matches_cpu(self):
        # A struct value built by wp.static() is a struct constant; Metal
        # declared its per-field locals zero-initialized and ignored the value.
        snippet = textwrap.dedent(
            """
            import numpy as np

            @wp.struct
            class Inner:
                mat: wp.mat33
                vec: wp.vec3
                i: int

            @wp.struct
            class Outer:
                s: Inner
                tf: wp.transform
                q: wp.quat

            @wp.func
            def make(mat: wp.mat33, vec: wp.vec3, i: int, tf: wp.transform, q: wp.quat):
                o = Outer()
                o.s.mat = mat
                o.s.vec = vec
                o.s.i = i
                o.tf = tf
                o.q = q
                return o

            @wp.kernel(module="unique")
            def k(res: wp.array(dtype=Outer), out: wp.array(dtype=float)):
                c = wp.static(make(wp.mat33(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), wp.vec3(1.0, 2.0, 3.0), 7,
                                   wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5)),
                                   wp.quat(0.1, 0.2, 0.3, 0.9)))
                res[0] = c
                out[0] = c.s.vec[1] + c.s.mat[2, 1] + float(c.s.i) + c.tf.p[2] + c.q[3]

            r = {}
            for dev in ("cpu", "metal:0"):
                res = wp.zeros(1, dtype=Outer, device=dev)
                out = wp.zeros(1, dtype=float, device=dev)
                wp.launch(k, dim=1, inputs=[res, out], device=dev)
                r[dev] = (res.numpy(), out.numpy())
            assert r["cpu"][0].tobytes() == r["metal:0"][0].tobytes(), r
            np.testing.assert_array_equal(r["metal:0"][1], r["cpu"][1])
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_tile_from_thread_with_lanes_raises(self):
        # Serial-model kernels run one thread per block on Metal, so
        # tile_from_thread() cannot read another lane's value; block_dim > 1
        # used to return each thread's own value (CUDA / CPU with
        # enable_cpu_blocks broadcast the source lane). Now loud; block_dim=1
        # (the only lane is the source) still runs.
        snippet = textwrap.dedent(
            """
            import numpy as np

            @wp.kernel(module="unique")
            def k(output: wp.array(dtype=int)):
                idx = wp.tid()
                t = wp.tile_from_thread(shape=(1,), value=idx * 10, thread_idx=wp.block_dim() - 1, storage="shared")
                output[idx] = wp.tile_extract(t, 0) + idx

            out = wp.zeros(16, dtype=int, device="metal:0")
            try:
                wp.launch(k, dim=[16], inputs=[out], block_dim=16, device="metal:0")
            except RuntimeError as e:
                assert "tile_from_thread" in str(e), e
            else:
                raise AssertionError("tile_from_thread with block_dim=16 did not raise on Metal")
            res = {}
            for dev in ("cpu", "metal:0"):
                o = wp.zeros(16, dtype=int, device=dev)
                wp.launch(k, dim=[16], inputs=[o], block_dim=1, device=dev)
                res[dev] = o.numpy()
            np.testing.assert_array_equal(res["metal:0"], res["cpu"])
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_custom_func_grad_matches_cpu(self):
        # @wp.func_grad replaces a function's adjoint. Metal inlined the
        # automatic adjoint of the primal body instead, so a clipped gradient
        # came out unclipped and side effects in the gradient never ran.
        snippet = textwrap.dedent(
            """
            import numpy as np
            @wp.func
            def clipped_sq(x: float) -> float:
                return x * x
            @wp.func_grad(clipped_sq)
            def adj_clipped_sq(x: float, adj_ret: float):
                wp.adjoint[x] += wp.clamp(2.0 * x * adj_ret, -1.0, 1.0)

            @wp.func
            def two(x: float, y: float):
                return x * y, x + y
            @wp.func_grad(two)
            def adj_two(x: float, y: float, adj_r0: float, adj_r1: float):
                wp.adjoint[x] += 10.0 * adj_r0 + adj_r1
                wp.adjoint[y] += 100.0 * adj_r0

            @wp.func
            def counter_increment(counter: wp.array(dtype=int), idx: int):
                return wp.atomic_add(counter, idx, 1)
            @wp.func_replay(counter_increment)
            def replay_counter_increment(counter: wp.array(dtype=int), idx: int):
                return idx

            @wp.kernel(module="unique")
            def k(x: wp.array(dtype=float), counter: wp.array(dtype=int), y: wp.array(dtype=float)):
                i = wp.tid()
                a, b = two(x[i], 2.0 * x[i])
                slot = counter_increment(counter, 0)
                y[i] = clipped_sq(x[i]) + a + b + x[i] * float(slot)
            r = {}
            for d in ("cpu", "metal:0"):
                x = wp.array(np.linspace(-2, 2, 8).astype(np.float32), dtype=float, device=d, requires_grad=True)
                c = wp.zeros(1, dtype=int, device=d)
                y = wp.zeros(8, dtype=float, device=d, requires_grad=True)
                t = wp.Tape()
                with t:
                    wp.launch(k, dim=8, inputs=[x, c, y], device=d)
                t.backward(grads={y: wp.ones(8, dtype=float, device=d)})
                r[d] = (y.numpy(), x.grad.numpy(), c.numpy())
            for a, b in zip(r["cpu"], r["metal:0"]):
                np.testing.assert_allclose(b, a, rtol=1e-6)

            @wp.func
            def side_effect(counter: wp.array(dtype=int), v: float) -> float:
                wp.atomic_add(counter, 0, 1)
                return v

            @wp.func
            def scale(counter: wp.array(dtype=int), x: float, s: float) -> float:
                return x * s

            @wp.func_grad(scale)
            def adj_scale(counter: wp.array(dtype=int), x: float, s: float, adj_ret: float):
                wp.adjoint[x] += side_effect(counter, adj_ret * s)
                wp.adjoint[s] += adj_ret * x

            @wp.kernel(module="unique")
            def k2(counter: wp.array(dtype=int), x: wp.array(dtype=float), s: wp.array(dtype=float), out: wp.array(dtype=float)):
                i = wp.tid()
                out[i] = scale(counter, x[i], s[i])

            for d in ("cpu", "metal:0"):
                counter = wp.zeros(1, dtype=int, device=d)
                x = wp.array([3.0], dtype=float, device=d, requires_grad=True)
                s = wp.array([2.0], dtype=float, device=d, requires_grad=True)
                out = wp.zeros(1, dtype=float, device=d, requires_grad=True)
                t = wp.Tape()
                with t:
                    wp.launch(k2, dim=1, inputs=[counter, x, s, out], device=d)
                t.backward(grads={out: wp.ones(1, dtype=float, device=d)})
                assert counter.numpy()[0] == 1, (d, counter.numpy())
                assert x.grad.numpy()[0] == 2.0 and s.grad.numpy()[0] == 3.0, d
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_numpy_integer_launch_dims(self):
        # CPU/CUDA canonicalize dim with operator.index(); Metal raised a
        # TypeError for dim=np.int64(n) (and numpy entries in a sequence).
        snippet = textwrap.dedent(
            """
            import numpy as np

            @wp.kernel(module="unique")
            def k(a: wp.array(dtype=float)):
                i = wp.tid()
                a[i] = float(i)

            @wp.kernel(module="unique")
            def k2(a: wp.array2d(dtype=float)):
                i, j = wp.tid()
                a[i, j] = float(i * 10 + j)

            for dim in (np.int64(4), [np.int32(4)], (np.int64(4),), np.uint16(4)):
                res = {}
                for dev in ("cpu", "metal:0"):
                    a = wp.zeros(4, dtype=float, device=dev)
                    wp.launch(k, dim=dim, inputs=[a], device=dev)
                    res[dev] = a.numpy()
                np.testing.assert_array_equal(res["metal:0"], res["cpu"], err_msg=repr(dim))
            res = {}
            for dev in ("cpu", "metal:0"):
                a = wp.zeros((3, 4), dtype=float, device=dev)
                wp.launch(k2, dim=(np.int64(3), np.int32(4)), inputs=[a], device=dev)
                res[dev] = a.numpy()
            np.testing.assert_array_equal(res["metal:0"], res["cpu"])
            # empty launches with numpy extents are no-ops (no argument validation), as on CPU
            for dim in (np.int64(0), (np.int32(0), np.int64(3))):
                for dev in ("cpu", "metal:0"):
                    wp.launch(k, dim=dim, inputs=[wp.zeros(4, dtype=wp.vec3, device=dev)], device=dev)
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_cubql_bvh_rebuild_matches_cpu(self):
        # Bvh.rebuild() of a cuBQL tree called the native CUDA rebuild with
        # the Metal descriptor id (a no-op here), so queries kept using the
        # stale tree after the items moved.
        snippet = textwrap.dedent(
            """
            import numpy as np

            if not wp.is_cubql_available():
                raise SystemExit(0)

            @wp.kernel(module="unique")
            def q(bvh: wp.uint64, lo: wp.vec3, hi: wp.vec3, hits: wp.array(dtype=int)):
                query = wp.bvh_query_aabb(bvh, lo, hi)
                i = int(0)
                while wp.bvh_query_next(query, i):
                    hits[i] = 1

            rng = np.random.default_rng(123)
            lo0 = rng.random((100, 3)) * 5.0
            hi0 = lo0 + rng.random((100, 3)) * 5.0
            lo1 = rng.random((100, 3)) * 5.0
            hi1 = lo1 + rng.random((100, 3)) * 5.0
            res = {}
            for dev in ("cpu", "metal:0"):
                lo = wp.array(lo0, dtype=wp.vec3, device=dev)
                hi = wp.array(hi0, dtype=wp.vec3, device=dev)
                bvh = wp.Bvh(lo, hi, constructor="cubql", leaf_size=1)
                lo.assign(lo1)
                hi.assign(hi1)
                bvh.rebuild()
                hits = wp.zeros(100, dtype=int, device=dev)
                wp.launch(q, dim=1, inputs=[bvh.id, wp.vec3(2.0), wp.vec3(8.0), hits], device=dev)
                res[dev] = hits.numpy()
            np.testing.assert_array_equal(res["metal:0"], res["cpu"])
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)


if __name__ == "__main__":
    unittest.main(verbosity=2)
