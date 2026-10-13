# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal kernels that compiled and ran but returned values different from CPU.

Each test launches the same kernel on ``cpu`` and ``metal:0`` and compares.
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_enabled


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalMismatch(unittest.TestCase):
    def test_negative_index_wraps(self):
        # Native ``index()`` maps ``arr[-1]`` to the last element; Newton reads
        # ``gravity[world]`` with ``world == -1`` for global entities.
        snippet = textwrap.dedent(
            """
            import numpy as np

            @wp.kernel
            def k(g: wp.array[wp.vec3], w: wp.array[wp.int32], g2: wp.array2d[float],
                  out: wp.array[wp.vec3], out2: wp.array[float], out3: wp.array[float]):
                tid = wp.tid()
                out[tid] = g[w[tid]]
                out2[tid] = g2[w[tid], -1]
                wp.atomic_add(out3, w[tid], 1.0)

            res = {}
            for dev in ("cpu", "metal:0"):
                g = wp.array([[1.0, 2.0, 3.0], [0.0, 0.0, -9.81]], dtype=wp.vec3, device=dev)
                g2 = wp.array(np.arange(6, dtype=np.float32).reshape(2, 3), device=dev)
                w = wp.array([0, -1], dtype=wp.int32, device=dev)
                out = wp.zeros(2, dtype=wp.vec3, device=dev)
                out2 = wp.zeros(2, dtype=float, device=dev)
                out3 = wp.zeros(2, dtype=float, device=dev)
                wp.launch(k, dim=2, inputs=[g, w, g2], outputs=[out, out2, out3], device=dev)
                res[dev] = (out.numpy(), out2.numpy(), out3.numpy())
            for a, b in zip(res["cpu"], res["metal:0"]):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_allclose(res["metal:0"][0][1], [0.0, 0.0, -9.81], rtol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_func_native_snippet_is_called(self):
        # The Python body of a ``@wp.func_native`` is a stub; inlining it left
        # the result local unassigned (silently 0). The snippet's non-CUDA
        # branch must run instead.
        snippet = textwrap.dedent(
            '''
            import numpy as np

            @wp.func_native("""
            uint32_t i = reinterpret_cast<uint32_t&>(f);
            uint32_t mask = (uint32_t)(-(int)(i >> 31)) | 0x80000000;
            return i ^ mask;
            """)
            def float_flip(f: float) -> wp.uint32: ...

            @wp.func_native("""
            uint32_t mask = ((u >> 31) - 1) | 0x80000000;
            uint32_t i = u ^ mask;
            return reinterpret_cast<float&>(i);
            """)
            def ifloat_flip(u: wp.uint32) -> float: ...

            @wp.func_native("""return 0x7FFFFFFFFFFFFFFFull;""")
            def sentinel() -> wp.uint64: ...

            @wp.func_native("""
            #if defined(__CUDA_ARCH__)
            return nextafterf(value, INFINITY);
            #else
            return __builtin_nextafterf(value, INFINITY);
            #endif
            """)
            def next_up(value: float) -> float: ...

            @wp.kernel
            def k(x: wp.array[float], flipped: wp.array[wp.uint32], back: wp.array[float],
                  up: wp.array[float], s: wp.array[wp.uint64]):
                tid = wp.tid()
                u = float_flip(x[tid])
                flipped[tid] = u
                back[tid] = ifloat_flip(u)
                up[tid] = next_up(x[tid])
                s[tid] = sentinel()

            res = {}
            for dev in ("cpu", "metal:0"):
                x = wp.array([1.0, -1.0, 0.5, 0.0], dtype=float, device=dev)
                outs = [wp.zeros(4, dtype=t, device=dev) for t in (wp.uint32, float, float, wp.uint64)]
                wp.launch(k, dim=4, inputs=[x], outputs=outs, device=dev)
                res[dev] = [o.numpy() for o in outs]
            for a, b in zip(res["cpu"], res["metal:0"]):
                np.testing.assert_array_equal(a, b)
            assert res["metal:0"][3][0] == 0x7FFFFFFFFFFFFFFF
            '''
        )
        _run_with_metal_enabled(self, snippet)

    def test_inlined_transform_and_quat_mul(self):
        # A transform / quat local of an inlined ``@wp.func`` (reassigned in an
        # ``if``) lowered ``a * b`` to the component-wise vec7 / float4
        # product instead of rigid composition / the Hamilton product.
        snippet = textwrap.dedent(
            """
            import numpy as np

            @wp.func
            def f(parent: int, a: wp.transform, b: wp.transform, qa: wp.quat, qb: wp.quat):
                x = b
                q = qb
                if parent >= 0:
                    x = a * b
                    q = qa * qb
                return x * b, q * qb

            @wp.kernel
            def k(xs: wp.array[wp.transform], qs: wp.array[wp.quat],
                  out: wp.array[wp.transform], outq: wp.array[wp.quat]):
                x, q = f(0, xs[0], xs[1], qs[0], qs[1])
                out[0] = x
                outq[0] = q

            rng = np.random.default_rng(0)
            q = rng.normal(size=(2, 4)).astype(np.float32)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            xs = np.concatenate([rng.normal(size=(2, 3)).astype(np.float32), q], axis=1)
            res = {}
            for dev in ("cpu", "metal:0"):
                o = wp.zeros(1, dtype=wp.transform, device=dev)
                oq = wp.zeros(1, dtype=wp.quat, device=dev)
                wp.launch(k, dim=1, inputs=[wp.array(xs, dtype=wp.transform, device=dev),
                          wp.array(q, dtype=wp.quat, device=dev)], outputs=[o, oq], device=dev)
                res[dev] = (o.numpy(), oq.numpy())
            for a, b in zip(res["cpu"], res["metal:0"]):
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_large_struct_helpers_stay_correct(self):
        # Apple's Metal compiler drops stores when a struct over ~144 bytes
        # (wp_mat7x7 and up) is passed by value to a helper it does not
        # inline; all helpers are now forced inline.
        snippet = textwrap.dedent(
            """
            import numpy as np
            N = 8
            M = wp.types.matrix(shape=(N, N), dtype=wp.float32)


            @wp.func
            def f(A: M, n: int):
                at = wp.transpose(A)
                vt = wp.identity(n=N, dtype=float)
                ut = wp.identity(n=N, dtype=float)
                for i in range(n):
                    ut[i] = at[i] / wp.length(at[i])
                return wp.transpose(ut), wp.transpose(vt)


            @wp.kernel
            def k(a: wp.array[M], n: int, u: wp.array[M], v: wp.array[M]):
                uu, vv = f(a[0], n)
                u[0] = uu
                v[0] = vv


            a_np = np.random.default_rng(32).normal(size=(1, N, N)).astype(np.float32)
            ref = a_np[0] / np.linalg.norm(a_np[0], axis=0, keepdims=True)
            for dev in ("cpu", "metal:0"):
                u = wp.zeros(1, dtype=M, device=dev)
                v = wp.zeros(1, dtype=M, device=dev)
                wp.launch(k, dim=1, inputs=[wp.array(a_np, dtype=M, device=dev), N], outputs=[u, v], device=dev)
                np.testing.assert_allclose(u.numpy()[0], ref, atol=1e-5)
                np.testing.assert_allclose(v.numpy()[0], np.eye(N))
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_column_tile_load_store(self):
        # A ``shape=(R, 1)`` tile is 2-D: offset (row, col) and rows walk the
        # array's leading dim. It was lowered as a 1-D tile (row stride 1),
        # which broke Kamino's semi-sparse Cholesky solve with >1 RHS.
        snippet = textwrap.dedent(
            """
            import numpy as np

            @wp.kernel
            def k(src: wp.array3d[float], dst: wp.array3d[float], acc: wp.array2d[float]):
                b, c, _t = wp.tid()
                s = src[b]
                d = dst[b]
                for i in range(0, 8, 4):
                    t = wp.tile_load(s, shape=(4, 1), offset=(i, c))
                    wp.tile_store(d, t, offset=(i, c))
                    wp.tile_atomic_add(acc, t, offset=(i, c))

            src_np = np.arange(2 * 8 * 5, dtype=np.float32).reshape(2, 8, 5)
            res = {}
            for dev in ("cpu", "metal:0"):
                dst = wp.zeros((2, 8, 5), dtype=float, device=dev)
                acc = wp.zeros((8, 5), dtype=float, device=dev)
                wp.launch_tiled(k, dim=[2, 5], inputs=[wp.array(src_np, device=dev)], outputs=[dst, acc],
                                block_dim=64, device=dev)
                res[dev] = (dst.numpy(), acc.numpy())
            np.testing.assert_array_equal(res["cpu"][0], src_np)
            for a, b in zip(res["cpu"], res["metal:0"]):
                np.testing.assert_array_equal(a, b)
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_negative_index_struct_fields_views_components(self):
        # Struct-arg array fields, component stores, views handed to tiles,
        # int64 indices and loop-carried decrements must all wrap like
        # native ``index()``.
        snippet = textwrap.dedent(
            """
            import numpy as np

            @wp.struct
            class S:
                a: wp.array[float]
                b: wp.array2d[float]

            @wp.kernel
            def k_struct(s: S, w: wp.array[wp.int32], o: wp.array[float]):
                t = wp.tid()
                o[t] = s.a[w[t]] + 10.0 * s.b[w[t], -1]

            @wp.kernel
            def k_comp(w: wp.array[wp.int32], a: wp.array[wp.vec3]):
                t = wp.tid()
                a[w[t]][1] = 5.0

            @wp.kernel
            def k_view(w: wp.array[wp.int32], a: wp.array2d[float], o: wp.array2d[float]):
                i = wp.tid()
                t = wp.tile_load(a[w[0]], shape=3)
                wp.tile_store(o[0], t)

            @wp.kernel
            def k_i64(w: wp.array[wp.int64], a: wp.array[float], o: wp.array[float]):
                t = wp.tid()
                o[t] = a[w[t]]

            @wp.kernel
            def k_loop(a: wp.array[float], o: wp.array[float]):
                t = wp.tid()
                s = float(0.0)
                for k in range(3):
                    s += a[t] * float(k + 1)
                    t = t - 1
                o[0] = s

            def cases(d):
                s = S()
                s.a = wp.array([1.0, 2.0, 3.0, 4.0], dtype=float, device=d)
                s.b = wp.array(np.arange(6, dtype=np.float32).reshape(2, 3), device=d)
                w = wp.array([0, -1], dtype=wp.int32, device=d)
                o1 = wp.zeros(2, dtype=float, device=d)
                wp.launch(k_struct, dim=2, inputs=[s, w], outputs=[o1], device=d)
                v = wp.zeros(3, dtype=wp.vec3, device=d)
                wp.launch(k_comp, dim=1, inputs=[wp.array([-1], dtype=wp.int32, device=d), v], device=d)
                o2 = wp.zeros((2, 3), dtype=float, device=d)
                wp.launch(k_view, dim=1, inputs=[wp.array([-1], dtype=wp.int32, device=d), s.b], outputs=[o2], device=d)
                o3 = wp.zeros(2, dtype=float, device=d)
                wp.launch(k_i64, dim=2, inputs=[wp.array([1, -1], dtype=wp.int64, device=d), s.a], outputs=[o3], device=d)
                o4 = wp.zeros(1, dtype=float, device=d)
                wp.launch(k_loop, dim=1, inputs=[s.a], outputs=[o4], device=d)
                return [o1.numpy(), v.numpy(), o2.numpy(), o3.numpy(), o4.numpy()]

            cpu, metal = cases("cpu"), cases("metal:0")
            for a, b in zip(cpu, metal):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(metal[1][2], [0.0, 5.0, 0.0])
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_func_native_vector_params(self):
        snippet = textwrap.dedent(
            '''
            import numpy as np

            @wp.func_native("""return v[0] + 2.0f * v[1] + 3.0f * v[2] + q[3];""")
            def wsum(v: wp.vec3, q: wp.quat) -> float: ...

            @wp.func_native("""return v * 2.0f;""")
            def dbl(v: wp.vec3) -> wp.vec3: ...

            @wp.kernel
            def k(a: wp.array[wp.vec3], o: wp.array[float], o2: wp.array[wp.vec3]):
                t = wp.tid()
                o[t] = wsum(a[t], wp.quat(0.0, 0.0, 0.0, 1.0))
                o2[t] = dbl(a[t])

            res = {}
            for d in ("cpu", "metal:0"):
                a = wp.array(np.arange(6, dtype=np.float32).reshape(2, 3), dtype=wp.vec3, device=d)
                o = wp.zeros(2, dtype=float, device=d)
                o2 = wp.zeros(2, dtype=wp.vec3, device=d)
                wp.launch(k, dim=2, inputs=[a], outputs=[o, o2], device=d)
                res[d] = (o.numpy(), o2.numpy())
            for a, b in zip(res["cpu"], res["metal:0"]):
                np.testing.assert_array_equal(a, b)
            '''
        )
        _run_with_metal_enabled(self, snippet)

    def test_func_native_barriers(self):
        # ``WP_TILE_SYNC()`` / ``__syncthreads()`` are no-ops in the serial
        # model and real threadgroup barriers in distributed-tile kernels.
        snippet = textwrap.dedent(
            '''
            import numpy as np

            @wp.func_native("""
            #if defined(__CUDA_ARCH__)
            __syncthreads();
            #endif
            """)
            def sync_threads(): ...

            @wp.func_native(snippet="WP_TILE_SYNC();")
            def tile_sync(): ...

            @wp.kernel
            def k_plain(a: wp.array[float], o: wp.array[float]):
                i = wp.tid()
                tile_sync()
                o[i] = a[i]

            @wp.kernel
            def k_coop(vals: wp.array2d[float], scratch: wp.array2d[float], out: wp.array[float]):
                w, tid = wp.tid()
                acc = float(0.0)
                for i in range(tid, 64, wp.block_dim()):
                    acc += vals[w, i]
                scratch[w, tid] = acc
                sync_threads()
                tile_sync()
                tot = wp.tile_reduce(wp.add, wp.tile(acc))
                if tid == 0:
                    s = float(0.0)
                    for j in range(wp.block_dim()):
                        s += scratch[w, j]
                    out[w] = tot[0] - s + 1.0

            for d in ("cpu", "metal:0"):
                a = wp.array(np.arange(8, dtype=np.float32), device=d)
                o = wp.zeros(8, dtype=float, device=d)
                wp.launch(k_plain, dim=8, inputs=[a], outputs=[o], device=d)
                np.testing.assert_array_equal(o.numpy(), np.arange(8, dtype=np.float32))
                v = wp.array(np.arange(3 * 64, dtype=np.float32).reshape(3, 64), device=d)
                sc = wp.zeros((3, 32), dtype=float, device=d)
                out = wp.zeros(3, dtype=float, device=d)
                wp.launch_tiled(k_coop, dim=3, inputs=[v, sc], outputs=[out], block_dim=32, device=d)
                np.testing.assert_array_equal(out.numpy(), [1.0, 1.0, 1.0])
            art = k_coop._metal_artifact
            assert art.coop_block_n == 32, art.coop_block_n
            assert "threadgroup_barrier(metal::mem_flags::mem_threadgroup" in art.header
            '''
        )
        _run_with_metal_enabled(self, snippet)

    def test_func_native_lane_primitives_rejected(self):
        # A CUDA-only shuffle cannot run faithfully when lanes are real
        # threads: a launch_tiled kernel without tile semantics and a
        # distributed-tile kernel both raise instead of using the host branch.
        snippet = textwrap.dedent(
            '''
            import numpy as np
            from warp._src.codegen_metal import MetalCodegenError

            @wp.func_native("""
            #if defined(__CUDA_ARCH__)
            return __shfl_sync(0xffffffffu, value, 0, 32);
            #else
            return value;
            #endif
            """)
            def bcast(value: float) -> float: ...

            @wp.kernel
            def k_lanes(a: wp.array[float], o: wp.array[float]):
                i, lane = wp.tid()
                o[lane] = bcast(a[lane])

            @wp.kernel
            def k_coop(vals: wp.array2d[float], out: wp.array[float]):
                w, tid = wp.tid()
                acc = bcast(vals[w, tid])
                tot = wp.tile_reduce(wp.add, wp.tile(acc))
                if tid == 0:
                    out[w] = tot[0]

            a = wp.array(np.arange(64, dtype=np.float32), device="metal:0")
            o = wp.zeros(64, dtype=float, device="metal:0")
            wp.launch_tiled(k_lanes, dim=[1], inputs=[a], outputs=[o], block_dim=1, device="metal:0")
            try:
                wp.launch_tiled(k_lanes, dim=[1], inputs=[a], outputs=[o], block_dim=64, device="metal:0")
                raise AssertionError("expected RuntimeError")
            except RuntimeError as e:
                assert "intra-block primitives" in str(e), e
            v = wp.zeros((2, 32), dtype=float, device="metal:0")
            out = wp.zeros(2, dtype=float, device="metal:0")
            try:
                wp.launch_tiled(k_coop, dim=2, inputs=[v], outputs=[out], block_dim=32, device="metal:0")
                raise AssertionError("expected MetalCodegenError")
            except MetalCodegenError as e:
                assert "CUDA-only branch" in str(e), e
            '''
        )
        _run_with_metal_enabled(self, snippet)

    def test_func_native_adjoint_raises(self):
        # Native adjoint snippets are not lowered; raising beats silently
        # returning zero gradients.
        snippet = textwrap.dedent(
            """
            from warp._src.codegen_metal import MetalCodegenError

            @wp.func_native("return 2.0f * x;", adj_snippet="adj_x += 2.0f * adj_ret;")
            def dbl(x: float) -> float: ...

            @wp.kernel
            def k(a: wp.array[float], o: wp.array[float]):
                t = wp.tid()
                o[t] = dbl(a[t])

            a = wp.array([1.0, 2.0], dtype=float, device="metal:0", requires_grad=True)
            o = wp.zeros(2, dtype=float, device="metal:0", requires_grad=True)
            tape = wp.Tape()
            with tape:
                wp.launch(k, dim=2, inputs=[a], outputs=[o], device="metal:0")
            assert o.numpy().tolist() == [2.0, 4.0]
            try:
                tape.backward(grads={o: wp.ones(2, dtype=float, device="metal:0")})
                raise AssertionError("expected MetalCodegenError")
            except MetalCodegenError as e:
                assert "adjoint of a native snippet" in str(e), e
            """
        )
        _run_with_metal_enabled(self, snippet)

    def test_func_native_writable_pun_and_name_collision(self):
        # ``reinterpret_cast<T&>(x) ^= m`` needs a ``thread`` reference, and
        # two snippets sharing a Python name must get distinct helpers.
        snippet = textwrap.dedent(
            '''
            import numpy as np

            @wp.func_native("""reinterpret_cast<uint32_t&>(f) ^= 0x80000000u; return f;""")
            def neg_bits(f: float) -> float: ...

            def make(scale):
                @wp.func_native(f"return x * {scale:.1f}f;")
                def f(x: float) -> float: ...

                return f

            f2, f3 = make(2.0), make(3.0)

            @wp.kernel
            def k(x: wp.array[float], o: wp.array[float], o2: wp.array[float]):
                i = wp.tid()
                o[i] = neg_bits(x[i])
                o2[i] = f2(x[i]) + 10.0 * f3(x[i])

            res = {}
            for d in ("cpu", "metal:0"):
                x = wp.array([1.0, -2.5, 0.0], dtype=float, device=d)
                o = wp.zeros(3, dtype=float, device=d)
                o2 = wp.zeros(3, dtype=float, device=d)
                wp.launch(k, dim=3, inputs=[x], outputs=[o, o2], device=d)
                res[d] = (o.numpy(), o2.numpy())
            for a, b in zip(res["cpu"], res["metal:0"]):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(res["metal:0"][1], [32.0, -80.0, 0.0])
            '''
        )
        _run_with_metal_enabled(self, snippet)

    def test_func_native_tile_snippet_folds_block(self):
        # A ``WP_TILE_*`` snippet gives the kernel block semantics even with
        # no tile op or wp.block_dim() call: like CPU it runs one thread per
        # block instead of keeping block_dim independent lanes.
        snippet = textwrap.dedent(
            '''
            import numpy as np

            @wp.func_native(snippet="WP_TILE_SYNC();")
            def sync(): ...

            @wp.func_native(snippet="return WP_TILE_BLOCK_DIM;")
            def block_width() -> int: ...

            @wp.kernel
            def k(count: wp.array[float], width: wp.array[int]):
                w, t = wp.tid()
                sync()
                wp.atomic_add(count, w, 1.0)
                width[w] = block_width()

            for d in ("cpu", "metal:0"):
                count = wp.zeros(3, dtype=float, device=d)
                width = wp.zeros(3, dtype=int, device=d)
                wp.launch_tiled(k, dim=3, inputs=[count, width], block_dim=64, device=d)
                np.testing.assert_array_equal(count.numpy(), [1.0, 1.0, 1.0])
                np.testing.assert_array_equal(width.numpy(), [1, 1, 1])
            '''
        )
        _run_with_metal_enabled(self, snippet)

    def test_func_native_without_adjoint_has_zero_gradient(self):
        snippet = textwrap.dedent(
            '''
            import numpy as np

            @wp.func_native("""return 1.0f / sqrtf(value);""")
            def rsqrt_rn(value: float) -> float: ...

            @wp.kernel
            def k(x: wp.array[float], o: wp.array[float]):
                i = wp.tid()
                o[i] = rsqrt_rn(x[i]) + x[i] * x[i]

            for d in ("cpu", "metal:0"):
                x = wp.array([1.0, 4.0], dtype=float, device=d, requires_grad=True)
                o = wp.zeros(2, dtype=float, device=d, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(k, dim=2, inputs=[x], outputs=[o], device=d)
                tape.backward(grads={o: wp.ones(2, dtype=float, device=d)})
                np.testing.assert_array_equal(x.grad.numpy(), [2.0, 8.0])
            '''
        )
        _run_with_metal_enabled(self, snippet)


if __name__ == "__main__":
    unittest.main(verbosity=2)
