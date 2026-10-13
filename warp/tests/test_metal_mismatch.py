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


if __name__ == "__main__":
    unittest.main(verbosity=2)
