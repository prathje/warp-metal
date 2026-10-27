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


if __name__ == "__main__":
    unittest.main(verbosity=2)
