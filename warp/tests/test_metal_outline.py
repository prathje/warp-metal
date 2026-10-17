# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal ``@wp.func`` outlining (``warp.config.metal_outline_funcs``).

Each test runs a kernel on ``cpu`` and ``metal:0`` in a subprocess with
``metal_outline_funcs = "all"`` (every eligible inlined body becomes a
separate ``noinline`` MSL function), checks the outputs match exactly and
that the kernel really was outlined.
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_enabled

_COMMON = textwrap.dedent(
    """
    import numpy as np
    from typing import Any

    wp.config.metal_outline_funcs = "all"

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

    def assert_outlined(kernel, min_funcs=1):
        src = getattr(kernel, "_metal_native_wrapped_source", None)
        assert src is not None, "no native source"
        n = src.count("__attribute__((noinline)) void wp_outl_")
        assert n >= min_funcs, f"expected >= {min_funcs} outlined functions, got {n}"
    """
)


def _run(test_case, body: str, timeout: int = 240):
    _run_with_metal_enabled(test_case, _COMMON + textwrap.dedent(body), timeout=timeout)


@unittest.skipUnless(_is_apple_silicon() and _has_mlx(), "Metal requires Apple Silicon")
class TestMetalOutline(unittest.TestCase):
    def test_values_nested_and_returns(self):
        _run(
            self,
            """
            @wp.func
            def leaf(x: wp.vec3, s: float):
                if s > 1.0:
                    return x * s
                acc = wp.vec3()
                for i in range(int(s * 10.0)):
                    acc = acc + x * float(i)
                    if wp.length(acc) > 50.0:
                        return acc
                return acc

            @wp.func
            def mid(a: wp.array[wp.vec3], i: int, s: float):
                v = a[i]
                return leaf(v, s) + leaf(v, 2.0 * s), wp.length(v)

            @wp.kernel
            def k(a: wp.array[wp.vec3], s: float, b: wp.array[wp.vec3], c: wp.array[float]):
                i = wp.tid()
                r, l = mid(a, i, s + float(i) * 0.01)
                b[i] = r + leaf(a[i], s)
                c[i] = l

            def launch(dev):
                rng = np.random.default_rng(1)
                a = wp.array(rng.standard_normal((257, 3)).astype(np.float32), dtype=wp.vec3, device=dev)
                b = wp.zeros(257, dtype=wp.vec3, device=dev)
                c = wp.zeros(257, dtype=float, device=dev)
                wp.launch(k, 257, inputs=[a, 0.45], outputs=[b, c], device=dev)
                return [b.numpy(), c.numpy()]

            cpu, metal = both(launch)
            assert_same(cpu, metal, rtol=1e-6, atol=1e-6)
            assert_outlined(k, 2)
            """,
        )

    def test_mutated_params_are_by_value(self):
        # A callee that writes its value parameters must not leak the
        # writes into the caller's variables.
        _run(
            self,
            """
            @wp.struct
            class S:
                a: wp.vec3
                b: float
                n: int

            @wp.func
            def clobber(s: S, x: float, v: wp.vec3):
                s.a = wp.vec3(0.0)
                s.b = s.b * 2.0
                x = x + 1.0
                v[1] = 7.0
                return s.b + x + v[1] + wp.length(s.a)

            @wp.kernel
            def k(inp: wp.array[float], out: wp.array[float]):
                i = wp.tid()
                s = S()
                s.a = wp.vec3(1.0, 2.0, 3.0)
                s.b = inp[i]
                s.n = i
                x = inp[i] * 3.0
                v = wp.vec3(inp[i])
                r = clobber(s, x, v)
                out[i] = r + s.b + x + v[1] + s.a[2] + float(s.n)

            def launch(dev):
                inp = wp.array(np.linspace(-1, 1, 64).astype(np.float32), device=dev)
                out = wp.zeros(64, dtype=float, device=dev)
                wp.launch(k, 64, inputs=[inp], outputs=[out], device=dev)
                return out.numpy()

            cpu, metal = both(launch)
            assert_same(cpu, metal)
            assert_outlined(k)
            """,
        )

    def test_large_matrices_and_structs(self):
        # Values above Apple's ~144-byte by-value miscompile threshold
        # (mat77, mat88, a struct holding them) cross outlined calls by
        # reference.
        _run(
            self,
            """
            mat77 = wp.types.matrix(shape=(7, 7), dtype=wp.float32)
            mat88 = wp.types.matrix(shape=(8, 8), dtype=wp.float32)

            @wp.struct
            class Big:
                m: mat88
                n: mat77
                s: float

            @wp.func
            def transpose_add(m: mat88, k: float):
                t = wp.transpose(m)
                for r in range(8):
                    for c in range(8):
                        t[r, c] = t[r, c] + k * float(r - c)
                return t

            @wp.func
            def big_update(b: Big, k: float):
                out = Big()
                out.m = transpose_add(b.m, k)
                n = b.n * 2.0
                for r in range(7):
                    n[r, r] = n[r, r] + b.s
                out.n = n
                out.s = b.s + k
                return out

            @wp.kernel
            def k(a8: wp.array[mat88], a7: wp.array[mat77], o8: wp.array[mat88], o7: wp.array[mat77],
                  os: wp.array[float]):
                i = wp.tid()
                b = Big()
                b.m = a8[i]
                b.n = a7[i]
                b.s = float(i)
                b2 = big_update(b, 0.5)
                b3 = big_update(b2, -0.25)
                o8[i] = b3.m
                o7[i] = b3.n
                os[i] = b3.s

            def launch(dev):
                rng = np.random.default_rng(2)
                a8 = wp.array(rng.standard_normal((9, 8, 8)).astype(np.float32), dtype=mat88, device=dev)
                a7 = wp.array(rng.standard_normal((9, 7, 7)).astype(np.float32), dtype=mat77, device=dev)
                o8 = wp.zeros(9, dtype=mat88, device=dev)
                o7 = wp.zeros(9, dtype=mat77, device=dev)
                os_ = wp.zeros(9, dtype=float, device=dev)
                wp.launch(k, 9, inputs=[a8, a7], outputs=[o8, o7, os_], device=dev)
                return [o8.numpy(), o7.numpy(), os_.numpy()]

            cpu, metal = both(launch)
            assert_same(cpu, metal)
            assert_outlined(k)
            """,
        )

    def test_generic_overloads_and_arrays(self):
        _run(
            self,
            """
            @wp.func
            def accum(a: wp.array[Any], lo: int, hi: int, scale: Any):
                s = a.dtype(0)
                for j in range(lo, hi):
                    s += a[j] * scale
                return s

            @wp.func
            def write_back(a: wp.array[float], i: int, v: float):
                a[i] = v
                wp.atomic_add(a, 0, 1.0)

            @wp.kernel
            def k(f: wp.array[float], v: wp.array[wp.vec3], of: wp.array[float], ov: wp.array[wp.vec3],
                  acc: wp.array[float]):
                i = wp.tid()
                lo = wp.max(i - 3, 0)
                of[i] = accum(f, lo, i + 1, 2.0)
                ov[i] = accum(v, lo, i + 1, 0.5)
                write_back(acc, i + 1, of[i])

            def launch(dev):
                f = wp.array(np.arange(32, dtype=np.float32), device=dev)
                v = wp.array(np.arange(96, dtype=np.float32).reshape(32, 3), dtype=wp.vec3, device=dev)
                of = wp.zeros(32, dtype=float, device=dev)
                ov = wp.zeros(32, dtype=wp.vec3, device=dev)
                acc = wp.zeros(33, dtype=float, device=dev)
                wp.launch(k, 32, inputs=[f, v], outputs=[of, ov, acc], device=dev)
                return [of.numpy(), ov.numpy(), acc.numpy()]

            cpu, metal = both(launch)
            assert_same(cpu, metal)
            assert_outlined(k, 2)
            """,
        )

    def test_off_mode_emits_no_functions(self):
        _run(
            self,
            """
            wp.config.metal_outline_funcs = "off"

            @wp.func
            def f(x: float):
                y = x
                for i in range(4):
                    y = y * x + float(i)
                return y

            @wp.kernel
            def k(a: wp.array[float]):
                i = wp.tid()
                a[i] = f(float(i))

            a = wp.zeros(8, dtype=float, device="metal:0")
            wp.launch(k, 8, inputs=[a], device="metal:0")
            src = k._metal_native_wrapped_source
            assert "wp_outl_" not in src and "@@wp_outline" not in src
            b = wp.zeros(8, dtype=float, device="cpu")
            wp.launch(k, 8, inputs=[b], device="cpu")
            np.testing.assert_array_equal(a.numpy(), b.numpy())
            """,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
