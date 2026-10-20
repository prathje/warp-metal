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

    def test_stores_through_kernel_arrays_are_outlined(self):
        # Bodies that store or atomically update a kernel array argument
        # (Newton's write_contact_to_reducer) take the buffer pointer by
        # value and are outlined too.
        _run(
            self,
            """
            @wp.func
            def emit(counts: wp.array[int], vals: wp.array[wp.vec3], flags: wp.array[int], i: int, v: wp.vec3):
                slot = wp.atomic_add(counts, i % 4, 1)
                vals[i] = v * float(i % 4 + 1)
                if wp.length(v) > 1.0:
                    flags[i] = slot * 0 + 1
                    wp.atomic_max(counts, 4, i)
                else:
                    flags[i] = -1
                for j in range(3):
                    wp.atomic_add(vals, (i + j + 1) % vals.shape[0], v * 0.0)
                return slot * 0 + i % 4

            @wp.kernel
            def k(inp: wp.array[wp.vec3], counts: wp.array[int], vals: wp.array[wp.vec3], flags: wp.array[int],
                  bins: wp.array[int]):
                i = wp.tid()
                bins[i] = emit(counts, vals, flags, i, inp[i])

            def launch(dev):
                rng = np.random.default_rng(3)
                inp = wp.array(rng.standard_normal((200, 3)).astype(np.float32), dtype=wp.vec3, device=dev)
                counts = wp.zeros(5, dtype=int, device=dev)
                vals = wp.zeros(200, dtype=wp.vec3, device=dev)
                flags = wp.zeros(200, dtype=int, device=dev)
                bins = wp.zeros(200, dtype=int, device=dev)
                wp.launch(k, 200, inputs=[inp], outputs=[counts, vals, flags, bins], device=dev)
                return [counts.numpy(), vals.numpy(), flags.numpy(), bins.numpy()]

            cpu, metal = both(launch)
            assert_same(cpu, metal)
            assert_outlined(k)
            src = k._metal_native_wrapped_source
            kernel_body = src[src.index("[[kernel]]"):]
            assert "atomic_fetch_add" not in kernel_body, "store/atomic region stayed inline"
            """,
        )

    def test_outlined_bodies_count_as_kernel_code_for_launch_shape(self):
        # A lock loop in a @wp.func keeps the kernel off grid-only
        # threadgroup spanning whether the body is inlined or outlined;
        # library helpers keep their exemption.
        _run(
            self,
            """
            from warp._src import codegen_metal as cm

            head = "inline int h_cas(device atomic_int* p) { int e = 0; atomic_compare_exchange_weak_explicit(p, &e, 1, memory_order_relaxed, memory_order_relaxed); return e; }\\n"
            outl = (
                "__attribute__((noinline)) void wp_outl_0123456789abcdef(device atomic_int* p0, thread int& p1) {\\n"
                "    int l0 = 0; atomic_compare_exchange_weak_explicit(p0, &l0, 1, memory_order_relaxed, memory_order_relaxed); p1 = l0;\\n"
                "}\\n"
            )
            sig = "[[kernel]] void custom_kernel_k(device atomic_int* a [[buffer(0)]], device int* b [[buffer(1)]]) {"
            assert cm._grid_only_source(head + sig + "\\nb[0] = h_cas(a);\\n}\\n")
            assert not cm._grid_only_source(head + outl + sig + "\\nint r = 0; wp_outl_0123456789abcdef(a, r); b[0] = r;\\n}\\n")

            # 64-bit atomics lower to a lock loop (``wp_a64_*``) in the code
            # that uses them, here the outlined body.
            @wp.func
            def bump(acc: wp.array[wp.int64], i: int, v: float):
                s = float(0.0)
                for j in range(4):
                    s += v * float(j)
                wp.atomic_add(acc, i % 3, wp.int64(i))
                return s

            @wp.kernel
            def k(acc: wp.array[wp.int64], out: wp.array[float]):
                i = wp.tid()
                out[i] = bump(acc, i, float(i))

            def launch(dev):
                acc = wp.zeros(3, dtype=wp.int64, device=dev)
                out = wp.zeros(64, dtype=float, device=dev)
                wp.launch(k, 64, inputs=[acc], outputs=[out], device=dev)
                return [acc.numpy(), out.numpy()]

            cpu, metal = both(launch)
            assert_same(cpu, metal)
            assert_outlined(k)
            src = k._metal_native_wrapped_source
            assert "wp_a64_" not in src[src.index("[[kernel]]"):]
            assert not cm._grid_only_source(src)
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
