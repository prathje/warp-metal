# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal-vs-CPU checks for whole-struct stores into struct arrays with array members.

On Metal each array member of a struct element is stored as the host's
``array_t`` record and read through launch-time pointer tables; a store such
as ``data[i] = src[j]`` must copy those records like the native by-value
struct copy. Each case runs in a subprocess (see ``test_metal_launch``).
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_enabled

_PRELUDE = textwrap.dedent(
    """
    import numpy as np

    @wp.struct
    class OnlyArr:
        a: wp.array[float]

    @wp.struct
    class Mixed:
        x: float
        a: wp.array[float]
        k: int
        b: wp.array[wp.vec3]

    @wp.struct
    class Inner:
        a: wp.array[float]
        s: float

    @wp.struct
    class Outer:
        i: Inner
        t: int
        c: wp.array[int]

    KEEP = []

    def mk_only(dev, n, base):
        items = []
        for e in range(n):
            s = OnlyArr()
            s.a = wp.array(np.arange(e + 1, dtype=np.float32) + base + 10 * e, device=dev)
            items.append(s)
        KEEP.append(items)
        return wp.array(items, dtype=OnlyArr, device=dev)

    def mk_mixed(dev, n, base, shape=None):
        items = []
        for e in range(n):
            s = Mixed()
            s.x = base + e + 0.5
            s.k = int(base) + 7 * e
            s.a = wp.array(np.arange(e + 2, dtype=np.float32) + base + 10 * e, device=dev)
            s.b = wp.array(np.full((e + 1, 3), base + e, dtype=np.float32), dtype=wp.vec3, device=dev)
            items.append(s)
        KEEP.append(items)
        arr = wp.array(items, dtype=Mixed, device=dev)
        return arr.reshape(shape) if shape else arr

    def mk_outer(dev, n, base):
        items = []
        for e in range(n):
            s = Outer()
            s.i.a = wp.array(np.arange(e + 1, dtype=np.float32) + base + 10 * e, device=dev)
            s.i.s = base + e
            s.t = int(base) + e
            s.c = wp.array(np.arange(e + 3, dtype=np.int32) + int(base), device=dev)
            items.append(s)
        KEEP.append(items)
        return wp.array(items, dtype=Outer, device=dev)

    @wp.kernel
    def read_only(d: wp.array[OnlyArr], out: wp.array2d[float]):
        i = wp.tid()
        s = d[i]
        out[i, 0] = float(s.a.shape[0])
        if s.a.shape[0] > 0:
            out[i, 1] = s.a[s.a.shape[0] - 1]

    @wp.kernel
    def read_mixed(d: wp.array[Mixed], out: wp.array2d[float]):
        i = wp.tid()
        s = d[i]
        out[i, 0] = s.x
        out[i, 1] = float(s.k)
        out[i, 2] = float(s.a.shape[0])
        out[i, 3] = float(s.b.shape[0])
        if s.a.shape[0] > 0:
            out[i, 4] = s.a[s.a.shape[0] - 1]
        if s.b.shape[0] > 0:
            out[i, 5] = s.b[0][1]

    def read(d, dev):
        o = wp.zeros((d.shape[0], 6), dtype=float, device=dev)
        wp.launch(read_mixed if d.dtype is Mixed else read_only, d.shape[0], [d, o], device=dev)
        return o.numpy()

    def compare(run):
        res = {dev: run(dev) for dev in ("cpu", "metal:0")}
        for a, b in zip(res["cpu"], res["metal:0"]):
            np.testing.assert_array_equal(np.asarray(b), np.asarray(a))
        return res

    def expect_raise(fn, text):
        from warp._src.codegen_metal import MetalCodegenError

        try:
            fn()
        except MetalCodegenError as e:
            assert text in str(e), e
        else:
            raise AssertionError(f"expected a MetalCodegenError containing {text!r}")
    """
)


def _run(test_case, body: str, timeout: int = 300):
    _run_with_metal_enabled(test_case, _PRELUDE + textwrap.dedent(body), timeout=timeout)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR5StructCopy(unittest.TestCase):
    def test_element_store_copies_array_members(self):
        # ``data[i] = src[j]`` and friends: the array members must be copied
        # (they were dropped, and a struct of only arrays stored nothing).
        _run(
            self,
            """
            @wp.kernel
            def same(d: wp.array[OnlyArr]):
                d[0] = d[2]

            @wp.kernel
            def other(d: wp.array[OnlyArr], src: wp.array[OnlyArr]):
                i = wp.tid()
                d[i] = src[2 - i]

            @wp.kernel
            def local(d: wp.array[OnlyArr], src: wp.array[OnlyArr]):
                i = wp.tid()
                s = src[i]
                d[2 - i] = s

            @wp.kernel
            def default(d: wp.array[OnlyArr]):
                d[1] = OnlyArr()

            @wp.kernel
            def mixed(d: wp.array[Mixed], src: wp.array[Mixed]):
                i = wp.tid()
                d[i] = src[2 - i]

            @wp.kernel
            def mixed_mod(d: wp.array[Mixed], src: wp.array[Mixed]):
                i = wp.tid()
                s = src[i]
                s.x = 99.0
                d[(i + 1) % 3] = s

            @wp.kernel
            def swap(d: wp.array[Mixed]):
                a = d[0]
                b = d[1]
                d[0] = b
                d[1] = a

            @wp.kernel
            def neg_src(d: wp.array[Mixed], src: wp.array[Mixed], off: int):
                i = wp.tid()
                d[i] = src[i - off]

            @wp.kernel
            def from_2d(d: wp.array[Mixed], src: wp.array2d[Mixed]):
                i = wp.tid()
                d[i] = src[i % 2, 1 - i % 2]

            @wp.kernel
            def from_arg(d: wp.array[Mixed], p: Mixed):
                i = wp.tid()
                if i == 0:
                    d[i] = p
                if i == 2:
                    s = p
                    s.x = 7.0
                    d[i] = s

            def run(dev):
                out = []
                for k in (same, default):
                    d = mk_only(dev, 3, 100.0)
                    wp.launch(k, 1, [d], device=dev)
                    out.append(read(d, dev))
                for k in (other, local):
                    d, s = mk_only(dev, 3, 100.0), mk_only(dev, 3, 200.0)
                    wp.launch(k, 3, [d, s], device=dev)
                    out.append(read(d, dev))
                for k in (mixed, mixed_mod):
                    d, s = mk_mixed(dev, 3, 100.0), mk_mixed(dev, 3, 200.0)
                    wp.launch(k, 3, [d, s], device=dev)
                    out.append(read(d, dev))
                d = mk_mixed(dev, 3, 100.0)
                wp.launch(swap, 1, [d], device=dev)
                out.append(read(d, dev))
                d, s = mk_mixed(dev, 3, 100.0), mk_mixed(dev, 3, 200.0)
                wp.launch(neg_src, 3, [d, s, 1], device=dev)
                out.append(read(d, dev))
                d, s = mk_mixed(dev, 3, 100.0), mk_mixed(dev, 4, 200.0, shape=(2, 2))
                wp.launch(from_2d, 3, [d, s], device=dev)
                out.append(read(d, dev))
                d = mk_mixed(dev, 3, 100.0)
                p = Mixed()
                p.x = 9.0
                p.k = 4
                p.a = wp.array(np.arange(5, dtype=np.float32) + 0.5, device=dev)
                KEEP.append(p)
                wp.launch(from_arg, 3, [d, p], device=dev)
                out.append(read(d, dev))
                return out

            res = compare(run)
            # Element 0 of ``same`` now holds element 2's three-element array.
            np.testing.assert_array_equal(res["metal:0"][0][0, :2], [3.0, 122.0])
            """,
        )

    def test_element_store_through_where_loops_and_funcs(self):
        _run(
            self,
            """
            @wp.kernel
            def cond(d: wp.array[Mixed], src: wp.array[Mixed], flag: int):
                i = wp.tid()
                s = Mixed()
                if i == flag:
                    s = src[0]
                d[i] = s

            @wp.kernel
            def loop_pick(d: wp.array[Mixed], src: wp.array[Mixed], n: int):
                i = wp.tid()
                s = Mixed()
                for j in range(n):
                    if j == (i + 1) % n:
                        s = src[j]
                d[i] = s

            @wp.func
            def get(src: wp.array[Mixed], j: int):
                return src[j]

            @wp.func
            def pick(src: wp.array[Mixed], j: int, c: bool):
                if c:
                    return src[j]
                return Mixed()

            @wp.func
            def put(d: wp.array[Mixed], src: wp.array[Mixed], i: int):
                d[i] = src[2 - i]

            @wp.kernel
            def via_funcs(d: wp.array[Mixed], e: wp.array[Mixed], f: wp.array[Mixed], src: wp.array[Mixed]):
                i = wp.tid()
                d[i] = get(src, 2 - i)
                e[i] = pick(src, 2 - i, i != 1)
                put(f, src, i)

            @wp.kernel
            def read_const(d: wp.array[Mixed], out: wp.array[float]):
                out[0] = float(d[1].a.shape[0])
                out[1] = d[1].a[0]

            def run(dev):
                out = []
                d, s = mk_mixed(dev, 3, 100.0), mk_mixed(dev, 3, 200.0)
                wp.launch(cond, 3, [d, s, 1], device=dev)
                out.append(read(d, dev))
                d = mk_mixed(dev, 3, 100.0)
                wp.launch(loop_pick, 3, [d, s, 3], device=dev)
                out.append(read(d, dev))
                d, e, f = (mk_mixed(dev, 3, 100.0) for _ in range(3))
                wp.launch(via_funcs, 3, [d, e, f, s], device=dev)
                out += [read(d, dev), read(e, dev), read(f, dev)]
                o = wp.zeros(2, dtype=float, device=dev)
                wp.launch(read_const, 1, [d, o], device=dev)
                out.append(o.numpy())
                return out

            compare(run)
            """,
        )

    def test_nested_struct_element_store_copies_records(self):
        # Array members of nested structs: compare the stored records with
        # the source's on each device (reading nested array members through
        # an element is not supported on Metal).
        _run(
            self,
            """
            @wp.kernel
            def cp(d: wp.array[Outer], src: wp.array[Outer]):
                i = wp.tid()
                d[i] = src[2 - i]

            @wp.kernel
            def sub(d: wp.array[Outer], src: wp.array[Outer]):
                i = wp.tid()
                o = d[i]
                o.i = src[2 - i].i
                d[i] = o

            def recs(d, s, perm, d0):
                dn, sn = d.numpy(), s.numpy()
                return np.array(
                    [
                        [
                            float(dn[e]["i"]["a"].tobytes() == sn[perm[e]]["i"]["a"].tobytes()),
                            float(dn[e]["c"].tobytes() == sn[perm[e]]["c"].tobytes()),
                            float(dn[e]["c"].tobytes() == d0[e]["c"].tobytes()),
                            float(dn[e]["i"]["s"]),
                            float(dn[e]["t"]),
                        ]
                        for e in range(3)
                    ]
                )

            def run(dev):
                out = []
                for k in (cp, sub):
                    d, s = mk_outer(dev, 3, 100.0), mk_outer(dev, 3, 200.0)
                    d0 = d.numpy().copy()
                    wp.launch(k, 3, [d, s], device=dev)
                    out.append(recs(d, s, [2, 1, 0], d0))
                return out

            res = compare(run)
            np.testing.assert_array_equal(res["metal:0"][0][:, 0], [1.0, 1.0, 1.0])
            np.testing.assert_array_equal(res["metal:0"][1][:, 2], [1.0, 1.0, 1.0])
            """,
        )

    def test_record_words_copied_bit_exact(self):
        # Record words travel through float32 storage and ``uint`` locals;
        # NaN-like bit patterns (pointer halves) must not be canonicalised.
        _run(
            self,
            """
            @wp.kernel
            def cp(d: wp.array[Mixed], src: wp.array[Mixed]):
                i = wp.tid()
                s = src[i]
                t = s
                d[2 - i] = t

            pats = [0x7FA0000100000001, 0xFFA00001FF800001, 0x7F8000017FC00001]
            z = wp.zeros(3, dtype=Mixed, device="cpu").numpy()
            for e in range(3):
                z[e]["a"]["data"] = pats[e]
                z[e]["a"]["grad"] = pats[(e + 1) % 3]
                z[e]["a"]["shape"][:] = [0x7F800001, -1, -8388607, 5]
                z[e]["b"]["data"] = pats[(e + 2) % 3]
                z[e]["x"] = np.frombuffer(np.uint32(0x7FA00001).tobytes(), np.float32)[0]
            for dev in ("cpu", "metal:0"):
                src = wp.array(z, dtype=Mixed, device=dev)
                d = wp.zeros(3, dtype=Mixed, device=dev)
                wp.launch(cp, 3, [d, src], device=dev)
                out = d.numpy()
                for e in range(3):
                    for f in ("x", "a", "k", "b"):
                        assert out[2 - e][f].tobytes() == z[e][f].tobytes(), (dev, e, f)
            """,
        )

    def test_element_store_in_graph_capture(self):
        # The copy is replayed against the records as they are at replay time;
        # a graph that also reads the stored array's members is rejected.
        _run(
            self,
            """
            @wp.kernel
            def cp(d: wp.array[Mixed], src: wp.array[Mixed]):
                i = wp.tid()
                d[i] = src[2 - i]

            def run(dev):
                d, s, s2 = mk_mixed(dev, 3, 100.0), mk_mixed(dev, 3, 200.0), mk_mixed(dev, 3, 300.0)
                if dev == "cpu":
                    wp.launch(cp, 3, [d, s], device=dev)
                    r1 = read(d, dev)
                    s.assign(s2)
                    wp.launch(cp, 3, [d, s], device=dev)
                    return [r1, read(d, dev)]
                with wp.ScopedCapture(device=dev) as cap:
                    wp.launch(cp, 3, [d, s], device=dev)
                wp.capture_launch(cap.graph)
                r1 = read(d, dev)
                s.assign(s2)
                wp.capture_launch(cap.graph)
                return [r1, read(d, dev)]

            compare(run)

            from warp._src.metal_dispatch import MetalDispatchError

            d, s = mk_mixed("metal:0", 3, 100.0), mk_mixed("metal:0", 3, 200.0)
            o = wp.zeros((3, 6), dtype=float, device="metal:0")
            try:
                with wp.ScopedCapture(device="metal:0") as cap:
                    wp.launch(cp, 3, [d, s], device="metal:0")
                    wp.launch(read_mixed, 3, [d, o], device="metal:0")
            except MetalDispatchError as e:
                assert "same graph writes" in str(e), e
            else:
                raise AssertionError("a graph storing and reading struct-array members was not rejected")
            """,
        )

    def test_read_modify_write_with_member_reads(self):
        # ``s = d[i]; s.k = len(s.a); d[i] = s`` stores the records back
        # unchanged, so reading the members through the launch-time tables
        # is still right; a store of another element's records is not.
        _run(
            self,
            """
            @wp.kernel
            def rmw(d: wp.array[Mixed]):
                i = wp.tid()
                s = d[i]
                s.k = s.a.shape[0] + s.b.shape[0] * 10
                s.x = s.a[0]
                d[i] = s

            @wp.kernel
            def shifted(d: wp.array[Mixed], out: wp.array[float]):
                i = wp.tid()
                s = d[i]
                out[i] = float(s.a.shape[0])
                d[(i + 1) % 3] = s

            def run(dev):
                d = mk_mixed(dev, 3, 100.0)
                wp.launch(rmw, 3, [d], device=dev)
                return [read(d, dev)]

            compare(run)
            dev = "metal:0"
            expect_raise(
                lambda: wp.launch(shifted, 3, [mk_mixed(dev, 3, 1.0), wp.zeros(3, dtype=float, device=dev)], device=dev),
                "before the launch",
            )
            """,
        )

    def test_unsupported_element_stores_raise(self):
        _run(
            self,
            """
            @wp.kernel
            def store_then_read(d: wp.array[Mixed], src: wp.array[Mixed], out: wp.array[float]):
                i = wp.tid()
                d[i] = src[2 - i]
                s = d[i]
                out[i] = float(s.a.shape[0])

            @wp.kernel
            def from_kernel_array(d: wp.array[OnlyArr], x: wp.array[float]):
                s = OnlyArr()
                s.a = x
                d[0] = s

            @wp.kernel
            def from_copied_local(d: wp.array[Mixed], src: wp.array[Mixed], x: wp.array[float]):
                i = wp.tid()
                s = src[i]
                t = s
                t.a = x
                d[i] = t

            @wp.kernel
            def from_ctor(d: wp.array[Mixed], x: wp.array[float], y: wp.array[wp.vec3]):
                i = wp.tid()
                d[i] = Mixed(2.0, x, 3, y)

            dev = "metal:0"
            d, s = mk_mixed(dev, 3, 100.0), mk_mixed(dev, 3, 200.0)
            x = wp.zeros(4, dtype=float, device=dev)
            y = wp.zeros(4, dtype=wp.vec3, device=dev)
            expect_raise(lambda: wp.launch(from_ctor, 3, [d, x, y], device=dev), "set from an array")
            expect_raise(lambda: wp.launch(store_then_read, 3, [d, s, x], device=dev), "before the launch")
            expect_raise(
                lambda: wp.launch(from_kernel_array, 1, [mk_only(dev, 3, 1.0), x], device=dev), "set from an array"
            )
            expect_raise(lambda: wp.launch(from_copied_local, 3, [d, s, x], device=dev), "set from an array")
            """,
        )

    def test_negative_dynamic_element_index_reads_array_members(self):
        # ``s = data[i - off]`` with a negative runtime index reads the last
        # element's array members, as on CPU (the pointer-table row index
        # was not wrapped).
        _run(
            self,
            """
            @wp.kernel
            def k(d: wp.array[Mixed], off: int, out: wp.array2d[float]):
                i = wp.tid()
                s = d[i - off]
                out[i, 0] = s.x
                out[i, 1] = float(s.a.shape[0])
                if s.a.shape[0] > 0:
                    out[i, 2] = s.a[0]

            def run(dev):
                d = mk_mixed(dev, 3, 100.0)
                o = wp.zeros((3, 3), dtype=float, device=dev)
                wp.launch(k, 3, [d, 1, o], device=dev)
                return [o.numpy()]

            res = compare(run)
            np.testing.assert_array_equal(res["metal:0"][0][0], [102.5, 4.0, 120.0])
            """,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
