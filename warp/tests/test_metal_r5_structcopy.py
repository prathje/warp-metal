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
