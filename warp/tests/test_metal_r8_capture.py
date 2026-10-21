# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal-vs-CPU checks for graph capture of kernels that read array members of
constant struct-array elements (``s = data[0]; s.a[i]``).

The captured command binds the arrays the element references at capture time;
before every replay the graph checks that the element records still resolve to
them. Changes a CUDA graph also sees at replay (the member arrays' contents, the
element's scalar fields, records rewritten to the same arrays) must match an
eager CPU run; a record rebound to a different array must raise instead of
reading the captured one. Each case runs in a subprocess (see
``test_metal_launch``).
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_enabled

_PRELUDE = textwrap.dedent(
    """
    import numpy as np

    @wp.struct
    class Info:
        x: float
        k: int
        a: wp.array[float]
        b: wp.array[wp.vec3]
        c: wp.array2d[wp.int32]

    KEEP = []

    def mk_info(dev, base, n=5):
        s = Info()
        s.x = base + 0.5
        s.k = int(base)
        s.a = wp.array(np.arange(n, dtype=np.float32) + base, device=dev)
        s.b = wp.array(np.arange(3 * (n - 1), dtype=np.float32).reshape(n - 1, 3) + base, dtype=wp.vec3, device=dev)
        s.c = wp.array(np.arange(2 * n, dtype=np.int32).reshape(n, 2) + int(base), device=dev)
        KEEP.append(s)
        return s

    @wp.kernel
    def read0(d: wp.array[Info], out: wp.array2d[float]):
        i = wp.tid()
        s = d[0]
        out[i, 0] = s.x + float(s.k)
        out[i, 1] = float(s.a.shape[0]) + 10.0 * float(s.b.shape[0]) + 100.0 * float(s.c.shape[1])
        if i < s.a.shape[0]:
            out[i, 2] = s.a[i]
        if i < s.b.shape[0]:
            out[i, 3] = s.b[i][2]
        if i < s.c.shape[0]:
            out[i, 4] = float(s.c[i, 1])

    @wp.kernel
    def read1(d: wp.array[Info], out: wp.array2d[float]):
        # second constant element, read twice in one kernel
        i = wp.tid()
        e = d[1]
        f = d[1]
        if i < e.a.shape[0]:
            out[i, 5] = e.a[i] * 2.0 + f.a[0]

    @wp.kernel
    def accum0(d: wp.array[Info]):
        # writes through a constant element's member array
        i = wp.tid()
        s = d[0]
        if i < s.a.shape[0]:
            wp.atomic_add(s.a, i, 1.0)

    def run_steps(dev, steps):
        # ``steps``: callables ``(d, items) -> None`` applied between launches.
        items = [mk_info(dev, 100.0), mk_info(dev, 200.0, 4)]
        d = wp.array(items, dtype=Info, device=dev)
        out = wp.zeros((6, 6), dtype=float, device=dev)

        def body():
            out.zero_()
            wp.launch(accum0, 6, [d], device=dev)
            wp.launch(read0, 6, [d, out], device=dev)
            wp.launch(read1, 6, [d, out], device=dev)

        graph = None
        if dev != "cpu":
            with wp.ScopedCapture(device=dev) as cap:
                body()
            graph = cap.graph
        res = []
        for step in [None, *steps]:
            if step is not None:
                step(d, items)
            if graph is None:
                body()
            else:
                wp.capture_launch(graph)
            res.append(out.numpy().copy())
            res.append(items[0].a.numpy().copy())
        return res

    def compare(run):
        res = {dev: run(dev) for dev in ("cpu", "metal:0")}
        for a, b in zip(res["cpu"], res["metal:0"]):
            np.testing.assert_array_equal(np.asarray(b), np.asarray(a))
        return res
    """
)


def _run(test_case, body: str, timeout: int = 300):
    _run_with_metal_enabled(test_case, _PRELUDE + textwrap.dedent(body), timeout=timeout)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR8Capture(unittest.TestCase):
    def test_capture_replay_matches_cpu(self):
        # Capture, then replay after changes a CUDA graph sees at replay:
        # member array contents (host fill, assign, an eager kernel), the
        # element's scalar fields, and records rewritten with the same arrays.
        _run(
            self,
            """
            @wp.kernel
            def bump_b(d: wp.array[Info]):
                i = wp.tid()
                s = d[0]
                s.b[i] = s.b[i] + wp.vec3(0.0, 0.0, 3.0)

            def fill_a(d, items):
                items[0].a.fill_(7.0)

            def assign_c(d, items):
                items[0].c.assign(np.full((5, 2), 9, dtype=np.int32))

            def kernel_b(d, items):
                wp.launch(bump_b, 4, [d], device=d.device)

            def scalars(d, items):
                items[0].x = -3.25
                items[0].k = 40
                items[1].a.fill_(2.0)
                d.assign(wp.array(items, dtype=Info, device=d.device))

            def rebuild_same(d, items):
                # a new struct instance holding the same arrays
                s = Info()
                s.x, s.k, s.a, s.b, s.c = 1.0, 2, items[0].a, items[0].b, items[0].c
                items[0] = s
                d.assign(wp.array(items, dtype=Info, device=d.device))

            res = compare(lambda dev: run_steps(dev, [None, fill_a, assign_c, kernel_b, scalars, rebuild_same]))
            outs = res["metal:0"][0::2]
            assert not np.array_equal(outs[0], outs[2]) and not np.array_equal(outs[4], outs[5])
            """,
        )

    def test_rebound_record_raises_until_restored(self):
        # A record rebound to another array (CUDA would follow it) raises at
        # replay instead of reading the captured array; putting the original
        # back makes the graph usable again. Freeing a member array (its record
        # now dangles; undefined on CUDA) also raises.
        _run(
            self,
            """
            dev = "metal:0"
            items = [mk_info(dev, 100.0), mk_info(dev, 200.0, 4)]
            d = wp.array(items, dtype=Info, device=dev)
            out = wp.zeros((6, 6), dtype=float, device=dev)
            with wp.ScopedCapture(device=dev) as cap:
                wp.launch(read0, 6, [d, out], device=dev)
                wp.launch(read1, 6, [d, out], device=dev)
            wp.capture_launch(cap.graph)
            ref = out.numpy().copy()

            def expect_rebind_error():
                try:
                    wp.capture_launch(cap.graph)
                except RuntimeError as e:
                    assert "references a different array than when the graph was captured" in str(e), e
                else:
                    raise AssertionError("replay after rebinding did not raise")

            for field, elem in (("a", 0), ("b", 0), ("c", 0), ("a", 1)):
                orig = items[elem]
                s = Info()
                s.x, s.k, s.a, s.b, s.c = orig.x, orig.k, orig.a, orig.b, orig.c
                fresh = mk_info(dev, 500.0, 5 if elem == 0 else 4)
                setattr(s, field, getattr(fresh, field))
                items[elem] = s
                d.assign(wp.array(items, dtype=Info, device=dev))
                expect_rebind_error()
                items[elem] = orig
                d.assign(wp.array(items, dtype=Info, device=dev))
                out.zero_()
                wp.capture_launch(cap.graph)
                np.testing.assert_array_equal(out.numpy(), ref)

            # a view of the same buffer at another offset is a different array too
            s = Info()
            s.x, s.k, s.a, s.b, s.c = items[0].x, items[0].k, items[0].a[1:], items[0].b, items[0].c
            d.assign(wp.array([s, items[1]], dtype=Info, device=dev))
            expect_rebind_error()

            # free the member array the record still points at
            d.assign(wp.array(items, dtype=Info, device=dev))
            wp.capture_launch(cap.graph)
            wp.synchronize_device(dev)
            KEEP.clear()
            items[1].a = None
            items[1] = None
            import gc

            gc.collect()
            expect_rebind_error()
            """,
        )

    def test_dynamic_index_rebind_follows_record(self):
        # Pointer tables (runtime element index) are rebuilt before every
        # replay, so a rebound record is followed like on CUDA.
        _run(
            self,
            """
            @wp.kernel
            def readi(d: wp.array[Info], out: wp.array2d[float]):
                i = wp.tid()
                s = d[i]
                out[i, 0] = float(s.a.shape[0])
                out[i, 1] = s.a[s.a.shape[0] - 1] + s.x

            def run(dev):
                items = [mk_info(dev, 100.0), mk_info(dev, 200.0, 4)]
                d = wp.array(items, dtype=Info, device=dev)
                out = wp.zeros((2, 2), dtype=float, device=dev)
                if dev == "cpu":
                    go = lambda: wp.launch(readi, 2, [d, out], device=dev)
                else:
                    with wp.ScopedCapture(device=dev) as cap:
                        wp.launch(readi, 2, [d, out], device=dev)
                    go = lambda: wp.capture_launch(cap.graph)
                go()
                r = [out.numpy().copy()]
                items[1] = mk_info(dev, 300.0, 7)
                d.assign(wp.array(items, dtype=Info, device=dev))
                go()
                r.append(out.numpy().copy())
                return r

            compare(run)
            """,
        )

    def test_graph_writing_the_struct_array_is_rejected(self):
        # The pre-replay check reads the records on the host, so it cannot see
        # writes the graph itself makes to them: rejected at capture end.
        _run(
            self,
            """
            from warp._src.metal_dispatch import MetalDispatchError

            @wp.kernel
            def store(d: wp.array[Info], src: wp.array[Info]):
                d[0] = src[0]

            dev = "metal:0"
            d = wp.array([mk_info(dev, 100.0), mk_info(dev, 200.0, 4)], dtype=Info, device=dev)
            src = wp.array([mk_info(dev, 300.0)], dtype=Info, device=dev)
            out = wp.zeros((6, 6), dtype=float, device=dev)
            for writer in (
                lambda: wp.launch(store, 1, [d, src], device=dev),
                lambda: wp.copy(d, src, count=1),
            ):
                try:
                    with wp.ScopedCapture(device=dev) as cap:
                        writer()
                        wp.launch(read0, 6, [d, out], device=dev)
                except MetalDispatchError as e:
                    assert "same graph writes" in str(e), e
                else:
                    raise AssertionError("capture of a graph writing the struct array it reads did not raise")
            # the reads alone still capture
            with wp.ScopedCapture(device=dev) as cap:
                wp.launch(read0, 6, [d, out], device=dev)
            wp.capture_launch(cap.graph)
            assert out.numpy()[0, 2] == 100.0
            """,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
