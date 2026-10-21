# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal translation of rolled constant-range loops, old and new IR shapes.

Warp <= 1.18 lowers a constant ``range()`` longer than ``max_unroll`` through a
``wp::range_t`` (``iter_cmp`` / ``iter_next`` / ``iter_reverse``). Upstream
Warp main (aa0c70e7b, GH-1954) emits a plain ``int32`` counter instead:
``if (var_X >= STOP) goto end_for_K; var_Y = var_X; var_X += STEP;`` and a
reverse pass that carries its own bounds. The translator folds both into a
native MSL ``for``.

- ``TestMetalLoopIRFixtures`` checks the new shape against upstream's IR,
  frozen verbatim in ``aux_test_metal_loopir_ir`` (no upstream tree needed).
- ``TestMetalLoopIRDevice`` compares CPU and Metal forward results and Tape
  gradients for the same kernels on whatever Warp is running (old IR shape on
  1.18; the new shape on a tree that carries GH-1954).
"""

import re
import textwrap
import unittest

import warp as wp
from warp._src.codegen_metal_ast import (
    For,
    ForIterCmp,
    Goto,
    If,
    Label,
    MetalASTParseError,
    While,
    _DoWhileZero,
    assemble_adjoint_body,
    canonicalize_constant_loops,
    emit,
    fold,
    parse,
)
from warp.tests.aux_test_metal_loopir_ir import UPSTREAM_IR
from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_enabled


def _walk(nodes):
    for n in nodes:
        yield n
        if isinstance(n, (If, For, While, _DoWhileZero)):
            yield from _walk(n.body)


def _for_bounds(nodes):
    return [(int(n.start), int(n.stop), int(n.step)) for n in _walk(nodes) if isinstance(n, For)]


def _assert_structured(test, nodes):
    leftovers = [n.raw for n in _walk(nodes) if isinstance(n, (Goto, Label, ForIterCmp))]
    test.assertEqual(leftovers, [], "goto / label / iter_cmp left after fold")


# Python ``range`` arguments of the rolled loops in each fixture, in IR order.
_FORWARD_RANGES = {
    "k_long": [(0, 32, 1), (60, 2, -3), (1, 40, 2), (0, 17, 1), (0, 18, 1)],
    "f_loop": [(0, 20, 1)],
    "f_ret": [(0, 24, 1)],
}


class TestMetalLoopIRFixtures(unittest.TestCase):
    """Upstream (GH-1954) IR folds to the right native loops."""

    def test_forward_folds_to_native_for(self):
        for name, ranges in _FORWARD_RANGES.items():
            with self.subTest(name=name):
                nodes, _ = fold(parse(UPSTREAM_IR[name]["forward"]))
                _assert_structured(self, nodes)
                self.assertEqual(_for_bounds(nodes), ranges)

    def test_reverse_visits_forward_iterations_backwards(self):
        for name, ranges in _FORWARD_RANGES.items():
            with self.subTest(name=name):
                ir = UPSTREAM_IR[name]
                nodes, _ = assemble_adjoint_body(ir["replay"], ir["reverse"], {}, name)
                _assert_structured(self, nodes)
                # The reverse sweep runs the loops in reverse program order
                # (nested loops stay nested, so outer-before-inner holds).
                rev = _for_bounds(nodes)
                fwd = list(ranges)
                if name == "k_long":
                    fwd = [fwd[3], fwd[4], fwd[2], fwd[1], fwd[0]]
                self.assertEqual(len(rev), len(fwd))
                for r, f in zip(rev, fwd, strict=True):
                    self.assertEqual(list(range(*r)), list(reversed(range(*f))), (r, f))

    def test_counter_stays_declared(self):
        # The replay assigns the counter on its own (``var_3 = 0;``), so the
        # fold must not suppress its declaration as it does a range object.
        ir = UPSTREAM_IR["k_long"]
        _, skip = assemble_adjoint_body(ir["replay"], ir["reverse"], {}, "k_long")
        self.assertIn("var_3 = 0;", ir["replay"])
        self.assertNotIn("3", skip)
        _, skip = fold(parse(ir["forward"]))
        self.assertNotIn("3", skip)

    def test_emitted_loop_text(self):
        nodes, _ = fold(parse(UPSTREAM_IR["k_long"]["forward"]))
        text = "\n".join(emit(nodes))
        self.assertIn("for (var_4 = 0; var_4 < 32; ++var_4) {", text)
        self.assertIn("for (var_13 = 60; var_13 < 2; var_13 += -3) {", text)  # ``>`` fixed later
        self.assertNotIn("goto", text)

    def test_dynamic_loop_shape_unchanged(self):
        # ``continue`` keeps the ``wp::range_t`` form upstream too.
        lines = UPSTREAM_IR["k_cont"]["forward"]
        self.assertTrue(any("iter_cmp" in ln for ln in lines))
        self.assertEqual(canonicalize_constant_loops(lines), list(lines))
        nodes, _ = fold(parse(lines))
        _assert_structured(self, nodes)
        loops = [n for n in _walk(nodes) if isinstance(n, For)]
        self.assertEqual(len(loops), 1)
        self.assertTrue(loops[0].stop.startswith("var_"))  # bounds are constant locals, not literals

    def test_running_warp_ir(self):
        # Whatever shape the running Warp emits: the ``wp::range_t`` form
        # (<= 1.18) passes through untouched, the rolled-counter form is
        # rewritten so that no counter test survives.
        @wp.kernel
        def long_loops(a: wp.array(dtype=float)):
            s = float(0.0)
            for i in range(40):
                s += float(i)
            for j in range(50, 3, -2):
                s -= float(j)
            a[0] = s

        long_loops.adj.build(builder=None, default_builder_options={"enable_backward": True})
        block = long_loops.adj.blocks[0]
        replay, reverse = list(block.body_replay), list(reversed(block.body_reverse))
        for lines in (block.body_forward, replay, reverse):
            out = canonicalize_constant_loops(lines)
            if any("iter_cmp(" in ln for ln in lines):
                self.assertEqual(out, list(lines))
            else:
                self.assertFalse([ln for ln in out if re.search(r"if \(var_\w+ [<>]=? -?\d+\) goto end_for", ln)])
        nodes, _ = fold(parse(block.body_forward))
        _assert_structured(self, nodes)
        const_ints = {
            str(v.label): v.constant
            for v in long_loops.adj.variables
            if isinstance(v.constant, int) and not isinstance(v.constant, bool)
        }
        nodes, _ = assemble_adjoint_body(replay, reverse, const_ints, "long_loops")
        _assert_structured(self, nodes)

    def test_inconsistent_opener_raises(self):
        lines = [
            "var_3 = 0;",
            "start_for_0:;",
            "    if (var_3 >= 32) goto end_for_0;",
            "    var_4 = var_3;",
            "    var_3 -= 1;",
            "    goto start_for_0;",
            "end_for_0:;",
        ]
        with self.assertRaises(MetalASTParseError):
            parse(lines)


_LOOP_SNIPPET = textwrap.dedent(
    """
    import numpy as np
    from warp._src import codegen_metal as cm

    @wp.func
    def f_loop(x: float):
        s = float(0.0)
        for i in range(20):
            s += x * float(i) * 0.01
        return s

    @wp.func
    def f_ret(x: float):
        s = float(0.0)
        for i in range(24):
            if x * float(i) > 50.0:
                return s
            s += x * 0.1
        return s

    @wp.kernel
    def k_long(a: wp.array(dtype=float), out: wp.array(dtype=float)):
        tid = wp.tid()
        s = float(0.0)
        for i in range(32):
            s += a[tid] * float(i) * 0.01
        for j in range(60, 2, -3):
            s += wp.sin(a[tid] * float(j) * 0.01)
        if tid > 2:
            for k in range(1, 40, 2):
                s = s * 0.9 + a[tid] * 0.1
        for p in range(17):
            for q in range(18):
                s += a[tid] * float(p - q) * 0.001
        s += f_loop(a[tid])
        s += f_ret(a[tid])
        out[tid] = s * s

    @wp.kernel
    def k_vec(a: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        tid = wp.tid()
        v = a[tid]
        acc = wp.vec3(0.0)
        for i in range(48):
            acc = acc * 0.95 + v * float(i % 5)
        out[tid] = wp.dot(acc, v)

    rng = np.random.default_rng(0)
    for k, data in (
        (k_long, rng.uniform(0.5, 2.0, 64).astype(np.float32)),
        (k_vec, rng.uniform(-1.0, 1.0, (64, 3)).astype(np.float32)),
    ):
        res = {}
        for dev in ("cpu", "metal:0"):
            dtype = wp.vec3 if data.ndim == 2 else float
            a = wp.array(data, dtype=dtype, device=dev, requires_grad=True)
            out = wp.zeros(64, dtype=float, device=dev, requires_grad=True)
            tape = wp.Tape()
            with tape:
                wp.launch(k, dim=64, inputs=[a], outputs=[out], device=dev)
            tape.backward(grads={out: wp.ones(64, dtype=float, device=dev)})
            res[dev] = (out.numpy(), a.grad.numpy())
        for idx in (0, 1):
            c, m = res["cpu"][idx], res["metal:0"][idx]
            assert np.abs(c).max() > 0, (k.key, idx)
            np.testing.assert_allclose(m, c, rtol=1e-4, atol=1e-4, err_msg=f"{k.key} {'grad' if idx else 'fwd'}")
        for adjoint in (False, True):
            src = cm.generate_msl_kernel(k, adjoint=adjoint, grad_args=("a", "out") if adjoint else ()).source
            for bad in ("goto ", "iter_cmp", "iter_next", "iter_reverse", "range_t"):
                assert bad not in src, (k.key, adjoint, bad)
    print("loopir ok")
    """
)


_REASSIGN_SNIPPET = textwrap.dedent(
    """
    import numpy as np

    # Re-binding the loop target in the body (``i = i + 3``) makes Warp copy
    # the new value back into the loop variable at the end of the body. A C
    # loop driven by that variable skipped iterations forward and never
    # terminated in reverse (the copy outran the decrement): a GPU hang.
    @wp.kernel
    def k_const(a: wp.array(dtype=float), out: wp.array(dtype=float)):
        t = wp.tid()
        s = float(0.0)
        for i in range(20):
            i = i + 3
            s += a[t] * float(i)
        out[t] = s + float(i)

    @wp.kernel
    def k_dyn(a: wp.array(dtype=float), n: int, out: wp.array(dtype=float)):
        t = wp.tid()
        s = float(0.0)
        for i in range(n, 0, -1):
            s += a[t] * float(i)
            i += 2
            s += a[t] * float(i) * 0.5
        out[t] = s

    data = np.random.default_rng(3).uniform(-1.0, 1.0, 16).astype(np.float32)
    for k in (k_const, k_dyn):
        res = {}
        for dev in ("cpu", "metal:0"):
            a = wp.array(data, dtype=float, device=dev, requires_grad=True)
            out = wp.zeros(16, dtype=float, device=dev, requires_grad=True)
            args = [a, 9, out] if k is k_dyn else [a, out]
            tape = wp.Tape()
            with tape:
                wp.launch(k, dim=16, inputs=args, device=dev)
            tape.backward(grads={out: wp.ones(16, dtype=float, device=dev)})
            res[dev] = (out.numpy(), a.grad.numpy())
        for idx in (0, 1):
            np.testing.assert_allclose(
                res["metal:0"][idx], res["cpu"][idx], rtol=1e-5, atol=1e-5, err_msg=f"{k.key} {idx}"
            )
    print("reassign ok")
    """
)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalLoopIRDevice(unittest.TestCase):
    def test_long_constant_loops_match_cpu(self):
        _run_with_metal_enabled(self, _LOOP_SNIPPET, timeout=600)

    def test_reassigned_loop_target_matches_cpu(self):
        _run_with_metal_enabled(self, _REASSIGN_SNIPPET, timeout=300)


_BIT_COUNT_SNIPPET = textwrap.dedent(
    """
    import numpy as np

    types = (wp.int8, wp.uint8, wp.int16, wp.uint16, wp.int32, wp.uint32, wp.int64, wp.uint64)
    kernels = {}
    for T in types:

        @wp.kernel(module="unique")
        def count(x: wp.array(dtype=T), out: wp.array(dtype=wp.int32)):
            i = wp.tid()
            out[i] = wp.bit_count(x[i]) + wp.bit_count(T(5)) * 100

        kernels[T] = count

    for T in types:
        np_t = wp._src.types.warp_type_to_np_dtype[T]
        info = np.iinfo(np_t)
        vals = np.array([0, 1, 2, 3, 45, info.max, info.min, info.max - 1, info.min + 1], dtype=np_t)
        if info.min < 0:
            vals = np.concatenate([vals, np.array([-1, -2, -45], dtype=np_t)])
        res = {}
        for dev in ("cpu", "metal:0"):
            x = wp.array(vals, dtype=T, device=dev)
            out = wp.zeros(len(vals), dtype=wp.int32, device=dev)
            wp.launch(kernels[T], dim=len(vals), inputs=[x], outputs=[out], device=dev)
            res[dev] = out.numpy()
        bits = np.dtype(np_t).itemsize * 8
        expect = [bin(int(v) & ((1 << bits) - 1)).count("1") + 200 for v in vals]
        np.testing.assert_array_equal(res["cpu"], expect, err_msg=str(T))
        np.testing.assert_array_equal(res["metal:0"], res["cpu"], err_msg=str(T))
    print("bit_count ok")
    """
)

_TEXTURE_ADJ_SNIPPET = textwrap.dedent(
    """
    import numpy as np
    from warp._src.codegen_metal import MetalCodegenError

    @wp.kernel
    def s2(tex: wp.Texture2D, uv: wp.array[wp.vec2], o: wp.array[float]):
        i = wp.tid()
        o[i] = wp.texture_sample(tex, uv[i], dtype=float) * 2.0

    data = np.random.default_rng(0).standard_normal((5, 7)).astype(np.float32)
    grads = {}
    for dev in ("cpu", "metal:0"):
        tex = wp.Texture2D(data, filter_mode=wp.TextureFilterMode.LINEAR, device=dev)
        uv = wp.array(np.full((4, 2), 0.37, np.float32), dtype=wp.vec2, device=dev, requires_grad=True)
        o = wp.zeros(4, dtype=float, device=dev, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(s2, 4, [tex, uv, o], device=dev)
        try:
            tape.backward(grads={o: wp.ones(4, dtype=float, device=dev)})
            grads[dev] = uv.grad.numpy()
        except MetalCodegenError as e:
            assert dev == "metal:0" and "texture_sample" in str(e), e
            grads[dev] = None
    if np.any(grads["cpu"] != 0.0):
        # Differentiable texture_sample (Warp > 1.18, GH-1301): Metal has no
        # adjoint yet and must say so instead of returning zeros.
        assert grads["metal:0"] is None, grads
    else:
        # Non-differentiable texture_sample (Warp <= 1.18): zero on both.
        assert grads["metal:0"] is not None, "unexpected raise"
        np.testing.assert_array_equal(grads["metal:0"], grads["cpu"])
    print("texture adjoint ok")
    """
)

_POISSON_SNIPPET = textwrap.dedent(
    """
    from warp._src.codegen_metal import MetalCodegenError

    @wp.kernel
    def counts(seed: int, rate: float, out: wp.array[wp.uint32]):
        i = wp.tid()
        rng = wp.rand_init(seed, i)
        out[i] = wp.poisson(rng, rate)

    out = wp.zeros(16, dtype=wp.uint32, device="metal:0")
    try:
        wp.launch(counts, out.shape, [7, 15.0, out], device="metal:0")
        out.numpy()
    except MetalCodegenError as e:
        assert "poisson" in str(e), e
    else:
        raise AssertionError("wp.poisson ran on Metal; compare it with the CPU sampler (GH-2083) before allowing it")
    print("poisson loud")
    """
)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalUpstreamBuiltins(unittest.TestCase):
    """Built-ins whose upstream (> 1.18) form reaches the Metal translator."""

    @unittest.skipUnless(hasattr(wp, "bit_count"), "wp.bit_count needs Warp > 1.18 (GH-2071)")
    def test_bit_count_matches_cpu(self):
        _run_with_metal_enabled(self, _BIT_COUNT_SNIPPET, timeout=600)

    def test_poisson_is_loud(self):
        # No Metal sampler yet; upstream also changed the CPU one (GH-2083).
        _run_with_metal_enabled(self, _POISSON_SNIPPET, timeout=300)

    def test_texture_sample_adjoint_not_silent(self):
        _run_with_metal_enabled(self, _TEXTURE_ADJ_SNIPPET, timeout=300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
