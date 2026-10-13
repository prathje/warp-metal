# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step 3c — MSL code generation from Warp kernels.

Verifies that ``warp._src.codegen_metal.generate_msl_kernel`` produces an
MSL function body that Apple's Metal compiler accepts. The verification is
a 1-element dispatch via ``mx.fast.metal_kernel`` — sufficient to confirm
the generated source compiles. Bit-exact CPU-vs-Metal comparison comes in
step 3d when ``wp.launch`` is wired up.
"""

import platform
import sys
import unittest

import numpy as np

import warp as wp
from warp._src.codegen_metal import (
    MetalCodegenError,
    _msl_constant_str,
    generate_msl_kernel,
)


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _has_mlx() -> bool:
    try:
        import mlx.core as mx  # noqa: PLC0415

        return mx.metal.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Kernel fixtures (must live at module scope so inspect.getsourcelines works)
# ---------------------------------------------------------------------------


@wp.kernel
def _vector_add(
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    c: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]


@wp.kernel
def _vector_mul_int(
    a: wp.array(dtype=wp.int32),
    b: wp.array(dtype=wp.int32),
    c: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    c[tid] = a[tid] * b[tid]


@wp.kernel
def _scale_then_subtract(
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    c: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    c[tid] = a[tid] * 3.0 - b[tid]


# ``wp.constant`` wrapping a Warp-typed FLOAT scalar (not a Python float).
# warp.fem builds its reference-cell measure this way.
_HALF = wp.constant(wp.float32(0.5))
_QUARTER = wp.constant(wp.float32(0.25))


@wp.kernel
def _typed_float_constants(
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    c: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    c[tid] = a[tid] * _HALF + b[tid] * _QUARTER


@wp.kernel
def _all_inputs_no_output(
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    _ = a[tid] + b[tid]


@wp.kernel
def _vector_add_fp64(
    a: wp.array(dtype=wp.float64),
    b: wp.array(dtype=wp.float64),
    c: wp.array(dtype=wp.float64),
):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]


# Spin-lock fixtures for the SIMD-lane serialisation pass. ``_spin_lock_counter``
# is the shape of ``warp/examples/core/example_spin_lock.py``; the others are
# lookalikes the pass must leave alone.
@wp.func
def _spin_acquire(lock: wp.array(dtype=int)):
    while wp.atomic_cas(lock, 0, 0, 1) == 1:
        pass


@wp.func
def _spin_release(lock: wp.array(dtype=int)):
    wp.atomic_exch(lock, 0, 0)


@wp.kernel
def _spin_lock_counter(counter: wp.array(dtype=int), lock: wp.array(dtype=int)):
    _spin_acquire(lock)
    value = wp.atomic_exch(counter, 0, 0)
    counter[0] = value + 1
    _spin_release(lock)


@wp.kernel
def _spin_lock_no_release(counter: wp.array(dtype=int), lock: wp.array(dtype=int)):
    _spin_acquire(lock)
    wp.atomic_add(counter, 0, 1)


@wp.kernel
def _spin_lock_release_in_branch(counter: wp.array(dtype=int), lock: wp.array(dtype=int), flag: int):
    _spin_acquire(lock)
    wp.atomic_add(counter, 0, 1)
    if flag == 1:
        _spin_release(lock)


@wp.kernel
def _atomic_in_for_loop(counter: wp.array(dtype=int), n: int):
    for _i in range(n):
        wp.atomic_add(counter, 0, 1)


@wp.kernel
def _ticket_lock_counter(counter: wp.array(dtype=int), next_ticket: wp.array(dtype=int), serving: wp.array(dtype=int)):
    ticket = wp.atomic_add(next_ticket, 0, 1)
    while wp.atomic_add(serving, 0, 0) != ticket:
        pass
    value = wp.atomic_exch(counter, 0, 0)
    counter[0] = value + 1
    wp.atomic_add(serving, 0, 1)


# ---------------------------------------------------------------------------
# Codegen tests (no MLX needed, just AST -> MSL string)
# ---------------------------------------------------------------------------


class TestMetalCodegenStructure(unittest.TestCase):
    """Exercises the generator without executing on the GPU."""

    def test_classifies_inputs_and_outputs(self):
        artifact = generate_msl_kernel(_vector_add)
        self.assertEqual(artifact.input_names, ["a", "b"])
        self.assertEqual(artifact.output_names, ["c"])

    def test_emits_msl_types(self):
        artifact = generate_msl_kernel(_vector_add)
        # The emitted body should mention MSL primitive types and the magic
        # ``thread_position_in_grid`` builtin — a regression test against the
        # CUDA-style intrinsics being emitted unchanged.
        self.assertIn("float", artifact.source)
        self.assertIn("thread_position_in_grid", artifact.source)
        self.assertNotIn("wp::", artifact.source)
        self.assertNotIn("builtin_", artifact.source)
        self.assertNotIn("__global__", artifact.source)

    def test_collapses_address_load_chains(self):
        # The intermediate pointer locals (``var_1``, ``var_2`` in the IR) must
        # be folded into direct subscripts so MSL doesn't see address-space
        # mismatches between ``const constant`` inputs and ``device`` outputs.
        src = generate_msl_kernel(_vector_add).source
        self.assertNotIn("&a[", src)
        self.assertNotIn("(*", src)
        self.assertIn("a[var_", src)
        self.assertIn("b[var_", src)

    def test_no_output_array_raises(self):
        # Only the MLX path needs an output buffer; native dispatch runs
        # read-only kernels (e.g. ones that only check ``wp.expect_eq``).
        saved = wp.config.metal_native_dispatch
        try:
            wp.config.metal_native_dispatch = False
            with self.assertRaises(MetalCodegenError):
                generate_msl_kernel(_all_inputs_no_output)
            wp.config.metal_native_dispatch = True
            # A kernel without writes compiles to a no-op (empty source).
            self.assertEqual(generate_msl_kernel(_all_inputs_no_output).source, "")
        finally:
            wp.config.metal_native_dispatch = saved

    def test_typed_float_constants_are_not_truncated(self):
        # ``wp.float32(0.5)`` is neither a Python ``float`` nor a Python
        # ``int``, so it used to fall through to an ``int(value)`` fallback
        # and emit ``0`` — silently zeroing warp.fem's integration weights
        # with no diagnostic. Both the literal and the result must survive.
        src = generate_msl_kernel(_typed_float_constants).source
        self.assertIn("0.5f", src)
        self.assertIn("0.25f", src)

    def test_msl_constant_str_typed_scalars(self):
        for scalar_type in (wp.float16, wp.float32, wp.float64):
            with self.subTest(scalar_type=scalar_type.__name__):
                self.assertEqual(_msl_constant_str(scalar_type(0.5)), "0.5f")
        # Typed ints must keep taking the integer path.
        self.assertEqual(_msl_constant_str(wp.uint32(2)), "2")
        self.assertEqual(_msl_constant_str(wp.int32(-7)), "-7")

    def test_float64_raises(self):
        # MSL has no native float64; this should be rejected at codegen time
        # rather than producing invalid MSL.
        with self.assertRaises(MetalCodegenError):
            generate_msl_kernel(_vector_add_fp64)


class TestMetalSpinLockSerialization(unittest.TestCase):
    """Apple GPUs give no independent forward progress between the lanes of a
    SIMD group, so a lock spun on by two sibling lanes deadlocks. The codegen
    pass rewrites acquire / critical section / release spans to run one lane
    per SIMD group at a time; these tests pin what it does and does not touch.
    """

    @staticmethod
    def _generate(kernel):
        import contextlib  # noqa: PLC0415
        import io  # noqa: PLC0415

        import warnings  # noqa: PLC0415

        # Warp's logger routes warnings to stderr; capture both streams and
        # disable the warnings registry dedup so each test sees its warning.
        out = io.StringIO()
        with warnings.catch_warnings(), contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            warnings.simplefilter("always")
            src = generate_msl_kernel(kernel).source
        return src, out.getvalue()

    def test_lock_span_is_serialised_across_simd_lanes(self):
        src, printed = self._generate(_spin_lock_counter)
        self.assertNotIn("could not be serialised", printed)
        self.assertIn("simd_is_first()", src)
        self.assertIn("simd_any(wp_spin_pending_0)", src)
        # Acquire loop, critical section, and release all sit inside the
        # elected-lane branch, bracketed by device-scope fences.
        elected = src.index("simd_is_first()")
        acquire = src.index("wp_atomic_cas(&lock[", elected)
        release = src.index("wp_atomic_exch(&lock[", acquire)
        critical = src.index("atomic_store_explicit(&counter[", acquire)
        self.assertLess(critical, release)
        fences = [i for i in range(len(src)) if src.startswith("atomic_thread_fence(", i)]
        self.assertEqual(len(fences), 2)
        self.assertLess(acquire, fences[0])
        self.assertLess(fences[0], critical)
        self.assertLess(critical, fences[1])
        self.assertLess(fences[1], release)
        self.assertIn("memory_order_seq_cst", src)
        self.assertIn("wp_spin_pending_0 = false;", src[release:])

    def test_missing_release_warns_and_leaves_body(self):
        src, printed = self._generate(_spin_lock_no_release)
        self.assertNotIn("simd_is_first", src)
        self.assertIn("could not be serialised", printed)
        self.assertIn("no matching release", printed)

    def test_release_in_nested_branch_warns_and_leaves_body(self):
        src, printed = self._generate(_spin_lock_release_in_branch)
        self.assertNotIn("simd_is_first", src)
        self.assertIn("could not be serialised", printed)

    def test_for_loop_with_atomic_is_not_a_spin_lock(self):
        src, printed = self._generate(_atomic_in_for_loop)
        self.assertNotIn("simd_is_first", src)
        self.assertNotIn("could not be serialised", printed)

    def test_ticket_lock_is_left_alone(self):
        # The ticket is taken before the spin, so electing the first pending
        # lane could pick a lane whose turn never comes. Only test-and-set
        # locks (``wp.atomic_cas`` / ``wp.atomic_exch``) are rewritten.
        src, printed = self._generate(_ticket_lock_counter)
        self.assertNotIn("simd_is_first", src)
        self.assertNotIn("could not be serialised", printed)

    def test_kill_switch_disables_rewrite(self):
        saved = wp.config.metal_serialize_spin_locks
        wp.config.metal_serialize_spin_locks = False
        try:
            src, _ = self._generate(_spin_lock_counter)
        finally:
            wp.config.metal_serialize_spin_locks = saved
        self.assertNotIn("simd_is_first", src)
        self.assertIn("wp_atomic_cas(&lock[", src)


# ---------------------------------------------------------------------------
# MLX compile-launch tests (verifies generated MSL compiles)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalCodegenCompiles(unittest.TestCase):
    """Feeds the generated MSL into ``mx.fast.metal_kernel`` and runs a
    1-element dispatch — enough to confirm the source compiles and produces
    the expected element-wise result for a degenerate input.
    """

    def _run_one_element(self, kernel, np_a, np_b, expected_c, dtype):
        import mlx.core as mx  # noqa: PLC0415

        artifact = generate_msl_kernel(kernel)
        mlx_kernel = mx.fast.metal_kernel(
            name=artifact.name,
            input_names=artifact.input_names,
            output_names=artifact.output_names,
            source=artifact.source,
        )
        a_mx = mx.array(np_a)
        b_mx = mx.array(np_b)
        (out,) = mlx_kernel(
            inputs=[a_mx, b_mx],
            grid=(len(np_a), 1, 1),
            threadgroup=(len(np_a), 1, 1),
            output_shapes=[(len(np_a),)],
            output_dtypes=[dtype],
        )
        mx.eval(out)
        np.testing.assert_array_equal(np.array(out), expected_c)

    def test_vector_add_compiles_and_runs(self):
        import mlx.core as mx  # noqa: PLC0415

        self._run_one_element(
            _vector_add,
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([10.0, 20.0, 30.0], dtype=np.float32),
            np.array([11.0, 22.0, 33.0], dtype=np.float32),
            mx.float32,
        )

    def test_vector_mul_int_compiles_and_runs(self):
        import mlx.core as mx  # noqa: PLC0415

        self._run_one_element(
            _vector_mul_int,
            np.array([2, 3, 4], dtype=np.int32),
            np.array([5, 6, 7], dtype=np.int32),
            np.array([10, 18, 28], dtype=np.int32),
            mx.int32,
        )

    def test_scale_then_subtract_compiles_and_runs(self):
        import mlx.core as mx  # noqa: PLC0415

        self._run_one_element(
            _scale_then_subtract,
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([0.5, 1.5], dtype=np.float32),
            np.array([2.5, 4.5], dtype=np.float32),  # a*3 - b
            mx.float32,
        )

    def test_typed_float_constants_compile_and_run(self):
        import mlx.core as mx  # noqa: PLC0415

        self._run_one_element(
            _typed_float_constants,
            np.array([2.0, 4.0], dtype=np.float32),
            np.array([8.0, 16.0], dtype=np.float32),
            np.array([3.0, 6.0], dtype=np.float32),  # a*0.5 + b*0.25
            mx.float32,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
