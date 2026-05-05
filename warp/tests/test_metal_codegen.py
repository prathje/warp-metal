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
        with self.assertRaises(MetalCodegenError):
            generate_msl_kernel(_all_inputs_no_output)

    def test_float64_raises(self):
        # MSL has no native float64; this should be rejected at codegen time
        # rather than producing invalid MSL.
        with self.assertRaises(MetalCodegenError):
            generate_msl_kernel(_vector_add_fp64)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
