# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the experimental Metal backend on macOS / Apple Silicon.

Step 3a verifies device discovery; step 3b verifies allocator + array
round-trip. Kernel launch is not yet wired up.

The flag ``warp.config.enable_metal`` must be set before :func:`warp.init`,
which means it can't be toggled inside a single test process. Tests that
need the flag set spawn a subprocess and use ``unittest.TestCase``-style
assertions inside; tests that only need the device disabled run in-process.
"""

import platform
import subprocess
import sys
import textwrap
import unittest

import warp as wp


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _has_mlx() -> bool:
    try:
        import mlx.core as mx  # noqa: PLC0415

        return mx.metal.is_available()
    except Exception:
        return False


def _run_with_metal_enabled(snippet: str, timeout: int = 30):
    """Run a Python snippet with ``enable_metal=True`` set before init.

    Returns (returncode, stdout, stderr).
    """
    code = "import warp as wp\nwp.config.enable_metal = True\nwp.init()\n" + snippet
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def _run_assertion_subprocess(test_case, snippet: str, timeout: int = 30):
    """Run a snippet with metal enabled; fail the test if exit code is non-zero."""
    rc, out, err = _run_with_metal_enabled(snippet, timeout=timeout)
    test_case.assertEqual(
        rc,
        0,
        f"subprocess exited with {rc}\n--- stdout ---\n{out}\n--- stderr ---\n{err}",
    )


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalDevice(unittest.TestCase):
    def test_metal_not_registered_by_default(self):
        # In-process: enable_metal defaults to False, so 'metal:0' should not resolve.
        with self.assertRaises(ValueError):
            wp.get_device("metal:0")

    def test_metal_device_registered_when_enabled(self):
        rc, out, err = _run_with_metal_enabled(
            "d = wp.get_device('metal:0')\n"
            "print('alias=', d.alias)\n"
            "print('kind=', d.kind)\n"
            "print('is_metal=', d.is_metal)\n"
            "print('is_cpu=', d.is_cpu)\n"
            "print('is_cuda=', d.is_cuda)\n"
        )
        self.assertEqual(rc, 0, f"subprocess failed:\nstdout={out}\nstderr={err}")
        self.assertIn("alias= metal:0", out)
        self.assertIn("kind= metal", out)
        self.assertIn("is_metal= True", out)
        self.assertIn("is_cpu= False", out)
        self.assertIn("is_cuda= False", out)

    def test_metal_alias_without_index_resolves(self):
        rc, out, err = _run_with_metal_enabled(
            "d = wp.get_device('metal')\nprint('alias=', d.alias)\nprint('is_metal=', d.is_metal)\n"
        )
        self.assertEqual(rc, 0, f"subprocess failed:\nstdout={out}\nstderr={err}")
        self.assertIn("alias= metal:0", out)
        self.assertIn("is_metal= True", out)

    def test_metal_uva_attribute(self):
        # Apple Silicon has unified memory; we expect is_uva=True so callers can
        # take the unified-memory codepath when applicable.
        rc, out, err = _run_with_metal_enabled("d = wp.get_device('metal:0')\nprint('is_uva=', d.is_uva)\n")
        self.assertEqual(rc, 0, f"subprocess failed:\nstdout={out}\nstderr={err}")
        self.assertIn("is_uva= True", out)

    def test_default_device_is_not_metal(self):
        # Even when Metal is registered, we don't make it the default — the
        # backend is incomplete and opt-in only.
        rc, out, err = _run_with_metal_enabled(
            "print('default=', wp.get_device().alias)\nprint('default_is_metal=', wp.get_device().is_metal)\n"
        )
        self.assertEqual(rc, 0, f"subprocess failed:\nstdout={out}\nstderr={err}")
        self.assertIn("default_is_metal= False", out)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalAllocator(unittest.TestCase):
    """Step 3b — MLX-backed allocator and host-readable array round-trip."""

    def test_empty_allocates_with_nonzero_ptr(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            a = wp.empty(64, dtype=wp.float32, device='metal:0')
            assert a.ptr != 0, f'Expected non-zero ptr, got {a.ptr}'
            assert a.size == 64, f'Expected size 64, got {a.size}'
            assert a.device.is_metal
            """
        )
        _run_assertion_subprocess(self, snippet)

    def test_zeros_returns_zeros(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np
            a = wp.zeros(128, dtype=wp.float32, device='metal:0')
            n = a.numpy()
            assert n.shape == (128,), n.shape
            assert (n == 0).all(), f'Expected all zeros, got non-zero count {(n != 0).sum()}'
            """
        )
        _run_assertion_subprocess(self, snippet)

    def test_round_trip_float32(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np
            rng = np.random.default_rng(seed=0xC0FFEE)
            ref = rng.standard_normal(1024).astype(np.float32)
            a = wp.array(ref, dtype=wp.float32, device='metal:0')
            got = a.numpy()
            np.testing.assert_array_equal(got, ref)
            """
        )
        _run_assertion_subprocess(self, snippet)

    def test_round_trip_int32(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np
            rng = np.random.default_rng(seed=42)
            ref = rng.integers(-100000, 100000, size=512, dtype=np.int32)
            a = wp.array(ref, dtype=wp.int32, device='metal:0')
            np.testing.assert_array_equal(a.numpy(), ref)
            """
        )
        _run_assertion_subprocess(self, snippet)

    def test_round_trip_float64(self):
        # Storage round-trip must work for fp64 even though MLX has no fp64 GPU
        # ops — the allocator is dtype-agnostic at the byte level.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np
            rng = np.random.default_rng(seed=7)
            ref = rng.standard_normal(256).astype(np.float64)
            a = wp.array(ref, dtype=wp.float64, device='metal:0')
            np.testing.assert_array_equal(a.numpy(), ref)
            """
        )
        _run_assertion_subprocess(self, snippet)

    def test_round_trip_2d(self):
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np
            rng = np.random.default_rng(seed=11)
            ref = rng.standard_normal((32, 48)).astype(np.float32)
            a = wp.array(ref, dtype=wp.float32, device='metal:0')
            assert a.shape == (32, 48), a.shape
            np.testing.assert_array_equal(a.numpy(), ref)
            """
        )
        _run_assertion_subprocess(self, snippet)

    def test_a_b_against_cpu_allocation_and_readback(self):
        # Same data routed through CPU and Metal should yield identical numpy
        # arrays — the canonical CPU-vs-Metal verification this step targets.
        snippet = textwrap.dedent(
            """
            import warp as wp
            import numpy as np
            rng = np.random.default_rng(seed=2026)
            ref = rng.standard_normal(2048).astype(np.float32)
            a_cpu = wp.array(ref, dtype=wp.float32, device='cpu').numpy()
            a_metal = wp.array(ref, dtype=wp.float32, device='metal:0').numpy()
            np.testing.assert_array_equal(a_cpu, a_metal)
            np.testing.assert_array_equal(a_metal, ref)
            """
        )
        _run_assertion_subprocess(self, snippet)

    def test_buffer_registry_releases_on_del(self):
        # Allocations on Metal stash the underlying mx.array in a module-level
        # registry; deleting the wp.array must remove it so the buffer can be
        # released.
        snippet = textwrap.dedent(
            """
            import warp as wp
            from warp._src.context import _metal_buffer_registry
            initial = len(_metal_buffer_registry)
            a = wp.empty(1024, dtype=wp.float32, device='metal:0')
            assert len(_metal_buffer_registry) == initial + 1, (initial, len(_metal_buffer_registry))
            ptr = a.ptr
            assert ptr in _metal_buffer_registry
            del a
            import gc; gc.collect()
            assert ptr not in _metal_buffer_registry, 'buffer not released after del'
            assert len(_metal_buffer_registry) == initial
            """
        )
        _run_assertion_subprocess(self, snippet)


class TestMetalDisabled(unittest.TestCase):
    """Sanity check: pre-existing CPU/CUDA paths still work with the new ``kind`` plumbing."""

    def test_cpu_device_kind(self):
        d = wp.get_device("cpu")
        self.assertTrue(d.is_cpu)
        self.assertFalse(d.is_cuda)
        self.assertFalse(d.is_metal)
        self.assertEqual(d.kind, "cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
