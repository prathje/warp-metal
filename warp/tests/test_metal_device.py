# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the experimental Metal backend on macOS / Apple Silicon.

Step 3a verifies device discovery only — allocator and kernel launch are not
yet wired up, so these tests exercise only the Device-side plumbing.

The flag ``warp.config.enable_metal`` must be set before :func:`warp.init`,
which means it can't be toggled inside a single test process. Each test that
needs the flag set spawns a subprocess.
"""

import platform
import subprocess
import sys
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
