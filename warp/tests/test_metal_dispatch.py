# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone tests for the native Metal dispatcher.

These exercise :class:`warp._src.metal_dispatch.MetalDispatcher` without
going through Warp's ``wp.launch`` path — the dispatcher is independently
useful and we want to know it's correct before Phase 2b wires it into
``launch_metal_kernel``.

Skipped on non-Apple-Silicon hosts; the module imports PyObjC Metal which
is only available on macOS.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np


def _can_run() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import Metal  # noqa: F401
    except ImportError:
        return False
    try:
        from warp._src.metal_dispatch import MetalDispatcher  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(_can_run(), "MetalDispatcher requires macOS + pyobjc-framework-metal")
class TestMetalDispatcher(unittest.TestCase):
    def setUp(self) -> None:
        from warp._src.metal_dispatch import MetalDispatcher

        self.d = MetalDispatcher()

    def _read_buffer(self, addr: int, nbytes: int) -> np.ndarray:
        return np.frombuffer((ctypes.c_byte * nbytes).from_address(addr), dtype=np.float32)

    def test_alloc_returns_stable_cpu_addressable_pointer(self):
        # Writes through the CPU pointer must be visible to the GPU
        # (and vice versa) thanks to shared storage + unified memory.
        N = 8
        buf, addr = self.d.alloc(N * 4)
        view = self._read_buffer(addr, N * 4)
        view[:] = np.arange(N, dtype=np.float32)
        # Re-resolve the address from the same buffer — should match.
        view2 = np.frombuffer(buf.contents().as_buffer(N * 4), dtype=np.float32)
        np.testing.assert_array_equal(view2, np.arange(N, dtype=np.float32))

    def test_compile_caches_pipeline_state(self):
        src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void noop(uint id [[thread_position_in_grid]]) {}
        """
        pso_a = self.d.compile(src, "noop")
        pso_b = self.d.compile(src, "noop")
        # Same PSO object — both library and pipeline caches hit.
        self.assertIs(pso_a, pso_b)

    def test_compile_raises_on_bad_msl(self):
        from warp._src.metal_dispatch import MetalDispatchError

        with self.assertRaises(MetalDispatchError):
            self.d.compile("this is not MSL", "nope")

    def test_dispatch_writes_in_place(self):
        # The whole point: kernel modifies *our* buffer, no fresh
        # allocation, no host memcpy.
        N = 32
        buf, addr = self.d.alloc(N * 4)
        view = self._read_buffer(addr, N * 4)
        view[:] = np.arange(N, dtype=np.float32)
        src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void add_one(device float* x [[buffer(0)]],
                            uint id [[thread_position_in_grid]]) {
            x[id] += 1.0;
        }
        """
        pso = self.d.compile(src, "add_one")
        self.d.dispatch(pso, [buf], grid=(N, 1, 1), threadgroup=(N, 1, 1))
        self.d.sync()
        np.testing.assert_array_equal(
            self._read_buffer(addr, N * 4),
            np.arange(N, dtype=np.float32) + 1.0,
        )

    def test_dispatch_batches_until_sync(self):
        # Many dispatches issued before sync — should produce
        # cumulative result. Validates the command-buffer-batching
        # path that gives us the per-launch speedup.
        N = 16
        NLAUNCH = 50
        buf, addr = self.d.alloc(N * 4)
        view = self._read_buffer(addr, N * 4)
        view[:] = 0.0
        src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void add_one(device float* x [[buffer(0)]],
                            uint id [[thread_position_in_grid]]) {
            x[id] += 1.0;
        }
        """
        pso = self.d.compile(src, "add_one")
        for _ in range(NLAUNCH):
            self.d.dispatch(pso, [buf], grid=(N, 1, 1), threadgroup=(N, 1, 1))
        self.d.sync()
        np.testing.assert_array_equal(
            self._read_buffer(addr, N * 4),
            np.full(N, NLAUNCH, dtype=np.float32),
        )

    def test_multiple_buffers_bound_in_order(self):
        # Two inputs + one output. Validates that arg-slot ordering is
        # honored by setBuffer:offset:atIndex:.
        N = 16
        buf_a, addr_a = self.d.alloc(N * 4)
        buf_b, addr_b = self.d.alloc(N * 4)
        buf_out, addr_out = self.d.alloc(N * 4)
        a = self._read_buffer(addr_a, N * 4)
        b = self._read_buffer(addr_b, N * 4)
        a[:] = np.arange(N, dtype=np.float32)
        b[:] = np.arange(N, dtype=np.float32) * 10
        src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void axpy(device const float* a [[buffer(0)]],
                         device const float* b [[buffer(1)]],
                         device float*       o [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
            o[id] = a[id] + b[id];
        }
        """
        pso = self.d.compile(src, "axpy")
        self.d.dispatch(pso, [buf_a, buf_b, buf_out], grid=(N, 1, 1), threadgroup=(N, 1, 1))
        self.d.sync()
        np.testing.assert_array_equal(
            self._read_buffer(addr_out, N * 4),
            np.arange(N, dtype=np.float32) * 11,
        )


class TestMetalGpuArchiverReaper(unittest.TestCase):
    """Bounding of Apple's ``com.apple.gpuarchiver`` cache.

    Pure filesystem logic against a synthetic cache directory, so it runs
    everywhere the module imports — no GPU needed.
    """

    def setUp(self) -> None:
        try:
            from warp._src.metal_dispatch import MetalDispatcher  # noqa: PLC0415
        except Exception as exc:
            self.skipTest(f"metal_dispatch unavailable: {exc!r}")
        self.reap = MetalDispatcher._reap_gpu_archiver_cache
        self.base = tempfile.mkdtemp(prefix="wp_reaper_test_")
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.arch = os.path.join(self.base, "C", "com.apple.gpuarchiver")
        os.makedirs(self.arch)
        self._env = {
            "DARWIN_USER_CACHE_DIR": os.path.join(self.base, "C"),
            "WARP_METAL_GPU_ARCHIVER_MAX_MB": "8192",
            "WARP_METAL_GPU_ARCHIVER_MAX_AGE_DAYS": "3",
            "WARP_METAL_GPU_ARCHIVER_MAX_ENTRIES": "16",
        }
        for k, v in self._env.items():
            self.addCleanup(self._restore, k, os.environ.get(k))
            os.environ[k] = v

    @staticmethod
    def _restore(key, value) -> None:
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    def _entry(self, name: str, age_days: float, nbytes: int = 1024) -> str:
        path = os.path.join(self.arch, name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "blob"), "wb") as f:
            f.write(b"\0" * nbytes)
        stamp = time.time() - age_days * 86400
        os.utime(path, (stamp, stamp))
        return path

    def _names(self, prefix: str = "gpuarchiver-") -> list[str]:
        return sorted(n for n in os.listdir(self.arch) if n.startswith(prefix))

    def test_age_cap_drops_stale_entries_only(self):
        self._entry("gpuarchiver-stale", age_days=10)
        self._entry("gpuarchiver-recent", age_days=1)
        unrelated = os.path.join(self.arch, "not-an-entry")
        os.makedirs(unrelated)
        self.reap()
        self.assertEqual(self._names(), ["gpuarchiver-recent"])
        self.assertTrue(os.path.isdir(unrelated), "non-gpuarchiver directories must be left alone")

    def test_entries_younger_than_an_hour_always_survive(self):
        # A live process in another terminal must keep its own entry.
        self._entry("gpuarchiver-live", age_days=0)
        self._entry("gpuarchiver-stale", age_days=99)
        os.environ["WARP_METAL_GPU_ARCHIVER_MAX_ENTRIES"] = "1"
        os.environ["WARP_METAL_GPU_ARCHIVER_MAX_MB"] = "1"
        self.reap()
        self.assertIn("gpuarchiver-live", self._names())

    def test_count_cap_keeps_the_newest(self):
        for i in range(5):
            self._entry(f"gpuarchiver-c{i}", age_days=2.5 - i * 0.1)  # c4 newest
        os.environ["WARP_METAL_GPU_ARCHIVER_MAX_ENTRIES"] = "2"
        self.reap()
        self.assertEqual(self._names(), ["gpuarchiver-c3", "gpuarchiver-c4"])

    def test_size_cap_evicts_oldest_until_under_budget(self):
        for i in range(3):
            self._entry(f"gpuarchiver-s{i}", age_days=2.5 - i * 0.1, nbytes=2 * 1024 * 1024)
        os.environ["WARP_METAL_GPU_ARCHIVER_MAX_MB"] = "3"  # 3 MB budget, 6 MB present
        self.reap()
        survivors = self._names()
        self.assertLessEqual(len(survivors), 1)
        if survivors:
            self.assertEqual(survivors, ["gpuarchiver-s2"], "the newest entry is the one to keep")

    def test_zero_max_mb_disables_reaping(self):
        self._entry("gpuarchiver-ancient", age_days=999)
        os.environ["WARP_METAL_GPU_ARCHIVER_MAX_MB"] = "0"
        self.reap()
        self.assertEqual(self._names(), ["gpuarchiver-ancient"])

    def test_missing_cache_directory_is_not_an_error(self):
        os.environ["DARWIN_USER_CACHE_DIR"] = os.path.join(self.base, "does-not-exist")
        self.reap()  # must not raise


if __name__ == "__main__":
    unittest.main()
