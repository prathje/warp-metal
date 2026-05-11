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
import sys
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
        return np.frombuffer(
            (ctypes.c_byte * nbytes).from_address(addr), dtype=np.float32
        )

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


if __name__ == "__main__":
    unittest.main()
