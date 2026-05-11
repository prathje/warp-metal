# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct Metal-API dispatcher for Warp kernels.

Bypasses ``mx.fast.metal_kernel`` so kernels write in-place into Warp-owned
``MTLBuffer``s, matching CUDA's "kernel writes into user-allocated memory"
semantics. The MLX path always allocates fresh outputs and forces a host
memcpy per launch; this dispatcher eliminates both.

Prototype scope (Phase 2a):
    * Compile MSL via ``MTLDevice newLibraryWithSource``.
    * Cache ``MTLLibrary`` + ``MTLComputePipelineState`` by source hash.
    * Allocate ``MTLBuffer``s via ``MTLDevice newBufferWithLength`` (shared
      storage so the host pointer is the same as the GPU's view).
    * Encode dispatches into a per-frame ``MTLCommandBuffer`` and commit at
      sync points or when explicitly flushed.

Not yet wired into ``launch_metal_kernel`` — Phase 2b. This module is
importable on its own so the standalone tests in
``test_metal_dispatch.py`` can exercise the API.
"""

from __future__ import annotations

import ctypes
import hashlib
import threading
from typing import Any


_dispatcher_singleton: MetalDispatcher | None = None
_dispatcher_lock = threading.Lock()


def get_dispatcher() -> MetalDispatcher:
    """Return the process-wide :class:`MetalDispatcher`, creating it on first call."""
    global _dispatcher_singleton
    if _dispatcher_singleton is None:
        with _dispatcher_lock:
            if _dispatcher_singleton is None:
                _dispatcher_singleton = MetalDispatcher()
    return _dispatcher_singleton


class MetalDispatchError(RuntimeError):
    """Raised on Metal API failure (compile, dispatch, allocate)."""


class MetalDispatcher:
    """Encapsulates the Apple Metal API state Warp needs for direct dispatch.

    Owns the ``MTLDevice``, the command queue, and the pipeline caches.
    Compiles MSL on demand, hands back ``MTLComputePipelineState``
    instances, and dispatches encoded compute passes onto a
    lazily-created ``MTLCommandBuffer``.
    """

    def __init__(self) -> None:
        # Lazy import — keeps a non-macOS install of Warp from failing at
        # module load when this file is imported through the package.
        import Metal  # noqa: PLC0415

        self._Metal = Metal
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            raise MetalDispatchError("MTLCreateSystemDefaultDevice returned None — no Metal GPU available")
        if not device.hasUnifiedMemory():
            # Warp's Metal path assumes unified memory so the buffer's
            # CPU pointer aliases the GPU view. A discrete Metal device
            # would need an explicit blit pass, which we don't support.
            raise MetalDispatchError(
                f"Metal device {device.name()!r} does not have unified memory"
            )
        self._device = device
        self._command_queue = device.newCommandQueue()
        if self._command_queue is None:
            raise MetalDispatchError("MTLDevice newCommandQueue returned None")
        # Storage mode for ``newBufferWithLength`` — ``MTLResourceStorageModeShared``
        # gives us a CPU-addressable pointer via ``buffer.contents()`` that
        # the GPU sees too. The integer constant matches Apple's headers
        # so we don't have to look it up via PyObjC every call.
        self._shared_storage = Metal.MTLResourceStorageModeShared
        # PSO and library caches keyed by source SHA-256 (full source +
        # entry-point name). MSL compilation is the slow step — ~50-500ms
        # per unique kernel — so caching is critical.
        self._library_cache: dict[str, Any] = {}
        self._pipeline_cache: dict[tuple[str, str], Any] = {}
        # Lazily-created command buffer + a single live compute encoder
        # shared across dispatches between sync points. Many launches
        # encode through the same encoder; we only call ``endEncoding`` +
        # ``commit`` at :meth:`flush` / :meth:`sync`. Each ObjC call has
        # fixed PyObjC overhead, so collapsing encoder lifecycle from
        # per-dispatch to per-batch is a measurable win.
        self._cmd_buf: Any = None
        self._encoder: Any = None
        # Refs to MTLBuffers bound to the current cmd buffer — must stay
        # alive until the buffer completes.
        self._inflight_refs: list = []
        # Cache the NSRange constructor — avoids one PyObjC lookup per
        # batch-bind.
        self._NSMakeRange = Metal.NSMakeRange
        self._lock = threading.Lock()

    @property
    def device(self):
        """The wrapped :class:`MTLDevice` (PyObjC proxy)."""
        return self._device

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(self, source: str, entry_point: str):
        """Return a ``MTLComputePipelineState`` for ``entry_point`` in ``source``.

        Caches the compiled library and pipeline state — subsequent
        compiles with the same source are O(dict lookup).
        """
        key = (self._hash(source), entry_point)
        cached = self._pipeline_cache.get(key)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._pipeline_cache.get(key)
            if cached is not None:
                return cached
            lib = self._library_cache.get(key[0])
            if lib is None:
                lib, err = self._device.newLibraryWithSource_options_error_(source, None, None)
                if lib is None:
                    raise MetalDispatchError(
                        f"MSL compilation failed for entry point {entry_point!r}: {err}"
                    )
                self._library_cache[key[0]] = lib
            fn = lib.newFunctionWithName_(entry_point)
            if fn is None:
                raise MetalDispatchError(
                    f"MSL library has no function named {entry_point!r}"
                )
            pso, err = self._device.newComputePipelineStateWithFunction_error_(fn, None)
            if pso is None:
                raise MetalDispatchError(
                    f"newComputePipelineStateWithFunction failed for {entry_point!r}: {err}"
                )
            self._pipeline_cache[key] = pso
            return pso

    @staticmethod
    def _hash(source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def alloc(self, nbytes: int):
        """Allocate a shared-storage ``MTLBuffer`` of ``nbytes`` bytes.

        Returns ``(mtl_buffer, cpu_ptr)``. The CPU pointer aliases the
        same bytes as the GPU view (unified memory + shared storage),
        so host code can read/write through ``cpu_ptr`` and any
        subsequent compute pass that binds the buffer sees the latest
        bytes.

        Allocation is uninitialised — callers that need a zero-filled
        buffer should follow up with ``ctypes.memset(cpu_ptr, 0, n)``
        or queue a fill-kernel dispatch.
        """
        if nbytes <= 0:
            raise ValueError(f"MetalDispatcher.alloc: nbytes must be positive, got {nbytes}")
        buf = self._device.newBufferWithLength_options_(nbytes, self._shared_storage)
        if buf is None:
            raise MetalDispatchError(
                f"MTLDevice newBufferWithLength failed for {nbytes} bytes"
            )
        # ``contents()`` returns a PyObjC ``objc.varlist``. ``as_buffer(n)``
        # gives a memoryview of the unified-memory bytes; the address
        # is stable for the MTLBuffer's lifetime and is what the GPU
        # sees too.
        import numpy as np  # noqa: PLC0415

        contents = buf.contents()
        mv = contents.as_buffer(nbytes)
        # Resolve to an integer address via numpy's array interface —
        # cheaper than constructing a ctypes type.
        addr = int(np.frombuffer(mv, dtype=np.uint8).__array_interface__["data"][0])
        return buf, addr

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        pso,
        bindings: list,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
    ) -> None:
        """Encode one compute dispatch onto the in-flight command buffer.

        ``bindings`` is a list whose i-th entry becomes argument-slot
        ``i`` in the kernel. Each entry is either:

        * An ``MTLBuffer`` — bound via ``setBuffer:offset:atIndex:``.
        * A ``(bytes-like, length)`` tuple — bound via
          ``setBytes:length:atIndex:`` (cheaper for small const args
          ≤4KB; avoids a buffer allocation).

        The caller is responsible for ordering bindings to match the
        kernel's signature.

        This does not commit the command buffer — call :meth:`flush`
        to schedule it for execution, or :meth:`sync` to wait for it.
        """
        Metal = self._Metal
        if self._encoder is None:
            if self._cmd_buf is None:
                self._cmd_buf = self._command_queue.commandBuffer()
                if self._cmd_buf is None:
                    raise MetalDispatchError("MTLCommandQueue commandBuffer returned None")
                self._inflight_refs = []
            self._encoder = self._cmd_buf.computeCommandEncoder()
            if self._encoder is None:
                raise MetalDispatchError("MTLCommandBuffer computeCommandEncoder returned None")
        encoder = self._encoder
        encoder.setComputePipelineState_(pso)
        # Batch-bind buffers. ``setBytes`` entries are passed one at a
        # time (each pushes a separate small allocation into the
        # encoder's command stream) but consecutive MTLBuffer entries
        # collapse into a single ``setBuffers:offsets:withRange:`` call
        # — fewer PyObjC bridge crossings = lower per-launch overhead.
        run_start = 0
        run_buffers: list = []
        run_offsets: list = []

        def _flush_run(end_exclusive: int) -> None:
            if not run_buffers:
                return
            count = len(run_buffers)
            encoder.setBuffers_offsets_withRange_(
                run_buffers, run_offsets, self._NSMakeRange(end_exclusive - count, count)
            )
            self._inflight_refs.extend(run_buffers)
            run_buffers.clear()
            run_offsets.clear()

        for idx, entry in enumerate(bindings):
            if isinstance(entry, tuple) and len(entry) == 2:
                _flush_run(idx)
                data, length = entry
                encoder.setBytes_length_atIndex_(data, length, idx)
            else:
                run_buffers.append(entry)
                run_offsets.append(0)
        _flush_run(len(bindings))

        gx, gy, gz = grid
        tx, ty, tz = threadgroup
        encoder.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(gx, gy, gz),
            Metal.MTLSizeMake(tx, ty, tz),
        )

    def _end_encoder(self) -> None:
        """Close the live compute encoder, if any."""
        if self._encoder is not None:
            self._encoder.endEncoding()
            self._encoder = None

    def flush(self) -> None:
        """Commit the in-flight command buffer (fire and forget).

        Returns immediately; the GPU work runs asynchronously. The
        bound buffer references are held until the command buffer
        signals completion.
        """
        if self._cmd_buf is None:
            return
        self._end_encoder()
        cmd_buf = self._cmd_buf
        refs = self._inflight_refs
        self._cmd_buf = None
        self._inflight_refs = []

        def _release(_):
            # Capture & release on completion so refs survive until
            # the GPU is done with them.
            del refs

        cmd_buf.addCompletedHandler_(_release)
        cmd_buf.commit()

    def sync(self) -> None:
        """Flush the in-flight command buffer and block until completion.

        Call before any host read of buffer contents (``.numpy()``,
        explicit ``wp.synchronize_device``, etc.).
        """
        if self._cmd_buf is None:
            return
        self._end_encoder()
        cmd_buf = self._cmd_buf
        refs = self._inflight_refs
        self._cmd_buf = None
        self._inflight_refs = []
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()
        del refs
