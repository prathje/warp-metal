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

import atexit
import ctypes
import hashlib
import os
import threading
from typing import Any


_dispatcher_singleton: MetalDispatcher | None = None
_dispatcher_lock = threading.Lock()


class MetalGraph:
    """Replayable graph of Metal compute dispatches.

    Captured by :meth:`MetalDispatcher.begin_record` /
    :meth:`MetalDispatcher.end_record`, replayed by
    :meth:`MetalDispatcher.replay`.

    The captured form is an :class:`MTLIndirectCommandBuffer` with N
    pre-encoded commands (one per ``dispatch()`` call made during
    recording). Each command has its PSO + bindings + grid +
    threadgroup baked in, so replay skips Warp's entire per-launch
    Python path and Metal's encoder-level state-set calls — only a
    sequence of ``executeCommandsInBuffer`` calls hit the GPU driver.

    Replay correctness for data-dependent kernel chains
    --------------------------------------------------
    Apple's compute ICB commands run **concurrently** by design
    (``MTLIndirectCommandType.ConcurrentDispatchThreads`` — there is
    no serial type for compute, verified against the SDK header). To
    serialise commands that have read-after-write dependencies on the
    same buffer, :meth:`MetalDispatcher.replay` splits the ICB into
    one-command ranges and inserts a ``memoryBarrierWithScope:`` on
    the outer encoder between every pair. That keeps semantics
    identical to per-launch direct dispatch while still skipping the
    Python launch path — measured at ~8 µs per "launch" vs ~52 µs
    for direct ``dispatcher.dispatch`` (6.5× faster).
    """

    __slots__ = (
        "_icb",
        "_count",
        "_resources",
        "_owned_buffers",
        "_signature",
    )

    def __init__(self, icb, count: int, resources: list, owned_buffers: list, signature: Any = None):
        self._icb = icb
        self._count = count
        # MTLBuffers referenced by the recorded commands. The replay
        # encoder must call ``useResource:usage:`` on each so Metal
        # makes them resident before the dispatch starts. Bindings are
        # baked into the ICB itself, but ``useResource`` is the only
        # way the parent encoder learns which resources the indirect
        # commands touch.
        self._resources = resources
        # Small MTLBuffers we allocated to hold ``setBytes``-style
        # data (scalar args, packed shape ints, etc.) on behalf of the
        # recorded launches. Keep refs alive for the graph's lifetime
        # — the ICB doesn't retain them, but they're referenced by
        # the GPU on every replay.
        self._owned_buffers = owned_buffers
        # Opaque caller-supplied tag (used by higher layers, e.g. a
        # Warp graph wrapper, to invalidate the cache when the source
        # workload changes). Untouched by the dispatcher.
        self._signature = signature

    @property
    def count(self) -> int:
        return self._count

    @property
    def signature(self) -> Any:
        return self._signature


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
        # alive until the buffer completes. The completion handler hands
        # ownership to the GPU command-buffer lifecycle so we can drop
        # them eagerly at :meth:`flush` (no host-side blocking).
        self._inflight_refs: list = []
        # All in-flight (fire-and-forget) command buffers committed via
        # :meth:`flush`. :meth:`sync` must wait on every one of these
        # before returning. Apple's docs say command buffers on a single
        # ``MTLCommandQueue`` *commit* in order, but execution can
        # overlap — waiting on the newest does NOT imply prior ones have
        # finished writing to shared-storage buffers. Without explicit
        # waits on each, the host reads land 1-2 steps behind reality
        # (observed empirically on mujoco_warp's ~683-dispatch step,
        # which autoflushes ~3x and leaves the older two cmd buffers
        # in flight at sync time).
        self._pending_commits: list = []
        # Auto-flush threshold. ``launch_metal_kernel_native`` enqueues
        # per-launch transient buffers (packed shapes / ints / floats /
        # struct args) into ``_inflight_refs``; if a workload runs many
        # launches without an explicit sync, these would accumulate
        # unbounded (a single G1 step issues ~700K). Flushing the
        # command buffer mid-stream commits the queued work, drops the
        # refs via the completion handler, and starts a fresh cmd
        # buffer — bounded memory at the cost of one ``commit`` per
        # ``_AUTOFLUSH_EVERY`` launches.
        self._dispatch_count = 0
        # Per-dispatch commit is empirically faster than batched
        # commits with intra-encoder ``memoryBarrierWithScope``: on a
        # G1 step (~1000 launches) we measured 493 ms/step at
        # ``autoflush=1`` vs 629-842 ms/step at 16-256. Apple's
        # implicit cross-cmd-buffer hazard tracking pipelines
        # non-dependent launches better than serial in-encoder
        # barriers do. Keep the barrier code (used when batching is
        # enabled by future workloads) but default to 1.
        self._autoflush_every = 1
        # Cache the NSRange constructor — avoids one PyObjC lookup per
        # batch-bind.
        self._NSMakeRange = Metal.NSMakeRange
        # Cache the buffer-scope barrier value — used between every
        # dispatch in the same encoder when batching is enabled.
        self._barrier_scope_buffers = Metal.MTLBarrierScopeBuffers
        # ``WARP_METAL_CANARY=1`` enables OOB-write detection: alloc
        # fills the guard region with a sentinel pattern and stores
        # ``buf -> (data_nbytes, guard_nbytes)`` here. After a sync we
        # scan every recorded buffer for sentinel violations.
        self._canaries: dict = {}
        self._lock = threading.Lock()
        # ------------------------------------------------------------
        # MTLBinaryArchive on-disk PSO cache.
        #
        # Mirrors Warp's CUDA path: ``warp.config.kernel_cache_dir`` is
        # a versioned directory; we write one ``metal_pso_archive``
        # binary archive into it. On first use of each PSO, Apple's
        # back-end compile populates the archive; subsequent process
        # starts skip the back-end compile entirely (single-PSO probe
        # showed ~57 ms cold vs ~0.1 ms warm). Without this every
        # ``MetalDispatcher`` start re-paid the full PSO build cost
        # — for mjlab G1 that's ~700 unique kernels × ~50 ms each
        # = ~40 s of compile time on every process launch.
        #
        # The archive holds *compiled* PSOs only — the front-end
        # ``newLibraryWithSource_options_error_`` still parses MSL each
        # time. In practice that step is ~25× cheaper than PSO
        # construction on a real kernel, and Metal's in-process library
        # cache handles repeated-source compiles within one run.
        self._archive_url = None
        self._archive = None
        self._archive_dirty = False
        try:
            self._init_binary_archive()
        except Exception as exc:  # noqa: BLE001
            # Don't let cache setup failures take down the dispatcher —
            # fall back to in-memory only.
            import warnings  # noqa: PLC0415
            warnings.warn(
                f"MetalDispatcher: binary archive disabled ({exc!r}); "
                f"PSOs will be re-compiled every run.",
                stacklevel=2,
            )
        if self._archive is not None:
            # Persist on interpreter shutdown so the next process gets a
            # warm cache. ``atexit`` runs handlers in LIFO order, after
            # the main script returns but before Python tears down
            # extension state, so PyObjC bridges are still alive.
            atexit.register(self._serialize_archive_if_dirty)

        # ------------------------------------------------------------
        # Indirect-command-buffer recording state.
        #
        # When :meth:`begin_record` has been called, every subsequent
        # :meth:`dispatch` writes its PSO + bindings + grid + threadgroup
        # into the next slot of ``_record_icb`` instead of issuing live
        # encoder commands. :meth:`end_record` rolls those slots up into
        # a :class:`MetalGraph` that callers can replay cheaply.
        #
        # Holds either ``None`` (not recording) or a small dict with the
        # in-flight recording state.
        self._record_state: dict | None = None
        # Cached descriptor used to allocate new ICBs. We reuse it
        # across captures (mutating the slot count per call) to avoid
        # the PyObjC alloc/init overhead on every record session.
        self._icb_desc = None
        # DEBUG: per-kernel dispatch profiling. Disabled by default
        # (one branch + dict lookup per dispatch is cheap but not free).
        self._profile_dispatch = False
        self._pso_name: dict[int, str] = {}
        # entry_point -> [count, total_ns, total_grid]
        self._dispatch_stats: dict[str, list] = {}

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

        Builds the PSO via ``MTLComputePipelineDescriptor`` so we can
        attach the on-disk binary archive: PSOs already serialized into
        the archive reload in ~0.1 ms instead of paying ~20 ms back-end
        compile each time.
        """
        key = (self._hash(source), entry_point)
        cached = self._pipeline_cache.get(key)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._pipeline_cache.get(key)
            if cached is not None:
                return cached
            Metal = self._Metal
            lib = self._library_cache.get(key[0])
            if lib is None:
                # Match MLX's compile options as closely as we can.
                # MLX disables fast-math by default (so ``normalize((0,0,0))``
                # returns ``0`` instead of ``NaN`` per IEEE 0/0 rules), which
                # mujoco_warp's ``quat_integrate`` depends on for the
                # ``angle = 0`` corner case to integrate cleanly.
                opts = Metal.MTLCompileOptions.alloc().init()
                opts.setFastMathEnabled_(False)
                lib, err = self._device.newLibraryWithSource_options_error_(
                    source, opts, None
                )
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
            pso_desc = Metal.MTLComputePipelineDescriptor.alloc().init()
            pso_desc.setComputeFunction_(fn)
            # Allow the PSO to be encoded into a ``MTLIndirectCommandBuffer``
            # (see :class:`MetalGraph` later in this file). Without this
            # flag, ``indirectComputeCommand.setComputePipelineState:``
            # silently produces an invalid command and the GPU crashes
            # on execute. Apple's docs warn the flag may marginally hurt
            # runtime perf for kernels that never end up in an ICB; we
            # measured no observable regression on the existing
            # ``test_metal_*`` suite, so enable it unconditionally.
            pso_desc.setSupportIndirectCommandBuffers_(True)
            if self._archive is not None:
                pso_desc.setBinaryArchives_([self._archive])
            pso, err = self._device.newComputePipelineStateWithDescriptor_error_(
                pso_desc, None
            )
            if pso is None:
                raise MetalDispatchError(
                    f"newComputePipelineStateWithDescriptor failed for {entry_point!r}: {err}"
                )
            if self._archive is not None:
                # If this PSO wasn't already in the archive, add it so the
                # next process start can reload it without a back-end
                # compile. ``addComputePipelineFunctionsWithDescriptor_``
                # returns False when the entry is already present — that's
                # a no-op, not an error, so we just track whether *any*
                # add succeeded to decide whether to re-serialize.
                added, _ = (
                    self._archive.addComputePipelineFunctionsWithDescriptor_error_(
                        pso_desc, None
                    )
                )
                if added:
                    self._archive_dirty = True
            self._pipeline_cache[key] = pso
            self._pso_name[id(pso)] = entry_point
            return pso

    # ------------------------------------------------------------------
    # MTLBinaryArchive helpers
    # ------------------------------------------------------------------

    def _init_binary_archive(self) -> None:
        """Open the on-disk PSO archive, creating one if it doesn't exist.

        Archive lives at ``<warp.config.kernel_cache_dir>/metal_pso_archive``
        — same versioned cache root the CUDA path uses, so a Warp
        upgrade naturally invalidates the archive.
        """
        import warp.config as _wp_cfg  # noqa: PLC0415

        cache_dir = getattr(_wp_cfg, "kernel_cache_dir", None)
        if not cache_dir:
            # ``warp.init()`` not yet called — archive disabled.
            return
        archive_path = os.path.join(cache_dir, "metal_pso_archive")
        os.makedirs(cache_dir, exist_ok=True)

        Metal = self._Metal
        try:
            import Foundation  # noqa: PLC0415
        except ImportError as exc:
            raise MetalDispatchError(
                "Foundation unavailable; cannot construct NSURL for archive path"
            ) from exc

        desc = Metal.MTLBinaryArchiveDescriptor.alloc().init()
        if os.path.exists(archive_path):
            # Apple errors if the URL points to a non-existent file, so
            # only set it when the file is present. On a missing file we
            # fall through and create an empty archive.
            url = Foundation.NSURL.fileURLWithPath_(archive_path)
            desc.setUrl_(url)
        archive, err = self._device.newBinaryArchiveWithDescriptor_error_(desc, None)
        if archive is None:
            # Treat as a soft failure — disable archiving for this run.
            raise MetalDispatchError(
                f"newBinaryArchiveWithDescriptor failed for {archive_path!r}: {err}"
            )
        self._archive = archive
        self._archive_url = Foundation.NSURL.fileURLWithPath_(archive_path)

    def _serialize_archive_if_dirty(self) -> None:
        """Write the archive to disk if any new PSO was added this run.

        Called from ``atexit``; failures are logged but never raised so
        a stale cache can't crash the interpreter at shutdown.
        """
        if (
            self._archive is None
            or self._archive_url is None
            or not self._archive_dirty
        ):
            return
        try:
            ok, err = self._archive.serializeToURL_error_(self._archive_url, None)
            if not ok:
                import warnings  # noqa: PLC0415
                warnings.warn(
                    f"MetalDispatcher: failed to serialize binary archive: {err}",
                    stacklevel=2,
                )
            else:
                self._archive_dirty = False
        except Exception as exc:  # noqa: BLE001
            import warnings  # noqa: PLC0415
            warnings.warn(
                f"MetalDispatcher: archive serialize raised {exc!r}",
                stacklevel=2,
            )

    @staticmethod
    def _hash(source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def fill_zero(self, mtl_buf, nbytes: int) -> None:
        """Zero a region of an ``MTLBuffer`` via a blit encoder.

        Used by ``launch_metal_kernel_native`` to match MLX's
        ``init_value=0`` semantics for atomic-output kernels whose
        outputs would otherwise inherit the user's prior data
        (because the wp.array's MTLBuffer is bound directly instead
        of a fresh MLX allocation).

        Reuses the in-flight command buffer so the fill is part of
        the same async stream as the surrounding launches.
        """
        if nbytes <= 0:
            return
        Metal = self._Metal
        # End the live compute encoder, if any, so we can switch to a
        # blit encoder. Reopen the compute encoder after — minor
        # ObjC churn but only on atomic-output launches.
        had_encoder = self._encoder is not None
        if had_encoder:
            self._encoder.endEncoding()
            self._encoder = None
        if self._cmd_buf is None:
            self._cmd_buf = self._command_queue.commandBuffer()
            if self._cmd_buf is None:
                raise MetalDispatchError("MTLCommandQueue commandBuffer returned None")
            self._inflight_refs = []
        blit = self._cmd_buf.blitCommandEncoder()
        blit.fillBuffer_range_value_(mtl_buf, Metal.NSMakeRange(0, nbytes), 0)
        blit.endEncoding()
        self._inflight_refs.append(mtl_buf)

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

        ``WARP_METAL_ALLOC_GUARD_BYTES``: add that many extra bytes
        beyond ``nbytes`` to every allocation (defaults to ``0``).
        Useful as a diagnostic for out-of-bounds writes — bump it up
        to e.g. ``4096`` to test whether a numerical regression is
        masked by Metal's tight unified-memory packing.
        """
        if nbytes <= 0:
            raise ValueError(f"MetalDispatcher.alloc: nbytes must be positive, got {nbytes}")
        import os as _os  # noqa: PLC0415

        # Default guard: 4 KiB. Several mujoco_warp kernels (notably
        # ``_efc_contact_init``, ``_efc_contact_update``,
        # ``update_constraint_efc``, ``solve_done``, …) write one
        # element past the end of their nominal output array — confirmed
        # via the canary sanitiser. MLX's larger memory pool masked the
        # overshoot; ``MTLDevice newBufferWithLength`` packs allocations
        # tightly so the OOB writes corrupted adjacent ``wp.array``s.
        # Padding every allocation pushes the overshoot into a harmless
        # guard region until the upstream kernels are fixed.
        guard = int(_os.environ.get("WARP_METAL_ALLOC_GUARD_BYTES", "4096") or "0")
        # ``WARP_METAL_CANARY``: pre-fill the guard region with a sentinel
        # byte (default ``0xAB``). Combined with the dispatcher's
        # post-launch scan (see ``_check_canaries``), this points the
        # finger at any kernel that wrote past its bound MTLBuffer.
        canary_enabled = bool(int(_os.environ.get("WARP_METAL_CANARY", "0") or "0"))
        # Round up to a 4-byte boundary so an atomic store of a 32-bit
        # type (``atomic_bool`` etc., which MSL implements as a 32-bit
        # atomic regardless of the logical element size) doesn't write
        # past the end of a small allocation. The wp.array's reported
        # size stays at ``nbytes``; the extra bytes only live in the
        # MTLBuffer to absorb size-mismatched stores.
        padded = (nbytes + 3) & ~3
        alloc_size = padded + guard
        buf = self._device.newBufferWithLength_options_(alloc_size, self._shared_storage)
        if buf is None:
            raise MetalDispatchError(
                f"MTLDevice newBufferWithLength failed for {alloc_size} bytes"
            )
        # ``contents()`` returns a PyObjC ``objc.varlist``. ``as_buffer(n)``
        # gives a memoryview of the unified-memory bytes; the address
        # is stable for the MTLBuffer's lifetime and is what the GPU
        # sees too.
        import numpy as np  # noqa: PLC0415

        contents = buf.contents()
        mv = contents.as_buffer(alloc_size)
        # Resolve to an integer address via numpy's array interface —
        # cheaper than constructing a ctypes type.
        view = np.frombuffer(mv, dtype=np.uint8)
        addr = int(view.__array_interface__["data"][0])
        if canary_enabled and guard > 0:
            # Fill the guard region with a recognisable sentinel pattern.
            # ``_check_canaries`` later scans for any byte that isn't the
            # sentinel — that's an OOB write that needs investigation.
            # The 4-byte alignment padding stays out of the canary region
            # so atomic_bool writes (which span 4 bytes) don't trip it.
            view[padded:].fill(0xAB)
            self._canaries[buf] = (padded, alloc_size - padded)
        return buf, addr

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def check_canaries(self, label: str, bindings: list, slot_names: list | None = None) -> list[tuple]:
        """Scan ``bindings`` for any guard-region bytes that aren't the
        sentinel. Returns a list of
        ``(slot_index, slot_name, data_nbytes, first_bad_offset)`` for
        offenders. Empty if all canaries are intact (or canary mode
        is off).

        Use as a post-launch / post-sync diagnostic to identify which
        kernel writes past one of its bound ``MTLBuffer``s. The check
        forces a ``sync()`` because the GPU-side write to the guard
        region needs to be observable from the host.
        """
        if not self._canaries:
            return []
        import numpy as np  # noqa: PLC0415

        self.sync()
        offenders: list[tuple] = []
        for idx, entry in enumerate(bindings):
            if isinstance(entry, tuple):
                continue  # ``setBytes`` payloads carry no guard region.
            info = self._canaries.get(entry)
            if info is None:
                continue
            data_nbytes, guard_nbytes = info
            mv = entry.contents().as_buffer(data_nbytes + guard_nbytes)
            arr = np.frombuffer(mv, dtype=np.uint8)
            guard_slice = arr[data_nbytes:]
            if not np.all(guard_slice == 0xAB):
                first_bad = int(np.argmin(guard_slice == 0xAB))
                slot_name = slot_names[idx] if slot_names and idx < len(slot_names) else "?"
                offenders.append((idx, slot_name, data_nbytes, first_bad))
        if offenders:
            for slot, name, data_nbytes, first_bad in offenders:
                print(
                    f"[canary] {label}: slot {slot} {name!r} "
                    f"(data={data_nbytes}B) clobbered guard at +{first_bad}B",
                    flush=True,
                )
        return offenders

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

        When :meth:`begin_record` has been called the dispatch is
        encoded into the in-flight ``MTLIndirectCommandBuffer`` slot
        instead, and *not* executed. Resolve via :meth:`end_record`
        and replay via :meth:`replay`.
        """
        Metal = self._Metal
        if self._profile_dispatch:
            import time as _time  # noqa: PLC0415
            _prof_t0 = _time.perf_counter_ns()
        # Recording path: write the dispatch into an ICB slot. No
        # encoder commands hit the live cmd buffer until replay.
        if self._record_state is not None:
            self._record_dispatch(pso, bindings, grid, threadgroup)
            if self._profile_dispatch:
                self._bump_stats(pso, grid, _time.perf_counter_ns() - _prof_t0)
            return
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
        # Apple's compute encoder does NOT auto-synchronise back-to-back
        # ``dispatchThreads`` calls — kernel B may see kernel A's
        # pre-write buffer view unless we insert this. The earlier
        # confusion about "memoryBarrierWithScope_ doesn't fix it" came
        # from the per-thread output-init prologue racing inside the
        # kernel *body*, which the barrier can't see across; that
        # prologue is now stripped by the native-dispatch wrapper. With
        # it gone the barrier is sufficient and we can batch dispatches
        # behind a single commit instead of paying one cmd-buffer per
        # launch (the previous policy was correct for safety but ~10×
        # slower on G1).
        encoder.memoryBarrierWithScope_(self._barrier_scope_buffers)
        self._dispatch_count += 1
        if self._dispatch_count >= self._autoflush_every:
            # Bounded-memory dispatch: commit periodically so transient
            # refs in ``_inflight_refs`` get released by the GPU
            # completion handler. Doesn't block — a fresh cmd buffer
            # picks up the next dispatch on the same queue (commit order
            # is preserved, ``sync()`` drains the full chain).
            self.flush()
            self._dispatch_count = 0
        if self._profile_dispatch:
            self._bump_stats(pso, grid, _time.perf_counter_ns() - _prof_t0)

    def _bump_stats(self, pso, grid: tuple[int, int, int], elapsed_ns: int) -> None:
        """Record one dispatch's wall time into the per-kernel stats dict."""
        name = self._pso_name.get(id(pso), "<unknown>")
        slot = self._dispatch_stats.get(name)
        if slot is None:
            slot = [0, 0, 0]
            self._dispatch_stats[name] = slot
        slot[0] += 1
        slot[1] += elapsed_ns
        slot[2] += grid[0] * grid[1] * grid[2]

    def reset_dispatch_stats(self) -> None:
        """Clear accumulated per-kernel dispatch timings."""
        self._dispatch_stats.clear()

    def enable_dispatch_profile(self, enabled: bool = True) -> None:
        """Toggle per-kernel dispatch timing collection."""
        self._profile_dispatch = bool(enabled)

    def dispatch_stats(self, top: int | None = None) -> list[tuple[str, int, int, int]]:
        """Return per-kernel stats sorted by total time, descending.

        Each row is ``(name, count, total_ns, total_grid_elems)``.
        ``top`` truncates to that many rows; ``None`` returns all.
        """
        rows = [
            (name, slot[0], slot[1], slot[2])
            for name, slot in self._dispatch_stats.items()
        ]
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows if top is None else rows[:top]

    def _end_encoder(self) -> None:
        """Close the live compute encoder, if any."""
        if self._encoder is not None:
            self._encoder.endEncoding()
            self._encoder = None

    def flush(self) -> None:
        """Commit the in-flight command buffer (fire and forget).

        Returns immediately; the GPU work runs asynchronously. The
        bound buffer references are held until the command buffer
        signals completion via the closure capture in the completion
        handler — Metal releases the handler block after invocation,
        which drops the closure and lets the buffers be GC'd.
        """
        if self._cmd_buf is None:
            return
        self._end_encoder()
        cmd_buf = self._cmd_buf
        refs = self._inflight_refs
        self._cmd_buf = None
        self._inflight_refs = []

        # ``_release`` does nothing explicit — it just exists to *capture*
        # ``refs`` in its closure so the bound buffers stay alive until
        # Metal calls back and releases the handler block. Avoid
        # ``del refs`` here: PyObjC re-invokes the handler from a non-
        # Python dispatch queue and a ``del`` of a free var raises
        # ``UnboundLocalError`` mid-callback (it tries to shadow the
        # cell as a local).
        def _release(_cmd_buf):
            refs  # noqa: B018, keep closure ref alive

        cmd_buf.addCompletedHandler_(_release)
        cmd_buf.commit()
        # Remember every fire-and-forget commit so :meth:`sync` can
        # block until *each* one has finished. See ``_pending_commits``
        # docstring for why a single newest-only tracker is not enough.
        self._pending_commits.append(cmd_buf)

    def sync(self) -> None:
        """Block until every committed command buffer is complete.

        Call before any host read of buffer contents (``.numpy()``,
        explicit ``wp.synchronize_device``, etc.).

        Must explicitly wait on *each* in-flight commit. Apple's
        ``MTLCommandQueue`` schedules command buffers in commit order,
        but execution can overlap — ``waitUntilCompleted`` on a newer
        buffer does **not** imply earlier ones are also finished, so the
        host can read stale unified-memory bytes from a still-running
        prior cmd buffer (observed empirically: 1-2 step lag on mjw step
        before this loop was added).
        """
        if self._cmd_buf is not None:
            self._end_encoder()
            cmd_buf = self._cmd_buf
            refs = self._inflight_refs
            self._cmd_buf = None
            self._inflight_refs = []
            cmd_buf.commit()
            self._pending_commits.append(cmd_buf)
            del refs
        # Drain every fire-and-forget cmd buffer the dispatcher has
        # outstanding. They were committed to the same queue in commit
        # order; we wait sequentially so any host read after ``sync()``
        # sees a consistent post-execution view of every shared buffer.
        for cb in self._pending_commits:
            cb.waitUntilCompleted()
        self._pending_commits = []

    # ------------------------------------------------------------------
    # Indirect-command-buffer graph capture & replay
    # ------------------------------------------------------------------

    def begin_record(self, max_commands: int = 4096, signature: Any = None) -> None:
        """Enter ICB recording mode.

        Subsequent :meth:`dispatch` calls write into an
        ``MTLIndirectCommandBuffer`` slot rather than executing live.
        Call :meth:`end_record` to return the captured graph.

        ``max_commands`` is the slot count of the ICB allocation; if
        recording exceeds it the dispatcher raises
        :class:`MetalDispatchError`. Pick a value comfortably above
        the dispatch count of whatever you're capturing — over-
        allocation is cheap, expansion is not.

        ``signature`` is an opaque caller-supplied tag stored on the
        graph so higher-level cache layers can detect when the
        captured workload has changed and invalidate.

        Nested recording is forbidden; live dispatches inside a
        capture region would split GPU state across two paths.
        """
        if self._record_state is not None:
            raise MetalDispatchError("Already recording an ICB graph; call end_record() first")
        Metal = self._Metal
        # Flush any in-flight encoder so live work doesn't accidentally
        # get encoded alongside the recording. The graph itself is
        # GPU-driven, so the caller still has to call ``replay`` +
        # ``sync`` to observe its effects.
        self._end_encoder()
        # ICB descriptors are immutable once handed to ``newIndirectCommandBuffer``,
        # so we always build a fresh one — but cache the constructor
        # call across captures.
        if self._icb_desc is None:
            self._icb_desc = Metal.MTLIndirectCommandBufferDescriptor.alloc().init()
            self._icb_desc.setCommandTypes_(Metal.MTLIndirectCommandTypeConcurrentDispatchThreads)
            self._icb_desc.setInheritBuffers_(False)
            self._icb_desc.setInheritPipelineState_(False)
        # ``maxKernelBufferBindCount`` must cover the widest kernel
        # signature we'll ever record. mujoco_warp kernels rarely
        # exceed 30 buffer slots; 31 is Metal's documented hardware
        # limit, so use that as a safe default.
        self._icb_desc.setMaxKernelBufferBindCount_(31)
        icb = self._device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options_(
            self._icb_desc, max_commands, 0
        )
        if icb is None:
            raise MetalDispatchError(
                f"newIndirectCommandBufferWithDescriptor returned None for max_commands={max_commands}"
            )
        self._record_state = {
            "icb": icb,
            "max": max_commands,
            "count": 0,
            "resources_set": set(),  # id(buffer) → buffer, for dedupe
            "resources": [],
            "owned_buffers": [],
            "signature": signature,
        }

    def end_record(self) -> MetalGraph:
        """Exit recording mode and return the captured :class:`MetalGraph`."""
        if self._record_state is None:
            raise MetalDispatchError("Not currently recording an ICB graph")
        st = self._record_state
        self._record_state = None
        return MetalGraph(
            icb=st["icb"],
            count=st["count"],
            resources=st["resources"],
            owned_buffers=st["owned_buffers"],
            signature=st["signature"],
        )

    def replay(self, graph: MetalGraph) -> None:
        """Execute ``graph`` on the GPU (fire-and-forget).

        Encodes one ``executeCommandsInBuffer`` call per command in
        the graph, with a ``memoryBarrierWithScope:`` between each
        pair. The per-command-range split is required because Apple's
        compute ICB executes commands concurrently by spec — a
        single multi-command range would race on any kernel chain
        with shared-buffer R/W dependencies.

        Replay piggybacks on the dispatcher's existing autoflush /
        sync machinery: the encoded executeCommandsInBuffer calls
        share the in-flight :class:`MTLCommandBuffer` with any
        non-graph dispatches and respect the same commit / wait
        semantics as :meth:`dispatch`.
        """
        if self._record_state is not None:
            raise MetalDispatchError("Cannot replay while a recording is open")
        if graph.count == 0:
            return
        Metal = self._Metal
        if self._cmd_buf is None:
            self._cmd_buf = self._command_queue.commandBuffer()
            if self._cmd_buf is None:
                raise MetalDispatchError("MTLCommandQueue commandBuffer returned None")
            self._inflight_refs = []
        if self._encoder is None:
            self._encoder = self._cmd_buf.computeCommandEncoder()
            if self._encoder is None:
                raise MetalDispatchError("MTLCommandBuffer computeCommandEncoder returned None")
        encoder = self._encoder
        # Tell Metal all the buffers we're about to indirectly touch.
        # ``MTLResourceUsageRead | MTLResourceUsageWrite`` is the
        # conservative superset — declaring it strictly correctly per
        # binding would let the driver do tighter hazard tracking, but
        # we'd need to thread per-binding usage through the recording
        # path. Leave that as a follow-up.
        usage = Metal.MTLResourceUsageRead | Metal.MTLResourceUsageWrite
        for r in graph._resources:
            encoder.useResource_usage_(r, usage)
        # Keep both the graph and its owned buffers alive until the
        # cmd buffer completes — Metal can dereference them at any
        # point during GPU execution.
        self._inflight_refs.append(graph._icb)
        if graph._owned_buffers:
            self._inflight_refs.extend(graph._owned_buffers)
        # Per-command range execution with a barrier between, so
        # data-dependent kernel chains still produce correct output
        # (see :class:`MetalGraph` docstring for why per-range).
        nsrange = self._NSMakeRange
        scope = self._barrier_scope_buffers
        icb = graph._icb
        for i in range(graph.count):
            encoder.executeCommandsInBuffer_withRange_(icb, nsrange(i, 1))
            encoder.memoryBarrierWithScope_(scope)
        # Count this as ``graph.count`` logical dispatches against
        # the autoflush threshold so the in-flight cmd buffer commits
        # at the same cadence as direct ``dispatch`` use.
        self._dispatch_count += graph.count
        if self._dispatch_count >= self._autoflush_every:
            self.flush()
            self._dispatch_count = 0

    def _record_dispatch(
        self,
        pso,
        bindings: list,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
    ) -> None:
        """Encode one dispatch into the active recording's next ICB slot.

        ``setBytes``-style bindings (``(bytes, length)`` tuples) are
        materialised as fresh shared-storage MTLBuffers so the ICB
        can reference them via ``setKernelBuffer:offset:atIndex:``.
        Apple's ICB compute commands have no setBytes equivalent.
        """
        st = self._record_state
        assert st is not None  # caller checked
        if st["count"] >= st["max"]:
            raise MetalDispatchError(
                f"ICB recording overflowed allocation of {st['max']} commands — "
                "raise begin_record(max_commands=...) for this workload."
            )
        Metal = self._Metal
        cmd = st["icb"].indirectComputeCommandAtIndex_(st["count"])
        cmd.setComputePipelineState_(pso)
        res_set = st["resources_set"]
        resources = st["resources"]
        owned = st["owned_buffers"]
        for idx, entry in enumerate(bindings):
            if isinstance(entry, tuple) and len(entry) == 2:
                # setBytes equivalent: stash the data in a fresh
                # shared-storage buffer the ICB can reference. The
                # buffer must outlive the graph; ``owned`` holds the
                # ref. The data is captured by value here — caller
                # changes to ``entry`` after begin_record() do not
                # propagate into the graph.
                data, length = entry
                buf, addr = self.alloc(length)
                ctypes.memmove(addr, data, length)
                owned.append(buf)
                cmd.setKernelBuffer_offset_atIndex_(buf, 0, idx)
                # ``alloc`` returned a fresh buffer not yet known to
                # the caller; add it to the resource set unconditionally.
                bid = id(buf)
                if bid not in res_set:
                    res_set.add(bid)
                    resources.append(buf)
            else:
                cmd.setKernelBuffer_offset_atIndex_(entry, 0, idx)
                bid = id(entry)
                if bid not in res_set:
                    res_set.add(bid)
                    resources.append(entry)
        gx, gy, gz = grid
        tx, ty, tz = threadgroup
        cmd.concurrentDispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(gx, gy, gz),
            Metal.MTLSizeMake(tx, ty, tz),
        )
        st["count"] += 1
