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

# Built-in compute kernels for host-op equivalents (fill / copy / pattern
# tile). Dispatched through the ordinary :meth:`MetalDispatcher.dispatch`
# path, so unlike blit-encoder ops they can be recorded into an
# MTLIndirectCommandBuffer and replayed as part of a captured graph.
# ``dispatchThreads`` grids are exact (no partial-threadgroup overrun), so
# the kernels need no bounds check — every thread maps 1:1 to an element.
_DEVICE_OPS_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void wp_fill4(device uint* dst [[buffer(0)]],
                     constant uint& value [[buffer(1)]],
                     constant uint& off [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
    dst[off + tid] = value;
}

kernel void wp_fill1(device uchar* dst [[buffer(0)]],
                     constant uint& value [[buffer(1)]],
                     constant uint& off [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
    dst[off + tid] = uchar(value);
}

kernel void wp_copy4(device const uint* src [[buffer(0)]],
                     device uint* dst [[buffer(1)]],
                     constant uint& src_off [[buffer(2)]],
                     constant uint& dst_off [[buffer(3)]],
                     uint tid [[thread_position_in_grid]]) {
    dst[dst_off + tid] = src[src_off + tid];
}

kernel void wp_copy1(device const uchar* src [[buffer(0)]],
                     device uchar* dst [[buffer(1)]],
                     constant uint& src_off [[buffer(2)]],
                     constant uint& dst_off [[buffer(3)]],
                     uint tid [[thread_position_in_grid]]) {
    dst[dst_off + tid] = src[src_off + tid];
}

kernel void wp_tile4(device const uint* pat [[buffer(0)]],
                     device uint* dst [[buffer(1)]],
                     constant uint& patlen [[buffer(2)]],
                     constant uint& off [[buffer(3)]],
                     uint tid [[thread_position_in_grid]]) {
    dst[off + tid] = pat[tid % patlen];
}

kernel void wp_tile1(device const uchar* pat [[buffer(0)]],
                     device uchar* dst [[buffer(1)]],
                     constant uint& patlen [[buffer(2)]],
                     constant uint& off [[buffer(3)]],
                     uint tid [[thread_position_in_grid]]) {
    dst[off + tid] = pat[tid % patlen];
}

// Gate for conditionally-executed graph regions. One thread per chunk of
// the gated region: copies the chunk's full execution range, zeroing its
// length when the int32 flag reads 0. The replay encoder points each
// chunk's ``executeCommandsInBuffer:indirectBuffer:`` at ``ranges``, so
// the command processor resolves skip-vs-run on the GPU timeline — the
// Metal equivalent of a CUDA conditional graph node.
struct WpExecRange { uint location; uint length; };
kernel void wp_icb_gate(device const int* flag [[buffer(0)]],
                        device WpExecRange* ranges [[buffer(1)]],
                        const device WpExecRange* full [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {
    WpExecRange r = full[tid];
    if (flag[0] == 0) { r.length = 0u; }
    ranges[tid] = r;
}
"""


def _u32(value: int) -> tuple[bytes, int]:
    """Pack an int as a 4-byte setBytes binding tuple."""
    return ((value & 0xFFFFFFFF).to_bytes(4, "little"), 4)


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
        "_chunks",
        "_count",
        "_icb",
        "_names",
        "_owned_buffers",
        "_regions",
        "_resources",
        "_signature",
    )

    def __init__(
        self,
        icb,
        count: int,
        resources: list,
        owned_buffers: list,
        signature: Any = None,
        chunks: list[tuple[int, int]] | None = None,
        regions: list[tuple] | None = None,
        names: list[str] | None = None,
    ):
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
        # Dependency-aware chunking: each entry is ``(start, length)``
        # of consecutive ICB commands with no internal R/W dependency
        # on the same buffer. Replay executes each chunk as a single
        # concurrent ``executeCommandsInBuffer`` range and only emits
        # ``memoryBarrierWithScope:`` BETWEEN chunks (not after every
        # command). When ``chunks`` is ``None`` the replay falls back
        # to per-command ranges with barriers (safe but slow).
        self._chunks = chunks
        # GPU-conditional regions (see ``begin_gated_region``). Each
        # entry is ``(chunk_lo, chunk_hi, flag_buf, flag_offset,
        # ranges_buf, full_buf)``: at replay a ``wp_icb_gate`` dispatch
        # reads the int32 flag and writes ``ranges_buf`` with either the
        # chunks' real execution ranges or zero-length ones, and every
        # chunk in ``[chunk_lo, chunk_hi)`` executes via
        # ``executeCommandsInBuffer:indirectBuffer:`` against it.
        self._regions = regions
        # Per-command kernel entry names, parallel to the ICB slots.
        # Only used for diagnostics (``replay_timed`` attribution);
        # ``None`` for graphs recorded before names were tracked.
        self._names = names

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
            raise MetalDispatchError(f"Metal device {device.name()!r} does not have unified memory")
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
        # GPU-resident resources: MTLBuffers reached only through
        # descriptor-embedded ``gpuAddress``es (BVH node / item arrays,
        # descriptor buffers — see ``warp/_src/metal_bvh.py``). They are
        # never bound to an encoder slot, so every compute encoder must
        # declare them via ``useResource:usage:`` before work that
        # dereferences them. ``_resident_applied``/``_resident_encoder``
        # track how many entries the *current* encoder has seen so the
        # per-dispatch cost is one identity check.
        self._resident_resources: list = []
        self._resident_ids: set[int] = set()
        self._resident_applied = 0
        self._resident_encoder: Any = None
        self._usage_read_write = Metal.MTLResourceUsageRead | Metal.MTLResourceUsageWrite
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
        self._archive_path = None
        self._archive = None
        self._archive_dirty = False
        try:
            self._init_binary_archive()
        except Exception as exc:
            # Don't let cache setup failures take down the dispatcher —
            # fall back to in-memory only.
            import warnings  # noqa: PLC0415

            warnings.warn(
                f"MetalDispatcher: binary archive disabled ({exc!r}); PSOs will be re-compiled every run.",
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
        # ``WARP_METAL_PROFILE_DISPATCH=1`` turns it on from the
        # environment; ``enable_dispatch_profile()`` toggles at runtime.
        self._profile_dispatch = bool(int(os.environ.get("WARP_METAL_PROFILE_DISPATCH", "0") or "0"))
        self._pso_name: dict[int, str] = {}
        # entry_point -> [count, total_ns, total_grid]
        self._dispatch_stats: dict[str, list] = {}
        # Allocation guard / canary settings — read once here rather than
        # per ``alloc()`` call (os.environ lookups on the hot allocation
        # path are measurable at mujoco_warp's allocation counts). See
        # :meth:`alloc` for what they do.
        self._guard_bytes = int(os.environ.get("WARP_METAL_ALLOC_GUARD_BYTES", "4096") or "0")
        self._canary_enabled = bool(int(os.environ.get("WARP_METAL_CANARY", "0") or "0"))

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
                lib, err = self._device.newLibraryWithSource_options_error_(source, opts, None)
                if lib is None:
                    raise MetalDispatchError(f"MSL compilation failed for entry point {entry_point!r}: {err}")
                self._library_cache[key[0]] = lib
            fn = lib.newFunctionWithName_(entry_point)
            if fn is None:
                raise MetalDispatchError(f"MSL library has no function named {entry_point!r}")
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
            pso, err = self._device.newComputePipelineStateWithDescriptor_error_(pso_desc, None)
            if pso is None:
                raise MetalDispatchError(f"newComputePipelineStateWithDescriptor failed for {entry_point!r}: {err}")
            if self._archive is not None:
                # If this PSO wasn't already in the archive, add it so the
                # next process start can reload it without a back-end
                # compile. ``addComputePipelineFunctionsWithDescriptor_``
                # returns False when the entry is already present — that's
                # a no-op, not an error, so we just track whether *any*
                # add succeeded to decide whether to re-serialize.
                added, _ = self._archive.addComputePipelineFunctionsWithDescriptor_error_(pso_desc, None)
                if added:
                    self._archive_dirty = True
            self._pipeline_cache[key] = pso
            self._pso_name[id(pso)] = entry_point
            return pso

    # ------------------------------------------------------------------
    # MTLBinaryArchive helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _archive_salt() -> str:
        """Content hash of the Metal codegen + dispatch sources.

        The archive filename embeds this salt so a codegen change starts
        a FRESH archive instead of appending to the old one. Without the
        salt every codegen iteration re-added its full PSO set to one
        ever-growing file — observed at 4.1 GB after a few iterations,
        at which point Apple's ``serializeToURL:`` segfaulted the
        interpreter at shutdown.
        """
        import warp._src.codegen_metal as _cg  # noqa: PLC0415
        import warp._src.codegen_metal_ast as _cga  # noqa: PLC0415

        h = hashlib.sha256()
        for mod_file in (_cg.__file__, _cga.__file__, __file__):
            with open(mod_file, "rb") as f:
                h.update(f.read())
        return h.hexdigest()[:16]

    def _init_binary_archive(self) -> None:
        """Open the on-disk PSO archive, creating one if it doesn't exist.

        Archive lives at ``<warp.config.kernel_cache_dir>/metal_pso_archive_<salt>``
        — same versioned cache root the CUDA path uses, so a Warp
        upgrade naturally invalidates the archive; the salt (see
        :meth:`_archive_salt`) does the same for codegen changes. Stale
        archives from other salts are deleted here to bound disk usage.
        """
        import warp.config as _wp_cfg  # noqa: PLC0415

        cache_dir = getattr(_wp_cfg, "kernel_cache_dir", None)
        if not cache_dir:
            # ``warp.init()`` not yet called — archive disabled.
            return
        archive_path = os.path.join(cache_dir, f"metal_pso_archive_{self._archive_salt()}")
        os.makedirs(cache_dir, exist_ok=True)

        # Drop archives from previous codegen versions (including the
        # legacy unsalted ``metal_pso_archive`` and any orphaned
        # serialize temp files) — they'd never be read again.
        import glob  # noqa: PLC0415

        for old in glob.glob(os.path.join(cache_dir, "metal_pso_archive*")):
            if old != archive_path:
                try:
                    os.remove(old)
                except OSError:
                    pass

        # Size sanity cap: a runaway archive is worse than a cold cache
        # (multi-GB serialize at every exit, and Apple's serializer has
        # been observed to crash on very large files).
        max_bytes = int(os.environ.get("WARP_METAL_PSO_ARCHIVE_MAX_MB", "1024") or "0") * 1024 * 1024
        if max_bytes > 0 and os.path.exists(archive_path) and os.path.getsize(archive_path) > max_bytes:
            try:
                os.remove(archive_path)
            except OSError:
                pass

        Metal = self._Metal
        try:
            import Foundation  # noqa: PLC0415
        except ImportError as exc:
            raise MetalDispatchError("Foundation unavailable; cannot construct NSURL for archive path") from exc

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
            raise MetalDispatchError(f"newBinaryArchiveWithDescriptor failed for {archive_path!r}: {err}")
        self._archive = archive
        self._archive_path = archive_path

    def _serialize_archive_if_dirty(self) -> None:
        """Write the archive to disk if any new PSO was added this run.

        Called from ``atexit``; failures are logged but never raised so
        a stale cache can't crash the interpreter at shutdown. The write
        goes to a temp file first and is renamed into place so a process
        killed mid-serialize (training runs, test timeouts) can't leave
        a truncated archive for the next process to choke on.
        """
        if self._archive is None or self._archive_path is None or not self._archive_dirty:
            return
        try:
            import Foundation  # noqa: PLC0415

            tmp_path = f"{self._archive_path}.tmp.{os.getpid()}"
            tmp_url = Foundation.NSURL.fileURLWithPath_(tmp_path)
            ok, err = self._archive.serializeToURL_error_(tmp_url, None)
            if not ok:
                import warnings  # noqa: PLC0415

                warnings.warn(
                    f"MetalDispatcher: failed to serialize binary archive: {err}",
                    stacklevel=2,
                )
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            else:
                os.replace(tmp_path, self._archive_path)
                self._archive_dirty = False
        except Exception as exc:
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

    def add_resident_resource(self, mtl_buf) -> None:
        """Mark ``mtl_buf`` as reachable via raw ``gpuAddress`` dereference.

        Buffers registered here are declared to every subsequent compute
        encoder with ``useResource:usage:`` (read+write) so kernels that
        chase descriptor-embedded addresses (BVH / mesh queries) get
        residency and hazard tracking without an encoder binding. Idempotent
        per buffer object; entries live for the dispatcher's lifetime unless
        removed via :meth:`remove_resident_resource`.
        """
        if id(mtl_buf) in self._resident_ids:
            return
        self._resident_ids.add(id(mtl_buf))
        self._resident_resources.append(mtl_buf)

    def remove_resident_resource(self, mtl_buf) -> None:
        """Drop ``mtl_buf`` from the resident set (e.g. on ``Bvh.__del__``)."""
        if id(mtl_buf) not in self._resident_ids:
            return
        self._resident_ids.discard(id(mtl_buf))
        self._resident_resources = [r for r in self._resident_resources if r is not mtl_buf]
        # Force a full re-apply on the next dispatch — the applied-count
        # bookkeeping is positional and just went stale.
        self._resident_encoder = None
        self._resident_applied = 0

    def _apply_resident_resources(self, encoder) -> None:
        """Declare any not-yet-declared resident resources to ``encoder``."""
        n = len(self._resident_resources)
        if n == 0:
            return
        if self._resident_encoder is encoder and self._resident_applied >= n:
            return
        start = self._resident_applied if self._resident_encoder is encoder else 0
        usage = self._usage_read_write
        for r in self._resident_resources[start:]:
            encoder.useResource_usage_(r, usage)
        self._resident_encoder = encoder
        self._resident_applied = n

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
        if self._record_state is not None:
            # Blit commands cannot be encoded into a compute ICB, but the
            # compute-kernel fill can — record it so the zero re-runs on
            # every replay (a replayed atomic accumulator must restart at 0).
            self.device_fill(mtl_buf, 0, nbytes)
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

    # ------------------------------------------------------------------
    # Device-side host-op equivalents (recordable into ICB graphs)
    # ------------------------------------------------------------------

    def _device_op(self, entry_point: str):
        return self.compile(_DEVICE_OPS_SOURCE, entry_point)

    @staticmethod
    def _op_tg(n: int) -> tuple[int, int, int]:
        return (min(256, n), 1, 1)

    def device_fill(self, mtl_buf, value: int, nbytes: int, offset: int = 0) -> None:
        """Fill ``nbytes`` of ``mtl_buf`` starting at byte ``offset`` with the
        byte ``value`` (C ``memset`` semantics) via a compute dispatch.

        Goes through :meth:`dispatch`, so inside a recording it is encoded
        into the ICB and re-runs on every replay — the correct semantics
        for fills captured inside a graph, which the host/blit paths can't
        provide.
        """
        if nbytes <= 0:
            return
        value &= 0xFF
        if offset % 4 == 0 and nbytes % 4 == 0:
            word = value * 0x01010101
            n = nbytes // 4
            pso = self._device_op("wp_fill4")
            bindings = [mtl_buf, _u32(word), _u32(offset // 4)]
        else:
            n = nbytes
            pso = self._device_op("wp_fill1")
            bindings = [mtl_buf, _u32(value), _u32(offset)]
        self.dispatch(pso, bindings, (n, 1, 1), self._op_tg(n), binding_modes=["w", None, None])

    def device_memtile(self, mtl_buf, pattern: bytes, reps: int, offset: int = 0) -> None:
        """Tile ``pattern`` ``reps`` times into ``mtl_buf`` at byte ``offset``
        via a compute dispatch (``wp_memtile_host`` equivalent, recordable)."""
        patlen = len(pattern)
        nbytes = patlen * reps
        if nbytes <= 0:
            return
        # All-same-byte patterns collapse to a plain fill (covers zeros and
        # e.g. float32 0x01010101-style splats).
        if pattern == bytes([pattern[0]]) * patlen:
            self.device_fill(mtl_buf, pattern[0], nbytes, offset)
            return
        if offset % 4 == 0 and patlen % 4 == 0:
            n = nbytes // 4
            pso = self._device_op("wp_tile4")
            bindings = [(pattern, patlen), mtl_buf, _u32(patlen // 4), _u32(offset // 4)]
        else:
            n = nbytes
            pso = self._device_op("wp_tile1")
            bindings = [(pattern, patlen), mtl_buf, _u32(patlen), _u32(offset)]
        self.dispatch(pso, bindings, (n, 1, 1), self._op_tg(n), binding_modes=[None, "w", None, None])

    def device_copy(self, dst_buf, src_buf, nbytes: int, dst_off: int = 0, src_off: int = 0) -> None:
        """Copy ``nbytes`` from ``src_buf`` (+``src_off``) to ``dst_buf``
        (+``dst_off``) via a compute dispatch (recordable ``memcpy``)."""
        if nbytes <= 0:
            return
        if dst_off % 4 == 0 and src_off % 4 == 0 and nbytes % 4 == 0:
            n = nbytes // 4
            pso = self._device_op("wp_copy4")
            bindings = [src_buf, dst_buf, _u32(src_off // 4), _u32(dst_off // 4)]
        else:
            n = nbytes
            pso = self._device_op("wp_copy1")
            bindings = [src_buf, dst_buf, _u32(src_off), _u32(dst_off)]
        self.dispatch(pso, bindings, (n, 1, 1), self._op_tg(n), binding_modes=["r", "w", None, None])

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
        beyond ``nbytes`` to every allocation (defaults to ``4096`` —
        see the guard comment below; set to ``0`` to disable). Useful
        as a diagnostic for out-of-bounds writes and as protection
        against known one-element overshoots in upstream kernels.
        """
        if nbytes <= 0:
            raise ValueError(f"MetalDispatcher.alloc: nbytes must be positive, got {nbytes}")
        # Default guard: 4 KiB. Several mujoco_warp kernels (notably
        # ``_efc_contact_init``, ``_efc_contact_update``,
        # ``update_constraint_efc``, ``solve_done``, …) write one
        # element past the end of their nominal output array — confirmed
        # via the canary sanitiser. MLX's larger memory pool masked the
        # overshoot; ``MTLDevice newBufferWithLength`` packs allocations
        # tightly so the OOB writes corrupted adjacent ``wp.array``s.
        # Padding every allocation pushes the overshoot into a harmless
        # guard region until the upstream kernels are fixed. Both settings
        # are read from the environment once, in ``__init__``.
        guard = self._guard_bytes
        # ``WARP_METAL_CANARY``: pre-fill the guard region with a sentinel
        # byte (default ``0xAB``). Combined with the dispatcher's
        # post-launch scan (see ``_check_canaries``), this points the
        # finger at any kernel that wrote past its bound MTLBuffer.
        canary_enabled = self._canary_enabled
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
            raise MetalDispatchError(f"MTLDevice newBufferWithLength failed for {alloc_size} bytes")
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
                    f"[canary] {label}: slot {slot} {name!r} (data={data_nbytes}B) clobbered guard at +{first_bad}B",
                    flush=True,
                )
        return offenders

    def dispatch(
        self,
        pso,
        bindings: list,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        binding_modes: list | None = None,
        reads_resident: bool = False,
        extra_resources: list | None = None,
    ) -> None:
        """Encode one compute dispatch onto the in-flight command buffer.

        ``bindings`` is a list whose i-th entry becomes argument-slot
        ``i`` in the kernel. Each entry is either:

        * An ``MTLBuffer`` — bound via ``setBuffer:offset:atIndex:``.
        * A ``(bytes-like, length)`` tuple — bound via
          ``setBytes:length:atIndex:`` (cheaper for small const args
          ≤4KB; avoids a buffer allocation).

        ``binding_modes`` is an optional parallel list of access modes
        (one per binding) used **only** by ICB recording to compute
        dependency chunks. Each entry is one of ``None`` /
        ``"r"`` / ``"w"`` / ``"rw"``. ``None`` (or ``binding_modes``
        omitted) keeps the safe per-command-barrier replay path.

        ``reads_resident`` declares that the kernel may read any of the
        dispatcher's resident resources through descriptor-embedded
        ``gpuAddress``es (BVH / mesh query kernels). Direct dispatch needs
        no extra work (residency is applied per encoder), but ICB
        recording folds the resident set into this command's resource
        list and read set so replay gets residency and RAW barriers
        against refit kernels that write those buffers.

        ``extra_resources`` is an optional list of ``(mtl_buffer, mode)``
        pairs for buffers the kernel accesses through raw ``gpuAddress``
        pointers (the ``__arg_ptrs`` bindless table) rather than bound
        argument slots. Each buffer is made resident for the dispatch
        (``useResource:usage:``) and participates in ICB dependency
        chunking with the given mode (``"r"`` / ``"w"`` / ``"rw"``), so
        RAW hazards against kernels that write those buffers through
        normal bindings still get barriers on replay.

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
            self._record_dispatch(
                pso,
                bindings,
                grid,
                threadgroup,
                binding_modes,
                reads_resident=reads_resident,
                extra_resources=extra_resources,
            )
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
        self._apply_resident_resources(encoder)
        if extra_resources:
            # Bindless (``__arg_ptrs``) accesses: the buffers are not bound
            # to argument slots, so the encoder must be told explicitly to
            # make them resident before the dispatch. ``useResource`` is
            # encoder-scoped, so repeats across dispatches on the same
            # encoder are redundant but harmless.
            usage_r = Metal.MTLResourceUsageRead
            usage_w = Metal.MTLResourceUsageWrite
            for buf, mode in extra_resources:
                usage = usage_r if mode == "r" else (usage_w if mode == "w" else (usage_r | usage_w))
                encoder.useResource_usage_(buf, usage)
                # Retain until the command buffer completes — bindless
                # buffers never enter the bound-buffer runs below, so
                # they'd otherwise miss the in-flight ref that keeps
                # transient allocations alive while the GPU reads them.
                self._inflight_refs.append(buf)
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
        rows = [(name, slot[0], slot[1], slot[2]) for name, slot in self._dispatch_stats.items()]
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

    def has_pending_work(self) -> bool:
        """Return ``True`` if any recorded-but-unsynced GPU work exists
        (an open command buffer or committed-but-undrained buffers)."""
        return self._cmd_buf is not None or bool(self._pending_commits)

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
            # Dependency-aware chunking. Each closed chunk is a
            # (start, length) tuple of consecutive command indices
            # known to have NO read/write conflict on any shared
            # buffer; commands within a chunk can execute concurrently
            # at replay time. ``chunk_start`` is the index of the
            # first command in the still-open chunk; ``chunk_reads``
            # / ``chunk_writes`` track which buffer pointers the open
            # chunk's commands have touched so far. The next dispatch
            # closes the open chunk if it would conflict.
            "chunks": [],
            "chunk_start": 0,
            "chunk_reads": set(),
            "chunk_writes": set(),
            # Set to ``True`` the first time a caller passes
            # ``binding_modes``. When ``False``, we fall back to the
            # safe per-command-range replay; chunking is meaningless
            # unless someone actually told us which bindings are
            # read-only vs written.
            "has_modes": False,
            # GPU-conditional regions: closed entries are
            # ``(cmd_lo, cmd_hi, flag_buf, flag_offset)`` command-index
            # ranges; ``region_open`` holds the in-progress
            # ``(cmd_lo, flag_buf, flag_offset)`` between
            # ``begin_gated_region`` and ``end_gated_region``.
            "regions": [],
            "region_open": None,
            # Per-command kernel entry names for diagnostics
            # (``replay_timed`` attribution).
            "names": [],
        }

    def end_record(self) -> MetalGraph:
        """Exit recording mode and return the captured :class:`MetalGraph`."""
        if self._record_state is None:
            raise MetalDispatchError("Not currently recording an ICB graph")
        st = self._record_state
        self._record_state = None
        if st["region_open"] is not None:
            raise MetalDispatchError("end_record() with a gated region still open; call end_gated_region() first")
        # Close the still-open chunk. Only emit chunk info if at least
        # one dispatch passed ``binding_modes``; otherwise the chunks
        # would all conservatively collapse to length-1 anyway and we
        # gain nothing -- falling back to the per-command-range replay
        # is simpler and identical in behaviour.
        chunks: list[tuple[int, int]] | None
        if st["has_modes"]:
            if st["count"] > st["chunk_start"]:
                st["chunks"].append((st["chunk_start"], st["count"] - st["chunk_start"]))
            chunks = st["chunks"] or None
        else:
            chunks = None
        regions = None
        if st["regions"]:
            if chunks is None:
                # No ``binding_modes`` ever recorded, so ``st["chunks"]``
                # holds region-boundary closes without any conflict
                # analysis inside them — unsafe to execute concurrently.
                # Synthesize the per-command fallback explicitly so gated
                # regions still map onto executable ranges.
                chunks = [(i, 1) for i in range(st["count"])]
            regions = self._build_gated_regions(st["regions"], chunks, st["owned_buffers"])
        return MetalGraph(
            icb=st["icb"],
            count=st["count"],
            resources=st["resources"],
            owned_buffers=st["owned_buffers"],
            signature=st["signature"],
            chunks=chunks,
            regions=regions,
            names=st["names"],
        )

    @staticmethod
    def _close_open_chunk(st) -> None:
        """Close the recording's open dependency chunk (region boundaries)."""
        if st["count"] > st["chunk_start"]:
            st["chunks"].append((st["chunk_start"], st["count"] - st["chunk_start"]))
            st["chunk_start"] = st["count"]
            st["chunk_reads"] = set()
            st["chunk_writes"] = set()

    def begin_gated_region(self, flag_buf, flag_offset: int = 0) -> None:
        """Start a GPU-conditional region inside an active recording.

        Every dispatch recorded until the matching
        :meth:`end_gated_region` executes on replay only while the
        int32 at byte ``flag_offset`` of ``flag_buf`` reads nonzero —
        evaluated on the GPU timeline at the region's position in the
        graph, so a command *earlier in the same graph* (or an earlier
        replay) can flip the flag and skip the region's work. This is
        the Metal equivalent of a CUDA conditional graph node; the
        canonical use is an iterative solver whose per-world
        convergence kernel decrements a "worlds still solving" counter.

        Skipped regions still cost their inter-chunk barriers plus one
        tiny gate dispatch, but no kernel threads launch — the
        ``executeCommandsInBuffer:indirectBuffer:`` ranges collapse to
        zero length.

        Regions cannot nest. An empty region (no dispatches recorded
        inside) is dropped silently.
        """
        st = self._record_state
        if st is None:
            raise MetalDispatchError("begin_gated_region() is only valid while recording")
        if st["region_open"] is not None:
            raise MetalDispatchError("Gated regions cannot nest")
        # Region boundaries must coincide with chunk boundaries: the
        # gate's range table covers whole chunks only.
        self._close_open_chunk(st)
        st["region_open"] = (st["count"], flag_buf, flag_offset)

    def end_gated_region(self) -> None:
        """Close the gated region opened by :meth:`begin_gated_region`."""
        st = self._record_state
        if st is None or st["region_open"] is None:
            raise MetalDispatchError("end_gated_region() without a matching begin_gated_region()")
        cmd_lo, flag_buf, flag_offset = st["region_open"]
        st["region_open"] = None
        if st["count"] == cmd_lo:
            return
        self._close_open_chunk(st)
        st["regions"].append((cmd_lo, st["count"], flag_buf, flag_offset))

    def _build_gated_regions(self, cmd_regions: list, chunks: list, owned_buffers: list) -> list:
        """Map recorded command-index regions onto chunk indices and
        allocate each region's execution-range buffers.

        ``full_buf`` holds the chunks' real ``{location, length}``
        ranges (immutable); ``ranges_buf`` is the gate kernel's output
        the replay encoder actually points
        ``executeCommandsInBuffer:indirectBuffer:`` at. Both live as
        long as the graph via ``owned_buffers``.
        """
        Metal = self._Metal
        regions = []
        for cmd_lo, cmd_hi, flag_buf, flag_offset in cmd_regions:
            # begin/end_gated_region close the open chunk, so region
            # bounds align exactly with chunk starts.
            chunk_lo = next(i for i, (s, _n) in enumerate(chunks) if s == cmd_lo)
            chunk_hi = chunk_lo
            while chunk_hi < len(chunks) and chunks[chunk_hi][0] < cmd_hi:
                chunk_hi += 1
            n = chunk_hi - chunk_lo
            full_buf, full_addr = self.alloc(8 * n)
            ranges_buf, _ = self.alloc(8 * n)
            packed = (ctypes.c_uint32 * (2 * n)).from_address(full_addr)
            for j in range(n):
                start, length = chunks[chunk_lo + j]
                packed[2 * j] = start
                packed[2 * j + 1] = length
            owned_buffers.extend((full_buf, ranges_buf))
            gate_grid = Metal.MTLSizeMake(n, 1, 1)
            gate_tg = Metal.MTLSizeMake(min(n, 64), 1, 1)
            regions.append((chunk_lo, chunk_hi, flag_buf, flag_offset, ranges_buf, full_buf, gate_grid, gate_tg))
        return regions

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
        # BVH / mesh buffers reached via descriptor gpuAddresses — the
        # recorded resource list covers those known at record time; this
        # covers ones registered since.
        self._apply_resident_resources(encoder)
        # Keep both the graph and its owned buffers alive until the
        # cmd buffer completes — Metal can dereference them at any
        # point during GPU execution.
        self._inflight_refs.append(graph._icb)
        if graph._owned_buffers:
            self._inflight_refs.extend(graph._owned_buffers)
        # Chunked range execution: one ``executeCommandsInBuffer``
        # per chunk of mutually-independent commands, with a barrier
        # only BETWEEN chunks. Without chunks (caller never passed
        # ``binding_modes``), fall back to per-command ranges + per-
        # command barriers. See :class:`MetalGraph` docstring for the
        # correctness argument.
        nsrange = self._NSMakeRange
        scope = self._barrier_scope_buffers
        icb = graph._icb
        chunks = graph._chunks
        if chunks is not None and graph._regions:
            # Region-aware walk: chunks inside a gated region execute
            # via GPU-resolved indirect ranges, preceded (once per
            # region) by the gate dispatch that writes them. The gate
            # must observe flag writes from earlier commands in this
            # same encoder — memoryBarrierWithScope: covers both that
            # and the command processor's later range read (verified
            # empirically; see test_metal_launch.py gate tests).
            gate_pso = self._device_op("wp_icb_gate")
            exec_indirect = encoder.executeCommandsInBuffer_indirectBuffer_indirectBufferOffset_
            regions = graph._regions
            nregions = len(regions)
            ridx = 0
            last = len(chunks) - 1
            for i, (start, length) in enumerate(chunks):
                if ridx < nregions:
                    reg = regions[ridx]
                    if i == reg[0]:
                        _lo, _hi, flag_buf, flag_off, ranges_buf, full_buf, gate_grid, gate_tg = reg
                        encoder.setComputePipelineState_(gate_pso)
                        encoder.setBuffer_offset_atIndex_(flag_buf, flag_off, 0)
                        encoder.setBuffer_offset_atIndex_(ranges_buf, 0, 1)
                        encoder.setBuffer_offset_atIndex_(full_buf, 0, 2)
                        encoder.dispatchThreads_threadsPerThreadgroup_(gate_grid, gate_tg)
                        encoder.memoryBarrierWithScope_(scope)
                    if reg[0] <= i < reg[1]:
                        exec_indirect(icb, reg[4], (i - reg[0]) * 8)
                        if i + 1 == reg[1]:
                            ridx += 1
                        if i < last:
                            encoder.memoryBarrierWithScope_(scope)
                        continue
                encoder.executeCommandsInBuffer_withRange_(icb, nsrange(start, length))
                if i < last:
                    encoder.memoryBarrierWithScope_(scope)
        elif chunks is not None:
            last = len(chunks) - 1
            for i, (start, length) in enumerate(chunks):
                encoder.executeCommandsInBuffer_withRange_(icb, nsrange(start, length))
                if i < last:
                    encoder.memoryBarrierWithScope_(scope)
        else:
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

    # Hardware bound on ``MTLCounterSampleBuffer`` length is 32 KiB =
    # 4096 uint64 timestamps = 2048 encoders (verified empirically on
    # M3 Max). Batch profiled commands so any graph size works.
    _TIMED_BATCH = 2048

    def replay_timed(self, graph: MetalGraph, iters: int = 1) -> list[tuple[str, float]]:
        """Execute ``graph`` once per ``iters`` with per-command GPU timing.

        Diagnostics-only replay: each ICB command runs in its own
        compute-pass encoder with GPU timestamp samples at the encoder
        stage boundaries, fence-chained so commands serialize exactly
        like the per-command-barrier replay. Returns one
        ``(entry_name, gpu_ns)`` row per command in graph order, with
        durations averaged over ``iters``.

        Compared to :meth:`replay` this measures pure GPU execution
        time per kernel — kernel cost separated from the Python /
        encoder dispatch overhead ``dispatch_stats`` mixes in. Expect
        a per-encoder floor of ~5-10 µs (stage-boundary sampling wraps
        the whole encoder); rows at the floor are dispatch-bound, rows
        above it are real GPU work.

        Caveats: gated regions are ignored (every command executes,
        ungated), and timing runs synchronously (``waitUntilCompleted``
        per batch of ≤2048 commands). Apple GPUs only support
        stage-boundary counter sampling; raises
        :class:`MetalDispatchError` where even that is unavailable.
        """
        if self._record_state is not None:
            raise MetalDispatchError("Cannot replay while a recording is open")
        if graph.count == 0:
            return []
        Metal = self._Metal
        if not self._device.supportsCounterSampling_(Metal.MTLCounterSamplingPointAtStageBoundary):
            raise MetalDispatchError("Device does not support stage-boundary counter sampling")
        ts_set = None
        for cs in self._device.counterSets():
            if str(cs.name()) == "timestamp":
                ts_set = cs
                break
        if ts_set is None:
            raise MetalDispatchError("Device exposes no timestamp counter set")
        # Drain in-flight work so the timed run measures only the graph.
        self.sync()

        names = graph._names or ["<unknown>"] * graph.count
        totals = [0.0] * graph.count
        usage = Metal.MTLResourceUsageRead | Metal.MTLResourceUsageWrite
        fence = self._device.newFence()
        sdesc = Metal.MTLCounterSampleBufferDescriptor.alloc().init()
        sdesc.setCounterSet_(ts_set)
        sdesc.setStorageMode_(Metal.MTLStorageModeShared)
        nsrange = self._NSMakeRange

        for _ in range(iters):
            for batch_lo in range(0, graph.count, self._TIMED_BATCH):
                batch_n = min(self._TIMED_BATCH, graph.count - batch_lo)
                sdesc.setSampleCount_(2 * batch_n)
                sbuf, err = self._device.newCounterSampleBufferWithDescriptor_error_(sdesc, None)
                if sbuf is None:
                    raise MetalDispatchError(f"newCounterSampleBufferWithDescriptor failed: {err}")
                cmd_buf = self._command_queue.commandBuffer()
                if cmd_buf is None:
                    raise MetalDispatchError("MTLCommandQueue commandBuffer returned None")
                for j in range(batch_n):
                    pdesc = Metal.MTLComputePassDescriptor.computePassDescriptor()
                    att = pdesc.sampleBufferAttachments().objectAtIndexedSubscript_(0)
                    att.setSampleBuffer_(sbuf)
                    att.setStartOfEncoderSampleIndex_(2 * j)
                    att.setEndOfEncoderSampleIndex_(2 * j + 1)
                    encoder = cmd_buf.computeCommandEncoderWithDescriptor_(pdesc)
                    if encoder is None:
                        raise MetalDispatchError("computeCommandEncoderWithDescriptor returned None")
                    if j > 0:
                        encoder.waitForFence_(fence)
                    for r in graph._resources:
                        encoder.useResource_usage_(r, usage)
                    self._apply_resident_resources(encoder)
                    encoder.executeCommandsInBuffer_withRange_(graph._icb, nsrange(batch_lo + j, 1))
                    encoder.updateFence_(fence)
                    encoder.endEncoding()
                # Correlated CPU/GPU timestamp pairs around execution
                # turn raw GPU ticks into nanoseconds (identity on
                # Apple silicon, but don't bake that assumption in).
                cpu_a, gpu_a = self._device.sampleTimestamps_gpuTimestamp_(None, None)
                cmd_buf.commit()
                cmd_buf.waitUntilCompleted()
                cpu_b, gpu_b = self._device.sampleTimestamps_gpuTimestamp_(None, None)
                scale = (cpu_b - cpu_a) / (gpu_b - gpu_a) if gpu_b != gpu_a else 1.0
                data = sbuf.resolveCounterRange_(nsrange(0, 2 * batch_n))
                if data is None:
                    raise MetalDispatchError("resolveCounterRange returned None")
                stamps = (ctypes.c_uint64 * (2 * batch_n)).from_buffer_copy(data.bytes().tobytes())
                for j in range(batch_n):
                    totals[batch_lo + j] += (stamps[2 * j + 1] - stamps[2 * j]) * scale

        return [(names[i], totals[i] / iters) for i in range(graph.count)]

    def _record_dispatch(
        self,
        pso,
        bindings: list,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        binding_modes: list | None = None,
        reads_resident: bool = False,
        extra_resources: list | None = None,
    ) -> None:
        """Encode one dispatch into the active recording's next ICB slot.

        ``setBytes``-style bindings (``(bytes, length)`` tuples) are
        materialised as fresh shared-storage MTLBuffers so the ICB
        can reference them via ``setKernelBuffer:offset:atIndex:``.
        Apple's ICB compute commands have no setBytes equivalent.

        ``binding_modes`` -- when non-``None`` -- enables dependency
        chunking: each entry is ``None`` / ``"r"`` / ``"w"`` / ``"rw"``
        for the corresponding binding. setBytes args ignore the mode
        (every setBytes allocates a private MTLBuffer per command, so
        it can never alias another command's buffer). If this command
        conflicts on any shared buffer with the still-open chunk, the
        chunk is closed and a new one starts here.
        """
        st = self._record_state
        assert st is not None  # caller checked
        if st["count"] >= st["max"]:
            raise MetalDispatchError(
                f"ICB recording overflowed allocation of {st['max']} commands — "
                "raise begin_record(max_commands=...) for this workload."
            )
        # Compute this command's read/write sets over shared buffers.
        # Done BEFORE encoding the command so we know whether to close
        # the open chunk first. We use ``id(entry)`` as the buffer key
        # -- two ICB commands binding the same Python-level MTLBuffer
        # object identify aliasing. setBytes args produce a fresh
        # MTLBuffer per command (see below) so they never alias.
        cmd_reads: set = set()
        cmd_writes: set = set()
        if reads_resident:
            # Descriptor-mediated reads (BVH / mesh queries): the kernel can
            # touch any resident buffer without binding it. Fold the whole
            # resident set into this command's read set so the chunker
            # serialises it against refit kernels that WRITE those buffers
            # through normal bindings, and into the graph resources so
            # replay declares them via ``useResource``.
            for r in self._resident_resources:
                cmd_reads.add(id(r))
        if extra_resources:
            # Bindless (``__arg_ptrs``) buffers: precise per-buffer modes,
            # folded exactly like bound-argument modes so RAW/WAR/WAW
            # hazards against other commands get chunk barriers.
            for buf, mode in extra_resources:
                ptr = id(buf)
                if mode in ("r", "rw"):
                    cmd_reads.add(ptr)
                if mode in ("w", "rw"):
                    cmd_writes.add(ptr)
        if binding_modes is not None:
            st["has_modes"] = True
            # A short modes list would silently drop trailing buffers from
            # the conflict analysis — dependency chunks would then omit
            # barriers for exactly those buffers. Fail loudly instead.
            if len(binding_modes) != len(bindings):
                raise MetalDispatchError(
                    f"binding_modes length {len(binding_modes)} != bindings length "
                    f"{len(bindings)} — dependency chunking would silently miss "
                    "conflicts on the unpaired buffers."
                )
            for entry, mode in zip(bindings, binding_modes):
                if mode is None or isinstance(entry, tuple):
                    continue
                ptr = id(entry)
                if mode in ("r", "rw"):
                    cmd_reads.add(ptr)
                if mode in ("w", "rw"):
                    cmd_writes.add(ptr)
            # Conflict check vs the still-open chunk.
            chunk_reads = st["chunk_reads"]
            chunk_writes = st["chunk_writes"]
            # RAW: this cmd reads something the chunk wrote.
            # WAR: this cmd writes something the chunk read.
            # WAW: this cmd writes something the chunk wrote.
            conflict = (
                bool(cmd_reads & chunk_writes) or bool(cmd_writes & chunk_reads) or bool(cmd_writes & chunk_writes)
            )
            if conflict and st["count"] > st["chunk_start"]:
                st["chunks"].append((st["chunk_start"], st["count"] - st["chunk_start"]))
                st["chunk_start"] = st["count"]
                st["chunk_reads"] = cmd_reads.copy()
                st["chunk_writes"] = cmd_writes.copy()
            else:
                chunk_reads |= cmd_reads
                chunk_writes |= cmd_writes
        # Encode the command.
        Metal = self._Metal
        cmd = st["icb"].indirectComputeCommandAtIndex_(st["count"])
        cmd.setComputePipelineState_(pso)
        res_set = st["resources_set"]
        resources = st["resources"]
        owned = st["owned_buffers"]
        if reads_resident:
            for r in self._resident_resources:
                rid = id(r)
                if rid not in res_set:
                    res_set.add(rid)
                    resources.append(r)
        if extra_resources:
            for buf, _mode in extra_resources:
                bid = id(buf)
                if bid not in res_set:
                    res_set.add(bid)
                    resources.append(buf)
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
        st["names"].append(self._pso_name.get(id(pso), "<unknown>"))
        st["count"] += 1
