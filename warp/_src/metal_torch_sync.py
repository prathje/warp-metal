# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-side ordering between Torch (MPS) and Warp's native Metal dispatcher.

Metal does not order work across ``MTLCommandQueue`` objects. Warp and Torch
each had their own queue, so a Torch kernel could run concurrently with a
committed Warp command buffer that writes or reads the same buffer (a stale
``wp.to_torch`` read, or a Torch write clobbering the input of a pending
launch).

Once a buffer is shared with Torch, Warp encodes its command buffers on
Torch's MPS command queue instead, and keeps one invariant: **Torch's pending
command buffer is committed right before Warp commits a command buffer that
touches a shared buffer** (see ``MetalDispatcher._commit_cmd_buf``).
Commit order on one queue then matches program order in both directions:

* Torch -> Warp: Torch work issued before a launch is committed before the
  launch's command buffer.
* Warp -> Torch: launches that touch a shared buffer are committed when
  ``wp.launch`` returns, so Torch work issued afterwards is committed later.

Ordering between command buffers on one queue is the same guarantee Torch
relies on between its own command buffers (MPS commits every few ops), so
Warp's work gets exactly Torch's own same-stream semantics. A cross-queue
``MTLSharedEvent`` handshake also orders the work, but every wait costs a GPU
round trip between the queues (~140 us per launch measured on an M-series GPU,
against ~1.5 us for a commit), so the shared queue is used instead.

Torch exposes neither its queue nor a commit in Python. ``at::mps::MPSHooks``
exports both as ``TORCH_API`` C++ symbols (Torch 2.8 to 2.14), which this
module calls through ctypes (no compiler needed):

* ``MPSHooks::commitStream()``: ``synchronize(SyncType::COMMIT)`` on the
  default stream (Torch keeps encoding into a fresh command buffer).
* ``MPSHooks::getDispatchQueue()``: the stream's serial dispatch queue.
* ``MPSHooks::getCommandBuffer()``: the stream's ``MPSCommandBuffer``, whose
  ``commandQueue`` is the queue Warp encodes on.

Torch encodes only on its serial dispatch queue, and ``MPSStream`` itself
takes no lock, so every call into the stream runs on that queue through
``dispatch_sync_f`` (calling ``commitStream`` from another thread while a
Torch op encodes crashed). Code already running on the queue calls directly
(``dispatch_get_specific``), as ``dispatch_sync`` onto the current serial
queue would deadlock. A CPU wait for Torch's work goes through a
``torch.mps.Event`` (record, then synchronize): Torch records it on its queue
and reports device errors as Python exceptions.

Binding runs a canary: a Torch op, a commit, and a check that the command
buffer holding the op was committed. If any of this fails (another Torch
build or version), Warp falls back to CPU fences (a Torch wait before, and a
Warp device sync after, every command buffer that touches a shared buffer):
correct but slower. ``WARP_METAL_TORCH_SYNC`` selects the mode:

* ``gpu`` (default): the shared queue, falling back to ``cpu`` if unavailable.
* ``cpu``: always use CPU fences.
* ``off``: no ordering beyond committing shared launches at once (the racy
  pre-fix behaviour; for A/B measurements only).
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
import threading
import warnings

_MODES = ("gpu", "cpu", "off")

# Torch releases with the MPSHooks entry points verified to match (symbols,
# no locking in ``MPSStream::synchronize``, ``commitStream`` = COMMIT).
_VERIFIED_TORCH = ((2, 8), (2, 9), (2, 11), (2, 13), (2, 14))

_SYM_COMMIT_STREAM = "_ZNK2at3mps8MPSHooks12commitStreamEv"
_SYM_GET_DISPATCH_QUEUE = "_ZNK2at3mps8MPSHooks16getDispatchQueueEv"
_SYM_GET_COMMAND_BUFFER = "_ZNK2at3mps8MPSHooks16getCommandBufferEv"

# MTLCommandBufferStatus: NotEnqueued, Enqueued, Committed, Scheduled, ...
_STATUS_COMMITTED = 2


def requested_mode() -> str:
    """The ordering mode requested through ``WARP_METAL_TORCH_SYNC``."""
    raw = os.environ.get("WARP_METAL_TORCH_SYNC", "").strip().lower()
    if not raw:
        return "gpu"
    if raw not in _MODES:
        warnings.warn(
            f"Ignoring WARP_METAL_TORCH_SYNC={raw!r}: expected one of {', '.join(_MODES)}; using 'gpu'.",
            stacklevel=2,
        )
        return "gpu"
    return raw


def torch_mps_in_use() -> bool:
    """Whether Torch is imported and has a usable MPS device.

    Never imports Torch: a buffer shared through DLPack with something other
    than Torch has no Torch queue to order against.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _torch_version(version: str) -> tuple[int, int] | None:
    m = re.match(r"(\d+)\.(\d+)", version)
    return (int(m.group(1)), int(m.group(2))) if m else None


_waits = threading.local()


def wait_for_torch() -> None:
    """Block until Torch's MPS work issued so far has completed.

    Records a per-thread ``torch.mps.Event`` (on Torch's dispatch queue) and
    waits for it; ``torch.mps.synchronize()`` touches the stream without the
    queue and is not safe while another thread runs Torch ops.
    """
    import torch  # noqa: PLC0415

    ev = getattr(_waits, "event", None)
    if ev is None:
        ev = torch.mps.Event()
        _waits.event = ev
    ev.record()
    ev.synchronize()


class TorchMPSStream:
    """ctypes binding to Torch's default MPS stream (see the module docstring)."""

    def __init__(self) -> None:
        import objc  # noqa: PLC0415
        import torch  # noqa: PLC0415

        version = _torch_version(torch.__version__)
        if version not in _VERIFIED_TORCH:
            warnings.warn(
                f"Warp's Metal/Torch ordering is verified with Torch "
                f"{', '.join('.'.join(map(str, v)) for v in _VERIFIED_TORCH)}, not {torch.__version__}; "
                "binding Torch's MPS queue anyway (a canary check falls back to CPU fences on mismatch).",
                stacklevel=4,
            )
        lib = ctypes.CDLL(os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cpu.dylib"))
        # ``const`` member functions that ignore ``this`` (they act on the
        # default stream), so ``dispatch_sync_f`` can call them with a null
        # context as plain ``void (*)(void*)`` functions.
        self._commit_fn = ctypes.cast(getattr(lib, _SYM_COMMIT_STREAM), ctypes.c_void_p).value
        get_queue = getattr(lib, _SYM_GET_DISPATCH_QUEUE)
        get_queue.restype = ctypes.c_void_p
        get_queue.argtypes = [ctypes.c_void_p]
        get_cb = getattr(lib, _SYM_GET_COMMAND_BUFFER)
        get_cb.restype = ctypes.c_void_p
        get_cb.argtypes = [ctypes.c_void_p]
        self._get_cb = get_cb

        libdispatch = ctypes.CDLL("/usr/lib/system/libdispatch.dylib")
        self._dispatch_sync_f = libdispatch.dispatch_sync_f
        self._dispatch_sync_f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self._dispatch_sync_f.restype = None
        self._get_specific = libdispatch.dispatch_get_specific
        self._get_specific.argtypes = [ctypes.c_void_p]
        self._get_specific.restype = ctypes.c_void_p
        set_specific = libdispatch.dispatch_queue_set_specific
        set_specific.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        set_specific.restype = None
        self._call_fn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(self._commit_fn)
        self._objc = objc

        # Torch creates its MPS stream lazily; a tiny op forces it into
        # existence before its queue is read.
        torch.empty(1, device="mps")
        self._queue = get_queue(None)
        if not self._queue:
            raise RuntimeError("at::mps::MPSHooks::getDispatchQueue() returned null")
        # Mark Torch's queue so ``_on_queue`` can tell when it already runs
        # on it. The key is the address of a buffer this binding keeps alive.
        self._key = ctypes.create_string_buffer(1)
        self._key_ptr = ctypes.addressof(self._key)
        set_specific(self._queue, self._key_ptr, self._key_ptr, None)

    def _on_queue(self, fn_ptr: int, call) -> None:
        if self._get_specific(self._key_ptr):
            call(None)
        else:
            self._dispatch_sync_f(self._queue, None, fn_ptr)

    def command_buffer(self):
        """Torch's current ``MPSCommandBuffer``, read on its dispatch queue."""
        out = []

        # A Python callback on Torch's queue: needs the GIL, which Torch ops
        # release before entering the queue. Only used while binding.
        @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        def read(_ctx):
            out.append(self._get_cb(None))

        fn_ptr = ctypes.cast(read, ctypes.c_void_p).value
        self._on_queue(fn_ptr, read)
        if not out or not out[0]:
            raise RuntimeError("at::mps::MPSHooks::getCommandBuffer() returned null")
        return self._objc.objc_object(c_void_p=out[0])

    def command_queue(self):
        """Torch's ``MTLCommandQueue`` (the queue of its MPSCommandBuffer)."""
        queue = self.command_buffer().commandQueue()
        if queue is None:
            raise RuntimeError("Torch's MPS command buffer has no command queue")
        return queue

    def commit(self) -> None:
        """Commit Torch's pending command buffer without waiting, on Torch's
        dispatch queue."""
        self._on_queue(self._commit_fn, self._call_fn)

    def check(self) -> None:
        """Canary: a Torch op followed by :meth:`commit` must leave the
        command buffer that holds the op committed."""
        import torch  # noqa: PLC0415

        x = torch.ones(16, device="mps")
        x.add_(1.0)
        cb = self.command_buffer().commandBuffer()
        if cb is None:
            raise RuntimeError("Torch's MPSCommandBuffer has no underlying MTLCommandBuffer")
        self.commit()
        status = int(cb.status())
        if status < _STATUS_COMMITTED:
            raise RuntimeError(f"MPSHooks::commitStream() did not commit Torch's command buffer (status {status})")
        wait_for_torch()


_stream_binding: TorchMPSStream | None = None
_stream_error: str | None = None
_stream_lock = threading.Lock()


def get_torch_stream() -> TorchMPSStream | None:
    """The process-wide :class:`TorchMPSStream`, or ``None`` when it cannot be
    bound or its canary fails (the reason is in :func:`unavailable_reason`)."""
    global _stream_binding, _stream_error
    with _stream_lock:
        if _stream_binding is None and _stream_error is None:
            try:
                stream = TorchMPSStream()
                stream.check()
                _stream_binding = stream
            except Exception as e:
                _stream_error = f"{type(e).__name__}: {e}"
    return _stream_binding


def unavailable_reason() -> str | None:
    return _stream_error
