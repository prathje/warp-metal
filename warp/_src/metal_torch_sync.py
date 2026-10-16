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

Torch exposes neither its queue nor a commit in Python. ``at::mps`` exports
both as ``TORCH_API`` C++ symbols, which this module calls through ctypes (no
compiler needed):

* ``at::mps::getCurrentMPSStream()``
* ``at::mps::MPSStream::commandBuffer()`` (its ``commandQueue`` is the queue)
* ``at::mps::MPSStream::synchronize(SyncType::COMMIT)``

If any of them cannot be resolved (another Torch build or version), Warp
falls back to CPU fences (``torch.mps.synchronize()`` before, and a Warp
device sync after, every command buffer that touches a shared buffer):
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
import warnings

_MODES = ("gpu", "cpu", "off")

# at::mps::SyncType::COMMIT in ATen/mps/MPSStream.h
# (NONE, COMMIT, COMMIT_AND_WAIT, COMMIT_AND_CONTINUE, COMMIT_ADAPTIVE).
_SYNC_TYPE_COMMIT_DEFAULT = 1

_SYM_GET_STREAM = "_ZN2at3mps19getCurrentMPSStreamEv"
_SYM_COMMAND_BUFFER = "_ZN2at3mps9MPSStream13commandBufferEv"
_SYM_SYNCHRONIZE = "_ZN2at3mps9MPSStream11synchronizeENS0_8SyncTypeE"


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


def _sync_type_commit(torch_dir: str) -> int:
    """``SyncType::COMMIT``'s value, read from the installed header when
    present (guards against a reordered enum), else the known default."""
    header = os.path.join(torch_dir, "include", "ATen", "mps", "MPSStream.h")
    try:
        with open(header, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return _SYNC_TYPE_COMMIT_DEFAULT
    m = re.search(r"enum\s+class\s+SyncType\s*\{(.*?)\}", text, re.S)
    if m is None:
        raise RuntimeError("at::mps::SyncType not found in MPSStream.h")
    body = re.sub(r"//[^\n]*", "", m.group(1))
    names = [n.strip() for n in body.split(",") if n.strip()]
    if any("=" in n for n in names) or "COMMIT" not in names:
        raise RuntimeError(f"unexpected at::mps::SyncType layout: {names}")
    return names.index("COMMIT")


class TorchMPSStream:
    """ctypes binding to Torch's current MPS stream (see the module docstring)."""

    def __init__(self) -> None:
        import objc  # noqa: PLC0415
        import torch  # noqa: PLC0415

        torch_dir = os.path.dirname(torch.__file__)
        lib_path = os.path.join(torch_dir, "lib", "libtorch_cpu.dylib")
        lib = ctypes.CDLL(lib_path)
        self._get_stream = getattr(lib, _SYM_GET_STREAM)
        self._get_stream.restype = ctypes.c_void_p
        self._get_stream.argtypes = []
        self._command_buffer = getattr(lib, _SYM_COMMAND_BUFFER)
        self._command_buffer.restype = ctypes.c_void_p
        self._command_buffer.argtypes = [ctypes.c_void_p]
        self._synchronize = getattr(lib, _SYM_SYNCHRONIZE)
        self._synchronize.restype = None
        self._synchronize.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._commit_type = _sync_type_commit(torch_dir)
        self._objc = objc
        # Torch creates its MPS stream lazily; a tiny op forces it (and its
        # device) into existence before its pointer is read.
        torch.empty(1, device="mps")
        self._stream = self._get_stream()
        if not self._stream:
            raise RuntimeError("at::mps::getCurrentMPSStream() returned null")

    def stream_ptr(self) -> int:
        return int(self._get_stream() or 0)

    def command_queue(self):
        """Torch's ``MTLCommandQueue`` (the queue of its MPSCommandBuffer)."""
        cb_ptr = self._command_buffer(self._stream)
        if not cb_ptr:
            raise RuntimeError("at::mps::MPSStream::commandBuffer() returned null")
        cb = self._objc.objc_object(c_void_p=cb_ptr)
        queue = cb.commandQueue()
        if queue is None:
            raise RuntimeError("Torch's MPS command buffer has no command queue")
        return queue

    def commit(self) -> None:
        """Commit Torch's pending command buffer without waiting
        (``synchronize(SyncType::COMMIT)``; Torch keeps encoding into a fresh
        one)."""
        self._synchronize(self._stream, self._commit_type)


_stream_binding: TorchMPSStream | None = None
_stream_error: str | None = None


def get_torch_stream() -> TorchMPSStream | None:
    """The process-wide :class:`TorchMPSStream`, or ``None`` when it cannot be
    bound (the reason is in :func:`unavailable_reason`)."""
    global _stream_binding, _stream_error
    if _stream_binding is None and _stream_error is None:
        try:
            _stream_binding = TorchMPSStream()
        except Exception as e:
            _stream_error = f"{type(e).__name__}: {e}"
    return _stream_binding


def unavailable_reason() -> str | None:
    return _stream_error
