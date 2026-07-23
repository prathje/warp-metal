# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import weakref
from typing import TYPE_CHECKING

import numpy

import warp
import warp._src.context

if TYPE_CHECKING:
    import torch

_wp_module_name_ = "warp.torch"


# return the warp device corresponding to a torch device
def device_from_torch(torch_device: torch.device | str) -> warp.Device:
    """Return the Warp device corresponding to a Torch device.

    Args:
        torch_device: Torch device identifier

    Raises:
        RuntimeError: Torch device does not have a corresponding Warp device
    """
    if type(torch_device) is str:
        warp_device = warp._src.context.runtime.device_map.get(torch_device)
        if warp_device is not None:
            return warp_device
        elif torch_device == "cuda":
            return warp._src.context.runtime.get_current_cuda_device()
        else:
            raise RuntimeError(f"Unsupported Torch device {torch_device}")
    else:
        try:
            if torch_device.type == "cuda":
                return warp._src.context.runtime.cuda_devices[torch_device.index]
            elif torch_device.type == "cpu":
                return warp._src.context.runtime.cpu_device
            elif torch_device.type == "mps":
                # Torch calls the Apple GPU "mps"; Warp registers the same
                # alias in its device map when the Metal backend is enabled.
                metal_device = warp._src.context.runtime.device_map.get("mps")
                if metal_device is not None:
                    return metal_device
                raise RuntimeError(
                    "Torch device 'mps' has no corresponding Warp device — enable the Metal "
                    "backend with wp.config.enable_metal = True before wp.init()"
                )
            else:
                raise RuntimeError(f"Unsupported Torch device type {torch_device.type}")
        except Exception as e:
            import torch  # noqa: PLC0415

            if not isinstance(torch_device, torch.device):
                raise ValueError("Argument must be a torch.device object or a string") from e
            raise


def device_to_torch(warp_device: warp.DeviceLike) -> str:
    """Return the Torch device string corresponding to a Warp device.

    Args:
        warp_device: An identifier that can be resolved to a :class:`warp.Device`.

    Raises:
        RuntimeError: The Warp device is not compatible with PyTorch.
    """
    device = warp.get_device(warp_device)
    if device.is_cpu or device.is_primary:
        return str(device)
    elif device.is_cuda and device.is_uva:
        # it's not a primary context, but torch can access the data ptr directly thanks to UVA
        return f"cuda:{device.ordinal}"
    elif device.is_metal:
        # Torch calls Apple's GPU "mps", not "metal".
        return "mps"
    raise RuntimeError(f"Warp device {device} is not compatible with torch")


def dtype_to_torch(warp_dtype):
    """Return the Torch dtype corresponding to a Warp dtype.

    Args:
        warp_dtype: A Warp data type that has a corresponding ``torch.dtype``.
            ``warp.uint16``, ``warp.uint32``, and ``warp.uint64`` are mapped
            to the signed integer ``torch.dtype`` of the same width.
    Raises:
        TypeError: Unable to find a corresponding PyTorch data type.
    """
    # initialize lookup table on first call to defer torch import
    if dtype_to_torch.type_map is None:
        import torch  # noqa: PLC0415

        dtype_to_torch.type_map = {
            warp.float16: torch.float16,
            warp.bfloat16: torch.bfloat16,
            warp.float32: torch.float32,
            warp.float64: torch.float64,
            warp.int8: torch.int8,
            warp.int16: torch.int16,
            warp.int32: torch.int32,
            warp.int64: torch.int64,
            warp.uint8: torch.uint8,
            # torch doesn't support unsigned ints bigger than 8 bits
            warp.uint16: torch.int16,
            warp.uint32: torch.int32,
            warp.uint64: torch.int64,
            warp.bool: torch.bool,
        }

    torch_dtype = dtype_to_torch.type_map.get(warp_dtype)
    if torch_dtype is not None:
        return torch_dtype
    else:
        raise TypeError(f"Cannot convert {warp_dtype} to a Torch type")


def dtype_from_torch(torch_dtype):
    """Return the Warp dtype corresponding to a Torch dtype.

    Args:
        torch_dtype: A ``torch.dtype`` that has a corresponding Warp data type.
            Currently ``torch.complex64`` and ``torch.complex128`` are not
            supported.

    Raises:
        TypeError: Unable to find a corresponding Warp data type.
    """
    # initialize lookup table on first call to defer torch import
    if dtype_from_torch.type_map is None:
        import torch  # noqa: PLC0415

        dtype_from_torch.type_map = {
            torch.float16: warp.float16,
            torch.bfloat16: warp.bfloat16,
            torch.float32: warp.float32,
            torch.float64: warp.float64,
            torch.int8: warp.int8,
            torch.int16: warp.int16,
            torch.int32: warp.int32,
            torch.int64: warp.int64,
            torch.uint8: warp.uint8,
            torch.bool: warp.bool,
            # currently unsupported by Warp
            # torch.complex64:
            # torch.complex128:
        }

    warp_dtype = dtype_from_torch.type_map.get(torch_dtype)

    if warp_dtype is not None:
        return warp_dtype
    else:
        raise TypeError(f"Cannot convert {torch_dtype} to a Warp type")


def dtype_is_compatible(torch_dtype, warp_dtype) -> bool:
    """Evaluate whether the given torch dtype is compatible with the given Warp dtype."""
    # initialize lookup table on first call to defer torch import
    if dtype_is_compatible.compatible_sets is None:
        import torch  # noqa: PLC0415

        dtype_is_compatible.compatible_sets = {
            torch.float64: {warp.float64},
            torch.float32: {warp.float32},
            torch.float16: {warp.float16},
            torch.bfloat16: {warp.bfloat16},
            # allow aliasing integer tensors as signed or unsigned integer arrays
            torch.int64: {warp.int64, warp.uint64},
            torch.int32: {warp.int32, warp.uint32},
            torch.int16: {warp.int16, warp.uint16},
            torch.int8: {warp.int8, warp.uint8},
            torch.uint8: {warp.uint8, warp.int8},
            torch.bool: {warp.bool, warp.uint8, warp.int8},
            # currently unsupported by Warp
            # torch.complex64:
            # torch.complex128:
        }

    compatible_set = dtype_is_compatible.compatible_sets.get(torch_dtype)

    if compatible_set is not None:
        if warp_dtype in compatible_set:
            return True
        # check if it's a vector or matrix type
        if hasattr(warp_dtype, "_wp_scalar_type_"):
            return warp_dtype._wp_scalar_type_ in compatible_set

    return False


# lookup tables initialized when needed
dtype_from_torch.type_map = None
dtype_to_torch.type_map = None
dtype_is_compatible.compatible_sets = None


# wrap a torch tensor to a wp array, data is not copied
def _from_torch_mps(t, dtype, shape, strides, requires_grad, grad, return_ctype, device):
    """Wrap an MPS Torch tensor as a zero-copy Warp array on the Metal device.

    Torch's MPS ``data_ptr()`` is not a data pointer: it is the ObjC
    ``id<MTLBuffer>`` of the storage plus the byte storage offset (see
    ``getMTLBufferStorage`` in Torch's MPS backend). For an offset-0 tensor
    whose buffer uses *shared* storage we can recover the buffer object, take
    its ``contents()`` host address (Apple unified memory), and register the
    buffer with Warp's Metal registry so native-dispatch launches can bind it.

    Torch's own MPS allocations use *private* storage and cannot be wrapped —
    only tensors whose buffer originated elsewhere qualify, most usefully a
    Warp array round-tripped through ``wp.to_torch`` (zero-copy on the native
    dispatch path). The idiomatic Metal pattern is therefore Warp-owned
    memory viewed from Torch, with Torch writing back in place through the
    view — the reverse of the CUDA habit of wrapping Torch-owned tensors.

    Coherency contract (same as the round-5 host-op rules): pending Torch
    GPU work is drained here so the initial contents are visible; after
    that, callers must synchronize the writing side before the other side
    reads (``torch.mps.synchronize()`` for Torch writes,
    ``wp.synchronize_device()`` for Warp writes) — the two frameworks
    dispatch on separate MTLCommandQueues with no cross-queue ordering.
    """
    import numpy as np  # noqa: PLC0415
    import objc  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from warp._src.context import _metal_register_foreign_buffer  # noqa: PLC0415

    if not warp.config.metal_native_dispatch:
        raise RuntimeError(
            "wp.from_torch on an MPS tensor requires the native Metal dispatch path — "
            "set wp.config.metal_native_dispatch = True before wp.init(). The MLX "
            "dispatch path cannot bind Torch-owned MTLBuffers to kernels."
        )
    if requires_grad or (requires_grad is None and t.requires_grad) or grad is not None:
        raise RuntimeError(
            "wp.from_torch on an MPS tensor does not support gradients — the Metal "
            "backend has no adjoint kernel support. Pass requires_grad=False (and "
            "detach the tensor if it requires grad)."
        )
    if return_ctype:
        raise RuntimeError("wp.from_torch(..., return_ctype=True) is not supported for MPS tensors")
    if t.storage_offset() != 0:
        raise RuntimeError(
            "wp.from_torch on an MPS tensor requires storage_offset() == 0 — Metal kernel "
            "launches bind whole MTLBuffers. Wrap the base tensor, or make an offset-free "
            "copy with .clone()."
        )
    if not t.is_contiguous():
        raise RuntimeError(
            "wp.from_torch on an MPS tensor requires a contiguous tensor — Metal kernels "
            "index arrays as row-major contiguous. Call .contiguous() first."
        )

    mtl_buf = objc.objc_object(c_void_p=t.data_ptr())
    try:
        buf_nbytes = int(mtl_buf.length())
        storage_mode = int(mtl_buf.storageMode())
    except Exception as e:
        raise RuntimeError(
            "wp.from_torch could not interpret the MPS tensor's storage as an MTLBuffer — "
            "this Torch version may have changed its MPS data_ptr() convention."
        ) from e
    if storage_mode != 0:  # MTLStorageModeShared
        raise RuntimeError(
            "wp.from_torch: this MPS tensor's MTLBuffer uses private storage (Torch's MPS "
            "allocator default), so its bytes are not host-addressable and Warp cannot wrap "
            "it. Allocate the array in Warp instead and view it in Torch via wp.to_torch "
            "(zero-copy on the native Metal dispatch path) — Torch ops can write back "
            "in place through that view."
        )
    data_nbytes = t.numel() * t.element_size()
    if buf_nbytes < data_nbytes:
        raise RuntimeError(
            f"wp.from_torch: MPS tensor claims {data_nbytes} bytes but its MTLBuffer holds "
            f"only {buf_nbytes} — refusing to wrap."
        )
    mv = mtl_buf.contents().as_buffer(buf_nbytes)
    addr = int(np.frombuffer(mv, dtype=np.uint8).__array_interface__["data"][0])

    # Drain pending Torch GPU work so the wrapped array's initial contents
    # are coherent (see the coherency contract in the docstring).
    torch.mps.synchronize()

    _metal_register_foreign_buffer(addr, mtl_buf)

    a = warp.array(
        ptr=addr,
        dtype=dtype,
        shape=shape,
        strides=strides,
        device=device,
        copy=False,
    )
    # Keep the tensor alive: it owns the storage; if it died, Torch's MPS
    # allocator could hand the MTLBuffer to a new tensor while Warp still
    # reads/writes through it.
    a._tensor = t
    weakref.finalize(a, warp._src.context._metal_release_foreign_buffer, addr)
    return a


def from_torch(
    t: torch.Tensor,
    dtype: type | None = None,
    requires_grad: bool | None = None,
    grad=None,
    return_ctype: bool = False,
    retain_grad: bool = False,
) -> warp.array | warp._src.types.array_t:
    """Convert a Torch tensor to a Warp array without copying the data.

    MPS (Apple GPU) tensors can only be wrapped when their ``MTLBuffer`` uses
    shared storage — in practice, tensors that view Warp-owned memory (e.g.
    obtained from ``wp.to_torch``), since Torch's own MPS allocations use
    private storage. The idiomatic Metal pattern is the reverse of CUDA's:
    allocate in Warp, view in Torch via ``wp.to_torch`` (zero-copy on the
    native dispatch path), and let Torch write back in place through the view.
    Torch and Warp dispatch on separate Metal command queues: synchronize the
    writing side (``torch.mps.synchronize()`` / ``wp.synchronize_device()``)
    before the other side reads.

    Args:
        t: The torch tensor to wrap.
        dtype: The target data type of the resulting Warp array. Defaults to the tensor value type mapped to a Warp array value type.
        requires_grad: Whether the resulting array should wrap the tensor's gradient,
          if it exists (the grad tensor will be allocated otherwise). Defaults to the tensor's ``requires_grad`` value.
        grad: Optional gradient array to attach to the result. Can be a Warp array or Torch tensor.
          If not provided and ``requires_grad`` is True, the tensor's gradient will be wrapped or allocated.
        return_ctype: Whether to return a low-level array descriptor instead of a ``wp.array`` object (faster).
          The descriptor can be passed to Warp kernels.
        retain_grad: Whether to preserve gradients during backward instead of zeroing after read.

    Returns:
        The wrapped array or array descriptor.
    """
    if dtype is None:
        dtype = dtype_from_torch(t.dtype)
    elif not dtype_is_compatible(t.dtype, dtype):
        raise RuntimeError(f"Cannot convert Torch type {t.dtype} to Warp type {dtype}")

    # get size of underlying data type to compute strides
    ctype_size = ctypes.sizeof(dtype._type_)

    shape = tuple(t.shape)
    strides = tuple(s * ctype_size for s in t.stride())

    # if target is a vector or matrix type
    # then check if trailing dimensions match
    # the target type and update the shape
    if hasattr(dtype, "_shape_"):
        dtype_shape = dtype._shape_
        dtype_dims = len(dtype._shape_)
        # ensure inner shape matches
        if dtype_dims > len(shape) or dtype_shape != shape[-dtype_dims:]:
            raise RuntimeError(
                f"Could not convert Torch tensor with shape {shape} to Warp array with dtype={dtype}, ensure that source inner shape is {dtype_shape}"
            )
        # ensure inner strides are contiguous
        if strides[-1] != ctype_size or (dtype_dims > 1 and strides[-2] != ctype_size * dtype_shape[-1]):
            raise RuntimeError(
                f"Could not convert Torch tensor with shape {shape} to Warp array with dtype={dtype}, because the source inner strides are not contiguous"
            )
        # trim shape and strides
        shape = tuple(shape[:-dtype_dims]) or (1,)
        strides = tuple(strides[:-dtype_dims]) or (ctype_size,)

    if t.device.type == "mps":
        # Apple-GPU tensors need special pointer handling (their data_ptr()
        # is not a data pointer) — see _from_torch_mps.
        return _from_torch_mps(t, dtype, shape, strides, requires_grad, grad, return_ctype, device_from_torch(t.device))

    # gradient
    # - if return_ctype is False, we set `grad` to a wp.array or None
    # - if return_ctype is True, we set `grad_ptr` and set `grad` as the owner (wp.array or torch.Tensor)
    requires_grad = t.requires_grad if requires_grad is None else requires_grad
    grad_ptr = 0
    if grad is not None:
        if isinstance(grad, warp.array):
            if return_ctype:
                if grad.strides != strides:
                    raise RuntimeError(
                        f"Gradient strides must match array strides, expected {strides} but got {grad.strides}"
                    )
                grad_ptr = grad.ptr
        else:
            # assume grad is a torch.Tensor
            if return_ctype:
                if t.stride() != grad.stride():
                    raise RuntimeError(
                        f"Gradient strides must match array strides, expected {t.stride()} but got {grad.stride()}"
                    )
                grad_ptr = grad.data_ptr()
            else:
                grad = from_torch(grad, dtype=dtype, requires_grad=False)
    elif requires_grad:
        # wrap the tensor gradient, allocate if necessary
        if t.grad is not None:
            if return_ctype:
                grad = t.grad
                if t.stride() != grad.stride():
                    raise RuntimeError(
                        f"Gradient strides must match array strides, expected {t.stride()} but got {grad.stride()}"
                    )
                grad_ptr = grad.data_ptr()
            else:
                grad = from_torch(t.grad, dtype=dtype, requires_grad=False)
        else:
            # allocate a zero-filled gradient if it doesn't exist
            # Note: we use Warp to allocate the shared gradient with compatible strides
            grad = warp.zeros(dtype=dtype, shape=shape, strides=strides, device=device_from_torch(t.device))
            t.grad = to_torch(grad, requires_grad=False)
            grad_ptr = grad.ptr

    if return_ctype:
        ptr = t.data_ptr()

        # create array descriptor
        array_ctype = warp._src.types.array_t(ptr, grad_ptr, len(shape), shape, strides)

        # keep data and gradient alive
        array_ctype._ref = t
        array_ctype._gradref = grad

        return array_ctype

    else:
        a = warp.array(
            ptr=t.data_ptr(),
            dtype=dtype,
            shape=shape,
            strides=strides,
            device=device_from_torch(t.device),
            copy=False,
            grad=grad,
            requires_grad=requires_grad,
            retain_grad=retain_grad,
        )

        # save a reference to the source tensor, otherwise it may get deallocated
        a._tensor = t

        return a


def to_torch(a: warp.array, requires_grad: bool | None = None):
    """Convert a Warp array to a Torch tensor without copying the data.

    On Metal devices the conversion is zero-copy on the native dispatch path
    (the result is an MPS view of the Warp array's unified memory, so Torch
    ops can write back in place); the MLX dispatch path falls back to a copy.

    Args:
        a: The Warp array to convert.
        requires_grad: Whether the resulting tensor should convert the array's
          gradient, if it exists, to a grad tensor. Defaults to the array's
          ``requires_grad`` value.

    Returns:
        torch.Tensor: The converted tensor.
    """
    import torch  # noqa: PLC0415

    if requires_grad is None:
        requires_grad = a.requires_grad

    # Torch does not support structured arrays
    if isinstance(a.dtype, warp._src.codegen.Struct):
        raise RuntimeError("Cannot convert structured Warp arrays to Torch.")

    # bfloat16 is not representable via __array_interface__ or __cuda_array_interface__
    # (it exposes as uint16), so use DLPack for a correct dtype round-trip.
    # This also covers compound types (vectors/matrices) whose scalar type is bfloat16.
    scalar_type = getattr(a.dtype, "_wp_scalar_type_", a.dtype)
    if scalar_type is warp.bfloat16:
        t = torch.from_dlpack(warp.to_dlpack(a))
        t.requires_grad = requires_grad
        if requires_grad and a.requires_grad:
            t.grad = torch.from_dlpack(warp.to_dlpack(a.grad))
            t.grad._warp_grad_array = a.grad
        # prevent GC of the Warp array while the tensor is alive
        t._warp_array = a
        return t

    if a.device.is_cpu:
        # Torch has an issue wrapping CPU objects
        # that support the __array_interface__ protocol
        # in this case we need to workaround by going
        # to an ndarray first, see https://pearu.github.io/array_interface_pytorch.html
        t = torch.as_tensor(numpy.asarray(a))
        t.requires_grad = requires_grad
        if requires_grad and a.requires_grad:
            t.grad = torch.as_tensor(numpy.asarray(a.grad))
        return t

    elif a.device.is_cuda:
        # Torch does support the __cuda_array_interface__
        # correctly, but we must be sure to maintain a reference
        # to the owning object to prevent memory allocs going out of scope
        t = torch.as_tensor(a, device=device_to_torch(a.device))
        t.requires_grad = requires_grad
        if requires_grad and a.requires_grad:
            t.grad = torch.as_tensor(a.grad, device=device_to_torch(a.device))
        return t

    elif a.device.is_metal:
        # Native dispatch path: zero-copy. Export the array's MTLBuffer as a
        # kDLMetal DLPack capsule and let Torch wrap it as an MPS tensor
        # aliasing the same unified memory — Torch ops write back in place.
        # Falls through to the copy path when the pointer is not a base
        # allocation (offset views) or when gradient wiring is requested.
        if warp.config.metal_native_dispatch and not (requires_grad and a.requires_grad):
            mtl_buf = warp._src.context._metal_get_buffer(a.ptr)
            if mtl_buf is not None:
                t = torch.from_dlpack(warp.to_dlpack(a))
                # Torch's MPS DLPack import normalises stride metadata, which
                # erases Warp's zero strides (mujoco_warp's per-world
                # broadcast convention — see the long comment on the copy
                # path below). Restore them; for extent-1 dims the memory
                # layout is identical, only the stride *value* changes, and
                # consumers like mjlab detect broadcast fields by
                # ``tensor.stride(0) == 0``.
                if any(s == 0 for s in a.strides):
                    torch_stride = list(t.stride())
                    for i, wp_s in enumerate(a.strides):
                        if wp_s == 0 and i < len(torch_stride):
                            torch_stride[i] = 0
                    t = torch.as_strided(t, t.shape, torch_stride)
                t.requires_grad = requires_grad
                return t

        # MLX-managed unified memory is host-readable. Round-trip through
        # numpy and let torch place the result on MPS. NB: this is a
        # *copy* in both directions — host-side mutations on the torch
        # tensor will not propagate back to the Warp array. mjlab and
        # similar host frameworks that round-trip between the two need
        # an explicit writeback.
        t = torch.as_tensor(a.numpy(), device="mps")
        # ``a.numpy()`` discards Warp's zero strides — the numpy round-
        # trip produces a contiguous copy with normal strides, even for
        # arrays whose ``.strides[k] == 0`` (the per-world broadcast
        # convention mujoco_warp's ``put_model`` uses for fields like
        # ``jnt_range``). Frameworks built around the DLPack-zero-copy
        # CUDA path detect "this is per-asset data, expand to num_envs"
        # via ``tensor.stride(0) == 0`` (e.g. mjlab's ``TorchArray``);
        # without the stride hint they leave the tensor at shape
        # ``(1, …)`` and indexing it with ``env_ids = [0, 1, …]``
        # raises ``index N is out of bounds: 0, range 0 to 1``. Restore
        # the broadcast by forcing stride 0 on every wp.array dim that
        # had it. ``as_strided`` is zero-copy and only updates the
        # tensor's stride metadata.
        if any(s == 0 for s in a.strides):
            torch_stride = list(t.stride())
            for i, wp_s in enumerate(a.strides):
                if wp_s == 0 and i < len(torch_stride):
                    torch_stride[i] = 0
            t = torch.as_strided(t, t.shape, torch_stride)
        t.requires_grad = requires_grad
        if requires_grad and a.requires_grad:
            t.grad = torch.as_tensor(a.grad.numpy(), device="mps")
        return t

    else:
        raise RuntimeError("Unsupported device")


def stream_from_torch(stream_or_device=None):
    """Convert from a Torch CUDA stream to a Warp CUDA stream."""
    import torch  # noqa: PLC0415

    if isinstance(stream_or_device, torch.cuda.Stream):
        stream = stream_or_device
    else:
        # assume arg is a torch device
        stream = torch.cuda.current_stream(stream_or_device)

    device = device_from_torch(stream.device)

    warp_stream = warp.Stream(device, cuda_stream=stream.cuda_stream)

    # save a reference to the source stream, otherwise it may be destroyed
    warp_stream._torch_stream = stream

    return warp_stream


def stream_to_torch(stream_or_device=None):
    """Convert from a Warp CUDA stream to a Torch CUDA stream."""
    import torch  # noqa: PLC0415

    if isinstance(stream_or_device, warp.Stream):
        stream = stream_or_device
    else:
        # assume arg is a warp device
        stream = warp.get_device(stream_or_device).stream

    device = device_to_torch(stream.device)

    torch_stream = torch.cuda.ExternalStream(stream.cuda_stream, device=device)

    # save a reference to the source stream, otherwise it may be destroyed
    torch_stream._warp_stream = stream

    return torch_stream
