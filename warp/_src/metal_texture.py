# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Buffer-backed ``wp.Texture1D/2D/3D`` storage for the Metal backend.

Apple GPUs have hardware samplers, but Warp's texture semantics are defined by
the CPU software sampler in ``warp/native/texture.h`` (address modes applied to
the neighbor indices, ``u / width`` for unnormalized coordinates on every mip
level, border texels reading 0, the normalization of integer formats, mip
blending). Rather than approximating those with an ``MTLSamplerState``, a
Metal texture is a plain buffer of texels plus a descriptor, and the MSL
helpers in ``codegen_metal_texture`` port the CPU sampler line for line.

Layout:

- ``data``: one Warp ``uint8`` array on the Metal device holding every mip
  level back to back, each level row-major ``(depth, height, width, channels)``
  like the CPU texture.
- descriptor: a small dispatcher buffer (``_DESC_FMT``); its gpuAddress is the
  texture's ``id`` / the ``tex`` field of ``texture{1,2,3}d_t`` that kernels
  receive. It records the geometry, modes, and the gpuAddress of every level.

Both buffers join the dispatcher's resident set, as BVH buffers do, because
kernels reach them only through addresses.
"""

from __future__ import annotations

import ctypes
import struct

import numpy as np

MAX_MIP_LEVELS = 16
# 14 int32 header fields, 16 level addresses, then per-level width/height/depth.
_DESC_FMT = "<14i16Q16i16i16i"
DESC_NBYTES = struct.calcsize(_DESC_FMT)


def _gpu_address(arr):
    from warp._src.context import _metal_find_buffer  # noqa: PLC0415

    buf, off = _metal_find_buffer(arr.ptr)
    if buf is None:
        raise RuntimeError("Texture storage is not a registered Metal buffer")
    return int(buf.gpuAddress()) + int(off), buf


class MetalTextureStorage:
    """Texel buffer + descriptor for one Metal texture."""

    def __init__(
        self,
        device,
        ndim: int,
        mip_shapes: list[tuple[int, int, int]],
        num_channels: int,
        dtype_code: int,
        bytes_per_channel: int,
        filter_mode: int,
        mip_filter_mode: int,
        address_modes: tuple[int, int, int],
        normalized_coords: bool,
    ):
        import warp._src.context  # noqa: PLC0415
        import warp.config as _cfg  # noqa: PLC0415
        from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415
        from warp._src.types import uint8  # noqa: PLC0415

        if not _cfg.metal_native_dispatch:
            raise RuntimeError(
                "wp.Texture on Metal requires native dispatch (wp.config.metal_native_dispatch=True): "
                "kernels reach texture storage through descriptor addresses"
            )
        self.device = device
        self._dispatcher = get_dispatcher()
        self.level_offsets: list[int] = []
        self.level_nbytes: list[int] = []
        total = 0
        for w, h, d in mip_shapes:
            nbytes = int(w) * int(h) * int(d) * int(num_channels) * int(bytes_per_channel)
            # Keep every level 16-byte aligned so a typed view of it is valid.
            total = (total + 15) & ~15
            self.level_offsets.append(total)
            self.level_nbytes.append(nbytes)
            total += nbytes
        self.data = warp._src.context.zeros(max(total, 16), dtype=uint8, device=device)
        data_ga, data_buf = _gpu_address(self.data)

        self._desc_buf, self._desc_ptr = self._dispatcher.alloc(DESC_NBYTES)
        self._resident = [self._desc_buf, data_buf]
        for buf in self._resident:
            self._dispatcher.add_resident_resource(buf)

        base = mip_shapes[0]
        addrs = [0] * MAX_MIP_LEVELS
        ws = [0] * MAX_MIP_LEVELS
        hs = [0] * MAX_MIP_LEVELS
        ds = [0] * MAX_MIP_LEVELS
        for level, (w, h, d) in enumerate(mip_shapes):
            addrs[level] = data_ga + self.level_offsets[level]
            ws[level], hs[level], ds[level] = int(w), int(h), int(d)
        header = [
            int(ndim),
            int(num_channels),
            int(dtype_code),
            int(filter_mode),
            int(mip_filter_mode),
            len(mip_shapes),
            int(address_modes[0]),
            int(address_modes[1]),
            int(address_modes[2]),
            1 if normalized_coords else 0,
            int(base[0]),
            int(base[1]) if ndim > 1 else 0,
            int(base[2]) if ndim > 2 else 0,
            0,
        ]
        packed = struct.pack(_DESC_FMT, *header, *addrs, *ws, *hs, *ds)
        ctypes.memmove(self._desc_ptr, packed, len(packed))
        self.id = int(self._desc_buf.gpuAddress())

    def level_array(self, level: int, shape, dtype):
        """A typed Warp array aliasing mip ``level``'s texels."""
        from warp._src.types import array  # noqa: PLC0415

        return array(
            ptr=self.data.ptr + self.level_offsets[level],
            shape=shape,
            dtype=dtype,
            device=self.device,
        )

    def upload_level(self, level: int, data_np: np.ndarray) -> None:
        data_np = np.ascontiguousarray(data_np)
        if data_np.nbytes != self.level_nbytes[level]:
            raise ValueError(
                f"Texture mip level {level}: expected {self.level_nbytes[level]} bytes, got {data_np.nbytes}"
            )
        # Unified memory: drain queued GPU work on the buffer, then write it
        # through its host mapping.
        self._dispatcher.sync(torch=False)
        ctypes.memmove(self.data.ptr + self.level_offsets[level], data_np.ctypes.data, data_np.nbytes)

    def release(self) -> None:
        disp = self._dispatcher
        if disp is None:
            return
        for buf in self._resident:
            try:
                disp.remove_resident_resource(buf)
            except (TypeError, AttributeError):
                pass
        self._resident = []
        self._dispatcher = None
