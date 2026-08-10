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

"""Metal-side ``wp.HashGrid`` support (float32 grids only).

The CUDA/CPU backends pass a host or device ``HashGrid_t*`` as the
``uint64`` id a kernel dereferences. Metal kernels can't chase host
pointers, so — exactly like ``metal_bvh.py`` — the Metal model is:

* Build the cell tables ON THE HOST. The build is a direct NumPy
  transcription of ``hash_grid_update_host_impl`` (warp/native/hashgrid.cpp):
  per-point cell index (same ``floor(p * cell_width_inv)`` + origin-offset +
  modulo math, in float32), a stable sort of (cell, point-id) pairs, then a
  cell start/end scan over the sorted cells.
* Store ``point_cells`` / ``point_ids`` / ``cell_starts`` / ``cell_ends`` as
  Metal-side ``wp.array`` storage and pack a descriptor buffer whose fields
  are the arrays' ``gpuAddress``es (see ``wp_hash_grid_desc_t`` in
  ``codegen_metal._HASHGRID_HELPERS``). ``grid.id`` IS the descriptor
  buffer's own gpuAddress, stable across rebuilds.
* Register every reached buffer with the dispatcher's resident-resource set;
  kernels using hash-grid builtins dispatch with ``reads_resident=True``
  (the ``uses_bvh`` artifact flag covers hash grids too).

``build()`` re-sorts on the host every call (it syncs to read the points),
so unlike CUDA it is NOT graph-capturable. Before the first ``build()`` the
descriptor's ``point_cells`` address is 0 and queries return no results,
matching the native unbuilt-grid behavior.
"""

from __future__ import annotations

import ctypes
import struct

import numpy as np

import warp
from warp._src.metal_bvh import _gpu_address, _require_native_dispatch

__all__ = ["MetalHashGrid"]

# 4 gpu addresses + dim_x/dim_y/dim_z/num_points/max_points/pad +
# cell_width/cell_width_inv. Must match ``wp_hash_grid_desc_t``.
_HASH_GRID_DESC_FMT = "<4Q6i2f"
_HASH_GRID_DESC_NBYTES = struct.calcsize(_HASH_GRID_DESC_FMT)


class MetalHashGrid:
    """Metal-side cell tables + descriptor for one ``wp.HashGrid``.
    ``id`` is the descriptor buffer's gpuAddress."""

    def __init__(self, dim_x: int, dim_y: int, dim_z: int, device):
        _require_native_dispatch("wp.HashGrid")
        from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

        self.device = device
        self.dim_x = int(dim_x)
        self.dim_y = int(dim_y)
        self.dim_z = int(dim_z)
        if self.dim_x <= 0 or self.dim_y <= 0 or self.dim_z <= 0:
            raise ValueError(f"Hash grid dimensions must be positive, got ({dim_x}, {dim_y}, {dim_z})")
        self.num_cells = self.dim_x * self.dim_y * self.dim_z
        self._dispatcher = get_dispatcher()
        self._resident: list = []
        self._resident_ids: set[int] = set()

        # The descriptor buffer outlives rebuilds — its gpuAddress is the
        # public id, baked into launched kernels and captured graphs.
        self._desc_buf, self._desc_ptr = self._dispatcher.alloc(_HASH_GRID_DESC_NBYTES)
        self._register(self._desc_buf)
        self.id = int(self._desc_buf.gpuAddress())

        self.point_cells = None
        self.point_ids = None
        self.cell_starts = None
        self.cell_ends = None
        self.num_points = 0
        self.cell_width = 0.0
        self._write_descriptor()

    # -- residency ------------------------------------------------------

    def _register(self, mtl_buf) -> None:
        if id(mtl_buf) in self._resident_ids:
            return
        self._resident_ids.add(id(mtl_buf))
        self._dispatcher.add_resident_resource(mtl_buf)
        self._resident.append(mtl_buf)

    def release(self) -> None:
        """Unregister this grid's buffers from the resident set. Called from
        ``HashGrid.__del__``; safe to call more than once."""
        disp = self._dispatcher
        if disp is None:
            return
        for buf in self._resident:
            try:
                disp.remove_resident_resource(buf)
            except (TypeError, AttributeError):
                pass  # interpreter shutdown
        self._resident = []
        self._resident_ids = set()
        self._dispatcher = None

    # -- build ----------------------------------------------------------

    def build(self, points, radius: float) -> None:
        """Host-side rebuild of the cell tables (see module docstring)."""
        pts = points.numpy().reshape(-1, 3).astype(np.float32, copy=False)  # syncs the device
        num_points = pts.shape[0]
        self.num_points = num_points
        self.cell_width = float(radius)

        # Per-point cell index — float32 math matching hash_grid_index():
        # floor toward -inf, +2^20 origin offset, clamp, modulo dims.
        cell_width_inv = np.float32(1.0) / np.float32(radius)
        ijk = np.floor(pts * cell_width_inv).astype(np.int64) + (1 << 20)
        np.maximum(ijk, 0, out=ijk)
        cx = ijk[:, 0] % self.dim_x
        cy = ijk[:, 1] % self.dim_y
        cz = ijk[:, 2] % self.dim_z
        cells = (cz * (self.dim_x * self.dim_y) + cy * self.dim_x + cx).astype(np.int32)

        # Sort (cell, point-id) pairs; stable matches the native radix sort.
        order = np.argsort(cells, kind="stable").astype(np.int32)
        sorted_cells = cells[order]

        # Cell start/end scan. Unvisited cells keep start == end == 0
        # (an empty range), matching the native memset.
        starts = np.zeros(self.num_cells, dtype=np.int32)
        ends = np.zeros(self.num_cells, dtype=np.int32)
        if num_points:
            vals, first_idx, counts = np.unique(sorted_cells, return_index=True, return_counts=True)
            starts[vals] = first_idx
            ends[vals] = first_idx + counts

        with warp.ScopedDevice(self.device):
            self.point_cells = warp.array(sorted_cells, dtype=warp.int32)
            self.point_ids = warp.array(order, dtype=warp.int32)
            self.cell_starts = warp.array(starts, dtype=warp.int32)
            self.cell_ends = warp.array(ends, dtype=warp.int32)

        self._write_descriptor()
        self._dispatcher.sync()

    def _write_descriptor(self) -> None:
        addrs = []
        for arr in (self.point_cells, self.point_ids, self.cell_starts, self.cell_ends):
            if arr is None:
                addrs.append(0)
                continue
            ga, buf = _gpu_address(arr)
            addrs.append(ga)
            self._register(buf)
        cell_width = float(self.cell_width)
        cell_width_inv = float(np.float32(1.0) / np.float32(cell_width)) if cell_width > 0.0 else 0.0
        data = struct.pack(
            _HASH_GRID_DESC_FMT,
            *addrs,
            self.dim_x,
            self.dim_y,
            self.dim_z,
            self.num_points,
            self.num_points,
            0,
            cell_width,
            cell_width_inv,
        )
        ctypes.memmove(self._desc_ptr, data, len(data))
