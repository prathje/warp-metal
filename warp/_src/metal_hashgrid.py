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

* Build the cell tables ON THE GPU with ordinary Warp kernels (a counting
  sort): per-point cell index (same ``floor(p * cell_width_inv)`` +
  origin-offset + modulo math as ``hash_grid_index()``), a per-cell
  histogram via ``atomic_add``, a two-level exclusive scan over the cells,
  an atomic scatter of point ids into their cell ranges, and finally a
  per-cell sort of the ids so every cell lists its points in ascending
  index order — the same order the native stable radix sort produces.
  Nothing syncs the device; ``build()`` is a fixed sequence of launches.
* Store ``point_cells`` / ``point_ids`` / ``cell_starts`` / ``cell_ends`` as
  Metal-side ``wp.array`` storage and pack a descriptor buffer whose fields
  are the arrays' ``gpuAddress``es (see ``wp_hash_grid_desc_t`` in
  ``codegen_metal._HASHGRID_HELPERS``). ``grid.id`` IS the descriptor
  buffer's own gpuAddress, stable across rebuilds.
* Register every reached buffer with the dispatcher's resident-resource set;
  kernels using hash-grid builtins dispatch with ``reads_resident=True``
  (the ``uses_bvh`` artifact flag covers hash grids too).

The point tables are reallocated only when a build sees more points than
the tables hold; the descriptor is rewritten (after draining the queue)
only when an address, the point count, or the cell width changes. Before
the first ``build()`` the descriptor's ``point_cells`` address is 0 and
queries return no results, matching the native unbuilt-grid behavior.
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

# Cells per thread in the two-level exclusive scan.
_SCAN_BLOCK = 1024


@warp.kernel(enable_backward=False)
def _hg_count_cells(
    points: warp.array(dtype=warp.vec3),
    cell_width_inv: warp.float32,
    dim_x: int,
    dim_y: int,
    dim_z: int,
    point_cell: warp.array(dtype=warp.int32),
    counts: warp.array(dtype=warp.int32),
):
    # Mirrors hash_grid_index() in warp/native/hashgrid.h: floor toward -inf,
    # +2^20 origin offset, clamp at 0, modulo the grid dims.
    i = warp.tid()
    p = points[i]
    x = warp.max(0, int(warp.floor(p[0] * cell_width_inv)) + 1048576)
    y = warp.max(0, int(warp.floor(p[1] * cell_width_inv)) + 1048576)
    z = warp.max(0, int(warp.floor(p[2] * cell_width_inv)) + 1048576)
    cell = (z % dim_z) * (dim_x * dim_y) + (y % dim_y) * dim_x + (x % dim_x)
    point_cell[i] = cell
    warp.atomic_add(counts, cell, 1)


@warp.kernel(enable_backward=False)
def _hg_block_sums(
    counts: warp.array(dtype=warp.int32),
    num_cells: int,
    block_sums: warp.array(dtype=warp.int32),
):
    b = warp.tid()
    start = b * 1024
    end = warp.min(start + 1024, num_cells)
    s = int(0)
    for k in range(start, end):
        s += counts[k]
    block_sums[b] = s


@warp.kernel(enable_backward=False)
def _hg_scan_block_sums(block_sums: warp.array(dtype=warp.int32), num_blocks: int):
    # Single thread: in-place exclusive scan of the (few) block totals.
    s = int(0)
    for b in range(num_blocks):
        v = block_sums[b]
        block_sums[b] = s
        s += v


@warp.kernel(enable_backward=False)
def _hg_scan_within_block(
    counts: warp.array(dtype=warp.int32),
    num_cells: int,
    block_offsets: warp.array(dtype=warp.int32),
    cell_starts: warp.array(dtype=warp.int32),
    cell_ends: warp.array(dtype=warp.int32),
    cursor: warp.array(dtype=warp.int32),
):
    b = warp.tid()
    start = b * 1024
    end = warp.min(start + 1024, num_cells)
    s = block_offsets[b]
    for k in range(start, end):
        c = counts[k]
        counts[k] = 0  # leave the histogram zeroed for the next build
        cell_starts[k] = s
        cursor[k] = s
        s += c
        cell_ends[k] = s


@warp.kernel(enable_backward=False)
def _hg_scatter(
    point_cell: warp.array(dtype=warp.int32),
    cursor: warp.array(dtype=warp.int32),
    point_ids: warp.array(dtype=warp.int32),
    sorted_cells: warp.array(dtype=warp.int32),
):
    i = warp.tid()
    c = point_cell[i]
    pos = warp.atomic_add(cursor, c, 1)
    point_ids[pos] = i
    sorted_cells[pos] = c


@warp.kernel(enable_backward=False)
def _hg_sort_cell_ids(
    cell_starts: warp.array(dtype=warp.int32),
    cell_ends: warp.array(dtype=warp.int32),
    point_ids: warp.array(dtype=warp.int32),
):
    # The atomic scatter places a cell's points in arbitrary order; sort each
    # cell's range ascending so the tables equal a stable sort by cell.
    c = warp.tid()
    lo = cell_starts[c]
    hi = cell_ends[c]
    for k in range(lo + 1, hi):
        v = point_ids[k]
        j = k - 1
        while j >= lo:
            if point_ids[j] <= v:
                break
            point_ids[j + 1] = point_ids[j]
            j -= 1
        point_ids[j + 1] = v


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
        self._num_blocks = (self.num_cells + _SCAN_BLOCK - 1) // _SCAN_BLOCK
        self._dispatcher = get_dispatcher()
        self._resident: list = []
        self._resident_ids: set[int] = set()

        # The descriptor buffer outlives rebuilds — its gpuAddress is the
        # public id, baked into launched kernels and captured graphs.
        self._desc_buf, self._desc_ptr = self._dispatcher.alloc(_HASH_GRID_DESC_NBYTES)
        self._register(self._desc_buf)
        self.id = int(self._desc_buf.gpuAddress())

        # Per-cell tables are sized once; per-point tables grow on demand.
        with warp.ScopedDevice(self.device):
            self.cell_starts = warp.zeros(self.num_cells, dtype=warp.int32)
            self.cell_ends = warp.zeros(self.num_cells, dtype=warp.int32)
            self._counts = warp.zeros(self.num_cells, dtype=warp.int32)
            self._cursor = warp.zeros(self.num_cells, dtype=warp.int32)
            self._block_sums = warp.zeros(self._num_blocks, dtype=warp.int32)
        self.point_cells = None
        self.point_ids = None
        self._point_cell = None
        self._capacity = 0
        self.num_points = 0
        self.cell_width = 0.0
        self._desc_data = None
        self._retired: list = []
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

    def _reserve_points(self, num_points: int) -> None:
        if num_points <= self._capacity:
            return
        # A previous build's tables may still be read by in-flight queries;
        # keep the old arrays alive until the descriptor rewrite drains the
        # queue (``_write_descriptor`` clears ``_retired`` after its sync).
        self._retired = [self.point_ids, self.point_cells, self._point_cell]
        with warp.ScopedDevice(self.device):
            self.point_ids = warp.zeros(num_points, dtype=warp.int32)
            self.point_cells = warp.zeros(num_points, dtype=warp.int32)
            self._point_cell = warp.zeros(num_points, dtype=warp.int32)
        self._capacity = num_points

    def build(self, points, radius: float) -> None:
        """GPU rebuild of the cell tables (see module docstring)."""
        num_points = points.shape[0]
        self._reserve_points(num_points)
        self.num_points = num_points
        self.cell_width = float(radius)
        self._write_descriptor()  # no-op unless an address / count / width changed

        if num_points == 0:
            return
        dev = self.device
        cell_width_inv = float(np.float32(1.0) / np.float32(radius))
        # Leading-slice views keep the launch dims honest when the tables are
        # larger than the current point count.
        point_ids = self.point_ids[:num_points]
        point_cells = self.point_cells[:num_points]
        point_cell = self._point_cell[:num_points]
        warp.launch(
            _hg_count_cells,
            dim=num_points,
            inputs=[points, cell_width_inv, self.dim_x, self.dim_y, self.dim_z],
            outputs=[point_cell, self._counts],
            device=dev,
        )
        warp.launch(
            _hg_block_sums,
            dim=self._num_blocks,
            inputs=[self._counts, self.num_cells],
            outputs=[self._block_sums],
            device=dev,
        )
        warp.launch(
            _hg_scan_block_sums,
            dim=1,
            inputs=[self._block_sums, self._num_blocks],
            device=dev,
        )
        warp.launch(
            _hg_scan_within_block,
            dim=self._num_blocks,
            inputs=[self._counts, self.num_cells, self._block_sums],
            outputs=[self.cell_starts, self.cell_ends, self._cursor],
            device=dev,
        )
        warp.launch(
            _hg_scatter,
            dim=num_points,
            inputs=[point_cell, self._cursor],
            outputs=[point_ids, point_cells],
            device=dev,
        )
        warp.launch(
            _hg_sort_cell_ids,
            dim=self.num_cells,
            inputs=[self.cell_starts, self.cell_ends],
            outputs=[point_ids],
            device=dev,
        )

    def _write_descriptor(self) -> None:
        addrs = []
        bufs = []
        for arr in (self.point_cells, self.point_ids, self.cell_starts, self.cell_ends):
            if arr is None:
                addrs.append(0)
                continue
            ga, buf = _gpu_address(arr)
            addrs.append(ga)
            bufs.append(buf)
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
        if data == self._desc_data:
            return
        for buf in bufs:
            self._register(buf)
        if self._desc_data is not None:
            # Kernels from earlier launches may still read the old descriptor
            # (and, after a regrow, the old tables); the host write below is
            # immediate, so drain the queue first.
            self._dispatcher.sync(torch=False)
        self._retired = []
        ctypes.memmove(self._desc_ptr, data, len(data))
        self._desc_data = data
