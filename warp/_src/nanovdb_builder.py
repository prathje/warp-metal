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

"""Host-side (NumPy) construction of NanoVDB ``ValueOnIndex`` grids.

The native builders behind :meth:`warp.Volume.allocate_by_voxels` live in
``volume_builder.cu`` and need CUDA. This module writes the same on-disk
layout with NumPy so index volumes can be created on devices without a
native builder (Metal, CPU). Layout constants follow ``PNanoVDB.h``
(ABI major version 32): a 672-byte grid header, a 64-byte tree header, the
root node with its tiles, then upper (32^3), lower (16^3) and leaf (8^3)
nodes in breadth-first order. Only ``ValueOnIndex`` grids are produced:
every active voxel gets a 1-based index (0 is the background), assigned in
node order so ``Volume.get_voxels()`` enumerates them consistently.
"""

from __future__ import annotations

import numpy as np

_GRID_SIZE = 672
_TREE_SIZE = 64
_ROOT_SIZE = 96
_ROOT_TILE_SIZE = 32
_UPPER_SIZE = 270400
_LOWER_SIZE = 33856
_LEAF_SIZE = 96
_UPPER_OFF_TABLE = 8256
_LOWER_OFF_TABLE = 1088
_UPPER_OFF_VALUE_MASK = 32
_UPPER_OFF_CHILD_MASK = 4128
_LOWER_OFF_VALUE_MASK = 32
_LOWER_OFF_CHILD_MASK = 544

_MAGIC_GRID = 0x314244566F6E614E  # "NanoVDB1"
_VERSION = (32 << 21) | (8 << 10) | 0
_FLAGS = (1 << 1) | (1 << 5)  # HasBBox | IsBreadthFirst
_GRID_CLASS_INDEX = 8
_GRID_TYPE_ONINDEX = 20
_NODE_FLAGS_BBOX = 1 << 1
_LEAF_FLAGS = (1 << 1) | (1 << 5)  # has bbox, breadth-first (matches the CUDA builder)


def _coord_to_key(origin: np.ndarray) -> np.ndarray:
    """Root tile key of ``pnanovdb_coord_to_key`` for upper-node origins ``(N, 3)``."""
    u = origin.astype(np.int64).astype(np.uint32).astype(np.uint64) >> np.uint64(12)
    return u[:, 2] | (u[:, 1] << np.uint64(21)) | (u[:, 0] << np.uint64(42))


def _bit_offsets(ijk: np.ndarray, mask: int, shift: int, bits: int) -> np.ndarray:
    """Child/voxel slot of ``ijk`` inside its node (``pnanovdb_*_coord_to_offset``)."""
    x = (ijk[:, 0] & mask) >> shift
    y = (ijk[:, 1] & mask) >> shift
    z = (ijk[:, 2] & mask) >> shift
    return (x << (2 * bits)) | (y << bits) | z


def _set_mask_bits(mask_words: np.ndarray, node_of: np.ndarray, slot: np.ndarray) -> None:
    """Set bit ``slot[i]`` of the 64-bit-word mask of node ``node_of[i]`` (in place)."""
    words = mask_words.reshape(mask_words.shape[0], -1)
    np.bitwise_or.at(words, (node_of, slot >> 6), np.uint64(1) << (slot & 63).astype(np.uint64))


def build_onindex_grid(ijk: np.ndarray, transform: np.ndarray, translation: np.ndarray, name: str = "") -> np.ndarray:
    """Return the bytes of a NanoVDB ``ValueOnIndex`` grid whose active voxels are ``ijk``.

    Args:
        ijk: ``(N, 3)`` integer index-space voxel coordinates (duplicates allowed).
        transform: ``(3, 3)`` row-major index-to-world linear map.
        translation: ``(3,)`` index-to-world translation.
        name: Grid name (at most 255 bytes).

    Returns:
        A one-dimensional ``uint8`` array holding the grid.
    """
    ijk = np.ascontiguousarray(ijk, dtype=np.int32).reshape(-1, 3)
    # Deduplicate with a lexsort + mask: ``np.unique(axis=0)`` sorts a void
    # view of the rows and is ~15x slower, and NumPy's hash-based 1-D unique
    # degrades badly on packed 63-bit keys. Any int32 coordinate is allowed.
    if ijk.shape[0] > 1:
        ijk = ijk[np.lexsort((ijk[:, 2], ijk[:, 1], ijk[:, 0]))]
        ijk = ijk[np.concatenate(([True], np.any(ijk[1:] != ijk[:-1], axis=1)))]
    transform = np.asarray(transform, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    if ijk.shape[0] == 0:
        raise ValueError("Cannot build a NanoVDB grid with no active voxels")

    # Breadth-first node order: uppers by root key, lowers/leaves by parent then slot.
    upper_origin = (ijk >> 12) << 12
    lower_origin = (ijk >> 7) << 7
    leaf_origin = (ijk >> 3) << 3
    key = _coord_to_key(upper_origin)
    upper_slot = _bit_offsets(ijk, 4095, 7, 5)
    lower_slot = _bit_offsets(ijk, 127, 3, 4)
    leaf_slot = _bit_offsets(ijk, 7, 0, 3)
    order = np.lexsort((leaf_slot, lower_slot, upper_slot, key))
    ijk, key = ijk[order], key[order]
    upper_slot, lower_slot, leaf_slot = upper_slot[order], lower_slot[order], leaf_slot[order]
    upper_origin, lower_origin, leaf_origin = upper_origin[order], lower_origin[order], leaf_origin[order]

    _, upper_start, upper_of = np.unique(key, return_index=True, return_inverse=True)
    upper_start = np.sort(upper_start)
    upper_of = np.searchsorted(upper_start, np.arange(ijk.shape[0]), side="right") - 1
    lower_key = upper_of.astype(np.int64) * 32768 + upper_slot
    _, lower_start = np.unique(lower_key, return_index=True)
    lower_start = np.sort(lower_start)
    lower_of = np.searchsorted(lower_start, np.arange(ijk.shape[0]), side="right") - 1
    leaf_key = lower_of.astype(np.int64) * 4096 + lower_slot
    _, leaf_start = np.unique(leaf_key, return_index=True)
    leaf_start = np.sort(leaf_start)
    leaf_of = np.searchsorted(leaf_start, np.arange(ijk.shape[0]), side="right") - 1

    n_vox = ijk.shape[0]
    n_upper, n_lower, n_leaf = len(upper_start), len(lower_start), len(leaf_start)
    if n_upper * _UPPER_SIZE + n_lower * _LOWER_SIZE + n_leaf * _LEAF_SIZE > (1 << 31):
        raise ValueError(
            f"NanoVDB grid with {n_vox} voxels spanning {n_upper} upper nodes would exceed 2 GB "
            "(coordinates range over " + str(ijk.min(axis=0).tolist()) + " to " + str(ijk.max(axis=0).tolist()) + ")"
        )
    root_off = _TREE_SIZE
    upper_off = root_off + _ROOT_SIZE + n_upper * _ROOT_TILE_SIZE
    lower_off = upper_off + n_upper * _UPPER_SIZE
    leaf_off = lower_off + n_lower * _LOWER_SIZE
    total = _GRID_SIZE + leaf_off + n_leaf * _LEAF_SIZE
    buf = np.zeros(total, dtype=np.uint8)

    def bbox(starts):
        # Inclusive index-space bounding boxes of the active voxels under each node.
        mins = np.minimum.reduceat(ijk, starts, axis=0)
        maxs = np.maximum.reduceat(ijk, starts, axis=0)
        return mins, maxs

    # ---- grid header ----------------------------------------------------
    hdr = np.zeros(
        1,
        dtype=[
            ("magic", "<u8"),
            ("checksum", "<u8"),
            ("version", "<u4"),
            ("flags", "<u4"),
            ("grid_index", "<u4"),
            ("grid_count", "<u4"),
            ("grid_size", "<u8"),
            ("name", "S256"),
            ("matf", "<f4", 9),
            ("invmatf", "<f4", 9),
            ("vecf", "<f4", 3),
            ("taperf", "<f4"),
            ("matd", "<f8", 9),
            ("invmatd", "<f8", 9),
            ("vecd", "<f8", 3),
            ("taperd", "<f8"),
            ("world_bbox", "<f8", 6),
            ("voxel_size", "<f8", 3),
            ("grid_class", "<u4"),
            ("grid_type", "<u4"),
            ("blind_off", "<u8"),
            ("blind_count", "<u4"),
            ("data0", "<u4"),
            ("data1", "<u8"),
            ("data2", "<u8"),
        ],
    )
    assert hdr.itemsize == _GRID_SIZE
    inv = np.linalg.inv(transform)
    h = hdr[0]
    h["magic"] = _MAGIC_GRID
    h["checksum"] = np.uint64(0xFFFFFFFFFFFFFFFF)  # checksum disabled
    h["version"] = _VERSION
    h["flags"] = _FLAGS
    h["grid_count"] = 1
    h["grid_size"] = total
    h["name"] = name.encode()[:255]
    h["matf"] = transform.reshape(-1)
    h["invmatf"] = inv.reshape(-1)
    h["vecf"] = translation
    h["taperf"] = 1.0
    h["matd"] = transform.reshape(-1)
    h["invmatd"] = inv.reshape(-1)
    h["vecd"] = translation
    h["taperd"] = 1.0
    ijk_min, ijk_max = ijk.min(axis=0).astype(np.int64), ijk.max(axis=0).astype(np.int64)
    corners = np.array(
        [
            [x, y, z]
            for x in (ijk_min[0], ijk_max[0] + 1)
            for y in (ijk_min[1], ijk_max[1] + 1)
            for z in (ijk_min[2], ijk_max[2] + 1)
        ],
        dtype=np.float64,
    )
    world = corners @ transform.T + translation
    h["world_bbox"] = np.concatenate([world.min(axis=0), world.max(axis=0)])
    h["voxel_size"] = np.linalg.norm(transform, axis=0)
    h["grid_class"] = _GRID_CLASS_INDEX
    h["grid_type"] = _GRID_TYPE_ONINDEX
    h["blind_off"] = total
    h["data1"] = n_vox + 1  # values indexed, background included
    buf[:_GRID_SIZE] = hdr.view(np.uint8)

    # ---- tree ------------------------------------------------------------
    tree = np.array(
        [(leaf_off, lower_off, upper_off, root_off, n_leaf, n_lower, n_upper, n_leaf, n_lower, n_upper, n_vox)],
        dtype="<u8,<u8,<u8,<u8,<u4,<u4,<u4,<u4,<u4,<u4,<u8",
    )
    assert tree.itemsize == _TREE_SIZE
    buf[_GRID_SIZE : _GRID_SIZE + _TREE_SIZE] = tree.view(np.uint8)

    # ---- root + tiles ----------------------------------------------------
    base = _GRID_SIZE + root_off
    # bbox min/max (6 x i32), table size (u32), 4 pad bytes, then background /
    # min / max / average / std-dev (u64 each, all 0 for OnIndex); padded to 96.
    root = np.zeros(_ROOT_SIZE, dtype=np.uint8)
    root[0:24] = np.concatenate([ijk_min, ijk_max]).astype("<i4").view(np.uint8)
    root[24:28] = np.array([n_upper], dtype="<u4").view(np.uint8)
    buf[base : base + _ROOT_SIZE] = root
    tiles = np.zeros(n_upper, dtype="<u8,<i8,<u4,<u4,<u8")
    assert tiles.itemsize == _ROOT_TILE_SIZE
    tiles["f0"] = key[upper_start]
    tiles["f1"] = (upper_off - root_off) + np.arange(n_upper, dtype=np.int64) * _UPPER_SIZE
    tb = base + _ROOT_SIZE
    buf[tb : tb + n_upper * _ROOT_TILE_SIZE] = tiles.view(np.uint8)

    # ---- internal nodes --------------------------------------------------
    def write_internal(
        node_start,
        node_of,
        child_slot,
        child_start_idx,
        node_size,
        off_value_mask,
        off_child_mask,
        off_table,
        n_mask_words,
        n_table,
        self_off,
        child_off,
        child_size,
        region_base,
    ):
        n = len(node_start)
        nodes = np.zeros((n, node_size), dtype=np.uint8)
        mins, maxs = bbox(node_start)
        nodes[:, 0:12] = mins.astype("<i4").view(np.uint8).reshape(n, 12)
        nodes[:, 12:24] = maxs.astype("<i4").view(np.uint8).reshape(n, 12)
        nodes[:, 24:32] = np.array([_NODE_FLAGS_BBOX], dtype="<u8").view(np.uint8)
        child_mask = np.zeros((n, n_mask_words), dtype=np.uint64)
        # One child per distinct (node, slot): take the slot of the first voxel of every child node.
        first = child_start_idx
        _set_mask_bits(child_mask, node_of[first], child_slot[first])
        nodes[:, off_child_mask : off_child_mask + 8 * n_mask_words] = child_mask.view(np.uint8)
        table = np.zeros((n, n_table), dtype=np.int64)
        child_idx = np.arange(len(first), dtype=np.int64)
        table[node_of[first], child_slot[first]] = (child_off + child_idx * child_size) - (
            self_off + node_of[first].astype(np.int64) * node_size
        )
        nodes[:, off_table : off_table + 8 * n_table] = table.view(np.uint8)
        buf[region_base : region_base + n * node_size] = nodes.reshape(-1)

    write_internal(
        upper_start,
        upper_of,
        upper_slot,
        lower_start,
        _UPPER_SIZE,
        _UPPER_OFF_VALUE_MASK,
        _UPPER_OFF_CHILD_MASK,
        _UPPER_OFF_TABLE,
        512,
        32768,
        upper_off,
        lower_off,
        _LOWER_SIZE,
        _GRID_SIZE + upper_off,
    )
    write_internal(
        lower_start,
        lower_of,
        lower_slot,
        leaf_start,
        _LOWER_SIZE,
        _LOWER_OFF_VALUE_MASK,
        _LOWER_OFF_CHILD_MASK,
        _LOWER_OFF_TABLE,
        64,
        4096,
        lower_off,
        leaf_off,
        _LEAF_SIZE,
        _GRID_SIZE + lower_off,
    )

    # ---- leaves ----------------------------------------------------------
    leaves = np.zeros((n_leaf, _LEAF_SIZE), dtype=np.uint8)
    mins, maxs = bbox(leaf_start)
    leaves[:, 0:12] = mins.astype("<i4").view(np.uint8).reshape(n_leaf, 12)
    dif = (maxs - mins).astype(np.uint8)
    leaves[:, 12:15] = dif
    leaves[:, 15] = _LEAF_FLAGS
    value_mask = np.zeros((n_leaf, 8), dtype=np.uint64)
    _set_mask_bits(value_mask, leaf_of, leaf_slot)
    leaves[:, 16:80] = value_mask.view(np.uint8)
    counts = np.diff(np.append(leaf_start, n_vox))
    offsets = 1 + np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.uint64)
    leaves[:, 80:88] = offsets.astype("<u8").view(np.uint8).reshape(n_leaf, 8)
    # 9-bit running popcounts of mask words 0..k for k = 0..6.
    popcnt = np.zeros((n_leaf, 8), dtype=np.uint64)
    for w in range(8):
        v = value_mask[:, w].copy()
        c = np.zeros(n_leaf, dtype=np.uint64)
        for _ in range(64):
            c += v & np.uint64(1)
            v >>= np.uint64(1)
        popcnt[:, w] = c
    cum = np.cumsum(popcnt, axis=1)
    prefix = np.zeros(n_leaf, dtype=np.uint64)
    for k in range(7):
        prefix |= (cum[:, k] & np.uint64(511)) << np.uint64(9 * k)
    leaves[:, 88:96] = prefix.astype("<u8").view(np.uint8).reshape(n_leaf, 8)
    lb = _GRID_SIZE + leaf_off
    buf[lb : lb + n_leaf * _LEAF_SIZE] = leaves.reshape(-1)
    return buf
