# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-side sparse topology ops for the Metal backend.

``bsr_set_transpose`` builds the transposed topology with a native call
(``wp_bsr_transpose_host`` / ``_device``). On Metal the host variant reads
unified memory in place, but a host op cannot be replayed from a Metal graph
capture. This module builds the same compact topology with dispatches only:
key every active source block by ``(col, row)``, radix-sort the keys with the
block indices, then scatter columns, row offsets and the block map. The
result matches ``wp_bsr_transpose_host`` exactly: blocks ordered by
destination row, then destination column.
"""

from typing import Any

import warp


@warp.kernel(enable_backward=False, module="unique")
def _bsr_transpose_keys_metal(
    row_count: int,
    nnz: int,
    sentinel: Any,
    offsets: warp.array[int],
    row_counts: warp.array[int],
    has_row_counts: int,
    columns: warp.array[int],
    keys: warp.array[Any],
):
    # One thread per source row; blocks outside every active row range keep
    # the sentinel key and sort to the end.
    row = warp.tid()
    beg = offsets[row]
    end = offsets[row + 1]
    if has_row_counts != 0:
        end = warp.min(beg + row_counts[row], end)
    end = warp.min(end, nnz)
    for block in range(beg, end):
        keys[block] = type(sentinel)(columns[block]) * type(sentinel)(row_count) + type(sentinel)(row)


@warp.kernel(enable_backward=False, module="unique")
def _bsr_transpose_init_keys_metal(sentinel: Any, keys: warp.array[Any], indices: warp.array[int]):
    i = warp.tid()
    keys[i] = sentinel
    indices[i] = i


@warp.kernel(enable_backward=False, module="unique")
def _bsr_transpose_emit_metal(
    nnz: int,
    src_row_count: int,
    sentinel: Any,
    keys: warp.array[Any],
    sorted_blocks: warp.array[int],
    dest_columns: warp.array[int],
    block_index_map: warp.array[int],
    row_ends: warp.array[int],
):
    i = warp.tid()
    key = keys[i]
    if key == sentinel:
        return
    dest_row = int(key / type(sentinel)(src_row_count))
    dest_columns[i] = int(key - type(sentinel)(dest_row) * type(sentinel)(src_row_count))
    block_index_map[i] = sorted_blocks[i]
    row_last = i + 1 == nnz
    if not row_last:
        nxt = keys[i + 1]
        row_last = nxt == sentinel or int(nxt / type(sentinel)(src_row_count)) != dest_row
    if row_last:
        row_ends[dest_row] = i + 1


@warp.kernel(enable_backward=False, module="unique")
def _bsr_transpose_offsets_metal(row_ends: warp.array[int], offsets: warp.array[int]):
    # ``row_ends`` holds each non-empty destination row's end (else -1); an
    # empty row ends where the previous non-empty one does.
    r = warp.tid()
    if r == 0:
        offsets[0] = 0
    end = row_ends[r]
    k = r
    while end < 0 and k > 0:
        k -= 1
        end = row_ends[k]
    offsets[r + 1] = warp.max(end, 0)


def bsr_transpose_topology_metal(src, dest, nnz: int) -> "warp.array":
    """Write the compact transposed topology of ``src`` into ``dest``.

    ``nnz`` is the host-side upper bound of ``src``'s stored blocks; the
    active count is read on the device from ``src.offsets`` /
    ``src.row_counts``, so the build records into a Metal graph capture.
    ``dest`` must already have ``src.ncol`` rows and room for ``nnz`` blocks.

    Returns the block map consumed by ``_bsr_transpose_values``: entry ``k``
    is the source block stored at destination slot ``k``.
    """
    from warp._src import metal_utils as _mu  # noqa: PLC0415

    device = dest.device
    span = src.nrow * src.ncol
    key_type = warp.uint32 if span < 2**32 - 1 else warp.uint64
    sentinel = key_type(span)
    end_bit = max(int(span).bit_length(), 1)
    # The radix sort ping-pongs through the second half.
    keys = warp.empty(2 * nnz, dtype=key_type, device=device)
    indices = warp.empty(2 * nnz, dtype=int, device=device)
    warp.launch(
        _bsr_transpose_init_keys_metal, dim=nnz, device=device, inputs=[sentinel, keys, indices], record_tape=False
    )
    has_row_counts = src.row_counts is not None
    row_counts = src.row_counts if has_row_counts else warp.empty(1, dtype=int, device=device)
    warp.launch(
        _bsr_transpose_keys_metal,
        dim=src.nrow,
        device=device,
        inputs=[src.nrow, nnz, sentinel, src.offsets, row_counts, int(has_row_counts), src.columns, keys],
        record_tape=False,
    )
    _mu.radix_sort_pairs(keys, indices, nnz, 0, end_bit)
    block_index_map = warp.full(nnz, -1, dtype=int, device=device)
    row_ends = warp.full(dest.nrow, -1, dtype=int, device=device)
    warp.launch(
        _bsr_transpose_emit_metal,
        dim=nnz,
        device=device,
        inputs=[nnz, src.nrow, sentinel, keys, indices, dest.columns, block_index_map, row_ends],
        record_tape=False,
    )
    warp.launch(
        _bsr_transpose_offsets_metal, dim=dest.nrow, device=device, inputs=[row_ends, dest.offsets], record_tape=False
    )
    # Like the native transpose, the topology build stays off any wp.Tape.
    if device.is_capturing:
        graph = warp._src.context.runtime._metal_capture_graph
        graph._metal_retain.extend((keys, indices, row_ends, row_counts, block_index_map))
    return block_index_map
