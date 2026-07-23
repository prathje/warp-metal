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

"""Metal-side ``wp.Bvh`` / ``wp.Mesh`` support.

The CUDA/CPU backends pass a host or device ``BVH*`` as the ``uint64`` id a
kernel dereferences. Metal kernels can't chase host pointers, so the Metal
model is:

* Build the tree ON THE HOST with the native ``wp_bvh_create_host`` builder,
  pointed straight at the Metal arrays' unified-memory pointers (valid after
  a dispatcher sync). This reuses the exact SAH/median topology and the
  grouped-BVH leaf ordering CPU trees get.
* Mirror the resulting ``wp::BVH`` struct via ctypes, copy the node arrays
  into Metal-side ``wp.array`` storage, then free the host tree.
* Pack a descriptor buffer whose fields are the ``MTLBuffer.gpuAddress()``es
  of those arrays (see ``wp_bvh_desc_t`` in ``codegen_metal._BVH_HELPERS``).
  ``bvh.id`` IS the descriptor buffer's own gpuAddress — MSL query helpers
  ``reinterpret_cast`` their way from it to the node data.
* Register every reached buffer with the dispatcher's resident-resource set:
  they are never bound to an encoder slot, so residency + hazard tracking
  come from ``useResource`` (and, for ICB graphs, from the launcher passing
  ``reads_resident=True`` for kernels that use BVH builtins).

``refit()`` must be graph-capturable (mjlab captures ``refit_bvh`` +
raycast inside one graph), so it is a fixed sequence of small kernel
launches: one over the leaf nodes, then one per tree level from the deepest
internal level up to the root. Node bounds live in flat ``float32`` arrays
(4 floats per packed node half); the kernels only ever write components
0-2, leaving the packed child-index/leaf-flag word intact, which keeps the
whole scheme free of float<->int bitcast builtins.
"""

from __future__ import annotations

import ctypes

import numpy as np

import warp

__all__ = ["MetalBvh", "MetalMesh"]


class _NativeBvhMirror(ctypes.Structure):
    """ctypes mirror of ``wp::BVH`` (warp/native/bvh.h) for host readout."""

    _fields_ = [
        ("node_lowers", ctypes.c_void_p),
        ("node_uppers", ctypes.c_void_p),
        ("node_parents", ctypes.c_void_p),
        ("node_counts", ctypes.c_void_p),
        ("primitive_indices", ctypes.c_void_p),
        ("max_depth", ctypes.c_int),
        ("max_nodes", ctypes.c_int),
        ("num_nodes", ctypes.c_int),
        ("num_leaf_nodes", ctypes.c_int),
        ("root", ctypes.c_void_p),
        ("item_lowers", ctypes.c_void_p),
        ("item_uppers", ctypes.c_void_p),
        ("item_groups", ctypes.c_void_p),
        ("num_items", ctypes.c_int),
        ("leaf_size", ctypes.c_int),
        ("context", ctypes.c_void_p),
    ]


@warp.kernel(enable_backward=False)
def _bvh_refit_leaves(
    leaf_nodes: warp.array(dtype=warp.int32),
    leaf_prim_start: warp.array(dtype=warp.int32),
    leaf_prim_end: warp.array(dtype=warp.int32),
    primitive_indices: warp.array(dtype=warp.int32),
    item_lowers: warp.array(dtype=warp.vec3),
    item_uppers: warp.array(dtype=warp.vec3),
    node_lowers: warp.array(dtype=warp.float32),
    node_uppers: warp.array(dtype=warp.float32),
):
    tid = warp.tid()
    node = leaf_nodes[tid]
    start = leaf_prim_start[tid]
    end = leaf_prim_end[tid]
    idx = primitive_indices[start]
    lo = item_lowers[idx]
    hi = item_uppers[idx]
    for k in range(start + 1, end):
        idx = primitive_indices[k]
        lo = warp.min(lo, item_lowers[idx])
        hi = warp.max(hi, item_uppers[idx])
    base = node * 4
    node_lowers[base + 0] = lo[0]
    node_lowers[base + 1] = lo[1]
    node_lowers[base + 2] = lo[2]
    node_uppers[base + 0] = hi[0]
    node_uppers[base + 1] = hi[1]
    node_uppers[base + 2] = hi[2]


@warp.kernel(enable_backward=False)
def _bvh_refit_level(
    level_nodes: warp.array(dtype=warp.int32),
    child_left: warp.array(dtype=warp.int32),
    child_right: warp.array(dtype=warp.int32),
    node_lowers: warp.array(dtype=warp.float32),
    node_uppers: warp.array(dtype=warp.float32),
):
    tid = warp.tid()
    node = level_nodes[tid]
    lb = child_left[node] * 4
    rb = child_right[node] * 4
    base = node * 4
    node_lowers[base + 0] = warp.min(node_lowers[lb + 0], node_lowers[rb + 0])
    node_lowers[base + 1] = warp.min(node_lowers[lb + 1], node_lowers[rb + 1])
    node_lowers[base + 2] = warp.min(node_lowers[lb + 2], node_lowers[rb + 2])
    node_uppers[base + 0] = warp.max(node_uppers[lb + 0], node_uppers[rb + 0])
    node_uppers[base + 1] = warp.max(node_uppers[lb + 1], node_uppers[rb + 1])
    node_uppers[base + 2] = warp.max(node_uppers[lb + 2], node_uppers[rb + 2])


@warp.kernel(enable_backward=False)
def _mesh_compute_tri_aabbs(
    points: warp.array(dtype=warp.vec3),
    indices: warp.array(dtype=warp.int32),
    lowers: warp.array(dtype=warp.vec3),
    uppers: warp.array(dtype=warp.vec3),
):
    tid = warp.tid()
    p = points[indices[tid * 3 + 0]]
    q = points[indices[tid * 3 + 1]]
    r = points[indices[tid * 3 + 2]]
    lowers[tid] = warp.min(warp.min(p, q), r)
    uppers[tid] = warp.max(warp.max(p, q), r)


def _require_native_dispatch(what: str):
    import warp.config as _cfg  # noqa: PLC0415

    if not _cfg.metal_native_dispatch:
        raise RuntimeError(
            f"{what} on Metal requires native dispatch (wp.config.metal_native_dispatch=True or "
            "WARP_METAL_NATIVE_DISPATCH=1): the MLX launch path can't declare the BVH buffers "
            "GPU-resident for descriptor-mediated reads."
        )


def _gpu_address(arr) -> tuple[int, object]:
    """``(gpu_va, mtl_buffer)`` for a Metal ``wp.array``'s storage."""
    from warp._src.context import _metal_find_buffer  # noqa: PLC0415

    buf, off = _metal_find_buffer(arr.ptr)
    if buf is None:
        raise RuntimeError(
            f"No registered Metal buffer for array at ptr={arr.ptr} — was it allocated on the Metal device?"
        )
    return int(buf.gpuAddress()) + int(off), buf


_BVH_DESC_FMT = "<7Q6i"  # 7 gpu addresses + root/num_nodes/num_leaf_nodes/num_items/leaf_size/pad
_BVH_DESC_NBYTES = 80
_MESH_DESC_FMT = "<2Q2i"  # appended to the bvh desc: points/indices addresses + num_points/num_tris
_MESH_DESC_NBYTES = 104


class MetalBvh:
    """Metal-side tree storage + descriptor for one ``wp.Bvh`` (or a mesh's
    internal BVH). ``id`` is the descriptor buffer's gpuAddress."""

    def __init__(self, lowers, uppers, constructor: int, groups, leaf_size: int, desc_reserve: int = 0):
        _require_native_dispatch("wp.Bvh / wp.Mesh")
        from warp._src.metal_dispatch import get_dispatcher  # noqa: PLC0415

        self.device = lowers.device
        self.lowers = lowers
        self.uppers = uppers
        self.groups = groups
        self.constructor = int(constructor)
        self.leaf_size = int(leaf_size)
        self._dispatcher = get_dispatcher()
        self._resident: list = []
        self._resident_ids: set[int] = set()

        # The descriptor buffer outlives rebuilds — its gpuAddress is the
        # public id, baked into launched kernels and captured graphs.
        desc_nbytes = max(_BVH_DESC_NBYTES + int(desc_reserve), _BVH_DESC_NBYTES)
        self._desc_buf, self._desc_ptr = self._dispatcher.alloc(desc_nbytes)
        self._register(self._desc_buf)
        self.id = int(self._desc_buf.gpuAddress())

        self._build()

    # -- residency ------------------------------------------------------

    def _register(self, mtl_buf) -> None:
        if id(mtl_buf) in self._resident_ids:
            return
        self._resident_ids.add(id(mtl_buf))
        self._dispatcher.add_resident_resource(mtl_buf)
        self._resident.append(mtl_buf)

    def release(self) -> None:
        """Unregister this tree's buffers from the resident set. Called from
        ``Bvh.__del__``; safe to call more than once."""
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

    def _build(self) -> None:
        """Host-build the tree from the current lowers/uppers and (re)upload."""
        from warp._src.context import runtime  # noqa: PLC0415

        num_items = len(self.lowers)
        if num_items == 0:
            raise RuntimeError("wp.Bvh on Metal requires at least one item")

        # The host builder reads the item bounds through their unified-memory
        # pointers — drain any queued GPU writes first.
        self._dispatcher.sync()

        groups_ptr = ctypes.c_void_p(self.groups.ptr) if self.groups is not None else ctypes.c_void_p(0)
        host_id = runtime.core.wp_bvh_create_host(
            ctypes.c_void_p(self.lowers.ptr),
            ctypes.c_void_p(self.uppers.ptr),
            num_items,
            self.constructor,
            groups_ptr,
            self.leaf_size,
        )
        try:
            native = _NativeBvhMirror.from_address(host_id)
            num_nodes = int(native.num_nodes)
            self.num_nodes = num_nodes
            self.num_leaf_nodes = int(native.num_leaf_nodes)
            self.num_items = num_items

            raw_lowers = np.frombuffer(ctypes.string_at(native.node_lowers, 16 * num_nodes), dtype=np.float32).copy()
            raw_uppers = np.frombuffer(ctypes.string_at(native.node_uppers, 16 * num_nodes), dtype=np.float32).copy()
            parents = np.frombuffer(ctypes.string_at(native.node_parents, 4 * num_nodes), dtype=np.int32).copy()
            prim_indices = np.frombuffer(
                ctypes.string_at(native.primitive_indices, 4 * num_items), dtype=np.int32
            ).copy()
            self.root = int(np.frombuffer(ctypes.string_at(native.root, 4), dtype=np.int32)[0])
        finally:
            runtime.core.wp_bvh_destroy_host(host_id)

        # Decode topology from the packed-bits word (i:31 | b:1).
        bits_lower = raw_lowers.reshape(num_nodes, 4)[:, 3].view(np.uint32)
        bits_upper = raw_uppers.reshape(num_nodes, 4)[:, 3].view(np.uint32)
        is_leaf = (bits_lower >> 31).astype(bool)
        left = (bits_lower & 0x7FFFFFFF).astype(np.int32)
        right = (bits_upper & 0x7FFFFFFF).astype(np.int32)

        # Level buckets for the capturable refit: breadth-first frontiers
        # from the root. Only reachable nodes matter (the builder can leave
        # unused slots below ``max_nodes``).
        levels: list[np.ndarray] = []
        leaves: list[np.ndarray] = []
        frontier = np.array([self.root], dtype=np.int32)
        while frontier.size:
            leaf_mask = is_leaf[frontier]
            leaves.append(frontier[leaf_mask])
            internal = frontier[~leaf_mask]
            if internal.size:
                levels.append(internal)
            frontier = np.concatenate([left[internal], right[internal]]) if internal.size else np.empty(0, np.int32)
        leaf_nodes = np.concatenate(leaves) if leaves else np.empty(0, np.int32)

        # Metal-side storage. NB: ``rebuild`` reallocates these, so unlike
        # CUDA's in-place rebuild it is NOT safe for graphs captured before
        # the rebuild (the recorded refit bindings would go stale). Queries
        # stay valid — they go through the descriptor, which is rewritten.
        dev = self.device
        with warp.ScopedDevice(dev):
            self.node_lowers = warp.array(raw_lowers, dtype=warp.float32)
            self.node_uppers = warp.array(raw_uppers, dtype=warp.float32)
            self.node_parents = warp.array(parents, dtype=warp.int32)
            self.primitive_indices = warp.array(prim_indices, dtype=warp.int32)
            self.child_left = warp.array(left, dtype=warp.int32)
            self.child_right = warp.array(right, dtype=warp.int32)
            self.leaf_nodes = warp.array(leaf_nodes, dtype=warp.int32)
            self.level_nodes = [warp.array(lv, dtype=warp.int32) for lv in reversed(levels)]
            leaf_start = left[leaf_nodes] if leaf_nodes.size else np.empty(0, np.int32)
            leaf_end = right[leaf_nodes] if leaf_nodes.size else np.empty(0, np.int32)
            self.leaf_prim_start = warp.array(leaf_start, dtype=warp.int32)
            self.leaf_prim_end = warp.array(leaf_end, dtype=warp.int32)

        self._write_descriptor()
        self._dispatcher.sync()

    def _desc_bytes(self) -> bytes:
        import struct  # noqa: PLC0415

        addrs = []
        for arr in (self.node_lowers, self.node_uppers, self.node_parents, self.primitive_indices):
            ga, buf = _gpu_address(arr)
            addrs.append(ga)
            self._register(buf)
        for arr in (self.lowers, self.uppers):
            ga, buf = _gpu_address(arr)
            addrs.append(ga)
            self._register(buf)
        if self.groups is not None:
            ga, buf = _gpu_address(self.groups)
            addrs.append(ga)
            self._register(buf)
        else:
            addrs.append(0)
        return struct.pack(
            _BVH_DESC_FMT,
            *addrs,
            self.root,
            self.num_nodes,
            self.num_leaf_nodes,
            self.num_items,
            self.leaf_size,
            0,
        )

    def _write_descriptor(self) -> None:
        data = self._desc_bytes()
        ctypes.memmove(self._desc_ptr, data, len(data))

    # -- refit / rebuild --------------------------------------------------

    def refit(self) -> None:
        """Tighten node bounds to the current item bounds, topology unchanged.

        A fixed sequence of kernel launches (leaves, then one per level up
        to the root), so it records into Metal ICB graph captures.
        """
        dev = self.device
        n_leaf = self.leaf_nodes.shape[0]
        if n_leaf:
            warp.launch(
                _bvh_refit_leaves,
                dim=n_leaf,
                inputs=[
                    self.leaf_nodes,
                    self.leaf_prim_start,
                    self.leaf_prim_end,
                    self.primitive_indices,
                    self.lowers,
                    self.uppers,
                ],
                outputs=[self.node_lowers, self.node_uppers],
                device=dev,
            )
        for level in self.level_nodes:
            warp.launch(
                _bvh_refit_level,
                dim=level.shape[0],
                inputs=[level, self.child_left, self.child_right],
                outputs=[self.node_lowers, self.node_uppers],
                device=dev,
            )

    def rebuild(self, constructor: int | None = None) -> None:
        """Re-run the host builder on the current item bounds. The descriptor
        buffer (and therefore ``id``) is stable; node storage is reused when
        the new tree fits, else reallocated and the descriptor rewritten."""
        if constructor is not None:
            self.constructor = int(constructor)
        self._build()


class MetalMesh:
    """Metal-side ``wp.Mesh``: a MetalBvh over per-triangle AABBs plus a
    mesh descriptor extending the BVH descriptor with points/indices."""

    def __init__(self, points, indices, constructor: int, leaf_size: int, groups):
        _require_native_dispatch("wp.Mesh")
        self.device = points.device
        self.points = points
        self.indices = indices
        self.num_points = len(points)
        self.num_tris = int(indices.size // 3)
        if self.num_tris == 0:
            raise RuntimeError("wp.Mesh on Metal requires at least one triangle")

        with warp.ScopedDevice(self.device):
            self.tri_lowers = warp.empty(self.num_tris, dtype=warp.vec3)
            self.tri_uppers = warp.empty(self.num_tris, dtype=warp.vec3)
        self._compute_tri_aabbs()

        # The BVH descriptor sits at offset 0 of a 104-byte mesh descriptor,
        # so a mesh id doubles as a bvh id for group-root queries.
        self.bvh = MetalBvh(
            self.tri_lowers,
            self.tri_uppers,
            constructor,
            groups,
            leaf_size,
            desc_reserve=_MESH_DESC_NBYTES - _BVH_DESC_NBYTES,
        )
        self.id = self.bvh.id
        self._write_mesh_fields()
        self.bvh._dispatcher.sync()

    def _compute_tri_aabbs(self) -> None:
        warp.launch(
            _mesh_compute_tri_aabbs,
            dim=self.num_tris,
            inputs=[self.points, self.indices],
            outputs=[self.tri_lowers, self.tri_uppers],
            device=self.device,
        )

    def _write_mesh_fields(self) -> None:
        import struct  # noqa: PLC0415

        points_ga, points_buf = _gpu_address(self.points)
        indices_ga, indices_buf = _gpu_address(self.indices)
        self.bvh._register(points_buf)
        self.bvh._register(indices_buf)
        tail = struct.pack(_MESH_DESC_FMT, points_ga, indices_ga, self.num_points, self.num_tris)
        ctypes.memmove(self.bvh._desc_ptr + _BVH_DESC_NBYTES, tail, len(tail))

    def set_points(self, points_new) -> None:
        self.points = points_new
        self._write_mesh_fields()
        self.refit()

    def refit(self) -> None:
        """Recompute triangle AABBs from the current points, then refit the
        BVH. Capturable (kernel launches only)."""
        self._compute_tri_aabbs()
        self.bvh.refit()

    def release(self) -> None:
        self.bvh.release()
