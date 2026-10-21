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

"""Solid-angle node data for ``wp.Mesh(support_winding_number=True)`` on Metal.

A port of ``precompute_triangle_solid_angle_props`` /
``combine_precomputed_solid_angle_props`` (``warp/native/solid_angle.h``) and
the host refit in ``mesh.cpp``. It runs as a CPU Warp kernel over the Metal
mesh's own BVH (node arrays read through unified memory), and the result is
uploaded as a flat ``float[num_nodes * 36]`` array whose layout matches the
MSL ``wp_solid_angle_props_t`` in ``codegen_metal``.
"""

from __future__ import annotations

import numpy as np

import warp

PROPS_FLOATS = 36


@warp.struct
class _SAProps:
    average_p: warp.vec3
    normal: warp.vec3
    n_ij_diag: warp.vec3
    n_ijk_diag: warp.vec3
    sum_permute_n_xyz: float
    two_n_xxy_n_yxx: float
    two_n_xxz_n_zxx: float
    two_n_yyz_n_zyy: float
    two_n_yyx_n_xyy: float
    two_n_zzx_n_xzz: float
    two_n_zzy_n_yzz: float
    n_xy: float
    n_yx: float
    n_yz: float
    n_zy: float
    n_zx: float
    n_xz: float
    box_lower: warp.vec3
    box_upper: warp.vec3
    area_p: warp.vec3
    area: float
    max_p_dist_sq: float


@warp.func
def _integrals(a: warp.vec3, b: warp.vec3, c: warp.vec3, P: warp.vec3, i: int, want_j: bool, want_k: bool):
    # Returns (integral_ii, integral_ij, integral_ik); see compute_integrals().
    oab = b - a
    oac = c - a
    ocb = b - c
    t = oab[i] / oac[i]
    j = int(0)
    if i != 2:
        j = i + 1
    k = int(0)
    if j != 2:
        k = j + 1
    jdiff = t * oac[j] - oab[j]
    kdiff = t * oac[k] - oab[k]
    cross_a = warp.vec3(jdiff * oab[k] - kdiff * oab[j], kdiff * oab[i], jdiff * oab[i])
    cross_c = warp.vec3(jdiff * ocb[k] - kdiff * ocb[j], kdiff * ocb[i], jdiff * ocb[i])
    area_scale_a = warp.length(cross_a)
    area_scale_c = warp.length(cross_c)
    Pai = a[i] - P[i]
    Pci = c[i] - P[i]
    int_ii_a = area_scale_a * (0.5 * Pai * Pai + (2.0 / 3.0) * Pai * oab[i] + 0.25 * oab[i] * oab[i])
    int_ii_c = area_scale_c * (0.5 * Pci * Pci + (2.0 / 3.0) * Pci * ocb[i] + 0.25 * ocb[i] * ocb[i])
    ii = int_ii_a + int_ii_c
    res_j = float(0.0)
    res_k = float(0.0)
    for it in range(2):
        jk = j
        diff = jdiff
        want = want_j
        if it == 1:
            jk = k
            diff = kdiff
            want = want_k
        if want:
            obmidj = b[jk] + 0.5 * diff
            oabmidj = obmidj - a[jk]
            ocbmidj = obmidj - c[jk]
            Paj = a[jk] - P[jk]
            Pcj = c[jk] - P[jk]
            int_ij_a = area_scale_a * (
                0.5 * Pai * Paj + (1.0 / 3.0) * Pai * oabmidj + (1.0 / 3.0) * Paj * oab[i] + 0.25 * oab[i] * oabmidj
            )
            int_ij_c = area_scale_c * (
                0.5 * Pci * Pcj + (1.0 / 3.0) * Pci * ocbmidj + (1.0 / 3.0) * Pcj * ocb[i] + 0.25 * ocb[i] * ocbmidj
            )
            if it == 0:
                res_j = int_ij_a + int_ij_c
            else:
                res_k = int_ij_a + int_ij_c
    return warp.vec3(ii, res_j, res_k)


@warp.func
def _sort3(a: warp.vec3, b: warp.vec3, c: warp.vec3, axis: int):
    o0 = int(0)
    o1 = int(1)
    o2 = int(2)
    vals = warp.matrix_from_rows(a, b, c)
    if a[axis] > b[axis]:
        tmp = o0
        o0 = o1
        o1 = tmp
    if vals[o0, axis] > c[axis]:
        tmp = o0
        o0 = o2
        o2 = tmp
    if vals[o1, axis] > vals[o2, axis]:
        tmp = o1
        o1 = o2
        o2 = tmp
    return warp.vec3i(o0, o1, o2)


@warp.func
def _tri_props(a: warp.vec3, b: warp.vec3, c: warp.vec3) -> _SAProps:
    d = _SAProps()
    ab = b - a
    ac = c - a
    N = 0.5 * warp.cross(ab, ac)
    area2 = warp.length_sq(N)
    area = warp.sqrt(area2)
    P = (a + b + c) / 3.0
    d.box_lower = warp.min(warp.min(a, b), c)
    d.box_upper = warp.max(warp.max(a, b), c)
    d.average_p = P
    d.area_p = P * area
    d.normal = N
    d.area = area
    if area == 0.0:
        return d
    n = N / area
    vals = warp.matrix_from_rows(a, b, c)
    ox = _sort3(a, b, c, 0)
    oy = _sort3(a, b, c, 1)
    oz = _sort3(a, b, c, 2)
    dx = vals[ox[2], 0] - vals[ox[0], 0]
    dy = vals[oy[2], 1] - vals[oy[0], 1]
    dz = vals[oz[2], 2] - vals[oz[0], 2]
    ixx = float(0.0)
    ixy = float(0.0)
    iyy = float(0.0)
    iyz = float(0.0)
    izz = float(0.0)
    izx = float(0.0)
    if dx > 0.0:
        r = _integrals(vals[ox[0]], vals[ox[1]], vals[ox[2]], P, 0, dx >= dy and dy > 0.0, dx >= dz and dz > 0.0)
        ixx = r[0]
        if dx >= dy and dy > 0.0:
            ixy = r[1]
        if dx >= dz and dz > 0.0:
            izx = r[2]
    if dy > 0.0:
        r = _integrals(vals[oy[0]], vals[oy[1]], vals[oy[2]], P, 1, dy >= dz and dz > 0.0, dx < dy and dx > 0.0)
        iyy = r[0]
        if dy >= dz and dz > 0.0:
            iyz = r[1]
        if dx < dy and dx > 0.0:
            ixy = r[2]
    if dz > 0.0:
        r = _integrals(vals[oz[0]], vals[oz[1]], vals[oz[2]], P, 2, dx < dz and dx > 0.0, dy < dz and dy > 0.0)
        izz = r[0]
        if dx < dz and dx > 0.0:
            izx = r[1]
        if dy < dz and dy > 0.0:
            iyz = r[2]
    d.n_ijk_diag = warp.cw_mul(warp.vec3(ixx, iyy, izz), n)
    d.sum_permute_n_xyz = 2.0 * (n[0] * iyz + n[1] * izx + n[2] * ixy)
    d.two_n_xxy_n_yxx = 2.0 * n[0] * ixy + n[1] * ixx
    d.two_n_xxz_n_zxx = 2.0 * n[0] * izx + n[2] * ixx
    d.two_n_yyz_n_zyy = 2.0 * n[1] * iyz + n[2] * iyy
    d.two_n_yyx_n_xyy = 2.0 * n[1] * ixy + n[0] * iyy
    d.two_n_zzx_n_xzz = 2.0 * n[2] * izx + n[0] * izz
    d.two_n_zzy_n_yzz = 2.0 * n[2] * iyz + n[1] * izz
    return d


@warp.func
def _accum_child(m: _SAProps, c: _SAProps) -> _SAProps:
    disp = c.average_p - m.average_p
    N = c.normal
    m.n_ij_diag = m.n_ij_diag + warp.cw_mul(N, disp)
    Nxy = c.n_xy + N[0] * disp[1]
    Nyx = c.n_yx + N[1] * disp[0]
    Nyz = c.n_yz + N[1] * disp[2]
    Nzy = c.n_zy + N[2] * disp[1]
    Nzx = c.n_zx + N[2] * disp[0]
    Nxz = c.n_xz + N[0] * disp[2]
    m.n_xy = m.n_xy + Nxy
    m.n_yx = m.n_yx + Nyx
    m.n_yz = m.n_yz + Nyz
    m.n_zy = m.n_zy + Nzy
    m.n_zx = m.n_zx + Nzx
    m.n_xz = m.n_xz + Nxz
    m.n_ijk_diag = m.n_ijk_diag + 2.0 * warp.cw_mul(disp, c.n_ij_diag) + warp.cw_mul(disp, warp.cw_mul(disp, N))
    m.sum_permute_n_xyz = m.sum_permute_n_xyz + (disp[0] * (Nyz + Nzy) + disp[1] * (Nzx + Nxz) + disp[2] * (Nxy + Nyx))
    m.two_n_xxy_n_yxx = (
        m.two_n_xxy_n_yxx
        + 2.0 * (disp[1] * c.n_ij_diag[0] + disp[0] * c.n_xy + N[0] * disp[0] * disp[1])
        + 2.0 * c.n_yx * disp[0]
        + N[1] * disp[0] * disp[0]
    )
    m.two_n_xxz_n_zxx = (
        m.two_n_xxz_n_zxx
        + 2.0 * (disp[2] * c.n_ij_diag[0] + disp[0] * c.n_xz + N[0] * disp[0] * disp[2])
        + 2.0 * c.n_zx * disp[0]
        + N[2] * disp[0] * disp[0]
    )
    m.two_n_yyz_n_zyy = (
        m.two_n_yyz_n_zyy
        + 2.0 * (disp[2] * c.n_ij_diag[1] + disp[1] * c.n_yz + N[1] * disp[1] * disp[2])
        + 2.0 * c.n_zy * disp[1]
        + N[2] * disp[1] * disp[1]
    )
    m.two_n_yyx_n_xyy = (
        m.two_n_yyx_n_xyy
        + 2.0 * (disp[0] * c.n_ij_diag[1] + disp[1] * c.n_yx + N[1] * disp[1] * disp[0])
        + 2.0 * c.n_xy * disp[1]
        + N[0] * disp[1] * disp[1]
    )
    m.two_n_zzx_n_xzz = (
        m.two_n_zzx_n_xzz
        + 2.0 * (disp[0] * c.n_ij_diag[2] + disp[2] * c.n_zx + N[2] * disp[2] * disp[0])
        + 2.0 * c.n_xz * disp[2]
        + N[0] * disp[2] * disp[2]
    )
    m.two_n_zzy_n_yzz = (
        m.two_n_zzy_n_yzz
        + 2.0 * (disp[1] * c.n_ij_diag[2] + disp[2] * c.n_zy + N[2] * disp[2] * disp[1])
        + 2.0 * c.n_yz * disp[2]
        + N[1] * disp[2] * disp[2]
    )
    return m


@warp.func
def _combine(l: _SAProps, r: _SAProps, has_r: bool) -> _SAProps:
    m = _SAProps()
    N = l.normal
    areaP = l.area_p
    area = l.area
    lo = l.box_lower
    hi = l.box_upper
    if has_r:
        N = N + r.normal
        areaP = areaP + r.area_p
        area = area + r.area
        lo = warp.min(lo, r.box_lower)
        hi = warp.max(hi, r.box_upper)
    m.normal = N
    m.area_p = areaP
    m.area = area
    m.box_lower = lo
    m.box_upper = hi
    if area > 0.0:
        m.average_p = areaP / area
    else:
        m.average_p = 0.5 * (lo + hi)
    m.n_ij_diag = l.n_ij_diag
    m.n_ijk_diag = l.n_ijk_diag
    m.sum_permute_n_xyz = l.sum_permute_n_xyz
    m.two_n_xxy_n_yxx = l.two_n_xxy_n_yxx
    m.two_n_xxz_n_zxx = l.two_n_xxz_n_zxx
    m.two_n_yyz_n_zyy = l.two_n_yyz_n_zyy
    m.two_n_yyx_n_xyy = l.two_n_yyx_n_xyy
    m.two_n_zzx_n_xzz = l.two_n_zzx_n_xzz
    m.two_n_zzy_n_yzz = l.two_n_zzy_n_yzz
    if has_r:
        m.n_ij_diag = m.n_ij_diag + r.n_ij_diag
        m.n_ijk_diag = m.n_ijk_diag + r.n_ijk_diag
        m.sum_permute_n_xyz = m.sum_permute_n_xyz + r.sum_permute_n_xyz
        m.two_n_xxy_n_yxx = m.two_n_xxy_n_yxx + r.two_n_xxy_n_yxx
        m.two_n_xxz_n_zxx = m.two_n_xxz_n_zxx + r.two_n_xxz_n_zxx
        m.two_n_yyz_n_zyy = m.two_n_yyz_n_zyy + r.two_n_yyz_n_zyy
        m.two_n_yyx_n_xyy = m.two_n_yyx_n_xyy + r.two_n_yyx_n_xyy
        m.two_n_zzx_n_xzz = m.two_n_zzx_n_xzz + r.two_n_zzx_n_xzz
        m.two_n_zzy_n_yzz = m.two_n_zzy_n_yzz + r.two_n_zzy_n_yzz
    m = _accum_child(m, l)
    if has_r:
        m = _accum_child(m, r)
    m.max_p_dist_sq = warp.length_sq(warp.max(m.average_p - m.box_lower, m.box_upper - m.average_p))
    return m


@warp.kernel(enable_backward=False, module="unique")
def _solid_angle_props_kernel(
    root: int,
    left: warp.array[int],
    right: warp.array[int],
    is_leaf: warp.array[int],
    prims: warp.array[int],
    points: warp.array[warp.vec3],
    indices: warp.array[int],
    stack: warp.array[int],
    state: warp.array[int],
    props: warp.array[_SAProps],
):
    # Iterative post-order walk (``bvh_refit_with_solid_angle_recursive_host``).
    count = int(1)
    stack[0] = root
    state[root] = 0
    while count > 0:
        node = stack[count - 1]
        if is_leaf[node] != 0:
            for pc in range(left[node], right[node]):
                p = prims[pc]
                t = _tri_props(points[indices[p * 3 + 0]], points[indices[p * 3 + 1]], points[indices[p * 3 + 2]])
                if pc == left[node]:
                    props[node] = t
                else:
                    props[node] = _combine(props[node], t, True)
            count -= 1
        elif state[node] == 0:
            state[node] = 1
            stack[count] = left[node]
            state[left[node]] = 0
            count += 1
            if right[node] != left[node]:
                stack[count] = right[node]
                state[right[node]] = 0
                count += 1
        else:
            l = left[node]
            r = right[node]
            props[node] = _combine(props[l], props[r], l != r)
            count -= 1


def compute_solid_angle_props(bvh, points, indices) -> np.ndarray:
    """``float32[num_nodes, 36]`` solid-angle data for a :class:`MetalBvh`."""
    lowers = bvh.node_lowers.numpy().view(np.uint32).reshape(-1, 4)
    uppers = bvh.node_uppers.numpy().view(np.uint32).reshape(-1, 4)
    num_nodes = lowers.shape[0]
    left = (lowers[:, 3] & 0x7FFFFFFF).astype(np.int32)
    right = (uppers[:, 3] & 0x7FFFFFFF).astype(np.int32)
    leaf = ((lowers[:, 3] & 0x80000000) != 0).astype(np.int32)
    cpu = "cpu"
    props = warp.zeros(num_nodes, dtype=_SAProps, device=cpu)
    warp.launch(
        _solid_angle_props_kernel,
        dim=1,
        inputs=[
            int(bvh.root),
            warp.array(left, dtype=int, device=cpu),
            warp.array(right, dtype=int, device=cpu),
            warp.array(leaf, dtype=int, device=cpu),
            warp.array(bvh.primitive_indices.numpy().astype(np.int32), dtype=int, device=cpu),
            warp.array(points.numpy(), dtype=warp.vec3, device=cpu),
            warp.array(indices.numpy().reshape(-1).astype(np.int32), dtype=int, device=cpu),
            warp.zeros(num_nodes + 2, dtype=int, device=cpu),
            warp.zeros(num_nodes, dtype=int, device=cpu),
        ],
        outputs=[props],
        device=cpu,
        record_tape=False,
    )
    out = props.numpy().view(np.float32).reshape(num_nodes, -1)
    if out.shape[1] != PROPS_FLOATS:
        raise RuntimeError(f"Unexpected solid-angle record size {out.shape}")
    return np.ascontiguousarray(out)
