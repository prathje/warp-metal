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

"""Metal-vs-CPU checks for tile ops, textures and mesh features added for Newton.

Every kernel runs on ``cpu`` and ``metal:0`` and the outputs must agree. The
Metal device must be enabled before ``wp.init()``, so each case runs in a
subprocess (see ``test_metal_launch``).
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_native

_COMPARE = textwrap.dedent(
    """
    import numpy as np

    def compare(kernel, dim, make_args, out_idx, block_dim=32, tiled=True, atol=1e-5):
        outs = {}
        for dev in ("cpu", "metal:0"):
            args = make_args(dev)
            if tiled:
                wp.launch_tiled(kernel, dim=dim, inputs=args, block_dim=block_dim, device=dev)
            else:
                wp.launch(kernel, dim=dim, inputs=args, device=dev)
            outs[dev] = [args[i].numpy() for i in out_idx]
        for a, b in zip(outs["cpu"], outs["metal:0"]):
            np.testing.assert_allclose(b, a, rtol=1e-4, atol=atol)
    """
)


def _run(test_case, body: str, timeout: int = 300):
    _run_with_metal_native(test_case, _COMPARE + textwrap.dedent(body), timeout=timeout)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalTilesExt(unittest.TestCase):
    def test_scatter_add_and_masked(self):
        _run(
            self,
            """
            T = 8

            @wp.kernel
            def k(x: wp.array2d[float], out: wp.array2d[float], out2: wp.array3d[float]):
                b, t = wp.tid()
                s = wp.tile_zeros(shape=T, dtype=float, storage="shared")
                m = wp.tile_zeros(shape=(T, 2), dtype=float, storage="shared")
                for i in range(t, T, wp.block_dim()):
                    wp.tile_scatter_add(s, i % 4, x[b, i], True)
                    wp.tile_scatter_masked(m, i, 1, x[b, i] * 2.0, i % 2 == 0)
                wp.tile_store(out[b], s)
                wp.tile_store(out2[b], m)

            v = np.random.default_rng(0).standard_normal((4, T)).astype(np.float32)
            compare(k, [4], lambda d: [wp.array(v, device=d), wp.zeros((4, T), dtype=float, device=d),
                                       wp.zeros((4, T, 2), dtype=float, device=d)], [1, 2])
            """,
        )

    def test_scatter_masked_progress_counter(self):
        # Only lane 0 advances the shared counter: running the extra lanes of a
        # launch_tiled block serially would loop forever on Metal.
        _run(
            self,
            """
            @wp.kernel
            def k(n: int, out: wp.array[int]):
                b, t = wp.tid()
                progress = wp.tile_zeros(shape=1, dtype=int, storage="shared")
                wp.tile_scatter_masked(progress, 0, 0, t == 0)
                count = int(0)
                while wp.tile_extract(progress, 0) < n:
                    count += 1
                    wp.tile_scatter_masked(progress, 0, wp.tile_extract(progress, 0) + 3, t == 0)
                if t == 0:
                    out[b] = count

            compare(k, [4], lambda d: [10, wp.zeros(4, dtype=int, device=d)], [1])
            """,
            timeout=120,
        )

    def test_load_indexed_reshape_view(self):
        _run(
            self,
            """
            T = 8

            @wp.kernel
            def k_li(a: wp.array2d[float], idx: wp.array[int], out: wp.array2d[float], out2: wp.array2d[float]):
                w, t = wp.tid()
                ti = wp.tile_load(idx, shape=(4,), storage="shared")
                row = wp.tile_load_indexed(a[w], ti, shape=(4,), storage="shared")
                wp.tile_store(out[w], row)
                blk = wp.tile_load_indexed(a, indices=ti, shape=(4, T), offset=(0, 0), axis=0)
                wp.tile_store(out2, blk)

            @wp.kernel
            def k_rv(x: wp.array2d[float], a: wp.array3d[float], out: wp.array2d[float], out3: wp.array3d[float]):
                b, t = wp.tid()
                tl = wp.tile_load(x[b], shape=T, storage="shared")
                col = wp.tile_reshape(tl, shape=(T, 1))
                m = wp.tile_matmul(wp.tile_ones(shape=(T, T), dtype=float), col)
                wp.tile_store(out[b], wp.tile_reshape(m, shape=(T,)))
                A = wp.tile_load(a[b], shape=(T, T), storage="shared")
                sub = wp.tile_view(A, offset=(0, 4), shape=(T, 4))
                sub += wp.tile_ones(shape=(T, 4), dtype=float)
                o = wp.tile_zeros(shape=(T, T), dtype=float)
                wp.tile_assign(o, wp.tile_view(A, offset=(2, 2), shape=(4, 4)), offset=(1, 3))
                wp.tile_store(out3[b], wp.tile_map(wp.add, A, o))

            rng = np.random.default_rng(1)
            v = rng.standard_normal((4, T)).astype(np.float32)
            A = rng.standard_normal((4, T, T)).astype(np.float32)
            compare(k_li, [4], lambda d: [wp.array(v, device=d), wp.array([3, 1, -1, 2], dtype=int, device=d),
                                          wp.zeros((4, 4), dtype=float, device=d),
                                          wp.zeros((4, T), dtype=float, device=d)], [2, 3])
            compare(k_rv, [4], lambda d: [wp.array(v, device=d), wp.array(A, device=d),
                                          wp.zeros((4, T), dtype=float, device=d),
                                          wp.zeros((4, T, T), dtype=float, device=d)], [2, 3])
            """,
        )

    def test_column_tiles_and_unit_leading_3d(self):
        _run(
            self,
            """
            @wp.kernel
            def k_col(b3: wp.array3d[float], out3: wp.array3d[float]):
                batch, rhs, _t = wp.tid()
                t = wp.tile_load(b3[batch], shape=(4, 1), offset=(2, rhs))
                wp.tile_store(out3[batch], wp.tile_map(wp.mul, t, t), offset=(1, rhs))
                wp.tile_atomic_add(out3[batch], t, offset=(3, rhs))

            @wp.kernel
            def k_3d(a: wp.array3d[float], out: wp.array3d[float]):
                w, t = wp.tid()
                t2 = wp.tile_reshape(wp.tile_load(a, shape=(1, 4, 4), offset=(w, 2, 1)), (4, 4))
                acc = wp.tile_zeros(shape=(4, 4), dtype=float)
                wp.tile_matmul(t2, wp.tile_transpose(t2), acc)
                wp.tile_store(out, wp.tile_reshape(acc, (1, 4, 4)), offset=(w, 1, 2))

            rng = np.random.default_rng(2)
            b = rng.standard_normal((2, 8, 5)).astype(np.float32)
            a = rng.standard_normal((3, 8, 8)).astype(np.float32)
            compare(k_col, [2, 5], lambda d: [wp.array(b, device=d), wp.zeros((2, 8, 5), dtype=float, device=d)], [1])
            compare(k_3d, [3], lambda d: [wp.array(a, device=d), wp.zeros((3, 8, 8), dtype=float, device=d)], [1])
            """,
        )

    def test_tile_stack_and_transform_tiles(self):
        _run(
            self,
            """
            @wp.kernel
            def k_stack(x: wp.array2d[int], out: wp.array2d[int]):
                b, t = wp.tid()
                st = wp.tile_stack(capacity=16, dtype=int)
                for i in range(t, 8, wp.block_dim()):
                    wp.tile_stack_push(st, x[b, i], x[b, i] % 2 == 0)
                n = wp.tile_stack_count(st)
                while wp.tile_stack_count(st) > 0:
                    val, slot = wp.tile_stack_pop(st)
                    if slot >= 0:
                        wp.atomic_add(out, b, 1, val)
                wp.tile_stack_clear(st)
                if t == 0:
                    out[b, 0] = n + wp.tile_stack_count(st)

            @wp.kernel
            def k_xf(xs: wp.array[wp.transform], out: wp.array[wp.transform]):
                b, t = wp.tid()
                buf = wp.tile_zeros(shape=8, dtype=wp.transform, storage="shared")
                for i in range(8):
                    wp.tile_scatter_masked(buf, (i * 3) % 8, xs[i], i % 3 != 0)
                for i in range(8):
                    out[b * 8 + i] = buf[i] * xs[0]

            rng = np.random.default_rng(3)
            xi = rng.integers(0, 5, (4, 8)).astype(np.int32)
            q = rng.standard_normal((8, 4))
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            xf = np.concatenate([rng.standard_normal((8, 3)), q], axis=1).astype(np.float32)
            compare(k_stack, [4], lambda d: [wp.array(xi, dtype=int, device=d), wp.zeros((4, 2), dtype=int, device=d)], [1])
            compare(k_xf, [2], lambda d: [wp.array(xf, dtype=wp.transform, device=d),
                                          wp.zeros(16, dtype=wp.transform, device=d)], [1])
            """,
        )

    def test_tile_mesh_query_aabb(self):
        _run(
            self,
            """
            @wp.kernel
            def k(mesh: wp.uint64, lo: wp.array[wp.vec3], hi: wp.array[wp.vec3], out: wp.array2d[int]):
                b, t = wp.tid()
                q = wp.tile_mesh_query_aabb(mesh, lo[b], hi[b])
                while wp.tile_query_valid(q):
                    tri = wp.untile(wp.tile_mesh_query_aabb_next(q))
                    if tri >= 0:
                        out[b, tri] = 1

            g = np.stack(np.meshgrid(np.arange(5), np.arange(5), indexing="ij"), -1).reshape(-1, 2)
            pts = np.concatenate([g, np.zeros((25, 1))], 1).astype(np.float32)
            tris = []
            for i in range(4):
                for j in range(4):
                    a = i * 5 + j
                    tris += [a, a + 1, a + 5, a + 1, a + 6, a + 5]
            keep = []

            def make(d):
                m = wp.Mesh(points=wp.array(pts, dtype=wp.vec3, device=d),
                            indices=wp.array(np.array(tris, np.int32), device=d))
                keep.append(m)
                lo = wp.array([[0.5, 0.5, -1], [2.2, 1.1, -1], [9, 9, 9]], dtype=wp.vec3, device=d)
                hi = wp.array([[1.5, 1.5, 1], [3.7, 3.2, 1], [10, 10, 10]], dtype=wp.vec3, device=d)
                return [m.id, lo, hi, wp.zeros((3, 32), dtype=int, device=d)]

            compare(k, [3], make, [3])
            """,
        )

    def test_tile_index_read_from_lane_table(self):
        # An index loaded from a table addressed by the lane is data, not a
        # block_dim-derived index (Newton's level-scheduled tiled FK).
        _run(
            self,
            """
            @wp.kernel
            def k(table: wp.array[int], out: wp.array[float]):
                b, t = wp.tid()
                buf = wp.tile_zeros(shape=8, dtype=float, storage="shared")
                for c in range((8 + wp.block_dim() - 1) // wp.block_dim()):
                    lane = c * wp.block_dim() + t
                    if lane < 8:
                        slot = table[lane]
                        wp.tile_scatter_masked(buf, lane, float(slot) * 2.0, True)
                        out[b * 8 + lane] = buf[table[7 - lane]]

            compare(k, [2], lambda d: [wp.array([3, 1, 4, 1, 5, 2, 6, 0], dtype=int, device=d),
                                       wp.zeros(16, dtype=float, device=d)], [1], block_dim=1)
            """,
        )

    def test_volume_voxel_count(self):
        _run(
            self,
            """
            @wp.kernel
            def k(v: wp.uint64, o: wp.array[int]):
                o[0] = wp.volume_voxel_count(v)

            pts = np.random.default_rng(0).random((200, 3)).astype(np.float32) * 20
            keep = []

            def make(d):
                vol = wp.Volume.allocate_by_voxels(wp.array(pts, dtype=wp.vec3, device=d), voxel_size=0.5, device=d)
                keep.append(vol)
                return [vol.id, wp.zeros(1, dtype=int, device=d)]

            compare(k, 1, make, [1], tiled=False)
            """,
        )

    def test_tile_map_builtin_adjoint(self):
        _run(
            self,
            """
            @wp.kernel
            def k(x: wp.array[float], out: wp.array[float]):
                b, t = wp.tid()
                tl = wp.tile_load(x, shape=8, offset=b * 8)
                m = wp.tile_map(wp.mul, wp.tile_map(wp.sin, tl), tl)
                wp.tile_atomic_add(out, wp.tile_sum(m), offset=b)

            v = np.random.default_rng(4).standard_normal(32).astype(np.float32)
            g = {}
            for dev in ("cpu", "metal:0"):
                x = wp.array(v, device=dev, requires_grad=True)
                o = wp.zeros(4, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch_tiled(k, dim=[4], inputs=[x, o], block_dim=32, device=dev)
                o.grad = wp.ones(4, dtype=float, device=dev)
                tape.backward()
                g[dev] = x.grad.numpy()
            np.testing.assert_allclose(g["metal:0"], g["cpu"], rtol=1e-4, atol=1e-5)
            """,
        )


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalTextures(unittest.TestCase):
    def test_texture_sample_matches_cpu(self):
        _run(
            self,
            """
            import itertools

            @wp.kernel
            def s2(tex: wp.Texture2D, uv: wp.array[wp.vec2], lod: float, o1: wp.array[float], o4: wp.array[wp.vec4]):
                i = wp.tid()
                o1[i] = wp.texture_sample(tex, uv[i], dtype=float, lod=lod)
                o4[i] = wp.texture_sample(tex, uv[i][0], uv[i][1], dtype=wp.vec4f, lod=lod)

            @wp.kernel
            def s3(tex: wp.Texture3D, p: wp.array[wp.vec3], lod: float, o1: wp.array[float], o4: wp.array[wp.vec4]):
                i = wp.tid()
                o1[i] = wp.texture_sample(tex, p[i], dtype=float, lod=lod)
                o4[i] = wp.texture_sample(tex, p[i], dtype=wp.vec4f, lod=lod)

            rng = np.random.default_rng(0)
            for ndim, dt, filt, addr, norm, mips in itertools.product(
                (2, 3), (np.float32, np.uint8, np.int16), (0, 1), (0, 1, 2, 3), (True, False), (1, 0)
            ):
                if mips != 1 and not norm:
                    continue
                shape = (5, 7) if ndim == 2 else (4, 5, 6)
                if dt == np.float32:
                    data = rng.standard_normal((*shape, 4)).astype(dt)
                else:
                    info = np.iinfo(dt)
                    data = rng.integers(info.min, info.max, (*shape, 4), endpoint=True).astype(dt)
                lo, hi = (-0.4, 1.4) if norm else (-3.0, 12.0)
                coords = (rng.random((32, ndim)) * (hi - lo) + lo).astype(np.float32)
                lod = 1.3 if mips != 1 else -1.0
                outs = {}
                for dev in ("cpu", "metal:0"):
                    cls = wp.Texture2D if ndim == 2 else wp.Texture3D
                    tex = cls(data, filter_mode=filt, address_mode=addr, normalized_coords=norm,
                              num_mip_levels=mips, device=dev)
                    c = wp.array(coords, dtype=wp.vec2 if ndim == 2 else wp.vec3, device=dev)
                    o1 = wp.zeros(32, dtype=float, device=dev)
                    o4 = wp.zeros(32, dtype=wp.vec4, device=dev)
                    wp.launch(s2 if ndim == 2 else s3, dim=32, inputs=[tex, c, lod, o1, o4], device=dev)
                    outs[dev] = (o1.numpy(), o4.numpy())
                    if mips == 1:
                        back = np.zeros_like(data)
                        tex.copy_to(back)
                        np.testing.assert_array_equal(back, data)
                for a, b in zip(outs["cpu"], outs["metal:0"]):
                    np.testing.assert_allclose(b, a, rtol=1e-3, atol=2e-3)
            """,
            timeout=600,
        )

    def test_texture_struct_fields(self):
        _run(
            self,
            """
            @wp.struct
            class SDF:
                coarse: wp.Texture3D
                fine: wp.Texture3D
                scale: float

            @wp.kernel
            def k(s: SDF, sa: wp.array[SDF], p: wp.array[wp.vec3], o: wp.array[float], w: wp.array[int]):
                i = wp.tid()
                e = sa[i % 2]
                o[i] = (wp.texture_sample(s.coarse, p[i], dtype=float) + wp.texture_sample(e.fine, p[i], dtype=float)) * s.scale
                w[i] = s.coarse.width * 100 + e.fine.depth

            rng = np.random.default_rng(1)
            a = rng.standard_normal((3, 4, 5)).astype(np.float32)
            b = rng.standard_normal((2, 3, 6)).astype(np.float32)
            keep = []

            def make(d):
                s = SDF()
                s.coarse = wp.Texture3D(a, device=d)
                s.fine = wp.Texture3D(b, device=d)
                s.scale = 2.5
                s2 = SDF()
                s2.coarse, s2.fine, s2.scale = s.fine, s.coarse, -1.0
                keep.append((s, s2))
                p = wp.array(np.linspace(0, 1, 30).reshape(10, 3).astype(np.float32), dtype=wp.vec3, device=d)
                return [s, wp.array([s, s2], dtype=SDF, device=d), p,
                        wp.zeros(10, dtype=float, device=d), wp.zeros(10, dtype=int, device=d)]

            compare(k, 10, make, [3, 4], tiled=False, atol=1e-4)
            """,
        )


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalMeshWinding(unittest.TestCase):
    def test_sign_winding_number_matches_cpu(self):
        _run(
            self,
            """
            @wp.kernel
            def k(mesh: wp.uint64, pts: wp.array[wp.vec3], s1: wp.array[float], s2: wp.array[float]):
                i = wp.tid()
                s1[i] = wp.mesh_query_point_sign_winding_number(mesh, pts[i], 10.0, 2.0, 0.5).sign
                s2[i] = wp.mesh_query_point_sign_winding_number(mesh, pts[i], 10.0, 2.0, 0.25).sign

            nu, nv = 24, 16
            verts = [(0, 0, 1)]
            for j in range(1, nv):
                th = np.pi * j / nv
                for i in range(nu):
                    ph = 2 * np.pi * i / nu
                    verts.append((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))
            verts.append((0, 0, -1))
            tris = [(0, 1 + i, 1 + (i + 1) % nu) for i in range(nu)]
            for j in range(nv - 2):
                for i in range(nu):
                    a = 1 + j * nu + i
                    b = 1 + j * nu + (i + 1) % nu
                    tris += [(a, a + nu, b), (b, a + nu, b + nu)]
            last, base = len(verts) - 1, 1 + (nv - 2) * nu
            tris += [(last, base + (i + 1) % nu, base + i) for i in range(nu)]
            v = np.array(verts, np.float32)
            t = np.array(tris, np.int32).reshape(-1)
            # Drop a cap so the mesh is open: winding numbers then differ from parity.
            t = t[nu * 3:]
            p = (np.random.default_rng(0).random((400, 3)) * 3 - 1.5).astype(np.float32)
            res = {}
            for dev in ("cpu", "metal:0"):
                m = wp.Mesh(wp.array(v, dtype=wp.vec3, device=dev), wp.array(t, dtype=int, device=dev),
                            support_winding_number=True)
                pa = wp.array(p, dtype=wp.vec3, device=dev)
                outs = []
                for scale in (1.0, 1.3):
                    m.points.assign(wp.array(v * scale, dtype=wp.vec3, device=dev))
                    m.refit()
                    s1 = wp.zeros(400, dtype=float, device=dev)
                    s2 = wp.zeros(400, dtype=float, device=dev)
                    wp.launch(k, dim=400, inputs=[m.id, pa, s1, s2], device=dev)
                    outs += [s1.numpy(), s2.numpy()]
                res[dev] = outs
            for a, b in zip(res["cpu"], res["metal:0"]):
                np.testing.assert_array_equal(b, a)
            """,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
