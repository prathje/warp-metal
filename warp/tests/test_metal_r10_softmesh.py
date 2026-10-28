# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal ports needed by Newton's full-surface mesh soft contacts.

BVH sphere / capsule queries, ``mesh_get_bvh``, the native ``wp::`` API for
``@wp.func_native`` snippets, and ``closest_point_edge_edge`` (+ adjoint).

Each test runs the same kernels on ``cpu`` and ``metal:0`` in a subprocess and
compares the results exactly, or pins that Metal fails loudly.
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_native

_COMMON = textwrap.dedent(
    """
    import numpy as np

    def run_both(fn):
        return {dev: fn(dev) for dev in ("cpu", "metal:0")}

    def random_boxes(rng, n, spread=4.0, size=0.6):
        lo = rng.uniform(-spread, spread, (n, 3)).astype(np.float32)
        hi = (lo + rng.uniform(0.0, size, (n, 3))).astype(np.float32)
        return lo, hi
    """
)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR10SoftMesh(unittest.TestCase):
    def test_bvh_query_sphere_matches_cpu(self):
        # Same item set per query as CPU, for the global root, an explicit
        # group root, zero and negative radii, and tangential contact.
        snippet = _COMMON + textwrap.dedent(
            """
            MAXH = 64

            @wp.kernel(module="unique")
            def k_sphere(bvh: wp.uint64, centers: wp.array[wp.vec3], radii: wp.array[float], root: int,
                         hits: wp.array2d[int], counts: wp.array[int]):
                t = wp.tid()
                q = wp.bvh_query_sphere(bvh, centers[t], radii[t], root)
                item = int(0)
                n = int(0)
                while wp.bvh_query_next(q, item):
                    if n < MAXH:
                        hits[t, n] = item
                    n += 1
                counts[t] = n

            @wp.kernel(module="unique")
            def k_root(bvh: wp.uint64, group: int, out: wp.array[int]):
                out[0] = wp.bvh_get_group_root(bvh, group)

            rng = np.random.default_rng(7)
            lo, hi = random_boxes(rng, 500)
            groups = (np.arange(500) % 3).astype(np.int32)
            n = 4000
            centers = rng.uniform(-5, 5, (n, 3)).astype(np.float32)
            radii = rng.uniform(-0.2, 1.5, n).astype(np.float32)
            radii[:50] = 0.0
            # tangential: center at distance exactly 0.5 from box 0 along x
            centers[50] = np.array([hi[0, 0] + 0.5, lo[0, 1], lo[0, 2]], np.float32)
            radii[50] = 0.5

            def go(dev):
                bvh = wp.Bvh(wp.array(lo, dtype=wp.vec3, device=dev), wp.array(hi, dtype=wp.vec3, device=dev),
                             groups=wp.array(groups, dtype=int, device=dev))
                c = wp.array(centers, dtype=wp.vec3, device=dev)
                r = wp.array(radii, dtype=float, device=dev)
                out = []
                root_arr = wp.zeros(1, dtype=int, device=dev)
                wp.launch(k_root, 1, [bvh.id, 1, root_arr], device=dev)
                for root in (-1, int(root_arr.numpy()[0])):
                    hits = wp.full((n, MAXH), -1, dtype=int, device=dev)
                    counts = wp.zeros(n, dtype=int, device=dev)
                    wp.launch(k_sphere, n, [bvh.id, c, r, root, hits, counts], device=dev)
                    h = hits.numpy()
                    out.append((counts.numpy(), [sorted(h[i][h[i] >= 0].tolist()) for i in range(n)]))
                return out

            res = run_both(go)
            for (cc, ch), (mc, mh) in zip(res["cpu"], res["metal:0"]):
                assert np.array_equal(cc, mc), np.nonzero(cc != mc)
                assert cc.max() <= MAXH
                assert ch == mh
            assert res["cpu"][0][0][50] >= 1
            # the group root restricts the result
            assert res["cpu"][1][0].sum() < res["cpu"][0][0].sum()
            assert res["cpu"][0][0][:50].sum() > 0  # zero radius still finds containing boxes
            """
        )
        _run_with_metal_native(self, snippet, timeout=300)

    def test_bvh_query_capsule_matches_cpu(self):
        # Closed max_dist endpoint, axis-aligned directions (zero components),
        # negative radius clamp, and the default (infinite) max_dist.
        snippet = _COMMON + textwrap.dedent(
            """
            @wp.kernel(module="unique")
            def k_capsule(bvh: wp.uint64, starts: wp.array[wp.vec3], dirs: wp.array[wp.vec3],
                          radii: wp.array[float], max_dist: wp.array[float], counts: wp.array[int],
                          sums: wp.array[int], counts_inf: wp.array[int]):
                t = wp.tid()
                q = wp.bvh_query_capsule(bvh, starts[t], dirs[t], radii[t])
                item = int(0)
                n = int(0)
                s = int(0)
                while wp.bvh_query_next(q, item, max_dist[t]):
                    n += 1
                    s += item * item
                counts[t] = n
                sums[t] = s
                q2 = wp.bvh_query_capsule(bvh, starts[t], dirs[t], radii[t], -1)
                n = int(0)
                while wp.bvh_query_next(q2, item):
                    n += 1
                counts_inf[t] = n

            rng = np.random.default_rng(11)
            lo, hi = random_boxes(rng, 400)
            n = 4000
            starts = rng.uniform(-5, 5, (n, 3)).astype(np.float32)
            dirs = rng.uniform(-3, 3, (n, 3)).astype(np.float32)
            axis = rng.integers(0, 3, n)
            sel = rng.random(n) < 0.3
            for k in range(3):
                dirs[sel & (axis != k), k] = 0.0  # axis-aligned segments
            radii = rng.uniform(-0.1, 0.8, n).astype(np.float32)
            max_dist = rng.choice(np.array([0.25, 0.5, 1.0, 2.0], np.float32), n)
            # closed endpoint: segment ends exactly on an inflated box face
            starts[0] = np.array([lo[0, 0] - 1.0, 0.5 * (lo[0, 1] + hi[0, 1]), 0.5 * (lo[0, 2] + hi[0, 2])], np.float32)
            dirs[0] = np.array([1.0, 0.0, 0.0], np.float32)
            radii[0] = 0.0
            max_dist[0] = 1.0

            def go(dev):
                bvh = wp.Bvh(wp.array(lo, dtype=wp.vec3, device=dev), wp.array(hi, dtype=wp.vec3, device=dev))
                arrs = [wp.array(starts, dtype=wp.vec3, device=dev), wp.array(dirs, dtype=wp.vec3, device=dev),
                        wp.array(radii, dtype=float, device=dev), wp.array(max_dist, dtype=float, device=dev)]
                outs = [wp.zeros(n, dtype=int, device=dev) for _ in range(3)]
                wp.launch(k_capsule, n, [bvh.id, *arrs, *outs], device=dev)
                return [o.numpy() for o in outs]

            res = run_both(go)
            for c, m in zip(res["cpu"], res["metal:0"]):
                assert np.array_equal(c, m), np.nonzero(c != m)
            assert res["cpu"][0][0] >= 1
            assert (res["cpu"][2] >= res["cpu"][0]).all() and res["cpu"][2].sum() > res["cpu"][0].sum()
            """
        )
        _run_with_metal_native(self, snippet, timeout=300)

    def test_mesh_get_bvh_queries_match_cpu(self):
        # ``mesh_get_bvh`` ids drive bvh_query_{aabb,sphere,ray} over the
        # mesh's triangle BVH; results are face indices, as on CPU.
        snippet = _COMMON + textwrap.dedent(
            """
            @wp.kernel(module="unique")
            def k_mesh(mesh: wp.uint64, pts: wp.array[wp.vec3], r: float, out: wp.array2d[int]):
                t = wp.tid()
                bvh = wp.mesh_get_bvh(mesh)
                q = wp.bvh_query_sphere(bvh, pts[t], r)
                face = int(0)
                n = int(0)
                s = int(0)
                while wp.bvh_query_next(q, face):
                    n += 1
                    s += face
                out[t, 0] = n
                out[t, 1] = s
                qa = wp.bvh_query_aabb(bvh, pts[t] - wp.vec3(r), pts[t] + wp.vec3(r))
                n = int(0)
                while wp.bvh_query_next(qa, face):
                    n += 1
                out[t, 2] = n
                qr = wp.bvh_query_ray(bvh, pts[t], wp.vec3(0.3, -0.2, 1.0))
                n = int(0)
                while wp.bvh_query_next(qr, face):
                    n += 1
                out[t, 3] = n

            rng = np.random.default_rng(5)
            # a bumpy 24x24 height-field mesh
            g = 24
            xs, ys = np.meshgrid(np.linspace(-1, 1, g), np.linspace(-1, 1, g), indexing="ij")
            zs = 0.1 * np.sin(4 * xs) * np.cos(3 * ys)
            verts = np.stack([xs, ys, zs], -1).reshape(-1, 3).astype(np.float32)
            tris = []
            for i in range(g - 1):
                for j in range(g - 1):
                    a, b, c, d = i * g + j, (i + 1) * g + j, (i + 1) * g + j + 1, i * g + j + 1
                    tris += [a, b, c, a, c, d]
            tris = np.array(tris, np.int32)
            n = 3000
            pts = rng.uniform(-1.2, 1.2, (n, 3)).astype(np.float32)

            def go(dev):
                mesh = wp.Mesh(wp.array(verts, dtype=wp.vec3, device=dev), wp.array(tris, dtype=int, device=dev))
                out = wp.zeros((n, 4), dtype=int, device=dev)
                wp.launch(k_mesh, n, [mesh.id, wp.array(pts, dtype=wp.vec3, device=dev), 0.15, out], device=dev)
                return out.numpy()

            res = run_both(go)
            assert np.array_equal(res["cpu"], res["metal:0"]), np.nonzero(res["cpu"] != res["metal:0"])
            assert res["cpu"][:, 0].sum() > 0 and (res["cpu"][:, 2] >= res["cpu"][:, 0]).all()
            """
        )
        _run_with_metal_native(self, snippet, timeout=300)

    def test_native_snippet_mesh_bvh_api_matches_cpu(self):
        # A snippet walking the mesh BVH through wp::mesh_get / bvh_load_node
        # (Newton's partitioned mesh query) and the struct-returning sign
        # queries reproduce CPU exactly.
        snippet = _COMMON + textwrap.dedent(
            """
            @wp.func_native(\"\"\"
            const wp::Mesh mesh = wp::mesh_get(mesh_id);
            int root = *mesh.bvh.root;
            if (root < 0) return -2;
            for (int level = 0; level < depth; ++level) {
                const auto lower = wp::bvh_load_node(mesh.bvh.node_lowers, root);
                if (lower.b) return (lane & ((1 << (depth - level)) - 1)) == 0 ? root : -2;
                const auto upper = wp::bvh_load_node(mesh.bvh.node_uppers, root);
                root = ((lane >> (depth - 1 - level)) & 1) ? upper.i : lower.i;
            }
            return root;
            \"\"\")
            def partition(mesh_id: wp.uint64, lane: int, depth: int) -> int: ...

            @wp.func_native(\"\"\"
            const auto query = parity ? wp::mesh_query_point_sign_parity(mesh_id, point, radius)
                                      : wp::mesh_query_point_sign_normal(mesh_id, point, radius);
            return query.result ? (query.sign < 0.0f ? -1 : 1) * (query.face + 1) : 0;
            \"\"\")
            def signed_face(mesh_id: wp.uint64, point: wp.vec3, radius: float, parity: bool) -> int: ...

            @wp.func_native(\"\"\"
            const wp::Mesh mesh = wp::mesh_get(mesh_id);
            return mesh.num_tris * 1000 + mesh.bvh.num_items;
            \"\"\")
            def mesh_counts(mesh_id: wp.uint64) -> int: ...

            @wp.kernel(module="unique")
            def k_part(mesh: wp.uint64, depth: int, out: wp.array[int], counts: wp.array[int]):
                lane = wp.tid()
                root = partition(mesh, lane, depth)
                out[lane] = root
                n = int(0)
                if root != -2:
                    q = wp.bvh_query_sphere(wp.mesh_get_bvh(mesh), wp.vec3(0.1, 0.2, 0.0), 0.5, root)
                    face = int(0)
                    while wp.bvh_query_next(q, face):
                        n += 1
                counts[lane] = n

            @wp.kernel(module="unique")
            def k_sign(mesh: wp.uint64, pts: wp.array[wp.vec3], out: wp.array2d[int]):
                t = wp.tid()
                out[t, 0] = signed_face(mesh, pts[t], 10.0, True)
                out[t, 1] = signed_face(mesh, pts[t], 10.0, False)
                out[t, 2] = signed_face(mesh, pts[t], 0.05, False)
                out[t, 3] = mesh_counts(mesh)

            # closed box-like mesh: a subdivided cube
            def cube(m):
                verts, tris = [], []
                lin = np.linspace(-0.5, 0.5, m + 1)
                for axis in range(3):
                    for side in (-0.5, 0.5):
                        base = len(verts)
                        for a in lin:
                            for b in lin:
                                p = [0.0, 0.0, 0.0]
                                p[axis] = side
                                p[(axis + 1) % 3] = a
                                p[(axis + 2) % 3] = b
                                verts.append(p)
                        for i in range(m):
                            for j in range(m):
                                v0 = base + i * (m + 1) + j
                                v1, v2, v3 = v0 + (m + 1), v0 + (m + 1) + 1, v0 + 1
                                quad = [v0, v1, v2, v0, v2, v3] if side > 0 else [v0, v2, v1, v0, v3, v2]
                                tris += quad
                return np.array(verts, np.float32), np.array(tris, np.int32)

            verts, tris = cube(6)
            rng = np.random.default_rng(3)
            n = 2000
            pts = rng.uniform(-0.8, 0.8, (n, 3)).astype(np.float32)

            def go(dev):
                mesh = wp.Mesh(wp.array(verts, dtype=wp.vec3, device=dev), wp.array(tris, dtype=int, device=dev))
                parts = []
                for depth in (0, 1, 3, 5):
                    out = wp.zeros(1 << depth, dtype=int, device=dev)
                    counts = wp.zeros(1 << depth, dtype=int, device=dev)
                    wp.launch(k_part, 1 << depth, [mesh.id, depth, out, counts], device=dev)
                    parts.append((out.numpy(), counts.numpy()))
                out = wp.zeros((n, 4), dtype=int, device=dev)
                wp.launch(k_sign, n, [mesh.id, wp.array(pts, dtype=wp.vec3, device=dev), out], device=dev)
                return parts, out.numpy()

            res = run_both(go)
            for (co, cc), (mo, mc) in zip(res["cpu"][0], res["metal:0"][0]):
                assert np.array_equal(co, mo), (co, mo)
                assert np.array_equal(cc, mc), (cc, mc)
            # every lane partition together covers the full query exactly once
            full = res["cpu"][0][0][1][0]
            for _out, counts in res["cpu"][0]:
                assert counts.sum() == full, (counts.sum(), full)
            assert np.array_equal(res["cpu"][1], res["metal:0"][1]), np.nonzero(res["cpu"][1] != res["metal:0"][1])
            sign = res["cpu"][1]
            assert (sign[:, 0] < 0).any() and (sign[:, 0] > 0).any() and (sign[:, 2] == 0).any()
            assert sign[0, 3] == len(tris) // 3 * 1001
            """
        )
        _run_with_metal_native(self, snippet, timeout=300)

    def test_native_snippet_ray_tri_woop_matches_cpu_bitwise(self):
        # wp::intersect_ray_tri_woop in a snippet: watertight edge / vertex
        # hits must give the same exact-zero weights (binary64 refinement)
        # and the same unfused float results as CPU.
        snippet = _COMMON + textwrap.dedent(
            """
            @wp.func_native(\"\"\"
            const wp::vec3 direction = q - p;
            float t, u, v, sign;
            wp::vec3 n;
            if (wp::dot(direction, direction) == 0.0f ||
                !wp::intersect_ray_tri_woop(p, direction, a, b, c, t, u, v, sign, &n) || t > 1.0f)
                return wp::vec4(-1.0f, 0.0f, 0.0f, 0.0f);
            float cyclic_t, cyclic_u, w, cyclic_sign;
            if (!wp::intersect_ray_tri_woop(p, direction, b, c, a, cyclic_t, cyclic_u, w, cyclic_sign, nullptr))
                return wp::vec4(-1.0f, 0.0f, 0.0f, 0.0f);
            return wp::vec4(t, u, v, w) + wp::vec4(0.0f, 0.0f, 0.0f, 0.0f * (sign + wp::dot(n, n)));
            \"\"\")
            def seg_hit(p: wp.vec3, q: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec4: ...

            @wp.kernel(module="unique")
            def k_hit(p: wp.array[wp.vec3], q: wp.array[wp.vec3], a: wp.array[wp.vec3], b: wp.array[wp.vec3],
                      c: wp.array[wp.vec3], out: wp.array[wp.vec4]):
                t = wp.tid()
                out[t] = seg_hit(p[t], q[t], a[t], b[t], c[t])

            rng = np.random.default_rng(13)
            n = 200000
            cases = []
            # random float segments aimed at triangle vertices / edge midpoints
            for target in ("vertex", "edge"):
                P, A, B, C = (rng.uniform(-1, 1, (n, 3)).astype(np.float32) for _ in range(4))
                T = A if target == "vertex" else (0.5 * (A + B)).astype(np.float32)
                cases.append((P, (2 * T - P).astype(np.float32), A, B, C))
            # coarse grid geometry: many exact edge / vertex crossings
            g = lambda: (rng.integers(-3, 4, (n, 3)) * 0.3).astype(np.float32)
            cases.append((g(), g(), g(), g(), g()))

            for P, Q, A, B, C in cases:
                def go(dev):
                    arrs = [wp.array(x, dtype=wp.vec3, device=dev) for x in (P, Q, A, B, C)]
                    out = wp.zeros(n, dtype=wp.vec4, device=dev)
                    wp.launch(k_hit, n, [*arrs, out], device=dev)
                    return out.numpy()
                res = run_both(go)
                c, m = res["cpu"], res["metal:0"]
                bad = np.any(c.view(np.int32) != m.view(np.int32), axis=1)
                assert not bad.any(), (bad.sum(), c[bad][:3], m[bad][:3])
                hit = c[:, 0] >= 0
                assert hit.sum() > n // 10 and (hit & np.any(c[:, 1:] == 0.0, axis=1)).sum() > 100
            """
        )
        _run_with_metal_native(self, snippet, timeout=300)

    def test_closest_point_edge_edge_matches_cpu(self):
        # Forward bitwise and adjoint of wp.closest_point_edge_edge, incl. the
        # degenerate (point) edges, near-parallel and clamped branches. The
        # Metal port predated Warp's stabilized version (unclamped t for a
        # point edge 1, uncompensated denominators) and had no adjoint.
        snippet = _COMMON + textwrap.dedent(
            """
            @wp.kernel(module="unique")
            def k_ee(p1: wp.array[wp.vec3], q1: wp.array[wp.vec3], p2: wp.array[wp.vec3], q2: wp.array[wp.vec3],
                     eps: float, out: wp.array[wp.vec3], loss: wp.array[float]):
                t = wp.tid()
                r = wp.closest_point_edge_edge(p1[t], q1[t], p2[t], q2[t], eps)
                out[t] = r
                wp.atomic_add(loss, 0, 0.3 * r[0] - 0.7 * r[1] + 1.3 * r[2])

            rng = np.random.default_rng(17)
            n = 60000
            P1, Q1, P2, Q2 = (rng.uniform(-1, 1, (n, 3)).astype(np.float32) for _ in range(4))
            k = n // 6
            Q1[:k] = P1[:k] + rng.normal(scale=1e-4, size=(k, 3)).astype(np.float32)          # point edge 1
            Q2[k:2 * k] = P2[k:2 * k]                                                          # point edge 2
            Q1[2 * k:3 * k] = P1[2 * k:3 * k]
            Q2[2 * k:3 * k] = P2[2 * k:3 * k]                                                  # both points
            d = (Q1 - P1)[3 * k:4 * k]
            Q2[3 * k:4 * k] = P2[3 * k:4 * k] + d * rng.uniform(0.5, 2, (k, 1)).astype(np.float32)  # parallel
            Q2[4 * k:5 * k] = (P2[4 * k:5 * k] + (d + rng.normal(scale=1e-3, size=(k, 3))) ).astype(np.float32)

            def go(dev):
                arrs = [wp.array(x, dtype=wp.vec3, device=dev, requires_grad=True) for x in (P1, Q1, P2, Q2)]
                out = wp.zeros(n, dtype=wp.vec3, device=dev, requires_grad=True)
                loss = wp.zeros(1, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(k_ee, n, [*arrs, 1.0e-6, out, loss], device=dev)
                tape.backward(loss)
                return out.numpy(), [a.grad.numpy() for a in arrs]

            res = run_both(go)
            c, m = res["cpu"][0], res["metal:0"][0]
            bad = np.any(c.view(np.int32) != m.view(np.int32), axis=1)
            assert not bad.any(), (bad.sum(), np.nonzero(bad)[0][:5], c[bad][:3], m[bad][:3])
            assert (c[:k, 1] >= 0).all() and (c[:k, 1] <= 1).all()
            for gc, gm in zip(res["cpu"][1], res["metal:0"][1]):
                gbad = np.any(gc.view(np.int32) != gm.view(np.int32), axis=1)
                assert not gbad.any(), (gbad.sum(), gc[gbad][:3], gm[gbad][:3])
                assert np.abs(gc).sum() > 0
            """
        )
        _run_with_metal_native(self, snippet, timeout=300)

    def test_native_snippet_cross_and_woop_normal_match_cpu_bitwise(self):
        # wp::cross and the normal output of wp::intersect_ray_tri_woop are
        # computed unfused, as native vec.h cross() on CPU (metal::cross fuses).
        snippet = _COMMON + textwrap.dedent(
            """
            @wp.func_native(\"\"\"
            float t, u, v, sign;
            wp::vec3 n;
            if (!wp::intersect_ray_tri_woop(p, d, a, b, c, t, u, v, sign, &n))
                return wp::vec3(0.0f, 0.0f, 0.0f);
            return n;
            \"\"\")
            def woop_normal(p: wp.vec3, d: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3: ...

            @wp.func_native(\"\"\"
            return wp::cross(a, b);
            \"\"\")
            def native_cross(a: wp.vec3, b: wp.vec3) -> wp.vec3: ...

            @wp.kernel(module="unique")
            def k(p: wp.array[wp.vec3], d: wp.array[wp.vec3], a: wp.array[wp.vec3], b: wp.array[wp.vec3],
                  c: wp.array[wp.vec3], out: wp.array2d[wp.vec3]):
                t = wp.tid()
                out[t, 0] = woop_normal(p[t], d[t], a[t], b[t], c[t])
                out[t, 1] = native_cross(a[t], b[t])

            rng = np.random.default_rng(29)
            n = 100000
            P, A, B, C = (rng.uniform(-1, 1, (n, 3)).astype(np.float32) for _ in range(4))
            W = rng.dirichlet([1, 1, 1], n).astype(np.float32)
            D = (W[:, :1] * A + W[:, 1:2] * B + W[:, 2:] * C - P).astype(np.float32)

            def go(dev):
                arrs = [wp.array(x, dtype=wp.vec3, device=dev) for x in (P, D, A, B, C)]
                out = wp.zeros((n, 2), dtype=wp.vec3, device=dev)
                wp.launch(k, n, [*arrs, out], device=dev)
                return out.numpy()

            res = run_both(go)
            c, m = res["cpu"], res["metal:0"]
            assert np.any(c[:, 0] != 0.0)
            bad = np.any(c.view(np.int32) != m.view(np.int32), axis=2)
            assert not bad.any(), (bad.sum(axis=0), c[bad.any(1)][:3], m[bad.any(1)][:3])
            """
        )
        _run_with_metal_native(self, snippet, timeout=300)

    def test_mesh_parity_sign_and_ray_on_surface_match_cpu(self):
        # Points exactly on a watertight mesh (faces, edges, vertices) and ray
        # casts through its edges / vertices: the ray / triangle test must not
        # FP-contract and must refine exact-zero edge functions in binary64 like
        # native intersect_ray_tri_woop, or the parity sign of on-surface points
        # (Newton's full-surface soft contacts on watertight meshes) flips.
        snippet = _COMMON + textwrap.dedent(
            """
            @wp.kernel(module="unique")
            def k_sign(mesh: wp.uint64, pts: wp.array[wp.vec3], out: wp.array[wp.vec4]):
                t = wp.tid()
                q = wp.mesh_query_point_sign_parity(mesh, pts[t], 10.0)
                out[t] = wp.vec4(wp.where(q.result, 1.0, 0.0), q.sign, float(q.face), q.u + 2.0 * q.v)

            @wp.kernel(module="unique")
            def k_ray(mesh: wp.uint64, starts: wp.array[wp.vec3], dirs: wp.array[wp.vec3],
                      out: wp.array2d[wp.vec4]):
                t = wp.tid()
                q = wp.mesh_query_ray(mesh, starts[t], dirs[t], 100.0)
                out[t, 0] = wp.vec4(wp.where(q.result, 1.0, 0.0), q.t, q.u, q.v)
                out[t, 1] = wp.vec4(q.sign, q.normal[0], q.normal[1], q.normal[2])

            def cube(m):
                verts, tris = [], []
                lin = np.linspace(-0.5, 0.5, m + 1)
                for axis in range(3):
                    for side in (-0.5, 0.5):
                        base = len(verts)
                        for a in lin:
                            for b in lin:
                                p = [0.0, 0.0, 0.0]
                                p[axis] = side
                                p[(axis + 1) % 3] = a
                                p[(axis + 2) % 3] = b
                                verts.append(p)
                        for i in range(m):
                            for j in range(m):
                                v0 = base + i * (m + 1) + j
                                v1, v2, v3 = v0 + (m + 1), v0 + (m + 1) + 1, v0 + 1
                                tris += [v0, v1, v2, v0, v2, v3] if side > 0 else [v0, v2, v1, v0, v3, v2]
                return np.array(verts, np.float32), np.array(tris, np.int32)

            verts, tris = cube(4)
            rng = np.random.default_rng(31)
            grid = np.linspace(-0.75, 0.75, 13).astype(np.float32)
            pts = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
            face = rng.uniform(-0.45, 0.45, (4000, 3)).astype(np.float32)
            face[:, 2] = 0.5
            pts = np.concatenate([pts, face]).astype(np.float32)
            g = lambda n: (rng.integers(-4, 5, (n, 3)) * 0.25).astype(np.float32)
            # starts outside the cube; targets on its face / edge / vertex lattice
            starts = g(20000) * np.float32(2.0)
            starts[:, 0] = np.where(np.abs(starts[:, 0]) < 1.0, np.float32(2.0), starts[:, 0])
            dirs = g(20000) - starts
            dirs[np.all(dirs == 0, axis=1)] = 1.0

            def go(dev):
                mesh = wp.Mesh(wp.array(verts, dtype=wp.vec3, device=dev), wp.array(tris, dtype=int, device=dev))
                out = wp.zeros(len(pts), dtype=wp.vec4, device=dev)
                wp.launch(k_sign, len(pts), [mesh.id, wp.array(pts, dtype=wp.vec3, device=dev), out], device=dev)
                ray = wp.zeros((len(starts), 2), dtype=wp.vec4, device=dev)
                wp.launch(k_ray, len(starts), [mesh.id, wp.array(starts, dtype=wp.vec3, device=dev),
                                               wp.array(dirs, dtype=wp.vec3, device=dev), ray], device=dev)
                return out.numpy(), ray.numpy()

            res = run_both(go)
            c, m = res["cpu"][0], res["metal:0"][0]
            flips = np.sign(c[:, 1]) != np.sign(m[:, 1])
            assert not flips.any(), (flips.sum(), pts[flips][:4])
            assert (c[:, 1] < 0).any() and (c[:, 1] > 0).any()
            cr, mr = res["cpu"][1], res["metal:0"][1]
            assert cr[:, 0, 0].sum() > 1000
            # Hit / miss everywhere; every output where the hit is unique (interior of a face).
            # Hits on shared edges / vertices pick their face (and its ulp-different t) by
            # traversal order, which differs between the native (near-child-first) and the
            # Metal BVH walk.
            assert np.array_equal(cr[:, 0, 0], mr[:, 0, 0])
            np.testing.assert_allclose(mr[:, 0, 1], cr[:, 0, 1], rtol=1e-6, atol=0.0)
            u, v = cr[:, 0, 2], cr[:, 0, 3]
            unique = (cr[:, 0, 0] > 0) & (u > 0) & (v > 0) & (1.0 - u - v > 1e-6)
            assert unique.sum() > 500
            bad = unique & np.any(cr.view(np.int32) != mr.view(np.int32), axis=(1, 2))
            assert not bad.any(), (bad.sum(), starts[bad][:3], dirs[bad][:3], cr[bad][:2], mr[bad][:2])
            """
        )
        _run_with_metal_native(self, snippet, timeout=300)

    def test_native_snippet_unknown_wp_api_fails_loudly(self):
        # Native members without a Metal counterpart (here the mesh's points
        # array) must fail to compile, not read something else.
        snippet = textwrap.dedent(
            """
            import numpy as np

            @wp.func_native(\"\"\"
            const wp::Mesh mesh = wp::mesh_get(mesh_id);
            return mesh.points[0][0];
            \"\"\")
            def first_x(mesh_id: wp.uint64) -> float: ...

            @wp.kernel(module="unique")
            def k(mesh: wp.uint64, out: wp.array[float]):
                out[0] = first_x(mesh)

            dev = "metal:0"
            mesh = wp.Mesh(wp.array(np.eye(3, dtype=np.float32), dtype=wp.vec3, device=dev),
                           wp.array(np.array([0, 1, 2], np.int32), dtype=int, device=dev))
            out = wp.zeros(1, dtype=float, device=dev)
            try:
                wp.launch(k, 1, [mesh.id, out], device=dev)
                wp.synchronize_device(dev)
            except Exception as e:
                assert "points" in str(e), e
            else:
                raise AssertionError("unsupported wp::Mesh member compiled on Metal")
            """
        )
        _run_with_metal_native(self, snippet, timeout=300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
