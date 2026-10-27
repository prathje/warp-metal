# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metal parity with the CPU for behaviour Warp 1.19 changed.

Each test runs the same kernel on ``cpu`` and ``metal:0`` in a subprocess and
compares the results, or pins that Metal rejects a construct loudly
(``MetalCodegenError``) instead of computing something different from the CPU.
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_native


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalUpstream119(unittest.TestCase):
    def test_tile_matmul_beta_zero_does_not_read_out(self):
        # GH-2079: with beta == 0 the product must not read ``out``, so NaN or
        # Inf already in ``out`` cannot reach the result through 0 * out.
        snippet = textwrap.dedent(
            """
            import numpy as np

            MM = 8

            @wp.kernel(module="unique")
            def k_lit(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float), c: wp.array2d(dtype=float)):
                ta = wp.tile_load(a, shape=(MM, MM))
                tb = wp.tile_load(b, shape=(MM, MM))
                tc = wp.tile_load(c, shape=(MM, MM))
                wp.tile_matmul(ta, tb, tc, alpha=2.0, beta=0.0)
                wp.tile_store(c, tc)

            @wp.kernel(module="unique")
            def k_dyn(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float), c: wp.array2d(dtype=float),
                      beta: float):
                ta = wp.tile_load(a, shape=(MM, MM))
                tb = wp.tile_load(b, shape=(MM, MM))
                tc = wp.tile_load(c, shape=(MM, MM))
                wp.tile_matmul(ta, tb, tc, alpha=1.0, beta=beta)
                wp.tile_store(c, tc)

            rng = np.random.default_rng(3)
            a_np = rng.standard_normal((MM, MM)).astype(np.float32)
            b_np = rng.standard_normal((MM, MM)).astype(np.float32)
            c_np = np.ones((MM, MM), np.float32)
            c_np[0, 0] = np.nan
            c_np[3, 4] = np.inf
            c_np[5, 1] = -np.inf
            cases = [
                (k_lit, [], 2.0 * (a_np @ b_np)),
                (k_dyn, [0.0], a_np @ b_np),
                (k_dyn, [0.5], a_np @ b_np + 0.5 * c_np),
            ]
            for k, extra, expected in cases:
                outs = {}
                for dev in ("cpu", "metal:0"):
                    a = wp.array(a_np, device=dev)
                    b = wp.array(b_np, device=dev)
                    c = wp.array(c_np, device=dev)
                    for bd in (1, 64):
                        c.assign(c_np)
                        wp.launch_tiled(k, dim=1, inputs=[a, b, c, *extra], block_dim=bd, device=dev)
                        outs[(dev, bd)] = c.numpy().copy()
                for key, got in outs.items():
                    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5, err_msg=f"{k.key} {extra} {key}")
                for bd in (1, 64):
                    np.testing.assert_array_equal(np.isfinite(outs[("cpu", bd)]), np.isfinite(outs[("metal:0", bd)]))
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_tile_matmul_tf32_warns_and_falls_back(self):
        # GH-937: precision="tf32" is honored only by cuBLASDx. Like the CPU,
        # Metal warns and computes at the operand precision.
        snippet = textwrap.dedent(
            """
            import os

            import numpy as np
            import warp._src.builtins as wpb

            os.environ["WARP_METAL_DISABLE_ARTIFACT_CACHE"] = "1"
            MM = 8

            @wp.kernel(module="unique")
            def k(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float), c: wp.array2d(dtype=float)):
                ta = wp.tile_load(a, shape=(MM, MM))
                tb = wp.tile_load(b, shape=(MM, MM))
                wp.tile_store(c, wp.tile_matmul(ta, tb, alpha=1.5, precision="tf32"))

            rng = np.random.default_rng(4)
            a_np = rng.standard_normal((MM, MM)).astype(np.float32)
            b_np = rng.standard_normal((MM, MM)).astype(np.float32)
            a = wp.array(a_np, device="metal:0")
            b = wp.array(b_np, device="metal:0")
            c = wp.zeros((MM, MM), dtype=float, device="metal:0")
            caught = []
            wpb.log_warning = lambda msg, *args, **kwargs: caught.append(msg)
            wp.launch_tiled(k, dim=1, inputs=[a, b, c], block_dim=64, device="metal:0")
            wp.synchronize_device("metal:0")
            assert any("precision='tf32'" in m for m in caught), caught
            np.testing.assert_allclose(c.numpy(), 1.5 * (a_np @ b_np), rtol=1e-5, atol=1e-5)
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_tile_map_adjoint_with_forward_result(self):
        # GH-2031: tile_map over builtins whose adjoint takes the forward
        # result (exp, sqrt, tanh, pow, div) — gradients match the CPU.
        snippet = textwrap.dedent(
            """
            import numpy as np

            M, N = 4, 8

            @wp.kernel(module="unique")
            def k_unary(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
                t = wp.tile_load(x, shape=(M, N))
                e = wp.tile_map(wp.exp, t)
                s = wp.tile_map(wp.sqrt, t)
                h = wp.tile_map(wp.tanh, t)
                wp.tile_store(y, wp.tile_map(wp.add, wp.tile_map(wp.add, e, s), h))

            @wp.kernel(module="unique")
            def k_binary(x: wp.array2d(dtype=float), z: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
                t = wp.tile_load(x, shape=(M, N))
                u = wp.tile_load(z, shape=(M, N))
                wp.tile_store(y, wp.tile_map(wp.add, wp.tile_map(wp.pow, t, u), wp.tile_map(wp.div, t, u)))

            rng = np.random.default_rng(11)
            x_np = rng.uniform(0.5, 2.0, (M, N)).astype(np.float32)
            z_np = rng.uniform(1.0, 3.0, (M, N)).astype(np.float32)
            g_np = rng.uniform(-1.0, 1.0, (M, N)).astype(np.float32)
            for name, k, ins in (("unary", k_unary, [x_np]), ("binary", k_binary, [x_np, z_np])):
                res = {}
                for dev in ("cpu", "metal:0"):
                    arrs = [wp.array(v, device=dev, requires_grad=True) for v in ins]
                    y = wp.zeros((M, N), dtype=float, device=dev, requires_grad=True)
                    tape = wp.Tape()
                    with tape:
                        wp.launch_tiled(k, dim=1, inputs=[*arrs, y], block_dim=64, device=dev)
                    y.grad = wp.array(g_np, device=dev)
                    tape.backward()
                    res[dev] = [y.numpy()] + [a.grad.numpy() for a in arrs]
                for i, (c, m) in enumerate(zip(res["cpu"], res["metal:0"])):
                    np.testing.assert_allclose(m, c, rtol=2e-5, atol=1e-5, err_msg=f"{name} output {i}")
                assert np.abs(res["cpu"][1]).sum() > 0
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_texture_sampling_matches_cpu_bit_exact(self):
        # GH-1301 / GH-2065: the 1.19 CPU sampler reduces periodic
        # coordinates first, rounds CLOSEST lookups from the original
        # coordinate in double precision, keeps BORDER lookups at large or
        # tiny coordinates outside the texture, and samples 32-bit integer
        # textures (closest only) without normalization. The texture
        # parameter struct now packs the format into one metadata word.
        snippet = textwrap.dedent(
            """
            import itertools

            import numpy as np

            @wp.kernel
            def k1(tex: wp.Texture1D, u: wp.array(dtype=float), o: wp.array(dtype=wp.vec2f)):
                i = wp.tid()
                o[i] = wp.texture_sample(tex, u[i], dtype=wp.vec2f)

            @wp.kernel
            def k2(tex: wp.Texture2D, u: wp.array(dtype=wp.vec2f), lod: float, o: wp.array(dtype=float)):
                i = wp.tid()
                o[i] = wp.texture_sample(tex, u[i], lod=lod, dtype=float)

            @wp.kernel
            def k2a(texs: wp.array(dtype=wp.Texture2D), u: wp.array(dtype=wp.vec2f), o: wp.array(dtype=float),
                    w: wp.array(dtype=int)):
                i = wp.tid()
                t = texs[i % 2]
                o[i] = wp.texture_sample(t, u[i], dtype=float)
                w[i] = t.width * 100 + t.height

            def coords(rng, n, size, normalized):
                scale = 1.0 if normalized else float(size)
                k = rng.integers(-3 * size, 3 * size, n)
                edge = (k / size * scale).astype(np.float32)
                c = [rng.uniform(-3, 3, n) * scale, edge, np.nextafter(edge, np.float32(np.inf)),
                     np.nextafter(edge, np.float32(-np.inf)), ((k + 0.5) / size) * scale, rng.uniform(-1e7, 1e7, n),
                     np.array([1e30, -1e30, np.inf, -np.inf, np.nan, -1e-45, 1e-45, -1e-40, 0.0, -0.0])]
                return np.concatenate(c).astype(np.float32)

            def same(a, b):
                return bool(((a.view(np.uint32) == b.view(np.uint32)) | (np.isnan(a) & np.isnan(b))).all())

            data_rng = np.random.default_rng(0)
            dtypes = [(np.float32, False), (np.uint8, False), (np.float16, False), (np.int32, True), (np.uint32, True)]
            modes = list(wp.TextureAddressMode)
            filters = list(wp.TextureFilterMode)
            for (npdt, int32), mode, filt, norm in itertools.product(dtypes, modes, filters, (True, False)):
                if int32 and filt == wp.TextureFilterMode.LINEAR:
                    continue
                tag = f"{np.dtype(npdt).name} {mode.name} {filt.name} normalized={norm}"
                if np.issubdtype(npdt, np.integer):
                    info = np.iinfo(npdt)
                    d1 = data_rng.integers(max(info.min, -2**30), min(info.max, 2**30), (7, 2)).astype(npdt)
                    d2 = data_rng.integers(max(info.min, -2**30), min(info.max, 2**30), (5, 6)).astype(npdt)
                else:
                    d1 = data_rng.standard_normal((7, 2)).astype(npdt)
                    d2 = data_rng.standard_normal((5, 6)).astype(npdt)
                mips = 1 if int32 else 2
                res = {}
                for dev in ("cpu", "metal:0"):
                    rng = np.random.default_rng(1)
                    kw = dict(filter_mode=filt, address_mode=mode, normalized_coords=norm, device=dev)
                    t1 = wp.Texture1D(d1, **kw)
                    t2 = wp.Texture2D(d2, num_mip_levels=mips, mip_filter_mode=filt, **kw)
                    u1 = coords(rng, 24, 7, norm)
                    o1 = wp.zeros(len(u1), dtype=wp.vec2f, device=dev)
                    wp.launch(k1, dim=len(u1), inputs=[t1, wp.array(u1, device=dev), o1], device=dev)
                    out = [o1.numpy()]
                    u2 = np.stack([coords(rng, 24, 6, norm), np.roll(coords(rng, 24, 5, norm), 7)], 1)
                    for lod in (-1.0, 0.0, 0.5, 1.0) if mips > 1 else (-1.0,):
                        o2 = wp.zeros(len(u2), dtype=float, device=dev)
                        wp.launch(k2, dim=len(u2), inputs=[t2, wp.array(u2, dtype=wp.vec2f, device=dev), lod, o2],
                                  device=dev)
                        out.append(o2.numpy())
                    res[dev] = out
                for i, (a, b) in enumerate(zip(res["cpu"], res["metal:0"])):
                    assert same(a, b), f"{tag} output {i}: cpu {a.ravel()[:8]} metal {b.ravel()[:8]}"

            # Arrays of textures read the packed descriptor struct from a buffer.
            a = data_rng.standard_normal((4, 5)).astype(np.float32)
            b = data_rng.standard_normal((6, 3)).astype(np.float32)
            res = {}
            for dev in ("cpu", "metal:0"):
                ta = wp.Texture2D(a, address_mode=wp.TextureAddressMode.WRAP, device=dev)
                tb = wp.Texture2D(b, filter_mode=wp.TextureFilterMode.CLOSEST, device=dev)
                texs = wp.array([ta, tb], dtype=wp.Texture2D, device=dev)
                u = wp.array(np.random.default_rng(2).uniform(-2, 2, (16, 2)).astype(np.float32), dtype=wp.vec2f,
                             device=dev)
                o = wp.zeros(16, dtype=float, device=dev)
                w = wp.zeros(16, dtype=int, device=dev)
                wp.launch(k2a, dim=16, inputs=[texs, u, o, w], device=dev)
                res[dev] = (o.numpy(), w.numpy())
            assert same(res["cpu"][0], res["metal:0"][0]), res
            np.testing.assert_array_equal(res["cpu"][1], res["metal:0"][1])
            """
        )
        _run_with_metal_native(self, snippet, timeout=600)

    def test_handle_mesh_ids_match_cpu(self):
        # GH-2114: ``wp.handle`` is the uint64 object-id type APIC remaps. Mesh
        # ids in handle arrays, handle params, and handle struct fields
        # (also inside struct arrays) work as on the CPU.
        snippet = textwrap.dedent(
            """
            import numpy as np

            @wp.struct
            class MS:
                mesh: wp.handle
                scale: float

            @wp.kernel
            def k(ids: wp.array(dtype=wp.handle), h: wp.handle, s: MS, sa: wp.array(dtype=MS),
                  p: wp.array(dtype=wp.vec3), o: wp.array(dtype=float), oid: wp.array(dtype=wp.handle)):
                i = wp.tid()
                q0 = wp.mesh_query_point_no_sign(ids[0], p[i], 10.0)
                q1 = wp.mesh_query_point_no_sign(h, p[i], 10.0)
                q2 = wp.mesh_query_point_no_sign(s.mesh, p[i], 10.0)
                q3 = wp.mesh_query_point_no_sign(sa[0].mesh, p[i], 10.0)
                o[i] = float(q0.face) + 10.0 * q1.u + 100.0 * q2.v * s.scale + 1000.0 * float(q3.result) * sa[0].scale
                oid[i] = ids[0]

            res = {}
            for d in ("cpu", "metal:0"):
                pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
                m = wp.Mesh(points=wp.array(pts, dtype=wp.vec3, device=d),
                            indices=wp.array([0, 1, 2, 1, 3, 2], dtype=int, device=d))
                s = MS()
                s.mesh = m.id
                s.scale = 2.0
                sa = wp.array([s], dtype=MS, device=d)
                p = wp.array(np.random.default_rng(0).uniform(-1, 2, (16, 3)).astype(np.float32), dtype=wp.vec3,
                             device=d)
                o = wp.zeros(16, dtype=float, device=d)
                oid = wp.zeros(16, dtype=wp.handle, device=d)
                wp.launch(k, 16, [wp.array([m.id], dtype=wp.handle, device=d), m.id, s, sa, p, o, oid], device=d)
                res[d] = o.numpy()
                assert (oid.numpy() == m.id).all()
            np.testing.assert_allclose(res["metal:0"], res["cpu"], rtol=1e-6, atol=1e-5)
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_apic_graph_load_on_metal_raises(self):
        # GH-1724 adds capture_load_from_bytes(). APIC graphs are CPU/CUDA
        # only (capture_begin(apic=True) already raises on Metal); loading
        # one onto a Metal device must say so instead of failing inside the
        # native loader as a "CUDA graph".
        snippet = textwrap.dedent(
            """
            import os
            import tempfile

            @wp.kernel
            def k(a: wp.array(dtype=float)):
                i = wp.tid()
                a[i] = a[i] + 1.0

            a = wp.zeros(4, dtype=float, device="cpu")
            wp.load_module(device="cpu")
            with wp.ScopedCapture(device="cpu", apic=True) as cap:
                wp.launch(k, 4, [a], device="cpu")
            path = os.path.join(tempfile.mkdtemp(), "g.wrp")
            wp.capture_save(cap.graph, path)
            for load in (lambda: wp.capture_load(path, device="metal:0"),
                         lambda: wp.capture_load_from_bytes(open(path, "rb").read(), device="metal:0")):
                try:
                    load()
                except RuntimeError as e:
                    assert "Metal" in str(e), e
                else:
                    raise AssertionError("APIC graph loaded onto a Metal device")
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_float16_scalar_params_match_cpu(self):
        # warp.optim.linear.FSAI (GH-2039) launches kernels with float16
        # scalar params; native dispatch must bind their half bits (it raised
        # a TypeError packing a wp.float16 value).
        snippet = textwrap.dedent(
            """
            import numpy as np

            @wp.kernel
            def k(a: wp.array(dtype=wp.float16), s: wp.float16, o: wp.array(dtype=wp.float16)):
                i = wp.tid()
                o[i] = a[i] * s + s

            res = {}
            for d in ("cpu", "metal:0"):
                a = wp.array(np.linspace(-2, 2, 9).astype(np.float16), device=d)
                o = wp.zeros(9, dtype=wp.float16, device=d)
                out = []
                for s in (wp.float16(1.5), 0.25, np.float16(-3.0)):
                    wp.launch(k, 9, [a, s, o], device=d)
                    out.append(o.numpy().copy())
                res[d] = out
            for c, m in zip(res["cpu"], res["metal:0"]):
                np.testing.assert_array_equal(m, c)
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_range_with_runtime_step_matches_cpu(self):
        # ``range(start, stop, step)`` with a step known only at run time:
        # the MSL loop used ``i < stop`` whatever the sign, so a negative
        # step ran no iterations and a zero step never terminated
        # (upstream test_codegen test_range_dynamic_reverse*). Covers the
        # kernel body, an inlined @wp.func and a loop variable read after
        # the loop.
        snippet = textwrap.dedent(
            """
            import numpy as np

            @wp.func
            def count(start: int, stop: int, step: int):
                n = int(0)
                for _i in range(start, stop, step):
                    n += 1
                return n

            @wp.kernel
            def k(start: int, stop: int, step: int, out: wp.array(dtype=int), n: wp.array(dtype=int)):
                c = int(0)
                last = int(-99)
                for i in range(start, stop, step):
                    if c < 8:
                        out[c] = i
                    c += 1
                    last = i
                n[0] = c
                n[1] = count(start, stop, step)
                n[2] = last

            cases = [(0, 4, 1), (4, 0, -1), (8, 0, -2), (0, 8, 3), (3, 3, -1), (0, 5, 0), (5, 0, 0), (-3, -10, -3)]
            for args in cases:
                res = {}
                for d in ("cpu", "metal:0"):
                    out = wp.full(8, -1, dtype=int, device=d)
                    n = wp.zeros(3, dtype=int, device=d)
                    wp.launch(k, 1, [*args, out, n], device=d)
                    res[d] = (out.numpy(), n.numpy())
                for c, m in zip(res["cpu"], res["metal:0"]):
                    np.testing.assert_array_equal(m, c, err_msg=str(args))
                expected = list(range(*args)) if args[2] != 0 else []
                assert res["cpu"][1][0] == len(expected), (args, res["cpu"])
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_bsr_transpose_under_tape_matches_cpu(self):
        # 1.19's BSR expression evaluation builds the transposed topology
        # through bsr_set_from_triplets under an active wp.Tape. The Metal
        # triplet builder's integer dispatches must stay off the tape (their
        # adjoints have no outputs), as the native CPU/CUDA builder does.
        snippet = textwrap.dedent(
            """
            import numpy as np
            import warp.sparse as sp

            res = {}
            for d in ("cpu", "metal:0"):
                rows = wp.array([0, 1, 2, 3, 2, 1], dtype=int, device=d)
                cols = wp.array([1, 4, 1, 3, 0, 2], dtype=int, device=d)
                vals_np = np.random.default_rng(123).random(size=(6, 1, 3)).astype(np.float32)
                vals = wp.array(vals_np, dtype=wp.types.matrix(shape=(1, 3), dtype=float), device=d)
                A = sp.bsr_zeros(65, 67, wp.types.matrix(shape=(1, 3), dtype=float), device=d)
                sp.bsr_set_from_triplets(A, rows, cols, vals)
                A.values.requires_grad = True
                with wp.Tape() as tape:
                    T = (2.0 * A).transpose().eval()
                seed = wp.array(np.ones_like(T.values.numpy()), dtype=T.values.dtype, device=d)
                tape.backward(grads={T.values: seed})
                T.nnz_sync()
                n = int(T.offsets.numpy()[T.nrow])
                res[d] = (T.offsets.numpy(), T.columns.numpy()[:n], T.values.numpy()[:n], A.values.grad.numpy())
            for c, m in zip(res["cpu"], res["metal:0"]):
                np.testing.assert_allclose(m, c, rtol=1e-6)
            np.testing.assert_allclose(res["metal:0"][3], 2.0)
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)

    def test_unported_tile_reductions_raise(self):
        # Metal has no lowering for these yet. 1.19 changed their CPU/CUDA
        # results (GH-2000 tile_sum axis gradients, GH-2003/GH-1781 scan
        # min/max seeds and uint32, GH-2001 broadcast tile_assign gradients,
        # GH-2010 eigenvalue-only eig3 gradients); Metal must keep refusing
        # them instead of computing something else.
        snippet = textwrap.dedent(
            """
            import numpy as np
            from warp._src.codegen_metal import MetalCodegenError

            M, N, L = 4, 8, 16

            @wp.kernel(module="unique")
            def k_sum_axis(a: wp.array2d(dtype=float), r: wp.array(dtype=float)):
                t = wp.tile_load(a, shape=(M, N))
                wp.tile_store(r, wp.tile_sum(t, axis=0))

            @wp.kernel(module="unique")
            def k_reduce_axis(a: wp.array2d(dtype=float), r: wp.array(dtype=float)):
                t = wp.tile_load(a, shape=(M, N))
                wp.tile_store(r, wp.tile_reduce(wp.max, t, axis=1))

            @wp.kernel(module="unique")
            def k_scan_max(a: wp.array(dtype=float), r: wp.array(dtype=float)):
                wp.tile_store(r, wp.tile_scan_max_inclusive(wp.tile_load(a, shape=L)))

            @wp.kernel(module="unique")
            def k_scan_min_u32(a: wp.array(dtype=wp.uint32), r: wp.array(dtype=wp.uint32)):
                wp.tile_store(r, wp.tile_scan_min_inclusive(wp.tile_load(a, shape=L)))

            @wp.kernel(module="unique")
            def k_assign_bcast(x: wp.array(dtype=float), out: wp.array2d(dtype=float)):
                v = wp.tile_load(x, shape=N)
                dst = wp.tile_zeros(shape=(M, N), dtype=float)
                wp.tile_assign(dst, wp.tile_broadcast(v, shape=(M, N)), offset=(0, 0))
                wp.tile_store(out, dst)

            @wp.kernel(module="unique")
            def k_eig(A: wp.array(dtype=wp.mat33), out: wp.array(dtype=float)):
                i = wp.tid()
                Q = wp.mat33()
                d = wp.vec3()
                wp.eig3(A[i], Q, d)
                wp.atomic_add(out, 0, d[0] + d[1] + d[2])

            dev = "metal:0"
            fwd = [
                (k_sum_axis, lambda: [wp.zeros((M, N), dtype=float, device=dev), wp.zeros(N, dtype=float, device=dev)]),
                (k_reduce_axis, lambda: [wp.zeros((M, N), dtype=float, device=dev), wp.zeros(M, dtype=float, device=dev)]),
                (k_scan_max, lambda: [wp.zeros(L, dtype=float, device=dev), wp.zeros(L, dtype=float, device=dev)]),
                (k_scan_min_u32, lambda: [wp.zeros(L, dtype=wp.uint32, device=dev), wp.zeros(L, dtype=wp.uint32, device=dev)]),
            ]
            for k, mk in fwd:
                try:
                    wp.launch_tiled(k, dim=1, inputs=mk(), block_dim=64, device=dev)
                    wp.synchronize_device(dev)
                except MetalCodegenError:
                    continue
                raise AssertionError(f"{k.key} ran on Metal; it needs a CPU-parity test now")

            def bwd_assign():
                x = wp.zeros(N, dtype=float, device=dev, requires_grad=True)
                out = wp.zeros((M, N), dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch_tiled(k_assign_bcast, dim=1, inputs=[x, out], block_dim=64, device=dev)
                out.grad = wp.ones((M, N), dtype=float, device=dev)
                tape.backward()

            def bwd_eig():
                A = wp.array([wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0)], dtype=wp.mat33, device=dev,
                             requires_grad=True)
                out = wp.zeros(1, dtype=float, device=dev, requires_grad=True)
                tape = wp.Tape()
                with tape:
                    wp.launch(k_eig, dim=1, inputs=[A, out], device=dev)
                tape.backward(loss=out)

            for name, fn in (("adj tile_assign(broadcast)", bwd_assign), ("adj eig3", bwd_eig)):
                try:
                    fn()
                    wp.synchronize_device(dev)
                except MetalCodegenError:
                    continue
                raise AssertionError(f"{name} ran on Metal; it needs a CPU-parity test now")
            """
        )
        _run_with_metal_native(self, snippet, timeout=240)


if __name__ == "__main__":
    unittest.main(verbosity=2)
