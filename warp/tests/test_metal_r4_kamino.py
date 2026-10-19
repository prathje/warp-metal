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

"""Metal-vs-CPU checks for gaps found running Newton's Kamino solver on Metal.

Every case runs on ``cpu`` and ``metal:0`` and the results must agree. The
Metal device must be enabled before ``wp.init()``, so each case runs in a
subprocess (see ``test_metal_launch``).
"""

import textwrap
import unittest

from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon, _run_with_metal_native

_PRELUDE = textwrap.dedent(
    """
    import numpy as np

    def both(fn):
        return {dev: fn(dev) for dev in ("cpu", "metal:0")}

    def assert_same(res, rtol=1e-5, atol=1e-5):
        a, b = res["cpu"], res["metal:0"]
        if not isinstance(a, (tuple, list)):
            a, b = [a], [b]
        for x, y in zip(a, b):
            np.testing.assert_allclose(np.asarray(y), np.asarray(x), rtol=rtol, atol=atol)
    """
)


def _run(test_case, body: str, timeout: int = 600):
    _run_with_metal_native(test_case, _PRELUDE + textwrap.dedent(body), timeout=timeout)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
class TestMetalR4Kamino(unittest.TestCase):
    def test_blocked_cholesky_64_transposes(self):
        # Kamino's blocked LLT at block_size=64: transposed matmul operands and
        # ``transpose(solve(transpose(A)))``. Materialising every transpose
        # overflowed the per-thread stack and crashed the GPU compiler.
        _run(
            self,
            """
            B = 64

            @wp.kernel
            def factor(A: wp.array2d[wp.float32], L: wp.array2d[wp.float32], n: int):
                _b, _lane = wp.tid()
                for k in range(0, n, B):
                    akk = wp.tile_load(A, shape=(B, B), offset=(k, k), storage="shared")
                    for j in range(0, k, B):
                        lb = wp.tile_load(L, shape=(B, B), offset=(k, j))
                        lbt = wp.tile_transpose(lb)
                        wp.tile_matmul(lb, lbt, akk, alpha=-1.0)
                    wp.tile_cholesky_inplace(akk)
                    wp.tile_store(L, akk, offset=(k, k))
                    for i in range(k + B, n, B):
                        aik = wp.tile_load(A, shape=(B, B), offset=(i, k), storage="shared")
                        for j in range(0, k, B):
                            l1 = wp.tile_load(L, shape=(B, B), offset=(i, j))
                            l2 = wp.tile_load(L, shape=(B, B), offset=(k, j))
                            l2t = wp.tile_transpose(l2)
                            wp.tile_matmul(l1, l2t, aik, alpha=-1.0)
                        t = wp.tile_transpose(aik)
                        wp.tile_lower_solve_inplace(akk, t)
                        sol = wp.tile_transpose(t)
                        wp.tile_store(L, sol, offset=(i, k))

            n = 2 * B
            rng = np.random.default_rng(3)
            d = rng.standard_normal((n, n)).astype(np.float32)
            a_np = d @ d.T + n * np.eye(n, dtype=np.float32)

            def run(dev):
                a = wp.array(a_np, device=dev)
                l = wp.zeros_like(a)
                wp.launch_tiled(factor, dim=1, inputs=[a, l, n], block_dim=1, device=dev)
                return np.tril(l.numpy())

            res = both(run)
            assert_same(res, rtol=1e-4, atol=1e-4)
            lm = res["metal:0"]
            np.testing.assert_allclose(lm @ lm.T, a_np, rtol=1e-4, atol=1e-2)
            """,
        )

    def test_blocked_cholesky_64_upper_factor(self):
        # Kamino's RCM variant: an upper-mode factor read through its
        # transpose as the solve's L and as a tile_store source, plus a
        # zero tile that is only stored.
        _run(
            self,
            """
            B = 64

            @wp.kernel
            def factor(A: wp.array2d[wp.float32], L: wp.array2d[wp.float32], n: int, skip: int):
                _b, _lane = wp.tid()
                for k in range(0, n, B):
                    akk = wp.tile_load(A, shape=(B, B), offset=(k, k), storage="shared")
                    for j in range(0, k, B):
                        lb = wp.tile_load(L, shape=(B, B), offset=(k, j))
                        wp.tile_matmul(lb, wp.tile_transpose(lb), akk, alpha=-1.0)
                    wp.tile_cholesky_inplace(akk, fill_mode="upper")
                    lkk = wp.tile_transpose(akk)
                    wp.tile_store(L, lkk, offset=(k, k))
                    for i in range(k + B, n, B):
                        if i // B == skip:
                            zeros = wp.tile_zeros(shape=(B, B), dtype=wp.float32)
                            wp.tile_store(L, zeros, offset=(i, k))
                            continue
                        aik = wp.tile_load(A, shape=(B, B), offset=(i, k), storage="shared")
                        for j in range(0, k, B):
                            l1 = wp.tile_load(L, shape=(B, B), offset=(i, j))
                            l2 = wp.tile_load(L, shape=(B, B), offset=(k, j))
                            wp.tile_matmul(l1, wp.tile_transpose(l2), aik, alpha=-1.0)
                        t = wp.tile_transpose(aik)
                        wp.tile_lower_solve_inplace(lkk, t)
                        wp.tile_store(L, wp.tile_transpose(t), offset=(i, k))

            n = 2 * B
            rng = np.random.default_rng(5)
            d = rng.standard_normal((n, n)).astype(np.float32)
            a_np = d @ d.T + n * np.eye(n, dtype=np.float32)

            def run(dev, skip):
                a = wp.array(a_np, device=dev)
                l = wp.full((n, n), 7.0, dtype=wp.float32, device=dev)
                wp.launch_tiled(factor, dim=1, inputs=[a, l, n, skip], block_dim=1, device=dev)
                return np.tril(l.numpy())

            res = both(lambda dev: run(dev, -1))
            assert_same(res, rtol=1e-4, atol=1e-4)
            lm = res["metal:0"]
            np.testing.assert_allclose(lm @ lm.T, a_np, rtol=1e-4, atol=1e-2)
            res = both(lambda dev: run(dev, 1))
            assert_same(res, rtol=1e-4, atol=1e-4)
            np.testing.assert_array_equal(res["metal:0"][B:, :B], 0.0)
            """,
        )

    def test_module_fp64_emulate(self):
        # ``metal_fp64="emulate"`` as a module option gives that module's
        # float64 locals real double precision while the process-wide mode
        # stays ``"error"`` (Kamino's APGD natural-map residual).
        _run(
            self,
            """
            wp.set_module_options({"metal_fp64": "emulate"})

            @wp.kernel
            def k(a: wp.array[float], out: wp.array[float]):
                i = wp.tid()
                x = wp.float64(a[i]) * wp.float64(1.0e8) + wp.float64(1.0)
                out[i] = wp.float32(x - wp.float64(a[i]) * wp.float64(1.0e8))

            def run(dev):
                a = wp.array([1.0, 3.0, 7.0], dtype=float, device=dev)
                o = wp.zeros(3, dtype=float, device=dev)
                wp.launch(k, dim=3, inputs=[a, o], device=dev)
                return o.numpy()

            assert wp.config.metal_fp64 == "error"
            res = both(run)
            assert_same(res, rtol=0.0, atol=0.0)
            np.testing.assert_array_equal(res["metal:0"], 1.0)
            """,
        )

    def test_graph_replay_after_large_private_kernel(self):
        # Once an ICB-capable pipeline with a large private tile has run, the
        # Apple GPU driver fails every later indirect-command-buffer execution
        # with "Insufficient Memory". Such kernels dispatch eagerly through a
        # direct-only pipeline.
        _run(
            self,
            """
            R = 128

            @wp.kernel
            def big(idx: wp.array[int], out: wp.array2d[float]):
                _b, _l = wp.tid()
                t = wp.tile_load(out, shape=(R, 64))
                for i in range(idx.shape[0]):
                    t[idx[i] % R, idx[i] % 64] = float(i)
                wp.tile_store(out, t)

            @wp.kernel
            def inc(a: wp.array[float]):
                a[wp.tid()] += 1.0

            dev = "metal:0"
            a = wp.zeros(16, dtype=float, device=dev)
            with wp.ScopedCapture(device=dev) as c:
                wp.launch(inc, dim=16, inputs=[a], device=dev)
            wp.capture_launch(c.graph)
            o = wp.zeros((R, 64), dtype=float, device=dev)
            wp.launch_tiled(big, dim=1, inputs=[wp.array(np.arange(7), dtype=int, device=dev), o], block_dim=1, device=dev)
            wp.capture_launch(c.graph)
            np.testing.assert_array_equal(a.numpy(), 2.0)
            assert o.numpy()[6, 6] == 6.0
            """,
        )

    def test_graph_capture_of_large_private_kernel(self):
        # A kernel with more private memory than an ICB supports used to be
        # encoded into the graph's ICB; replay failed with "Insufficient
        # Memory" and every later graph in the process failed too. Graph
        # capture now dispatches it directly, in order with the ICB commands
        # around it, also inside capture_if / capture_while. The replayed
        # results must match an eager run of the same launches on the CPU.
        _run(
            self,
            """
            from warp._src import metal_dispatch as md

            R, C = 96, 64

            @wp.kernel
            def big(a: wp.array2d[float], out: wp.array2d[float]):
                _b, _l = wp.tid()
                t = wp.tile_load(a, shape=(R, C))
                o = wp.tile_load(out, shape=(R, C))
                u = wp.tile_map(wp.add, o, wp.tile_map(wp.mul, t, t))
                wp.tile_store(out, u)

            @wp.kernel
            def inc(a: wp.array2d[float], v: float):
                i, j = wp.tid()
                a[i, j] = a[i, j] + v

            @wp.kernel
            def count_down(flag: wp.array[int], cnt: wp.array[int]):
                cnt[0] = cnt[0] - 1
                if cnt[0] <= 0:
                    flag[0] = 0

            @wp.kernel
            def locals_big(a: wp.array[float], out: wp.array[float]):
                i = wp.tid()
                s = wp.zeros(shape=(6144,), dtype=float)
                for j in range(6144):
                    s[j] = a[j % a.shape[0]] + float(j % 97)
                acc = float(0.0)
                for j in range(6144):
                    acc += s[(j * 7 + i) % 6144]
                out[i] = out[i] + acc

            rng = np.random.default_rng(3)
            A = (0.1 * rng.standard_normal((R, C))).astype(np.float32)

            def make(dev):
                return dict(
                    a=wp.array(A, device=dev),
                    out=wp.zeros((R, C), dtype=float, device=dev),
                    flag=wp.zeros(1, dtype=int, device=dev),
                    cnt=wp.zeros(1, dtype=int, device=dev),
                    off=wp.zeros(1, dtype=int, device=dev),
                    small=wp.zeros(16, dtype=float, device=dev),
                    sa=wp.array(np.arange(16, dtype=np.float32), device=dev),
                )

            def step(v, dev, capture):
                wp.launch(inc, dim=(R, C), inputs=[v["a"], 1.0], device=dev)
                wp.launch_tiled(big, dim=1, inputs=[v["a"], v["out"]], block_dim=64, device=dev)
                wp.launch(inc, dim=(R, C), inputs=[v["out"], 0.5], device=dev)

                def body():
                    wp.launch_tiled(big, dim=1, inputs=[v["a"], v["out"]], block_dim=64, device=dev)
                    wp.launch(count_down, dim=1, inputs=[v["flag"], v["cnt"]], device=dev)

                def skipped():
                    wp.launch_tiled(big, dim=1, inputs=[v["a"], v["out"]], block_dim=64, device=dev)

                if capture:
                    with wp.ScopedMetalCaptureWhileBudget(8):
                        wp.capture_while(v["flag"], while_body=body)
                    wp.capture_if(v["off"], on_true=skipped)
                else:
                    while v["flag"].numpy()[0]:
                        body()
                wp.launch(locals_big, dim=16, inputs=[v["sa"], v["small"]], device=dev)

            cpu = make("cpu")
            met = make("metal:0")
            with wp.ScopedCapture(device="metal:0") as cap:
                step(met, "metal:0", capture=True)
            for loops in (3, 2):
                for v in (cpu, met):
                    v["flag"].fill_(1)
                    v["cnt"].fill_(loops)
                step(cpu, "cpu", capture=False)
                wp.capture_launch(cap.graph)
                for name in ("out", "a", "small", "cnt"):
                    np.testing.assert_allclose(met[name].numpy(), cpu[name].numpy(), rtol=1e-5, atol=1e-4, err_msg=name)

            disp = md.get_dispatcher()
            names = {str(disp._pso_name.get(k)) for k in disp._direct_only}
            assert any("locals_big" in n for n in names), names
            assert any("big" in n and "locals" not in n for n in names), names

            # Later graphs in the process still replay.
            with wp.ScopedCapture(device="metal:0") as cap2:
                wp.launch(inc, dim=(R, C), inputs=[met["a"], 1.0], device="metal:0")
            before = met["a"].numpy()
            wp.capture_launch(cap2.graph)
            np.testing.assert_allclose(met["a"].numpy(), before + 1.0, rtol=1e-6)
            """,
        )

    def test_short_leading_dim_launch(self):
        # ``dim=(num_worlds, n)`` with few worlds: threadgroups now span the
        # second axis for kernels that only index by the grid position.
        _run(
            self,
            """
            @wp.kernel
            def k(a: wp.array2d[float], s: wp.array[float]):
                i, j = wp.tid()
                a[i, j] = float(i * 10000 + j)
                wp.atomic_add(s, i, 1.0)

            @wp.kernel
            def k3(a: wp.array3d[int]):
                i, j, l = wp.tid()
                a[i, j, l] = i * 1000000 + j * 1000 + l

            def run(dev):
                a = wp.zeros((2, 3001), dtype=float, device=dev)
                s = wp.zeros(2, dtype=float, device=dev)
                wp.launch(k, dim=(2, 3001), inputs=[a, s], device=dev)
                b = wp.zeros((1, 37, 129), dtype=int, device=dev)
                wp.launch(k3, dim=(1, 37, 129), inputs=[b], device=dev)
                return a.numpy(), s.numpy(), b.numpy()

            assert_same(both(run), rtol=0.0, atol=0.0)
            """,
        )

    def test_module_fp64_option_after_first_launch(self):
        # Setting ``metal_fp64`` after the kernel already ran on Metal must
        # rebuild it instead of reusing the artifact of the old mode.
        _run(
            self,
            """
            @wp.kernel
            def k(a: wp.array[float], out: wp.array[float]):
                i = wp.tid()
                x = wp.float64(a[i]) * wp.float64(1.0e8) + wp.float64(1.0)
                out[i] = wp.float32(x - wp.float64(a[i]) * wp.float64(1.0e8))

            def run(dev):
                a = wp.array([1.0, 3.0], dtype=float, device=dev)
                o = wp.zeros(2, dtype=float, device=dev)
                wp.launch(k, dim=2, inputs=[a, o], device=dev)
                return o.numpy()

            demoted = run("metal:0")
            wp.set_module_options({"metal_fp64": "emulate"})
            np.testing.assert_array_equal(run("metal:0"), 1.0)
            np.testing.assert_array_equal(run("cpu"), 1.0)
            assert not np.array_equal(demoted, 1.0)
            """,
        )

    def test_metal_module_options_do_not_change_module_hash(self):
        # ``metal_*`` module options are Metal-only: CPU/CUDA module hashes
        # (and so their kernel caches) must not change when one is set.
        _run(
            self,
            """
            @wp.kernel
            def k(a: wp.array[float]):
                a[wp.tid()] = 1.0

            a = wp.zeros(2, dtype=float, device="cpu")
            wp.launch(k, dim=2, inputs=[a], device="cpu")
            mod = wp.get_module(k.module.name)
            h0 = mod.get_module_hash()
            wp.set_module_options({"metal_fp64": "emulate"})
            mod.hashers = {}
            assert mod.get_module_hash() == h0
            """,
        )

    def test_transposed_matmul_operands(self):
        # ``tile_matmul(transpose(a), transpose(b), c)`` with rectangular tiles,
        # and a transpose read by ``tile_store`` after a double transpose.
        _run(
            self,
            """
            @wp.kernel
            def k(a: wp.array2d[wp.float32], b: wp.array2d[wp.float32], c: wp.array2d[wp.float32],
                  d: wp.array2d[wp.float32]):
                _i, _lane = wp.tid()
                ta = wp.tile_load(a, shape=(3, 5))
                tb = wp.tile_load(b, shape=(4, 3))
                tc = wp.tile_zeros(shape=(5, 4), dtype=wp.float32)
                wp.tile_matmul(wp.tile_transpose(ta), wp.tile_transpose(tb), tc)
                wp.tile_store(c, tc)
                tt = wp.tile_transpose(ta)
                wp.tile_store(d, wp.tile_transpose(tt))

            rng = np.random.default_rng(0)
            a_np = rng.standard_normal((3, 5)).astype(np.float32)
            b_np = rng.standard_normal((4, 3)).astype(np.float32)

            def run(dev):
                a = wp.array(a_np, device=dev)
                b = wp.array(b_np, device=dev)
                c = wp.zeros((5, 4), dtype=wp.float32, device=dev)
                d = wp.zeros((3, 5), dtype=wp.float32, device=dev)
                wp.launch_tiled(k, dim=1, inputs=[a, b, c, d], block_dim=1, device=dev)
                return c.numpy(), d.numpy()

            res = both(run)
            assert_same(res)
            np.testing.assert_allclose(res["metal:0"][0], a_np.T @ b_np.T, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(res["metal:0"][1], a_np)
            """,
        )

    def test_tile_transpose_is_a_view(self):
        # ``wp.tile_transpose`` is a view of its source on CPU/CUDA: writes to
        # the source after the transpose show through it, and writes through
        # it change the source. Metal used to keep a stale copy whenever the
        # transpose could not be folded into its uses.
        _run(
            self,
            """
            M, N, K = 3, 4, 2

            @wp.func
            def f_read(t: wp.tile(dtype=float, shape=(N, M))):
                return wp.tile_sum(t)

            @wp.func
            def f_write(t: wp.tile(dtype=float, shape=(N, M)), x: wp.tile(dtype=float, shape=(N, K)),
                        y: wp.tile(dtype=float, shape=(K, M))):
                wp.tile_matmul(x, y, t)

            @wp.func
            def f_tmm(l: wp.tile(dtype=float, shape=(M, N)), x: wp.tile(dtype=float, shape=(M, K)),
                      acc: wp.tile(dtype=float, shape=(N, K))):
                lt = wp.tile_transpose(l)
                wp.tile_matmul(lt, x, acc)

            @wp.kernel(enable_backward=False)
            def k(a: wp.array2d[float], x: wp.array2d[float], y: wp.array2d[float], xt: wp.array2d[float],
                  yt: wp.array2d[float], out: wp.array3d[float], sums: wp.array[float], ext: wp.array[float],
                  n: int, out7: wp.array2d[float]):
                _i, _l = wp.tid()
                tx = wp.tile_load(x, shape=(M, K))
                ty = wp.tile_load(y, shape=(K, N))
                txt = wp.tile_load(xt, shape=(N, K), storage="shared")
                tyt = wp.tile_load(yt, shape=(K, M), storage="shared")
                # 0: source written (matmul) after the transpose
                s0 = wp.tile_load(a, shape=(M, N))
                t0 = wp.tile_transpose(s0)
                wp.tile_matmul(tx, ty, s0)
                wp.tile_store(out[0], t0)
                # 1: transpose written (matmul C), source read
                s1 = wp.tile_load(a, shape=(M, N))
                t1 = wp.tile_transpose(s1)
                wp.tile_matmul(txt, tyt, t1)
                wp.tile_store(out[1], wp.tile_transpose(s1))
                # 2: transpose written by a wp.func, element writes through it
                s2 = wp.tile_load(a, shape=(M, N))
                t2 = wp.tile_transpose(s2)
                f_write(t2, txt, tyt)
                t2[1, 2] = 42.0
                t2 += wp.tile_load(xt, shape=(N, M))
                wp.tile_store(out[2], wp.tile_transpose(s2))
                # 3: source element write and in-place add, transpose read by a
                # wp.func, an element read and tile_map
                s3 = wp.tile_load(a, shape=(M, N))
                t3 = wp.tile_transpose(s3)
                s3[2, 1] = 7.0
                s3 += wp.tile_load(a, shape=(M, N))
                wp.tile_store(sums, f_read(t3))
                ext[0] = t3[3, 1]
                wp.tile_store(out[3], wp.tile_map(wp.mul, t3, t3))
                # 4: transpose taken before a loop that updates the source
                s4 = wp.tile_load(a, shape=(M, N))
                t4 = wp.tile_transpose(s4)
                acc = wp.tile_zeros(shape=(K, M), dtype=float)
                for _it in range(n):
                    wp.tile_matmul(ty, t4, acc)
                    wp.tile_matmul(tx, ty, s4)
                wp.tile_store(out[4], wp.tile_transpose(acc))
                # 5: transpose re-taken in a loop, source updated after it
                s5 = wp.tile_load(a, shape=(M, N))
                acc5 = wp.tile_zeros(shape=(N, M), dtype=float)
                for _it in range(n):
                    t5 = wp.tile_transpose(s5)
                    s5 += wp.tile_load(a, shape=(M, N))
                    acc5 += wp.tile_map(wp.mul, t5, t5)
                wp.tile_store(out[5], acc5)
                # 6: assignment copies (``u = t`` is ``wp::copy``): writes to the
                # copy do not reach the source
                s6 = wp.tile_load(a, shape=(M, N))
                t6 = wp.tile_transpose(s6)
                u6 = t6
                u6[0, 0] = -1.0
                wp.tile_store(out[6], wp.tile_transpose(s6))
                # 7: transpose of a loop-local tile inside a wp.func (inlined
                # locals are declared up front)
                txs = wp.tile_load(x, shape=(M, K), storage="shared")
                acc7 = wp.tile_zeros(shape=(N, K), dtype=float, storage="shared")
                for _it in range(n):
                    l7 = wp.tile_load(a, shape=(M, N), storage="shared")
                    f_tmm(l7, txs, acc7)
                wp.tile_store(out7, acc7)

            rng = np.random.default_rng(1)
            a_np = rng.standard_normal((M, N)).astype(np.float32)
            arrs = [rng.standard_normal(sh).astype(np.float32) for sh in ((M, K), (K, N), (N, K), (K, M))]

            def run(dev):
                ins = [wp.array(a_np, device=dev)] + [wp.array(v, device=dev) for v in arrs]
                out = wp.zeros((7, N, M), dtype=float, device=dev)
                sums = wp.zeros(1, dtype=float, device=dev)
                ext = wp.zeros(1, dtype=float, device=dev)
                out7 = wp.zeros((N, K), dtype=float, device=dev)
                wp.launch_tiled(k, dim=1, inputs=[*ins, out, sums, ext, 3, out7], block_dim=1, device=dev)
                return out.numpy(), sums.numpy(), ext.numpy(), out7.numpy()

            res = both(run)
            assert_same(res, rtol=1e-5, atol=1e-4)
            # The CPU reference really is a view (case 0 sees the update).
            o = res["cpu"][0]
            np.testing.assert_allclose(o[0], (a_np + arrs[0] @ arrs[1]).T, rtol=1e-5, atol=1e-5)
            """,
        )

    def test_tile_transpose_views_metal_cannot_lower_raise(self):
        # Where the view semantics cannot be reproduced, Metal must refuse
        # instead of computing something else.
        _run(
            self,
            """
            from warp._src.codegen_metal import MetalCodegenError

            M, N = 3, 4

            @wp.kernel(enable_backward=False, module="unique")
            def k_alias(a: wp.array2d[float], out: wp.array2d[float]):
                _i, _l = wp.tid()
                s = wp.tile_load(a, shape=(N, N))
                t = wp.tile_transpose(s)
                wp.tile_matmul(t, wp.tile_load(a, shape=(N, N)), s)
                wp.tile_store(out, s)

            @wp.kernel(enable_backward=False, module="unique")
            def k_view(a: wp.array2d[float], out: wp.array2d[float]):
                _i, _l = wp.tid()
                s = wp.tile_load(a, shape=(M, N))
                t = wp.tile_transpose(s)
                v = wp.tile_view(t, offset=(1, 0), shape=(2, M))
                wp.tile_assign(v, wp.tile_ones(shape=(1, M), dtype=float), offset=(0, 0))
                wp.tile_store(out, t)

            @wp.kernel(enable_backward=False, module="unique")
            def k_row(a: wp.array2d[float], out: wp.array2d[float]):
                _i, _l = wp.tid()
                s = wp.tile_load(a, shape=(1, N))
                t = wp.tile_transpose(s)
                s += wp.tile_load(a, shape=(1, N))
                wp.tile_store(out, t)

            @wp.kernel(enable_backward=False, module="unique")
            def k_row_ok(a: wp.array2d[float], out: wp.array2d[float]):
                _i, _l = wp.tid()
                s = wp.tile_load(a, shape=(1, N))
                t = wp.tile_transpose(s)
                wp.tile_store(out, t)
                s += wp.tile_load(a, shape=(1, N))

            a = wp.ones((N, N), dtype=float, device="metal:0")
            out = wp.zeros((N, N), dtype=float, device="metal:0")
            for kern in (k_alias, k_view, k_row):
                try:
                    wp.launch_tiled(kern, dim=1, inputs=[a, out], block_dim=1, device="metal:0")
                    wp.synchronize_device("metal:0")
                except MetalCodegenError as e:
                    assert "tile_transpose" in str(e), e
                else:
                    raise AssertionError(f"{kern.key} did not raise")

            def run(dev):
                a = wp.array(np.arange(N, dtype=np.float32).reshape(1, N), device=dev)
                out = wp.zeros((N, 1), dtype=float, device=dev)
                wp.launch_tiled(k_row_ok, dim=1, inputs=[a, out], block_dim=1, device=dev)
                return out.numpy()

            assert_same(both(run))
            """,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
