# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordering of Warp Metal launches against Torch MPS work (``metal_torch_sync``).

Warp and Torch shared memory zero-copy but encoded on separate
MTLCommandQueues, which Metal does not order: Torch work could run before
committed Warp launches it followed, or after Warp launches it preceded.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

import warp as wp
from warp.tests.test_metal_launch import _has_mlx, _is_apple_silicon


def _has_torch_mps() -> bool:
    try:
        import torch  # noqa: PLC0415

        return torch.backends.mps.is_available()
    except Exception:
        return False


_PREFIX = "import warp as wp\nwp.config.enable_metal = True\nwp.config.quiet = True\nwp.init()\n"


def _run(test_case, code: str, env_extra=None, timeout: int = 300) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="warp_metal_torch_") as f:
        f.write(_PREFIX + textwrap.dedent(code))
        path = f.name
    env = dict(os.environ, WARP_METAL_PSO_ARCHIVE="0")
    env.pop("WARP_METAL_TORCH_SYNC", None)
    env.update(env_extra or {})
    try:
        result = subprocess.run(
            [sys.executable, path], check=False, capture_output=True, text=True, timeout=timeout, env=env
        )
    finally:
        os.unlink(path)
    test_case.assertEqual(
        result.returncode,
        0,
        f"exit {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
    )
    return result.stdout + result.stderr


# Torch and Warp take turns on two shared buffers (one Warp-owned via
# wp.to_torch, one Torch-owned via wp.from_torch) with no synchronization,
# while busy kernels on both sides keep the GPU saturated. Per iteration:
# x -> x+1 (Warp) -> 2x+2 (Torch) -> 2x+3 (Warp) -> x+1.5 (Torch).
_INTERLEAVE = """
import torch
from warp._src.metal_dispatch import get_dispatcher

dev = "metal:0"
N = 1 << 16
ITERS = {iters}

@wp.kernel
def inc(a: wp.array[float]):
    a[wp.tid()] += 1.0

@wp.kernel
def busy(a: wp.array[float], iters: int):
    i = wp.tid()
    x = a[i]
    for _k in range(iters):
        x = wp.sin(x) * 0.5 + 0.25
    a[i] = x

scratch = wp.zeros(N, dtype=float, device=dev)
a = wp.zeros(N, dtype=float, device=dev)
t = wp.to_torch(a)
tt = torch.zeros(N, device="mps")
b = wp.from_torch(tt)
big = torch.randn(1024, 1024, device="mps")
bad = []
for it in range(ITERS):
    if it % 3 == 0:
        wp.launch(busy, dim=N, inputs=[scratch, 3000], device=dev)
    wp.launch(inc, dim=N, inputs=[a], device=dev)
    wp.launch(inc, dim=N, inputs=[b], device=dev)
    if it % 3 == 1:
        big = (big @ big).tanh()
    t.mul_(2.0)
    tt.mul_(2.0)
    wp.launch(inc, dim=N, inputs=[a], device=dev)
    wp.launch(inc, dim=N, inputs=[b], device=dev)
    t.sub_(t.clone() * 0.5)
    tt.sub_(tt.clone() * 0.5)
    if it % 10 == 9:
        exp = 1.5 * (it + 1)
        v1, v2 = float(t[0].item()), float(tt[-1].item())
        if v1 != exp or v2 != exp:
            bad.append((it, v1, v2, exp))
            t.fill_(exp)
            tt.fill_(exp)

# Host reads through Warp right after Torch writes (wp.synchronize_device
# drains Torch too, like cudaDeviceSynchronize).
for it in range(ITERS // 10):
    big = (big @ big).tanh()
    t.fill_(float(it))
    tt.fill_(float(it))
    if a.numpy()[0] != it or b.numpy()[-1] != it:
        bad.append(("host", it))

mode = get_dispatcher()._torch_mode
assert mode == "{mode}", mode
assert not bad, bad
print("OK", mode)
"""


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
@unittest.skipUnless(_has_torch_mps(), "Torch with MPS is not installed")
class TestMetalTorchSync(unittest.TestCase):
    def test_interleaved_gpu_ordering(self):
        _run(self, _INTERLEAVE.format(iters=400, mode="gpu"))

    def test_interleaved_cpu_fences(self):
        _run(self, _INTERLEAVE.format(iters=100, mode="cpu"), env_extra={"WARP_METAL_TORCH_SYNC": "cpu"})

    def test_fallback_when_torch_queue_unavailable(self):
        # Another Torch build without the symbols: warn once, then order with
        # CPU fences instead of racing.
        code = (
            "import warnings\n"
            "import warp._src.metal_torch_sync as mts\n"
            "mts._SYM_COMMAND_BUFFER = '_ZN2at3mps9MPSStream14noSuchSymbolEv'\n"
            "warnings.simplefilter('always')\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    import torch\n"
            "    _probe = wp.to_torch(wp.zeros(4, dtype=float, device='metal:0'))\n"
            "assert any('CPU fences' in str(w.message) for w in caught), [str(w.message) for w in caught]\n"
        ) + _INTERLEAVE.format(iters=60, mode="cpu")
        _run(self, code)

    def test_interop_does_not_drain(self):
        # On Torch's queue neither wp.to_torch nor wp.from_torch nor a launch
        # on shared arrays waits on the CPU.
        code = """
        import torch
        from warp._src.metal_dispatch import get_dispatcher

        @wp.kernel
        def inc(a: wp.array[float]):
            a[wp.tid()] += 1.0

        a = wp.zeros(64, dtype=float, device="metal:0")
        t = wp.to_torch(a)  # binds Torch's queue
        disp = get_dispatcher()
        assert disp._torch_mode == "gpu", disp._torch_mode
        calls = {"torch": 0, "warp": 0}
        real_torch_sync, real_sync = torch.mps.synchronize, disp.sync

        def torch_sync():
            calls["torch"] += 1
            real_torch_sync()

        def warp_sync(*args, **kwargs):
            # A sync with nothing queued (wp.zeros' host memset) does not wait.
            calls["warp"] += int(disp.has_pending_work())
            real_sync(*args, **kwargs)

        torch.mps.synchronize = torch_sync
        disp.sync = warp_sync
        for _ in range(20):
            x = wp.zeros(64, dtype=float, device="metal:0")
            tx = wp.to_torch(x)
            tx.add_(1.0)
            w = wp.from_torch(torch.ones(64, device="mps"))
            wp.launch(inc, dim=64, inputs=[w], device="metal:0")
            wp.launch(inc, dim=64, inputs=[x], device="metal:0")
            wp.launch(inc, dim=64, inputs=[a], device="metal:0")
            t.mul_(2.0)
        assert calls == {"torch": 0, "warp": 0}, calls
        torch.mps.synchronize = real_torch_sync
        disp.sync = real_sync
        exp = 0.0
        for _ in range(20):
            exp = 2.0 * (exp + 1.0)
        assert float(t.sum().item()) == exp * 64, (float(t.sum().item()), exp * 64)
        assert float(tx.sum().item()) == 2.0 * 64
        """
        _run(self, code)

    def test_batch_defers_commit(self):
        code = """
        import torch
        from warp._src.metal_dispatch import get_dispatcher

        @wp.kernel
        def inc(a: wp.array[float]):
            a[wp.tid()] += 1.0

        @wp.kernel
        def busy(a: wp.array[float], iters: int):
            i = wp.tid()
            x = a[i]
            for _k in range(iters):
                x = wp.sin(x) * 0.5 + 0.25
            a[i] = x

        N = 1 << 12
        a = wp.zeros(N, dtype=float, device="metal:0")
        t = wp.to_torch(a)
        disp = get_dispatcher()
        scratch = wp.zeros(1 << 16, dtype=float, device="metal:0")
        for step in range(12):
            t.add_(1.0)
            if step == 0:
                # Keep the GPU busy so idle-GPU commits don't kick in.
                wp.launch(busy, dim=scratch.shape[0], inputs=[scratch, 20000], device="metal:0")
            with wp.metal_torch_batch("metal:0"):
                for _ in range(10):
                    wp.launch(inc, dim=N, inputs=[a], device="metal:0")
                if step == 0:
                    # Batched: still open, not committed per launch.
                    assert disp._cmd_buf is not None and disp._cb_shared
            assert disp._cmd_buf is None and not disp._cb_shared
            t.mul_(2.0)
        # x -> 2 * (x + 11) per step
        exp = 0.0
        for _ in range(12):
            exp = 2.0 * (exp + 11.0)
        assert float(t[0].item()) == exp, (float(t[0].item()), exp)
        """
        _run(self, code)

    def test_fence_commits_pending_work(self):
        code = """
        import torch
        from warp._src.metal_dispatch import get_dispatcher

        @wp.kernel
        def busy(a: wp.array[float], iters: int):
            i = wp.tid()
            x = a[i]
            for _k in range(iters):
                x = wp.sin(x) * 0.5 + 0.25
            a[i] = x

        @wp.kernel
        def setv(a: wp.array[float], v: float):
            a[wp.tid()] = v

        _bind = wp.to_torch(wp.zeros(1, dtype=float, device="metal:0"))
        disp = get_dispatcher()
        assert disp._torch_mode == "gpu"
        scratch = wp.zeros(1 << 16, dtype=float, device="metal:0")
        b = wp.zeros(1 << 12, dtype=float, device="metal:0")
        wp.launch(busy, dim=scratch.shape[0], inputs=[scratch, 20000], device="metal:0")
        # Not shared with Torch and the GPU is busy: the launch stays batched.
        wp.launch(setv, dim=b.shape[0], inputs=[b, 3.0], device="metal:0")
        assert disp._cmd_buf is not None
        wp.metal_torch_fence("metal:0")
        assert disp._cmd_buf is None and not disp._cb_shared
        wp.metal_torch_fence("metal:0")  # nothing pending: commits Torch only
        assert float(b.numpy().sum()) == 3.0 * b.shape[0]
        """
        _run(self, code)


class TestMetalTorchSyncHost(unittest.TestCase):
    """Host-side parts that run without a Metal device."""

    def test_helpers_are_noops_off_metal(self):
        wp.metal_torch_fence("cpu")
        with wp.metal_torch_batch("cpu"):
            pass

    def test_requested_mode(self):
        import warnings  # noqa: PLC0415

        from warp._src import metal_torch_sync as mts  # noqa: PLC0415

        old = os.environ.get("WARP_METAL_TORCH_SYNC")
        try:
            os.environ.pop("WARP_METAL_TORCH_SYNC", None)
            self.assertEqual(mts.requested_mode(), "gpu")
            os.environ["WARP_METAL_TORCH_SYNC"] = "CPU"
            self.assertEqual(mts.requested_mode(), "cpu")
            os.environ["WARP_METAL_TORCH_SYNC"] = "sometimes"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assertEqual(mts.requested_mode(), "gpu")
            self.assertTrue(any("WARP_METAL_TORCH_SYNC" in str(w.message) for w in caught))
        finally:
            if old is None:
                os.environ.pop("WARP_METAL_TORCH_SYNC", None)
            else:
                os.environ["WARP_METAL_TORCH_SYNC"] = old

    def test_sync_type_from_header(self):
        from warp._src import metal_torch_sync as mts  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "include", "ATen", "mps"))
            header = os.path.join(d, "include", "ATen", "mps", "MPSStream.h")
            self.assertEqual(mts._sync_type_commit(d), mts._SYNC_TYPE_COMMIT_DEFAULT)
            with open(header, "w") as f:
                f.write("enum class SyncType {\n  NONE, // none\n  COMMIT, // commit\n  COMMIT_AND_WAIT,\n};\n")
            self.assertEqual(mts._sync_type_commit(d), 1)
            with open(header, "w") as f:
                f.write("enum class SyncType {\n  COMMIT_AND_WAIT,\n  NONE,\n  COMMIT,\n};\n")
            self.assertEqual(mts._sync_type_commit(d), 2)
            with open(header, "w") as f:
                f.write("enum class SyncType {\n  NONE = 3,\n  COMMIT,\n};\n")
            with self.assertRaises(RuntimeError):
                mts._sync_type_commit(d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
