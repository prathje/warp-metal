# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reentrancy and thread-safety of the Metal buffer registry.

A garbage collection can run at almost any bytecode boundary, including
inside the registry's critical sections, and the finalizers it runs free
Metal arrays (``wp.array.__del__`` -> ``MetalDefaultAllocator.deallocate``).
A release that blocked on the registry lock deadlocked the thread that was
already holding it (seen as a permanent hang in Newton's
``test_xpbd_tetrahedra_use_control_activation``).

Every scenario runs in a subprocess with a timeout so a regression fails
the test instead of hanging the suite. The GC is forced at a fixed point
inside the locked section by wrapping a helper the section calls, which
makes the hang deterministic without the fix.
"""

import platform
import subprocess
import sys
import textwrap
import unittest

_PRELUDE = textwrap.dedent(
    """
    import bisect, gc, sys, threading, types
    import warp as wp
    wp.config.quiet = True
    wp.config.enable_metal = True
    wp.init()
    from warp._src import context as ctx

    class Cycle:
        # Holds a Metal array in a reference cycle, so only the cyclic GC frees it.
        def __init__(self, n=16):
            self.a = wp.empty(n, dtype=wp.float32, device="metal:0")
            self.me = self

    def make_garbage(k=4):
        ptrs = []
        for _ in range(k):
            c = Cycle()
            ptrs.append(c.a.ptr)
        return ptrs

    def check_index():
        with ctx._metal_buffer_lock:
            bases = list(ctx._metal_buffer_bases)
            assert bases == sorted(bases), "base index not sorted"
            assert len(set(bases)) == len(bases), "duplicate bases"
            assert set(bases) == set(ctx._metal_buffer_nbytes), "base index out of step with extents"
            assert set(bases) <= set(ctx._metal_buffer_registry), "indexed base missing from registry"
        assert not ctx._metal_pending_releases, "releases left pending with the lock free"
    """
)


def _metal_available() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core as mx  # noqa: PLC0415

        return mx.metal.is_available()
    except Exception:
        return False


def _run(test_case, snippet: str, timeout: int = 120):
    code = _PRELUDE + textwrap.dedent(snippet) + "\nprint('OK')\n"
    try:
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        test_case.fail(
            f"subprocess hung for {timeout} s (registry deadlock?)\n--- stdout ---\n{out}\n--- stderr ---\n{err}"
        )
    test_case.assertEqual(
        result.returncode,
        0,
        f"subprocess exited with {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
    )
    test_case.assertIn("OK", result.stdout)
    test_case.assertNotIn("Exception ignored", result.stderr, result.stderr)
    return result


@unittest.skipUnless(_metal_available(), "Metal backend requires Apple Silicon and MLX")
class TestMetalBufferRegistryReentrancy(unittest.TestCase):
    def test_gc_inside_register_does_not_deadlock(self):
        # GC (freeing cycle-held Metal arrays) inside ``_metal_register_buffer``'s
        # locked section, while it indexes the new buffer.
        _run(
            self,
            """
            orig = ctx._metal_buffer_length
            fired = []
            gc.collect()
            n0 = len(ctx._metal_buffer_registry)
            garbage = make_garbage()  # cycles become garbage, freed by the collect below
            def length_with_gc(buf):
                if not fired:
                    fired.append(garbage)
                    gc.collect()
                return orig(buf)
            ctx._metal_buffer_length = length_with_gc
            try:
                b = wp.empty(64, dtype=wp.float32, device="metal:0")
            finally:
                ctx._metal_buffer_length = orig
            assert fired
            for p in fired[0]:
                assert p not in ctx._metal_buffer_registry, "deferred release was not applied"
            assert b.ptr in ctx._metal_buffer_registry
            assert ctx._metal_find_buffer(b.ptr + 4)[1] == 4
            del b
            gc.collect()
            assert len(ctx._metal_buffer_registry) == n0, (len(ctx._metal_buffer_registry), n0)
            check_index()
            """,
        )

    def test_gc_inside_find_bisect_does_not_deadlock(self):
        # GC inside ``_metal_find_buffer``'s interior-pointer bisect (the
        # path in the Newton hang). The deferred releases must not move the
        # base table under the outer lookup: the result must stay correct.
        _run(
            self,
            """
            live = [wp.empty(32, dtype=wp.float32, device="metal:0") for _ in range(64)]
            target = live[31]
            real_bisect = ctx.bisect
            state = {"armed": True, "freed": make_garbage(8)}
            def bisect_right_with_gc(a, x, *args, **kw):
                i = real_bisect.bisect_right(a, x, *args, **kw)
                if state["armed"]:
                    state["armed"] = False
                    gc.collect()
                return i
            shim = types.SimpleNamespace(
                bisect_right=bisect_right_with_gc,
                bisect_left=real_bisect.bisect_left,
                insort=real_bisect.insort,
            )
            ctx.bisect = shim
            try:
                buf, off = ctx._metal_find_buffer(target.ptr + 12)
            finally:
                ctx.bisect = real_bisect
            assert not state["armed"]
            assert buf is ctx._metal_buffer_registry[target.ptr], "interior lookup returned the wrong buffer"
            assert off == 12, off
            for p in state["freed"]:
                assert p not in ctx._metal_buffer_registry, "deferred release was not applied"
            check_index()
            """,
        )

    def test_gc_inside_zero_does_not_deadlock(self):
        # End-to-end shape of the Newton stack: ``array.zero_()`` ->
        # ``_metal_host_memset`` -> ``_metal_find_buffer`` with a GC freeing
        # Metal arrays from inside the locked lookup.
        _run(
            self,
            """
            import numpy as np
            live = [wp.empty(32, dtype=wp.float32, device="metal:0") for _ in range(16)]
            a = wp.full(64, 3.0, dtype=wp.float32, device="metal:0")
            view = a[8:]
            real_bisect = ctx.bisect
            calls = []
            freed = make_garbage(4)
            def bisect_right_with_gc(lst, x, *args, **kw):
                calls.append(1)
                gc.collect()
                return real_bisect.bisect_right(lst, x, *args, **kw)
            ctx.bisect = types.SimpleNamespace(
                bisect_right=bisect_right_with_gc,
                bisect_left=real_bisect.bisect_left,
                insort=real_bisect.insort,
            )
            try:
                view.zero_()
            finally:
                ctx.bisect = real_bisect
            assert calls
            for p in freed:
                assert p not in ctx._metal_buffer_registry, "deferred release was not applied"
            out = a.numpy()
            assert np.all(out[:8] == 3.0) and np.all(out[8:] == 0.0), out
            check_index()
            """,
        )

    def test_gc_with_low_threshold(self):
        # No instrumentation: a tiny GC threshold with garbage cycles holding
        # Metal arrays, while the hot registry paths run. Hits the locked
        # sections at whatever point the collector happens to run.
        _run(
            self,
            """
            live = [wp.empty(32, dtype=wp.float32, device="metal:0") for _ in range(32)]
            gc.collect()
            n0 = len(ctx._metal_buffer_registry)
            gc.set_threshold(1, 1, 1)
            try:
                for i in range(3000):
                    Cycle(8)
                    x = wp.empty(16, dtype=wp.float32, device="metal:0")
                    live[i % 32][4:].zero_()
                    ctx._metal_find_buffer(live[(i * 7) % 32].ptr + 8)
                    del x
            finally:
                gc.set_threshold(700, 10, 10)
            gc.collect()
            assert len(ctx._metal_buffer_registry) == n0, (len(ctx._metal_buffer_registry), n0)
            check_index()
            """,
            timeout=240,
        )

    def test_foreign_release_from_finalizer_inside_lock(self):
        # ``_metal_release_foreign_buffer`` (a weakref.finalize callback for
        # DLPack/Torch wrappers) running from a GC inside a locked section.
        _run(
            self,
            """
            import weakref
            disp_owner = wp.empty(256, dtype=wp.uint8, device="metal:0")
            mtl = ctx._metal_buffer_registry[disp_owner.ptr]
            from warp._src.metal_dispatch import get_dispatcher
            buf, fptr = get_dispatcher().alloc(256)
            ctx._metal_register_foreign_buffer(fptr, buf)
            ctx._metal_register_foreign_buffer(fptr, buf)  # refcount 2

            class Holder:
                pass
            h = Holder(); h.me = h
            weakref.finalize(h, ctx._metal_release_foreign_buffer, fptr)
            h2 = Holder(); h2.me = h2
            weakref.finalize(h2, ctx._metal_release_foreign_buffer, fptr)
            del h, h2, buf

            orig = ctx._metal_buffer_length
            def length_with_gc(b):
                gc.collect()
                return orig(b)
            ctx._metal_buffer_length = length_with_gc
            try:
                c = wp.empty(32, dtype=wp.float32, device="metal:0")
            finally:
                ctx._metal_buffer_length = orig
            assert fptr not in ctx._metal_buffer_registry, "foreign release not applied"
            assert fptr not in ctx._metal_foreign_refcounts
            assert c.ptr in ctx._metal_buffer_registry
            check_index()
            """,
        )

    def test_nested_registration_is_loud(self):
        # A finalizer that *allocates* Metal memory inside a registry critical
        # section cannot be deferred; it must raise, never deadlock or
        # corrupt the outer section's view of the index.
        _run(
            self,
            """
            orig = ctx._metal_buffer_length
            fired = []
            def nested_alloc(buf):
                if not fired:
                    fired.append(1)
                    wp.empty(8, dtype=wp.float32, device="metal:0")
                return orig(buf)
            ctx._metal_buffer_length = nested_alloc
            try:
                try:
                    wp.empty(8, dtype=wp.float32, device="metal:0")
                except RuntimeError as e:
                    assert "critical section" in str(e), e
                else:
                    raise AssertionError("nested registration did not raise")
            finally:
                ctx._metal_buffer_length = orig
            gc.collect()
            check_index()
            """,
        )

    def test_release_with_lock_held_by_other_thread(self):
        # A release while another thread holds the lock is deferred, not
        # blocked; the holder applies it when it leaves the critical section.
        _run(
            self,
            """
            a = wp.empty(16, dtype=wp.float32, device="metal:0")
            ptr = a.ptr
            entered, go = threading.Event(), threading.Event()
            def holder():
                with ctx._metal_buffer_lock:
                    entered.set()
                    go.wait(10)
                ctx._metal_drain_pending_releases()
            t = threading.Thread(target=holder)
            t.start()
            assert entered.wait(10)
            del a  # must return immediately
            assert ptr in ctx._metal_buffer_registry, "release ran while another thread held the lock"
            assert ctx._metal_pending_releases
            go.set()
            t.join(10)
            assert not t.is_alive()
            assert ptr not in ctx._metal_buffer_registry
            check_index()
            """,
        )

    def test_multithreaded_alloc_free_stress(self):
        # Several threads allocate, look up and free Metal arrays (some via
        # GC cycles) with a tiny GC threshold. Must finish, leave the index
        # consistent, and release every buffer.
        _run(
            self,
            """
            gc.collect()
            n0 = len(ctx._metal_buffer_registry)
            errors = []
            def worker(seed):
                try:
                    keep = []
                    for i in range(600):
                        a = wp.empty(8 + (i + seed) % 32, dtype=wp.float32, device="metal:0")
                        buf, off = ctx._metal_find_buffer(a.ptr + 4)
                        assert buf is not None and off == 4, (buf, off)
                        Cycle(4)
                        keep.append(a)
                        if len(keep) > 8:
                            keep.pop(seed % len(keep))
                except BaseException as e:
                    errors.append(repr(e))
            gc.set_threshold(5, 2, 2)
            try:
                threads = [threading.Thread(target=worker, args=(s,)) for s in range(6)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(180)
                assert not any(t.is_alive() for t in threads), "worker thread hung"
            finally:
                gc.set_threshold(700, 10, 10)
            assert not errors, errors
            gc.collect()
            ctx._metal_drain_pending_releases()
            assert len(ctx._metal_buffer_registry) == n0, (len(ctx._metal_buffer_registry), n0)
            check_index()
            """,
            timeout=300,
        )

    def test_shutdown_with_lock_held_does_not_hang(self):
        # A daemon thread parks inside the critical section at exit; the
        # finalizers that run during interpreter shutdown must not block.
        _run(
            self,
            """
            keep = [Cycle(8) for _ in range(16)]
            arrays = [wp.empty(16, dtype=wp.float32, device="metal:0") for _ in range(16)]
            entered = threading.Event()
            def park():
                ctx._metal_buffer_lock.acquire()
                entered.set()
                threading.Event().wait()
            threading.Thread(target=park, daemon=True).start()
            assert entered.wait(10)
            del arrays
            """,
            timeout=60,
        )


@unittest.skipUnless(_metal_available(), "Metal backend requires Apple Silicon and MLX")
class TestMetalResidentSetReentrancy(unittest.TestCase):
    def test_remove_during_apply_keeps_later_adds(self):
        # ``remove_resident_resource`` runs from finalizers (Bvh/Mesh/HashGrid
        # ``__del__``). A removal while ``_apply_resident_resources`` declares
        # the set to an encoder must not leave a stale applied-count that
        # makes the next encoder skip a newly added resource.
        _run(
            self,
            """
            from warp._src.metal_dispatch import MetalDispatcher
            d = types.SimpleNamespace(
                _resident_resources=[], _resident_ids=set(), _resident_applied=0,
                _resident_encoder=None, _usage_read_write=3,
            )
            add = lambda r: MetalDispatcher.add_resident_resource(d, r)
            rem = lambda r: MetalDispatcher.remove_resident_resource(d, r)
            res = [object() for _ in range(4)]
            for r in res:
                add(r)
            class Enc:
                def __init__(self, hook=None):
                    self.used, self.hook = [], hook
                def useResource_usage_(self, r, usage):
                    self.used.append(r)
                    if self.hook is not None:
                        h, self.hook = self.hook, None
                        h()
            enc = Enc(hook=lambda: rem(res[0]))  # a finalizer fires mid-apply
            MetalDispatcher._apply_resident_resources(d, enc)
            new = object()
            add(new)
            MetalDispatcher._apply_resident_resources(d, enc)
            assert new in enc.used, "resource added after a mid-apply removal was never declared"
            """,
        )

    def test_nested_removal_is_not_undone(self):
        # A finalizer removal that interrupts another removal's list rebuild
        # must survive it (else the removed buffer stays resident forever).
        _run(
            self,
            """
            from warp._src.metal_dispatch import MetalDispatcher
            d = types.SimpleNamespace(
                _resident_resources=[], _resident_ids=set(), _resident_applied=0,
                _resident_encoder=None, _usage_read_write=3,
            )
            res = [object() for _ in range(4)]
            for r in res:
                MetalDispatcher.add_resident_resource(d, r)
            class Hooked(list):
                fired = False
                def __iter__(self):
                    if not Hooked.fired:
                        Hooked.fired = True
                        MetalDispatcher.remove_resident_resource(d, res[2])  # "finalizer"
                    return super().__iter__()
            d._resident_resources = Hooked(d._resident_resources)
            MetalDispatcher.remove_resident_resource(d, res[0])
            assert Hooked.fired
            assert res[2] not in d._resident_resources, "nested removal was undone"
            assert res[0] not in d._resident_resources
            assert list(d._resident_resources) == [res[1], res[3]], d._resident_resources
            """,
        )

    def test_ptab_release_from_finalizer_during_update(self):
        # ``_ptab_release`` (a weakref.finalize callback) interrupting a
        # pointer-table residency update is queued and applied, never lost.
        _run(
            self,
            """
            from warp._src import codegen_metal as cm
            buf, other = object(), object()
            log = []
            class FakeDispatcher:
                hook = None
                def add_resident_resource(self, b):
                    log.append(("add", b))
                    if self.hook is not None:
                        h, self.hook = self.hook, None
                        h()
                def remove_resident_resource(self, b):
                    log.append(("remove", b))
            d = FakeDispatcher()
            cm._ptab_retain(d, [buf])           # buf: 1
            cm._ptab_retain(d, [buf])           # buf: 2
            # While retaining ``other``, two finalizers release ``buf``.
            d.hook = lambda: (cm._ptab_release(d, [buf]), cm._ptab_release(d, [buf]))
            cm._ptab_retain(d, [other])
            assert not cm._PTAB_PENDING
            assert id(buf) not in cm._PTAB_RESIDENT_REFS, cm._PTAB_RESIDENT_REFS
            assert cm._PTAB_RESIDENT_REFS[id(other)] == 1
            assert ("remove", buf) in log
            cm._ptab_release(d, [other])
            assert id(other) not in cm._PTAB_RESIDENT_REFS
            assert not cm._PTAB_LOCK.locked()
            """,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
