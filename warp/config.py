# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global configuration settings for Warp.

This module provides settings to control compilation behavior, debugging, performance,
and runtime behavior of Warp kernels and modules. Settings exist at the global, module,
and kernel levels, with more specific scopes taking precedence.

Settings can be modified by direct assignment before or after calling :func:`warp.init`,
though some settings only take effect if set prior to initialization. See individual
setting documentation for details.

For information on module-level and kernel-level settings, see :doc:`/user_guide/configuration`.
"""

import os as _os
import sys as _sys
import types as _types
from enum import IntEnum as _IntEnum

from warp._src.logger import LOG_INFO as _LOG_INFO


class _ConfigModule(_types.ModuleType):
    def __setattr__(self, name, value):
        if name == "launch_array_access_mode" and not isinstance(value, LaunchArrayAccessMode):
            raise ValueError(
                f"warp.config.launch_array_access_mode must be a warp.config.LaunchArrayAccessMode value, got {value!r}"
            )
        if name == "deterministic" and not isinstance(value, DeterministicMode):
            raise ValueError(f"warp.config.deterministic must be a warp.DeterministicMode value, got {value!r}")
        super().__setattr__(name, value)


def _install_config_module_hooks() -> None:
    _sys.modules[__name__].__class__ = _ConfigModule


class LaunchArrayAccessMode(_IntEnum):
    """Array-access verification modes for kernel launches."""

    RELAXED = 0
    """Perform no launch array access checks before launching kernels."""

    CHECKED = 1
    """Detect cross-device Warp array access issues before launching kernels where possible."""

    STRICT = 2
    """Require every Warp array argument to be allocated on the launch device."""


class DeterministicMode(_IntEnum):
    """Deterministic execution modes for supported atomic operations."""

    NOT_GUARANTEED = 0
    """Use normal atomic execution without constraining atomic ordering."""

    RUN_TO_RUN = 1
    """Produce bit-exact repeated results on the same GPU architecture."""

    GPU_TO_GPU = 2
    """Use a stronger path intended to preserve results across GPU architectures."""


launch_array_access_mode: LaunchArrayAccessMode = LaunchArrayAccessMode.RELAXED
"""Kernel launch array access verification mode.

``LaunchArrayAccessMode.RELAXED`` performs no launch array access checks and is
the default. ``LaunchArrayAccessMode.STRICT`` requires every Warp array argument
to be allocated on the launch device, matching Warp's original behavior.
``LaunchArrayAccessMode.CHECKED`` raises an error before launch when Warp can
determine that a cross-device Warp array argument is not accessible from the
launch device. When Warp cannot verify access because an array uses an unknown
custom allocator or externally wrapped allocation, checked mode emits a warning
once for the launch pattern and allows the launch to proceed.

Unlike ``verify_cuda``, this setting can be used during CUDA graph capture
because checks run before each launch is recorded. For cross-GPU graph capture,
enable peer or memory-pool access with Warp APIs before capture begins.

Note: Strict and checked modes impact performance.
"""


def _env_flag(name: str, default: bool = False) -> bool:
    value = _os.environ.get(name, "").strip().lower()
    if value == "":
        return default
    return value not in ("0", "false", "off")


version: str = "1.18.0"
"""Warp version string"""

verify_fp: bool = False
"""Enable floating-point verification for inputs and outputs.

When enabled, checks if all values are finite before and after operations.

Note: Enabling this flag impacts performance.
"""

verify_cuda: bool = False
"""Enable CUDA error checking after kernel launches.

This setting cannot be used during graph capture

Note: Enabling this flag impacts performance
"""

print_launches: bool = False
"""Enable detailed kernel launch logging.

Prints information about each kernel launch including:

- Launch dimensions
- Input/output parameters
- Target device

Note: Enabling this flag impacts performance.
"""

mode: str = "release"
"""Compilation mode for Warp kernels.

Args:
    mode: Either ``"release"`` or ``"debug"``.

Note: Debug mode may impact performance.

This setting can be overridden at the module level by setting the ``"mode"`` module option.
"""

optimization_level: int | None = None
"""Optimization level for Warp kernels.

Args:
    optimization_level: An integer representing the optimization level (0-3), or ``None`` for
        target-specific defaults (``-O2`` for CPU, ``-O3`` for CUDA).

Note: Higher optimization levels increase compilation time but may improve run-time performance.

This setting can be overridden at the module level by setting the ``"optimization_level"`` module option.
"""

verbose_warnings: bool = False
"""Enable extended warning messages with source location information."""

log_level: int = _LOG_INFO
"""Log level threshold for Warp's logging infrastructure.

Messages below this level are suppressed. Use the ``LOG_DEBUG``, ``LOG_INFO``,
``LOG_WARNING``, and ``LOG_ERROR`` constants from the ``warp`` module.

Default is ``warp.LOG_INFO`` (20).
"""

verify_autograd_array_access: bool = False
"""Enable warnings for array overwrites that may affect gradient computation."""

enable_vector_component_overwrites: bool = False
"""Allow multiple writes to vector/matrix/quaternion components.

Note: Enabling this may significantly increase kernel compilation time.
"""

legacy_scalar_return_types: bool = False
"""Use legacy scalar return types from built-in functions and indexing.

When ``False`` (the default), built-in function calls and vector/matrix
indexing return Warp scalar instances (e.g. ``wp.float64``, ``wp.int16``)
that match the input types. The types ``wp.int32``, ``wp.float32``, and
``wp.bool`` are aliases for Python's ``int``, ``float``, and ``bool``, so
those continue to return Python built-in values.

Set to ``True`` to restore the pre-1.12 behavior where all scalar
operations return Python built-in types (``int``, ``float``, ``bool``).
"""

cache_kernels: bool = True
"""Enable kernel caching between application launches."""

kernel_cache_dir: str | None = None
"""Directory path for storing compiled kernel cache.

If ``None``, the path is determined in the following order:

1. ``WARP_CACHE_PATH`` environment variable.
2. System's user cache directory (via ``appdirs.user_cache_directory``).

A version-specific subdirectory is automatically appended to the resolved
base path to prevent cache collisions between different Warp versions.

Note: Subdirectories prefixed with ``wp_`` will be created in this location.
"""

cuda_output: str | None = None
"""Preferred CUDA output format for kernel compilation.

Args:
    cuda_output: One of {``None``, ``"ptx"``, ``"cubin"``}. If ``None``, format is auto-determined.
"""

ptx_target_arch: int | None = None
"""Target architecture version for PTX generation, e.g., ``ptx_target_arch = 75``.

If ``None``, the architecture is determined by devices present in the system.
"""

cuda_arch_suffix: str | None = None
"""CUDA architecture suffix for kernel compilation.

Controls whether architecture-specific or family-specific suffixes are
appended to the ``--gpu-architecture`` flag passed to NVRTC.

Args:
    cuda_arch_suffix: One of {``None``, ``"a"``, ``"f"``}.
        ``None`` disables suffixes (default, current behavior).
        ``"a"`` enables architecture-specific features (requires sm_90+).
        ``"f"`` enables family-specific features (requires sm_100+ and CUDA 12.9+).
"""

lineinfo: bool = False
"""Enable the compilation of modules with line information.

Modules compiled for GPU execution will be compiled with the
``--generate-line-info`` compiler option, which generates line-number
information for device code. Line-number information is always included when
compiling a module in ``"debug"`` mode regardless of this setting.

This setting can be overridden at the module level by setting the ``"lineinfo"`` module option.
"""

line_directives: bool = True
"""Enable Python source line mapping in generated code.

If ``True``, ``#line`` directives are inserted in generated code for modules
compiled with line information to map back to the original Python source file.
"""

compile_time_trace: bool = False
"""Enable the generation of Trace Event Format files for runtime module compilation.

These are JSON files that can be opened by tools like ``edge://tracing/`` and
``chrome://tracing/``.

This setting is currently only effective when compiling modules for the GPU with NVRTC (CUDA 12.8+).

This setting can be overridden at the module level by setting the ``"compile_time_trace"`` module option.
"""

enable_backward: bool = True
"""Enable compilation of kernel backward passes.

This setting can be overridden at the module level by setting the ``"enable_backward"`` module option.
"""

enable_cpu_blocks: bool = False
"""Enable experimental CPU kernel launches with more than one logical thread per block.

When ``False``, CPU launch block dimensions greater than one resolve to one,
preserving the historical CPU execution model. When ``True``, explicit CPU
block dimensions from 2 through 1024 run with cooperative fibers.
"""

default_grid_stride: bool = True
"""Default ``grid_stride`` for kernels that make no explicit choice.

``False`` opts kernels into the lean launch path (no grid-stride loop, lower per-thread overhead and
register pressure, but ``max_blocks`` cannot be capped). Lowest precedence: a per-kernel
``@wp.kernel(grid_stride=...)`` argument or the module ``"default_grid_stride"`` option overrides it.
"""

enable_mathdx_gemm: bool = True
"""Use libmathdx (cuBLASDx) for tile_matmul on GPU when available.

When False, tile_matmul falls back to a scalar GEMM implementation, which avoids
the slow libmathdx LTO compilation at the cost of runtime performance.

This setting can be overridden at the module level by setting the
``"enable_mathdx_gemm"`` module option.
"""

enable_mathdx_solver: bool = True
"""Use libmathdx (cuSolverDx) for tile solver ops on GPU when available.

Controls all cuSolverDx-backed ops: :func:`tile_cholesky <warp._src.lang.tile_cholesky>`
(and its adjoint), :func:`tile_cholesky_solve <warp._src.lang.tile_cholesky_solve>`,
:func:`tile_lower_solve <warp._src.lang.tile_lower_solve>`, and
:func:`tile_upper_solve <warp._src.lang.tile_upper_solve>`.

When False, these ops fall back to cooperative shared-memory implementations
that do not require libmathdx, at the cost of runtime performance.

This setting can be overridden at the module level by setting the
``"enable_mathdx_solver"`` module option.
"""

enable_mathdx_fft: bool = True
"""Use libmathdx (cuFFTDx) for :func:`tile_fft <warp._src.lang.tile_fft>` and
:func:`tile_ifft <warp._src.lang.tile_ifft>` on GPU when available.

When False, these ops use a fallback that supports power-of-two FFT sizes only,
trading runtime performance for faster kernel compile times.

This setting can be overridden at the module level by setting the
``"enable_mathdx_fft"`` module option.
"""

cpu_compiler_flags: str | None = None
"""Flags controlling CPU kernel compilation.

Warp acts as a compiler driver for the embedded Clang frontend. The flag
``-march=native`` is intercepted and triggers host CPU feature detection
(equivalent to ``llvm::sys::getHostCPUName()`` + ``getHostCPUFeatures()``).
All other flags are passed through to the Clang frontend as-is.

The value controls both CPU target detection and extra compiler flags:

- ``None`` (default): detect host CPU features (equivalent to ``"-march=native"``).
- ``""``: disable host CPU detection; compile for a generic target.
- ``"-march=native"``: explicitly detect host CPU features.
- ``"-march=native -fno-vectorize"``: detect host CPU + pass ``-fno-vectorize``.
- ``"-fno-vectorize"``: generic target + pass ``-fno-vectorize``.

Changing this setting invalidates the kernel cache.
"""

llvm_cuda: bool = False
"""Use Clang/LLVM compiler instead of NVRTC for CUDA compilation."""

enable_graph_capture_module_load_by_default: bool = True
"""Enable automatic module loading before graph capture.

Only affects systems with CUDA driver versions below 12.3.
"""

enable_mempools_at_init: bool = _os.environ.get("WARP_ENABLE_MEMPOOLS_AT_INIT", "1").lower() not in ("0", "false", "no")
"""Enable CUDA memory pools during device initialization when supported.

Set ``WARP_ENABLE_MEMPOOLS_AT_INIT`` to ``0``, ``false``, or ``no`` before
importing Warp to disable automatic memory pool initialization (e.g. when
sharing the process with a framework that owns its own GPU allocator).
"""

enable_metal: bool = _env_flag("WARP_ENABLE_METAL")
"""Enable the experimental Metal backend on macOS (Apple Silicon).

Defaults to the ``WARP_ENABLE_METAL`` environment variable (unset → ``False``),
so applications that never touch ``warp.config`` can opt in from the shell.

When ``True``, Warp registers a ``"metal:0"`` device. Requires ``mlx`` to be
importable; raises ``RuntimeError`` at :func:`warp.init` time if it is not.
Kernels are dispatched through Apple's Metal API directly when
``pyobjc-framework-metal`` is installed (see :data:`metal_native_dispatch`),
and through the `MLX <https://github.com/ml-explore/mlx>`_ runtime otherwise.

This setting must be assigned before :func:`warp.init` is called (or before the
first import-driven device enumeration). The Metal backend is incomplete:
Apple GPUs have no ``float64`` hardware (see :data:`metal_fp64`), and
some builtins and tile operations are not yet supported.
"""

metal_fp64: str = _os.environ.get("WARP_METAL_FP64", "").strip().lower() or (
    "demote" if _env_flag("WARP_METAL_DEMOTE_FP64") else "error"
)
"""How the Metal backend handles ``float64`` (Apple GPUs have no fp64 hardware).

Defaults to the ``WARP_METAL_FP64`` environment variable (unset → ``"error"``,
or ``"demote"`` when the older ``WARP_METAL_DEMOTE_FP64=1`` is set), so
applications that never touch ``warp.config`` can choose from the shell.

* ``"error"``: ``float64`` arrays, parameters, and values are rejected at
  code-generation time. Use ``float32`` on Metal.
* ``"demote"``: compute on ``float64`` data in ``float32``. Arrays are still
  *stored* as 8-byte IEEE-754 doubles — host-side ``numpy()`` round-trips are
  unchanged — but every value is converted to ``float32`` when a kernel reads
  it and back when it writes, so all arithmetic happens at single precision.
  Warp warns the first time a kernel is generated this way. ``float64``
  kernel parameters, tiles, and vector/matrix values demote the same way. This
  is the fast option; it silently changes results relative to a CUDA or CPU
  run, and algorithms that chose ``float64`` for conditioning reasons may
  converge differently or not at all.
* ``"emulate"``: compute in real IEEE-754 binary64 using integer arithmetic
  (the bundled `metal-softfloat <https://github.com/guyfischman/metal-softfloat>`_
  routines). Addition, subtraction, multiplication, division, square root,
  ``fma``, comparisons, conversions, and ``floor``/``ceil``/``trunc``/``round``
  are correctly rounded and bit-exact with a CPU run. Transcendental functions
  (``exp``, ``log``, ``sin``, ``cos``, ``pow``, ...) are evaluated in ``float32``
  and widened, so they carry single-precision error. Roughly 30x slower than
  ``float32`` on the same GPU. Requires native dispatch
  (:data:`metal_native_dispatch`); reverse-mode (adjoint) kernels over
  ``float64`` values are not supported in this mode.

In both non-default modes ``wp.atomic_add``/``sub``/``min``/``max`` on a
``float64`` array element are emulated with a per-element lock (Metal has no
64-bit atomics); they are exact but much slower than a native atomic, and the
MLX dispatch path rejects them.

Setting :data:`metal_demote_fp64` to ``True`` is equivalent to ``"demote"``
and is kept for backward compatibility.
"""

metal_demote_fp64: bool = False
"""Deprecated alias for ``metal_fp64 = "demote"`` (see :data:`metal_fp64`).

Setting this to ``True`` selects the ``"demote"`` mode whenever
:data:`metal_fp64` is still ``"error"``. Reading it does not reflect a
``metal_fp64`` assignment; prefer the string setting.
"""

metal_print: str = _os.environ.get("WARP_METAL_PRINT", "").strip().lower() or "warn"
"""How Metal kernels handle ``wp.print()`` / ``wp.printf()``.

Metal compute kernels have no ``printf``, so the calls are compiled out:

* ``"warn"`` (default): drop them and warn once per process, naming the first
  kernel affected.
* ``"ignore"``: drop them silently.
* ``"error"``: raise when a kernel containing one is compiled for Metal.

Initialized from the ``WARP_METAL_PRINT`` environment variable. Does not
affect CPU or CUDA kernels.
"""

metal_native_dispatch: bool = _env_flag("WARP_METAL_NATIVE_DISPATCH", default=True)
"""Route Metal launches through Apple's Metal API directly instead of MLX.

Defaults to ``True``; set the ``WARP_METAL_NATIVE_DISPATCH`` environment
variable to ``0`` to fall back to MLX dispatch (an A/B kill switch).

When ``True`` (and :data:`enable_metal` is also true), Warp's Metal backend
bypasses ``mlx.core.fast.metal_kernel`` for dispatch — kernels write in-place
into Warp-owned ``MTLBuffer``s the same way CUDA kernels write into
``cudaMalloc``-allocated memory. This eliminates the per-launch ``mx.eval`` +
host memcpy that the MLX path requires (≈660 µs/launch on M3) and is ≈30x
faster on dispatch-bound workloads. Graph capture, :class:`warp.Mesh`,
:class:`warp.HashGrid`, :class:`warp.Volume`, and Torch/DLPack interop on
Metal are only available on this path.

Requires ``pyobjc-framework-metal``. If it is not importable Warp warns at
:func:`warp.init` and falls back to MLX dispatch. Set before :func:`warp.init`.
"""

metal_capture_while_max_iters: int = int(_os.environ.get("WARP_METAL_CAPTURE_WHILE_MAX_ITERS", "") or 128)
"""Default iteration budget of :func:`warp.capture_while` inside a Metal graph capture.

Metal command buffers have no device-side loop construct, so a captured
``capture_while`` is replayed as its body unrolled this many times (use
:class:`warp.ScopedMetalCaptureWhileBudget` to override it for one loop). The
value is read when ``capture_while`` is recorded. At the start of each unrolled iteration the GPU latches the loop
condition, and the iteration's kernels only run while it is nonzero, so
iterations after the loop has terminated launch no kernel threads. The body is
recorded once, whatever the budget.

A loop that is still running after its budget is an error: the next sync point
(:func:`warp.synchronize_device`, ``array.numpy()``, ...) or the next
:func:`warp.capture_launch` of the graph raises a ``RuntimeError``, unless
:data:`metal_capture_while_truncate` is set.

Each unrolled iteration costs host time on every :func:`warp.capture_launch`:
about a dozen Metal API calls (condition latch, gate, one indirect execute and
three barriers) per gated command run, independent of how many kernels the run
contains, so budgets of a few hundred iterations cost on the order of a
millisecond per launch. Can be set with the ``WARP_METAL_CAPTURE_WHILE_MAX_ITERS``
environment variable. Outside a capture, ``capture_while`` runs eagerly and is
unbounded, as on other devices.
"""

metal_capture_while_truncate: bool = _env_flag("WARP_METAL_CAPTURE_WHILE_TRUNCATE")
"""Accept truncated captured :func:`warp.capture_while` loops on Metal.

When ``False`` (default), a replayed loop that exhausts its unrolled iteration
budget raises at the next sync point. When ``True`` it only warns once and the
loop's results are those after ``max_iters`` iterations (a fixed-iteration
solver).
"""

metal_threadgroup_tiles: bool = _os.environ.get("WARP_METAL_THREADGROUP_TILES", "").strip().lower() not in (
    "0",
    "false",
    "off",
)
"""Keep Warp tiles resident in Metal threadgroup memory.

Defaults to ``True``; set the ``WARP_METAL_THREADGROUP_TILES`` environment
variable to ``0`` to disable (an A/B kill switch).

When ``True``, a tile whose every use is supported — produced by
:func:`warp.tile_load` or :func:`warp.tile_cholesky`, consumed as
``tile_cholesky``'s input, ``tile_cholesky_solve``'s factor, or
:func:`warp.tile_store`'s source — lives in threadgroup memory for its whole
lifetime instead of being copied through a per-thread private struct in each
of the 32 cooperating lanes. At N=35 the private form costs 4.9 KB per lane
(157 KB per threadgroup, which spills to device-backed scratch) and is ~23x
slower. Any tile with an unsupported use keeps the private representation,
so this only ever adds a fast path.

Scope: promotion currently only happens in kernels that already dispatch 32
cooperating lanes per tile, which today means kernels containing a
``tile_cholesky`` or vector-RHS ``tile_cholesky_solve`` on a square tile of
24 <= N <= 88. A kernel whose tile work is, say, only ``tile_load`` and
``tile_store`` keeps the private representation because it is launched one
thread per tile. Widening that would mean running such kernels with 32 lanes,
which would also execute their scalar body 32 times over -- the
multiply-counting hazard that ``block_dim()`` lowering to ``1`` exists to
avoid -- so it needs per-kernel analysis rather than a blanket switch.
"""

metal_serialize_spin_locks: bool = _os.environ.get("WARP_METAL_SERIALIZE_SPIN_LOCKS", "").strip().lower() not in (
    "0",
    "false",
    "off",
)
"""Rewrite spin locks so they cannot deadlock on Apple GPUs.

Defaults to ``True``; set the ``WARP_METAL_SERIALIZE_SPIN_LOCKS`` environment
variable to ``0`` to disable (an A/B kill switch).

Apple GPUs do not guarantee independent forward progress between the lanes
of a SIMD group, so a mutual-exclusion lock of the form
``while wp.atomic_cas(lock, i, 0, 1) == 1: pass`` followed by a critical
section and a releasing ``wp.atomic_exch(lock, i, 0)`` deadlocks when two
lanes of one SIMD group contend for it. When ``True``, the Metal backend
recognises that shape and runs acquire, critical section, and release one
lane per SIMD group at a time, with device-scope memory fences around the
critical section. Spans it cannot prove safe to rewrite (no release in the
same block, a ``return`` inside the critical section, SIMD-group or tile
operations in it) are left as written with a warning.
"""

metal_coop_tiles: bool = _os.environ.get("WARP_METAL_COOP_TILES", "").strip().lower() not in (
    "0",
    "false",
    "off",
)
"""Run eligible ``wp.launch_tiled`` kernels with real cooperating lanes.

Defaults to ``True``; set the ``WARP_METAL_COOP_TILES`` environment
variable to ``0`` to disable (an A/B kill switch).

A kernel launched via :func:`warp.launch_tiled` with ``block_dim=32`` whose
tile usage is limited to ``wp.tile()`` register-tile construction,
``wp.tile_reduce`` (``wp.add``/``wp.min``/``wp.max``), and
``wp.tile_extract`` of the reduce result is compiled with an honest
``block_dim`` and dispatched as one 32-lane threadgroup per tile:
``wp.block_dim()`` lowers to ``32``, per-lane strided loops partition work
exactly as on CUDA, register tiles hold one element per lane, and
reductions lower to ``simd_sum``/``simd_min``/``simd_max``. Semantics match
CUDA thread-for-thread, so the multiply-counting hazard of the serial model
does not apply. Kernels outside the supported pattern keep the serial
(one-thread-per-tile) model.
"""

disable_metal_solver_icb_capture: bool = False
"""Disable the mujoco_warp solver's MTLIndirectCommandBuffer fast path.

When :data:`metal_native_dispatch` is ``True``, mujoco_warp's
``_solve`` captures the body of each iteration into a
:class:`warp._src.metal_dispatch.MetalGraph` and replays it
``opt.iterations`` times — ~7.5× faster per iteration on G1 than
the plain Python loop because replay skips Warp's per-launch type
checks and ObjC encoder calls. Set this knob to ``True`` to fall
back to the pure-Python ``for _ in range(opt.iterations)`` loop,
e.g. when debugging a solver kernel whose behavior changes between
iterations of the same ``_solve()`` call.
"""

track_memory: bool = False
"""Enable tracking of memory allocations at initialization.

When ``True`` at the time :func:`warp.init` is called, a
:class:`~warp.ScopedMemoryTracker` is automatically activated for all
devices. For on-demand tracking, use :class:`~warp.ScopedMemoryTracker`
directly as a context manager.

Note: Impacts performance when active due to call-stack introspection.
"""

max_unroll: int = 16
"""Maximum unroll factor for loops.

Note that ``max_unroll`` does not consider the total number of iterations in
nested loops. This can result in a large amount of automatically generated code
if each nested loop is below the ``max_unroll`` threshold.

This setting can be overridden at the module level by setting the ``"max_unroll"`` module option.
"""

enable_tiles_in_stack_memory: bool | None = True
"""Use stack memory instead of static memory for tile allocations on the CPU.

Static memory in kernels is not well supported on some architectures (notably AArch64). We work
around it by reserving stack memory on kernel entry and pointing a reserved callee-saved
register to it (AArch64) or a single static pointer (x86-64).

When set to ``None``, this flag automatically enables stack memory on ``aarch64``
platforms (Linux ARM) and disables it on other architectures. To explicitly enable or disable
this behavior regardless of architecture, set this flag to ``True`` or ``False``.
"""

use_precompiled_headers: bool = True
"""Enable the use of precompiled headers during kernel compilation.
"""

legacy_cpu_linker: bool = False
"""Use the legacy RTDyld linker instead of JITLink for CPU kernel loading.

The default JITLink linker is more robust against virtual address space
fragmentation (e.g. caused by the CUDA driver). Set this to ``True`` to
use the older RTDyld linker, which supports step-through debugging of CPU
kernels with pre-built LLVM libraries that lack the JITLink debug symbols.

This setting can be changed at runtime.  Each linker has its own JIT
instance, created lazily on first use and kept alive so that previously
loaded CPU modules remain valid.

.. note::

   Step-through debugging with JITLink requires building Warp with
   ``--build-llvm`` to get LLVM 21+ with the necessary ORC runtime symbols.

.. warning::

   This flag is experimental and may be removed without warning in a
   future release.
"""

load_module_max_workers: int | None = 0
"""Default number of worker threads for compiling and loading modules in parallel.

For ``wp.load_module()`` and ``wp.force_load()``, if the ``max_workers`` parameter is not specified,
the default number of worker threads is determined by this setting. ``0`` means serial loading.
If ``None``, Warp determines the behavior (currently equal to ``min(os.cpu_count(), 4)``).
"""

deterministic: DeterministicMode = DeterministicMode.NOT_GUARANTEED
"""Determinism guarantee for supported atomic operations.

Accepted values are:

- ``wp.DeterministicMode.NOT_GUARANTEED``: Default behavior.
- ``wp.DeterministicMode.RUN_TO_RUN``: Bit-exact repeated results on the same GPU
  architecture.
- ``wp.DeterministicMode.GPU_TO_GPU``: Stronger cross-GPU reproducibility path.

Set this before module creation/import for it to apply broadly. Existing
modules can be changed by setting the ``"deterministic"`` module option.
See :doc:`/user_guide/execution_and_performance/deterministic_execution` for supported patterns,
performance considerations, and limitations.
"""

deterministic_max_records: int = 0
"""Default per-target, per-thread record bound for deterministic atomics.

The default ``0`` uses the code-generated lower bound. Increase this before
module creation/import when kernels have data-dependent loops or repeated visits
to the same atomic site that static analysis cannot bound.

Modules can override this by setting the ``"deterministic_max_records"`` module
option.
"""

deterministic_debug: bool = False
"""Enable debug diagnostics for deterministic execution mode.

When enabled, deterministic scatter overflows may emit device-side diagnostics.
This setting is intended for debugging capacity issues and should remain disabled
for normal execution, especially when CUDA graph capture is performance-critical.
"""

_git_commit_hash: str | None = None
"""Git commit hash associated with the Warp installation.

Set automatically by CI, do not modify.
"""


_install_config_module_hooks()
