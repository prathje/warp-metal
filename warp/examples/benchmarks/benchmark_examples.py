# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the Warp examples across devices and backend configurations.

Every example registered in ``warp/tests/test_examples.py`` and
``warp/tests/fem/test_fem_examples.py`` is run as a separate subprocess under
each requested configuration, with the same command-line options the test
harness uses (so the workloads match the test suite). Each run records

* the wall time of the whole process,
* the time to import and initialize Warp,
* the time spent in ``Example.__init__`` (where the example defines an
  ``Example`` class), and
* every call of the public ``Example`` methods (``step``, ``render``, ...),
  each followed by a device synchronization, so that the first call (kernel
  module load / JIT) and the steady state can be told apart.

Configurations:

* ``cpu``        -- ``--device cpu``
* ``metal``      -- ``--device metal:0`` with ``WARP_ENABLE_METAL=1``
* ``metal_fp64`` -- as ``metal`` plus ``WARP_METAL_FP64=emulate`` (software
  IEEE float64 for the examples that compute in double precision)

Usage::

    uv run warp/examples/benchmarks/benchmark_examples.py [--configs cpu,metal,metal_fp64]
        [--timeout 600] [--out-dir DIR] [--force] [names...]

``names`` are substrings matched against ``group.example_name``. Results are
written to ``DIR/results.json`` (resumable: finished runs are skipped unless
``--force``) and summarized in ``DIR/report.md``. Use ``--report-only`` to
rebuild the report from an existing ``results.json``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import signal
import statistics
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# name -> (device, extra environment, which harness option set to use)
CONFIGS = {
    "cpu": ("cpu", {}, "cpu"),
    "metal": ("metal:0", {"WARP_ENABLE_METAL": "1"}, "cuda"),
    "metal_fp64": ("metal:0", {"WARP_ENABLE_METAL": "1", "WARP_METAL_FP64": "emulate"}, "cuda"),
}
# Environment variables that must not leak from the caller into a run.
_SCRUB_ENV = ("WARP_ENABLE_METAL", "WARP_METAL_FP64", "WARP_METAL_DEMOTE_FP64", "WARP_METAL_NATIVE_DISPATCH")

# Registered examples that cannot run here (multi-GPU / CUDA-only APIs).
SKIP = {"fem.example_diffusion_mgpu", "core.example_custom_allocator"}
# Harness option keys that are not example command-line flags.
_META_KEYS = ("usd_required", "torch_required", "pillow_required", "test_timeout")

_TEST_FILES = ("warp/tests/test_examples.py", "warp/tests/fem/test_fem_examples.py")


# ---------------------------------------------------------------------------
# Runner (executed in the subprocess)
# ---------------------------------------------------------------------------


def _run_example(module: str, out_path: str, argv: list[str]) -> None:
    """Run ``module`` as ``__main__`` with the ``Example`` class instrumented."""
    import builtins  # noqa: PLC0415
    import functools  # noqa: PLC0415
    import runpy  # noqa: PLC0415
    import traceback  # noqa: PLC0415
    import types  # noqa: PLC0415

    record: dict = {"module": module, "argv": argv, "ok": False, "methods": {}}
    t_proc = time.perf_counter()

    def dump():
        record["t_total"] = time.perf_counter() - t_proc
        with open(out_path, "w") as f:
            json.dump(record, f, indent=1)

    t0 = time.perf_counter()
    import warp as wp  # noqa: PLC0415

    record["t_import_warp"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    wp.init()
    record["t_warp_init"] = time.perf_counter() - t0

    methods: dict[str, list[float]] = record["methods"]
    depth = [0]

    def sync():
        try:
            wp.synchronize_device()
        except Exception:  # best effort, e.g. during graph capture
            pass

    def wrap(label, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            depth[0] += 1
            t_call = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                if depth[0] == 1:
                    sync()
                methods.setdefault(label, []).append(time.perf_counter() - t_call)
                depth[0] -= 1

        return wrapper

    orig_build_class = builtins.__build_class__

    def build_class(func, name, *bases, **kwargs):
        cls = orig_build_class(func, name, *bases, **kwargs)
        if name == "Example" and func.__globals__.get("__name__") == "__main__":
            for attr, value in list(vars(cls).items()):
                if isinstance(value, types.FunctionType) and (attr == "__init__" or not attr.startswith("_")):
                    setattr(cls, attr, wrap(attr, value))
        return cls

    builtins.__build_class__ = build_class

    sys.argv = [module.replace(".", "/") + ".py", *argv]
    t0 = time.perf_counter()
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        sync()
        record["ok"] = True
    except SystemExit as e:
        sync()
        record["ok"] = e.code in (None, 0)
        record["error"] = f"SystemExit({e.code})"
    except BaseException:
        record["error"] = traceback.format_exc()[-4000:]
    record["t_main"] = time.perf_counter() - t0
    record["device"] = str(wp.get_device())
    dump()
    if not record["ok"]:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def registered_examples() -> dict[str, dict]:
    """Collect the ``add_*example_test`` registrations (name, options, device policy) from the test files."""
    examples: dict[str, dict] = {}
    for rel in _TEST_FILES:
        with open(os.path.join(ROOT, rel)) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", "").endswith("example_test")):
                continue
            entry = {"test_options": {}, "test_options_cpu": {}, "test_options_cuda": {}, "devices": ""}
            name = None
            for kw in node.keywords:
                if kw.arg == "name":
                    name = ast.literal_eval(kw.value)
                elif kw.arg in ("test_options", "test_options_cpu", "test_options_cuda"):
                    entry[kw.arg] = ast.literal_eval(kw.value)
                elif kw.arg == "devices":
                    entry["devices"] = ast.unparse(kw.value)
            if name and name not in SKIP:
                # The raymarch example is registered twice (macOS CPU-only variant first); keep the general one.
                examples[name] = entry
    return examples


def harness_policy(devices_expr: str) -> str:
    """Human-readable summary of which devices the test harness runs an example on."""
    if "fp64_exact" in devices_expr:
        base = "fp64 exact"
    elif "fp64" in devices_expr:
        base = "fp64"
    else:
        base = ""
    if "gpu" in devices_expr:
        scope = "GPU only"
    elif "cuda" in devices_expr:
        scope = "CUDA only"
    elif devices_expr.startswith("["):
        scope = "CPU only"
    else:
        scope = "CPU + GPU"
    return f"{scope}{', ' + base if base else ''}"


def build_options(entry: dict, option_set: str) -> dict:
    opts = entry["test_options"] | entry[f"test_options_{option_set}"]
    return {k: v for k, v in opts.items() if k not in _META_KEYS}


def options_to_argv(options: dict) -> list[str]:
    argv = []
    for key, value in options.items():
        if key == "headless":
            if value:
                argv.append("--headless")
        else:
            argv.extend(["--" + key.replace("_", "-"), str(value)])
    return argv


# Iteration-count flags forced to 1 for warm-up runs; every example parses with ``parse_known_args`` so
# unknown flags are ignored, and a later flag overrides the harness value.
_WARMUP_ARGV = ["--num-frames", "1", "--train-iters", "1", "--num-steps", "1", "--num-iters", "1"]


def run_one(name: str, config: str, entry: dict, out_dir: str, timeout: float, warmup: bool = False) -> dict:
    """Run one example under one configuration; ``warmup`` runs a one-iteration variant to fill the kernel caches."""
    device, extra_env, option_set = CONFIGS[config]
    options = build_options(entry, option_set)
    argv = ["--device", device, "--stage-path", "None", *options_to_argv(options)]
    if warmup:
        argv += _WARMUP_ARGV
    tag = f"{name}.{config}" + (".warmup" if warmup else "")
    json_path = os.path.join(out_dir, "runs", tag + ".json")
    log_path = os.path.join(out_dir, "runs", tag + ".log")
    if os.path.exists(json_path):
        os.remove(json_path)

    env = {k: v for k, v in os.environ.items() if k not in _SCRUB_ENV}
    env.update(extra_env)
    cmd = [sys.executable, os.path.abspath(__file__), "--_run", f"warp.examples.{name}", json_path, "--", *argv]

    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc: int | str = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            rc = "TIMEOUT"
    wall = time.perf_counter() - t0

    result = {
        "name": name,
        "config": config,
        "device": device,
        "options": options,
        "harness": harness_policy(entry["devices"]),
        "rc": rc,
        "wall": wall,
        "timeout": timeout,
        "log": os.path.relpath(log_path, out_dir),
    }
    if os.path.exists(json_path):
        with open(json_path) as f:
            result.update(json.load(f))
    else:
        result["ok"] = False
    with open(log_path, errors="replace") as f:
        log_text = f.read()
    if result["ok"] and "only runs on CUDA devices" in log_text:
        # Some examples print a notice and exit successfully on unsupported devices.
        result["ok"] = False
        result["error"] = "This example only runs on CUDA devices."
    if not result["ok"] and "error" not in result:
        result["error"] = log_text[-2000:]
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def primary_method(methods: dict[str, list[float]]) -> str | None:
    """The method whose repeated calls represent the example's steady state."""
    public = {k: v for k, v in methods.items() if k != "__init__"}
    if not public:
        return None
    if "step" in public:
        return "step"
    return max(public, key=lambda k: (len(public[k]), sum(public[k])))


def summarize(result: dict) -> dict:
    """Derive the headline numbers of a run."""
    methods = result.get("methods", {})
    out = {
        "ok": result.get("ok", False),
        "rc": result.get("rc"),
        "status": classify(result),
        "wall": result.get("wall"),
        "main": result.get("t_main"),
        "init": (methods.get("__init__") or [None])[0],
    }
    m = primary_method(methods)
    if m:
        calls = methods[m]
        out.update(
            {
                "method": m,
                "calls": len(calls),
                "first": calls[0],
                "steady": statistics.fmean(calls[1:]) if len(calls) > 1 else None,
                "steady_median": statistics.median(calls[1:]) if len(calls) > 1 else None,
            }
        )
    return out


def error_line(result: dict) -> str:
    """The most informative line of a failed run's error text."""
    tail = [ln.strip() for ln in (result.get("error") or "").strip().splitlines() if ln.strip()]
    if not tail:
        return ""
    for ln in reversed(tail):
        if "Error" in ln or "Mismatch" in ln or "only runs" in ln:
            return ln[:200]
    return tail[-1][:200]


def _fmt_s(x) -> str:
    return "" if x is None else f"{x:.2f}"


def _fmt_ms(x) -> str:
    if x is None:
        return ""
    ms = x * 1e3
    return f"{ms:.3g}" if ms < 100 else f"{ms:.0f}"


_UNSUPPORTED_MARKERS = ("Failed to find forward kernel", "only runs on CUDA devices", "only supported on CUDA")
_NEEDS_FP64_MARKERS = ("MSL has no fp64", "element type 'wp::float64'")


def classify(result: dict) -> str:
    """Status of a run: ``ok``, ``timeout``, ``needs fp64`` (Metal without an fp64 mode), ``unsupported``
    (the example does not support the device) or ``failed``."""
    if result.get("ok"):
        return "ok"
    if result.get("rc") == "TIMEOUT":
        return "timeout"
    err = result.get("error") or ""
    if "MetalCodegenError" in err and any(m in err for m in _NEEDS_FP64_MARKERS):
        return "needs fp64"
    if any(m in err for m in _UNSUPPORTED_MARKERS):
        return "unsupported"
    return "failed"


def _status(s: dict) -> str:
    return s["status"]


def _ratio(summaries: dict[str, dict], num: str, den: str) -> str:
    """Ratio of the steady-state (else first-call, marked ``*``) times of two configurations."""
    sn, sd = summaries.get(num), summaries.get(den)
    if not (sn and sd and sn["ok"] and sd["ok"]):
        return ""
    if sn.get("steady") and sd.get("steady"):
        return f"{sn['steady'] / sd['steady']:.2f}x"
    if sn.get("first") and sd.get("first"):
        return f"{sn['first'] / sd['first']:.2f}x*"
    return ""


def write_report(results: dict[str, dict], configs: list[str], path: str) -> None:
    names = sorted({r["name"] for r in results.values()})
    lines = ["# Warp examples benchmark", ""]
    lines.append(
        "Each example runs as its own process with the options of the test harness. `total` is the wall time of "
        "the whole process (interpreter start-up, imports, kernel module load and the example itself). `init` is "
        "`Example.__init__`. `first`/`steady` are the first call and the mean of the remaining calls of the "
        "example's main method (`step` where it exists), each followed by a device synchronization; the first "
        "call includes the kernel module load. `warm-up` columns come from the one-iteration run that precedes each "
        "measured run; its first call includes code generation and compilation whenever the kernel cache was cold. "
        "Times are seconds unless marked ms."
    )
    lines.append("")

    # Steady-state comparison
    lines.append("## Per-step time (ms, steady state)")
    lines.append("")
    lines.append(
        "`*` marks examples whose main method runs once: the number is that single (warm-cache) call, which "
        "includes loading the kernel modules. Ratios > 1 in `cpu / metal` mean Metal is faster; "
        "`metal_fp64 / metal` is the cost of running with fp64 emulation enabled."
    )
    lines.append("")
    header = ["example", "method", *[f"{c} (calls)" for c in configs]]
    if "cpu" in configs and "metal" in configs:
        header.append("cpu / metal")
    if "metal" in configs and "metal_fp64" in configs:
        header.append("metal_fp64 / metal")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for name in names:
        summaries = {c: summarize(results[f"{name}::{c}"]) for c in configs if f"{name}::{c}" in results}
        method = next((s.get("method") for s in summaries.values() if s.get("method")), None)
        if method is None:
            continue
        row = [name, method]
        for c in configs:
            s = summaries.get(c)
            if s is None:
                row.append("")
            elif not s["ok"]:
                row.append(_status(s))
            elif s.get("steady") is not None:
                row.append(f"{_fmt_ms(s['steady'])} ({s['calls']})")
            elif s.get("first") is not None:
                row.append(f"{_fmt_ms(s['first'])}* (1)")
            else:
                row.append("n/a")

        if "cpu" in configs and "metal" in configs:
            row.append(_ratio(summaries, "cpu", "metal"))  # >1: Metal is faster than the CPU
        if "metal" in configs and "metal_fp64" in configs:
            row.append(_ratio(summaries, "metal_fp64", "metal"))  # >1: fp64 emulation costs extra
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Full detail
    lines.append("## Whole-run detail")
    lines.append("")
    header = [
        "example",
        "harness",
        "config",
        "status",
        "total s",
        "init s",
        "first ms",
        "steady ms",
        "calls",
        "warm-up total s",
        "warm-up first ms",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for name in names:
        for c in configs:
            r = results.get(f"{name}::{c}")
            if r is None:
                continue
            s = summarize(r)
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        r.get("harness", ""),
                        c,
                        _status(s),
                        _fmt_s(s["wall"]),
                        _fmt_s(s["init"]),
                        _fmt_ms(s.get("first")),
                        _fmt_ms(s.get("steady")),
                        str(s.get("calls", "")),
                        _fmt_s((r.get("warmup") or {}).get("wall")),
                        _fmt_ms((r.get("warmup") or {}).get("first")),
                    ]
                )
                + " |"
            )
    lines.append("")

    failures = [r for r in results.values() if not r.get("ok")]
    if failures:
        lines.append("## Failures")
        lines.append("")
        for r in failures:
            lines.append(f"- `{r['name']}` [{r['config']}]: {_status(summarize(r))} -- `{error_line(r)}`")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------


def main() -> None:
    # Internal runner mode: benchmark_examples.py --_run MODULE OUT_JSON -- <example arguments>
    if len(sys.argv) > 4 and sys.argv[1] == "--_run" and sys.argv[4] == "--":
        _run_example(sys.argv[2], sys.argv[3], sys.argv[5:])
        return

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", default=",".join(CONFIGS), help="Comma-separated subset of " + ",".join(CONFIGS))
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-run timeout in seconds.")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "benchmark_examples_out"))
    parser.add_argument(
        "--no-warmup", action="store_true", help="Skip the one-iteration warm-up run per configuration."
    )
    parser.add_argument("--force", action="store_true", help="Re-run examples that already have results.")
    parser.add_argument("--report-only", action="store_true", help="Only rebuild the report from results.json.")
    parser.add_argument("--list", action="store_true", help="List the examples and options, then exit.")
    parser.add_argument("names", nargs="*", help="Substrings of group.example_name to select.")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        parser.error(f"Unknown configs: {unknown}")

    examples = registered_examples()
    names = sorted(examples)
    if args.names:
        names = [n for n in names if any(s in n for s in args.names)]

    if args.list:
        for n in names:
            print(f"{n:45s} {harness_policy(examples[n]['devices']):20s} {build_options(examples[n], 'cuda')}")
        return

    os.makedirs(os.path.join(args.out_dir, "runs"), exist_ok=True)
    results_path = os.path.join(args.out_dir, "results.json")
    results: dict[str, dict] = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    if not args.report_only:
        total = len(names) * len(configs)
        done = 0
        for name in names:
            for config in configs:
                done += 1
                key = f"{name}::{config}"
                if key in results and not args.force:
                    continue
                print(f"[{done}/{total}] {name} [{config}] ...", end=" ", flush=True)
                warm = None
                if not args.no_warmup:
                    warm = run_one(name, config, examples[name], args.out_dir, args.timeout, warmup=True)
                    print(f"warm-up {_status(summarize(warm))} {warm['wall']:.1f}s;", end=" ", flush=True)
                if warm is not None and warm["rc"] == "TIMEOUT":
                    result = warm  # the full run cannot finish either
                else:
                    result = run_one(name, config, examples[name], args.out_dir, args.timeout)
                if warm is not None:
                    ws = summarize(warm)
                    result["warmup"] = {
                        "ok": warm["ok"],
                        "wall": warm["wall"],
                        "init": ws["init"],
                        "first": ws.get("first"),
                    }
                results[key] = result
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=1)
                s = summarize(result)
                print(
                    f"{_status(s):10s} total {s['wall']:7.1f}s"
                    + (f"  {s['method']}: first {_fmt_ms(s['first'])} ms" if s.get("method") else "")
                    + (f", steady {_fmt_ms(s['steady'])} ms x{s['calls'] - 1}" if s.get("steady") is not None else ""),
                    flush=True,
                )

    report_path = os.path.join(args.out_dir, "report.md")
    write_report({k: v for k, v in results.items() if k.split("::")[1] in configs}, configs, report_path)
    print(f"\nResults: {results_path}\nReport:  {report_path}")


if __name__ == "__main__":
    main()
