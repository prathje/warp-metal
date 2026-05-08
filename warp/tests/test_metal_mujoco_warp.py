# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CPU vs Metal physics integration via mujoco_warp.

These tests run ``mujoco_warp.step()`` on small MuJoCo models on both
``cpu`` and ``metal:0`` and assert agreement between the resulting
dynamics state arrays. They protect the Metal backend's high-level
codegen against regressions that only surface when many kernels
interact (smooth dynamics, constraint solver, contact pipeline).

Skipped when:
  - Not on Apple Silicon (no Metal device)
  - MLX not installed
  - mujoco_warp not on the import path
  - Stock ``warp-lang`` (no ``enable_metal`` config flag)

Each test runs in a fresh subprocess because ``warp.config.enable_metal``
must be set BEFORE ``wp.init()`` runs, and ``wp.init()`` is process-
global.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import unittest


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _has_mlx() -> bool:
    try:
        import mlx.core as mx  # noqa: PLC0415

        return mx.metal.is_available()
    except Exception:
        return False


def _has_metal_warp() -> bool:
    try:
        import warp.config  # noqa: PLC0415

        return hasattr(warp.config, "enable_metal")
    except Exception:
        return False


def _has_mujoco_warp() -> bool:
    return importlib.util.find_spec("mujoco_warp") is not None


def _run_subprocess(test_case: unittest.TestCase, snippet: str, timeout: int = 240) -> None:
    """Run ``snippet`` with ``enable_metal=True``; fail if exit != 0."""
    code = (
        "import warp as wp\n"
        "wp.config.enable_metal = True\n"
        "wp.init()\n"
    ) + snippet
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="warp_mjw_test_") as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    test_case.assertEqual(
        result.returncode,
        0,
        f"subprocess exited with {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
    )


_COMPARE_FIELDS = ("qpos", "qvel", "qacc", "act", "time", "xpos", "xquat")


_COMPARE_SNIPPET = textwrap.dedent(
    """
    import mujoco
    import numpy as np
    import warp as wp
    import mujoco_warp as mjw

    # The simple-vs-blocked Cholesky threshold: bump it past nv so models
    # with nv up to 64 use the (now correct) single-tile path on Metal,
    # avoiding the cooperative blocked-Cholesky kernels which still need
    # follow-up parallelism work to be efficient.
    if _BUMP_CHOLESKY_THRESHOLD:
        import mujoco_warp._src.solver as _mjw_solver
        _mjw_solver._BLOCK_CHOLESKY_DIM = 64

    mjm = mujoco.MjModel.from_xml_string(_XML)
    # Force sparse Jacobian — the dense path uses cooperative tiles of
    # ``vec_t<6, float>`` (spatial vectors) which the Metal backend
    # doesn't lower yet (see project task list).
    mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetData(mjm, mjd)
    mujoco.mj_forward(mjm, mjd)

    with wp.ScopedDevice('cpu'):
        m_cpu = mjw.put_model(mjm)
        d_cpu = mjw.put_data(mjm, mjd)
        for _ in range(_NSTEPS):
            mjw.step(m_cpu, d_cpu)
        cpu_state = {f: getattr(d_cpu, f).numpy().copy() for f in _COMPARE_FIELDS if hasattr(d_cpu, f)}

    with wp.ScopedDevice('metal:0'):
        m_metal = mjw.put_model(mjm)
        d_metal = mjw.put_data(mjm, mjd)
        for _ in range(_NSTEPS):
            mjw.step(m_metal, d_metal)
        metal_state = {f: getattr(d_metal, f).numpy().copy() for f in _COMPARE_FIELDS if hasattr(d_metal, f)}

    for f in cpu_state:
        a = cpu_state[f]
        b = metal_state[f]
        np.testing.assert_allclose(
            a, b,
            atol=_ATOL, rtol=_RTOL,
            err_msg=f'{f} differs between cpu and metal:0 (shape={a.shape})',
        )
    print('OK')
    """
)


@unittest.skipUnless(_is_apple_silicon(), "Metal backend requires macOS / Apple Silicon")
@unittest.skipUnless(_has_mlx(), "MLX is not installed (required for Metal backend)")
@unittest.skipUnless(_has_metal_warp(), "Warp build does not expose enable_metal — using stock warp-lang?")
@unittest.skipUnless(_has_mujoco_warp(), "mujoco_warp is not installed; skipping integration tests")
class TestMetalMujocoWarp(unittest.TestCase):
    """End-to-end mujoco_warp.step() agreement between CPU and Metal."""

    def _run(
        self,
        xml: str,
        nsteps: int = 1,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        bump_cholesky_threshold: bool = False,
    ) -> None:
        snippet = textwrap.dedent(
            f"""
            _XML = {xml!r}
            _NSTEPS = {nsteps}
            _ATOL = {atol}
            _RTOL = {rtol}
            _COMPARE_FIELDS = {_COMPARE_FIELDS!r}
            _BUMP_CHOLESKY_THRESHOLD = {bump_cholesky_threshold}
            """
        ) + _COMPARE_SNIPPET
        _run_subprocess(self, snippet)

    def test_freejoint_sphere_drops_under_gravity(self):
        # Simplest model: one body, free joint, sphere geom. Tests
        # gravity, integration, kinematics; no contact.
        xml = """
        <mujoco>
          <worldbody>
            <body name="b" pos="0 0 1">
              <freejoint/>
              <geom size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        self._run(xml, nsteps=5)

    def test_pendulum_swings(self):
        # Single hinge pendulum. Exercises the constraint solver
        # minimally and a non-trivial kinematic chain.
        xml = """
        <mujoco>
          <worldbody>
            <body name="link" pos="0 0 1">
              <joint type="hinge" axis="0 1 0"/>
              <geom type="capsule" size="0.05" fromto="0 0 0  0 0 -0.5"/>
            </body>
          </worldbody>
        </mujoco>
        """
        self._run(xml, nsteps=10)

    def test_pendula_chain_exercises_cooperative_cholesky(self):
        # 30-link hinge chain — nv=30, well above the cooperative
        # ``_COOP_CHOL_MIN_N=24`` threshold. ``update_gradient_cholesky``
        # for nv=30 routes through the SIMD-cooperative variant of
        # ``tile_cholesky``. The 32-thread threadgroup, threadgroup-
        # memory scratch, and ``threadgroup_position_in_grid`` worldid
        # remap all participate. This is the primary end-to-end check
        # that the cooperative codegen produces the same result as
        # the single-thread path.
        N = 30
        # Build a nested chain XML: each link contains the next.
        opens = "".join(
            f'<body name="l{i}" pos="0 0 {-0.0 if i == 1 else -0.2}">'
            f'<joint type="hinge" axis="0 1 0"/>'
            f'<geom type="capsule" size="0.02" fromto="0 0 0  0 0 -0.2"/>'
            for i in range(1, N + 1)
        )
        closes = "</body>" * N
        xml = f"""
        <mujoco>
          <worldbody>
            <body name="root" pos="0 0 1">
              {opens}{closes}
            </body>
          </worldbody>
        </mujoco>
        """
        # 5 steps with the standard atol — float32 Cholesky drift
        # accumulates faintly across the iterative-solver linesearch
        # but stays at ~1e-4 over this horizon.
        self._run(xml, nsteps=5)

    def test_cartpole_with_contact_geoms(self):
        # mjlab's cartpole model. Exercises the full contact pipeline
        # because there are static (worldbody-attached) geoms — floor
        # + 2 rails — alongside the cart and pole bodies. Regression
        # for the early-return seed bug: ``_geom_local_to_global``
        # returns early for static geoms, so their ``geom_xpos`` /
        # ``geom_xmat`` slots have to be seeded from the user's
        # ``put_data`` values rather than left zero/stale.
        xml = """
        <mujoco>
          <option gravity="0 0 -9.81"/>
          <worldbody>
            <geom name="floor" type="plane" size="2 2 0.1"/>
            <geom name="rail1" type="capsule" size="0.02"
                  fromto="-1 0.07 1   1 0.07 1"/>
            <geom name="rail2" type="capsule" size="0.02"
                  fromto="-1 -0.07 1   1 -0.07 1"/>
            <body name="cart" pos="0 0 1">
              <joint name="slider" type="slide" axis="1 0 0"
                     range="-1 1" damping="0.1"/>
              <geom name="cart_geom" type="box" size="0.1 0.05 0.05"/>
              <body name="pole" euler="180 0 0">
                <joint name="hinge" type="hinge" axis="0 1 0"
                       damping="0.05"/>
                <geom name="pole_geom" type="capsule" size="0.02"
                      fromto="0 0 0  0 0 0.5"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <motor joint="slider" gear="1" ctrlrange="-1 1"/>
          </actuator>
        </mujoco>
        """
        self._run(xml, nsteps=5)

    def test_two_bodies_with_actuator(self):
        # Two-link arm with position actuators. End-to-end actuation.
        xml = """
        <mujoco>
          <worldbody>
            <body name="link1" pos="0 0 1">
              <joint name="j1" type="hinge" axis="0 1 0"/>
              <geom type="capsule" size="0.05" fromto="0 0 0  0 0 -0.4"/>
              <body name="link2" pos="0 0 -0.4">
                <joint name="j2" type="hinge" axis="0 1 0"/>
                <geom type="capsule" size="0.05" fromto="0 0 0  0 0 -0.4"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <position joint="j1" kp="50" ctrlrange="-1 1"/>
            <position joint="j2" kp="50" ctrlrange="-1 1"/>
          </actuator>
        </mujoco>
        """
        self._run(xml, nsteps=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
