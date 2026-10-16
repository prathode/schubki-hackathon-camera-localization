"""Geometry / filter unit tests (no dataset required)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from camloc.calibration.projection import pose_to_T_map_bl
from camloc.localization.ekf import PoseEKF, wrap
from camloc.localization.motion import MotionIntegrator
from camloc.data.mcap_source import StateSample


def test_wrap():
    assert abs(wrap(3 * np.pi)) - np.pi < 1e-9
    assert abs(wrap(-np.pi / 2) + np.pi / 2) < 1e-9


def test_pose_transform_roundtrip():
    pose = np.array([10.0, -5.0, 0.7])
    T = pose_to_T_map_bl(pose)
    # a point at base_link origin maps to (x,y)
    p = T @ np.array([0, 0, 0, 1.0])
    assert np.allclose(p[:2], pose[:2])
    # a point 1 m forward in base_link
    pf = T @ np.array([1, 0, 0, 1.0])
    assert np.allclose(pf[:2], pose[:2] + [np.cos(0.7), np.sin(0.7)], atol=1e-9)


def test_motion_straight_line():
    cfg = {"filter": {"speed_scale": 1.0}}
    mi = MotionIntegrator(cfg)
    s = StateSample(stamp_ns=0, fields={"vx_mps": 10.0, "vy_mps": 0.0,
                                        "wz_radps": 0.0})
    pose = np.array([0.0, 0.0, 0.0])
    out = mi.integrate(pose, s, dt=1.0)
    assert np.allclose(out, [10.0, 0.0, 0.0], atol=1e-6)


def test_motion_yaw_rate():
    cfg = {"filter": {"speed_scale": 1.0}}
    mi = MotionIntegrator(cfg)
    s = StateSample(stamp_ns=0, fields={"vx_mps": 0.0, "vy_mps": 0.0,
                                        "wz_radps": 1.0})
    out = mi.integrate(np.array([0.0, 0.0, 0.0]), s, dt=0.5)
    assert abs(out[2] - 0.5) < 1e-9


def test_ekf_position_update_pulls_toward_measurement():
    ekf = PoseEKF({"filter": {"q_pos": 0.05, "q_yaw": 0.002, "r_lateral": 0.25,
                              "r_yaw": 0.02, "gate_residual_m": 1.5}})
    ekf.initialize(0, 0, 0, p_pos=1.0, p_yaw=0.1)
    ekf.predict(0, 0, 0)
    before = ekf.pose.copy()
    ekf.update_position(1.0, 0.0, r_pos=0.3)
    assert ekf.pose[0] > before[0]          # moved toward the measurement
    assert ekf.pose[0] < 1.0                # but not all the way (filtered)


def test_ekf_yaw_ref_update():
    ekf = PoseEKF({"filter": {"q_pos": 0.05, "q_yaw": 0.002, "r_lateral": 0.25,
                              "r_yaw": 0.02, "gate_residual_m": 1.5}})
    ekf.initialize(0, 0, 0.1, p_pos=1.0, p_yaw=0.1)
    ekf.update_yaw_ref(0.0, r_yaw=0.02)
    assert 0.0 <= ekf.pose[2] < 0.1          # pulled toward 0
