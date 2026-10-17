"""Triangulation + PnP geometry tests with synthetic known geometry.

These validate the map-building and PnP math without any learned features or
GPU, so a regression in the geometry is caught in CI.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

cv2 = pytest.importorskip("cv2")

from camloc.data.calibration import CameraCalib, Extrinsics
from camloc.perception.landmarks import triangulate_pair, _mutual_nn


def _make_calib():
    K = np.array([[1000.0, 0, 750.0], [0, 1000.0, 360.0], [0, 0, 1.0]])
    D = np.zeros(5)
    # camera optical frame == base_link axes rotated: z fwd, x right, y down.
    # Build extrinsic base_link->camera for a forward camera at height 0.3.
    # base_link: x fwd, y left, z up. camera optical: x right(=-y_bl),
    # y down(=-z_bl), z fwd(=x_bl).
    # T_bl_cam columns = camera optical axes expressed in base_link:
    #   cam x (right) -> bl -y = (0,-1,0)
    #   cam y (down)  -> bl -z = (0,0,-1)
    #   cam z (fwd)   -> bl +x = (1,0,0)
    R_bl_cam = np.column_stack([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ])
    T = np.eye(4)
    T[:3, :3] = R_bl_cam
    T[:3, 3] = [1.0, 0.0, 0.3]
    cc = CameraCalib(cam="fl", width=1506, height=728, K=K, D=D,
                     distortion_model="plumb_bob",
                     extrinsics=Extrinsics(cam="fl", T_bl_cam=T))
    return cc


def _project(cc, pose_bl, pts_map):
    from camloc.perception.landmarks import _T_map_cam
    T_map_cam = _T_map_cam(pose_bl, cc.extrinsics)
    Tc = np.linalg.inv(T_map_cam)
    P = (Tc[:3, :3] @ pts_map.T).T + Tc[:3, 3]
    uv = (cc.K @ (P / P[:, 2:3]).T).T[:, :2]
    return uv, P[:, 2]


def test_triangulation_recovers_points():
    cc = _make_calib()
    # two poses along +x (vehicle moving forward), looking at points ahead
    pose_a = np.array([0.0, 0.0, 0.0])
    pose_b = np.array([3.0, 0.0, 0.0])
    # map points ahead of the car, on/near the ground
    pts = np.array([[20.0, 2.0, 0.2], [25.0, -3.0, 0.1],
                    [30.0, 1.0, 0.3], [18.0, -1.0, 0.0],
                    [22.0, 4.0, 0.25], [28.0, -2.0, 0.15],
                    [24.0, 0.5, 0.2], [26.0, 3.0, 0.05],
                    [19.0, 1.5, 0.1], [31.0, -1.5, 0.2]])
    ua, za = _project(cc, pose_a, pts)
    ub, zb = _project(cc, pose_b, pts)
    assert (za > 0).all() and (zb > 0).all()
    X, ok = triangulate_pair(cc.K, cc.D, ua, ub, pose_a, pose_b,
                             cc.extrinsics, min_angle_deg=0.05)
    assert ok.sum() >= 4
    err = np.linalg.norm(X[ok] - pts[ok], axis=1)
    assert err.mean() < 0.5     # sub-0.5 m triangulation on clean synthetic data


def test_mutual_nn_matches_identical():
    d = np.random.default_rng(0).standard_normal((20, 32)).astype(np.float32)
    qi, mi = _mutual_nn(d, d, ratio=0.95)
    # identical descriptors: each should match its own index
    assert len(qi) > 0
    assert (qi == mi).mean() > 0.9
