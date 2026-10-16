"""Proof that the PnP relocalization path can reach sub-0.5 m accuracy.

This simulates the inference-time PnP solve with REALISTIC feature-matching
noise (pixel noise, landmark/triangulation noise, gross-outlier fraction) and
a representative forward-camera intrinsic. It is the evidence that the
GPU relocalization path is *capable* of the sub-0.5 m target the challenge
asks for -- the geometry is not the limiting factor; feature-match quality is.

The actual night-image match quality (and hence the final RMSE) must be
measured on a GPU with ALIKED+LightGlue; this test only bounds the geometry.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

cv2 = pytest.importorskip("cv2")


def _fl_intrinsic():
    # representative narrow front camera (close to the dataset fl camera)
    K = np.array([[2601.0, 0, 775.0], [0, 2590.0, 521.0], [0, 0, 1.0]])
    D = np.array([-0.39, 0.12, -0.007, -0.001, 0.0])
    W, H = 1506, 728
    # extrinsic base_link->camera (optical: z fwd, x right, y down)
    R_bl_cam = np.column_stack([[0.0, -1.0, 0.0],
                                [0.0, 0.0, -1.0],
                                [1.0, 0.0, 0.0]])
    T = np.eye(4)
    T[:3, :3] = R_bl_cam
    T[:3, 3] = [0.94, 0.20, 0.34]
    return K, D, W, H, T


def _T_map_cam(pose, T_bl_cam):
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    T_map_bl = np.eye(4)
    T_map_bl[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T_map_bl[:3, 3] = [x, y, 0.0]
    return T_map_bl @ T_bl_cam


def _trial(rng, K, D, W, H, T_bl_cam, px_noise, lm_noise, out_frac, reproj):
    true = np.array([rng.uniform(-50, 50), rng.uniform(-50, 50),
                     rng.uniform(-np.pi, np.pi)])
    N = 400
    fwd = np.array([np.cos(true[2]), np.sin(true[2])])
    lat = np.array([-np.sin(true[2]), np.cos(true[2])])
    d = rng.uniform(6, 40, N)
    off = rng.uniform(-10, 10, N)
    hz = rng.uniform(-0.3, 3.0, N)
    pts3 = np.column_stack([true[:2] + fwd * d[:, None] + lat * off[:, None], hz])
    Tc = np.linalg.inv(_T_map_cam(true, T_bl_cam))
    P = (Tc[:3, :3] @ pts3.T).T + Tc[:3, 3]
    front = P[:, 2] > 0.5
    if front.sum() < 25:
        return None
    proj = cv2.projectPoints(P[front].reshape(-1, 1, 3), np.zeros(3),
                             np.zeros(3), K, D)[0].reshape(-1, 2)
    inf = (proj[:, 0] >= 0) & (proj[:, 0] < W) & (proj[:, 1] >= 0) & (proj[:, 1] < H)
    pf, uf = pts3[front][inf], proj[inf]
    if len(pf) < 25:
        return None
    uvn = uf + rng.normal(0, px_noise, uf.shape)
    p3n = pf + rng.normal(0, lm_noise, pf.shape)
    nout = int(out_frac * len(uf))
    if nout:
        oi = rng.choice(len(uf), nout, replace=False)
        uvn[oi] += rng.uniform(-200, 200, (nout, 2))
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        p3n.astype(np.float64), uvn.astype(np.float64), K, D,
        reprojectionError=reproj, confidence=0.9999, iterationsCount=500,
        flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < 12:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(p3n[inl.ravel()].astype(np.float64),
                                      uvn[inl.ravel()].astype(np.float64),
                                      K, D, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    Tcm = np.eye(4); Tcm[:3, :3] = R; Tcm[:3, 3] = tvec.ravel()
    Tmap = np.linalg.inv(Tcm) @ np.linalg.inv(T_bl_cam)
    est = np.array([Tmap[0, 3], Tmap[1, 3]])
    return float(np.linalg.norm(est - true[:2]))


def test_pnp_reaches_submetre_with_good_features():
    """Good GPU-grade features (0.5 px, 5 cm landmarks, 15% outliers)."""
    K, D, W, H, T = _fl_intrinsic()
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(400):
        e = _trial(rng, K, D, W, H, T, px_noise=0.5, lm_noise=0.05,
                   out_frac=0.15, reproj=2.0)
        if e is not None:
            errs.append(e)
    errs = np.array(errs)
    assert len(errs) > 100
    # median well under 0.5 m, and the large majority within 0.5 m
    assert np.median(errs) < 0.5
    assert np.mean(errs < 0.5) > 0.9
