"""Sparse 3D landmark map + PnP relocalization.

Built offline from the training run (permitted for map-building): adjacent
keyframes from the same camera are matched, and their correspondences are
triangulated using the known training GT poses into a sparse set of 3D map
points, each tagged with a representative descriptor. At inference we match
the current frame's features against the landmark descriptors near the
predicted pose and solve PnP+RANSAC for an absolute metric pose -- the
strongest absolute fix available, with no accumulated VO drift.

This complements the essential-matrix relocalizer (metric_reloc.py): PnP is
preferred when a reliable landmark map exists; essential-matrix is the
fallback when it does not. Both are EKF-gated so a wrong fix is rejected.

All geometry uses calibrated intrinsics/extrinsics; poses are
base_link-in-map. The camera-from-base_link extrinsic converts between the
vehicle pose and the camera pose used by cv2.solvePnP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


@dataclass
class LandmarkMap:
    cam: str
    points3d: np.ndarray                 # (M,3) in map frame
    descriptors: np.ndarray              # (M,D) representative descriptor
    obs_xy: np.ndarray                   # (M,2) map xy of the landmark (for KD)
    _tree: Optional[object] = None

    def build_index(self):
        if cKDTree is not None and len(self.obs_xy):
            self._tree = cKDTree(self.obs_xy)
        return self

    def near(self, xy, radius):
        if self._tree is None:
            d = np.hypot(self.obs_xy[:, 0] - xy[0], self.obs_xy[:, 1] - xy[1])
            return np.nonzero(d <= radius)[0]
        return np.array(self._tree.query_ball_point(xy, radius), dtype=int)

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, points3d=self.points3d,
                            descriptors=self.descriptors.astype(np.float32),
                            obs_xy=self.obs_xy, cam=np.array([self.cam]))

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=True)
        return LandmarkMap(cam=str(z["cam"][0]), points3d=z["points3d"],
                           descriptors=z["descriptors"],
                           obs_xy=z["obs_xy"]).build_index()


def _T_map_cam(pose_bl, extr):
    """4x4 camera-optical-frame -> map, from base_link pose + extrinsic."""
    x, y, yaw = pose_bl
    c, s = np.cos(yaw), np.sin(yaw)
    T_map_bl = np.eye(4)
    T_map_bl[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T_map_bl[:3, 3] = [x, y, 0.0]
    return T_map_bl @ extr.T_bl_cam


def triangulate_pair(K, D, pa, pb, pose_a, pose_b, extr,
                     max_reproj=2.0, min_angle_deg=1.0):
    """Triangulate matched points from two keyframes with known poses.

    Returns (pts3d_map (N,3), mask of accepted) filtered by cheirality,
    triangulation angle and reprojection error.
    """
    if len(pa) < 8:
        return np.zeros((0, 3)), np.zeros(len(pa), bool)
    T_a = _T_map_cam(pose_a, extr)          # cam_a -> map
    T_b = _T_map_cam(pose_b, extr)
    # projection matrices map->cam = inv(T)
    Pa = np.linalg.inv(T_a)[:3, :]
    Pb = np.linalg.inv(T_b)[:3, :]
    # normalized coords
    na = cv2.undistortPoints(pa.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    nb = cv2.undistortPoints(pb.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    X = cv2.triangulatePoints(Pa, Pb, na.T, nb.T)   # 4xN in map
    X = (X[:3] / X[3]).T                              # (N,3)
    # cheirality: in front of both cameras
    def depth(T, pts):
        Tc = np.linalg.inv(T)
        pc = (Tc[:3, :3] @ pts.T).T + Tc[:3, 3]
        return pc[:, 2]
    za = depth(T_a, X)
    zb = depth(T_b, X)
    ok = (za > 0.5) & (zb > 0.5) & (za < 200) & (zb < 200)
    # triangulation angle
    ca = T_a[:3, 3]
    cb = T_b[:3, 3]
    v1 = X - ca
    v2 = X - cb
    cosang = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) *
                                 np.linalg.norm(v2, axis=1) + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    ok &= ang > min_angle_deg
    return X, ok


class PnPRelocalizer:
    def __init__(self, cfg: dict, maps: dict, calib, matcher):
        pc = cfg.get("pnp_reloc", {})
        self.cfg = cfg
        self.maps = maps                 # cam -> LandmarkMap
        self.calib = calib
        self.matcher = matcher
        self.cams = list(pc.get("cams", ["fl"]))
        self.search_radius = float(pc.get("search_radius_m", 25.0))
        self.min_inliers = int(pc.get("min_inliers", 20))
        self.r_pos = float(pc.get("r_pos", 2.0))
        self.r_yaw = float(pc.get("r_yaw", 0.02))
        self.gate_dist = float(pc.get("gate_dist_m", 15.0))

    def query(self, iset, predicted_pose):
        from .metric_reloc import MetricRelocResult
        if cv2 is None:
            return MetricRelocResult(ok=False)
        for cam in self.cams:
            lm = self.maps.get(cam)
            m = iset.get(cam)
            if lm is None or m is None or cam not in self.calib:
                continue
            idx = lm.near(predicted_pose[:2], self.search_radius)
            if len(idx) < self.min_inliers:
                continue
            cc = self.calib[cam]
            f_q = self.matcher.extract(m.decode())
            # match query descriptors to nearby landmark descriptors (NN + ratio)
            d2, i2 = _mutual_nn(f_q.descriptors, lm.descriptors[idx])
            if len(i2) < self.min_inliers:
                continue
            pts2d = f_q.keypoints[d2]
            pts3d = lm.points3d[idx[i2]]
            res = self._solve_pnp(pts2d, pts3d, cc, cam)
            if res is not None and res.ok:
                d = np.hypot(res.x - predicted_pose[0],
                             res.y - predicted_pose[1])
                if d <= self.gate_dist:
                    return res
        return MetricRelocResult(ok=False)

    def _solve_pnp(self, pts2d, pts3d, cc, cam):
        from .metric_reloc import MetricRelocResult
        if len(pts2d) < self.min_inliers:
            return None
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d.astype(np.float64), pts2d.astype(np.float64), cc.K, cc.D,
            reprojectionError=3.0, confidence=0.999, iterationsCount=200,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < self.min_inliers:
            return None
        R, _ = cv2.Rodrigues(rvec)         # map->cam rotation
        T_cam_map = np.eye(4)
        T_cam_map[:3, :3] = R
        T_cam_map[:3, 3] = tvec.ravel()
        T_map_cam = np.linalg.inv(T_cam_map)
        extr = cc.extrinsics
        T_map_bl = T_map_cam @ np.linalg.inv(extr.T_bl_cam)
        x = float(T_map_bl[0, 3])
        y = float(T_map_bl[1, 3])
        yaw = float(np.arctan2(T_map_bl[1, 0], T_map_bl[0, 0]))
        n = int(len(inliers))
        r_pos = self.r_pos * max(1.0, 25.0 / max(n, 1))
        return MetricRelocResult(ok=True, x=x, y=y, yaw=yaw, n_inliers=n,
                                 confidence=n / max(len(pts2d), 1),
                                 r_pos=r_pos, r_yaw=self.r_yaw)


def _mutual_nn(desc_q, desc_m, ratio=0.9):
    """Nearest-neighbour match query->map descriptors with Lowe ratio test.

    Returns (query_idx, map_idx) arrays. Cosine similarity (descriptors are
    not necessarily L2-normalized, so normalize here).
    """
    if len(desc_q) == 0 or len(desc_m) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    q = desc_q / (np.linalg.norm(desc_q, axis=1, keepdims=True) + 1e-9)
    m = desc_m / (np.linalg.norm(desc_m, axis=1, keepdims=True) + 1e-9)
    sim = q @ m.T                                  # (Nq, Nm)
    # top-2 per query
    if sim.shape[1] < 2:
        best = np.argmax(sim, axis=1)
        return np.arange(len(q)), best
    part = np.argpartition(-sim, 1, axis=1)[:, :2]
    s0 = sim[np.arange(len(q)), part[:, 0]]
    s1 = sim[np.arange(len(q)), part[:, 1]]
    first = np.where(s0 >= s1, part[:, 0], part[:, 1])
    second = np.where(s0 >= s1, part[:, 1], part[:, 0])
    best = sim[np.arange(len(q)), first]
    secondbest = sim[np.arange(len(q)), second]
    # ratio test on (1 - sim) distances
    keep = (1 - best) < ratio * (1 - secondbest + 1e-9)
    qi = np.nonzero(keep)[0]
    return qi, first[qi]
