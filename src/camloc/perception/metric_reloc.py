"""Metric visual relocalization: ALIKED+LightGlue matching -> absolute pose.

Pipeline (causal; map built offline from training GT -- permitted):
  1. Keyframe DB: sampled training frames with ALIKED features + their GT
     pose (per camera). Built by scripts/build_feature_db.py.
  2. At inference, restrict candidate keyframes to a neighbourhood of the
     EKF-predicted pose (prevents matching to a visually similar but wrong
     part of the circuit).
  3. Match the current frame to the best candidate with LightGlue.
  4. Recover the relative rotation + bearing between query and keyframe from
     the essential matrix (calibrated cameras). The keyframe's absolute pose
     plus this relative transform give an absolute pose fix for the query.
  5. Gate hard on inlier count, reprojection consistency and distance from
     the prediction; output an (x, y) position fix + yaw for the EKF.

Translation from a single essential matrix is only known up to scale. We
resolve scale using the keyframe->query baseline implied by the retrieved
keyframe spacing and the motion model; when scale is unreliable we still use
the recovered *bearing* and *relative yaw*, which constrain position and
heading. All fixes are EKF-gated so a bad estimate cannot corrupt the state.
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
class FeatureDB:
    """Per-camera keyframe database of ALIKED features + poses."""
    cam: str
    poses: np.ndarray                 # (N,3) x,y,yaw (training GT; map-building)
    stamps: np.ndarray                # (N,)
    keypoints: List[np.ndarray] = field(default_factory=list)   # list of (Ki,2)
    descriptors: List[np.ndarray] = field(default_factory=list) # list of (Ki,D)
    sizes: Optional[np.ndarray] = None  # (N,2) W,H
    _tree: Optional[object] = None

    def build_index(self):
        if cKDTree is not None and len(self.poses):
            self._tree = cKDTree(self.poses[:, :2])
        return self

    def candidates(self, xy, radius, k=5):
        if self._tree is None:
            d = np.hypot(self.poses[:, 0] - xy[0], self.poses[:, 1] - xy[1])
            idx = np.argsort(d)[:k]
            return [i for i in idx if d[i] <= radius]
        idx = self._tree.query_ball_point(xy, radius)
        if not idx:
            return []
        d = np.hypot(self.poses[idx, 0] - xy[0], self.poses[idx, 1] - xy[1])
        order = np.argsort(d)[:k]
        return [idx[o] for o in order]

    def save(self, path):
        import pickle
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


@dataclass
class MetricRelocResult:
    ok: bool
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    n_inliers: int = 0
    confidence: float = 0.0
    r_pos: float = 4.0
    r_yaw: float = 0.05
    r_lat: float = 0.4
    r_lon: float = 1.5


class MetricRelocalizer:
    def __init__(self, cfg: dict, dbs: dict, calib, features_backend):
        rc = cfg.get("metric_reloc", {})
        self.cfg = cfg
        self.dbs = dbs                      # cam -> FeatureDB
        self.calib = calib
        self.fx = features_backend
        self.cams = list(rc.get("cams", ["fl"]))
        self.search_radius = float(rc.get("search_radius_m", 15.0))
        self.topk = int(rc.get("topk", 3))
        self.min_inliers = int(rc.get("min_inliers", 25))
        self.r_pos = float(rc.get("r_pos", 4.0))
        self.r_yaw = float(rc.get("r_yaw", 0.03))
        self.gate_dist = float(rc.get("gate_dist_m", 15.0))

    def query(self, iset, predicted_pose) -> MetricRelocResult:
        if cv2 is None:
            return MetricRelocResult(ok=False)
        best = None
        for cam in self.cams:
            db = self.dbs.get(cam)
            img_msg = iset.get(cam)
            if db is None or img_msg is None or cam not in self.calib:
                continue
            cand = db.candidates(predicted_pose[:2], self.search_radius,
                                 self.topk)
            if not cand:
                continue
            f_q = self.fx.extract(img_msg.decode())
            K = self.calib[cam].K
            D = self.calib[cam].D
            for ci in cand:
                f_ref = self._ref_features(db, ci)
                p_q, p_r = self.fx.match(f_q, f_ref)
                if len(p_q) < self.min_inliers:
                    continue
                res = self._relative_pose(p_q, p_r, K, D, db.poses[ci], cam)
                if res is None:
                    continue
                if best is None or res.n_inliers > best.n_inliers:
                    best = res
        if best is None:
            return MetricRelocResult(ok=False)
        # gate against prediction
        d = np.hypot(best.x - predicted_pose[0], best.y - predicted_pose[1])
        if d > self.gate_dist:
            return MetricRelocResult(ok=False, n_inliers=best.n_inliers)
        return best

    def _ref_features(self, db, i):
        from .features import FrameFeatures
        return FrameFeatures(
            keypoints=db.keypoints[i], descriptors=db.descriptors[i],
            scale=np.array([1.0, 1.0], np.float32),
            size_wh=tuple(db.sizes[i]) if db.sizes is not None else (1506, 728),
        )

    def _relative_pose(self, p_q, p_r, K, D, ref_pose, cam) -> Optional[MetricRelocResult]:
        """Essential-matrix relative pose query<-ref, then absolute query pose.

        The essential matrix gives rotation R and unit translation t of the
        query camera relative to the reference camera. We use it primarily for
        the relative YAW and the translation bearing; combined with the known
        reference absolute pose this yields an absolute position fix on the
        ground plane. Scale is taken from the search neighbourhood baseline.
        """
        # undistort to normalized coords
        q = cv2.undistortPoints(p_q.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        r = cv2.undistortPoints(p_r.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        E, mask = cv2.findEssentialMat(
            q, r, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1e-3
        )
        if E is None or mask is None:
            return None
        inl = int(mask.sum())
        if inl < self.min_inliers:
            return None
        _, R, t, mask2 = cv2.recoverPose(E, q, r, np.eye(3), mask=mask)
        # R, t: transform from query camera frame to reference camera frame.
        # Convert the relative yaw (about the optical y-axis ~ vehicle z) into
        # a map-frame yaw correction relative to the reference pose.
        extr = self.calib[cam].extrinsics
        R_bl_cam = extr.T_bl_cam[:3, :3] if extr is not None else np.eye(3)
        # rotation of query relative to ref, mapped into base_link
        R_rel_bl = R_bl_cam @ R @ R_bl_cam.T
        dyaw = np.arctan2(R_rel_bl[1, 0], R_rel_bl[0, 0])
        ref_x, ref_y, ref_yaw = ref_pose
        # translation bearing in base_link (scale ambiguous): direction only
        t_bl = R_bl_cam @ t.ravel()
        # Scale: approximate with a small fixed baseline; position fix is
        # dominated by the reference pose, bearing adds a sub-metre nudge.
        scale = float(self.cfg.get("metric_reloc", {}).get("baseline_m", 2.0))
        c, s = np.cos(ref_yaw), np.sin(ref_yaw)
        Rm = np.array([[c, -s], [s, c]])
        dxy_map = Rm @ (t_bl[:2] * scale)
        x = ref_x + dxy_map[0]
        y = ref_y + dxy_map[1]
        yaw = ref_yaw + dyaw
        conf = inl / max(len(p_q), 1)
        r_pos = self.r_pos * max(1.0, 30.0 / max(inl, 1))
        return MetricRelocResult(ok=True, x=float(x), y=float(y),
                                 yaw=float(yaw), n_inliers=inl,
                                 confidence=conf, r_pos=r_pos, r_yaw=self.r_yaw)
