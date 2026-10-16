"""Map-tangent yaw correction (drift anchor, no pixel detection needed).

The dominant error of the motion-only estimator is slow yaw drift. On a race
track the vehicle heading follows the local track direction closely except at
corner apices. Given an approximate position (from the motion estimate), the
track-border map yields an unambiguous local *tangent* direction. We use that
tangent as an absolute heading reference and feed a gated yaw correction to
the EKF.

This is map-based (uses only track_bounds + the current estimate), causal, and
generalizes to unseen runs. It is gated hard so it only nudges the heading
where the geometry supports it:
  * apply only where the local track is nearly straight (low curvature), since
    on straights the racing line is ~parallel to the borders;
  * apply only when the tangent agrees with the current heading within a
    window (reject corners / gross position errors);
  * small per-update gain so noise averages out over many frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class TangentResult:
    ok: bool
    yaw_ref: float = 0.0
    curvature: float = 0.0
    agree_deg: float = 0.0


class TangentYawCorrector:
    def __init__(self, cfg: dict, track):
        tc = cfg.get("tangent_yaw", {})
        self.enabled = bool(tc.get("enabled", True))
        self.win = int(tc.get("tangent_window", 8))         # polyline samples
        self.max_curv = float(tc.get("max_curvature", 0.015))  # 1/m
        self.agree_deg = float(tc.get("agree_deg", 6.0))
        self.r_yaw = float(tc.get("r_yaw", 0.03))
        self.max_dist = float(tc.get("max_border_dist_m", 25.0))
        # build per-border KD-trees on the raw (ordered) polylines
        self._borders = []
        for name in ("track_left", "track_right"):
            pts = track.borders.get(name)
            if pts is not None and len(pts) > self.win + 2:
                tree = cKDTree(pts) if cKDTree is not None else None
                self._borders.append((pts, tree))

    def reference_yaw(self, pose) -> TangentResult:
        if not self.enabled or not self._borders:
            return TangentResult(ok=False)
        p = np.asarray(pose[:2])
        heading = pose[2]
        tangents = []
        curvs = []
        dists = []
        for pts, tree in self._borders:
            if tree is None:
                d = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
                i = int(np.argmin(d))
                dist = float(d[i])
            else:
                dist, i = tree.query(p)
            if dist > self.max_dist:
                continue
            a = max(i - self.win, 0)
            b = min(i + self.win, len(pts) - 1)
            seg = pts[b] - pts[a]
            norm = np.linalg.norm(seg)
            if norm < 1e-3:
                continue
            t = seg / norm
            ty = np.arctan2(t[1], t[0])
            # resolve 180deg ambiguity against current heading
            if abs(_wrap(heading - ty)) > np.pi / 2:
                ty = _wrap(ty + np.pi)
            # local curvature from three samples
            c = max(i - self.win, 0)
            d_ = min(i + self.win, len(pts) - 1)
            mid = i
            v1 = pts[mid] - pts[c]
            v2 = pts[d_] - pts[mid]
            ang = _wrap(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))
            seglen = np.linalg.norm(v1) + np.linalg.norm(v2)
            curv = abs(ang) / (seglen + 1e-6)
            tangents.append(ty)
            curvs.append(curv)
            dists.append(dist)
        if not tangents:
            return TangentResult(ok=False)
        # combine left/right tangents (circular mean)
        c = np.mean(np.cos(tangents))
        s = np.mean(np.sin(tangents))
        yaw_ref = np.arctan2(s, c)
        curv = float(np.mean(curvs))
        agree = abs(np.degrees(_wrap(heading - yaw_ref)))
        if curv > self.max_curv or agree > self.agree_deg:
            return TangentResult(ok=False, yaw_ref=yaw_ref, curvature=curv,
                                 agree_deg=agree)
        return TangentResult(ok=True, yaw_ref=float(yaw_ref), curvature=curv,
                             agree_deg=agree)
