"""Track-border map matching -> lateral + yaw correction for the EKF.

Given the predicted pose and an image set, for the reliable front/ground
cameras we:
  1. detect near-field track-edge pixels,
  2. project them to the ground plane in the map frame using the predicted
     pose (flat-track assumption),
  3. associate each projected point with its nearest map-border point,
  4. solve a small rigid correction (lateral shift + yaw) that best aligns
     the projected cloud to the map, robustly (trimmed least squares),
  5. gate the result on inlier count and residual so bad matches are
     rejected.

The correction is returned relative to the predicted pose:
  * lat  : how far (and which way) to move perpendicular to heading,
  * yaw  : heading correction,
and consumed by PoseEKF.update_lateral_yaw.

Longitudinal correction is intentionally NOT applied from border matching on
straights (borders are parallel -> along-track unobservable). It is applied
only where the geometry supports it (corners), detected via the spread of
border-normal directions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .border_detect import detect_border_pixels
from ..calibration.projection import ground_project_pixels


@dataclass
class MatchResult:
    ok: bool
    lat: float = 0.0
    yaw: float = 0.0
    n_pts: int = 0
    residual: float = 0.0
    r_lat: Optional[float] = None
    r_yaw: Optional[float] = None
    lon: float = 0.0
    lon_observable: bool = False

    def as_dict(self):
        return dict(ok=self.ok, lat=self.lat, yaw=self.yaw, n_pts=self.n_pts,
                    residual=self.residual, r_lat=self.r_lat, r_yaw=self.r_yaw,
                    lon=self.lon, lon_observable=self.lon_observable)


class BorderMatcher:
    def __init__(self, cfg: dict, calib, track, segmenter=None):
        self.cfg = cfg
        bm = cfg["border_match"]
        self.cams = list(bm["cams"])
        self.max_range = float(bm["max_range_m"])
        self.min_points = int(bm["min_points"])
        self.icp_iters = int(bm.get("icp_iters", 8))
        self.gate = float(cfg["filter"].get("gate_residual_m", 1.5))
        self.calib = calib
        self.track = track
        self.r_lateral = float(cfg["filter"]["r_lateral"])
        self.r_yaw = float(cfg["filter"]["r_yaw"])
        # optional learned drivable-surface segmenter -> clean border pixels.
        self.segmenter = segmenter
        self._seg_boundary = None
        if segmenter is not None:
            from .segmentation import RoadBoundaryFromMask
            self._seg_boundary = RoadBoundaryFromMask()

    def _detect(self, img):
        if self.segmenter is not None:
            from .segmentation import detect_border_pixels_seg
            return detect_border_pixels_seg(img, self.segmenter,
                                            self._seg_boundary)
        return detect_border_pixels(img)

    def _collect_points(self, iset, pose) -> np.ndarray:
        clouds = []
        for cam in self.cams:
            m = iset.get(cam)
            if m is None or cam not in self.calib:
                continue
            cc = self.calib[cam]
            if cc.extrinsics is None:
                continue
            img = m.decode(color=True)
            px = self._detect(img)
            if len(px) == 0:
                continue
            xy, _ = ground_project_pixels(px, cc, pose, z_ground_bl=0.0)
            if len(xy) == 0:
                continue
            d = np.hypot(xy[:, 0] - pose[0], xy[:, 1] - pose[1])
            keep = d <= self.max_range
            if keep.any():
                clouds.append(xy[keep])
        if not clouds:
            return np.zeros((0, 2))
        return np.concatenate(clouds, axis=0)

    def match(self, iset, pose) -> Optional[dict]:
        pts = self._collect_points(iset, pose)
        if len(pts) < self.min_points:
            return MatchResult(ok=False, n_pts=len(pts)).as_dict()

        heading = pose[2]
        # unit vectors: along-heading and lateral(left)
        along = np.array([np.cos(heading), np.sin(heading)])
        lateral = np.array([-np.sin(heading), np.cos(heading)])

        cur_lat = 0.0
        cur_yaw = 0.0
        residual = np.inf
        n_in = 0
        for _ in range(self.icp_iters):
            # apply current correction to the cloud about the car position
            R = _rot(cur_yaw)
            rel = pts - pose[:2]
            moved = (R @ rel.T).T + pose[:2] + lateral * cur_lat
            d, nn = self.track.nearest(moved)
            # robust inlier set via MAD
            med = np.median(d)
            mad = np.median(np.abs(d - med)) + 1e-6
            inliers = d < (med + 2.5 * mad)
            inliers &= d < self.gate * 2
            if inliers.sum() < self.min_points // 2:
                break
            P = moved[inliers]
            Q = nn[inliers]
            # residual vectors
            res = Q - P
            # lateral component of residual -> lateral shift
            dlat = float(np.median(res @ lateral))
            # yaw: fit small rotation about car that reduces residual
            rel_in = P - pose[:2]
            cross = rel_in[:, 0] * res[:, 1] - rel_in[:, 1] * res[:, 0]
            dot = (rel_in * rel_in).sum()
            dyaw = float(np.sum(cross) / (dot + 1e-6))
            dyaw = float(np.clip(dyaw, -0.05, 0.05))
            cur_lat += dlat
            cur_yaw += dyaw
            residual = float(np.sqrt(np.mean(d[inliers] ** 2)))
            n_in = int(inliers.sum())
            if abs(dlat) < 1e-3 and abs(dyaw) < 1e-4:
                break

        if n_in < self.min_points // 2 or residual > self.gate:
            return MatchResult(ok=False, n_pts=len(pts), residual=residual).as_dict()

        # along-track observability: spread of border normal directions.
        d_final, nn_final = self.track.nearest(pts)
        lon_obs = self._lon_observable(pts, nn_final)

        # scale measurement noise by residual quality
        r_lat = self.r_lateral * max(1.0, residual / 0.3)
        r_yaw = self.r_yaw * max(1.0, residual / 0.3)
        return MatchResult(
            ok=True, lat=cur_lat, yaw=cur_yaw, n_pts=n_in,
            residual=residual, r_lat=r_lat, r_yaw=r_yaw,
            lon_observable=lon_obs,
        ).as_dict()

    def _lon_observable(self, pts, nn) -> bool:
        """Corners make along-track observable: check border tangent spread."""
        if len(nn) < 10:
            return False
        # approximate local tangent direction spread of matched map points
        order = np.argsort(nn[:, 0])
        q = nn[order]
        diffs = np.diff(q, axis=0)
        ang = np.arctan2(diffs[:, 1], diffs[:, 0])
        spread = np.ptp(np.unwrap(ang)) if len(ang) else 0.0
        return spread > 0.5  # radians; enough curvature to pin along-track


def _rot(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])
