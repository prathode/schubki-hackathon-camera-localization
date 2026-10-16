"""Causal localization pipeline: assemble image sets, predict, update, emit.

Flow per the challenge's Method 4 (causal filter combining motion + map fix):

  1. Consume the time-ordered stream of state samples and image sets.
  2. On each state sample: integrate motion to advance a running pose and
     feed the increment to the EKF prediction step.
  3. On each image set: run the (optional) track-border matcher to produce a
     lateral+yaw correction and apply it as a gated EKF update, then emit the
     current pose for that image set's timestamp.

Every image set yields exactly one pose (10 Hz output, full coverage). If the
border match fails or is disabled, the emitted pose is the pure motion
prediction -- the filter never skips a frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from ..data.mcap_source import McapSource, StateSample
from ..data.image_set import ImageSetAssembler, ImageSet
from .motion import MotionIntegrator
from .ekf import PoseEKF, wrap


@dataclass
class PoseEstimate:
    stamp_ns: int
    x: float
    y: float
    yaw: float
    source: str = "predict"   # predict | border | reloc
    n_border_pts: int = 0


class Localizer:
    def __init__(self, cfg: dict, border_matcher=None, relocalizer=None,
                 tangent_yaw=None, metric_reloc=None):
        self.cfg = cfg
        self.motion = MotionIntegrator(cfg)
        self.ekf = PoseEKF(cfg)
        self.border = border_matcher
        self.reloc = relocalizer
        self.metric_reloc = metric_reloc
        self.tangent = tangent_yaw
        self._last_state_ns: Optional[int] = None
        self._set_count = 0
        self.log_every = int(cfg.get("runtime", {}).get("log_every", 0))
        # zero-velocity yaw-rate bias calibration state
        f = cfg.get("filter", {})
        self._auto_bias = bool(f.get("auto_yaw_bias", True))
        self._bias_v_thresh = float(f.get("bias_v_thresh", 0.3))
        self._bias_sum = 0.0
        self._bias_n = 0

    def initialize(self, x: float, y: float, yaw: float):
        self.ekf.initialize(x, y, yaw)

    def run(self, source: McapSource, start_ns=None, end_ns=None,
            on_log=None) -> Iterator[PoseEstimate]:
        asm = ImageSetAssembler(self.cfg)
        n_emitted = 0
        last_state: Optional[StateSample] = None

        for kind, msg in asm.stream(source, start_ns, end_ns):
            if kind == "state":
                self._on_state(msg)
                last_state = msg
            else:  # image set
                est = self._on_image_set(msg, last_state)
                n_emitted += 1
                if self.log_every and on_log and n_emitted % self.log_every == 0:
                    on_log(n_emitted, est)
                yield est

    def _on_state(self, s: StateSample):
        if not self.ekf._init:
            return
        # ---- zero-velocity yaw-rate bias calibration (causal, allowed) ----
        # The run starts ~5 s before first motion. While stationary, any
        # non-zero yaw rate is pure gyro bias; average it and feed it to the
        # motion model. This is standard ZUPT calibration, not ground truth.
        if self._auto_bias and abs(s.get("v_mps", 1.0)) < self._bias_v_thresh:
            self._bias_sum += s.get("wz_radps", 0.0)
            self._bias_n += 1
            if self._bias_n >= 20:
                self.motion.yaw_rate_bias = self._bias_sum / self._bias_n
        if self._last_state_ns is None:
            self._last_state_ns = s.stamp_ns
            return
        dt = (s.stamp_ns - self._last_state_ns) / 1e9
        self._last_state_ns = s.stamp_ns
        if dt <= 0 or dt > 0.5:   # guard against gaps
            return
        pose = self.ekf.pose
        new = self.motion.integrate(pose, s, dt)
        self.ekf.predict(new[0] - pose[0], new[1] - pose[1],
                         wrap(new[2] - pose[2]))

    def _on_image_set(self, iset: ImageSet,
                      last_state: Optional[StateSample]) -> PoseEstimate:
        self._set_count += 1
        source = "predict"
        n_pts = 0
        decoded = None  # lazily decoded images shared by reloc + border

        # ---- map-tangent yaw correction: anchors slow yaw drift (gated) ----
        # Only safe when position is still trustworthy: if the filter is
        # uncertain (large position covariance), the nearest-border tangent
        # may lock onto the wrong track segment and diverge. Gate on the
        # position covariance trace.
        if self.tangent is not None and self.cfg.get("tangent_yaw", {}).get(
            "enabled", True
        ):
            pos_cov = self.ekf.P[0, 0] + self.ekf.P[1, 1]
            max_cov = float(self.cfg["tangent_yaw"].get("max_pos_cov", 4.0))
            if pos_cov <= max_cov:
                tr = self.tangent.reference_yaw(self.ekf.pose)
                if tr.ok:
                    self.ekf.update_yaw_ref(tr.yaw_ref, self.tangent.r_yaw)
                    source = "tangent"

        # ---- metric relocalization (ALIKED+LightGlue): absolute fix (gated) ----
        if (self.metric_reloc is not None
                and self.cfg.get("metric_reloc", {}).get("enabled")):
            stride = int(self.cfg["metric_reloc"].get("apply_every", 1))
            if self._set_count % stride == 0:
                mr = self.metric_reloc.query(iset, self.ekf.pose)
                if mr.ok and mr.n_inliers >= self.cfg["metric_reloc"]["min_inliers"]:
                    self.ekf.update_position(mr.x, mr.y, mr.r_pos)
                    self.ekf.update_yaw_ref(mr.yaw, mr.r_yaw)
                    source = "metric_reloc"
                    n_pts = mr.n_inliers

        # ---- relocalization: coarse absolute position fix (gated) ----
        if self.reloc is not None and self.cfg.get("reloc", {}).get("enabled"):
            stride = int(self.cfg["reloc"].get("apply_every", 1))
            if self._set_count % stride == 0:
                decoded = {c: m.decode() for c, m in iset.images.items()}
                rr = self.reloc.query(decoded, self.ekf.pose)
                if rr.ok:
                    if self.ekf.update_position(rr.x, rr.y, rr.r_pos):
                        source = "reloc"

        # ---- track-border matching: lateral + yaw correction (gated) ----
        if self.border is not None and self.cfg["border_match"]["enabled"]:
            result = self.border.match(iset, self.ekf.pose)
            if result is not None and result.get("ok"):
                applied = self.ekf.update_lateral_yaw(
                    lat_resid=result["lat"],
                    yaw_resid=result["yaw"],
                    heading=self.ekf.pose[2],
                    r_lat=result.get("r_lat"),
                    r_yaw=result.get("r_yaw"),
                )
                if applied:
                    source = "border" if source == "predict" else "reloc+border"
                n_pts = result.get("n_pts", 0)

        p = self.ekf.pose
        return PoseEstimate(
            stamp_ns=iset.stamp_ns, x=float(p[0]), y=float(p[1]),
            yaw=float(p[2]), source=source, n_border_pts=n_pts,
        )
