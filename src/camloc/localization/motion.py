"""Causal motion integration from ALLOWED vehicle-state fields.

We dead-reckon the pose using body-frame velocity and yaw rate, which are
permitted motion signals (not ground-truth pose). The bicycle/odometry
update between two state samples at times t0 -> t1 (dt = t1 - t0) is:

    x  += (vx*cos(yaw) - vy*sin(yaw)) * dt
    y  += (vx*sin(yaw) + vy*cos(yaw)) * dt
    yaw+= wz * dt

where (vx, vy) is the body-frame velocity (x forward, y left) and wz the
yaw rate. If vy is unavailable we fall back to speed v_mps with sideslip
beta: vx = v*cos(beta), vy = v*sin(beta).

This is the prediction model of the EKF; it is also usable standalone as a
pure dead-reckoning baseline.
"""
from __future__ import annotations

import numpy as np

from ..data.mcap_source import StateSample


class MotionIntegrator:
    def __init__(self, cfg: dict):
        f = cfg.get("filter", {})
        self.speed_scale = float(f.get("speed_scale", 1.0))
        # Constant yaw-rate (gyro) bias in rad/s, a sensor calibration
        # constant estimated on the training run and applied forward. This is
        # NOT ground truth -- it is a fixed scalar like an extrinsic offset.
        self.yaw_rate_bias = float(f.get("yaw_rate_bias", 0.0))

    def body_velocity(self, s: StateSample) -> tuple[float, float, float]:
        """Return (vx, vy, wz) in the body frame from allowed fields.

        Use wz_radps as the yaw rate. NOTE: yaw_vel_rad in this dataset is
        NOT a usable yaw rate (it integrates to absurd values) and must not
        be used. A small constant yaw-rate bias (a sensor calibration
        constant, not ground truth) is subtracted if configured.
        """
        wz = s.get("wz_radps", 0.0) - self.yaw_rate_bias
        if "vx_mps" in s.fields:
            vx = s.get("vx_mps")
            vy = s.get("vy_mps", 0.0)
        else:
            v = s.get("v_mps", s.get("v_raw_mps", 0.0))
            beta = s.get("beta_rad", 0.0)
            vx = v * np.cos(beta)
            vy = v * np.sin(beta)
        return vx * self.speed_scale, vy * self.speed_scale, wz

    def integrate(
        self, pose: np.ndarray, s: StateSample, dt: float
    ) -> np.ndarray:
        """Advance pose=[x,y,yaw] by dt seconds using state sample s."""
        if dt <= 0:
            return pose
        vx, vy, wz = self.body_velocity(s)
        x, y, yaw = pose
        # midpoint heading for a slightly better arc integration
        yaw_mid = yaw + 0.5 * wz * dt
        cx, sx = np.cos(yaw_mid), np.sin(yaw_mid)
        x = x + (vx * cx - vy * sx) * dt
        y = y + (vx * sx + vy * cx) * dt
        yaw = yaw + wz * dt
        return np.array([x, y, yaw])
