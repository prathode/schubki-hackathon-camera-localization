"""Causal EKF over the 2D pose state [x, y, yaw] in the map frame.

Prediction uses the allowed body velocity + yaw rate (odometry model).
Updates come from the track-border matcher as a (lateral, yaw) correction
expressed relative to the current predicted pose, plus optional absolute
(x, y) corrections. All updates are gated by innovation magnitude so a bad
border match cannot corrupt the state.

The EKF never consumes future data: prediction and update are both driven by
messages with stamps <= the current output time.
"""
from __future__ import annotations

import numpy as np


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class PoseEKF:
    def __init__(self, cfg: dict):
        f = cfg["filter"]
        self.q_pos = float(f["q_pos"])
        self.q_yaw = float(f["q_yaw"])
        self.r_lateral = float(f["r_lateral"])
        self.r_yaw = float(f["r_yaw"])
        self.gate = float(f.get("gate_residual_m", 1.5))
        self.x = np.zeros(3)                 # [x, y, yaw]
        self.P = np.diag([1.0, 1.0, 0.05])   # initial covariance
        self._init = False

    def initialize(self, x: float, y: float, yaw: float,
                   p_pos: float = 0.3, p_yaw: float = 0.02):
        self.x = np.array([x, y, yaw], dtype=float)
        self.P = np.diag([p_pos ** 2, p_pos ** 2, p_yaw ** 2])
        self._init = True

    @property
    def pose(self) -> np.ndarray:
        return self.x.copy()

    def predict(self, dx: float, dy: float, dyaw: float):
        """Propagate with an odometry increment in the map frame.

        dx, dy, dyaw are the pose change produced by MotionIntegrator over dt.
        The Jacobian w.r.t. state is identity for this additive model; process
        noise scales with the magnitude of motion.
        """
        self.x[0] += dx
        self.x[1] += dy
        self.x[2] = wrap(self.x[2] + dyaw)
        step = np.hypot(dx, dy)
        Q = np.diag([
            (self.q_pos * max(step, 1e-3)) ** 2 + 1e-6,
            (self.q_pos * max(step, 1e-3)) ** 2 + 1e-6,
            (self.q_yaw * max(abs(dyaw), 1e-3) + self.q_yaw) ** 2,
        ])
        self.P = self.P + Q

    def update_lateral_yaw(self, lat_resid: float, yaw_resid: float,
                           heading: float, r_lat: float = None,
                           r_yaw: float = None) -> bool:
        """Correct lateral position (perpendicular to `heading`) and yaw.

        lat_resid: measured lateral offset of the true pose relative to the
                   current estimate, in metres (positive = left).
        yaw_resid: measured yaw correction, radians.
        Returns True if the update was applied (passed the gate).
        """
        if abs(lat_resid) > self.gate:
            return False
        r_lat = self.r_lateral if r_lat is None else r_lat
        r_yaw = self.r_yaw if r_yaw is None else r_yaw
        # Measurement: lateral = -sin(h)*x + cos(h)*y (relative), yaw.
        h = heading
        H = np.array([
            [-np.sin(h), np.cos(h), 0.0],
            [0.0, 0.0, 1.0],
        ])
        R = np.diag([r_lat ** 2, r_yaw ** 2])
        z = np.array([lat_resid, yaw_resid])      # innovation directly
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ z
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P
        return True

    def update_yaw_ref(self, yaw_ref: float, r_yaw: float) -> bool:
        """Absolute heading correction toward yaw_ref (gated by caller)."""
        innov = wrap(yaw_ref - self.x[2])
        H = np.array([[0.0, 0.0, 1.0]])
        R = np.array([[r_yaw ** 2]])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K * innov).ravel()
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P
        return True

    def update_position(self, x_meas: float, y_meas: float,
                        r_pos: float) -> bool:
        """Absolute (x,y) correction, e.g. from relocalization."""
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        R = np.diag([r_pos ** 2, r_pos ** 2])
        z = np.array([x_meas - self.x[0], y_meas - self.x[1]])
        if np.hypot(*z) > 5.0:  # hard outlier gate
            return False
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ z
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P
        return True
