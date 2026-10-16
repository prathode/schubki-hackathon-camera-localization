"""Camera intrinsics + extrinsics parsing and projection utilities.

Frame conventions (from the dataset README):
  * camera optical frames `constructor0/camera_*`: z forward, x right, y down.
  * `base_link`: x forward, y left, z up (REP 103).

tf_static provides base_link -> camera transforms. We build, per camera:
  * K (3x3 intrinsics), D (plumb_bob distortion),
  * T_bl_cam : 4x4 transform of a point expressed in the camera optical
    frame into base_link,
  * and its inverse T_cam_bl.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class Extrinsics:
    cam: str
    T_bl_cam: np.ndarray  # 4x4: point in camera optical frame -> base_link

    @property
    def T_cam_bl(self) -> np.ndarray:
        return np.linalg.inv(self.T_bl_cam)

    @property
    def cam_origin_bl(self) -> np.ndarray:
        """Camera centre in base_link coordinates (x fwd, y left, z up)."""
        return self.T_bl_cam[:3, 3]


@dataclass
class CameraCalib:
    cam: str
    width: int
    height: int
    K: np.ndarray            # 3x3
    D: np.ndarray            # (k1,k2,p1,p2,k3)
    distortion_model: str
    extrinsics: Optional[Extrinsics] = None

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """Undistort pixel points -> normalized camera rays (x,y,1).

        pts: (N,2) pixel coordinates. Returns (N,3) normalized directions in
        the optical frame (z forward).
        """
        import cv2
        if len(pts) == 0:
            return np.zeros((0, 3))
        p = pts.reshape(-1, 1, 2).astype(np.float64)
        und = cv2.undistortPoints(p, self.K, self.D)  # -> normalized (x,y)
        und = und.reshape(-1, 2)
        rays = np.concatenate([und, np.ones((len(und), 1))], axis=1)
        return rays


@dataclass
class Calibration:
    cams: Dict[str, CameraCalib]

    def __getitem__(self, cam: str) -> CameraCalib:
        return self.cams[cam]

    def __contains__(self, cam: str) -> bool:
        return cam in self.cams

    def keys(self):
        return self.cams.keys()


def load_calibration(mcap_path: str, cfg: dict) -> Calibration:
    """Read camera_info (intrinsics) + tf_static (extrinsics) only.

    Both are latched/static, so the first message of each is sufficient.
    """
    d = cfg["dataset"]
    cams: List[str] = list(d["cameras"])
    info_topics = {d["camera_info_topic"].format(cam=c): c for c in cams}
    base = d["base_link_frame"]

    calib: Dict[str, CameraCalib] = {}
    extr: Dict[str, Extrinsics] = {}

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        # --- intrinsics ---
        seen = set()
        for _, channel, _, msg in reader.iter_decoded_messages(
            topics=list(info_topics.keys())
        ):
            c = info_topics[channel.topic]
            if c in seen:
                continue
            seen.add(c)
            calib[c] = CameraCalib(
                cam=c,
                width=int(msg.width),
                height=int(msg.height),
                K=np.array(msg.k, dtype=np.float64).reshape(3, 3),
                D=np.array(msg.d, dtype=np.float64),
                distortion_model=str(msg.distortion_model),
            )
            if len(seen) == len(cams):
                break

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, _, msg in reader.iter_decoded_messages(topics=[d["tf_static_topic"]]):
            for t in msg.transforms:
                if t.header.frame_id != base:
                    continue
                child = t.child_frame_id  # e.g. constructor0/camera_fl
                cam_name = child.split("/")[-1].replace("camera_", "")
                if cam_name not in cams:
                    continue
                R = quat_to_R(
                    t.transform.rotation.x, t.transform.rotation.y,
                    t.transform.rotation.z, t.transform.rotation.w,
                )
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [
                    t.transform.translation.x,
                    t.transform.translation.y,
                    t.transform.translation.z,
                ]
                extr[cam_name] = Extrinsics(cam=cam_name, T_bl_cam=T)
            break  # latched: first message has all transforms

    for c, cc in calib.items():
        cc.extrinsics = extr.get(c)
    return Calibration(cams=calib)
