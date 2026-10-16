"""MCAP ingestion for the Constructor Labs camera-localization dataset.

This module provides a *causal*, streaming reader over the ROS 2 MCAP file.
Messages are yielded in log-time order so that inference can consume them
exactly as they would arrive on the real car.

CAUSALITY / GROUND-TRUTH ISOLATION
----------------------------------
The inference path must only read:
  * camera compressed_image messages,
  * camera_info (intrinsics),
  * tf_static (extrinsics),
  * track_bounds (map prior),
  * the ALLOWED motion fields of state_estimation (see StateSample).

The absolute pose in state_estimation (x_m/y_m/z_m/roll/pitch/yaw) and the
sn_map_state block are GROUND TRUTH and are *not* exposed by StateSample.
The /tf topic (map -> base_link) is ground truth and is only read by the
evaluation-only helpers, never by the inference reader.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

try:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "mcap and mcap-ros2-support are required: pip install mcap mcap-ros2-support"
    ) from exc


def stamp_to_ns(stamp) -> int:
    """Convert a ROS2 builtin_interfaces/Time to int nanoseconds."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


# ---- ALLOWED motion fields only. NO absolute pose, NO sn_map_state. ----
# This is the single source of truth for the inference allowlist.
ALLOWED_STATE_FIELDS: Sequence[str] = (
    "v_mps",
    "v_raw_mps",
    "vx_mps",
    "vy_mps",
    "wz_radps",
    "yaw_vel_rad",
    "beta_rad",
    "ax_mps2",
    "ay_mps2",
    "delta_wheel_rad",
    "kappa_radpm",
)

# Fields that are ground truth and must never leak into inference.
PROHIBITED_STATE_FIELDS: Sequence[str] = (
    "x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad",
    "pos_accuracy", "sn_map_state",
)


@dataclass
class ImageMsg:
    """A single decoded camera frame (JPEG bytes kept lazily)."""
    cam: str
    stamp_ns: int
    jpeg: bytes

    def decode(self, color: bool = True):
        import cv2
        flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
        return cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), flag)


@dataclass
class StateSample:
    """Allowed vehicle-motion signals at a single timestamp.

    Only fields on ALLOWED_STATE_FIELDS are populated. Absolute pose and
    track s,n are intentionally absent.
    """
    stamp_ns: int
    fields: Dict[str, float] = field(default_factory=dict)

    def get(self, name: str, default: float = 0.0) -> float:
        return self.fields.get(name, default)


class McapSource:
    """Streaming, causal reader over the dataset MCAP.

    Use `iter_messages()` to get a single time-ordered stream of tagged
    events for the inference pipeline. Static/latched data (intrinsics,
    extrinsics, track bounds) can be loaded up-front with the loaders in
    `calibration.py` / `track.py`, which read only their own topics.
    """

    def __init__(self, path: str, cfg: dict):
        self.path = path
        self.cfg = cfg
        d = cfg["dataset"]
        self.cameras: List[str] = list(d["cameras"])
        self.image_topics = {
            d["camera_image_topic"].format(cam=c): c for c in self.cameras
        }
        self.state_topic = d["state_estimation_topic"]
        self._allow = list(cfg.get("state_allowlist", ALLOWED_STATE_FIELDS))

    # ---- inference stream: images + allowed state only ----
    def iter_messages(
        self,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        include_state: bool = True,
    ) -> Iterator[tuple]:
        """Yield ('image', ImageMsg) and ('state', StateSample) in time order.

        Only inference-permitted topics are opened here.
        """
        topics = list(self.image_topics.keys())
        if include_state:
            topics.append(self.state_topic)
        with open(self.path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, ros_msg in reader.iter_decoded_messages(
                topics=topics, start_time=start_ns, end_time=end_ns
            ):
                topic = channel.topic
                if topic in self.image_topics:
                    cam = self.image_topics[topic]
                    yield (
                        "image",
                        ImageMsg(
                            cam=cam,
                            stamp_ns=stamp_to_ns(ros_msg.header.stamp),
                            jpeg=bytes(ros_msg.data),
                        ),
                    )
                elif topic == self.state_topic:
                    s = StateSample(stamp_ns=stamp_to_ns(ros_msg.header.stamp))
                    for name in self._allow:
                        if hasattr(ros_msg, name):
                            s.fields[name] = float(getattr(ros_msg, name))
                    yield ("state", s)

    def time_bounds(self) -> tuple[int, int]:
        with open(self.path, "rb") as f:
            reader = make_reader(f)
            stats = reader.get_summary().statistics
            return int(stats.message_start_time), int(stats.message_end_time)

    def first_state_pose_eval_only(self) -> Optional[dict]:
        """EVALUATION / INITIALIZATION ONLY.

        Returns the very first ground-truth pose (x, y, yaw) from /tf, which
        the challenge explicitly permits as the initial pose. This reads the
        ground-truth topic and must ONLY be called to seed the initial state,
        never inside the per-frame inference loop.
        """
        tf_topic = self.cfg["dataset"]["tf_topic"]
        map_frame = self.cfg["dataset"]["map_frame"]
        base = self.cfg["dataset"]["base_link_frame"]
        with open(self.path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, _, _, msg in reader.iter_decoded_messages(topics=[tf_topic]):
                for t in msg.transforms:
                    if t.header.frame_id == map_frame and t.child_frame_id == base:
                        q = t.transform.rotation
                        p = t.transform.translation
                        yaw = np.arctan2(
                            2 * (q.w * q.z + q.x * q.y),
                            1 - 2 * (q.y * q.y + q.z * q.z),
                        )
                        return {
                            "stamp_ns": stamp_to_ns(t.header.stamp),
                            "x": float(p.x),
                            "y": float(p.y),
                            "yaw": float(yaw),
                        }
        return None
