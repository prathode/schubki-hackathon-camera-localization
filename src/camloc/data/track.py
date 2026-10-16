"""Track-border map prior.

track_bounds is a MarkerArray of LINE_STRIPs in the `map` frame:
  track_left, track_right, pitlane_left, pitlane_right.

We load them into dense polylines and build a KD-tree over a resampled set
of border points for fast nearest-neighbour / chamfer matching against
projected camera border detections.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


def _resample_polyline(pts: np.ndarray, step: float = 0.5) -> np.ndarray:
    """Resample a polyline to roughly uniform spacing `step` metres."""
    if len(pts) < 2:
        return pts
    seg = np.diff(pts, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0], np.cumsum(seglen)])
    total = cum[-1]
    if total <= 0:
        return pts
    n = max(int(total / step), 2)
    s = np.linspace(0, total, n)
    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    return np.stack([x, y], axis=1)


@dataclass
class TrackMap:
    borders: Dict[str, np.ndarray]           # name -> (N,2) in map frame
    include_pit: bool = True
    _tree: Optional[object] = field(default=None, repr=False)
    _pts: Optional[np.ndarray] = field(default=None, repr=False)

    def build_index(self, step: float = 0.5, include_pit: bool = True):
        chunks = []
        for name, pts in self.borders.items():
            if not include_pit and name.startswith("pitlane"):
                continue
            chunks.append(_resample_polyline(pts, step))
        self._pts = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 2))
        if cKDTree is not None and len(self._pts):
            self._tree = cKDTree(self._pts)
        return self

    def nearest(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest border point for each query point (map frame).

        Returns (distances, nearest_points).
        """
        if self._tree is None:
            raise RuntimeError("call build_index() first")
        d, idx = self._tree.query(query)
        return d, self._pts[idx]

    @property
    def all_points(self) -> np.ndarray:
        return self._pts if self._pts is not None else np.zeros((0, 2))


def load_track_bounds(mcap_path: str, cfg: dict) -> TrackMap:
    topic = cfg["dataset"]["track_bounds_topic"]
    borders: Dict[str, np.ndarray] = {}
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, _, msg in reader.iter_decoded_messages(topics=[topic]):
            for mk in msg.markers:
                if len(mk.points) == 0:
                    continue
                borders[mk.ns] = np.array(
                    [[p.x, p.y] for p in mk.points], dtype=np.float64
                )
            break  # latched
    return TrackMap(borders=borders)
