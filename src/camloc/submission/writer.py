"""Strict CSV writer for Option A submissions: timestamp_ns,x,y,yaw."""
from __future__ import annotations

import csv
import os
from typing import Iterable

from ..localization.pipeline import PoseEstimate


def write_submission(path: str, estimates: Iterable[PoseEstimate]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ns", "x", "y", "yaw"])
        for e in estimates:
            w.writerow([e.stamp_ns, f"{e.x:.4f}", f"{e.y:.4f}", f"{e.yaw:.6f}"])
            n += 1
    return n
