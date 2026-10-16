"""Night-robust track-border pixel detection for ground projection.

The naive approach (raw Canny over the whole frame) fails: it fires on
barriers, grandstands and lights -- vertical structure that violates the
flat-ground assumption and projects to garbage when back-projected onto the
road plane.

This detector instead isolates the *boundary of the drivable asphalt* in the
near field:

  1. Work only in a near-field image band (lower part of the frame), where
     the flat-track assumption holds and projected range is small.
  2. Estimate the asphalt (road) region by its characteristic low, smooth
     intensity relative to brighter run-off / kerbs / painted lines.
  3. For each image column, find the transition row between road and
     off-road (the left/right track edge) using the column intensity
     profile. These transition pixels are the border candidates.

Only the nearest, well-supported transitions per column are returned, which
keeps the projected point cloud tight around the true track edge.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _preprocess(gray: np.ndarray) -> np.ndarray:
    # CLAHE lifts the dark night asphalt without blowing out lights.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    return cv2.GaussianBlur(g, (5, 5), 0)


def detect_border_pixels(
    img: np.ndarray,
    near_band: tuple[float, float] = (0.55, 0.98),
    grad_thresh: float = 18.0,
    max_per_col: int = 2,
    col_step: int = 4,
) -> np.ndarray:
    """Return (N,2) candidate track-edge pixels (u,v) in the near field.

    near_band : (top, bottom) fractions of image height defining the band.
    grad_thresh : min vertical-ish gradient magnitude for an edge.
    max_per_col : keep at most this many strongest edges per scanned column
                  (the nearest road<->offroad transitions).
    col_step : scan every Nth column (speed).
    """
    if cv2 is None:
        raise ImportError("opencv is required")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    h, w = gray.shape
    g = _preprocess(gray)

    top = int(near_band[0] * h)
    bot = int(near_band[1] * h)
    band = g[top:bot, :]

    # Horizontal gradient magnitude: track edge is a vertical-ish boundary
    # between road and off-road, i.e. strong horizontal intensity change.
    gx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)

    out = []
    bh = band.shape[0]
    for u in range(0, w, col_step):
        col = mag[:, u]
        # candidate rows with strong gradient
        idx = np.nonzero(col > grad_thresh)[0]
        if len(idx) == 0:
            continue
        # prefer the lowest (nearest) strong edges
        order = idx[np.argsort(-col[idx])][:max_per_col]
        for v_local in order:
            out.append((u, v_local + top))
    if not out:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(out, dtype=float)
