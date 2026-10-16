"""Unified feature-matcher interface + backends.

A matcher takes two BGR images (or precomputed features) and returns matched
pixel coordinates (pts_a, pts_b) in each image's ORIGINAL pixel coordinates.
This lets the metric relocalizer and the benchmark harness swap backends:

  * "aliked_lightglue" -- ALIKED extractor + LightGlue matcher (default GPU
    path). Supports precomputed per-image features, so a keyframe DB can store
    descriptors once and only extract the query at inference.
  * "loma" / "loma-b" / "loma-l" / "loma-g" / "loma-r" -- LoMa
    (davnords/LoMa, ECCV26). Dense/learned matcher; matches image pairs
    directly (no reusable sparse DB), so it is primarily a benchmark/accuracy
    reference and a drop-in for pairwise matching.

All backends are GPU-optional via perception/device.py (CPU fallback). All are
DECLARED EXTERNAL MODELS (pretrained):
  ALIKED BSD-3, LightGlue Apache-2.0, LoMa MIT (matcher inherits Apache-2.0).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class BaseMatcher:
    name = "base"
    supports_feature_cache = False

    def extract(self, img_bgr: np.ndarray):
        raise NotImplementedError

    def match_images(self, img_a_bgr, img_b_bgr) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def match_features(self, feat_a, feat_b) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class AlikedLightGlueMatcher(BaseMatcher):
    name = "aliked_lightglue"
    supports_feature_cache = True

    def __init__(self, cfg: dict):
        from .features import AlikedLightGlue
        self.backend = AlikedLightGlue(cfg)
        self.device = self.backend.device

    def extract(self, img_bgr):
        return self.backend.extract(img_bgr)

    def match_features(self, feat_a, feat_b):
        return self.backend.match(feat_a, feat_b)

    def match_images(self, img_a_bgr, img_b_bgr):
        fa = self.backend.extract(img_a_bgr)
        fb = self.backend.extract(img_b_bgr)
        return self.backend.match(fa, fb)


class LoMaMatcher(BaseMatcher):
    """LoMa pairwise matcher (davnords/LoMa).

    LoMa's public API matches by file path and returns matched keypoints. We
    wrap it to accept in-memory BGR images by writing to a temp file when a
    path is not available. Dense matchers do not expose a reusable sparse
    descriptor cache, so this backend is used for pairwise matching /
    benchmarking rather than a precomputed keyframe DB.
    """
    name = "loma"
    supports_feature_cache = False

    _VARIANTS = {
        "loma": "LoMaB", "loma-b": "LoMaB", "loma-b128": "LoMaB128",
        "loma-l": "LoMaL", "loma-g": "LoMaG", "loma-r": "LoMaR",
    }

    def __init__(self, cfg: dict, variant: str = "loma-b"):
        from .device import resolve_device
        import loma as _loma  # noqa
        self._loma = _loma
        self.name = variant
        cls_name = self._VARIANTS.get(variant.lower(), "LoMaB")
        backbone_cls = getattr(_loma, cls_name)
        self.device = resolve_device(cfg.get("runtime", {}).get("device", "auto"))
        self.model = _loma.LoMa(backbone_cls())
        # LoMa manages its own device internally; move if the API allows it.
        try:
            self.model = self.model.to(self.device)
        except Exception:
            pass

    def _match_paths(self, pa, pb):
        return self.model.match(pa, pb)

    def match_images(self, img_a_bgr, img_b_bgr):
        import os
        import tempfile
        import cv2
        da = tempfile.mkdtemp()
        pa = os.path.join(da, "a.png")
        pb = os.path.join(da, "b.png")
        cv2.imwrite(pa, img_a_bgr)
        cv2.imwrite(pb, img_b_bgr)
        kptsA, kptsB = self.model.match(pa, pb)
        kptsA = np.asarray(kptsA, np.float64)
        kptsB = np.asarray(kptsB, np.float64)
        return kptsA, kptsB


class SiftMatcher(BaseMatcher):
    """Classical SIFT extractor + ratio-test matcher (CPU, no GPU/weights).

    This is the exact feature used by the reference winning run (SIFT + PnP).
    It runs on CPU, so the full landmark + PnP pipeline is reproducible and
    measurable without a GPU. Descriptors are 128-D float32, matched with a
    FLANN/BF nearest-neighbour + Lowe ratio test.
    """
    name = "sift"
    supports_feature_cache = True

    def __init__(self, cfg: dict):
        import cv2
        fc = cfg.get("features", {})
        self.max_kp = int(fc.get("max_keypoints", 2000))
        self.ratio = float(fc.get("sift_ratio", 0.75))
        self.resize = int(fc.get("resize", 0))  # 0 = full resolution
        self.sift = cv2.SIFT_create(nfeatures=self.max_kp)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.device = "cpu"

    def extract(self, img_bgr):
        import cv2
        from .features import FrameFeatures
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        H, W = g.shape[:2]
        scale = 1.0
        if self.resize and max(H, W) > self.resize:
            scale = self.resize / max(H, W)
            g = cv2.resize(g, (int(W * scale), int(H * scale)))
        kps, desc = self.sift.detectAndCompute(g, None)
        if desc is None or len(kps) == 0:
            return FrameFeatures(np.zeros((0, 2)), np.zeros((0, 128), np.float32),
                                 np.array([1 / scale, 1 / scale], np.float32),
                                 (W, H))
        pts = np.array([k.pt for k in kps], np.float32) / scale
        return FrameFeatures(keypoints=pts, descriptors=desc.astype(np.float32),
                             scale=np.array([1 / scale, 1 / scale], np.float32),
                             size_wh=(W, H))

    def match_features(self, f0, f1):
        if len(f0.descriptors) == 0 or len(f1.descriptors) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
        knn = self.bf.knnMatch(f0.descriptors, f1.descriptors, k=2)
        good = [m for pair in knn if len(pair) == 2
                for m, n in [pair] if m.distance < self.ratio * n.distance]
        if not good:
            return np.zeros((0, 2)), np.zeros((0, 2))
        p0 = np.array([f0.keypoints[m.queryIdx] for m in good])
        p1 = np.array([f1.keypoints[m.trainIdx] for m in good])
        return p0, p1

    def match_images(self, img_a_bgr, img_b_bgr):
        return self.match_features(self.extract(img_a_bgr),
                                   self.extract(img_b_bgr))


def build_matcher(cfg: dict, name: Optional[str] = None) -> BaseMatcher:
    name = (name or cfg.get("metric_reloc", {}).get("matcher",
            "aliked_lightglue")).lower()
    if name == "sift":
        return SiftMatcher(cfg)
    if name.startswith("loma"):
        return LoMaMatcher(cfg, variant=name)
    return AlikedLightGlueMatcher(cfg)
