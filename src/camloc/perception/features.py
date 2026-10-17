"""Learned local-feature extraction + matching (ALIKED + LightGlue).

DECLARED EXTERNAL MODELS (pretrained, frozen, GPU-optional):
  * ALIKED (Shiaoming/ALIKED, BSD-3-Clause) -- keypoint + descriptor network.
  * LightGlue (cvg/LightGlue, Apache-2.0) -- learned feature matcher.
We deliberately use the ALIKED pairing (not SuperPoint) to avoid SuperPoint's
restrictive licence.

This backend is used by the metric relocalizer. It is GPU-optional: on a
Pascal-class card (e.g. P5200) FlashAttention is unavailable, so LightGlue's
automatic fallback is relied upon and mixed precision is left off. It also
runs on CPU for correctness checks (slowly).

Install: `pip install lightglue @ git+https://github.com/cvg/LightGlue.git`
(or clone and `pip install -e .`). Weights download on first use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FrameFeatures:
    """Extracted features for one image, in CPU numpy for serialization."""
    keypoints: np.ndarray        # (N,2) pixel coords (x,y) at the resized res
    descriptors: np.ndarray      # (N,D) float32
    scale: np.ndarray            # (2,) x,y scale from resized -> original px
    size_wh: tuple               # original (W,H)


def _lazy_import():
    import torch
    from lightglue import LightGlue, ALIKED
    return torch, LightGlue, ALIKED


class AlikedLightGlue:
    def __init__(self, cfg: dict):
        fc = cfg.get("features", {})
        self.max_kp = int(fc.get("max_keypoints", 1024))
        self.resize = int(fc.get("resize", 1024))
        self.min_conf = float(fc.get("filter_threshold", 0.1))
        requested = cfg.get("runtime", {}).get("device", "auto")
        torch, LightGlue, ALIKED = _lazy_import()
        self.torch = torch
        from .device import to_device
        # Route every model through to_device so a GPU problem degrades to CPU
        # instead of crashing. The matcher follows whatever device the
        # extractor actually ended up on (keep both consistent).
        extractor = ALIKED(max_num_keypoints=self.max_kp).eval()
        self.extractor, self.device = to_device(extractor, requested)
        # flash auto-detected (off on Pascal); mp off for Pascal safety.
        matcher = LightGlue(features="aliked", filter_threshold=self.min_conf,
                            mp=False).eval()
        self.matcher, self.device = to_device(matcher, self.device)

    # ---- extraction ----
    def _to_tensor(self, img_bgr: np.ndarray):
        import cv2
        torch = self.torch
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        scale = 1.0
        if self.resize and max(H, W) > self.resize:
            scale = self.resize / max(H, W)
            rgb = cv2.resize(rgb, (int(round(W * scale)), int(round(H * scale))))
        t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        return t.to(self.device), (W, H), scale

    def extract(self, img_bgr: np.ndarray) -> FrameFeatures:
        torch = self.torch
        t, (W, H), scale = self._to_tensor(img_bgr)
        with torch.no_grad():
            feats = self.extractor.extract(t[None])  # batched
        kp = feats["keypoints"][0].cpu().numpy()          # (N,2) in resized px
        desc = feats["descriptors"][0].cpu().numpy()      # (N,D)
        inv = 1.0 / scale
        return FrameFeatures(
            keypoints=kp * inv,                            # back to original px
            descriptors=desc.astype(np.float32),
            scale=np.array([inv, inv], np.float32),
            size_wh=(W, H),
        )

    # ---- matching ----
    def match(self, f0: FrameFeatures, f1: FrameFeatures):
        """Return (pts0, pts1) matched pixel coords in each image (original px)."""
        torch = self.torch
        from lightglue.utils import rbd

        def pack(f):
            kp = torch.from_numpy((f.keypoints / f.scale)).float()  # resized px
            d = torch.from_numpy(f.descriptors).float()
            return {
                "keypoints": kp[None].to(self.device),
                "descriptors": d[None].to(self.device),
                "image_size": torch.tensor(
                    [[f.size_wh[0] * f.scale[0], f.size_wh[1] * f.scale[1]]]
                ).float().to(self.device),
            }

        with torch.no_grad():
            out = self.matcher({"image0": pack(f0), "image1": pack(f1)})
        out = rbd(out)
        m = out["matches"].cpu().numpy()                 # (K,2)
        if len(m) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
        p0 = f0.keypoints[m[:, 0]]
        p1 = f1.keypoints[m[:, 1]]
        return p0, p1
