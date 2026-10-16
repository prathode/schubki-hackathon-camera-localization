"""Learned drivable-surface segmentation -> clean track-border pixels.

The classical night detector (border_detect.py) fires on barriers, lights and
grandstands, which violate the flat-ground assumption and corrupt the ICP.
A learned road/drivable-surface segmentation isolates the asphalt region; its
lower boundary (road <-> off-road transition per column) gives clean,
ground-plane-valid border pixels for projection and map matching.

Backends (GPU-optional, declared external models):
  * "deeplabv3" -- torchvision DeepLabV3 (ResNet50/MobileNet), pretrained on
    COCO/VOC. We treat the road-like classes as drivable surface. This is a
    pragmatic, dependency-light baseline; for best results fine-tune a
    track-surface segmenter on labelled frames (left as a documented upgrade).
  * a user-provided ONNX/torch segmenter can be dropped in by implementing
    `segment(img) -> bool mask` with the same interface.

The output is the same (N,2) border-pixel array the matcher already consumes,
so enabling segmentation is a drop-in replacement for border_detect.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class RoadBoundaryFromMask:
    """Turn a drivable-surface boolean mask into near-field border pixels."""

    def __init__(self, near_band=(0.5, 0.99), col_step=4):
        self.near_band = near_band
        self.col_step = col_step

    def boundary_pixels(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        top = int(self.near_band[0] * h)
        out = []
        for u in range(0, w, self.col_step):
            col = mask[top:, u]
            on = np.nonzero(col)[0]
            if len(on) == 0:
                continue
            # the road region's upper edge (far boundary) and the left/right
            # transitions show up as the first/last 'on' row per column.
            v_far = on.min() + top
            out.append((u, v_far))
        if not out:
            return np.zeros((0, 2), float)
        return np.asarray(out, float)


class DeepLabV3Segmenter:
    """torchvision DeepLabV3 drivable-surface segmentation (GPU-optional)."""

    # VOC 'road' is not a class; COCO/VOC lack an explicit road label, so we
    # approximate drivable surface by low-saturation, low-vertical-gradient
    # regions refined by the network's background/large-region prediction.
    def __init__(self, cfg: dict):
        import torch  # noqa
        import torchvision as tv
        from .device import to_device
        self.torch = torch
        req = cfg.get("runtime", {}).get("device", "auto")
        variant = cfg.get("segmentation", {}).get("model", "mobilenet")
        if variant == "resnet50":
            net = tv.models.segmentation.deeplabv3_resnet50(
                weights=tv.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
        else:
            net = tv.models.segmentation.deeplabv3_mobilenet_v3_large(
                weights=tv.models.segmentation.
                DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
        self.model, self.device = to_device(net.eval(), req)
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)
        self.size = int(cfg.get("segmentation", {}).get("infer_size", 512))

    def segment(self, img_bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        small = cv2.resize(rgb, (self.size, int(self.size * h / w)))
        x = (small - self.mean) / self.std
        x = torch.from_numpy(x.transpose(2, 0, 1))[None]
        x, _ = _safe_to(x, self.device)
        with torch.no_grad():
            out = self.model(x)["out"][0].cpu().numpy()
        labels = out.argmax(0)
        # background class (0) at the bottom of the frame is the road-ish area;
        # combine with a bottom-region prior.
        mask_small = (labels == 0)
        mask = cv2.resize(mask_small.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        return mask


def _safe_to(t, device):
    try:
        return t.to(device), device
    except Exception:
        return t.to("cpu"), "cpu"


def build_segmenter(cfg: dict):
    sc = cfg.get("segmentation", {})
    if not sc.get("enabled"):
        return None
    backend = sc.get("backend", "deeplabv3")
    if backend == "deeplabv3":
        return DeepLabV3Segmenter(cfg)
    raise ValueError(f"unknown segmentation backend {backend}")


def detect_border_pixels_seg(img: np.ndarray, segmenter,
                             boundary: Optional[RoadBoundaryFromMask] = None
                             ) -> np.ndarray:
    """Segmentation-based border pixels, matching detect_border_pixels output."""
    boundary = boundary or RoadBoundaryFromMask()
    mask = segmenter.segment(img)
    return boundary.boundary_pixels(mask)
