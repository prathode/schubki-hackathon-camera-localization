"""Global image descriptors for place recognition / relocalization.

Two backends, selected by config/availability:

  * "thumb" (default, CPU): a fast, dependency-free descriptor formed from
    normalized multi-camera grayscale thumbnails. No GPU, no weights. Used in
    the CPU sandbox and as a universal fallback.

  * "dinov2" (GPU, optional): global token from a pretrained DINOv2 ViT
    (declared external model). Much stronger place recognition; runs on the
    single-GPU workstation (e.g. P5200, 16 GB). Enabled via
    reloc.descriptor=dinov2 and requires torch + timm.

Both return an L2-normalized vector per image set so cosine similarity =
dot product. The descriptor is deliberately pluggable so the relocalization
map and the query path always use the same backend.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class ThumbnailDescriptor:
    """CPU multi-camera place descriptor (intensity grid + edge grid + hist).

    Combines, per camera: a 4x4 intensity grid, a 4x4 Canny-edge-density grid,
    and a 16-bin intensity histogram. More discriminative than a raw thumbnail
    for night place recognition (validated ~6 m retrieval with a motion prior),
    while staying cheap and dependency-free. Used as a CPU drift bound; the
    metric/PnP paths provide the sub-metre fix on a GPU.
    """

    GRID = 4
    HIST = 16

    def __init__(self, cams: List[str], tw: int = 32, th: int = 16):
        self.cams = cams
        self._per = self.GRID * self.GRID * 2 + self.HIST
        self.dim = len(cams) * self._per

    def __call__(self, images: Dict[str, np.ndarray]) -> np.ndarray:
        vecs = []
        for c in self.cams:
            im = images.get(c)
            if im is None:
                vecs.append(np.zeros(self._per, np.float32))
                continue
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
            gs = cv2.resize(g, (self.GRID, self.GRID)).astype(np.float32).ravel()
            e = cv2.Canny(g, 50, 150)
            eg = cv2.resize(e, (self.GRID, self.GRID)).astype(np.float32).ravel()
            h = cv2.calcHist([g], [0], None, [self.HIST], [0, 256]).astype(
                np.float32).ravel()
            vecs.append(np.concatenate([gs, eg, h]))
        v = np.concatenate(vecs).astype(np.float32)
        v = (v - v.mean()) / (v.std() + 1e-6)
        n = np.linalg.norm(v)
        return v / (n + 1e-9)


class DinoV2Descriptor:
    """Pretrained DINOv2 global descriptor (GPU, optional).

    DECLARED EXTERNAL MODEL: facebookresearch/dinov2 (ViT-S/14 by default),
    pretrained, used only as a frozen feature extractor for place
    recognition. No ground truth involved.
    """

    def __init__(self, cams: List[str], device: str = "auto",
                 model: str = "dinov2_vits14", img_size: int = 224):
        import torch  # noqa
        from .device import to_device
        self.torch = torch
        self.cams = cams
        self.img_size = img_size
        net = torch.hub.load("facebookresearch/dinov2", model).eval()
        # fail-safe device move: GPU trouble -> CPU instead of a crash.
        self.model, self.device = to_device(net, device)
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)
        # per-camera embedding dim known after a dry run
        self.dim = None

    def _prep(self, im: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        g = cv2.resize(g, (self.img_size, self.img_size))
        g = (g - self.mean) / self.std
        return g.transpose(2, 0, 1)

    def __call__(self, images: Dict[str, np.ndarray]) -> np.ndarray:
        torch = self.torch
        batch = []
        order = []
        for c in self.cams:
            im = images.get(c)
            if im is None:
                continue
            batch.append(self._prep(im))
            order.append(c)
        if not batch:
            return np.zeros(self.dim or 384 * len(self.cams), np.float32)
        x = torch.from_numpy(np.stack(batch)).to(self.device)
        with torch.no_grad():
            feats = self.model(x).cpu().numpy()  # (n,emb)
        emb = feats.shape[1]
        if self.dim is None:
            self.dim = emb * len(self.cams)
        # place each cam's feature in its slot (zeros for missing)
        out = np.zeros(emb * len(self.cams), np.float32)
        for c, f in zip(order, feats):
            i = self.cams.index(c)
            out[i * emb:(i + 1) * emb] = f
        n = np.linalg.norm(out)
        return out / (n + 1e-9)


def build_descriptor(cfg: dict):
    cams = list(cfg["dataset"]["cameras"])
    rc = cfg.get("reloc", {})
    kind = rc.get("descriptor", "thumb")
    if kind == "dinov2":
        device = cfg.get("runtime", {}).get("device", "auto")
        return DinoV2Descriptor(cams, device=device,
                                model=rc.get("dino_model", "dinov2_vits14"))
    return ThumbnailDescriptor(cams)
