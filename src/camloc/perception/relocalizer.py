"""Visual relocalization against a training-run database.

A relocalization map is built OFFLINE from the training run: for a set of
sampled image sets we store (global descriptor, pose) where the pose is the
training ground truth. This is permitted -- training GT may be used to build
a relocalization map; it is never read on the inference path. The database is
serialized to disk and loaded read-only at inference.

At inference, each query image set's descriptor is matched (cosine) against
the database; the top-k poses are aggregated (similarity-weighted) into a
coarse absolute (x, y) position fix with a confidence. The EKF consumes this
as a gated position update, so it only ever *bounds* drift and never
overrides a confident local estimate.

Causality: the database is static (built beforehand). Querying uses only the
current image set. No future data, no live ground truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RelocMap:
    descs: np.ndarray        # (N, D) L2-normalized
    poses: np.ndarray        # (N, 3) x,y,yaw (training GT, map-building only)
    stamps: np.ndarray       # (N,) ns
    descriptor_kind: str

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, descs=self.descs.astype(np.float32),
                            poses=self.poses.astype(np.float64),
                            stamps=self.stamps.astype(np.int64),
                            kind=np.array([self.descriptor_kind]))

    @staticmethod
    def load(path: str) -> "RelocMap":
        z = np.load(path, allow_pickle=True)
        return RelocMap(descs=z["descs"], poses=z["poses"], stamps=z["stamps"],
                        descriptor_kind=str(z["kind"][0]))


@dataclass
class RelocResult:
    ok: bool
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    confidence: float = 0.0
    r_pos: float = 10.0


class Relocalizer:
    def __init__(self, cfg: dict, reloc_map: RelocMap, descriptor):
        self.cfg = cfg
        rc = cfg.get("reloc", {})
        self.map = reloc_map
        self.descriptor = descriptor
        self.topk = int(rc.get("topk", 5))
        self.min_sim = float(rc.get("min_sim", 0.5))
        self.max_spread = float(rc.get("max_spread_m", 8.0))
        self.r_pos = float(rc.get("r_pos", 6.0))
        self.gate_dist = float(rc.get("gate_dist_m", 20.0))

    def query(self, images, predicted_pose) -> RelocResult:
        if self.map.descs.shape[0] == 0:
            return RelocResult(ok=False)
        q = self.descriptor(images)
        # Restrict candidates to the motion-prior neighbourhood FIRST. This is
        # what makes coarse retrieval reliable: a global top-k often matches a
        # visually similar but wrong part of the circuit, whereas within a
        # small radius of the predicted pose the best descriptor match is the
        # right place. (Validated: ~6 m retrieval with a prior vs ~18 m global.)
        if predicted_pose is not None:
            d = np.hypot(self.map.poses[:, 0] - predicted_pose[0],
                         self.map.poses[:, 1] - predicted_pose[1])
            local = np.nonzero(d <= self.gate_dist)[0]
            if len(local) < 2:
                return RelocResult(ok=False)
            sims_local = self.map.descs[local] @ q
            k = min(self.topk, len(local))
            order = np.argsort(-sims_local)[:k]
            top = local[order]
            s = sims_local[order]
        else:
            sims = self.map.descs @ q
            k = min(self.topk, len(sims))
            top = np.argpartition(-sims, k - 1)[:k]
            top = top[np.argsort(-sims[top])]
            s = sims[top]
        if s[0] < self.min_sim:
            return RelocResult(ok=False, confidence=float(s[0]))
        cand = self.map.poses[top]            # (k,3)
        w = np.clip(s, 0, None)
        w = w / (w.sum() + 1e-9)
        est = (w[:, None] * cand[:, :2]).sum(axis=0)
        # spread of candidates -> reject multimodal/ambiguous retrievals
        spread = np.sqrt(((cand[:, :2] - est) ** 2).sum(axis=1)).mean()
        if spread > self.max_spread:
            return RelocResult(ok=False, confidence=float(s[0]))
        # consistency with prediction (gate against gross mismatch)
        if predicted_pose is not None:
            d = np.hypot(est[0] - predicted_pose[0], est[1] - predicted_pose[1])
            if d > self.gate_dist:
                return RelocResult(ok=False, confidence=float(s[0]))
        # yaw from best match (for diagnostics; EKF uses position only)
        yaw = float(cand[0, 2])
        conf = float(s[0])
        r = self.r_pos * max(1.0, spread / 2.0)
        return RelocResult(ok=True, x=float(est[0]), y=float(est[1]),
                           yaw=yaw, confidence=conf, r_pos=r)
