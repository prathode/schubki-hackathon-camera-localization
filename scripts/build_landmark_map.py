#!/usr/bin/env python3
"""Build a sparse 3D landmark map from the TRAINING run for PnP relocalization.

Matches adjacent same-camera keyframes (ALIKED+LightGlue) and triangulates the
correspondences using the known training GT poses (permitted for map-building).
The map (3D points + descriptors) is loaded read-only at inference.

    python scripts/build_landmark_map.py --mcap train.mcap \
        --out outputs/landmarks_fl.npz --set runtime.device=cuda \
        --cam fl --exclude-start-s 680 --exclude-end-s 1063

Requires lightglue (ALIKED) + torch; a GPU is recommended.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from camloc.config import load_config
from camloc.data.mcap_source import McapSource, stamp_to_ns
from camloc.data.image_set import ImageSetAssembler
from camloc.data.calibration import load_calibration
from camloc.perception.features import AlikedLightGlue
from camloc.perception.landmarks import LandmarkMap, triangulate_pair

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def load_gt(mcap, cfg):
    tf = cfg["dataset"]["tf_topic"]; mapf = cfg["dataset"]["map_frame"]
    base = cfg["dataset"]["base_link_frame"]
    ts, xs, ys, yaws = [], [], [], []
    with open(mcap, "rb") as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, _, msg in r.iter_decoded_messages(topics=[tf]):
            for t in msg.transforms:
                if t.header.frame_id != mapf or t.child_frame_id != base:
                    continue
                q = t.transform.rotation; p = t.transform.translation
                yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y * q.y + q.z * q.z))
                ts.append(stamp_to_ns(t.header.stamp))
                xs.append(p.x); ys.append(p.y); yaws.append(yaw)
    ts = np.array(ts, np.int64); o = np.argsort(ts)
    return ts[o], np.stack([np.array(xs)[o], np.array(ys)[o],
                            np.array(yaws)[o]], 1)


def interp(gt_t, gt_p, s):
    i = min(max(int(np.searchsorted(gt_t, s)), 1), len(gt_t) - 1)
    a = (s - gt_t[i - 1]) / ((gt_t[i] - gt_t[i - 1]) or 1)
    xy = gt_p[i - 1, :2] + a * (gt_p[i, :2] - gt_p[i - 1, :2])
    return np.array([xy[0], xy[1], gt_p[i - 1, 2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default="fl")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--min-baseline-m", type=float, default=1.0)
    ap.add_argument("--max-baseline-m", type=float, default=8.0)
    ap.add_argument("--exclude-start-s", type=float, default=None)
    ap.add_argument("--exclude-end-s", type=float, default=None)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    cam = args.cam
    calib = load_calibration(args.mcap, cfg)
    cc = calib[cam]
    gt_t, gt_p = load_gt(args.mcap, cfg)
    fx = AlikedLightGlue(cfg)

    src = McapSource(args.mcap, cfg)
    t0, _ = src.time_bounds()
    ex0 = None if args.exclude_start_s is None else t0 + int(args.exclude_start_s * 1e9)
    ex1 = None if args.exclude_end_s is None else t0 + int(args.exclude_end_s * 1e9)
    asm = ImageSetAssembler(cfg)

    prev = None  # (pose, feats)
    pts_all, desc_all = [], []
    n = 0
    tic = time.time()
    for k, msg in asm.stream(src):
        if k != "set":
            continue
        n += 1
        if n % args.stride:
            continue
        if ex0 is not None and ex0 <= msg.stamp_ns <= ex1:
            prev = None
            continue
        m = msg.get(cam)
        if m is None:
            continue
        pose = interp(gt_t, gt_p, msg.stamp_ns)
        feats = fx.extract(m.decode())
        if prev is not None:
            base = np.hypot(pose[0] - prev[0][0], pose[1] - prev[0][1])
            if args.min_baseline_m <= base <= args.max_baseline_m:
                pa, pb = fx.match(prev[1], feats)
                if len(pa) >= 10:
                    X, ok = triangulate_pair(cc.K, cc.D, pa, pb,
                                             prev[0], pose, cc.extrinsics)
                    if ok.any():
                        pts_all.append(X[ok])
                        # descriptor from the current frame's matched points
                        from camloc.perception.landmarks import _mutual_nn  # noqa
                        # use prev-frame descriptors for the accepted matches
                        # (align by the match order)
                        # pa corresponds to prev feats keypoints; approximate
                        # descriptor as prev matched descriptors.
                        # recover indices via nearest keypoint:
                        desc_all.append(_desc_for(prev[1], pa[ok]))
        prev = (pose, feats)
        if len(pts_all) and n % (args.stride * 200) < args.stride:
            tot = sum(len(p) for p in pts_all)
            print(f"  {tot} landmarks so far ({time.time()-tic:.0f}s)")

    if not pts_all:
        print("[landmark] no landmarks triangulated; check baseline range")
        return
    P = np.concatenate(pts_all, 0)
    Dd = np.concatenate(desc_all, 0)
    lm = LandmarkMap(cam=cam, points3d=P, descriptors=Dd,
                     obs_xy=P[:, :2].copy()).build_index()
    lm.save(args.out)
    print(f"[landmark] saved {len(P)} landmarks (cam {cam}) -> {args.out} "
          f"in {time.time()-tic:.0f}s")


def _desc_for(feats, pts):
    """Nearest-keypoint descriptor lookup for a set of pixel points."""
    kp = feats.keypoints
    out = np.zeros((len(pts), feats.descriptors.shape[1]), np.float32)
    for i, p in enumerate(pts):
        j = int(np.argmin(np.hypot(kp[:, 0] - p[0], kp[:, 1] - p[1])))
        out[i] = feats.descriptors[j]
    return out


if __name__ == "__main__":
    main()
