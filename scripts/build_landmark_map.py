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
from camloc.perception.matchers import build_matcher
from camloc.perception.landmarks import (
    LandmarkMap, triangulate_pair, voxel_dedup,
)

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
    ap.add_argument("--matcher", default=None,
                    help="sift | aliked_lightglue (default from config)")
    ap.add_argument("--pair-gap", type=int, default=3,
                    help="frames between triangulation pairs (3 = 0.3 s)")
    ap.add_argument("--voxel-m", type=float, default=0.5,
                    help="voxel size for landmark dedup")
    ap.add_argument("--max-height-m", type=float, default=30.0,
                    help="drop landmarks more than this above the track")
    ap.add_argument("--exclude-start-s", type=float, default=None)
    ap.add_argument("--exclude-end-s", type=float, default=None)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    cam = args.cam
    calib = load_calibration(args.mcap, cfg)
    cc = calib[cam]
    gt_t, gt_p = load_gt(args.mcap, cfg)
    fx = build_matcher(cfg, args.matcher)
    print(f"[landmark] matcher={fx.name} cam={cam} pair_gap={args.pair_gap}")

    src = McapSource(args.mcap, cfg)
    t0, _ = src.time_bounds()
    ex0 = None if args.exclude_start_s is None else t0 + int(args.exclude_start_s * 1e9)
    ex1 = None if args.exclude_end_s is None else t0 + int(args.exclude_end_s * 1e9)
    asm = ImageSetAssembler(cfg)

    from collections import deque
    buf = deque(maxlen=args.pair_gap + 1)  # recent (pose, feats)
    pts_all, desc_all, vy_all = [], [], []
    n = 0
    tic = time.time()
    for k, msg in asm.stream(src):
        if k != "set":
            continue
        n += 1
        if n % args.stride:
            continue
        if ex0 is not None and ex0 <= msg.stamp_ns <= ex1:
            buf.clear()
            continue
        m = msg.get(cam)
        if m is None:
            continue
        pose = interp(gt_t, gt_p, msg.stamp_ns)
        feats = fx.extract(m.decode())
        # pair with the frame `pair_gap` steps back (0.3 s at 10 Hz)
        if len(buf) == buf.maxlen:
            pose0, feats0 = buf[0]
            base = np.hypot(pose[0] - pose0[0], pose[1] - pose0[1])
            if args.min_baseline_m <= base <= args.max_baseline_m:
                pa, pb = fx.match_features(feats0, feats)
                if len(pa) >= 10:
                    X, ok = triangulate_pair(cc.K, cc.D, pa, pb,
                                             pose0, pose, cc.extrinsics)
                    if ok.any():
                        Xok = X[ok]
                        # drop points too high above the track (floodlights)
                        hmask = Xok[:, 2] < args.max_height_m
                        if hmask.any():
                            pts_all.append(Xok[hmask])
                            desc_all.append(_desc_for(feats0, pa[ok][hmask]))
                            vy_all.append(np.full(int(hmask.sum()), pose0[2]))
        buf.append((pose, feats))
        if len(pts_all) and n % (args.stride * 300) < args.stride:
            tot = sum(len(p) for p in pts_all)
            print(f"  {tot} raw landmarks so far ({time.time()-tic:.0f}s)")

    if not pts_all:
        print("[landmark] no landmarks triangulated; check baseline range")
        return
    P = np.concatenate(pts_all, 0)
    Dd = np.concatenate(desc_all, 0)
    VY = np.concatenate(vy_all, 0)
    print(f"[landmark] {len(P)} raw landmarks; voxel dedup @ {args.voxel_m} m ...")
    P, Dd, VY = voxel_dedup(P, Dd, args.voxel_m, VY)
    lm = LandmarkMap(cam=cam, points3d=P, descriptors=Dd,
                     obs_xy=P[:, :2].copy(), view_yaw=VY).build_index()
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
