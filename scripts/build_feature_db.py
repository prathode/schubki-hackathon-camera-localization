#!/usr/bin/env python3
"""Build the ALIKED feature keyframe DB from the TRAINING run.

Permitted: training ground-truth poses may build a relocalization map. The
resulting DB (ALIKED keypoints + descriptors + poses, per camera) is loaded
read-only at inference; no live ground truth is used.

    python scripts/build_feature_db.py --mcap train.mcap \
        --out outputs/feature_db.pkl --set runtime.device=cuda

Use --exclude-start-s/--exclude-end-s to keep held-out validation laps out.
Requires lightglue (ALIKED) + torch; a GPU is strongly recommended.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from camloc.config import load_config
from camloc.data.mcap_source import McapSource
from camloc.data.image_set import ImageSetAssembler
from camloc.perception.features import AlikedLightGlue
from camloc.perception.metric_reloc import FeatureDB

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from camloc.data.mcap_source import stamp_to_ns


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
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--exclude-start-s", type=float, default=None)
    ap.add_argument("--exclude-end-s", type=float, default=None)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    out = args.out or cfg["metric_reloc"]["db_path"]
    stride = args.stride or int(cfg["metric_reloc"]["build_stride"])
    cams = list(cfg["metric_reloc"]["cams"])

    print(f"[feat-db] device={cfg['runtime']['device']} cams={cams} stride={stride}")
    gt_t, gt_p = load_gt(args.mcap, cfg)
    fx = AlikedLightGlue(cfg)

    src = McapSource(args.mcap, cfg)
    t0, _ = src.time_bounds()
    ex0 = None if args.exclude_start_s is None else t0 + int(args.exclude_start_s * 1e9)
    ex1 = None if args.exclude_end_s is None else t0 + int(args.exclude_end_s * 1e9)

    acc = {c: {"poses": [], "stamps": [], "kp": [], "desc": [], "sz": []}
           for c in cams}
    asm = ImageSetAssembler(cfg)
    n = 0
    tic = time.time()
    for k, msg in asm.stream(src):
        if k != "set":
            continue
        n += 1
        if n % stride:
            continue
        if ex0 is not None and ex0 <= msg.stamp_ns <= ex1:
            continue
        pose = interp(gt_t, gt_p, msg.stamp_ns)
        for c in cams:
            m = msg.get(c)
            if m is None:
                continue
            f = fx.extract(m.decode())
            a = acc[c]
            a["poses"].append(pose); a["stamps"].append(msg.stamp_ns)
            a["kp"].append(f.keypoints.astype(np.float32))
            a["desc"].append(f.descriptors.astype(np.float32))
            a["sz"].append(f.size_wh)
        if sum(len(acc[c]["poses"]) for c in cams) % 200 < len(cams):
            done = len(acc[cams[0]]["poses"])
            print(f"  {done} keyframes/cam  ({time.time()-tic:.0f}s)")

    dbs = {}
    for c in cams:
        a = acc[c]
        db = FeatureDB(cam=c, poses=np.array(a["poses"], np.float64),
                       stamps=np.array(a["stamps"], np.int64),
                       keypoints=a["kp"], descriptors=a["desc"],
                       sizes=np.array(a["sz"], np.int32)).build_index()
        dbs[c] = db
        print(f"  cam {c}: {len(db.poses)} keyframes")

    import pickle
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(dbs, f)
    print(f"[feat-db] saved -> {out} in {time.time()-tic:.0f}s")


if __name__ == "__main__":
    main()
