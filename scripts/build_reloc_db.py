#!/usr/bin/env python3
"""Build the visual relocalization database from the TRAINING run.

Permitted: training ground-truth poses may be used to build a relocalization
map. The resulting DB (descriptors + poses) is loaded read-only at inference;
no live ground truth is ever read on the inference path.

    python scripts/build_reloc_db.py --mcap train.mcap --out outputs/reloc_db.npz

Optionally restrict the time range (to exclude held-out validation laps from
the DB) with --exclude-start-s/--exclude-end-s.
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
from camloc.perception.descriptors import build_descriptor
from camloc.perception.relocalizer import RelocMap

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def load_gt_poses(mcap, cfg):
    """Read /tf map->base_link GT (map-building only). Returns (stamps, xyyaw)."""
    tf = cfg["dataset"]["tf_topic"]
    mapf = cfg["dataset"]["map_frame"]
    base = cfg["dataset"]["base_link_frame"]
    ts, xs, ys, yaws = [], [], [], []
    with open(mcap, "rb") as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, _, msg in r.iter_decoded_messages(topics=[tf]):
            for t in msg.transforms:
                if t.header.frame_id != mapf or t.child_frame_id != base:
                    continue
                q = t.transform.rotation
                p = t.transform.translation
                yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y * q.y + q.z * q.z))
                ts.append(stamp_to_ns(t.header.stamp))
                xs.append(p.x); ys.append(p.y); yaws.append(yaw)
    ts = np.array(ts, np.int64)
    o = np.argsort(ts)
    return ts[o], np.stack([np.array(xs)[o], np.array(ys)[o], np.array(yaws)[o]], 1)


def interp_pose(gt_t, gt_p, stamp):
    i = int(np.searchsorted(gt_t, stamp))
    i = min(max(i, 1), len(gt_t) - 1)
    denom = (gt_t[i] - gt_t[i - 1]) or 1
    a = (stamp - gt_t[i - 1]) / denom
    xy = gt_p[i - 1, :2] + a * (gt_p[i, :2] - gt_p[i - 1, :2])
    yaw = gt_p[i - 1, 2]
    return np.array([xy[0], xy[1], yaw])


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
    out = args.out or cfg["reloc"]["db_path"]
    stride = args.stride or int(cfg["reloc"]["build_stride"])

    print(f"[reloc-db] GT poses from {args.mcap}")
    gt_t, gt_p = load_gt_poses(args.mcap, cfg)
    print(f"[reloc-db] {len(gt_t)} GT poses")

    src = McapSource(args.mcap, cfg)
    t0, _ = src.time_bounds()
    ex0 = None if args.exclude_start_s is None else t0 + int(args.exclude_start_s * 1e9)
    ex1 = None if args.exclude_end_s is None else t0 + int(args.exclude_end_s * 1e9)

    desc_fn = build_descriptor(cfg)
    kind = cfg["reloc"]["descriptor"]
    asm = ImageSetAssembler(cfg)

    descs, poses, stamps = [], [], []
    n = 0
    tic = time.time()
    for k, msg in asm.stream(src):
        if k != "set":
            continue
        n += 1
        if n % stride:
            continue
        if ex0 is not None and ex0 <= msg.stamp_ns <= ex1:
            continue  # keep validation laps out of the DB
        imgs = {c: m.decode() for c, m in msg.images.items()}
        descs.append(desc_fn(imgs))
        poses.append(interp_pose(gt_t, gt_p, msg.stamp_ns))
        stamps.append(msg.stamp_ns)
        if len(descs) % 500 == 0:
            print(f"  built {len(descs)} entries  ({time.time()-tic:.0f}s)")

    rm = RelocMap(descs=np.array(descs, np.float32),
                  poses=np.array(poses, np.float64),
                  stamps=np.array(stamps, np.int64),
                  descriptor_kind=kind)
    rm.save(out)
    print(f"[reloc-db] saved {len(descs)} entries ({kind}) -> {out} "
          f"in {time.time()-tic:.0f}s")


if __name__ == "__main__":
    main()
