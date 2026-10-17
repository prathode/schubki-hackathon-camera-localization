#!/usr/bin/env python3
"""Benchmark feature matchers on identical dataset frame pairs.

Compares ALIKED+LightGlue vs LoMa (and variants) on cross-lap image pairs
drawn from the training run, reporting: number of matches, RANSAC inlier
ratio (essential matrix), relative-pose error vs GT, and runtime per pair.
This is the Module-D "measure, don't assume" benchmark: only promote a
matcher into the pipeline if it measurably wins here.

    python scripts/benchmark_matchers.py --mcap train.mcap \
        --matchers aliked_lightglue loma-b --cam fl --pairs 40 \
        --set runtime.device=cuda

GT poses are used only to form cross-lap pairs and to score relative-pose
error (evaluation, not inference).
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

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

try:
    import cv2
except Exception:
    cv2 = None


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
                ts.append(stamp_to_ns(t.header.stamp)); xs.append(p.x)
                ys.append(p.y); yaws.append(yaw)
    ts = np.array(ts, np.int64); o = np.argsort(ts)
    return ts[o], np.stack([np.array(xs)[o], np.array(ys)[o],
                            np.array(yaws)[o]], 1)


def collect_frames(cfg, mcap, cam, gt_t, gt_p, n_target):
    """Collect frames tagged with GT pose for pairing."""
    src = McapSource(mcap, cfg); asm = ImageSetAssembler(cfg)
    out = []
    n = 0
    for k, msg in asm.stream(src):
        if k != "set":
            continue
        n += 1
        if n % 3:
            continue
        m = msg.get(cam)
        if m is None:
            continue
        i = min(max(int(np.searchsorted(gt_t, msg.stamp_ns)), 1), len(gt_t) - 1)
        pose = gt_p[i]
        out.append((pose.copy(), m.decode()))
        if len(out) >= n_target:
            break
    return out


def make_pairs(frames, n_pairs, min_dt_idx=50, max_dist=6.0):
    """Cross-lap pairs: same place (close xy) but temporally separated."""
    poses = np.array([f[0][:2] for f in frames])
    pairs = []
    rng = np.random.default_rng(0)
    order = rng.permutation(len(frames))
    for i in order:
        d = np.hypot(poses[:, 0] - poses[i, 0], poses[:, 1] - poses[i, 1])
        cand = np.nonzero((d < max_dist) & (np.abs(np.arange(len(frames)) - i)
                                            > min_dt_idx))[0]
        if len(cand):
            j = cand[np.argmin(d[cand])]
            pairs.append((i, int(j)))
        if len(pairs) >= n_pairs:
            break
    return pairs


def rel_pose_err(K, D, pa, pb, pose_a, pose_b):
    if len(pa) < 8:
        return np.nan, 0
    na = cv2.undistortPoints(pa.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    nb = cv2.undistortPoints(pb.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    E, mask = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.RANSAC,
                                   prob=0.999, threshold=1e-3)
    if E is None or mask is None:
        return np.nan, 0
    inl = int(mask.sum())
    _, R, t, _ = cv2.recoverPose(E, na, nb, np.eye(3), mask=mask)
    yaw_rel_est = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    yaw_rel_gt = np.degrees(np.arctan2(np.sin(pose_b[2] - pose_a[2]),
                                       np.cos(pose_b[2] - pose_a[2])))
    err = abs(np.angle(np.exp(1j * np.radians(yaw_rel_est - yaw_rel_gt))))
    return np.degrees(err), inl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--matchers", nargs="+", default=["aliked_lightglue"])
    ap.add_argument("--cam", default="fl")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default="reports/matcher_benchmark.csv")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    calib = load_calibration(args.mcap, cfg)
    cc = calib[args.cam]
    gt_t, gt_p = load_gt(args.mcap, cfg)
    print(f"[bench] collecting up to {args.frames} frames (cam {args.cam})")
    frames = collect_frames(cfg, args.mcap, args.cam, gt_t, gt_p, args.frames)
    pairs = make_pairs(frames, args.pairs)
    print(f"[bench] {len(pairs)} cross-lap pairs")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = ["matcher,n_pairs,mean_matches,mean_inliers,mean_inlier_ratio,"
            "median_yaw_err_deg,mean_ms_per_pair"]
    for mname in args.matchers:
        print(f"\n[bench] === {mname} ===")
        try:
            matcher = build_matcher(cfg, mname)
        except Exception as e:
            print(f"  skip {mname}: {e!r}")
            continue
        nm, ninl, rat, errs, times = [], [], [], [], []
        for (i, j) in pairs:
            ia = frames[i][1]; ib = frames[j][1]
            t0 = time.time()
            try:
                pa, pb = matcher.match_images(ia, ib)
            except Exception as e:
                print(f"  match failed: {e!r}")
                continue
            times.append((time.time() - t0) * 1000)
            nm.append(len(pa))
            err, inl = rel_pose_err(cc.K, cc.D, pa, pb,
                                    frames[i][0], frames[j][0])
            ninl.append(inl)
            rat.append(inl / max(len(pa), 1))
            if not np.isnan(err):
                errs.append(err)
        if not nm:
            continue
        row = (f"{mname},{len(nm)},{np.mean(nm):.0f},{np.mean(ninl):.0f},"
               f"{np.mean(rat):.3f},{np.median(errs) if errs else float('nan'):.2f},"
               f"{np.mean(times):.1f}")
        print("  " + row)
        rows.append(row)

    with open(args.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\n[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
