#!/usr/bin/env python3
"""Single entry point: raw MCAP -> submission CSV (causal, camera-only).

Example
-------
    python run.py --mcap path/to/run.mcap --out outputs/submission.csv

The initial pose is taken from the first ground-truth pose of the run, which
the challenge explicitly permits as the initialization. No other ground-truth
data is read on the inference path.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# make src/ importable without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from camloc.config import load_config
from camloc.data.mcap_source import McapSource
from camloc.data.calibration import load_calibration
from camloc.data.track import load_track_bounds
from camloc.localization.pipeline import Localizer
from camloc.submission.writer import write_submission


def build_border_matcher(cfg, calib, track):
    if not cfg["border_match"]["enabled"]:
        return None
    from camloc.perception.border_matcher import BorderMatcher
    return BorderMatcher(cfg, calib, track)


def build_metric_reloc(cfg, calib):
    mc = cfg.get("metric_reloc", {})
    if not mc.get("enabled"):
        return None
    db_path = mc["db_path"]
    if not os.path.exists(db_path):
        print(f"[metric_reloc] DB not found at {db_path}; disabling. "
              f"Build it with scripts/build_feature_db.py")
        return None
    import pickle
    from camloc.perception.features import AlikedLightGlue
    from camloc.perception.metric_reloc import MetricRelocalizer
    with open(db_path, "rb") as f:
        dbs = pickle.load(f)
    for db in dbs.values():
        db.build_index()
    fx = AlikedLightGlue(cfg)
    print(f"[metric_reloc] loaded DBs: "
          f"{{ {', '.join(f'{c}:{len(db.poses)}' for c, db in dbs.items())} }}")
    return MetricRelocalizer(cfg, dbs, calib, fx)


def build_relocalizer(cfg):
    rc = cfg.get("reloc", {})
    if not rc.get("enabled"):
        return None
    db_path = rc["db_path"]
    if not os.path.exists(db_path):
        print(f"[reloc] DB not found at {db_path}; disabling relocalization. "
              f"Build it with scripts/build_reloc_db.py")
        return None
    from camloc.perception.relocalizer import Relocalizer, RelocMap
    from camloc.perception.descriptors import build_descriptor
    rm = RelocMap.load(db_path)
    desc = build_descriptor(cfg)
    print(f"[reloc] loaded DB with {rm.descs.shape[0]} entries "
          f"({rm.descriptor_kind})")
    return Relocalizer(cfg, rm, desc)


def _first_gt_pose_at(mcap, cfg, start_ns):
    """GT pose of base_link at/after start_ns (validation-window init only)."""
    import numpy as np
    from camloc.data.mcap_source import stamp_to_ns
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    tf = cfg["dataset"]["tf_topic"]
    mapf = cfg["dataset"]["map_frame"]
    base = cfg["dataset"]["base_link_frame"]
    with open(mcap, "rb") as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, _, _, msg in r.iter_decoded_messages(topics=[tf], start_time=start_ns):
            for t in msg.transforms:
                if t.header.frame_id == mapf and t.child_frame_id == base:
                    q = t.transform.rotation
                    p = t.transform.translation
                    yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                                     1 - 2 * (q.y * q.y + q.z * q.z))
                    return {"x": float(p.x), "y": float(p.y), "yaw": float(yaw)}
    return None


def main():
    ap = argparse.ArgumentParser(description="Camera-based race-track localization")
    ap.add_argument("--mcap", required=True, help="ROS 2 MCAP recording")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None, help="output submission CSV")
    ap.add_argument("--start-s", type=float, default=None,
                    help="start offset seconds (for validation laps)")
    ap.add_argument("--end-s", type=float, default=None,
                    help="end offset seconds (for validation laps)")
    ap.add_argument("--init-xyyaw", default=None,
                    help="override initial pose as 'x,y,yaw' (else first GT pose)")
    ap.add_argument("--set", action="append", default=[],
                    help="config override key=value (repeatable)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    cfg["dataset"]["mcap"] = args.mcap
    out_path = args.out or cfg["output"]["path"]

    print(f"[load] calibration + track from {args.mcap}")
    calib = load_calibration(args.mcap, cfg)
    track = load_track_bounds(args.mcap, cfg)
    track.build_index(step=0.5, include_pit=True)

    src = McapSource(args.mcap, cfg)
    t0, t1 = src.time_bounds()
    start_ns = t0 if args.start_s is None else t0 + int(args.start_s * 1e9)
    end_ns = t1 if args.end_s is None else t0 + int(args.end_s * 1e9)

    # ---- initial pose (permitted: first pose of the run / window) ----
    if args.init_xyyaw:
        x, y, yaw = (float(v) for v in args.init_xyyaw.split(","))
        print(f"[init] using supplied initial pose ({x:.3f},{y:.3f},{yaw:.4f})")
    elif args.start_s is not None:
        # Validation lap starting mid-run: the legitimate analogue of "the
        # first pose of the run" is the pose at the window start. This is the
        # single permitted GT pose used for initialization only.
        seed = _first_gt_pose_at(args.mcap, cfg, start_ns)
        if seed is None:
            print("ERROR: could not read window initial pose", file=sys.stderr)
            sys.exit(1)
        x, y, yaw = seed["x"], seed["y"], seed["yaw"]
        print(f"[init] window-start seed ({x:.3f},{y:.3f},{yaw:.4f})")
    else:
        seed = src.first_state_pose_eval_only()
        if seed is None:
            print("ERROR: could not read initial pose", file=sys.stderr)
            sys.exit(1)
        x, y, yaw = seed["x"], seed["y"], seed["yaw"]
        print(f"[init] first-pose seed ({x:.3f},{y:.3f},{yaw:.4f})")

    matcher = build_border_matcher(cfg, calib, track)
    relocalizer = build_relocalizer(cfg)
    tangent = None
    if cfg.get("tangent_yaw", {}).get("enabled", True):
        from camloc.localization.tangent_yaw import TangentYawCorrector
        tangent = TangentYawCorrector(cfg, track)
    metric = build_metric_reloc(cfg, calib)
    loc = Localizer(cfg, border_matcher=matcher, relocalizer=relocalizer,
                    tangent_yaw=tangent, metric_reloc=metric)
    loc.initialize(x, y, yaw)

    def on_log(n, est):
        print(f"  [{n:6d}] t={est.stamp_ns} pose=({est.x:8.2f},{est.y:8.2f},"
              f"{est.yaw:+.3f}) src={est.source} npts={est.n_border_pts}")

    print("[run] streaming causal localization ...")
    tic = time.time()
    estimates = list(loc.run(src, start_ns, end_ns, on_log=on_log))
    elapsed = time.time() - tic

    n = write_submission(out_path, estimates)
    per_frame_ms = 1000 * elapsed / max(n, 1)
    print(f"[done] wrote {n} poses -> {out_path}")
    print(f"[runtime] {elapsed:.1f}s total, {per_frame_ms:.1f} ms/frame "
          f"({len(estimates)} frames)")


if __name__ == "__main__":
    main()
