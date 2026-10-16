#!/usr/bin/env python3
"""Generate trajectory + error-timeline plots for a submission vs GT.

    python scripts/make_plots.py --mcap run.mcap --gt gt.csv --sub submission.csv \
        --prefix outputs/val
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from camloc.config import load_config
from camloc.data.track import load_track_bounds
from camloc.evaluation.plots import plot_trajectory, plot_error_timeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--prefix", default="outputs/val")
    ap.add_argument("--title", default="Camera localization")
    args = ap.parse_args()

    cfg = load_config(args.config)
    track = load_track_bounds(args.mcap, cfg)
    t = plot_trajectory(args.gt, args.sub, track.borders,
                        args.prefix + "_trajectory.png",
                        title=args.title + " — trajectory")
    e = plot_error_timeline(args.gt, args.sub, args.prefix + "_error.png",
                            title=args.title + " — error over time")
    print("wrote", t)
    print("wrote", e)


if __name__ == "__main__":
    main()
