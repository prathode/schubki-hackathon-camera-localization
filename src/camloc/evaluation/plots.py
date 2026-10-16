"""Validation visualization: estimated trajectory vs ground truth.

Evaluation-only. Ground truth is read here solely for plotting/metrics and is
never exposed to the inference path.
"""
from __future__ import annotations

import numpy as np


def _interp_gt(gt_t, gt_x, gt_y, gt_yaw, stamps):
    i = np.clip(np.searchsorted(gt_t, stamps), 1, len(gt_t) - 1)
    a = (stamps - gt_t[i - 1]) / (gt_t[i] - gt_t[i - 1] + 1e-9)
    gx = gt_x[i - 1] + a * (gt_x[i] - gt_x[i - 1])
    gy = gt_y[i - 1] + a * (gt_y[i] - gt_y[i - 1])
    gyaw_u = np.unwrap(gt_yaw)
    gyaw = gyaw_u[i - 1] + a * (gyaw_u[i] - gyaw_u[i - 1])
    return gx, gy, gyaw


def plot_trajectory(gt_csv, sub_csv, track_borders, out_png,
                    title="Estimated vs GT trajectory"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = np.genfromtxt(gt_csv, delimiter=",", names=True)
    sb = np.genfromtxt(sub_csv, delimiter=",", names=True)
    gt_t = gt["timestamp_ns"].astype(np.int64)
    st = sb["timestamp_ns"].astype(np.int64)
    gx, gy, _ = _interp_gt(gt_t, gt["x"], gt["y"], gt["yaw"], st)

    fig, ax = plt.subplots(figsize=(10, 12))
    for name, col in [("track_left", "0.6"), ("track_right", "0.6"),
                      ("pitlane_left", "orange"), ("pitlane_right", "orange")]:
        b = track_borders.get(name)
        if b is not None:
            ax.plot(b[:, 0], b[:, 1], color=col, lw=0.4)
    ax.plot(gx, gy, "g-", lw=1.2, label="ground truth")
    ax.plot(sb["x"], sb["y"], "r-", lw=1.0, label="estimate")
    ax.plot(sb["x"][0], sb["y"][0], "ko", ms=8, label="start")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    return out_png


def plot_error_timeline(gt_csv, sub_csv, out_png, title="Error over time"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = np.genfromtxt(gt_csv, delimiter=",", names=True)
    sb = np.genfromtxt(sub_csv, delimiter=",", names=True)
    gt_t = gt["timestamp_ns"].astype(np.int64)
    st = sb["timestamp_ns"].astype(np.int64)
    gx, gy, gyaw = _interp_gt(gt_t, gt["x"], gt["y"], gt["yaw"], st)
    dx, dy = sb["x"] - gx, sb["y"] - gy
    lat = -np.sin(gyaw) * dx + np.cos(gyaw) * dy
    lon = np.cos(gyaw) * dx + np.sin(gyaw) * dy
    pos = np.hypot(dx, dy)
    dyaw = np.degrees(np.angle(np.exp(1j * (sb["yaw"] - gyaw))))
    secs = (st - gt_t[0]) / 1e9

    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axs[0].plot(secs, pos, label="position")
    axs[0].plot(secs, np.abs(lat), label="|lateral|")
    axs[0].plot(secs, np.abs(lon), label="|longitudinal|")
    axs[0].set_ylabel("error [m]")
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    axs[1].plot(secs, dyaw, "C3")
    axs[1].set_ylabel("yaw error [deg]")
    axs[1].set_xlabel("time [s]")
    axs[1].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    return out_png
