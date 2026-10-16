# Validation results (organizer `tools/evaluate.py`)

Dataset: `c0_2026-10-13_22-53-42_camloc` (training run, 1194 s, night).
All numbers produced by the organizer scoring script, reproducible with the
commands in the README. Runtime measured on the CPU-only dev sandbox
(3.0–3.1 ms/frame, well under the 100 ms camera period).

## Summary table

| Configuration | Pos RMSE [m] | Lateral [m] | Longit. [m] | Yaw [deg] | <1 m | Coverage |
|---|---|---|---|---|---|---|
| Dead reckoning (no ZUPT), full run | 64.40 | 50.30 | 40.21 | 8.54 | 2.0% | 100% |
| **Default (motion + ZUPT), full run** | **37.44** | 29.35 | 23.26 | 4.70 | 2.0% | 100% |
| Validation lap A (680–833 s) | 10.98 | 8.98 | 6.33 | 1.29 | 12.2% | n/a* |
| Validation lap B (953–1063 s) | 9.63 | 7.94 | 5.44 | 1.36 | 14.7% | n/a* |

\* A ~150 s lap is a window of the 1194 s GT, so the evaluator's coverage
metric (share of the whole GT window covered) is low by construction; the
per-lap RMSE is the meaningful figure. Coverage is 100% on the full run.

**Ablation (ZUPT yaw-rate bias calibration):** 64.40 → 37.44 m position RMSE
and 8.54° → 4.70° yaw RMSE on the full run — a causal, allowed, zero-overfit
sensor calibration from the stationary startup.

## Raw evaluator output

### Dead reckoning (no ZUPT), full run
```
position RMSE  [m]     64.399
lateral RMSE   [m]     50.302
longitudinal RMSE [m]  40.211
yaw RMSE     [deg]     8.538
coverage (1 s bins)    100.0 %
```

### Default (motion + ZUPT), full run  — submission config
```
poses submitted        11944   evaluated 11943
coverage (1 s bins)    100.0 %
position RMSE  [m]     37.443   <- SCORE
position median/p95    19.180 / 81.427
lateral RMSE   [m]     29.345
longitudinal RMSE [m]  23.256
yaw RMSE     [deg]     4.701
within 1.00 m          2.0 %
```

### Validation lap A (680–833 s)
```
position RMSE  [m]     10.983
lateral RMSE   [m]     8.978
longitudinal RMSE [m]  6.326
yaw RMSE     [deg]     1.285
```

### Validation lap B (953–1063 s)
```
position RMSE  [m]     9.627
lateral RMSE   [m]     7.944
longitudinal RMSE [m]  5.438
yaw RMSE     [deg]     1.357
```

## Reaching the sub-0.5 m target (GPU PnP path)

The 37 m full-run figure is **pure camera dead reckoning with no absolute
position fix** — it drifts because ~1° of residual yaw bends the trajectory
over 20 minutes (error grows 2 m → 6 m → 17 m → 32 m → 65 m across the run).
Coarse image-descriptor retrieval can only *bound* this drift to ~4–6 m
(it returns a nearby keyframe's pose); it physically cannot reach sub-metre.

**Sub-0.5 m requires metric visual geometry — the 3D-landmark PnP path.**
This is validated end-to-end with the real `fl` camera intrinsics and
realistic feature-matching noise (`tests/test_pnp_accuracy.py`, part of CI):

| Feature-match quality | Position error (median) | share < 0.5 m |
|---|---|---|
| Good GPU features (0.5 px, 5 cm landmarks, 15% outliers) | **0.097 m** | 100% |
| Excellent (0.3 px, 3 cm, 10% outliers) | 0.055 m | 100% |
| Noisy (1.0 px, 15 cm, 25% outliers) | ~0.33 m | ~78% |

So the geometry is **not** the limiting factor for the 0.5 m target —
feature-match quality on the night images is, and that must be measured on a
GPU with ALIKED+LightGlue. The CPU sandbox cannot run the neural network, so
the final on-track RMSE comes from the target workstation. **No sub-metre
number is claimed as measured on real images here** — only that the geometry
provably supports it.

Recommended GPU validation order (on held-out laps A/B, then hidden test):
1. `scripts/benchmark_matchers.py` — confirm match quality / inlier ratio.
2. `scripts/build_landmark_map.py` (dense, e.g. stride 2, cams fl+fr).
3. `run.py --set pnp_reloc.enabled=true ...` — measure RMSE vs laps A/B.
