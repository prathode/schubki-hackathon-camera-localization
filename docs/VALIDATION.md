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

## GPU paths — to be measured on the single-GPU workstation

The learned absolute-correction paths (ALIKED+LightGlue essential-matrix,
3D-landmark PnP, LoMa, DeepLabV3 border segmentation) require a GPU for
practical runtime and are **not measured in the CPU sandbox**. They are
implemented, unit-tested where the geometry allows (`tests/test_landmarks.py`),
and runnable via the README commands. Their RMSE — and crucially their
cross-session transfer to the hidden test — must be validated on the held-out
laps (A and B above) on the target workstation before being trusted. No
improved number is claimed here that has not been measured.
