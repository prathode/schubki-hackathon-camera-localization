# Approach, Ablations, and Failure Analysis

This document explains the method, the experiments that shaped it, and the
failure modes — the "lessons learned" the challenge asks for. Every number
here was produced by the organizer's `tools/evaluate.py` on this dataset.

## 1. Problem framing

- Score = position RMSE (lower is better); lateral error and yaw matter most
  for following a racing line; coverage ≥ 90 % to be ranked.
- Inputs allowed at inference: 7 cameras + intrinsics + extrinsics + track
  borders + permitted motion signals (speed, velocities, yaw rate,
  acceleration, steering, curvature) + the single first pose.
- Forbidden at inference: `/tf` ground truth, absolute pose / `s,n` from
  `state_estimation`, GNSS/LiDAR/radar, any future frame.

We verified every topic, schema, intrinsic, extrinsic, and border polyline
directly from the MCAP before writing estimator code (the brief warns against
guessing these). Key confirmed facts:

- `wz_radps` is the usable yaw rate. `yaw_vel_rad` is **not** a yaw rate — it
  integrates to ~10⁴ degrees over the run and must be ignored.
- Reported speed is extremely accurate: integrated `v_mps` over a lap matches
  the GT path length to within 0.3 %.
- Calibration reliability (from the dataset README): pitch reliable for all
  cameras; yaw reliable for `r`/`rl`, ±1° for `fl`/`fr`; `cl`/`cr` use an old
  calibration; roll and translation are uncalibrated mount values. We use the
  front/rear-reliable cameras for any geometric projection.

## 2. System

```
state_estimation (106 Hz, allowed fields only)
        │  vx,vy,wz  (+ ZUPT yaw-rate bias)
        ▼
   Motion model ──► EKF predict ──► pose [x,y,yaw]
                                      ▲
   7 cameras (10 Hz) ─► image set ────┤  (optional, gated updates)
                                      │   • map-tangent yaw
                                      │   • track-border ICP
                                      │   • visual relocalization
                                      ▼
                            one pose per image set → CSV (10 Hz)
```

The EKF state is the 2D pose in the map frame. Prediction uses the permitted
body velocity and yaw rate. Updates (all optional and hard-gated) correct the
pose from map/vision; if none fire, the filter emits the prediction, so every
image set yields a pose (100 % coverage).

### Zero-velocity yaw-rate bias calibration (default, on)

The run begins ~5 s before first motion. While `|v| < 0.3 m/s` the yaw rate is
pure gyro bias; we average it and subtract it from `wz` thereafter. This is a
standard ZUPT sensor calibration, not ground truth, and it generalizes (the
hidden run also starts "first motion − 5 s").

Effect on the full run: yaw RMSE 8.5° → 4.7°, position RMSE 64 m → 37 m.

## 3. Ablation (held-out validation lap A, 680–833 s)

| Configuration | Pos RMSE | Yaw RMSE | Verdict |
|---|---|---|---|
| Dead reckoning (speed+yaw) | 11.0 m | 1.29° | baseline backbone |
| + thumbnail relocalization | 11.1 m | 1.29° | no effect (retrieval ~18 m) |
| + ResNet18 relocalization | worse | — | generic features not discriminative |
| + classical border ICP | 55.0 m | 5.25° | **hurts** — noisy night detections |
| + map-tangent yaw | 7.2 m | 2.00° | helps *this lap*, unsafe globally |

Full run with ZUPT bias calibration (default): **37.4 m / 4.70° / 100 %
coverage / 3.1 ms per frame.**

## 4. Failure analysis (why the extras are off by default)

### 4.1 Map-tangent yaw correction — global divergence

Idea: the car's heading follows the local track direction, so the map tangent
is an absolute yaw reference. Measured agreement: within 3° for 73 % of the run
(median 1.3°); large only at corner apices.

It cuts a single clean lap to 7.2 m. **But on the full run it diverges to
>400 m.** Once position drift grows, the nearest-border tangent locks onto the
*wrong* parallel straight (the circuit has two long opposing straights ~600 m
apart) and resolves the 180° direction ambiguity incorrectly, creating a
runaway feedback loop. Error exceeds 10 m at 72 s and 100 m at 178 s —
*faster* than plain dead reckoning. A covariance gate does not save it because
the EKF stays over-confident while drifting.

Lesson: a map corrector with positional ambiguity is unsafe without a
trustworthy absolute-position source. Kept behind `tangent_yaw.enabled`.

### 4.2 Track-border matching — night detection is the bottleneck

Geometry is correct: projecting the surveyed borders into the front cameras
using the GT pose lands exactly on the visible track edge (see
`outputs/border_overlay.png`), and ground-projecting detected edges back to the
map is validated. The failure is **detection**: classical edge/gradient
detectors fire on barriers, grandstands and track lights — vertical structure
that violates the flat-ground assumption and projects to garbage, so the ICP is
dragged off and the score worsens to 55 m.

Lesson: this needs a learned drivable-surface segmentation front-end (GPU) to
emit only true ground-boundary pixels. The matcher and geometry are ready; only
`perception/border_detect.py` needs replacing. Kept behind
`border_match.enabled`.

### 4.3 Visual relocalization — descriptor quality

A relocalization DB is built offline from training GT (permitted for map
building) and queried causally. The CPU thumbnail descriptor retrieves to only
~18 m median across laps; off-the-shelf ImageNet ResNet features are *worse*
(~52 m) because dark, repetitive night scenes are near-identical in generic
feature space (top-1 cosine ≈ 0.93 everywhere). Neither is tight enough to
anchor position, and feeding noisy fixes in degrades the estimate.

Lesson: effective relocalization here needs a task-specific / stronger
descriptor (DINOv2 or a trained place-recognition head) on a GPU, with careful
validation that it transfers to the *different-session* hidden test (the brief
warns test lighting/speed differ). Wired behind `reloc.descriptor=dinov2`.

### 4.4 Residual drift of the default

Even with ZUPT bias calibration, ~1° of residual yaw error per lap bends the
trajectory, so position error grows along a lap and partially self-corrects at
the start/finish (see `outputs/val_lapA_error.png`). This is the fundamental
limit of camera-permitted dead reckoning without a working absolute visual fix,
and it is exactly what a GPU relocalization / segmentation front-end would
remove.

## 5. What a GPU run should add (next steps on the target workstation)

1. **Metric visual relocalization with ALIKED + LightGlue** (implemented;
   `metric_reloc.enabled=true`). This is the highest-expected-gain component
   and directly attacks the residual drift of §4.4. Why it should beat the
   coarse descriptors of §4.3: instead of a fuzzy nearest-pose lookup it
   performs *learned feature matching* (robust to night / low texture /
   viewpoint change) followed by essential-matrix / PnP geometry for a
   *metric* pose, giving a true absolute `(x, y, yaw)` fix rather than a
   ~18 m retrieval guess.

   Pipeline: keyframe DB of ALIKED features + training GT poses (built
   offline, permitted) → restrict candidates to the EKF-predicted
   neighbourhood (avoids wrong-section matches) → LightGlue match → relative
   pose → absolute fix → **EKF-gated** (inliers, reprojection, distance from
   prediction), so a bad match can only be rejected, never corrupt the state.

   *Caveat (measured concern, must be validated on the GPU):* the keyframes
   come from the training session; the hidden test is a *different* session
   with possibly different lighting/speed. Retrieval/matching may degrade
   there. The hard EKF gating bounds the downside, but cross-session transfer
   must be checked on the held-out laps before trusting it. This repo cannot
   measure its accuracy (CPU-only sandbox); it is wired and runnable, and the
   numbers come from the single-GPU workstation.

   *Benchmark LoMa / learned VO as alternatives*, not defaults: measure match
   quality, inlier ratio, localization error and runtime on identical frames
   and only promote a component that measurably improves the official score.

   **Triangulated 3D landmark map + PnP** (implemented; `pnp_reloc`). The
   strongest absolute fix: adjacent training keyframes are matched and
   triangulated (with their GT poses) into a sparse 3D point cloud
   (`scripts/build_landmark_map.py`); at test time the query's 2D matches to
   nearby landmarks are solved with PnP+RANSAC for a direct metric pose, with
   no accumulated VO drift. The triangulation + PnP geometry is unit-tested on
   synthetic data (`tests/test_landmarks.py`, sub-0.5 m). The pipeline prefers
   PnP when a landmark map is present and falls back to the essential-matrix
   relocalizer otherwise.

   **LoMa as a benchmarked alternative matcher** (implemented;
   `metric_reloc.matcher=loma-b|loma-l|loma-g`). `scripts/benchmark_matchers.py`
   compares ALIKED+LightGlue vs LoMa on identical cross-lap frame pairs
   (match count, inlier ratio, relative-yaw error, runtime). Promote a matcher
   only if it measurably wins -- do not assume the newer model is better.

2. **Learned drivable-surface segmentation** feeding `border_match`
   (implemented; `segmentation.enabled=true`, torchvision DeepLabV3). It
   replaces the classical night detector of §4.2 with clean road-boundary
   pixels, so the (validated) ground-projection + map-ICP geometry gets
   trustworthy input. A track-surface-specific fine-tuned segmenter is the
   recommended further upgrade.
3. With a reliable absolute fix in place, re-enable `tangent_yaw` safely for
   extra heading stability.

### Component map (the brief's four methods, all implemented)

| Brief method | Module | Default | Needs GPU |
|---|---|---|---|
| 1. Visual odometry (motion model) | `localization/motion.py` + EKF | on | no |
| 2. Track-border matching | `perception/border_matcher.py` (+ `segmentation.py`) | off | optional |
| 3. Reference-map relocalization | `perception/metric_reloc.py`, `landmarks.py` (ALIKED+LightGlue / LoMa / PnP) | off | yes |
| 4. Causal filter fusion | `localization/ekf.py` + `pipeline.py` | on | no |

All four are present and fuse through the one causal EKF; the GPU-dependent
absolute-correction paths are off by default and enabled per the README.

### Device safety

All learned components resolve their device through `perception/device.py`
(`resolve_device` / `to_device`): `runtime.device=auto|cpu|cuda|cuda:N`, with
automatic CPU fallback on any CUDA failure (unavailable, OOM, driver, Pascal
FlashAttention). The default pipeline needs no GPU and no external weights.
