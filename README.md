# Camera-based localization on a race track (Yas Marina)

Our solution for the Constructor Labs challenge at the National AI Hackathon 2026: estimate the race
car's pose (x, y, yaw in the `map` frame) at 10 Hz from its cameras, causally, starting from the first
pose of the run. The task description and data format are in [DATASET.md](DATASET.md).

## Results

Validation laps 3 and 7 of the training run. They were never used to build maps or to choose anything,
and were scored with the organizers' `tools/evaluate.py`.

| Version | Cameras | Lap 3 RMSE | Lap 7 RMSE (40 m/s) | Lateral RMSE | Time / frame |
|---|---|---|---|---|---|
| Car-state dead reckoning only | — | 43.6 m | 102.0 m | 33.2 / 79.2 m | <1 ms |
| Car state + track map (particle filter) | — | 4.02 m | 4.24 m | 3.09 / 2.61 m | 2 ms |
| + SIFT landmark relocalization (precomputed) | fl | 0.41 m | 0.43 m | 0.18 / 0.21 m | ~150 ms |
| + ALIKED + LightGlue, online | fl | 0.46 m | 0.49 m | 0.18 / 0.21 m | 102 ms |
| + rear-left camera, parameters chosen on held-out laps 2 and 6 | fl + rl | 0.21 m | 0.27 m | 0.13 / 0.11 m | 201 ms |
| **Current: + RoMa fallback when LightGlue fails (chosen on laps 2 and 6)** | **fl + rl** | **0.21 m** | **0.27 m** | **0.13 / 0.11 m** | **198 ms** |

On normal (night) images the RoMa fallback is rarely needed (104 calls in the whole run) and changes
nothing beyond noise (0.211 / 0.273 → 0.207 / 0.265 m). It is there for appearance changes:

### Robustness to appearance changes (synthetic)

Online runs on the validation laps with every image transformed (`camloc/augment.py`), started 20 s
before the lap from our own causal estimate. Night maps, no retraining. The settings were chosen on
laps 2 and 6 only; this table is the single run on laps 3 and 7.

| Images | Lap 3 RMSE | Lap 7 RMSE | Within 1 m (3 / 7) | Time / frame |
|---|---|---|---|---|
| brighter, overexposed, low contrast, lamps removed, motion blur (no RoMa) | 0.20–0.29 m | 0.20–0.29 m | ~100 % | ~200 ms |
| inverted (negative), no RoMa | 1.91 m | 3.90 m | 44 / 5 % | 177 ms |
| **inverted, with RoMa fallback** | **0.40 m** | **0.58 m** | **99.8 / 94 %** | **372 ms** |

Inverted images are a crude stand-in for daylight reversing the night shading (LightGlue then finds a
fix in only ~12 % of the frames). This is not a test on real daylight images, which we do not have.

![Validation lap 7, inverted images: error with and without RoMa, and which fixes the filter received](docs/figures/lap7_invert_roma.png)

What RoMa does on one frame (lap 7, 48 s): the inverted image is matched densely to the stored night
keyframe; the matches lie on fixed structures (stands, light rows, bridge), and PnP on them gives the pose.

![RoMa matches between an inverted image and a night keyframe, with the certainty map](docs/figures/roma_match_example.png)

Timing is per image set on an NVIDIA GeForce RTX 2070 under WSL2. Coverage is 100 % in all versions.
Full analysis, failure cases and next steps: [RESULTS17.10morning.md](RESULTS17.10morning.md);
method and test procedure: [METHOD.md](METHOD.md).

## Method

1. **Motion model:** car-state speed and yaw rate (`vx`, `vy`, `wz` from `/constructor0/state_estimation`,
   allowed by the task) integrated between image stamps.
2. **Track-map prior:** the car must be inside the track or pit-lane borders, heading along them.
3. **Landmark relocalization (slide 16, method 3a):** keyframe maps built from the training laps by
   triangulating ALIKED keypoints with the GT poses. Online, each image is matched with LightGlue to the
   keyframes nearest the predicted pose, and PnP gives an absolute pose fix.
4. **Particle filter (slide 17, method 4):** fuses 1–3; outlier fixes are gated. Output at every image set.
5. **RoMa fallback (`--roma-fallback 1`):** when LightGlue gives no accepted fix, at most every 0.5 s,
   the front-left image is matched densely with RoMa to the nearest keyframe image; the warp is read at
   the keyframe's 3D keypoints and PnP gives a fix (looser sigmas than LightGlue fixes).

Ground truth is used only to build the maps and to score; at inference only the first pose of the run is used.

**Pretrained models used:** ALIKED (`aliked-n16`) and LightGlue (`aliked_lightglue`) from
[cvg/LightGlue](https://github.com/cvg/LightGlue); RoMa outdoor (`romatch` 0.1.2,
[Parskatt/RoMa](https://github.com/Parskatt/RoMa)) for the fallback. No external datasets.

## Data split

| Use | Laps (see `cache/laps.csv`) |
|---|---|
| Build maps | training laps 0 (out-lap), 1, 2, 4, 5, 6, 8 (in-lap) |
| Choose cameras and parameters | laps 2 and 6, using maps built **without** them (`maps_no26/`) |
| Validation (final report only) | laps 3 and 7 |

## Setup

```bash
conda create -n camloc python=3.12 -y && conda activate camloc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git
pip install romatch==0.1.2      # RoMa fallback
```

Put the dataset (`c0_2026-10-13_22-53-42_camloc_0.mcap`) in the repository root. It is not in git
(licence: hackathon use only).

## Reproduce

```bash
BAG=c0_2026-10-13_22-53-42_camloc_0.mcap
# 1. training data (once): calibration, borders, state, sample images, GT, lap split
python tools/export_cache.py $BAG cache --img-every 10
python tools/extract_gt.py $BAG gt.csv
python scripts/split_laps.py gt.csv cache
# 2. keyframe maps from the training laps (~11 min per camera)
python scripts/build_kf_map.py $BAG cache gt.csv maps/kf_fl.npz --cam fl
python scripts/build_kf_map.py $BAG cache gt.csv maps/kf_rl.npz --cam rl
python scripts/export_kf_images.py $BAG cache gt.csv maps/kf_fl.npz --cam fl   # keyframe images for RoMa
# 3. localization: one command, raw mcap -> submission CSV (causal, online)
python run.py $BAG out/final_roma/final.csv --cams fl,rl --pf-config configs/fl_rl.json --roma-fallback 1
# 4. score (full run and each validation lap)
python scripts/eval_laps.py gt.csv out/final_roma/final.csv cache
# 5. robustness tests on the validation laps (synthetic appearance changes)
python scripts/stress_test.py $BAG --conds invert --roma 0,1 --out out/stress_val_roma
```

RoMa settings were chosen on laps 2 and 6 with `maps_no26` (`out/roma_tune/summary*.csv`):
certainty threshold 0.2 → 0.05 and period 1 s → 0.5 s took inverted lap 6 from 2.03 m to 0.38 m.

`scripts/overnight.sh` runs the whole held-out selection (maps without laps 2 and 6, online run with
three cameras, parameter and camera sweep on laps 2 and 6) and then the final run.

## Where the results are

The submission file and the result files are committed under `results/` (copied from `out/`, which is
not in git). Maps, caches and the dataset are **not in git** (size, dataset licence).

| Path | Contents |
|---|---|
| `results/submission.csv` | **submission file** (`timestamp_ns,x,y,yaw`), RoMa fallback on; same as `results/final_roma/final.csv` |
| `results/final_roma/final_eval.txt` | `evaluate.py` report: full run and validation laps 3, 7 |
| `results/final_roma/final_timing.json` | mean / median / p95 time per frame, GPU |
| `results/final_roma/final_log.csv` | every relocalization fix per camera and frame (`fl_roma` = RoMa) |
| `results/final_no_roma/` | previous version without RoMa (same files, plus `final_overview.png`, `selected_config.json`) |
| `results/stress/summary.csv` | photometric stress tests on laps 3, 7 (no RoMa) |
| `results/stress_val_roma/summary.csv` | inverted images on laps 3, 7, with and without RoMa |
| `results/roma_tune/summary*.csv` | RoMa settings compared on laps 2, 6 |

Running the commands above writes the same files to `out/`. Other outputs stay on the local disk only:

| Path | Contents |
|---|---|
| `out/roma/exp_roma.csv` | offline LightGlue vs RoMa matcher comparison on laps 2, 6 |
| `out/lolo/run3_log_sweep.csv` | camera-set × parameter sweep on held-out laps 2 and 6 |
| `out/lolo/run3_log_best.json` | the selected configuration (copied to `configs/fl_rl.json`) |
| `out/map_pf*`, `out/reloc_pf*`, `out/aliked_online*` | earlier versions (rows 2–4 of the results table) |
| `out/variants/results.jsonl` | speed-up variants scored on laps 2 and 6 (in progress) |
| `maps/kf_*.npz` | keyframe maps from all training laps (0.3–0.5 GB each) |
| `maps/kf_fl_img/`, `maps_no26/kf_fl_img/` | keyframe images (half resolution) for RoMa |
| `maps_no26/kf_*.npz` | keyframe maps without laps 2 and 6, for tuning |
| `cache/` | data exported from the bag, lap split, sample images |

## Repository layout

| Path | Contents |
|---|---|
| `run.py` | online pipeline: bag → submission CSV, with per-frame timing |
| `camloc/` | library: `pf.py` particle filter, `trackmap.py` track prior, `motion.py` odometry, `features.py` ALIKED + LightGlue, `kfmap.py` keyframe map + PnP, `camera.py` calibration, `body_masks.py` car-body masks, `roma_fallback.py` RoMa fallback, `augment.py` synthetic appearance changes |
| `scripts/` | map building, lap split, evaluation, parameter sweeps, profiling, experiments |
| `tools/` | organizers' `evaluate.py` and `extract_gt.py`, plus our export and plotting tools |
| `configs/fl_rl.json` | cameras and filter parameters selected on laps 2 and 6 |

## Open items

- Runtime is ~200 ms/frame on the RTX 2070 (~370 ms when RoMa is active on every LightGlue failure);
  full runtime points need ≤ 100 ms. Not pursued further: the test GPU is expected to be much faster.
- The maps are night-only; a daylight test run is the main risk. The RoMa fallback helps strongly on
  synthetic changes, but real daylight is not verified.
