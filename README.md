# Camera-Based Race-Track Localization

Causal, **camera-only** localization of an autonomous race car on the Yas
Marina North circuit — Constructor Labs challenge, National AI Hackathon 2026
(SchubKI).

It estimates the pose of `constructor0/base_link` in the `map` frame
(`timestamp_ns, x, y, yaw`) at camera rate (10 Hz) from the onboard cameras,
their calibration, the track-border map, and the **permitted** vehicle-motion
signals only. No GNSS, LiDAR, radar, or ground-truth pose is used at inference
(only the single permitted initial pose).

---

## TL;DR — one command

```bash
# 1. environment
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 2. produce the submission CSV from the raw MCAP
python run.py --mcap /path/to/run.mcap --out outputs/submission.csv
```

That writes `outputs/submission.csv` with the header
`timestamp_ns,x,y,yaw` — one pose per image set, 10 Hz, 100 % coverage.

Score it with the organizer's evaluator:

```bash
python3 tools/evaluate.py gt.csv outputs/submission.csv   # tools/ is in the dataset zip
```

---

## Results (held-out validation, this repo's default config)

Default = motion EKF with zero-velocity yaw-rate bias calibration. Fully
causal, camera-permitted, **no ground truth at inference**, no training.

| Run | Pos RMSE | Lateral RMSE | Yaw RMSE | Coverage | Runtime |
|---|---|---|---|---|---|
| Full run (1194 s) | **37.4 m** | 29.3 m | 4.70° | 100 % | **3.1 ms/frame** |
| Validation lap A (680–833 s) | 11.0 m | 9.0 m | 1.29° | n/a* | 3.1 ms/frame |
| Validation lap B (953–1063 s) | 9.6 m | 7.9 m | 1.36° | n/a* | 3.1 ms/frame |

\* A single ~150 s lap is a window of the 1194 s GT, so the evaluator's
coverage metric (fraction of the whole GT window covered) is low by
construction; the per-lap RMSE is the meaningful number there. Coverage is
100 % on the full run.

Runtime measured on the **CPU-only** development sandbox (no GPU); well under
the 100 ms camera period, so real-time on the target single-GPU workstation.

See `docs/APPROACH.md` for the full method, ablations, and **failure analysis**
(including why border matching and relocalization are off by default).

---

## What the system does

1. **Motion prediction (the backbone).** Integrate the permitted body-frame
   velocity (`vx_mps`/`vy_mps` or `v_mps`+`beta_rad`) and yaw rate
   (`wz_radps`) between 106 Hz state samples. An EKF carries the pose
   `[x, y, yaw]` in the map frame.
2. **Zero-velocity yaw-rate bias calibration.** The recording starts ~5 s
   before first motion. While stationary, any non-zero yaw rate is pure gyro
   bias; we average it and subtract it forward. This is standard ZUPT
   calibration (a sensor constant, not ground truth) and cuts full-run yaw
   drift from 8.5° to 4.7° and position RMSE from 64 m to 37 m.
3. **Causal output.** One pose is emitted for every image set, so coverage is
   100 %. When a (optional) visual correction is unavailable or rejected, the
   filter keeps predicting — it never drops a frame.

Optional, **disabled-by-default** correctors (see `docs/APPROACH.md`):

- **Metric visual relocalization (ALIKED + LightGlue)** — match the live
  frame to training keyframes, recover relative pose (essential matrix /
  PnP), and apply an absolute `(x, y, yaw)` fix. This is the primary path to
  beat the motion baseline; it needs a GPU and the feature DB. See
  "GPU relocalization" below.
- **Map-tangent yaw correction** — anchors heading to the track direction.
  Helps on a clean lap but diverges globally without a reliable position
  source; kept behind a flag.
- **Track-border matching** — geometry is validated (ground projection + map
  ICP), but classical night-time edge detection is too noisy; ready for a
  learned segmentation front-end on a GPU.
- **Coarse retrieval relocalization** — thumbnail (CPU) or DINOv2 (GPU)
  global descriptor; a cheaper coarse prior than the metric path.

### Device handling (GPU-optional, fail-safe)

`runtime.device` accepts `auto` (default), `cpu`, `cuda`, or `cuda:N`. All
learned models are moved through a single `to_device()` helper that **falls
back to CPU automatically** if CUDA is unavailable or the GPU move fails
(driver mismatch, OOM, Pascal/FlashAttention quirks) — the pipeline never
crashes because of the GPU. On a Pascal card (P5200) FlashAttention is
auto-disabled and mixed precision is left off.

### GPU relocalization — on your workstation

Install the optional deps (CUDA torch + LightGlue, which bundles ALIKED;
optionally LoMa):

```bash
pip install "lightglue @ git+https://github.com/cvg/LightGlue.git"
# optional alternative matcher:
# pip install "loma @ git+https://github.com/davnords/LoMa.git"
```

**Option 1 — essential-matrix relocalization (ALIKED + LightGlue):**

```bash
python scripts/build_feature_db.py --mcap "$MCAP" \
    --out outputs/feature_db.pkl --set runtime.device=cuda \
    --exclude-start-s 680 --exclude-end-s 1063
python run.py --mcap "$MCAP" --out outputs/submission.csv \
    --set metric_reloc.enabled=true --set runtime.device=cuda
```

**Option 2 — PnP against a triangulated 3D landmark map (strongest):**

```bash
python scripts/build_landmark_map.py --mcap "$MCAP" --cam fl \
    --out outputs/landmarks_fl.npz --set runtime.device=cuda \
    --exclude-start-s 680 --exclude-end-s 1063
python run.py --mcap "$MCAP" --out outputs/submission.csv \
    --set pnp_reloc.enabled=true \
    --set 'pnp_reloc.map_paths={fl: outputs/landmarks_fl.npz}' \
    --set runtime.device=cuda
```

**Option 3 — learned border segmentation (lateral/yaw map fix):**

```bash
python run.py --mcap "$MCAP" --out outputs/submission.csv \
    --set border_match.enabled=true --set segmentation.enabled=true \
    --set runtime.device=cuda
```

**Benchmark matchers before promoting one:**

```bash
python scripts/benchmark_matchers.py --mcap "$MCAP" \
    --matchers aliked_lightglue loma-b --cam fl --pairs 40 \
    --set runtime.device=cuda      # -> reports/matcher_benchmark.csv
```

These paths can be combined (e.g. PnP + segmentation border matching); all
corrections fuse through the one causal EKF and are hard-gated.

---

## Repository layout

```
run.py                     # single entry point: MCAP -> submission CSV
configs/default.yaml       # all thresholds, topics, allowlist, toggles
requirements.txt
src/camloc/
  config.py                # YAML config + dotted overrides
  data/
    mcap_source.py         # causal MCAP reader; STRICT state allowlist
    calibration.py         # intrinsics + extrinsics (base_link<->camera)
    track.py               # track-border map + KD-tree
    image_set.py           # causal 7-camera image-set assembler
  calibration/projection.py# ground projection + map<->image transforms
  perception/
    border_detect.py       # classical night border detector (optional)
    border_matcher.py      # projected-border ICP -> lateral/yaw (optional)
    descriptors.py         # thumbnail (CPU) / DINOv2 (GPU) descriptors
    relocalizer.py         # retrieval against a training DB (optional)
  localization/
    motion.py              # permitted-signal motion model + bias
    ekf.py                 # 2D pose EKF with gated updates
    tangent_yaw.py         # map-tangent yaw corrector (optional)
    pipeline.py            # predict/update/emit orchestration
  submission/writer.py     # strict Option-A CSV writer
  evaluation/plots.py      # trajectory + error-timeline plots
scripts/
  build_reloc_db.py        # build the relocalization DB from training GT
  make_plots.py            # validation visualizations
tests/                     # causality + GT-isolation + geometry tests
docs/APPROACH.md           # method, ablations, failure analysis
docs/PRESENTATION.md       # <=6-slide outline
```

---

## Reproducing the reported numbers

```bash
. .venv/bin/activate
MCAP=/path/to/c0_2026-10-13_22-53-42_camloc_0.mcap

# ground truth for scoring (organizer tool, from the dataset zip)
python3 tools/extract_gt.py "$MCAP" outputs/gt.csv

# full run (default config)
python run.py --mcap "$MCAP" --out outputs/full.csv
python3 tools/evaluate.py outputs/gt.csv outputs/full.csv

# held-out validation laps (window-start pose used as the permitted init)
python run.py --mcap "$MCAP" --out outputs/val_lapA.csv --start-s 680 --end-s 833
python run.py --mcap "$MCAP" --out outputs/val_lapB.csv --start-s 953 --end-s 1063
python3 tools/evaluate.py outputs/gt.csv outputs/val_lapA.csv
python3 tools/evaluate.py outputs/gt.csv outputs/val_lapB.csv

# visualization
python scripts/make_plots.py --mcap "$MCAP" --gt outputs/gt.csv \
    --sub outputs/val_lapA.csv --prefix outputs/val_lapA
```

The two validation laps (680–833 s and 953–1063 s) are the held-out laps; the
default pipeline has no trained parameters tuned on them.

---

## Compliance (disqualification guards)

- **Causality.** Output at time `t` uses only data stamped `<= t`. The
  image-set assembler never bundles a future frame (tested).
- **Ground-truth isolation.** `src/camloc/data/mcap_source.py` defines an
  explicit `ALLOWED_STATE_FIELDS` allowlist; the reader copies only those
  fields out of `state_estimation`. `/tf` and absolute pose / `sn_map_state`
  are never read on the inference path. Automated tests fail if a prohibited
  field appears in the stream (`tests/test_compliance.py`).
- **Permitted initial pose.** The first pose of the run (from `/tf`) seeds the
  filter and nothing else. For a mid-run validation lap, the pose at the
  window start is the analogue, used only to initialize.
- **Coverage.** One pose per image set → 100 % of 1 s bins covered.
- **Reproducibility.** Pinned `requirements.txt`, a single `run.py`, no
  network or hidden state at inference. The default path uses no GPU and no
  external model weights.

---

## Hardware & runtime

- Developed and measured on a **CPU-only** sandbox (Python 3.11): 3.1 ms/frame
  for the default pipeline.
- Target reproduction machine: single-GPU workstation. The optional GPU
  relocalization path (DINOv2, a **declared external pretrained model**) is
  sized for a 16 GB Pascal-class GPU (e.g. Quadro P5200); it avoids
  Ampere-only features.

## External models / datasets

- **Default pipeline: none.** No pretrained weights, no external datasets.
- **Optional:** DINOv2 (`facebookresearch/dinov2`, pretrained, frozen feature
  extractor) for the relocalization descriptor — declared here, downloaded on
  demand only when `reloc.descriptor=dinov2` is set.

## Licence / data

Dataset is for National AI Hackathon 2026 use only (Constructor Labs); not
redistributed in this repo. The dataset and large artifacts are gitignored.
