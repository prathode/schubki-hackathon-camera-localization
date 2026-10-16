# Presentation outline (≤ 6 slides, ≤ 5 minutes)

### Slide 1 — Problem & constraints
- Camera-only localization of a race car on Yas Marina North, 10 Hz, causal.
- No GNSS/LiDAR/radar/GT at inference (only the first pose). Score = position
  RMSE; lateral error + yaw matter most for the racing line.
- Hard parts: night, motion blur, low-texture asphalt, repetitive features,
  two long opposing straights (weak along-track observability).

### Slide 2 — System diagram
- EKF over `[x, y, yaw]` in the map frame.
- Predict from permitted motion signals (`vx, vy, wz`) at 106 Hz.
- Zero-velocity yaw-rate bias calibration at the stationary start (ZUPT).
- One gated pose per image set → 10 Hz, 100 % coverage; never drops a frame.
- Optional gated correctors (map-tangent yaw, border ICP, relocalization).
- (Use the ASCII diagram in docs/APPROACH.md §2.)

### Slide 3 — Results
- Full run: **37.4 m pos RMSE, 4.70° yaw, 100 % coverage, 3.1 ms/frame (CPU)**.
- Validation laps: ~10 m pos RMSE, **~1.3° yaw**.
- Trajectory plot: estimate vs GT on the track map (`outputs/val_lapA_trajectory.png`).

### Slide 4 — What moved the needle
- Verified data first: `wz_radps` is the real yaw rate; `yaw_vel_rad` is
  unusable; speed is accurate to 0.3 %.
- ZUPT yaw-rate bias calibration: 64 m → 37 m, 8.5° → 4.7° — causal, zero
  overfit, generalizes.

### Slide 5 — Failure cases & lessons (the honest part)
- Map-tangent yaw: great on one lap (7.2 m) but diverges globally (>400 m) via
  wrong-straight lock — ambiguity needs a trusted position source.
- Border matching: geometry validated (overlay image), but night edge
  detection is too noisy → needs learned segmentation.
- Relocalization: off-the-shelf descriptors too coarse (~18–52 m) on repetitive
  night scenes → needs a task-specific descriptor.
- Error-over-time plot showing drift ↔ yaw accumulation
  (`outputs/val_lapA_error.png`).

### Slide 6 — Next steps / GPU path
- DINOv2 relocalization + learned drivable-surface segmentation on the
  single-GPU workstation (P5200, 16 GB), then safely re-enable tangent-yaw.
- All three are implemented behind config flags; default stays robust and
  reproducible.

**Artifacts to show:** trajectory video/plot, error-over-time plot, border
overlay (geometry check), evaluator output.
