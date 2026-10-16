# Validation results (tools/evaluate.py)

Produced by the organizer scoring script on this repo's default config.
Dataset: c0_2026-10-13_22-53-42_camloc (training run, 1194 s, night).

## Full run (submission.csv, default config)
```
poses submitted        11944   evaluated 11943
coverage (1 s bins)    100.0 %   (>= 90 % to be ranked)
position RMSE  [m]     37.443   <- SCORE (lower is better)
position median/p95    19.180 / 81.427
lateral RMSE   [m]     29.345
longitudinal RMSE [m]  23.256
yaw RMSE     [deg]     4.701
within 0.25 m          0.6 %
within 0.50 m          0.8 %
within 1.00 m          2.0 %
```

## Held-out validation lap A (680-833 s)
```
poses submitted        1530   evaluated 1530
coverage (1 s bins)    12.9 %   (>= 90 % to be ranked)
position RMSE  [m]     10.983   <- SCORE (lower is better)
position median/p95    5.905 / 21.557
lateral RMSE   [m]     8.978
longitudinal RMSE [m]  6.326
yaw RMSE     [deg]     1.285
within 0.25 m          2.7 %
within 0.50 m          7.2 %
within 1.00 m          12.2 %
NOT RANKED: coverage below threshold
```

## Held-out validation lap B (953-1063 s)
```
poses submitted        1100   evaluated 1100
coverage (1 s bins)    9.3 %   (>= 90 % to be ranked)
position RMSE  [m]     9.627   <- SCORE (lower is better)
position median/p95    6.658 / 18.384
lateral RMSE   [m]     7.944
longitudinal RMSE [m]  5.438
yaw RMSE     [deg]     1.357
within 0.25 m          1.5 %
within 0.50 m          7.1 %
within 1.00 m          14.7 %
NOT RANKED: coverage below threshold
```
