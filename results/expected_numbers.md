# Expected values

Every number published in the article, for checking a reproduction run.
Baseline arm: `path2_winning_v2_results.json`. Conformal arm:
`path2_conformal_v3_results.json`. Both arms complete all 135 runs
(45 trajectories x 3 sensor-noise seeds); statistics are over all 135.

## Table 2: overall rendezvous accuracy

| metric | baseline | conformal |
|---|---|---|
| mean position (cm) | 5.64 | 4.70 |
| median position (cm) | 4.84 | 4.04 |
| 95th percentile position (cm) | 10.94 | 9.55 |
| mean orientation (deg) | 11.88 | 12.40 |
| median orientation (deg) | 7.22 | 4.02 |
| 95th percentile orientation (deg) | 39.18 | 59.40 |
| capture-ready (< 10 cm, < 15 deg) | 77% | 82% |
| strict capture (< 5 cm, < 5 deg) | 24% | 47% |

## Table 4: joint threshold grid (baseline % / conformal %)

5/5: 24 / 47 · 5/10: 39 / 59 · 10/10: 65 / 79 · 10/15: 77 / 82 ·
15/15: 81 / 83 · 15/20: 84 / 88

## Figure 3: cumulative distribution anchors

position < 5 cm: 51 -> 67 · orientation < 5 deg: 35 -> 61 ·
position < 10 cm: 93 -> 95 · orientation < 15 deg: 81 -> 83

## Table 3: per-tumble-rate breakdown, median (IQR)

| bin | n | baseline position | conformal position | baseline orientation | conformal orientation |
|---|---|---|---|---|---|
| low (< 3 deg/s) | 18 | 4.45 (2.94-5.80) | 3.93 (3.27-4.73) | 3.73 (1.86-7.28) | 1.95 (1.54-3.66) |
| mid (3-10) | 48 | 4.99 (3.69-7.66) | 4.07 (2.49-6.25) | 7.58 (4.27-13.48) | 4.62 (2.77-8.90) |
| high (10-20) | 48 | 4.79 (3.41-6.42) | 3.45 (2.60-5.38) | 7.23 (4.32-11.19) | 3.66 (1.85-6.21) |
| extreme (> 20) | 21 | 5.57 (4.21-6.61) | 4.48 (2.55-6.24) | 8.12 (6.32-13.01) | 5.90 (3.81-11.38) |

Median orientation improvement by bin: -48%, -39%, -49%, -27%.
Mean orientation, extreme versus mid bin: 10.03 vs 13.63 deg (baseline),
12.59 vs 13.81 deg (conformal).

## Table 1: calibration constants and coverage

Hierarchical repeated subsampling (B = 500) over 200 independently generated
calibration trajectories, grasp-point position scores, alpha = 0.10.

| lookahead (s) | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|
| orientation bound (deg) | 18.2 | 34.4 | 60.7 | 87.6 |
| position bound (cm) | 14.9 | 30.0 | 61.6 | 89.9 |
| orientation coverage (%) | 91.4 | 91.3 | 91.2 | 91.6 |
| position coverage (%) | 90.4 | 90.4 | 90.4 | 90.7 |

Calibration samples: 4800 / 4800 / 3600 / 3000 (200 trajectories each).
Per-trajectory-maximum variant (reported, not deployed): 40.7 / 84.5 /
131.5 / 166.6 deg, with 93-98% uniform-per-trajectory coverage for
orientation and 84% for position.
The orientation bound crosses the 40 deg rejection threshold at a lookahead
of about 1.11 s.

## Mechanism statistics (conformal arm)

* Rejections: mean 2.28 per run, 89% of runs at most five, maximum 12.
* Rejection strata (capture-ready rate): 0 rejections n = 35, 83%;
  1-3 rejections n = 67, 91%; 4 or more n = 33, 64%.
* Contingency: 13 activations across 11 runs; those runs reach 18%
  capture-ready; worst-case position error over all 135 runs is 24.2 cm.
* Calibrated bound at the last accepted replan: at most 39.9 deg in every run.
* Torque saturation: 341 of 5400 steps (6.31%); 42 runs with at least one
  clip; median run zero.
* Terminal orientation error above 40 deg: 8 of 135 runs (5.9%).

## Showcase runs (Figures 7-10)

* Success: trajectory 13, seed 2, final 0.73 cm / 6.7 deg.
* Failure: trajectory 33, seed 1, final 9.32 cm / 161.8 deg, with 12
  threshold exceedances and 2 contingency activations.

## Ablations

**Lookahead schedule** (`path2_tau_schedule.py`): 134 of 135 runs complete;
capture-ready 81% (vs 82%); strict 46% (vs 47%); medians 3.83 cm / 4.11 deg;
rejections 2.19 per run; 10 contingency activations in 9 runs; mean maximum
bound seen 71.9 deg (vs 87.6 deg calibrated at 2 s).

**Out of distribution, 30-50 deg/s** (`ood_ablation.py`): both arms complete
135 of 135. Conformal: 67% capture-ready, median 5.37 cm / 9.51 deg, 95th
percentile 13.42 cm / 82.97 deg, worst-case position 22.4 cm, rejections 1.65
per run. Baseline: 62% capture-ready, median 6.13 cm / 10.64 deg, 95th
percentile 12.12 cm / 41.22 deg. Coverage of the deployed bounds at 1 s
lookahead falls to 30.4% (orientation) and 66.7% (position).

**Exact orientation law** (`c6_exact_xi.py`): on the ten-run subset the
omitted Jacobian-rate term measures 0.3-9 rad/s^2 in nominally tracking
terminal windows and 200-405 rad/s^2 during saturation and rejection churn.
The deployed law reproduces bit-for-bit; the exact-law variant is unstable in
the high-term cases and its outcomes vary between platforms, so only the
deployed-law column is expected to match exactly.

## Notes on reproducibility

The calibration, benchmark, and first two ablations are deterministic given
the shipped datasets and weights. Figures 2-6 are computed from the two
result files; Figures 7-10 re-simulate the two showcase runs.
