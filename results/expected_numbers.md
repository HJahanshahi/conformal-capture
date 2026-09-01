# Expected values

Every number published in the article, for checking a reproduction run.
Baseline arm: `path2_winning_v2_results.json`. Conformal arm:
`path2_conformal_v3_results.json`. Both arms complete all 135 runs
(45 trajectories x 3 sensor-noise seeds); statistics are over all 135.

Deployed configuration: exact reduced free-floating dynamics, plant advanced
in 20 substeps per 10 Hz control period with zero-order-hold torque,
K_p^ori = 10 modulated to 1 across the 20-40 deg band and evaluated at every
control step, K_d = 0.6 K_p, blend window 1.5 s, torque limit 20 N m,
rejection threshold 40 deg, contingency after 4 consecutive rejections.

## Table 2: overall rendezvous accuracy

| metric | baseline | conformal |
|---|---|---|
| mean position (cm) | 6.71 | 5.71 |
| median position (cm) | 5.18 | 4.37 |
| 95th percentile position (cm) | 15.32 | 16.48 |
| mean orientation (deg) | 9.81 | 7.46 |
| median orientation (deg) | 5.06 | 2.48 |
| 95th percentile orientation (deg) | 31.69 | 27.88 |
| capture-ready (< 10 cm, < 15 deg) | 78% | 89% |
| strict capture (< 5 cm, < 5 deg) | 30% | 49% |

## Table 4: joint threshold grid (baseline % / conformal %)

5/5: 30 / 49 · 5/10: 39 / 60 · 10/10: 67 / 84 · 10/15: 78 / 89 ·
15/15: 83 / 91 · 15/20: 87 / 91

## Figure 3: cumulative distribution anchors

position < 5 cm: 47 -> 64 · position < 10 cm: 86 -> 92 ·
orientation < 5 deg: 50 -> 70 · orientation < 15 deg: 84 -> 93

## Table 3: per-tumble-rate breakdown, median (IQR)

| bin | n | baseline position | conformal position | baseline orientation | conformal orientation |
|---|---|---|---|---|---|
| low (< 3 deg/s) | 18 | 4.60 (3.06-6.26) | 3.93 (2.36-5.02) | 3.63 (2.04-6.13) | 1.52 (1.25-4.47) |
| mid (3-10) | 48 | 5.76 (3.95-11.21) | 4.45 (2.88-6.90) | 6.94 (3.68-16.25) | 2.19 (1.49-5.00) |
| high (10-20) | 48 | 4.54 (3.34-6.58) | 4.36 (2.91-5.68) | 4.30 (2.68-8.60) | 2.89 (1.72-5.52) |
| extreme (> 20) | 21 | 6.92 (4.60-9.06) | 4.37 (2.29-8.45) | 7.80 (3.61-10.23) | 4.21 (2.29-9.12) |

Median improvement by bin: orientation -58%, -69%, -33%, -46%;
position -15%, -23%, -4%, -37%. Mean orientation in the extreme bin is
higher for the conformal arm (16.19 against 10.31 deg) because one run in
that bin of 21 ends 173 deg misaligned.

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
131.5 / 166.6 deg. The orientation bound crosses the 40 deg rejection
threshold at a lookahead of about 1.11 s.

## Table 5: mechanism component study (validation split, 15 traj x 2 seeds)

| configuration | capture-ready | strict | mean ori (deg) | p95 ori (deg) |
|---|---|---|---|---|
| no mechanisms | 87% | 23% | 11.66 | 41.4 |
| deployed system | 87% | 50% | 7.24 | 27.2 |
| gain set once per plan | 70% | 43% | 21.06 | 103.4 |
| no rejection or contingency | 43% | 7% | 24.25 | 86.8 |

## Mechanism statistics (conformal arm, test set)

* Rejections: mean 2.30 per run, 88% of runs at most five, maximum 11.
* Rejection strata (capture-ready): 0 rejections n = 36, 92%;
  1-3 rejections n = 67, 96%; 4 or more n = 32, 72%.
* Contingency: 17 activations across 15 runs; those runs reach 47%
  capture-ready.
* Calibrated bound at the last accepted replan: at most 39.9 deg in every run.
* Torque saturation: 5.96% of control steps clip at 20 N m; 50 runs with at
  least one clip; median run zero.
* Certificate violations: position 1 of 135 (0.7%), orientation above 40 deg
  4 of 135 (3.0%).
* Run-by-run comparison: 21 encounters the baseline fails are captured, and
  6 encounters the baseline handles are lost.

## Showcase runs (Figures 7-10)

* Success: trajectory 13, seed 2, final 2.13 cm / 3.3 deg, 2 rejections.
* Difficult: trajectory 42, seed 1, final 25.00 cm / 11.6 deg, 10 rejections,
  2 contingency activations (baseline on the same run: 28.5 cm).

## Ablations

**Lookahead schedule** (`path2_tau_schedule.py`): 135 of 135 complete;
capture-ready 86% (deployed 89%); medians 4.33 cm / 2.66 deg; 95th percentile
orientation 34.95 deg (deployed 27.88); mean maximum bound 73.7 deg (deployed
82.9); rejections 2.39; 18 contingency activations in 16 runs.

**Out of distribution, 30-50 deg/s** (`ood_ablation.py`): conformal completes
135 of 135 and reaches 77% capture-ready with medians 6.07 cm / 6.17 deg;
baseline completes 133 of 135 and reaches 59% with medians 7.44 cm /
8.31 deg. Coverage of the deployed bounds at 1 s lookahead falls to 30.4%
(orientation) and 66.7% (position).

**Exact orientation law** (`c6_exact_xi.py`): the omitted Jacobian-rate term
measures 0.3-13 rad/s^2 in nominally tracking runs, where including it
usually improves terminal errors slightly, and grows to roughly 1500 rad/s^2
during rejection churn, where feeding it back destabilises two runs that
complete under the deployed law (86 cm and 24 cm). Only the deployed-law
column is expected to match exactly.

## Notes on reproducibility

The calibration, benchmark, and the first two ablations are deterministic
given the shipped datasets and weights. Figures 2-6 are computed from the two
result files; Figures 7-10 re-simulate the two showcase runs and reproduce
their terminal errors exactly (2.1 cm and 25.0 cm in Figure 7).
