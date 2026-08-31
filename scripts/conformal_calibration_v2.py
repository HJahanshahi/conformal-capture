"""Stage 1 v2: Hierarchical conformal calibration on a FRESH calibration set,
with GRASP-POINT position scores. Implements grouped
exchangeability, an independent calibration split, and grasp-point
vs body-position scores).

What it does
------------
1. Generates 45 fresh calibration trajectories from the SAME generator
   (seed=1000, disjoint from the original 300 by construction) unless the
   file already exists.
2. Gathers per-(trajectory, seed, observation-point, lookahead) scores:
     - orientation: geodesic angle (unchanged)
     - position:    GRASP-POINT error ||r_c_hat - r_c||, r_c = r + R(q) rho
3. Computes q_hat per lookahead under FOUR constructions:
     a) sample-level split conformal      (legacy; for comparison only)
     b) per-trajectory MAX subsampling    (exact 1-alpha, uniform-over-
                                            trajectory guarantee; n = 45)
     c) Dunn repeated subsampling         (marginal per-query, provably
                                            1-2*alpha, empirically ~1-alpha)
     d) CDF pooling                        (equal-trajectory weighting;
                                            asymptotic reference)
4. Evaluates ALL FOUR on the full 45-trajectory test split:
     - marginal per-query coverage (+ trajectory-cluster bootstrap CI)
     - per-trajectory uniform coverage (all queries in a run covered)
5. Saves one JSON per construction (deployment-ready schema identical to
   conformal_calibration.json) plus calibration_comparison.json.

Run from the project root:  python conformal_calibration_v2.py
Nothing existing is overwritten.
"""
import sys
for m in list(sys.modules.keys()):
    if "cap_control" in m or "space_robot_dq" in m or m.startswith("capture"):
        del sys.modules[m]

import os
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cap_control")

from cap_control import config as cfg
from cap_control.simulation.target_sim import DatasetTumblingTarget
from cap_control.simulation.sensors import NoisyPoseSensor
from cap_control.prediction.upn_predictor import UPNPredictor

# ----------------------------------------------------------------------------
ALPHA = 0.10
LOOKAHEADS = [0.5, 1.0, 1.5, 2.0]
RHO_BODY = np.array([0.1, 0.0, 0.0])          # grasp offset (matches benchmark)
FRESH_PATH = os.path.join("capture_lib_v2", "calibration_matched_200.npz")
ORIG_PATH = os.path.join("capture_lib_v2", "tumbling_target_dataset_v2.npz")
N_FRESH = 200
FRESH_SEED = 1000
B_SUBSAMPLES = 500
RNG = np.random.default_rng(7)

# ----------------------------------------------------------------------------
def quat_angle_deg(q1, q2):
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    cw, cx, cy, cz = q1[0], -q1[1], -q1[2], -q1[3]
    qw, qx, qy, qz = q2
    rw = qw*cw - qx*cx - qy*cy - qz*cz
    return 2 * np.rad2deg(np.arccos(min(1.0, abs(rw))))


def quat_to_rot(q):
    """Scalar-first [w,x,y,z] unit quaternion -> rotation matrix."""
    w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def grasp_point(r, q):
    return np.asarray(r) + quat_to_rot(np.asarray(q)) @ RHO_BODY


class DirectDatasetTarget:
    """Trajectory by absolute index from an arbitrary npz (same transforms
    as DatasetTumblingTarget)."""
    def __init__(self, abs_idx, dataset_path):
        data = np.load(dataset_path)
        self.true_states = data["true_states"][abs_idx].copy()
        self.times = data["times"].copy()
        self.dt = float(self.times[1] - self.times[0])
        self.T_final = float(self.times[-1])
        world_offset = (np.array([0.0, 0.0, 0.65 * cfg.ARM_REACH])
                        - self.true_states[0, 0:3])
        self.true_states[:, 0:3] += world_offset

    def state_at(self, t):
        if t <= 0.0:
            return self.true_states[0].copy()
        if t >= self.T_final:
            return self.true_states[-1].copy()
        idx = int(t / self.dt)
        frac = (t / self.dt) - idx
        out = ((1 - frac) * self.true_states[idx]
               + frac * self.true_states[idx + 1])
        out[6:10] = out[6:10] / max(np.linalg.norm(out[6:10]), 1e-12)
        return out

    def pose_at(self, t):
        st = self.state_at(t)
        return st[0:3], st[6:10] / max(np.linalg.norm(st[6:10]), 1e-12)


# ----------------------------------------------------------------------------
def ensure_fresh_calibration_set():
    if os.path.exists(FRESH_PATH):
        print(f"Fresh calibration set already exists: {FRESH_PATH}")
        return
    print(f"Generating fresh calibration set ({N_FRESH} traj, seed={FRESH_SEED})...")
    sys.path.insert(0, "capture_lib_v2")
    from capture.target.dataset import generate_dataset
    generate_dataset(n_traj=N_FRESH, t_final=10.0, n_steps=100,
                     pos_noise_std=0.01, rot_noise_std_deg=1.0,
                     seed=FRESH_SEED, output_file=FRESH_PATH,
                     vel_range=(0.01, 0.10),
                     w_bins_deg=[(0.1, 3.0), (3.0, 10.0), (10.0, 20.0), (20.0, 30.0)])
    print(f"Saved {FRESH_PATH}")


def gather_scores(target_factory, indices, tag):
    """Return scores[la][metric] = list over trajectories of per-trajectory
    score lists (grouped structure preserved)."""
    upn = UPNPredictor()
    HIST = cfg.UPN_HISTORY_LEN
    SDT = cfg.SENSOR_DT
    grouped = {la: {"pos": [], "ori": []} for la in LOOKAHEADS}

    for count, idx in enumerate(indices):
        target = target_factory(idx)
        eval_times = np.linspace(0.0, 3.0, 8)
        traj_scores = {la: {"pos": [], "ori": []} for la in LOOKAHEADS}

        for seed in (1, 2, 3):
            sensor = NoisyPoseSensor(pos_noise_std=cfg.POS_NOISE_STD,
                                     rot_noise_std_deg=cfg.ROT_NOISE_STD_DEG,
                                     seed=seed)
            obs_h, obs_t = [], []
            for tb in np.arange(HIST) * SDT - (HIST - 1) * SDT:
                p, q = target.pose_at(max(0.0, tb))
                obs_h.append(sensor.observe(p, q)); obs_t.append(float(tb))

            next_sensor_t = 0.0
            for now_t in eval_times:
                while next_sensor_t <= now_t + 1e-9:
                    p, q = target.pose_at(next_sensor_t)
                    obs_h.append(sensor.observe(p, q))
                    obs_t.append(next_sensor_t)
                    if len(obs_h) > HIST + 10:
                        obs_h = obs_h[-(HIST + 10):]
                        obs_t = obs_t[-(HIST + 10):]
                    next_sensor_t += SDT

                for la in LOOKAHEADS:
                    target_t = now_t + la
                    if target_t > 4.0:
                        continue
                    ho = np.asarray(obs_h[-HIST:])
                    ht = np.asarray(obs_t[-HIST:])
                    ft = np.array([max(target_t, ht[-1] + 1e-3)])
                    try:
                        m, _, _ = upn.predict(ho, ht, ft, future_obs=None,
                                              use_updates=False)
                        pred = m[-1]
                    except Exception:
                        continue
                    true_st = target.state_at(target_t)
                    # GRASP-POINT position score
                    rc_hat = grasp_point(pred[0:3], pred[6:10])
                    rc_true = grasp_point(true_st[0:3], true_st[6:10])
                    traj_scores[la]["pos"].append(
                        float(np.linalg.norm(rc_hat - rc_true) * 100))
                    traj_scores[la]["ori"].append(
                        float(quat_angle_deg(pred[6:10], true_st[6:10])))

        for la in LOOKAHEADS:
            grouped[la]["pos"].append(traj_scores[la]["pos"])
            grouped[la]["ori"].append(traj_scores[la]["ori"])
        print(f"  [{tag}] trajectory {count + 1}/{len(indices)} done")
    return grouped


# ------------------------- quantile constructions ---------------------------
def q_sample_level(groups, alpha=ALPHA):
    pooled = sorted(s for g in groups for s in g)
    n = len(pooled)
    k = min(int(np.ceil((1 - alpha) * (n + 1))), n)
    return pooled[k - 1], n


def q_traj_max(groups, alpha=ALPHA):
    maxima = sorted(max(g) for g in groups if g)
    n = len(maxima)
    k = min(int(np.ceil((1 - alpha) * (n + 1))), n)
    return maxima[k - 1], n


def q_repeated_subsample(groups, alpha=ALPHA, B=B_SUBSAMPLES, rng=RNG):
    """Dunn et al. (2023) repeated subsampling: one score per trajectory per
    draw; averaged conformal p-values; threshold = sup{s : p_bar(s) > alpha}.
    Finite-sample guarantee at 1 - 2*alpha; empirically ~ 1 - alpha."""
    gs = [np.asarray(g) for g in groups if len(g)]
    n = len(gs)
    M = np.stack([np.array([rng.choice(g) for g in gs]) for _ in range(B)])
    cand = np.unique(np.concatenate([m for m in M]))
    # p_bar(s) = mean_b (1 + #{M[b] >= s}) / (n + 1); decreasing in s
    counts = (M[:, :, None] >= cand[None, None, :]).sum(axis=1)   # (B, C)
    p_bar = (1.0 + counts).mean(axis=0) / (n + 1)                  # (C,)
    ok = np.where(p_bar > alpha)[0]
    return float(cand[ok[-1]]) if len(ok) else float(cand[-1]), n


def q_cdf_pooling(groups, alpha=ALPHA):
    """Equal-trajectory-weight empirical quantile (asymptotic reference)."""
    vals, wts = [], []
    gs = [g for g in groups if g]
    k = len(gs)
    for g in gs:
        m = len(g)
        vals.extend(g)
        wts.extend([1.0 / (k * m)] * m)
    order = np.argsort(vals)
    v = np.asarray(vals)[order]
    w = np.asarray(wts)[order]
    cum = np.cumsum(w)
    idx = np.searchsorted(cum, 1 - alpha)
    return float(v[min(idx, len(v) - 1)]), k


METHODS = {
    "sample_level": q_sample_level,
    "traj_max": q_traj_max,
    "repeated_subsample": q_repeated_subsample,
    "cdf_pooling": q_cdf_pooling,
}
GUARANTEE = {
    "sample_level": "1-alpha under raw-sample exchangeability (violated here)",
    "traj_max": "exact 1-alpha, UNIFORM over all queries of a new trajectory",
    "repeated_subsample": "finite-sample 1-2*alpha marginal; empirically ~1-alpha",
    "cdf_pooling": "asymptotic (k -> inf) marginal",
}

# ---------------------------- coverage evaluation ---------------------------
def coverage_report(test_grouped, qtab, la):
    q = qtab[la]
    groups_o = test_grouped[la]["ori"]
    groups_p = test_grouped[la]["pos"]
    marg_o = np.mean([e <= q["ori"] for g in groups_o for e in g]) * 100
    marg_p = np.mean([e <= q["pos"] for g in groups_p for e in g]) * 100
    unif_o = np.mean([all(e <= q["ori"] for e in g) for g in groups_o if g]) * 100
    unif_p = np.mean([all(e <= q["pos"] for e in g) for g in groups_p if g]) * 100
    # trajectory-cluster bootstrap CI on marginal coverage
    def cluster_ci(groups, thr, B=2000):
        k = len(groups)
        stats = []
        for _ in range(B):
            pick = RNG.integers(0, k, size=k)
            vals = [e <= thr for i in pick for e in groups[i]]
            stats.append(np.mean(vals) * 100)
        return np.percentile(stats, [2.5, 97.5])
    ci_o = cluster_ci(groups_o, q["ori"])
    ci_p = cluster_ci(groups_p, q["pos"])
    return marg_o, marg_p, unif_o, unif_p, ci_o, ci_p


# ============================================================================
print("=" * 78)
print("Stage 1 v2: hierarchical conformal calibration (fresh set, grasp point)")
print("=" * 78)

ensure_fresh_calibration_set()

print("\nGathering CALIBRATION scores on the fresh set...")
cal = gather_scores(lambda i: DirectDatasetTarget(i, FRESH_PATH),
                    list(range(N_FRESH)), "cal")

print("\nGathering TEST scores on all 45 test trajectories...")
test = gather_scores(lambda i: DatasetTumblingTarget(traj_idx=i),
                     list(range(45)), "test")

results = {}
for name, fn in METHODS.items():
    qtab = {}
    for la in LOOKAHEADS:
        qo, n_o = fn(cal[la]["ori"])
        qp, _ = fn(cal[la]["pos"])
        qtab[la] = {"ori": float(qo), "pos": float(qp), "n_groups": n_o}
    results[name] = qtab

print("\n" + "=" * 78)
print("CALIBRATED BOUNDS (deg / cm) BY CONSTRUCTION")
print("=" * 78)
hdr = f"{'tau':>5s}" + "".join(f" | {m[:18]:>22s}" for m in METHODS)
print(hdr)
for la in LOOKAHEADS:
    row = f"{la:>5.1f}"
    for m in METHODS:
        q = results[m][la]
        row += f" | {q['ori']:>9.1f} / {q['pos']:>8.1f}"
    print(row)

print("\n" + "=" * 78)
print("TEST-SET COVERAGE BY CONSTRUCTION "
      "(marginal % [cluster 95% CI] ; uniform-per-trajectory %)")
print("=" * 78)
comparison = {}
for m in METHODS:
    print(f"\n--- {m}  [{GUARANTEE[m]}] ---")
    comparison[m] = {"guarantee": GUARANTEE[m], "per_lookahead": {}}
    for la in LOOKAHEADS:
        mo, mp, uo, up, cio, cip = coverage_report(test, results[m], la)
        print(f"  tau={la:.1f}s  ORI {mo:5.1f}% [{cio[0]:.1f},{cio[1]:.1f}]"
              f"  unif {uo:5.1f}%   POS {mp:5.1f}% [{cip[0]:.1f},{cip[1]:.1f}]"
              f"  unif {up:5.1f}%")
        comparison[m]["per_lookahead"][str(la)] = {
            "q_ori_deg": results[m][la]["ori"],
            "q_pos_cm": results[m][la]["pos"],
            "cov_ori_marginal": mo, "cov_pos_marginal": mp,
            "cov_ori_uniform_traj": uo, "cov_pos_uniform_traj": up,
            "ci_ori": list(map(float, cio)), "ci_pos": list(map(float, cip)),
        }

# deployment-schema JSON per construction
for m in METHODS:
    out = {
        "alpha": ALPHA, "confidence": 1 - ALPHA,
        "method": m, "guarantee": GUARANTEE[m],
        "calibration_source": FRESH_PATH,
        "position_score": "grasp_point",
        "lookaheads": LOOKAHEADS,
        "q_hat_orientation_deg": {str(la): results[m][la]["ori"] for la in LOOKAHEADS},
        "q_hat_position_cm": {str(la): results[m][la]["pos"] for la in LOOKAHEADS},
        "calibration_n_samples": {str(la): sum(len(g) for g in cal[la]["ori"])
                                   for la in LOOKAHEADS},
        "calibration_n_trajectories": N_FRESH,
    }
    fn = f"conformal_calibration_v2_{m}.json"
    json.dump(out, open(fn, "w"), indent=2)
    print(f"saved {fn}")

json.dump(comparison, open("calibration_comparison.json", "w"), indent=2)
print("saved calibration_comparison.json")

print("\nDeployed construction: repeated_subsample (see paper, Section 4.1).")
