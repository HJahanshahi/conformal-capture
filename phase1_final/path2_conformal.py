import os
"""Stage 2+3: Conformal-aware controller with uncertainty-adaptive logic.

Architecture additions over path2_full_winning.py:

  At each replan, we compute the calibrated orientation bound q_hat for
  the current lookahead (tf - t_now). Based on this bound:

  STRATEGY 1 - Trajectory rejection (high uncertainty):
    If q_hat_ori > REJECT_THRESHOLD (40 deg), reject the new plan and
    keep the previous trajectory. This avoids deploying high-uncertainty plans.

  STRATEGY 2 - Gain modulation (medium uncertainty):
    Linearly interpolate Kp_ori between [1.0 at q_hat=40 deg] and
    [5.0 at q_hat=20 deg]. Below 20: full gains. Above 40: rejection
    triggers anyway.

  STRATEGY 3 - Predicted feasibility flag:
    Track which runs were "high-confidence" vs "low-confidence" based on
    bound at deployment. Allows statistical capture-rate claim.
"""
import sys
for m in list(sys.modules.keys()):
    if "cap_control" in m or "space_robot_dq" in m:
        del sys.modules[m]

import time
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cap_control")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Hamiltonian.*")

from cap_control import config as cfg
from cap_control.dynamics.free_floating import FreeFloatingChaser
from cap_control.controller.feedback_linearization import FeedbackLinearizationController
from cap_control.control.rendezvous_trajectory import (
    solve_rendezvous_trajectory, _grapple_kinematics,
)
from cap_control.simulation.target_sim import DatasetTumblingTarget
from cap_control.simulation.sensors import NoisyPoseSensor
from cap_control.prediction.upn_predictor import UPNPredictor


def quat_angle_deg(q1, q2):
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    cw, cx, cy, cz = q1[0], -q1[1], -q1[2], -q1[3]
    qw, qx, qy, qz = q2
    rw = qw*cw - qx*cx - qy*cy - qz*cz
    return 2 * np.rad2deg(np.arccos(min(1.0, abs(rw))))


# Load conformal calibration
with open("conformal_calibration.json", "r") as f:
    CALIB = json.load(f)
print(f"Loaded conformal calibration: confidence={CALIB['confidence']}")

# Calibration q_hat for orientation, by lookahead (interpolatable)
CALIB_LA = CALIB["lookaheads"]
CALIB_Q_ORI = [CALIB["q_hat_orientation_deg"][str(la)] for la in CALIB_LA]
CALIB_Q_POS = [CALIB["q_hat_position_cm"][str(la)] for la in CALIB_LA]


def q_hat_ori(lookahead):
    """Linearly interpolated calibrated orientation bound."""
    if lookahead <= CALIB_LA[0]:
        return CALIB_Q_ORI[0]
    if lookahead >= CALIB_LA[-1]:
        return CALIB_Q_ORI[-1]
    # Linear interp
    return float(np.interp(lookahead, CALIB_LA, CALIB_Q_ORI))


def kp_ori_from_uncertainty(q_bound, q_low=20.0, q_high=40.0,
                              kp_max=10.0, kp_min=1.0):
    """Linearly interpolate Kp_ori based on calibrated uncertainty bound.

    Below q_low: full gain (kp_max=5)
    Between q_low and q_high: linear ramp down
    Above q_high: minimum gain (kp_min=1) - but rejection should trigger first
    """
    if q_bound <= q_low:
        return kp_max
    if q_bound >= q_high:
        return kp_min
    frac = (q_bound - q_low) / (q_high - q_low)
    return kp_max * (1 - frac) + kp_min * frac


RHO_BODY = np.array([0.1, 0.0, 0.0])
IC = np.diag([10.0, 10.0, 10.0])

# Conformal-control parameters
T_BLEND = 1.5
REJECT_THRESHOLD = 40.0       # reject plan if q_hat_ori > 40 deg
GAIN_LOW_THRESHOLD = 20.0     # below this, full gain
GAIN_HIGH_THRESHOLD = 40.0    # at/above this, min gain
KP_ORI_MAX = 10.0
KP_ORI_MIN = 1.0
KD_ORI_RATIO = 0.6            # Kd = ratio * Kp


def make_upn_propagator(upn, obs_h, obs_t, t_offset):
    HIST = cfg.UPN_HISTORY_LEN
    def prop(t):
        ho = np.asarray(obs_h[-HIST:])
        ht = np.asarray(obs_t[-HIST:])
        future_t = np.array([max(t_offset + t, ht[-1] + 1e-3)])
        m, _, _ = upn.predict(ho, ht, future_t, future_obs=None, use_updates=False)
        s = m[-1]
        q = s[6:10] / max(np.linalg.norm(s[6:10]), 1e-12)
        return (q, s[10:13], s[0:3], s[3:6])
    return prop


def run_one(traj_idx, seed=1, t_final=4.0, dt=0.1):
    """Run with conformal-aware logic. Returns (errs_pos, errs_ori, n_rejected, max_q_hat)."""
    chaser = FreeFloatingChaser()
    state = chaser.home()
    target_sim = DatasetTumblingTarget(traj_idx=traj_idx)
    sensor = NoisyPoseSensor(pos_noise_std=cfg.POS_NOISE_STD,
                             rot_noise_std_deg=cfg.ROT_NOISE_STD_DEG, seed=seed)
    upn = UPNPredictor()

    HIST = cfg.UPN_HISTORY_LEN; SDT = cfg.SENSOR_DT
    obs_h, obs_t = [], []
    for tb in np.arange(HIST) * SDT - (HIST - 1) * SDT:
        p, q = target_sim.pose_at(max(0.0, tb))
        obs_h.append(sensor.observe(p, q)); obs_t.append(float(tb))
    next_sensor_t = 0.0

    rh, _ = chaser.fk_world(state)
    rhdot = None  # filled below
    
    # Initial plan
    try:
        prop = make_upn_propagator(upn, obs_h, obs_t, t_offset=0.0)
        # Use temporary controller to compute initial rhdot
        temp_c = FeedbackLinearizationController(
            chaser=chaser, Kp_pos=20.0, Kd_pos=8.0,
            Kp_ori=KP_ORI_MAX, Kd_ori=KP_ORI_MAX * KD_ORI_RATIO,
            tau_limit=20.0, t_blend_ori=T_BLEND)
        rhdot = temp_c.compute_ee_velocity(state)
        traj = solve_rendezvous_trajectory(
            rh, rhdot, target_state=None, target_inertia=IC,
            rho_body=RHO_BODY, w1=1.0, w2=1.0, target_propagator=prop)
    except Exception:
        return None, None, None, None
    plan_t0 = 0.0

    # Initial q_hat for the upcoming traj
    initial_q_hat = q_hat_ori(traj.tf)
    # Initial gains based on this
    current_kp_ori = kp_ori_from_uncertainty(initial_q_hat,
                                                GAIN_LOW_THRESHOLD,
                                                GAIN_HIGH_THRESHOLD,
                                                KP_ORI_MAX, KP_ORI_MIN)
    controller = FeedbackLinearizationController(
        chaser=chaser, Kp_pos=20.0, Kd_pos=8.0,
        Kp_ori=current_kp_ori, Kd_ori=current_kp_ori * KD_ORI_RATIO,
        tau_limit=20.0, t_blend_ori=T_BLEND)

    n_steps = int(round(t_final / dt))
    errs_pos, errs_ori = [], []
    n_rejected = 0
    max_q_hat = initial_q_hat

    for k in range(n_steps):
        t_now = k * dt
        while next_sensor_t <= t_now + 1e-9:
            p, q = target_sim.pose_at(next_sensor_t)
            obs_h.append(sensor.observe(p, q)); obs_t.append(next_sensor_t)
            if len(obs_h) > HIST + 10:
                obs_h = obs_h[-(HIST + 10):]; obs_t = obs_t[-(HIST + 10):]
            next_sensor_t += SDT

        time_in = t_now - plan_t0
        if time_in >= 1.0 or time_in >= traj.tf - dt:
            rh, _ = chaser.fk_world(state)
            rhdot = controller.compute_ee_velocity(state)
            prop = make_upn_propagator(upn, obs_h, obs_t, t_offset=t_now)
            try:
                new_traj = solve_rendezvous_trajectory(
                    rh, rhdot, target_state=None, target_inertia=IC,
                    rho_body=RHO_BODY, w1=1.0, w2=1.0, target_propagator=prop)
                # Check calibrated uncertainty for the new plan's lookahead
                new_q_hat = q_hat_ori(new_traj.tf)
                max_q_hat = max(max_q_hat, new_q_hat)

                if new_q_hat > REJECT_THRESHOLD:
                    # Strategy 1: REJECT - keep using previous trajectory
                    n_rejected += 1
                else:
                    # Accept plan
                    traj = new_traj
                    plan_t0 = t_now
                    # Strategy 2: GAIN MODULATION based on calibrated uncertainty
                    current_kp_ori = kp_ori_from_uncertainty(
                        new_q_hat,
                        GAIN_LOW_THRESHOLD, GAIN_HIGH_THRESHOLD,
                        KP_ORI_MAX, KP_ORI_MIN)
                    controller = FeedbackLinearizationController(
                        chaser=chaser, Kp_pos=20.0, Kd_pos=8.0,
                        Kp_ori=current_kp_ori,
                        Kd_ori=current_kp_ori * KD_ORI_RATIO,
                        tau_limit=20.0, t_blend_ori=T_BLEND)
            except Exception:
                pass

        local_t = t_now - plan_t0
        rh_des, rhdot_des, rhddot_des = traj.evaluate(local_t)
        ref = {"rh_des": rh_des, "rhdot_des": rhdot_des, "rhddot_des": rhddot_des}

        time_to_go = traj.tf - local_t
        # --- PER-STEP GAIN MODULATION: the calibrated bound is a function of
        # lookahead, and the lookahead shrinks while a plan runs, so the
        # orientation gain is recomputed every control step.
        _q_now = q_hat_ori(max(float(time_to_go), 1e-3))
        _kp_now = kp_ori_from_uncertainty(
            _q_now, GAIN_LOW_THRESHOLD, GAIN_HIGH_THRESHOLD,
            KP_ORI_MAX, KP_ORI_MIN)
        if abs(_kp_now - current_kp_ori) > 1e-9:
            current_kp_ori = _kp_now
            controller = FeedbackLinearizationController(
                chaser=chaser, Kp_pos=20.0, Kd_pos=8.0,
                Kp_ori=current_kp_ori,
                Kd_ori=current_kp_ori * KD_ORI_RATIO,
                tau_limit=controller.tau_limit, t_blend_ori=T_BLEND)
        future_t = t_now + time_to_go
        ho = np.asarray(obs_h[-HIST:])
        ht = np.asarray(obs_t[-HIST:])
        ft = np.array([max(future_t, ht[-1] + 1e-3)])
        try:
            m, _, _ = upn.predict(ho, ht, ft, future_obs=None, use_updates=False)
            s_pred = m[-1]
            q_des = s_pred[6:10] / max(np.linalg.norm(s_pred[6:10]), 1e-12)
            ref["q_des"] = q_des
            ref["omega_des"] = s_pred[10:13]
            ref["omega_dot_des"] = np.zeros(3)
            ref["time_to_go"] = float(time_to_go)
        except Exception:
            pass

        tau, _ = controller.solve(state, ref)
        state = chaser.dynamic_step(state, tau, dt, include_coriolis=True)
        if not np.all(np.isfinite(state.qdot)) or np.max(np.abs(state.qdot)) > 50:
            return None, None, None, None

        rh_actual, q_ee = chaser.fk_world(state)
        true_st = target_sim.state_at(t_now)
        q_true = true_st[6:10] / max(np.linalg.norm(true_st[6:10]), 1e-12)
        rc_actual, _, _ = _grapple_kinematics(q_true, true_st[10:13], true_st[0:3],
                                                RHO_BODY, IC)
        errs_pos.append(np.linalg.norm(rh_actual - rc_actual) * 100)
        errs_ori.append(quat_angle_deg(q_ee, q_true))

    return errs_pos, errs_ori, n_rejected, max_q_hat


print("=" * 78)
print(f"Stage 2+3: Conformal-aware controller benchmark (45 traj x 3 seeds)")
print(f"  Reject threshold: q_hat_ori > {REJECT_THRESHOLD} deg")
print(f"  Gain modulation: Kp_ori = {KP_ORI_MAX} (q_hat <= {GAIN_LOW_THRESHOLD} deg)")
print(f"                            -> {KP_ORI_MIN} (q_hat >= {GAIN_HIGH_THRESHOLD} deg)")
print(f"  Confidence level: {1 - CALIB['alpha']} ({CALIB['confidence']*100}%)")
print("=" * 78)

data = np.load("capture_lib_v2/tumbling_target_dataset_v2.npz")
true_states = data["true_states"]
n_train = int(0.70 * true_states.shape[0])
n_val = int(0.15 * true_states.shape[0])
n_test = true_states.shape[0] - n_train - n_val
omegas = []
for i in range(n_test):
    om = np.rad2deg(np.linalg.norm(true_states[n_train + n_val + i, :, 10:13], axis=1)).max()
    omegas.append(om)

results = []
SEEDS = [1, 2, 3]
t0 = time.time()

for i in range(n_test):
    for seed in SEEDS:
        errs_p, errs_o, n_rej, mq = run_one(i, seed=seed)
        if errs_p is None:
            results.append((i, seed, omegas[i], None, None, None, None, None, None, False))
        else:
            results.append((i, seed, omegas[i],
                              float(np.mean(errs_p[-10:])),
                              float(np.mean(errs_o[-10:])),
                              float(errs_p[-1]),
                              float(errs_o[-1]),
                              int(n_rej),
                              float(mq), True))
    last3 = results[-3:]
    summary = ", ".join(f"{r[5]:5.2f}cm/{r[6]:5.1f}deg" if r[9] else "DIV"
                          for r in last3)
    print(f"  traj {i:2d} (omega={omegas[i]:5.2f} deg/s): {summary}")

elapsed = time.time() - t0
print(f"\nTotal: {elapsed:.1f}s = {elapsed/60:.1f} min")

pos_tf = [r[5] for r in results if r[9]]
ori_tf = [r[6] for r in results if r[9]]
n_rejs = [r[7] for r in results if r[9]]
max_qs = [r[8] for r in results if r[9]]
n_total = len(results)
n_success = sum(1 for r in results if r[9])

print()
print("=" * 78)
print("RESULTS WITH CONFORMAL")
print("=" * 78)
print(f"Success: {n_success}/{n_total} = {100*n_success/n_total:.1f}%")
print()
print("Position tracking AT TF:")
print(f"  Mean: {np.mean(pos_tf):.2f} +/- {np.std(pos_tf):.2f} cm")
print(f"  Median: {np.median(pos_tf):.2f}, 95th: {np.percentile(pos_tf, 95):.2f} cm")
print()
print("Orientation tracking AT TF:")
print(f"  Mean: {np.mean(ori_tf):.2f} +/- {np.std(ori_tf):.2f} deg")
print(f"  Median: {np.median(ori_tf):.2f}, 95th: {np.percentile(ori_tf, 95):.2f} deg")

print()
print("Conformal stats:")
print(f"  Mean rejections per run:  {np.mean(n_rejs):.2f}")
print(f"  Max rejections in any run: {max(n_rejs)}")
print(f"  Mean max-q_hat seen:       {np.mean(max_qs):.2f} deg")

print()
print("Distribution at tf:")
for thresh in [5, 10, 15, 20, 30, 60]:
    n_pos = sum(1 for p in pos_tf if p < thresh)
    n_ori = sum(1 for o in ori_tf if o < thresh)
    print(f"  Pos<{thresh}cm:  {n_pos:3d}/{n_total} = {100*n_pos/n_total:.0f}%   "
          f"Ori<{thresh}deg: {n_ori:3d}/{n_total} = {100*n_ori/n_total:.0f}%")

print()
print("BOTH-thresholds capture-ready rate:")
for p_th, o_th in [(5, 5), (5, 10), (10, 10), (10, 15), (15, 15), (15, 20)]:
    n = sum(1 for p, o in zip(pos_tf, ori_tf) if p < p_th and o < o_th)
    print(f"  Pos<{p_th}cm AND Ori<{o_th}deg: {n}/{n_total} = {100*n/n_total:.0f}%")

print()

# --- baseline column read from the actual baseline run, if present ---
_BASELINE_FILE = "path2_winning_v2_results.json"
_b_pos = _b_ori = _b_cr = None
try:
    with open(_BASELINE_FILE) as _f:
        _b = [r for r in json.load(_f) if r["success"]]
    _b_pos = float(np.mean([r["pos_tf"] for r in _b]))
    _b_ori = float(np.mean([r["ori_tf"] for r in _b]))
    _b_cr = 100.0 * sum(1 for r in _b
                        if r["pos_tf"] < 10 and r["ori_tf"] < 15) / 135.0
except Exception:
    pass

print("=" * 78)
print("COMPARISON to the non-adaptive baseline")
print("=" * 78)
print(f"               BASELINE                CONFORMAL")
print(f"Pos at tf      {_b_pos:6.2f} cm                 {np.mean(pos_tf):.2f} cm")
print(f"Ori at tf      {_b_ori:6.2f} deg               {np.mean(ori_tf):.2f} deg")
print(f"Cap-ready 10/15 {_b_cr:3.0f}%                   {100*sum(1 for p, o in zip(pos_tf, ori_tf) if p < 10 and o < 15)/n_total:.0f}%")

# Save
out = []
for r in results:
    out.append({
        "traj": int(r[0]), "seed": int(r[1]), "omega": float(r[2]),
        "pos_last1s": r[3], "ori_last1s": r[4],
        "pos_tf": r[5], "ori_tf": r[6],
        "n_rejected": r[7], "max_q_hat": r[8],
        "success": bool(r[9])
    })
with open("path2_conformal_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nRaw saved to path2_conformal_results.json")
