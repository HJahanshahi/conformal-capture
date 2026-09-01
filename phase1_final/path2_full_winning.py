import os
"""Path 2 full benchmark with WINNING config: Kp_ori=5, Kd_ori=4, t_blend=1.5s.

If results hold across all 45 trajectories, this is our 6DOF headline.
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


RHO_BODY = np.array([0.1, 0.0, 0.0])
IC = np.diag([10.0, 10.0, 10.0])

# WINNING CONFIG
KP_ORI = 10.0
KD_ORI = 6.0
T_BLEND = 1.5


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
    chaser = FreeFloatingChaser()
    state = chaser.home()
    target_sim = DatasetTumblingTarget(traj_idx=traj_idx)
    sensor = NoisyPoseSensor(pos_noise_std=cfg.POS_NOISE_STD,
                             rot_noise_std_deg=cfg.ROT_NOISE_STD_DEG, seed=seed)
    upn = UPNPredictor()
    controller = FeedbackLinearizationController(
        chaser=chaser, Kp_pos=20.0, Kd_pos=8.0,
        Kp_ori=KP_ORI, Kd_ori=KD_ORI,
        tau_limit=20.0, t_blend_ori=T_BLEND)

    HIST = cfg.UPN_HISTORY_LEN; SDT = cfg.SENSOR_DT
    obs_h, obs_t = [], []
    for tb in np.arange(HIST) * SDT - (HIST - 1) * SDT:
        p, q = target_sim.pose_at(max(0.0, tb))
        obs_h.append(sensor.observe(p, q)); obs_t.append(float(tb))
    next_sensor_t = 0.0

    rh, _ = chaser.fk_world(state)
    rhdot = controller.compute_ee_velocity(state)
    try:
        prop = make_upn_propagator(upn, obs_h, obs_t, t_offset=0.0)
        traj = solve_rendezvous_trajectory(
            rh, rhdot, target_state=None, target_inertia=IC,
            rho_body=RHO_BODY, w1=1.0, w2=1.0, target_propagator=prop)
    except Exception:
        return None, None
    plan_t0 = 0.0

    n_steps = int(round(t_final / dt))
    errs_pos, errs_ori = [], []
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
                traj = solve_rendezvous_trajectory(
                    rh, rhdot, target_state=None, target_inertia=IC,
                    rho_body=RHO_BODY, w1=1.0, w2=1.0, target_propagator=prop)
                plan_t0 = t_now
            except Exception:
                pass

        local_t = t_now - plan_t0
        rh_des, rhdot_des, rhddot_des = traj.evaluate(local_t)
        ref = {"rh_des": rh_des, "rhdot_des": rhdot_des, "rhddot_des": rhddot_des}

        time_to_go = traj.tf - local_t
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
            return None, None

        rh_actual, q_ee = chaser.fk_world(state)
        true_st = target_sim.state_at(t_now)
        q_true = true_st[6:10] / max(np.linalg.norm(true_st[6:10]), 1e-12)
        rc_actual, _, _ = _grapple_kinematics(q_true, true_st[10:13], true_st[0:3],
                                                RHO_BODY, IC)
        errs_pos.append(np.linalg.norm(rh_actual - rc_actual) * 100)
        errs_ori.append(quat_angle_deg(q_ee, q_true))

    return errs_pos, errs_ori


print("=" * 78)
print(f"Path 2 FULL with WINNING CONFIG: Kp_ori={KP_ORI}, Kd_ori={KD_ORI}, t_blend={T_BLEND}s")
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
        errs_p, errs_o = run_one(i, seed=seed)
        if errs_p is None:
            results.append((i, seed, omegas[i], None, None, None, None, False))
        else:
            results.append((i, seed, omegas[i],
                              float(np.mean(errs_p[-10:])),
                              float(np.mean(errs_o[-10:])),
                              float(errs_p[-1]),
                              float(errs_o[-1]), True))
    last3 = results[-3:]
    p_summary = ", ".join(f"{r[5]:5.2f}cm/{r[6]:5.1f}deg" if r[7] else "DIV"
                            for r in last3)
    print(f"  traj {i:2d} (omega={omegas[i]:5.2f} deg/s): {p_summary}")

elapsed = time.time() - t0
print(f"\nTotal: {elapsed:.1f}s = {elapsed/60:.1f} min")

# Aggregate
pos_last1s = [r[3] for r in results if r[7]]
ori_last1s = [r[4] for r in results if r[7]]
pos_tf = [r[5] for r in results if r[7]]
ori_tf = [r[6] for r in results if r[7]]
n_total = len(results)
n_success = sum(1 for r in results if r[7])

print()
print("=" * 78)
print("RESULTS")
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

# Distribution
print()
print("Distribution at tf:")
for thresh in [5, 10, 15, 20, 30, 60]:
    n_pos = sum(1 for p in pos_tf if p < thresh)
    n_ori = sum(1 for o in ori_tf if o < thresh)
    print(f"  Pos<{thresh}cm:  {n_pos:3d}/{n_total} = {100*n_pos/n_total:.0f}%   "
          f"Ori<{thresh}deg: {n_ori:3d}/{n_total} = {100*n_ori/n_total:.0f}%")

# Joint success rate (BOTH thresholds)
print()
print("BOTH-thresholds success rate (capture-ready):")
for p_th, o_th in [(5, 5), (5, 10), (10, 10), (10, 15), (15, 15), (15, 20)]:
    n = sum(1 for p, o in zip(pos_tf, ori_tf) if p < p_th and o < o_th)
    print(f"  Pos<{p_th}cm AND Ori<{o_th}deg: {n}/{n_total} = {100*n/n_total:.0f}%")

# Per-tumble bin
def bin_omega(o):
    if o < 3: return "LOW"
    elif o < 10: return "MID"
    elif o < 20: return "HIGH"
    else: return "EXTREME"

bins_data = {"LOW": [], "MID": [], "HIGH": [], "EXTREME": []}
for r in results:
    if r[7]:
        bins_data[bin_omega(r[2])].append((r[5], r[6]))

print()
print("Per-tumble breakdown at tf:")
print(f"  {'Bin':<10s} {'N':>4s} {'Pos (cm)':>14s} {'Ori (deg)':>16s}")
for bn, vals in bins_data.items():
    if vals:
        ps = [v[0] for v in vals]; os = [v[1] for v in vals]
        print(f"  {bn:<10s} {len(vals):>4d}  {np.mean(ps):>5.2f}+-{np.std(ps):.2f}    "
              f"{np.mean(os):>6.2f}+-{np.std(os):.2f}")

# Save
out = []
for r in results:
    out.append({
        "traj": int(r[0]), "seed": int(r[1]), "omega": float(r[2]),
        "pos_last1s": r[3], "ori_last1s": r[4],
        "pos_tf": r[5], "ori_tf": r[6], "success": bool(r[7])
    })
with open("path2_winning_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nRaw saved to path2_winning_results.json")
