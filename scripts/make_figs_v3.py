"""Sync make_all_figures.py to the v3 combined-rerun results. Run once:
    python make_figs_v3.py
then regenerate:  python make_all_figures.py

Edits: (1) result/calibration filenames -> v2/v3; (2) coverage panel reads
calibration_comparison.json (repeated_subsample) with trajectory-cluster CIs;
(3) failure exemplar -> traj 33 seed 1; (4) internal re-simulation synced to
the deployed v3 controller (smoothstep gains, rejection-overflow contingency,
aligned evaluation); (5) table captions updated.
Backup saved as make_all_figures.py.bak
"""
import shutil, py_compile

PATH = "make_all_figures.py"
shutil.copy(PATH, PATH + ".bak")
src = open(PATH, encoding="utf-8").read()
applied, missed = [], []


def rep(name, old, new, count=1):
    global src
    c = src.count(old)
    if c != count:
        missed.append(f"{name} (found {c}, expected {count})")
        return
    src = src.replace(old, new)
    applied.append(name)


# ---- 1. filenames ----
rep("baseline file", 'load_results("path2_winning_results.json")',
    'load_results("path2_winning_v2_results.json")')
rep("conformal file", 'load_results("path2_conformal_results.json")',
    'load_results("path2_conformal_v3_results.json")')
rep("calibration file", 'with open("conformal_calibration.json") as f:',
    'with open("conformal_calibration_v2_repeated_subsample.json") as f:')

# ---- 2. coverage source -> calibration_comparison.json ----
rep("coverage loader",
    '''def load_test_coverage(path="conformal_coverage_full.json"):''',
    '''def load_test_coverage(path="calibration_comparison.json"):''')
rep("coverage loader body",
    '''        return json.load(f)["per_lookahead"]''',
    '''        d = json.load(f)["repeated_subsample"]["per_lookahead"]
    return {la: {"coverage_ori_pct": v["cov_ori_marginal"],
                  "coverage_pos_pct": v["cov_pos_marginal"],
                  "ci_ori": v["ci_ori"], "ci_pos": v["ci_pos"]}
            for la, v in d.items()}''')
rep("coverage loader error msg",
    '"conformal_coverage_full.json not found. Copy it from the "',
    '"calibration_comparison.json not found. Copy it from the "')

# ---- 3. fig2 panel b: cluster CIs replace binomial ----
rep("drop n_samples",
    '''    # n for the CIs = number of TEST samples the coverage was measured on
    n_samples = [covd[str(la)]["n_test_samples"] for la in las]''',
    '''    ci_lo_ori = [covd[str(la)]["ci_ori"][0] for la in las]
    ci_hi_ori = [covd[str(la)]["ci_ori"][1] for la in las]
    ci_lo_pos = [covd[str(la)]["ci_pos"][0] for la in las]
    ci_hi_pos = [covd[str(la)]["ci_pos"][1] for la in las]''')
rep("binomial -> cluster",
    '''    def binomial_ci(p_pct, n):
        p = p_pct / 100.0
        z = 1.96
        se = np.sqrt(p * (1 - p) / n)
        lo = max(0, p - z * se) * 100
        hi = min(1, p + z * se) * 100
        return p_pct - lo, hi - p_pct

    ci_ori = np.array([binomial_ci(c, n)
                          for c, n in zip(test_cov_ori, n_samples)]).T
    ci_pos = np.array([binomial_ci(c, n)
                          for c, n in zip(test_cov_pos, n_samples)]).T''',
    '''    # trajectory-level cluster-bootstrap intervals (from the calibration run)
    ci_ori = np.array([[c - lo for c, lo in zip(test_cov_ori, ci_lo_ori)],
                        [hi - c for c, hi in zip(test_cov_ori, ci_hi_ori)]])
    ci_pos = np.array([[c - lo for c, lo in zip(test_cov_pos, ci_lo_pos)],
                        [hi - c for c, hi in zip(test_cov_pos, ci_hi_pos)]])''')

# ---- 4. failure exemplar -> traj 33 seed 1 ----
rep("failure run call",
    '''    print("\\nRecording FAILURE run (traj 42, seed 1)...")
    log_failure = run_with_logging(traj_idx=42, seed=1)''',
    '''    print("\\nRecording FAILURE run (traj 33, seed 1)...")
    log_failure = run_with_logging(traj_idx=33, seed=1)''')
rep("fig7 title", '"(b) Failure case (traj 42)"', '"(b) Failure case (traj 33)"')
rep("fig10 title", '"Failure case (traj 42)"', '"Failure case (traj 33)"')

# ---- 5a. smoothstep gain map ----
rep("smoothstep",
    '''    if qb <= q_low: return kp_max
    if qb >= q_high: return kp_min
    frac = (qb - q_low) / (q_high - q_low)
    return kp_max * (1 - frac) + kp_min * frac''',
    '''    frac = float(np.clip((qb - q_low) / (q_high - q_low), 0.0, 1.0))
    sm = frac * frac * (3.0 - 2.0 * frac)
    return float(kp_max - (kp_max - kp_min) * sm)''')

# ---- 5b. contingency in the logging sim ----
rep("extend rt imports",
    """from cap_control.control.rendezvous_trajectory import (
        solve_rendezvous_trajectory, _grapple_kinematics)""",
    """from cap_control.control.rendezvous_trajectory import (
        solve_rendezvous_trajectory, _grapple_kinematics,
        _trajectory_coefficients, RendezvousTrajectory)""")
rep("tau contingency + fixed planner",
    '''    def make_upn_propagator(upn, obs_h, obs_t, t_offset):''',
    '''    taus_grid = np.linspace(0.5, 2.0, 301)
    TAU_CONTINGENCY = float(max([t for t in taus_grid if q_hat_ori(t) <= 40.0],
                                 default=0.5))

    def plan_fixed_horizon(rh, rhdot, prop, tf_fix, sigma=1.0):
        q_tf, omega_tf, r_tf, _ = prop(tf_fix)
        q_tf = q_tf / max(np.linalg.norm(q_tf), 1e-12)
        rc_tf, rcdot_tf, _ = _grapple_kinematics(q_tf, omega_tf, r_tf,
                                                   RHO_BODY, IC)
        k0, k1, k2, k3 = _trajectory_coefficients(rh, rhdot, rc_tf, rcdot_tf,
                                                    tf_fix, sigma)
        return RendezvousTrajectory(k0=k0, k1=k1, k2=k2, k3=k3, sigma=sigma,
                                      tf=tf_fix, rc_tf=rc_tf,
                                      rcdot_tf=rcdot_tf)

    def make_upn_propagator(upn, obs_h, obs_t, t_offset):''')
rep("consec init",
    '''        initial_q_hat = q_hat_ori(traj.tf)
        current_kp_ori = kp_ori_from_uncertainty(initial_q_hat)''',
    '''        initial_q_hat = q_hat_ori(traj.tf)
        current_kp_ori = kp_ori_from_uncertainty(initial_q_hat)
        consec_rej = 0''')
rep("contingency logic",
    '''                    nq = q_hat_ori(new_traj.tf)
                    if nq > reject_threshold:
                        log["rejected_events"].append(t_now)
                    else:
                        traj = new_traj
                        plan_t0 = t_now
                        log["replan_events"].append(t_now)
                        current_kp_ori = kp_ori_from_uncertainty(nq)''',
    '''                    nq = q_hat_ori(new_traj.tf)
                    if nq > reject_threshold and consec_rej < 4:
                        log["rejected_events"].append(t_now)
                        consec_rej += 1
                    else:
                        if nq > reject_threshold:
                            # Mechanism 3: short fixed-horizon contingency plan
                            log["rejected_events"].append(t_now)
                            new_traj = plan_fixed_horizon(rh, rhdot, prop,
                                                            TAU_CONTINGENCY)
                            nq = q_hat_ori(TAU_CONTINGENCY)
                        consec_rej = 0
                        traj = new_traj
                        plan_t0 = t_now
                        log["replan_events"].append(t_now)
                        current_kp_ori = kp_ori_from_uncertainty(nq)''')

# ---- 5c. aligned evaluation ----
rep("aligned evaluation",
    "            true_st = target_sim.state_at(t_now)",
    "            true_st = target_sim.state_at(t_now + dt)"
    "  # aligned with post-step chaser")

# ---- 6. table captions ----
rep("table2 caption",
    r"Error statistics are computed over completed runs; success rates over all 135 runs (one diverged run counted as failure).",
    r"All 135 runs of both configurations complete; statistics are computed over all 135 runs.")
rep("calib table caption",
    r"with empirical coverage measured on the full held-out test set (45 trajectories; test sample counts equal $n$ by construction).",
    r"with empirical coverage measured on the full held-out test set (45 trajectories). Bounds use the hierarchical repeated-subsampling construction over 200 independently generated calibration trajectories.")

open(PATH, "w", encoding="utf-8").write(src)
print(f"APPLIED ({len(applied)}):")
for a in applied:
    print("  +", a)
if missed:
    print(f"MISSED ({len(missed)})")
    for m in missed:
        print("  !", m)
else:
    print("All edits applied.")
try:
    py_compile.compile(PATH, doraise=True)
    print("Compile check OK")
except py_compile.PyCompileError as e:
    print("COMPILE FAILED:", e)
