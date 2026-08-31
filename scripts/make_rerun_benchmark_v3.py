"""Build path2_conformal_v3.py: combined-rerun benchmark, revision 3.\n\nv3 vs v2: (a) gain map is a C1 smoothstep slope-matched to the original\nramp (v2 logistic was too aggressive in the 30-40 deg band); (b) the\ncontingency now plans at a FIXED short admissible horizon instead of\nforce-accepting the long uncertain candidate (v2 accepted tau=2.0 s\nplans with q_hat ~88 deg, causing the 76-80 cm tail cases).

Changes vs path2_conformal.py (original left untouched):
  1. Calibration source -> conformal_calibration_v2_repeated_subsample.json
     (fresh matched 200-trajectory set, grasp-point scores, hierarchical
      repeated-subsampling construction)
  2. Gain modulation: piecewise-linear -> smooth C1 schedule
  3. Mechanism 3: rejection-overflow contingency (force-accept the candidate
     after 4 consecutive rejections instead of tracking a staler plan)
  4. Torque-saturation logging per run
  5. Evaluation-time alignment: errors compare the post-step chaser state
     with the target at t_now + dt (removes the one-step bias) [Sec-G]
  6. Extra per-run fields: n_contingency, n_sat_steps, q_hat_last_accept
  7. Output -> path2_conformal_v3_results.json

Run:  python make_rerun_benchmark.py     then:  python path2_conformal_v3.py
"""
import py_compile

SRC = r"phase1_final\path2_conformal.py"
DST = "path2_conformal_v3.py"

src = open(SRC, encoding="utf-8").read()
applied, missed = [], []


def rep(name, old, new, count=1):
    global src
    c = src.count(old)
    if c != count:
        missed.append(f"{name} (found {c}, expected {count})")
        return
    src = src.replace(old, new)
    applied.append(name)


# ---- 1. calibration source -------------------------------------------------
rep("extend rt imports",
    """from cap_control.control.rendezvous_trajectory import (
    solve_rendezvous_trajectory, _grapple_kinematics,
)""",
    """from cap_control.control.rendezvous_trajectory import (
    solve_rendezvous_trajectory, _grapple_kinematics,
    _trajectory_coefficients, RendezvousTrajectory,
)""")

rep("calibration file",
    'with open("conformal_calibration.json", "r") as f:',
    'with open("conformal_calibration_v2_repeated_subsample.json", "r") as f:')

# ---- 2. smooth (logistic) gain modulation ----------------------------------
rep("sigmoid gain map",
    '''def kp_ori_from_uncertainty(q_bound, q_low=20.0, q_high=40.0,
                              kp_max=5.0, kp_min=1.0):
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
    return kp_max * (1 - frac) + kp_min * frac''',
    '''def kp_ori_from_uncertainty(q_bound, q_low=20.0, q_high=40.0,
                              kp_max=5.0, kp_min=1.0):
    """Smooth (C1 smoothstep) gain schedule on the calibrated bound.

    Same endpoints and midpoint as the piecewise-linear ramp, but with
    zero slope at q_low and q_high (removes the derivative discontinuities
    the ramp had). Midpoint slope is 1.5x the linear ramp's.
    """
    frac = float(np.clip((q_bound - q_low) / (q_high - q_low), 0.0, 1.0))
    sm = frac * frac * (3.0 - 2.0 * frac)     # C1 smoothstep, exact endpoints
    return float(kp_max - (kp_max - kp_min) * sm)''')

# ---- 3+6. run_one counters ---------------------------------------------------
rep("contingency helper",
    """KD_ORI_RATIO = 0.8            # Kd = ratio * Kp""",
    """KD_ORI_RATIO = 0.8            # Kd = ratio * Kp


def _largest_admissible_lookahead():
    \"\"\"Largest calibrated lookahead with q_hat_ori <= REJECT_THRESHOLD.\"\"\"
    taus = np.linspace(CALIB_LA[0], CALIB_LA[-1], 301)
    ok = [t for t in taus if q_hat_ori(t) <= REJECT_THRESHOLD]
    return float(ok[-1]) if ok else float(CALIB_LA[0])


TAU_CONTINGENCY = _largest_admissible_lookahead()


def plan_fixed_horizon(rh, rhdot, prop, tf_fix, sigma=1.0):
    \"\"\"Mechanism 3 planner: closed-form trajectory at a FIXED horizon
    tf_fix (bypasses the free-terminal-time root-finding), used when the
    rejection counter overflows: a short, admissible-uncertainty plan
    replaces an ever-staler one.\"\"\"
    q_tf, omega_tf, r_tf, _ = prop(tf_fix)
    q_tf = q_tf / max(np.linalg.norm(q_tf), 1e-12)
    rc_tf, rcdot_tf, _ = _grapple_kinematics(q_tf, omega_tf, r_tf,
                                               RHO_BODY, IC)
    k0, k1, k2, k3 = _trajectory_coefficients(rh, rhdot, rc_tf, rcdot_tf,
                                                tf_fix, sigma)
    return RendezvousTrajectory(k0=k0, k1=k1, k2=k2, k3=k3, sigma=sigma,
                                  tf=tf_fix, rc_tf=rc_tf,
                                  rcdot_tf=rcdot_tf)""")

rep("run_one counters",
    '''    n_steps = int(round(t_final / dt))
    errs_pos, errs_ori = [], []
    n_rejected = 0
    max_q_hat = initial_q_hat''',
    '''    n_steps = int(round(t_final / dt))
    errs_pos, errs_ori = [], []
    n_rejected = 0
    n_contingency = 0
    consec_rej = 0
    n_sat = 0
    q_hat_last_accept = float(initial_q_hat)
    max_q_hat = initial_q_hat''')

# ---- 3. rejection-overflow contingency -------------------------------------
rep("contingency logic",
    '''                if new_q_hat > REJECT_THRESHOLD:
                    # Strategy 1: REJECT - keep using previous trajectory
                    n_rejected += 1
                else:
                    # Accept plan
                    traj = new_traj''',
    '''                if new_q_hat > REJECT_THRESHOLD and consec_rej < 4:
                    # Strategy 1: REJECT - keep using previous trajectory
                    n_rejected += 1
                    consec_rej += 1
                else:
                    if new_q_hat > REJECT_THRESHOLD:
                        # Mechanism 3: rejection-overflow contingency.
                        # After 4 consecutive rejections, replace the stale
                        # plan with a SHORT fixed-horizon plan at the largest
                        # admissible lookahead (q_hat <= threshold), instead
                        # of accepting the long uncertain candidate.
                        n_rejected += 1
                        n_contingency += 1
                        new_traj = plan_fixed_horizon(rh, rhdot, prop,
                                                        TAU_CONTINGENCY)
                        new_q_hat = q_hat_ori(TAU_CONTINGENCY)
                    consec_rej = 0
                    q_hat_last_accept = float(new_q_hat)
                    # Accept plan
                    traj = new_traj''')

# ---- 4. saturation logging ---------------------------------------------------
rep("saturation logging",
    "        tau, _ = controller.solve(state, ref)",
    '''        tau, _info = controller.solve(state, ref)
        try:
            n_sat += int(bool(_info.get("tau_clipped", False)))
        except Exception:
            pass''')

# ---- 5. evaluation-time alignment -------------------------------------------
rep("aligned evaluation",
    "        true_st = target_sim.state_at(t_now)",
    "        true_st = target_sim.state_at(t_now + dt)"
    "  # aligned with post-step chaser state")

# ---- 6. returns --------------------------------------------------------------
rep("divergence returns",
    "            return None, None, None, None",
    "            return None, None, None, None, None, None, None", count=1)
rep("initial-plan return",
    "    except Exception:\n        return None, None, None, None",
    "    except Exception:\n        return None, None, None, None, None, None, None")
rep("run_one return",
    "    return errs_pos, errs_ori, n_rejected, max_q_hat",
    "    return (errs_pos, errs_ori, n_rejected, max_q_hat,\n"
    "            n_contingency, n_sat, q_hat_last_accept)")

# ---- 7. benchmark loop unpack + records --------------------------------------
rep("loop unpack",
    '''        errs_p, errs_o, n_rej, mq = run_one(i, seed=seed)
        if errs_p is None:
            results.append((i, seed, omegas[i], None, None, None, None, None, None, False))
        else:
            results.append((i, seed, omegas[i],
                              float(np.mean(errs_p[-10:])),
                              float(np.mean(errs_o[-10:])),
                              float(errs_p[-1]),
                              float(errs_o[-1]),
                              int(n_rej),
                              float(mq), True))''',
    '''        errs_p, errs_o, n_rej, mq, n_cont, n_sat, q_last = run_one(i, seed=seed)
        if errs_p is None:
            results.append((i, seed, omegas[i], None, None, None, None, None, None,
                            False, None, None, None))
        else:
            results.append((i, seed, omegas[i],
                              float(np.mean(errs_p[-10:])),
                              float(np.mean(errs_o[-10:])),
                              float(errs_p[-1]),
                              float(errs_o[-1]),
                              int(n_rej),
                              float(mq), True,
                              int(n_cont), int(n_sat), float(q_last)))''')

rep("conformal stats print",
    '''print(f"  Mean max-q_hat seen:       {np.mean(max_qs):.2f} deg")''',
    '''print(f"  Mean max-q_hat seen:       {np.mean(max_qs):.2f} deg")
n_conts = [r[10] for r in results if r[9]]
n_sats = [r[11] for r in results if r[9]]
print(f"  Contingency force-accepts: total {sum(n_conts)}, "
      f"runs with >=1: {sum(1 for c in n_conts if c > 0)}")
print(f"  Saturated steps: {sum(n_sats)} total; "
      f"runs with >=1 clip: {sum(1 for s in n_sats if s > 0)}")''')

rep("record writer",
    '''        "n_rejected": r[7], "max_q_hat": r[8],
        "success": bool(r[9])
    })
with open("path2_conformal_results.json", "w") as f:''',
    '''        "n_rejected": r[7], "max_q_hat": r[8],
        "success": bool(r[9]),
        "n_contingency": r[10], "n_sat_steps": r[11],
        "q_hat_last_accept": r[12]
    })
with open("path2_conformal_v3_results.json", "w") as f:''')

rep("save message",
    'print("\\nRaw saved to path2_conformal_results.json")',
    'print("\\nRaw saved to path2_conformal_v3_results.json")')

rep("header banner",
    'print(f"Stage 2+3: Conformal-aware controller benchmark (45 traj x 3 seeds)")',
    'print(f"Stage 2+3 v2: COMBINED-RERUN conformal benchmark (45 traj x 3 seeds)")\n'
    'print(f"  [v2] hierarchical grasp-point calibration | sigmoid gains | "\n'
    '      f"contingency | aligned eval | saturation log")')

# ---- write + verify ----------------------------------------------------------
open(DST, "w", encoding="utf-8").write(src)
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
    py_compile.compile(DST, doraise=True)
    print(f"Compile check OK -> {DST}")
except py_compile.PyCompileError as e:
    print("COMPILE FAILED:", e)
