"""Exact vs simplified orientation law on a 10-run subset.

A = deployed law: residual = omega_dot_cmd - J_ang @ qddot_primary
B = exact law:    residual = omega_dot_cmd - Jdot_ang @ qdot - J_ang @ qddot_primary
    with Jdot_ang @ qdot evaluated by finite difference along qdot
    (base-coupling rate terms remain approximated; see note in output).

Patches applied IN MEMORY only. Run:  python c6_exact_xi.py   (~4-6 min)
"""
import numpy as np
import cap_control.controller.feedback_linearization as fb

fb_src = open(fb.__file__, encoding="utf-8").read()

OLD = "            residual = omega_dot_cmd_base - J_ang_base @ qddot_primary"
NEW = """            # EXACT-LAW VARIANT: include the Jacobian-rate term.
            _eps = 1e-5
            _Jg2 = self.chaser.dyn.compute_generalized_jacobian(
                state.q + _eps * state.qdot)[0]
            _Jdot_qdot = (_Jg2[0:3, :] - J_ang_base) @ state.qdot / _eps
            XI_TERM_LOG.append(float(np.linalg.norm(_Jdot_qdot)))
            residual = (omega_dot_cmd_base - _Jdot_qdot
                        - J_ang_base @ qddot_primary)"""
assert fb_src.count(OLD) == 1, ""
fb_patched = fb_src.replace(OLD, NEW)
fb_patched = "XI_TERM_LOG = []\n" + fb_patched

ns = {}
exec(compile(fb_patched, fb.__file__ + " [exact-xi]", "exec"), ns)
ExactController = ns["FeedbackLinearizationController"]

SRC = "path2_conformal_v3.py"
src = open(SRC, encoding="utf-8").read()
code = src[:src.index('print("=" * 78)')]
g = {}
exec(compile(code, SRC, "exec"), g)
OrigController = g["FeedbackLinearizationController"]

CASES = [(13, 2), (13, 1), (39, 1), (22, 1), (31, 2),
         (5, 1), (0, 1), (44, 1), (35, 1), (33, 1)]


def run_variant(name, ctrl_cls):
    g["FeedbackLinearizationController"] = ctrl_cls
    out = {}
    for traj, seed in CASES:
        ns["XI_TERM_LOG"].clear() if name == "B" else None
        r = g["run_one"](traj, seed=seed)
        ep, eo, nrej = r[0], r[1], r[2]
        if ep is None:
            out[(traj, seed)] = None
            print(f"  [{name}] traj {traj:2d} s{seed}: DIVERGED")
        else:
            xi = (max(ns["XI_TERM_LOG"]) if (name == "B" and ns["XI_TERM_LOG"])
                  else float("nan"))
            out[(traj, seed)] = (ep[-1], eo[-1], nrej, xi)
            extra = f"  max|Jdot qdot|={xi:6.3f} rad/s^2" if name == "B" else ""
            print(f"  [{name}] traj {traj:2d} s{seed}: {ep[-1]:6.2f} cm / "
                  f"{eo[-1]:6.1f} deg (rej={nrej}){extra}")
    return out


print("Variant A: deployed (simplified) orientation law")
A = run_variant("A", OrigController)
print("\nVariant B: exact law with Jacobian-rate term")
B = run_variant("B", ExactController)

print("\n" + "=" * 74)
print(f"{'case':>9s} {'A pos':>7s} {'B pos':>7s} {'A ori':>7s} {'B ori':>7s} "
      f"{'dPos':>7s} {'dOri':>7s} {'max|Jw qd|':>11s}")
dps, dos = [], []
for k in CASES:
    a, b = A[k], B[k]
    if a is None or b is None:
        print(f"{str(k):>9s}  (divergence in one variant)")
        continue
    dp, do = b[0] - a[0], b[1] - a[1]
    dps.append(dp); dos.append(do)
    print(f"{str(k):>9s} {a[0]:7.2f} {b[0]:7.2f} {a[1]:7.1f} {b[1]:7.1f} "
          f"{dp:+7.2f} {do:+7.1f} {b[3]:11.3f}")
print(f"\nmean |delta|: position {np.mean(np.abs(dps)):.3f} cm, "
      f"orientation {np.mean(np.abs(dos)):.3f} deg")
print("Note: the finite-difference term captures the joint-space Jacobian "
      "rate; base-coupling\nrate terms of the generalized Jacobian remain "
      "approximated, so B bounds the correction's\nrelevance rather than "
      "implementing every rate term exactly.")
