"""Integration-convergence study for the truth simulator.

The plant advances one explicit step per 0.1 s control period. With the exact
reduced dynamics (Coriolis quadratic in joint rate) that step may be far from
converged. This script re-runs representative cases with the control torque
held constant over the period (zero-order hold) and the plant advanced in N
substeps, for increasing N, so the value of N at which terminal errors stop
moving can be read off.

Run from the repository root:
    .venv\\Scripts\\python convergence_test.py
"""
import time
import numpy as np

SRC = "path2_conformal_v3.py"
src = open(SRC, encoding="utf-8").read()
g = {}
exec(compile(src[:src.index('print("=" * 78)')], SRC, "exec"), g)

from cap_control.dynamics.free_floating import FreeFloatingChaser

_orig_step = FreeFloatingChaser.dynamic_step
N_SUB = 1


def dynamic_step_sub(self, state, tau, dt, include_coriolis=True):
    h = dt / N_SUB
    for _ in range(N_SUB):
        state = _orig_step(self, state, tau, h,
                           include_coriolis=include_coriolis)
        if not np.all(np.isfinite(state.qdot)):
            return state
    return state


FreeFloatingChaser.dynamic_step = dynamic_step_sub

CASES = [(13, 2), (39, 1), (35, 1),   # benign
         (33, 1), (26, 1), (5, 2)]    # hard
SUBSTEPS = [1, 2, 5, 10, 20, 40, 80]

print("terminal position (cm) / orientation (deg) versus substeps per "
      "control period\n")
header = "case      " + "".join(f"{'N=%d' % n:>20}" for n in SUBSTEPS)
print(header)
for traj, seed in CASES:
    row = f"({traj:2d},{seed})  "
    for n in SUBSTEPS:
        globals()["N_SUB"] = n
        out = g["run_one"](traj, seed=seed)
        if out[0] is None:
            row += f"{'DIVERGED':>20}"
        else:
            row += f"{out[0][-1]:9.2f} /{out[1][-1]:7.1f}  "
    print(row)

print("\nTiming for one run at each N (traj 13, seed 2):")
for n in SUBSTEPS:
    globals()["N_SUB"] = n
    t0 = time.time()
    g["run_one"](13, seed=2)
    el = time.time() - t0
    print(f"  N={n:3d}: {el:6.2f} s per run -> {el * 270 / 60:6.1f} min "
          f"for both arms")

FreeFloatingChaser.dynamic_step = _orig_step
print("\nRead the smallest N beyond which the terminal values stop changing "
      "materially;\nthat is the converged setting the benchmark should use.")
