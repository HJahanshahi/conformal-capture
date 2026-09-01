"""Re-tune the shared controller gains on the corrected plant.

The deployed gains were selected on a plant that used an incorrect Coriolis
term and a single explicit integration step per control period. Both are now
fixed, so the selection must be repeated. Two changes from the original
protocol:

  * the sweep runs on the VALIDATION split, never on the held-out test
    trajectories used for the reported benchmark;
  * the torque limit is swept alongside the gains, because clipping is
    active in a third of the hard runs once the plant is integrated properly.

The winner is adopted by BOTH arms (the conformal arm modulates the
orientation gain between K_min and the selected K_p), so the comparison
stays fair.

Run from the repository root:
    .venv\\Scripts\\python sweep_gains.py
"""
import itertools
import time

import numpy as np

from cap_control import config as cfg
from cap_control.simulation.sensors import NoisyPoseSensor  # noqa: F401

SRC = "path2_winning_v2.py"
src = open(SRC, encoding="utf-8").read()
g = {}
exec(compile(src[:src.index('print("=" * 78)')], SRC, "exec"), g)

DATA = np.load("capture_lib_v2/tumbling_target_dataset_v2.npz")
TRUE = DATA["true_states"]
N_TRAIN = int(0.70 * TRUE.shape[0])
N_VAL = int(0.15 * TRUE.shape[0])


class ValTarget:
    """Validation-split trajectory, same interface as the benchmark target."""

    def __init__(self, traj_idx=0):
        self.true_states = TRUE[N_TRAIN + int(traj_idx)].copy()
        self.times = DATA["times"].copy()
        self.dt = float(self.times[1] - self.times[0])
        self.T_final = float(self.times[-1])
        off = (np.array([0.0, 0.0, 0.65 * cfg.ARM_REACH])
               - self.true_states[0, 0:3])
        self.true_states[:, 0:3] += off

    def state_at(self, t):
        if t <= 0.0:
            return self.true_states[0].copy()
        if t >= self.T_final:
            return self.true_states[-1].copy()
        i = int(t / self.dt)
        fr = (t / self.dt) - i
        s = (1 - fr) * self.true_states[i] + fr * self.true_states[i + 1]
        s[6:10] = s[6:10] / max(np.linalg.norm(s[6:10]), 1e-12)
        return s

    def pose_at(self, t):
        s = self.state_at(t)
        return s[0:3], s[6:10] / max(np.linalg.norm(s[6:10]), 1e-12)


g["DatasetTumblingTarget"] = ValTarget

# validation trajectories spanning the tumble-rate range
w_val = [np.rad2deg(np.linalg.norm(TRUE[N_TRAIN + i, :, 10:13], axis=1)).max()
         for i in range(N_VAL)]
order = np.argsort(w_val)
PICK = [int(order[int(f * (N_VAL - 1))]) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
print("validation trajectories used:",
      [(i, round(w_val[i], 1)) for i in PICK])

Base = g["FeedbackLinearizationController"]
TAU_LIMIT = 10.0


class CtrlTau(Base):
    def __init__(self, *a, **k):
        k["tau_limit"] = TAU_LIMIT
        super().__init__(*a, **k)


g["FeedbackLinearizationController"] = CtrlTau

GRID = [(2.0, 2.0, 1.5), (5.0, 4.0, 1.5), (10.0, 6.0, 1.5), (20.0, 8.0, 1.5),
        (5.0, 4.0, 0.5), (5.0, 4.0, 1.0), (10.0, 6.0, 1.0), (5.0, 8.0, 1.5)]
SEEDS = [1, 2]
TAUS = [10.0, 20.0]

print(f"\n{'tau_lim':>8} {'Kp':>5} {'Kd':>5} {'blend':>6} {'n_ok':>5} "
      f"{'pos med':>8} {'ori med':>8} {'ori mean':>9} {'CR%':>5}")
results = []
t0 = time.time()
for tau_lim, (kp, kd, blend) in itertools.product(TAUS, GRID):
    globals()["TAU_LIMIT"] = TAU_LIMIT = tau_lim
    g["KP_ORI"], g["KD_ORI"], g["T_BLEND"] = kp, kd, blend
    pos, ori, nok = [], [], 0
    for ti in PICK:
        for sd in SEEDS:
            out = g["run_one"](ti, seed=sd)
            if out[0] is not None:
                pos.append(out[0][-1]); ori.append(out[1][-1]); nok += 1
    n_tot = len(PICK) * len(SEEDS)
    if not pos:
        print(f"{tau_lim:8.0f} {kp:5.1f} {kd:5.1f} {blend:6.1f} {nok:5d}  all diverged")
        continue
    cr = 100 * sum(1 for p, o in zip(pos, ori) if p < 10 and o < 15) / n_tot
    rec = dict(tau=tau_lim, kp=kp, kd=kd, blend=blend, n_ok=nok,
               pos_med=float(np.median(pos)), ori_med=float(np.median(ori)),
               ori_mean=float(np.mean(ori)), cr=cr)
    results.append(rec)
    print(f"{tau_lim:8.0f} {kp:5.1f} {kd:5.1f} {blend:6.1f} {nok:5d} "
          f"{rec['pos_med']:8.2f} {rec['ori_med']:8.2f} {rec['ori_mean']:9.2f} "
          f"{cr:5.0f}")

print(f"\nelapsed {(time.time() - t0)/60:.1f} min")
if results:
    best = sorted(results, key=lambda r: (-r["cr"], r["ori_mean"]))[:3]
    print("\nranked by capture-ready then mean orientation:")
    for r in best:
        print(f"  tau={r['tau']:.0f} Kp={r['kp']} Kd={r['kd']} "
              f"blend={r['blend']}: CR {r['cr']:.0f}%, ori mean "
              f"{r['ori_mean']:.2f}, {r['n_ok']}/{len(PICK)*len(SEEDS)} completed")
    print("\nAdopt the winner for BOTH arms, then rerun the full pipeline.")
