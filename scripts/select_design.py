"""Select the deployed mechanism design on a large validation sample.

The earlier five-trajectory validation set did not contain the long-horizon
regime that dominates the test failures, which is how a harmful design was
selected. This sweep uses 15 validation trajectories x 2 seeds per design.

Designs compared:
    off        no mechanisms (rejection off, no modulation)
    frozen     current design: rejection + gain set once per accepted plan
    step       rejection + gain recomputed each step from the remaining
               lookahead
    step_norej gain recomputed each step, no rejection or contingency
    step_k2    as step, with a shallower modulation floor (K_min = 2)

Run from the repository root:
    .venv\\Scripts\\python select_design.py
"""
import time

import numpy as np

from cap_control import config as cfg

SRC = "path2_conformal_v3.py"
head = open(SRC, encoding="utf-8").read()
head = head[:head.index('print("=" * 78)')].replace("\r\n", "\n")

c = "        time_to_go = traj.tf - local_t"
d = """        time_to_go = traj.tf - local_t
        if PER_STEP_GAIN:
            _q_now = q_hat_ori(max(float(time_to_go), 1e-3))
            _kp = kp_ori_from_uncertainty(
                _q_now, GAIN_LOW_THRESHOLD, GAIN_HIGH_THRESHOLD,
                KP_ORI_MAX, KP_ORI_MIN)
            if abs(_kp - current_kp_ori) > 1e-9:
                current_kp_ori = _kp
                controller = FeedbackLinearizationController(
                    chaser=chaser, Kp_pos=20.0, Kd_pos=8.0,
                    Kp_ori=current_kp_ori,
                    Kd_ori=current_kp_ori * KD_ORI_RATIO,
                    tau_limit=controller.tau_limit, t_blend_ori=T_BLEND)"""
assert head.count(c) == 1
head = "PER_STEP_GAIN = False\n" + head.replace(c, d)

g = {}
exec(compile(head, SRC, "exec"), g)

DATA = np.load("capture_lib_v2/tumbling_target_dataset_v2.npz")
TRUE = DATA["true_states"]
N_TRAIN = int(0.70 * TRUE.shape[0])
N_VAL = int(0.15 * TRUE.shape[0])


class ValTarget:
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

w_val = [np.rad2deg(np.linalg.norm(TRUE[N_TRAIN + i, :, 10:13], axis=1)).max()
         for i in range(N_VAL)]
PICK = list(np.argsort(w_val)[::3])[:15]      # 15 trajectories spanning the range
SEEDS = [1, 2]
print("validation trajectories:", [(int(i), round(w_val[i], 1)) for i in PICK])

DESIGNS = {
    "off":        dict(thr=1e9,  kmin=10.0, step=False),
    "frozen":     dict(thr=40.0, kmin=1.0,  step=False),
    "step":       dict(thr=40.0, kmin=1.0,  step=True),
    "step_norej": dict(thr=1e9,  kmin=1.0,  step=True),
    "step_k2":    dict(thr=40.0, kmin=2.0,  step=True),
}

print(f"\n{'design':>11} {'n_ok':>5} {'pos med':>8} {'ori med':>8} "
      f"{'ori mean':>9} {'pos p95':>8} {'ori p95':>8} {'CR%':>5} {'strict%':>8}")
t0 = time.time()
summary = {}
for name, d in DESIGNS.items():
    g["REJECT_THRESHOLD"] = d["thr"]
    g["KP_ORI_MIN"] = d["kmin"]
    g["PER_STEP_GAIN"] = d["step"]
    pos, ori = [], []
    for ti in PICK:
        for sd in SEEDS:
            out = g["run_one"](int(ti), seed=sd)
            if out[0] is not None:
                pos.append(out[0][-1]); ori.append(out[1][-1])
    n_tot = len(PICK) * len(SEEDS)
    p = np.array(pos); o = np.array(ori)
    cr = 100 * sum(1 for x, y in zip(p, o) if x < 10 and y < 15) / n_tot
    st = 100 * sum(1 for x, y in zip(p, o) if x < 5 and y < 5) / n_tot
    summary[name] = (cr, st, float(o.mean()))
    print(f"{name:>11} {len(p):5d} {np.median(p):8.2f} {np.median(o):8.2f} "
          f"{o.mean():9.2f} {np.percentile(p,95):8.2f} {np.percentile(o,95):8.2f} "
          f"{cr:5.0f} {st:8.0f}")

print(f"\nelapsed {(time.time()-t0)/60:.1f} min")
print("\nAdopt the design with the best capture-ready rate and no tail "
      "regression;\nthe comparison against 'off' is what the paper must be "
      "able to claim.")
