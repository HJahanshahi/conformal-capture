"""Component study of the conformal control mechanisms, validation split.

Crosses the gain-modulation depth against the rejection and contingency
mechanisms, so that each mechanism's contribution is measured rather than
assumed, and the deployed K_min is selected on data that is not the test set.

  modulation : K_min in {1, 2, 5, 10}; K_max = 10, so K_min = 10 disables it
  rejection  : off (threshold set beyond reach) or on (40 deg)
  contingency: off, or on (fires after 4 consecutive rejections)

Rejection-off implies contingency-off, so the grid is 4 x 3 = 12 cells.

Run from the repository root:
    .venv\\Scripts\\python sweep_mechanisms.py
"""
import time

import numpy as np

from cap_control import config as cfg

SRC = "path2_conformal_v3.py"
raw = open(SRC, encoding="utf-8").read()
head = raw[:raw.index('print("=" * 78)')]

# expose the contingency trigger as a global so it can be disabled
assert "consec_rej < 4" in head, "contingency trigger not found"
head = head.replace("consec_rej < 4", "consec_rej < CONT_TRIGGER")
head = "CONT_TRIGGER = 4\n" + head

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
order = np.argsort(w_val)
PICK = [int(order[int(f * (N_VAL - 1))]) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
SEEDS = [1, 2]
print("validation trajectories:", [(i, round(w_val[i], 1)) for i in PICK])
print(f"K_max = {g['KP_ORI_MAX']}, tau_limit = 20 N.m, "
      f"PLANT_SUBSTEPS = {cfg.PLANT_SUBSTEPS}\n")

MODES = [("reject off", 1e9, 10 ** 9),
         ("reject only", 40.0, 10 ** 9),
         ("reject+conting", 40.0, 4)]
KMINS = [1.0, 2.0, 5.0, 10.0]

print(f"{'mode':>15} {'K_min':>6} {'n_ok':>5} {'pos med':>8} {'ori med':>8} "
      f"{'ori mean':>9} {'CR%':>5} {'strict%':>8} {'rej':>5} {'cont':>5}")
rows = []
t0 = time.time()
for mode, thr, trig in MODES:
    for kmin in KMINS:
        g["REJECT_THRESHOLD"] = thr
        g["CONT_TRIGGER"] = trig
        g["KP_ORI_MIN"] = kmin
        pos, ori, nrej, ncont, nok = [], [], [], [], 0
        for ti in PICK:
            for sd in SEEDS:
                out = g["run_one"](ti, seed=sd)
                if out[0] is not None:
                    pos.append(out[0][-1]); ori.append(out[1][-1]); nok += 1
                    nrej.append(out[2]); ncont.append(out[4])
        n_tot = len(PICK) * len(SEEDS)
        if not pos:
            print(f"{mode:>15} {kmin:6.1f} {nok:5d}   all diverged")
            continue
        cr = 100 * sum(1 for p, o in zip(pos, ori) if p < 10 and o < 15) / n_tot
        st = 100 * sum(1 for p, o in zip(pos, ori) if p < 5 and o < 5) / n_tot
        rec = dict(mode=mode, kmin=kmin, n_ok=nok, n_tot=n_tot,
                   pos_med=float(np.median(pos)), ori_med=float(np.median(ori)),
                   ori_mean=float(np.mean(ori)), cr=cr, strict=st,
                   rej=float(np.mean(nrej)), cont=int(np.sum(ncont)))
        rows.append(rec)
        print(f"{mode:>15} {kmin:6.1f} {nok:5d} {rec['pos_med']:8.2f} "
              f"{rec['ori_med']:8.2f} {rec['ori_mean']:9.2f} {cr:5.0f} "
              f"{st:8.0f} {rec['rej']:5.2f} {rec['cont']:5d}")

print(f"\nelapsed {(time.time() - t0) / 60:.1f} min")
if rows:
    print("\nranked by capture-ready, then completions, then mean orientation:")
    for r in sorted(rows, key=lambda r: (-r["cr"], -r["n_ok"], r["ori_mean"]))[:5]:
        print(f"  {r['mode']:>15} K_min={r['kmin']:.0f}: CR {r['cr']:.0f}%, "
              f"strict {r['strict']:.0f}%, ori mean {r['ori_mean']:.2f}, "
              f"{r['n_ok']}/{r['n_tot']} completed")
    print("\nMechanism contributions at the best K_min (compare down a column):")
    best_k = sorted(rows, key=lambda r: (-r["cr"], r["ori_mean"]))[0]["kmin"]
    for r in [x for x in rows if x["kmin"] == best_k]:
        print(f"  {r['mode']:>15}: CR {r['cr']:.0f}%, ori mean "
              f"{r['ori_mean']:.2f}, completed {r['n_ok']}/{r['n_tot']}")
