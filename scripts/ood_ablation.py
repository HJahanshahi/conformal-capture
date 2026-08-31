"""Out-of-distribution ablation: 45 fresh trajectories at 30-50 deg/s (beyond the
0.1-30 deg/s training/calibration law), run through BOTH arms plus a quick
predictor-coverage check of the deployed bounds under the shift.

Run from the project root:  python ood_ablation.py     (~25-35 min)
Nothing existing is modified. Outputs: ood_results.json + console summary.

Steps:
  1. Generate capture_lib_v2/ood_fast_45.npz (w in 30-50 deg/s, seed 2000)
     via the matched generator with EXPLICIT kwargs (shadow-proof).
  2. Run the conformal-v3 arm and the baseline-v2 arm on all 45 x 3 runs,
     redirecting trajectory loading to the OOD file.
  3. Score UPN prediction errors at tau = 1.0 s on the OOD set and report
     empirical coverage of the deployed bounds (34.4 deg / 30.0 cm).
"""
import sys, os, json
import numpy as np

OOD_PATH = os.path.join("capture_lib_v2", "ood_fast_45.npz")

# ---------- 1. generate OOD set (explicit kwargs; import order matters) ----
if not os.path.exists(OOD_PATH):
    sys.path.insert(0, "capture_lib_v2")
    from capture.target.dataset import generate_dataset
    print("Generating OOD set (45 traj, |w0| in 30-50 deg/s, seed 2000)...")
    generate_dataset(n_traj=45, t_final=10.0, n_steps=100,
                     pos_noise_std=0.01, rot_noise_std_deg=1.0, seed=2000,
                     output_file=OOD_PATH,
                     vel_range=(0.01, 0.10),
                     w_bins_deg=[(30.0, 50.0)])
    d = np.load(OOD_PATH)
    wmax = np.rad2deg(np.linalg.norm(d["true_states"][:, :, 10:13], axis=2)).max(axis=1)
    print(f"  traj-max |w|: min {wmax.min():.1f} med {np.median(wmax):.1f} "
          f"max {wmax.max():.1f} deg/s")
else:
    print(f"OOD set exists: {OOD_PATH}")

from cap_control import config as cfg


class OODTarget:
    """DatasetTumblingTarget-compatible loader for the OOD npz."""
    def __init__(self, traj_idx=0):
        data = np.load(OOD_PATH)
        self.true_states = data["true_states"][int(traj_idx)].copy()
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


def load_arm(path):
    src = open(path, encoding="utf-8").read()
    code = src[:src.index('print("=" * 78)')]
    g = {}
    exec(compile(code, path, "exec"), g)
    g["DatasetTumblingTarget"] = OODTarget
    return g


def run_arm(g, name, conformal):
    rows = []
    for traj in range(45):
        for seed in (1, 2, 3):
            out = g["run_one"](traj, seed=seed)
            if conformal:
                ep, eo, nrej, mq, ncont, nsat, qlast = out
            else:
                ep, eo = out[0], out[1]
                nrej = ncont = None
            if ep is None:
                rows.append(dict(traj=traj, seed=seed, pos=None, ori=None,
                                 nrej=nrej, ncont=ncont, ok=False))
            else:
                rows.append(dict(traj=traj, seed=seed, pos=float(ep[-1]),
                                 ori=float(eo[-1]),
                                 nrej=(int(nrej) if nrej is not None else None),
                                 ncont=(int(ncont) if ncont is not None else None),
                                 ok=True))
        done = [r for r in rows if r["ok"]]
        print(f"  [{name}] traj {traj + 1:2d}/45 done "
              f"(completed {len(done)}/{len(rows)})")
    return rows


def summarize(rows, name):
    ok = [r for r in rows if r["ok"]]
    p = np.array([r["pos"] for r in ok])
    o = np.array([r["ori"] for r in ok])
    cr = np.mean([(r["pos"] < 10) and (r["ori"] < 15) for r in ok]) * 100 \
        * len(ok) / len(rows)
    print(f"\n{name}: completed {len(ok)}/{len(rows)}")
    print(f"  pos mean {p.mean():.2f} med {np.median(p):.2f} "
          f"p95 {np.percentile(p, 95):.2f} cm")
    print(f"  ori mean {o.mean():.2f} med {np.median(o):.2f} "
          f"p95 {np.percentile(o, 95):.2f} deg")
    print(f"  capture-ready (over all {len(rows)} runs): {cr:.0f}%")
    if rows[0]["nrej"] is not None:
        nr = [r["nrej"] for r in rows if r["nrej"] is not None]
        nc = [r["ncont"] for r in rows if r["ncont"] is not None]
        print(f"  rejections mean {np.mean(nr):.2f} max {max(nr)}; "
              f"contingency accepts total {sum(nc)} in "
              f"{sum(1 for c in nc if c > 0)} runs")
    return dict(name=name, n=len(rows), completed=len(ok),
                pos_mean=float(p.mean()), pos_med=float(np.median(p)),
                ori_mean=float(o.mean()), ori_med=float(np.median(o)),
                capture_ready_pct=float(cr))


print("\nLoading conformal-v3 arm...")
gc = load_arm(r"path2_conformal_v3.py")
print("Loading baseline-v2 arm...")
gb = load_arm(r"path2_winning_v2.py")

print("\nRunning CONFORMAL arm on OOD set...")
rc = run_arm(gc, "conf", conformal=True)
print("\nRunning BASELINE arm on OOD set...")
rb = run_arm(gb, "base", conformal=False)

sc = summarize(rc, "CONFORMAL (OOD 30-50 deg/s)")
sb = summarize(rb, "BASELINE (OOD 30-50 deg/s)")

# ---------- 3. predictor coverage under shift (tau = 1.0 s) ----------------
print("\nPredictor coverage of deployed bounds at tau = 1.0 s on OOD set...")
from cap_control.simulation.sensors import NoisyPoseSensor
from cap_control.prediction.upn_predictor import UPNPredictor


def quat_angle_deg(q1, q2):
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    rw = (q2[0]*q1[0] + q2[1]*q1[1] + q2[2]*q1[2] + q2[3]*q1[3])
    return 2 * np.rad2deg(np.arccos(min(1.0, abs(float(rw)))))


upn = UPNPredictor()
HIST, SDT = cfg.UPN_HISTORY_LEN, cfg.SENSOR_DT
Q_ORI, Q_POS = 34.4, 30.0
hits_o, hits_p, tot = 0, 0, 0
for ti in range(45):
    tgt = OODTarget(ti)
    sensor = NoisyPoseSensor(pos_noise_std=cfg.POS_NOISE_STD,
                             rot_noise_std_deg=cfg.ROT_NOISE_STD_DEG, seed=1)
    obs_h, obs_t = [], []
    for tb in np.arange(HIST) * SDT - (HIST - 1) * SDT:
        p, q = tgt.pose_at(max(0.0, tb))
        obs_h.append(sensor.observe(p, q)); obs_t.append(float(tb))
    tcur = 0.0
    for now_t in (0.0, 1.0, 2.0):
        while tcur <= now_t + 1e-9:
            p, q = tgt.pose_at(tcur)
            obs_h.append(sensor.observe(p, q)); obs_t.append(tcur)
            tcur += SDT
        ho, ht = np.asarray(obs_h[-HIST:]), np.asarray(obs_t[-HIST:])
        ft = np.array([now_t + 1.0])
        try:
            m, _, _ = upn.predict(ho, ht, ft, future_obs=None, use_updates=False)
        except Exception:
            continue
        true_st = tgt.state_at(now_t + 1.0)
        ep = np.linalg.norm(m[-1][0:3] - true_st[0:3]) * 100
        eo = quat_angle_deg(m[-1][6:10], true_st[6:10])
        hits_p += int(ep <= Q_POS); hits_o += int(eo <= Q_ORI); tot += 1
print(f"  OOD coverage at tau=1.0s: ORI {100*hits_o/tot:.1f}%  "
      f"POS {100*hits_p/tot:.1f}%  (in-distribution: 91.3% / 90.4%; "
      f"n = {tot} queries)")

json.dump(dict(conformal=rc, baseline=rb, summary=[sc, sb],
               coverage_ood_tau1={"ori_pct": 100*hits_o/tot,
                                    "pos_pct": 100*hits_p/tot, "n": tot}),
          open("ood_results.json", "w"), indent=2)
print("\nSaved ood_results.json")
