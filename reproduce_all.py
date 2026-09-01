#!/usr/bin/env python3
"""Reproduce every number, table, and figure in the paper.

Usage:
    python reproduce_all.py                 # full pipeline
    python reproduce_all.py --stage NAME    # datasets | calibrate | benchmark | ablations | figures
    python reproduce_all.py --stage train   # optional UPN retraining (approximate)

Everything runs with the repository root as the working directory, exactly
like the authors' original setup; generated benchmark scripts land in the
root. With the shipped dataset, weights, and seeds the outputs match
results/expected_numbers.md exactly.
"""
import argparse
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
S = lambda name: os.path.join("scripts", name)


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}")


REQUIRED_DYN_METHODS = ("compute_generalized_jacobian", "compute_base_velocity",
                         "compute_inertia_matrices", "compute_link_jacobians")


def preflight():
    """Fail fast and clearly if space_robot_dq lacks the dynamics API this
    paper's controller needs (the released 0.2.0 does not have it; use
    v0.3.0 or newer)."""
    try:
        import space_robot_dq
        from space_robot_dq.dynamics import SpaceRobotDynamics
    except Exception as e:
        sys.exit("space_robot_dq is not importable (%s).\n"
                 "Install it with:  pip install "
                 "\"space-robot-dq @ git+https://github.com/HJahanshahi/space-robot-dq@v0.3.1\"" % e)
    missing = [m for m in REQUIRED_DYN_METHODS
               if not hasattr(SpaceRobotDynamics, m)]
    if missing:
        sys.exit(
            "space_robot_dq at %s is missing: %s\n"
            "This build predates the API this study relies on.\n"
            "Install v0.3.1 or newer:  pip install --force-reinstall "
            "\"space-robot-dq @ git+https://github.com/HJahanshahi/space-robot-dq@v0.3.1\""
            % (os.path.dirname(space_robot_dq.__file__), ", ".join(missing)))
    print("preflight OK: space_robot_dq %s at %s"
          % (getattr(space_robot_dq, "__version__", "?"),
             os.path.dirname(space_robot_dq.__file__)))


def need(path, hint):
    if not os.path.exists(os.path.join(ROOT, path)):
        sys.exit(f"Missing {path}. {hint}")


def sha(path):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, path), "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def stage_datasets():
    need("capture_lib_v2/capture", "Copy the capture_lib_v2 package in first (see README).")
    # The benchmark dataset is SHIPPED (exact reproduction requires the identical file).
    need("capture_lib_v2/tumbling_target_dataset_v2.npz",
         "The benchmark dataset ships with the repository; restore it from git.")
    print("benchmark dataset sha256:", sha("capture_lib_v2/tumbling_target_dataset_v2.npz"))
    # Calibration set is regenerated from its fixed seed and explicit generator law.
    run([sys.executable, "-c", (
        "import sys; sys.path.insert(0,'capture_lib_v2');"
        "from capture.target.dataset import generate_dataset;"
        "generate_dataset(n_traj=200, t_final=10.0, n_steps=100,"
        " pos_noise_std=0.01, rot_noise_std_deg=1.0, seed=1000,"
        " vel_range=(0.01, 0.10),"
        " w_bins_deg=[(0.1,3.0),(3.0,10.0),(10.0,20.0),(20.0,30.0)],"
        " output_file='capture_lib_v2/calibration_regen_200.npz')")])
    import numpy as np
    A = np.load(os.path.join(ROOT, 'capture_lib_v2/calibration_regen_200.npz'))
    B = np.load(os.path.join(ROOT, 'capture_lib_v2/calibration_matched_200.npz'))
    same = all(k in B.files and np.array_equal(A[k], B[k]) for k in A.files)
    print('regenerated calibration set vs shipped (array-level):',
          'IDENTICAL' if same else 'DIFFERENT (generator nondeterminism - report this)')
    print("Calibration set regenerated (seed 1000) and verified against the shipped file. The OOD set is generated "
          "inside scripts/ood_ablation.py (seed 2000).")


def stage_train():
    print("Optional and approximate (GPU nondeterminism); shipped weights are the paper's.")
    need("cap_control/prediction", "Copy the cap_control package in first.")
    run([sys.executable, S("train_upn.py")])


def stage_calibrate():
    preflight()
    need("cap_control", "Copy the cap_control package in first.")
    run([sys.executable, S("conformal_calibration_v2.py")])


def stage_benchmark():
    preflight()
    need("phase1_final/path2_conformal.py", "Copy the phase-1 originals into phase1_final/.")
    need("phase1_final/path2_full_winning.py", "Copy the phase-1 originals into phase1_final/.")
    run([sys.executable, S("make_rerun_baseline.py")])
    run([sys.executable, "path2_winning_v2.py"])
    run([sys.executable, S("make_rerun_benchmark_v3.py")])
    run([sys.executable, "path2_conformal_v3.py"])


def stage_ablations():
    preflight()
    need("path2_conformal_v3.py", "Run --stage benchmark first.")
    run([sys.executable, S("make_tau_schedule.py")])
    run([sys.executable, "path2_tau_schedule.py"])
    run([sys.executable, S("ood_ablation.py")])
    run([sys.executable, S("c6_exact_xi.py")])


def stage_figures():
    need("calibration_comparison.json", "Run --stage calibrate first.")
    run([sys.executable, S("make_all_figures.py")])
    print("Figures and LaTeX tables written; compare against results/expected_numbers.md")


STAGES = {"datasets": stage_datasets, "train": stage_train,
          "calibrate": stage_calibrate, "benchmark": stage_benchmark,
          "ablations": stage_ablations, "figures": stage_figures}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES), default=None)
    a = ap.parse_args()
    for s in ([a.stage] if a.stage else
              ["datasets", "calibrate", "benchmark", "ablations", "figures"]):
        print(f"\n{'=' * 70}\nSTAGE: {s}\n{'=' * 70}")
        STAGES[s]()
    print("\nDone. Compare results against results/expected_numbers.md")
