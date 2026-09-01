"""Check the Coriolis convention on the actual chaser model, and quantify how
much the correction changes the deployed torques.

Run from the repository root:  python check_coriolis.py

Prints:
  1. the passivity residual (Hdot - 2C skew-symmetry) for both conventions;
  2. the size of the two c_tilde vectors at representative states;
  3. the resulting torque difference against the 10 N.m saturation limit.
"""
import numpy as np

import cap_control  # noqa: F401  (attaches the reduced-model methods)
from cap_control.dynamics.free_floating import FreeFloatingChaser

chaser = FreeFloatingChaser()
dyn = chaser.dyn
n = chaser.n_joints
rng = np.random.default_rng(0)


def dH_tensor(q, eps=1e-5):
    dH = np.zeros((n, n, n))
    for k in range(n):
        qp = q.copy(); qp[k] += eps
        qm = q.copy(); qm[k] -= eps
        dH[k] = (dyn.compute_effective_arm_inertia(qp)
                 - dyn.compute_effective_arm_inertia(qm)) / (2 * eps)
    return dH


def c_standard(q, qdot):
    dH = dH_tensor(q)
    chr_ = 0.5 * (np.einsum("jki->ijk", dH) + np.einsum("ikj->ijk", dH)
                  - np.einsum("kij->ijk", dH))
    return np.einsum("ijk,i,j->k", chr_, qdot, qdot), chr_


def c_legacy(q, qdot):
    dH = dH_tensor(q)
    chr_ = 0.5 * (np.einsum("jki->ijk", dH) + np.einsum("ikj->ijk", dH) - dH)
    return np.einsum("ijk,i,j->k", chr_, qdot, qdot), chr_


def passivity_residual(chr_, q, qdot, eps=1e-5):
    C = np.einsum("ijk,i->kj", chr_, qdot)
    Hdot = np.zeros((n, n))
    for k in range(n):
        qp = q.copy(); qp[k] += eps
        qm = q.copy(); qm[k] -= eps
        Hdot += ((dyn.compute_effective_arm_inertia(qp)
                  - dyn.compute_effective_arm_inertia(qm)) / (2 * eps)) * qdot[k]
    S = Hdot - 2 * C
    return np.abs(S + S.T).max()


print(f"{'|qdot|':>8} {'|c_std|':>9} {'|c_legacy|':>11} {'|diff|':>9} "
      f"{'skew_std':>10} {'skew_legacy':>12}")
state = chaser.home()
for scale in (0.5, 1.0, 2.0, 5.0):
    q = np.asarray(state.q, dtype=float).flatten()[:n] + 0.1 * rng.normal(size=n)
    qdot = scale * rng.normal(size=n)
    cs, chr_s = c_standard(q, qdot)
    cl, chr_l = c_legacy(q, qdot)
    print(f"{np.linalg.norm(qdot):8.2f} {np.linalg.norm(cs):9.3f} "
          f"{np.linalg.norm(cl):11.3f} {np.linalg.norm(cs - cl):9.3f} "
          f"{passivity_residual(chr_s, q, qdot):10.2e} "
          f"{passivity_residual(chr_l, q, qdot):12.2e}")

print("\nInterpretation: the two conventions cancel exactly in the closed loop "
      "(controller and plant\nuse the same term), so trajectories are "
      "unaffected except through torque saturation.\nCompare |diff| against "
      "the 10 N.m clip limit to judge whether saturation behaviour changes.")
