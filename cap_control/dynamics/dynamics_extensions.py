"""Reduced free-floating dynamics used by this study's computed-torque
controller.

The published space-robot-dq library provides the kinematics, the generalized
Jacobian, the base-velocity map, and the inertia matrices (H_b, H_bm). The
computed-torque law of this paper additionally needs the Umetani-Yoshida
reduced model,

    H_tilde(q) qddot + c_tilde(q, qdot) = tau,
    H_tilde = H_m - H_bm^T H_b^-1 H_bm,

with Coriolis terms obtained from Christoffel symbols of H_tilde. Those three
methods are implemented here and attached to SpaceRobotDynamics at import
time, so the library itself is used exactly as released.

Import once (cap_control/__init__.py does this) before constructing a chaser.
"""
import numpy as np

from space_robot_dq.dynamics import SpaceRobotDynamics

REQUIRED_BASE_API = ("compute_link_states", "compute_link_jacobians",
                     "compute_inertia_matrices")
_missing = [m for m in REQUIRED_BASE_API if not hasattr(SpaceRobotDynamics, m)]
if _missing:
    raise ImportError(
        "space_robot_dq is missing required methods: %s. Install v0.3.1 or "
        "newer:  pip install \"space-robot-dq @ "
        "git+https://github.com/HJahanshahi/space-robot-dq@v0.3.1\""
        % ", ".join(_missing))


def compute_arm_inertia(self, q):
    """Arm spatial inertia matrix with the base held fixed.

    H_m[a, b] is the contribution of joint motion to kinetic energy with a
    stationary base, summed over links as
        m_i (J_v_i^T J_v_i) + J_w_i^T I_i_world J_w_i.

    Returns:
        H_m: (N x N) symmetric positive-definite arm inertia matrix.
    """
    q = np.array(q).flatten()[:self.n_joints]
    _, rotations = self.compute_link_states(q)
    link_jacs = self.compute_link_jacobians(q)
    H_m = np.zeros((self.n_joints, self.n_joints))
    for i in range(self.n_joints):
        m_i = self.link_masses[i]
        I_i_world = rotations[i] @ self.link_inertias[i] @ rotations[i].T
        J_v_i = link_jacs[i][3:6, :]
        J_w_i = link_jacs[i][0:3, :]
        H_m += m_i * (J_v_i.T @ J_v_i) + J_w_i.T @ I_i_world @ J_w_i
    return H_m


def compute_effective_arm_inertia(self, q):
    """Effective arm inertia of the reduced free-floating model.

    For zero initial momentum the equations of motion reduce to
        H_tilde(q) qddot + c_tilde(q, qdot) = tau,
        H_tilde = H_m - H_bm^T H_b^-1 H_bm,
    the standard Umetani-Yoshida reduced free-floating model.

    Returns:
        H_tilde: (N x N) effective arm inertia matrix.
    """
    H_b, H_bm = self.compute_inertia_matrices(q)
    H_m = compute_arm_inertia(self, q)
    return H_m - H_bm.T @ np.linalg.solve(H_b, H_bm)


def compute_coriolis_term(self, q, qdot, eps=1e-5):
    """Coriolis and centrifugal term of the reduced free-floating model.

    c_tilde[k] = sum_{i,j} c_{ijk}(q) qdot[i] qdot[j], with Christoffel
    symbols derived from H_tilde,
        c_{ijk} = 0.5 (dH_tilde[k,i]/dq_j + dH_tilde[k,j]/dq_i
                       - dH_tilde[i,j]/dq_k).
    Partial derivatives use central finite differences, costing 2N
    evaluations of H_tilde per call.

    Args:
        q: (N,) joint angles.
        qdot: (N,) joint velocities.
        eps: finite-difference step.

    Returns:
        c_tilde: (N,) Coriolis and centrifugal force vector.
    """
    q = np.array(q, dtype=float).flatten()[:self.n_joints]
    qdot = np.array(qdot, dtype=float).flatten()[:self.n_joints]
    n = self.n_joints
    dH = np.zeros((n, n, n))
    for k in range(n):
        qp = q.copy(); qp[k] += eps
        qm = q.copy(); qm[k] -= eps
        H_p = compute_effective_arm_inertia(self, qp)
        H_m_ = compute_effective_arm_inertia(self, qm)
        dH[k] = (H_p - H_m_) / (2.0 * eps)
    # c_{ijk} = 0.5 (dH_ki/dq_j + dH_kj/dq_i - dH_ij/dq_k), with
    # dH[k, i, j] = dH_ij/dq_k. This convention satisfies the passivity
    # property: Hdot - 2C is skew-symmetric.
    christoffel = 0.5 * (
        np.einsum("jki->ijk", dH) +
        np.einsum("ikj->ijk", dH) -
        np.einsum("kij->ijk", dH)
    )
    return np.einsum("ijk,i,j->k", christoffel, qdot, qdot)


def forward_dynamics(self, q, qdot, tau, include_coriolis=True):
    """Forward dynamics of the reduced free-floating model.

    Solves qddot = H_tilde^-1 (tau - c_tilde). Assumes zero initial
    momentum; the base velocity follows from momentum conservation and is
    not a free parameter (see compute_base_velocity).

    Args:
        q: (N,) joint angles (rad).
        qdot: (N,) joint velocities (rad/s).
        tau: (N,) joint torques (N.m).
        include_coriolis: include c_tilde(q, qdot). Setting it False saves
            2N inertia evaluations per call; the truth simulator keeps it
            True.

    Returns:
        qddot: (N,) joint accelerations (rad/s^2).
    """
    q = np.array(q, dtype=float).flatten()[:self.n_joints]
    qdot = np.array(qdot, dtype=float).flatten()[:self.n_joints]
    tau = np.array(tau, dtype=float).flatten()[:self.n_joints]
    H_tilde = compute_effective_arm_inertia(self, q)
    if include_coriolis:
        c_tilde = compute_coriolis_term(self, q, qdot)
        return np.linalg.solve(H_tilde, tau - c_tilde)
    return np.linalg.solve(H_tilde, tau)


for _name, _fn in (("compute_arm_inertia", compute_arm_inertia),
                   ("compute_effective_arm_inertia", compute_effective_arm_inertia),
                   ("compute_coriolis_term", compute_coriolis_term),
                   ("forward_dynamics", forward_dynamics)):
    if not hasattr(SpaceRobotDynamics, _name):
        setattr(SpaceRobotDynamics, _name, _fn)

__all__ = ["compute_arm_inertia", "compute_effective_arm_inertia",
           "compute_coriolis_term", "forward_dynamics"]
