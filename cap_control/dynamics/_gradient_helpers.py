"""
Helpers that compute intermediate matrices needed for analytical MPC gradients.

All functions take the robot (kinematics + dynamics) and a joint-angle vector q
and return numpy arrays usable in chain-rule calculations.
"""
import numpy as np


def base_map_matrix(dyn, q):
    """
    M(q) such that [v_base_world; omega_base_body] = M(q) @ qdot  (assuming h0 = 0).

    Derivation: the library's compute_base_velocity(q, qdot) implements the
    momentum-conservation formula which is linear in qdot when h0 = 0. We
    extract M column-by-column using unit-qdot probes.

    This is an O(n_joints) operation per call but each probe is a single
    momentum-conservation evaluation, so it's cheap compared to an IK solve
    or a full forward dynamics step.
    """
    q = np.asarray(q, dtype=float)
    n = q.shape[0]
    M = np.zeros((6, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        M[:, j] = np.asarray(dyn.compute_base_velocity(q, e))
    return M


def ee_spatial_jacobian_base(kin, q, h=1e-6):
    """
    End-effector spatial Jacobian in the base frame:
        rows [0:3] = dp_ee_base / dq   (linear)
        rows [3:6] = dtheta_ee_base / dq (angular)

    Computed by central difference on forward_kinematics_6dof.

    NOTE (Step 10a was reverted): we tried switching to the library's
    calculate_jacobian for theoretical cleanliness but it caused regressions
    on Steps 6 and 7b due to a frame-convention mismatch (linear rows in
    forward_kinematics may be world-frame, not base-frame). The finite-diff
    version is what produced the Step 9 benchmark, so we keep it.
    """
    from cap_control.utils.transforms import pose_error
    q = np.asarray(q, dtype=float)
    n = q.shape[0]
    J = np.zeros((6, n))
    for j in range(n):
        qp = q.copy(); qp[j] += h
        qm = q.copy(); qm[j] -= h
        pp, qp_quat = kin.forward_kinematics_6dof(qp)
        pm, qm_quat = kin.forward_kinematics_6dof(qm)
        J[0:3, j] = (np.asarray(pp) - np.asarray(pm)) / (2.0 * h)
        _pos_err, rot_err = pose_error(
            (np.zeros(3), np.asarray(qp_quat)),
            (np.zeros(3), np.asarray(qm_quat)))
        J[3:6, j] = rot_err / (2.0 * h)
    return J


def quat_rotmat(q_wxyz):
    """[w,x,y,z] -> 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ])


def skew(v):
    return np.array([
        [   0.0, -v[2],  v[1]],
        [ v[2],    0.0, -v[0]],
        [-v[1],  v[0],    0.0],
    ])


def ee_jacobian_base_aw_layout(kin, q):
    """Library [angular;linear] layout. Kept for future omega-cost work."""
    return np.asarray(kin.calculate_jacobian(q))


def generalized_jacobian_aw_layout(dyn, q):
    """J_g in [angular;linear] base-frame layout. Kept for future work."""
    J_g, _J_m, _J_b = dyn.compute_generalized_jacobian(q)
    return np.asarray(J_g)
