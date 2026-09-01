"""
Feedback linearization controller for free-floating chaser.

Tracks a desired end-effector position trajectory using computed-torque control:

    tau = H_tilde(q) * qddot_cmd + c_tilde(q, qdot)

where qddot_cmd comes from inverse kinematics of the desired EE acceleration:

    qddot_cmd = J_lin^+ * (rhddot_cmd - J_lin_dot * qdot - base_term)

and rhddot_cmd is feedforward + PD:

    rhddot_cmd = rhddot* + Kd * (rhdot* - rhdot) + Kp * (rh* - rh)

This follows Aghili (2023) eq (14) but adapted for free-floating mode where
no base attitude actuators are available. Only EE position is tracked here;
orientation tracking will be added in a follow-up if needed.

Stability: with the Coriolis-free planning model and exact dynamics in the
truth simulator, the closed loop is locally stable for positive-definite
Kp, Kd. Aghili sec. 2 proves stability for the full 6DOF case.
"""
import numpy as np

from cap_control.utils.rotations import _quat_rotmat

from cap_control.controller.mpc_base import MPCBase


class FeedbackLinearizationController(MPCBase):
    """
    Computed-torque controller that tracks an EE position trajectory.

    Note: this is NOT an MPC. It inherits from MPCBase only for interface
    compatibility with the simulator (which expects a `solve(state, ref)`
    method). There is no optimization at runtime; just direct torque
    computation.

    Args:
        chaser: FreeFloatingChaser instance
        Kp_pos: (3,) or scalar - proportional gain on position error
        Kd_pos: (3,) or scalar - derivative gain on velocity error
        tau_limit: scalar - clip joint torques to [-tau_limit, +tau_limit]
        damping_lambda: scalar - Tikhonov damping for inverse Jacobian (handles
            singular configurations gracefully)
    """

    def __init__(self, chaser, Kp_pos=20.0, Kd_pos=8.0,
                 Kp_ori=20.0, Kd_ori=8.0,
                 tau_limit=20.0, damping_lambda=0.01,
                 t_blend_ori=0.5):
        super().__init__()
        self.chaser = chaser
        self.n_joints = chaser.n_joints
        # Convert scalar gains to diagonal (3,3) matrices
        if np.isscalar(Kp_pos):
            self.Kp_pos = float(Kp_pos) * np.eye(3)
        else:
            self.Kp_pos = np.diag(np.asarray(Kp_pos, dtype=float).flatten())
        if np.isscalar(Kd_pos):
            self.Kd_pos = float(Kd_pos) * np.eye(3)
        else:
            self.Kd_pos = np.diag(np.asarray(Kd_pos, dtype=float).flatten())
        # Orientation gains
        if np.isscalar(Kp_ori):
            self.Kp_ori = float(Kp_ori) * np.eye(3)
        else:
            self.Kp_ori = np.diag(np.asarray(Kp_ori, dtype=float).flatten())
        if np.isscalar(Kd_ori):
            self.Kd_ori = float(Kd_ori) * np.eye(3)
        else:
            self.Kd_ori = np.diag(np.asarray(Kd_ori, dtype=float).flatten())
        self.tau_limit = float(tau_limit)
        self.damping_lambda = float(damping_lambda)
        self.t_blend_ori = float(t_blend_ori)  # Window before tf to enable ori control

    def compute_ee_velocity(self, state):
        """
        Current EE linear velocity in world frame.

        Free-floating: rhdot = R_base * (J_lin * qdot + v_base_in_base_frame)
        Since base velocity follows from momentum conservation given qdot,
        we compute it explicitly via the library's compute_base_velocity.
        """
        # Library returns 6-vec [v_lin(0:3); omega(3:6)] in base frame
        nu_b_body = self.chaser.dyn.compute_base_velocity(state.q, state.qdot)
        v_base_body = nu_b_body[0:3]
        omega_base_body = nu_b_body[3:6]

        # EE position relative to base, in base frame: J_lin in base frame * qdot
        J_full = self.chaser.kin.calculate_jacobian(state.q)
        J_lin_base = J_full[3:6, :]   # rows 3:6 are linear part in base frame

        # EE velocity contribution from joint motion (in base frame)
        rhdot_arm_base = J_lin_base @ state.qdot

        # Total EE velocity in base frame:
        #   v_ee_base = v_base_body + omega_base_body x r_ee_in_base + rhdot_arm_base
        # We need r_ee_in_base. Compute via FK.
        # FK gives world-frame EE; we want EE position relative to base CoM.
        # In base frame: r_ee_base = R_base^T * (r_ee_world - r_base_world)
        R_base = _quat_rotmat(state.base_quat)
        rh_world, _ = self.chaser.fk_world(state)
        r_ee_in_base = R_base.T @ (rh_world - state.base_pos)

        v_ee_base = v_base_body + np.cross(omega_base_body, r_ee_in_base) + rhdot_arm_base
        # Convert to world frame
        v_ee_world = R_base @ v_ee_base
        return v_ee_world

    def compute_ee_angular_velocity(self, state):
        """
        Current EE angular velocity in world frame.

        omega_ee_world = R_base * (J_ang_base * qdot + omega_base_body)
        where the J_ang_base part is the angular component of the Jacobian
        (in base frame) and omega_base_body comes from momentum conservation.
        """
        nu_b_body = self.chaser.dyn.compute_base_velocity(state.q, state.qdot)
        omega_base_body = nu_b_body[3:6]

        J_full = self.chaser.kin.calculate_jacobian(state.q)
        J_ang_base = J_full[0:3, :]
        omega_arm_base = J_ang_base @ state.qdot

        # EE angular velocity in base frame = base angular velocity + arm contribution
        # (Both expressed in base frame, then rotated to world)
        omega_ee_base = omega_base_body + omega_arm_base

        R_base = _quat_rotmat(state.base_quat)
        return R_base @ omega_ee_base

    def solve(self, state, reference):
        """
        Compute torque command. Compatible with simulator's MPC interface.

        Args:
            state: FreeFloatingState
            reference: dict with keys
                "rh_des"      : (3,) desired EE position (world frame)
                "rhdot_des"   : (3,) desired EE velocity
                "rhddot_des"  : (3,) desired EE acceleration

        Returns:
            tau: (n_joints,) joint torques
            info: dict with diagnostic info
        """
        import time as _time
        t_start = _time.time()

        rh_des = np.asarray(reference["rh_des"], dtype=float)
        rhdot_des = np.asarray(reference["rhdot_des"], dtype=float)
        rhddot_des = np.asarray(reference["rhddot_des"], dtype=float)

        # 1. Current EE state
        rh, q_ee = self.chaser.fk_world(state)
        rhdot = self.compute_ee_velocity(state)

        # 2. Tracking errors (position)
        e_pos = rh_des - rh
        e_vel = rhdot_des - rhdot

        # 3. Commanded EE acceleration: feedforward + PD
        rhddot_cmd = rhddot_des + self.Kd_pos @ e_vel + self.Kp_pos @ e_pos

        # ---- Orientation tracking (optional, terminal-only) ----
        # Aghili 2023 sets omega*_h = omega_o at tf only. We follow this by
        # enabling orientation control only in the final t_blend_ori seconds.
        # Outside that window, orientation feedback is disabled to avoid
        # destabilizing dynamic feedback through the manipulator.
        time_to_go = reference.get("time_to_go", None)
        ori_enabled = (
            "q_des" in reference and reference["q_des"] is not None
            and (time_to_go is None or time_to_go <= self.t_blend_ori)
        )
        if ori_enabled:
            q_des = np.asarray(reference["q_des"], dtype=float)
            q_des = q_des / max(np.linalg.norm(q_des), 1e-12)
            omega_des = np.asarray(reference["omega_des"], dtype=float)
            omega_dot_des = np.asarray(reference.get("omega_dot_des", np.zeros(3)), dtype=float)

            # Current EE angular velocity (world frame)
            omega_ee = self.compute_ee_angular_velocity(state)

            # Quaternion error: q_err = q_des * conj(q_ee)
            # vec(q_err) gives a small-angle rotation vector for PD
            w_e, x_e, y_e, z_e = q_ee
            cw, cx, cy, cz = w_e, -x_e, -y_e, -z_e
            wd, xd, yd, zd = q_des
            err_w = wd*cw - xd*cx - yd*cy - zd*cz
            err_x = wd*cx + xd*cw + yd*cz - zd*cy
            err_y = wd*cy - xd*cz + yd*cw + zd*cx
            err_z = wd*cz + xd*cy - yd*cx + zd*cw
            # Take shortest rotation (handle quat double cover)
            if err_w < 0:
                err_x, err_y, err_z = -err_x, -err_y, -err_z
            e_rot = np.array([err_x, err_y, err_z])  # vec(q_err) ~ rot/2

            # Angular velocity error (world frame)
            e_omega = omega_des - omega_ee

            # Commanded angular acceleration: feedforward + PD
            omega_dot_cmd = omega_dot_des + self.Kd_ori @ e_omega + self.Kp_ori @ (2.0 * e_rot)
        else:
            omega_dot_cmd = None
            omega_ee = None

        # 4. Inverse mapping via generalized Jacobian.
        #    Position is the primary task. If orientation command exists, use
        #    task-priority IK (orientation in null space of position task).
        R_base = _quat_rotmat(state.base_quat)
        J_g_tuple = self.chaser.dyn.compute_generalized_jacobian(state.q)
        J_g = J_g_tuple[0]   # (6, n_joints): rows [angular(0:3); linear(3:6)] in base frame
        J_ang_base = J_g[0:3, :]
        J_lin_base = J_g[3:6, :]

        rhddot_cmd_base = R_base.T @ rhddot_cmd

        # Primary task: position. Damped LS pseudo-inverse of J_lin.
        n_joints = J_lin_base.shape[1]
        JJT_lin = J_lin_base @ J_lin_base.T
        damp3 = self.damping_lambda ** 2 * np.eye(3)
        J_lin_pinv = J_lin_base.T @ np.linalg.solve(JJT_lin + damp3, np.eye(3))  # (n,3)
        qddot_primary = J_lin_pinv @ rhddot_cmd_base  # (n,)

        if omega_dot_cmd is not None and (np.linalg.norm(self.Kp_ori) + np.linalg.norm(self.Kd_ori)) > 1e-9:
            # Secondary task: orientation, projected into null space of position task.
            #   N = I - J_lin_pinv @ J_lin_base
            #   qddot_secondary = (J_ang @ N)^+ * (omega_dot_cmd - J_ang @ qddot_primary)
            N = np.eye(n_joints) - J_lin_pinv @ J_lin_base
            J_ang_proj = J_ang_base @ N

            # Solve in null space
            JJT_ang = J_ang_proj @ J_ang_proj.T
            damp_ang = (self.damping_lambda * 5.0) ** 2 * np.eye(3)  # heavier damping
            omega_dot_cmd_base = R_base.T @ omega_dot_cmd
            residual = omega_dot_cmd_base - J_ang_base @ qddot_primary
            qddot_secondary = J_ang_proj.T @ np.linalg.solve(JJT_ang + damp_ang, residual)
            qddot_cmd = qddot_primary + qddot_secondary
        else:
            qddot_cmd = qddot_primary

        # 5. Torque via H_tilde * qddot + c_tilde (full computed-torque control).
        #    Coriolis compensation is essential when qdot is non-negligible
        #    (e.g., during orientation tracking where qdot peaks at 5+ rad/s).
        H_tilde = self.chaser.dyn.compute_effective_arm_inertia(state.q)
        c_tilde = self.chaser.dyn.compute_coriolis_term(state.q, state.qdot)
        tau = H_tilde @ qddot_cmd + c_tilde

        # 6. Clip to limits
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)

        info = {
            "solve_time_s": _time.time() - t_start,
            "cost": float(np.dot(e_pos, e_pos) + np.dot(e_vel, e_vel)),
            "n_iter": 0,
            "success": True,
            "e_pos_norm": float(np.linalg.norm(e_pos)),
            "e_vel_norm": float(np.linalg.norm(e_vel)),
            "qddot_norm": float(np.linalg.norm(qddot_cmd)),
            "tau_clipped": bool(np.any(np.abs(H_tilde @ qddot_cmd) > self.tau_limit)),
            "has_orientation_cmd": omega_dot_cmd is not None,
        }
        return tau, info


__all__ = ["FeedbackLinearizationController"]
