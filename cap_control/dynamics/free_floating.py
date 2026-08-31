"""
Free-floating chaser dynamics wrapper.

Wraps `space_robot_dq`'s SpaceRobotKinematics and SpaceRobotDynamics into a narrow
interface the MPC and simulator will use.

Conventions:
    State variables:
        q           - (7,) joint angles (rad)
        qdot        - (7,) joint rates (rad/s)
        base_pos    - (3,) chaser base position in world frame (m)
        base_quat   - (4,) chaser base orientation, [w,x,y,z]
        base_vel    - (3,) chaser base linear velocity in world frame (m/s)
        base_omega  - (3,) chaser base angular velocity in BODY frame (rad/s)

    Free-floating assumption: no thrusters during manipulation. Base motion is
    entirely a consequence of momentum conservation given initial momentum h0.
    For Phase 4 we assume h0 = 0 (the system started at rest).

    The Phase 4 planning model is KINEMATIC: the MPC commands joint velocities,
    we compute the induced base velocity from momentum conservation, and integrate.
    Full torque-level forward dynamics is deferred to Phase 7.
"""
from dataclasses import dataclass, field
import numpy as np

from space_robot_dq import SpaceRobotKinematics, SpaceRobotDynamics, LinkProperties, create_7dof_srs

from cap_control.utils.transforms import (
    quat_integrate, quat_normalize, quat_to_rotmat, pose_compose,
)


@dataclass
class FreeFloatingState:
    """Full state of the free-floating chaser."""
    q:          np.ndarray                      # (7,)
    qdot:       np.ndarray                      # (7,)
    base_pos:   np.ndarray                      # (3,), world frame
    base_quat:  np.ndarray                      # (4,), [w,x,y,z]
    base_vel:   np.ndarray                      # (3,), world frame
    base_omega: np.ndarray                      # (3,), body frame

    def copy(self):
        return FreeFloatingState(
            q=self.q.copy(), qdot=self.qdot.copy(),
            base_pos=self.base_pos.copy(), base_quat=self.base_quat.copy(),
            base_vel=self.base_vel.copy(), base_omega=self.base_omega.copy(),
        )


class FreeFloatingChaser:
    """
    Free-floating 7-DOF chaser wrapper.

    Exposes:
        fk(q)                        - EE pose in base frame (pos, quat)
        fk_world(state)              - EE pose in world frame
        generalized_jacobian(q)      - (6,7) generalized Jacobian for free-floating
        manipulability(q)            - scalar dynamic manipulability w_free
        compute_base_velocity(q, qd) - (linear, angular) base velocity induced by qd
        kinematic_step(s, qd, dt)    - integrate one step under kinematic model
        home()                       - default initial state
    """

    def __init__(self):
        self.kin = SpaceRobotKinematics()
        self.config = create_7dof_srs()
        self.n_joints = self.config.num_joints

        # Realistic mass distribution for a 1.18 m, 7-DOF SRS-style space arm
        # Pattern: heavier proximal cluster (shoulder), medium elbow,
        # progressively lighter distal links. Total arm mass ~21 kg.
        # Inertias from cylindrical-link approximation:
        #   for a thin rod of mass m and length L:
        #     I_xx = I_yy = m*L^2/12  (transverse axes)
        #     I_zz = m*r^2/2          (axial, with r << L)
        # Link lengths (between successive joints) approx 0.2 m.
        # We use diagonal inertias; off-diagonal terms are zero (standard
        # for axisymmetric link approximation).
        link_masses = np.array([5.5, 4.0, 4.0, 3.0, 2.5, 1.5, 1.0])  # 21.5 kg total
        link_lengths = np.array([0.20, 0.20, 0.20, 0.20, 0.20, 0.10, 0.10])
        link_radii = np.array([0.05, 0.04, 0.04, 0.04, 0.035, 0.03, 0.03])

        # Get default link COM positions (in home configuration, base frame)
        # from create_7dof_srs ee_position is at [0, 0, 1.178].
        # We approximate link COMs at midpoints between consecutive joint
        # positions (matching the library's default convention).
        joint_z = np.array([0.0, 0.0, 0.310, 0.310, 0.710, 0.710, 1.100])
        next_z  = np.array([0.310, 0.310, 0.710, 0.710, 1.100, 1.100, 1.178])
        com_z = 0.5 * (joint_z + next_z)

        link_props = []
        for i in range(7):
            m = float(link_masses[i])
            L = float(link_lengths[i])
            r = float(link_radii[i])
            # Diagonal inertia: floor each axis at 0.05 kg.m^2 to represent
            # the motor + harmonic-gear + structural inertia that always
            # accompanies a real space-arm joint, regardless of the link's
            # slender geometric extent. This prevents pathological cases at
            # joints whose rotation axis aligns with the link's long axis.
            I_trans = max(m * L * L / 12.0, 0.05)
            I_axial = max(0.5 * m * r * r, 0.05)
            inertia = np.diag([I_trans, I_trans, I_axial])
            link_props.append(LinkProperties(
                mass=m,
                com_home=[0.0, 0.0, float(com_z[i])],
                inertia=inertia,
            ))

        # Base mass 400 kg gives arm/base ratio ~0.054 (Orbital Express class)
        # Base inertia approximated as a 0.5 m radius solid sphere:
        #   I = (2/5) * M * r^2 = (2/5) * 400 * 0.25 = 40 kg.m^2 per axis
        base_inertia = np.diag([40.0, 40.0, 40.0])

        self.dyn = SpaceRobotDynamics(
            kinematics=self.kin,
            base_mass=400.0,
            base_inertia=base_inertia,
            link_properties=link_props,
        )

        self.q_min = np.asarray(self.config.q_min, dtype=float)
        self.q_max = np.asarray(self.config.q_max, dtype=float)

    # ---- Kinematics ----

    def fk(self, q):
        """Forward kinematics in the chaser base frame. Returns (pos, quat)."""
        pos, quat = self.kin.forward_kinematics_6dof(np.asarray(q, dtype=float))
        return np.asarray(pos), np.asarray(quat)

    def fk_world(self, state: FreeFloatingState):
        """
        End-effector pose in the world frame.
        Composes base pose (world_from_base) with EE pose (base_from_ee).
        """
        T_base = (state.base_pos, state.base_quat)
        T_ee_in_base = self.fk(state.q)
        return pose_compose(T_base, T_ee_in_base)

    def generalized_jacobian(self, q):
        """(6, 7) generalized Jacobian: joint rates -> EE twist, accounting for base reaction.

        space_robot_dq's compute_generalized_jacobian returns a tuple
        (J_g, J_m, H_bm). We only need J_g here; the manipulator Jacobian J_m
        and base-arm coupling H_bm are computed separately when needed.
        """
        J_g, _J_m, _H_bm = self.dyn.compute_generalized_jacobian(np.asarray(q, dtype=float))
        return np.asarray(J_g)

    def manipulability(self, q):
        """Scalar dynamic manipulability measure w_free."""
        out = self.dyn.compute_dynamic_manipulability(np.asarray(q, dtype=float))
        # Library returns (w_free, ...); first element is the scalar.
        if isinstance(out, tuple):
            return float(out[0])
        return float(out)

    # ---- Dynamics (kinematic-model step) ----

    def compute_base_velocity(self, q, qdot):
        """
        Base twist induced by joint velocities under momentum conservation (h0 = 0).
        Returns (base_vel_world: (3,), base_omega_body: (3,)).

        space_robot_dq's compute_base_velocity returns a 6-vector xb_dot. We assume
        the convention [linear; angular]; the zero-h0 case is what the function
        computes when called without an h0 argument.
        """
        xb_dot = np.asarray(self.dyn.compute_base_velocity(
            np.asarray(q, dtype=float), np.asarray(qdot, dtype=float)))
        if xb_dot.shape[-1] != 6:
            raise RuntimeError(f"compute_base_velocity returned shape {xb_dot.shape}; expected (6,).")
        base_vel = xb_dot[:3]
        base_omega = xb_dot[3:]
        return base_vel, base_omega

    def kinematic_step(self, state: FreeFloatingState, qdot_cmd: np.ndarray, dt: float):
        """
        Integrate one step of the kinematic free-floating model.

        The MPC commands joint velocities directly. Base motion is computed from
        momentum conservation (h0 = 0 assumed). Joints integrate linearly; base
        position integrates linearly; base orientation integrates via quaternion
        exponential.
        """
        qdot_cmd = np.asarray(qdot_cmd, dtype=float)

        # 1. Base motion induced by these joint rates
        base_vel, base_omega = self.compute_base_velocity(state.q, qdot_cmd)

        # 2. Integrate joints (forward Euler)
        q_new = state.q + dt * qdot_cmd

        # 3. Integrate base
        base_pos_new = state.base_pos + dt * base_vel
        base_quat_new = quat_integrate(state.base_quat, base_omega, dt)

        # 4. Assemble
        return FreeFloatingState(
            q=q_new, qdot=qdot_cmd.copy(),
            base_pos=base_pos_new, base_quat=quat_normalize(base_quat_new),
            base_vel=base_vel, base_omega=base_omega,
        )

    def dynamic_step(self, state: FreeFloatingState, tau: np.ndarray, dt: float,
                     include_coriolis: bool = True):
        """
        Integrate one step of the *torque-level* free-floating reduced model.

        The MPC commands joint torques. Joint accelerations are computed via
        forward dynamics (Umetani-Yoshida reduced model), velocities and angles
        are integrated via semi-implicit Euler, base motion follows from
        momentum conservation.

        Args:
            state: current FreeFloatingState
            tau:   (n_joints,) joint torques
            dt:    integration step (s)
            include_coriolis: if True, accurate truth model. Default True for
                simulator. Set False inside MPC rollouts for speed (consistent
                with Coriolis-frozen gradient approximation).
        """
        tau = np.asarray(tau, dtype=float)

        # 1. Forward dynamics: torques -> joint accelerations
        qddot = self.dyn.forward_dynamics(state.q, state.qdot, tau,
                                           include_coriolis=include_coriolis)

        # 2. Semi-implicit Euler: update velocity first, then position
        qdot_new = state.qdot + dt * qddot
        q_new = state.q + dt * qdot_new

        # 3. Base motion induced by the new joint rates
        base_vel, base_omega = self.compute_base_velocity(state.q, qdot_new)

        # 4. Integrate base
        base_pos_new = state.base_pos + dt * base_vel
        base_quat_new = quat_integrate(state.base_quat, base_omega, dt)

        return FreeFloatingState(
            q=q_new, qdot=qdot_new,
            base_pos=base_pos_new, base_quat=quat_normalize(base_quat_new),
            base_vel=base_vel, base_omega=base_omega,
        )

    # ---- Defaults ----

    def home(self, base_quat=None, base_pos=None):
        """
        Chaser at zero joint angles, zero velocities. Default base orientation
        is identity at world origin; pass base_quat / base_pos to override
        (useful for pre-aligning chaser to face a tumbling target).

        Args:
            base_quat: optional (4,) quaternion [w,x,y,z]. Default identity.
            base_pos:  optional (3,) world position. Default origin.
        """
        if base_quat is None:
            base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            base_quat = np.asarray(base_quat, dtype=float)
            base_quat = base_quat / max(np.linalg.norm(base_quat), 1e-12)
        if base_pos is None:
            base_pos = np.zeros(3)
        else:
            base_pos = np.asarray(base_pos, dtype=float)

        # Default home posture: most joints at 0, joint 4 (elbow) at 0.5 rad
        # to satisfy q_min[3] = 0.1 (single-sided elbow joint).
        q_home = np.zeros(self.n_joints)
        q_home[3] = 0.5
        return FreeFloatingState(
            q=q_home,
            qdot=np.zeros(self.n_joints),
            base_pos=base_pos,
            base_quat=base_quat,
            base_vel=np.zeros(3),
            base_omega=np.zeros(3),
        )


__all__ = ["FreeFloatingChaser", "FreeFloatingState"]
