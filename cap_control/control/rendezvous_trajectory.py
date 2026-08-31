"""
Rendezvous trajectory generation following Aghili (2023, arXiv:2303.10812).

Generates a closed-form end-effector trajectory r*_h(t) for a free-floating
chaser to intercept a tumbling target satellite. The trajectory:
  - starts at the current EE pose (rh0, rhdot0) at t=0
  - ends at the target grapple-fixture pose (rc(tf), rcdot(tf)) at t=tf
  - is optimal w.r.t. cost J = integral (1 + w1*||rhdot||^2 + w2*||rhddot||^2) dt
  - has free terminal time tf, found by enforcing optimal Hamiltonian H*(tf)=0

The trajectory takes the form (eq 37 in Aghili):
    r*_h(t) = k0 + k1*t + k2*exp(sigma*t) + k3*exp(-sigma*t)
where sigma = sqrt(w1/w2) and k0..k3 are 3-vectors solved from boundary conditions.

This module is purely the trajectory generator. The tracking controller
(feedback linearization) is in a separate module.
"""
import numpy as np
from dataclasses import dataclass
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Quaternion helpers (Hamilton convention, [w, x, y, z])
# ---------------------------------------------------------------------------

def _quat_mult(q1, q2):
    """Hamilton product q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_to_R(q):
    """Quaternion [w,x,y,z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


# ---------------------------------------------------------------------------
# Target state propagation (free rotation + linear motion)
# ---------------------------------------------------------------------------

def _propagate_target(q0, omega0, r0, rdot0, Ic, t, dt=0.005):
    """
    Forward integrate a target satellite under free dynamics (no external
    forces/torques) from t=0 to t.

    Uses Euler-step integration with quaternion renormalization.

    Args:
        q0:      (4,) initial quaternion [w,x,y,z]
        omega0:  (3,) initial body-frame angular velocity
        r0:      (3,) initial CoM position
        rdot0:   (3,) initial CoM linear velocity
        Ic:      (3,3) target inertia tensor in body frame
        t:       scalar, propagation time
        dt:      integration step (default 5ms)

    Returns:
        (q, omega, r, rdot) at time t
    """
    n_steps = max(1, int(np.ceil(t / dt)))
    actual_dt = t / n_steps  # Adjust dt to land exactly on t

    q = q0.copy().astype(float)
    omega = omega0.copy().astype(float)
    r = r0.copy().astype(float)
    rdot = rdot0.copy().astype(float)
    Ic_inv = np.linalg.inv(Ic)

    for _ in range(n_steps):
        # Euler equation for free rotation: omega_dot = -Ic^-1 (omega x Ic*omega)
        omega_dot = -Ic_inv @ np.cross(omega, Ic @ omega)
        omega = omega + actual_dt * omega_dot
        # Quaternion update: q_dot = 0.5 * omega_quat * q
        omega_quat = np.array([0.0, *omega])
        q = q + 0.5 * actual_dt * _quat_mult(omega_quat, q)
        q = q / max(np.linalg.norm(q), 1e-12)
        # Linear motion (force-free)
        r = r + actual_dt * rdot
        # rdot unchanged

    return q, omega, r, rdot


def _grapple_kinematics(q, omega, r, rho_body, Ic):
    """
    Compute grapple-fixture position, velocity, acceleration in world frame.

    Aghili eqs (23), (25), (44):
        rc      = ro + R(q) @ rho_body
        rcdot   = R(q) @ (omega x rho_body)
        rcddot  = R(q) @ (omega x (omega x rho_body) + phi x rho_body)
    where phi = -Ic^-1 (omega x Ic omega).

    Args:
        q, omega, r:    target state
        rho_body:       (3,) grapple offset in target body frame
        Ic:             (3,3) target inertia (for acceleration via Euler eq)

    Returns:
        (rc, rcdot, rcddot) each (3,) in world frame
    """
    R = _quat_to_R(q)
    rc = r + R @ rho_body

    omega_cross_rho = np.cross(omega, rho_body)
    rcdot = R @ omega_cross_rho

    Ic_inv = np.linalg.inv(Ic)
    phi = -Ic_inv @ np.cross(omega, Ic @ omega)
    rcddot = R @ (np.cross(omega, omega_cross_rho) + np.cross(phi, rho_body))

    return rc, rcdot, rcddot


# ---------------------------------------------------------------------------
# Aghili trajectory: solve for coefficients given boundary conditions
# ---------------------------------------------------------------------------

def _trajectory_coefficients(rh0, rhdot0, rc_tf, rcdot_tf, tf, sigma):
    """
    Solve for k0..k3 in r*_h(t) = k0 + k1*t + k2*exp(sigma*t) + k3*exp(-sigma*t).

    Boundary conditions:
        rh*(0)    = rh0
        rhdot*(0) = rhdot0
        rh*(tf)   = rc(tf)        # match grapple position
        rhdot*(tf)= rcdot(tf)     # match grapple velocity

    System is 4 equations per coordinate. The system matrix is the same for
    all 3 coordinates; we just solve it once and reuse.

    Returns:
        (k0, k1, k2, k3): each shape (3,)
    """
    e_p = np.exp(sigma * tf)
    e_n = np.exp(-sigma * tf)

    A = np.array([
        [1.0,  0.0,    1.0,        1.0],
        [0.0,  1.0,    sigma,     -sigma],
        [1.0,  tf,     e_p,        e_n],
        [0.0,  1.0,    sigma*e_p, -sigma*e_n],
    ])
    A_inv = np.linalg.inv(A)

    coeffs = np.zeros((4, 3))
    for i in range(3):
        b = np.array([rh0[i], rhdot0[i], rc_tf[i], rcdot_tf[i]])
        coeffs[:, i] = A_inv @ b
    return coeffs[0], coeffs[1], coeffs[2], coeffs[3]


def _hamiltonian_residual(tf, target_state, rho_body, Ic, rh0, rhdot0, w1, w2,
                            target_propagator=None):
    """
    Eq (43) in Aghili: optimality condition H*(tf) = 0.

    Used inside brentq to find the optimal terminal time.

    If target_propagator is None, uses analytical free-rotation propagation
    of `target_state` forward to tf. Otherwise calls target_propagator(tf)
    to get target state directly at time tf (e.g., from UPN).

    Returns:
        scalar residual that should be zero at optimal tf
    """
    sigma = np.sqrt(w1 / w2)
    if target_propagator is not None:
        q_tf, omega_tf, r_tf, rdot_tf = target_propagator(tf)
        q_tf = q_tf / max(np.linalg.norm(q_tf), 1e-12)
    else:
        q0, omega0, r0, rdot0 = target_state
        q_tf, omega_tf, r_tf, _ = _propagate_target(q0, omega0, r0, rdot0, Ic, tf)
    rc_tf, rcdot_tf, rcddot_tf = _grapple_kinematics(q_tf, omega_tf, r_tf, rho_body, Ic)

    k0, k1, k2, k3 = _trajectory_coefficients(rh0, rhdot0, rc_tf, rcdot_tf, tf, sigma)
    e_p = np.exp(sigma * tf)
    e_n = np.exp(-sigma * tf)

    # Eq (43)
    residual = (1.0
                + w1 * np.dot(k1, k1)
                - 4.0 * w1 / w2 * np.dot(k2, k3)
                + 2.0 * w1 * sigma * np.dot(k2 * e_p - k3 * e_n, k1)
                - 2.0 * w2 * sigma**2 * np.dot(k2 * e_p + k3 * e_n, rcddot_tf))
    return residual


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class RendezvousTrajectory:
    """
    Closed-form rendezvous trajectory with optional orientation reference.

    Position part: closed-form polynomial-exponential (Aghili 2023 eq 37):
        r*_h(t) = k0 + k1*t + k2*exp(sigma*t) + k3*exp(-sigma*t)

    Orientation part (optional, added when q_h0 / q_tf are provided):
        Linear interpolation in rotation-vector space from the chaser's
        initial EE orientation q_h0 toward the target orientation at tf,
        with angular velocity smoothly ramped to omega_o(tf) at the
        rendezvous instant. This follows Aghili 2023 sec. 3 where
        omega*_h = omega_o is set at tf only; the trajectory in between
        is a smooth interpolation that hits this terminal velocity match.
    """
    k0: np.ndarray   # (3,)
    k1: np.ndarray   # (3,)
    k2: np.ndarray   # (3,)
    k3: np.ndarray   # (3,)
    sigma: float
    tf: float
    rc_tf: np.ndarray       # grapple position at tf
    rcdot_tf: np.ndarray    # grapple velocity at tf

    # Orientation reference (optional - None means position-only mode)
    q_h0: np.ndarray = None      # (4,) chaser EE quat at t=0  [w,x,y,z]
    q_tf: np.ndarray = None      # (4,) target attitude quat at tf [w,x,y,z]
    omega_tf: np.ndarray = None  # (3,) target body-frame angular velocity at tf

    def has_orientation(self):
        return self.q_h0 is not None and self.q_tf is not None

    def evaluate(self, t):
        """
        Evaluate position trajectory at time t (clipped to [0, tf]).

        Returns:
            rh    : (3,) desired EE position
            rhdot : (3,) desired EE velocity
            rhddot: (3,) desired EE acceleration
        """
        t = float(np.clip(t, 0.0, self.tf))
        e_p = np.exp(self.sigma * t)
        e_n = np.exp(-self.sigma * t)
        rh    = self.k0 + self.k1 * t + self.k2 * e_p + self.k3 * e_n
        rhdot = self.k1 + self.sigma * (self.k2 * e_p - self.k3 * e_n)
        rhddot = self.sigma**2 * (self.k2 * e_p + self.k3 * e_n)
        return rh, rhdot, rhddot

    def evaluate_orientation(self, t, t_blend=None):
        """
        Evaluate orientation reference at time t (clipped to [0, tf]).

        For t < tf - t_blend: orientation reference is q_h0 (no rotation).
        For t in [tf - t_blend, tf]: smooth ramp from q_h0 to q_tf using quintic.
        At t = tf: q_des = q_tf, omega_des = omega_tf.

        This follows Aghili 2023 sec. 3 (orientation matched only at rendezvous
        instant) while providing a smooth blending profile.

        Args:
            t: query time
            t_blend: window before tf in which to apply orientation ramp.
                     None means use the full trajectory (legacy behavior).

        Returns:
            q_des     : (4,) desired EE quaternion [w,x,y,z]
            omega_des : (3,) desired EE angular velocity (world frame)
            omega_dot_des : (3,) desired EE angular acceleration (world frame)
        """
        if not self.has_orientation():
            raise ValueError("Trajectory has no orientation reference; provide "
                             "q_h0 and q_tf to solve_rendezvous_trajectory().")
        t = float(np.clip(t, 0.0, self.tf))

        # Determine effective ramp window
        if t_blend is None or t_blend >= self.tf:
            ramp_start = 0.0
            ramp_duration = self.tf
        else:
            ramp_start = self.tf - t_blend
            ramp_duration = t_blend

        # Before ramp: hold at q_h0, omega = 0
        if t < ramp_start:
            return self.q_h0.copy(), np.zeros(3), np.zeros(3)
        # In ramp: quintic interpolation from q_h0 to q_tf
        t_in_ramp = t - ramp_start
        # Slerp-style interpolation from q_h0 to q_tf via rotation vector.
        # Compute relative rotation r_rel = q_tf * conj(q_h0)
        q_h0 = self.q_h0
        q_tf = self.q_tf
        # Hamilton product: q_rel = q_tf * conj(q_h0)
        w0, x0, y0, z0 = q_h0
        wf, xf, yf, zf = q_tf
        # conj(q_h0) = [w0, -x0, -y0, -z0]
        cw, cx, cy, cz = w0, -x0, -y0, -z0
        rel_w = wf*cw - xf*cx - yf*cy - zf*cz
        rel_x = wf*cx + xf*cw + yf*cz - zf*cy
        rel_y = wf*cy - xf*cz + yf*cw + zf*cx
        rel_z = wf*cz + xf*cy - yf*cx + zf*cw
        rel = np.array([rel_w, rel_x, rel_y, rel_z])
        rel = rel / max(np.linalg.norm(rel), 1e-12)
        # Take shortest rotation (handle quaternion double cover)
        if rel[0] < 0:
            rel = -rel
        # Rotation vector: angle * axis
        cos_half = float(np.clip(rel[0], -1.0, 1.0))
        angle = 2.0 * np.arccos(cos_half)
        sin_half = np.sqrt(max(1.0 - cos_half**2, 0.0))
        if sin_half < 1e-9:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = rel[1:4] / sin_half
        rot_vec_full = angle * axis  # rotation vector for the full t=0->tf rotation

        # Quintic time-scaling on the ramp window [ramp_start, tf]
        # s_ramp(0)=0, s_ramp(t_blend)=1, smooth start AND end (C^2)
        u = t_in_ramp / ramp_duration  # in [0, 1]
        s_orient = 10.0*u**3 - 15.0*u**4 + 6.0*u**5
        s_orient_dot = (30.0*u**2 - 60.0*u**3 + 30.0*u**4) / ramp_duration
        s_orient_ddot = (60.0*u - 180.0*u**2 + 120.0*u**3) / ramp_duration**2

        # Interpolated rotation vector at t (from q_h0 frame)
        rot_vec_t = s_orient * rot_vec_full
        # Convert back to quaternion (q_des = exp(0.5 * rot_vec_t) * q_h0)
        rv_norm = np.linalg.norm(rot_vec_t)
        if rv_norm < 1e-9:
            dq_w = 1.0; dq_v = np.zeros(3)
        else:
            half = 0.5 * rv_norm
            dq_w = np.cos(half)
            dq_v = np.sin(half) * (rot_vec_t / rv_norm)
        # q_des = dq * q_h0  (Hamilton product)
        w0, x0, y0, z0 = q_h0
        q_des_w = dq_w*w0 - dq_v[0]*x0 - dq_v[1]*y0 - dq_v[2]*z0
        q_des_x = dq_w*x0 + dq_v[0]*w0 + dq_v[1]*z0 - dq_v[2]*y0
        q_des_y = dq_w*y0 - dq_v[0]*z0 + dq_v[1]*w0 + dq_v[2]*x0
        q_des_z = dq_w*z0 + dq_v[0]*y0 - dq_v[1]*x0 + dq_v[2]*w0
        q_des = np.array([q_des_w, q_des_x, q_des_y, q_des_z])
        q_des = q_des / max(np.linalg.norm(q_des), 1e-12)

        # Angular velocity reference: smoothly ramp from 0 to omega_tf using
        # the same quintic shape (so it is C^2 smooth at both endpoints)
        omega_des = s_orient * self.omega_tf
        omega_dot_des = s_orient_dot * self.omega_tf

        return q_des, omega_des, omega_dot_des

    def evaluate_grid(self, t_grid):
        """Evaluate position trajectory at multiple times. Returns (3, N) arrays."""
        t_grid = np.atleast_1d(t_grid)
        rh_arr = np.zeros((3, len(t_grid)))
        rhdot_arr = np.zeros((3, len(t_grid)))
        rhddot_arr = np.zeros((3, len(t_grid)))
        for i, t in enumerate(t_grid):
            rh_arr[:, i], rhdot_arr[:, i], rhddot_arr[:, i] = self.evaluate(t)
        return rh_arr, rhdot_arr, rhddot_arr


def solve_rendezvous_trajectory(rh0, rhdot0, target_state, target_inertia,
                                  rho_body=np.zeros(3), w1=1.0, w2=1.0,
                                  tf_search_range=(0.5, 5.0),
                                  tf_fallback=2.0,
                                  target_propagator=None,
                                  q_h0_chaser=None):
    """
    Generate an optimal closed-form rendezvous trajectory (Aghili 2023).

    Args:
        rh0:            (3,) chaser EE position at t=0
        rhdot0:         (3,) chaser EE velocity at t=0
        target_state:   tuple (q0, omega0, r0, rdot0) in world frame.
                        Used only when target_propagator is None.
        target_inertia: (3,3) target inertia tensor in body frame.
                        Used for grapple acceleration kinematics regardless
                        of which propagation method is used.
        rho_body:       (3,) grapple-fixture offset in target body frame.
        w1, w2:         cost weights (eq 28).
        tf_search_range: (tf_min, tf_max) for finding optimal tf via brentq.
        tf_fallback:    fallback tf if Hamiltonian residual doesn't change sign.
        target_propagator: optional callable(t) -> (q, omega, r, rdot) that
                        returns predicted target state at time t. When given,
                        the trajectory generator queries this for rc(tf) and
                        rcdot(tf) directly. Useful when target dynamics are
                        non-free or when you have a learned predictor (e.g. UPN)
                        whose forecast is more accurate than analytical
                        propagation of an initial state estimate.

    Returns:
        RendezvousTrajectory object
    """
    rh0 = np.asarray(rh0, dtype=float)
    rhdot0 = np.asarray(rhdot0, dtype=float)
    rho_body = np.asarray(rho_body, dtype=float)
    Ic = np.asarray(target_inertia, dtype=float)
    sigma = np.sqrt(w1 / w2)

    # 1) Find optimal tf
    args = (target_state, rho_body, Ic, rh0, rhdot0, w1, w2, target_propagator)
    tf_min, tf_max = tf_search_range
    h_min = _hamiltonian_residual(tf_min, *args)
    h_max = _hamiltonian_residual(tf_max, *args)

    if h_min * h_max < 0:
        tf_optimal = brentq(_hamiltonian_residual, tf_min, tf_max, args=args)
    else:
        import warnings
        warnings.warn(
            f"Hamiltonian residual does not change sign in [{tf_min}, {tf_max}] "
            f"(h_min={h_min:.3f}, h_max={h_max:.3f}). Using tf_fallback={tf_fallback}.",
            UserWarning,
        )
        tf_optimal = tf_fallback

    # 2) Evaluate target at tf_optimal and compute coefficients
    if target_propagator is not None:
        q_tf, omega_tf, r_tf, _ = target_propagator(tf_optimal)
        q_tf = q_tf / max(np.linalg.norm(q_tf), 1e-12)
    else:
        q0, omega0, r0, rdot0 = target_state
        q_tf, omega_tf, r_tf, _ = _propagate_target(q0, omega0, r0, rdot0, Ic, tf_optimal)
    rc_tf, rcdot_tf, _ = _grapple_kinematics(q_tf, omega_tf, r_tf, rho_body, Ic)
    k0, k1, k2, k3 = _trajectory_coefficients(rh0, rhdot0, rc_tf, rcdot_tf, tf_optimal, sigma)

    # Capture target orientation at tf if propagator/target_state available
    q_tf_target = None
    omega_tf_target = None
    if target_propagator is not None:
        q_t, om_t, _, _ = target_propagator(tf_optimal)
        q_tf_target = q_t / max(np.linalg.norm(q_t), 1e-12)
        omega_tf_target = om_t
    elif target_state is not None:
        q_tf_target = q_tf / max(np.linalg.norm(q_tf), 1e-12)
        omega_tf_target = omega_tf

    return RendezvousTrajectory(
        k0=k0, k1=k1, k2=k2, k3=k3,
        sigma=sigma, tf=tf_optimal,
        rc_tf=rc_tf, rcdot_tf=rcdot_tf,
        q_h0=q_h0_chaser,
        q_tf=q_tf_target,
        omega_tf=omega_tf_target,
    )


__all__ = ["RendezvousTrajectory", "solve_rendezvous_trajectory"]
