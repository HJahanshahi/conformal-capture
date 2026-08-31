"""
Rigid-body tumbling target simulator.

Simulates torque-free tumbling of a rigid body in 3D:
  - Translation: free flight (no forces in local orbital frame)
  - Rotation: Euler's equations with quaternion kinematics

State (13-D): [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
"""

import numpy as np
from scipy.integrate import solve_ivp


def quat_mult(q1, q2):
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def random_unit_quaternion(rng):
    """Uniform random rotation (Shoemake's method)."""
    u1, u2, u3 = rng.uniform(0, 1, size=3)
    return np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1)     * np.sin(2 * np.pi * u3),
        np.sqrt(u1)     * np.cos(2 * np.pi * u3),
    ])[[3, 0, 1, 2]]  # reorder to [w, x, y, z]


def perturb_quaternion(q_true, angle_std, rng):
    """Apply a small random rotation noise to a quaternion."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.normal(scale=angle_std)
    half = 0.5 * angle
    dq = np.array([np.cos(half), *(axis * np.sin(half))])
    q_noisy = quat_mult(dq, q_true)
    return q_noisy / np.linalg.norm(q_noisy)


class TumblingTargetSimulator:
    """
    Simulate torque-free tumbling of a rigid body.

    Parameters
    ----------
    inertia_diag : array (3,)
        Principal moments of inertia [Ix, Iy, Iz] in kg·m².
    """

    def __init__(self, inertia_diag):
        self.inertia = np.asarray(inertia_diag, dtype=float)

    def dynamics(self, t, state):
        """13-D state derivative."""
        v = state[3:6]
        q = state[6:10]
        w = state[10:13]

        p_dot = v
        v_dot = np.zeros(3)
        q_dot = 0.5 * quat_mult(q, np.array([0.0, w[0], w[1], w[2]]))
        w_dot = -np.cross(w, self.inertia * w) / self.inertia

        return np.concatenate([p_dot, v_dot, q_dot, w_dot])

    def simulate(self, state0, t_eval, rtol=1e-8, atol=1e-10):
        """
        Integrate one trajectory.

        Parameters
        ----------
        state0 : array (13,)
        t_eval : array (T,)

        Returns
        -------
        trajectory : array (T, 13)
        """
        sol = solve_ivp(
            fun=lambda t, s: self.dynamics(t, s),
            t_span=(t_eval[0], t_eval[-1]),
            y0=state0,
            t_eval=t_eval,
            method='RK45',
            rtol=rtol, atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        traj = sol.y.T
        # Re-normalise quaternions
        traj[:, 6:10] /= np.linalg.norm(traj[:, 6:10], axis=1, keepdims=True)
        return traj

    @staticmethod
    def sample_initial_conditions(rng, pos_range=2.0,
                                   vel_range=(0.01, 0.10),
                                   w_bins_deg=None,
                                   w_bin_weights=None,
                                   inertia_range=(1.0, 10.0)):
        """
        Sample random initial conditions and inertia (v2 generation).

        Parameters
        ----------
        pos_range     : initial position bound (m), uniform [-pr, pr] per axis
        vel_range     : (min, max) linear velocity magnitude in m/s
        w_bins_deg    : list of (lo_deg_per_s, hi_deg_per_s) bins for omega.
                        If None, defaults to 3 bins: slow (0.1-3),
                        moderate (3-10), fast (10-30) deg/s.
        w_bin_weights : sampling probability per bin. If None, uniform.
        inertia_range : (min, max) inertia diagonal entries (kg*m^2)

        Returns: p0, v0, q0, w0, inertia_diag
        """
        if w_bins_deg is None:
            w_bins_deg = [(0.1, 3.0), (3.0, 10.0), (10.0, 30.0)]
        if w_bin_weights is None:
            w_bin_weights = [1.0 / len(w_bins_deg)] * len(w_bins_deg)

        p0 = rng.uniform(-pos_range, pos_range, size=3)

        v_dir = rng.normal(size=3)
        v_dir /= np.linalg.norm(v_dir)
        v_min, v_max = float(vel_range[0]), float(vel_range[1])
        v0 = v_dir * rng.uniform(v_min, v_max)

        q0 = random_unit_quaternion(rng)

        # Pick a bin, then sample omega magnitude within it
        bin_idx = rng.choice(len(w_bins_deg), p=w_bin_weights)
        lo_deg, hi_deg = w_bins_deg[bin_idx]
        w_mag_rad = np.deg2rad(rng.uniform(lo_deg, hi_deg))
        w_dir = rng.normal(size=3)
        w_dir /= np.linalg.norm(w_dir)
        w0 = w_dir * w_mag_rad

        inertia = rng.uniform(*inertia_range, size=3)

        return p0, v0, q0, w0, inertia
