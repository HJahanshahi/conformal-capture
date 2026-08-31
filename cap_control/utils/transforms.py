"""
SE(3) and quaternion helpers used throughout cap_control.

Conventions:
    Quaternions: [w, x, y, z] (Hamilton, scalar-first). Same as capture_lib.
    Poses:       tuple (position: (3,), quaternion: (4,))
    Twists:      (linear: (3,), angular: (3,)) in the frame specified at call site
"""
import numpy as np


def quat_mult(q1, q2):
    """Hamilton product of two [w,x,y,z] quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q):
    """Conjugate of [w,x,y,z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q, eps=1e-12):
    n = np.linalg.norm(q)
    return q / max(n, eps)


def quat_to_rotmat(q):
    """[w,x,y,z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ])


def quat_rotate(q, v):
    """Rotate a 3-vector v by quaternion q."""
    return quat_to_rotmat(q) @ v


def quat_integrate(q, omega, dt):
    """
    First-order quaternion integration: q_{k+1} = normalize(q_k + 0.5 * q_k (x) [0, omega] * dt).
    omega in the body frame.
    """
    w_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    q_dot = 0.5 * quat_mult(q, w_quat)
    return quat_normalize(q + dt * q_dot)


def pose_compose(T_a, T_b):
    """
    Compose two poses: T_a then T_b (i.e. world_from_b = world_from_a * a_from_b).
    Each pose is (position, quaternion).
    """
    p_a, q_a = T_a
    p_b, q_b = T_b
    p_out = p_a + quat_rotate(q_a, p_b)
    q_out = quat_normalize(quat_mult(q_a, q_b))
    return p_out, q_out


def pose_error(T_desired, T_current):
    """
    6-D pose error (desired - current), expressed in the current frame.
    Returns (pos_err: (3,), rot_err: (3,)) where rot_err is the rotation vector.
    """
    p_d, q_d = T_desired
    p_c, q_c = T_current
    pos_err = p_d - p_c

    # Relative rotation: q_rel = q_c^-1 * q_d
    q_rel = quat_normalize(quat_mult(quat_conjugate(q_c), q_d))
    # Handle double-cover: take the shorter rotation
    if q_rel[0] < 0:
        q_rel = -q_rel
    # Rotation vector = 2 * vec_part for small angles; use acos form for robustness
    qw = np.clip(q_rel[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(qw)
    vec = q_rel[1:4]
    vec_norm = np.linalg.norm(vec)
    if vec_norm < 1e-9:
        rot_err = np.zeros(3)
    else:
        rot_err = angle * (vec / vec_norm)
    return pos_err, rot_err
