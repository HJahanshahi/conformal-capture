"""
Quaternion utilities for both NumPy (simulation) and PyTorch (training).

Convention: [w, x, y, z]
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation


# ---- PyTorch (batched) ----

def quat_conjugate(q):
    """Conjugate of [w, x, y, z] — batched."""
    return q * torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)


def quat_mult_batch(q1, q2):
    """Hamilton product — batched over dim 0."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=1)


def quaternion_angular_error(q_pred, q_true):
    """
    Geodesic distance between quaternions (degrees).
    Handles double-cover.
    q_pred, q_true: [..., 4]
    """
    q_pred = q_pred / q_pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    q_true = q_true / q_true.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dot = (q_pred * q_true).sum(dim=-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(dot))


# ---- NumPy (single) ----

def quat_mult(q1, q2):
    """Hamilton product of two quaternions [w,x,y,z] (NumPy, single)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate_np(q):
    """Conjugate of [w,x,y,z] (NumPy, single)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rotmat(q):
    """Quaternion [w,x,y,z] → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def quat_error(q_desired, q_current):
    """
    Orientation error as rotation vector (NumPy, single).
    Returns 3-vector in base frame.
    """
    R_d = Rotation.from_quat([q_desired[1], q_desired[2], q_desired[3], q_desired[0]])
    R_c = Rotation.from_quat([q_current[1], q_current[2], q_current[3], q_current[0]])
    return (R_d * R_c.inv()).as_rotvec()
