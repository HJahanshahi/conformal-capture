"""
Estimate full 13-D state from observation history.

Observations: [px, py, pz, qw, qx, qy, qz]  (7-D)
State:        [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]  (13-D)
"""

import numpy as np
import torch
from .quaternion import quat_conjugate, quat_mult_batch, quat_mult, quat_conjugate_np


def initialize_state_from_observations(history_obs, history_time):
    """
    Estimate full 13-D state from a history of 7-D observations.

    Works with both PyTorch tensors (batched) and NumPy arrays (single).

    Parameters
    ----------
    history_obs  : [B, T, 7] tensor or [T, 7] array
    history_time : [B, T] tensor or [T,] array

    Returns
    -------
    state : [B, 13] tensor or [13,] array
    """
    if isinstance(history_obs, torch.Tensor):
        return _init_state_torch(history_obs, history_time)
    else:
        return _init_state_numpy(history_obs, history_time)


def _init_state_torch(history_obs, history_time):
    """Batched PyTorch version."""
    pos  = history_obs[:, -1, 0:3]
    quat = history_obs[:, -1, 3:7]
    dt = (history_time[:, -1] - history_time[:, -2]).clamp(min=1e-6).unsqueeze(-1)

    vel = (pos - history_obs[:, -2, 0:3]) / dt

    q_curr = history_obs[:, -1, 3:7]
    q_prev = history_obs[:, -2, 3:7]
    dq = quat_mult_batch(q_curr, quat_conjugate(q_prev))
    sign = torch.sign(dq[:, 0:1])
    sign[sign == 0] = 1.0
    dq = dq * sign
    omega = 2.0 * dq[:, 1:4] / dt

    quat = quat / quat.norm(dim=1, keepdim=True)
    return torch.cat([pos, vel, quat, omega], dim=1)


def _init_state_numpy(history_obs, history_time):
    """Single-trajectory NumPy version."""
    dt = max(history_time[-1] - history_time[-2], 1e-6)

    pos = history_obs[-1, 0:3]
    vel = (history_obs[-1, 0:3] - history_obs[-2, 0:3]) / dt

    q2 = history_obs[-1, 3:7]
    q1 = history_obs[-2, 3:7]
    quat = q2 / np.linalg.norm(q2)

    # delta rotation: dq = q2 x conj(q1); handle double-cover by flipping if qw<0
    dq = quat_mult(q2, quat_conjugate_np(q1))
    if dq[0] < 0:
        dq = -dq
    omega = 2.0 * dq[1:4] / dt

    return np.concatenate([pos, vel, quat, omega])
