"""
Dataset generation and loading for tumbling target trajectories.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .simulator import TumblingTargetSimulator, perturb_quaternion


class TumblingTargetDataset(Dataset):
    """
    PyTorch Dataset of (history, future) sliding windows.

    Parameters
    ----------
    true_states  : (N_traj, T, 13)
    observations : (N_traj, T, 7)
    times        : (T,)
    history_len  : int
    future_len   : int
    stride       : int — stride between consecutive windows
    """

    def __init__(self, true_states, observations, times,
                 history_len=10, future_len=20, stride=1):
        self.windows = []
        min_len = history_len + future_len

        for traj_idx in range(true_states.shape[0]):
            T = true_states.shape[1]
            for start in range(0, T - min_len + 1, stride):
                h_end = start + history_len
                f_end = h_end + future_len
                self.windows.append({
                    'history_obs':   observations[traj_idx, start:h_end],
                    'history_time':  times[start:h_end],
                    'future_states': true_states[traj_idx, h_end:f_end],
                    'future_obs':    observations[traj_idx, h_end:f_end],
                    'future_time':   times[h_end:f_end],
                })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        return (
            torch.tensor(w['history_obs'],   dtype=torch.float32),
            torch.tensor(w['history_time'],  dtype=torch.float32),
            torch.tensor(w['future_states'], dtype=torch.float32),
            torch.tensor(w['future_obs'],    dtype=torch.float32),
            torch.tensor(w['future_time'],   dtype=torch.float32),
        )


def generate_dataset(n_traj=300, t_final=10.0, n_steps=100,
                     pos_noise_std=0.01, rot_noise_std_deg=1.0,
                     seed=0, output_file=None, **ic_kwargs):
    """
    Generate a full dataset of tumbling target trajectories.

    Parameters
    ----------
    n_traj         : number of trajectories
    t_final        : simulation duration (s)
    n_steps        : time steps per trajectory
    pos_noise_std  : position observation noise (m)
    rot_noise_std_deg : orientation noise (degrees)
    seed           : random seed
    output_file    : if given, save .npz here
    **ic_kwargs    : forwarded to sample_initial_conditions

    Returns
    -------
    true_states  : (n_traj, n_steps, 13)
    observations : (n_traj, n_steps, 7)
    times        : (n_steps,)
    inertias     : (n_traj, 3)
    """
    rng = np.random.default_rng(seed)
    rot_noise_std = np.deg2rad(rot_noise_std_deg)
    t_eval = np.linspace(0, t_final, n_steps)

    true_states  = np.zeros((n_traj, n_steps, 13))
    observations = np.zeros((n_traj, n_steps, 7))
    inertias     = np.zeros((n_traj, 3))

    for i in tqdm(range(n_traj), desc="Generating trajectories"):
        p0, v0, q0, w0, I = TumblingTargetSimulator.sample_initial_conditions(
            rng, **ic_kwargs)
        state0 = np.concatenate([p0, v0, q0, w0])

        sim = TumblingTargetSimulator(I)
        traj = sim.simulate(state0, t_eval)

        true_states[i] = traj
        inertias[i] = I

        # Noisy observations: position + quaternion
        observations[i, :, 0:3] = traj[:, 0:3] + rng.normal(
            scale=pos_noise_std, size=(n_steps, 3))
        observations[i, :, 3:7] = np.stack([
            perturb_quaternion(traj[k, 6:10], rot_noise_std, rng)
            for k in range(n_steps)
        ])

    if output_file is not None:
        np.savez_compressed(
            output_file,
            true_states=true_states,
            observations=observations,
            times=t_eval,
            inertias=inertias,
        )

    return true_states, observations, t_eval, inertias
