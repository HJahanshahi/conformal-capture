"""
Euler integration for UPN — fast alternative to odeint for training/inference.
"""

import torch
from upn.core.vech import vech, unvech


def euler_integrate(model, mu0, S0, time_points, state_dim=13):
    """
    Euler integration of UPN (pure prediction, no observations).

    Parameters
    ----------
    model       : UPN model
    mu0         : [B, state_dim] initial mean
    S0          : [B, state_dim, state_dim] initial covariance
    time_points : [T] time grid
    state_dim   : int

    Returns
    -------
    mean_traj : [T, B, state_dim]
    cov_traj  : [T, B, state_dim, state_dim]
    """
    sigma_v = vech(S0)
    z = torch.cat([mu0, sigma_v], dim=1)
    mean_traj, cov_traj = [mu0], [S0]

    for i in range(1, len(time_points)):
        dt = time_points[i] - time_points[i - 1]
        dz = model(time_points[i - 1], z)
        z = z + dt * dz
        mu  = z[:, :state_dim]
        sig = unvech(z[:, state_dim:], state_dim)
        mean_traj.append(mu)
        cov_traj.append(sig)

    return torch.stack(mean_traj), torch.stack(cov_traj)


def euler_integrate_with_updates(model, mu0, S0, time_points, observations,
                                  update_frequency=5, state_dim=13):
    """
    Euler integration with sparse Kalman updates.

    Parameters
    ----------
    model            : UPN model
    mu0              : [B, state_dim]
    S0               : [B, state_dim, state_dim]
    time_points      : [T]
    observations     : [B, T-1, obs_dim] noisy measurements
    update_frequency : apply Kalman update every N steps
    state_dim        : int

    Returns
    -------
    mean_traj : [T, B, state_dim]
    cov_traj  : [T, B, state_dim, state_dim]
    """
    sigma_v = vech(S0)
    z = torch.cat([mu0, sigma_v], dim=1)
    mean_traj, cov_traj = [mu0], [S0]

    for i in range(1, len(time_points)):
        dt = time_points[i] - time_points[i - 1]
        dz = model(time_points[i - 1], z)
        z = z + dt * dz
        mu  = z[:, :state_dim]
        sig = unvech(z[:, state_dim:], state_dim)

        if i % update_frequency == 0 and i - 1 < observations.shape[1]:
            mu, sig = model.kalman_update(mu, sig, observations[:, i - 1, :])
            z = torch.cat([mu, vech(sig)], dim=1)

        mean_traj.append(mu)
        cov_traj.append(sig)

    return torch.stack(mean_traj), torch.stack(cov_traj)
