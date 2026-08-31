"""
UPN predictor wrapper for receding-horizon MPC.

Loads a trained UPN checkpoint once, caches it, and exposes a single numpy-in
numpy-out `predict` method the MPC can call each control cycle.

Responsibilities:
  - Own the torch model and keep it in eval mode.
  - Handle state seeding from an observation history (uses capture_lib's helper).
  - Run Euler integration with or without Kalman measurement updates.
  - Return (mean_trajectory, covariance_trajectory, times) as numpy arrays.

The MPC does not touch torch. All tensor conversion is contained here.
"""
import numpy as np
import torch


from upn.core.upn import UPN

# Reuse the existing, battle-tested integrators from capture_lib.
from capture.utils.integration import euler_integrate, euler_integrate_with_updates
from capture.utils.state import initialize_state_from_observations

from cap_control import config as cfg


class UPNPredictor:
    """
    Thin wrapper around a trained UPN checkpoint.

    Parameters
    ----------
    model_path : str
        Path to a .pt checkpoint with `model_state_dict` key.
    device : str or torch.device, optional
        Defaults to 'cuda' if available, else 'cpu'.
    state_dim, obs_dim, hidden_dim, obs_indices : optional overrides.
        Defaults match cap_control.config.
    update_freq : int
        Kalman update frequency during integration (every N steps). Defaults to cfg.UPN_UPDATE_FREQ.
    initial_sigma : float
        Initial per-component std used to seed the covariance (S0 = sigma^2 * I).
        Defaults to 0.1 (matching capture_lib's S0 = 0.01 * I).
    """

    def __init__(self,
                 model_path=None,
                 device=None,
                 state_dim=None, obs_dim=None, hidden_dim=None, obs_indices=None,
                 update_freq=None,
                 initial_sigma=0.1):
        self.model_path = model_path or cfg.UPN_MODEL_PATH
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim if state_dim is not None else cfg.TARGET_STATE_DIM
        self.obs_dim = obs_dim if obs_dim is not None else cfg.TARGET_OBS_DIM
        self.hidden_dim = hidden_dim if hidden_dim is not None else cfg.UPN_HIDDEN_DIM
        self.obs_indices = list(obs_indices) if obs_indices is not None else list(cfg.TARGET_OBS_INDICES)
        self.update_freq = int(update_freq) if update_freq is not None else int(cfg.UPN_UPDATE_FREQ)
        self.initial_sigma = float(initial_sigma)

        self.model = self._load()

    def _load(self):
        ckpt = torch.load(self.model_path, weights_only=False, map_location=self.device)
        model = UPN(
            state_dim=self.state_dim, obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim, obs_indices=self.obs_indices,
        ).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    # ---- Seeding ----

    def initial_state(self, history_obs, history_time):
        """
        Estimate the initial 13-D state from a history of 7-D observations.

        Parameters
        ----------
        history_obs  : (H, 7) numpy array
        history_time : (H,)  numpy array

        Returns
        -------
        mu0 : (13,) numpy array
        """
        history_obs = np.asarray(history_obs, dtype=np.float64)
        history_time = np.asarray(history_time, dtype=np.float64)
        if history_obs.ndim != 2 or history_obs.shape[1] != self.obs_dim:
            raise ValueError(f"history_obs must be (H, {self.obs_dim}); got {history_obs.shape}")
        if history_time.ndim != 1 or history_time.shape[0] != history_obs.shape[0]:
            raise ValueError("history_time shape must match history_obs[0]")
        return initialize_state_from_observations(history_obs, history_time)

    # ---- Prediction ----

    def predict(self, history_obs, history_time, future_time,
                future_obs=None, use_updates=True):
        """
        Predict target trajectory over a future horizon with propagated uncertainty.

        Parameters
        ----------
        history_obs  : (H, 7) numpy array of past noisy observations
        history_time : (H,)  numpy array of past time stamps
        future_time  : (F,)  numpy array of future time stamps to predict at
        future_obs   : (F, 7) numpy array of future noisy observations OR None.
                       If provided AND use_updates=True, applied as Kalman updates
                       every `update_freq` steps during integration.
        use_updates  : bool. If False, pure prediction (no measurement updates).

        Returns
        -------
        mean_traj : (F+1, 13) numpy array — state mean at [t_hist_last, *future_time]
        cov_traj  : (F+1, 13, 13) numpy array — state covariance at the same instants
        time_grid : (F+1,) numpy array — [history_time[-1], *future_time]

        The first element of each array is the integration anchor (same as the
        seeded initial state). The MPC typically consumes trajectories starting
        from index 1.
        """
        history_obs = np.asarray(history_obs, dtype=np.float64)
        history_time = np.asarray(history_time, dtype=np.float64)
        future_time = np.asarray(future_time, dtype=np.float64)

        # Seed mean state from observation history
        mu0_np = self.initial_state(history_obs, history_time)

        # Build integration grid: [hist_last, *future]
        t_grid_np = np.concatenate([history_time[-1:], future_time])

        # Torch tensors for UPN
        mu0 = torch.tensor(mu0_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        S0 = torch.eye(self.state_dim, device=self.device).unsqueeze(0) * (self.initial_sigma ** 2)
        t_grid = torch.tensor(t_grid_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if use_updates:
                if future_obs is None:
                    raise ValueError("use_updates=True requires future_obs to be provided.")
                future_obs_np = np.asarray(future_obs, dtype=np.float64)
                if future_obs_np.shape != (future_time.shape[0], self.obs_dim):
                    raise ValueError(
                        f"future_obs must be (F={future_time.shape[0]}, {self.obs_dim}); "
                        f"got {future_obs_np.shape}")
                obs_t = torch.tensor(future_obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                mean_t, cov_t = euler_integrate_with_updates(
                    self.model, mu0, S0, t_grid, obs_t, self.update_freq,
                    state_dim=self.state_dim,
                )
            else:
                mean_t, cov_t = euler_integrate(
                    self.model, mu0, S0, t_grid, state_dim=self.state_dim,
                )

        mean_np = mean_t[:, 0].cpu().numpy()
        cov_np = cov_t[:, 0].cpu().numpy()
        return mean_np, cov_np, t_grid_np


__all__ = ["UPNPredictor"]
