"""
Sensor model: produces noisy 7-D observations of the target (position + quaternion).

Matches capture_lib's training conditions:
    - Position noise:     Gaussian sigma = POS_NOISE_STD meters
    - Orientation noise:  small random rotation, sigma = ROT_NOISE_STD_DEG degrees
"""
import numpy as np

from cap_control import config as cfg
from cap_control.utils.transforms import quat_mult, quat_normalize


class NoisyPoseSensor:
    """
    Emits a 7-D observation [p(3), q(4)] corrupted by Gaussian position noise
    and small-angle rotation noise, reproducing capture_lib's observation model.
    """

    def __init__(self, pos_noise_std=None, rot_noise_std_deg=None, seed=0):
        self.pos_std = float(pos_noise_std if pos_noise_std is not None else cfg.POS_NOISE_STD)
        self.rot_std = np.deg2rad(float(rot_noise_std_deg if rot_noise_std_deg is not None
                                        else cfg.ROT_NOISE_STD_DEG))
        self.rng = np.random.default_rng(seed)

    def observe(self, true_pos, true_quat):
        """
        Parameters
        ----------
        true_pos  : (3,)
        true_quat : (4,) [w,x,y,z]

        Returns
        -------
        obs : (7,) [p_noisy(3), q_noisy(4)]
        """
        pos_noisy = np.asarray(true_pos, dtype=float) + self.rng.normal(
            scale=self.pos_std, size=3)

        # Random small-axis rotation
        axis = self.rng.normal(size=3)
        axis /= max(np.linalg.norm(axis), 1e-12)
        angle = self.rng.normal(scale=self.rot_std)
        half = 0.5 * angle
        dq = np.array([np.cos(half), *(axis * np.sin(half))])

        q_noisy = quat_normalize(quat_mult(dq, np.asarray(true_quat, dtype=float)))
        return np.concatenate([pos_noisy, q_noisy])


__all__ = ["NoisyPoseSensor"]
