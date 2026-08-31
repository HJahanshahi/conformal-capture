"""
Real tumbling target for closed-loop simulation.

Wraps capture_lib's dataset: we pick a test trajectory and replay its
(true_states, observations) at the requested time.

For Step 7+ we want the target dynamics to match the data UPN was trained on
exactly, so we use the pre-computed trajectory rather than re-simulating.
"""
import os
import numpy as np

from cap_control import config as cfg


DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "capture_lib_v2", "tumbling_target_dataset_v2.npz")


class DatasetTumblingTarget:
    """
    Replay a pre-computed tumbling trajectory from capture_lib's dataset.

    Parameters
    ----------
    traj_idx : int
        Index into the TEST split of capture_lib's dataset.
    train_ratio, val_ratio : floats, matching capture_lib's split.
    world_offset : (3,) optional translation applied to target positions,
        so the target sits inside the chaser's reachable workspace.
    """

    def __init__(self, traj_idx=5, train_ratio=0.70, val_ratio=0.15,
                 world_offset=None, align_orientation=False):
        data = np.load(DATASET_PATH)
        N = data["true_states"].shape[0]
        n_train = int(train_ratio * N)
        n_val = int(val_ratio * N)
        test_idx = n_train + n_val + int(traj_idx)

        self.true_states = data["true_states"][test_idx].copy()     # (T, 13)
        self.times = data["times"].copy()                            # (T,)
        self.dt = float(self.times[1] - self.times[0])
        self.T_final = float(self.times[-1])

        # Translate position so the target starts at (0, 0, 0.65 * ARM_REACH) if no offset given
        if world_offset is None:
            world_offset = np.array([0.0, 0.0, 0.65 * cfg.ARM_REACH]) - self.true_states[0, 0:3]
        self.world_offset = np.asarray(world_offset, dtype=float)
        self.true_states[:, 0:3] += self.world_offset

        # Rotation offset: align initial orientation to identity.
        # The target's pose at t=0 may be far from chaser's home orientation
        # (paper-1's dataset samples uniform SO(3) initial orientations).
        # We rotate ALL trajectory quaternions by q_align = q0^-1 so that
        # the trajectory starts at identity. Tumble dynamics (relative
        # rotations) and body-frame angular velocity are unchanged.
        self.quat_align = np.array([1.0, 0.0, 0.0, 0.0])
        if align_orientation:
            q0 = self.true_states[0, 6:10].copy()
            q0 = q0 / max(np.linalg.norm(q0), 1e-12)
            # q_align = inverse of q0 = conjugate (since unit quaternion)
            q_align = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
            self.quat_align = q_align
            for t_idx in range(self.true_states.shape[0]):
                q = self.true_states[t_idx, 6:10]
                w1, x1, y1, z1 = q_align
                w2, x2, y2, z2 = q
                q_new = np.array([
                    w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2,
                ])
                q_new = q_new / max(np.linalg.norm(q_new), 1e-12)
                self.true_states[t_idx, 6:10] = q_new

    # ---- Query ----

    def state_at(self, t):
        """Interpolated 13-D state at world time t (clipped to trajectory length)."""
        t = float(np.clip(t, self.times[0], self.times[-1]))
        idx_f = (t - self.times[0]) / self.dt
        i0 = int(np.floor(idx_f))
        i0 = min(i0, len(self.times) - 2)
        alpha = idx_f - i0
        s0 = self.true_states[i0]
        s1 = self.true_states[i0 + 1]
        s = (1.0 - alpha) * s0 + alpha * s1
        # Re-normalize quaternion (linear interp then normalize)
        qn = s[6:10] / max(np.linalg.norm(s[6:10]), 1e-12)
        s[6:10] = qn
        return s

    def pose_at(self, t):
        """(position, quaternion) at world time t."""
        s = self.state_at(t)
        return s[0:3].copy(), s[6:10].copy()


__all__ = ["DatasetTumblingTarget", "DATASET_PATH"]
