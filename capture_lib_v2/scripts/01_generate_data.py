"""
Step 1: Generate tumbling target dataset.

Usage: python scripts/01_generate_data.py
"""

from capture.target import generate_dataset
from capture import config as cfg

if __name__ == "__main__":
    print("Generating tumbling target dataset...")
    true_states, observations, times, inertias = generate_dataset(
        n_traj=cfg.N_TRAJECTORIES,
        t_final=cfg.T_FINAL,
        n_steps=cfg.N_TIME_STEPS,
        pos_noise_std=cfg.POS_NOISE_STD,
        rot_noise_std_deg=cfg.ROT_NOISE_STD_DEG,
        seed=cfg.SEED,
        output_file="tumbling_target_dataset.npz",
    )
    print(f"✓ Saved {true_states.shape[0]} trajectories "
          f"({true_states.shape[1]} steps each)")
