"""
Step 2 v3: Train UPN with physics-consistency losses.

Adds three physics terms to the v2 training loss:
    L_kin  = ||d(pos)/dt - rdot||^2          (kinematic consistency)
    L_quat = ||d(quat)/dt - 0.5*omega(x)q||^2 (quaternion kinematics)
    L_acc  = ||d(rdot)/dt||^2                (force-free prior, soft)

This fixes UPN's velocity outputs which were unconstrained in v2.

Usage: python scripts/02_train_upn_v3.py
"""

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

from upn.core.upn import UPN
from upn.core.vech import vech, unvech
from capture.target import TumblingTargetDataset
from capture.utils import initialize_state_from_observations
from capture.utils.integration import euler_integrate, euler_integrate_with_updates
from capture import config as cfg

# Physics-consistency loss weights (v3 contribution)
LAMBDA_KIN  = 1.0    # weight on ||d(pos)/dt - rdot||^2
LAMBDA_QUAT = 1.0    # weight on quaternion kinematic consistency
LAMBDA_ACC  = 0.05   # weight on ||d(rdot)/dt||^2 (force-free prior)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Physics weights: kin={LAMBDA_KIN}, quat={LAMBDA_QUAT}, acc={LAMBDA_ACC}")

SAVE_DIR = "trained_models_target_v3"
os.makedirs(SAVE_DIR, exist_ok=True)


def quat_mult_batched(q1, q2):
    """Hamilton product of [w,x,y,z] quaternions, batched."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def compute_physics_loss(model, mean_pred, t_grid):
    """
    Penalize physical inconsistency in dynamics_net's outputs at predicted states.

    Args:
        model:     UPN
        mean_pred: (T, B, state_dim) predicted means
        t_grid:    (T,) time stamps for the predictions

    Returns:
        (L_kin, L_quat, L_acc): scalar physics losses
    """
    T, B, D = mean_pred.shape
    device = mean_pred.device

    # Subsample timesteps to reduce cost (~8 timesteps per trajectory)
    step = max(1, T // 8)
    sub_idx = list(range(0, T, step))
    n_sub = len(sub_idx)

    L_kin = torch.zeros((), device=device)
    L_quat = torch.zeros((), device=device)
    L_acc = torch.zeros((), device=device)

    for i in sub_idx:
        mu = mean_pred[i]                       # (B, D)
        S = torch.eye(D, device=device).unsqueeze(0).expand(B, -1, -1) * 0.01
        z = torch.cat([mu, vech(S)], dim=1)
        t_i = t_grid[i]

        dz = model(t_i, z)
        mu_dot = dz[:, :D]                      # (B, D)

        pos_dot   = mu_dot[:, 0:3]
        rdot_dot  = mu_dot[:, 3:6]
        quat_dot  = mu_dot[:, 6:10]
        rdot_state  = mu[:, 3:6]
        quat_state  = mu[:, 6:10]
        omega_state = mu[:, 10:13]

        # L_kin: d(pos)/dt = rdot
        L_kin = L_kin + ((pos_dot - rdot_state) ** 2).sum(dim=1).mean()

        # L_quat: d(q)/dt = 0.5 * omega_quat (x) q
        omega_quat = torch.cat([torch.zeros(B, 1, device=device), omega_state], dim=1)
        quat_dot_expected = 0.5 * quat_mult_batched(omega_quat, quat_state)
        L_quat = L_quat + ((quat_dot - quat_dot_expected) ** 2).sum(dim=1).mean()

        # L_acc: d(rdot)/dt ~ 0
        L_acc = L_acc + (rdot_dot ** 2).sum(dim=1).mean()

    return L_kin / n_sub, L_quat / n_sub, L_acc / n_sub


def compute_nll(model, true_obs, mean_pred, cov_pred):
    """NLL in observation space (unchanged from v2)."""
    T, B, _ = mean_pred.shape
    H = model.H.to(mean_pred.device)
    R = torch.diag(torch.exp(model.log_R_diag)).to(mean_pred.device)
    H_b = H.unsqueeze(0).expand(B, -1, -1)
    R_b = R.unsqueeze(0).expand(B, -1, -1)
    I_obs = 1e-4 * torch.eye(cfg.OBS_DIM, device=mean_pred.device)

    total = 0.0
    for t in range(T):
        obs_mu = torch.bmm(H_b, mean_pred[t].unsqueeze(-1)).squeeze(-1)
        obs_cov = torch.bmm(torch.bmm(H_b, cov_pred[t]), H_b.transpose(1, 2))
        obs_cov = obs_cov + R_b + I_obs
        try:
            dist = torch.distributions.MultivariateNormal(obs_mu, obs_cov)
            total = total + (-dist.log_prob(true_obs[:, t, :])).mean()
        except Exception:
            std = torch.sqrt(torch.diagonal(obs_cov, dim1=1, dim2=2)).clamp(min=1e-6)
            dist = torch.distributions.Normal(obs_mu, std)
            total = total + (-dist.log_prob(true_obs[:, t, :]).sum(dim=1)).mean()
    return total / T


def train_phase(model, optimizer, scheduler, train_loader, val_loader,
                num_epochs, use_updates, tag=""):
    best_val, best_state, patience = float('inf'), None, 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, n_b = 0.0, 0
        pbar = tqdm(train_loader, desc=f"[{tag}] Ep {epoch+1}/{num_epochs}", leave=False)
        for hist_obs, hist_t, fut_states, fut_obs, fut_t in pbar:
            hist_obs, hist_t = hist_obs.to(device), hist_t.to(device)
            fut_states, fut_obs, fut_t = fut_states.to(device), fut_obs.to(device), fut_t.to(device)

            B = hist_obs.shape[0]
            mu0 = initialize_state_from_observations(hist_obs, hist_t)
            S0 = torch.eye(cfg.STATE_DIM, device=device).unsqueeze(0).expand(B, -1, -1) * 0.01
            t_grid = torch.cat([hist_t[0, -1:], fut_t[0]])

            optimizer.zero_grad()
            if use_updates:
                mean_pred, cov_pred = euler_integrate_with_updates(
                    model, mu0, S0, t_grid,
                    observations=fut_obs, update_frequency=cfg.UPDATE_FREQ,
                )
            else:
                mean_pred, cov_pred = euler_integrate(model, mu0, S0, t_grid)

            mean_pred_obs, cov_pred_obs = mean_pred[1:], cov_pred[1:]
            nll = compute_nll(model, fut_states[:, :, cfg.OBS_INDICES],
                              mean_pred_obs, cov_pred_obs)

            # v3: physics losses on the FULL predicted trajectory (excluding anchor)
            L_kin, L_quat, L_acc = compute_physics_loss(model, mean_pred[1:], t_grid[1:])

            loss = nll + LAMBDA_KIN * L_kin + LAMBDA_QUAT * L_quat + LAMBDA_ACC * L_acc

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item(); n_b += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}", nll=f"{nll.item():.2f}",
                              kin=f"{L_kin.item():.3f}", quat=f"{L_quat.item():.3f}",
                              acc=f"{L_acc.item():.3f}")

        avg_train = epoch_loss / n_b
        train_losses.append(avg_train)

        # Validation
        model.eval()
        val_loss, n_v = 0.0, 0
        with torch.no_grad():
            for hist_obs, hist_t, fut_states, fut_obs, fut_t in val_loader:
                hist_obs, hist_t = hist_obs.to(device), hist_t.to(device)
                fut_states, fut_obs, fut_t = fut_states.to(device), fut_obs.to(device), fut_t.to(device)
                B = hist_obs.shape[0]
                mu0 = initialize_state_from_observations(hist_obs, hist_t)
                S0 = torch.eye(cfg.STATE_DIM, device=device).unsqueeze(0).expand(B, -1, -1) * 0.01
                t_grid = torch.cat([hist_t[0, -1:], fut_t[0]])
                if use_updates:
                    mp, cp = euler_integrate_with_updates(model, mu0, S0, t_grid,
                                                           observations=fut_obs,
                                                           update_frequency=cfg.UPDATE_FREQ)
                else:
                    mp, cp = euler_integrate(model, mu0, S0, t_grid)
                mp_obs, cp_obs = mp[1:], cp[1:]
                nll_v = compute_nll(model, fut_states[:, :, cfg.OBS_INDICES], mp_obs, cp_obs)
                Lk, Lq, La = compute_physics_loss(model, mp[1:], t_grid[1:])
                loss = nll_v + LAMBDA_KIN * Lk + LAMBDA_QUAT * Lq + LAMBDA_ACC * La
                val_loss += loss.item(); n_v += 1

        avg_val = val_loss / n_v
        val_losses.append(avg_val)
        scheduler.step(avg_val)
        print(f"  [{tag}] Ep {epoch+1}: train={avg_train:.3f}  val={avg_val:.3f}")

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0; print(f"    ok best (val={best_val:.3f})")
        else:
            patience += 1
            if patience >= cfg.EARLY_STOP_PATIENCE:
                print(f"    Early stop at epoch {epoch+1}"); break

    if best_state:
        model.load_state_dict(best_state)
    return train_losses, val_losses


def main():
    print("Loading data...")
    data = np.load("tumbling_target_dataset_v2.npz")
    N = data['true_states'].shape[0]
    n_train = int(cfg.TRAIN_RATIO * N)
    n_val = int(cfg.VAL_RATIO * N)

    train_ds = TumblingTargetDataset(
        data['true_states'][:n_train], data['observations'][:n_train],
        data['times'], cfg.HISTORY_LEN, cfg.FUTURE_LEN, cfg.WINDOW_STRIDE)
    val_ds = TumblingTargetDataset(
        data['true_states'][n_train:n_train+n_val],
        data['observations'][n_train:n_train+n_val],
        data['times'], cfg.HISTORY_LEN, cfg.FUTURE_LEN, cfg.WINDOW_STRIDE)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, pin_memory=True)
    print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    model = UPN(state_dim=cfg.STATE_DIM, obs_dim=cfg.OBS_DIM,
                hidden_dim=cfg.HIDDEN_DIM, obs_indices=cfg.OBS_INDICES).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Phase 1: prediction-only (no measurement updates)
    opt = optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, min_lr=1e-6)
    t0 = time.time()
    l1t, l1v = train_phase(model, opt, sched, train_loader, val_loader,
                           cfg.NUM_EPOCHS_P1, use_updates=False, tag="P1")
    print(f"Phase 1: {(time.time()-t0)/60:.1f} min")

    # Phase 2: with measurement updates
    opt = optim.Adam(model.parameters(), lr=cfg.LR*0.5, weight_decay=cfg.WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, min_lr=1e-6)
    t0 = time.time()
    l2t, l2v = train_phase(model, opt, sched, train_loader, val_loader,
                           cfg.NUM_EPOCHS_P2, use_updates=True, tag="P2")
    print(f"Phase 2: {(time.time()-t0)/60:.1f} min")

    torch.save({
        'model_state_dict': model.state_dict(),
        'losses_p1': {'train': l1t, 'val': l1v},
        'losses_p2': {'train': l2t, 'val': l2v},
        'config': {
            'state_dim': cfg.STATE_DIM, 'obs_dim': cfg.OBS_DIM,
            'obs_indices': cfg.OBS_INDICES, 'hidden_dim': cfg.HIDDEN_DIM,
            'history_len': cfg.HISTORY_LEN, 'future_len': cfg.FUTURE_LEN,
            'update_freq': cfg.UPDATE_FREQ,
            'lambda_kin': LAMBDA_KIN,
            'lambda_quat': LAMBDA_QUAT,
            'lambda_acc': LAMBDA_ACC,
        },
    }, os.path.join(SAVE_DIR, 'upn_target_v3.pt'))
    print(f"Model saved to {SAVE_DIR}/upn_target_v3.pt")


if __name__ == "__main__":
    main()
