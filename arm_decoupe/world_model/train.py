"""
Train the RSSM world model.

Usage:
    python -m world_model.train               # default hyperparameters
    python -m world_model.train --epochs 200  # override one param
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (TrajectoryDataset, collate_fn, target_dim,
                      split_by_session, _load_episode, _is_session_file,
                      _session_id, _load_session_history, _build_history,
                      HISTORY_K, HISTORY_DIM)
from .model import WorldModel


# ── default hyperparameters ────────────────────────────────────────────────────
DEFAULTS = dict(
    early_stopping       = 40,
    shape_embed_dim      = 256,
    h_dim                = 512,
    obs_dim              = 2,        # déterminé automatiquement depuis target_keys
    dropout              = 0.1,
    gru_layers           = 3,
    pe_dim               = 64,
    lr                   = 3e-4,
    weight_decay         = 1e-4,
    batch_size           = 32,
    epochs               = 100,
    grad_clip            = 5.0,
    val_split            = 0.1,
    seed                 = 42,
    num_workers          = 0,        # 0 = optimal : dataset entièrement pré-chargé en RAM
    subsample_factor     = 1,        # sous-échantillonnage temporel (ex: 10 = 1 step sur 10)
    use_amp              = True,     # mixed precision BF16 (A100/T4 uniquement)
    target_keys          = "q_error",  # q_error = q_real-q_des : prédit l'écart à la consigne, pas la trajectoire absolue
    lambda_dev           = 1.0,      # poids de la loss cut_deviation (MSE, espace normalisé)
    lambda_defect        = 0.5,      # poids de la loss cut_defect (BCE)
    lambda_vel           = 0.5,      # poids de la loss sur les différences temporelles (d(q)/dt)
    lambda_acc           = 0.0,      # désactivé : récompense le collapse (acc=0 sur 80% des steps)
    lambda_var           = 0.5,      # variance matching : pénalise si std(pred) ≠ std(target) dans le temps
    cut_weight           = 10.0,     # facteur multiplicatif sur les steps is_cutting dans la loss
    rollout_eval_every   = 10,       # évaluer le rollout autorégressif toutes les N epochs (0 = jamais)
    n_rollout_sessions   = 5,        # nombre de sessions de val utilisées pour le rollout
    save_dir             = "world_model/checkpoints",
    data_dir             = "dataset",
    db_path              = "pieces_database.json",
)


# ── loss ───────────────────────────────────────────────────────────────────────
def compute_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    seq_lengths: torch.Tensor,
    is_cutting: torch.Tensor | None = None,
    cut_weight: float = 1.0,
) -> torch.Tensor:
    """MSE masqué sur les séquences de longueurs variables.
    cut_weight > 1 : upweight les steps de découpe pour éviter le collapse sur la position initiale.
    """
    B, T, D = preds.shape
    device  = preds.device
    mask    = (torch.arange(T, device=device)[None, :] < seq_lengths[:, None]).float()
    if is_cutting is not None and cut_weight > 1.0:
        mask = mask * (1.0 + (cut_weight - 1.0) * is_cutting[:, :T])
    n       = mask.sum() * D + 1e-8
    return ((preds - targets).pow(2) * mask[..., None]).sum() / n


# ── device ─────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── training loop ──────────────────────────────────────────────────────────────
def train(cfg: dict | None = None):
    H = {**DEFAULTS, **(cfg or {})}
    torch.manual_seed(H["seed"])
    device = get_device()
    print(f"Device : {device}")

    # ── data ──────────────────────────────────────────────────────────────────
    with open(H["db_path"]) as f:
        db_json  = json.load(f)
    piece_db = db_json["pieces"]

    episode_paths = sorted(
        glob.glob(os.path.join(H["data_dir"], "episode_*.npz")) +
        glob.glob(os.path.join(H["data_dir"], "session_*_piece*.npz"))
    )
    print(f"Episodes : {len(episode_paths)}")

    os.makedirs(H["save_dir"], exist_ok=True)

    # Signaux cibles : liste depuis la chaîne CLI "q_real" ou "q_real,q_des" etc.
    raw_keys = H.get("target_keys", "q_real")
    target_keys = [k.strip() for k in raw_keys.split(",")] if isinstance(raw_keys, str) else raw_keys
    obs_dim = target_dim(target_keys)
    H["obs_dim"]     = obs_dim
    H["target_keys"] = target_keys
    print(f"Signaux cibles : {target_keys}  →  obs_dim={obs_dim}")

    # Split par session complète pour éviter la fuite d'information via les historiques
    train_paths, val_paths = split_by_session(episode_paths, H["val_split"], H["seed"])
    train_ds = TrajectoryDataset(train_paths, piece_db or None, target_keys=target_keys)
    val_ds   = TrajectoryDataset(val_paths,   piece_db or None,
                                 normalizer=train_ds.normalizer, target_keys=target_keys)
    train_ds.normalizer.save(os.path.join(H["save_dir"], "normalizer.npz"))

    nw  = H["num_workers"]
    pin = device.type == "cuda"   # pin_memory utile même avec num_workers=0 sur CUDA
    train_loader = DataLoader(
        train_ds, batch_size=H["batch_size"],
        shuffle=True, collate_fn=collate_fn,
        num_workers=nw, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=H["batch_size"],
        shuffle=False, collate_fn=collate_fn,
        num_workers=nw, pin_memory=pin,
    )
    print(f"Train : {len(train_ds)}  |  Val : {len(val_ds)}"
          f"  ({len({_session_id(p) for p in train_paths if _is_session_file(p)})} sess train"
          f" / {len({_session_id(p) for p in val_paths if _is_session_file(p)})} sess val)")

    # ── model ─────────────────────────────────────────────────────────────────
    model = WorldModel(
        shape_embed_dim = H["shape_embed_dim"],
        h_dim           = H["h_dim"],
        obs_dim         = H["obs_dim"],
        dropout         = H["dropout"],
        gru_layers      = H["gru_layers"],
        pe_dim          = H.get("pe_dim", 64),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=H["lr"], weight_decay=H["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=H["epochs"], eta_min=1e-5)

    best_val      = float("inf")
    history       = dict(train=[], val=[], lr=[], grad_norm=[], rollout=[])
    epochs_no_imp = 0
    t_start       = time.time()

    # Mixed precision
    use_amp = H["use_amp"] and device.type == "cuda"
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)
    amp_dtype = torch.bfloat16   # BF16 : pas de perte de dynamique sur A100/T4
    print(f"Mixed precision BF16 : {'ON' if use_amp else 'OFF'}")

    # Sous-échantillonnage temporel
    sf = max(1, int(H["subsample_factor"]))
    if sf > 1:
        print(f"Sous-échantillonnage : 1 step sur {sf} (T divisé par {sf})")

    # ── epoch loop ────────────────────────────────────────────────────────────
    epoch_bar = tqdm(range(1, H["epochs"] + 1), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        t_epoch = time.time()

        # --- train ---
        model.train()
        train_total     = 0.0
        total_grad_norm = 0.0

        for (wps, wp_len, speed, obs, seq_len,
             cut_dev, cut_defect, is_cutting,
             dev_hist, piece_count, cadence, temperature) in train_loader:
            wps, wp_len  = wps.to(device), wp_len.to(device)
            speed        = speed.to(device)
            obs, seq_len = obs.to(device), seq_len.to(device)
            cut_dev      = cut_dev.to(device)
            cut_defect   = cut_defect.to(device)
            is_cutting   = is_cutting.to(device)
            dev_hist     = dev_hist.to(device)
            piece_count  = piece_count.to(device)
            cadence      = cadence.to(device)
            temperature  = temperature.to(device)

            if sf > 1:
                obs        = obs[:, ::sf, :].contiguous()
                cut_dev    = cut_dev[:, ::sf].contiguous()
                cut_defect = cut_defect[:, ::sf].contiguous()
                is_cutting = is_cutting[:, ::sf].contiguous()
                seq_len    = (seq_len + sf - 1) // sf

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                preds = model(wps, wp_len, speed,
                              deviation_history=dev_hist,
                              piece_count=piece_count,
                              cadence=cadence,
                              temperature=temperature,
                              targets=obs)
                cw = H.get("cut_weight", 1.0)
                loss = compute_loss(preds, obs, seq_len, is_cutting, cw)
                if H["lambda_vel"] > 0:
                    vel_pred = preds[:, 1:] - preds[:, :-1]
                    vel_true = obs[:, 1:]   - obs[:, :-1]
                    loss = loss + H["lambda_vel"] * compute_loss(
                        vel_pred, vel_true, (seq_len - 1).clamp(min=0),
                        is_cutting[:, 1:], cw,
                    )
                if H.get("lambda_acc", 0) > 0:
                    acc_pred = preds[:, 2:] - 2 * preds[:, 1:-1] + preds[:, :-2]
                    acc_true = obs[:, 2:]   - 2 * obs[:, 1:-1]   + obs[:, :-2]
                    loss = loss + H["lambda_acc"] * compute_loss(
                        acc_pred, acc_true, (seq_len - 2).clamp(min=0),
                        is_cutting[:, 2:], cw,
                    )
                if H.get("lambda_var", 0) > 0:
                    pred_std = preds.std(dim=1)
                    true_std = obs.std(dim=1)
                    loss = loss + H["lambda_var"] * F.mse_loss(pred_std, true_std)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), H["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            train_total     += loss.item()
            total_grad_norm += grad_norm.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # --- validate --- (toujours en teacher forcing pour comparer proprement)
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for (wps, wp_len, speed, obs, seq_len,
                 cut_dev, cut_defect, is_cutting,
                 dev_hist, piece_count, cadence, temperature) in val_loader:
                wps, wp_len  = wps.to(device), wp_len.to(device)
                speed        = speed.to(device)
                obs, seq_len = obs.to(device), seq_len.to(device)
                cut_dev      = cut_dev.to(device)
                cut_defect   = cut_defect.to(device)
                is_cutting   = is_cutting.to(device)
                dev_hist     = dev_hist.to(device)
                piece_count  = piece_count.to(device)
                cadence      = cadence.to(device)
                temperature  = temperature.to(device)
                if sf > 1:
                    obs        = obs[:, ::sf, :].contiguous()
                    cut_dev    = cut_dev[:, ::sf].contiguous()
                    cut_defect = cut_defect[:, ::sf].contiguous()
                    is_cutting = is_cutting[:, ::sf].contiguous()
                    seq_len    = (seq_len + sf - 1) // sf
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    preds = model(wps, wp_len, speed,
                                  deviation_history=dev_hist,
                                  piece_count=piece_count,
                                  cadence=cadence,
                                  temperature=temperature,
                                  targets=obs)
                    val_loss = compute_loss(preds, obs, seq_len)
                    if H["lambda_vel"] > 0:
                        vel_pred = preds[:, 1:] - preds[:, :-1]
                        vel_true = obs[:, 1:]   - obs[:, :-1]
                        cw = H.get("cut_weight", 1.0)
                    val_loss = val_loss + H["lambda_vel"] * compute_loss(
                            vel_pred, vel_true, (seq_len - 1).clamp(min=0),
                            is_cutting[:, 1:], cw,
                        )
                    if H.get("lambda_acc", 0) > 0:
                        acc_pred = preds[:, 2:] - 2 * preds[:, 1:-1] + preds[:, :-2]
                        acc_true = obs[:, 2:]   - 2 * obs[:, 1:-1]   + obs[:, :-2]
                        val_loss = val_loss + H["lambda_acc"] * compute_loss(
                            acc_pred, acc_true, (seq_len - 2).clamp(min=0),
                            is_cutting[:, 2:], cw,
                        )
                    if H.get("lambda_var", 0) > 0:
                        pred_std = preds.std(dim=1)
                        true_std = obs.std(dim=1)
                        val_loss = val_loss + H["lambda_var"] * F.mse_loss(pred_std, true_std)
                    val_total += val_loss.item()

        avg_train     = train_total     / len(train_loader)
        avg_val       = val_total       / len(val_loader)
        avg_grad_norm = total_grad_norm / len(train_loader)
        epoch_time    = time.time() - t_epoch
        elapsed       = time.time() - t_start
        eta           = elapsed / epoch * (H["epochs"] - epoch)

        history["train"    ].append(avg_train)
        history["val"      ].append(avg_val)
        history["lr"       ].append(current_lr)
        history["grad_norm"].append(avg_grad_norm)

        rollout_mse = None
        improved = avg_val < best_val
        if improved:
            best_val      = avg_val
            epochs_no_imp = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "hyperparams": H,
                    "val_loss": avg_val,
                    "dev_mean": train_ds.dev_mean,
                    "dev_std":  train_ds.dev_std,
                },
                os.path.join(H["save_dir"], "best_model.pt"),
            )
        else:
            epochs_no_imp += 1

        rollout_str = f"  rollout={rollout_mse*1e6:.1f}µm²" if rollout_mse is not None else ""
        marker = " ★" if improved else f" ({epochs_no_imp})"
        epoch_bar.set_postfix(
            val=f"{avg_val:.4f}", best=f"{best_val:.4f}",
            lr=f"{current_lr:.2e}"
        )
        tqdm.write(
            f"[{epoch:3d}/{H['epochs']}] {epoch_time:5.1f}s  ETA {_fmt_time(eta)}  "
            f"lr={current_lr:.2e}  ‖g‖={avg_grad_norm:.2f}  "
            f"train={avg_train:.4f}  val={avg_val:.4f}{rollout_str}{marker}"
        )

        # Early stopping
        patience = H["early_stopping"]
        if patience > 0 and epochs_no_imp >= patience:
            tqdm.write(f"\nEarly stopping — pas d'amélioration depuis {patience} epochs.")
            break

    total_time = time.time() - t_start
    print(f"\nDone in {_fmt_time(total_time)}.  Best val loss : {best_val:.4f}")
    _save_curves(history, H["save_dir"])
    return model, train_ds.normalizer, H


# ── plot helpers ───────────────────────────────────────────────────────────────
def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _save_curves(history: dict, save_dir: str):
    has_rollout = bool(history.get("rollout"))
    n_plots = 4 if has_rollout else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    ax = axes[0]
    ax.plot(history["train"], label="Train")
    ax.plot(history["val"],   label="Val")
    ax.set_title("MSE loss (log)"); ax.legend(); ax.set_xlabel("Epoch")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(history["lr"])
    ax.set_title("Learning rate"); ax.set_xlabel("Epoch")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(history["grad_norm"])
    ax.set_title("Gradient norm (avg per epoch)"); ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    if has_rollout:
        ax = axes[3]
        epochs_r, mses = zip(*history["rollout"])
        ax.plot(epochs_r, [m * 1e6 for m in mses], marker='o', markersize=4)
        ax.set_title("Rollout MSE (µm²) — autorégressif"); ax.set_xlabel("Epoch")
        ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves → {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=str if k == "target_keys" else type(v), default=v)
    # Arguments supprimés — acceptés silencieusement pour compatibilité avec les anciennes commandes
    p.add_argument("--error_weight_gamma", type=float, default=None)
    p.add_argument("--ss_start_epoch",     type=int,   default=None)
    p.add_argument("--ss_end_epoch",       type=int,   default=None)
    p.add_argument("--ss_p_min",           type=float, default=None)
    args = vars(p.parse_args())
    args.pop("error_weight_gamma", None)
    args.pop("ss_start_epoch",     None)
    args.pop("ss_end_epoch",       None)
    args.pop("ss_p_min",           None)
    return args


if __name__ == "__main__":
    train(cfg=_parse_args())
