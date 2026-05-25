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
                      _session_id, HISTORY_K)
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
    target_keys          = "q_real",  # signaux à prédire : "q_real" | "q_error" (= q_real-q_des) | séparés par virgule
    lambda_dev           = 1.0,      # poids de la loss cut_deviation (MSE, espace normalisé)
    lambda_defect        = 0.5,      # poids de la loss cut_defect (BCE)
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
) -> torch.Tensor:
    """MSE masqué sur les séquences de longueurs variables."""
    B, T, D = preds.shape
    device  = preds.device
    mask    = (torch.arange(T, device=device)[None, :] < seq_lengths[:, None]).float()
    n       = mask.sum() * D + 1e-8
    return ((preds - targets).pow(2) * mask[..., None]).sum() / n


def compute_quality_loss(
    quality_preds: torch.Tensor,    # (B, T, 2)
    cut_dev: torch.Tensor,          # (B, T) — normalisé
    cut_defect: torch.Tensor,       # (B, T) — 0/1
    is_cutting: torch.Tensor,       # (B, T) — masque
    seq_lengths: torch.Tensor,
    lambda_dev: float,
    lambda_defect: float,
) -> torch.Tensor:
    B, T = cut_dev.shape
    device = quality_preds.device
    seq_mask = (torch.arange(T, device=device)[None, :] < seq_lengths[:, None]).float()
    cut_mask = is_cutting * seq_mask   # (B, T) — actif uniquement pendant la découpe
    n_cut    = cut_mask.sum() + 1e-8

    loss = torch.tensor(0.0, device=device)
    if lambda_dev > 0:
        loss = loss + lambda_dev * (
            (quality_preds[:, :, 0] - cut_dev).pow(2) * cut_mask
        ).sum() / n_cut
    if lambda_defect > 0:
        loss = loss + lambda_defect * (
            F.binary_cross_entropy_with_logits(
                quality_preds[:, :, 1], cut_defect, reduction="none"
            ) * cut_mask
        ).sum() / n_cut
    return loss


# ── device ─────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── rollout autorégressif ──────────────────────────────────────────────────────

def eval_rollout(model, val_paths, piece_db, target_keys,
                 normalizer, dev_mean, dev_std, device, n_sessions=5):
    """
    Évalue le modèle en déroulement autorégressif sur n_sessions sessions de val.

    À chaque pièce, l'historique est construit depuis les déviations que le modèle
    a lui-même prédites (pas les vraies valeurs) — ce qui simule l'usage réel où
    on prédit une session complète sans observer le comportement réel entre les pièces.

    Retourne le MSE moyen (en m²) entre la courbe de déviation prédite et la courbe réelle.
    """
    # Grouper les chemins de val par session
    sessions: dict[str, list[str]] = {}
    for p in val_paths:
        if _is_session_file(p):
            sessions.setdefault(_session_id(p), []).append(p)

    if not sessions:
        return None

    eval_sids = sorted(sessions.keys())[:n_sessions]
    all_session_mse = []

    model.eval()
    with torch.no_grad():
        for sid in eval_sids:
            # Trier les pièces par numéro croissant
            paths = sorted(
                sessions[sid],
                key=lambda p: int(re.search(r"piece(\d+)", p).group(1))
            )
            pred_devs: list[float] = []   # déviations prédites (en mètres)
            true_devs: list[float] = []

            for i, path in enumerate(paths):
                ep = _load_episode(path, piece_db, target_keys)

                # Historique depuis les prédictions passées (pas les vraies valeurs)
                hist_raw = pred_devs[max(0, i - HISTORY_K):i]
                pad      = [0.0] * (HISTORY_K - len(hist_raw))
                hist     = np.array(pad + hist_raw, dtype=np.float32)

                wps    = torch.from_numpy(ep["waypoints"]).unsqueeze(0).to(device)
                wp_len = torch.tensor([ep["waypoints"].shape[0]]).to(device)
                speed  = torch.tensor([[ep["speed"]]]).to(device)
                dev_h  = torch.from_numpy(hist).view(1, HISTORY_K, 1).to(device)
                pc     = torch.tensor([[float(ep["piece_count"])]]).to(device)
                cad    = torch.tensor([[float(ep["cadence"])]]).to(device)

                _, quality = model(wps, wp_len, speed,
                                   deviation_history=dev_h,
                                   piece_count=pc,
                                   cadence=cad,
                                   max_len=ep["length"])

                # Déviation prédite : moyenne quality[:, :, 0] sur les pas en découpe
                is_cut = torch.from_numpy(ep["is_cutting"]).to(device).bool()
                T_ep   = min(quality.shape[1], len(is_cut))
                q_cut  = quality[0, :T_ep, 0][is_cut[:T_ep]]
                pred_dev = float(q_cut.mean()) * dev_std + dev_mean if q_cut.numel() > 0 else 0.0
                pred_dev = max(0.0, pred_dev)

                pred_devs.append(pred_dev)
                true_devs.append(ep["mean_cut_deviation"])

            mse = float(np.mean((np.array(pred_devs) - np.array(true_devs)) ** 2))
            all_session_mse.append(mse)

    return float(np.mean(all_session_mse)) if all_session_mse else None


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
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
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
             dev_hist, piece_count, cadence) in train_loader:
            wps, wp_len  = wps.to(device), wp_len.to(device)
            speed        = speed.to(device)
            obs, seq_len = obs.to(device), seq_len.to(device)
            cut_dev      = cut_dev.to(device)
            cut_defect   = cut_defect.to(device)
            is_cutting   = is_cutting.to(device)
            dev_hist     = dev_hist.to(device)
            piece_count  = piece_count.to(device)
            cadence      = cadence.to(device)

            if sf > 1:
                obs        = obs[:, ::sf, :].contiguous()
                cut_dev    = cut_dev[:, ::sf].contiguous()
                cut_defect = cut_defect[:, ::sf].contiguous()
                is_cutting = is_cutting[:, ::sf].contiguous()
                seq_len    = (seq_len + sf - 1) // sf

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                preds, quality = model(wps, wp_len, speed,
                                       deviation_history=dev_hist,
                                       piece_count=piece_count,
                                       cadence=cadence,
                                       targets=obs)
                loss = compute_loss(preds, obs, seq_len)
                loss = loss + compute_quality_loss(
                    quality, cut_dev, cut_defect, is_cutting, seq_len,
                    H["lambda_dev"], H["lambda_defect"],
                )
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
                 dev_hist, piece_count, cadence) in val_loader:
                wps, wp_len  = wps.to(device), wp_len.to(device)
                speed        = speed.to(device)
                obs, seq_len = obs.to(device), seq_len.to(device)
                cut_dev      = cut_dev.to(device)
                cut_defect   = cut_defect.to(device)
                is_cutting   = is_cutting.to(device)
                dev_hist     = dev_hist.to(device)
                piece_count  = piece_count.to(device)
                cadence      = cadence.to(device)
                if sf > 1:
                    obs        = obs[:, ::sf, :].contiguous()
                    cut_dev    = cut_dev[:, ::sf].contiguous()
                    cut_defect = cut_defect[:, ::sf].contiguous()
                    is_cutting = is_cutting[:, ::sf].contiguous()
                    seq_len    = (seq_len + sf - 1) // sf
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    preds, quality = model(wps, wp_len, speed,
                                           deviation_history=dev_hist,
                                           piece_count=piece_count,
                                           cadence=cadence,
                                           targets=obs)
                    val_loss = compute_loss(preds, obs, seq_len)
                    val_loss = val_loss + compute_quality_loss(
                        quality, cut_dev, cut_defect, is_cutting, seq_len,
                        H["lambda_dev"], H["lambda_defect"],
                    )
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

        # Rollout autorégressif (évaluation sur sessions complètes de val)
        rollout_mse = None
        every = H.get("rollout_eval_every", 10)
        if every > 0 and epoch % every == 0:
            rollout_mse = eval_rollout(
                model, val_paths, piece_db, target_keys,
                train_ds.normalizer, train_ds.dev_mean, train_ds.dev_std,
                device, H.get("n_rollout_sessions", 5),
            )
            if rollout_mse is not None:
                history["rollout"].append((epoch, rollout_mse))

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
    return model, full_ds.normalizer, H


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
