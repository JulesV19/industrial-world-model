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
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import TrajectoryDataset, collate_fn, target_dim
from .model import WorldModel

# ── default hyperparameters ────────────────────────────────────────────────────
DEFAULTS = dict(
    early_stopping  = 40,
    shape_embed_dim = 256,
    h_dim           = 512,
    obs_dim         = 2,        # déterminé automatiquement depuis target_keys
    dropout         = 0.1,
    gru_layers      = 3,
    pe_dim          = 64,
    lr              = 3e-4,
    weight_decay    = 1e-4,
    batch_size      = 32,
    epochs          = 100,
    grad_clip       = 5.0,
    val_split       = 0.1,
    seed            = 42,
    num_workers     = 0,        # 0 = optimal : dataset entièrement pré-chargé en RAM
    subsample_factor= 1,        # sous-échantillonnage temporel (ex: 10 = 1 step sur 10)
    use_amp         = True,     # mixed precision BF16 (A100/T4 uniquement)
    target_keys     = "q_real",  # signaux à prédire (séparés par virgule)
    save_dir        = "world_model/checkpoints",
    data_dir        = "dataset",
    db_path         = "pieces_database.json",
)


# ── loss ───────────────────────────────────────────────────────────────────────
def compute_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    seq_lengths: torch.Tensor,
):
    """MSE masqué sur tous les signaux cibles (continus normalisés)."""
    B, T, D = preds.shape
    device  = preds.device
    mask    = (torch.arange(T, device=device)[None, :] < seq_lengths[:, None]).float()
    n       = mask.sum() * D + 1e-8
    mse     = ((preds - targets).pow(2) * mask[..., None]).sum() / n
    return mse


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

    episode_paths = sorted(glob.glob(os.path.join(H["data_dir"], "episode_*.npz")))
    print(f"Episodes : {len(episode_paths)}")

    os.makedirs(H["save_dir"], exist_ok=True)

    # Signaux cibles : liste depuis la chaîne CLI "q_real" ou "q_real,q_des" etc.
    raw_keys = H.get("target_keys", "q_real")
    target_keys = [k.strip() for k in raw_keys.split(",")] if isinstance(raw_keys, str) else raw_keys
    obs_dim = target_dim(target_keys)
    H["obs_dim"]     = obs_dim
    H["target_keys"] = target_keys
    print(f"Signaux cibles : {target_keys}  →  obs_dim={obs_dim}")

    full_ds = TrajectoryDataset(episode_paths, piece_db, target_keys=target_keys)
    full_ds.normalizer.save(os.path.join(H["save_dir"], "normalizer.npz"))

    n_val   = max(1, int(len(full_ds) * H["val_split"]))
    n_train = len(full_ds) - n_val
    g       = torch.Generator().manual_seed(H["seed"])
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

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
    print(f"Train : {n_train}  |  Val : {n_val}")

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
    history       = dict(train=[], val=[], lr=[], grad_norm=[])
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

        for wps, wp_len, speed, obs, seq_len in train_loader:
            wps, wp_len  = wps.to(device), wp_len.to(device)
            speed        = speed.to(device)
            obs, seq_len = obs.to(device), seq_len.to(device)

            # Sous-échantillonnage temporel : réduit T avant le GRU
            if sf > 1:
                obs     = obs[:, ::sf, :].contiguous()
                seq_len = (seq_len + sf - 1) // sf

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                preds, _ = model(wps, wp_len, speed, targets=obs)
                loss     = compute_loss(preds, obs, seq_len)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), H["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            train_total     += loss.item()
            total_grad_norm += grad_norm.item()


        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # --- validate ---
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for wps, wp_len, speed, obs, seq_len in val_loader:
                wps, wp_len  = wps.to(device), wp_len.to(device)
                speed        = speed.to(device)
                obs, seq_len = obs.to(device), seq_len.to(device)
                if sf > 1:
                    obs     = obs[:, ::sf, :].contiguous()
                    seq_len = (seq_len + sf - 1) // sf
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    preds, _ = model(wps, wp_len, speed, targets=obs)
                    val_total += compute_loss(preds, obs, seq_len).item()

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
                },
                os.path.join(H["save_dir"], "best_model.pt"),
            )
        else:
            epochs_no_imp += 1

        marker = " ★" if improved else f" ({epochs_no_imp})"
        epoch_bar.set_postfix(
            val=f"{avg_val:.4f}", best=f"{best_val:.4f}", lr=f"{current_lr:.2e}"
        )
        tqdm.write(
            f"[{epoch:3d}/{H['epochs']}] {epoch_time:5.1f}s  ETA {_fmt_time(eta)}  "
            f"lr={current_lr:.2e}  ‖g‖={avg_grad_norm:.2f}  "
            f"train={avg_train:.4f}  val={avg_val:.4f}{marker}"
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

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
    return vars(p.parse_args())


if __name__ == "__main__":
    train(cfg=_parse_args())
