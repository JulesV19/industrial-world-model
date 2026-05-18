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

from .dataset import BINARY_IDX, CONTINUOUS_SLICE, TrajectoryDataset, collate_fn
from .model import WorldModel

# ── default hyperparameters ────────────────────────────────────────────────────
DEFAULTS = dict(
    shape_embed_dim = 128,
    h_dim           = 256,
    z_dim           = 32,
    obs_dim         = 15,
    dropout         = 0.1,
    lr              = 3e-4,
    weight_decay    = 1e-5,
    beta_kl         = 0.5,      # KL weight at full warm-up
    kl_warmup       = 30,       # epochs to ramp KL from 0 to beta_kl
    batch_size      = 16,
    epochs          = 150,
    grad_clip       = 10.0,
    val_split       = 0.1,
    seed            = 42,
    save_dir        = "world_model/checkpoints",
    data_dir        = "dataset",
    db_path         = "pieces_database.json",
)


# ── loss ───────────────────────────────────────────────────────────────────────
def compute_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    seq_lengths: torch.Tensor,
    kl_loss: torch.Tensor,
    beta: float,
):
    """
    preds       : (B, T, 15) — raw decoder output
    targets     : (B, T, 15) — normalised observations
    seq_lengths : (B,)        — actual lengths (rest is padding)
    """
    B, T, _ = preds.shape
    device   = preds.device

    mask = (torch.arange(T, device=device)[None, :] < seq_lengths[:, None]).float()
    n    = mask.sum() + 1e-8

    # Continuous signals — MSE in normalised space
    mse = ((preds[..., CONTINUOUS_SLICE] - targets[..., CONTINUOUS_SLICE]).pow(2)
           * mask[..., None]).sum() / (n * 14)

    # is_cutting — BCE with logits (targets are already 0 / 1)
    bce = nn.functional.binary_cross_entropy_with_logits(
        preds[..., BINARY_IDX],
        targets[..., BINARY_IDX],
        reduction="none",
    )
    bce = (bce * mask).sum() / n

    total = mse + bce + beta * kl_loss
    return total, mse, bce, kl_loss


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
        piece_db = json.load(f)

    episode_paths = sorted(glob.glob(os.path.join(H["data_dir"], "episode_*.npz")))
    print(f"Episodes : {len(episode_paths)}")

    os.makedirs(H["save_dir"], exist_ok=True)

    full_ds = TrajectoryDataset(episode_paths, piece_db)
    full_ds.normalizer.save(os.path.join(H["save_dir"], "normalizer.npz"))

    n_val   = max(1, int(len(full_ds) * H["val_split"]))
    n_train = len(full_ds) - n_val
    g       = torch.Generator().manual_seed(H["seed"])
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds, batch_size=H["batch_size"],
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=H["batch_size"],
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    print(f"Train : {n_train}  |  Val : {n_val}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = WorldModel(
        shape_embed_dim = H["shape_embed_dim"],
        h_dim           = H["h_dim"],
        z_dim           = H["z_dim"],
        obs_dim         = H["obs_dim"],
        dropout         = H["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=H["lr"], weight_decay=H["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=H["epochs"], eta_min=1e-5)

    best_val      = float("inf")
    history       = dict(train=[], val=[], mse=[], bce=[], kl=[], lr=[], grad_norm=[])
    epochs_no_imp = 0
    t_start       = time.time()

    # ── epoch loop ────────────────────────────────────────────────────────────
    epoch_bar = tqdm(range(1, H["epochs"] + 1), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        beta = H["beta_kl"] * min(1.0, epoch / max(1, H["kl_warmup"]))
        t_epoch = time.time()

        # --- train ---
        model.train()
        train_total = train_mse = train_bce = train_kl = 0.0
        total_grad_norm = 0.0

        batch_bar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch:3d} train",
            leave=False,
            unit="batch",
        )
        for wps, wp_len, obs, seq_len in batch_bar:
            wps, wp_len  = wps.to(device), wp_len.to(device)
            obs, seq_len = obs.to(device), seq_len.to(device)

            optimizer.zero_grad()
            preds, kl = model(wps, wp_len, targets=obs)
            loss, mse, bce, kl_val = compute_loss(preds, obs, seq_len, kl, beta)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), H["grad_clip"])
            optimizer.step()

            train_total    += loss.item()
            train_mse      += mse.item()
            train_bce      += bce.item()
            train_kl       += kl_val.item()
            total_grad_norm += grad_norm.item()

            batch_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                mse=f"{mse.item():.4f}",
                bce=f"{bce.item():.4f}",
                kl=f"{kl_val.item():.4f}",
            )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # --- validate ---
        model.eval()
        val_total = val_mse = val_bce = val_kl_acc = 0.0
        with torch.no_grad():
            for wps, wp_len, obs, seq_len in val_loader:
                wps, wp_len  = wps.to(device), wp_len.to(device)
                obs, seq_len = obs.to(device), seq_len.to(device)
                preds, kl = model(wps, wp_len, targets=obs)
                loss, mse, bce, kl_val = compute_loss(preds, obs, seq_len, kl, beta)
                val_total  += loss.item()
                val_mse    += mse.item()
                val_bce    += bce.item()
                val_kl_acc += kl_val.item()

        n_train_b = len(train_loader)
        n_val_b   = len(val_loader)

        avg_train     = train_total    / n_train_b
        avg_train_mse = train_mse      / n_train_b
        avg_train_bce = train_bce      / n_train_b
        avg_train_kl  = train_kl       / n_train_b
        avg_grad_norm = total_grad_norm / n_train_b
        avg_val       = val_total      / n_val_b
        avg_mse       = val_mse        / n_val_b
        avg_bce       = val_bce        / n_val_b
        avg_kl        = val_kl_acc     / n_val_b
        epoch_time    = time.time() - t_epoch
        elapsed       = time.time() - t_start
        eta           = elapsed / epoch * (H["epochs"] - epoch)

        history["train"    ].append(avg_train)
        history["val"      ].append(avg_val)
        history["mse"      ].append(avg_mse)
        history["bce"      ].append(avg_bce)
        history["kl"       ].append(avg_kl)
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
                    "val_loss": best_val,
                },
                os.path.join(H["save_dir"], "best_model.pt"),
            )
        else:
            epochs_no_imp += 1

        # Update outer bar suffix with key metrics
        epoch_bar.set_postfix(
            val=f"{avg_val:.4f}",
            best=f"{best_val:.4f}",
            mse=f"{avg_mse:.4f}",
            bce=f"{avg_bce:.4f}",
            kl=f"{avg_kl:.4f}",
            lr=f"{current_lr:.2e}",
        )

        # Detailed line every epoch (below the bars)
        marker = " ★" if improved else f" ({epochs_no_imp})"
        tqdm.write(
            f"[{epoch:3d}/{H['epochs']}] {epoch_time:5.1f}s  ETA {_fmt_time(eta)}  "
            f"β={beta:.3f}  lr={current_lr:.2e}  ‖g‖={avg_grad_norm:.2f}\n"
            f"         train  loss={avg_train:.4f}  mse={avg_train_mse:.4f}  "
            f"bce={avg_train_bce:.4f}  kl={avg_train_kl:.4f}\n"
            f"         val    loss={avg_val:.4f}  mse={avg_mse:.4f}  "
            f"bce={avg_bce:.4f}  kl={avg_kl:.4f}{marker}"
        )

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
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes[0, 0]
    ax.plot(history["train"], label="Train")
    ax.plot(history["val"],   label="Val")
    ax.set_title("Total loss (log)"); ax.legend(); ax.set_xlabel("Epoch")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(history["mse"], label="MSE")
    ax.plot(history["bce"], label="BCE")
    ax.plot(history["kl"],  label="KL")
    ax.set_title("Val loss components"); ax.legend(); ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(history["lr"])
    ax.set_title("Learning rate"); ax.set_xlabel("Epoch")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
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
        p.add_argument(f"--{k}", type=type(v), default=v)
    return vars(p.parse_args())


if __name__ == "__main__":
    train(cfg=_parse_args())
