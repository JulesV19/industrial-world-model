"""
Entraîne le world model de perçage.

Usage:
    python -m world_model.train
    python -m world_model.train --epochs 200 --lr 1e-3
"""

import argparse
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import DrillDataset
from .model import DrillModel


DEFAULTS = dict(
    corner_embed_dim = 64,
    global_dim       = 256,
    dropout          = 0.1,
    lr               = 3e-4,
    weight_decay     = 1e-4,
    batch_size       = 64,
    epochs           = 150,
    grad_clip        = 5.0,
    val_split        = 0.1,
    seed             = 42,
    lambda_defect    = 0.5,   # poids de la loss BCE défaut
    early_stopping   = 40,
    save_dir         = "world_model/checkpoints",
    data_dir         = "dataset",
)


def get_device() -> torch.device:
    if torch.cuda.is_available():    return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _fmt_time(s: float) -> str:
    s = int(s); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def train(cfg: dict | None = None):
    H = {**DEFAULTS, **(cfg or {})}
    torch.manual_seed(H["seed"])
    device = get_device()
    print(f"Device : {device}")

    episode_paths = sorted(glob.glob(os.path.join(H["data_dir"], "episode_*.npz")))
    print(f"Épisodes : {len(episode_paths)}")
    os.makedirs(H["save_dir"], exist_ok=True)

    full_ds = DrillDataset(episode_paths)
    full_ds.normalizer.save(os.path.join(H["save_dir"], "normalizer.npz"))

    n_val   = max(1, int(len(full_ds) * H["val_split"]))
    n_train = len(full_ds) - n_val
    g       = torch.Generator().manual_seed(H["seed"])
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=H["batch_size"],
                              shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=H["batch_size"],
                              shuffle=False, pin_memory=pin)
    print(f"Train : {n_train}  |  Val : {n_val}")

    model = DrillModel(
        corner_embed_dim = H["corner_embed_dim"],
        global_dim       = H["global_dim"],
        dropout          = H["dropout"],
    ).to(device)
    print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=H["lr"], weight_decay=H["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=H["epochs"], eta_min=1e-5)

    best_val      = float("inf")
    history       = dict(train=[], val=[], lr=[], grad_norm=[])
    epochs_no_imp = 0
    t_start       = time.time()

    epoch_bar = tqdm(range(1, H["epochs"] + 1), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        t_ep = time.time()

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_total = 0.0
        total_gnorm = 0.0

        for corners, speed, offsets_target, defects_target in train_loader:
            corners        = corners.to(device)          # (B, 4, 2)
            speed          = speed.unsqueeze(-1).to(device)  # (B, 1)
            offsets_target = offsets_target.to(device)   # (B, 4, 2)
            defects_target = defects_target.to(device)   # (B, 4)

            optimizer.zero_grad()
            offsets_pred, defect_logits = model(corners, speed)

            loss_offset = F.mse_loss(offsets_pred, offsets_target)
            loss_defect = F.binary_cross_entropy_with_logits(
                defect_logits, defects_target
            )
            loss = loss_offset + H["lambda_defect"] * loss_defect

            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), H["grad_clip"])
            optimizer.step()

            train_total += loss.item()
            total_gnorm += gnorm.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for corners, speed, offsets_target, defects_target in val_loader:
                corners        = corners.to(device)
                speed          = speed.unsqueeze(-1).to(device)
                offsets_target = offsets_target.to(device)
                defects_target = defects_target.to(device)
                offsets_pred, defect_logits = model(corners, speed)
                loss = F.mse_loss(offsets_pred, offsets_target) + \
                       H["lambda_defect"] * F.binary_cross_entropy_with_logits(
                           defect_logits, defects_target)
                val_total += loss.item()

        avg_train = train_total / len(train_loader)
        avg_val   = val_total   / len(val_loader)
        avg_gnorm = total_gnorm / len(train_loader)
        epoch_time = time.time() - t_ep
        elapsed    = time.time() - t_start
        eta        = elapsed / epoch * (H["epochs"] - epoch)

        history["train"    ].append(avg_train)
        history["val"      ].append(avg_val)
        history["lr"       ].append(current_lr)
        history["grad_norm"].append(avg_gnorm)

        improved = avg_val < best_val
        if improved:
            best_val = avg_val
            epochs_no_imp = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "hyperparams": H,
                "val_loss":    avg_val,
                "norm_mean":   full_ds.normalizer.mean,
                "norm_std":    full_ds.normalizer.std,
            }, os.path.join(H["save_dir"], "best_model.pt"))
        else:
            epochs_no_imp += 1

        marker = " ★" if improved else f" ({epochs_no_imp})"
        epoch_bar.set_postfix(val=f"{avg_val:.4f}", best=f"{best_val:.4f}",
                              lr=f"{current_lr:.2e}")
        tqdm.write(
            f"[{epoch:3d}/{H['epochs']}] {epoch_time:5.1f}s  ETA {_fmt_time(eta)}  "
            f"lr={current_lr:.2e}  ‖g‖={avg_gnorm:.2f}  "
            f"train={avg_train:.4f}  val={avg_val:.4f}{marker}"
        )

        if H["early_stopping"] > 0 and epochs_no_imp >= H["early_stopping"]:
            tqdm.write(f"\nEarly stopping après {H['early_stopping']} epochs sans amélioration.")
            break

    print(f"\nDone in {_fmt_time(time.time() - t_start)}.  Best val : {best_val:.4f}")
    _save_curves(history, H["save_dir"])
    return model, full_ds.normalizer, H


def _save_curves(history: dict, save_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train"], label="Train")
    axes[0].plot(history["val"],   label="Val")
    axes[0].set_title("Loss (log)"); axes[0].legend()
    axes[0].set_yscale("log"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["lr"])
    axes[1].set_title("Learning rate")
    axes[1].set_yscale("log"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["grad_norm"])
    axes[2].set_title("Gradient norm"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Training curves → {path}")


def _parse_args():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    return vars(p.parse_args())


if __name__ == "__main__":
    train(cfg=_parse_args())
