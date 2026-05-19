"""
Evaluate a trained world model and visualise predictions.

Usage:
    python -m world_model.evaluate                        # evaluate all val episodes
    python -m world_model.evaluate --episode_idx 42       # plot one specific episode
    python -m world_model.evaluate --n_samples 5          # plot 5 random val episodes
"""

import argparse
import glob
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .dataset import (
    BINARY_IDX,
    CONTINUOUS_SLICE,
    METRIC_KEYS,
    OBS_DIM,
    Normalizer,
    TrajectoryDataset,
    collate_fn,
)
from .model import WorldModel
from .train import DEFAULTS, get_device


# ── signal labels (14 continuous + 1 binary) ──────────────────────────────────
_LABELS = []
for key, dim in METRIC_KEYS:
    for j in range(dim):
        suffix = f"[{j}]" if dim > 1 else ""
        _LABELS.append(f"{key}{suffix}")   # e.g. "q_real[0]", "is_cutting"


# ── load checkpoint ───────────────────────────────────────────────────────────
def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    H    = {**DEFAULTS, **ckpt.get("hyperparams", {})}
    model = WorldModel(
        shape_embed_dim = H["shape_embed_dim"],
        h_dim           = H["h_dim"],
        z_dim           = H["z_dim"],
        obs_dim         = H["obs_dim"],
        dropout         = 0.0,
        free_bits       = H.get("free_bits", 0.5),
        gru_layers      = H.get("gru_layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")
    return model, H


# ── per-metric MSE in original (denormalised) units ───────────────────────────
def compute_metrics(
    preds_norm: np.ndarray,         # (T, 15)
    targets_norm: np.ndarray,       # (T, 15)
    seq_len: int,
    normalizer: Normalizer,
) -> dict:
    p = normalizer.denormalize(preds_norm[:seq_len])
    t = normalizer.denormalize(targets_norm[:seq_len])

    # Apply sigmoid to is_cutting prediction
    p[:, BINARY_IDX] = 1 / (1 + np.exp(-p[:, BINARY_IDX]))

    rmse = {}
    for i, label in enumerate(_LABELS):
        rmse[label] = float(np.sqrt(np.mean((p[:, i] - t[:, i]) ** 2)))
    return rmse


# ── plot one episode ───────────────────────────────────────────────────────────
def plot_episode(
    pred_norm: np.ndarray,
    tgt_norm:  np.ndarray,
    seq_len:   int,
    normalizer: Normalizer,
    episode_idx: int,
    save_path: str | None = None,
):
    pred = normalizer.denormalize(pred_norm[:seq_len])
    tgt  = normalizer.denormalize(tgt_norm [:seq_len])
    pred[:, BINARY_IDX] = 1 / (1 + np.exp(-pred[:, BINARY_IDX]))

    t = np.arange(seq_len) * 0.01   # seconds (100 Hz)

    # Group signals for plotting
    groups = [
        ("Joint angles — real  [rad]",    ["q_real[0]",    "q_real[1]"   ]),
        ("Joint angles — sensed [rad]",   ["q_sensed[0]",  "q_sensed[1]" ]),
        ("Joint angles — desired [rad]",  ["q_des[0]",     "q_des[1]"    ]),
        ("Joint vel — real  [rad/s]",     ["dq_real[0]",   "dq_real[1]"  ]),
        ("Joint vel — sensed [rad/s]",    ["dq_sensed[0]", "dq_sensed[1]"]),
        ("Joint vel — desired [rad/s]",   ["dq_des[0]",    "dq_des[1]"   ]),
        ("Torques [N·m]",                 ["tau[0]",        "tau[1]"      ]),
        ("Is cutting",                    ["is_cutting"                   ]),
    ]

    n_rows = len(groups)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, n_rows * 2.2), sharex=True)
    fig.suptitle(f"Episode {episode_idx + 1:03d} — World model prediction", fontsize=13)

    for ax, (title, signals) in zip(axes, groups):
        for sig in signals:
            i = _LABELS.index(sig)
            ax.plot(t, tgt [:, i], lw=1.2, label=f"{sig} (true)")
            ax.plot(t, pred[:, i], lw=1.2, ls="--", label=f"{sig} (pred)")
        ax.set_ylabel(title, fontsize=8)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved → {save_path}")
    else:
        plt.show()


# ── main evaluation ────────────────────────────────────────────────────────────
def evaluate(args: argparse.Namespace):
    device     = get_device()
    save_dir   = args.save_dir
    ckpt_path  = os.path.join(save_dir, "best_model.pt")
    norm_path  = os.path.join(save_dir, "normalizer.npz")

    model, H = load_model(ckpt_path, device)
    normalizer = Normalizer.load(norm_path)

    with open(H["db_path"]) as f:
        piece_db = json.load(f)["pieces"]

    episode_paths = sorted(glob.glob(os.path.join(H["data_dir"], "episode_*.npz")))
    full_ds = TrajectoryDataset(episode_paths, piece_db, normalizer=normalizer)

    # Reproduce the same train/val split as in training
    n_val   = max(1, int(len(full_ds) * H["val_split"]))
    n_train = len(full_ds) - n_val
    g       = torch.Generator().manual_seed(H["seed"])
    _, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    val_loader = DataLoader(
        val_ds, batch_size=1,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    all_rmse: list[dict] = []
    plot_indices = set()

    if args.episode_idx is not None:
        plot_indices.add(args.episode_idx)
    elif args.n_samples > 0:
        plot_indices = set(random.sample(range(len(val_ds)), min(args.n_samples, len(val_ds))))

    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)

    with torch.no_grad():
        for i, (wps, wp_len, speed, obs, seq_len) in enumerate(val_loader):
            wps, wp_len = wps.to(device), wp_len.to(device)
            speed       = speed.to(device)
            T = obs.shape[1]
            preds, _ = model(wps, wp_len, speed, max_len=T)

            pred_np = preds[0].cpu().numpy()    # (T, 15)
            tgt_np  = obs  [0].numpy()          # (T, 15)
            L       = int(seq_len[0])

            rmse = compute_metrics(pred_np, tgt_np, L, normalizer)
            all_rmse.append(rmse)

            if i in plot_indices:
                orig_idx = val_ds.indices[i] if hasattr(val_ds, "indices") else i
                plot_episode(
                    pred_np, tgt_np, L, normalizer,
                    episode_idx=orig_idx,
                    save_path=os.path.join(save_dir, "plots", f"episode_{orig_idx+1:03d}.png"),
                )

    # ── aggregate RMSE report ──────────────────────────────────────────────
    print("\n── RMSE on validation set (original units) ──")
    all_keys = _LABELS
    for key in all_keys:
        vals = [r[key] for r in all_rmse]
        print(f"  {key:<18s}  mean {np.mean(vals):.4f}   std {np.std(vals):.4f}")

    _save_rmse_bar(all_rmse, os.path.join(save_dir, "plots", "rmse_summary.png"))


def _save_rmse_bar(all_rmse: list[dict], path: str):
    labels = _LABELS
    means  = [np.mean([r[k] for r in all_rmse]) for k in labels]
    stds   = [np.std ([r[k] for r in all_rmse]) for k in labels]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_title("RMSE per signal — validation set (original units)")
    ax.set_ylabel("RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"RMSE bar chart → {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir",     default=DEFAULTS["save_dir"])
    p.add_argument("--episode_idx",  type=int, default=None,
                   help="If set, plot this specific val episode (0-indexed)")
    p.add_argument("--n_samples",    type=int, default=3,
                   help="Number of random val episodes to plot")
    evaluate(p.parse_args())
