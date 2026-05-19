"""
Visualisation du world model — animation du bras 2-DOF.

Affiche côte à côte :
  - Gauche  : animation du bras (vrai vs prédit) + tracé de coupe
  - Droite  : 4 graphes de métriques (q, dq, tau, is_cutting)

Usage :
    python -m world_model.visualize                      # épisode aléatoire du val set
    python -m world_model.visualize --episode_idx 42     # épisode précis (index 0-based)
    python -m world_model.visualize --save_gif ep42.gif  # sauvegarde au lieu d'afficher
"""

import argparse
import glob
import json
import os
import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch.utils.data import random_split

from .dataset import (
    BINARY_IDX, METRIC_KEYS, Normalizer,
    TrajectoryDataset, build_obs_vector, collate_fn,
)
from .model import WorldModel
from .train import DEFAULTS, get_device


# ── cinématique directe ────────────────────────────────────────────────────────
L1, L2 = 1.0, 1.0   # longueurs des segments (depuis config.py)

def fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    q : (T, 2) angles joints
    Retourne base (T,2), coude (T,2), effecteur (T,2)
    """
    base  = np.zeros((len(q), 2))
    elbow = np.column_stack([
        L1 * np.cos(q[:, 0]),
        L1 * np.sin(q[:, 0]),
    ])
    tip = np.column_stack([
        L1 * np.cos(q[:, 0]) + L2 * np.cos(q[:, 0] + q[:, 1]),
        L1 * np.sin(q[:, 0]) + L2 * np.sin(q[:, 0] + q[:, 1]),
    ])
    return base, elbow, tip


# ── labels des métriques ───────────────────────────────────────────────────────
_LABELS = []
for key, dim in METRIC_KEYS:
    for j in range(dim):
        _LABELS.append(f"{key}{'['+str(j)+']' if dim > 1 else ''}")


# ── chargement du modèle ───────────────────────────────────────────────────────
def _load(save_dir: str, device: torch.device):
    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device)
    H    = {**DEFAULTS, **ckpt.get("hyperparams", {})}
    model = WorldModel(
        shape_embed_dim=H["shape_embed_dim"],
        h_dim=H["h_dim"], z_dim=H["z_dim"], obs_dim=H["obs_dim"], dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm = Normalizer.load(os.path.join(save_dir, "normalizer.npz"))
    return model, norm, H


# ── préparation d'un épisode ───────────────────────────────────────────────────
def _prepare_episode(episode_idx: int, save_dir: str, device: torch.device,
                     split: str = "val"):
    """
    split : "val"   → épisodes de validation uniquement
            "train" → épisodes d'entraînement uniquement
            "all"   → tous les épisodes (index sur l'ensemble complet)
    """
    model, norm, H = _load(save_dir, device)

    with open(H["db_path"]) as f:
        piece_db = json.load(f)

    episode_paths = sorted(glob.glob(os.path.join(H["data_dir"], "episode_*.npz")))
    full_ds = TrajectoryDataset(episode_paths, piece_db, normalizer=norm)

    # Reproduire le split train/val
    n_val   = max(1, int(len(full_ds) * H["val_split"]))
    n_train = len(full_ds) - n_val
    g       = torch.Generator().manual_seed(H["seed"])
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    if split == "val":
        subset = val_ds
        split_label = "val"
    elif split == "train":
        subset = train_ds
        split_label = "train"
    else:
        subset = full_ds
        split_label = "all"

    if episode_idx is None:
        episode_idx = random.randint(0, len(subset) - 1)
    episode_idx = episode_idx % len(subset)

    wps_t, obs_norm_t, seq_len = subset[episode_idx]
    wps_t  = wps_t.unsqueeze(0).to(device)
    wp_len = torch.tensor([wps_t.shape[1]]).to(device)
    T      = obs_norm_t.shape[0]

    # Prédiction prior (shape only)
    with torch.no_grad():
        pred_norm, _ = model(wps_t, wp_len, max_len=T)
    pred_norm = pred_norm[0].cpu().numpy()
    tgt_norm  = obs_norm_t.numpy()

    pred = norm.denormalize(pred_norm)
    tgt  = norm.denormalize(tgt_norm)
    pred[:, BINARY_IDX] = 1 / (1 + np.exp(-pred[:, BINARY_IDX]))

    # Waypoints de la pièce
    orig_idx        = subset.indices[episode_idx] if hasattr(subset, "indices") else episode_idx
    piece_waypoints = full_ds.samples[orig_idx]["waypoints"]

    return pred, tgt, piece_waypoints, seq_len, episode_idx, split_label


# ── animation principale ───────────────────────────────────────────────────────
def animate(episode_idx=None, save_dir="world_model/checkpoints",
            save_path=None, step=3, split="val"):
    """
    split : "val" | "train" | "all"
    step  : ne dessiner qu'un frame sur `step` (accélère l'animation).
    """
    device = get_device()
    pred, tgt, waypoints, seq_len, ep_idx, split_label = _prepare_episode(
        episode_idx, save_dir, device, split=split
    )

    # Cinématique
    q_pred = pred[:seq_len, :2]   # q_real prédit
    q_true = tgt [:seq_len, :2]   # q_real vrai

    _, elbow_p, tip_p = fk(q_pred)
    _, elbow_t, tip_t = fk(q_true)

    # Tracé de la forme coupée (waypoints avec is_cutting=True)
    cut_wp = [(x, y) for x, y, c in waypoints if c]
    cut_xy = np.array(cut_wp) if cut_wp else np.empty((0, 2))

    frames = range(0, seq_len, step)

    # ── layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"World model — épisode [{split_label}] #{ep_idx}  (shape only → full trajectory)",
                 fontsize=12)

    gs      = fig.add_gridspec(4, 2, left=0.05, right=0.97,
                               top=0.93, bottom=0.06, hspace=0.55, wspace=0.3)
    ax_arm  = fig.add_subplot(gs[:, 0])
    ax_q1   = fig.add_subplot(gs[0, 1])
    ax_q2   = fig.add_subplot(gs[1, 1])
    ax_tau  = fig.add_subplot(gs[2, 1])
    ax_cut  = fig.add_subplot(gs[3, 1])

    t_axis  = np.arange(seq_len) * 0.01   # secondes

    # ── bras ──────────────────────────────────────────────────────────────────
    arm_lim = L1 + L2 + 0.15
    ax_arm.set_xlim(-arm_lim, arm_lim)
    ax_arm.set_ylim(-arm_lim, arm_lim)
    ax_arm.set_aspect("equal")
    ax_arm.set_title("Bras 2-DOF")
    ax_arm.set_xlabel("x [m]"); ax_arm.set_ylabel("y [m]")
    ax_arm.axhline(0, color="gray", lw=0.5); ax_arm.axvline(0, color="gray", lw=0.5)
    ax_arm.grid(True, alpha=0.2)

    # Forme à couper
    if len(cut_xy):
        ax_arm.fill(cut_xy[:, 0], cut_xy[:, 1],
                    color="gold", alpha=0.25, zorder=1, label="pièce")
        ax_arm.plot(np.append(cut_xy[:, 0], cut_xy[0, 0]),
                    np.append(cut_xy[:, 1], cut_xy[0, 1]),
                    "k--", lw=1, zorder=2)

    # Tracé de l'effecteur (fond)
    ax_arm.plot(tip_t[:, 0], tip_t[:, 1], color="royalblue",
                lw=0.8, alpha=0.3, zorder=3)
    ax_arm.plot(tip_p[:, 0], tip_p[:, 1], color="tomato",
                lw=0.8, alpha=0.3, zorder=3)

    # Éléments animés
    line_true,  = ax_arm.plot([], [], "o-", color="royalblue",
                              lw=2.5, ms=6, zorder=5, label="vrai")
    line_pred,  = ax_arm.plot([], [], "o-", color="tomato",
                              lw=2.5, ms=6, zorder=5, label="prédit", ls="--")
    trace_true, = ax_arm.plot([], [], color="royalblue", lw=1.2, alpha=0.6, zorder=4)
    trace_pred, = ax_arm.plot([], [], color="tomato",    lw=1.2, alpha=0.6, zorder=4,
                              ls="--")
    time_text   = ax_arm.text(0.02, 0.97, "", transform=ax_arm.transAxes,
                              fontsize=9, va="top")
    ax_arm.legend(fontsize=8, loc="upper right")

    # ── graphes métriques ─────────────────────────────────────────────────────
    def _plot_metric(ax, idx_true, idx_pred, ylabel, title):
        ax.plot(t_axis, tgt [:seq_len, idx_true],
                color="royalblue", lw=1.2, label="vrai")
        ax.plot(t_axis, pred[:seq_len, idx_pred],
                color="tomato", lw=1.2, ls="--", label="prédit", alpha=0.85)
        ax.set_title(title, fontsize=9); ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3)
        vline = ax.axvline(0, color="gray", lw=1, ls=":")
        return vline

    vl_q1  = _plot_metric(ax_q1,  0,  0, "rad",   "q₁ réel")
    vl_q2  = _plot_metric(ax_q2,  1,  1, "rad",   "q₂ réel")
    vl_tau = _plot_metric(ax_tau, 12, 12, "N·m",  "τ₁ (couple)")
    ax_cut.plot(t_axis, tgt [:seq_len, BINARY_IDX],
                color="royalblue", lw=1.2, label="vrai")
    ax_cut.plot(t_axis, pred[:seq_len, BINARY_IDX],
                color="tomato", lw=1.2, ls="--", label="prédit", alpha=0.85)
    ax_cut.set_title("is_cutting", fontsize=9)
    ax_cut.set_ylabel("prob", fontsize=8); ax_cut.set_xlabel("t [s]", fontsize=8)
    ax_cut.legend(fontsize=7, loc="upper right"); ax_cut.grid(True, alpha=0.3)
    vl_cut = ax_cut.axvline(0, color="gray", lw=1, ls=":")

    # ── fonction de mise à jour ────────────────────────────────────────────────
    history_len = 120   # points de tracé derrière le bras

    def update(frame_idx):
        i = list(frames)[frame_idx]
        t = i * 0.01

        # Bras vrai
        bx, ey, tx = 0, elbow_t[i, 0], tip_t[i, 0]
        line_true.set_data([0, elbow_t[i, 0], tip_t[i, 0]],
                           [0, elbow_t[i, 1], tip_t[i, 1]])
        # Bras prédit
        line_pred.set_data([0, elbow_p[i, 0], tip_p[i, 0]],
                           [0, elbow_p[i, 1], tip_p[i, 1]])

        # Tracés récents
        start = max(0, i - history_len)
        trace_true.set_data(tip_t[start:i, 0], tip_t[start:i, 1])
        trace_pred.set_data(tip_p[start:i, 0], tip_p[start:i, 1])

        time_text.set_text(f"t = {t:.2f} s")

        # Lignes verticales
        for vl in (vl_q1, vl_q2, vl_tau, vl_cut):
            vl.set_xdata([t, t])

        return line_true, line_pred, trace_true, trace_pred, time_text, \
               vl_q1, vl_q2, vl_tau, vl_cut

    anim = animation.FuncAnimation(
        fig, update, frames=len(list(frames)),
        interval=30, blit=True,
    )

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".gif":
            writer = animation.PillowWriter(fps=30)
        else:
            writer = animation.FFMpegWriter(fps=30, bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Animation sauvegardée → {save_path}")
        plt.close()
    else:
        plt.show()

    return anim


# ── helper Colab (affiche dans le notebook) ───────────────────────────────────
def show_in_colab(episode_idx=None, save_dir="world_model/checkpoints",
                  step=3, split="val"):
    """
    Dans un notebook Colab/Jupyter :
        from world_model.visualize import show_in_colab
        show_in_colab(episode_idx=5, split="train")  # données d'entraînement
        show_in_colab(episode_idx=5, split="val")    # données de validation
        show_in_colab(episode_idx=5, split="all")    # index global
    """
    from IPython.display import HTML
    anim = animate(episode_idx=episode_idx, save_dir=save_dir,
                   save_path=None, step=step, split=split)
    plt.close()
    return HTML(anim.to_jshtml())


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episode_idx", type=int, default=None)
    p.add_argument("--save_dir",    default=DEFAULTS["save_dir"])
    p.add_argument("--save_gif",    default=None,
                   help="Chemin de sauvegarde (.gif ou .mp4)")
    p.add_argument("--step",        type=int, default=3,
                   help="1 frame sur N (3=×3 plus rapide)")
    p.add_argument("--split",       default="val",
                   choices=["val", "train", "all"],
                   help="Sous-ensemble à visualiser")
    args = p.parse_args()
    animate(episode_idx=args.episode_idx, save_dir=args.save_dir,
            save_path=args.save_gif, step=args.step, split=args.split)
