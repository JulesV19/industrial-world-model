"""
Visualisation du world model de perçage.

Affiche pour chaque épisode :
  - Gauche  : workspace 2D (contour découpe, cibles coins, vrais trous, prédits)
  - Milieu  : erreur par coin (réel vs prédit) + défaut (vrai vs prédit)
  - Droite  : distribution des losses MSE + rang

Usage :
    python -m world_model.visualize                  # browser interactif (défaut)
    python -m world_model.visualize --split train
    python -m world_model.visualize --split all
    python -m world_model.visualize --episode_idx 5  # démarre sur cet épisode
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np
import torch
from torch.utils.data import random_split

from .dataset import DrillDataset, Normalizer
from .model import DrillModel
from .train import DEFAULTS, get_device

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from record_dataset import DRILL_DEFECT_THRESHOLD

CORNER_COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]
CORNER_LABELS = ["Coin 1", "Coin 2", "Coin 3", "Coin 4"]


# ── chargement du modèle et du dataset ────────────────────────────────────────

def _load(save_dir: str, device: torch.device):
    ckpt  = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device, weights_only=False)
    H     = {**DEFAULTS, **ckpt.get("hyperparams", {})}
    model = DrillModel(
        corner_embed_dim = H.get("corner_embed_dim", 64),
        global_dim       = H.get("global_dim", 256),
        dropout          = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm = Normalizer.load(os.path.join(save_dir, "normalizer.npz"))
    return model, norm, H


def _get_split(save_dir, split, device):
    model, norm, H = _load(save_dir, device)
    episode_paths  = sorted(glob.glob(os.path.join(H["data_dir"], "episode_*.npz")))
    full_ds = DrillDataset(episode_paths, normalizer=norm)
    n_val   = max(1, int(len(full_ds) * H["val_split"]))
    n_train = len(full_ds) - n_val
    g       = torch.Generator().manual_seed(H["seed"])
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)
    subset  = {"val": val_ds, "train": train_ds}.get(split, full_ds)
    return model, norm, H, full_ds, subset, episode_paths


# ── inférence batch ────────────────────────────────────────────────────────────

def _run_inference_all(model, norm, full_ds, subset, episode_paths, device):
    """Inférence sur tous les épisodes du subset, retourne des arrays pré-calculés."""
    n_ep = len(subset)
    print(f"Calcul des prédictions sur {n_ep} épisodes...")

    all_offsets_pred = []
    all_defect_prob  = []
    all_losses       = []
    all_corners      = []
    all_drill_hits   = []
    all_errors_real  = []
    all_defects_real = []
    all_speeds       = []
    all_cut_x        = []
    all_cut_y        = []

    for i in range(n_ep):
        item             = subset[i]
        corners_t        = item[0].unsqueeze(0).to(device)          # (1, 4, 2)
        speed_scalar     = item[1]
        offsets_norm_tgt = item[2]                                   # (4, 2) tensor
        speed_t = speed_scalar.unsqueeze(0).unsqueeze(-1).to(device) # (1, 1)

        with torch.no_grad():
            offsets_norm_pred, defect_logits = model(corners_t, speed_t)

        offsets_pred_np = norm.denormalize(offsets_norm_pred[0].cpu().numpy())   # (4, 2) m
        defect_prob_np  = torch.sigmoid(defect_logits[0]).cpu().numpy()           # (4,)
        loss = float(torch.mean((offsets_norm_pred[0].cpu() - offsets_norm_tgt) ** 2))

        orig_idx = subset.indices[i] if hasattr(subset, "indices") else i
        ep_data  = np.load(episode_paths[orig_idx])
        s        = full_ds.samples[orig_idx]

        all_offsets_pred.append(offsets_pred_np)
        all_defect_prob.append(defect_prob_np)
        all_losses.append(loss)
        all_corners.append(s["corners"])
        all_drill_hits.append(ep_data["drill_hits"].astype(float))
        all_errors_real.append(ep_data["errors"].astype(float))
        all_defects_real.append(ep_data["defects"].astype(float))
        all_speeds.append(float(s["speed"]))
        all_cut_x.append(ep_data["cut_contour_x"].astype(float))
        all_cut_y.append(ep_data["cut_contour_y"].astype(float))

        if (i + 1) % 20 == 0 or (i + 1) == n_ep:
            print(f"  {i+1}/{n_ep}", end="\r")

    print("\nPrédictions calculées.")
    return dict(
        offsets_pred = np.array(all_offsets_pred),
        defect_prob  = np.array(all_defect_prob),
        losses       = np.array(all_losses),
        corners      = np.array(all_corners),
        drill_hits   = np.array(all_drill_hits),
        errors_real  = np.array(all_errors_real),
        defects_real = np.array(all_defects_real),
        speeds       = np.array(all_speeds),
        cut_x        = all_cut_x,
        cut_y        = all_cut_y,
    )


# ── browser interactif ─────────────────────────────────────────────────────────

def browse(save_dir="world_model/checkpoints", split="val", start_ep=0):
    device = get_device()
    model, norm, H, full_ds, subset, episode_paths = _get_split(save_dir, split, device)
    n_ep   = len(subset)
    data   = _run_inference_all(model, norm, full_ds, subset, episode_paths, device)

    losses       = data["losses"]
    valid_losses = losses[~np.isnan(losses)]
    sorted_idx   = np.argsort(valid_losses)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 9))
    gs  = fig.add_gridspec(4, 3, left=0.05, right=0.97,
                           top=0.92, bottom=0.22, hspace=0.65, wspace=0.35)
    ax_ws   = fig.add_subplot(gs[:, 0])
    ax_err  = fig.add_subplot(gs[:2, 1])
    ax_def  = fig.add_subplot(gs[2:, 1])
    ax_dist = fig.add_subplot(gs[:2, 2])
    ax_rank = fig.add_subplot(gs[2:, 2])

    # ── loss distribution (statique) ──────────────────────────────────────────
    ax_dist.hist(valid_losses, bins=40, color="steelblue", alpha=0.75,
                 edgecolor="white", lw=0.4)
    ax_dist.set_title("Distribution des losses (MSE offset)", fontsize=9)
    ax_dist.set_xlabel("MSE", fontsize=8)
    ax_dist.set_ylabel("# épisodes", fontsize=8)
    ax_dist.grid(True, alpha=0.25)
    loss_marker_v  = ax_dist.axvline(np.nan, color="tomato", lw=2, ls="--",
                                     label="pièce courante")
    loss_marker_tx = ax_dist.text(0.98, 0.95, "", transform=ax_dist.transAxes,
                                  fontsize=8, ha="right", va="top",
                                  color="tomato", fontweight="bold")
    ax_dist.legend(fontsize=7, loc="upper left")

    ax_rank.plot(np.arange(len(valid_losses)), valid_losses[sorted_idx],
                 color="steelblue", lw=1.2)
    ax_rank.set_title("Losses triées", fontsize=9)
    ax_rank.set_xlabel("Rang", fontsize=8)
    ax_rank.set_ylabel("MSE", fontsize=8)
    ax_rank.grid(True, alpha=0.25)
    rank_marker_h  = ax_rank.axhline(np.nan, color="tomato", lw=1.5, ls="--")
    rank_marker_pt, = ax_rank.plot([], [], "o", color="tomato", ms=7, zorder=5)

    # ── widgets ───────────────────────────────────────────────────────────────
    ax_sl   = fig.add_axes([0.05, 0.12, 0.60, 0.03])
    ax_bprv = fig.add_axes([0.68, 0.09, 0.09, 0.07])
    ax_bnxt = fig.add_axes([0.79, 0.09, 0.09, 0.07])

    sl      = mwidgets.Slider(ax_sl, "Pièce", 0, max(1, n_ep - 1),
                              valinit=start_ep, valstep=1)
    btn_prv = mwidgets.Button(ax_bprv, "◀ ep")
    btn_nxt = mwidgets.Button(ax_bnxt, "ep ▶")

    title_text = fig.suptitle("", fontsize=11)

    # ── éléments workspace (initialisés vides) ─────────────────────────────────
    ax_ws.set_aspect("equal")
    ax_ws.set_xlabel("x [m]", fontsize=8)
    ax_ws.set_ylabel("y [m]", fontsize=8)
    ax_ws.grid(True, alpha=0.2)

    theta_circ   = np.linspace(0, 2 * np.pi, 64)
    cut_line,    = ax_ws.plot([], [], ".", color="lightgray", ms=1.5, zorder=1,
                              label="contour découpe")
    tgt_pts      = [ax_ws.plot([], [], "o", color=CORNER_COLORS[k], ms=10,
                               markerfacecolor="none", markeredgewidth=2.5,
                               zorder=3, label=f"{CORNER_LABELS[k]} cible")[0]
                    for k in range(4)]
    real_pts     = [ax_ws.plot([], [], "s", color=CORNER_COLORS[k], ms=8,
                               zorder=5)[0] for k in range(4)]
    pred_pts     = [ax_ws.plot([], [], "^", color="#888888", ms=8, alpha=0.9,
                               zorder=4)[0] for k in range(4)]
    circ_lines   = [ax_ws.plot([], [], "--", color=CORNER_COLORS[k], alpha=0.35,
                               lw=0.9, zorder=2)[0] for k in range(4)]
    ax_ws.plot([], [], "s", color="gray", ms=8, label="réel")
    ax_ws.plot([], [], "^", color="#888888", ms=8, alpha=0.9, label="prédit")
    ax_ws.legend(fontsize=7, loc="upper right", ncol=2)
    speed_text = ax_ws.text(0.02, 0.02, "", transform=ax_ws.transAxes,
                            fontsize=8, va="bottom", color="dimgray")

    def _draw_episode(ep_idx):
        corners   = data["corners"][ep_idx]                        # (4, 2)
        hits      = data["drill_hits"][ep_idx]                     # (4, 2)
        off_pred  = data["offsets_pred"][ep_idx]                   # (4, 2)
        pred_hits = corners + off_pred                             # (4, 2)
        err_real  = data["errors_real"][ep_idx]                    # (4,) m
        err_pred  = np.linalg.norm(off_pred, axis=1)               # (4,) m
        def_real  = data["defects_real"][ep_idx]                   # (4,)
        def_prob  = data["defect_prob"][ep_idx]                    # (4,)
        speed     = data["speeds"][ep_idx]
        cut_x     = data["cut_x"][ep_idx]
        cut_y     = data["cut_y"][ep_idx]

        # Workspace
        cut_line.set_data(cut_x, cut_y)
        for k in range(4):
            tgt_pts[k].set_data([corners[k, 0]], [corners[k, 1]])
            real_pts[k].set_data([hits[k, 0]], [hits[k, 1]])
            pred_pts[k].set_data([pred_hits[k, 0]], [pred_hits[k, 1]])
            cx = corners[k, 0] + DRILL_DEFECT_THRESHOLD * np.cos(theta_circ)
            cy = corners[k, 1] + DRILL_DEFECT_THRESHOLD * np.sin(theta_circ)
            circ_lines[k].set_data(cx, cy)

        all_x  = np.concatenate([corners[:, 0], hits[:, 0], pred_hits[:, 0]])
        all_y  = np.concatenate([corners[:, 1], hits[:, 1], pred_hits[:, 1]])
        margin = 0.12
        ax_ws.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax_ws.set_ylim(all_y.min() - margin, all_y.max() + margin)
        ax_ws.set_title(f"Workspace — pièce {ep_idx}", fontsize=10)
        speed_text.set_text(f"Vitesse : {speed:.3f} s/seg")

        # Erreur par coin
        ax_err.cla()
        x, w = np.arange(4), 0.35
        ax_err.bar(x - w/2, err_real * 1000, width=w,
                   color=CORNER_COLORS, alpha=0.85, label="Réel")
        ax_err.bar(x + w/2, err_pred * 1000, width=w,
                   color="#888888", alpha=0.6, label="Prédit")
        ax_err.axhline(DRILL_DEFECT_THRESHOLD * 1000, color="red", lw=1.2,
                       ls="--", alpha=0.7,
                       label=f"Seuil {DRILL_DEFECT_THRESHOLD*1000:.0f} mm")
        ax_err.set_xticks(x)
        ax_err.set_xticklabels(CORNER_LABELS, fontsize=8)
        ax_err.set_ylabel("Erreur (mm)", fontsize=8)
        ax_err.set_title("Erreur de perçage par coin", fontsize=9)
        ax_err.legend(fontsize=7, loc="upper right")
        ax_err.grid(True, axis="y", alpha=0.3)
        ax_err.set_ylim(bottom=0)

        # Défaut par coin
        ax_def.cla()
        ax_def.bar(x, def_real, width=0.4, color=CORNER_COLORS, alpha=0.85,
                   label="Défaut réel (0/1)")
        ax_def.plot(x, def_prob, "^--", color="#444444", ms=8,
                    label="Prob. préd.")
        ax_def.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax_def.set_xticks(x)
        ax_def.set_xticklabels(CORNER_LABELS, fontsize=8)
        ax_def.set_ylabel("Défaut / probabilité", fontsize=8)
        ax_def.set_ylim(-0.05, 1.15)
        ax_def.set_title("Défauts par coin", fontsize=9)
        ax_def.legend(fontsize=7, loc="upper right")
        ax_def.grid(True, axis="y", alpha=0.3)

        # Loss distribution marker
        ep_loss        = losses[ep_idx]
        valid_mask     = ~np.isnan(losses)
        percentile     = float(np.mean(losses[valid_mask] <= ep_loss) * 100)
        rank_in_sorted = int(np.sum(valid_losses <= ep_loss))
        loss_marker_v.set_xdata([ep_loss, ep_loss])
        loss_marker_tx.set_text(f"MSE={ep_loss:.4f}\nPercentile {percentile:.0f}%")
        rank_marker_h.set_ydata([ep_loss, ep_loss])
        rank_marker_pt.set_data([rank_in_sorted], [ep_loss])

        n_def_real = int(def_real.sum())
        n_def_pred = int((def_prob > 0.5).sum())
        title_text.set_text(
            f"World model perçage — [{split}]  pièce {ep_idx} / {n_ep-1}  "
            f"({speed:.3f} s/seg)   "
            f"défauts : réel {n_def_real}/4  préd. {n_def_pred}/4"
        )
        fig.canvas.draw_idle()

    def on_slider(val):
        _draw_episode(int(round(sl.val)))

    def on_prev(_):
        sl.set_val(max(0, int(sl.val) - 1))

    def on_next(_):
        sl.set_val(min(n_ep - 1, int(sl.val) + 1))

    sl.on_changed(on_slider)
    btn_prv.on_clicked(on_prev)
    btn_nxt.on_clicked(on_next)

    _draw_episode(max(0, min(start_ep, n_ep - 1)))
    plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir",    default=DEFAULTS["save_dir"])
    p.add_argument("--split",       default="val", choices=["val", "train", "all"])
    p.add_argument("--episode_idx", type=int, default=0,
                   help="Indice de départ dans le browser")
    args = p.parse_args()
    browse(save_dir=args.save_dir, split=args.split, start_ep=args.episode_idx)
