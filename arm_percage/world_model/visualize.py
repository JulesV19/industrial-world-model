"""
Visualisation du world model de perçage — animation du bras 2-DOF.

Affiche pour chaque épisode :
  - Gauche  : animation du bras (vrai vs prédit) + cibles coins + trous
  - Milieu  : trajectoire q₁(t) et q₂(t) avec curseur temporel
  - Droite  : distribution des losses MSE + rang

Usage :
    python -m world_model.visualize
    python -m world_model.visualize --split train
    python -m world_model.visualize --episode_idx 5
    python -m world_model.visualize --save_gif ep5.gif
"""

import argparse
import glob
import math
import os
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np
import torch
from torch.utils.data import random_split

from .dataset import DrillDataset, Normalizer, TrajNormalizer
from .model import DrillWorldModel
from .train import DEFAULTS, get_device

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from record_dataset import DRILL_DEFECT_THRESHOLD

CORNER_COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]
CORNER_LABELS = ["Coin 1", "Coin 2", "Coin 3", "Coin 4"]

# ── cinématique directe (bras 2-DOF, L1 = L2 = √2) ───────────────────────────
L1 = L2 = math.sqrt(2)

def fk(q: np.ndarray):
    """q : (T, 2) → base (T,2), elbow (T,2), tip (T,2)"""
    base  = np.zeros((len(q), 2))
    elbow = np.column_stack([L1 * np.cos(q[:, 0]),
                             L1 * np.sin(q[:, 0])])
    tip   = np.column_stack([L1 * np.cos(q[:, 0]) + L2 * np.cos(q[:, 0] + q[:, 1]),
                             L1 * np.sin(q[:, 0]) + L2 * np.sin(q[:, 0] + q[:, 1])])
    return base, elbow, tip


# ── chargement ────────────────────────────────────────────────────────────────

def _load(save_dir: str, device: torch.device):
    ckpt  = torch.load(os.path.join(save_dir, "best_model.pt"),
                       map_location=device, weights_only=False)
    H     = {**DEFAULTS, **ckpt.get("hyperparams", {})}
    model = DrillWorldModel(
        corner_embed_dim = H.get("corner_embed_dim", 64),
        embed_dim        = H.get("embed_dim", 256),
        h_dim            = H.get("h_dim", 512),
        pe_dim           = H.get("pe_dim", 64),
        gru_layers       = H.get("gru_layers", 2),
        n_attn_heads     = H.get("n_attn_heads", 4),
        dropout          = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm      = Normalizer.load(os.path.join(save_dir, "normalizer.npz"))
    traj_norm = TrajNormalizer.load(os.path.join(save_dir, "traj_normalizer.npz"))
    return model, norm, traj_norm, H


def _get_split(save_dir, split, device):
    model, norm, traj_norm, H = _load(save_dir, device)
    episode_paths = sorted(glob.glob(os.path.join(H["data_dir"], "episode_*.npz")))
    full_ds = DrillDataset(episode_paths, normalizer=norm, traj_normalizer=traj_norm)
    n_val   = max(1, int(len(full_ds) * H["val_split"]))
    n_train = len(full_ds) - n_val
    g       = torch.Generator().manual_seed(H["seed"])
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)
    subset  = {"val": val_ds, "train": train_ds}.get(split, full_ds)
    return model, norm, traj_norm, H, full_ds, subset, episode_paths


# ── pré-calcul des losses (distribution) ─────────────────────────────────────

def _compute_losses(model, norm, subset, device):
    """MSE sur les offsets normalisés pour chaque épisode."""
    n_ep   = len(subset)
    losses = []
    print(f"Calcul des losses sur {n_ep} épisodes...")
    for i in range(n_ep):
        item         = subset[i]
        corners_t    = item[0].unsqueeze(0).to(device)
        speed_t      = item[1].unsqueeze(0).unsqueeze(-1).to(device)
        length       = item[3]
        off_norm_tgt = item[4]
        with torch.no_grad():
            _, off_pred, _ = model(corners_t, speed_t, T=length)
        losses.append(((off_pred[0].cpu() - off_norm_tgt) ** 2).mean().item())
        if (i + 1) % 50 == 0 or (i + 1) == n_ep:
            print(f"  {i+1}/{n_ep}", end="\r")
    print("\nDone.")
    return np.array(losses)


# ── inférence complète pour un épisode ────────────────────────────────────────

def _infer_episode(ep_idx, model, norm, traj_norm, full_ds,
                   subset, episode_paths, device):
    item      = subset[ep_idx]
    corners_t = item[0].unsqueeze(0).to(device)
    speed_t   = item[1].unsqueeze(0).unsqueeze(-1).to(device)
    length    = item[3]

    with torch.no_grad():
        traj_pred_t, off_pred_t, defect_logits = model(corners_t, speed_t, T=length)

    traj_pred   = traj_norm.denormalize(traj_pred_t[0].cpu().numpy())   # (T, 2)
    off_pred    = norm.denormalize(off_pred_t[0].cpu().numpy())          # (4, 2)
    defect_prob = torch.sigmoid(defect_logits[0]).cpu().numpy()          # (4,)

    orig_idx    = subset.indices[ep_idx] if hasattr(subset, "indices") else ep_idx
    ep_data     = np.load(episode_paths[orig_idx])
    s           = full_ds.samples[orig_idx]

    traj_real   = ep_data["q_real"].astype(np.float32)        # (T, 2)
    corners     = s["corners"]                                 # (4, 2) Cartésien
    drill_hits  = ep_data["drill_hits"].astype(np.float32)    # (4, 2)
    is_drilling = ep_data["is_drilling"].astype(np.float32)   # (T,)
    speed       = float(s["speed"])

    _, elbow_p, tip_p = fk(traj_pred)
    _, elbow_t, tip_t = fk(traj_real)

    return dict(
        traj_pred=traj_pred, traj_real=traj_real,
        off_pred=off_pred, defect_prob=defect_prob,
        corners=corners, drill_hits=drill_hits, is_drilling=is_drilling,
        speed=speed, length=length,
        elbow_p=elbow_p, elbow_t=elbow_t,
        tip_p=tip_p,     tip_t=tip_t,
    )


# ── browser interactif ────────────────────────────────────────────────────────

def browse(save_dir="world_model/checkpoints", split="val", start_ep=0):
    device = get_device()
    model, norm, traj_norm, H, full_ds, subset, episode_paths = _get_split(
        save_dir, split, device)
    n_ep = len(subset)

    all_losses   = _compute_losses(model, norm, subset, device)
    valid_losses = all_losses[~np.isnan(all_losses)]
    sorted_idx   = np.argsort(valid_losses)

    # ── état mutable partagé ──────────────────────────────────────────────────
    S = dict(ep=-1, fi=0, playing=False, ep_data=None)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs  = fig.add_gridspec(4, 3, left=0.05, right=0.97,
                           top=0.92, bottom=0.23, hspace=0.55, wspace=0.35)
    ax_arm  = fig.add_subplot(gs[:, 0])
    ax_q1   = fig.add_subplot(gs[:2, 1])
    ax_q2   = fig.add_subplot(gs[2:, 1])
    ax_dist = fig.add_subplot(gs[:2, 2])
    ax_rank = fig.add_subplot(gs[2:, 2])

    # ── distribution (statique) ───────────────────────────────────────────────
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
    ax_sl_ep  = fig.add_axes([0.05, 0.13, 0.60, 0.03])
    ax_sl_t   = fig.add_axes([0.05, 0.07, 0.60, 0.03])
    ax_btn_pp  = fig.add_axes([0.68, 0.06, 0.07, 0.07])
    ax_btn_prv = fig.add_axes([0.77, 0.06, 0.07, 0.07])
    ax_btn_nxt = fig.add_axes([0.86, 0.06, 0.07, 0.07])

    sl_ep   = mwidgets.Slider(ax_sl_ep, "Pièce", 0, max(1, n_ep - 1),
                              valinit=start_ep, valstep=1)
    sl_t    = mwidgets.Slider(ax_sl_t,  "Temps", 0, 1, valinit=0, valstep=1)
    btn_pp  = mwidgets.Button(ax_btn_pp,  "▶")
    btn_prv = mwidgets.Button(ax_btn_prv, "◀ ep")
    btn_nxt = mwidgets.Button(ax_btn_nxt, "ep ▶")

    title_text = fig.suptitle("", fontsize=11)

    # ── panneau bras (éléments initialisés vides) ─────────────────────────────
    ax_arm.set_aspect("equal")
    ax_arm.set_xlabel("x [m]", fontsize=8)
    ax_arm.set_ylabel("y [m]", fontsize=8)
    ax_arm.grid(True, alpha=0.2)

    theta_circ = np.linspace(0, 2 * np.pi, 64)

    # Cibles coins (cercle + seuil défaut)
    tgt_markers  = [ax_arm.plot([], [], "o", color=CORNER_COLORS[k], ms=10,
                                markerfacecolor="none", markeredgewidth=2.5,
                                zorder=4, label=f"{CORNER_LABELS[k]}")[0]
                    for k in range(4)]
    thr_circles  = [ax_arm.plot([], [], "--", color=CORNER_COLORS[k],
                                alpha=0.4, lw=0.9, zorder=3)[0] for k in range(4)]
    real_markers = [ax_arm.plot([], [], "s", color=CORNER_COLORS[k],
                                ms=8, zorder=5)[0] for k in range(4)]
    pred_markers = [ax_arm.plot([], [], "^", color="#888888",
                                ms=8, alpha=0.9, zorder=4)[0] for k in range(4)]

    # Tracé complet (estompé en fond)
    full_trail_t, = ax_arm.plot([], [], color="royalblue", lw=0.8, alpha=0.25, zorder=2)
    full_trail_p, = ax_arm.plot([], [], color="tomato",    lw=0.8, alpha=0.25, zorder=2)

    # Bras animé (courant)
    line_true,  = ax_arm.plot([], [], "o-",  color="royalblue",
                              lw=2.5, ms=6, zorder=6, label="vrai")
    line_pred,  = ax_arm.plot([], [], "o--", color="tomato",
                              lw=2.5, ms=6, zorder=6, label="prédit")
    # Trace récente
    trace_true, = ax_arm.plot([], [], color="royalblue", lw=1.5, alpha=0.6, zorder=5)
    trace_pred, = ax_arm.plot([], [], color="tomato",    lw=1.5, alpha=0.6,
                              zorder=5, ls="--")

    time_text    = ax_arm.text(0.02, 0.97, "", transform=ax_arm.transAxes,
                               fontsize=9, va="top")
    drill_text   = ax_arm.text(0.02, 0.90, "", transform=ax_arm.transAxes,
                               fontsize=8, va="top", color="tomato")
    speed_text   = ax_arm.text(0.02, 0.83, "", transform=ax_arm.transAxes,
                               fontsize=8, va="top", color="dimgray")
    ax_arm.plot([], [], "s", color="gray",    ms=8, label="trou réel")
    ax_arm.plot([], [], "^", color="#888888", ms=8, label="trou prédit")
    ax_arm.legend(fontsize=7, loc="upper right", ncol=2)

    # Lignes verticales recréées à chaque chargement d'épisode (ax.cla() les détruit)
    vlines = {"q1": None, "q2": None}

    # ── chargement d'un épisode ───────────────────────────────────────────────
    def _load_ep(ep_idx):
        d = _infer_episode(ep_idx, model, norm, traj_norm, full_ds,
                           subset, episode_paths, device)
        S.update(ep=ep_idx, fi=0, ep_data=d)

        T = d["length"]
        sl_t.eventson = False
        sl_t.valmax   = max(1, T - 1)
        sl_t.ax.set_xlim(0, max(1, T - 1))
        sl_t.set_val(0)
        sl_t.eventson = True

        # Statique : tracés complets en fond
        full_trail_t.set_data(d["tip_t"][:, 0], d["tip_t"][:, 1])
        full_trail_p.set_data(d["tip_p"][:, 0], d["tip_p"][:, 1])

        # Cibles, trous réels, trous prédits
        corners    = d["corners"]
        drill_hits = d["drill_hits"]
        pred_hits  = corners + d["off_pred"]
        for k in range(4):
            tgt_markers[k].set_data([corners[k, 0]], [corners[k, 1]])
            cx = corners[k, 0] + DRILL_DEFECT_THRESHOLD * np.cos(theta_circ)
            cy = corners[k, 1] + DRILL_DEFECT_THRESHOLD * np.sin(theta_circ)
            thr_circles[k].set_data(cx, cy)
            real_markers[k].set_data([drill_hits[k, 0]], [drill_hits[k, 1]])
            pred_markers[k].set_data([pred_hits[k, 0]], [pred_hits[k, 1]])

        # Axes du bras : centré sur le workspace des coins
        all_x = np.concatenate([d["tip_t"][:, 0], d["tip_p"][:, 0]])
        all_y = np.concatenate([d["tip_t"][:, 1], d["tip_p"][:, 1]])
        m = 0.3
        ax_arm.set_xlim(all_x.min() - m, all_x.max() + m)
        ax_arm.set_ylim(all_y.min() - m, all_y.max() + m)
        ax_arm.set_title(f"Bras 2-DOF — [{split}] pièce {ep_idx} / {n_ep-1}",
                         fontsize=10)
        speed_text.set_text(f"Vitesse : {d['speed']:.3f} s/seg")

        # Graphes de trajectoire
        t_ax = np.arange(T)
        ax_q1.cla()
        ax_q1.plot(t_ax, d["traj_real"][:, 0], color="royalblue", lw=1.2, label="réel")
        ax_q1.plot(t_ax, d["traj_pred"][:, 0], color="tomato",    lw=1.2,
                   ls="--", label="prédit", alpha=0.85)
        # Zone de perçage en fond
        _shade_drilling(ax_q1, d["is_drilling"], T)
        ax_q1.set_title("q₁(t)", fontsize=9)
        ax_q1.set_ylabel("rad", fontsize=8)
        ax_q1.legend(fontsize=7, loc="upper right")
        ax_q1.grid(True, alpha=0.25)
        ax_q1.set_xlim(0, T - 1)

        ax_q2.cla()
        ax_q2.plot(t_ax, d["traj_real"][:, 1], color="royalblue", lw=1.2, label="réel")
        ax_q2.plot(t_ax, d["traj_pred"][:, 1], color="tomato",    lw=1.2,
                   ls="--", label="prédit", alpha=0.85)
        _shade_drilling(ax_q2, d["is_drilling"], T)
        ax_q2.set_title("q₂(t)", fontsize=9)
        ax_q2.set_ylabel("rad", fontsize=8)
        ax_q2.set_xlabel("pas de temps", fontsize=8)
        ax_q2.legend(fontsize=7, loc="upper right")
        ax_q2.grid(True, alpha=0.25)
        ax_q2.set_xlim(0, T - 1)

        # Recréer les vlines après cla()
        vlines["q1"] = ax_q1.axvline(0, color="gray", lw=1, ls=":")
        vlines["q2"] = ax_q2.axvline(0, color="gray", lw=1, ls=":")

        # Titre + loss
        ep_loss    = all_losses[ep_idx]
        valid_mask = ~np.isnan(all_losses)
        percentile = float(np.mean(all_losses[valid_mask] <= ep_loss) * 100)
        rank       = int(np.sum(valid_losses <= ep_loss))
        loss_marker_v.set_xdata([ep_loss, ep_loss])
        loss_marker_tx.set_text(f"MSE={ep_loss:.4f}\nPercentile {percentile:.0f}%")
        rank_marker_h.set_ydata([ep_loss, ep_loss])
        rank_marker_pt.set_data([rank], [ep_loss])

        n_def_real = int(d["drill_hits"].shape[0] > 0 and
                         sum(1 for e in
                             np.linalg.norm(d["drill_hits"] - d["corners"], axis=1)
                             if e > DRILL_DEFECT_THRESHOLD))
        n_def_pred = int((d["defect_prob"] > 0.5).sum())
        title_text.set_text(
            f"World model perçage — [{split}]  pièce {ep_idx} / {n_ep-1}  "
            f"({T} pas · {d['speed']:.3f} s/seg)   "
            f"défauts : réel {n_def_real}/4  préd. {n_def_pred}/4"
        )
        drill_text.set_text("")

    def _shade_drilling(ax, is_drilling, T):
        """Fond gris clair sur les zones où is_drilling > 0.5."""
        in_drill = False
        t_start  = 0
        for t in range(T):
            if is_drilling[t] > 0.5 and not in_drill:
                t_start  = t
                in_drill = True
            elif is_drilling[t] <= 0.5 and in_drill:
                ax.axvspan(t_start, t, color="gray", alpha=0.10, zorder=0)
                in_drill = False
        if in_drill:
            ax.axvspan(t_start, T - 1, color="gray", alpha=0.10, zorder=0)

    # ── dessin d'un frame ─────────────────────────────────────────────────────
    def _draw_frame(fi):
        d = S["ep_data"]
        if d is None:
            return
        T  = d["length"]
        fi = max(0, min(fi, T - 1))
        S["fi"] = fi

        line_true.set_data([0, d["elbow_t"][fi, 0], d["tip_t"][fi, 0]],
                           [0, d["elbow_t"][fi, 1], d["tip_t"][fi, 1]])
        line_pred.set_data([0, d["elbow_p"][fi, 0], d["tip_p"][fi, 0]],
                           [0, d["elbow_p"][fi, 1], d["tip_p"][fi, 1]])
        s = max(0, fi - 20)
        trace_true.set_data(d["tip_t"][s:fi, 0], d["tip_t"][s:fi, 1])
        trace_pred.set_data(d["tip_p"][s:fi, 0], d["tip_p"][s:fi, 1])

        time_text.set_text(f"t = {fi * 0.1:.2f} s")
        drill_text.set_text("⚡ perçage" if d["is_drilling"][fi] > 0.5 else "")

        # Curseur temporel sur les graphes q1/q2
        for vl in (vlines["q1"], vlines["q2"]):
            if vl is not None:
                vl.set_xdata([fi, fi])

        sl_t.eventson = False
        sl_t.set_val(fi)
        sl_t.eventson = True

    # ── callbacks ─────────────────────────────────────────────────────────────
    def on_ep_change(val):
        ep = int(round(sl_ep.val))
        if ep == S["ep"]:
            return
        S["playing"] = False
        btn_pp.label.set_text("▶")
        _load_ep(ep)
        _draw_frame(0)
        fig.canvas.draw_idle()

    def on_t_change(val):
        fi = int(round(sl_t.val))
        if fi != S["fi"] and S["ep_data"] is not None:
            S["playing"] = False
            btn_pp.label.set_text("▶")
            _draw_frame(fi)
            fig.canvas.draw_idle()

    def on_play_pause(_):
        S["playing"] = not S["playing"]
        btn_pp.label.set_text("⏸" if S["playing"] else "▶")

    def on_prev(_):
        sl_ep.set_val(max(0, int(sl_ep.val) - 1))

    def on_next(_):
        sl_ep.set_val(min(n_ep - 1, int(sl_ep.val) + 1))

    sl_ep.on_changed(on_ep_change)
    sl_t.on_changed(on_t_change)
    btn_pp.on_clicked(on_play_pause)
    btn_prv.on_clicked(on_prev)
    btn_nxt.on_clicked(on_next)

    # ── timer d'animation (vitesse réelle : 100 ms / pas à 10 Hz) ────────────
    def tick(_):
        if S["playing"] and S["ep_data"] is not None:
            next_fi = S["fi"] + 1
            if next_fi >= S["ep_data"]["length"]:
                next_fi = 0
            _draw_frame(next_fi)
        return []

    anim = animation.FuncAnimation(
        fig, tick, interval=100, blit=False, cache_frame_data=False)

    on_ep_change(start_ep)
    plt.show()
    return anim


# ── sauvegarde GIF / MP4 ──────────────────────────────────────────────────────

def animate(episode_idx=0, save_dir="world_model/checkpoints",
            save_path=None, step=3, split="val"):
    device = get_device()
    model, norm, traj_norm, H, full_ds, subset, episode_paths = _get_split(
        save_dir, split, device)

    episode_idx = episode_idx % len(subset)
    d = _infer_episode(episode_idx, model, norm, traj_norm, full_ds,
                       subset, episode_paths, device)

    T      = d["length"]
    frames = list(range(0, T, step))
    t_ax   = np.arange(T)

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(
        f"World model perçage — [{split}] pièce {episode_idx}  "
        f"({T} pas · {d['speed']:.3f} s/seg)",
        fontsize=11)
    gs = fig.add_gridspec(4, 3, left=0.05, right=0.97,
                          top=0.93, bottom=0.06, hspace=0.55, wspace=0.35)
    ax_arm = fig.add_subplot(gs[:, 0])
    ax_q1  = fig.add_subplot(gs[:2, 1])
    ax_q2  = fig.add_subplot(gs[2:, 1])

    # Bras
    all_x = np.concatenate([d["tip_t"][:, 0], d["tip_p"][:, 0]])
    all_y = np.concatenate([d["tip_t"][:, 1], d["tip_p"][:, 1]])
    m = 0.3
    ax_arm.set_xlim(all_x.min() - m, all_x.max() + m)
    ax_arm.set_ylim(all_y.min() - m, all_y.max() + m)
    ax_arm.set_aspect("equal")
    ax_arm.set_title("Bras 2-DOF"); ax_arm.grid(True, alpha=0.2)
    ax_arm.plot(d["tip_t"][:, 0], d["tip_t"][:, 1],
                color="royalblue", lw=0.8, alpha=0.25)
    ax_arm.plot(d["tip_p"][:, 0], d["tip_p"][:, 1],
                color="tomato",   lw=0.8, alpha=0.25)

    theta_circ = np.linspace(0, 2 * np.pi, 64)
    corners    = d["corners"]
    drill_hits = d["drill_hits"]
    pred_hits  = corners + d["off_pred"]
    for k in range(4):
        ax_arm.plot(*corners[k],    "o", color=CORNER_COLORS[k], ms=10,
                    markerfacecolor="none", markeredgewidth=2.5, zorder=3)
        ax_arm.plot(*drill_hits[k], "s", color=CORNER_COLORS[k], ms=8, zorder=5)
        ax_arm.plot(*pred_hits[k],  "^", color="#888888",         ms=8, zorder=4)
        cx = corners[k, 0] + DRILL_DEFECT_THRESHOLD * np.cos(theta_circ)
        cy = corners[k, 1] + DRILL_DEFECT_THRESHOLD * np.sin(theta_circ)
        ax_arm.plot(cx, cy, "--", color=CORNER_COLORS[k], alpha=0.4, lw=0.9)

    line_true,  = ax_arm.plot([], [], "o-",  color="royalblue",
                              lw=2.5, ms=6, zorder=6, label="vrai")
    line_pred,  = ax_arm.plot([], [], "o--", color="tomato",
                              lw=2.5, ms=6, zorder=6, label="prédit")
    trace_true, = ax_arm.plot([], [], color="royalblue", lw=1.5, alpha=0.6, zorder=5)
    trace_pred, = ax_arm.plot([], [], color="tomato",    lw=1.5, alpha=0.6,
                              zorder=5, ls="--")
    time_text   = ax_arm.text(0.02, 0.97, "", transform=ax_arm.transAxes,
                              fontsize=9, va="top")
    ax_arm.legend(fontsize=8, loc="upper right")

    # Graphes q1, q2
    for ax, col, title in [(ax_q1, 0, "q₁(t)"), (ax_q2, 1, "q₂(t)")]:
        ax.plot(t_ax, d["traj_real"][:, col], color="royalblue", lw=1.2, label="réel")
        ax.plot(t_ax, d["traj_pred"][:, col], color="tomato", lw=1.2,
                ls="--", label="prédit", alpha=0.85)
        in_d, ts = False, 0
        for t in range(T):
            if d["is_drilling"][t] > 0.5 and not in_d:
                ts, in_d = t, True
            elif d["is_drilling"][t] <= 0.5 and in_d:
                ax.axvspan(ts, t, color="gray", alpha=0.10, zorder=0)
                in_d = False
        if in_d:
            ax.axvspan(ts, T - 1, color="gray", alpha=0.10, zorder=0)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("rad", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, T - 1)

    ax_q2.set_xlabel("pas de temps", fontsize=8)
    vl_q1 = ax_q1.axvline(0, color="gray", lw=1, ls=":")
    vl_q2 = ax_q2.axvline(0, color="gray", lw=1, ls=":")

    def update(frame_idx):
        i  = frames[frame_idx]
        s0 = max(0, i - 20)
        line_true.set_data([0, d["elbow_t"][i, 0], d["tip_t"][i, 0]],
                           [0, d["elbow_t"][i, 1], d["tip_t"][i, 1]])
        line_pred.set_data([0, d["elbow_p"][i, 0], d["tip_p"][i, 0]],
                           [0, d["elbow_p"][i, 1], d["tip_p"][i, 1]])
        trace_true.set_data(d["tip_t"][s0:i, 0], d["tip_t"][s0:i, 1])
        trace_pred.set_data(d["tip_p"][s0:i, 0], d["tip_p"][s0:i, 1])
        time_text.set_text(f"t = {i * 0.1:.2f} s")
        vl_q1.set_xdata([i, i])
        vl_q2.set_xdata([i, i])
        return (line_true, line_pred, trace_true, trace_pred,
                time_text, vl_q1, vl_q2)

    interval = max(1, int(step * 100))   # vitesse réelle : 100 ms par pas
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=interval, blit=True)

    if save_path:
        ext    = os.path.splitext(save_path)[1].lower()
        writer = (animation.PillowWriter(fps=10 // step)
                  if ext == ".gif"
                  else animation.FFMpegWriter(fps=10 // step, bitrate=1800))
        anim.save(save_path, writer=writer)
        print(f"Animation sauvegardée → {save_path}")
        plt.close()
    else:
        plt.show()
    return anim


# ── helper Colab ──────────────────────────────────────────────────────────────
def show_in_colab(episode_idx=0, save_dir="world_model/checkpoints",
                  step=3, split="val"):
    from IPython.display import HTML
    a = animate(episode_idx=episode_idx, save_dir=save_dir,
                save_path=None, step=step, split=split)
    plt.close()
    return HTML(a.to_jshtml())


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir",    default=DEFAULTS["save_dir"])
    p.add_argument("--split",       default="val",
                   choices=["val", "train", "all"])
    p.add_argument("--episode_idx", type=int, default=0)
    p.add_argument("--save_gif",    default=None,
                   help="Chemin de sauvegarde (.gif ou .mp4)")
    p.add_argument("--step",        type=int, default=3)
    args = p.parse_args()

    if args.save_gif or args.episode_idx != 0:
        animate(episode_idx=args.episode_idx, save_dir=args.save_dir,
                save_path=args.save_gif, step=args.step, split=args.split)
    else:
        browse(save_dir=args.save_dir, split=args.split, start_ep=args.episode_idx)
