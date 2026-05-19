"""
Visualisation du world model — animation du bras 2-DOF.

Affiche côte à côte :
  - Gauche  : animation du bras (vrai vs prédit) + tracé de coupe
  - Droite  : 4 graphes de métriques (q, dq, tau, is_cutting)

Usage :
    python -m world_model.visualize                        # browser interactif (défaut)
    python -m world_model.visualize --split train          # sur le train set
    python -m world_model.visualize --episode_idx 42       # épisode précis (animation fixe)
    python -m world_model.visualize --save_gif ep42.gif    # sauvegarde au lieu d'afficher
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
    TrajectoryDataset, build_obs_vector, collate_fn, target_dim,
)
from .model import WorldModel
from .train import DEFAULTS, get_device


# ── cinématique directe ────────────────────────────────────────────────────────
L1, L2 = 1.0, 1.0

def fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base  = np.zeros((len(q), 2))
    elbow = np.column_stack([L1 * np.cos(q[:, 0]),
                             L1 * np.sin(q[:, 0])])
    tip   = np.column_stack([L1 * np.cos(q[:, 0]) + L2 * np.cos(q[:, 0] + q[:, 1]),
                             L1 * np.sin(q[:, 0]) + L2 * np.sin(q[:, 0] + q[:, 1])])
    return base, elbow, tip


# ── labels des métriques ───────────────────────────────────────────────────────
_LABELS = []
for key, dim in METRIC_KEYS:
    for j in range(dim):
        _LABELS.append(f"{key}{'['+str(j)+']' if dim > 1 else ''}")


# ── chargement du modèle ───────────────────────────────────────────────────────
def _load(save_dir: str, device: torch.device):
    ckpt  = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device)
    H     = {**DEFAULTS, **ckpt.get("hyperparams", {})}
    model = WorldModel(
        shape_embed_dim = H.get("shape_embed_dim", 256),
        h_dim           = H.get("h_dim", 512),
        obs_dim         = H.get("obs_dim", 2),
        dropout         = 0.0,
        gru_layers      = H.get("gru_layers", 3),
        pe_dim          = H.get("pe_dim", 64),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm = Normalizer.load(os.path.join(save_dir, "normalizer.npz"))
    return model, norm, H


# ── helpers dataset ────────────────────────────────────────────────────────────
def _get_split(save_dir, split, device):
    """Retourne (model, norm, H, full_ds, subset) pour le split demandé."""
    model, norm, H = _load(save_dir, device)
    with open(H["db_path"]) as f:
        piece_db = json.load(f)
    episode_paths = sorted(glob.glob(os.path.join(H["data_dir"], "episode_*.npz")))
    full_ds = TrajectoryDataset(episode_paths, piece_db, normalizer=norm,
                                target_keys=H.get("target_keys", None))
    n_val   = max(1, int(len(full_ds) * H["val_split"]))
    n_train = len(full_ds) - n_val
    g       = torch.Generator().manual_seed(H["seed"])
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)
    subset  = {"val": val_ds, "train": train_ds}.get(split, full_ds)
    return model, norm, H, full_ds, subset


def _run_inference(model, norm, H, full_ds, subset, ep_idx, device):
    """Inférence sur un épisode → (pred, tgt, waypoints, seq_len, q_col)."""
    wps_t, obs_norm_t, seq_len = subset[ep_idx]
    wps_t  = wps_t.unsqueeze(0).to(device)
    wp_len = torch.tensor([wps_t.shape[1]]).to(device)
    T      = obs_norm_t.shape[0]

    with torch.no_grad():
        pred_norm, _ = model(wps_t, wp_len, max_len=T)
    pred_norm = pred_norm[0].cpu().numpy()
    tgt_norm  = obs_norm_t.numpy()

    pred = norm.denormalize(pred_norm)
    tgt  = norm.denormalize(tgt_norm)

    target_keys = H.get("target_keys") or [k for k, _ in METRIC_KEYS]
    if isinstance(target_keys, str):
        target_keys = [k.strip() for k in target_keys.split(",")]

    if "is_cutting" in target_keys:
        idx_cut = target_keys.index("is_cutting")
        pred[:, idx_cut] = 1 / (1 + np.exp(-pred[:, idx_cut]))
    elif not target_keys:
        pred[:, BINARY_IDX] = 1 / (1 + np.exp(-pred[:, BINARY_IDX]))

    orig_idx   = subset.indices[ep_idx] if hasattr(subset, "indices") else ep_idx
    waypoints  = full_ds.samples[orig_idx]["waypoints"]

    col, q_col = 0, None
    for k in target_keys:
        d = dict(METRIC_KEYS)[k]
        if k in ("q_des", "q_real"):
            q_col = col
            break
        col += d
    if q_col is None:
        raise ValueError(f"Aucun signal de position (q_des/q_real) dans target_keys={target_keys}")

    return pred, tgt, waypoints, seq_len, q_col, target_keys


# ── browser interactif ─────────────────────────────────────────────────────────
def browse(save_dir="world_model/checkpoints", split="val", step=1):
    """
    Browser interactif :
      • Slider « Pièce »  : navigue entre tous les épisodes du split
      • Slider « Temps »  : scrub manuel dans la trajectoire
      • Bouton ▶/⏸       : lecture automatique
      • Boutons ◀ / ▶     : épisode précédent / suivant
    """
    import matplotlib.widgets as mwidgets

    device = get_device()
    model, norm, H, full_ds, subset = _get_split(save_dir, split, device)
    n_ep   = len(subset)

    _UNITS = {"q_real":"rad","q_des":"rad","q_sensed":"rad",
              "dq_real":"rad/s","dq_des":"rad/s","dq_sensed":"rad/s",
              "tau":"N·m","is_cutting":"prob"}

    # ── état mutable ──────────────────────────────────────────────────────────
    S = dict(ep=-1, fi=0, playing=False,
             pred=None, tgt=None, waypoints=None, seq_len=None,
             frames=None, target_keys=None,
             elbow_t=None, elbow_p=None, tip_t=None, tip_p=None)

    # ── figure & axes ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(4, 2, left=0.05, right=0.97,
                           top=0.92, bottom=0.23, hspace=0.55, wspace=0.30)
    ax_arm      = fig.add_subplot(gs[:, 0])
    metric_axes = [fig.add_subplot(gs[i, 1]) for i in range(4)]

    # Axes widgets
    ax_sl_ep  = fig.add_axes([0.07, 0.13, 0.60, 0.03])
    ax_sl_t   = fig.add_axes([0.07, 0.07, 0.60, 0.03])
    ax_btn_pp = fig.add_axes([0.70, 0.06, 0.07, 0.07])
    ax_btn_prv= fig.add_axes([0.79, 0.06, 0.07, 0.07])
    ax_btn_nxt= fig.add_axes([0.88, 0.06, 0.07, 0.07])

    sl_ep   = mwidgets.Slider(ax_sl_ep,  "Pièce", 0, max(1, n_ep - 1),
                              valinit=0, valstep=1)
    sl_t    = mwidgets.Slider(ax_sl_t,   "Temps", 0, 1,
                              valinit=0, valstep=1)
    btn_pp  = mwidgets.Button(ax_btn_pp,  "▶")
    btn_prv = mwidgets.Button(ax_btn_prv, "◀ ep")
    btn_nxt = mwidgets.Button(ax_btn_nxt, "ep ▶")

    title_text = fig.suptitle("", fontsize=11)

    # ── bras : artists statiques & animés ─────────────────────────────────────
    arm_lim = L1 + L2 + 0.15
    ax_arm.set_xlim(-arm_lim, arm_lim)
    ax_arm.set_ylim(-arm_lim, arm_lim)
    ax_arm.set_aspect("equal")
    ax_arm.set_xlabel("x [m]"); ax_arm.set_ylabel("y [m]")
    ax_arm.axhline(0, color="gray", lw=0.5)
    ax_arm.axvline(0, color="gray", lw=0.5)
    ax_arm.grid(True, alpha=0.2)

    piece_poly    = mpatches.Polygon(np.zeros((3, 2)), closed=True,
                                     color="gold", alpha=0.25, zorder=1)
    ax_arm.add_patch(piece_poly)
    piece_outline, = ax_arm.plot([], [], "k--", lw=1, zorder=2, label="pièce")
    full_trail_t,  = ax_arm.plot([], [], color="royalblue", lw=0.8, alpha=0.3, zorder=3)
    full_trail_p,  = ax_arm.plot([], [], color="tomato",    lw=0.8, alpha=0.3, zorder=3)
    line_true,     = ax_arm.plot([], [], "o-", color="royalblue",
                                 lw=2.5, ms=6, zorder=5, label="vrai")
    line_pred,     = ax_arm.plot([], [], "o-", color="tomato",
                                 lw=2.5, ms=6, zorder=5, label="prédit", ls="--")
    trace_true,    = ax_arm.plot([], [], color="royalblue", lw=1.2, alpha=0.6, zorder=4)
    trace_pred,    = ax_arm.plot([], [], color="tomato",    lw=1.2, alpha=0.6,
                                 zorder=4, ls="--")
    time_text      = ax_arm.text(0.02, 0.97, "", transform=ax_arm.transAxes,
                                 fontsize=9, va="top")
    ax_arm.legend(fontsize=8, loc="upper right")

    m_vlines = []   # lignes verticales des graphes métriques (reconstruites par épisode)

    # ── chargement d'un épisode ───────────────────────────────────────────────
    def _load_ep(ep_idx):
        pred, tgt, waypoints, seq_len, q_col, target_keys = _run_inference(
            model, norm, H, full_ds, subset, ep_idx, device)
        q_pred = pred[:seq_len, q_col:q_col + 2]
        q_true = tgt [:seq_len, q_col:q_col + 2]
        _, elbow_p, tip_p = fk(q_pred)
        _, elbow_t, tip_t = fk(q_true)
        S.update(ep=ep_idx, fi=0, pred=pred, tgt=tgt,
                 waypoints=waypoints, seq_len=seq_len, target_keys=target_keys,
                 frames=list(range(0, seq_len, step)),
                 elbow_t=elbow_t, elbow_p=elbow_p, tip_t=tip_t, tip_p=tip_p)

    def _refresh_static():
        wp  = S["waypoints"]
        cut = np.array([(x, y) for x, y, c in wp if c])
        if len(cut):
            piece_poly.set_xy(np.vstack([cut, cut[0]]))
            piece_poly.set_visible(True)
            piece_outline.set_data(
                np.append(cut[:, 0], cut[0, 0]),
                np.append(cut[:, 1], cut[0, 1]))
        else:
            piece_poly.set_visible(False)
            piece_outline.set_data([], [])
        full_trail_t.set_data(S["tip_t"][:, 0], S["tip_t"][:, 1])
        full_trail_p.set_data(S["tip_p"][:, 0], S["tip_p"][:, 1])
        ax_arm.set_title(
            f"Bras 2-DOF — [{split}] pièce {S['ep']} / {n_ep - 1}", fontsize=10)

    def _rebuild_metrics():
        nonlocal m_vlines
        for ax in metric_axes:
            ax.cla()
        m_vlines = []
        pred, tgt, seq_len = S["pred"], S["tgt"], S["seq_len"]
        target_keys = S["target_keys"]
        t_ax = np.arange(seq_len) * 0.01
        col, done = 0, 0
        for key in target_keys:
            dim = dict(METRIC_KEYS)[key]
            for j in range(dim):
                if done >= len(metric_axes):
                    break
                lbl = f"{key}[{j}]" if dim > 1 else key
                ax  = metric_axes[done]
                ax.plot(t_ax, tgt [:seq_len, col + j],
                        color="royalblue", lw=1.2, label="vrai")
                ax.plot(t_ax, pred[:seq_len, col + j],
                        color="tomato", lw=1.2, ls="--", label="prédit", alpha=0.85)
                vl = ax.axvline(0, color="gray", lw=1, ls=":")
                ax.set_title(lbl, fontsize=9)
                ax.set_ylabel(_UNITS.get(key, ""), fontsize=8)
                ax.legend(fontsize=7, loc="upper right")
                ax.grid(True, alpha=0.3)
                m_vlines.append(vl)
                done += 1
            col += dim
            if done >= len(metric_axes):
                break
        metric_axes[-1].set_xlabel("t [s]", fontsize=8)

    def _draw_frame(fi):
        frames = S["frames"]
        fi = max(0, min(fi, len(frames) - 1))
        i  = frames[fi]
        t  = i * 0.01
        S["fi"] = fi

        line_true.set_data([0, S["elbow_t"][i, 0], S["tip_t"][i, 0]],
                           [0, S["elbow_t"][i, 1], S["tip_t"][i, 1]])
        line_pred.set_data([0, S["elbow_p"][i, 0], S["tip_p"][i, 0]],
                           [0, S["elbow_p"][i, 1], S["tip_p"][i, 1]])
        s = max(0, i - 120)
        trace_true.set_data(S["tip_t"][s:i, 0], S["tip_t"][s:i, 1])
        trace_pred.set_data(S["tip_p"][s:i, 0], S["tip_p"][s:i, 1])
        time_text.set_text(f"t = {t:.2f} s")
        for vl in m_vlines:
            vl.set_xdata([t, t])

        sl_t.eventson = False
        sl_t.set_val(fi)
        sl_t.eventson = True

    # ── callbacks widgets ─────────────────────────────────────────────────────
    def on_ep_change(val):
        ep = int(round(sl_ep.val))
        if ep == S["ep"]:
            return
        S["playing"] = False
        btn_pp.label.set_text("▶")

        _load_ep(ep)

        n_frames = len(S["frames"])
        sl_t.valmax = max(1, n_frames - 1)
        sl_t.ax.set_xlim(0, max(1, n_frames - 1))

        title_text.set_text(
            f"World model — [{split}]  pièce {ep} / {n_ep - 1}  "
            f"({S['seq_len']} pas,  shape → trajectoire complète)")

        _refresh_static()
        _rebuild_metrics()
        _draw_frame(0)
        fig.canvas.draw_idle()

    def on_t_change(val):
        fi = int(round(sl_t.val))
        if fi != S["fi"] and S["frames"]:
            S["playing"] = False
            btn_pp.label.set_text("▶")
            _draw_frame(fi)
            fig.canvas.draw_idle()

    def on_play_pause(event):
        S["playing"] = not S["playing"]
        btn_pp.label.set_text("⏸" if S["playing"] else "▶")

    def on_prev(event):
        sl_ep.set_val(max(0, int(sl_ep.val) - 1))

    def on_next_ep(event):
        sl_ep.set_val(min(n_ep - 1, int(sl_ep.val) + 1))

    sl_ep.on_changed(on_ep_change)
    sl_t.on_changed(on_t_change)
    btn_pp.on_clicked(on_play_pause)
    btn_prv.on_clicked(on_prev)
    btn_nxt.on_clicked(on_next_ep)

    # ── boucle animation (timer) ──────────────────────────────────────────────
    def tick(_):
        if S["playing"] and S["frames"]:
            next_fi = S["fi"] + 1
            if next_fi >= len(S["frames"]):
                next_fi = 0
            _draw_frame(next_fi)
        return []

    anim = animation.FuncAnimation(
        fig, tick, interval=30, blit=False, cache_frame_data=False)

    # Chargement initial
    on_ep_change(0)
    plt.show()
    return anim


# ── animation classique (mode --save_gif / épisode fixe) ──────────────────────
def animate(episode_idx=None, save_dir="world_model/checkpoints",
            save_path=None, step=3, split="val"):
    device = get_device()
    model, norm, H, full_ds, subset = _get_split(save_dir, split, device)

    if episode_idx is None:
        episode_idx = random.randint(0, len(subset) - 1)
    episode_idx = episode_idx % len(subset)

    pred, tgt, waypoints, seq_len, q_col, target_keys = _run_inference(
        model, norm, H, full_ds, subset, episode_idx, device)

    q_pred = pred[:seq_len, q_col:q_col + 2]
    q_true = tgt [:seq_len, q_col:q_col + 2]
    _, elbow_p, tip_p = fk(q_pred)
    _, elbow_t, tip_t = fk(q_true)

    cut_wp = [(x, y) for x, y, c in waypoints if c]
    cut_xy = np.array(cut_wp) if cut_wp else np.empty((0, 2))
    frames = range(0, seq_len, step)

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        f"World model — épisode [{split}] #{episode_idx}  (shape only → full trajectory)",
        fontsize=12)

    gs     = fig.add_gridspec(4, 2, left=0.05, right=0.97,
                              top=0.93, bottom=0.06, hspace=0.55, wspace=0.3)
    ax_arm = fig.add_subplot(gs[:, 0])
    metric_axes = [fig.add_subplot(gs[i, 1]) for i in range(4)]

    t_axis = np.arange(seq_len) * 0.01

    arm_lim = L1 + L2 + 0.15
    ax_arm.set_xlim(-arm_lim, arm_lim); ax_arm.set_ylim(-arm_lim, arm_lim)
    ax_arm.set_aspect("equal")
    ax_arm.set_title("Bras 2-DOF")
    ax_arm.set_xlabel("x [m]"); ax_arm.set_ylabel("y [m]")
    ax_arm.axhline(0, color="gray", lw=0.5); ax_arm.axvline(0, color="gray", lw=0.5)
    ax_arm.grid(True, alpha=0.2)

    if len(cut_xy):
        ax_arm.fill(cut_xy[:, 0], cut_xy[:, 1],
                    color="gold", alpha=0.25, zorder=1, label="pièce")
        ax_arm.plot(np.append(cut_xy[:, 0], cut_xy[0, 0]),
                    np.append(cut_xy[:, 1], cut_xy[0, 1]), "k--", lw=1, zorder=2)

    ax_arm.plot(tip_t[:, 0], tip_t[:, 1], color="royalblue", lw=0.8, alpha=0.3, zorder=3)
    ax_arm.plot(tip_p[:, 0], tip_p[:, 1], color="tomato",    lw=0.8, alpha=0.3, zorder=3)

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

    _UNITS = {"q_real":"rad","q_des":"rad","q_sensed":"rad",
              "dq_real":"rad/s","dq_des":"rad/s","dq_sensed":"rad/s",
              "tau":"N·m","is_cutting":"prob"}

    vlines = []
    col, done = 0, 0
    for key in target_keys:
        dim = dict(METRIC_KEYS)[key]
        for j in range(dim):
            if done >= len(metric_axes):
                break
            lbl = f"{key}[{j}]" if dim > 1 else key
            ax  = metric_axes[done]
            ax.plot(t_axis, tgt [:seq_len, col + j], color="royalblue", lw=1.2, label="vrai")
            ax.plot(t_axis, pred[:seq_len, col + j], color="tomato",    lw=1.2, ls="--",
                    label="prédit", alpha=0.85)
            vl = ax.axvline(0, color="gray", lw=1, ls=":")
            ax.set_title(lbl, fontsize=9)
            ax.set_ylabel(_UNITS.get(key, ""), fontsize=8)
            ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3)
            vlines.append(vl)
            done += 1
        col += dim
        if done >= len(metric_axes):
            break
    metric_axes[-1].set_xlabel("t [s]", fontsize=8)

    history_len = 120

    def update(frame_idx):
        i = list(frames)[frame_idx]
        t = i * 0.01
        line_true.set_data([0, elbow_t[i, 0], tip_t[i, 0]],
                           [0, elbow_t[i, 1], tip_t[i, 1]])
        line_pred.set_data([0, elbow_p[i, 0], tip_p[i, 0]],
                           [0, elbow_p[i, 1], tip_p[i, 1]])
        start = max(0, i - history_len)
        trace_true.set_data(tip_t[start:i, 0], tip_t[start:i, 1])
        trace_pred.set_data(tip_p[start:i, 0], tip_p[start:i, 1])
        time_text.set_text(f"t = {t:.2f} s")
        for vl in vlines:
            vl.set_xdata([t, t])
        return (line_true, line_pred, trace_true, trace_pred, time_text, *vlines)

    anim = animation.FuncAnimation(
        fig, update, frames=len(list(frames)), interval=30, blit=True)

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        writer = animation.PillowWriter(fps=30) if ext == ".gif" \
                 else animation.FFMpegWriter(fps=30, bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Animation sauvegardée → {save_path}")
        plt.close()
    else:
        plt.show()
    return anim


# ── helper Colab ───────────────────────────────────────────────────────────────
def show_in_colab(episode_idx=None, save_dir="world_model/checkpoints",
                  step=3, split="val"):
    from IPython.display import HTML
    anim = animate(episode_idx=episode_idx, save_dir=save_dir,
                   save_path=None, step=step, split=split)
    plt.close()
    return HTML(anim.to_jshtml())


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episode_idx", type=int, default=None,
                   help="Épisode précis (désactive le browser interactif)")
    p.add_argument("--save_dir",    default=DEFAULTS["save_dir"])
    p.add_argument("--save_gif",    default=None,
                   help="Chemin de sauvegarde (.gif ou .mp4)")
    p.add_argument("--step",        type=int, default=3,
                   help="1 frame sur N dans le mode animation fixe")
    p.add_argument("--split",       default="val",
                   choices=["val", "train", "all"])
    args = p.parse_args()

    if args.save_gif or args.episode_idx is not None:
        # Mode animation classique / export
        animate(episode_idx=args.episode_idx, save_dir=args.save_dir,
                save_path=args.save_gif, step=args.step, split=args.split)
    else:
        # Mode browser interactif (défaut)
        browse(save_dir=args.save_dir, split=args.split, step=1)
