"""
Visualisation du world model — animation du bras 2-DOF.

Affiche côte à côte :
  - Gauche  : animation du bras (vrai vs prédit) + tracé de coupe
  - Milieu  : métriques de trajectoire (q, dq, …) + qualité (cut_deviation, cut_defect)
  - Droite  : distribution des losses + rang

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
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import math
import numpy as np
import torch
from torch.utils.data import random_split

from .dataset import (
    BINARY_IDX, METRIC_KEYS, Normalizer,
    TrajectoryDataset, build_obs_vector, collate_fn, target_dim,
)
from .model import WorldModel
from .train import DEFAULTS, get_device

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm import inverse_kinematics


# ── cinématique directe ────────────────────────────────────────────────────────
L1, L2 = math.sqrt(2), math.sqrt(2)

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
    norm     = Normalizer.load(os.path.join(save_dir, "normalizer.npz"))
    dev_mean = float(ckpt.get("dev_mean", 0.0))
    dev_std  = float(ckpt.get("dev_std",  1.0))
    return model, norm, H, dev_mean, dev_std


# ── helpers dataset ────────────────────────────────────────────────────────────
def _get_split(save_dir, split, device):
    """Retourne (model, norm, H, full_ds, subset, dev_mean, dev_std)."""
    model, norm, H, dev_mean, dev_std = _load(save_dir, device)
    with open(H["db_path"]) as f:
        piece_db = json.load(f)["pieces"]
    episode_paths = sorted(glob.glob(os.path.join(H["data_dir"], "episode_*.npz")))
    full_ds = TrajectoryDataset(episode_paths, piece_db, normalizer=norm,
                                target_keys=H.get("target_keys", None))
    n_val   = max(1, int(len(full_ds) * H["val_split"]))
    n_train = len(full_ds) - n_val
    g       = torch.Generator().manual_seed(H["seed"])
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)
    subset  = {"val": val_ds, "train": train_ds}.get(split, full_ds)

    return model, norm, H, full_ds, subset, dev_mean, dev_std


def _compute_loss_distribution(model, norm, H, full_ds, subset, device,
                               batch_size: int = 32):
    """Calcule la MSE loss (sur q) pour chaque épisode — inférence batché."""
    n_ep = len(subset)
    print(f"Calcul de la distribution des losses sur {n_ep} épisodes...")

    target_keys = H.get("target_keys") or [k for k, _ in METRIC_KEYS]
    if isinstance(target_keys, str):
        target_keys = [k.strip() for k in target_keys.split(",")]
    q_col = 0
    for k in target_keys:
        if k in ("q_des", "q_real"):
            break
        q_col += dict(METRIC_KEYS)[k]

    sf = max(1, int(H.get("subsample_factor", 1)))

    losses = [float("nan")] * n_ep
    start = 0
    while start < n_ep:
        end   = min(start + batch_size, n_ep)
        items = [subset[i] for i in range(start, end)]
        B     = len(items)

        # items sont des 7-tuples : (wps, speed, obs, length, cut_dev, cut_defect, is_cutting)
        wps_list = [it[0] for it in items]
        speeds   = torch.stack([it[1] for it in items]).unsqueeze(1).to(device)
        obs_list = [it[2] for it in items]
        seq_lens = [it[3] for it in items]
        max_n_sub = max(math.ceil(T / sf) for T in seq_lens)
        max_W     = max(w.shape[0] for w in wps_list)

        wps_t   = torch.zeros(B, max_W, 3, device=device)
        wp_lens = torch.zeros(B, dtype=torch.long, device=device)
        for k, w in enumerate(wps_list):
            wps_t[k, :w.shape[0]] = w.to(device)
            wp_lens[k] = w.shape[0]

        try:
            with torch.no_grad():
                pred_norm, _ = model.predict(wps_t, wp_lens, speeds,
                                             max_len=max_n_sub)
            pred_np = norm.denormalize_tensor(pred_norm).cpu().numpy()

            for k in range(B):
                T      = seq_lens[k]
                n_sub  = math.ceil(T / sf)
                tgt_sub = norm.denormalize(obs_list[k][::sf].numpy())
                mse = float(np.mean(
                    (pred_np[k, :n_sub, q_col:q_col+2] - tgt_sub[:n_sub, q_col:q_col+2]) ** 2
                ))
                losses[start + k] = mse
        except Exception:
            pass

        print(f"  {end}/{n_ep}", end="\r")
        start = end

    print(f"\nDistribution calculée.")
    return np.array(losses)


def _upsample(arr: np.ndarray, sf: int, T_full: int) -> np.ndarray:
    """Interpole linéairement un tableau (n_sub,) ou (n_sub, D) vers T_full."""
    if sf <= 1:
        return arr
    n_sub    = arr.shape[0]
    sub_idx  = np.arange(n_sub) * sf
    full_idx = np.arange(T_full)
    if arr.ndim == 1:
        return np.interp(full_idx, sub_idx, arr)
    return np.stack(
        [np.interp(full_idx, sub_idx, arr[:, d]) for d in range(arr.shape[1])],
        axis=1,
    )


def _run_inference(model, norm, H, full_ds, subset, ep_idx, device,
                   dev_mean: float = 0.0, dev_std: float = 1.0):
    """Inférence sur un épisode.

    Retourne (pred, tgt, waypoints, seq_len, q_col, target_keys, quality) où
    quality est un dict avec les clés :
        dev_pred    (T,)  déviation prédite [m]
        defect_prob (T,)  probabilité de défaut prédite [0-1]
        dev_true    (T,)  déviation réelle [m]
        defect_true (T,)  défaut réel 0/1
        is_cutting  (T,)  masque de découpe 0/1
    """
    item    = subset[ep_idx]
    wps_t   = item[0].unsqueeze(0).to(device)
    speed_t = item[1]
    obs_norm_t = item[2]
    seq_len = item[3]
    speed   = speed_t.unsqueeze(0).unsqueeze(-1).to(device)
    wp_len  = torch.tensor([wps_t.shape[1]]).to(device)
    T       = obs_norm_t.shape[0]

    sf      = max(1, int(H.get("subsample_factor", 1)))
    n_steps = math.ceil(T / sf)

    with torch.no_grad():
        pred_norm, quality_norm = model.predict(wps_t, wp_len, speed,
                                                max_len=n_steps)

    pred_np    = pred_norm[0].cpu().numpy()         # (n_steps, D)
    quality_np = quality_norm[0].cpu().numpy()      # (n_steps, 2)
    tgt_norm   = obs_norm_t.numpy()                 # (T, D)

    pred    = _upsample(norm.denormalize(pred_np), sf, T)
    tgt     = norm.denormalize(tgt_norm)
    qual_up = _upsample(quality_np, sf, T)          # (T, 2)

    # Dénormalisation des prédictions de qualité
    cut_dev_pred    = qual_up[:, 0] * dev_std + dev_mean
    cut_defect_prob = 1.0 / (1.0 + np.exp(-qual_up[:, 1]))

    # Vérité terrain depuis le dataset (déjà dénormalisée pour cut_dev)
    orig_idx = subset.indices[ep_idx] if hasattr(subset, "indices") else ep_idx
    s = full_ds.samples[orig_idx]
    cut_dev_true    = s["cut_deviation"] * dev_std + dev_mean
    cut_defect_true = s["cut_defect"]
    is_cutting      = s["is_cutting"]

    target_keys = H.get("target_keys") or [k for k, _ in METRIC_KEYS]
    if isinstance(target_keys, str):
        target_keys = [k.strip() for k in target_keys.split(",")]

    if "is_cutting" in target_keys:
        ic  = target_keys.index("is_cutting")
        col = sum(dict(METRIC_KEYS)[k] for k in target_keys[:ic])
        pred[:, col] = 1 / (1 + np.exp(-pred[:, col]))

    waypoints = full_ds.samples[orig_idx]["waypoints"]

    col, q_col = 0, None
    for k in target_keys:
        d = dict(METRIC_KEYS)[k]
        if k in ("q_des", "q_real"):
            q_col = col
            break
        col += d
    if q_col is None:
        raise ValueError(f"Aucun signal de position (q_des/q_real) dans target_keys={target_keys}")

    quality = dict(
        dev_pred    = cut_dev_pred,
        defect_prob = cut_defect_prob,
        dev_true    = cut_dev_true,
        defect_true = cut_defect_true,
        is_cutting  = is_cutting,
    )
    return pred, tgt, waypoints, seq_len, q_col, target_keys, quality


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
    model, norm, H, full_ds, subset, dev_mean, dev_std = _get_split(
        save_dir, split, device)
    n_ep = len(subset)

    all_losses = _compute_loss_distribution(model, norm, H, full_ds, subset,
                                            device)

    _UNITS = {"q_real":"rad","q_des":"rad","q_sensed":"rad",
              "dq_real":"rad/s","dq_des":"rad/s","dq_sensed":"rad/s",
              "tau":"N·m","is_cutting":"prob",
              "cut_deviation":"m","cut_defect":"prob"}

    S = dict(ep=-1, fi=0, playing=False,
             pred=None, tgt=None, waypoints=None, seq_len=None,
             frames=None, target_keys=None, quality=None,
             elbow_t=None, elbow_p=None, tip_t=None, tip_p=None)

    fig = plt.figure(figsize=(18, 10))
    gs  = fig.add_gridspec(4, 3, left=0.05, right=0.97,
                           top=0.92, bottom=0.23, hspace=0.55, wspace=0.35)
    ax_arm      = fig.add_subplot(gs[:, 0])
    metric_axes = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    ax_dist     = fig.add_subplot(gs[:2, 2])
    ax_rank     = fig.add_subplot(gs[2:, 2])

    valid_losses = all_losses[~np.isnan(all_losses)]
    ax_dist.hist(valid_losses, bins=40, color="steelblue", alpha=0.75,
                 edgecolor="white", lw=0.4)
    ax_dist.set_title("Distribution des losses (MSE q)", fontsize=9)
    ax_dist.set_xlabel("MSE", fontsize=8)
    ax_dist.set_ylabel("# épisodes", fontsize=8)
    ax_dist.grid(True, alpha=0.25)
    loss_marker_v  = ax_dist.axvline(np.nan, color="tomato", lw=2, ls="--",
                                     label="pièce courante")
    loss_marker_tx = ax_dist.text(0.98, 0.95, "", transform=ax_dist.transAxes,
                                  fontsize=8, ha="right", va="top",
                                  color="tomato", fontweight="bold")
    ax_dist.legend(fontsize=7, loc="upper left")

    sorted_idx = np.argsort(valid_losses)
    ax_rank.plot(np.arange(len(valid_losses)), valid_losses[sorted_idx],
                 color="steelblue", lw=1.2)
    ax_rank.set_title("Losses triées", fontsize=9)
    ax_rank.set_xlabel("Rang", fontsize=8)
    ax_rank.set_ylabel("MSE", fontsize=8)
    ax_rank.grid(True, alpha=0.25)
    rank_marker_h  = ax_rank.axhline(np.nan, color="tomato", lw=1.5, ls="--")
    rank_marker_pt, = ax_rank.plot([], [], "o", color="tomato", ms=7, zorder=5)

    ax_sl_ep   = fig.add_axes([0.05, 0.13, 0.60, 0.03])
    ax_sl_t    = fig.add_axes([0.05, 0.07, 0.60, 0.03])
    ax_btn_pp  = fig.add_axes([0.68, 0.06, 0.07, 0.07])
    ax_btn_prv = fig.add_axes([0.77, 0.06, 0.07, 0.07])
    ax_btn_nxt = fig.add_axes([0.86, 0.06, 0.07, 0.07])

    sl_ep   = mwidgets.Slider(ax_sl_ep,  "Pièce", 0, max(1, n_ep - 1),
                              valinit=0, valstep=1)
    sl_t    = mwidgets.Slider(ax_sl_t,   "Temps", 0, 1,
                              valinit=0, valstep=1)
    btn_pp  = mwidgets.Button(ax_btn_pp,  "▶")
    btn_prv = mwidgets.Button(ax_btn_prv, "◀ ep")
    btn_nxt = mwidgets.Button(ax_btn_nxt, "ep ▶")

    title_text = fig.suptitle("", fontsize=11)

    arm_lim = L1 + L2 + 0.2
    ax_arm.set_xlim(-0.3, arm_lim); ax_arm.set_ylim(-0.3, arm_lim)
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
    defect_text    = ax_arm.text(0.02, 0.90, "", transform=ax_arm.transAxes,
                                 fontsize=8, va="top", color="tomato")
    ax_arm.legend(fontsize=8, loc="upper right")

    m_vlines = []

    def _load_ep(ep_idx):
        pred, tgt, waypoints, seq_len, q_col, target_keys, quality = _run_inference(
            model, norm, H, full_ds, subset, ep_idx, device,
            dev_mean, dev_std)
        q_pred = pred[:seq_len, q_col:q_col + 2]
        q_true = tgt [:seq_len, q_col:q_col + 2]
        _, elbow_p, tip_p = fk(q_pred)
        _, elbow_t, tip_t = fk(q_true)
        S.update(ep=ep_idx, fi=0, pred=pred, tgt=tgt,
                 waypoints=waypoints, seq_len=seq_len, target_keys=target_keys,
                 quality=quality,
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

        q = S["quality"]
        if q is not None:
            sl = S["seq_len"]
            dr_pred = float(np.mean(q["defect_prob"][:sl] > 0.5)) * 100
            dr_true = float(np.mean(q["defect_true"][:sl] > 0.5)) * 100
            defect_text.set_text(f"Défauts  préd. {dr_pred:.1f}%  /  réel {dr_true:.1f}%")

    def _rebuild_metrics():
        nonlocal m_vlines
        for ax in metric_axes:
            ax.cla()
        m_vlines = []
        pred, tgt, seq_len = S["pred"], S["tgt"], S["seq_len"]
        target_keys = S["target_keys"]
        t_ax = np.arange(seq_len) * 0.1
        col, done = 0, 0

        # — signaux de trajectoire —
        for key in target_keys:
            dim = dict(METRIC_KEYS)[key]
            for j in range(dim):
                if done >= len(metric_axes) - 2:   # réserve 2 slots pour la qualité
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
            if done >= len(metric_axes) - 2:
                break

        # — qualité : cut_deviation —
        q = S["quality"]
        if q is not None and done < len(metric_axes):
            ax = metric_axes[done]
            ax.plot(t_ax, q["dev_true"]   [:seq_len], color="royalblue",
                    lw=1.2, label="vrai")
            ax.plot(t_ax, q["dev_pred"]   [:seq_len], color="tomato",
                    lw=1.2, ls="--", label="prédit", alpha=0.85)
            ax.axhline(0.020, color="darkorange", lw=0.8, ls=":", label="seuil 20mm")
            vl = ax.axvline(0, color="gray", lw=1, ls=":")
            ax.set_title("cut_deviation", fontsize=9)
            ax.set_ylabel("m", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)
            m_vlines.append(vl)
            done += 1

        # — qualité : cut_defect —
        if q is not None and done < len(metric_axes):
            ax = metric_axes[done]
            ax.plot(t_ax, q["defect_true"][:seq_len], color="royalblue",
                    lw=1.2, label="vrai", drawstyle="steps-post")
            ax.plot(t_ax, q["defect_prob"][:seq_len], color="tomato",
                    lw=1.2, ls="--", label="prob. préd.", alpha=0.85)
            ax.axhline(0.5, color="gray", lw=0.8, ls=":")
            ax.set_ylim(-0.05, 1.1)
            vl = ax.axvline(0, color="gray", lw=1, ls=":")
            ax.set_title("cut_defect", fontsize=9)
            ax.set_ylabel("prob.", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)
            m_vlines.append(vl)

        metric_axes[-1].set_xlabel("t [s]", fontsize=8)

    def _draw_frame(fi):
        frames = S["frames"]
        fi = max(0, min(fi, len(frames) - 1))
        i  = frames[fi]
        t  = i * 0.1
        S["fi"] = fi

        line_true.set_data([0, S["elbow_t"][i, 0], S["tip_t"][i, 0]],
                           [0, S["elbow_t"][i, 1], S["tip_t"][i, 1]])
        line_pred.set_data([0, S["elbow_p"][i, 0], S["tip_p"][i, 0]],
                           [0, S["elbow_p"][i, 1], S["tip_p"][i, 1]])
        s = max(0, i - 12)
        trace_true.set_data(S["tip_t"][s:i, 0], S["tip_t"][s:i, 1])
        trace_pred.set_data(S["tip_p"][s:i, 0], S["tip_p"][s:i, 1])
        time_text.set_text(f"t = {t:.2f} s")
        for vl in m_vlines:
            vl.set_xdata([t, t])

        sl_t.eventson = False
        sl_t.set_val(fi)
        sl_t.eventson = True

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

        q = S["quality"]
        dr_pred = float(np.mean(q["defect_prob"][:S["seq_len"]] > 0.5)) * 100
        dr_true = float(np.mean(q["defect_true"][:S["seq_len"]] > 0.5)) * 100
        title_text.set_text(
            f"World model — [{split}]  pièce {ep} / {n_ep - 1}  "
            f"({S['seq_len']} pas)   "
            f"défauts préd. {dr_pred:.1f}% / réel {dr_true:.1f}%")

        ep_loss = all_losses[ep]
        loss_marker_v.set_xdata([ep_loss, ep_loss])
        valid_mask = ~np.isnan(all_losses)
        percentile = float(np.mean(all_losses[valid_mask] <= ep_loss) * 100)
        rank_in_sorted = int(np.sum(valid_losses <= ep_loss))
        loss_marker_tx.set_text(f"MSE={ep_loss:.4f}\nPercentile {percentile:.0f}%")
        rank_marker_h.set_ydata([ep_loss, ep_loss])
        rank_marker_pt.set_data([rank_in_sorted], [ep_loss])
        ax_dist.figure.canvas.draw_idle()

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

    def tick(_):
        if S["playing"] and S["frames"]:
            next_fi = S["fi"] + 1
            if next_fi >= len(S["frames"]):
                next_fi = 0
            _draw_frame(next_fi)
        return []

    anim = animation.FuncAnimation(
        fig, tick, interval=30, blit=False, cache_frame_data=False)

    on_ep_change(0)
    plt.show()
    return anim


# ── animation classique (mode --save_gif / épisode fixe) ──────────────────────
def animate(episode_idx=None, save_dir="world_model/checkpoints",
            save_path=None, step=3, split="val"):
    device = get_device()
    model, norm, H, full_ds, subset, dev_mean, dev_std = _get_split(
        save_dir, split, device)

    if episode_idx is None:
        episode_idx = random.randint(0, len(subset) - 1)
    episode_idx = episode_idx % len(subset)

    pred, tgt, waypoints, seq_len, q_col, target_keys, quality = _run_inference(
        model, norm, H, full_ds, subset, episode_idx, device,
        dev_mean, dev_std)

    q_pred = pred[:seq_len, q_col:q_col + 2]
    q_true = tgt [:seq_len, q_col:q_col + 2]
    _, elbow_p, tip_p = fk(q_pred)
    _, elbow_t, tip_t = fk(q_true)

    cut_wp = [(x, y) for x, y, c in waypoints if c]
    cut_xy = np.array(cut_wp) if cut_wp else np.empty((0, 2))
    frames = range(0, seq_len, step)

    dr_pred = float(np.mean(quality["defect_prob"][:seq_len] > 0.5)) * 100
    dr_true = float(np.mean(quality["defect_true"][:seq_len] > 0.5)) * 100

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(
        f"World model — [{split}] #{episode_idx}  "
        f"défauts préd. {dr_pred:.1f}% / réel {dr_true:.1f}%",
        fontsize=11)

    # 3 colonnes : bras | trajectoire | qualité
    gs = fig.add_gridspec(4, 3, left=0.05, right=0.97,
                          top=0.93, bottom=0.06, hspace=0.55, wspace=0.35)
    ax_arm      = fig.add_subplot(gs[:, 0])
    metric_axes = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    ax_dev      = fig.add_subplot(gs[:2, 2])
    ax_def      = fig.add_subplot(gs[2:, 2])

    t_axis = np.arange(seq_len) * 0.1

    arm_lim = L1 + L2 + 0.2
    ax_arm.set_xlim(-0.3, arm_lim); ax_arm.set_ylim(-0.3, arm_lim)
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
            ax.plot(t_axis, tgt [:seq_len, col + j], color="royalblue",
                    lw=1.2, label="vrai")
            ax.plot(t_axis, pred[:seq_len, col + j], color="tomato",
                    lw=1.2, ls="--", label="prédit", alpha=0.85)
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

    # — cut_deviation —
    ax_dev.plot(t_axis, quality["dev_true"]   [:seq_len], color="royalblue",
                lw=1.2, label="vrai")
    ax_dev.plot(t_axis, quality["dev_pred"]   [:seq_len], color="tomato",
                lw=1.2, ls="--", label="prédit", alpha=0.85)
    ax_dev.axhline(0.020, color="darkorange", lw=0.8, ls=":", label="seuil 20mm")
    vl_dev = ax_dev.axvline(0, color="gray", lw=1, ls=":")
    ax_dev.set_title("cut_deviation", fontsize=9)
    ax_dev.set_ylabel("m", fontsize=8)
    ax_dev.legend(fontsize=7); ax_dev.grid(True, alpha=0.3)

    # — cut_defect —
    ax_def.plot(t_axis, quality["defect_true"][:seq_len], color="royalblue",
                lw=1.2, label="vrai", drawstyle="steps-post")
    ax_def.plot(t_axis, quality["defect_prob"][:seq_len], color="tomato",
                lw=1.2, ls="--", label="prob. préd.", alpha=0.85)
    ax_def.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax_def.set_ylim(-0.05, 1.1)
    vl_def = ax_def.axvline(0, color="gray", lw=1, ls=":")
    ax_def.set_title("cut_defect", fontsize=9)
    ax_def.set_ylabel("prob.", fontsize=8)
    ax_def.set_xlabel("t [s]", fontsize=8)
    ax_def.legend(fontsize=7); ax_def.grid(True, alpha=0.3)

    history_len = 12

    def update(frame_idx):
        i = list(frames)[frame_idx]
        t = i * 0.1
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
        vl_dev.set_xdata([t, t])
        vl_def.set_xdata([t, t])
        return (line_true, line_pred, trace_true, trace_pred, time_text,
                *vlines, vl_dev, vl_def)

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
    p.add_argument("--episode_idx", type=int, default=None)
    p.add_argument("--save_dir",    default=DEFAULTS["save_dir"])
    p.add_argument("--save_gif",    default=None,
                   help="Chemin de sauvegarde (.gif ou .mp4)")
    p.add_argument("--step",        type=int, default=3)
    p.add_argument("--split",       default="val",
                   choices=["val", "train", "all"])
    args = p.parse_args()

    if args.save_gif or args.episode_idx is not None:
        animate(episode_idx=args.episode_idx, save_dir=args.save_dir,
                save_path=args.save_gif, step=args.step, split=args.split)
    else:
        browse(save_dir=args.save_dir, split=args.split, step=1)
