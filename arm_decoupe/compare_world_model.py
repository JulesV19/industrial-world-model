"""
Comparateur World Model — bras découpe.

Compare les vraies séries temporelles aux prédictions du world model.

Modes :
    python compare_world_model.py                            # mode session (défaut)
    python compare_world_model.py --mode session             # évolution des métriques sur la session
    python compare_world_model.py --mode piece               # animation pièce par pièce
    python compare_world_model.py --mode piece --session 000 --piece 5
"""

import argparse
import math
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, Button
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import l1, l2
from record_dataset import CUT_DEFECT_THRESHOLD
from world_model.dataset import (
    HISTORY_K, Normalizer, build_obs_vector, METRIC_KEYS,
)
from world_model.model import WorldModel
from world_model.train import DEFAULTS, get_device

DATASET_DIR  = "dataset"
CKPT_DIR     = "world_model/checkpoints"

# ── dark theme ─────────────────────────────────────────────────────────────────
BG_DARK  = "#1a1a2e"
BG_PANEL = "#16213e"
COL_BORDER = "#444466"
COL_TEXT   = "#ddddff"
COL_TICK   = "#aaaacc"
COL_TRUE   = "#5599ff"   # bleu  : vrai
COL_PRED   = "#ff6644"   # rouge : prédit
COL_GRID   = "#555577"


def _style(ax):
    ax.set_facecolor(BG_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(COL_BORDER)
    ax.tick_params(colors=COL_TICK, labelsize=8)
    ax.title.set_color(COL_TEXT)
    ax.xaxis.label.set_color(COL_TICK)
    ax.yaxis.label.set_color(COL_TICK)


# ── cinématique directe ────────────────────────────────────────────────────────

def _fk(q):
    """q : (T, 2) → tip_x, tip_y (T,)"""
    x = l1 * np.cos(q[:, 0]) + l2 * np.cos(q[:, 0] + q[:, 1])
    y = l1 * np.sin(q[:, 0]) + l2 * np.sin(q[:, 0] + q[:, 1])
    return x, y


def _fk_full(q):
    """Retourne (elbow_xy, tip_xy) avec shapes (T, 2)."""
    ex = l1 * np.cos(q[:, 0])
    ey = l1 * np.sin(q[:, 0])
    tx = ex + l2 * np.cos(q[:, 0] + q[:, 1])
    ty = ey + l2 * np.sin(q[:, 0] + q[:, 1])
    elbow = np.stack([ex, ey], axis=1)
    tip   = np.stack([tx, ty], axis=1)
    return elbow, tip


def _corridor(x, y, thickness):
    n = len(x)
    nx_arr, ny_arr = np.zeros(n), np.zeros(n)
    for i in range(n):
        if i == 0:
            dx, dy = x[1] - x[0], y[1] - y[0]
        elif i == n - 1:
            dx, dy = x[-1] - x[-2], y[-1] - y[-2]
        else:
            dx, dy = x[i + 1] - x[i - 1], y[i + 1] - y[i - 1]
        length = np.hypot(dx, dy)
        if length > 1e-9:
            nx_arr[i], ny_arr[i] = -dy / length, dx / length
    return (x + thickness * nx_arr, y + thickness * ny_arr,
            x - thickness * nx_arr, y - thickness * ny_arr)


# ── chargement modèle ──────────────────────────────────────────────────────────

def _load_model(ckpt_dir, device):
    path = os.path.join(ckpt_dir, "best_model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint introuvable : {path}")
    ckpt = torch.load(path, map_location=device)
    H    = {**DEFAULTS, **ckpt.get("hyperparams", {})}
    model = WorldModel(
        shape_embed_dim = H.get("shape_embed_dim", 256),
        h_dim           = H.get("h_dim",           512),
        obs_dim         = H.get("obs_dim",          2),
        dropout         = 0.0,
        gru_layers      = H.get("gru_layers",       3),
        pe_dim          = H.get("pe_dim",           64),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm     = Normalizer.load(os.path.join(ckpt_dir, "normalizer.npz"))
    dev_mean = float(ckpt.get("dev_mean", 0.0))
    dev_std  = float(ckpt.get("dev_std",  1.0))
    target_keys = H.get("target_keys", ["q_real"])
    if isinstance(target_keys, str):
        target_keys = [k.strip() for k in target_keys.split(",")]
    return model, norm, dev_mean, dev_std, target_keys


# ── helpers dataset ────────────────────────────────────────────────────────────

def _list_sessions(data_dir):
    files = os.listdir(data_dir)
    ids = sorted({
        re.search(r"session_(\d+)_piece", f).group(1)
        for f in files
        if re.match(r"session_\d+_piece\d+\.npz$", f)
    })
    return ids


def _list_pieces(data_dir, sid):
    files = sorted([
        f for f in os.listdir(data_dir)
        if re.match(rf"session_{sid}_piece\d+\.npz$", f)
    ])
    indices = [int(re.search(r"piece(\d+)", f).group(1)) for f in files]
    return indices


def _build_history(session_devs, piece_count):
    """Construit le vecteur d'historique (K,) depuis le tableau de déviations déjà chargé."""
    if session_devs is None:
        return np.zeros(HISTORY_K, dtype=np.float32)
    n     = int(piece_count)
    start = max(0, n - HISTORY_K)
    hist  = session_devs[start:n]
    pad   = np.zeros(HISTORY_K - len(hist), dtype=np.float32)
    return np.concatenate([pad, hist]).astype(np.float32)


def _load_session_devs(data_dir, sid):
    """Charge le tableau de déviations d'une session (une seule fois)."""
    p = os.path.join(data_dir, f"session_{sid}_deviations.npy")
    return np.load(p).astype(np.float32) if os.path.exists(p) else None


# ── inférence batched ──────────────────────────────────────────────────────────

def _infer_batch(model, norm_mean_t, norm_std_t, dev_mean, dev_std,
                 raw_batch, device):
    """
    Inférence sur un batch de pièces déjà chargées.

    raw_batch : liste de (pidx, data_npz, hist_np)
    Retourne  : liste de dicts résultat (même format que _run_inference).
    """
    B     = len(raw_batch)
    max_W = max(item[1]["waypoints"].shape[0] for item in raw_batch)
    max_T = max(len(item[1]["q_real"])         for item in raw_batch)

    # Construire les tenseurs sur CPU avant le transfert groupé
    wps_t   = torch.zeros(B, max_W, 3,        dtype=torch.float32)
    wp_lens = torch.zeros(B,                   dtype=torch.long)
    speeds  = torch.zeros(B, 1,                dtype=torch.float32)
    hists   = torch.zeros(B, HISTORY_K, 1,    dtype=torch.float32)
    pcs     = torch.zeros(B, 1,                dtype=torch.float32)
    cads    = torch.zeros(B, 1,                dtype=torch.float32)

    for k, (_, data, hist_np) in enumerate(raw_batch):
        wp = data["waypoints"].astype(np.float32)
        wps_t[k, :wp.shape[0]].copy_(torch.from_numpy(wp))
        wp_lens[k] = wp.shape[0]
        speeds[k, 0] = float(data["duration_per_segment"])
        hists[k, :, 0].copy_(torch.from_numpy(hist_np))
        pcs[k, 0]  = float(data["piece_count"]) if "piece_count" in data else 0.0
        cads[k, 0] = float(data["cadence"])     if "cadence"     in data else 0.0

    # Transfert device unique
    wps_t   = wps_t  .to(device, non_blocking=True)
    wp_lens = wp_lens.to(device, non_blocking=True)
    speeds  = speeds .to(device, non_blocking=True)
    hists   = hists  .to(device, non_blocking=True)
    pcs     = pcs    .to(device, non_blocking=True)
    cads    = cads   .to(device, non_blocking=True)

    with torch.inference_mode():
        embed              = model._encode(wps_t, wp_lens, speeds, hists, pcs, cads)
        traj_norm, qual_t  = model.decoder(embed, max_T)

    # Dénormaliser sur CPU en une opération vectorisée
    pred_np = (traj_norm.cpu() * norm_std_t + norm_mean_t).numpy()  # (B, max_T, D)
    qual_np = qual_t.cpu().numpy()                                    # (B, max_T, 2)

    results = []
    for k, (pidx, data, _) in enumerate(raw_batch):
        T         = len(data["q_real"])
        q_real    = data["q_real"].astype(np.float32)
        q_pred    = pred_np[k, :T, :2]
        dev_pred  = qual_np[k, :T, 0] * dev_std + dev_mean
        def_prob  = 1.0 / (1.0 + np.exp(-qual_np[k, :T, 1]))
        cut       = data["is_cutting"].astype(bool)

        elbow_true, tip_true = _fk_full(q_real)
        elbow_pred, tip_pred = _fk_full(q_pred)

        results.append(dict(
            pidx=pidx, T=T, cut=cut,
            q_real=q_real,
            q_des=data["q_des"].astype(np.float32),
            q_pred=q_pred,
            dq_real=data["dq_real"].astype(np.float32),
            tau=data["tau"].astype(np.float32),
            cut_dev_true=data["cut_deviation"].astype(np.float32),
            cut_dev_pred=dev_pred,
            cut_defect=data["cut_defect"].astype(np.float32),
            defect_prob=def_prob,
            elbow_true=elbow_true, tip_true=tip_true,
            elbow_pred=elbow_pred, tip_pred=tip_pred,
            waypoints=data["waypoints"],
            speed=float(data["duration_per_segment"]),
            piece_count=int(data["piece_count"]) if "piece_count" in data else 0,
            cadence=float(data["cadence"])       if "cadence"     in data else 0.0,
        ))
    return results


def _infer_session_batched(model, norm, dev_mean, dev_std,
                            data_dir, sid, device, batch_size=64):
    """
    Charge toutes les pièces de la session (I/O), puis fait l'inférence
    en batches (GPU). Retourne la liste de dicts résultat.
    """
    piece_indices = _list_pieces(data_dir, sid)
    if not piece_indices:
        return []

    # Pré-calculer les tenseurs de normalisation une seule fois
    norm_mean_t = torch.tensor(norm.mean, dtype=torch.float32)
    norm_std_t  = torch.tensor(norm.std,  dtype=torch.float32)

    # Charger les déviations de session une seule fois
    session_devs = _load_session_devs(data_dir, sid)

    # I/O : charger tous les .npz
    raw_items = []
    for pidx in piece_indices:
        path = os.path.join(data_dir, f"session_{sid}_piece{pidx:04d}.npz")
        if not os.path.exists(path):
            continue
        data     = np.load(path)
        pc       = int(data["piece_count"]) if "piece_count" in data else 0
        hist_np  = _build_history(session_devs, pc)
        raw_items.append((pidx, data, hist_np))

    N = len(raw_items)
    print(f"  {N} pièces chargées — inférence en batches de {batch_size}…")

    all_results = []
    for b_start in range(0, N, batch_size):
        batch = raw_items[b_start:b_start + batch_size]
        print(f"  batch {b_start // batch_size + 1}/{math.ceil(N / batch_size)}"
              f"  ({b_start + len(batch)}/{N})", end="\r")
        all_results.extend(_infer_batch(model, norm_mean_t, norm_std_t,
                                         dev_mean, dev_std, batch, device))

    print(f"\n  Terminé ({N} pièces).")
    return all_results


def _run_inference(model, norm, dev_mean, dev_std, target_keys,
                   data_dir, sid, piece_idx, device):
    """Inférence sur une seule pièce (mode piece)."""
    path = os.path.join(data_dir, f"session_{sid}_piece{piece_idx:04d}.npz")
    if not os.path.exists(path):
        return None
    data     = np.load(path)
    pc       = int(data["piece_count"]) if "piece_count" in data else 0
    hist_np  = _build_history(_load_session_devs(data_dir, sid), pc)
    norm_mean_t = torch.tensor(norm.mean, dtype=torch.float32)
    norm_std_t  = torch.tensor(norm.std,  dtype=torch.float32)
    results = _infer_batch(model, norm_mean_t, norm_std_t, dev_mean, dev_std,
                           [(piece_idx, data, hist_np)], device)
    return results[0] if results else None


# ── métriques session ─────────────────────────────────────────────────────────

def _compute_session_metrics(model, norm, dev_mean, dev_std, target_keys,
                              data_dir, sid, device):
    """Agrège les métriques par pièce depuis l'inférence batched."""
    results = _infer_session_batched(model, norm, dev_mean, dev_std,
                                      data_dir, sid, device)
    if not results:
        return None

    piece_counts, mean_dev_true, mean_dev_pred = [], [], []
    max_dev_true,  max_dev_pred  = [], []
    defect_pct_true, defect_pct_pred = [], []

    for r in results:
        cut = r["cut"]
        if not cut.any():
            continue
        dv_t = r["cut_dev_true"][cut]
        dv_p = r["cut_dev_pred"][cut]
        df_t = r["cut_defect"][cut]
        df_p = r["defect_prob"][cut]
        piece_counts.append(r["piece_count"])
        mean_dev_true.append(float(dv_t.mean()))
        mean_dev_pred.append(float(dv_p.mean()))
        max_dev_true.append(float(dv_t.max()))
        max_dev_pred.append(float(dv_p.max()))
        defect_pct_true.append(100.0 * float((df_t > 0.5).mean()))
        defect_pct_pred.append(100.0 * float((df_p > 0.5).mean()))

    order = np.argsort(piece_counts)
    def _s(lst): return np.array(lst)[order]

    return dict(
        piece_counts    = _s(piece_counts),
        mean_dev_true   = _s(mean_dev_true),   mean_dev_pred   = _s(mean_dev_pred),
        max_dev_true    = _s(max_dev_true),     max_dev_pred    = _s(max_dev_pred),
        defect_pct_true = _s(defect_pct_true),  defect_pct_pred = _s(defect_pct_pred),
    )


# ── vue session ────────────────────────────────────────────────────────────────

def view_session_compare(data_dir=DATASET_DIR, ckpt_dir=CKPT_DIR, init_sid=None):
    """
    Vue session : pour chaque pièce de la session sélectionnée, compare les
    métriques agrégées réelles vs prédites par le world model.
    """
    device = get_device()
    print("Chargement du modèle…")
    model, norm, dev_mean, dev_std, target_keys = _load_model(ckpt_dir, device)

    session_ids = _list_sessions(data_dir)
    if not session_ids:
        print(f"Aucune session dans '{data_dir}'.")
        return
    if init_sid is None or init_sid not in session_ids:
        init_sid = session_ids[0]

    n_sess = len(session_ids)

    # ── layout ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(19, 10))
    fig.canvas.manager.set_window_title("Comparateur Sessions — World Model")
    fig.patch.set_facecolor(BG_DARK)

    gs = fig.add_gridspec(2, 3,
                          left=0.07, right=0.97, top=0.91, bottom=0.14,
                          wspace=0.32, hspace=0.48)
    ax_mdev  = fig.add_subplot(gs[0, 0])   # déviation moyenne
    ax_xdev  = fig.add_subplot(gs[0, 1])   # déviation max
    ax_def   = fig.add_subplot(gs[0, 2])   # % défauts
    ax_all   = fig.add_subplot(gs[1, :2])  # scatter global (réel seul, aperçu)
    ax_info  = fig.add_subplot(gs[1, 2])   # panneau info

    for ax in [ax_mdev, ax_xdev, ax_def, ax_all, ax_info]:
        _style(ax)

    # Pré-charger les aperçus (deviations.npy, léger)
    all_mean_devs = {}
    for sid in session_ids:
        p = os.path.join(data_dir, f"session_{sid}_deviations.npy")
        if os.path.exists(p):
            all_mean_devs[sid] = np.load(p)

    cmap = plt.cm.plasma
    ax_all.set_title("Déviation moy. × sessions (réel, aperçu)", fontweight="bold")
    for i, sid in enumerate(session_ids):
        if sid in all_mean_devs:
            devs  = all_mean_devs[sid]
            color = cmap(i / max(1, n_sess - 1))
            ax_all.plot(np.arange(len(devs)), devs * 1000,
                        alpha=0.2, lw=0.8, color=color)
    ax_all.axhline(CUT_DEFECT_THRESHOLD * 1000, color="#ff4444", lw=1.2,
                   ls="--", label=f"seuil {CUT_DEFECT_THRESHOLD*1000:.0f} mm")
    ax_all.set_xlabel("N° pièce dans la session")
    ax_all.set_ylabel("Déviation (mm)")
    ax_all.legend(fontsize=8, labelcolor="white",
                  facecolor=BG_PANEL, edgecolor=COL_BORDER)
    ax_all.grid(True, linestyle=":", alpha=0.3, color=COL_GRID)

    highlight_line = [None]

    # Slider session
    ax_sl = fig.add_axes([0.20, 0.04, 0.60, 0.025], facecolor="#2a2a4e")
    sl = Slider(ax_sl, "Session", 1, n_sess, valinit=1, valstep=1, color="#5555cc")
    sl.label.set_color(COL_TICK)
    sl.valtext.set_color("#ccccff")

    # Cache des métriques déjà calculées
    metrics_cache = {}

    def draw_session(sid):
        print(f"\nCalcul des métriques pour la session {sid}…")
        if sid not in metrics_cache:
            metrics_cache[sid] = _compute_session_metrics(
                model, norm, dev_mean, dev_std, target_keys,
                data_dir, sid, device)

        m = metrics_cache[sid]
        if m is None:
            fig.suptitle(f"Session {sid} — aucune pièce trouvée",
                         color="#ff6666")
            fig.canvas.draw_idle()
            return

        xs = m["piece_counts"]
        n  = len(xs)

        for ax in [ax_mdev, ax_xdev, ax_def, ax_info]:
            ax.clear()
            _style(ax)

        # ── déviation moyenne ──────────────────────────────────────────────────
        ax_mdev.plot(xs, m["mean_dev_true"] * 1000, color=COL_TRUE, lw=1.8,
                     label="vrai")
        ax_mdev.plot(xs, m["mean_dev_pred"] * 1000, color=COL_PRED, lw=1.8,
                     ls="--", label="prédit", alpha=0.85)
        ax_mdev.fill_between(xs, 0, m["mean_dev_true"] * 1000,
                             alpha=0.10, color=COL_TRUE)
        ax_mdev.fill_between(xs, 0, m["mean_dev_pred"] * 1000,
                             alpha=0.08, color=COL_PRED)
        ax_mdev.axhline(CUT_DEFECT_THRESHOLD * 1000, color="#ff4444",
                        ls="--", lw=1.2)
        ax_mdev.set_title("Déviation moy. (mm)", fontweight="bold")
        ax_mdev.set_xlabel("N° pièce")
        ax_mdev.legend(fontsize=8, labelcolor="white",
                       facecolor=BG_PANEL, edgecolor=COL_BORDER)
        ax_mdev.grid(True, linestyle=":", alpha=0.3, color=COL_GRID)

        # ── déviation max ──────────────────────────────────────────────────────
        ax_xdev.plot(xs, m["max_dev_true"] * 1000, color=COL_TRUE, lw=1.8,
                     label="vrai")
        ax_xdev.plot(xs, m["max_dev_pred"] * 1000, color=COL_PRED, lw=1.8,
                     ls="--", label="prédit", alpha=0.85)
        ax_xdev.fill_between(xs, 0, m["max_dev_true"] * 1000,
                             alpha=0.10, color=COL_TRUE)
        ax_xdev.fill_between(xs, 0, m["max_dev_pred"] * 1000,
                             alpha=0.08, color=COL_PRED)
        ax_xdev.axhline(CUT_DEFECT_THRESHOLD * 1000, color="#ff4444",
                        ls="--", lw=1.2)
        ax_xdev.set_title("Déviation max (mm)", fontweight="bold")
        ax_xdev.set_xlabel("N° pièce")
        ax_xdev.legend(fontsize=8, labelcolor="white",
                       facecolor=BG_PANEL, edgecolor=COL_BORDER)
        ax_xdev.grid(True, linestyle=":", alpha=0.3, color=COL_GRID)

        # ── % défauts (barres groupées) ────────────────────────────────────────
        w = 0.4
        ax_def.bar(xs - w / 2, m["defect_pct_true"], width=w,
                   color=COL_TRUE, alpha=0.75, label="vrai")
        ax_def.bar(xs + w / 2, m["defect_pct_pred"], width=w,
                   color=COL_PRED, alpha=0.75, label="prédit")
        ax_def.axhline(50, color="#ffaa00", ls="--", lw=1.0)
        ax_def.set_ylim(0, 105)
        ax_def.set_title("% points défectueux", fontweight="bold")
        ax_def.set_xlabel("N° pièce")
        ax_def.set_ylabel("%")
        ax_def.legend(fontsize=8, labelcolor="white",
                      facecolor=BG_PANEL, edgecolor=COL_BORDER)
        ax_def.grid(True, linestyle=":", alpha=0.3, color=COL_GRID, axis="y")

        # ── panneau info ───────────────────────────────────────────────────────
        ax_info.axis("off")
        mae_mdev = float(np.mean(np.abs(m["mean_dev_true"] - m["mean_dev_pred"]))) * 1000
        mae_xdev = float(np.mean(np.abs(m["max_dev_true"]  - m["max_dev_pred"])))  * 1000
        mae_def  = float(np.mean(np.abs(m["defect_pct_true"] - m["defect_pct_pred"])))
        corr_mdev = float(np.corrcoef(m["mean_dev_true"], m["mean_dev_pred"])[0, 1]) \
                    if n > 1 else float("nan")

        cad_path = os.path.join(data_dir, f"session_{sid}_cadence.npy")
        cadence = float(np.load(cad_path)) if os.path.exists(cad_path) else 0.0

        info_lines = [
            f"Session   : {sid}",
            f"Pièces    : {n}",
            f"Cadence   : {cadence:.0f} pièces/h",
            "",
            f"MAE dév. moy  : {mae_mdev:.2f} mm",
            f"MAE dév. max  : {mae_xdev:.2f} mm",
            f"MAE % défauts : {mae_def:.1f} %",
            f"Corr. dév.moy : {corr_mdev:.3f}",
            "",
            f"Dév. moy finale réelle  : {m['mean_dev_true'][-1]*1000:.2f} mm",
            f"Dév. moy finale prédite : {m['mean_dev_pred'][-1]*1000:.2f} mm",
        ]
        for i, line in enumerate(info_lines):
            color = "#ffaa44" if "Session" in line or "Pièces" in line or "Cadence" in line \
                    else "#55ccff" if "MAE" in line or "Corr" in line \
                    else "#ff7777" if "défauts" in line and "MAE" not in line \
                    else COL_TEXT
            ax_info.text(0.05, 0.96 - i * 0.085, line,
                         transform=ax_info.transAxes,
                         fontsize=8.5, color=color, family="monospace")
        ax_info.set_title("Résumé session", fontweight="bold")

        # Highlight scatter global
        if highlight_line[0] is not None:
            try:
                highlight_line[0].remove()
            except Exception:
                pass
        if sid in all_mean_devs:
            devs = all_mean_devs[sid]
            ln, = ax_all.plot(np.arange(len(devs)), devs * 1000,
                              color="white", lw=2.0, zorder=10)
            highlight_line[0] = ln

        fig.suptitle(
            f"Comparateur sessions — {n_sess} sessions  |  "
            f"Session {sid} ({n} pièces)  —  "
            f"bleu = vrai   orange = prédit",
            color="#ccccff", fontsize=11, fontweight="bold"
        )
        fig.canvas.draw_idle()

    def on_slider(val):
        idx = int(sl.val) - 1
        draw_session(session_ids[idx])

    # Init
    init_idx = session_ids.index(init_sid) + 1 if init_sid in session_ids else 1
    sl.on_changed(on_slider)
    sl.set_val(init_idx)
    if not metrics_cache:   # set_val peut ne pas déclencher si valeur inchangée
        draw_session(init_sid)
    plt.show()


# ── vue principale ─────────────────────────────────────────────────────────────

def view_compare(data_dir=DATASET_DIR, ckpt_dir=CKPT_DIR,
                 init_sid=None, init_piece=None):

    device = get_device()
    print("Chargement du modèle…")
    model, norm, dev_mean, dev_std, target_keys = _load_model(ckpt_dir, device)
    print(f"Modèle chargé  |  target_keys={target_keys}")

    session_ids = _list_sessions(data_dir)
    if not session_ids:
        print(f"Aucune session trouvée dans '{data_dir}'.")
        return

    if init_sid is None or init_sid not in session_ids:
        init_sid = session_ids[0]

    piece_indices = _list_pieces(data_dir, init_sid)
    if not piece_indices:
        print(f"Aucune pièce trouvée pour la session {init_sid}.")
        return
    init_piece = piece_indices[0] if init_piece is None else init_piece

    # ── layout ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 10))
    fig.canvas.manager.set_window_title("Comparateur World Model — Bras Découpe")
    fig.patch.set_facecolor(BG_DARK)
    fig.subplots_adjust(left=0.05, bottom=0.17, right=0.97, top=0.92,
                        wspace=0.32, hspace=0.55)

    gs = fig.add_gridspec(4, 3,
                          left=0.05, right=0.97, top=0.91, bottom=0.17,
                          wspace=0.32, hspace=0.55)

    ax_xy   = fig.add_subplot(gs[:, 0])          # XY bras (gauche, toute la hauteur)
    ax_q1   = fig.add_subplot(gs[0, 1])
    ax_q2   = fig.add_subplot(gs[1, 1])
    ax_dq1  = fig.add_subplot(gs[2, 1])
    ax_dq2  = fig.add_subplot(gs[3, 1])
    ax_dev  = fig.add_subplot(gs[:2, 2])
    ax_def  = fig.add_subplot(gs[2:, 2])

    signal_axes = [ax_q1, ax_q2, ax_dq1, ax_dq2, ax_dev, ax_def]
    for ax in [ax_xy] + signal_axes:
        _style(ax)

    # ── sliders & boutons ──────────────────────────────────────────────────────
    n_sess = len(session_ids)
    max_piece = max(piece_indices)

    ax_sl_s  = fig.add_axes([0.05, 0.10, 0.35, 0.025], facecolor="#2a2a4e")
    ax_sl_p  = fig.add_axes([0.05, 0.06, 0.35, 0.025], facecolor="#2a2a4e")
    ax_btn   = fig.add_axes([0.48, 0.05, 0.09, 0.055], facecolor="#2a2a4e")
    ax_sl_sp = fig.add_axes([0.60, 0.06, 0.33, 0.025], facecolor="#2a2a4e")

    sl_sess  = Slider(ax_sl_s,  "Session", 1, n_sess,     valinit=1, valstep=1,
                      color="#5555cc")
    sl_piece = Slider(ax_sl_p,  "Pièce",   0, max_piece,  valinit=0, valstep=1,
                      color="#cc5555")
    sl_speed = Slider(ax_sl_sp, "Vitesse ×", 1, 20, valinit=1, valstep=1,
                      color="#55aa55")
    btn_play = Button(ax_btn, "▶ Play", color="#2a2a4e", hovercolor="#3a3a6e")

    for sl in [sl_sess, sl_piece, sl_speed]:
        sl.label.set_color(COL_TICK)
        sl.valtext.set_color("#ccccff")
    btn_play.label.set_color("#ccffcc")

    # ── état partagé ───────────────────────────────────────────────────────────
    S = dict(
        sid=init_sid, piece_idx=init_piece,
        result=None, anim=None, playing=False, frame=0,
        # artistes animés
        arm_true=None, arm_pred=None,
        trail_true=None, trail_pred=None,
        time_cursors=[],
    )

    # ── artistes statiques (créés une fois) ───────────────────────────────────
    arm_lim = l1 + l2 + 0.3
    ax_xy.set_xlim(-0.3, arm_lim)
    ax_xy.set_ylim(-0.5, arm_lim)
    ax_xy.set_aspect("equal", "box")
    ax_xy.grid(True, linestyle=":", alpha=0.2, color=COL_GRID)

    from matplotlib.patches import Polygon as MPoly
    piece_patch = MPoly(np.zeros((3, 2)), closed=True,
                        color="gold", alpha=0.18, zorder=1)
    ax_xy.add_patch(piece_patch)
    piece_outline, = ax_xy.plot([], [], "--", color="#888866", lw=1.0, zorder=2)

    # traînées complètes (fond transparent)
    full_true,  = ax_xy.plot([], [], color=COL_TRUE, lw=0.8, alpha=0.25, zorder=3)
    full_pred,  = ax_xy.plot([], [], color=COL_PRED, lw=0.8, alpha=0.25,
                             zorder=3, ls="--")
    # bras animés
    arm_true_ln, = ax_xy.plot([], [], "o-", color=COL_TRUE, lw=2.5, ms=6,
                               zorder=6, label="vrai")
    arm_pred_ln, = ax_xy.plot([], [], "o--", color=COL_PRED, lw=2.5, ms=6,
                               zorder=6, label="prédit")
    # trace courte animée
    trail_true_ln, = ax_xy.plot([], [], color=COL_TRUE, lw=1.5, alpha=0.7, zorder=5)
    trail_pred_ln, = ax_xy.plot([], [], color=COL_PRED, lw=1.5, alpha=0.7,
                                 zorder=5, ls="--")
    time_txt = ax_xy.text(0.03, 0.97, "", transform=ax_xy.transAxes,
                          fontsize=9, va="top", color=COL_TEXT)
    ax_xy.legend(fontsize=9, loc="upper right",
                 facecolor=BG_PANEL, edgecolor=COL_BORDER,
                 labelcolor=COL_TEXT)

    S["arm_true"]   = arm_true_ln
    S["arm_pred"]   = arm_pred_ln
    S["trail_true"] = trail_true_ln
    S["trail_pred"] = trail_pred_ln

    # ── dessin statique d'un résultat ─────────────────────────────────────────

    def _draw_static(result):
        T   = result["T"]
        t   = np.arange(T) * 0.1
        cut = result["cut"]

        # Pièce (waypoints de coupe)
        wp   = result["waypoints"]
        cut_wp = np.array([[x, y] for x, y, c in wp if c > 0.5])
        if len(cut_wp) > 2:
            poly_pts = np.vstack([cut_wp, cut_wp[0]])
            piece_patch.set_xy(poly_pts)
            piece_patch.set_visible(True)
            piece_outline.set_data(poly_pts[:, 0], poly_pts[:, 1])
            # Corridor de tolérance
            xu, yu, xl, yl = _corridor(cut_wp[:, 0], cut_wp[:, 1],
                                        CUT_DEFECT_THRESHOLD)
            ax_xy.fill(np.concatenate([xu, xl[::-1]]),
                       np.concatenate([yu, yl[::-1]]),
                       color="gold", alpha=0.10, zorder=1)
        else:
            piece_patch.set_visible(False)
            piece_outline.set_data([], [])

        full_true.set_data(result["tip_true"][:, 0], result["tip_true"][:, 1])
        full_pred.set_data(result["tip_pred"][:, 0], result["tip_pred"][:, 1])

        # ── signaux ───────────────────────────────────────────────────────────
        for ax in signal_axes:
            ax.clear()
            _style(ax)

        def _plot_sig(ax, true_sig, pred_sig, label, unit=""):
            ax.plot(t, true_sig, color=COL_TRUE,  lw=1.2, label="vrai")
            ax.plot(t, pred_sig, color=COL_PRED,   lw=1.2, ls="--",
                    label="prédit", alpha=0.85)
            ax.set_title(label, fontweight="bold")
            if unit:
                ax.set_ylabel(unit, fontsize=8)
            ax.legend(fontsize=7, loc="upper right",
                      facecolor=BG_PANEL, edgecolor=COL_BORDER,
                      labelcolor=COL_TEXT)
            ax.grid(True, linestyle=":", alpha=0.3, color=COL_GRID)
            return ax.axvline(0, color="white", lw=1.0, alpha=0.6)

        q1_true = np.degrees(result["q_real"][:, 0])
        q1_pred = np.degrees(result["q_pred"][:, 0])
        q2_true = np.degrees(result["q_real"][:, 1])
        q2_pred = np.degrees(result["q_pred"][:, 1])

        vl_q1  = _plot_sig(ax_q1,  q1_true,  q1_pred,  "Angle 1 (°)",    "°")
        vl_q2  = _plot_sig(ax_q2,  q2_true,  q2_pred,  "Angle 2 (°)",    "°")
        vl_dq1 = _plot_sig(ax_dq1, result["dq_real"][:, 0], result["q_pred"][:, 0] * 0,
                            "Vit. ang. 1 (rad/s)", "rad/s")
        # dq non prédit → tracer seulement le vrai
        ax_dq1.clear()
        _style(ax_dq1)
        ax_dq1.plot(t, result["dq_real"][:, 0], color=COL_TRUE, lw=1.2, label="vrai")
        ax_dq1.set_title("Vit. ang. 1 (rad/s)", fontweight="bold")
        ax_dq1.set_ylabel("rad/s", fontsize=8)
        ax_dq1.legend(fontsize=7, loc="upper right",
                      facecolor=BG_PANEL, edgecolor=COL_BORDER, labelcolor=COL_TEXT)
        ax_dq1.grid(True, linestyle=":", alpha=0.3, color=COL_GRID)
        vl_dq1 = ax_dq1.axvline(0, color="white", lw=1.0, alpha=0.6)

        ax_dq2.clear()
        _style(ax_dq2)
        ax_dq2.plot(t, result["dq_real"][:, 1], color=COL_TRUE, lw=1.2, label="vrai")
        ax_dq2.set_title("Vit. ang. 2 (rad/s)", fontweight="bold")
        ax_dq2.set_ylabel("rad/s", fontsize=8)
        ax_dq2.set_xlabel("Temps (s)", fontsize=8)
        ax_dq2.legend(fontsize=7, loc="upper right",
                      facecolor=BG_PANEL, edgecolor=COL_BORDER, labelcolor=COL_TEXT)
        ax_dq2.grid(True, linestyle=":", alpha=0.3, color=COL_GRID)
        vl_dq2 = ax_dq2.axvline(0, color="white", lw=1.0, alpha=0.6)

        # cut_deviation
        ax_dev.plot(t, result["cut_dev_true"] * 1000, color=COL_TRUE,  lw=1.2,
                    label="vrai (mm)")
        ax_dev.plot(t, result["cut_dev_pred"] * 1000, color=COL_PRED,   lw=1.2,
                    ls="--", label="prédit (mm)", alpha=0.85)
        ax_dev.fill_between(t, 0, result["cut_dev_true"] * 1000,
                            where=cut, color=COL_TRUE, alpha=0.05)
        ax_dev.axhline(CUT_DEFECT_THRESHOLD * 1000, color="#ffaa44",
                       lw=1.0, ls=":", label=f"seuil {CUT_DEFECT_THRESHOLD*1000:.0f} mm")
        ax_dev.set_title("Déviation coupe (mm)", fontweight="bold")
        ax_dev.set_ylabel("mm", fontsize=8)
        ax_dev.set_ylim(bottom=0)
        ax_dev.legend(fontsize=7, loc="upper right",
                      facecolor=BG_PANEL, edgecolor=COL_BORDER, labelcolor=COL_TEXT)
        ax_dev.grid(True, linestyle=":", alpha=0.3, color=COL_GRID)
        vl_dev = ax_dev.axvline(0, color="white", lw=1.0, alpha=0.6)

        # cut_defect
        ax_def.plot(t, result["cut_defect"], color=COL_TRUE, lw=1.2,
                    label="vrai", drawstyle="steps-post")
        ax_def.plot(t, result["defect_prob"], color=COL_PRED, lw=1.2, ls="--",
                    label="prob. préd.", alpha=0.85)
        ax_def.axhline(0.5, color="#888888", lw=0.8, ls=":")
        ax_def.set_ylim(-0.05, 1.1)
        ax_def.set_title("Défaut coupe (prob.)", fontweight="bold")
        ax_def.set_ylabel("prob.", fontsize=8)
        ax_def.set_xlabel("Temps (s)", fontsize=8)
        ax_def.legend(fontsize=7, loc="upper right",
                      facecolor=BG_PANEL, edgecolor=COL_BORDER, labelcolor=COL_TEXT)
        ax_def.grid(True, linestyle=":", alpha=0.3, color=COL_GRID)
        vl_def = ax_def.axvline(0, color="white", lw=1.0, alpha=0.6)

        S["time_cursors"] = [vl_q1, vl_q2, vl_dq1, vl_dq2, vl_dev, vl_def]

        # Suptitle
        n_def_true = int(result["cut_defect"][cut].sum()) if cut.any() else 0
        n_def_pred = int((result["defect_prob"][cut] > 0.5).sum()) if cut.any() else 0
        n_cut      = int(cut.sum())
        fig.suptitle(
            f"Session {S['sid']}  |  Pièce {S['piece_idx']}  "
            f"|  n={result['piece_count']}  cad={result['cadence']:.0f} p/h  "
            f"speed={result['speed']:.3f} s/seg  "
            f"|  défauts : vrai {n_def_true}/{n_cut}  préd {n_def_pred}/{n_cut}",
            color="#ccccff", fontsize=10, fontweight="bold"
        )

    # ── mise à jour d'une frame animée ────────────────────────────────────────

    def _update_frame(frame):
        result = S["result"]
        if result is None:
            return
        T   = result["T"]
        i   = min(frame, T - 1)
        t   = i * 0.1

        # bras
        et = result["elbow_true"][i]
        tt = result["tip_true"][i]
        ep = result["elbow_pred"][i]
        tp = result["tip_pred"][i]
        S["arm_true"].set_data([0, et[0], tt[0]], [0, et[1], tt[1]])
        S["arm_pred"].set_data([0, ep[0], tp[0]], [0, ep[1], tp[1]])

        # traces courtes
        s = max(0, i - 15)
        S["trail_true"].set_data(result["tip_true"][s:i, 0],
                                  result["tip_true"][s:i, 1])
        S["trail_pred"].set_data(result["tip_pred"][s:i, 0],
                                  result["tip_pred"][s:i, 1])
        time_txt.set_text(f"t = {t:.2f} s")

        # curseurs temporels
        for vl in S["time_cursors"]:
            vl.set_xdata([t, t])

        fig.canvas.draw_idle()

    # ── chargement + inférence ────────────────────────────────────────────────

    def load_and_draw():
        sid  = S["sid"]
        pidx = S["piece_idx"]

        # Nettoyer les patches corridor précédents (polygones remplis)
        for coll in list(ax_xy.collections):
            coll.remove()
        ax_xy.fill([], [], color="gold", alpha=0.10, zorder=1)  # reset

        result = _run_inference(model, norm, dev_mean, dev_std, target_keys,
                                data_dir, sid, pidx, device)
        if result is None:
            fig.suptitle(
                f"Épisode introuvable : session {sid} pièce {pidx}",
                color="#ff6666"
            )
            S["result"] = None
            fig.canvas.draw_idle()
            return

        S["result"] = result
        S["frame"]  = 0
        _draw_static(result)
        _update_frame(0)
        fig.canvas.draw_idle()

    # ── callbacks ─────────────────────────────────────────────────────────────

    def on_sess(val):
        idx = int(sl_sess.val) - 1
        S["sid"] = session_ids[idx]
        new_pieces = _list_pieces(data_dir, S["sid"])
        if new_pieces:
            S["piece_idx"] = new_pieces[0]
            sl_piece.valmax = max(new_pieces)
            sl_piece.ax.set_xlim(0, max(new_pieces))
            sl_piece.set_val(new_pieces[0])
        _stop_anim()
        load_and_draw()

    def on_piece(val):
        S["piece_idx"] = int(sl_piece.val)
        _stop_anim()
        load_and_draw()

    def _stop_anim():
        S["playing"] = False
        btn_play.label.set_text("▶ Play")
        if S["anim"] is not None:
            S["anim"].event_source.stop()
            S["anim"] = None

    def on_play(event):
        if S["result"] is None:
            return
        T = S["result"]["T"]

        if S["playing"]:
            _stop_anim()
            return

        S["playing"] = True
        btn_play.label.set_text("⏸ Pause")
        S["frame"] = 0

        def anim_func(_):
            if not S["playing"]:
                return
            _update_frame(S["frame"])
            S["frame"] = (S["frame"] + max(1, int(sl_speed.val))) % T

        S["anim"] = animation.FuncAnimation(
            fig, anim_func, interval=100, cache_frame_data=False
        )
        fig.canvas.draw_idle()

    sl_sess.on_changed(on_sess)
    sl_piece.on_changed(on_piece)
    btn_play.on_clicked(on_play)

    # Initialisation
    sidx = session_ids.index(init_sid) + 1 if init_sid in session_ids else 1
    sl_sess.set_val(sidx)
    sl_piece.set_val(init_piece)
    S["sid"]       = init_sid
    S["piece_idx"] = init_piece
    load_and_draw()

    plt.show()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",     choices=["session", "piece"], default="session",
                   help="session = métriques sur toute la session  |  "
                        "piece = animation pièce par pièce")
    p.add_argument("--session",  type=str,  default=None,
                   help="ID de session (ex: 000)")
    p.add_argument("--piece",    type=int,  default=None,
                   help="N° de pièce (mode piece uniquement)")
    p.add_argument("--data-dir", type=str,  default=DATASET_DIR)
    p.add_argument("--ckpt-dir", type=str,  default=CKPT_DIR)
    args = p.parse_args()

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir

    if not os.path.exists(data_dir):
        print(f"Dossier dataset '{data_dir}' introuvable.")
        sys.exit(1)
    if not os.path.exists(ckpt_dir):
        print(f"Dossier checkpoint '{ckpt_dir}' introuvable.")
        sys.exit(1)

    if args.mode == "session":
        view_session_compare(data_dir=data_dir, ckpt_dir=ckpt_dir,
                             init_sid=args.session)
    else:
        view_compare(data_dir=data_dir, ckpt_dir=ckpt_dir,
                     init_sid=args.session, init_piece=args.piece)


if __name__ == "__main__":
    main()
