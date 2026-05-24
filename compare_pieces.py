#!/usr/bin/env python3
"""
Comparateur Découpe → Perçage
Visualise côte-à-côte : résultats réels du bras vs prédictions des world models.
Montre la propagation des erreurs dans la chaîne découpe → perçage.

Usage (depuis la racine du projet) :
    python3 compare_pieces.py                  # toutes les pièces
    python3 compare_pieces.py 1 3 5            # pièces 1, 3, 5 (1-indexé)
    python3 compare_pieces.py --range 1 10     # pièces 1 à 10 inclus
"""

import sys
import os
import glob
import json
import argparse

import numpy as np
import math
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "arm_percage"))
sys.path.insert(0, os.path.join(BASE, "arm_decoupe"))

import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button

# ─── Constantes physiques (identiques dans les deux bras) ────────────────────
L1 = L2 = math.sqrt(2)

DECOUPE_DB    = os.path.join(BASE, "arm_decoupe", "pieces_database.json")
DECOUPE_DSET  = os.path.join(BASE, "arm_decoupe", "dataset")
DECOUPE_CKPT  = os.path.join(BASE, "arm_decoupe", "world_model", "checkpoints", "best_model.pt")
DECOUPE_NORM  = os.path.join(BASE, "arm_decoupe", "world_model", "checkpoints", "normalizer.npz")
PERCAGE_DSET  = os.path.join(BASE, "arm_percage", "dataset")
PERCAGE_CKPT  = os.path.join(BASE, "arm_percage", "world_model", "checkpoints", "best_model.pt")
PERCAGE_NORM  = os.path.join(BASE, "arm_percage", "world_model", "checkpoints", "normalizer.npz")
PERCAGE_TNORM = os.path.join(BASE, "arm_percage", "world_model", "checkpoints", "traj_normalizer.npz")

DRILL_DEFECT_THRESHOLD_MM = 20.0  # mm, cohérent avec record_dataset.py (0.02 m)
_DRILL_INSET          = 0.1      # m  — copié de arm_percage/piece_input.py
_PLACEMENT_NOISE_STD  = 0.002    # m  — idem


# ─── Cinématique directe ─────────────────────────────────────────────────────
def fk(q: np.ndarray) -> np.ndarray:
    """q : (T, 2) → xy : (T, 2)"""
    x = L1 * np.cos(q[:, 0]) + L2 * np.cos(q[:, 0] + q[:, 1])
    y = L1 * np.sin(q[:, 0]) + L2 * np.sin(q[:, 0] + q[:, 1])
    return np.stack([x, y], axis=1)


# ─── Chargement des modèles ───────────────────────────────────────────────────
def _load_decoupe_model():
    from arm_decoupe.world_model.model import WorldModel

    ck = torch.load(DECOUPE_CKPT, map_location="cpu", weights_only=False)
    H = ck["hyperparams"]
    model = WorldModel(
        shape_embed_dim=H["shape_embed_dim"],
        h_dim=H["h_dim"],
        obs_dim=H.get("obs_dim", 2),
        dropout=0.0,
        gru_layers=H.get("gru_layers", 3),
        pe_dim=H.get("pe_dim", 64),
    )
    model.load_state_dict(ck["model_state"])
    model.eval()

    n = np.load(DECOUPE_NORM, allow_pickle=True)
    q_err_mean = n["mean"].astype(np.float32)  # (2,)
    q_err_std  = n["std"].astype(np.float32)   # (2,)
    dev_mean   = float(ck["dev_mean"])
    dev_std    = float(ck["dev_std"])

    return model, q_err_mean, q_err_std, dev_mean, dev_std


def _load_percage_model():
    from arm_percage.world_model.model import DrillWorldModel

    ck = torch.load(PERCAGE_CKPT, map_location="cpu", weights_only=False)
    H = ck["hyperparams"]
    model = DrillWorldModel(
        corner_embed_dim=H.get("corner_embed_dim", 64),
        embed_dim=H.get("embed_dim", 256),
        h_dim=H.get("h_dim", 512),
        pe_dim=H.get("pe_dim", 64),
        gru_layers=H.get("gru_layers", 2),
        n_attn_heads=H.get("n_attn_heads", 4),
        dropout=0.0,
    )
    model.load_state_dict(ck["model_state"])
    model.eval()

    n = np.load(PERCAGE_NORM)
    return model, n["mean"].astype(np.float32), n["std"].astype(np.float32)


# ─── Inférence découpe ────────────────────────────────────────────────────────
@torch.no_grad()
def _predict_decoupe(model, q_err_mean, q_err_std, dev_mean, dev_std,
                     waypoints, speed: float, q_des: np.ndarray):
    T = len(q_des)
    wp = torch.tensor(np.array(waypoints, dtype=np.float32)).unsqueeze(0)
    wp_len = torch.tensor([len(waypoints)])
    spd = torch.tensor([[speed]], dtype=torch.float32)

    traj_norm, quality = model.predict(wp, wp_len, spd, max_len=T)

    T_pred = traj_norm.shape[1]
    T_use  = min(T_pred, T)

    q_err_pred = traj_norm[0, :T_use].numpy() * q_err_std + q_err_mean
    q_real_pred = q_des[:T_use] + q_err_pred

    qual = quality[0, :T_use].numpy()
    dev_pred  = qual[:, 0] * dev_std + dev_mean
    def_prob  = 1.0 / (1.0 + np.exp(-qual[:, 1]))

    return fk(q_real_pred), dev_pred, def_prob, T_use


# ─── Inférence perçage ────────────────────────────────────────────────────────
@torch.no_grad()
def _predict_percage(model, off_mean, off_std, corners: np.ndarray, speed: float):
    corners_t = torch.tensor(corners, dtype=torch.float32).unsqueeze(0)
    spd = torch.tensor([[speed]], dtype=torch.float32)

    _, offsets_norm, defect_logits = model(corners_t, spd)

    offsets    = offsets_norm[0].numpy() * off_std + off_mean
    pred_hits  = corners + offsets
    def_probs  = torch.sigmoid(defect_logits[0]).numpy()
    return pred_hits, def_probs


# ─── Comparateur principal ────────────────────────────────────────────────────
class PieceComparator:

    BG       = "#1a1a2e"
    AX_BG    = "#16213e"
    SPINE_C  = "#444466"
    TICK_C   = "#aaaaaa"
    REAL_C   = "#4488ff"
    MODEL_C  = "#ff8844"
    IDEAL_C  = "#666688"
    DEF_C    = "#ff4444"
    OK_C     = "#44ff88"

    def __init__(self, piece_indices=None):
        """
        piece_indices : liste d'indices 0-basés des pièces à afficher.
                        None = toutes les pièces.
        """
        print("Chargement des données et des modèles…")

        with open(DECOUPE_DB) as f:
            db = json.load(f)
        self.pieces_db = db["pieces"]
        total = len(self.pieces_db)

        if piece_indices is None:
            self.selected = list(range(total))
        else:
            self.selected = [i for i in piece_indices if 0 <= i < total]
            if not self.selected:
                raise ValueError(f"Aucun indice valide dans {piece_indices} (0–{total-1})")

        self.n_pieces = len(self.selected)
        print(f"  {self.n_pieces} pièce(s) sélectionnée(s) sur {total}.")

        # Index des épisodes de découpe par pièce (clé = indice réel dans pieces_db)
        self.decoupe_eps: dict[int, list[str]] = {}
        for path in sorted(glob.glob(os.path.join(DECOUPE_DSET, "episode_*.npz"))):
            fname = os.path.basename(path)
            piece_n = int(fname.split("_")[1]) - 1
            self.decoupe_eps.setdefault(piece_n, []).append(path)
        for k in self.decoupe_eps:
            self.decoupe_eps[k].sort()

        # Épisodes de perçage
        print("  Indexation des épisodes de perçage (peut prendre quelques secondes)…")
        self.percage_eps: list[dict] = []
        ep_paths = sorted(glob.glob(os.path.join(PERCAGE_DSET, "episode_*.npz")))
        for i, path in enumerate(ep_paths):
            if i % 500 == 0:
                print(f"    {i}/{len(ep_paths)}")
            d = np.load(path)
            self.percage_eps.append({
                "path":           path,
                "corner_targets": d["corner_targets"].astype(np.float32),
                "drill_hits":     d["drill_hits"].astype(np.float32),
                "errors":         d["errors"].astype(np.float32),
                "defects":        d["defects"].astype(np.float32),
                "speed":          float(d["duration_per_segment"]),
            })
        self._perc_corners_flat = np.stack(
            [ep["corner_targets"].ravel() for ep in self.percage_eps]
        )
        print(f"  {len(self.percage_eps)} épisodes de perçage indexés.")

        print("  Chargement du modèle de découpe…")
        (self.dec_model, self.dec_q_err_mean, self.dec_q_err_std,
         self.dec_dev_mean, self.dec_dev_std) = _load_decoupe_model()

        print("  Chargement du modèle de perçage…")
        (self.perc_model, self.perc_off_mean, self.perc_off_std) = _load_percage_model()

        # cursor = position dans self.selected
        self.cursor  = 0
        self.run_idx = 0

        # Cache des stats globales (calculé à la demande)
        self._global_stats_cache = None

        self._build_figure()
        self._update()

    # ── Propriété : indice réel de la pièce courante ──────────────────────────
    @property
    def piece_idx(self) -> int:
        return self.selected[self.cursor]

    # ── Recherche du meilleur épisode de perçage correspondant ────────────────
    def _best_percage(self, corners: np.ndarray):
        query = corners.ravel()
        dists = np.linalg.norm(self._perc_corners_flat - query, axis=1)
        idx   = int(np.argmin(dists))
        return self.percage_eps[idx], float(dists[idx])

    # ── Construction de la figure ─────────────────────────────────────────────
    def _build_figure(self):
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.patch.set_facecolor(self.BG)

        gs = gridspec.GridSpec(
            3, 2,
            figure=self.fig,
            left=0.06, right=0.97,
            top=0.92, bottom=0.09,
            hspace=0.42, wspace=0.28,
            height_ratios=[3, 2, 0.35],
        )

        self.ax_dec        = self.fig.add_subplot(gs[0, 0])
        self.ax_perc       = self.fig.add_subplot(gs[0, 1])
        self.ax_dec_stats  = self.fig.add_subplot(gs[1, 0])
        self.ax_perc_stats = self.fig.add_subplot(gs[1, 1])

        self._style_axes()

        bkw = dict(color="#333355", hovercolor="#555588")
        self.btn_pp    = Button(self.fig.add_axes([0.06, 0.02, 0.07, 0.04]), "◀◀ Pièce", **bkw)
        self.btn_pn    = Button(self.fig.add_axes([0.15, 0.02, 0.07, 0.04]), "Pièce ▶▶", **bkw)
        self.btn_rp    = Button(self.fig.add_axes([0.40, 0.02, 0.07, 0.04]), "◀ Run",    **bkw)
        self.btn_rn    = Button(self.fig.add_axes([0.49, 0.02, 0.07, 0.04]), "Run ▶",    **bkw)
        self.btn_glob  = Button(self.fig.add_axes([0.78, 0.02, 0.14, 0.04]), "Stats globales", **bkw)

        for btn in [self.btn_pp, self.btn_pn, self.btn_rp, self.btn_rn, self.btn_glob]:
            btn.label.set_color("white")
            btn.label.set_fontsize(9)

        self.btn_pp.on_clicked(lambda _: self._nav_piece(-1))
        self.btn_pn.on_clicked(lambda _: self._nav_piece(+1))
        self.btn_rp.on_clicked(lambda _: self._nav_run(-1))
        self.btn_rn.on_clicked(lambda _: self._nav_run(+1))
        self.btn_glob.on_clicked(lambda _: self._show_global_stats())

        self.suptitle = self.fig.suptitle("", color="white", fontsize=11, fontweight="bold")

    def _style_axes(self):
        for ax in [self.ax_dec, self.ax_perc, self.ax_dec_stats, self.ax_perc_stats]:
            ax.set_facecolor(self.AX_BG)
            ax.tick_params(colors=self.TICK_C, labelsize=8)
            for s in ax.spines.values():
                s.set_color(self.SPINE_C)

    # ── Navigation ────────────────────────────────────────────────────────────
    def _nav_piece(self, delta: int):
        self.cursor  = (self.cursor + delta) % self.n_pieces
        self.run_idx = 0
        self._update()

    def _nav_run(self, delta: int):
        n = len(self.decoupe_eps.get(self.piece_idx, []))
        if n:
            self.run_idx = (self.run_idx + delta) % n
        self._update()

    # ── Mise à jour complète ──────────────────────────────────────────────────
    def _update(self):
        p = self.piece_idx
        r = self.run_idx
        waypoints = self.pieces_db[p]
        runs = self.decoupe_eps.get(p, [])
        n_runs = len(runs)

        self.suptitle.set_text(
            f"Pièce {p + 1:02d}  [{self.cursor + 1}/{self.n_pieces}]"
            f"   Run {r + 1}/{n_runs}"
            f"   —   Comparaison bras réel vs world models"
        )

        if runs:
            with np.load(runs[r]) as dec_data:
                q_real   = dec_data["q_real"].astype(np.float64)
                q_des    = dec_data["q_des"].astype(np.float64)
                is_cut   = dec_data["is_cutting"].astype(bool)
                cut_dev  = dec_data["cut_deviation"].astype(np.float64)
                cut_def  = dec_data["cut_defect"].astype(np.float64)
                speed_d  = float(dec_data["duration_per_segment"])
            T = len(q_real)
        else:
            q_real = q_des = is_cut = cut_dev = cut_def = None
            speed_d = 2.0
            T = 400

        dec_pred = None
        if q_des is not None:
            xy_pred, dev_pred, def_prob, T_use = _predict_decoupe(
                self.dec_model, self.dec_q_err_mean, self.dec_q_err_std,
                self.dec_dev_mean, self.dec_dev_std,
                waypoints, speed_d, q_des,
            )
            is_cut_use = is_cut[:T_use]
            dec_pred = (xy_pred, dev_pred, def_prob, T_use, is_cut_use)

        real_corners = self._extract_corners(runs, r, waypoints)

        if dec_pred is not None and q_des is not None:
            xy_pred_cut, _, _, T_use, _ = dec_pred
            model_corners = self._extract_model_corners(
                waypoints, xy_pred_cut, q_des[:T_use]
            )
        else:
            model_corners = real_corners.copy()

        best_ep, _ = self._best_percage(real_corners.astype(np.float32))
        perc_speed = best_ep["speed"]

        pred_hits, pred_def_probs = _predict_percage(
            self.perc_model, self.perc_off_mean, self.perc_off_std,
            model_corners.astype(np.float32), perc_speed,
        )
        pred_errors_mm = np.linalg.norm(pred_hits - model_corners, axis=1) * 1000.0

        self._draw_decoupe(waypoints, q_real, is_cut, dec_pred)
        self._draw_percage(real_corners, model_corners, q_real, is_cut,
                           best_ep, pred_hits, pred_def_probs)
        self._draw_decoupe_stats(cut_dev, cut_def, is_cut, dec_pred)
        self._draw_percage_stats(real_corners, model_corners, best_ep,
                                 pred_errors_mm, pred_def_probs)

        self.fig.canvas.draw_idle()

    # ── Extraction des coins réels ────────────────────────────────────────────
    def _extract_corners(self, runs, r, waypoints):
        if runs:
            try:
                import piece_input  # type: ignore[import]
                rng = np.random.default_rng(42)
                return piece_input.extract_corners(runs[r], DECOUPE_DB, rng=rng)
            except ImportError:
                pass
        cut_pts = [(wp[0], wp[1]) for wp in waypoints if wp[2]]
        if len(cut_pts) >= 4:
            return np.array(cut_pts[:4], dtype=np.float64)
        return np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]], dtype=np.float64)

    # ── Extraction des coins "rêvés" ──────────────────────────────────────────
    def _extract_model_corners(self, waypoints, xy_pred: np.ndarray, q_des: np.ndarray):
        ideal = []
        seen: set = set()
        for wp in waypoints:
            if wp[2]:
                pt = (round(float(wp[0]), 9), round(float(wp[1]), 9))
                if pt not in seen:
                    seen.add(pt)
                    ideal.append([float(wp[0]), float(wp[1])])
        if len(ideal) < 4:
            return np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]], dtype=np.float64)
        ideal_corners = np.array(ideal[:4], dtype=np.float64)

        ee_des = fk(q_des)
        T_pred = len(xy_pred)

        model_corners = np.zeros((4, 2))
        for i, corner in enumerate(ideal_corners):
            dists = np.linalg.norm(ee_des - corner, axis=1)
            t     = int(np.argmin(dists))
            model_corners[i] = xy_pred[min(t, T_pred - 1)]

        center = model_corners.mean(axis=0)
        for i in range(4):
            d = center - model_corners[i]
            n = np.linalg.norm(d)
            if n > 1e-9:
                model_corners[i] += (d / n) * _DRILL_INSET

        rng = np.random.default_rng(42)
        model_corners += rng.normal(0.0, _PLACEMENT_NOISE_STD, size=(4, 2))

        return model_corners

    # ── Panel découpe ─────────────────────────────────────────────────────────
    def _draw_decoupe(self, waypoints, q_real, is_cut, dec_pred):
        ax = self.ax_dec
        ax.cla()
        ax.set_facecolor(self.AX_BG)
        ax.set_aspect("equal")
        ax.set_title("DÉCOUPE", color="white", fontsize=10, pad=4)
        ax.set_xlabel("x (m)", color=self.TICK_C, fontsize=8)
        ax.set_ylabel("y (m)", color=self.TICK_C, fontsize=8)
        ax.tick_params(colors=self.TICK_C, labelsize=7)

        cut_pts = [(wp[0], wp[1]) for wp in waypoints if wp[2]]
        if len(cut_pts) > 1:
            xs, ys = zip(*cut_pts)
            ax.plot(list(xs) + [xs[0]], list(ys) + [ys[0]],
                    "--", color=self.IDEAL_C, lw=1.2, label="Idéal", zorder=1)

        if q_real is not None:
            xy_r = fk(q_real)
            m = is_cut & np.isfinite(xy_r[:, 0])
            ax.scatter(xy_r[m, 0], xy_r[m, 1],
                       c=self.REAL_C, s=3, alpha=0.7, label="Bras réel", zorder=2)

        if dec_pred is not None:
            xy_p, _, _, _, is_cut_use = dec_pred
            m = is_cut_use & np.isfinite(xy_p[:, 0])
            ax.scatter(xy_p[m, 0], xy_p[m, 1],
                       c=self.MODEL_C, s=3, alpha=0.7, label="World model", zorder=3)

        ax.legend(fontsize=7, labelcolor="white", framealpha=0.3,
                  facecolor="#333355", loc="upper right")
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(0.0, 2.0)
        for s in ax.spines.values():
            s.set_color(self.SPINE_C)

    # ── Panel perçage ─────────────────────────────────────────────────────────
    def _draw_percage(self, real_corners, model_corners, q_real, is_cut,
                      best_ep, pred_hits, pred_def_probs):
        ax = self.ax_perc
        ax.cla()
        ax.set_facecolor(self.AX_BG)
        ax.set_aspect("equal")
        ax.set_title("PERÇAGE", color="white", fontsize=10, pad=4)
        ax.set_xlabel("x (m)", color=self.TICK_C, fontsize=8)
        ax.set_ylabel("y (m)", color=self.TICK_C, fontsize=8)
        ax.tick_params(colors=self.TICK_C, labelsize=7)

        if q_real is not None:
            xy_r = fk(q_real)
            ax.scatter(xy_r[is_cut, 0], xy_r[is_cut, 1],
                       c=self.IDEAL_C, s=1, alpha=0.35, zorder=1, label="Contour pièce")

        ax.scatter(real_corners[:, 0], real_corners[:, 1],
                   marker="s", s=25, c=self.REAL_C, zorder=3, label="Coins réels")
        for i, (cx, cy) in enumerate(real_corners):
            ax.annotate(f"{i+1}", (cx, cy), color="#aaccff", fontsize=6,
                        xytext=(3, 3), textcoords="offset points")

        ax.scatter(model_corners[:, 0], model_corners[:, 1],
                   marker="s", s=25, facecolors="none",
                   edgecolors=self.MODEL_C, linewidths=1.2,
                   zorder=3, label="Coins modèle")

        real_hits = best_ep["drill_hits"]
        real_defs = best_ep["defects"]
        colors_r  = [self.DEF_C if d else self.OK_C for d in real_defs]
        ax.scatter(real_hits[:, 0], real_hits[:, 1],
                   marker="o", s=35, c=colors_r, zorder=4, label="Trous réels")
        for i in range(4):
            ax.plot([real_corners[i, 0], real_hits[i, 0]],
                    [real_corners[i, 1], real_hits[i, 1]],
                    color=self.REAL_C, lw=0.7, alpha=0.6)

        colors_p = [self.DEF_C if p > 0.5 else self.MODEL_C for p in pred_def_probs]
        ax.scatter(pred_hits[:, 0], pred_hits[:, 1],
                   marker="^", s=35, c=colors_p, zorder=5, label="Trous modèle")
        for i in range(4):
            ax.plot([model_corners[i, 0], pred_hits[i, 0]],
                    [model_corners[i, 1], pred_hits[i, 1]],
                    color=self.MODEL_C, lw=0.7, alpha=0.6)

        ax.legend(fontsize=7, labelcolor="white", framealpha=0.3,
                  facecolor="#333355", loc="upper right")

        all_x = np.concatenate([real_corners[:, 0], model_corners[:, 0],
                                 real_hits[:, 0], pred_hits[:, 0]])
        all_y = np.concatenate([real_corners[:, 1], model_corners[:, 1],
                                 real_hits[:, 1], pred_hits[:, 1]])
        pad = 0.20
        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
        for s in ax.spines.values():
            s.set_color(self.SPINE_C)

    # ── Stats découpe ─────────────────────────────────────────────────────────
    def _draw_decoupe_stats(self, cut_dev, cut_def, is_cut, dec_pred):
        ax = self.ax_dec_stats
        ax.cla()
        ax.set_facecolor(self.AX_BG)
        ax.tick_params(colors=self.TICK_C, labelsize=8)
        ax.set_title("Stats Découpe — déviation de coupe", color="white", fontsize=9, pad=4)

        if cut_dev is None or dec_pred is None:
            ax.text(0.5, 0.5, "Aucune donnée", ha="center", va="center",
                    color=self.TICK_C, transform=ax.transAxes)
            for s in ax.spines.values():
                s.set_color(self.SPINE_C)
            return

        _, dev_pred, def_prob, T_use, is_cut_use = dec_pred
        mask_real = is_cut
        mask_pred = is_cut_use

        if not mask_real.any() or not mask_pred.any():
            ax.text(0.5, 0.5, "Aucun point de coupe dans cet épisode",
                    ha="center", va="center", color=self.TICK_C, transform=ax.transAxes)
            for s in ax.spines.values():
                s.set_color(self.SPINE_C)
            return

        r_dev_mm = cut_dev[mask_real] * 1000.0
        p_dev_mm = dev_pred[mask_pred] * 1000.0

        labels = ["Réel\nmoy (mm)", "Réel\nmax (mm)", "Modèle\nmoy (mm)", "Modèle\nmax (mm)"]
        vals   = [r_dev_mm.mean(), r_dev_mm.max(), p_dev_mm.mean(), p_dev_mm.max()]
        colors = [self.REAL_C, "#2255aa", self.MODEL_C, "#aa5522"]

        xs = range(len(labels))
        bars = ax.bar(xs, vals, color=colors, alpha=0.82, width=0.6)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, color=self.TICK_C, fontsize=8)
        ax.set_ylabel("mm", color=self.TICK_C, fontsize=8)

        label_offset = max(r_dev_mm.max(), p_dev_mm.max()) * 0.02
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + label_offset,
                    f"{h:.2f}", ha="center", va="bottom", color="white", fontsize=7)

        r_def_pct = cut_def[mask_real].mean() * 100
        p_def_pct = (def_prob[mask_pred] > 0.5).mean() * 100
        ax.text(0.98, 0.97,
                f"Taux défauts réels : {r_def_pct:.1f}%\n"
                f"Taux défauts modèle : {p_def_pct:.1f}%",
                transform=ax.transAxes, ha="right", va="top",
                color="#dddddd", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="#222244", alpha=0.75))
        for s in ax.spines.values():
            s.set_color(self.SPINE_C)

    # ── Stats perçage ─────────────────────────────────────────────────────────
    def _draw_percage_stats(self, real_corners, model_corners, best_ep,
                            pred_errors_mm, pred_def_probs):
        ax = self.ax_perc_stats
        ax.cla()
        ax.set_facecolor(self.AX_BG)
        ax.tick_params(colors=self.TICK_C, labelsize=8)
        ax.set_title("Stats Perçage — erreur par coin (mm)", color="white", fontsize=9, pad=4)

        real_errs_mm = best_ep["errors"] * 1000.0
        real_defs    = best_ep["defects"]
        pred_defs    = (pred_def_probs > 0.5).astype(float)
        corner_delta_mm = np.linalg.norm(model_corners - real_corners, axis=1) * 1000.0

        x_pos = np.arange(4)
        w = 0.28
        ax.bar(x_pos - w, real_errs_mm, w,
               color=self.REAL_C, alpha=0.82, label="Bras réel (vs coins réels)")
        ax.bar(x_pos,      pred_errors_mm, w,
               color=self.MODEL_C, alpha=0.82, label="World model (vs coins modèle)")
        ax.bar(x_pos + w,  corner_delta_mm, w,
               color="#aaaaff", alpha=0.65, label="Δ coins réel↔modèle")

        ax.axhline(DRILL_DEFECT_THRESHOLD_MM, color=self.DEF_C, lw=1.2,
                   linestyle="--", alpha=0.8, label=f"Seuil ({DRILL_DEFECT_THRESHOLD_MM:.0f} mm)")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Coin 1", "Coin 2", "Coin 3", "Coin 4"],
                           color=self.TICK_C, fontsize=8)
        ax.set_ylabel("mm", color=self.TICK_C, fontsize=8)
        ax.legend(fontsize=7, labelcolor="white", framealpha=0.3, facecolor="#333355")

        r_nd = int(real_defs.sum())
        p_nd = int(pred_defs.sum())
        ax.text(0.98, 0.97,
                f"Défauts réels   : {r_nd}/4\n"
                f"Défauts modèle : {p_nd}/4\n"
                f"Δ coins moy    : {corner_delta_mm.mean():.1f} mm",
                transform=ax.transAxes, ha="right", va="top",
                color="#dddddd", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="#222244", alpha=0.75))
        for s in ax.spines.values():
            s.set_color(self.SPINE_C)

    # ── Coins idéaux depuis la DB (position cible de conception) ─────────────
    def _ideal_corners(self, waypoints) -> np.ndarray:
        seen: set = set()
        pts = []
        for wp in waypoints:
            if wp[2]:
                pt = (round(float(wp[0]), 9), round(float(wp[1]), 9))
                if pt not in seen:
                    seen.add(pt)
                    pts.append([float(wp[0]), float(wp[1])])
        if len(pts) >= 4:
            return np.array(pts[:4], dtype=np.float64)
        return np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]], dtype=np.float64)

    # ── Calcul des stats globales (toutes les pièces sélectionnées) ───────────
    def _compute_global_stats(self):
        """
        Compare deux pipelines complets bout-en-bout :

        Pipeline réel  : bras coupe → coins réels  → bras perce → trous réels
        Pipeline rêvé  : world model coupe → coins prédits → world model perce → trous prédits

        Métrique clé : erreur finale des trous vs position idéale (cible de conception).
        """
        stats = {
            "piece_ids":              [],
            # Découpe
            "dec_dev_real_mean":      [],   # déviation réelle moyenne
            "dec_dev_model_mean":     [],   # déviation prédite moyenne
            "dec_def_real_pct":       [],   # taux défauts réels
            "dec_def_model_pct":      [],   # taux défauts prédits
            # Perçage — performance du vrai bras vs position espérée
            "perc_real_error_mean":   [],   # ||trous_réels - coins_cibles||
            # Perçage — écart réel vs imaginé (bout-en-bout)
            "perc_hit_error_mean":    [],   # ||trous_réels - trous_prédits||
            # Perçage — défauts
            "perc_def_real_pct":      [],
            "perc_def_model_pct":     [],
            # Propagation découpe → perçage
            "perc_corner_delta":      [],   # ||coins_réels - coins_prédits||
        }

        n = len(self.selected)
        for idx, p in enumerate(self.selected):
            print(f"  Stats globales : pièce {p+1:02d}  [{idx+1}/{n}]")
            waypoints = self.pieces_db[p]
            runs      = self.decoupe_eps.get(p, [])
            stats["piece_ids"].append(p + 1)

            # ── Pipeline réel : découpe ───────────────────────────────────────
            if not runs:
                stats["dec_dev_real_mean"].append(np.nan)
                stats["dec_dev_model_mean"].append(np.nan)
                stats["dec_def_real_pct"].append(np.nan)
                stats["dec_def_model_pct"].append(np.nan)
                real_corners  = self._extract_corners([], 0, waypoints)
                model_corners = real_corners.copy()
            else:
                run_path = runs[0]
                with np.load(run_path) as d:
                    q_des   = d["q_des"].astype(np.float64)
                    is_cut  = d["is_cutting"].astype(bool)
                    cut_dev = d["cut_deviation"].astype(np.float64)
                    cut_def = d["cut_defect"].astype(np.float64)
                    speed_d = float(d["duration_per_segment"])

                # Pipeline rêvé : inférence world model découpe
                xy_pred, dev_pred, def_prob, T_use = _predict_decoupe(
                    self.dec_model, self.dec_q_err_mean, self.dec_q_err_std,
                    self.dec_dev_mean, self.dec_dev_std,
                    waypoints, speed_d, q_des,
                )
                is_cut_use = is_cut[:T_use]

                mask_r = is_cut
                mask_p = is_cut_use

                r_dev = cut_dev[mask_r] * 1000.0 if mask_r.any() else np.array([np.nan])
                p_dev = dev_pred[mask_p] * 1000.0 if mask_p.any() else np.array([np.nan])

                stats["dec_dev_real_mean"].append(np.nanmean(r_dev))
                stats["dec_dev_model_mean"].append(np.nanmean(p_dev))
                stats["dec_def_real_pct"].append(
                    cut_def[mask_r].mean() * 100 if mask_r.any() else np.nan)
                stats["dec_def_model_pct"].append(
                    (def_prob[mask_p] > 0.5).mean() * 100 if mask_p.any() else np.nan)

                real_corners  = self._extract_corners(runs, 0, waypoints)
                model_corners = self._extract_model_corners(waypoints, xy_pred, q_des[:T_use])

            # ── Pipeline réel : perçage ───────────────────────────────────────
            best_ep, _    = self._best_percage(real_corners.astype(np.float32))
            perc_speed    = best_ep["speed"]
            real_hits     = best_ep["drill_hits"]          # (4, 2)

            # ── Pipeline rêvé : perçage ───────────────────────────────────────
            pred_hits, pred_def_probs = _predict_percage(
                self.perc_model, self.perc_off_mean, self.perc_off_std,
                model_corners.astype(np.float32), perc_speed,
            )
            # Performance du vrai bras : trous réels vs coins cibles (position espérée)
            real_error_mm   = best_ep["errors"].astype(np.float64) * 1000.0

            # Écart bout-en-bout : trous réels vs trous imaginés par le WM
            hit_error_mm    = np.linalg.norm(real_hits - pred_hits, axis=1) * 1000.0

            # Propagation : décalage entre coins réels et coins prédits par le WM découpe
            corner_delta_mm = np.linalg.norm(model_corners - real_corners, axis=1) * 1000.0

            stats["perc_real_error_mean"].append(real_error_mm.mean())
            stats["perc_hit_error_mean"].append(hit_error_mm.mean())
            stats["perc_def_real_pct"].append(best_ep["defects"].mean() * 100)
            stats["perc_def_model_pct"].append((pred_def_probs > 0.5).mean() * 100)
            stats["perc_corner_delta"].append(corner_delta_mm.mean())

        return stats

    # ── Fenêtre stats globales ────────────────────────────────────────────────
    def _show_global_stats(self):
        if self._global_stats_cache is None:
            print("Calcul des stats globales…")
            self._global_stats_cache = self._compute_global_stats()
            print("  Terminé.")

        s   = self._global_stats_cache
        ids = np.array(s["piece_ids"])
        x   = np.arange(len(ids))

        fig = plt.figure(figsize=(18, 11))
        fig.patch.set_facecolor(self.BG)
        fig.suptitle(
            f"Stats globales — {self.n_pieces} pièce(s)"
            f"   |   Pipeline réel  vs  Pipeline world-model (bout-en-bout)",
            color="white", fontsize=12, fontweight="bold",
        )

        gs = gridspec.GridSpec(
            3, 2, figure=fig,
            left=0.06, right=0.97,
            top=0.91, bottom=0.07,
            hspace=0.60, wspace=0.32,
        )

        axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
        for ax in axes:
            ax.set_facecolor(self.AX_BG)
            ax.tick_params(colors=self.TICK_C, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(self.SPINE_C)

        tick_kw = dict(color=self.TICK_C, fontsize=7, rotation=45, ha="right")
        xlabels = [f"P{i}" for i in ids]
        LREAL   = "Pipeline réel"
        LMODEL  = "Pipeline WM"

        def _bar2(ax, vals_r, vals_m, title, ylabel, threshold=None):
            w  = 0.38
            vr = np.array(vals_r, dtype=float)
            vm = np.array(vals_m, dtype=float)
            ax.bar(x - w/2, vr, w, color=self.REAL_C,  alpha=0.82, label=LREAL)
            ax.bar(x + w/2, vm, w, color=self.MODEL_C, alpha=0.82, label=LMODEL)
            if threshold is not None:
                ax.axhline(threshold, color=self.DEF_C, lw=1.2, ls=":",
                           alpha=0.85, label=f"Seuil {threshold:.0f} mm")
            ax.set_title(title, color="white", fontsize=9, pad=4)
            ax.set_ylabel(ylabel, color=self.TICK_C, fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, **tick_kw)
            ax.legend(fontsize=7, labelcolor="white", framealpha=0.3, facecolor="#333355")
            mr = np.nanmean(vr); mm = np.nanmean(vm)
            ax.axhline(mr, color=self.REAL_C,  lw=1.0, ls="--", alpha=0.55)
            ax.axhline(mm, color=self.MODEL_C, lw=1.0, ls="--", alpha=0.55)
            ax.text(0.01, 0.97,
                    f"moy réel={mr:.2f}   moy WM={mm:.2f}",
                    transform=ax.transAxes, color="#cccccc", fontsize=7,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round", facecolor="#222244", alpha=0.7))

        # ── Panel 0 : performance réelle + écart WM ───────────────────────────
        # Barre bleue  : ||trous_réels − coins_cibles||  (vrai bras vs position espérée)
        # Barre orange : ||trous_réels − trous_WM||      (écart réel↔WM, bout-en-bout)
        re_ = np.array(s["perc_real_error_mean"], dtype=float)
        he  = np.array(s["perc_hit_error_mean"],  dtype=float)
        w0  = 0.38
        axes[0].bar(x - w0/2, re_, w0, color=self.REAL_C,  alpha=0.82,
                    label="Bras réel vs cible  (||trous_réels − coins_cibles||)")
        axes[0].bar(x + w0/2, he,  w0, color=self.MODEL_C, alpha=0.82,
                    label="Réel vs WM           (||trous_réels − trous_WM||)")
        axes[0].axhline(np.nanmean(re_), color=self.REAL_C,  lw=1.0, ls="--", alpha=0.55)
        axes[0].axhline(np.nanmean(he),  color=self.MODEL_C, lw=1.0, ls="--", alpha=0.55)
        axes[0].axhline(DRILL_DEFECT_THRESHOLD_MM, color=self.DEF_C, lw=1.2,
                        ls=":", alpha=0.85, label=f"Seuil {DRILL_DEFECT_THRESHOLD_MM:.0f} mm")
        axes[0].set_title(
            "Perçage — erreur vs cible (bleu) et écart réel↔WM (orange)\n"
            "[ barre bleue basse = bras précis  |  barre orange basse = WM fidèle ]",
            color="white", fontsize=9, pad=4)
        axes[0].set_ylabel("mm", color=self.TICK_C, fontsize=8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(xlabels, **tick_kw)
        axes[0].legend(fontsize=7, labelcolor="white", framealpha=0.3,
                       facecolor="#333355", loc="upper right")
        axes[0].text(0.01, 0.97,
                     f"moy bras={np.nanmean(re_):.2f} mm   moy WM={np.nanmean(he):.2f} mm",
                     transform=axes[0].transAxes, color="#cccccc", fontsize=7,
                     va="top", ha="left",
                     bbox=dict(boxstyle="round", facecolor="#222244", alpha=0.7))
        for sp in axes[0].spines.values():
            sp.set_color(self.SPINE_C)

        # ── Panel 1 : taux de défauts perçage ─────────────────────────────────
        _bar2(axes[1],
              s["perc_def_real_pct"], s["perc_def_model_pct"],
              "Taux de défauts perçage (%)", "%")

        # ── Panel 2 : déviation de coupe ──────────────────────────────────────
        _bar2(axes[2],
              s["dec_dev_real_mean"], s["dec_dev_model_mean"],
              "Découpe — déviation de coupe moyenne (mm)", "mm")

        # ── Panel 3 : taux de défauts découpe ─────────────────────────────────
        _bar2(axes[3],
              s["dec_def_real_pct"], s["dec_def_model_pct"],
              "Découpe — taux de défauts (%)", "%")

        # ── Panel 4 : propagation Δ coins ─────────────────────────────────────
        cd = np.array(s["perc_corner_delta"], dtype=float)
        axes[4].bar(x, cd, color="#aaaaff", alpha=0.8)
        axes[4].axhline(np.nanmean(cd), color="#aaaaff", lw=1.2, ls="--", alpha=0.7)
        axes[4].set_title(
            "Propagation découpe → perçage\n"
            "Δ coins réels↔coins WM (mm)",
            color="white", fontsize=9, pad=4)
        axes[4].set_ylabel("mm", color=self.TICK_C, fontsize=8)
        axes[4].set_xticks(x)
        axes[4].set_xticklabels(xlabels, **tick_kw)
        axes[4].text(0.01, 0.97, f"moy={np.nanmean(cd):.2f} mm",
                     transform=axes[4].transAxes, color="#cccccc", fontsize=7,
                     va="top", ha="left",
                     bbox=dict(boxstyle="round", facecolor="#222244", alpha=0.7))
        for sp in axes[4].spines.values():
            sp.set_color(self.SPINE_C)

        # ── Panel 5 : scatter Δcoins vs écart trous (propagation d'erreur) ─────
        # Montre si l'erreur du WM découpe (Δcoins) explique l'écart final de perçage
        ax5   = axes[5]
        valid = np.isfinite(cd) & np.isfinite(he)
        if valid.any():
            ax5.scatter(cd[valid], he[valid], c=self.MODEL_C, s=40, alpha=0.85, zorder=3)
            if valid.sum() >= 2:
                corr = np.corrcoef(cd[valid], he[valid])[0, 1]
                ax5.text(0.97, 0.05, f"r = {corr:.3f}",
                         transform=ax5.transAxes, color="#cccccc", fontsize=9,
                         ha="right", va="bottom",
                         bbox=dict(boxstyle="round", facecolor="#222244", alpha=0.7))
            ax5.axhline(DRILL_DEFECT_THRESHOLD_MM, color=self.DEF_C, lw=1.0,
                        ls=":", alpha=0.7, label=f"Seuil {DRILL_DEFECT_THRESHOLD_MM:.0f} mm")
        ax5.set_title(
            "Δ coins WM  vs  écart trous réels↔WM\n"
            "[ r élevé → erreur découpe explique l'écart de perçage ]",
            color="white", fontsize=9, pad=4)
        ax5.set_xlabel("Δ coins réels↔WM (mm)", color=self.TICK_C, fontsize=8)
        ax5.set_ylabel("Écart trous réels↔WM (mm)", color=self.TICK_C, fontsize=8)
        ax5.legend(fontsize=7, labelcolor="white", framealpha=0.3, facecolor="#333355")
        for sp in ax5.spines.values():
            sp.set_color(self.SPINE_C)

        fig.canvas.draw_idle()
        fig.show()


# ─── Point d'entrée ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparateur Découpe → Perçage avec visualisation world-model."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pieces", nargs="+", type=int, metavar="N",
        help="Numéros de pièces à afficher (1-indexé). Exemples : --pieces 1 3 5"
    )
    group.add_argument(
        "--range", nargs=2, type=int, metavar=("DEBUT", "FIN"),
        help="Plage de pièces inclusive (1-indexé). Exemple : --range 1 10"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.range:
        a, b = args.range
        piece_indices = list(range(a - 1, b))
    elif args.pieces:
        piece_indices = [n - 1 for n in args.pieces]
    else:
        piece_indices = None  # toutes les pièces

    comp = PieceComparator(piece_indices=piece_indices)
    plt.show()


if __name__ == "__main__":
    main()
