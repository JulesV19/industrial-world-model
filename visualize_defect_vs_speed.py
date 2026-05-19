"""
Visualise la distribution du taux de défauts (ou vs vitesse) par forme.

Usage:
    python3 visualize_defect_vs_speed.py                              # scatter défauts vs vitesse
    python3 visualize_defect_vs_speed.py --plot dist                  # distribution des % de défauts
    python3 visualize_defect_vs_speed.py --plot dist --mode compare   # réel + prédictions modèle
    python3 visualize_defect_vs_speed.py --mode compare --model-path world_model/checkpoints/best_model.pt
"""

import argparse
import glob
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from config import l1, l2
from record_dataset import CUT_DEFECT_THRESHOLD

SHAPE_ORDER  = ["Square", "Circle", "Trapezoid"]
SHAPE_COLORS = {"Square": "#4CAF50", "Circle": "#2196F3", "Trapezoid": "#FF9800"}


# ── geometry helpers ───────────────────────────────────────────────────────────

def fk_batch(q: np.ndarray) -> np.ndarray:
    """q: (T, 2) → (T, 2) cartesian end-effector positions."""
    x = l1 * np.cos(q[:, 0]) + l2 * np.cos(q[:, 0] + q[:, 1])
    y = l1 * np.sin(q[:, 0]) + l2 * np.sin(q[:, 0] + q[:, 1])
    return np.stack([x, y], axis=1)


def detect_shape(waypoints: list) -> str:
    """
    Infer shape type from piece waypoints [[x, y, is_cutting], ...].
    Circle has 24 cutting waypoints; Square has 4 equal sides; Trapezoid has 4 unequal.
    """
    cutting = [wp for wp in waypoints if wp[2]]
    if len(cutting) >= 10:
        return "Circle"
    # [close(=pt0), pt1, pt2, pt3] — the close waypoint equals the first vertex
    pts = [(wp[0], wp[1]) for wp in cutting]
    p = [pts[-1], pts[0], pts[1], pts[2]]          # reorder to [pt0, pt1, pt2, pt3]
    sides = [math.hypot(p[(i+1)%4][0] - p[i][0],
                        p[(i+1)%4][1] - p[i][1]) for i in range(4)]
    mean_s = sum(sides) / 4
    rel_dev = max(abs(s - mean_s) / mean_s for s in sides)
    return "Square" if rel_dev < 0.1 else "Trapezoid"


# ── data loading ───────────────────────────────────────────────────────────────

def load_real_records(data_dir: str, piece_db: list) -> list[dict]:
    records  = []
    skipped  = 0
    n_pieces = len(piece_db)
    for ep_path in sorted(glob.glob(os.path.join(data_dir, "episode_*.npz"))):
        piece_idx = int(os.path.basename(ep_path).split("_")[1]) - 1
        if piece_idx >= n_pieces:
            skipped += 1
            continue
        data = np.load(ep_path)

        is_cutting = data["is_cutting"]
        cut_defect = data["cut_defect"]
        speed      = float(data["duration_per_segment"])
        cut_mask   = is_cutting == 1.0
        n_cut      = cut_mask.sum()
        defect_pct = 100.0 * cut_defect[cut_mask].sum() / max(1, n_cut)

        records.append({
            "shape":      detect_shape(piece_db[piece_idx]),
            "speed":      speed,
            "defect_pct": defect_pct,
        })
    if skipped:
        print(f"  Attention : {skipped} épisodes ignorés (piece_idx >= {n_pieces}).")
        print(f"  → Relancez generate_pieces.py puis record_dataset.py pour resynchroniser.")
    return records


def load_model_records(data_dir: str, piece_db: list, model_path: str,
                       infer_batch: int = 32) -> list[dict]:
    import torch
    from world_model.model import WorldModel
    from world_model.dataset import Normalizer

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {model_path}")

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available()          else
              torch.device("cpu"))
    print(f"  Device inférence : {device}")

    ckpt = torch.load(model_path, map_location=device)
    H    = ckpt["hyperparams"]

    model = WorldModel(
        shape_embed_dim = H["shape_embed_dim"],
        h_dim           = H["h_dim"],
        obs_dim         = H["obs_dim"],
        dropout         = 0.0,
        gru_layers      = H["gru_layers"],
        pe_dim          = H.get("pe_dim", 64),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    print(f"Modèle chargé (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")

    norm_path  = os.path.join(os.path.dirname(model_path), "normalizer.npz")
    normalizer = Normalizer.load(norm_path)

    # ── collect all episodes ──────────────────────────────────────────────────
    ep_paths = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
    all_wps        = []   # list of (W, 3) float32
    all_speeds     = []   # list of float
    all_q_des      = []   # list of (T, 2) float32
    all_is_cutting = []   # list of (T,) float32
    all_shapes     = []   # list of str

    for ep_path in ep_paths:
        piece_idx = int(os.path.basename(ep_path).split("_")[1]) - 1
        data      = np.load(ep_path)
        all_wps.append(np.array(piece_db[piece_idx], dtype=np.float32))
        all_speeds.append(float(data["duration_per_segment"]))
        all_q_des.append(data["q_des"].astype(np.float32))
        all_is_cutting.append(data["is_cutting"].astype(np.float32))
        all_shapes.append(detect_shape(piece_db[piece_idx]))

    sf    = max(1, int(H.get("subsample_factor", 1)))
    N     = len(all_wps)
    max_W = max(wp.shape[0] for wp in all_wps)
    ep_T  = [q.shape[0] for q in all_q_des]
    ep_n_sub = [math.ceil(T / sf) for T in ep_T]   # longueur après sous-échantillonnage
    max_sub  = max(ep_n_sub)

    # ── batched inference (à la résolution d'entraînement) ────────────────────
    q_pred_all = np.zeros((N, max_sub, H["obs_dim"]), dtype=np.float32)

    start = 0
    while start < N:
        end    = min(start + infer_batch, N)
        B      = end - start
        bt_sub = max(ep_n_sub[start:end])

        wps_t   = torch.zeros(B, max_W, 3, device=device)
        wp_lens = torch.zeros(B, dtype=torch.long, device=device)
        for k in range(B):
            W = all_wps[start + k].shape[0]
            wps_t[k, :W] = torch.from_numpy(all_wps[start + k]).to(device)
            wp_lens[k]   = W
        speeds_t = torch.tensor(
            [[all_speeds[start + k]] for k in range(B)], dtype=torch.float32, device=device
        )

        try:
            with torch.no_grad():
                pred_norm = model.predict(wps_t, wp_lens, speeds_t, max_len=bt_sub)
            pred_cpu = normalizer.denormalize_tensor(pred_norm).cpu().numpy()
            q_pred_all[start:end, :bt_sub, :] = pred_cpu
            print(f"  Inférence épisodes {start+1}–{end}/{N}…", end="\r")
            start = end
        except RuntimeError as e:
            if "buffer size" in str(e).lower() or "out of memory" in str(e).lower():
                infer_batch = max(1, infer_batch // 2)
                print(f"\n  OOM — batch réduit à {infer_batch}")
            else:
                raise

    print()

    # ── vectorised FK + defect (sur les pas sous-échantillonnés) ─────────────
    q_des_sub_padded = np.zeros((N, max_sub, 2), dtype=np.float32)
    ic_sub_padded    = np.zeros((N, max_sub),    dtype=np.float32)
    for i in range(N):
        n = ep_n_sub[i]
        q_des_sub_padded[i, :n] = all_q_des[i][::sf, :2]
        ic_sub_padded   [i, :n] = all_is_cutting[i][::sf]

    # Si le modèle prédit q_error (= q_real - q_des), reconstruire q_eff = q_des + delta
    target_keys = H.get("target_keys", ["q_real"])
    if isinstance(target_keys, str):
        target_keys = [k.strip() for k in target_keys.split(",")]
    predicts_error = target_keys == ["q_error"]

    q_pred_joints = q_pred_all[:, :, :2]
    if predicts_error:
        q_eff = q_des_sub_padded + q_pred_joints   # q_des + erreur simulée
    else:
        q_eff = q_pred_joints                      # q_real prédit directement

    fk_pred_all = fk_batch(q_eff.reshape(-1, 2)).reshape(N, max_sub, 2)
    fk_des_all  = fk_batch(q_des_sub_padded.reshape(-1, 2)).reshape(N, max_sub, 2)
    deviation_all = np.linalg.norm(fk_pred_all - fk_des_all, axis=2)   # (N, max_sub)

    records = []
    for i in range(N):
        n    = ep_n_sub[i]
        mask = ic_sub_padded[i, :n] == 1.0
        n_cut = mask.sum()
        if n_cut == 0:
            defect_pct = 0.0
        else:
            dev = deviation_all[i, :n]
            defect_pct = 100.0 * (dev[mask] > CUT_DEFECT_THRESHOLD).sum() / n_cut
        records.append({
            "shape":      all_shapes[i],
            "speed":      all_speeds[i],
            "defect_pct": float(defect_pct),
        })
    return records


# ── trend line ─────────────────────────────────────────────────────────────────

def poly_trend(speeds: np.ndarray, defects: np.ndarray, degree: int = 2):
    """Fit a polynomial trend and return (x_dense, y_fitted)."""
    idx  = np.argsort(speeds)
    sx, sy = speeds[idx], defects[idx]
    coeffs = np.polyfit(sx, sy, degree)
    x_dense = np.linspace(sx.min(), sx.max(), 200)
    y_fitted = np.clip(np.polyval(coeffs, x_dense), 0, 100)
    return x_dense, y_fitted


# ── plot ───────────────────────────────────────────────────────────────────────

def plot(real_records: list[dict], model_records: list[dict] | None):
    compare = model_records is not None
    title   = ("Taux de défauts vs vitesse d'exécution — réel"
               + (" & modèle" if compare else ""))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, shape in zip(axes, SHAPE_ORDER):
        color = SHAPE_COLORS[shape]

        real = [r for r in real_records if r["shape"] == shape]
        if real:
            sr = np.array([r["speed"]      for r in real])
            dr = np.array([r["defect_pct"] for r in real])
            ax.scatter(sr, dr, color=color, alpha=0.5, s=20, zorder=3,
                       label="Réel (épisodes)")
            if len(sr) >= 4:
                tx, ty = poly_trend(sr, dr)
                ax.plot(tx, ty, color=color, linewidth=2.5, zorder=4,
                        label="Tendance réelle")

        if compare:
            mdl = [r for r in model_records if r["shape"] == shape]
            if mdl:
                sm = np.array([r["speed"]      for r in mdl])
                dm = np.array([r["defect_pct"] for r in mdl])
                ax.scatter(sm, dm, color="#888888", alpha=0.35, s=20,
                           marker="^", zorder=2, label="Modèle (épisodes)")
                if len(sm) >= 4:
                    tx, ty = poly_trend(sm, dm)
                    ax.plot(tx, ty, "--", color="#444444", linewidth=2.0, zorder=3,
                            label="Tendance modèle")

        # Count per shape for subtitle
        n = len(real)
        ax.set_title(f"{shape}  ({n} épisodes)", fontweight="bold")
        ax.set_xlabel("Durée par segment (s)\n← rapide         lent →")
        ax.set_ylim(-5, 105)
        ax.axhline(50, color="red", linewidth=0.8, linestyle=":", alpha=0.4)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Défauts (%)")
    plt.tight_layout()
    plt.show()


# ── distribution plot ──────────────────────────────────────────────────────────

def plot_distribution(real_records: list[dict], model_records: list[dict] | None):
    compare = model_records is not None
    title   = ("Distribution des % de défauts par forme — réel"
               + (" & modèle" if compare else ""))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    bins = np.linspace(0, 100, 26)  # bins de 4 % chacun

    for ax, shape in zip(axes, SHAPE_ORDER):
        color = SHAPE_COLORS[shape]

        real = [r["defect_pct"] for r in real_records if r["shape"] == shape]
        if real:
            dr = np.array(real)
            ax.hist(dr, bins=bins, color=color, alpha=0.5, density=True,
                    label=f"Réel (n={len(dr)})", zorder=2)
            if len(dr) >= 3:
                kde = gaussian_kde(dr, bw_method="scott")
                x_dense = np.linspace(0, 100, 400)
                ax.plot(x_dense, kde(x_dense), color=color, linewidth=2.5, zorder=4)
            ax.axvline(np.mean(dr),   color=color, linewidth=1.5,
                       linestyle="--", label=f"Moyenne {np.mean(dr):.1f}%")
            ax.axvline(np.median(dr), color=color, linewidth=1.5,
                       linestyle=":",  label=f"Médiane {np.median(dr):.1f}%")

        if compare:
            mdl = [r["defect_pct"] for r in model_records if r["shape"] == shape]
            if mdl:
                dm = np.array(mdl)
                ax.hist(dm, bins=bins, color="#888888", alpha=0.35, density=True,
                        label=f"Modèle (n={len(dm)})", zorder=1)
                if len(dm) >= 3:
                    kde_m = gaussian_kde(dm, bw_method="scott")
                    ax.plot(x_dense, kde_m(x_dense), "--", color="#444444",
                            linewidth=2.0, zorder=3)

        ax.set_title(f"{shape}", fontweight="bold")
        ax.set_xlabel("Défauts (%)")
        ax.set_ylabel("Densité")
        ax.set_xlim(-2, 102)
        ax.axvline(50, color="red", linewidth=0.8, linestyle=":", alpha=0.4)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       choices=["real", "compare"], default="real")
    p.add_argument("--plot",       choices=["speed", "dist"],   default="speed")
    p.add_argument("--model-path", default="world_model/checkpoints/best_model.pt")
    p.add_argument("--data-dir",   default="dataset")
    p.add_argument("--db-path",    default="pieces_database.json")
    args = p.parse_args()

    with open(args.db_path) as f:
        piece_db = json.load(f)["pieces"]

    print("Chargement des données réelles...")
    real_records = load_real_records(args.data_dir, piece_db)
    print(f"  {len(real_records)} épisodes chargés.")

    model_records = None
    if args.mode == "compare":
        print("Inférence du modèle...")
        model_records = load_model_records(args.data_dir, piece_db, args.model_path)

    if args.plot == "dist":
        plot_distribution(real_records, model_records)
    else:
        plot(real_records, model_records)


if __name__ == "__main__":
    main()
