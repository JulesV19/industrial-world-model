"""
Visualise le taux de défauts de perçage par coin et par vitesse.

Usage:
    python3 visualize_defect_vs_speed.py                              # erreur (mm) vs vitesse, par coin
    python3 visualize_defect_vs_speed.py --plot dist                  # distribution des erreurs par coin
    python3 visualize_defect_vs_speed.py --mode compare               # réel + prédictions modèle
    python3 visualize_defect_vs_speed.py --mode compare --plot dist
    python3 visualize_defect_vs_speed.py --mode compare --model-path world_model/checkpoints/best_model.pt
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from record_dataset import DRILL_DEFECT_THRESHOLD

CORNER_COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]
CORNER_LABELS = ["Coin 1", "Coin 2", "Coin 3", "Coin 4"]


# ── data loading ───────────────────────────────────────────────────────────────

def load_real_records(data_dir: str) -> list[dict]:
    records = []
    for ep_path in sorted(glob.glob(os.path.join(data_dir, "episode_*.npz"))):
        data    = np.load(ep_path)
        errors  = data["errors"].astype(float)    # (4,) en mètres
        defects = data["defects"].astype(float)   # (4,) 0/1
        speed   = float(data["duration_per_segment"])
        records.append({"speed": speed, "errors": errors, "defects": defects})
    return records


def load_model_records(data_dir: str, model_path: str) -> list[dict]:
    import torch
    from world_model.model import DrillModel
    from world_model.dataset import Normalizer

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {model_path}")

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available()          else
              torch.device("cpu"))
    print(f"  Device inférence : {device}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    H    = ckpt.get("hyperparams", {})

    model = DrillModel(
        corner_embed_dim = H.get("corner_embed_dim", 64),
        global_dim       = H.get("global_dim", 256),
        dropout          = 0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    print(f"Modèle chargé (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")

    norm_path  = os.path.join(os.path.dirname(model_path), "normalizer.npz")
    normalizer = Normalizer.load(norm_path)

    records   = []
    ep_paths  = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
    for i, ep_path in enumerate(ep_paths):
        data    = np.load(ep_path)
        corners = data["corner_targets"].astype(np.float32)   # (4, 2)
        speed   = float(data["duration_per_segment"])

        corners_t = torch.from_numpy(corners).unsqueeze(0).to(device)
        speed_t   = torch.tensor([[speed]], dtype=torch.float32, device=device)

        with torch.no_grad():
            offsets_norm, defect_logits = model(corners_t, speed_t)

        offsets = normalizer.denormalize(offsets_norm[0].cpu().numpy())   # (4, 2) m
        errors  = np.linalg.norm(offsets, axis=1)                         # (4,) m
        defects = (torch.sigmoid(defect_logits[0]).cpu().numpy() > 0.5).astype(float)

        records.append({"speed": speed, "errors": errors, "defects": defects})
        if (i + 1) % 20 == 0 or (i + 1) == len(ep_paths):
            print(f"  Inférence épisodes {i+1}/{len(ep_paths)}…", end="\r")
    print()
    return records


# ── trend line ─────────────────────────────────────────────────────────────────

def poly_trend(speeds: np.ndarray, values: np.ndarray, degree: int = 2):
    idx = np.argsort(speeds)
    sx, sy = speeds[idx], values[idx]
    coeffs  = np.polyfit(sx, sy, degree)
    x_dense = np.linspace(sx.min(), sx.max(), 200)
    y_fitted = np.clip(np.polyval(coeffs, x_dense), 0, None)
    return x_dense, y_fitted


# ── scatter : erreur (mm) vs vitesse, un subplot par coin ─────────────────────

def plot_speed(real_records: list[dict], model_records: list[dict] | None):
    compare = model_records is not None
    title   = ("Erreur de positionnement vs vitesse d'exécution — réel"
               + (" & modèle" if compare else ""))

    speeds  = np.array([r["speed"]   for r in real_records])
    errors  = np.array([r["errors"]  for r in real_records])   # (N, 4) m
    defects = np.array([r["defects"] for r in real_records])   # (N, 4)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i, (ax, color, label) in enumerate(zip(axes, CORNER_COLORS, CORNER_LABELS)):
        err_mm = errors[:, i] * 1000
        ax.scatter(speeds, err_mm, color=color, alpha=0.5, s=20, zorder=3,
                   label="Réel (épisodes)")

        if len(speeds) >= 4:
            tx, ty = poly_trend(speeds, err_mm)
            ax.plot(tx, ty, color=color, linewidth=2.5, zorder=4, label="Tendance réelle")

        ax.axhline(DRILL_DEFECT_THRESHOLD * 1000, color="red", linewidth=1.2,
                   linestyle="--", alpha=0.7,
                   label=f"Seuil {DRILL_DEFECT_THRESHOLD*1000:.0f} mm")

        if compare:
            sm = np.array([r["speed"]    for r in model_records])
            em = np.array([r["errors"][i] for r in model_records]) * 1000
            ax.scatter(sm, em, color="#888888", alpha=0.35, s=20, marker="^",
                       zorder=2, label="Modèle (épisodes)")
            if len(sm) >= 4:
                tx, ty = poly_trend(sm, em)
                ax.plot(tx, ty, "--", color="#444444", linewidth=2.0, zorder=3,
                        label="Tendance modèle")

        defect_rate = defects[:, i].mean() * 100
        ax.set_title(f"{label}  —  {defect_rate:.1f}% défauts", fontweight="bold")
        ax.set_xlabel("Durée par segment (s)\n← rapide         lent →")
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Erreur de positionnement (mm)")
    plt.tight_layout()
    plt.show()


# ── distribution : histogramme des erreurs par coin ───────────────────────────

def plot_distribution(real_records: list[dict], model_records: list[dict] | None):
    compare = model_records is not None
    title   = ("Distribution des erreurs de perçage par coin — réel"
               + (" & modèle" if compare else ""))

    errors = np.array([r["errors"] for r in real_records]) * 1000  # (N, 4) mm

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    max_err = errors.max()
    bins    = np.linspace(0, max(max_err * 1.1, DRILL_DEFECT_THRESHOLD * 1100), 30)
    x_dense = np.linspace(0, bins[-1], 400)

    for i, (ax, color, label) in enumerate(zip(axes, CORNER_COLORS, CORNER_LABELS)):
        err_mm = errors[:, i]
        ax.hist(err_mm, bins=bins, color=color, alpha=0.5, density=True,
                label=f"Réel (n={len(err_mm)})", zorder=2)

        if len(err_mm) >= 3:
            kde = gaussian_kde(err_mm, bw_method="scott")
            ax.plot(x_dense, kde(x_dense), color=color, linewidth=2.5, zorder=4)

        ax.axvline(np.mean(err_mm),   color=color, linewidth=1.5,
                   linestyle="--", label=f"Moyenne {np.mean(err_mm):.1f} mm")
        ax.axvline(np.median(err_mm), color=color, linewidth=1.5,
                   linestyle=":",  label=f"Médiane {np.median(err_mm):.1f} mm")
        ax.axvline(DRILL_DEFECT_THRESHOLD * 1000, color="red", linewidth=1.2,
                   linestyle="--", alpha=0.7,
                   label=f"Seuil {DRILL_DEFECT_THRESHOLD*1000:.0f} mm")

        if compare:
            em = np.array([r["errors"][i] for r in model_records]) * 1000
            ax.hist(em, bins=bins, color="#888888", alpha=0.35, density=True,
                    label=f"Modèle (n={len(em)})", zorder=1)
            if len(em) >= 3:
                kde_m = gaussian_kde(em, bw_method="scott")
                ax.plot(x_dense, kde_m(x_dense), "--", color="#444444",
                        linewidth=2.0, zorder=3)

        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Erreur (mm)")
        ax.set_ylabel("Densité")
        ax.set_xlim(left=0)
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
    args = p.parse_args()

    print("Chargement des données réelles...")
    real_records = load_real_records(args.data_dir)
    if not real_records:
        print(f"Aucun épisode trouvé dans '{args.data_dir}'. Lancez record_dataset.py d'abord.")
        return
    print(f"  {len(real_records)} épisodes chargés.")

    model_records = None
    if args.mode == "compare":
        print("Inférence du modèle...")
        model_records = load_model_records(args.data_dir, args.model_path)

    if args.plot == "dist":
        plot_distribution(real_records, model_records)
    else:
        plot_speed(real_records, model_records)


if __name__ == "__main__":
    main()
