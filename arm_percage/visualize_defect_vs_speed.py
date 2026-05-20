"""
Visualise le taux de défauts de perçage par coin et par vitesse.

Usage:
    python3 visualize_defect_vs_speed.py                   # erreur (mm) vs vitesse, par coin
    python3 visualize_defect_vs_speed.py --plot dist        # distribution des erreurs par coin
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

def load_records(data_dir: str) -> list[dict]:
    records = []
    for ep_path in sorted(glob.glob(os.path.join(data_dir, "episode_*.npz"))):
        data    = np.load(ep_path)
        errors  = data["errors"].astype(float)    # (4,) en mètres
        defects = data["defects"].astype(float)   # (4,) 0/1
        speed   = float(data["duration_per_segment"])
        records.append({"speed": speed, "errors": errors, "defects": defects})
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

def plot_speed(records: list[dict]):
    speeds  = np.array([r["speed"]   for r in records])
    errors  = np.array([r["errors"]  for r in records])   # (N, 4) mm
    defects = np.array([r["defects"] for r in records])   # (N, 4)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    fig.suptitle("Erreur de positionnement vs vitesse d'exécution — par coin",
                 fontsize=13, fontweight="bold")

    for i, (ax, color, label) in enumerate(zip(axes, CORNER_COLORS, CORNER_LABELS)):
        err_mm = errors[:, i] * 1000
        ax.scatter(speeds, err_mm, color=color, alpha=0.5, s=20, zorder=3,
                   label="Épisodes")

        # Tendance polynomiale
        if len(speeds) >= 4:
            tx, ty = poly_trend(speeds, err_mm)
            ax.plot(tx, ty, color=color, linewidth=2.5, zorder=4, label="Tendance")

        # Seuil de défaut
        ax.axhline(DRILL_DEFECT_THRESHOLD * 1000, color="red", linewidth=1.2,
                   linestyle="--", alpha=0.7,
                   label=f"Seuil {DRILL_DEFECT_THRESHOLD*1000:.0f} mm")

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

def plot_distribution(records: list[dict]):
    errors = np.array([r["errors"] for r in records]) * 1000  # (N, 4) mm

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
    fig.suptitle("Distribution des erreurs de perçage par coin",
                 fontsize=13, fontweight="bold")

    max_err = errors.max()
    bins = np.linspace(0, max(max_err * 1.1, DRILL_DEFECT_THRESHOLD * 1100), 30)

    for i, (ax, color, label) in enumerate(zip(axes, CORNER_COLORS, CORNER_LABELS)):
        err_mm = errors[:, i]
        ax.hist(err_mm, bins=bins, color=color, alpha=0.5, density=True,
                label=f"n={len(err_mm)}", zorder=2)

        if len(err_mm) >= 3:
            kde = gaussian_kde(err_mm, bw_method="scott")
            x_dense = np.linspace(0, bins[-1], 400)
            ax.plot(x_dense, kde(x_dense), color=color, linewidth=2.5, zorder=4)

        ax.axvline(np.mean(err_mm),   color=color, linewidth=1.5,
                   linestyle="--", label=f"Moyenne {np.mean(err_mm):.1f} mm")
        ax.axvline(np.median(err_mm), color=color, linewidth=1.5,
                   linestyle=":",  label=f"Médiane {np.median(err_mm):.1f} mm")
        ax.axvline(DRILL_DEFECT_THRESHOLD * 1000, color="red", linewidth=1.2,
                   linestyle="--", alpha=0.7, label=f"Seuil {DRILL_DEFECT_THRESHOLD*1000:.0f} mm")

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
    p.add_argument("--plot",     choices=["speed", "dist"], default="speed")
    p.add_argument("--data-dir", default="dataset")
    args = p.parse_args()

    records = load_records(args.data_dir)
    if not records:
        print(f"Aucun épisode trouvé dans '{args.data_dir}'. Lancez record_dataset.py d'abord.")
        return
    print(f"{len(records)} épisodes chargés.")

    if args.plot == "dist":
        plot_distribution(records)
    else:
        plot_speed(records)


if __name__ == "__main__":
    main()
