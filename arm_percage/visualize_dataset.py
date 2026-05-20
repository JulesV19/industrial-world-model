"""
Visualiseur du dataset de perçage.

Usage :
    python3 visualize_dataset.py
    python3 visualize_dataset.py --dataset dataset
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from config import l1, l2
from record_dataset import DRILL_DEFECT_THRESHOLD


def _fk(q):
    x = l1 * np.cos(q[:, 0]) + l2 * np.cos(q[:, 0] + q[:, 1])
    y = l1 * np.sin(q[:, 0]) + l2 * np.sin(q[:, 0] + q[:, 1])
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset")
    args = parser.parse_args()

    episodes = sorted([
        f for f in os.listdir(args.dataset)
        if f.startswith("episode_") and f.endswith(".npz")
    ]) if os.path.exists(args.dataset) else []

    if not episodes:
        print(f"Aucun épisode dans '{args.dataset}'. Lancez record_dataset.py d'abord.")
        return

    fig = plt.figure(figsize=(18, 10))
    fig.canvas.manager.set_window_title("Visualiseur Dataset Perçage")
    fig.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.92,
                        wspace=0.25, hspace=0.35)

    ax_traj = plt.subplot(2, 3, 1)
    ax_q1   = plt.subplot(2, 3, 2)
    ax_q2   = plt.subplot(2, 3, 3)
    ax_tau  = plt.subplot(2, 3, 4)
    ax_dq1  = plt.subplot(2, 3, 5)
    ax_dq2  = plt.subplot(2, 3, 6)

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgoldenrodyellow')
    episode_slider = Slider(ax_slider, 'Épisode', 1, len(episodes), valinit=1, valstep=1)

    def update(val):
        idx = int(episode_slider.val) - 1
        fname = episodes[idx]
        filepath = os.path.join(args.dataset, fname)

        try:
            data = np.load(filepath)
        except Exception as e:
            print(f"Erreur : {e}")
            return

        q_real    = data['q_real']
        q_sensed  = data['q_sensed']
        q_des     = data['q_des']
        dq_real   = data['dq_real']
        dq_sensed = data['dq_sensed']
        tau       = data['tau']
        targets      = data['corner_targets']   # (4, 2)
        hits         = data['drill_hits']       # (4, 2)
        errors       = data['errors']           # (4,) mètres
        defects      = data['defects']          # (4,) 0/1
        is_drilling  = data['is_drilling'] > 0   # pour les plots temporels (phases 1 et 2)
        duration     = float(data['duration_per_segment'])

        T = len(q_real)
        t = np.arange(T) * (0.01 * 5)  # SUBSAMPLE=5, dt=0.01 → 20 Hz

        for ax in [ax_traj, ax_q1, ax_q2, ax_tau, ax_dq1, ax_dq2]:
            ax.clear()

        # ── Cinématique directe ──────────────────────────────────────────────
        x_real, y_real = _fk(q_real)
        x_des,  y_des  = _fk(q_des)

        # ── 1. TRAJECTOIRE XY ────────────────────────────────────────────────

        # Contour réel de la pièce découpée
        if 'cut_contour_x' in data:
            ax_traj.plot(data['cut_contour_x'], data['cut_contour_y'],
                         '-', color='#DD8800', lw=2, alpha=0.85, zorder=2,
                         label='Contour découpé')

        # Chemin désiré (gris pointillé)
        ax_traj.plot(x_des, y_des, '--', color='#999999', lw=1,
                     alpha=0.6, zorder=3, label='Chemin désiré')

        # 3 phases : 0 = approche/retour, 1 = transit entre coins, 2 = perçage actif
        is_drilling_raw = data['is_drilling']
        moving   = is_drilling_raw == 0.0
        transit  = is_drilling_raw == 1.0
        active   = is_drilling_raw == 2.0

        ax_traj.scatter(x_real[moving], y_real[moving],
                        c='#6699CC', s=2, alpha=0.4, zorder=4,
                        label='Approche / retour')
        ax_traj.plot(x_real[transit], y_real[transit],
                     '-', color='#5588CC', lw=1.5, alpha=0.7, zorder=5,
                     label='Transit entre coins')
        ax_traj.scatter(x_real[active], y_real[active],
                        c='#FF6600', s=6, alpha=0.8, zorder=6,
                        label='Perçage actif (3s)')

        # Couloir de tolérance autour des cibles
        for i in range(4):
            color = '#CC2222' if defects[i] else '#22AA44'
            circ = plt.Circle(targets[i], DRILL_DEFECT_THRESHOLD,
                              color='gold', fill=False, lw=1.2, alpha=0.7, zorder=5)
            ax_traj.add_patch(circ)
            # Cible
            ax_traj.plot(*targets[i], 'o', color='gold', ms=6, zorder=6)
            # Trou réel
            ax_traj.plot(*hits[i], 'x', color=color, ms=10, mew=2.5, zorder=7)
            # Flèche erreur
            ax_traj.annotate('', xy=hits[i], xytext=targets[i],
                             arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
            ax_traj.text(hits[i][0] + 0.015, hits[i][1] + 0.015,
                         f"{errors[i]*1000:.1f}mm", fontsize=7, color=color, zorder=8)

        n_def = int(defects.sum())
        piece_id = fname.split('_')[1]
        ax_traj.set_title(
            f"Pièce {piece_id}  |  {duration:.2f} s/seg  |  "
            f"Défauts {n_def}/4  |  err moy {errors.mean()*1000:.1f} mm",
            fontweight='bold', fontsize=9
        )
        ax_traj.set_aspect('equal', 'box')
        ax_traj.legend(loc='upper right', fontsize=7)
        ax_traj.grid(True, linestyle=':', alpha=0.5)
        ax_traj.set_xlabel("x (m)")
        ax_traj.set_ylabel("y (m)")

        # ── 2. ANGLES ────────────────────────────────────────────────────────
        ax_q1.plot(t, np.degrees(q_des[:, 0]),    'k--', alpha=0.6, label='Désiré')
        ax_q1.plot(t, np.degrees(q_real[:, 0]),   'b-',  linewidth=1.5, label='Réel')
        ax_q1.plot(t, np.degrees(q_sensed[:, 0]), 'g-',  alpha=0.5, label='Capteur')
        ax_q1.set_title("Angle 1 (°)", fontweight='bold')
        ax_q1.legend(fontsize=7)
        ax_q1.grid(True, linestyle=':', alpha=0.5)

        ax_q2.plot(t, np.degrees(q_des[:, 1]),    'k--', alpha=0.6, label='Désiré')
        ax_q2.plot(t, np.degrees(q_real[:, 1]),   'b-',  linewidth=1.5, label='Réel')
        ax_q2.plot(t, np.degrees(q_sensed[:, 1]), 'g-',  alpha=0.5, label='Capteur')
        ax_q2.set_title("Angle 2 (°)", fontweight='bold')
        ax_q2.legend(fontsize=7)
        ax_q2.grid(True, linestyle=':', alpha=0.5)

        # ── 3. VITESSES ANGULAIRES ───────────────────────────────────────────
        ax_dq1.plot(t, q_des[:, 0] * 0,    'k--', alpha=0)  # axe commun
        ax_dq1.plot(t, dq_real[:, 0],   'b-',  linewidth=1.5, label='Réel')
        ax_dq1.plot(t, dq_sensed[:, 0], 'g-',  alpha=0.5, label='Capteur')
        ax_dq1.set_title("Vitesse ang. 1 (rad/s)", fontweight='bold')
        ax_dq1.set_xlabel("Temps (s)")
        ax_dq1.legend(fontsize=7)
        ax_dq1.grid(True, linestyle=':', alpha=0.5)

        ax_dq2.plot(t, dq_real[:, 1],   'b-',  linewidth=1.5, label='Réel')
        ax_dq2.plot(t, dq_sensed[:, 1], 'g-',  alpha=0.5, label='Capteur')
        ax_dq2.set_title("Vitesse ang. 2 (rad/s)", fontweight='bold')
        ax_dq2.set_xlabel("Temps (s)")
        ax_dq2.legend(fontsize=7)
        ax_dq2.grid(True, linestyle=':', alpha=0.5)

        # ── 4. COUPLES + marqueurs de perçage ────────────────────────────────
        ax_tau.plot(t, tau[:, 0], 'm-', alpha=0.8, label='Moteur 1')
        ax_tau.plot(t, tau[:, 1], 'y-', alpha=0.8, label='Moteur 2')

        # Zone colorée pendant le perçage (comme is_cutting dans arm_decoupe)
        ylims = ax_tau.get_ylim()
        ax_tau.fill_between(t, ylims[0], ylims[1],
                            where=is_drilling, color='red', alpha=0.08,
                            label='Perçage actif')

        ax_tau.set_title("Couples moteurs (N·m)", fontweight='bold')
        ax_tau.legend(fontsize=7)
        ax_tau.grid(True, linestyle=':', alpha=0.5)

        fig.canvas.draw_idle()

    update(1)
    episode_slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()
