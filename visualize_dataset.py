import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
import os

from config import l1, l2
from record_dataset import CUT_DEFECT_THRESHOLD


def _fk(q):
    x = l1 * np.cos(q[:, 0]) + l2 * np.cos(q[:, 0] + q[:, 1])
    y = l1 * np.sin(q[:, 0]) + l2 * np.sin(q[:, 0] + q[:, 1])
    return x, y


def _corridor(x, y, thickness):
    """Retourne les courbes offset supérieure et inférieure d'une polyligne."""
    n = len(x)
    nx_arr, ny_arr = np.zeros(n), np.zeros(n)
    for i in range(n):
        if i == 0:
            dx, dy = x[1] - x[0], y[1] - y[0]
        elif i == n - 1:
            dx, dy = x[-1] - x[-2], y[-1] - y[-2]
        else:
            dx, dy = x[i+1] - x[i-1], y[i+1] - y[i-1]
        length = np.hypot(dx, dy)
        if length > 1e-9:
            nx_arr[i], ny_arr[i] = -dy / length, dx / length
    return (x + thickness * nx_arr, y + thickness * ny_arr,
            x - thickness * nx_arr, y - thickness * ny_arr)


def main():
    dataset_dir = "dataset"
    episodes = sorted([
        f for f in os.listdir(dataset_dir)
        if f.startswith("episode_") and f.endswith(".npz")
    ]) if os.path.exists(dataset_dir) else []

    if not episodes:
        print(f"Aucun épisode trouvé dans '{dataset_dir}'.")
        print("Lancez d'abord 'python3 record_dataset.py'.")
        return

    num_episodes = len(episodes)

    fig = plt.figure(figsize=(18, 10))
    fig.canvas.manager.set_window_title("Visualiseur de Dataset World Model")
    fig.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.92,
                        wspace=0.25, hspace=0.35)

    ax_traj = plt.subplot(2, 3, 1)
    ax_q1   = plt.subplot(2, 3, 2)
    ax_q2   = plt.subplot(2, 3, 3)
    ax_tau  = plt.subplot(2, 3, 4)
    ax_dq1  = plt.subplot(2, 3, 5)
    ax_dq2  = plt.subplot(2, 3, 6)

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgoldenrodyellow')
    episode_slider = Slider(ax_slider, 'Épisode', 1, num_episodes, valinit=1, valstep=1)

    def update(val):
        idx = int(episode_slider.val) - 1
        fname = episodes[idx]
        filepath = os.path.join(dataset_dir, fname)

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
        dq_des    = data['dq_des']
        tau       = data['tau']
        is_cutting    = data['is_cutting']
        cut_deviation = data['cut_deviation']
        cut_defect    = data['cut_defect']
        duration      = float(data['duration_per_segment'])

        T = len(q_real)
        t = np.arange(T) * 0.1

        for ax in [ax_traj, ax_q1, ax_q2, ax_dq1, ax_dq2, ax_tau]:
            ax.clear()

        # ── Cinématique directe ──────────────────────────────────────────────
        x_real, y_real = _fk(q_real)
        x_des,  y_des  = _fk(q_des)

        cut  = is_cutting == 1.0
        nocut = ~cut

        # ── 1. TRAJECTOIRE XY ────────────────────────────────────────────────
        # Chemin désiré complet (gris pointillé)
        ax_traj.plot(x_des, y_des, '--', color='#999999', linewidth=1,
                     alpha=0.6, zorder=1, label='Chemin désiré')

        # Corridor de tolérance autour de la partie découpe uniquement
        x_cut_des, y_cut_des = x_des[cut], y_des[cut]
        if len(x_cut_des) > 2:
            xu, yu, xl, yl = _corridor(x_cut_des, y_cut_des, CUT_DEFECT_THRESHOLD)
            ax_traj.fill(
                np.concatenate([xu, xl[::-1]]),
                np.concatenate([yu, yl[::-1]]),
                color='gold', alpha=0.35, zorder=2,
                label=f'Tolérance ±{CUT_DEFECT_THRESHOLD*100:.1f} cm'
            )
            ax_traj.plot(x_cut_des, y_cut_des, '-', color='#CC8800',
                         linewidth=1.2, alpha=0.8, zorder=3)

        # Déplacement hors découpe (bleu pâle)
        ax_traj.scatter(x_real[nocut], y_real[nocut],
                        c='#6699CC', s=2, alpha=0.4, zorder=4,
                        label='Déplacement (hors coupe)')

        # Chemin de découpe réel coloré par défaut (vert=ok, rouge=défaut)
        x_cut_real, y_cut_real = x_real[cut], y_real[cut]
        defect_cut = cut_defect[cut]
        if len(x_cut_real) > 1:
            pts = np.array([x_cut_real, y_cut_real]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            seg_colors = ['#CC2222' if d > 0.5 else '#22AA44' for d in defect_cut[:-1]]
            lc = LineCollection(segs, colors=seg_colors, linewidths=2.0, zorder=5)
            ax_traj.add_collection(lc)

        n_defects = int(defect_cut.sum())
        n_cut     = int(cut.sum())
        defect_pct = 100 * n_defects / n_cut if n_cut > 0 else 0.0

        # Lecture du nom de forme depuis le nom de fichier
        piece_id = fname.split('_')[1]   # ex. "001"
        ax_traj.set_title(
            f"Pièce {piece_id}  |  {duration:.2f} s/seg  |  "
            f"Défauts {defect_pct:.1f}% ({n_defects}/{n_cut})",
            fontweight='bold', fontsize=9
        )
        ax_traj.set_aspect('equal', 'box')
        ax_traj.legend(loc='upper right', fontsize=7)
        ax_traj.grid(True, linestyle=':', alpha=0.5)

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
        ax_dq1.plot(t, dq_des[:, 0],    'k--', alpha=0.6, label='Désiré')
        ax_dq1.plot(t, dq_real[:, 0],   'b-',  linewidth=1.5, label='Réel')
        ax_dq1.plot(t, dq_sensed[:, 0], 'g-',  alpha=0.5, label='Capteur')
        ax_dq1.set_title("Vitesse ang. 1 (rad/s)", fontweight='bold')
        ax_dq1.set_xlabel("Temps (s)")
        ax_dq1.grid(True, linestyle=':', alpha=0.5)

        ax_dq2.plot(t, dq_des[:, 1],    'k--', alpha=0.6, label='Désiré')
        ax_dq2.plot(t, dq_real[:, 1],   'b-',  linewidth=1.5, label='Réel')
        ax_dq2.plot(t, dq_sensed[:, 1], 'g-',  alpha=0.5, label='Capteur')
        ax_dq2.set_title("Vitesse ang. 2 (rad/s)", fontweight='bold')
        ax_dq2.set_xlabel("Temps (s)")
        ax_dq2.grid(True, linestyle=':', alpha=0.5)

        # ── 4. COUPLES + DÉVIATION ───────────────────────────────────────────
        ax_tau.plot(t, tau[:, 0], 'm-', alpha=0.8, label='Moteur 1')
        ax_tau.plot(t, tau[:, 1], 'y-', alpha=0.8, label='Moteur 2')
        ylims = ax_tau.get_ylim()
        ax_tau.fill_between(t, ylims[0], ylims[1],
                            where=cut, color='red', alpha=0.08, label='Laser actif')
        ax_tau.set_title("Couples moteurs (N·m)", fontweight='bold')
        ax_tau.legend(fontsize=7)
        ax_tau.grid(True, linestyle=':', alpha=0.5)

        fig.canvas.draw_idle()

    update(1)
    episode_slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()
