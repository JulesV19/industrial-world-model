"""
Visualiseur de dataset — bras découpe.

Modes :
    python visualize_dataset.py                          # vue sessions (si dispo) sinon legacy
    python visualize_dataset.py --mode session           # évolution des indicateurs par session
    python visualize_dataset.py --mode episode           # animation temps réel d'un épisode
    python visualize_dataset.py --mode legacy            # ancien slider épisode-par-épisode
    python visualize_dataset.py --mode episode --session 3 --piece 42   # direct
"""

import argparse
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

from config import l1, l2
from record_dataset import CUT_DEFECT_THRESHOLD

DATASET_DIR = "dataset"


# ─── helpers FK & corridor ─────────────────────────────────────────────────────

def _fk(q):
    x = l1 * np.cos(q[:, 0]) + l2 * np.cos(q[:, 0] + q[:, 1])
    y = l1 * np.sin(q[:, 0]) + l2 * np.sin(q[:, 0] + q[:, 1])
    return x, y


def _fk1(q):
    """Retourne position de toutes les articulations : origine, joint2, effecteur."""
    j2x = l1 * np.cos(q[0])
    j2y = l1 * np.sin(q[0])
    eex = j2x + l2 * np.cos(q[0] + q[1])
    eey = j2y + l2 * np.sin(q[0] + q[1])
    return (0.0, j2x, eex), (0.0, j2y, eey)


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


# ─── data helpers ─────────────────────────────────────────────────────────────

def _list_sessions(data_dir):
    files = os.listdir(data_dir)
    ids = sorted({
        re.search(r"session_(\d+)_piece", f).group(1)
        for f in files
        if re.match(r"session_\d+_piece\d+\.npz$", f)
    })
    return ids


def _load_session_metrics(data_dir, sid):
    """Charge les métriques pièce par pièce d'une session."""
    dev_path = os.path.join(data_dir, f"session_{sid}_deviations.npy")
    cad_path = os.path.join(data_dir, f"session_{sid}_cadence.npy")

    cadence = float(np.load(cad_path)) if os.path.exists(cad_path) else 0.0
    mean_devs = np.load(dev_path) if os.path.exists(dev_path) else None

    # Charger les épisodes pour des métriques plus détaillées
    piece_files = sorted([
        f for f in os.listdir(data_dir)
        if re.match(rf"session_{sid}_piece\d+\.npz$", f)
    ])

    piece_counts, max_devs, defect_pcts = [], [], []
    for pf in piece_files:
        d = np.load(os.path.join(data_dir, pf))
        pc = int(d["piece_count"]) if "piece_count" in d else len(piece_counts)
        cut = d["is_cutting"].astype(bool)
        cut_dev = d["cut_deviation"][cut]
        defects = d["cut_defect"][cut]
        piece_counts.append(pc)
        max_devs.append(float(cut_dev.max()) if len(cut_dev) > 0 else 0.0)
        defect_pcts.append(
            100.0 * defects.sum() / len(defects) if len(defects) > 0 else 0.0
        )

    piece_counts = np.array(piece_counts)
    order = np.argsort(piece_counts)
    piece_counts = piece_counts[order]
    max_devs     = np.array(max_devs)[order]
    defect_pcts  = np.array(defect_pcts)[order]

    return dict(
        cadence     = cadence,
        piece_counts= piece_counts,
        mean_devs   = mean_devs if mean_devs is not None else np.zeros(len(piece_counts)),
        max_devs    = max_devs,
        defect_pcts = defect_pcts,
        n_pieces    = len(piece_counts),
    )


def _load_episode(data_dir, sid, piece_idx):
    fname = f"session_{sid}_piece{piece_idx:04d}.npz"
    path  = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        return None
    return np.load(path)


# ─── mode : session ───────────────────────────────────────────────────────────

def view_sessions(data_dir):
    session_ids = _list_sessions(data_dir)
    if not session_ids:
        print("Aucun fichier session trouvé. Lancez d'abord record_dataset.py --mode session")
        return

    n_sess = len(session_ids)

    fig = plt.figure(figsize=(18, 10))
    fig.canvas.manager.set_window_title("Vue Sessions — Bras Découpe")
    fig.patch.set_facecolor("#1a1a2e")

    # Layout : 3 plots de métriques + 1 info + slider en bas
    fig.subplots_adjust(left=0.07, bottom=0.14, right=0.97, top=0.91,
                        wspace=0.32, hspace=0.45)

    ax_dev  = fig.add_subplot(2, 3, 1)
    ax_max  = fig.add_subplot(2, 3, 2)
    ax_def  = fig.add_subplot(2, 3, 3)
    ax_all  = fig.add_subplot(2, 3, (4, 5))  # scatter toutes sessions
    ax_info = fig.add_subplot(2, 3, 6)

    DARK = "#1a1a2e"
    for ax in [ax_dev, ax_max, ax_def, ax_all, ax_info]:
        ax.set_facecolor("#16213e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.tick_params(colors="#aaaacc", labelsize=8)
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        ax.title.set_color("#ddddff")

    # Pré-charger aperçu rapide via les fichiers deviations.npy (léger)
    all_mean_devs = {}
    for sid in session_ids:
        p = os.path.join(data_dir, f"session_{sid}_deviations.npy")
        if os.path.exists(p):
            all_mean_devs[sid] = np.load(p)

    # Scatter toutes sessions (déviation moy vs n° pièce)
    ax_all.set_title("Déviation moy. × sessions (aperçu)", fontweight='bold')
    cmap = plt.cm.plasma
    for i, sid in enumerate(session_ids):
        if sid in all_mean_devs:
            devs = all_mean_devs[sid]
            color = cmap(i / max(1, n_sess - 1))
            ax_all.plot(np.arange(len(devs)), devs * 1000,
                        alpha=0.25, linewidth=0.8, color=color)
    # Ligne de seuil
    ax_all.axhline(CUT_DEFECT_THRESHOLD * 1000, color='#ff4444', linewidth=1.2,
                   linestyle='--', label=f'Seuil {CUT_DEFECT_THRESHOLD*1000:.0f} mm')
    ax_all.set_xlabel("N° pièce dans la session")
    ax_all.set_ylabel("Déviation (mm)")
    ax_all.legend(fontsize=8, labelcolor='white', facecolor='#16213e', edgecolor='#444466')
    ax_all.grid(True, linestyle=':', alpha=0.3, color='#555577')

    # Slider session
    ax_sl = fig.add_axes([0.2, 0.04, 0.6, 0.025], facecolor='#2a2a4e')
    sl = Slider(ax_sl, 'Session', 1, n_sess, valinit=1, valstep=1,
                color='#5555cc')
    sl.label.set_color('#aaaacc')
    sl.valtext.set_color('#ccccff')

    highlight_line = [None]

    def draw_session(sid):
        m = _load_session_metrics(data_dir, sid)
        n = m["n_pieces"]
        xs = m["piece_counts"]

        for ax in [ax_dev, ax_max, ax_def, ax_info]:
            ax.clear()
            ax.set_facecolor("#16213e")
            for sp in ax.spines.values():
                sp.set_edgecolor("#444466")
            ax.tick_params(colors="#aaaacc", labelsize=8)
            ax.title.set_color("#ddddff")
            ax.xaxis.label.set_color("#aaaacc")
            ax.yaxis.label.set_color("#aaaacc")

        # Courbe déviation moyenne
        ax_dev.plot(xs, m["mean_devs"] * 1000, color='#66aaff', linewidth=1.8)
        ax_dev.fill_between(xs, 0, m["mean_devs"] * 1000, alpha=0.15, color='#66aaff')
        ax_dev.axhline(CUT_DEFECT_THRESHOLD * 1000, color='#ff4444',
                       linestyle='--', linewidth=1.2)
        ax_dev.set_title("Déviation moy. (mm)", fontweight='bold')
        ax_dev.set_xlabel("N° pièce")
        ax_dev.grid(True, linestyle=':', alpha=0.3, color='#555577')

        # Courbe déviation max
        ax_max.plot(xs, m["max_devs"] * 1000, color='#ffaa44', linewidth=1.8)
        ax_max.fill_between(xs, 0, m["max_devs"] * 1000, alpha=0.15, color='#ffaa44')
        ax_max.axhline(CUT_DEFECT_THRESHOLD * 1000, color='#ff4444',
                       linestyle='--', linewidth=1.2)
        ax_max.set_title("Déviation max (mm)", fontweight='bold')
        ax_max.set_xlabel("N° pièce")
        ax_max.grid(True, linestyle=':', alpha=0.3, color='#555577')

        # Courbe % défauts
        colors_def = ['#ff4444' if v > 50 else '#44cc77' for v in m["defect_pcts"]]
        ax_def.bar(xs, m["defect_pcts"], color=colors_def, width=0.8, alpha=0.8)
        ax_def.axhline(50, color='#ffaa00', linestyle='--', linewidth=1.0)
        ax_def.set_ylim(0, 105)
        ax_def.set_title("% points défectueux", fontweight='bold')
        ax_def.set_xlabel("N° pièce")
        ax_def.set_ylabel("%")
        ax_def.grid(True, linestyle=':', alpha=0.3, color='#555577', axis='y')

        # Panneau info
        ax_info.axis('off')
        total_def = m["defect_pcts"].mean() if n > 0 else 0
        fm_end = 1.0 + 0.8 * (n - 1) / ((n - 1) + 500) if n > 1 else 1.0
        info_lines = [
            f"Session   : {sid}",
            f"Pièces    : {n}",
            f"Cadence   : {m['cadence']:.0f} pièces/h",
            "",
            f"Dév. moy finale : {m['mean_devs'][-1]*1000:.2f} mm" if n > 0 else "",
            f"Dév. max finale : {m['max_devs'][-1]*1000:.2f} mm" if n > 0 else "",
            f"% défauts moyen : {total_def:.1f} %",
            "",
            f"Friction mult.  : ×{fm_end:.2f}  (n={n-1})",
            f"Bruit mult.     : ×{1 + 0.5*(m['cadence']/60):.2f}  (cad={m['cadence']:.0f})",
        ]
        for i, line in enumerate(info_lines):
            color = '#ffaa44' if ':' in line and i < 3 else \
                    '#ff7777' if 'défauts' in line else '#ddddff'
            ax_info.text(0.05, 0.92 - i * 0.09, line,
                         transform=ax_info.transAxes,
                         fontsize=9, color=color, family='monospace')
        ax_info.set_title("Infos session", fontweight='bold')

        # Highlight dans le scatter global
        if highlight_line[0] is not None:
            try:
                highlight_line[0].remove()
            except Exception:
                pass
        if sid in all_mean_devs:
            devs = all_mean_devs[sid]
            line, = ax_all.plot(np.arange(len(devs)), devs * 1000,
                                color='white', linewidth=2.0, zorder=10)
            highlight_line[0] = line

        fig.suptitle(f"Dataset sessions — {n_sess} sessions  |  Session {sid} sélectionnée",
                     color='#ccccff', fontsize=11, fontweight='bold')
        fig.canvas.draw_idle()

    def on_slider(val):
        idx = int(sl.val) - 1
        draw_session(session_ids[idx])

    draw_session(session_ids[0])
    sl.on_changed(on_slider)
    plt.show()


# ─── mode : episode (animation) ───────────────────────────────────────────────

def animate_episode(data_dir, sid=None, piece_idx=None):
    session_ids = _list_sessions(data_dir)
    if not session_ids:
        print("Aucune session trouvée.")
        return

    if sid is None:
        sid = session_ids[0]
    if piece_idx is None:
        piece_idx = 0

    fig = plt.figure(figsize=(18, 9))
    fig.canvas.manager.set_window_title("Animation épisode — Bras Découpe")
    fig.patch.set_facecolor("#1a1a2e")
    fig.subplots_adjust(left=0.05, bottom=0.18, right=0.97, top=0.92,
                        wspace=0.3, hspace=0.45)

    ax_xy   = fig.add_subplot(1, 3, 1)
    ax_dev  = fig.add_subplot(3, 3, 2)
    ax_tau  = fig.add_subplot(3, 3, 3)
    ax_q1   = fig.add_subplot(3, 3, 5)
    ax_dq1  = fig.add_subplot(3, 3, 6)
    ax_q2   = fig.add_subplot(3, 3, 8)
    ax_dq2  = fig.add_subplot(3, 3, 9)

    for ax in [ax_xy, ax_dev, ax_tau, ax_q1, ax_dq1, ax_q2, ax_dq2]:
        ax.set_facecolor("#16213e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")
        ax.tick_params(colors="#aaaacc", labelsize=8)
        ax.title.set_color("#ddddff")
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")

    # Sliders session / pièce
    n_sess   = len(session_ids)
    ax_sl_s  = fig.add_axes([0.07, 0.09, 0.35, 0.025], facecolor='#2a2a4e')
    ax_sl_p  = fig.add_axes([0.07, 0.05, 0.35, 0.025], facecolor='#2a2a4e')
    ax_btn   = fig.add_axes([0.50, 0.05, 0.10, 0.055], facecolor='#2a2a4e')
    ax_sl_sp = fig.add_axes([0.65, 0.05, 0.28, 0.025], facecolor='#2a2a4e')

    sl_sess  = Slider(ax_sl_s, 'Session', 1, n_sess, valinit=1, valstep=1, color='#5555cc')
    sl_piece = Slider(ax_sl_p, 'Pièce',   0, 99,     valinit=0, valstep=1, color='#cc5555')
    # Vitesse 1 = temps réel (données à 10 Hz = 100 ms/frame)
    sl_speed = Slider(ax_sl_sp, 'Vitesse ×', 1, 20, valinit=1, valstep=1, color='#55aa55')
    btn_play = Button(ax_btn, '▶ Play', color='#2a2a4e', hovercolor='#3a3a6e')

    for sl in [sl_sess, sl_piece, sl_speed]:
        sl.label.set_color('#aaaacc')
        sl.valtext.set_color('#ccccff')
    btn_play.label.set_color('#ccffcc')

    # État partagé entre les callbacks
    state = {
        'sid': sid,
        'piece_idx': piece_idx,
        'data': None,
        'anim': None,
        'playing': False,
        'frame': 0,
        # Artistes animés — recréés à chaque load_and_draw
        'arm_line': None,
        'trail_coll': None,
        'time_cursors': {},
    }

    def _style_ax(ax):
        ax.set_facecolor("#16213e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")
        ax.tick_params(colors="#aaaacc", labelsize=8)
        ax.title.set_color("#ddddff")
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")

    def _setup_static(data):
        """Dessine les éléments statiques et recrée les artistes animés."""
        q_real  = data['q_real']
        q_des   = data['q_des']
        cut     = data['is_cutting'].astype(bool)
        T       = len(q_real)
        t       = np.arange(T) * 0.1   # 10 Hz → pas de 100 ms

        x_des, y_des   = _fk(q_des)
        cut_dev         = data['cut_deviation']
        tau             = data['tau']
        dq_real         = data['dq_real']

        for ax in [ax_xy, ax_dev, ax_tau, ax_q1, ax_dq1, ax_q2, ax_dq2]:
            ax.clear()
            _style_ax(ax)

        # ── XY statique ──────────────────────────────────────────────────────
        ax_xy.plot(x_des, y_des, '--', color='#555577', linewidth=0.8,
                   alpha=0.6, zorder=2)
        if cut.any():
            x_cut, y_cut = x_des[cut], y_des[cut]
            if len(x_cut) > 2:
                xu, yu, xl, yl = _corridor(x_cut, y_cut, CUT_DEFECT_THRESHOLD)
                ax_xy.fill(np.concatenate([xu, xl[::-1]]),
                           np.concatenate([yu, yl[::-1]]),
                           color='gold', alpha=0.18, zorder=1)
        ax_xy.set_aspect('equal', 'box')
        ax_xy.grid(True, linestyle=':', alpha=0.2, color='#555577')
        ax_xy.set_xlim(-0.3, 2.5)
        ax_xy.set_ylim(-0.5, 2.5)
        ax_xy.set_title("Trajectoire XY", fontweight='bold')

        # Artistes animés recréés proprement après le clear()
        arm_line, = ax_xy.plot([], [], 'o-', color='#88aaff', linewidth=3,
                               markersize=7, zorder=10)
        # LineCollection pour la traînée colorée segment par segment
        trail_coll = LineCollection([], linewidths=2.0, zorder=5, alpha=0.85)
        ax_xy.add_collection(trail_coll)
        state['arm_line']   = arm_line
        state['trail_coll'] = trail_coll

        # ── Signaux ──────────────────────────────────────────────────────────
        ax_dev.plot(t, cut_dev * 1000, color='#66aaff', linewidth=1.0, alpha=0.4)
        ax_dev.axhline(CUT_DEFECT_THRESHOLD * 1000, color='#ff4444',
                       linestyle='--', linewidth=1.0)
        ax_dev.fill_between(t, 0, cut_dev * 1000,
                            where=cut, color='#ff4444', alpha=0.08)
        ax_dev.set_title("Déviation (mm)", fontweight='bold')
        ax_dev.set_ylim(bottom=0)
        ax_dev.grid(True, linestyle=':', alpha=0.3, color='#555577')

        ax_tau.plot(t, tau[:, 0], color='#cc88ff', linewidth=0.8, alpha=0.5)
        ax_tau.plot(t, tau[:, 1], color='#ffcc44', linewidth=0.8, alpha=0.5)
        ax_tau.set_title("Couples (N·m)", fontweight='bold')
        ax_tau.grid(True, linestyle=':', alpha=0.3, color='#555577')

        ax_q1.plot(t, np.degrees(data['q_des'][:, 0]), '--', color='#888888',
                   linewidth=0.8, alpha=0.5)
        ax_q1.plot(t, np.degrees(q_real[:, 0]), color='#66aaff', linewidth=0.8, alpha=0.6)
        ax_q1.set_title("Angle 1 (°)", fontweight='bold')
        ax_q1.grid(True, linestyle=':', alpha=0.3, color='#555577')

        ax_q2.plot(t, np.degrees(data['q_des'][:, 1]), '--', color='#888888',
                   linewidth=0.8, alpha=0.5)
        ax_q2.plot(t, np.degrees(q_real[:, 1]), color='#66aaff', linewidth=0.8, alpha=0.6)
        ax_q2.set_title("Angle 2 (°)", fontweight='bold')
        ax_q2.set_xlabel("Temps (s)")
        ax_q2.grid(True, linestyle=':', alpha=0.3, color='#555577')

        ax_dq1.plot(t, dq_real[:, 0], color='#ffaa44', linewidth=0.8, alpha=0.6)
        ax_dq1.set_title("Vit. ang. 1 (rad/s)", fontweight='bold')
        ax_dq1.grid(True, linestyle=':', alpha=0.3, color='#555577')

        ax_dq2.plot(t, dq_real[:, 1], color='#ffaa44', linewidth=0.8, alpha=0.6)
        ax_dq2.set_title("Vit. ang. 2 (rad/s)", fontweight='bold')
        ax_dq2.set_xlabel("Temps (s)")
        ax_dq2.grid(True, linestyle=':', alpha=0.3, color='#555577')

        # Curseurs temporels (recréés après clear)
        cursors = {}
        for name, ax in [('dev', ax_dev), ('tau', ax_tau), ('q1', ax_q1),
                          ('q2', ax_q2), ('dq1', ax_dq1), ('dq2', ax_dq2)]:
            cursors[name] = ax.axvline(0, color='white', linewidth=1.0, alpha=0.7)
        state['time_cursors'] = cursors

        pc    = int(data['piece_count'])          if 'piece_count' in data else 0
        cad   = float(data['cadence'])            if 'cadence'     in data else 0.0
        dur   = float(data['duration_per_segment'])
        n_def = int(data['cut_defect'].sum())
        fig.suptitle(
            f"Session {state['sid']}  |  Pièce {state['piece_idx']}  |  "
            f"n={pc}  cadence={cad:.0f} p/h  dur={dur:.2f} s/seg  défauts={n_def}",
            color='#ccccff', fontsize=10, fontweight='bold'
        )

    def _update_frame(frame):
        data = state['data']
        arm_line   = state['arm_line']
        trail_coll = state['trail_coll']
        if data is None or arm_line is None:
            return

        q_real  = data['q_real']
        cut     = data['is_cutting'].astype(bool)
        cut_dev = data['cut_deviation']
        T       = len(q_real)
        frame   = min(frame, T - 1)
        t_now   = frame * 0.1

        # Position du bras (3 points : origine, joint2, effecteur)
        xs, ys = _fk1(q_real[frame])
        arm_line.set_data(xs, ys)

        # Traînée colorée segment par segment via LineCollection
        # Couleur : gris=transit, vert=découpe OK, rouge=découpe défaut
        if frame > 0:
            x_all, y_all = _fk(q_real[:frame + 1])
            pts  = np.stack([x_all, y_all], axis=1).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)   # (frame, 1, 2, 2)

            colors = []
            for i in range(len(segs)):
                j = i + 1   # point d'arrivée du segment
                if cut[j] and cut_dev[j] > CUT_DEFECT_THRESHOLD:
                    colors.append('#ff4444')   # rouge : hors tolérance
                elif cut[j]:
                    colors.append('#44cc77')   # vert  : découpe OK
                else:
                    colors.append('#445588')   # bleu-gris : transit

            trail_coll.set_segments(segs)
            trail_coll.set_colors(colors)
        else:
            trail_coll.set_segments([])

        # Curseurs temporels synchronisés
        for vl in state['time_cursors'].values():
            vl.set_xdata([t_now, t_now])

        fig.canvas.draw_idle()

    def load_and_draw():
        data = _load_episode(data_dir, state['sid'], state['piece_idx'])
        if data is None:
            fig.suptitle(
                f"Épisode introuvable : session {state['sid']} pièce {state['piece_idx']}",
                color='#ff6666'
            )
            fig.canvas.draw_idle()
            return
        state['data']  = data
        state['frame'] = 0
        _setup_static(data)
        _update_frame(0)

    def on_sess(val):
        idx = int(sl_sess.val) - 1
        state['sid'] = session_ids[idx]
        state['playing'] = False
        btn_play.label.set_text('▶ Play')
        if state['anim'] is not None:
            state['anim'].event_source.stop()
        load_and_draw()

    def on_piece(val):
        state['piece_idx'] = int(sl_piece.val)
        state['playing'] = False
        btn_play.label.set_text('▶ Play')
        if state['anim'] is not None:
            state['anim'].event_source.stop()
        load_and_draw()

    def on_play(event):
        if state['data'] is None:
            return
        T = len(state['data']['q_real'])

        if state['playing']:
            state['playing'] = False
            btn_play.label.set_text('▶ Play')
            if state['anim'] is not None:
                state['anim'].event_source.stop()
            return

        state['playing'] = True
        btn_play.label.set_text('⏸ Pause')
        state['frame'] = 0

        def anim_func(_):
            if not state['playing']:
                return
            actual = state['frame']
            _update_frame(actual)
            # Avancer de `speed` frames de données par rafraîchissement écran
            # interval=100 ms + speed=1 → temps réel (données à 10 Hz)
            state['frame'] = (actual + max(1, int(sl_speed.val))) % T

        state['anim'] = animation.FuncAnimation(
            fig, anim_func, interval=100, cache_frame_data=False
        )
        fig.canvas.draw_idle()

    # Initialisation
    if sid in session_ids:
        sl_sess.set_val(session_ids.index(sid) + 1)
    sl_piece.set_val(max(0, piece_idx))
    state['sid']       = sid
    state['piece_idx'] = piece_idx
    load_and_draw()

    sl_sess.on_changed(on_sess)
    sl_piece.on_changed(on_piece)
    btn_play.on_clicked(on_play)
    plt.show()


# ─── mode : legacy ────────────────────────────────────────────────────────────

def view_legacy(data_dir):
    episodes = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith("episode_") and f.endswith(".npz")
    ]) if os.path.exists(data_dir) else []

    if not episodes:
        print(f"Aucun épisode legacy trouvé dans '{data_dir}'.")
        return

    num_episodes = len(episodes)
    fig = plt.figure(figsize=(18, 10))
    fig.canvas.manager.set_window_title("Visualiseur Legacy — Bras Découpe")
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
        idx    = int(episode_slider.val) - 1
        data   = np.load(os.path.join(data_dir, episodes[idx]))
        q_real    = data['q_real'];  q_sensed  = data['q_sensed']; q_des = data['q_des']
        dq_real   = data['dq_real']; dq_sensed = data['dq_sensed']; dq_des = data['dq_des']
        tau = data['tau']; is_cutting = data['is_cutting']
        cut_deviation = data['cut_deviation']; cut_defect = data['cut_defect']
        duration = float(data['duration_per_segment'])
        T = len(q_real); t = np.arange(T) * 0.1

        for ax in [ax_traj, ax_q1, ax_q2, ax_dq1, ax_dq2, ax_tau]:
            ax.clear()

        x_real, y_real = _fk(q_real); x_des, y_des = _fk(q_des)
        cut = is_cutting == 1.0; nocut = ~cut

        ax_traj.plot(x_des, y_des, '--', color='#999999', linewidth=1, alpha=0.6,
                     zorder=1, label='Désiré')
        x_cut_des, y_cut_des = x_des[cut], y_des[cut]
        if len(x_cut_des) > 2:
            xu, yu, xl, yl = _corridor(x_cut_des, y_cut_des, CUT_DEFECT_THRESHOLD)
            ax_traj.fill(np.concatenate([xu, xl[::-1]]), np.concatenate([yu, yl[::-1]]),
                         color='gold', alpha=0.35, zorder=2)
            ax_traj.plot(x_cut_des, y_cut_des, '-', color='#CC8800', linewidth=1.2,
                         alpha=0.8, zorder=3)
        ax_traj.scatter(x_real[nocut], y_real[nocut], c='#6699CC', s=2, alpha=0.4,
                        zorder=4, label='Hors coupe')
        x_cut_real, y_cut_real = x_real[cut], y_real[cut]
        defect_cut = cut_defect[cut]
        if len(x_cut_real) > 1:
            pts  = np.array([x_cut_real, y_cut_real]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            seg_colors = ['#CC2222' if d > 0.5 else '#22AA44' for d in defect_cut[:-1]]
            ax_traj.add_collection(LineCollection(segs, colors=seg_colors,
                                                  linewidths=2.0, zorder=5))
        n_def = int(defect_cut.sum()); n_cut = int(cut.sum())
        pct   = 100 * n_def / n_cut if n_cut > 0 else 0.0
        piece_id = episodes[idx].split('_')[1]
        ax_traj.set_title(f"Pièce {piece_id}  {duration:.2f}s/seg  "
                          f"Défauts {pct:.1f}% ({n_def}/{n_cut})",
                          fontweight='bold', fontsize=9)
        ax_traj.set_aspect('equal', 'box'); ax_traj.legend(loc='upper right', fontsize=7)
        ax_traj.grid(True, linestyle=':', alpha=0.5)

        for ax, key, label in [(ax_q1, 0, "Angle 1 (°)"), (ax_q2, 1, "Angle 2 (°)")]:
            ax.plot(t, np.degrees(q_des[:, key]),    'k--', alpha=0.6, label='Désiré')
            ax.plot(t, np.degrees(q_real[:, key]),   'b-',  linewidth=1.5, label='Réel')
            ax.plot(t, np.degrees(q_sensed[:, key]), 'g-',  alpha=0.5, label='Capteur')
            ax.set_title(label, fontweight='bold'); ax.legend(fontsize=7)
            ax.grid(True, linestyle=':', alpha=0.5)

        for ax, key, label in [(ax_dq1, 0, "Vit. ang. 1 (rad/s)"),
                               (ax_dq2, 1, "Vit. ang. 2 (rad/s)")]:
            ax.plot(t, dq_des[:, key],    'k--', alpha=0.6, label='Désiré')
            ax.plot(t, dq_real[:, key],   'b-',  linewidth=1.5, label='Réel')
            ax.plot(t, dq_sensed[:, key], 'g-',  alpha=0.5, label='Capteur')
            ax.set_title(label, fontweight='bold'); ax.set_xlabel("Temps (s)")
            ax.grid(True, linestyle=':', alpha=0.5)

        ax_tau.plot(t, tau[:, 0], 'm-', alpha=0.8, label='Moteur 1')
        ax_tau.plot(t, tau[:, 1], 'y-', alpha=0.8, label='Moteur 2')
        ylims = ax_tau.get_ylim()
        ax_tau.fill_between(t, ylims[0], ylims[1], where=cut,
                            color='red', alpha=0.08, label='Laser actif')
        ax_tau.set_title("Couples (N·m)", fontweight='bold'); ax_tau.legend(fontsize=7)
        ax_tau.grid(True, linestyle=':', alpha=0.5)

        fig.canvas.draw_idle()

    update(1)
    episode_slider.on_changed(update)
    plt.show()


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",    choices=["session", "episode", "legacy"],
                   default=None, help="Mode de visualisation")
    p.add_argument("--session", type=str, default=None,
                   help="ID de session (ex: 003) pour le mode episode")
    p.add_argument("--piece",   type=int, default=0,
                   help="N° de pièce dans la session pour le mode episode")
    p.add_argument("--data-dir", type=str, default=DATASET_DIR)
    args = p.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Dossier '{data_dir}' introuvable.")
        sys.exit(1)

    # Détection automatique du mode si non précisé
    if args.mode is None:
        session_ids = _list_sessions(data_dir)
        args.mode = "session" if session_ids else "legacy"

    if args.mode == "session":
        view_sessions(data_dir)

    elif args.mode == "episode":
        session_ids = _list_sessions(data_dir)
        sid = args.session
        if sid is None and session_ids:
            sid = session_ids[0]
        animate_episode(data_dir, sid=sid, piece_idx=args.piece)

    elif args.mode == "legacy":
        view_legacy(data_dir)


if __name__ == "__main__":
    main()
