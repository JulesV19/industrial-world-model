"""
Enregistre des épisodes de perçage dans dataset/.

Pour chaque épisode :
  - les 4 coins réels sont extraits d'un .npz de découpe (via piece_input)
  - le bras visite chaque coin séquentiellement
  - la position réelle du foret est enregistrée à chaque transition de segment
  - défaut = distance réelle vs cible > DRILL_DEFECT_THRESHOLD

Usage :
    python3 record_dataset.py
    python3 record_dataset.py --decoupe-dataset ../arm_decoupe/dataset --n-episodes 200
"""

import argparse
import glob
import os
import sys

import numpy as np

from arm import PhysicsArmEnv, RobotController, TrajectoryPlanner, inverse_kinematics
from piece_input import extract_corners
from config import l1, l2

DRILL_DEFECT_THRESHOLD = 0.02        # mètres
SUBSAMPLE = 5                        # 100 Hz → log 20 Hz
HOME = np.array([0.1, 0.1])          # position de repos du bras
DRILL_DWELL_TIME = 3.0               # secondes d'immobilisation par trou


def fk(q):
    x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
    y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    return np.array([x, y])


def record_episode(corners: np.ndarray, duration: float) -> dict:
    """
    Simule un épisode complet de perçage :
      HOME (False) → coin0 (True) → coin1 (True) → coin2 (True) → coin3 (True) → HOME (False)

    Le bras part de HOME, s'approche du premier coin, perce les 4 coins dans
    l'ordre, puis revient à HOME. La position réelle du foret est enregistrée
    à chaque transition de segment correspondant à une arrivée sur un coin
    (segments 1 à 4).
    """
    wp = (
        [[HOME[0], HOME[1], False]] +
        [[c[0], c[1], True] for c in corners] +
        [[HOME[0], HOME[1], False]]
    )

    planner = TrajectoryPlanner(wp, duration_per_segment=duration)
    env = PhysicsArmEnv(initial_q=planner.last_q)  # last_q = IK(HOME)
    controller = RobotController()

    log_q_real,    log_dq_real    = [], []
    log_q_sensed,  log_dq_sensed  = [], []
    log_tau                       = []
    log_q_des                     = []
    log_is_drilling               = []

    drill_hits   = []
    prev_segment = planner.current_segment
    step         = 0

    while not planner.done:
        q_sensed, dq_sensed        = env.read_sensors()
        q_des, dq_des, ddq_des     = planner.get_desired_state(env.dt)
        tau                        = controller.compute_torque(q_des, dq_des, ddq_des,
                                                               q_sensed, dq_sensed)
        env.step(tau)

        # Transition de segment : arrivée sur un coin → dwell de perçage
        if planner.current_segment != prev_segment:
            new_seg = planner.current_segment
            if 1 <= new_seg <= 4:
                # Position tenue pendant le dwell = IK du coin courant
                q_hold = planner.last_q.copy()
                drill_hits.append(fk(env.state[:2]).copy())

                # Immobilisation DRILL_DWELL_TIME secondes (perçage simulé)
                dwell_steps = int(DRILL_DWELL_TIME / env.dt)
                for d in range(dwell_steps):
                    q_s, dq_s = env.read_sensors()
                    tau_d = controller.compute_torque(
                        q_hold, np.zeros(2), np.zeros(2), q_s, dq_s
                    )
                    env.step(tau_d)
                    step += 1
                    if step % SUBSAMPLE == 0:
                        log_q_real.append(env.state[:2].copy())
                        log_dq_real.append(env.state[2:].copy())
                        log_q_sensed.append(q_s.copy())
                        log_dq_sensed.append(dq_s.copy())
                        log_tau.append(tau_d.copy())
                        log_q_des.append(q_hold.copy())
                        log_is_drilling.append(2.0)   # 2 = phase de perçage actif

            prev_segment = new_seg

        step += 1
        if step % SUBSAMPLE == 0:
            log_q_real.append(env.state[:2].copy())
            log_dq_real.append(env.state[2:].copy())
            log_q_sensed.append(q_sensed.copy())
            log_dq_sensed.append(dq_sensed.copy())
            log_tau.append(tau.copy())
            log_q_des.append(q_des.copy())
            log_is_drilling.append(1.0 if planner.is_cutting else 0.0)

    drill_hits = np.array(drill_hits[:4], dtype=np.float32)   # (4, 2)
    corners_f  = corners.astype(np.float32)
    errors     = np.linalg.norm(drill_hits - corners_f, axis=1)
    defects    = (errors > DRILL_DEFECT_THRESHOLD).astype(np.float32)

    return dict(
        q_real           = np.array(log_q_real,       dtype=np.float32),
        dq_real          = np.array(log_dq_real,      dtype=np.float32),
        q_sensed         = np.array(log_q_sensed,     dtype=np.float32),
        dq_sensed        = np.array(log_dq_sensed,    dtype=np.float32),
        tau              = np.array(log_tau,           dtype=np.float32),
        q_des            = np.array(log_q_des,         dtype=np.float32),
        is_drilling      = np.array(log_is_drilling,   dtype=np.float32),
        corner_targets   = corners_f,
        drill_hits       = drill_hits,
        errors           = errors.astype(np.float32),
        defects          = defects,
        duration_per_segment = np.float32(duration),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes",      type=int,   default=100)
    parser.add_argument("--decoupe-dataset", type=str,   default="../arm_decoupe/dataset")
    parser.add_argument("--decoupe-db",      type=str,   default="../arm_decoupe/pieces_database.json")
    parser.add_argument("--speed-min",       type=float, default=1.0)
    parser.add_argument("--speed-max",       type=float, default=3.0)
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--out",             type=str,   default="dataset")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    decoupe_files = sorted(glob.glob(os.path.join(args.decoupe_dataset, "*.npz")))
    if not decoupe_files:
        print(f"Erreur : aucun .npz dans {args.decoupe_dataset}")
        sys.exit(1)

    n = min(args.n_episodes, len(decoupe_files))
    print(f"Enregistrement de {n} épisodes de perçage...")
    print(f"  DB découpe : {args.decoupe_db}")
    print(f"  Seuil défaut : {DRILL_DEFECT_THRESHOLD} m")

    for i, npz_path in enumerate(decoupe_files[:n]):
        corners  = extract_corners(npz_path, args.decoupe_db, rng=rng)
        duration = float(rng.uniform(args.speed_min, args.speed_max))
        episode  = record_episode(corners, duration)

        # Contour réel de la découpe (pour visualisation)
        cut_data = np.load(npz_path)
        cut_q    = cut_data["q_real"].astype(np.float32)
        cut_mask = cut_data["is_cutting"].astype(bool)
        cut_x = l1 * np.cos(cut_q[:, 0]) + l2 * np.cos(cut_q[:, 0] + cut_q[:, 1])
        cut_y = l1 * np.sin(cut_q[:, 0]) + l2 * np.sin(cut_q[:, 0] + cut_q[:, 1])
        episode["cut_contour_x"] = cut_x[cut_mask].astype(np.float32)
        episode["cut_contour_y"] = cut_y[cut_mask].astype(np.float32)

        out_path = os.path.join(args.out, f"episode_{i+1:03d}.npz")
        np.savez_compressed(out_path, **episode)

        n_def = int(episode["defects"].sum())
        errs  = episode["errors"]
        print(f"  [{i+1:03d}/{n}] {os.path.basename(npz_path):30s} "
              f"spd={duration:.2f}s  "
              f"err_moy={errs.mean()*1000:.1f}mm  "
              f"défauts={n_def}/4")

    print(f"Dataset généré dans '{args.out}/'.")


if __name__ == "__main__":
    main()
