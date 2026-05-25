"""
Enregistre des épisodes de perçage dans dataset/.

Usage :
    python3 record_dataset.py
    python3 record_dataset.py --mode session --n-sessions 200 --n-pieces 100
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

N_SESSIONS           = 200
N_PIECES_PER_SESSION = 100
CADENCE_MIN          = 20.0          # pièces/heure
CADENCE_MAX          = 120.0         # pièces/heure


def fk(q):
    x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
    y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    return np.array([x, y])


def record_episode(corners: np.ndarray, duration: float, machine_state=None) -> tuple[dict, float]:
    """
    Simule un épisode complet de perçage.
    Retourne (dict_arrays, mean_error).
    """
    wp = (
        [[HOME[0], HOME[1], False]] +
        [[c[0], c[1], True] for c in corners] +
        [[HOME[0], HOME[1], False]]
    )

    planner = TrajectoryPlanner(wp, duration_per_segment=duration)
    env = PhysicsArmEnv(initial_q=planner.last_q, machine_state=machine_state)
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

        if planner.current_segment != prev_segment:
            new_seg = planner.current_segment
            if 1 <= new_seg <= 4:
                q_hold = planner.last_q.copy()
                drill_hits.append(fk(env.state[:2]).copy())

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
                        log_is_drilling.append(2.0)

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

    drill_hits = np.array(drill_hits[:4], dtype=np.float32)
    corners_f  = corners.astype(np.float32)
    errors     = np.linalg.norm(drill_hits - corners_f, axis=1)
    defects    = (errors > DRILL_DEFECT_THRESHOLD).astype(np.float32)
    mean_error = float(errors.mean())

    result = dict(
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
        mean_error       = np.float32(mean_error),
        piece_count      = np.int32(machine_state['piece_count'] if machine_state else 0),
        cadence          = np.float32(machine_state['cadence']   if machine_state else 0.0),
    )
    return result, mean_error


def _run_single_mode(args, rng):
    out_dir = args.out
    decoupe_files = sorted(glob.glob(os.path.join(args.decoupe_dataset, "*.npz")))
    if not decoupe_files:
        print(f"Erreur : aucun .npz dans {args.decoupe_dataset}")
        sys.exit(1)

    n = min(args.n_episodes, len(decoupe_files))
    print(f"Mode single : {n} épisodes de perçage")
    print(f"  DB découpe : {args.decoupe_db}  |  Seuil défaut : {DRILL_DEFECT_THRESHOLD} m")

    for i, npz_path in enumerate(decoupe_files[:n]):
        corners  = extract_corners(npz_path, args.decoupe_db, rng=rng)
        duration = float(rng.uniform(args.speed_min, args.speed_max))
        episode, _ = record_episode(corners, duration)

        cut_data = np.load(npz_path)
        cut_q    = cut_data["q_real"].astype(np.float32)
        cut_mask = cut_data["is_cutting"].astype(bool)
        cut_x = l1 * np.cos(cut_q[:, 0]) + l2 * np.cos(cut_q[:, 0] + cut_q[:, 1])
        cut_y = l1 * np.sin(cut_q[:, 0]) + l2 * np.sin(cut_q[:, 0] + cut_q[:, 1])
        episode["cut_contour_x"] = cut_x[cut_mask].astype(np.float32)
        episode["cut_contour_y"] = cut_y[cut_mask].astype(np.float32)

        out_path = os.path.join(out_dir, f"episode_{i+1:03d}.npz")
        np.savez_compressed(out_path, **episode)

        n_def = int(episode["defects"].sum())
        errs  = episode["errors"]
        print(f"  [{i+1:03d}/{n}] {os.path.basename(npz_path):30s} "
              f"spd={duration:.2f}s  "
              f"err_moy={errs.mean()*1000:.1f}mm  "
              f"défauts={n_def}/4")


def _run_session_mode(args, rng):
    out_dir = args.out
    decoupe_dataset = args.decoupe_dataset

    # Pour le mode session on a besoin des fichiers session de découpe
    decoupe_session_files = sorted(glob.glob(
        os.path.join(decoupe_dataset, "session_*_piece*.npz")
    ))
    if not decoupe_session_files:
        # Fallback sur les épisodes classiques
        decoupe_session_files = sorted(glob.glob(os.path.join(decoupe_dataset, "*.npz")))
    if not decoupe_session_files:
        print(f"Erreur : aucun .npz dans {decoupe_dataset}")
        sys.exit(1)

    n_sessions = args.n_sessions
    n_pieces   = args.n_pieces

    old = glob.glob(os.path.join(out_dir, "*.npz")) + \
          glob.glob(os.path.join(out_dir, "session_*_deviations.npy")) + \
          glob.glob(os.path.join(out_dir, "session_*_cadence.npy"))
    if old:
        for f in old:
            os.remove(f)
        print(f"  {len(old)} ancien(s) fichier(s) supprimé(s).")

    print(f"Mode session : {n_sessions} sessions × {n_pieces} pièces")
    print(f"  Cadence ∈ U[{CADENCE_MIN}, {CADENCE_MAX}] pièces/h")

    total = n_sessions * n_pieces
    idx = 0
    for sess in range(n_sessions):
        cadence  = float(rng.uniform(CADENCE_MIN, CADENCE_MAX))
        duration = float(rng.uniform(args.speed_min, args.speed_max))
        session_errors = []

        for n in range(n_pieces):
            machine_state = {'piece_count': n, 'cadence': cadence}
            npz_path = decoupe_session_files[int(rng.integers(0, len(decoupe_session_files)))]
            corners  = extract_corners(npz_path, args.decoupe_db, rng=rng)
            idx += 1

            episode, mean_err = record_episode(corners, duration, machine_state)
            session_errors.append(mean_err)

            filename = os.path.join(out_dir, f"session_{sess:03d}_piece{n:04d}.npz")
            np.savez_compressed(filename, **episode)

            n_def = int(episode["defects"].sum())
            print(f"  [{idx:5d}/{total}] sess{sess:03d} n={n:03d} "
                  f"cad={cadence:.0f}p/h spd={duration:.2f}s "
                  f"err_moy={mean_err*1000:.2f}mm "
                  f"défauts={n_def}/4")

        np.save(os.path.join(out_dir, f"session_{sess:03d}_deviations.npy"),
                np.array(session_errors, dtype=np.float32))
        np.save(os.path.join(out_dir, f"session_{sess:03d}_cadence.npy"),
                np.float32(cadence))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",           choices=["single", "session"], default="single")
    parser.add_argument("--n-episodes",     type=int,   default=100)
    parser.add_argument("--n-sessions",     type=int,   default=N_SESSIONS)
    parser.add_argument("--n-pieces",       type=int,   default=N_PIECES_PER_SESSION)
    parser.add_argument("--decoupe-dataset",type=str,   default="../arm_decoupe/dataset")
    parser.add_argument("--decoupe-db",     type=str,   default="../arm_decoupe/pieces_database.json")
    parser.add_argument("--speed-min",      type=float, default=1.0)
    parser.add_argument("--speed-max",      type=float, default=3.0)
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--out",            type=str,   default="dataset")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.mode == "single":
        _run_single_mode(args, rng)
    else:
        _run_session_mode(args, rng)

    print(f"Dataset généré dans '{args.out}/'.")


if __name__ == "__main__":
    main()
