import argparse
import glob
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from arm import PhysicsArmEnv, RobotController, TrajectoryPlanner, inverse_kinematics
from piece import load_shape_database
from config import l1, l2
from degradation import ThermalModel

CUT_DEFECT_THRESHOLD = 0.02

N_SESSIONS          = 200
N_PIECES_PER_SESSION = 100
CADENCE_MIN         = 20.0   # pièces/heure
CADENCE_MAX         = 120.0  # pièces/heure


def fk(q):
    x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
    y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    return np.array([x, y])


def record_episode(waypoints: list, duration: float, machine_state=None,
                   temperature: float = 20.0):
    planner = TrajectoryPlanner(waypoints, duration_per_segment=duration)
    env = PhysicsArmEnv(initial_q=planner.last_q, machine_state=machine_state,
                        temperature=temperature)
    controller = RobotController()

    factory_state = "ARRIVING"
    conveyor_offset = -800.0

    log_q_real, log_dq_real       = [], []
    log_q_sensed, log_dq_sensed   = [], []
    log_tau                        = []
    log_q_des, log_dq_des          = [], []
    log_is_cutting                 = []
    log_cut_deviation              = []

    episode_done = False
    step_count = 0
    SUBSAMPLE = 10  # simulation 100 Hz → log 10 Hz

    while not episode_done:
        if factory_state == "CUTTING":
            q_des, dq_des, ddq_des = planner.get_desired_state(env.dt)
            if planner.done:
                factory_state = "EVACUATING"

        elif factory_state == "EVACUATING":
            q_des, dq_des, ddq_des = planner.get_desired_state(env.dt)
            conveyor_offset += 500 * env.dt
            if conveyor_offset > 900:
                episode_done = True

        elif factory_state == "ARRIVING":
            q_des, dq_des, ddq_des = planner.last_q, np.zeros(2), np.zeros(2)
            conveyor_offset += 500 * env.dt
            if conveyor_offset >= 0:
                conveyor_offset, factory_state = 0.0, "CUTTING"

        if episode_done:
            break

        q_sensed, dq_sensed = env.read_sensors()
        tau = controller.compute_torque(q_des, dq_des, ddq_des, q_sensed, dq_sensed)
        state = env.step(tau)

        q_real = state[0:2]
        cutting = (factory_state == "CUTTING" and planner.is_cutting)

        cut_dev = float(np.linalg.norm(fk(q_real) - fk(q_des))) if cutting else 0.0

        step_count += 1
        if step_count % SUBSAMPLE == 0:
            log_q_real.append(q_real.copy())
            log_dq_real.append(state[2:4].copy())
            log_q_sensed.append(q_sensed.copy())
            log_dq_sensed.append(dq_sensed.copy())
            log_tau.append(tau.copy())
            log_q_des.append(q_des.copy())
            log_dq_des.append(dq_des.copy())
            log_is_cutting.append(1.0 if cutting else 0.0)
            log_cut_deviation.append(cut_dev)

    cut_deviation = np.array(log_cut_deviation, dtype=np.float32)
    cut_defect    = (cut_deviation > CUT_DEFECT_THRESHOLD).astype(np.float32)

    cutting_mask = np.array(log_is_cutting, dtype=bool)
    cutting_devs = cut_deviation[cutting_mask]
    mean_cut_deviation = float(cutting_devs.mean()) if cutting_devs.size > 0 else 0.0

    # Agrégats mesurables en temps réel depuis les capteurs existants
    tau_arr     = np.array(log_tau,    dtype=np.float32)
    q_real_arr  = np.array(log_q_real, dtype=np.float32)
    q_des_arr   = np.array(log_q_des,  dtype=np.float32)
    q_error_arr = q_real_arr - q_des_arr
    if cutting_mask.any():
        tau_cut_rms     = float(np.sqrt(np.mean(tau_arr[cutting_mask] ** 2)))
        q_error_cut_rms = float(np.sqrt(np.mean(q_error_arr[cutting_mask] ** 2)))
    else:
        tau_cut_rms     = 0.0
        q_error_cut_rms = 0.0

    result = dict(
        q_real        = q_real_arr,
        dq_real       = np.array(log_dq_real,   dtype=np.float32),
        q_sensed      = np.array(log_q_sensed,  dtype=np.float32),
        dq_sensed     = np.array(log_dq_sensed, dtype=np.float32),
        tau           = tau_arr,
        q_des         = q_des_arr,
        dq_des        = np.array(log_dq_des,     dtype=np.float32),
        is_cutting    = np.array(log_is_cutting, dtype=np.float32),
        cut_deviation = cut_deviation,
        cut_defect    = cut_defect,
        duration_per_segment = np.float32(duration),
        mean_cut_deviation   = np.float32(mean_cut_deviation),
        tau_cut_rms          = np.float32(tau_cut_rms),
        q_error_cut_rms      = np.float32(q_error_cut_rms),
        piece_count   = np.int32(machine_state['piece_count'] if machine_state else 0),
        cadence       = np.float32(machine_state['cadence']   if machine_state else 0.0),
        temperature   = np.float32(temperature),
        waypoints     = np.array(waypoints,      dtype=np.float32),
    )
    return result, mean_cut_deviation


def _worker_single(args):
    piece_idx, waypoints, run, duration = args
    arrays, _ = record_episode(waypoints, duration)
    return piece_idx, run, arrays, duration


def _run_single_mode(data, rng, out_dir):
    pieces    = data["pieces"]
    n_runs    = data["n_runs"]
    speed_min = data["speed_min"]
    speed_max = data["speed_max"]

    old = glob.glob(os.path.join(out_dir, "*.npz"))
    if old:
        for f in old:
            os.remove(f)
        print(f"  {len(old)} ancien(s) fichier(s) supprimé(s).")

    # Pré-génération séquentielle des paramètres aléatoires (ordre identique à l'original)
    jobs = []
    for piece_idx, waypoints in enumerate(pieces):
        for run in range(n_runs):
            duration = float(rng.uniform(speed_min, speed_max))
            jobs.append((piece_idx, waypoints, run, duration))

    total = len(jobs)
    print(f"Mode single : {len(pieces)} pièces × {n_runs} runs = {total} épisodes")
    print(f"  Vitesse ∈ U[{speed_min}, {speed_max}] s/seg  |  Seuil défaut : {CUT_DEFECT_THRESHOLD} m")

    results = {}
    n_workers = max(1, (os.cpu_count() or 1) - 1)
    total_defects = 0
    total_cutting = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker_single, job): job for job in jobs}
        with tqdm(total=total, unit="ep", desc="Simulation") as pbar:
            for future in as_completed(futures):
                piece_idx, run, arrays, duration = future.result()
                results[(piece_idx, run)] = (arrays, duration)
                n_cutting = int(arrays["is_cutting"].sum())
                n_defects = int(arrays["cut_defect"].sum())
                total_cutting += n_cutting
                total_defects += n_defects
                defect_pct = 100 * total_defects / total_cutting if total_cutting > 0 else 0.0
                pbar.set_postfix(défauts=f"{defect_pct:.1f}%")
                pbar.update(1)

    # Écriture dans l'ordre original
    with tqdm(total=total, unit="ep", desc="Écriture  ") as pbar:
        for piece_idx, waypoints, run, duration in jobs:
            arrays, _ = results[(piece_idx, run)]
            filename = os.path.join(out_dir, f"episode_{piece_idx+1:03d}_run{run:02d}.npz")
            np.savez_compressed(filename, **arrays)
            pbar.update(1)


def _worker_session(args):
    sess, n, waypoints, duration, cadence, temperature = args
    machine_state = {'piece_count': n, 'cadence': cadence}
    arrays, mean_dev = record_episode(waypoints, duration, machine_state,
                                      temperature=temperature)
    return sess, n, arrays, mean_dev, cadence, duration, temperature


def _run_session_mode(data, rng, out_dir, n_sessions, n_pieces):
    pieces    = data["pieces"]
    speed_min = data["speed_min"]
    speed_max = data["speed_max"]

    old = glob.glob(os.path.join(out_dir, "*.npz")) + \
          glob.glob(os.path.join(out_dir, "session_*_deviations.npy")) + \
          glob.glob(os.path.join(out_dir, "session_*_cadence.npy"))
    if old:
        for f in old:
            os.remove(f)
        print(f"  {len(old)} ancien(s) fichier(s) supprimé(s).")

    print(f"Mode session : {n_sessions} sessions × {n_pieces} pièces")
    print(f"  Cadence ∈ U[{CADENCE_MIN}, {CADENCE_MAX}] pièces/h")
    print(f"  Vitesse ∈ U[{speed_min}, {speed_max}] s/seg  |  Seuil défaut : {CUT_DEFECT_THRESHOLD} m")

    # Pré-génération séquentielle des paramètres aléatoires (ordre identique à l'original)
    # Le modèle thermique est instancié ici (séquentiellement) pour être déterministe :
    # la température de chaque pièce est calculée analytiquement avant de lancer les workers.
    jobs = []
    session_meta = {}  # sess -> (cadence, duration)
    for sess in range(n_sessions):
        cadence  = float(rng.uniform(CADENCE_MIN, CADENCE_MAX))
        duration = float(rng.uniform(speed_min, speed_max))
        session_meta[sess] = (cadence, duration)

        # Calcul déterministe de la température de chaque pièce dans la session
        thermal = ThermalModel(cadence=cadence)
        for n in range(n_pieces):
            piece_idx = int(rng.integers(0, len(pieces)))
            waypoints = pieces[piece_idx]
            # Température au début de cette pièce (avant que la pièce chauffe davantage)
            piece_temp = thermal.temperature
            # Avancer le modèle thermique de la durée d'une pièce
            thermal.advance_piece(duration_s=duration)
            jobs.append((sess, n, waypoints, duration, cadence, piece_temp))

    total = len(jobs)
    results = {}
    n_workers = max(1, (os.cpu_count() or 1) - 1)
    total_defects = 0
    total_cutting = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker_session, job): job for job in jobs}
        with tqdm(total=total, unit="ep", desc="Simulation") as pbar:
            for future in as_completed(futures):
                sess, n, arrays, mean_dev, cadence, duration, temperature = future.result()
                results[(sess, n)] = (arrays, mean_dev, cadence, duration, temperature)
                n_cutting = int(arrays["is_cutting"].sum())
                n_defects = int(arrays["cut_defect"].sum())
                total_cutting += n_cutting
                total_defects += n_defects
                defect_pct = 100 * total_defects / total_cutting if total_cutting > 0 else 0.0
                pbar.set_postfix(défauts=f"{defect_pct:.1f}%", dev=f"{mean_dev*1000:.2f}mm")
                pbar.update(1)

    # Écriture dans l'ordre original + reconstruction des historiques par session
    with tqdm(total=total, unit="ep", desc="Écriture  ") as pbar:
        for sess in range(n_sessions):
            cadence, _ = session_meta[sess]
            # Historique (N, 3) : [mean_cut_deviation, tau_cut_rms, q_error_cut_rms]
            session_history = []
            for n in range(n_pieces):
                arrays, mean_dev, cadence, duration, temperature = results[(sess, n)]
                session_history.append([
                    mean_dev,
                    float(arrays["tau_cut_rms"]),
                    float(arrays["q_error_cut_rms"]),
                ])
                filename = os.path.join(out_dir, f"session_{sess:03d}_piece{n:04d}.npz")
                np.savez_compressed(filename, **arrays)
                pbar.update(1)

            hist_arr = np.array(session_history, dtype=np.float32)   # (N, 3)
            np.save(os.path.join(out_dir, f"session_{sess:03d}_history.npy"), hist_arr)
            # Garder deviations.npy (colonne 0) pour la vue session du comparateur
            np.save(os.path.join(out_dir, f"session_{sess:03d}_deviations.npy"),
                    hist_arr[:, 0])
            np.save(os.path.join(out_dir, f"session_{sess:03d}_cadence.npy"),
                    np.float32(cadence))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "session"], default="single",
                        help="single : comportement d'origine | session : sessions avec dégradation")
    parser.add_argument("--n-sessions",  type=int,   default=N_SESSIONS)
    parser.add_argument("--n-pieces",    type=int,   default=N_PIECES_PER_SESSION)
    parser.add_argument("--speed-min",   type=float, default=None)
    parser.add_argument("--speed-max",   type=float, default=None)
    parser.add_argument("--out",         type=str,   default="dataset")
    parser.add_argument("--seed",        type=int,   default=0)
    args = parser.parse_args()

    try:
        data = load_shape_database("pieces_database.json")
    except FileNotFoundError:
        print("Erreur : 'pieces_database.json' introuvable.")
        print("Lancez d'abord 'python3 generate_pieces.py'.")
        sys.exit(1)

    if args.speed_min is not None:
        data["speed_min"] = args.speed_min
    if args.speed_max is not None:
        data["speed_max"] = args.speed_max

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.mode == "single":
        _run_single_mode(data, rng, args.out)
    else:
        _run_session_mode(data, rng, args.out, args.n_sessions, args.n_pieces)

    print(f"Dataset généré dans '{args.out}/'.")


if __name__ == "__main__":
    main()
