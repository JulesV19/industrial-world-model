import glob
import os
import sys
import numpy as np
from arm import PhysicsArmEnv, RobotController, TrajectoryPlanner, inverse_kinematics
from piece import load_shape_database
from config import l1, l2

# Seuil en mètres au-delà duquel un point de découpe est considéré comme défectueux.
# À ajuster selon les résultats visuels.
CUT_DEFECT_THRESHOLD = 0.02


def fk(q):
    """Cinématique directe → position cartésienne de l'effecteur."""
    x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
    y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    return np.array([x, y])


def record_episode(waypoints, duration):
    planner = TrajectoryPlanner(waypoints, duration_per_segment=duration)
    env = PhysicsArmEnv(initial_q=planner.last_q)
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

        # Déviation cartésienne effecteur réel vs désiré (uniquement pendant la découpe)
        if cutting:
            cut_dev = float(np.linalg.norm(fk(q_real) - fk(q_des)))
        else:
            cut_dev = 0.0

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

    return dict(
        q_real        = np.array(log_q_real,    dtype=np.float32),
        dq_real       = np.array(log_dq_real,   dtype=np.float32),
        q_sensed      = np.array(log_q_sensed,  dtype=np.float32),
        dq_sensed     = np.array(log_dq_sensed, dtype=np.float32),
        tau           = np.array(log_tau,        dtype=np.float32),
        q_des         = np.array(log_q_des,      dtype=np.float32),
        dq_des        = np.array(log_dq_des,     dtype=np.float32),
        is_cutting    = np.array(log_is_cutting, dtype=np.float32),
        cut_deviation = cut_deviation,
        cut_defect    = cut_defect,
        duration_per_segment = np.float32(duration),
    )


def main():
    try:
        data = load_shape_database("pieces_database.json")
    except FileNotFoundError:
        print("Erreur : 'pieces_database.json' introuvable.")
        print("Lancez d'abord 'python3 generate_pieces.py'.")
        sys.exit(1)

    pieces    = data["pieces"]
    n_runs    = data["n_runs"]
    speed_min = data["speed_min"]
    speed_max = data["speed_max"]

    os.makedirs("dataset", exist_ok=True)
    old = glob.glob("dataset/*.npz")
    if old:
        for f in old:
            os.remove(f)
        print(f"  {len(old)} ancien(s) fichier(s) supprimé(s).")

    total = len(pieces) * n_runs
    print(f"Enregistrement de {len(pieces)} pièces × {n_runs} runs = {total} épisodes...")
    print(f"  Vitesse ∈ U[{speed_min}, {speed_max}] s/seg  |  Seuil défaut : {CUT_DEFECT_THRESHOLD} m")

    rng = np.random.default_rng(seed=0)
    idx = 0
    for piece_idx, waypoints in enumerate(pieces):
        for run in range(n_runs):
            duration = float(rng.uniform(speed_min, speed_max))
            idx += 1
            arrays = record_episode(waypoints, duration)

            filename = f"dataset/episode_{piece_idx+1:03d}_run{run:02d}.npz"
            np.savez_compressed(filename, **arrays)

            n_steps   = len(arrays["tau"])
            n_cutting = int(arrays["is_cutting"].sum())
            n_defects = int(arrays["cut_defect"].sum())
            defect_pct = 100 * n_defects / n_cutting if n_cutting > 0 else 0.0
            print(f"  [{idx:4d}/{total}] pièce {piece_idx+1:03d} run{run:02d} "
                  f"spd={duration:.3f}s → {n_steps} steps, "
                  f"défauts {n_defects}/{n_cutting} ({defect_pct:.1f}%)")

    print("Dataset généré dans 'dataset/'.")


if __name__ == "__main__":
    main()
