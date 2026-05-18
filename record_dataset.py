import os
import sys
import numpy as np
from arm import PhysicsArmEnv, RobotController, TrajectoryPlanner
from piece import load_shape_database

def main():
    try:
        database = load_shape_database("pieces_database.json")
    except FileNotFoundError:
        print("Erreur : Le fichier 'pieces_database.json' est introuvable.")
        print("Veuillez d'abord lancer 'python3 generate_pieces.py' pour le créer.")
        sys.exit(1)

    os.makedirs("dataset", exist_ok=True)
    
    print(f"Début de l'enregistrement du dataset pour {len(database)} pièces...")
    
    for piece_idx, waypoints in enumerate(database):
        # On recrée l'environnement pour chaque pièce
        planner = TrajectoryPlanner(waypoints, duration_per_segment=0.8)
        env = PhysicsArmEnv(initial_q=planner.last_q)
        controller = RobotController()
        
        factory_state = "ARRIVING"
        conveyor_offset = -800.0
        
        # Initialisation des logs
        log_q_real, log_dq_real = [], []
        log_q_sensed, log_dq_sensed = [], []
        log_tau = []
        log_q_des, log_dq_des = [], []
        log_cutting = []
        
        episode_done = False
        
        while not episode_done:
            # 1. Logique Usine
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
                
            # 2. Lecture et Commande
            q_sensed, dq_sensed = env.read_sensors()
            tau = controller.compute_torque(q_des, dq_des, ddq_des, q_sensed, dq_sensed)
            state = env.step(tau)
            
            # Enregistrement
            log_q_real.append(state[0:2].copy())
            log_dq_real.append(state[2:4].copy())
            log_q_sensed.append(q_sensed.copy())
            log_dq_sensed.append(dq_sensed.copy())
            log_tau.append(tau.copy())
            log_q_des.append(q_des.copy())
            log_dq_des.append(dq_des.copy())
            log_cutting.append(1.0 if (factory_state == "CUTTING" and planner.is_cutting) else 0.0)

        # Sauvegarde de l'épisode
        filename = f"dataset/episode_{piece_idx+1:03d}.npz"
        np.savez_compressed(
            filename,
            q_real=np.array(log_q_real, dtype=np.float32),
            dq_real=np.array(log_dq_real, dtype=np.float32),
            q_sensed=np.array(log_q_sensed, dtype=np.float32),
            dq_sensed=np.array(log_dq_sensed, dtype=np.float32),
            tau=np.array(log_tau, dtype=np.float32),
            q_des=np.array(log_q_des, dtype=np.float32),
            dq_des=np.array(log_dq_des, dtype=np.float32),
            is_cutting=np.array(log_cutting, dtype=np.float32)
        )
        print(f"Pièce {piece_idx+1}/{len(database)} -> Sauvegardée dans {filename} ({len(log_tau)} steps)")

    print("Dataset généré avec succès dans le dossier 'dataset/' !")

if __name__ == "__main__":
    main()
