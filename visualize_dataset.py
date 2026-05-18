import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

from config import l1, l2

def main():
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        print(f"Erreur : Aucun dataset trouvé dans '{dataset_dir}'.")
        print("Veuillez d'abord exécuter 'python3 record_dataset.py'.")
        return

    # Compter le nombre de fichiers .npz
    episodes = [f for f in os.listdir(dataset_dir) if f.startswith("episode_") and f.endswith(".npz")]
    num_episodes = len(episodes)
    
    if num_episodes == 0:
        print("Aucun fichier d'épisode valide trouvé.")
        return

    # Configuration de la fenêtre Matplotlib (très grande pour tout voir)
    fig = plt.figure(figsize=(18, 10))
    fig.canvas.manager.set_window_title("Visualiseur de Dataset World Model")
    
    # Ajustement des marges pour laisser de la place au slider en bas
    fig.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.92, wspace=0.25, hspace=0.35)

    # Création de 6 sous-graphiques (2 lignes, 3 colonnes)
    ax_traj = plt.subplot(2, 3, 1)
    ax_q1 = plt.subplot(2, 3, 2)
    ax_q2 = plt.subplot(2, 3, 3)
    ax_tau = plt.subplot(2, 3, 4)
    ax_dq1 = plt.subplot(2, 3, 5)
    ax_dq2 = plt.subplot(2, 3, 6)

    # Création du Slider
    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor=axcolor)
    episode_slider = Slider(
        ax=ax_slider,
        label='Numéro de la Pièce (Épisode) ',
        valmin=1,
        valmax=num_episodes,
        valinit=1,
        valstep=1
    )

    def update(val):
        idx = int(episode_slider.val)
        filename = os.path.join(dataset_dir, f"episode_{idx:03d}.npz")
        
        try:
            data = np.load(filename)
        except Exception as e:
            print(f"Erreur lors du chargement de {filename}: {e}")
            return

        # Extraction des données
        q_real = data['q_real']
        q_sensed = data['q_sensed']
        q_des = data['q_des']
        
        dq_real = data['dq_real']
        dq_sensed = data['dq_sensed']
        dq_des = data['dq_des']
        
        tau = data['tau']
        is_cutting = data['is_cutting']
        
        T = len(q_real)
        t = np.arange(T) * 0.01 # dt = 0.01s (100 Hz)

        # Nettoyage des graphes
        for ax in [ax_traj, ax_q1, ax_q2, ax_dq1, ax_dq2, ax_tau]:
            ax.clear()

        # --- 1. TRAJECTOIRE XY (End-effector) ---
        # Cinématique directe basique (attention au repère visuel inversé de Pygame, ici on trace en math pur)
        x_real = l1 * np.cos(q_real[:, 0]) + l2 * np.cos(q_real[:, 0] + q_real[:, 1])
        y_real = l1 * np.sin(q_real[:, 0]) + l2 * np.sin(q_real[:, 0] + q_real[:, 1])
        
        x_des = l1 * np.cos(q_des[:, 0]) + l2 * np.cos(q_des[:, 0] + q_des[:, 1])
        y_des = l1 * np.sin(q_des[:, 0]) + l2 * np.sin(q_des[:, 0] + q_des[:, 1])

        ax_traj.plot(x_des, y_des, 'k--', label='Désiré (Plan)', alpha=0.5, linewidth=1)
        
        # Séparation des moments de coupe et de non-coupe
        cut_idx = is_cutting == 1.0
        nocut_idx = is_cutting == 0.0
        
        ax_traj.scatter(x_real[nocut_idx], y_real[nocut_idx], c='blue', s=2, label='Déplacement (Hors Coupe)', alpha=0.5)
        ax_traj.scatter(x_real[cut_idx], y_real[cut_idx], c='red', s=5, label='Découpe Active')
        
        ax_traj.set_title(f"Trajectoire Outil (Pièce {idx})", fontweight='bold')
        ax_traj.set_aspect('equal', 'box')
        ax_traj.legend(loc='upper right', fontsize='small')
        ax_traj.grid(True, linestyle=':', alpha=0.6)

        # --- 2. ANGLES (Q1 et Q2) ---
        ax_q1.plot(t, np.degrees(q_des[:, 0]), 'k--', label='Désiré', alpha=0.7)
        ax_q1.plot(t, np.degrees(q_real[:, 0]), 'b-', label='Réel (Caché)', linewidth=2)
        ax_q1.plot(t, np.degrees(q_sensed[:, 0]), 'g-', label='Mesure (Capteur)', alpha=0.6)
        ax_q1.set_title("Angle 1 (Degrés)", fontweight='bold')
        ax_q1.set_ylabel("Degrés")
        ax_q1.legend(loc='best', fontsize='small')
        ax_q1.grid(True, linestyle=':', alpha=0.6)

        ax_q2.plot(t, np.degrees(q_des[:, 1]), 'k--', label='Désiré', alpha=0.7)
        ax_q2.plot(t, np.degrees(q_real[:, 1]), 'b-', label='Réel (Caché)', linewidth=2)
        ax_q2.plot(t, np.degrees(q_sensed[:, 1]), 'g-', label='Mesure (Capteur)', alpha=0.6)
        ax_q2.set_title("Angle 2 (Degrés)", fontweight='bold')
        ax_q2.legend(loc='best', fontsize='small')
        ax_q2.grid(True, linestyle=':', alpha=0.6)

        # --- 3. VITESSES ANGULAIRES (dQ1 et dQ2) ---
        ax_dq1.plot(t, dq_des[:, 0], 'k--', label='Désiré', alpha=0.7)
        ax_dq1.plot(t, dq_real[:, 0], 'b-', label='Réel', linewidth=1.5)
        ax_dq1.plot(t, dq_sensed[:, 0], 'g-', label='Mesure (Bruitée)', alpha=0.5)
        ax_dq1.set_title("Vitesse Angulaire 1 (rad/s)", fontweight='bold')
        ax_dq1.set_xlabel("Temps (s)")
        ax_dq1.set_ylabel("rad/s")
        ax_dq1.grid(True, linestyle=':', alpha=0.6)

        ax_dq2.plot(t, dq_des[:, 1], 'k--', label='Désiré', alpha=0.7)
        ax_dq2.plot(t, dq_real[:, 1], 'b-', label='Réel', linewidth=1.5)
        ax_dq2.plot(t, dq_sensed[:, 1], 'g-', label='Mesure (Bruitée)', alpha=0.5)
        ax_dq2.set_title("Vitesse Angulaire 2 (rad/s)", fontweight='bold')
        ax_dq2.set_xlabel("Temps (s)")
        ax_dq2.grid(True, linestyle=':', alpha=0.6)

        # --- 4. COUPLES MOTEURS (Tau) ET ÉTAT D'USINE ---
        ax_tau.plot(t, tau[:, 0], 'm-', label='Moteur 1', alpha=0.8)
        ax_tau.plot(t, tau[:, 1], 'y-', label='Moteur 2', alpha=0.8)
        
        # Ajout d'un fond coloré pour montrer quand le laser est actif
        ax_tau.fill_between(t, ax_tau.get_ylim()[0], ax_tau.get_ylim()[1], where=(is_cutting==1.0), color='red', alpha=0.1, label='Laser Actif')
        
        ax_tau.set_title("Couples envoyés ($a_t$)", fontweight='bold')
        ax_tau.set_ylabel("Newton-mètres")
        ax_tau.legend(loc='best', fontsize='small')
        ax_tau.grid(True, linestyle=':', alpha=0.6)

        fig.canvas.draw_idle()

    # Initialisation au premier épisode
    update(1)
    
    # Association de l'événement du slider à la fonction de mise à jour
    episode_slider.on_changed(update)

    plt.show()

if __name__ == "__main__":
    main()
