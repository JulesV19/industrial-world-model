import pygame
import numpy as np
import math
import sys
import random
from collections import deque

# --- PARAMÈTRES PHYSIQUES IDÉAUX ---
m1, m2 = 2.0, 1.5
l1, l2 = 1.0, 1.0
r1, r2 = 0.5, 0.5
I1, I2 = 0.1, 0.08
g = 9.81

class PhysicsArmEnv:
    def __init__(self):
        self.state = np.array([math.pi/4, -math.pi/2, 0.0, 0.0])
        self.dt = 0.01
        
    def get_matrices(self, q, dq):
        M11 = m1*r1**2 + m2*(l1**2 + r2**2 + 2*l1*r2*np.cos(q[1])) + I1 + I2
        M12 = m2*(r2**2 + l1*r2*np.cos(q[1])) + I2
        M = np.array([[M11, M12], [M12, m2*r2**2 + I2]])
        
        h = -m2 * l1 * r2 * np.sin(q[1])
        V = np.array([h * dq[1]**2 + 2 * h * dq[0] * dq[1], -h * dq[0]**2])
        
        G1 = (m1*r1 + m2*l1) * g * np.cos(q[0]) + m2*r2 * g * np.cos(q[0] + q[1])
        G2 = m2*r2 * g * np.cos(q[0] + q[1])
        G = np.array([G1, G2])
        return M, V, G

    def apply_physics_imperfections(self, tau, dq):
        # 1. Frottements visqueux (proportionnels à la vitesse)
        f_visc = np.array([2.5, 1.5]) * dq
        
        # 2. Frottements de Coulomb (constants avec le signe de la vitesse)
        f_coulomb = np.array([5.0, 3.0]) * np.tanh(dq * 10.0) 
        
        # 3. Effet Stribeck (pic d'adhérence au démarrage, chute quand on prend de la vitesse)
        f_stribeck = np.array([8.0, 5.0]) * np.exp(-np.abs(dq) * 15.0) * np.tanh(dq * 100.0)
        
        # 4. Torque ripple (défaut électromagnétique des moteurs)
        ripple = tau * 0.05 * np.sin(50 * self.state[0:2]) 
        
        # 5. BRUIT ACTUATEUR (Non-déterministe, Process Noise)
        # Erreur de gain (le moteur ne donne pas exactement ce qu'on demande)
        gain_error = np.random.normal(1.0, 0.03, 2) # +-3% d'erreur de gain
        # Bruit blanc additif (imperfections de l'électronique de puissance)
        motor_noise = np.random.normal(0, 2.0, 2)
        
        # 6. Zone morte (Deadband - jeu mécanique où les petites commandes n'ont pas d'effet)
        deadband = 1.5
        tau_effective = np.where(np.abs(tau) > deadband, tau - np.sign(tau)*deadband, 0.0)

        # Calcul du couple final appliqué au système
        tau_real = tau_effective * gain_error + motor_noise - f_visc - f_coulomb - f_stribeck + ripple
        return tau_real

    def step(self, tau):
        q, dq = self.state[0:2], self.state[2:4]
        tau_real = self.apply_physics_imperfections(tau, dq)
        M, V, G = self.get_matrices(q, dq)
        ddq = np.linalg.inv(M) @ (tau_real - V - G)
        dq_new = dq + ddq * self.dt
        q_new = q + dq_new * self.dt
        self.state = np.concatenate([q_new, dq_new])
        return self.state

    def read_sensors(self):
        q_real, dq_real = self.state[0:2], self.state[2:4]
        encoder_res = (2 * math.pi) / 8192
        
        # Bruit de mesure continu
        q_noisy = q_real + np.random.normal(0, 0.002, 2)
        dq_noisy = dq_real + np.random.normal(0, 0.08, 2)
        
        # Quantification liée à la résolution de l'encodeur optique
        q_sensed = np.round(q_noisy / encoder_res) * encoder_res
        
        # GLITCH CAPTEUR (Non-déterministe, Observation Noise ponctuel)
        # Pour forcer le World Model à apprendre à ignorer les aberrations (outliers)
        if np.random.random() < 0.01:  # 1% de chance d'avoir un glitch sur la trame capteur
            q_sensed += np.random.normal(0, 0.05, 2)
            dq_noisy += np.random.normal(0, 2.0, 2)

        return q_sensed, dq_noisy

# --- CINÉMATIQUE ET PLANIFICATION ---
def inverse_kinematics(x, y):
    D = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    D = np.clip(D, -1.0, 1.0)
    q2 = -math.acos(D) 
    q1 = math.atan2(y, x) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    return np.array([q1, q2])

class TrajectoryPlanner:
    def __init__(self, waypoints, duration_per_segment):
        self.points = [(wp[0], wp[1]) for wp in waypoints]
        self.laser_states = [wp[2] for wp in waypoints] 
        self.T = duration_per_segment
        self.reset()

    def reset(self):
        self.current_segment = 0
        self.t = 0.0
        self.done = False
        self.is_cutting = False
        self.last_q = inverse_kinematics(self.points[0][0], self.points[0][1])
        self.last_dq = np.zeros(2)

    def get_desired_state(self, dt):
        if self.done: return self.last_q, np.zeros(2), np.zeros(2)
        self.t += dt
        if self.t >= self.T:
            self.t = 0.0
            self.current_segment += 1
            if self.current_segment >= len(self.points) - 1:
                self.done = True
                self.is_cutting = False
                return self.last_q, np.zeros(2), np.zeros(2)

        self.is_cutting = self.laser_states[self.current_segment + 1]
        tau = self.t / self.T
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        
        P0 = np.array(self.points[self.current_segment])
        P1 = np.array(self.points[self.current_segment + 1])
        X_des = P0 + (P1 - P0) * s
        q_des = inverse_kinematics(X_des[0], X_des[1])
        
        dq_des = (q_des - self.last_q) / dt
        ddq_des = (dq_des - self.last_dq) / dt
        self.last_q, self.last_dq = q_des, dq_des
        return q_des, dq_des, ddq_des

class RobotController:
    def __init__(self):
        self.filtered_dq = np.zeros(2)
        self.alpha = 0.15 

    def compute_torque(self, q_des, dq_des, ddq_des, q_sensed, dq_sensed):
        self.filtered_dq = self.alpha * dq_sensed + (1 - self.alpha) * self.filtered_dq
        M11 = m1*r1**2 + m2*(l1**2 + r2**2 + 2*l1*r2*np.cos(q_sensed[1])) + I1 + I2
        M12 = m2*(r2**2 + l1*r2*np.cos(q_sensed[1])) + I2
        M = np.array([[M11, M12], [M12, m2*r2**2 + I2]])
        h = -m2 * l1 * r2 * np.sin(q_sensed[1])
        V = np.array([h * self.filtered_dq[1]**2 + 2 * h * self.filtered_dq[0] * self.filtered_dq[1], -h * self.filtered_dq[0]**2])
        G1 = (m1*r1 + m2*l1) * g * np.cos(q_sensed[0]) + m2*r2 * g * np.cos(q_sensed[0] + q_sensed[1])
        G2 = m2*r2 * g * np.cos(q_sensed[0] + q_sensed[1])
        G = np.array([G1, G2])
        
        Kp = np.array([[1200.0, 0.0], [0.0, 1200.0]])
        Kd = np.array([[120.0, 0.0], [0.0, 120.0]])  
        e = q_des - q_sensed
        de = dq_des - self.filtered_dq
        
        tau = M @ (ddq_des + Kd @ de + Kp @ e) + V + G
        return np.clip(tau, -400, 400)

# --- GÉNÉRATION PROCÉDURALE DE LA BASE DE DONNÉES (100 Formes Agrandies) ---
def generate_shape_database(num_shapes=100):
    db = []
    base_x, base_y = 1.0, -0.1 # Centre de l'espace de travail
    safe_z = 0.6 # Position de repli (hauteur)

    for _ in range(num_shapes):
        shape_type = random.choice(["L_SHAPE", "RECTANGLE", "TRIANGLE", "HEXAGON"])
        # Facteur d'agrandissement (les pièces seront 50% à 100% plus grandes)
        scale = random.uniform(0.4, 0.7) 
        
        path = [(0.4, safe_z, False), (base_x, safe_z, False)] # Approche
        
        if shape_type == "RECTANGLE":
            w, h = scale, scale * random.uniform(0.6, 1.2)
            path.extend([
                (base_x - w/2, base_y - h/2, False), # Descente
                (base_x + w/2, base_y - h/2, True),  # Découpe...
                (base_x + w/2, base_y + h/2, True),
                (base_x - w/2, base_y + h/2, True),
                (base_x - w/2, base_y - h/2, True)   # Fermeture
            ])
            
        elif shape_type == "L_SHAPE":
            th = scale * 0.3 # Épaisseur des branches
            path.extend([
                (base_x - scale/2, base_y - scale/2, False),
                (base_x + scale/2, base_y - scale/2, True),
                (base_x + scale/2, base_y - scale/2 + th, True),
                (base_x - scale/2 + th, base_y - scale/2 + th, True),
                (base_x - scale/2 + th, base_y + scale/2, True),
                (base_x - scale/2, base_y + scale/2, True),
                (base_x - scale/2, base_y - scale/2, True)
            ])
            
        elif shape_type == "TRIANGLE":
            path.extend([
                (base_x - scale/2, base_y - scale/2, False),
                (base_x + scale/2, base_y - scale/2, True),
                (base_x, base_y + scale/2, True),
                (base_x - scale/2, base_y - scale/2, True)
            ])
            
        elif shape_type == "HEXAGON":
            pts = []
            for i in range(7): # 6 côtés + fermeture
                angle = i * (math.pi / 3)
                pts.append((base_x + scale/2 * math.cos(angle), base_y + scale/2 * math.sin(angle)))
            path.append((pts[0][0], pts[0][1], False))
            for pt in pts[1:]:
                path.append((pt[0], pt[1], True))
                
        path.append((base_x, safe_z, False)) # Dégagement
        path.append((0.4, safe_z, False))    # Retour base
        db.append(path)
    return db

# --- SETUP PYGAME ---
def draw_graph(surface, font, x, y, w, h, data_list, colors, labels, title, min_val, max_val):
    pygame.draw.rect(surface, (20, 20, 25), (x, y, w, h))
    pygame.draw.rect(surface, (100, 100, 100), (x, y, w, h), 1)
    surface.blit(font.render(title, True, (255, 255, 255)), (x + 10, y + 5))
    if min_val < 0 < max_val:
        zero_y = y + h - ((0 - min_val) / (max_val - min_val)) * h
        pygame.draw.line(surface, (50, 50, 50), (x, zero_y), (x + w, zero_y))

    for i, data in enumerate(data_list):
        if len(data) > 1:
            pts = [(x + (j / data.maxlen) * w, y + h - np.clip((val - min_val) / (max_val - min_val), 0, 1) * h) for j, val in enumerate(data)]
            pygame.draw.lines(surface, colors[i], False, pts, 2)
            surface.blit(font.render(labels[i], True, colors[i]), (x + 10, y + 25 + i * 15))

pygame.init()
WIDTH, HEIGHT = 1400, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Jumeau Numérique - Production de 100 Pièces")
clock = pygame.time.Clock()
font = pygame.font.SysFont("monospace", 12)

env = PhysicsArmEnv()
controller = RobotController()
scale_px = 220 # Légèrement zoomé par rapport à avant
origin = (350, 600)

# Initialisation de la base de données et du planificateur
database_100_shapes = generate_shape_database(100)
current_piece_index = 0
planner = TrajectoryPlanner(database_100_shapes[current_piece_index], duration_per_segment=0.8)

factory_state = "ARRIVING"
conveyor_offset = -800.0
completed_cuts, current_cut = [], []

HISTORY_LEN = 300
hist_q1, hist_q1_sensed, hist_q2 = deque(maxlen=HISTORY_LEN), deque(maxlen=HISTORY_LEN), deque(maxlen=HISTORY_LEN)
hist_dq1, hist_dq2 = deque(maxlen=HISTORY_LEN), deque(maxlen=HISTORY_LEN)
hist_tau1, hist_tau2 = deque(maxlen=HISTORY_LEN), deque(maxlen=HISTORY_LEN)

running = True
while running:
    # 1. Logique Usine
    if factory_state == "CUTTING":
        q_des, dq_des, ddq_des = planner.get_desired_state(env.dt)
        if planner.done: factory_state = "EVACUATING"
    
    elif factory_state == "EVACUATING":
        q_des, dq_des, ddq_des = planner.get_desired_state(env.dt) 
        conveyor_offset += 500 * env.dt # Vitesse d'évacuation augmentée
        if conveyor_offset > 900:
            conveyor_offset = -800
            completed_cuts.clear(); current_cut.clear()
            # On passe à la forme suivante dans la DB
            current_piece_index = (current_piece_index + 1) % len(database_100_shapes)
            planner = TrajectoryPlanner(database_100_shapes[current_piece_index], duration_per_segment=0.8)
            factory_state = "ARRIVING"
            
    elif factory_state == "ARRIVING":
        q_des, dq_des, ddq_des = planner.last_q, np.zeros(2), np.zeros(2)
        conveyor_offset += 500 * env.dt
        if conveyor_offset >= 0:
            conveyor_offset, factory_state = 0, "CUTTING"

    # 2. Lecture et Commande
    q_sensed, dq_sensed = env.read_sensors()
    tau = controller.compute_torque(q_des, dq_des, ddq_des, q_sensed, dq_sensed)
    state = env.step(tau)
    
    # Historique Télémétrie
    hist_q1.append(math.degrees(state[0]))
    hist_q1_sensed.append(math.degrees(q_sensed[0]))
    hist_q2.append(math.degrees(state[1]))
    hist_dq1.append(state[2])
    hist_dq2.append(state[3])
    hist_tau1.append(tau[0])
    hist_tau2.append(tau[1])

    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False

    # --- RENDU VISUEL ---
    screen.fill((30, 30, 40))
    q1, q2 = state[0], state[1]
    x_end = origin[0] + scale_px * (l1 * math.cos(q1) + l2 * math.cos(q1 + q2))
    y_end = origin[1] - scale_px * (l1 * math.sin(q1) + l2 * math.sin(q1 + q2))
    
    if factory_state == "CUTTING" and planner.is_cutting and not planner.done:
        if len(current_cut) == 0: current_cut.append((x_end, y_end))
        current_cut.append((x_end, y_end))
    elif factory_state == "CUTTING" and not planner.is_cutting:
        if len(current_cut) > 1: completed_cuts.append(current_cut)
        current_cut = []

    pygame.draw.rect(screen, (80, 80, 80), (0, origin[1] + 30, 900, 40))
    # Tôle agrandie pour accueillir les grandes formes
    tôle_rect = (origin[0] + 0.4*scale_px + conveyor_offset, origin[1] - 0.6*scale_px, 1.2*scale_px, 1.0*scale_px)
    pygame.draw.rect(screen, (100, 100, 110), tôle_rect)
    
    for cut in completed_cuts:
        pygame.draw.lines(screen, (30, 30, 35), False, [(px + conveyor_offset, py) for px, py in cut], 4)
    if len(current_cut) > 1:
        pygame.draw.lines(screen, (255, 100, 50), False, [(px + conveyor_offset, py) for px, py in current_cut], 4)
        pygame.draw.line(screen, (255, 255, 255), (x_end, y_end-20), (x_end, y_end), 2)

    x1 = origin[0] + scale_px * l1 * math.cos(q1)
    y1 = origin[1] - scale_px * l1 * math.sin(q1)
    pygame.draw.line(screen, (200, 150, 50), origin, (x1, y1), 16)
    pygame.draw.line(screen, (200, 150, 50), (x1, y1), (x_end, y_end), 12)
    pygame.draw.circle(screen, (50, 50, 50), origin, 14)
    pygame.draw.circle(screen, (50, 50, 50), (int(x1), int(y1)), 10)
    pygame.draw.circle(screen, (255, 255, 255) if planner.is_cutting else (100,100,100), (int(x_end), int(y_end)), 6)

    # --- TÉLÉMÉTRIE ---
    gx, gw, gh = 900, 480, 240
    
    draw_graph(screen, font, gx, 20, gw, gh, 
               [hist_q1, hist_q1_sensed], [(255, 100, 100), (100, 255, 100)], 
               ["Théta 1 Réel", "Théta 1 Capteur"], "ANGLES & BRUIT CAPTEUR", -180, 180)
               
    draw_graph(screen, font, gx, 280, gw, gh, 
               [hist_dq1, hist_dq2], [(255, 150, 50), (50, 255, 150)], 
               ["Omega 1", "Omega 2"], "VITESSES ANGULAIRES", -5, 5)
               
    draw_graph(screen, font, gx, 540, gw, gh, 
               [hist_tau1, hist_tau2], [(255, 50, 255), (255, 255, 50)], 
               ["Couple M1", "Couple M2"], "COMMANDES MOTEURS (Bruyantes)", -350, 350)

    # UI : Affichage du numéro de pièce
    screen.blit(font.render(f"PIÈCE : {current_piece_index + 1} / 100", True, (255, 255, 255)), (20, 45))
    status_color = (100, 255, 100) if factory_state == "CUTTING" else (255, 200, 50)
    screen.blit(font.render(f"ÉTAT USINE : {factory_state}", True, status_color), (20, 20))

    pygame.display.flip()
    clock.tick(100)

pygame.quit()
sys.exit()