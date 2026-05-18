# main.py
import pygame
import numpy as np
import math
import sys
from collections import deque

from config import WIDTH, HEIGHT, scale_px, origin, HISTORY_LEN, l1, l2
from arm import PhysicsArmEnv, RobotController, TrajectoryPlanner
from piece import load_shape_database

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

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Jumeau Numérique - Production de 100 Pièces")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 12)

    current_piece_index = 0
    planner = TrajectoryPlanner(database_100_shapes[current_piece_index], duration_per_segment=0.8)

    env = PhysicsArmEnv(initial_q=planner.last_q)
    controller = RobotController()

    factory_state = "ARRIVING"
    conveyor_offset = -800.0
    completed_cuts, current_cut = [], []

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

if __name__ == "__main__":
    main()
