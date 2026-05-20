"""
Simulation pygame du bras de perçage.

    python3 main.py
    python3 main.py --episode ../arm_decoupe/dataset/episode_001_run00.npz
    python3 main.py --episode ../arm_decoupe/dataset/episode_001_run00.npz --speed 1.5
"""

import argparse
import math
import sys

import numpy as np
import pygame
from collections import deque

from config import WIDTH, HEIGHT, scale_px, origin, HISTORY_LEN, l1, l2
from arm import PhysicsArmEnv, RobotController, TrajectoryPlanner
from piece_input import extract_corners

DECOUPE_DB = "../arm_decoupe/pieces_database.json"


def fk(q):
    x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
    y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    return np.array([x, y])


def to_screen(pos):
    return (int(origin[0] + pos[0] * scale_px),
            int(origin[1] - pos[1] * scale_px))


def draw_arm(surface, q):
    base = origin
    j1 = to_screen([l1 * math.cos(q[0]), l1 * math.sin(q[0])])
    ee = to_screen(fk(q))
    pygame.draw.line(surface, (180, 180, 180), base, j1, 5)
    pygame.draw.line(surface, (140, 140, 200), j1, ee, 4)
    pygame.draw.circle(surface, (255, 80, 80), ee, 6)


def draw_piece(surface, corners, font):
    # Contour de la pièce
    pts = [to_screen(c) for c in corners]
    pts_closed = pts + [pts[0]]
    pygame.draw.lines(surface, (60, 120, 60), False, pts_closed, 1)

    # Cibles (coins idéaux perçus)
    for i, c in enumerate(corners):
        sc = to_screen(c)
        pygame.draw.circle(surface, (255, 200, 0), sc, 7, 2)
        surface.blit(font.render(str(i+1), True, (255, 200, 0)), (sc[0]+8, sc[1]-8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=str, default=None,
                        help="Chemin vers un .npz de découpe")
    parser.add_argument("--speed", type=float, default=2.0)
    args = parser.parse_args()

    if args.episode:
        corners = extract_corners(args.episode, DECOUPE_DB)
    else:
        corners = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])

    wp = [[c[0], c[1], True] for c in corners]
    planner = TrajectoryPlanner(wp, duration_per_segment=args.speed)
    env = PhysicsArmEnv(initial_q=planner.last_q)
    controller = RobotController()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Bras de perçage")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    trail = deque(maxlen=HISTORY_LEN)
    drill_hits = [fk(env.state[:2]).copy()]  # coin 0 = position initiale
    prev_segment = planner.current_segment
    done_pause = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        screen.fill((15, 15, 20))
        draw_piece(screen, corners, font)

        if not planner.done:
            q_sensed, dq_sensed = env.read_sensors()
            q_des, dq_des, ddq_des = planner.get_desired_state(env.dt)
            tau = controller.compute_torque(q_des, dq_des, ddq_des, q_sensed, dq_sensed)
            env.step(tau)

            if planner.current_segment != prev_segment:
                drill_hits.append(fk(env.state[:2]).copy())
                prev_segment = planner.current_segment
        else:
            done_pause += 1
            if done_pause > 200:
                running = False

        ee = fk(env.state[:2])
        trail.append(to_screen(ee))

        if len(trail) > 1:
            pygame.draw.lines(screen, (60, 60, 160), False, list(trail), 2)

        draw_arm(screen, env.state[:2])

        # Trous percés (position réelle)
        for i, hit in enumerate(drill_hits):
            sh = to_screen(hit)
            err = np.linalg.norm(hit - corners[i]) * 1000
            color = (0, 220, 120) if err < 20 else (220, 60, 60)
            pygame.draw.circle(screen, color, sh, 5)

        # HUD
        n_done = len(drill_hits)
        hud = font.render(
            f"Coins percés : {n_done}/4   "
            + (f"err moy : {np.mean([np.linalg.norm(drill_hits[i]-corners[i])*1000 for i in range(n_done)]):.1f} mm"
               if n_done else ""),
            True, (200, 200, 200)
        )
        screen.blit(hud, (20, 20))

        pygame.display.flip()
        clock.tick(100)

    pygame.quit()


if __name__ == "__main__":
    main()
