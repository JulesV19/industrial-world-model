"""
Visualisation pygame du convoyeur.

    python3 main.py

Commandes :
  ESPACE   – ajouter une pièce manuellement
  ESC / Q  – quitter
"""

import sys
import math
import numpy as np
import pygame
from collections import deque

from conveyor import ConveyorBelt, PieceState
from config import (
    STATION_DECOUPE, STATION_PERCAGE, PIECE_DISAPPEAR,
    BELT_SPEED_NOMINAL, SAFETY_DIST, DT, WIDTH, HEIGHT,
)

# ---------------------------------------------------------------------------
#  Constantes d'affichage
# ---------------------------------------------------------------------------

BELT_Y   = 420          # centre vertical du tapis
BELT_H   = 55           # épaisseur tapis (px)
BX0      = 60           # x pixel correspondant à x_m = 0
BX1      = 1000         # x pixel correspondant à x_m = PIECE_DISAPPEAR
BLEN     = BX1 - BX0   # longueur en pixels
M2PX     = BLEN / PIECE_DISAPPEAR   # facteur de conversion m → px

PIECE_PX_W = int(0.90 * M2PX)   # largeur pièce (~90 cm)
PIECE_PX_H = 42                  # hauteur pièce (px)

PANEL_X  = 1020          # x de début du panneau télémétrie

# Couleurs par état
STATE_COLOR = {
    PieceState.ENTERING:        (80,  160, 255),
    PieceState.AT_DECOUPE:      (255, 200,  30),
    PieceState.LEAVING_DECOUPE: (80,  220, 120),
    PieceState.AT_PERCAGE:      (255, 110,  40),
    PieceState.LEAVING_PERCAGE: (180, 110, 255),
    PieceState.EXITED:          (60,   60,  60),
}

# Timings de traitement (simulés automatiquement)
DECOUPE_PROCESS_TIME = 9.0   # s
PERCAGE_PROCESS_TIME = 6.0   # s
PIECE_SPAWN_INTERVAL = 4.0   # s  entre deux nouvelles pièces


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def m2px(x: float) -> int:
    return int(BX0 + x * M2PX)


def draw_graph(surf, font, x, y, w, h, data: deque, color, title: str,
               y_min: float, y_max: float, unit: str = "") -> None:
    pygame.draw.rect(surf, (22, 22, 30), (x, y, w, h))
    pygame.draw.rect(surf, (80, 80, 80), (x, y, w, h), 1)
    surf.blit(font.render(title, True, (200, 200, 200)), (x + 6, y + 4))
    if len(data) > 1:
        span = y_max - y_min or 1
        zero_y = y + h - int((0 - y_min) / span * h)
        if y_min < 0 < y_max:
            pygame.draw.line(surf, (60, 60, 60), (x, zero_y), (x + w, zero_y))
        pts = [
            (x + int(i / data.maxlen * w),
             y + h - int(np.clip((v - y_min) / span, 0, 1) * h))
            for i, v in enumerate(data)
        ]
        pygame.draw.lines(surf, color, False, pts, 2)
    if data:
        last = list(data)[-1]
        surf.blit(font.render(f"{last:+.2f}{unit}", True, color),
                  (x + w - 80, y + 4))


def draw_machine(surf, sx: int, label: str, color, processing: bool,
                 progress: float, font) -> None:
    """Dessine une machine simplifiée au-dessus du tapis."""
    arm_h = 120
    base_y = BELT_Y - BELT_H // 2
    # Colonne
    pygame.draw.rect(surf, (70, 70, 70), (sx - 10, base_y - arm_h, 20, arm_h))
    # Tête outil
    head_col = color if processing else (80, 80, 80)
    pygame.draw.rect(surf, head_col, (sx - 24, base_y - arm_h - 20, 48, 28))
    pygame.draw.rect(surf, (255, 255, 255),
                     (sx - 24, base_y - arm_h - 20, 48, 28), 1)
    # Label
    surf.blit(font.render(label, True, color),
              (sx - 28, base_y - arm_h - 46))
    # Barre de progression
    if processing:
        bw = 80
        bh = 8
        bx = sx - bw // 2
        by = base_y - arm_h - 62
        pygame.draw.rect(surf, (40, 40, 40), (bx, by, bw, bh))
        pygame.draw.rect(surf, color, (bx, by, int(bw * progress), bh))
        pygame.draw.rect(surf, (120, 120, 120), (bx, by, bw, bh), 1)


def draw_belt_texture(surf, zones, t: float) -> None:
    """Lignes animées simulant le mouvement du tapis."""
    zone_bounds = [
        (0,              STATION_DECOUPE),
        (STATION_DECOUPE, STATION_PERCAGE),
        (STATION_PERCAGE, PIECE_DISAPPEAR),
    ]
    for i, (x_start, x_end) in enumerate(zone_bounds):
        px0 = m2px(x_start)
        px1 = m2px(x_end)
        col = ZONE_COLORS[i]
        pygame.draw.rect(surf, col,
                         (px0, BELT_Y - BELT_H // 2, px1 - px0, BELT_H))
        # Lignes animées
        v = zones[i].vel
        spacing = 50
        offset = int((t * v * M2PX) % spacing)
        x = px0 + offset - spacing
        while x < px1:
            if x > px0:
                pygame.draw.line(surf, (c + 20 for c in col),
                                 (x, BELT_Y - BELT_H // 2),
                                 (x, BELT_Y + BELT_H // 2), 1)
                pygame.draw.line(surf,
                                 tuple(min(255, c + 20) for c in col),
                                 (x, BELT_Y - BELT_H // 2),
                                 (x, BELT_Y + BELT_H // 2), 1)
            x += spacing


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Convoyeur — Simulation Physique")
    clock = pygame.time.Clock()
    font     = pygame.font.SysFont("monospace", 12)
    font_b   = pygame.font.SysFont("monospace", 13, bold=True)

    conv = ConveyorBelt()
    conv.add_piece()

    # Signaux de machines (niveau haut tant que pièce en cours ou traitée)
    decoupe_done = False
    percage_done  = False

    # Minuteries de traitement
    decoupe_timer = 0.0
    percage_timer  = 0.0

    # Spawn automatique
    spawn_timer = 0.0

    # Télémétrie
    hist_errors: deque = deque(maxlen=60)   # mm
    hist_d_vel:  deque = deque(maxlen=300)  # m/s tapis

    sim_time = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                if event.key == pygame.K_SPACE:
                    if conv._in_state(PieceState.ENTERING) is None:
                        conv.add_piece()

        # -------------------------------------------------------------------
        #  Génération automatique des signaux « machine terminée »
        # -------------------------------------------------------------------
        p_d = conv._in_state(PieceState.AT_DECOUPE)
        p_p = conv._in_state(PieceState.AT_PERCAGE)

        if p_d is not None:
            decoupe_timer += DT
            if decoupe_timer >= DECOUPE_PROCESS_TIME:
                decoupe_done  = True
                decoupe_timer = 0.0
        else:
            decoupe_timer = 0.0
            decoupe_done  = False

        if p_p is not None:
            percage_timer += DT
            if percage_timer >= PERCAGE_PROCESS_TIME:
                percage_done  = True
                percage_timer = 0.0
        else:
            percage_timer = 0.0
            percage_done  = False

        # -------------------------------------------------------------------
        #  Spawn automatique de nouvelles pièces
        # -------------------------------------------------------------------
        spawn_timer += DT
        if spawn_timer >= PIECE_SPAWN_INTERVAL:
            if conv._in_state(PieceState.ENTERING) is None:
                conv.add_piece()
            spawn_timer = 0.0

        # -------------------------------------------------------------------
        #  Étape simulation
        # -------------------------------------------------------------------
        err_before = conv.get_percage_placement_error()
        conv.step(DT, decoupe_done=decoupe_done, percage_done=percage_done)
        err_after  = conv.get_percage_placement_error()

        # Enregistre l'erreur quand une nouvelle pièce vient de s'arrêter
        if err_after is not None and err_after != err_before:
            hist_errors.append(err_after * 1000)   # → mm

        hist_d_vel.append(conv.belt.vel)
        sim_time += DT

        # -------------------------------------------------------------------
        #  Rendu
        # -------------------------------------------------------------------
        screen.fill((18, 18, 26))

        # Tapis unique
        belt_col = (50, 90, 130) if conv.belt.vel > 0.05 else (55, 55, 65)
        pygame.draw.rect(screen, belt_col,
                         (BX0, BELT_Y - BELT_H // 2, BLEN, BELT_H))

        # Texture animée (lignes suivant la vitesse du tapis)
        v_belt = conv.belt.vel
        if v_belt > 0.01:
            spacing = 55
            offset = int((sim_time * v_belt * M2PX) % spacing)
            lx = BX0 + offset - spacing
            while lx < BX1:
                if lx > BX0:
                    lc = tuple(min(255, c + 28) for c in belt_col)
                    pygame.draw.line(screen, lc,
                                     (lx, BELT_Y - BELT_H // 2),
                                     (lx, BELT_Y + BELT_H // 2), 1)
                lx += spacing

        # Bords du tapis
        pygame.draw.rect(screen, (130, 130, 130),
                         (BX0, BELT_Y - BELT_H // 2, BLEN, BELT_H), 1)

        # Marqueurs de postes sur le tapis (pointillés verticaux)
        for xs in (STATION_DECOUPE, STATION_PERCAGE):
            lx = m2px(xs)
            for dy in range(BELT_Y - BELT_H // 2, BELT_Y + BELT_H // 2, 8):
                pygame.draw.line(screen, (140, 140, 140),
                                 (lx, dy), (lx, min(dy + 4, BELT_Y + BELT_H // 2)))

        # Zones de sécurité anti-collision (hachures devant chaque poste occupé)
        at_d_vis = conv._in_state(PieceState.AT_DECOUPE) is not None
        at_p_vis = conv._in_state(PieceState.AT_PERCAGE)  is not None
        for occupied, station in ((at_d_vis, STATION_DECOUPE),
                                   (at_p_vis, STATION_PERCAGE)):
            if occupied:
                sx0 = m2px(station - SAFETY_DIST)
                sx1 = m2px(station)
                for hx in range(sx0, sx1, 7):
                    pygame.draw.line(screen, (180, 60, 60),
                                     (hx, BELT_Y - BELT_H // 2 + 2),
                                     (hx + 4, BELT_Y + BELT_H // 2 - 2), 1)

        # Vitesse du tapis (unique)
        belt_col_lbl = (80, 200, 80) if v_belt > 0.05 else (180, 80, 80)
        lbl_v = font.render(
            f"TAPIS  {v_belt:.3f} m/s  {'▶ EN MARCHE' if v_belt > 0.05 else '■ ARRÊTÉ'}",
            True, belt_col_lbl)
        screen.blit(lbl_v, (BX0 + BLEN // 2 - lbl_v.get_width() // 2,
                             BELT_Y + BELT_H // 2 + 6))

        # Marqueurs de poste
        for label, xs, col in (
            ("DÉCOUPE", STATION_DECOUPE, (255, 210, 40)),
            ("PERÇAGE", STATION_PERCAGE, (255, 120, 50)),
        ):
            lx = m2px(xs)
            pygame.draw.line(screen, col,
                             (lx, BELT_Y - BELT_H // 2 - 130),
                             (lx, BELT_Y + BELT_H // 2 + 30), 2)

        # Machines
        p_d_proc = p_d is not None
        p_p_proc = p_p is not None
        draw_machine(screen, m2px(STATION_DECOUPE), "DÉCOUPE",
                     (255, 210, 40), p_d_proc,
                     min(decoupe_timer / DECOUPE_PROCESS_TIME, 1.0), font_b)
        draw_machine(screen, m2px(STATION_PERCAGE), "PERÇAGE",
                     (255, 120, 50), p_p_proc,
                     min(percage_timer / PERCAGE_PROCESS_TIME, 1.0), font_b)

        # Marqueurs entrée / sortie
        pygame.draw.line(screen, (100, 200, 255),
                         (BX0, BELT_Y - 50), (BX0, BELT_Y + 50), 2)
        screen.blit(font.render("ENTRÉE", True, (100, 200, 255)),
                    (BX0 - 8, BELT_Y - 65))
        pygame.draw.line(screen, (220, 60, 60),
                         (BX1, BELT_Y - 50), (BX1, BELT_Y + 50), 2)
        screen.blit(font.render("SORTIE", True, (220, 60, 60)),
                    (BX1 - 18, BELT_Y - 65))

        # -------------------------------------------------------------------
        #  Pièces
        # -------------------------------------------------------------------
        for piece in conv.pieces:
            if piece.state == PieceState.EXITED:
                continue
            px = m2px(piece.x)
            col = STATE_COLOR[piece.state]
            rect = pygame.Rect(px - PIECE_PX_W // 2,
                               BELT_Y - PIECE_PX_H // 2,
                               PIECE_PX_W, PIECE_PX_H)
            pygame.draw.rect(screen, col, rect)
            pygame.draw.rect(screen, (230, 230, 230), rect, 1)

            # ID et vitesse
            lbl = font.render(
                f"#{piece.piece_id}  {piece.vel:.2f} m/s", True, (20, 20, 28))
            screen.blit(lbl, (px - lbl.get_width() // 2,
                               BELT_Y - PIECE_PX_H // 2 - 17))

            # Erreur de placement (au poste de perçage)
            if piece.state == PieceState.AT_PERCAGE:
                err_mm  = piece.placement_error * 1000
                err_col = (40, 220, 80) if abs(err_mm) < 8 else (230, 60, 60)
                et = font.render(f"Δ = {err_mm:+.1f} mm", True, err_col)
                screen.blit(et, (px - et.get_width() // 2,
                                  BELT_Y + PIECE_PX_H // 2 + 24))

        # -------------------------------------------------------------------
        #  Panneau télémétrie
        # -------------------------------------------------------------------
        tx, ty = PANEL_X, 20

        screen.blit(font_b.render("TÉLÉMÉTRIE CONVOYEUR", True, (220, 220, 220)),
                    (tx, ty));  ty += 24

        # Graphe erreurs de placement
        draw_graph(screen, font, tx, ty, 360, 130,
                   hist_errors, (100, 210, 255),
                   "Erreur placement perçage", -30, 30, " mm")
        ty += 140

        if hist_errors:
            errs = list(hist_errors)
            moy  = np.mean(errs)
            sig  = np.std(errs)
            mx   = max(abs(e) for e in errs)
            for lbl, val, uc in (
                ("Moyenne", moy,  " mm"),
                ("Écart-type", sig, " mm"),
                ("Max |err|",  mx,  " mm"),
            ):
                c = (40, 200, 80) if abs(val) < 8 else (220, 80, 60)
                screen.blit(font.render(f"{lbl:<14}{val:+.2f}{uc}", True, c),
                            (tx, ty));  ty += 16
        ty += 10

        # Graphe vitesse du tapis unique
        draw_graph(screen, font, tx, ty, 360, 80,
                   hist_d_vel, (80, 190, 255),
                   "Vitesse tapis (moteur unique)",
                   0, BELT_SPEED_NOMINAL + 0.1, " m/s")
        ty += 98

        # Liste des pièces actives
        screen.blit(font_b.render("PIÈCES ACTIVES", True, (200, 200, 200)),
                    (tx, ty));  ty += 18
        for piece in conv.pieces[-10:]:
            if piece.state == PieceState.EXITED:
                continue
            col = STATE_COLOR[piece.state]
            txt = (f"#{piece.piece_id:02d}  x={piece.x:5.2f} m  "
                   f"v={piece.vel:.2f} m/s  {piece.state.name}")
            screen.blit(font.render(txt, True, col), (tx, ty));  ty += 15

        # Légende
        ty = HEIGHT - 110
        screen.blit(font_b.render("LÉGENDE", True, (160, 160, 160)),
                    (tx, ty));  ty += 16
        for state, col in STATE_COLOR.items():
            if state == PieceState.EXITED:
                continue
            screen.blit(font.render(state.name, True, col), (tx, ty))
            ty += 14

        screen.blit(font.render("ESPACE : nouvelle pièce   ESC : quitter",
                                 True, (100, 100, 100)),
                    (BX0, HEIGHT - 20))

        pygame.display.flip()
        clock.tick(100)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
