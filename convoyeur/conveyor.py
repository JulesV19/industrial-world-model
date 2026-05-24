"""
Convoyeur à tapis unique — un moteur, friction pièce/tapis.

  ENTRÉE ──── DÉCOUPE (4 m) ──── PERÇAGE (8 m) ──── SORTIE (12.5 m)

Physique
--------
BELT_ACCEL = 2.5 m/s² > μs·g ≈ 1.96 m/s² : les pièces glissent lors des
accélérations et freinages. L'erreur de positionnement résulte de ce glissement.

Commande tapis
--------------
  • Le tapis tourne quand aucune pièce n'est en cours de traitement ET
    qu'aucune pièce n'a atteint son seuil de freinage.
  • Sécurité : si un poste est occupé, le tapis s'arrête à SAFETY_DIST du poste.
  • Le tapis ne redémarre que quand tous les postes sont libres.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from config import (
    BELT_SPEED_NOMINAL, BELT_ACCEL, BELT_DECEL,
    MU_STATIC, MU_KINETIC, G,
    STATION_DECOUPE, STATION_PERCAGE, PIECE_DISAPPEAR,
    BRAKE_DISTANCE, BRAKE_TRIGGER_NOISE_STD, SAFETY_DIST,
)


class PieceState(Enum):
    ENTERING         = auto()   # en transit vers découpe
    AT_DECOUPE       = auto()   # bridée au poste de découpe
    LEAVING_DECOUPE  = auto()   # en transit vers perçage
    AT_PERCAGE       = auto()   # bridée au poste de perçage
    LEAVING_PERCAGE  = auto()   # en transit vers la sortie
    EXITED           = auto()   # sortie du tapis


@dataclass
class Piece:
    piece_id: int
    x:   float = 0.0
    vel: float = 0.0
    state: PieceState = PieceState.ENTERING
    placement_error: float = 0.0
    # Position à partir de laquelle le tapis déclenche l'arrêt pour cette pièce.
    brake_x: float = 0.0


class Belt:
    def __init__(self) -> None:
        self.vel: float = 0.0
        self._target: float = 0.0

    def command(self, run: bool) -> None:
        self._target = BELT_SPEED_NOMINAL if run else 0.0

    def step(self, dt: float) -> None:
        if self.vel < self._target:
            self.vel = min(self.vel + BELT_ACCEL * dt, self._target)
        else:
            self.vel = max(self.vel - BELT_DECEL * dt, self._target)

    @property
    def accel(self) -> float:
        if self.vel < self._target: return  BELT_ACCEL
        if self.vel > self._target: return -BELT_DECEL
        return 0.0


class ConveyorBelt:

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.belt = Belt()
        self.pieces: list[Piece] = []
        self._next_id = 0
        self._rng = rng if rng is not None else np.random.default_rng()
        self._prev_decoupe_done = False
        self._prev_percage_done = False

    # ------------------------------------------------------------------
    #  API publique
    # ------------------------------------------------------------------

    def add_piece(self) -> Piece:
        p = Piece(
            piece_id=self._next_id,
            brake_x=STATION_DECOUPE - BRAKE_DISTANCE + self._noise(),
        )
        self._next_id += 1
        self.pieces.append(p)
        return p

    def get_percage_placement_error(self) -> Optional[float]:
        p = self._in_state(PieceState.AT_PERCAGE)
        return p.placement_error if p is not None else None

    def step(self, dt: float, decoupe_done: bool, percage_done: bool) -> None:

        # -- Fronts montants : libération des postes -----------------------
        d_rise = decoupe_done and not self._prev_decoupe_done
        p_rise  = percage_done  and not self._prev_percage_done
        self._prev_decoupe_done = decoupe_done
        self._prev_percage_done  = percage_done

        if d_rise:
            p = self._in_state(PieceState.AT_DECOUPE)
            if p is not None:
                p.state   = PieceState.LEAVING_DECOUPE
                p.vel     = 0.0
                p.brake_x = STATION_PERCAGE - BRAKE_DISTANCE + self._noise()

        if p_rise:
            p = self._in_state(PieceState.AT_PERCAGE)
            if p is not None:
                p.state = PieceState.LEAVING_PERCAGE
                p.vel   = 0.0

        # -- État des postes -----------------------------------------------
        at_d = self._in_state(PieceState.AT_DECOUPE) is not None
        at_p = self._in_state(PieceState.AT_PERCAGE)  is not None

        # -- Commande tapis ------------------------------------------------
        #  Arrêt si :
        #   (a) une pièce approche du poste cible (freinage)
        #   (b) une pièce est trop près d'un poste occupé (sécurité)
        #  Run seulement quand tous les postes sont libres + aucun freinage.
        should_stop = False
        advancing   = False

        for p in self.pieces:
            if p.state in (PieceState.AT_DECOUPE, PieceState.AT_PERCAGE, PieceState.EXITED):
                continue
            advancing = True

            if p.state == PieceState.ENTERING:
                if at_d and p.x >= STATION_DECOUPE - SAFETY_DIST:
                    should_stop = True                          # sécurité
                elif not at_d and p.x >= p.brake_x:
                    should_stop = True                          # freinage
            elif p.state == PieceState.LEAVING_DECOUPE:
                if at_p and p.x >= STATION_PERCAGE - SAFETY_DIST:
                    should_stop = True
                elif not at_p and p.x >= p.brake_x:
                    should_stop = True

        # Le tapis ne redémarre que si AUCUN poste n'est occupé.
        belt_run = advancing and not should_stop and not at_d and not at_p
        self.belt.command(belt_run)
        self.belt.step(dt)

        # -- Dynamique des pièces (friction tapis/pièce) -------------------
        for p in self.pieces:
            if p.state in (PieceState.AT_DECOUPE, PieceState.AT_PERCAGE, PieceState.EXITED):
                continue
            a    = self._friction_accel(p.vel, self.belt.vel, self.belt.accel)
            p.vel = max(0.0, p.vel + a * dt)
            p.x  += p.vel * dt

        # -- Bridage au poste (tapis ET pièce arrêtés) --------------------
        if self.belt.vel < 0.02:
            p = self._in_state(PieceState.ENTERING)
            if p is not None and p.vel < 0.02 and p.x > STATION_DECOUPE - 0.5:
                p.vel   = 0.0
                p.state = PieceState.AT_DECOUPE

            p = self._in_state(PieceState.LEAVING_DECOUPE)
            if p is not None and p.vel < 0.02 and p.x > STATION_PERCAGE - 0.5:
                p.vel             = 0.0
                p.placement_error = p.x - STATION_PERCAGE
                p.state           = PieceState.AT_PERCAGE

        # -- Sortie --------------------------------------------------------
        for p in self.pieces:
            if p.state == PieceState.LEAVING_PERCAGE and p.x >= PIECE_DISAPPEAR:
                p.state = PieceState.EXITED

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _in_state(self, state: PieceState) -> Optional[Piece]:
        for p in self.pieces:
            if p.state == state:
                return p
        return None

    def _friction_accel(self, piece_vel: float, belt_vel: float, belt_accel: float) -> float:
        """Accélération due au frottement tapis/pièce (m/s²)."""
        rel = belt_vel - piece_vel
        if abs(rel) < 1e-3:                             # synchronisés
            if abs(belt_accel) <= MU_STATIC * G:
                return belt_accel                       # friction statique suit
            return MU_KINETIC * G * np.sign(belt_accel) # seuil dépassé → glissement
        return MU_KINETIC * G * np.sign(rel)            # friction cinétique

    def _noise(self) -> float:
        return float(self._rng.normal(0.0, BRAKE_TRIGGER_NOISE_STD))
