import math
from config import (
    FRICTION_DEGRAD_ALPHA, FRICTION_DEGRAD_HALFLIFE,
    TEMP_NOISE_GAMMA, CADENCE_REF,
    T_AMBIENT, T_EQ_SLOPE, THERMAL_TAU, THERMAL_COOLDOWN,
)


def friction_multiplier(piece_count: int) -> float:
    """Saturation Hill : 1.0 à neuf → 1+alpha à saturation, moitié à piece_count=halflife."""
    return 1.0 + FRICTION_DEGRAD_ALPHA * piece_count / (piece_count + FRICTION_DEGRAD_HALFLIFE)


def noise_multiplier(cadence: float) -> float:
    """Conservé pour rétrocompatibilité. Non utilisé quand ThermalModel est actif."""
    return 1.0 + TEMP_NOISE_GAMMA * (cadence / CADENCE_REF)


class ThermalModel:
    """
    Modèle thermique déterministe d'un bras robotique industriel.

    La température monte exponentiellement vers un équilibre T_eq(cadence),
    puis redescend vers T_AMBIENT entre les sessions avec une constante de temps
    plus longue (inertie thermique de la mécanique).

    Toutes les transitions sont continues et entièrement déterministes :
    à cadence et piece_count identiques, la trajectoire de T est identique.

    Usage typique :
        thermal = ThermalModel(cadence=60.0)
        for piece in range(N):
            thermal.advance_piece(duration_s=30.0)   # chauffage pendant la pièce
            T = thermal.temperature
        thermal.cool_between_sessions(elapsed_s=3600.0)
    """

    def __init__(self, cadence: float, initial_temp: float = T_AMBIENT):
        """
        cadence      : pièces/heure — détermine la température d'équilibre
        initial_temp : température de départ (°C), T_AMBIENT si machine à froid
        """
        self.cadence     = cadence
        self.temperature = float(initial_temp)
        # Température d'équilibre déterministe : plus la cadence est haute, plus il fait chaud
        self.T_eq        = T_AMBIENT + T_EQ_SLOPE * cadence

    def step(self, dt: float) -> float:
        """
        Intègre le modèle thermique sur dt secondes (Euler explicite).
        Appeler à chaque step de simulation (~0.01 s) pour une montée lisse.

        dT/dt = (T_eq - T) / THERMAL_TAU
        """
        self.temperature += (self.T_eq - self.temperature) / THERMAL_TAU * dt
        return self.temperature

    def advance_piece(self, duration_s: float):
        """
        Avance le modèle thermique d'une durée correspondant à une pièce entière.
        Utilise la solution analytique exacte (pas d'erreur d'intégration Euler).

        T(t) = T_eq + (T0 - T_eq) * exp(-duration_s / THERMAL_TAU)
        """
        alpha = math.exp(-duration_s / THERMAL_TAU)
        self.temperature = self.T_eq + (self.temperature - self.T_eq) * alpha
        return self.temperature

    def cool_between_sessions(self, elapsed_s: float = THERMAL_COOLDOWN * 3):
        """
        Refroidissement entre deux sessions de production (solution analytique).
        elapsed_s : durée du refroidissement en secondes (défaut : 3x la constante de temps)
        """
        alpha = math.exp(-elapsed_s / THERMAL_COOLDOWN)
        self.temperature = T_AMBIENT + (self.temperature - T_AMBIENT) * alpha
        return self.temperature

    @property
    def delta_T(self) -> float:
        """Écart à la température ambiante ΔT = T - T_AMBIENT (toujours >= 0)."""
        return max(0.0, self.temperature - T_AMBIENT)
