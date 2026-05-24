# config.py — Convoyeur

BELT_SPEED_NOMINAL = 0.8    # m/s
BELT_ACCEL         = 2.5    # m/s²  — > μs·g ≈ 1.96 → glissement en accélération
BELT_DECEL         = 2.0    # m/s²  — > μs·g → glissement en freinage

MU_STATIC  = 0.20
MU_KINETIC = 0.15
G          = 9.81

STATION_DECOUPE = 4.0       # m
STATION_PERCAGE = 8.0       # m
PIECE_DISAPPEAR = 12.5      # m

# Distance de freinage : quand une pièce dépasse (station − BRAKE_DISTANCE),
# le tapis s'arrête. Avec friction (μk·g = 1.47 m/s²) la pièce glisse encore
# ~0.22 m après l'arrêt du tapis et s'immobilise au poste.
BRAKE_DISTANCE          = 0.22   # m
BRAKE_TRIGGER_NOISE_STD = 0.004  # m  (σ = 4 mm → erreur placement ±4 mm)

# Zone de sécurité : tapis s'arrête à SAFETY_DIST avant un poste occupé.
# Séparation effective après glissement = SAFETY_DIST − 0.22 m.
# Condition sans chevauchement : SAFETY_DIST − 0.22 > 0.9 m  →  > 1.12 m.
SAFETY_DIST = 1.4           # m   séparation effective 1.18 m > 0.9 m ✓

DT            = 0.01        # s  (100 Hz)
WIDTH, HEIGHT = 1400, 720
