# config.py

# --- PARAMETRES PHYSIQUES IDEAUX ---
m1, m2 = 2.0, 1.5
import math as _math
l1, l2 = _math.sqrt(2), _math.sqrt(2)
r1, r2 = 0.5, 0.5
I1, I2 = 0.1, 0.08
g = 9.81

# --- CONSTANTES UI ET ENVIRONNEMENT ---
WIDTH, HEIGHT = 1400, 800
scale_px = 220
origin = (350, 600)
HISTORY_LEN = 300

# --- IMPERFECTIONS PHYSIQUES ET BRUIT ---
# Frottements (valeurs a froid, T = T_AMBIENT)
visc_friction_coeffs = [2.5*2, 1.5*2]
coulomb_friction_coeffs = [5.0/2, 3.0/2]
stribeck_friction_coeffs = [8.0/2, 5.0/2]

# Moteurs et Actuateurs
ripple_amplitude = 0.05
gain_error_std = 0.001     # Reduit x10 : l'effet temperature est dominant
motor_noise_std = 0.1      # Reduit x10 : bruit thermique modelise via T
deadband_torque = 0.8      # Zone morte de base (a T_AMBIENT)

# Capteurs
encoder_resolution = 8192
sensor_noise_q_std = 0.0002    # Reduit x10 : la degradation thermique est dominante
sensor_noise_dq_std = 0.0008   # Reduit x10
sensor_glitch_prob = 0.0       # Supprime : remplace par degradation continue liee a T
sensor_glitch_q_std = 0.0
sensor_glitch_dq_std = 0.0
vel_noise_scale_k = 0.4

# --- CONTROLEUR ---
Kp_gain = 1200.0
Kd_gain = 120.0
alpha_filter = 0.15

# --- DEGRADATION TEMPORELLE (usure mecanique via piece_count) ---
FRICTION_DEGRAD_ALPHA    = 1
FRICTION_DEGRAD_HALFLIFE = 300
TEMP_NOISE_GAMMA         = 0.2
CADENCE_REF              = 60.0

# --- MODELE THERMIQUE ---
# La temperature est une variable d'etat deterministe qui couple cadence et usure
# aux imprecisions physiques du bras.
#
# T_eq(cadence) = T_AMBIENT + T_EQ_SLOPE * cadence
#   -> temperature d'equilibre atteinte a cadence constante
#   -> ex : cadence=60 pieces/h -> T_eq ~ 20 + 1.2*60 = 92 degC
#
# La montee suit : dT/dt = (T_eq - T) / THERMAL_TAU
#   -> THERMAL_TAU = 300 s : la machine atteint 63% de T_eq en 5 min
#
T_AMBIENT         = 20.0   # degC -- temperature ambiante / depart a froid
T_EQ_SLOPE        = 1.2    # degC par piece/heure de cadence
THERMAL_TAU       = 300.0  # s   -- constante de temps thermique (montee)
THERMAL_COOLDOWN  = 600.0  # s   -- constante de refroidissement entre sessions

# Effets de la temperature sur les parametres physiques
# (tous proportionnels a l'ecart DeltaT = T - T_AMBIENT)
TEMP_VISC_COEFF     = 0.04   # Delta visc par degC -- graisse chaude lubrifie moins bien
TEMP_NOISE_COEFF    = 0.008  # Delta motor_noise_std par degC -- bruit thermique enroulements
TEMP_DEADBAND_COEFF = 0.005  # Delta deadband par degC -- dilatation mecanique
TEMP_SENSOR_COEFF   = 0.00002  # Delta sensor_noise_q_std par degC -- dilatation disque encodeur
