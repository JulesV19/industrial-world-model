# config.py

# --- PARAMÈTRES PHYSIQUES IDÉAUX ---
m1, m2 = 2.0, 1.5
import math as _math
l1, l2 = _math.sqrt(2), _math.sqrt(2)   # portée totale = 2√2, atteint le coin (2,2)
r1, r2 = 0.5, 0.5
I1, I2 = 0.1, 0.08
g = 9.81

# --- CONSTANTES UI ET ENVIRONNEMENT ---
WIDTH, HEIGHT = 1400, 800
scale_px = 220
origin = (350, 600)
HISTORY_LEN = 300

# --- IMPERFECTIONS PHYSIQUES ET BRUIT ---
# Frottements
visc_friction_coeffs = [2.5*2, 1.5*2]
coulomb_friction_coeffs = [5.0, 3.0]
stribeck_friction_coeffs = [8.0, 5.0]

# Moteurs et Actuateurs
ripple_amplitude = 0.05
gain_error_std = 0.01      # Ecart-type de l'erreur de gain des moteurs
motor_noise_std = 1.0      # Ecart-type du bruit blanc additif
deadband_torque = 0.8      # Zone morte (couple minimum pour bouger)

# Capteurs
encoder_resolution = 8192
sensor_noise_q_std = 0.002
sensor_noise_dq_std = 0.008
sensor_glitch_prob = 0.001  # Probabilité d'avoir un pic de bruit aberrant
sensor_glitch_q_std = 0.05
sensor_glitch_dq_std = 0.5
vel_noise_scale_k = 0.4    # Amplification du bruit capteur par rad/s de vitesse articulaire

# --- CONTRÔLEUR ---
Kp_gain = 1200.0
Kd_gain = 120.0
alpha_filter = 0.15

# --- DÉGRADATION TEMPORELLE ---
# ALPHA    : amplification totale max des frottements (1 + ALPHA à saturation)
#            → augmenter pour des défauts plus sévères en fin de vie
# HALFLIFE : nombre de pièces pour atteindre 50 % de la dégradation max
#            → augmenter pour repousser l'apparition des défauts
#            → diminuer pour les faire apparaître plus tôt
# GAMMA    : amplification max du bruit moteur à CADENCE_REF
#            → augmenter si la cadence doit aggraver les défauts
FRICTION_DEGRAD_ALPHA    = 4.0    # +300 % de frottements à saturation
FRICTION_DEGRAD_HALFLIFE = 1500   # 50 % de dégradation atteint à ~1500 pièces
TEMP_NOISE_GAMMA         = 0.2    # bruit moteur peu sensible à la cadence
CADENCE_REF              = 60.0   # pièces/heure de référence