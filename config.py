# config.py

# --- PARAMÈTRES PHYSIQUES IDÉAUX ---
m1, m2 = 2.0, 1.5
l1, l2 = 1.0, 1.0
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
visc_friction_coeffs = [2.5, 1.5]
coulomb_friction_coeffs = [5.0, 3.0]
stribeck_friction_coeffs = [8.0, 5.0]

# Moteurs et Actuateurs
ripple_amplitude = 0.05
gain_error_std = 0.03      # Ecart-type de l'erreur de gain des moteurs
motor_noise_std = 2.0      # Ecart-type du bruit blanc additif
deadband_torque = 1.5      # Zone morte (couple minimum pour bouger)

# Capteurs
encoder_resolution = 8192
sensor_noise_q_std = 0.0002
sensor_noise_dq_std = 0.008
sensor_glitch_prob = 0.001  # Probabilité d'avoir un pic de bruit aberrant
sensor_glitch_q_std = 0.05
sensor_glitch_dq_std = 0.5

# --- CONTRÔLEUR ---
Kp_gain = 1200.0
Kd_gain = 120.0
alpha_filter = 0.15