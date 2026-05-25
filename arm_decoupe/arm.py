# arm.py
import numpy as np
import math
from config import (
    m1, m2, l1, l2, r1, r2, I1, I2, g,
    visc_friction_coeffs, coulomb_friction_coeffs, stribeck_friction_coeffs,
    ripple_amplitude, gain_error_std, motor_noise_std, deadband_torque,
    encoder_resolution, sensor_noise_q_std, sensor_noise_dq_std,
    sensor_glitch_prob, sensor_glitch_q_std, sensor_glitch_dq_std,
    Kp_gain, Kd_gain, alpha_filter, vel_noise_scale_k
)
from degradation import friction_multiplier, noise_multiplier

class PhysicsArmEnv:
    def __init__(self, initial_q=None, machine_state=None):
        if initial_q is None:
            self.state = np.array([math.pi/4, -math.pi/2, 0.0, 0.0])
        else:
            self.state = np.array([initial_q[0], initial_q[1], 0.0, 0.0])
        self.dt = 0.01

        fm = friction_multiplier(machine_state['piece_count']) if machine_state else 1.0
        nm = noise_multiplier(machine_state['cadence'])        if machine_state else 1.0
        self.visc              = np.array(visc_friction_coeffs)     * fm
        self.coulomb           = np.array(coulomb_friction_coeffs)  * fm
        self.stribeck          = np.array(stribeck_friction_coeffs) * fm
        self.motor_noise_std_eff = motor_noise_std * nm
        self.deadband_eff        = deadband_torque * nm
        
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
        f_visc = self.visc * dq

        # 2. Frottements de Coulomb (constants avec le signe de la vitesse)
        f_coulomb = self.coulomb * np.tanh(dq * 10.0)

        # 3. Effet Stribeck (pic d'adhérence au démarrage, chute quand on prend de la vitesse)
        f_stribeck = self.stribeck * np.exp(-np.abs(dq) * 15.0) * np.tanh(dq * 100.0)

        # 4. Torque ripple (défaut électromagnétique des moteurs)
        ripple = tau * ripple_amplitude * np.sin(50 * self.state[0:2])

        # 5. BRUIT ACTUATEUR (Non-déterministe, Process Noise)
        gain_error  = np.random.normal(1.0, gain_error_std, 2)
        motor_noise = np.random.normal(0, self.motor_noise_std_eff, 2)

        # 6. Zone morte (Deadband - jeu mécanique où les petites commandes n'ont pas d'effet)
        tau_effective = np.where(np.abs(tau) > self.deadband_eff,
                                 tau - np.sign(tau) * self.deadband_eff, 0.0)

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
        encoder_res = (2 * math.pi) / encoder_resolution

        # À haute vitesse articulaire, chaque tick encodeur couvre un plus grand angle
        # et l'estimation de vitesse par différences finies se dégrade → bruit scalé par ||dq||
        vel_scale = 1.0 + vel_noise_scale_k * np.linalg.norm(dq_real)

        q_noisy = q_real + np.random.normal(0, sensor_noise_q_std * vel_scale, 2)
        dq_noisy = dq_real + np.random.normal(0, sensor_noise_dq_std * vel_scale, 2)

        # Quantification liée à la résolution de l'encodeur optique
        q_sensed = np.round(q_noisy / encoder_res) * encoder_res

        if np.random.random() < sensor_glitch_prob:
            q_sensed += np.random.normal(0, sensor_glitch_q_std, 2)
            dq_noisy += np.random.normal(0, sensor_glitch_dq_std, 2)

        return q_sensed, dq_noisy

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
        self.alpha = alpha_filter 

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
        
        Kp = np.array([[Kp_gain, 0.0], [0.0, Kp_gain]])
        Kd = np.array([[Kd_gain, 0.0], [0.0, Kd_gain]])  
        e = q_des - q_sensed
        de = dq_des - self.filtered_dq
        
        tau = M @ (ddq_des + Kd @ de + Kp @ e) + V + G
        return np.clip(tau, -400, 400)
