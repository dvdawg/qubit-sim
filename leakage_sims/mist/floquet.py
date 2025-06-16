import numpy as np
import matplotlib.pyplot as plt

tlist = np.linspace(0, 10, 1000)

def calculate_steady_state(sigma_z, params, chi, delta_r, kappa, g, delta_resonator):
    Omega_q_mag, phi_q = params
    Omega_q = Omega_q_mag * np.exp(1j * phi_q)
    Omega_r = 0.0

    epsilon = (1j * Omega_r - (Omega_q * chi * sigma_z / g))
    delta_eff = chi * sigma_z

    return epsilon / (1j * kappa/2 + delta_eff + delta_resonator)

def alpha_traj(t, sigma_z, params, chi, kappa, g, delta_resonator):
    Omega_q_mag, phi_q = params
    Omega_q = Omega_q_mag * np.exp(1j * phi_q)
    Omega_r = 0.0
    
    epsilon = 1j * Omega_r - (Omega_q * chi * sigma_z / g)
    delta_eff = delta_resonator + chi * sigma_z
    decay = np.exp(-(1j * delta_eff + kappa / 2) * t)
    alpha_ss = epsilon / (1j * kappa / 2 + delta_eff)
    return alpha_ss + (0 - alpha_ss) * decay

params = [1, 0]
chi = 1
kappa = 1
g = 1
delta_resonator = 1

n_crit = np.full_like(tlist, delta_resonator / (4*g**2))
plt.plot(tlist, alpha_traj(tlist, 1, params, chi, kappa, g, delta_resonator) * alpha_traj(tlist, 1, params, chi, kappa, g, delta_resonator).conj())
plt.plot(tlist, n_crit)
plt.show()