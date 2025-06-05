import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1000)
# params
kappa = 5.0
chi_1 = 1.5
chi_2 = 1.5
g_1 = 1.0
g_2 = 1.0
Omega_q = 0.2
Omega_r = 0.0
delta_r = 0.0

# drive term
def epsilon_eff(sigma_z1, sigma_z2):
    return Omega_r - 1j * Omega_q * (chi_1 * sigma_z1 / g_1 + chi_2 * sigma_z2 / g_2)

# alpha trajectory
def alpha_traj(t, sigma_z1, sigma_z2, alpha0=0):
    chi_total = chi_1 * sigma_z1 + chi_2 * sigma_z2
    decay = np.exp(-(1j * (delta_r + chi_total) + kappa / 2) * t)
    alpha_ss = epsilon_eff(sigma_z1, sigma_z2) / (kappa / 2 + 1j * (delta_r + chi_total))
    return alpha_ss + (alpha0 - alpha_ss) * decay

states = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
labels = [r"$|%d%d\rangle$" % ((1-s1)//2, (1-s2)//2) for s1, s2 in states]

# plot
plt.figure(figsize=(8, 6))
for (s1, s2), label in zip(states, labels):
    alpha = alpha_traj(t, s1, s2)
    plt.plot(alpha.real, alpha.imag, label=label)

plt.xlabel("Re(α) [I quadrature]")
plt.ylabel("Im(α) [Q quadrature]")
plt.title("Pointer State Trajectories in IQ Plane (Two Qubits)")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()
