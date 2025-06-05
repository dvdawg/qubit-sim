import numpy as np
import matplotlib.pyplot as plt

# params
chi = 1.0
kappa = 3.0
g = 1.0
Omega_q = 0.2
Omega_r = 0.0
t = np.linspace(0, 10, 1000)

def alpha(t, sigma_z, alpha0=0):
    decay = np.exp(-(1j * chi * sigma_z + kappa / 2) * t)
    alpha_ss = (Omega_r + 1j * Omega_q * chi / g * sigma_z) / (kappa / 2 + 1j * chi * sigma_z)
    return alpha_ss + (alpha0 - alpha_ss) * decay

# plot
for sz, label in [(-1, r"$|0⟩$"), (1, r"$|1⟩$")]:
    traj = alpha(t, sz)
    plt.plot(traj.real, traj.imag, label=label)

plt.scatter(-Omega_q/g, 0, label='vo', color='green')

plt.xlabel("Real")
plt.ylabel("Imag")
plt.title("Pointer State Trajectories")
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
