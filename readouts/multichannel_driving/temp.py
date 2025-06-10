# Final attempt: ensure LaTeX is fully disabled
import matplotlib.pyplot as plt
import numpy as np

# Parameters
kappa = 3.0
chi = 1.0
g = 1.0
Omega_q = 0.3
Omega_r = 0.4
t = np.linspace(0, 10, 1000)

# Pointer state trajectory function
def alpha(t, s, alpha0=0):
    decay = np.exp(-(kappa / 2 - 1j * chi * s) * t)
    alpha_ss = (Omega_r + 1j * Omega_q * chi / g * s) / (kappa / 2 - 1j * chi * s)
    return alpha_ss + (alpha0 - alpha_ss) * decay

# s values to consider
s_vals = [-1, 0, 1]
labels = ["<sigma_z> = -1", "<sigma_z> = 0", "<sigma_z> = 1"]

# Plotting
plt.figure(figsize=(7, 6))
for s, label in zip(s_vals, labels):
    traj = alpha(t, s)
    plt.plot(traj.real, traj.imag, label=label)

plt.xlabel("Re(alpha)")
plt.ylabel("Im(alpha)")
plt.title("Pointer State Trajectories in Phase Space")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
