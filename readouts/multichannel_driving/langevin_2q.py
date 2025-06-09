import numpy as np
import matplotlib.pyplot as plt

# Time array
t = np.linspace(0, 10, 1000)

kappa = 5.0

g_1 = 0.4
g_2 = 0.1

delta_r = 1.0  

chi_1 = g_1**2 / delta_r
chi_2 = g_2**2 / delta_r

phi_q1 = 0
phi_q2 = np.pi / 2

Omega_q1_mag = 0.1
Omega_q1 = Omega_q1_mag * np.exp(1j * phi_q1)

ratio = chi_1 * g_2 / (chi_2 * g_1)
Omega_q2_mag = Omega_q1_mag * ratio
Omega_q2 = Omega_q2_mag * np.exp(1j * phi_q2)

Omega_r = 0.0  

def epsilon_eff(sigma_z1, sigma_z2):
    return 1j * Omega_r - (Omega_q1 * chi_1 * sigma_z1 / g_1 + Omega_q2 * chi_2 * sigma_z2 / g_2)

def alpha_traj(t, sigma_z1, sigma_z2, alpha0=0):
    delta_eff = delta_r + chi_1 * sigma_z1 + chi_2 * sigma_z2
    decay = np.exp(-(1j * delta_eff + kappa / 2) * t)
    alpha_ss = epsilon_eff(sigma_z1, sigma_z2) / (1j * kappa / 2 + delta_eff)
    return alpha_ss + (alpha0 - alpha_ss) * decay

def calculate_steady_state(sigma_z1, sigma_z2):
    delta_eff = delta_r + chi_1 * sigma_z1 + chi_2 * sigma_z2
    return epsilon_eff(sigma_z1, sigma_z2) / (1j * kappa / 2 + delta_eff)

def calculate_snr(state1, state2, n_avg=1000):
    alpha1 = calculate_steady_state(*state1)
    alpha2 = calculate_steady_state(*state2)
    separation = np.abs(alpha1 - alpha2)

    # photon shot noise
    n1 = np.abs(alpha1)**2
    n2 = np.abs(alpha2)**2

    noise = np.sqrt(n1 + n2) / np.sqrt(n_avg)
    
    return separation / noise

states = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
labels = [r"$|%d%d\rangle$" % ((1 - s1) // 2, (1 - s2) // 2) for s1, s2 in states]

snr_matrix = np.zeros((4, 4))
for i, state1 in enumerate(states):
    for j, state2 in enumerate(states):
        if i != j:
            snr_matrix[i, j] = calculate_snr(state1, state2)

plt.figure(figsize=(10, 8))
colors = {'00': 'blue', '01': 'red', '10': 'green', '11': 'purple'}
for (s1, s2), label in zip(states, labels):
    alpha = alpha_traj(t, s1, s2)
    state = f"{(1 - s1) // 2}{(1 - s2) // 2}"
    plt.plot(alpha.real, alpha.imag, label=label, color=colors[state])
    alpha_ss = calculate_steady_state(s1, s2)
    plt.plot(alpha_ss.real, alpha_ss.imag, 'o', color=colors[state])

plt.xlabel("Real")
plt.ylabel("Imag")
plt.title("Pointer State Trajectories")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()

print("\nSNR Matrix:")
print("           |00⟩    |01⟩    |10⟩    |11⟩")
print("-" * 40)
for i, state1 in enumerate(states):
    row = f"|{(1-state1[0])//2}{(1-state1[1])//2}⟩ |"
    for j, state2 in enumerate(states):
        if i == j:
            row += "   -    "
        else:
            row += f" {snr_matrix[i,j]:6.2f} "
    print(row)

# phase response graphs

# omega_probe = np.linspace(-30, 30, 300)
# epsilon = 0.5 * kappa * np.sqrt(2.0)
# alpha_phases = {}

# states = {'00': (-1, -1), '01': (-1, +1), '10': (+1, -1), '11': (+1, +1)}

# for label, (sz1_val, sz2_val) in states.items():
#     delta_eff = omega_probe - (chi_1 * sz1_val + chi_2 * sz2_val)
#     alpha = epsilon / (kappa / 2 + 1j * delta_eff)
#     alpha_phases[label] = np.angle(alpha)

# plt.figure(figsize=(8, 6))
# for label, phase in alpha_phases.items():
#     plt.plot(omega_probe, phase, label=f"|{label}⟩")
# plt.title("Phase Response vs Probe Frequency")
# plt.xlabel("ω - ω_r (MHz)")
# plt.ylabel("Arg(α) [rad]")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
