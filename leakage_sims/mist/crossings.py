import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# === SYSTEM PARAMETERS === #
g = 120e6 * 2 * np.pi                # Coupling (Hz)
kappa = 5e6 * 2 * np.pi              # Resonator linewidth (Hz)
delta_resonator = -0.1e9 * 2 * np.pi # Resonator detuning (Hz)
Omega_r = 1.0e6 * 2 * np.pi          # Resonator drive amplitude (Hz)

# Transmon Hamiltonian (fixed structure)
EC = 0.22e9 * 2 * np.pi              # Charging energy (Hz)
EJ = 110 * EC                        # Josephson energy (Hz)
ng = 0.0
N_charge = 15
N_F = 5

# === CHARGE BASIS === #
n = np.arange(-N_charge, N_charge + 1)
dim = len(n)
H0 = np.diag(4 * EC * (n - ng)**2)
cos_phi = np.zeros((dim, dim))
for i in range(dim - 1):
    cos_phi[i, i+1] = cos_phi[i+1, i] = -0.5 * EJ
H0 += cos_phi

# Diagonalize
Evals, U = eigh(H0)
n_op = np.diag(n)
n_mat = U.T @ n_op @ U

# floquet hamiltonian
def build_HF(Evals, n_mat, epsilon_t, omega_d, N_F):
    d = len(Evals)
    D = d * (2 * N_F + 1)
    HF = np.zeros((D, D))
    for m in range(-N_F, N_F + 1):
        for i in range(d):
            idx = (m + N_F) * d + i
            HF[idx, idx] = Evals[i] + m * omega_d
    for m in range(-N_F, N_F):
        for i in range(d):
            for j in range(d):
                idx1 = (m + N_F) * d + i
                idx2 = (m + 1 + N_F) * d + j
                HF[idx1, idx2] += 0.5 * epsilon_t * n_mat[i, j]
                HF[idx2, idx1] += 0.5 * epsilon_t * n_mat[j, i]
    return HF

# === SWEEP n_r FROM PHYSICAL DRIVE === #
omega_d = 7.515e9 * 2 * np.pi  # drive freq
n_r_vals = np.linspace(0, 200, 150)
quasienergies = []

for n_r in n_r_vals:
    epsilon_t = 2 * g * np.sqrt(n_r)
    HF = build_HF(Evals[:10], n_mat[:10, :10], epsilon_t, omega_d, N_F)
    eigvals, _ = eigh(HF)
    folded = ((eigvals + omega_d / 2) % omega_d) - omega_d / 2
    quasienergies.append(np.sort(folded[:10]))

quasienergies = np.array(quasienergies)

plt.figure(figsize=(10, 6))
for i in range(10):
    plt.plot(n_r_vals, quasienergies[:, i] / (2 * np.pi * 1e9), label=f"level {i}")
plt.xlabel(r"Resonator photon number $|\alpha|^2$")
plt.ylabel("Floquet quasienergies (GHz)")
plt.title("Floquet Quasienergies vs. Resonator Photon Number")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
