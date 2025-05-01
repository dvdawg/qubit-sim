
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# ------------------------------------------------------------------
# 1. Physical and numerical parameters (all angular frequencies in MHz)
# ------------------------------------------------------------------
kappa = 5.0          # resonator linewidth  (κ/2π)
chi   = 2.5          # dispersive shift     (χ/2π)  — choose χ = κ/2 for near-optimal SNR
n_bar = 2.0          # desired steady-state photons when qubit = |g>

delta_r = 0.0        # probe on bare cavity resonance  (ω_d = ω_r)
epsilon = 0.5 * kappa * np.sqrt(n_bar)     # drive amplitude so that |α_g|^2 ≈ n_bar

Ncav = 12            # Fock-space truncation for the cavity
t_final = 8.0 / kappa        # integration window  (≈ 8/κ  is plenty)
Nt = 4000
tlist = np.linspace(0.0, t_final, Nt)

# ------------------------------------------------------------------
# 2. Operators
# ------------------------------------------------------------------
a  = tensor(destroy(Ncav), qeye(2))            # cavity annihilation
sm = tensor(qeye(Ncav), destroy(2))            # qubit lowering
sz = 2*sm.dag()*sm - tensor(qeye(Ncav), qeye(2))

# ------------------------------------------------------------------
# 3. Hamiltonian  H/ħ = χ σ_z a†a + ε (a + a†)   in frame rotating at ω_r
# ------------------------------------------------------------------
H = chi * sz * a.dag() * a + epsilon * (a + a.dag())

# ------------------------------------------------------------------
# 4. Dissipation: κ  for cavity,   (small γ, γφ could be added if desired)
# ------------------------------------------------------------------
c_ops = [np.sqrt(kappa) * a]

# ------------------------------------------------------------------
# 5. Initial states  |vac〉⊗|g〉  and  |vac〉⊗|e〉
# ------------------------------------------------------------------
vac = basis(Ncav, 0)
g   = basis(2, 0)
e   = basis(2, 1)
psi_g0 = tensor(vac, g)
psi_e0 = tensor(vac, e)

# ------------------------------------------------------------------
# 6. Solve the master equation
# ------------------------------------------------------------------
e_ops = [a, (a + a.dag())/2]     #   α(t)   and   X(t) = (a + a†)/2
res_g = mesolve(H, psi_g0, tlist, c_ops, e_ops, progress_bar=None)
res_e = mesolve(H, psi_e0, tlist, c_ops, e_ops, progress_bar=None)

alpha_g = res_g.expect[0]
alpha_e = res_e.expect[0]
Xg = res_g.expect[1]
Xe = res_e.expect[1]

# ------------------------------------------------------------------
# 7. “What the scope sees” – integrate X with optimal weight |X_e − X_g|
# ------------------------------------------------------------------
weight = np.abs(Xe - Xg)
M_g = np.trapz(weight * Xg, tlist)
M_e = np.trapz(weight * Xe, tlist)

# Vacuum-noise variance (single quadrature, quantum-limited amp, η = 1)
variance = 0.5 * np.trapz(weight**2, tlist)
SNR = np.abs(M_e - M_g) / np.sqrt(2 * variance)   # denominator uses both |g〉,|e〉 variances

# ------------------------------------------------------------------
# 8. Plots
# ------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# (a) Pointer trajectories in phase space
ax[0].plot(np.real(alpha_g), np.imag(alpha_g), label='|g⟩')
ax[0].plot(np.real(alpha_e), np.imag(alpha_e), label='|e⟩')
ax[0].scatter(np.real(alpha_g[-1]), np.imag(alpha_g[-1]), s=25)
ax[0].scatter(np.real(alpha_e[-1]), np.imag(alpha_e[-1]), s=25)
ax[0].set_xlabel(r'Re$\langle a\rangle$')
ax[0].set_ylabel(r'Im$\langle a\rangle$')
ax[0].set_title('Pointer-state trajectories')
ax[0].legend(frameon=False)

# (b) X-quadrature versus time
ax[1].plot(tlist, Xg, label='X  |g⟩')
ax[1].plot(tlist, Xe, label='X  |e⟩')
ax[1].set_xlabel('Time (µs)')
ax[1].set_ylabel(r'$\langle X\rangle$')
ax[1].set_title('X-quadrature during read-out')
ax[1].legend(frameon=False)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 9. Console report
# ------------------------------------------------------------------
print(f"Integrated homodyne signal :  M_g = {M_g:.3f} ,  M_e = {M_e:.3f}")
print(f"Vacuum-limited SNR         :  {SNR:.2f}")
