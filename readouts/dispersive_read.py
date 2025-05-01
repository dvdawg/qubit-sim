
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

omega_cav   = 2*np.pi*7.0   # cavity bare frequency  (rad/µs)  ≈ 7 GHz
omega_q   = 2*np.pi*5.0   # qubit bare frequency   (rad/µs)  ≈ 5 GHz
g    = 2 * np.pi * 0.10  # coupling strength      (rad/µs)  ≈ 100 MHz
kappa    = 1/0.25 # cavity κ  = 4 µs⁻¹  (≈ 1/(250 ns))
gamma1   = 1/50 # qubit T₁⁻¹           (≈ 50 µs)
gamma_phi= 1/80 # pure dephasing rate   (≈ 80 µs)
N_cav = 10 # photon Fock-space truncation

wd = omega_cav # on-resonant drive with cavity freq
eps = 2*np.pi*0.004 # drive amplitude (rad/µs)  ≈ 4 MHz


a  = tensor(destroy(N_cav), qeye(2)) # cavity annihilation
adag = a.dag()
sm = tensor(qeye(N_cav), sigmam()) # qubit lowering
sp = sm.dag()
sz = tensor(qeye(N_cav), sigmaz())


H0 = omega_cav*adag*a + 0.5*omega_q*sz + g*(adag*sm + a*sp)

def eps_t(t, args): # square pulse of length t_p
    t_p = args["t_p"]
    return args["eps"] if 0 <= t <= t_p else 0.0

H_drive = [ a + adag, eps_t ]
H = [H0, H_drive]


c_ops = []
if kappa   > 0.0: c_ops.append( np.sqrt(kappa) * a )
if gamma1  > 0.0: c_ops.append( np.sqrt(gamma1) * sm )
if gamma_phi > 0.0: c_ops.append( np.sqrt(gamma_phi) * sz )


g_state = tensor( basis(N_cav,0), basis(2,0) )
e_state = tensor( basis(N_cav,0), basis(2,1) )


t_pulse = 0.7        # µs — drive length
t_final = 2.0        # µs — simulate until cavity relaxes
t_steps = 2001
tlist = np.linspace(0, t_final, t_steps)

args = dict(eps=eps, t_p=t_pulse)

expect_ops = [ a, a+adag ]      # <a>,  X-quadrature
res_g = mesolve(H, g_state, tlist, c_ops, expect_ops, args=args, progress_bar=None)
res_e = mesolve(H, e_state, tlist, c_ops, expect_ops, args=args, progress_bar=None)

a_g  = res_g.expect[0]
a_e  = res_e.expect[0]
X_g  = 0.5*res_g.expect[1]       # Re{a} because <a+a†> = 2 Re<a>
X_e  = 0.5*res_e.expect[1]
dX   = X_e - X_g                 # pointer separation


# Shot-noise-limited homodyne (vacuum variance σ² = 1/4):
sigma = 0.5                   # std.-dev. of X quadrature
SNR_inst = np.abs(dX)/sigma
# Integrated SNR over [0, t]  ->  sqrt( ∫ (dX/σ)² dt )
snr_cum = np.sqrt(np.cumsum(SNR_inst**2) * (tlist[1]-tlist[0]))


fig, ax = plt.subplots(2,1, sharex=True, figsize=(6,5))
ax[0].plot(tlist, X_g, label='X |g⟩')
ax[0].plot(tlist, X_e, label='X |e⟩')
ax[0].set_ylabel('⟨X⟩  (dimensionless)')
ax[0].legend(loc='best')

ax[1].plot(tlist, snr_cum, color='black')
ax[1].set_ylabel('Integrated SNR')
ax[1].set_xlabel('time  (µs)')
ax[1].set_ylim(bottom=0)

plt.tight_layout()
plt.show()
