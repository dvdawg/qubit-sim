import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# params
kappa = 3.0
g = 1.0
Delta = 0.2
chi = g**2 / Delta
Omega_q = 0.2
Omega_r = 0.0  # compensation tone

Ncav = 12
t_final = 8.0 / kappa
Nt = 4000
tlist = np.linspace(0.0, t_final, Nt)

# operators
a = tensor(destroy(Ncav), qeye(2))
sm = tensor(qeye(Ncav), sigmam())
sp = sm.dag()
sz = tensor(qeye(Ncav), sigmaz())

# hamiltonian
H = ((Delta + chi) * sp * sm
     - chi * sz * a.dag() * a
     + ((Omega_q + 1j * Omega_r * chi / g) * sp + ((Omega_q + 1j * Omega_r * chi / g) * sp).dag())
     + ((1j * Omega_r - Omega_q * chi / g) * sz * a.dag() + ((1j * Omega_r - Omega_q * chi / g) * sz * a.dag()).dag()))

# collapse operator
c_ops = [np.sqrt(kappa) * a]

# initial states
init_g = tensor(basis(Ncav, 0), basis(2, 0))
init_e = tensor(basis(Ncav, 0), basis(2, 1))

# simulation
res_g = mesolve(H, init_g, tlist, c_ops, [a])
res_e = mesolve(H, init_e, tlist, c_ops, [a])

# plot
plt.figure(figsize=(8, 6))
plt.plot(np.real(res_g.expect[0]), np.imag(res_g.expect[0]), label='|0⟩', color='blue')
plt.plot(np.real(res_e.expect[0]), np.imag(res_e.expect[0]), label='|1⟩', color='red')
plt.scatter(-Omega_q/g, 0, label='vo', color='green')
plt.scatter(np.real(res_g.expect[0][-1]), np.imag(res_g.expect[0][-1]), color='blue')
plt.scatter(np.real(res_e.expect[0][-1]), np.imag(res_e.expect[0][-1]), color='red')
plt.xlabel("Real")
plt.ylabel("Imag")
plt.title("Pointer State Trajectories Multichannel Driving")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
