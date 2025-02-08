import numpy as np
from qutip import *
import matplotlib.pyplot as plt

omega_d = 2 * np.pi * 5.0
omega_q = 2 * np.pi * 5.0
alpha = -2 * np.pi * 0.2 
Omega = 2 * np.pi * 0.03
T_pi = np.pi / (2 * Omega)
duration = 13
tlist = np.linspace(0, duration, 1000)

def drive(t, args):
    if t <= T_pi:
        return 1
    else:
        return 0

def simulate(N):
    a = destroy(N) 
    H_drive = Omega * a + a.dag() * np.conj(Omega)
    H_qubit = (omega_q - omega_d) * a.dag() * a  
    H_anharm = (alpha / 2) * a.dag() * a.dag() * a * a 

    H = [H_qubit, H_anharm, [H_drive, drive]]

    psi0 = basis(N, 0)
    result = sesolve(H, psi0, tlist, e_ops=[basis(N, i) * basis(N, i).dag() for i in range(N)])

    populations = result.expect
    return populations

populations_2 = simulate(2)
populations_3 = simulate(3)
populations_4 = simulate(4)
populations_5 = simulate(5)

plt.figure(figsize=(13, 8))

plt.subplot(2, 2, 1)
for i, pop in enumerate(populations_2):
    plt.plot(tlist, pop, label=f"|{i}⟩")
plt.xlabel("t (ns)")
plt.ylabel("Population")
plt.title("2-Level System")
plt.legend()

plt.subplot(2, 2, 2)
for i, pop in enumerate(populations_3):
    plt.plot(tlist, pop, label=f"|{i}⟩")
plt.xlabel("t (ns)")
plt.ylabel("Population")
plt.title("3-Level System")
plt.legend()

plt.subplot(2, 2, 3)
for i, pop in enumerate(populations_4):
    plt.plot(tlist, pop, label=f"|{i}⟩")
plt.xlabel("t (ns)")
plt.ylabel("Population")
plt.title("4-Level System")
plt.legend()

plt.subplot(2, 2, 4)
for i, pop in enumerate(populations_5):
    plt.plot(tlist, pop, label=f"|{i}⟩")
plt.xlabel("t (ns)")
plt.ylabel("Population")
plt.title("5-Level System")
plt.legend()

plt.tight_layout()
plt.show()
