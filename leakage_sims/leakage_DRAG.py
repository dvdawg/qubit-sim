import numpy as np
from qutip import *
import matplotlib.pyplot as plt

omega_d = 2 * np.pi * 5.0
omega_q = 2 * np.pi * 5.01
alpha = -2 * np.pi * 0.2 
Omega = 2 * np.pi * 0.03
T_pi = np.pi / (2 * Omega)
duration = 13
tlist = np.linspace(0, duration, 1000)

pulse_width = 50

def drive(t, args):
    if t <= T_pi:
        return np.e**(-t**2 / (2 * pulse_width**2))
    else:
        return 0
    
def drive_DRAG(t, args):
    if t <= T_pi:
        return -1j * 200 * (-t / (pulse_width**2)) * np.exp(-t**2 / (2 * pulse_width**2))
    else:
        return 0


def simulate_drag(N):
    a = destroy(N) 
    H_drive = Omega * a + a.dag() * np.conj(Omega)
    H_qubit = (omega_q - omega_d) * a.dag() * a  
    H_anharm = (alpha / 2) * a.dag() * a.dag() * a * a 
    H_drag =  Omega * a + (1j) * a.dag() * np.conj(Omega)

    H = [H_qubit, H_anharm, [H_drive, drive], [H_drag, drive_DRAG]]

    psi0 = basis(N, 0)
    result = sesolve(H, psi0, tlist, e_ops=[basis(N, i) * basis(N, i).dag() for i in range(N)])

    populations = result.expect
    print("Final populations for DRAG simulation:")
    for i, pop in enumerate(populations):
        print(f"State |{i}\u27E9: {pop[-1]:.4f}")
    return populations

def simulate(N):
    a = destroy(N) 
    H_drive = Omega * a + a.dag() * np.conj(Omega)
    H_qubit = (omega_q - omega_d) * a.dag() * a  
    H_anharm = (alpha / 2) * a.dag() * a.dag() * a * a 

    H = [H_qubit, H_anharm, [H_drive, drive]]

    psi0 = basis(N, 0)
    result = sesolve(H, psi0, tlist, e_ops=[basis(N, i) * basis(N, i).dag() for i in range(N)])

    populations = result.expect
    print("Final populations for standard simulation:")
    for i, pop in enumerate(populations):
        print(f"State |{i}\u27E9: {pop[-1]:.4f}")
    return populations

populations_3 = simulate(3)
populations_3_drag = simulate_drag(3)

plt.figure(figsize=(13, 8))

plt.subplot(1, 2, 1)
for i, pop in enumerate(populations_3):
    plt.plot(tlist, pop, label=f"|{i}\u27E9")
plt.xlabel("t (ns)")
plt.ylabel("Population")
plt.title("3-Level System")
plt.legend()

plt.subplot(1, 2, 2)
for i, pop in enumerate(populations_3_drag):
    plt.plot(tlist, pop, label=f"|{i}\u27E9")
plt.xlabel("t (ns)")
plt.ylabel("Population")
plt.title("3-Level System with DRAG")
plt.legend()

plt.show()
