import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from qutip import * 
import qutip as qt

# params
omega_q = 1.0 
A = 1
omega_d = omega_q
phi = 0.4 # phase
alpha = 1 / np.sqrt(2)
beta = 1 / np.sqrt(2)
t_list = np.linspace(0, 100, 100)

def driving_field(t, args):
    return A * np.cos(omega_d * t + phi)

Omega = A * constants.e**(1j * phi)
H_rwa = (1/2) * (create(2) * Omega + destroy(2) * np.conj(Omega))

def qubit_time_evolution_qutip(hamiltonian, alpha, beta, tlist):
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    alpha, beta = alpha / norm, beta / norm
    
    initial_state = alpha * basis(2, 0) + beta * basis(2, 1)
    
    result = sesolve(hamiltonian, initial_state, tlist)

    bloch = Bloch()
    bloch.add_states(result.states)
    
    bloch.make_sphere()
    plt.show()

qubit_time_evolution_qutip(H_rwa, alpha, beta, t_list)