import numpy as np
import matplotlib.pyplot as plt
from qutip import * 
import qutip as qt

# params
omega = 1.0 # H = omega * sigma_z
alpha = 1
beta = 1  # psi = alpha |0> + beta |1>
t_list = np.linspace(0, 400)  # time points
psi0 = alpha * basis(2, 0) + beta * basis(2, 1)  # qobj init state

H = omega * sigmaz()

def qubit_time_evolution(hamiltonian, alpha, beta, tlist):
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    alpha, beta = alpha / norm, beta / norm
    
    initial_state = alpha * basis(2, 0) + beta * basis(2, 1)
    
    result = sesolve(hamiltonian, initial_state, tlist)

    bloch = Bloch()
    bloch.add_states(result.states)
    
    bloch.make_sphere()
    plt.show()

def qubit_time_evolution2 (hamiltonian, psi0, tlist):
    result = mesolve(hamiltonian, psi0, tlist, [], [])
    b = Bloch()

    for state in result.states:
        rho = state.proj()
        x = 2 * np.real(rho[0, 1])
        y = 2 * np.imag(rho[1, 0])
        z = rho[0, 0] - rho[1, 1]
        
        b.add_points([x, y, z])
        b.make_sphere()
    plt.show()

def qubit_time_evolution3 (hamiltonian, psi0, tlist):
    print ('a')

qubit_time_evolution(H, alpha, beta, t_list)