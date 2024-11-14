import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from qutip import * 
import qutip as qt

# params
omega = 1.0 # H = omega * sigma_z
alpha = 1 / 2
beta = np.sqrt(3) / 2 # psi = alpha |0> + beta |1>
t_list = np.linspace(0, 100, 100)  # time points
psi0 = alpha * basis(2, 0) + beta * basis(2, 1)  # qobj init state

H = omega * sigmaz()

def qubit_time_evolution_qutip(hamiltonian, alpha, beta, tlist):
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    alpha, beta = alpha / norm, beta / norm
    
    initial_state = alpha * basis(2, 0) + beta * basis(2, 1)
    
    result = sesolve(hamiltonian, initial_state, tlist)

    bloch = Bloch()
    bloch.add_states(result.states)
    
    bloch.make_sphere()
    plt.show()

def qubit_time_evolution_matplot(alpha, beta):
    ax = plt.figure().add_subplot(projection='3d')
    t = np.linspace(0, 100, 100)
    alpha_r = alpha * (np.cos(omega*t/constants.hbar) - 1j * np.sin(omega*t/constants.hbar))
    beta_r = beta * (np.cos(omega*t/constants.hbar) + 1j * np.sin(omega*t/constants.hbar)) 
    
    z = alpha**2 - beta**2
    y = 2 * np.imag(np.conjugate(alpha_r) * beta_r)
    x = 2 * np.real(np.conjugate(alpha_r) * beta_r)

    ax.plot(x, y, z, label='parametric curve')
    ax.legend()

    plt.show()
    
qubit_time_evolution_matplot(alpha, beta)
qubit_time_evolution_qutip(H, alpha, beta, t_list)