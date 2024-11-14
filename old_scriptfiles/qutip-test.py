import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from qutip import * 
import qutip as qt

def qubit_time_evolution(hamiltonian, alpha, beta, tlist):
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    alpha, beta = alpha / norm, beta / norm
    
    initial_state = alpha * basis(2, 0) + beta * basis(2, 1)
    
    result = sesolve(hamiltonian, initial_state, tlist)

    bloch = Bloch()
    bloch.add_states(result.states)
    
    bloch.make_sphere()
    plt.show()

    def output_points(alpha, beta):
        theta = 2 * np.arccos(alpha)
        