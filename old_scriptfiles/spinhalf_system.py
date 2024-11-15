import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from qutip import * 
import qutip as qt

# params
omega_q = 3.0 # qubit frequency
omega_d = 3.0 # driving frequency
A = 1
phi = 0.7 # phase
alpha = 1/2 # psi = alpha |0> + beta |1>
beta = np.sqrt(3) / 2

psi_0 = alpha * basis(2, 0) + beta * basis(2, 1)

t_list = np.linspace(0, 100, 100)

def hamiltonian(t):
    return omega_q * sigmaz() + A * np.cos(omega_d * t + phi)

Omega = A * np.cos(phi) + np.sin(phi) # Omega = e^(i * phi)

def plot_bloch():
    ax = plt.figure().add_subplot(projection='3d')
    result = sesolve(hamiltonian, psi_0, t_list)

    print("finished!!!!")