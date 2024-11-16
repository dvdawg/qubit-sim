import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from qutip import * 
import qutip as qt

# params
omega_q = 1.0
omega_d = 1.5
A = 0.1
phi = 0
alpha = 1
beta = 1

# normalize and define psi_0
norm = np.sqrt(np.power(alpha, 2) + np.power(beta, 2))
alpha, beta = alpha / norm, beta / norm
psi_0 = alpha * basis(2, 0) + beta * basis(2, 1)

# substitution variables
Omega = A * np.cos(phi) + np.sin(phi) # Omega = e^(i * phi)

# Hamiltonian(s)
def driving(t, args):
    return  + A * np.cos(omega_d * t + phi)

H0 = (omega_q * sigmaz()) / 2
H1 = sigmax()
H = [H0, [H1, driving]]

# time variable
t_list = np.linspace(0, 7, 100)

# plotting
def plot_bloch():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    result = sesolve(H, psi_0, t_list)
    
    alpha = [result.states[i][0] for i in range(len(result.states))]
    beta = [result.states[i][1] for i in range(len(result.states))]

    x = 2 * np.real(np.conjugate(alpha) * beta)
    y = 2 * np.imag(np.conjugate(alpha) * beta)
    z = alpha * np.conjugate(alpha) - beta * np.conjugate(beta)

    ax.plot(x, y, z, label='qubit')
    
    plt.title("Qubit Bloch Sphere")
    plt.show()

def plot_bloch_qutip():
    result = sesolve(H, psi_0, t_list)

    bloch = Bloch()
    bloch.add_states(result.states)
    
    bloch.make_sphere()
    plt.title("Qubit Bloch Sphere")
    plt.show()


plot_bloch()