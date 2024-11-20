import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from qutip import * 
import qutip as qt

# params
omega_q = 5.0
omega_d = 5.0
A = 0.126
phi = 0
alpha = 1
beta = 0

# normalize and define psi_0
norm = np.sqrt(alpha * alpha + beta * beta)
alpha = alpha / norm
beta = beta / norm
psi_0 = alpha * basis(2, 0) + beta * basis(2, 1)

# substitution variables
Omega = A * np.cos(phi) + A * 1j * np.sin(phi) # Omega = e^(i * phi)

# Hamiltonian(s)
def driving(t, args):
    return  A * np.cos(omega_d * t + phi)

def counter_plus(t, args):
    return np.cos(2 * omega_d * t) + 1j * np.sin(2 * omega_d * t)
def counter_minus(t, args):
    return np.cos(2 * omega_d * t) - 1j * np.sin(2 * omega_d * t)

H0 = (omega_q * sigmaz()) / 2
H1 = sigmax()
H = [H0, [H1, driving]]

H_rwa = (1/2) * Omega * create(2) + (1/2) * np.conj(Omega) * destroy(2)
H_counter_plus = (1/2) * Omega * create(2)
H_counter_minus = (1/2) * np.conj(Omega) * destroy(2)
H_qubit = [H_rwa, [H_counter_plus, counter_plus], [H_counter_minus, counter_minus]]

# time variable
t_list = np.linspace(0, 50, 5000)

def plot_bloch():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
 
    result = sesolve(H_qubit, psi_0, t_list)
    result2 = sesolve(H_qubit, psi_0, t_list, sigmax())
    print(max(result2.expect[0].tolist()))
    
        
    alpha = [result.states[i][0] for i in range(len(result.states))]
    beta = [result.states[i][1] for i in range(len(result.states))]

    x = 2 * np.real(np.conjugate(alpha) * beta)
    y = 2 * np.imag(np.conjugate(alpha) * beta)
    z = alpha * np.conjugate(alpha) - beta * np.conjugate(beta)

    ax.plot(x, y, z, label='qubit')

    phi, theta = np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)
    sp_x = np.sin(theta) * np.cos(phi)
    sp_y = np.sin(theta) * np.sin(phi)
    sp_z = np.cos(theta)

    ax.plot_wireframe(sp_x, sp_y, sp_z, rstride=4, cstride=4, color='#d3d3d3', edgecolor='k', alpha=0.3)
    ax.text(0, 0, 1.2, '|0⟩')
    ax.text(0, 0, -1.2, '|1⟩')
    
    plt.title("Qubit Bloch Sphere")
    plt.show()

plot_bloch()

def find_deco():
    
    result = sesolve(H_qubit, psi_0, t_list, sigmax())
    print(max(result.expect[0].tolist()))
