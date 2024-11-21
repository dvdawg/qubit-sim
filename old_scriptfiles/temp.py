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
time = 100
t_list = np.linspace(0, time, 5000)
A = 0.010
deviation = 0

while (deviation < 0.05):
    A += 0.001
    result = sesolve(H_qubit, psi_0, t_list, sigmax())
    vals = result.expect[0].tolist()
    deviation = max(vals) - min(vals)

print(A)
