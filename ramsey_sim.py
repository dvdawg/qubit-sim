import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from qutip import * 
import qutip as qt

# params
omega_q = 5.0 
omega_d = 5.0
A = np.pi/24 # A for π/2 pulse
phi = 1
alpha = 1
beta = 0

# normalize and define initial state
norm = np.sqrt(alpha * alpha + beta * beta)
alpha = alpha / norm
beta = beta / norm
psi_0 = alpha * basis(2, 0) + beta * basis(2, 1)

Omega = A * np.cos(phi) + A * 1j * np.sin(phi)

# time-dependent functions
def driving(t, args):
    return A * np.cos(omega_d * t + phi)

def counter_plus(t, args):
    return np.exp(2j * omega_d * t)

def counter_minus(t, args):
    return np.exp(-2j * omega_d * t)

# hamiltonians
H0 = (omega_q * sigmaz()) / 2
H1 = sigmax()

# RWA Hamiltonian in rotating frame
H_rwa = (1/2) * Omega * create(2) + (1/2) * np.conj(Omega) * destroy(2)

# counter-rotating terms
H_counter_plus = (1/2) * Omega * create(2)
H_counter_minus = (1/2) * np.conj(Omega) * destroy(2)

# off-resonant Hamiltonian with counter-rotating terms
H_offres = [(omega_q - omega_d) * sigmaz() / 2, H_rwa, [H_counter_plus, counter_plus], [H_counter_minus, counter_minus]]

# time params
pulse_time = 12  # π/2 pulse duration
free_time = 50   # free evolution time

def create_bloch_coords(Hamiltonian, time, psi):
    result = sesolve(Hamiltonian, psi, np.linspace(0, time, 100))
    result_expect = sesolve(Hamiltonian, psi, np.linspace(0, time, 100), [sigmax(), sigmay(), sigmaz()])
    
    states = result.states
    alpha = [state[0] for state in states]
    beta = [state[1] for state in states]
    
    x = 2 * np.real(np.conjugate(alpha) * beta)
    y = 2 * np.imag(np.conjugate(alpha) * beta)
    z = np.array([np.abs(a)**2 - np.abs(b)**2 for a, b in zip(alpha, beta)])
    
    return x, y, z, result_expect, result

# first π/2 pulse
x1, y1, z1, result1, state1 = create_bloch_coords(H_offres, pulse_time, psi_0)
final_state1 = state1.states[-1]

# free evolution
H_free = [(omega_q - omega_d) * sigmaz() / 2]
x2, y2, z2, result2, state2 = create_bloch_coords(H_free, free_time, final_state1)
final_state2 = state2.states[-1]

# second π/2 pulse
x3, y3, z3, result3, state3 = create_bloch_coords(H_offres, pulse_time, final_state2)

# combine trajectories
times1 = np.linspace(0, pulse_time, 100)
times2 = np.linspace(pulse_time, pulse_time + free_time, 100)
times3 = np.linspace(pulse_time + free_time, 2*pulse_time + free_time, 100)

x_total = np.concatenate([x1, x2, x3])
y_total = np.concatenate([y1, y2, y3])
z_total = np.concatenate([z1, z2, z3])
times_total = np.concatenate([times1, times2, times3])

# plotting
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(3, 3, figure=fig, hspace=0.6, wspace=0.5)

ax = fig.add_subplot(gs[:, :2], projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

ax.plot(x_total, y_total, z_total, label='qubit')

phi, theta = np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 50)
phi, theta = np.meshgrid(phi, theta)
sp_x = np.sin(theta) * np.cos(phi)
sp_y = np.sin(theta) * np.sin(phi)
sp_z = np.cos(theta)

ax.plot_wireframe(sp_x, sp_y, sp_z, rstride=4, cstride=4, color='#d3d3d3', edgecolor='k', alpha=0.3)
ax.text(0, 0, 1.2, '|0⟩')
ax.text(0, 0, -1.2, '|1⟩')
ax.set_title("Qubit Bloch Sphere")

# expectation values
expect_x = np.concatenate([result1.expect[0], result2.expect[0], result3.expect[0]])
expect_y = np.concatenate([result1.expect[1], result2.expect[1], result3.expect[1]])
expect_z = np.concatenate([result1.expect[2], result2.expect[2], result3.expect[2]])

# plot expectation values
ax_x = fig.add_subplot(gs[0, 2:])
ax_x.plot(times_total, expect_x, label='X')
ax_x.set_title("⟨X⟩")
ax_x.set_xlabel("Time (ns)")
ax_x.set_ylabel("x")

ax_y = fig.add_subplot(gs[1, 2:])
ax_y.plot(times_total, expect_y, label='Y')
ax_y.set_title("⟨Y⟩")
ax_y.set_xlabel("Time (ns)")
ax_y.set_ylabel("y")

ax_z = fig.add_subplot(gs[2, 2:])
ax_z.plot(times_total, expect_z, label='Z')
ax_z.set_title("⟨Z⟩")
ax_z.set_xlabel("Time (ns)")
ax_z.set_ylabel("z")

# add vertical lines to mark pulse boundaries
for ax in [ax_x, ax_y, ax_z]:
    ax.axvline(x=pulse_time, color='r', linestyle='--', alpha=0.3)
    ax.axvline(x=pulse_time + free_time, color='r', linestyle='--', alpha=0.3)

plt.show()
