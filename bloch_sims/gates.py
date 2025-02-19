import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmay, sigmaz, sesolve, Qobj
from matplotlib.gridspec import GridSpec

def plot_bloch(Hamiltonian, psi_0, t_list):
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 3, figure=fig, hspace=0.6, wspace=0.5)
    
    ax = fig.add_subplot(gs[:, :2], projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
 
    result = sesolve(Hamiltonian, psi_0, t_list)
    result_expect = sesolve(Hamiltonian, psi_0, t_list, [sigmax(), sigmay(), sigmaz()])
    
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
    
    ax_x = fig.add_subplot(gs[0, 2:])
    ax_x.plot(t_list, result_expect.expect[0], label='X')
    ax_x.set_title("⟨X⟩")
    ax_x.set_xlabel("Time (ns)")
    ax_x.set_ylabel("x")
    ax_y = fig.add_subplot(gs[1, 2:])
    ax_y.plot(t_list, result_expect.expect[1], label='Y')
    ax_y.set_title("⟨Y⟩")
    ax_y.set_xlabel("Time (ns)")
    ax_y.set_ylabel("y")
    ax_z = fig.add_subplot(gs[2, 2:])
    ax_z.plot(t_list, result_expect.expect[2], label='Z')
    ax_z.set_title("⟨Z⟩")
    ax_z.set_xlabel("Time (ns)")
    ax_z.set_ylabel("z")

    ax.set_title("Qubit Bloch Sphere")
    plt.show()

psi_0 = basis(2, 0)

t_list = np.linspace(0, 2 * np.pi, 100)

gate_hamiltonian = (1/np.sqrt(2)) * Qobj([[1, 0], [0, 1j]])
plot_bloch(gate_hamiltonian, psi_0, t_list)
