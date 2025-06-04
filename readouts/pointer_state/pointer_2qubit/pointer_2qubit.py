import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import seaborn as sns

def create_hamiltonian(g1, g2, omega_r, omega_q1, omega_q2):
    """
    Create Hamiltonian for 2 qubits coupled to a resonator
    H = ω_r a^†a + ω_q1 σ^z_1/2 + ω_q2 σ^z_2/2 + g1(a^†σ^-_1 + aσ^+_1) + g2(a^†σ^-_2 + aσ^+_2)
    """
    # Define operators
    a = destroy(3)  # resonator annihilation operator (increased dimension for better pointer states)
    sm1 = tensor(sigmam(), qeye(2), qeye(2))  # qubit 1 lowering operator
    sm2 = tensor(qeye(2), sigmam(), qeye(2))  # qubit 2 lowering operator
    sz1 = tensor(sigmaz(), qeye(2), qeye(2))  # qubit 1 sigma z
    sz2 = tensor(qeye(2), sigmaz(), qeye(2))  # qubit 2 sigma z
    
    # Build Hamiltonian
    H = (omega_r * a.dag() * a + 
         omega_q1/2 * sz1 + 
         omega_q2/2 * sz2 + 
         g1 * (a.dag() * sm1 + a * sm1.dag()) + 
         g2 * (a.dag() * sm2 + a * sm2.dag()))
    
    return H

def create_collapse_operators(kappa, gamma1, gamma2):
    """
    Create collapse operators for dissipation
    kappa: resonator decay rate
    gamma1, gamma2: qubit decay rates
    """
    a = destroy(3)
    sm1 = tensor(sigmam(), qeye(2), qeye(2))
    sm2 = tensor(qeye(2), sigmam(), qeye(2))
    
    c_ops = [
        np.sqrt(kappa) * a,  # resonator decay
        np.sqrt(gamma1) * sm1,  # qubit 1 decay
        np.sqrt(gamma2) * sm2,  # qubit 2 decay
    ]
    
    return c_ops

def create_initial_state(qubit_state):
    """
    Create initial state for the system
    qubit_state: '00', '01', '10', or '11'
    """
    # Create basis states
    g = basis(2, 0)  # ground state
    e = basis(2, 1)  # excited state
    vac = basis(3, 0)  # vacuum state of resonator
    
    # Create initial state based on qubit_state
    if qubit_state == '00':
        psi0 = tensor(vac, g, g)
    elif qubit_state == '01':
        psi0 = tensor(vac, g, e)
    elif qubit_state == '10':
        psi0 = tensor(vac, e, g)
    elif qubit_state == '11':
        psi0 = tensor(vac, e, e)
    else:
        raise ValueError("qubit_state must be '00', '01', '10', or '11'")
    
    return psi0

def simulate_trajectory(H, psi0, times, c_ops):
    """
    Simulate the time evolution of the system
    """
    # Solve master equation
    result = mesolve(H, psi0, times, c_ops)
    
    # Calculate expectation values
    a = destroy(3)
    sm1 = tensor(sigmam(), qeye(2), qeye(2))
    sm2 = tensor(qeye(2), sigmam(), qeye(2))
    sz1 = tensor(sigmaz(), qeye(2), qeye(2))
    sz2 = tensor(qeye(2), sigmaz(), qeye(2))
    
    # Get expectation values
    a_expect = [expect(a, state) for state in result.states]
    sm1_expect = [expect(sm1, state) for state in result.states]
    sm2_expect = [expect(sm2, state) for state in result.states]
    sz1_expect = [expect(sz1, state) for state in result.states]
    sz2_expect = [expect(sz2, state) for state in result.states]
    
    return times, a_expect, sm1_expect, sm2_expect, sz1_expect, sz2_expect

def plot_trajectories(times, a_expect, sm1_expect, sm2_expect, sz1_expect, sz2_expect, qubit_state):
    """
    Plot the trajectories
    """
    plt.figure(figsize=(15, 10))
    
    # Plot resonator
    plt.subplot(3, 2, 1)
    plt.plot(times, np.real(a_expect), label='Re')
    plt.plot(times, np.imag(a_expect), label='Im')
    plt.title(f'Resonator (Initial state: |{qubit_state}⟩)')
    plt.xlabel('Time')
    plt.ylabel('⟨a⟩')
    plt.legend()
    
    # Plot qubit 1
    plt.subplot(3, 2, 3)
    plt.plot(times, np.real(sm1_expect), label='Re')
    plt.plot(times, np.imag(sm1_expect), label='Im')
    plt.title('Qubit 1')
    plt.xlabel('Time')
    plt.ylabel('⟨σ⁻⟩')
    plt.legend()
    
    # Plot qubit 2
    plt.subplot(3, 2, 5)
    plt.plot(times, np.real(sm2_expect), label='Re')
    plt.plot(times, np.imag(sm2_expect), label='Im')
    plt.title('Qubit 2')
    plt.xlabel('Time')
    plt.ylabel('⟨σ⁻⟩')
    plt.legend()
    
    # Plot qubit 1 z-component
    plt.subplot(3, 2, 4)
    plt.plot(times, sz1_expect)
    plt.title('Qubit 1 Z')
    plt.xlabel('Time')
    plt.ylabel('⟨σᶻ⟩')
    
    # Plot qubit 2 z-component
    plt.subplot(3, 2, 6)
    plt.plot(times, sz2_expect)
    plt.title('Qubit 2 Z')
    plt.xlabel('Time')
    plt.ylabel('⟨σᶻ⟩')
    
    plt.tight_layout()
    plt.show()

def main():
    # System parameters
    g1 = 0.05  # coupling strength qubit 1
    g2 = 0.05  # coupling strength qubit 2
    omega_r = 1.0  # resonator frequency
    omega_q1 = 1.0  # qubit 1 frequency
    omega_q2 = 1.0  # qubit 2 frequency
    
    # Dissipation parameters
    kappa = 0.01  # resonator decay rate
    gamma1 = 0.001  # qubit 1 decay rate
    gamma2 = 0.001  # qubit 2 decay rate
    
    # Time points
    times = np.linspace(0, 100, 1000)
    
    # Create Hamiltonian and collapse operators
    H = create_hamiltonian(g1, g2, omega_r, omega_q1, omega_q2)
    c_ops = create_collapse_operators(kappa, gamma1, gamma2)
    
    # Simulate for each initial state
    for qubit_state in ['00', '01', '10', '11']:
        # Create initial state
        psi0 = create_initial_state(qubit_state)
        
        # Simulate trajectory
        times, a_expect, sm1_expect, sm2_expect, sz1_expect, sz2_expect = simulate_trajectory(H, psi0, times, c_ops)
        
        # Plot results
        plot_trajectories(times, a_expect, sm1_expect, sm2_expect, sz1_expect, sz2_expect, qubit_state)

if __name__ == "__main__":
    main()
