import numpy as np
import matplotlib.pyplot as plt

def simulate_ramsey(phi, omega_q=5.0, omega_d=4.9, total_time=74):
    # System parameters
    A = np.pi/24  # π/2 pulse amplitude
    pulse_time = 12  # Duration of π/2 pulses
    free_time = 50   # Free evolution time
    
    # Time points for evolution
    dt = 0.1
    t = np.arange(0, total_time, dt)
    
    # Initial state (|0⟩)
    state = np.array([1, 0], dtype=complex)
    
    # First π/2 pulse
    for i in range(int(pulse_time/dt)):
        # Rotation matrix for π/2 pulse
        theta = A * dt
        R = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])
        state = R @ state
    
    # Free evolution with phase accumulation
    delta = omega_q - omega_d
    for i in range(int(free_time/dt)):
        # Phase evolution
        phase = delta * dt
        R = np.array([[np.exp(-1j*phase/2), 0], [0, np.exp(1j*phase/2)]])
        state = R @ state
    
    # Second π/2 pulse with variable phase
    for i in range(int(pulse_time/dt)):
        theta = A * dt
        R = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)*np.exp(-1j*phi)],
                     [-1j*np.sin(theta/2)*np.exp(1j*phi), np.cos(theta/2)]])
        state = R @ state
    
    # Calculate Z expectation value
    z_expect = np.abs(state[0])**2 - np.abs(state[1])**2
    return z_expect

# Calculate Z expectation for different phases
phases = np.linspace(0, 2*np.pi, 100)
z_values = [simulate_ramsey(phi) for phi in phases]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(phases, z_values, 'b-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('relative phase (rads)')
plt.ylabel('final z-expectation value')
plt.title('Ramsey interferometry')


plt.ylim(-1.1, 1.1)
plt.show()