import numpy as np
import matplotlib.pyplot as plt

omega_q = 5.0
omega_d = 4.9

def simulate_ramsey_time_sweep(free_time, phi=np.pi/2, omega_q=5.0, omega_d=4.9):
    # System parameters
    A = np.pi/24  # π/2 pulse amplitude
    pulse_time = 12  # Duration of π/2 pulses
    total_time = 2*pulse_time + free_time

    dt = 0.1
    state = np.array([1, 0], dtype=complex)

    # First pi/2 pulse
    for i in range(int(pulse_time/dt)):
        theta = A * dt
        R = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], 
                     [-1j*np.sin(theta/2), np.cos(theta/2)]])
        state = R @ state

    # Free evolution with phase accumulation
    delta = omega_q - omega_d
    for i in range(int(free_time/dt)):
        phase = delta * dt
        R = np.array([[np.exp(-1j*phase/2), 0], 
                     [0, np.exp(1j*phase/2)]])
        state = R @ state

    # Second pi/2 pulse with fixed phase
    for i in range(int(pulse_time/dt)):
        theta = A * dt
        R = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)*np.exp(-1j*phi)],
                     [-1j*np.sin(theta/2)*np.exp(1j*phi), np.cos(theta/2)]])
        state = R @ state

    # Calculate Z expectation value
    z_expect = np.abs(state[0])**2 - np.abs(state[1])**2
    return z_expect

free_times = np.linspace(0, 200, 200)  
z_values = [simulate_ramsey_time_sweep(t) for t in free_times]

z_rel = np.sin((omega_d - omega_q) * free_times)
plt.figure(figsize=(12, 8))

# z expect plot
plt.subplot(2, 1, 1)
plt.plot(free_times, z_values, 'b-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('time (ns)')
plt.ylabel('final z-expectation value')
plt.title('Ramsey interferometry varying free precession time')
plt.ylim(-1.1, 1.1)

# relative phase
plt.subplot(2, 1, 2)
plt.plot(free_times, z_rel, 'r-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('time (ns)')
plt.ylabel('final z-expectation value')
plt.title(f'relative phase {omega_d-omega_q} GHz graph with time')
plt.ylim(-1.1, 1.1)

# Show the plots
plt.tight_layout()
plt.show()

print("Done")