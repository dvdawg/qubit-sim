import numpy as np
import matplotlib.pyplot as plt

def F1(Omega, A, omega):
    """ Analytic FT of A exp(-t^2/(2*omega)). """
    return A * np.sqrt(2.0*np.pi*omega) * np.exp(-0.5 * omega * Omega**2)

def F2(Omega, A, omega, beta):
    """ Analytic FT of f2(t). """
    # Common Gaussian factor:
    gauss = np.sqrt(2.0*np.pi*omega) * np.exp(-0.5 * omega * Omega**2)
    return A*gauss + A*beta*(Omega/omega)*gauss

# Make a frequency (Omega) grid for plotting
Omega_vals = np.linspace(-10, 10, 1000)  # for example

# Compute the transforms
F1_vals = F1(Omega_vals, A=1.0, omega=1.0)
F2_vals = F2(Omega_vals, A=1.0, omega=1.0, beta=1.0)

# Plot
plt.figure(figsize=(8,6))
plt.plot(Omega_vals, np.abs(F1_vals))
plt.plot(Omega_vals, np.abs(F2_vals))
plt.title("Analytic Fourier Transforms")
plt.xlabel(r'$\Omega$')
plt.ylabel('Magnitude')
plt.legend()
plt.tight_layout()
plt.show()
