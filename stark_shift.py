import numpy as np
import matplotlib.pyplot as plt

omega_delta_ratio = np.linspace(0, 200, 500) 

Delta = 1.0 

Omega = omega_delta_ratio * Delta

shift_minus_sqrt = np.sqrt(Delta**2 + Omega**2) - np.abs(Delta)
shift_neg_plus_sqrt = -np.sqrt(Delta**2 + Omega**2) + np.abs(Delta)

plt.figure(figsize=(10, 6))
plt.plot(omega_delta_ratio, shift_minus_sqrt, label=r"$+|\Delta| - \sqrt{\Delta^2 + \Omega^2}$", color="green")
plt.plot(omega_delta_ratio, shift_neg_plus_sqrt, label=r"$-|\Delta| + \sqrt{\Delta^2 + \Omega^2}$", color="orange")

plt.xlabel(r"$\frac{\Omega}{\Delta}$", fontsize=14)
plt.ylabel("Frequency Shift", fontsize=14)
plt.title("Frequency Shift with respect to $\\frac{\\Omega}{\\Delta}$", fontsize=16)
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.legend(fontsize=12)
plt.grid(alpha=0.5)

plt.tight_layout()
plt.show()
