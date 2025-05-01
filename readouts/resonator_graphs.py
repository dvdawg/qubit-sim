import numpy as np
import matplotlib.pyplot as plt

delta_omega = np.linspace(-10, 10, 500) 

# Coupling rates in MHz
kappa_e = 1.0
kappa_1 = 1.0
kappa_2 = 1.0

# Internal loss rates to compare
kappa_i_values = [0.0, 0.9, 2.0]
colors = ['blue', 'orange', 'green']
labels = [f'κᵢ = {ki} MHz' for ki in kappa_i_values]

# --- S-parameter definitions ---
def s11_reflection(delta_omega, kappa_i, kappa_e):
    numerator = delta_omega - 1j * (kappa_i - kappa_e) / 2
    denominator = delta_omega - 1j * (kappa_i + kappa_e) / 2
    return numerator / denominator

def s21_hanger(delta_omega, kappa_i, kappa_e):
    kappa_total = kappa_i + kappa_e
    denominator = kappa_total / 2.0 + 1j * delta_omega
    return 1.0 - kappa_e / denominator

def s21_transmission(delta_omega, kappa_i, kappa_1, kappa_2):
    numerator = -1j * np.sqrt(kappa_1 * kappa_2) / 2
    denominator = delta_omega - 1j * (kappa_i + kappa_1 + kappa_2) / 2
    return numerator / denominator

# --- Plotting helper ---
def plot_resonator(fig, s_param_func, title, s_param_label, *args):
    ax_complex = fig.add_subplot(1, 3, 1)
    ax_phase = fig.add_subplot(1, 3, 2)
    ax_logmag = fig.add_subplot(1, 3, 3)

    fig.suptitle(title, fontsize=14)

    for i, kappa_i in enumerate(kappa_i_values):
        s_param = s_param_func(delta_omega, kappa_i, *args)
        color = colors[i]
        label = labels[i]

        ax_complex.plot(np.real(s_param), np.imag(s_param))
        ax_phase.plot(delta_omega, np.angle(s_param))
        ax_logmag.plot(delta_omega, 20 * np.log10(np.abs(s_param)))

    ax_complex.set_xlabel(f'Real')
    ax_complex.set_ylabel(f'Imag')
    ax_complex.set_xlim(-1.15, 1.15)
    ax_complex.set_ylim(-1.15, 1.15)
    ax_complex.grid(True)
    ax_complex.axhline(0, color='black', linewidth=0.5)
    ax_complex.axvline(0, color='black', linewidth=0.5)
    ax_complex.set_aspect('equal', adjustable='box')
    ax_complex.legend(fontsize='small')
    ax_complex.set_title('Complex Plane')

    ax_phase.set_xlabel('omega - omega_0 (MHz)')
    ax_phase.set_ylabel('Arg (rad)')
    ax_phase.set_ylim(-np.pi - 0.2, np.pi + 0.2)
    ax_phase.grid(True)
    ax_phase.set_xlim(-10, 10)
    ax_phase.set_title('Phase')

    ax_logmag.set_xlabel('omega - omega_0 (MHz)')
    ax_logmag.set_ylabel('Log Magn (dB)')
    ax_logmag.grid(True)
    ax_logmag.set_xlim(-10, 10)
    ax_logmag.set_title('Logarithmic Magnitude')

    return ax_logmag

# --- Create Figures: side-by-side layout ---
fig1 = plt.figure(figsize=(16, 4))
ax_logmag_ref = plot_resonator(fig1, s11_reflection, 'Reflection Resonator (S11)', 'S11', kappa_e)
ax_logmag_ref.set_ylim(-5, 0.5)

fig2 = plt.figure(figsize=(16, 4))
ax_logmag_han = plot_resonator(fig2, s21_hanger, 'Hanger Resonator (S21)', 'S21', kappa_e)
ax_logmag_han.set_ylim(-30, 0.5)

fig3 = plt.figure(figsize=(16, 4))
ax_logmag_tra = plot_resonator(fig3, s21_transmission, 'Transmission Resonator (S21)', 'S21', kappa_1, kappa_2)
ax_logmag_tra.set_ylim(-30, 0.5)

# --- Layout and show ---
for fig in [fig1, fig2, fig3]:
    fig.tight_layout(rect=[0, 0, 1, 0.9])

plt.show()
