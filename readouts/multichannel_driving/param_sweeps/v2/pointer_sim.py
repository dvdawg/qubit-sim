# pointer_sim.py

import numpy as np
from scipy.optimize import minimize

# ─── Measurement parameters ───────────────────────────────────────────────────
eta = 0.5          # total measurement efficiency (0 < η ≤ 1)
tau = 200       # integration time window (200 ns)

# ─── Dispersive shifts ────────────────────────────────────────────────────────
def compute_chis(g_1, g_2, delta_r1, delta_r2):
    """Compute χ₁ = g₁²/Δ₁ and χ₂ = g₂²/Δ₂."""
    chi_1 = g_1**2 / delta_r1
    chi_2 = g_2**2 / delta_r2
    return chi_1, chi_2

# ─── Steady-state pointer amplitude ───────────────────────────────────────────
def calculate_steady_state(sigma_z1, sigma_z2, params,
                           chi_1, chi_2,
                           delta_r1, delta_r2,
                           kappa, g_1, g_2, delta_resonator):
    """
    Return α_ss for the resonator given qubit σ_z’s and drive params.
    """
    Omega_q1_mag, phi_q1, Omega_q2_mag, phi_q2 = params
    Omega_q1 = Omega_q1_mag * np.exp(1j * phi_q1)
    Omega_q2 = Omega_q2_mag * np.exp(1j * phi_q2)
    Omega_r = 0.0

    epsilon = (1j * Omega_r
               - (Omega_q1 * chi_1 * sigma_z1 / g_1
                  + Omega_q2 * chi_2 * sigma_z2 / g_2))
    delta_eff = chi_1 * sigma_z1 + chi_2 * sigma_z2

    return epsilon / (1j * kappa/2 + delta_eff + delta_resonator)

# ─── Instantaneous (steady-state) SNR ─────────────────────────────────────────
def calculate_snr(state1, state2, params,
                  chi_1, chi_2,
                  delta_r1, delta_r2,
                  kappa, g_1, g_2, delta_resonator,
                  n_avg=1):
    """
    |α₁–α₂| / noise, where noise = √(n₁ + n₂)/√n_avg.
    """
    α1 = calculate_steady_state(*state1, params,
                                chi_1, chi_2,
                                delta_r1, delta_r2,
                                kappa, g_1, g_2, delta_resonator)
    α2 = calculate_steady_state(*state2, params,
                                chi_1, chi_2,
                                delta_r1, delta_r2,
                                kappa, g_1, g_2, delta_resonator)

    return np.abs(α1 - α2) * np.sqrt(2 * kappa * tau)

# # ─── Time-integrated SNR over τ ────────────────────────────────────────────────
# def integrated_snr(state1, state2, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator):
#     """
#     SNR(τ) = 4 η κ τ · [separation/noise] · (1 – e^{-κτ/2})²
#     """
#     ss_snr = calculate_snr(state1, state2, params,
#                             chi_1, chi_2,
#                             delta_r1, delta_r2,
#                             kappa, g_1, g_2, delta_resonator)
#     ringup = (1 - np.exp(-kappa * tau / 2))**2
#     return 4 * eta * kappa * tau * ss_snr * ringup

# ─── Objective: maximize worst-case integrated SNR ───────────────────────────
def objective_function(params, state_pairs, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator):
    snrs = [
        calculate_snr(s1, s2, params, chi_1, chi_2, delta_r1, delta_r2, kappa, g_1, g_2, delta_resonator)
        for s1, s2 in state_pairs
    ]
    return -min(snrs)

# ─── Wrapper: find optimal drive params for given device params ──────────────
def optimize_parameters(delta_r1, delta_r2,
                        g_1, g_2,
                        kappa, delta_resonator):
    chi_1, chi_2 = compute_chis(g_1, g_2, delta_r1, delta_r2)

    # Pairs of logical states whose pointer separation we care about
    state_pairs = [
        ((-1, -1), (1,  1)),
        ((-1,  1), (1, -1)),
        ((-1, -1), (1, -1)),
        ((-1,  1), (1,  1))
    ]

    # Initial guess [Ωq1_mag, φq1, Ωq2_mag, φq2]
    initial = [3.0, np.pi, 2.0, np.pi/2]
    bounds  = [(0.001, 2.0),
               (0, 2*np.pi),
               (0.001, 2.0),
               (0, 2*np.pi)]

    res = minimize(
        objective_function,
        initial,
        args=(state_pairs,
              chi_1, chi_2,
              delta_r1, delta_r2,
              kappa, g_1, g_2, delta_resonator),
        bounds=bounds
    )

    opt = res.x
    final_snrs = [
        calculate_snr(s1, s2, opt,
                       chi_1, chi_2,
                       delta_r1, delta_r2,
                       kappa, g_1, g_2, delta_resonator)
        for s1, s2 in state_pairs
    ]

    return {
        'delta_r1': delta_r1,
        'delta_r2': delta_r2,
        'g_1'      : g_1,
        'g_2'      : g_2,
        'kappa'    : kappa,
        'delta_resonator': delta_resonator,
        'Omega_q1_mag': opt[0],
        'phi_q1'     : opt[1],
        'Omega_q2_mag': opt[2],
        'phi_q2'     : opt[3],
        'min_snr'    : np.min(final_snrs),
        'avg_snr'    : np.mean(final_snrs),
        'max_snr'    : np.max(final_snrs),
        'std_snr'    : np.std(final_snrs)
    }
