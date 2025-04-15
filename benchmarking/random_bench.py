import numpy as np
from qutip import *
import matplotlib.pyplot as plt

omega_d = 2 * np.pi * 5.0
omega_q = 2 * np.pi * 5.0
alpha = -2 * np.pi * 0.2 
Omega = 1.9 * np.pi * 0.03
T_pi = np.pi / (2 * Omega)
duration = 20
tlist = np.linspace(0, duration, 1000)

pulse_width = 50