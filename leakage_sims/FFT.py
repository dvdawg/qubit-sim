import numpy as np
import matplotlib.pyplot as plt

A = 1880
sigma = 1  
beta = 1  

t = np.linspace(-5*sigma, 5*sigma, num=100) 

Omega_t = A * np.exp(-t**2 / (2 * sigma))

dOmega_dt = (t / sigma**2) * Omega_t  
Omega_DRAG_t = Omega_t + 1j * A * beta * dOmega_dt

freqs = np.fft.fftfreq(t.size, d=(t[1] - t[0])) 
Omega_fft = np.fft.fft(Omega_t)  
Omega_DRAG_fft = np.fft.fft(Omega_DRAG_t)

freqs_shifted = np.fft.fftshift(freqs)
Omega_fft_shifted = np.fft.fftshift(Omega_fft)
Omega_DRAG_fft_shifted = np.fft.fftshift(Omega_DRAG_fft)

plt.figure(figsize=(10, 5))
plt.plot(freqs_shifted, np.abs(Omega_fft_shifted), label=r'$\Omega(t)$')
plt.plot(freqs_shifted, np.abs(Omega_DRAG_fft_shifted), label=r'$\Omega_{DRAG}(t)$')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT of Standard and DRAG Pulses")
plt.legend()
plt.grid()
plt.show()
