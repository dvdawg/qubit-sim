import numpy as np
import matplotlib.pyplot as plt

sigma = 1.0
A = 1.0

def my_pulse(t):
    return np.exp(-(t**2)/(2*sigma)) * A
def my_pulse_DRAG(t):
    return np.exp(-(t**2)/(2*sigma)) * 200 * (-t / (sigma**2))

def compute_power_spectrum(signal, dt):
    N = len(signal)                          # number of samples
    fft_result = np.fft.fft(signal)          # compute the FFT
    fft_shifted = np.fft.fftshift(fft_result)  # shift zero frequency to center

    freq = np.fft.fftfreq(N, d=dt)           # frequency bins
    freq_shifted = np.fft.fftshift(freq)

    power_spectrum = np.abs(fft_shifted)**2  # |FFT|^2

    return freq_shifted, power_spectrum

def main():
    # Time parameters
    t_min = -10
    t_max = 10
    dt = 0.001  # sampling interval
    t = np.arange(t_min, t_max, dt)

    # Define the pulse in time domain using the custom function
    # You can change sigma, f0, or the definition of the function entirely.
    pulse = my_pulse(t)

    # Compute the power spectrum
    freq, power_spec = compute_power_spectrum(pulse, dt)

    # Plot the time-domain pulse
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t, pulse, 'b-')
    plt.title("Time-Domain Pulse")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    # Plot the power spectrum
    plt.subplot(1, 2, 2)
    plt.plot(freq, power_spec, 'r-')
    plt.title("Power Spectrum")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
