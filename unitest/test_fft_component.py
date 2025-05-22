import torch
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 480  # Signal length
sampling_rate = 480  # Sampling rate (assume 480 Hz)

# Compute FFT frequency axis
freqs = torch.fft.fftfreq(L, d=1/sampling_rate)

# print("this is freqs", freqs)
# quit()

# Select some frequency indices (mix of low and high frequencies)
selected_indices = torch.tensor([-1, -2, -3, -4, -5])  

# Generate time axis
t = np.arange(L) / sampling_rate  # Normalize time axis (seconds)

# Plot setup
plt.figure(figsize=(10, 5))

# Process each selected frequency separately
for idx in selected_indices:
    # Initialize frequency-domain signal (all components zero)
    fft_signal = torch.zeros(L, dtype=torch.complex64)
    fft_signal[idx] = 1 + 0j  # Set only one frequency component to 1

    # Compute IFFT to get time-domain signal
    time_signal = torch.fft.ifft(fft_signal).real  # Take only real part

    # Plot current frequency component waveform
    plt.plot(t, time_signal.numpy(), label=f"Freq {freqs[idx]:.1f} Hz")

# Add legend and labels
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("IFFT of Selected Frequency Components (Separated)")
plt.legend()
plt.grid()

# Save the image
plt.savefig("./figs/ifft_waveform_separated.png", dpi=300)

