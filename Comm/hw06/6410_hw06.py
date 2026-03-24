# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:42:05 2026

@author: brend
"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    
    # Parameters
    N = 10        # Period of sine
    nx = 64       # Number of samples
    A = 1        # Amplitude of sine
    SNR_dB = 0    # Desired SNR in dB
    
    # Create the clean signal x[n]
    n = np.arange(nx)
    x_n = A * np.sin(2 * np.pi * n / N)
    
    # Generate additive uniform noise
    # Uniform noise range for 0dB SNR
    signal_power = np.mean(x_n**2)
    noise_power = signal_power / (10**(SNR_dB / 10))
    # Uniform noise range [-b, b] -> power = b^2 / 3
    b = np.sqrt(3 * noise_power)
    w = np.random.uniform(-b, b, nx)
    
    # Observed noisy signal
    y_n = x_n + w
    
    # Plot signals
    #---------------------------------------------
    plt.figure(figsize=(14, 14))
    # Subplot 1: Original signal
    plt.subplot(3, 1, 1)  # 3 rows, 1 column, plot 1
    plt.stem(n, x_n, use_line_collection=True)
    plt.title("Original Signal x[n]")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Subplot 2: Noise
    plt.subplot(3, 1, 2)  # 3 rows, 1 column, plot 2
    plt.stem(n, w, use_line_collection=True)
    plt.title("Noise w[n]")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Subplot 3: Noisy signal
    plt.subplot(3, 1, 3)  # 3 rows, 1 column, plot 3
    plt.stem(n, y_n, use_line_collection=True)
    plt.title("Noisy Signal y[n]")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.tight_layout()  # Adjust spacing so titles/labels don't overlap
    plt.show()
    
    # calculate autocorrelation
    r_yy = np.correlate(y_n, y_n, mode='full')
    lags = np.arange(-nx + 1, nx)
    
    plt.figure(figsize=(8, 4))
    plt.stem(np.arange(len(r_yy)), r_yy, use_line_collection=True)
    plt.title(r"Autocorrelation $R_{yy}$")
    plt.ylabel(r"$R_{yy}[n]$")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()