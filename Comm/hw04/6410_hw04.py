# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:44:06 2026

@author: brend
"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    p1()
    pass

def p1():
    # constants
    f1 = 1e3        # hertz
    fc = 20e3       # hertz
    B = 5           # beta
    ns = 256        # samples
    A = 1           # amplitude (set to 1 for now)
    
    # fs = n * f1
    fs = 100e3      
    fs = ns *f1     # sampling frequency
    n = np.arange(ns)
    t = n / fs
    
    # calculate omega values
    w1 = 2*np.pi*f1
    wLO = 2*np.pi*fc
    
    # plug into given equation
    x_n = A * np.cos(wLO * t + B * np.sin(w1 * t))
    #x_n = A * np.cos(2 * np.pi * fc * t + B * np.sin(2 * np.pi * f1 * t))
    
    # plot
    plt.figure(figsize=(12, 6))
    plt.plot(n, x_n)
    plt.title(r'Reproduced FM Signal')
    plt.xlabel('Sample Index ($n$)')
    plt.ylabel('$x(n)$')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Frequency domain
    
    # calc FFT
    X_f = np.fft.fft(x_n)
    freqs = np.fft.fftfreq(ns, 1/fs)
    
    X_mag = np.abs(X_f)
    
    # dB conversion
    X_db = 20 * np.log10(X_mag/20)
    # normalize to 30 dB as shown in HW
    X_db_norm = X_db - np.max(X_db) + 30
    
    # plotting 
    plt.figure(figsize=(12, 6))
    plt.stem(freqs / 1000, X_db_norm)
    plt.title(f'Frequency Domain')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.xlim(-40, 40) 
    plt.ylim(0.1, 31)
    plt.show()

if __name__ == "__main__":
    main()