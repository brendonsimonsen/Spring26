# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 16:53:09 2026

@author: brend
"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    p1()
    pass

def p1():
    fc = 20e3                   # hertz
    fs = 100e3                  # hertz
    fm = 1e3                    # message frequency
    T = 1/fm                    # period
    T10 = T*10                  # go for 10 periods 
    t = np.arange(0, T10, 1/fs)
    
    # band-limited square wave
    m = np.zeros_like(t)        # array of 0's size of t
    K = 5                       # 5 odd harmonics; 1,3,5,7,9
    
    # k = 1,3,5,7,9
    for k in range(1, 2*K, 2):
        # equation for square wave build from Fourier series
        m += (4/(np.pi*k))*np.sin(2*np.pi*k*fm*t)
    
    # normalize square wave
    m = m / np.max(np.abs(m))
    
    
    plt.figure()
    plt.plot(t*1000, m)
    plt.xlim(0, 3)     # show 3 ms (about 3 periods of a 1 kHz wave)
    plt.xlabel("Time (ms)")
    plt.ylabel("m(t)")
    plt.title("Normalized Band-limited 1 kHz square wave")
    plt.grid()
    plt.show()
    
    # create carrier signal
    c = np.cos(2*np.pi*fc*t)
    
    # implement balanced modulator
    s = m * c
    
    # plot time domain
    plt.figure()
    plt.plot(t*1000, s)
    plt.xlim(0, 2)   # zoom in (carrier is fast)
    plt.xlabel("Time (ms)")
    plt.ylabel("s(t)")
    plt.title("DSB-SC signal (Time-Domain)")
    plt.grid()
    plt.show()
    
    # compute FFT
    N = len(s)          # number of samples
    # compute FFT and shift zero frequency to center
    s_freq = np.fft.fftshift(np.fft.fft(s))  
    # corresponding frequency vector
    frequency = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    
    # plot in freqency domain
    plt.figure()
    plt.plot(frequency/1000, np.abs(s_freq)/N)  # divide by N to normalize
                                                # frequency/1000 easier plot
    plt.xlim(-40, 40)               # -40 to 40 kHz
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("|S(f)|")
    plt.title("DSB-SC Signal Spectrum")
    plt.grid()
    plt.show()
    

if __name__ == "__main__":
    main()