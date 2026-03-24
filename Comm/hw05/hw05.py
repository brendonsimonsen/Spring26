# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:30:36 2026

@author: brend
"""

import numpy as np
import matplotlib.pyplot as plt

def main():
    p1()
    p2()
    p3()

def p1():
    x = np.linspace(-5, 7, 800)
    y = (1/np.sqrt(2*np.pi)) * np.exp(-(x-2)**2 / 2)
    
    # plot
    plt.plot(x, y, label="N(2, 1)")
    plt.title("Normal (Gaussian) Distribution")
    plt.xlabel("x")
    plt.ylabel("fx(x)")
    plt.legend()
    plt.grid()
    
    plt.show()
    
    # numerically determine <x>
    moment_1 = np.trapz(x * y, x)
    
    # numerically determine <x^2>
    moment_2 = np.trapz(x**2 * y, x)
    
    # print values
    print("Normal Distribution values:")
    print("First moment:", moment_1)
    print("Second moment:", moment_2)
    
def p2():
    x = np.linspace(0, 2*np.pi, 500)
    y = np.ones_like(x) * (1 / (2 * np.pi))
    
    # plot
    plt.plot(x, y, label="[0, 2π]")
    plt.title("Uniform Distribution")
    plt.xlabel("x")
    plt.ylabel("fx(x)")
    plt.ylim(0, 1/(2*np.pi) + 0.1)  # small padding on y-axis
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # numerically determine <x>
    moment_1 = np.trapz(x * y, x)
    
    # numerically determine <x^2>
    moment_2 = np.trapz(x**2 * y, x)
    
    # print values
    print("Uniform Distribution values:")
    print("First moment:", moment_1)
    print("Second moment:", moment_2)
    
def p3():
    x = np.linspace(0, 5, 800)
    alpha = 2
    y = (2*x / alpha) * np.exp(-x**2 / alpha)
    y1 = (x / alpha**2) * np.exp(-x**2/(2*alpha**2))
    
    # plot
    plt.plot(x, y, label="[0,∞]")
    plt.title("Rayleigh Distribution")
    plt.xlabel("x")
    plt.ylabel("fx(x)")
    plt.legend()
    plt.grid()
    plt.show()
    
    # numerically determine <x>
    moment_1 = np.trapz(x * y, x)
    
    # numerically determine <x^2>
    moment_2 = np.trapz(x**2 * y, x)
    
    # print values
    print("Rayleigh Distribution values:")
    print("First moment:", moment_1)
    print("Second moment:", moment_2)

if __name__ == "__main__":
    main()