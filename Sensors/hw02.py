# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 15:55:38 2026

@author: brend
"""
import matplotlib.pyplot as plt
import numpy as np

def main():
    #p2()
    p3()
    pass

def p3a():
    # Parameters
    Vref = 10       # reference voltage
    R0 = 120        # base resistance of active gauge
    R_dummy = 120   # dummy gauge or opposite arm
    strain = np.linspace(0, 0.02, 11)  # 0 to 2% strain
    
    # Active gauge resistance
    R_active = R0 * (1 + strain)
    print(R_active)
    
    # Wheatstone bridge output (half-bridge)
    Vout = Vref * (R_dummy / (R_active + R_dummy) - 0.5)
    
    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(strain*100, Vout, 'o-', color='purple')
    plt.xlabel('Strain (%)')
    plt.ylabel('Bridge Output Vout (V)')
    plt.title('Wheatstone Bridge Output vs Strain')
    plt.grid(True)
    plt.show()
    
def p3():
    orig_r = np.array([350,441,553,700,879,1106,1393,1754,2209,3497])
    r_plus_2 = orig_r + 2
    r0 = 350 
    
    vout = orig_r / (orig_r + r0)
    vout_plus_2 = r_plus_2 / (r_plus_2 + r0)
    
    delta_vout = vout_plus_2 - vout
    
    sensitivity = delta_vout * 350 / 2
    
    k = np.array([1.006,1.26,1.58,2,2.51,3.16,3.98,5.01,6.31,9.997])
    
    # plot
    plt.figure(figsize=(8,5))
    plt.plot(k, sensitivity, marker='o', linestyle='-', color='blue', label='Sensitivity')
    
    plt.xlabel('Bridge Ratio K')
    plt.ylabel('Sensitivity [V/V]')
    plt.title('Quarter Bridge Sensitivity')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    

def p2():
    r_0 = 350                   # Ohms
    r_s = np.array([352, 441, 553, 700, 879, 1106, 1393, 1754, 2209, 3499])  # sensor resistances
    v_out = np.array([0.014, 0.575, 1.124, 1.667, 2.152, 2.596, 2.992, 3.336, 3.632, 4.091])  # measured output
    delta_r = r_s - r_0
    
    # Sensitivity calculation
    sensitivity = v_out / delta_r
    
    
    # Calculate k values
    k_values = r_s / r_0
    
    # Plot Sensitivity vs k
    plt.figure(figsize=(8,5))
    plt.plot(k_values, sensitivity, 'o-', color='purple')
    plt.xlabel('Bridge Ratio k')
    plt.ylabel('Sensitivity S (V/Ω)')
    plt.title('Wheatstone Bridge Sensitivity vs Bridge Ratio')
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
    

if __name__ == "__main__":
    main()