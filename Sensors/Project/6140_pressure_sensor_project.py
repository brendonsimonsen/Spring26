# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:59:31 2026

@author: brend
"""

import numpy as np
import matplotlib.pyplot as plt

# recorded data
temperature = np.array([22, 26.8, 32, 33, 34, 36.4, 37.3, 38.3])            # °C
vout = np.array([-1.3, -0.95, -0.88, -0.86, -0.82, -0.73, -0.72, -0.65])    # volts

# linear line of best fit
slope, intercept = np.polyfit(temperature, vout, 1)

# reference voltage
V0 = 5          # voltage supplying the bridge circuit

# temperature coefficient
alpha = slope / V0

print("Slope (dV/dT):", slope)
print("Temperature Coefficient (1/°C):", alpha)
print("Temperature Coefficient (ppm/°C):", alpha * 1e6)

# plot 
plt.scatter(temperature, vout, label="Measured Data")
plt.plot(temperature, slope*temperature + intercept, label="Linear Fit")

plt.xlabel("Temperature (°C)")
plt.ylabel("Bridge Output Voltage (V)")
plt.grid(True)
plt.legend()
plt.show()