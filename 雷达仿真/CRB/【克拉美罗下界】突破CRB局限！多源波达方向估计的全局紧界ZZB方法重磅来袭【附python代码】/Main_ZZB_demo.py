# -*- coding: utf-8 -*-
# @Author: wentao.yu
# @Date:   2025-03-16 17:40:21
# @Last Modified by:   wentao.yu
# @Last Modified time: 2025-03-20 13:04:26

import numpy as np
import matplotlib.pyplot as plt
from ZZB_DOAs import ZZB_DOAs

# Simulation parameters
M = 20  # The number of antennas/sensors
lambda_ = 2  # The wave length
Array = np.arange(M) * lambda_ / 2  # ULA
SNR = np.concatenate((np.arange(-40, -9.5, 0.5), np.arange(-8, 21, 2)))  # SNR in dB
T = 40  # The number of snapshots
K = 5  # The number of single-antenna sources
num_MC = 1000  # The number of Monte Carlo trials
vartheta_min = -60  # Minimum value of DOAs range (in deg)
vartheta_max = 60  # Maximum value of DOAs range (in deg)
resolution = 10  # Minimum separation between DOAs (in degrees)

# Calculate bounds
RAPB, RCRB, RZZB_Generalized, RZZB = ZZB_DOAs(M, lambda_, Array, SNR, T, K, num_MC, vartheta_min, vartheta_max, resolution)

# Plotting
plt.figure()
plt.semilogy(SNR, RAPB, 'k-.', linewidth=1.5, label='APB')
plt.semilogy(SNR, RCRB, 'b--', linewidth=1.5, label='CRB')
plt.semilogy(SNR, RZZB_Generalized, 'g--', linewidth=1.5, label='Generalized ZZB')
plt.semilogy(SNR, RZZB, 'r', linewidth=1.5, label='ZZB')
plt.legend()
plt.grid(True)
plt.axis([-40, 20, 1e-2, 1e2])
plt.xlabel('SNR (dB)')
plt.ylabel('RMSE (deg)')
plt.show()