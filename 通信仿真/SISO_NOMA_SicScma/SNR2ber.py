#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 17:33:57 2024

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate



Joint_fastfading_6u_wo_powdiv = np.array([[0.00, 1.00000000, 0.37083984, 0.66777344, 0.00000000, 8.333],
                                        [1.00, 1.00000000, 0.36387370, 0.66738281, 0.00000000, 8.333],
                                        [2.00, 1.00000000, 0.35673828, 0.65824219, 0.00000000, 8.333],
                                        [3.00, 1.00000000, 0.34981120, 0.65503906, 0.00000000, 8.333],
                                        [4.00, 1.00000000, 0.34153646, 0.64300781, 0.00000000, 8.333],
                                        [5.00, 1.00000000, 0.33445964, 0.63210937, 0.00000000, 8.333],
                                        [6.00, 1.00000000, 0.32084635, 0.62257812, 0.00000000, 8.333],
                                        [7.00, 1.00000000, 0.31232422, 0.61121094, 0.00000000, 8.333],
                                        [8.00, 1.00000000, 0.29671224, 0.58941406, 0.00000000, 8.333],
                                        [9.00, 1.00000000, 0.28407552, 0.56863281, 0.00000000, 8.333],
                                        [10.00, 1.00000000, 0.26832031, 0.53761719, 0.00000000, 8.333],
                                        [11.00, 1.00000000, 0.24811198, 0.50742188, 0.00000000, 8.333],
                                        [12.00, 1.00000000, 0.22247396, 0.46437500, 0.00000000, 8.333],
                                        [13.00, 1.00000000, 0.18227865, 0.40273437, 0.00000000, 8.333],
                                        [13.40, 1.00000000, 0.15913411, 0.36027344, 0.00000000, 8.333],
                                        [13.80, 0.73529412, 0.09713446, 0.22865924, 0.00000000, 7.326],
                                        [14.00, 0.15875527, 0.01760285, 0.04270916, 0.00000000, 4.055],
                                        [14.20, 0.01456171, 0.00159719, 0.00378404, 0.00000000, 2.329],
                                        [14.4, 0.0006996677, 0.0000789176, 0.000203, 0.00000000, 1.329],
                                        ])

## 不等功率分配, 3倍等间隔
sic_fastfading_6u_w_powdiv_3 = np.array([[0.00, 1.00000000, 0.39220703, 0.73699219, 0.00000000, 50.000],
                                        [1.00, 1.00000000, 0.38479167, 0.73953125, 0.00000000, 50.000],
                                        [2.00, 1.00000000, 0.37159505, 0.73445313, 0.00000000, 50.000],
                                        [3.00, 0.99673203, 0.36057496, 0.72422641, 0.00000000, 49.980],
                                        [4.00, 0.97756410, 0.34500200, 0.72581130, 0.00000000, 49.359],
                                        [5.00, 0.91818182, 0.32346117, 0.71590909, 0.00000000, 47.548],
                                        [6.00, 0.84444444, 0.30217014, 0.70172526, 0.00000000, 44.797],
                                        [7.00, 0.83611111, 0.29174262, 0.69768880, 0.00000000, 43.558],
                                        [8.00, 0.83333333, 0.27942708, 0.68730469, 0.00000000, 42.947],
                                        [9.00, 0.81451613, 0.26494246, 0.67990801, 0.00000000, 42.341],
                                        [10.00, 0.72463768, 0.23678574, 0.65800498, 0.00000000, 39.053],
                                        [15.00, 0.53191489, 0.15652704, 0.56025598, 0.00000000, 30.784],
                                        [20.00, 0.34597701, 0.08153287, 0.38201778, 0.00000000, 21.732],
                                        [25.00, 0.16393443, 0.01926016, 0.11540727, 0.00000000, 12.407],
                                        [26.00, 0.07911392, 0.00625134, 0.03750804, 0.00000000, 9.472],
                                        [27, 0.0017472267, 0.0001057433, 0.000640, 0.00000000, 5.33],
                                        ])

f1 = interpolate.interp1d(Joint_fastfading_6u_wo_powdiv[:,0], Joint_fastfading_6u_wo_powdiv[:,2])
snr1 = np.arange(0, 14.5, 0.1)
ber1 = f1(snr1)
snrber1 = np.vstack((snr1, ber1)).T

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
plt.semilogy(Joint_fastfading_6u_wo_powdiv[:,0], Joint_fastfading_6u_wo_powdiv[:,2], 'o', snr1, ber1, '-')
plt.show()




f2 = interpolate.interp1d(sic_fastfading_6u_w_powdiv_3[:,0], sic_fastfading_6u_w_powdiv_3[:,2])
snr2 = np.arange(0, 27.1, 0.1)
ber2 = f2(snr2)
snrber2 = np.vstack((snr2, ber2)).T

fig, axs = plt.subplots(1, 1, figsize=(6, 4), constrained_layout = True)
plt.semilogy(sic_fastfading_6u_w_powdiv_3[:,0], sic_fastfading_6u_w_powdiv_3[:,2], 'o', snr2, ber2, '-')
plt.show()



















































































