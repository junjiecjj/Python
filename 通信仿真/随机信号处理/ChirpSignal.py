#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 00:41:25 2025

@author: jack
"""

import numpy as np
from matplotlib.pyplot import tight_layout
from scipy.signal import chirp, square
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt
N, T = 1000, 0.01  # number of samples and sampling interval for 10 s signal
t = np.arange(N) * T  # timestamps
x_lin = chirp(t, f0 = 6, f1 = 1, t1 = 10, method='linear')
fg0, ax0 = plt.subplots()
ax0.set_title(r"Linear Chirp from $f(0)=6\,$Hz to $f(10)=1\,$Hz")
ax0.set_xlabel( "Time $t$ in Seconds", )
ax0.set_ylabel( r"Amplitude $x_{lin}(t)$")

ax0.plot(t, x_lin)
plt.show()



#%%
def chirp_signal(t, f0, t1, f1, phase = 0):
    t0 = t[0]
    T = t1 - t0
    k = (f1-f0)/T
    g = np.cos(2 * np.pi * (k/2 * t + f0)*t + phase)
    return g

fs = 500
t = np.arange(0, 1, 1/fs)
f0 = 1
f1 = fs/20
g = chirp_signal(t, f0, 1, f1)
fg0, ax0 = plt.subplots()
ax0.set_title(r"Linear Chirp from $f(0)=1\,$Hz to $f(1)=25\,$Hz")
ax0.set_xlabel( "Time $t$ in Seconds", )
ax0.set_ylabel( r"Amplitude $x_{lin}(t)$")

ax0.plot(t, g)
plt.show()












