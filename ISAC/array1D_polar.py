#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
import scipy.constants as CONSTANTS

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

# https://www.bilibili.com/read/cv17750843/
#%% Basic Electromagnetic Parameters
Frequency = 10e9
Lightspeed = CONSTANTS.c
Wavelength = Lightspeed/Frequency
Wavenumber = 2 * np.pi/Wavelength

### Array Parameters
N = 12
theta0 = math.radians(30)

### ArrayFactor Samping
theta = np.arange(0, 2*np.pi, 0.001)
# theta = np.arange(0, np.pi, 0.001)
psi = np.pi * (np.sin(theta) - np.sin(theta0))

P = np.abs(np.sin(N * psi/2)/np.sin(psi/2))/N

peaks, _ =  scipy.signal.find_peaks(P)

### 画图
fig, axs = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': 'polar'})
axs.plot(theta, P, color='b', linestyle='-', lw = 3, label='',  )
# axs.plot(theta[peaks], P[peaks], linestyle='', marker = 'o', color='r', markersize = 12)

plt.show()



#%% Basic Electromagnetic Parameters
Frequency = 10e9
Lightspeed = CONSTANTS.c
Wavelength = Lightspeed/Frequency
Wavenumber = 2 * np.pi/Wavelength

### Array Parameters
N = 12
A = np.ones(N)
theta0 = np.pi/6
# wt = A * np.ones(N, )   # 权重向量
wt = A * np.exp(-1j * (np.pi * np.arange(N) * np.sin(theta0)) )
alpha = np.zeros(N, )


### ArrayFactor Samping
Ns = 1000                  # Sampling number
theta = np.linspace(0, 2*np.pi, Ns)
Ptheta = np.zeros(Ns, )
mini_a = 1e-5
for num in range(Ns):
    # rad = math.radians(theta[num])
    Atheta = np.exp(-1j * (np.pi * (np.arange(N) + 1) * np.sin(theta[num])) + alpha )  # 导向/方向矢量
    Ptheta[num] = np.abs(wt @ Atheta.T.conjugate()) + mini_a
    # Ptheta[num] = np.abs(np.sum(wt * Atheta.T.conjugate())) + mini_a

# Ptheta = 20 * np.log10(Ptheta)
Ptheta /= N
peaks, _ =  scipy.signal.find_peaks(Ptheta)



### 画图
fig, axs = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': 'polar'})
axs.plot(theta, Ptheta, color='b', linestyle='-', lw = 3, label='',  )
axs.plot(theta[peaks], Ptheta[peaks], linestyle='', marker = 'o', color='r', markersize = 12)


plt.show()



#%% https://blog.csdn.net/qq_23176133/article/details/120056777

















































