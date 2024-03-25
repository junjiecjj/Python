#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 19:11:17 2022

@author: jack
"""

import numpy as np

fft=np.fft.fft(arrayTemp,512)
fftshift=np.fft.fftshift(fft)
amp=abs(fftshift)/len(fft)
pha=np.angle(fftshift)
fre=np.fft.fftshift(np.fft.fftfreq(d=1,n=512))
 
import matplotlib.pyplot as plt

plt.figure()
plt.plot(fre,amp)
plt.xlabel('Frequency');plt.ylabel('Amplitude')

plt.figure()
plt.plot(fre,pha)
plt.xlabel('Frequency');plt.ylabel('Phase')