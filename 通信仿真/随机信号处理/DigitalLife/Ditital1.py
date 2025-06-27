#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 22:07:29 2025

@author: jack
"""


import time
import numpy as np
import scipy
import matplotlib.pyplot as plt


fig, axes = plt.subplots(1, 1, figsize = (10, 8), facecolor='k',  constrained_layout = True)
# figure(facecolor='k', newfig=False)
# axes(position=[0,0,1,1], aspect='equal')
# axes([0,400,0,400])
n = 2e4
i = np.arange(0, n, dtype='int')
sp1 = axes.scatter(i, i, s=1, facecolor='w', edgecolor=None, alpha=0.4)
t = 0
x = np.mod(i, 100)
y = np.floor(i/100)
k = x/4 - 12.5
e = y/9 + 5
o = np.linalg.norm(np.row_stack((k,e)), axis=0)/9
for _ in range(100):
    t = t + np.pi/90
    q = x + 99 + np.tan(1./k) + o*k*(np.cos(e*9)/4 + np.cos(y/2))*np.sin(o*4 - t)
    c = o*e/30 - t/8
    sp1.xdata = (q*0.7*np.sin(c)) + 9*np.cos(y/19 + t) + 200
    sp1.ydata = 200 + (q/2*np.cos(c))
    plt.draw()
    time.sleep(0.05)
