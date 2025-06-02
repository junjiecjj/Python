#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:44:54 2025

@author: jack
"""

#%% Bk3_Ch5_01

import matplotlib.pyplot as plt
import numpy as np

x = [4,  0, -5, -4,  6, 0]
y = [2, -2,  7, -6, -5, 0]

labels = ['A', 'B', 'C', 'D', 'E', 'O']

fig, ax = plt.subplots()

plt.plot(x, y, 'x')

for label, i, j in zip(labels, x, y):
   plt.text(i, j+0.5, label + ' ({}, {})'.format(i, j))

plt.xlabel('x'); plt.ylabel('y')
plt.axhline(y=0, color='k', linestyle='-')
plt.axvline(x=0, color='k', linestyle='-')
plt.xticks(np.arange(-8, 8 + 1, step=1))
plt.yticks(np.arange(-8, 8 + 1, step=1))
plt.axis('scaled')

ax.set_xlim(-8,8); ax.set_ylim(-8,8)
ax.spines['top'].set_visible(False); ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False); ax.spines['right'].set_visible(False)

ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])


#%% Bk3_Ch5_02

import numpy as np
import matplotlib.pyplot as plt

def fig_decor(ax):

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.hlines(y=0, xmin=-8, xmax=8, color='k')
    ax.vlines(x=0, ymin=-8, ymax=8, color='k')

    ax.set_xticks(np.arange(-8, 8 + 1, step=1))
    ax.set_yticks(np.arange(-8, 8 + 1, step=1))

    ax.axis('scaled')
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

    ax.set_xbound(lower = -8, upper = 8)
    ax.set_ybound(lower = -8, upper = 8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

x_array = np.linspace(-8,8)
y_array = np.linspace(-8,8)

# orthogonal
fig, ax = plt.subplots()

y1 = 0.5*x_array + 2
y2 = -2*x_array - 1

ax.plot(x_array, y1)
ax.plot(x_array, y2)
fig_decor(ax)

# parallel
fig, ax = plt.subplots()

y1 = 0.5*x_array + 2
y2 = 0.5*x_array - 4

ax.plot(x_array, y1)
ax.plot(x_array, y2)
fig_decor(ax)

# horizontal
fig, ax = plt.subplots()

y1 = 0*x_array + 2
y2 = 0*x_array - 4

ax.plot(x_array, y1)
# axhline
ax.plot(x_array, y2)
fig_decor(ax)

# vertical
fig, ax = plt.subplots()

x1 = 0*y_array + 2
x2 = 0*y_array - 4

ax.plot(x1, y_array)
# axvline
ax.plot(x2, y_array)
fig_decor(ax)

#%% Bk3_Ch5_03

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

x_array = np.arange(0, 30 + 1, step=1)
y_line_1 = 35 - x_array
y_line_2 = (94 - 2*x_array)/4

fig, ax = plt.subplots()

plt.plot(x_array, y_line_1, color = '#0070C0')
plt.plot(x_array, y_line_2, color = 'g')

# solution of linear equations
plt.plot(23,12,marker = 'x', markersize = 12)
plt.axvline(x=23, color='r', linestyle='--')
plt.axhline(y=12, color='r', linestyle='--')

plt.xlabel('$x_1$ (number of chickens)')
plt.ylabel('$x_2$ (number of rabbits)')
plt.axhline(y=0, color='k', linestyle='-')
plt.axvline(x=0, color='k', linestyle='-')
plt.xticks(np.arange(0, 30 + 1, step=5))
plt.yticks(np.arange(0, 30 + 1, step=5))
plt.axis('scaled')
plt.minorticks_on()
ax.grid(which='minor', linestyle=':',
        linewidth='0.5', color=[0.8, 0.8, 0.8])
ax.set_xlim(0,30); ax.set_ylim(0,30)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])


#%% Bk3_Ch5_04

import numpy as np
import matplotlib.pyplot as plt

def plot_polar(theta, r):

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, r)
    # set radial axis limit
    ax.set_rmax(20)
    # set radial axis ticks
    ax.set_rticks([5, 10, 15, 20])
    # position radial labels
    ax.set_rlabel_position(-45)
    ax.set_thetagrids(np.arange(0.0, 360.0, 45.0));
    plt.show()

# circle
theta = np.linspace(0, 6*np.pi, 1000)

r = 10 + theta*0
plot_polar(theta, r)

# Archimedes' spiral
r = 1*theta
plot_polar(theta, r)

# Rose
r = 10*np.cos(6*theta) + 10
plot_polar(theta, r)


#%% Bk3_Ch5_05

import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,2*np.pi,100)

# parametric equation of unit circle
x1 = np.cos(t)
x2 = np.sin(t)

fig, ax = plt.subplots()
# plot the circle
plt.plot(x1,x2)

plt.show()
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_xticks(np.arange(-2, 2 + 1, step=1))
ax.set_yticks(np.arange(-2, 2 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = -2, upper = 2); ax.set_ybound(lower = -2, upper = 2)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.axhline(y=0, color='k', linestyle='-')
plt.axvline(x=0, color='k', linestyle='-')


#%% Bk3_Ch5_06

from sympy import *
from sympy.plotting import plot_parametric
import math

t = symbols('t')

# parametric equation of unit circle
x1 = cos(t)
x2 = sin(t)

# plot the circle
plot_parametric(x1, x2, (t, 0, 2*pi), size = (10,10))






#%%





#%%





#%%





