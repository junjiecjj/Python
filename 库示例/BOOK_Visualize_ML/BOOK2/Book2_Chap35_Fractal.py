

#%%
# Barnsley fern
# https://en.wikipedia.org/wiki/Barnsley_fern

import sys
import numpy as np
import random
import matplotlib.pyplot as plt

def f1(x, y):
    return np.array([[0, 0], [0, 0.16]]).dot(np.array([x, y]))


def f2(x, y):
    return (np.array([[0.85, 0.04], [-0.04, 0.85]]).dot(np.array([x, y])) + np.array([0, 1.6]))

def f3(x, y):
    return (np.array([[0.20, -0.26], [0.23, 0.22]]).dot(np.array([x, y])) + np.array([0, 1.6]))

def f4(x, y):
    return (np.array([[-0.15, 0.28], [0.26, 0.24]]).dot(np.array([x, y])) + np.array([0, 0.44]))

def barnsley_fern(n):

    x, y = [0], [0]
    for _ in range(n):
        r = random.random()
        if r < 0.01:
            dot = f1(x[-1], y[-1])
        elif r < 0.86:
            dot = f2(x[-1], y[-1])
        elif r < 0.93:
            dot = f3(x[-1], y[-1])
        else:
            dot = f4(x[-1], y[-1])
        x.append(dot[0])
        y.append(dot[1])
    return x, y

fig = plt.figure(figsize = (5,8))
for idx in range(6):
    ax = plt.subplot(3,2,idx + 1)
    x,y = barnsley_fern(2**(idx + 6))
    ax.plot(x, y, '.', markersize=2, color='g')
    ax.set_axis_off()
plt.show()
#%%
"""
Dragon curve
https://en.wikipedia.org/wiki/Dragon_curve
"""
import sys
from math import sqrt, cos, sin, pi
import numpy as np
import random
import matplotlib.pyplot as plt

def f1(x, y):
    return (1 / sqrt(2)) * np.array( [[cos(pi/4), -sin(pi/4)], [sin(pi/4), cos(pi/4)]]).dot( np.array([x, y]))


def f2(x, y):
    return (1 / sqrt(2)) * np.array( [[cos(3*pi/4), -sin(3*pi/4)], [sin(3*pi/4), cos(3*pi/4)]]).dot( np.array([x, y])) + np.array([1, 0])

def dragon_curve(n):
    x, y = [0], [0]
    for _ in range(n):
        r = random.random()
        if r <= 0.5:
            dot = f1(x[-1], y[-1])
        else:
            dot = f2(x[-1], y[-1])
        x.append(dot[0])
        y.append(dot[1])
    return x, y

fig = plt.figure(figsize = (5,8))
for idx in range(6):
    ax = plt.subplot(3,2,idx + 1)
    x,y = dragon_curve(2**(idx + 6))

    ax.plot(x, y, '.', markersize=2, color='g')
    ax.set_axis_off()
plt.show()
#%%

# 参考：
# https://mathworld.wolfram.com/JuliaSet.html

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

# m = 480
# n = 320

m = 2**11
n = m
iterations = 2**8
s = m * 0.8

def julia_set(iterations, c):
    c = np.full((n, m), c)
    x = np.linspace(-m / s, m / s, num=m).reshape((1, m))
    y = np.linspace(-n / s, n / s, num=n).reshape((n, 1))
    Z = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))
    M = np.full((n, m), True, dtype=bool)
    N = np.zeros((n, m))
    for i in range(iterations):
        Z[M] = Z[M] * Z[M] + c[M]
        M[np.abs(Z) > 2] = False
        N[M] = i
    return np.flipud(N)


c = 0.285 + 0.01 * 1j
N = julia_set(iterations, c)
fig, ax = plt.subplots(figsize = (10,10))
ax.imshow(N, cmap='RdYlBu_r')
ax.set_axis_off()


c = -0.4 + 0.6j
N = julia_set(iterations, c)

fig, ax = plt.subplots(figsize = (10,10))
ax.imshow(N, cmap='RdYlBu_r')
ax.set_axis_off()

c = -0.70176 - 0.3842 * 1j
N = julia_set(iterations, c)
fig, ax = plt.subplots(figsize = (10,10))

ax.imshow(N, cmap='RdYlBu_r')
ax.set_axis_off()

c = -0.7269 - 0.1889 * 1j

N = julia_set(iterations, c)
fig, ax = plt.subplots(figsize = (10,10))
ax.imshow(N, cmap='RdYlBu_r')
ax.set_axis_off()

c = -0.835 - 0.2321 * 1j
N = julia_set(iterations, c)
fig, ax = plt.subplots(figsize = (10,10))
ax.imshow(N, cmap='RdYlBu_r')
ax.set_axis_off()

for theta_i in np.linspace(0, np.pi, 7):
    c = 0.7885 * np.exp(1j * theta_i)
    N = julia_set(iterations, c)
    fig, ax = plt.subplots(figsize = (10,10))
    ax.imshow(N, cmap='RdYlBu_r')
    ax.set_axis_off()
plt.show()
#%%
"""
Koch snowflake
https://en.wikipedia.org/wiki/Koch_snowflake
"""
import sys
from math import sqrt
import matplotlib.pyplot as plt


def kochCurve(n, xA, yA, xB, yB):
    if n != 0:
        xC = xA + (xB - xA) / 3
        yC = yA + (yB - yA) / 3
        xD = xA + 2 * (xB - xA) / 3
        yD = yA + 2 * (yB - yA) / 3
        xE = (xC + xD) / 2 - (yD - yC) * sqrt(3) / 2
        yE = (yC + yD) / 2 + (xD - xC) * sqrt(3) / 2
        kochCurve(n - 1, xA, yA, xC, yC)
        kochCurve(n - 1, xC, yC, xE, yE)
        kochCurve(n - 1, xE, yE, xD, yD)
        kochCurve(n - 1, xD, yD, xB, yB)
    else:
        plt.plot([xA, xB], [yA, yB], 'b')

def kockCurveConstruction(n):
    kochCurve(n, 0, 0, 1, 0)
    plt.axis("equal")
    plt.show()

def kochSnowflake(n):
    xA, yA = 0, 0
    xB, yB = 1 / 2, sqrt(0.75)
    xC, yC = 1, 0
    kochCurve(n, xA, yA, xB, yB)
    kochCurve(n, xB, yB, xC, yC)
    kochCurve(n, xC, yC, xA, yA)
    plt.axis("equal")
    plt.tight_layout()

fig = plt.figure(figsize = (5,8))
for idx in range(6):
    ax = plt.subplot(3,2,idx + 1)
    kochSnowflake(idx)
    ax.set_axis_off()
plt.show()

#%%

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 18:01:29 2023

@author: james
"""
# https://carpentries-incubator.github.io/lesson-parallel-python/06b-exercise-fractals/index.html
from matplotlib import pyplot as plt
import numpy as np


width = 2**8
height = 2**8
center = -0.8+0.0j
extent = 3.0+3.0j

def Mandelbrot_set(max_iter):
    result = np.zeros((height, width), int)
    for j in range(height):
        for i in range(width):
            c = center + (i - width // 2 + (j - height // 2)*1j) * scale
            z = 0
            for k in range(max_iter):
                z = z**2 + c
                if (z * z.conjugate()).real > 4.0:
                    break
            result[j, i] = k

    return result


fig = plt.figure(figsize = (5,8))
for idx in range(6):
    scale = max((extent / width).real, (extent / height).imag)
    ax = plt.subplot(3,2,idx + 1)
    plot_extent = (width + 1j * height) * scale
    z1 = center - plot_extent / 2
    z2 = z1 + plot_extent
    max_iter = 2**(idx + 3)

    result = Mandelbrot_set(max_iter)
    ax.imshow(result**(1/3), origin='lower', extent=(z1.real, z2.real, z1.imag, z2.imag))
    ax.set_axis_off()
plt.show()
#%%
"""
Pythagoras tree
https://en.wikipedia.org/wiki/Pythagoras_tree_(fractal)
"""
import sys
from math import sin, cos, pi
import matplotlib.pyplot as plt


def pythagorasTree(x, y, length, alpha, angle, order, counter):
    dx = length * sin(alpha)
    dy = length * cos(alpha)
    X, Y = [x], [y]

    x1 = x + dx
    y1 = y - dy
    X.append(x1)
    Y.append(y1)

    x2 = x + dx - dy
    y2 = y - dy - dx
    X.append(x2)
    Y.append(y2)

    x3 = x - dy
    y3 = y - dx
    X.append(x3)
    Y.append(y3)

    x4 = x - dy + length * cos(angle) * sin(alpha - angle)
    y4 = y - dx - length * cos(angle) * cos(alpha - angle)

    plt.fill(X, Y, color=(0, counter / color_index, 0))
    plt.axis('equal')

    if order > 0:
        pythagorasTree( x4, y4, length * sin(angle), alpha - angle + pi / 2, angle, order - 1, counter + 1)
        pythagorasTree( x3, y3, length * cos(angle), alpha - angle, angle, order - 1, counter + 1)

x, y = 0, 0
length = 1
angle = pi / int(sys.argv[2]) if len(sys.argv) == 3 else pi / 3
alpha = -pi / 2
fig = plt.figure(figsize = (5,8))
for idx in range(6):
    order = idx
    ax = plt.subplot(3,2,idx + 1)
    color_index = order + 1
    pythagorasTree(x - 1, y - 1, length, -pi / 2, angle * 3 / 4, order, 1)
    ax.set_axis_off()
plt.show()

#%%
"""
Reference:
https://github.com/Quentin18/Matplotlib-fractals
"""
import sys
import numpy as np
import matplotlib.pyplot as plt


def sierpinskiCarpet(n):
    n += 1
    T = np.ones((3**n, 3**n))
    a = n
    start = 1
    step = 3
    size = 1
    while a > 0:
        for i in range(start, 3**n, step):
            for j in range(start, 3**n, step):
                for k in range(size):
                    for l in range(size):
                        T[i + k, j + l] = 0
        a -= 1
        start *= 3
        step *= 3
        size *= 3

    return T

fig = plt.figure(figsize = (5,8))
for idx in range(6):
    ax = plt.subplot(3,2,idx + 1)
    T = sierpinskiCarpet(idx)

    ax.pcolormesh(T, cmap='Blues_r', rasterized = True)
    ax.set_axis_off()
plt.show()


#%%
"""
Sierpinsky triangle version 1
https://en.wikipedia.org/wiki/Sierpi%C5%84ski_triangle
"""
import sys
from math import sqrt
import matplotlib.pyplot as plt

def sierpinskyTriangle(n, x, y, c):
    if n != 0:
        xA, yA = x, y
        xB, yB = x + c, y
        xC, yC = x + c / 2, y + c * sqrt(3)/2
        xE, yE = (xA + xB) / 2, (yA + yB) / 2
        xF, yF = (xB + xC) / 2, (yB + yC) / 2
        xG, yG = (xA + xC) / 2, (yA + yC) / 2
        # Central triangle
        plt.fill([xE, xF, xG], [yE, yF, yG], 'w')
        # Small triangles
        sierpinskyTriangle(n - 1, x, y, c / 2)
        sierpinskyTriangle(n - 1, xG, yG, c / 2)
        sierpinskyTriangle(n - 1, xE, yE, c/2)
    else:
        plt.fill([x, x + c, x + c / 2], [y, y, y + c * sqrt(3) / 2], 'b')

fig = plt.figure(figsize = (5,8))
for idx in range(6):
    ax = plt.subplot(3,2,idx + 1)
    sierpinskyTriangle(idx, 0, 0, 10)

    ax.set_axis_off()
plt.show()

#%%
# 参考：
# https://mathworld.wolfram.com/JuliaSet.html

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import streamlit as st

# m = 480
# n = 320

m = 2**9
n = m
iterations = 2**8
s = m * 0.8

def julia_set(iterations, c):
    c = np.full((n, m), c)
    x = np.linspace(-m / s, m / s, num=m).reshape((1, m))
    y = np.linspace(-n / s, n / s, num=n).reshape((n, 1))
    Z = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))
    M = np.full((n, m), True, dtype=bool)
    N = np.zeros((n, m))

    for i in range(iterations):
        Z[M] = Z[M] * Z[M] + c[M]
        M[np.abs(Z) > 2] = False
        N[M] = i
    return np.flipud(N)


with st.sidebar:
    st.title('朱利亚集合')
    theta = st.slider('角度', 0.0,2*np.pi,0.0,0.01)

c = 0.7885 * np.exp(1j * theta)

N = julia_set(iterations, c)
fig, ax = plt.subplots(figsize = (5,5))
ax.imshow(N, cmap='RdYlBu_r')
ax.set_aspect('equal', adjustable='box')
ax.set_axis_off()
st.pyplot(fig)
plt.show()

#%%

import matplotlib.pyplot as plt
def draw_fractal(ax, levels=4, x=0, y=0, size=1):
    if levels == 0:
        ax.add_patch(plt.Rectangle((x, y), size, size, color='navy'))
    else:
        size3 = size / 3
        for i in range(3):
            for j in range(3):
                if (i + j) % 2 == 0:
                    draw_fractal(ax, levels - 1, x + i * size3, y + j * size3, size3)

fig = plt.figure(figsize = (5,8))
for idx in range(6):
    ax = plt.subplot(3,2,idx + 1)
    draw_fractal(ax, idx)
    ax.set_axis_off()
plt.show()
#%%
# -*- coding: utf-8 -*-
"""
https://en.wikipedia.org/wiki/Vicsek_fractal

"""

import matplotlib.pyplot as plt

def draw_fractal(ax, levels=4, x=0, y=0, size=1):
    if levels == 0:
        ax.add_patch(plt.Rectangle((x, y), size, size, color='navy'))
    else:
        size3 = size / 3
        for i in range(3):
            for j in range(3):
                if i == 1 or j == 1:
                    draw_fractal(ax, levels - 1, x + i * size3, y + j * size3, size3)
fig = plt.figure(figsize = (5,8))

for idx in range(6):

    ax = plt.subplot(3,2,idx + 1)
    draw_fractal(ax, idx)
    ax.set_axis_off()
plt.show()



#%%





























