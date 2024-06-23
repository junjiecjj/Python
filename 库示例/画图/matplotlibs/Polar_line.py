#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 14:49:20 2024

@author: jack
"""

# 导入包
import numpy as np
import matplotlib.pyplot as plt

import os

# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

## 正圆
theta_array = np.linspace(0,2*np.pi, 200)
r_array     = np.ones_like(theta_array)

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='polar')
ax.plot(theta_array, r_array)
ax.set_rmax(2)
ax.set_rmin(0)
ax.set_rlabel_position(22.5)  # Move radial labels away from plotted line
ax.grid(True)
ax.set_yticklabels([])
fig.savefig('Figures/正圆.svg', format='svg')


# 阿基米德螺线
r_array = np.arange(0, 4, 0.01)
theta_array = 2 * np.pi * r_array

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='polar')
ax.plot(theta_array, r_array)
ax.set_rmin(r_array.min())
ax.set_rlabel_position(22.5)
ax.grid(True)
ax.set_yticklabels([])
fig.savefig('Figures/阿基米德螺线.svg', format='svg')



# 心形曲线
theta_array = np.linspace(0,2*np.pi, 2000)
r_array     = 2 + 2*np.cos(theta_array)

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='polar')

ax.plot(theta_array, r_array)
ax.set_rmin(r_array.min())

ax.set_rlabel_position(22.5)
ax.grid(True)
ax.set_yticklabels([])
fig.savefig('Figures/心形曲线.svg', format='svg')


# 椭圆
theta_array = np.linspace(0,2*np.pi, 2000)
r_array     = 1/(1 + 0.5*np.cos(theta_array))

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='polar')

ax.plot(theta_array, r_array)
ax.set_rmin(0)

ax.set_rlabel_position(22.5)
ax.grid(True)
ax.set_yticklabels([])
fig.savefig('Figures/椭圆.svg', format='svg')



# 玫瑰线
theta_array = np.linspace(0,2*np.pi, 2000)
r_array     = 2*np.sin(6*theta_array)

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='polar')

ax.plot(theta_array, r_array)
ax.set_rmin(r_array.min())

ax.set_rlabel_position(22.5)
ax.grid(True)
ax.set_yticklabels([])
fig.savefig('Figures/玫瑰线.svg', format='svg')


# 玫瑰线，有理数
theta_array = np.linspace(0,7*np.pi, 2000)
r_array     = 2*np.sin(11/3*theta_array)

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='polar')

ax.plot(theta_array, r_array)
ax.set_rmin(r_array.min())

ax.set_rlabel_position(22.5)
ax.grid(True)
ax.set_yticklabels([])
fig.savefig('Figures/玫瑰线，有理数.svg', format='svg')


# 双纽线

a = 2
t = np.linspace(0, 2*np.pi, 200)
sint = np.sin(t)
cost = np.cos(t)
theta = np.arctan2(sint*cost, cost)
r = a*np.abs(cost) / np.sqrt(1 + sint**2)

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='polar')

ax.plot(theta, r)
ax.set_rmin(r_array.min())

ax.set_rlabel_position(22.5)
ax.grid(True)
ax.set_yticklabels([])
fig.savefig('Figures/双纽线.svg', format='svg')




# 蝴蝶翼
theta_array = np.linspace(0,2*np.pi, 2000)
r_array = 1 - np.cos(theta_array)*np.sin(3*theta_array)

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(projection='polar')

ax.plot(theta_array, r_array)
ax.set_rlabel_position(22.5)
ax.set_rmin(r_array.min())
ax.grid(True)
ax.set_yticklabels([])
fig.savefig('Figures/蝴蝶翼.svg', format='svg')





































































































































































































































































































































