#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 25 03:28:04 2025

@author: jack
"""

import sympy
import numpy as np
import matplotlib.pyplot as plt


#%% Bk2_Ch24_01 可视化复数


import sympy
import numpy as np
import matplotlib.pyplot as plt


## 产生数据
# 颗粒度高，用来颜色渲染
xx1, xx2 = np.meshgrid(np.linspace(-2, 2, 128),
                       np.linspace(-2, 2, 128))
zz = xx1 + xx2 * 1j
zz_angle = np.angle(zz)
zz_norm  = np.abs(zz)

# 颗粒度低，用来绘制箭头图
xx1_, xx2_ = np.meshgrid(np.linspace(-2, 2, 17),
                         np.linspace(-2, 2, 17))
zz_ = xx1_ + xx2_ * 1j
zz_angle_ = np.angle(zz_)
zz_norm_ = np.abs(zz_)

# 向量起始点 (0,0)
zeros = np.zeros_like(xx1_)

## 绘制箭头图
fig, ax = plt.subplots(figsize = (5,5))
ax.scatter(xx1_, xx2_, marker = '.')
ax.quiver (zeros, zeros, xx1_, xx2_,
           color = [0.6, 0.6, 0.6],
           angles='xy', scale_units='xy', scale = 1,
           edgecolor='none')
ax.axhline(y = 0, c = 'k'); ax.axvline(x = 0, c = 'k')
ax.set_xlim(-2,2); ax.set_ylim(-2,2)
ax.set_xticks(np.arange(-2,3)); ax.set_yticks(np.arange(-2,3))
ax.set_xlabel(r"$Re(z)$"); ax.set_ylabel(r"$Im(z)$")
plt.show()

## 辐角
fig, ax = plt.subplots(figsize = (5,5))
ax.pcolormesh(xx1, xx2, zz_angle,
              cmap='hsv', shading = 'auto',
              rasterized = True)
ax.set_xlim(-2,2);
ax.set_ylim(-2,2)
ax.set_xticks(np.arange(-2,3));
ax.set_yticks(np.arange(-2,3))
ax.axhline(y = 0, c = 'k');
ax.axvline(x = 0, c = 'k')
plt.show()

fig, ax = plt.subplots(figsize = (5,5))
ax.quiver(zeros, zeros, xx1_, xx2_, zz_angle_,
          angles='xy', scale_units='xy', scale = 1,
          edgecolor='none', alpha=0.8, cmap = 'hsv')
ax.set_xlim(-2,2);
ax.set_ylim(-2,2)
ax.axhline(y = 0, c = 'k');
ax.axvline(x = 0, c = 'k')
ax.set_xticks(np.arange(-2,3));
ax.set_yticks(np.arange(-2,3))
plt.show()

## 复数模
fig, ax = plt.subplots(figsize = (5,5))

plt.contour(xx1, xx2, zz_norm,
            levels = np.linspace(0,5,26),
            colors = [[0.8, 0.8, 0.8, 1]])
plt.pcolormesh(xx1, xx2, zz_norm, cmap='RdYlBu_r',
               shading = 'auto', rasterized = True)
ax.set_xlim(-2,2); ax.set_ylim(-2,2)
ax.axhline(y = 0, c = 'k'); ax.axvline(x = 0, c = 'k')
ax.set_xticks(np.arange(-2,3)); ax.set_yticks(np.arange(-2,3))
plt.show()

# 三维曲面渲染
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(xx1, xx2, zz_norm,
                cmap="RdYlBu_r", shade=True, alpha=1)
ax.set_xlabel("$Re(z)$"); ax.set_ylabel("$Im(z)$")
ax.set_proj_type('ortho')
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.view_init(azim=-120, elev=30); ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max());  ax.set_ylim(xx2.min(), xx2.max())
plt.show()

## 2D 幅角+模, 细粒度
fig, ax = plt.subplots(figsize = (5,5))
ax.pcolormesh(xx1, xx2, zz_angle, cmap='hsv',
              shading = 'auto', rasterized = True)
ax.contour(xx1, xx2, zz_norm,
           levels = np.linspace(0,5,26),
           colors = [[0.8, 0.8, 0.8, 1]])
ax.set_xlim(-2,2); ax.set_ylim(-2,2)
ax.set_xticks(np.arange(-2,3)); ax.set_yticks(np.arange(-2,3))
ax.axhline(y = 0, c = 'k'); ax.axvline(x = 0, c = 'k')
plt.show()


## 2D 幅角+模, c粗粒度
fig, ax = plt.subplots(figsize = (5,5))
plt.quiver(zeros, zeros, xx1_, xx2_, zz_angle_,
            angles='xy', scale_units='xy', scale = 1,
            edgecolor='none', alpha=0.8, cmap = 'hsv')
plt.contour(xx1, xx2, zz_norm,
            levels = np.linspace(0,5,26),
            colors = [[0.8, 0.8, 0.8, 1]])
ax.axhline(y = 0, c = 'k'); ax.axvline(x = 0, c = 'k')
ax.set_xlim(-2,2); ax.set_ylim(-2,2)
ax.set_xticks(np.arange(-2,3)); ax.set_yticks(np.arange(-2,3))


# 网格
fig, ax = plt.subplots(figsize = (5,5))

plt.quiver(zeros, zeros, xx1_, xx2_, zz_angle_,
            angles='xy', scale_units='xy', scale = 1,
            edgecolor='none', alpha=0.8, cmap = 'hsv')

ax.contour(xx1, xx2, np.abs(xx1 - np.round(xx1)), levels = 1, colors="black", linewidths=0.25)
ax.contour(xx1, xx2, np.abs(xx2 - np.round(xx2)), levels = 1, colors="black", linewidths=0.25)

ax.axhline(y = 0, c = 'k'); ax.axvline(x = 0, c = 'k')
ax.set_xlim(-2,2); ax.set_ylim(-2,2)
ax.set_xticks(np.arange(-2,3)); ax.set_yticks(np.arange(-2,3))
plt.show()


fig, ax = plt.subplots(figsize = (5,5))
plt.pcolormesh(xx1, xx2, zz_angle, cmap='hsv', shading = 'auto', rasterized = True)
ax.contour(xx1, xx2, np.abs(xx1 - np.round(xx1)), levels = 1, colors="black", linewidths=0.25)
ax.contour(xx1, xx2, np.abs(xx2 - np.round(xx2)), levels = 1, colors="black", linewidths=0.25)

ax.axhline(y = 0, c = 'k'); ax.axvline(x = 0, c = 'k')
ax.set_xlim(-2,2); ax.set_ylim(-2,2)
ax.set_xticks(np.arange(-2,3)); ax.set_yticks(np.arange(-2,3))
plt.show()

#%% Bk2_Ch24_02 复数函数

## 可视化函数
def visualize(xx1, xx2, cc):
    # 计算复数模、辐角
    cc_norm = np.abs(cc);
    cc_angle = np.angle(cc)
    # 分离实部、虚部
    cc_xx1 = cc.real;
    cc_xx2 = cc.imag

    fig = plt.figure(figsize=(6, 3))
    # 第一幅子图
    ax = fig.add_subplot(1, 2, 1)
    ax.pcolormesh(xx1, xx2, cc_angle, cmap='hsv', shading = 'auto', rasterized = True)
    ax.contour(xx1, xx2, cc_norm, levels = np.linspace(0,np.mean(cc_norm) + 5* np.std(cc_norm),31), colors = 'w', linewidths = 0.25)
    ax.set_xlim(-2,2);
    ax.set_ylim(-2,2)
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.axhline(y = 0, c = 'k');
    ax.axvline(x = 0, c = 'k')
    ax.set_aspect('equal')
    # 第二幅子图
    ax = fig.add_subplot(1, 2, 2)
    ax.pcolormesh(cc_angle, cmap='hsv', shading = 'auto', rasterized = True)
    ax.contour(np.abs(cc_xx1 - np.round(cc_xx1)), levels = 1, colors="black", linewidths=0.25)
    ax.contour(np.abs(cc_xx2 - np.round(cc_xx2)), levels = 1, colors="black", linewidths=0.25)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')


## 产生数据
xx1, xx2 = np.meshgrid(np.linspace(-2, 2, 2**4),np.linspace(-2, 2, 2**4))
zz = xx1 + xx2 * 1j

# 旋转
cc = zz * np.exp(np.pi/6 * 1j)
visualize(xx1, xx2, cc)

cc = zz * np.exp(np.pi/3 * 1j)
visualize(xx1, xx2, cc)

# 旋转 + 缩放
cc = zz * 2 * np.exp(np.pi*3/4 * 1j)
visualize(xx1, xx2, cc)

## 多项式
cc = zz**2
visualize(xx1, xx2, cc)

cc = zz**3
visualize(xx1, xx2, cc)

cc = zz**3 + zz**2 + 1
visualize(xx1, xx2, cc)

cc = zz**6 + 1
visualize(xx1, xx2, cc)

cc = zz**6 - 1
visualize(xx1, xx2, cc)

cc = zz**(-6) + 1
visualize(xx1, xx2, cc)

## 分式
cc = 1/zz
visualize(xx1, xx2, cc)

cc = 1/(1 - zz)
visualize(xx1, xx2, cc)


cc = (zz**2 - 1)/zz
visualize(xx1, xx2, cc)

cc = 1/(zz**2 - 1)
visualize(xx1, xx2, cc)

cc = 1/(zz**4 + 1)
visualize(xx1, xx2, cc)

cc = zz/(zz**2 + zz + 1)
visualize(xx1, xx2, cc)


## 根式
cc = np.sqrt(zz + 1)
visualize(xx1, xx2, cc)


cc = np.sqrt(zz**2 + 1)
visualize(xx1, xx2, cc)

cc = 1/np.sqrt(zz**3 + 1)
visualize(xx1, xx2, cc)


## 三角
cc = np.sin(zz)
visualize(xx1, xx2, cc)


cc = np.sin(1/zz)
visualize(xx1, xx2, cc)


cc = zz * np.sin(1/zz)
visualize(xx1, xx2, cc)

## 乘幂
cc = zz ** zz
visualize(xx1, xx2, cc)

cc = (1/zz) ** zz
visualize(xx1, xx2, cc)

cc = zz ** (1/zz)
visualize(xx1, xx2, cc)


## 自然指数
cc = np.exp(zz)
visualize(xx1, xx2, cc)
cc = np.exp(-zz ** 2)
visualize(xx1, xx2, cc)

cc = np.exp(1/zz)
visualize(xx1, xx2, cc)

## 自然对数
cc = np.log(zz)
visualize(xx1, xx2, cc)

cc = np.log(zz**2 - 1)
visualize(xx1, xx2, cc)

cc = np.log(zz**4 + 1)
visualize(xx1, xx2, cc)





































































