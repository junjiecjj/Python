#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 16:59:38 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzg4MTcwODk5Ng==&mid=2247497816&idx=1&sn=a420495a78cc901ad6223028282cfbcd&chksm=ce7259d4d36c6a3bb5cee6611db606b77eb162fa3db442e1c035cd470e13cbc44a45e6e2ba2e&mpshare=1&scene=1&srcid=0913SesPmVtsRuJxcqiNW3e0&sharer_shareinfo=4f9bdddf11db96cf9e0aef93b05b16e1&sharer_shareinfo_first=4f9bdddf11db96cf9e0aef93b05b16e1&exportkey=n_ChQIAhIQgrsFXxU4eBUDH3bL7hzY1BKfAgIE97dBBAEAAAAAAJxXNsyoqhgAAAAOpnltbLcz9gKNyK89dVj0PisOrjmgAbPhw7240jRHUDtSUpVKaw8TY0kJGE8kCa1PZ9HrjsBzU2%2BvEwb%2FQVboLtuoa8g8uKkFnznPQqy0uDyp%2Fm%2F4qcITHgKEfJsNd7qToOG9P7EeBfktqexqhKJlOarc3%2FbeSbexfxEkL2cRe%2BDaA%2Fb21109pLudn9Nve3XEZkvwPj8dVWHfrLm2nFWSsMO2%2F7UpFAib6swwCp2O5xjIGrGNEQbgZwpGWX%2B%2Fm4vcrEl%2B1zXTQFYvfGo%2BN3ZppcskCMEOsTarDuu9XPxR%2Budc5%2FxHVCUZVtVWXRChG85SerL0PqBdktXvhJoR40hpWAshSwyG4rZw&acctmode=0&pass_ticket=BQum%2Fp8wNnwJAbNTG7rb%2BLI8mmoZ0gi0pPQb8I4dayqbHHmRHboG9mHbN01G2Vin&wx_header=0#rd

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math

def shitshape():
    # 路径采样点数
    points = 400
    # 控制收缩曲线（后期快速变细）
    z = np.linspace(0, 1, points)
    z[0:50] = z[0:50] + z[49] * np.linspace(1, 0, 50)
    z[200:400] = z[200:400] - 0.12 * np.linspace(0, 1, 200)
    s = np.linspace(0, 1, points)
    # 螺旋路径的半径变化（路径本身）
    path_radius = np.linspace(0.7, 0.02, points) * (np.cos(np.linspace(0, np.pi/2, points)) + 1.5) / 3.5
    # 桶体粗细也收缩（同步缩小）
    tube_radius = 0.18 * (1 - s**2.5) + 0.001
    tube_radius[0:50] = np.sin(np.linspace(0, np.pi/2, 50)) * tube_radius[49]
    # 螺旋角度
    theta = np.linspace(0, 2*np.pi*4, points)
    # 横截面圆点
    circle_pts = 40
    circle_theta = np.linspace(0, 2*np.pi, circle_pts)
    # 初始化
    X = np.zeros((circle_pts, points))
    Y = np.zeros((circle_pts, points))
    Z = np.zeros((circle_pts, points))

    for i in range(points - 1):
        # 当前螺旋中心位置（路径本身在缩小）
        R = path_radius[i]
        center_x = R * np.cos(theta[i])
        center_y = R * np.sin(theta[i])
        center_z = z[i]
        # 切向量（方向）
        dx = path_radius[i+1] * np.cos(theta[i+1]) - center_x
        dy = path_radius[i+1] * np.sin(theta[i+1]) - center_y
        dz = z[i+1] - center_z
        tangent = np.array([dx, dy, dz])
        tangent = tangent / np.linalg.norm(tangent)
        # 横截面正交平面
        ref = np.array([0, 0, 1])
        if abs(np.dot(ref, tangent)) > 0.99:
            ref = np.array([1, 0, 0])
        normal1 = np.cross(tangent, ref)
        normal1 = normal1 / np.linalg.norm(normal1)
        normal2 = np.cross(tangent, normal1)
        # 横截面圆柱壳
        r_tube = tube_radius[i]
        for j in range(circle_pts):
            offset = r_tube * (np.cos(circle_theta[j]) * normal1 + np.sin(circle_theta[j]) * normal2)
            X[j, i] = center_x + offset[0]
            Y[j, i] = center_y + offset[1]
            Z[j, i] = center_z + offset[2]

    Z[:, -1] = Z[:, -2]
    return X, Y, Z

def setColorByH(H, cList):
    X_val = (H - np.min(H)) / (np.max(H) - np.min(H))
    xx = np.linspace(0, 1, len(cList))
    y1 = cList[:, 0]
    y2 = cList[:, 1]
    y3 = cList[:, 2]

    cMap = np.zeros((H.shape[0], H.shape[1], 3))
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            cMap[i, j, 0] = np.interp(X_val[i, j], xx, y1)
            cMap[i, j, 1] = np.interp(X_val[i, j], xx, y2)
            cMap[i, j, 2] = np.interp(X_val[i, j], xx, y3)

    return cMap

def rotateXYZ(X, Y, Z, R):
    nX = np.zeros_like(X)
    nY = np.zeros_like(Y)
    nZ = np.zeros_like(Z)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            v = np.array([X[i, j], Y[i, j], Z[i, j]])
            nv = np.dot(R, v)
            nX[i, j] = nv[0]
            nY[i, j] = nv[1]
            nZ[i, j] = nv[2]

    return nX, nY, nZ

def bezierCurve(pnts, N):
    """修正的贝塞尔曲线函数"""
    t = np.linspace(0, 1, N)
    n = len(pnts) - 1  # 控制点数量减1

    # 计算二项式系数
    C = [math.comb(n, i) for i in range(n + 1)]

    # 计算贝塞尔曲线点
    curve = np.zeros((N, 3))
    for i in range(N):
        for j in range(n + 1):
            curve[i] += C[j] * (t[i] ** j) * ((1 - t[i]) ** (n - j)) * pnts[j]

    return curve

def drawStraw(ax, X, Y, Z):
    """简化的花杆绘制函数"""
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    m, n = min_idx
    x1 = X[m, n]
    y1 = Y[m, n]
    z1 = Z[m, n] + 0.03

    # 简化的直线花杆
    xx = np.array([x1, 0])
    yy = np.array([y1, 0])
    zz = np.array([z1, -1.5])

    ax.plot(xx, yy, zz, color=[88/255, 130/255, 126/255], linewidth=2)

def R_x(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def R_y(theta):
    return np.array([
        [np.cos(theta), 0, -np.sin(theta)],
        [0, 1, 0],
        [np.sin(theta), 0, np.cos(theta)]
    ])

# icecream 1
print("绘制 icecream 1...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = shitshape()
ax.plot_surface(X, Y, Z, color=[0.4, 0.2, 0], edgecolor='none')
ax.set_box_aspect([1, 1, 1])
ax.axis('off')
ax.view_init(elev=10, azim=-60)
plt.title('Icecream 1')
plt.show()

# icecream 2
print("绘制 icecream 2...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.axis('off')

X, Y, Z = shitshape()
ax.plot_surface(X + 0.1, Y, Z + 1.06, color=[0.4, 0.2, 0], edgecolor='none')
ax.plot_surface(X/1.3, Y/1.3, -Z + 1.1, color=[0.4, 0.2, 0], edgecolor='none')

# 甜筒数据构造
N = 500
n = 30
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X_mesh, Y_mesh = np.meshgrid(x, y)
Z_mesh = np.zeros_like(X_mesh)
Z_mesh[0::n, :] = 0.01
Z_mesh[:, 0::n] = 0.01
Z_mesh[1::n, :] = 0.01
Z_mesh[:, 1::n] = 0.01
Z_mesh[X_mesh**2 + Y_mesh**2 > 1] = np.nan

nXYZ = np.dot(np.column_stack([X_mesh.flatten(), Y_mesh.flatten(), Z_mesh.flatten()]), R_x(-np.pi/2.9))
nX = nXYZ[:, 0].reshape(X_mesh.shape)
nY = nXYZ[:, 1].reshape(Y_mesh.shape)
nZ = nXYZ[:, 2].reshape(Z_mesh.shape)
nZ = nZ - np.min(nZ)
nY = nY - np.min(nY)
nT = nX / (nZ/2.5 + 1e-10)
nR = nY
nX_new = np.cos(nT) * nR
nY_new = np.sin(nT) * nR

ax.plot_surface(nX_new, nY_new, nZ, color=[228/255, 200/255, 142/255], edgecolor='none')

# 绘制饼干
def create_cylinder(radius, height, resolution=100):
    theta = np.linspace(0, 2*np.pi, resolution)
    z = np.linspace(0, height, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = radius * np.cos(theta_grid)
    y = radius * np.sin(theta_grid)
    return x, y, z_grid

X_cyl, Y_cyl, Z_cyl = create_cylinder(0.04, 1, 100)

# 第一个饼干
nXYZ = np.dot(np.column_stack([X_cyl.flatten(), Y_cyl.flatten(), Z_cyl.flatten()]), R_x(np.pi/5))
nX = nXYZ[:, 0].reshape(X_cyl.shape)
nY = nXYZ[:, 1].reshape(Y_cyl.shape)
nZ = nXYZ[:, 2].reshape(Z_cyl.shape)
ax.plot_surface(nX, nY, nZ + 1.3, color=[0.8, 0.6, 0.4], edgecolor='none')

# 第二个饼干
nXYZ = np.dot(np.column_stack([X_cyl.flatten(), Y_cyl.flatten(), Z_cyl.flatten()]), R_x(np.pi/6))
nX = nXYZ[:, 0].reshape(X_cyl.shape)
nY = nXYZ[:, 1].reshape(Y_cyl.shape)
nZ = nXYZ[:, 2].reshape(Z_cyl.shape)
nXYZ = np.dot(np.column_stack([nX.flatten(), nY.flatten(), nZ.flatten()]), R_y(np.pi/7))
nX = nXYZ[:, 0].reshape(X_cyl.shape)
nY = nXYZ[:, 1].reshape(Y_cyl.shape)
nZ = nXYZ[:, 2].reshape(Z_cyl.shape)
ax.plot_surface(nX, nY, nZ + 1.2, color=[0.8, 0.6, 0.4], edgecolor='none')

ax.view_init(elev=20, azim=30)
plt.title('Icecream 2')
plt.show()

# icecream 3 - 简化版本，跳过复杂的花杆绘制
print("绘制 icecream 3 (简化版)...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

[Xa, Ya, Za] = shitshape()

# 百合花部分
rb = np.arange(0, 1.01, 0.01)
tb = np.linspace(0, 2, 151)
wb = rb[:, np.newaxis] * ((np.abs((1 - np.mod(tb*5, 2))))/2 + 0.3)
xb = wb * np.cos(np.pi * tb)
yb = wb * np.sin(np.pi * tb)
zb_func = lambda a: (-np.cos(np.pi * wb * a) + 1)**0.2
Zb = zb_func(1.2)

# 颜色映射表
colorList = np.array([
    [0.3300, 0.3300, 0.6900],
    [0.5300, 0.4000, 0.6800],
    [0.6800, 0.4200, 0.6300],
    [0.7800, 0.4200, 0.5700],
    [0.9100, 0.4900, 0.4700],
    [0.9600, 0.7300, 0.4400]
])
colorMapb = setColorByH(Zb, colorList * 0.4 + 0.6)

# 旋转矩阵定义
yaw_z = 72 * np.pi/180
roll_x_1 = np.pi/8
roll_x_2 = np.pi/9

R_z_2 = np.array([
    [np.cos(yaw_z), -np.sin(yaw_z), 0],
    [np.sin(yaw_z), np.cos(yaw_z), 0],
    [0, 0, 1]
])

R_z_1 = np.array([
    [np.cos(yaw_z/2), -np.sin(yaw_z/2), 0],
    [np.sin(yaw_z/2), np.cos(yaw_z/2), 0],
    [0, 0, 1]
])

R_z_3 = np.array([
    [np.cos(yaw_z/3), -np.sin(yaw_z/3), 0],
    [np.sin(yaw_z/3), np.cos(yaw_z/3), 0],
    [0, 0, 1]
])

R_x_1 = np.array([
    [1, 0, 0],
    [0, np.cos(roll_x_1), -np.sin(roll_x_1)],
    [0, np.sin(roll_x_1), np.cos(roll_x_1)]
])

R_x_2 = np.array([
    [1, 0, 0],
    [0, np.cos(roll_x_2), -np.sin(roll_x_2)],
    [0, np.sin(roll_x_2), np.cos(roll_x_2)]
])

# 曲面旋转及绘制 - 简化版，只绘制主要形状
ax.plot_surface(Xa, Ya, Za + 0.7, color=[0.4, 0.2, 0], alpha=0.95, edgecolor='black', linewidth=0.1)

[nXr, nYr, nZr] = rotateXYZ(Xa, Ya, Za + 0.7, R_x_1)
nYr = nYr - 0.4
ax.plot_surface(nXr, nYr, nZr - 0.1, color=[0.4, 0.2, 0], alpha=0.95, edgecolor='black', linewidth=0.1)

for k in range(4):
    [nXr, nYr, nZr] = rotateXYZ(nXr, nYr, nZr, R_z_2)
    ax.plot_surface(nXr, nYr, nZr - 0.1, color=[0.4, 0.2, 0], alpha=0.95, edgecolor='black', linewidth=0.1)

# 百合花部分
[nXb, nYb, nZb] = rotateXYZ(xb/2.5, yb/2.5, Zb/2.5 + 0.32, R_x_2)
nYb = nYb - 1.35

for k in range(5):
    [nXb, nYb, nZb] = rotateXYZ(nXb, nYb, nZb, R_z_2)
    ax.plot_surface(nXb, nYb, nZb, facecolors=colorMapb, edgecolor='none')

[nXb, nYb, nZb] = rotateXYZ(xb/2.5, yb/2.5, Zb/2.5 + 0.32, R_x_2)
nYb = nYb - 1.15
[nXb, nYb, nZb] = rotateXYZ(nXb, nYb, nZb, R_z_1)

for k in range(5):
    [nXb, nYb, nZb] = rotateXYZ(nXb, nYb, nZb, R_z_2)
    ax.plot_surface(nXb, nYb, nZb, facecolors=colorMapb, edgecolor='none')

[nXb, nYb, nZb] = rotateXYZ(xb/2.5, yb/2.5, Zb/2.5 + 0.32, R_x_2)
nYb = nYb - 1.25
[nXb, nYb, nZb] = rotateXYZ(nXb, nYb, nZb, R_z_3)

for k in range(5):
    [nXb, nYb, nZb] = rotateXYZ(nXb, nYb, nZb, R_z_2)
    ax.plot_surface(nXb, nYb, nZb, facecolors=colorMapb, edgecolor='none')

[nXb, nYb, nZb] = rotateXYZ(xb/2.5, yb/2.5, Zb/2.5 + 0.32, R_x_2)
nYb = nYb - 1.25
[nXb, nYb, nZb] = rotateXYZ(nXb, nYb, nZb, R_z_3)
[nXb, nYb, nZb] = rotateXYZ(nXb, nYb, nZb, R_z_3)

for k in range(5):
    [nXb, nYb, nZb] = rotateXYZ(nXb, nYb, nZb, R_z_2)
    ax.plot_surface(nXb, nYb, nZb, facecolors=colorMapb, edgecolor='none')

ax.set_box_aspect([1, 1, 1])
ax.axis('off')
ax.view_init(elev=35, azim=-15)
plt.title('Icecream 3 (简化版)')
plt.show()

print("所有图形绘制完成！")
