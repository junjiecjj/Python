#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:07:13 2024

@author: jack
"""


#%% 各种参数方程曲面
import numpy as np
import matplotlib.pyplot as plt
# import os
# # 如果文件夹不存在，创建文件夹
# if not os.path.isdir("Figures"):
#     os.makedirs("Figures")


# 正球体
u = np.linspace(0, np.pi, 101)
v = np.linspace(0, 2 * np.pi, 101)

x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot_surface(x, y, z, rstride=2, cstride=2, cmap = 'RdYlBu', edgecolors='k')
ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))
# fig.savefig('Figures/正球体.svg', format='svg')
plt.show()


# 游泳圈
n = 100

theta = np.linspace(0, 2.*np.pi, n)
phi = np.linspace(0, 2.*np.pi, n)
theta, phi = np.meshgrid(theta, phi)
c, a = 2, 1
x = (c + a*np.cos(theta)) * np.cos(phi)
y = (c + a*np.cos(theta)) * np.sin(phi)
z = a * np.sin(theta)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot_surface(x, y, z,
                rstride=2, cstride=2,
                cmap = 'RdYlBu', edgecolors='k')
ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
ax.set_zlim(-3,3)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))

# fig.savefig('Figures/游泳圈.svg', format='svg')
plt.show()




#%% “花篮”
dphi, dtheta = np.pi / 250.0, np.pi / 250.0
[phi, theta] = np.mgrid[0:np.pi + dphi * 1.5:dphi,
                        0:2 * np.pi + dtheta * 1.5:dtheta]
m0 = 4; m1 = 3; m2 = 2; m3 = 3;
m4 = 6; m5 = 2; m6 = 6; m7 = 4;

# 参数方程
r = (np.sin(m0 * phi) ** m1 + np.cos(m2 * phi) ** m3 +
     np.sin(m4 * theta) ** m5 + np.cos(m6 * theta) ** m7)
x = r * np.sin(phi) * np.cos(theta)
y = r * np.cos(phi)
z = r * np.sin(phi) * np.sin(theta)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot_surface(x, y, z,
                rstride=3, cstride=3,
                cmap = 'RdYlBu', edgecolors='k')
ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))
# fig.savefig('Figures/花篮.svg', format='svg')
plt.show()


#%% “螺旋”
s = np.linspace(0, 2 * np.pi, 240)
t = np.linspace(0, np.pi, 240)
tGrid, sGrid = np.meshgrid(s, t)

r = 2 + np.sin(7 * sGrid + 5 * tGrid)  # r = 2 + sin(7s+5t)
x = r * np.cos(sGrid) * np.sin(tGrid)  # x = r*cos(s)*sin(t)
y = r * np.sin(sGrid) * np.sin(tGrid)  # y = r*sin(s)*sin(t)
z = r * np.cos(tGrid)                  # z = r*cos(t)


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot_surface(x, y, z,
                rstride=3, cstride=3,
                cmap = 'RdYlBu', edgecolors='k')
ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))
plt.show()





#%% 三角网格

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# 正球体，规则三角网格
u = np.linspace(0, np.pi, 26)
v = np.linspace(0, 2 * np.pi, 26)
u, v = np.meshgrid(u, v)
u, v = u.flatten(), v.flatten()

r = 1
# z轴坐标网格数据
z = r*np.cos(u)

# x轴坐标网格数据
x = r*np.sin(u)*np.cos(v)

# y轴坐标网格数据
y = r*np.sin(u)*np.sin(v)

# 三角网格化
tri = mtri.Triangulation(u, v)
# 基于u和v，相当于theta-phi平面的网格点的三角化

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot_trisurf(x, y, z,
                triangles=tri.triangles,
                cmap=plt.cm.RdYlBu,
                edgecolor = 'k')
ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))

# fig.savefig('Figures/正球体.svg', format='svg')
plt.show()


# 游泳圈，规则三角网格
n = 21

theta = np.linspace(0, 2.*np.pi, n)
phi = np.linspace(0, 2.*np.pi, n)
theta, phi = np.meshgrid(theta, phi)
theta = theta.flatten()
phi = phi.flatten()

c, a = 2, 1
x = (c + a*np.cos(theta)) * np.cos(phi)
y = (c + a*np.cos(theta)) * np.sin(phi)
z = a * np.sin(theta)

# 三角网格化
tri = mtri.Triangulation(theta, phi)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot_trisurf(x, y, z,
                triangles=tri.triangles,
                cmap=plt.cm.RdYlBu,
                edgecolor = 'k')
ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
ax.set_zlim(-3,3)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))
# fig.savefig('Figures/游泳圈.svg', format='svg')
plt.show()


# 莫比乌斯环，规则三角网格
u = np.linspace(0, 2.0 * np.pi, endpoint=True, num=75)
v = np.linspace(-0.5, 0.5, endpoint=True, num=15)
u, v = np.meshgrid(u, v)
u, v = u.flatten(), v.flatten()

x = (1 + 0.5 * v * np.cos(u / 2.0)) * np.cos(u)
y = (1 + 0.5 * v * np.cos(u / 2.0)) * np.sin(u)
z = 0.5 * v * np.sin(u / 2.0)

# 三角网格化
tri = mtri.Triangulation(u, v)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot_trisurf(x, y, z,
                triangles=tri.triangles,
                cmap=plt.cm.RdYlBu,
                edgecolor = 'k')
ax.set_zlim(-0.6, 0.6)
ax.view_init(azim=-30, elev=45)
ax.set_proj_type('ortho')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))
# fig.savefig('Figures/莫比乌斯环.svg', format='svg')
plt.show()



# 正球体，不规则三角网格
u = np.random.uniform(low=0.0, high=np.pi, size=1000)
v = np.random.uniform(low=0.0, high=2*np.pi, size=1000)

r = 1
# z轴坐标网格数据
z = r*np.cos(u)

# x轴坐标网格数据
x = r*np.sin(u)*np.cos(v)

# y轴坐标网格数据
y = r*np.sin(u)*np.sin(v)

# 三角网格化
tri = mtri.Triangulation(u, v)
# 基于u和v，相当于theta-phi平面的网格点的三角化

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot_trisurf(x, y, z,
                triangles=tri.triangles,
                cmap=plt.cm.RdYlBu,
                edgecolor = 'k')
ax.view_init(azim=-120, elev=30)
# ax.view_init(azim=-30, elev=45)
ax.set_proj_type('ortho')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))

# fig.savefig('Figures/正球体，不规则三角网格.svg', format='svg')
plt.show()































































































































































































































































































































































































































































































































































































































































































































































































































