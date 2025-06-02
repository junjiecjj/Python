#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:46:19 2025

@author: jack
"""

#%% Bk3_Ch6_01

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_surf(xx, yy, zz, caption):
    norm_plt = plt.Normalize(zz.min(), zz.max())
    colors = cm.RdYlBu_r(norm_plt(zz))

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
    surf = ax.plot_surface(xx, yy, zz, facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))

    plt.show()
    ax.set_proj_type('ortho')

    if xx.min() == xx.max():
        ax.set_xlim(xx.min() - 4, xx.min() + 4)
    else:
        ax.set_xlim(xx.min(), xx.max())

    if yy.min() == yy.max():
        ax.set_ylim(yy.min() - 4, yy.min() + 4)
    else:
        ax.set_ylim(yy.min(), yy.max())

    if zz.min() == zz.max():
        ax.set_zlim(zz.min() - 4, zz.min() + 4)
    else:
        ax.set_zlim(zz.min(), zz.max())
    ax.set_xlabel(r'$\it{x}$')
    ax.set_ylabel(r'$\it{y}$')
    ax.set_zlabel(r'$\it{z}$')
    ax.set_title(caption)
    ax.view_init(azim=-135, elev=30)
    ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
    plt.tight_layout()

num = 33
x = np.linspace(-4, 4, num)
y = np.linspace(-4, 4, num)
xx, yy = np.meshgrid(x, y)

plt.close('all')

## z - 2 = 0
zz = 2 + xx*0;
caption = r'$z - 2 = 0$';
plot_surf (xx, yy, zz, caption)

## y - z = 0
zz = yy;
caption = r'$z - y = 0$';
plot_surf (xx, yy, zz, caption)

## x - z = 0
zz = xx;
caption = r'$x - z = 0$';
plot_surf (xx, yy, zz, caption)

## x + y - z = 0
zz = xx + yy;
caption = r'$x + y - z = 0$';
plot_surf (xx, yy, zz, caption)

## vertical mesh plot

x = np.linspace(-4, 4, num)
z = np.linspace(-4, 4, num)
xx,zz = np.meshgrid(x, z);

## y - 2 = 0
yy = 2 - xx*0
caption = r'$y - 2 = 0$';
plot_surf (xx, yy, zz, caption)

## x + y - 2 = 0
yy = 2 - xx
caption = r'$x + y - 2 = 0$';
plot_surf (xx, yy, zz, caption)

## x + 2 = 0
y = np.linspace(-4, 4, num)
z = np.linspace(-4, 4, num)
yy,zz = np.meshgrid(y,z);

xx = -2 - yy*0
caption = r'$x + 2 = 0$';
plot_surf (xx, yy, zz, caption)


#%% Bk3_Ch6_02

import math
import numpy as np
import matplotlib.pyplot as plt

num = 33
x = np.linspace(-4, 4, num)
y = np.linspace(-4, 4, num)
xx, yy = np.meshgrid(x, y)

zz1 = xx + yy;
zz2 = 2*xx - yy;

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
CS = ax.contour(xx,yy, zz1 - zz2, levels = [0], colors = '#339933') # 绘制交面
ax.cla()
ax.plot_wireframe(xx, yy, zz1, color = '#BDD6EE')
# rstride=10, cstride=10
ax.plot_wireframe(xx, yy, zz2, color = '#ECCCC0')
# plot the intersection line
for i in range(0,len(CS.allsegs[0])):
    contour_points_x_y = CS.allsegs[0][i]
    contour_points_z = (contour_points_x_y[:,0] + contour_points_x_y[:,1])
    ax.plot3D(contour_points_x_y[:,0], contour_points_x_y[:,1], contour_points_z, color = 'k', linewidth = 4)

ax.set_proj_type('ortho')
ax.set_xlim(xx.min(),xx.max())
ax.set_ylim(yy.min(),yy.max())

ax.set_xlabel(r'$\it{x}$')
ax.set_ylabel(r'$\it{y}$')
ax.set_zlabel(r'$\it{z}$')

ax.view_init(azim=-135, elev=30)
ax.grid(False)

plt.tight_layout()
plt.close('all')

#%% Bk3_Ch6_03

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# define three visualization tools
# ========================
# 3D contour plot of zz
# ========================
def plot_3D_f_xy(xx,yy,zz):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
    ax.plot_wireframe(xx, yy, zz, color = [0.75,0.75,0.75], cmap='RdYlBu_r', rstride=20, cstride=20, linewidth = 0.25)
    l_max = max(np.max(zz),-np.min(zz))
    levels = np.linspace(-l_max,l_max,21)
    ax.contour(xx, yy, zz, levels = levels, cmap = 'RdYlBu_r')

    # plot decision boundary
    ax.contour(xx, yy, zz, levels = [0], colors=['k'])

    ax.set_proj_type('ortho')

    ax.set_xlim(xx.min(),xx.max())
    ax.set_ylim(yy.min(),yy.max())

    plt.tight_layout()
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('f($x_1$,$x_2$)')
    ax.view_init(azim=-120, elev=30)
    ax.grid(False)

# ========================
# Wireframe plot of mask
# ========================
def plot_3D_mask(xx, yy, mask):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))

    ax.plot_wireframe(xx, yy, mask, cmap='RdYlBu_r', rstride=20, cstride=20, linewidth = 0.25)

    ax.set_proj_type('ortho')

    ax.set_xlim(xx.min(),xx.max())
    ax.set_ylim(yy.min(),yy.max())
    ax.set_zlim(0,1.2)

    plt.tight_layout()
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('[0,1]')
    ax.set_zticks([0,1])
    ax.view_init(azim=-120, elev=30)
    ax.grid(False)

# ========================
# 2D contour plot
# ========================
def plot_2D_contour(xx,yy,zz,mask):
    # Create color maps
    rgb = [[255, 238, 255],  # red
           [219, 238, 244]]  # blue
    rgb = np.array(rgb)/255.0

    cmap_light = ListedColormap(rgb)

    fig, ax = plt.subplots(figsize = (8,8))
    l_max = max(np.max(zz),-np.min(zz))
    levels = np.linspace(-l_max,l_max,21)
    ax.contourf(xx, yy, mask, cmap=cmap_light)
    ax.contour(xx, yy, zz, levels = levels, cmap = 'RdYlBu_r')

    # plot decision boundary
    ax.contour(xx, yy, zz, levels = [0], colors=['k'])

    # Figure decorations
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    # plt.axis('equal')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.show()

#%%
plt.close('all')
num = 500
x = np.linspace(-4,4,num)
y = np.linspace(-4,4,num)
xx,yy = np.meshgrid(x,y);

## x1 + 1 > 0
zz = -xx - 1
# satisfy the inequality: 1
# otherwise: 0
mask_less_than_0 = (zz < 0) + 0

plot_3D_f_xy(xx, yy, zz)
plot_3D_mask(xx, yy, mask_less_than_0)
plot_2D_contour(xx, yy, zz, mask_less_than_0)

## -1 < x1 < 2
zz = np.abs(xx - 0.5) - 1.5
mask_less_than_0 = (zz < 0) + 0

plot_3D_f_xy(xx, yy, zz)
plot_3D_mask(xx, yy, mask_less_than_0)
plot_2D_contour(xx, yy, zz, mask_less_than_0)

## x2 < 0 or x2 > 2
zz = -np.abs(yy - 1) + 1
mask_less_than_0 = (zz < 0) + 0

plot_3D_f_xy(xx, yy, zz)
plot_3D_mask(xx, yy, mask_less_than_0)
plot_2D_contour(xx, yy, zz, mask_less_than_0)

## x1 - x2 + 1 < 0
zz = xx - yy + 1
mask_less_than_0 = (zz < 0) + 0

plot_3D_f_xy(xx, yy, zz)
plot_3D_mask(xx, yy, mask_less_than_0)
plot_2D_contour(xx, yy, zz, mask_less_than_0)

## x1 > 2*x2
zz = - xx + 2*yy
mask_less_than_0 = (zz < 0) + 0

plot_3D_f_xy(xx, yy, zz)
plot_3D_mask(xx, yy, mask_less_than_0)
plot_2D_contour(xx, yy, zz, mask_less_than_0)

## |x1 + x2| < 1
zz = np.abs(xx + yy) - 1
mask_less_than_0 = (zz < 0) + 0

plot_3D_f_xy(xx, yy, zz)
plot_3D_mask(xx, yy, mask_less_than_0)
plot_2D_contour(xx, yy, zz, mask_less_than_0)

## |x1| + |x2| < 2
zz = np.abs(xx) + np.abs(yy) - 2
mask_less_than_0 = (zz < 0) + 0

plot_3D_f_xy(xx, yy, zz)
plot_3D_mask(xx, yy, mask_less_than_0)
plot_2D_contour(xx, yy, zz, mask_less_than_0)

## x1**2 + x2**2 < 4
zz = xx**2 + yy**2 - 4
mask_less_than_0 = (zz < 0) + 0

plot_3D_f_xy(xx, yy, zz)
plot_3D_mask(xx, yy, mask_less_than_0)
plot_2D_contour(xx, yy, zz, mask_less_than_0)


#%% Bk3_Ch6_04
import numpy as np
import matplotlib.pyplot as plt

u = np.linspace(0, np.pi, 30)
v = np.linspace(0, 2 * np.pi, 30)

x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (12, 12))
# ax.set_aspect('equal')

ax.plot_wireframe(x, y, z, colors = [0.6, 0.6, 0.6], lw = 0.6)
ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{x_3}$')
ax.view_init(azim=-125, elev=30)
plt.show()
plt.close()

#%% Bk3_Ch6_05
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,8*np.pi, 200)

# parametric equation of spiral
x1 = np.cos(t)
x2 = np.sin(t)
x3 = t

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
ax.plot(x1, x2, x3)

plt.show()
ax.set_proj_type('ortho')

ax.set_xlim(-1.5,1.5)
ax.set_ylim(-1.5,1.5)
ax.set_zlim(0,t.max())

plt.tight_layout()
ax.set_xlabel(r'$\it{x_1}$')
ax.set_ylabel(r'$\it{x_2}$')
ax.set_zlabel(r'$\it{x_3}$')

ax.view_init(azim=-135, elev=30)
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})


#%%  Bk3_Ch6_06
from sympy import *
from sympy.plotting import plot3d_parametric_line
import math

t = symbols('t')

# parametric equation of spiral
x1 = cos(t)
x2 = sin(t)
x3 = t

plot3d_parametric_line(x1, x2, x3, (t, 0, 8*math.pi))






#%%




#%%




#%%




#%%




#%%




#%%




#%%




#%%




#%%




#%%




#%%




#%%




#%%




#%%




#%%







