#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:38:19 2024

@author: jack



"""
#%%======================================================================================================================
############     Chapter 26 SciPy 数学运算 | Book 1《编程不难》
##=======================================================================================================================


import numpy as np
# from scipy.stats import gumbel_r
import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
# import matplotlib
from matplotlib.font_manager import FontProperties
# from pylab import tick_params
# import copy
# from matplotlib.pyplot import MultipleLocator

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)

fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

#%% 一元高斯分布 PDF，均值 µ 影响
x_array = np.linspace(-6, 6, 200)
# 设定均值一系列取值
mu_array = np.linspace(-4, 4, 9)

# 颜色映射
colors = cm.RdYlBu(np.linspace(0, 1, len(mu_array)))

# 均值对一元高斯分布PDF影响
fig, ax = plt.subplots(figsize = (5,4))
for idx, mu_idx in enumerate(mu_array):
    pdf_idx = norm.pdf(x_array, scale = 1, loc = mu_idx)
    legend_idx = '$\mu$ = ' + str(mu_idx)
    ax.plot(x_array, pdf_idx, color=colors[idx], label = legend_idx)

ax.legend(ncol = 3)
ax.set_xlim(x_array.min(), x_array.max())
ax.set_ylim(0, 1)
ax.set_xlabel('x')
ax.set_ylabel('PDF, $f_X(x)$')
plt.show()


#%% 一元高斯分布 PDF, 标准差 σ  影响
sigma_array = np.linspace(0.5,5,10)
# 设定标准差一系列取值

colors = cm.RdYlBu(np.linspace(0,1,len(sigma_array)))

# 标准差对一元高斯分布PDF影响
fig, ax = plt.subplots(figsize = (5,4))
for idx, sigma_idx in enumerate(sigma_array):
    pdf_idx = norm.pdf(x_array, scale = sigma_idx)
    legend_idx = '$\sigma$ = ' + str(sigma_idx)
    plt.plot(x_array, pdf_idx, color=colors[idx], label = legend_idx)

plt.legend()
ax.set_xlim(x_array.min(),x_array.max())
ax.set_ylim(0,1)
ax.set_xlabel('x')
ax.set_ylabel('PDF, $f_X(x)$')

plt.show()


#%% 二元高斯函数, sympy
from sympy import symbols, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt

x1, x2 = symbols('x1 x2')
# 定义符号变量
f_gaussian_x1x2 = exp(-x1**2 - x2**2)
# 将符号表达式转换为Python函数
f_gaussian_x1x2_fcn = lambdify([x1, x2], f_gaussian_x1x2)
xx1, xx2 = np.meshgrid(np.linspace(-3, 3, 201), np.linspace(-3, 3, 201))
ff = f_gaussian_x1x2_fcn(xx1, xx2)
# 可视化
fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(8, 8))
axs.plot_wireframe(xx1, xx2, ff, rstride = 5, cstride = 5, color = [0.8,0.8,0.8], linewidth = 1)
axs.contour(xx1, xx2, ff, levels = 20, cmap = 'RdYlBu_r', linewidth = 3)

axs.set_proj_type('ortho')
axs.view_init(azim = -120, elev = 30)
axs.grid(False)

font = FontProperties(fname = fontpath1 + "Times_New_Roman.ttf", size = 24)
axs.set_xlabel('x1', fontproperties = font)
axs.set_ylabel('x2', fontproperties = font)
axs.set_zlabel(r'$f(x1,x2)$', fontproperties = font)
axs.set_xlim(-3, 3)
axs.set_ylim(-3, 3)
axs.set_zlim(0, 1)
axs.set_box_aspect(aspect = (1, 1, 1))
# fig.savefig('二元高斯函数.svg', format='svg')
plt.show()



#%% 可视化二元高斯分布PDF, scipy.stats.multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

rho_array = [-0.9, -0.7, -0.5, -0.3, 0, 0.3, 0.5, 0.7, 0.9]

sigma_X = 1; sigma_Y = 1 # 标准差
mu_X = 0;    mu_Y = 0    # 期望

width = 4
X = np.linspace(-width, width, 321)
Y = np.linspace(-width, width, 321)
XX, YY = np.meshgrid(X, Y) # ￼￼(321, 321 )
XXYY = np.dstack((XX, YY))  # ￼￼(321, 321, 2)

# 曲面
fig = plt.figure(figsize = (15, 15))
for idx, rho_idx in enumerate(rho_array):
    # 质心
    mu    = [mu_X, mu_Y]
    # 协方差
    Sigma = [[sigma_X**2, sigma_X * sigma_Y * rho_idx],
            [sigma_X * sigma_Y * rho_idx, sigma_Y**2]]
    # 二元高斯分布
    bi_norm = multivariate_normal(mu, Sigma)
    f_X_Y_joint = bi_norm.pdf(XXYY) # ￼￼(321, 321 )

    ax = fig.add_subplot(3, 3, idx + 1, projection = '3d')
    ax.plot_wireframe(XX, YY, f_X_Y_joint, rstride=10, cstride=10, color = [0.3,0.3,0.3], linewidth = 0.25)
    ax.contour(XX, YY, f_X_Y_joint, 15, cmap = 'RdYlBu_r')

    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_zlabel('$f_{X,Y}(x,y)$')
    ax.view_init(azim=-120, elev=30)
    ax.set_proj_type('ortho')

    ax.set_xlim(-width, width)
    ax.set_ylim(-width, width)
    ax.set_zlim(f_X_Y_joint.min(), f_X_Y_joint.max())
    # ax.axis('off')

plt.tight_layout()
# fig.savefig('二元高斯分布，曲面.svg', format='svg')
plt.show()

# 平面填充等高线
fig = plt.figure(figsize = (8,8))
for idx, rho_idx in enumerate(rho_array):
    mu = [mu_X, mu_Y]
    Sigma = [[sigma_X**2, sigma_X*sigma_Y*rho_idx],
             [sigma_X*sigma_Y*rho_idx, sigma_Y**2]]

    bi_norm = multivariate_normal(mu, Sigma)
    f_X_Y_joint = bi_norm.pdf(XXYY)
    ax = fig.add_subplot(3, 3, idx + 1)
    ax.contourf(XX, YY, f_X_Y_joint, levels = 20, cmap = 'RdYlBu_r')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(-width, width)
    ax.set_ylim(-width, width)
    ax.axis('off')
plt.tight_layout()
# fig.savefig('二元高斯分布， 等高线.svg', format='svg')
plt.show()


#%%












































































































































































