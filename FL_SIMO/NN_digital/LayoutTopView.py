#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:14:30 2024

@author: jack
"""


# import sys
import numpy as np
# import scipy
# import cvxpy as cpy
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# from matplotlib.pyplot import MultipleLocator
import random

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


UserNumAroundAp = 4

r1 = 100

BS_locate = np.array([0, 0]) # (APx, APy) = (0, 0)
# (RISx, RISy) = (51, 0)

## AP和RIS的圆
theta = np.arange(0, 2 * np.pi, 0.01)
xAP = BS_locate[0] + r1 * np.cos(theta)
yAP = BS_locate[1] + r1 * np.sin(theta)


## AP周围的用户
K = 100
radius = np.random.rand(K, 1) * 80 + 20
angle = np.random.rand(K, 1) * 2 * np.pi
users_locate_x = radius * np.cos(angle) + BS_locate[0]
users_locate_y = radius * np.sin(angle) + BS_locate[1]
users_locate = np.hstack((users_locate_x, users_locate_y))



#%% 画图
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
axs.plot(xAP, yAP, color = 'gray', linestyle='--', lw = 4,   )
axs.plot(BS_locate[0], BS_locate[1], linestyle='none', marker = "s",  markersize = 20, mew = 4, mec = 'green', mfc = 'cyan', )

idx = list(range(K))
activate_idx = random.sample(idx, 6)
axs.plot(users_locate_x[activate_idx], users_locate_y[activate_idx], color='red', linestyle='none',  marker = "o",  markersize = 17,  )
deactivate_idx = [x for x in idx if x not in activate_idx]
axs.plot(users_locate_x[deactivate_idx], users_locate_y[deactivate_idx], color='gray', linestyle='none',  marker = "o",  markersize = 12,  )

# 圆心和圆心字母
font1 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30, }
font1 = FontProperties(fname=fontpath+"simsun.ttf", size=30)
# axs.annotate(f'AP({BS_locate[0]},{BS_locate[1]})',xy=(BS_locate[0], BS_locate[1]), xytext=(BS_locate[0],BS_locate[1]-12), textcoords='data', fontproperties = font1, )
axs.text(BS_locate[0]-12, BS_locate[1]-12, f'基站({BS_locate[0]},{BS_locate[1]})', fontproperties = font1, color= "k" )

## 圆的半径
theta1 =  np.radians(60)
x1 = BS_locate[0] + r1 * np.cos(theta1)
y1 = BS_locate[1] + r1 * np.sin(theta1)
axs.annotate("", xy=(BS_locate[0], BS_locate[1]),  xytext=(x1, y1), size=20, va="center", ha="center",  arrowprops=dict(color='orange',  arrowstyle="<->",  connectionstyle="arc3,rad=0",  linewidth = 3, alpha = 0.6))
font1 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30,}
axs.text((BS_locate[0]+x1)/2, (BS_locate[1]+y1)/2, r'$R$'+'=100 m', fontproperties = font1, color= "k" )

axs.set_xticks([])
axs.set_yticks([])

axs.spines['top'].set_visible(False)
axs.spines['bottom'].set_visible(False)
axs.spines['left'].set_visible(False)
axs.spines['right'].set_visible(False)

out_fig = plt.gcf()
# out_fig.savefig('./Figures/TopView.pdf' )
plt.show()




























































































































































































































