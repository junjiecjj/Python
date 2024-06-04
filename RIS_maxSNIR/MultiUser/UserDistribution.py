#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 14:26:45 2024

@author: jack
"""

import sys
import numpy as np
import scipy
import cvxpy as cpy
import matplotlib.pyplot as plt
import math
import matplotlib
from matplotlib.font_manager import FontProperties
# from pylab import tick_params
from matplotlib.pyplot import MultipleLocator
# import scipy.constants as CONSTANTS


filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


UserNumAroundAp = 4
UserNumAroundRIS = 4
r1 = 20
r2 = 10
(APx, APy) = (0, 0)
(RISx, RISy) = (51, 0)


## AP和RIS的圆
theta = np.arange(0, 2 * np.pi, 0.01)
xAP = APx + r1 * np.cos(theta)
yAP = APy + r1 * np.sin(theta)
xRIS = RISx + r2 * np.cos(theta)
yRIS = RISy + r2 * np.sin(theta)


## AP周围的用户
# thetaAP = np.random.uniform(0, 2*np.pi, UserNumAroundAp)  # 随机
# thetaAP = np.random.rand(UserNumAroundAp) * np.pi * 2       # 随机
thetaAP = np.arange(np.pi/UserNumAroundAp, 2*np.pi,  2*np.pi/UserNumAroundAp)  # 均匀
dAP = np.random.uniform(r1 - 0, r1, UserNumAroundAp)
# dAP = np.random.rand(UserNumAroundAp) * r1
xAPuser = APx + dAP*np.cos(thetaAP)
yAPuser = APy + dAP*np.sin(thetaAP)


## RIS周围的用户
# thetaRIS = np.random.uniform(np.pi/2, 1.5 * np.pi, UserNumAroundRIS)   # 随机
# thetaRIS = np.random.rand(UserNumAroundRIS) * np.pi * 2                    # 随机
thetaRIS = np.arange(np.pi/2+np.pi/(UserNumAroundRIS+1), 3*np.pi/2-0.01, np.pi/(UserNumAroundRIS+1))  # 均匀
# dRIS = np.random.rand(UserNumAroundRIS) * r2
dRIS = np.random.uniform(r2 - 0, r2, UserNumAroundRIS)
xRISuser = RISx + dRIS*np.cos(thetaRIS)
yRISuser = RISy + dRIS*np.sin(thetaRIS)



## 用户的名字
# UserName = [f"user{i}" for i in range(UserNumAroundAp+UserNumAroundRIS)]
UserName = [f"user{i}" for i in [3,5,7,1,8,6,4,2]]
x = np.append(xAPuser, xRISuser)
y = np.append(yAPuser, yRISuser)



#%% 画图
fig, axs = plt.subplots(1, 1, figsize=(20, 10))
axs.plot(APx, APy, linestyle='none', marker = "s",  markersize = 20, markeredgewidth = 4, markerfacecolor='gray',  markerfacecoloralt='red',  markeredgecolor='brown', fillstyle='full',)
axs.plot(RISx, RISy, linestyle='none', marker = "s", markersize = 20, markeredgewidth = 4, markerfacecolor='gray',  markerfacecoloralt='red',  markeredgecolor='brown', fillstyle='full',)

axs.plot(xAP, yAP, color = 'b', linestyle='--', lw = 4,   )
axs.plot(xRIS, yRIS, color='b', linestyle='--',  lw = 4, )

axs.plot(xAPuser, yAPuser, color='r', linestyle='none',  marker = "o",  markersize = 20,  )
axs.plot(xRISuser, yRISuser, color='r', linestyle='none',  marker = "o",  markersize = 20, )

# 两个圆心和圆心字母
font1 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22, }
axs.annotate(f'AP({APx},{APy})',xy=(APx, APy), xytext=(APx-1,APy-3), textcoords='data', fontproperties = font1, )
axs.annotate(f'RIS({RISx},{RISy})',xy=(RISx, RISy), xytext=(RISx-1,RISy-3), textcoords='data', fontproperties = font1, )

## 两圆心之间的连线和d0
axs.annotate("", xy=(APx, APy),  xytext=(RISx, RISy),  size=20, va="center", ha="center",  arrowprops=dict(color='#373331',  arrowstyle="<->",  connectionstyle="arc3,rad=0",  linewidth = 3, alpha = 0.6))
font1 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30,}
axs.text((APx+RISx)/2, (APy+RISy)/2+1, r'$d_0$', fontproperties = font1, color= "k" )


## 第一个圆的半径
theta1 =  np.radians(60)
x1 = APx + r1 * np.cos(theta1)
y1 = APy + r1 * np.sin(theta1)
axs.annotate("", xy=(APx, APy),  xytext=(x1, y1), size=20, va="center", ha="center",  arrowprops=dict(color='#373331',  arrowstyle="<->",  connectionstyle="arc3,rad=0",  linewidth = 3, alpha = 0.6))
font1 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30,}
axs.text((APx+x1)/2, (APy+y1)/2, r'$d_1$', fontproperties = font1, color= "k" )


## 第二个圆的半径
theta1 =  np.radians(30)
x2 = RISx + r2 * np.cos(theta1)
y2 = RISy + r2 * np.sin(theta1)
axs.annotate("", xy=(RISx, RISy),  xytext=(x2, y2), size=20, va="center", ha="center", arrowprops=dict(color='#373331',  arrowstyle="<->",  connectionstyle="arc3,rad=0",  linewidth = 3, alpha = 0.6))
font1 = {'family': 'Times New Roman', 'style': 'normal', 'size': 30,}
axs.text((RISx+x2)/2-1, (RISy+y2)/2+1, r'$d_2$', fontproperties = font1, color= "k" )




font1 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22, }
for i in range(len(UserName)):
    axs.annotate(UserName[i], xy = (x[i], y[i]), xytext = (x[i]-1, y[i]-2), fontproperties = font1,  ) # 这里xy是需要标记的坐标，xytext是对应的标签坐标


# font1 = { 'style': 'normal', 'size': 22, 'color':'blue',}
font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 30)
# axs.set_xlabel( "X(m)", fontproperties=font2, ) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
# axs.set_ylabel('Y(m)', fontproperties=font2, )
axs.set_title("Simulation setup of the multiuser case (top view)", fontproperties=font2)

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 20}
# # font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
# legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

# x_major_locator = MultipleLocator(5)               #把x轴的刻度间隔设置为1，并存在变量里
# axs.xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
# axs.tick_params(direction='in', axis='both', top=True, right=True,labelsize=25, width=3,)
# labels = axs.get_xticklabels() + axs.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(25) for label in labels]  # 刻度值字号

# axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )
# axs.spines['bottom'].set_linewidth(1.5)    ###设置底部坐标轴的粗细
# axs.spines['left'].set_linewidth(1.5)      ####设置左边坐标轴的粗细
# axs.spines['right'].set_linewidth(1.5)     ###设置右边坐标轴的粗细
# axs.spines['top'].set_linewidth(1.5)       ####设置上部坐标轴的粗细

axs.set_xticks([])
axs.set_yticks([])

axs.spines['top'].set_visible(False)
axs.spines['bottom'].set_visible(False)
axs.spines['left'].set_visible(False)
axs.spines['right'].set_visible(False)

out_fig = plt.gcf()
out_fig.savefig('fig6.eps' )
plt.show()




























































































































































































































