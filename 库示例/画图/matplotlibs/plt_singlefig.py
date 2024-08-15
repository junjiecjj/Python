#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:39:20 2023

@author: jack
"""

import matplotlib
# matplotlib.get_backend()
# matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)

fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
##==========================================  1 ===================================================

X = np.arange(0, 2, 0.1)

s1 = np.sin(2*np.pi*X)
s2 = np.cos(2*np.pi*X)
s3 = np.tan(2*np.pi*X)

losslog = np.zeros((len(X),3))
losslog[:,0] = s1
losslog[:,1] = s2
losslog[:,2] = s3

loss = "MSE"
fig = plt.figure(figsize=(8, 6), constrained_layout=True)
for i, l in enumerate(loss):
    label = '{} Loss'.format(l)
    # fig = plt.figure(constrained_layout=True)
    plt.plot(X, losslog[:, i], label=label,  marker='o', markersize = 12, markevery=10)


font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 30)
plt.xlabel('Epoch',fontproperties=font, labelpad=2.5) # labelpad：类型为浮点数，默认值为None，即标签与坐标轴的距离。
plt.ylabel('Training loss',fontproperties=font)
plt.title(label,fontproperties=font)

#font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# 图脊 (spine) 是指图形中的边框线，用于界定图形的边界。图脊由四条边框线组成：上脊 (top spine)、下脊 (bottom spine)、左脊 (left spine) 和右脊 (right spine)。这些脊线可以通过 Matplotlib 的 Axes.spines 属性进行访问和定制.
ax=plt.gca()#获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
ax.spines['left'].set_color('b')  ### 设置边框线颜色
ax.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(5);   ###设置上部坐标轴的粗细
ax.spines['top'].set_color('m')  ### 设置边框线颜色
ax.spines['right'].set_color('r')  ### 设置边框线颜色

ax.tick_params( axis='both', direction='in', top=True, right=True, width=10, length = 10, color='blue', labelsize=25, labelcolor = "red",  rotation=25, pad = 2, ) # pad 刻度线与刻度值之间的距离
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(20) for label in labels] #刻度值字号

ax.grid(color = 'black', alpha = 0.3, linestyle = (0, (5, 10)), linewidth = 1.5 )
filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
# out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()


##==========================================  2 ===================================================

X = np.arange(0, 2, 0.1)

s1 = np.sin(2*np.pi*X)
s2 = np.cos(2*np.pi*X)
s3 = np.tan(2*np.pi*X)



losslog = np.zeros((len(X),3))
losslog[:,0] = s1
losslog[:,1] = s2
losslog[:,2] = s3

loss = "MSE"

fig, axs = plt.subplots(1,1, figsize=(8, 6), constrained_layout=True)
for i, l in enumerate(loss):
    label = '{} Loss'.format(l)
    # fig = plt.figure(constrained_layout=True)
    axs.plot(X, losslog[:, i], label=label)

font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs.set_xlabel('Epoch',fontproperties=font)
axs.set_ylabel('Training loss',fontproperties=font)
axs.set_title(label, fontproperties=font)
#font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
#  edgecolor='black',
# facecolor = 'y', # none设置图例legend背景透明
legend1 = axs.legend(loc='best',  prop=font1, bbox_to_anchor=(0.5, -0.2), ncol = 3, facecolor = 'y', edgecolor = 'b', labelcolor = 'r', borderaxespad=0,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
axs.spines['left'].set_color('b')     ### 设置边框线颜色
axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细
axs.spines['top'].set_color('r')      ### 设置边框线颜色

axs.tick_params(direction='in',axis='both',top=True,right=True, labelsize=16, width=6, labelcolor = "red", color='blue', rotation=25, pad = 20)
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels] #刻度值字号


filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
# out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()



##==========================================  2 ===================================================















