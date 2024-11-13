#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:53:49 2023

@author: jack

画出给定的离散数据的概率密度函数和累计分布函数。

"""

import os
import sys
import matplotlib
# matplotlib.get_backend()
# matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
from matplotlib.font_manager import FontProperties
# from pylab import tick_params
# import copy
from matplotlib.pyplot import MultipleLocator

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

now = "2023-12-04-13:09:31_FLSemantic"
root_dir = f"/home/jack/FL_semantic/results/{now}/cdf_pdf"

r1 = 1
c1 = 99
r1_pdf = torch.load(os.path.join(root_dir, f"round_{r1}/round={r1}_client{c1}.pt"), weights_only=False)
# r1_pdf[torch.where(abs(r1_pdf)>0.1)] = 0

r2 = 5
c2 = 35
r2_pdf = torch.load(os.path.join(root_dir, f"round_{r2}/round={r2}_client{c2}.pt"), weights_only=False)
# r2_pdf[torch.where(abs(r2_pdf)>0.1)] = 0


r3 = 10
c3 = 75
r3_pdf = torch.load(os.path.join(root_dir, f"round_{r3}/round={r3}_client{c3}.pt"), weights_only=False)
# r3_pdf[torch.where(abs(r3_pdf)>0.1)] = 0


# r4 = 20
# c4 = 58
# r4_pdf = torch.load(os.path.join(root_dir, f"round_{r4}/round={r4}_client{c4}.pt"), weights_only=False)
# # r4_pdf[torch.where(abs(r4_pdf)>0.1)] = 0


# r5 = 100
# c5 = 34
# r5_pdf = torch.load(os.path.join(root_dir, f"round_{r5}/round={r5}_client{c5}.pt"), weights_only=False)
# # r5_pdf[torch.where(abs(r5_pdf)>0.1)] = 0


# r6 = 200
# c6 = 78
# r6_pdf = torch.load(os.path.join(root_dir, f"round_{r6}/round={r6}_client{c6}.pt"), weights_only=False)
# # r6_pdf[torch.where(abs(r6_pdf)>0.1)] = 0



fig, axs = plt.subplots(1,2, figsize=(12, 5), constrained_layout=True)
##===================================================== 1 =================================================

axs[0].hist(r1_pdf, bins = 1000, density = True, histtype='step',color='red', linewidth = 3,  alpha=0.8,  label='Round 1')

# axs[0].hist(r2_pdf, bins = 500, density = True, histtype='step',color='#1E90FF', alpha=0.75, label='round 5')

axs[0].hist(r3_pdf, bins = 1000, density = True, histtype='step',color='blue', linewidth = 1.5,  alpha=0.75, label='Round 100')

# axs[0].hist(r4_pdf, bins = 10, density = True, histtype='step',color='#0000FF', alpha=0.75, label='round 20')

# axs[0].hist(r5_pdf, bins = 1000, density = True, histtype='step', color='#FF00FF', alpha=0.75, label='round 100')

# axs[0].hist(r6_pdf, bins = 1000, density = True, histtype='step',color='#800080', alpha=0.8, label='round 200')

##=====================================================

# hist, bin_edges = np.histogram(r1_pdf, bins = 500, ) # density = True
# axs[0].stairs(hist, bin_edges,  linestyle='-', linewidth = 3,  color = "red", label = 'Round 1',)

# hist, bin_edges = np.histogram(r3_pdf, bins = 500, ) # density = True
# axs[0].stairs(hist, bin_edges,  linestyle='-', linewidth = 3,  color = "blue", label = 'Round 100',)



# 设置图表属性
font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 35)
# axs[0].set_title('RUNOOB hist() TEST', fontproperties = font2)
# axs[0].set_xlabel('Value', fontproperties = font2)
axs[0].set_ylabel('概率密度', fontproperties = font2)
axs[0].grid(linestyle = (0, (5, 10)), linewidth = 0.5 )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 27}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs[0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

axs[0].tick_params(direction='in', axis='both', top=True, right=True, labelsize=30, width=3,)
labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(30) for label in labels]  # 刻度值字号

axs[0].spines['bottom'].set_linewidth(2.5); ###设置底部坐标轴的粗细
axs[0].spines['left'].set_linewidth(2.5);   ####设置左边坐标轴的粗细
axs[0].spines['right'].set_linewidth(2.5);  ###设置右边坐标轴的粗细
axs[0].spines['top'].set_linewidth(2.5);    ####设置上部坐标轴的粗细

axs[0].set_xlim(-0.1, 0.1)  #拉开坐标轴范围显示投影
##===================================================== 2 =================================================
# 绘制直方图
axs[1].hist(r1_pdf, bins = 1000, density = True, histtype='step',color='red', linewidth = 3, cumulative=True, alpha=0.75, label='round 1')

# axs[1].hist(r2_pdf, bins = 200, density = True, histtype='step',color='#1E90FF', cumulative=True, alpha=0.75, label='round 5')

axs[1].hist(r3_pdf, bins = 1000, density = True, histtype='step',color='blue',linewidth = 3,  cumulative=True, alpha=0.75, label='round 100')

# axs[1].hist(r4_pdf, bins = 1000, density = True, histtype='step',color='#0000FF', cumulative=True, alpha=0.75, label='round 20')

# axs[1].hist(r5_pdf, bins = 1000, density = True, histtype='step',color='#FF00FF', cumulative=True, alpha=0.75, label='round 100')

# axs[1].hist(r6_pdf, bins = 200, density = True, histtype='step',color='#800080', cumulative=True, alpha=0.75, label='round 200')

# 设置图表属性
font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 30)
# axs[1].set_title('RUNOOB hist() TEST', fontproperties = font2)
axs[1].set_xlabel('值', fontproperties = font2)
axs[1].set_ylabel('概率分布', fontproperties = font2)
axs[1].grid(linestyle = (0, (5, 10)), linewidth = 0.5 )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 25}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs[1].legend(loc='lower right', borderaxespad=0, edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明

# x_major_locator=MultipleLocator(0.02)               #把x轴的刻度间隔设置为1，并存在变量里
# axs[1].xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
axs[1].tick_params(direction='in', axis='both', top=True,right=True,labelsize=30, width=3,)
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(30) for label in labels]  # 刻度值字号

axs[1].spines['bottom'].set_linewidth(2.5);###设置底部坐标轴的粗细
axs[1].spines['left'].set_linewidth(2.5);####设置左边坐标轴的粗细
axs[1].spines['right'].set_linewidth(2.5);###设置右边坐标轴的粗细
axs[1].spines['top'].set_linewidth(2.5);####设置上部坐标轴的粗细

axs[0].set_xlim(-0.1, 0.09)  #拉开坐标轴范围显示投影
x_major_locator=MultipleLocator(0.04)               #把x轴的刻度间隔设置为1，并存在变量里
axs[0].xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数

axs[1].set_xlim(-0.1, 0.09)  #拉开坐标轴范围显示投影
x_major_locator=MultipleLocator(0.04)               #把x轴的刻度间隔设置为1，并存在变量里
axs[1].xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数

fontt = {'family':'Times New Roman','style':'normal','size':30}
# plt.suptitle("non-IID MNIST,  AutoEncoder", fontproperties = fontt, )
out_fig = plt.gcf()

# out_fig.savefig('./eps/fig9.eps',   bbox_inches = 'tight')
out_fig.savefig('./fig9.png', dpi = 1000,  bbox_inches = 'tight')
# out_fig.savefig('./eps/fig9.jpg', dpi = 1000,  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()
plt.close()




























































































































































































































































































































































































































