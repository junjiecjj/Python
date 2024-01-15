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
matplotlib.use('TkAgg')
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

now = "2023-12-01-22:57:36_FLSemantic"
root_dir = f"/home/jack/FL_semantic/results/{now}/cdf_pdf"

r1 = 1
c1 = 99
r1_pdf = torch.load(os.path.join(root_dir, f"round_{r1}/round={r1}_client{c1}_pdf.pt"))
r1_cdf = torch.load(os.path.join(root_dir, f"round_{r1}/round={r1}_client{c1}_cdf.pt"))

r2 = 5
c2 = 35
r2_pdf = torch.load(os.path.join(root_dir, f"round_{r2}/round={r2}_client{c2}_pdf.pt"))
r2_cdf = torch.load(os.path.join(root_dir, f"round_{r2}/round={r2}_client{c2}_cdf.pt"))


r3 = 10
c3 = 75
r3_pdf = torch.load(os.path.join(root_dir, f"round_{r3}/round={r3}_client{c3}_pdf.pt"))
r3_cdf = torch.load(os.path.join(root_dir, f"round_{r3}/round={r3}_client{c3}_cdf.pt"))


r4 = 20
c4 = 58
r4_pdf = torch.load(os.path.join(root_dir, f"round_{r4}/round={r4}_client{c4}_pdf.pt"))
r4_cdf = torch.load(os.path.join(root_dir, f"round_{r4}/round={r4}_client{c4}_cdf.pt"))



r5 = 100
c5 = 34
r5_pdf = torch.load(os.path.join(root_dir, f"round_{r5}/round={r5}_client{c5}_pdf.pt"))
r5_cdf = torch.load(os.path.join(root_dir, f"round_{r5}/round={r5}_client{c5}_cdf.pt"))



r6 = 200
c6 = 78
r6_pdf = torch.load(os.path.join(root_dir, f"round_{r6}/round={r6}_client{c6}_pdf.pt"))
r6_cdf = torch.load(os.path.join(root_dir, f"round_{r6}/round={r6}_client{c6}_cdf.pt"))





fig, axs = plt.subplots(2,1, figsize=(8, 10), constrained_layout=True)



# 生成三组随机数据
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(2, 1, 1000)
data3 = np.random.normal(-2, 1, 1000)


##===================================================== 1 =================================================
# 绘制直方图
# axs[0].hist(data1, bins = 100, density = True, histtype='bar',color='r', alpha=0.75, label='Data 1')
# axs[0].hist(data2, bins = 100, density = True, histtype='bar',color='g', alpha=0.75, label='Data 2')
# axs[0].hist(data3, bins = 100, density = True, histtype='bar',color='b', alpha=0.75, label='Data 3')

hist  = r1_pdf[0]
bin_edges = r1_pdf[1]
axs[0].stairs(hist, bin_edges,  linestyle='-', linewidth = 2,  color = 'red', label = 'round 1',)

# hist  = r2_pdf[0]
# bin_edges = r2_pdf[1]
# axs[0].stairs(hist, bin_edges,  linestyle='-', linewidth = 2,  color = '#FF6347', label = 'round 5',)


hist  = r3_pdf[0]
bin_edges = r3_pdf[1]
axs[0].stairs(hist, bin_edges,  linestyle='-', linewidth = 2,  color = '#00FF00', label = 'round 10',)


# hist  = r4_pdf[0]
# bin_edges = r4_pdf[1]
# axs[0].stairs(hist, bin_edges,  linestyle='-', linewidth = 2,  color = '#0000FF', label = 'round 20',)

# hist  = r5_pdf[0]
# bin_edges = r5_pdf[1]
# axs[0].stairs(hist, bin_edges,  linestyle='-', linewidth = 2,  color = '#FF00FF', label = 'round 100',)

# hist  = r6_pdf[0]
# bin_edges = r6_pdf[1]
# axs[0].stairs(hist, bin_edges,  linestyle='-', linewidth = 2,  color = '#800080', label = 'round 200',)

# 设置图表属性
font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 22)
axs[0].set_title('RUNOOB hist() TEST', fontproperties = font2)
axs[0].set_xlabel('Value', fontproperties = font2)
axs[0].set_ylabel('Frequency', fontproperties = font2)
axs[0].grid()

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs[0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# x_major_locator=MultipleLocator(0.02)               #把x轴的刻度间隔设置为1，并存在变量里
# axs[0].xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
axs[0].tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

axs[0].spines['bottom'].set_linewidth(1.5); ###设置底部坐标轴的粗细
axs[0].spines['left'].set_linewidth(1.5);   ####设置左边坐标轴的粗细
axs[0].spines['right'].set_linewidth(1.5);  ###设置右边坐标轴的粗细
axs[0].spines['top'].set_linewidth(1.5);    ####设置上部坐标轴的粗细

axs[0].set_xlim(-0.1, 0.1)  #拉开坐标轴范围显示投影
##===================================================== 2 =================================================
# 绘制直方图
hist  = r1_cdf[0]
bin_edges = r1_cdf[1]
axs[1].stairs(hist, bin_edges,  linestyle='-', linewidth = 2,  color = '#1E90FF', label = 'round 1',)

# hist  = r2_cdf[0]
# bin_edges = r2_cdf[1]
# axs[1].stairs(hist, bin_edges,  linestyle='-', linewidth = 2,  color = '#FF6347', label = 'round 5',)


hist  = r3_cdf[0]
bin_edges = r3_cdf[1]
axs[1].stairs(hist, bin_edges,  linestyle='-', linewidth = 2,  color = '#00FF00', label = 'round 10',)


# hist  = r4_cdf[0]
# bin_edges = r4_cdf[1]
# axs[1].stairs(hist, bin_edges,  linestyle='-', linewidth = 2,  color = '#0000FF', label = 'round 20',)

# hist  = r5_cdf[0]
# bin_edges = r5_cdf[1]
# axs[1].stairs(hist, bin_edges,  linestyle='-', linewidth = 2,  color = '#FF00FF', label = 'round 100',)


# hist  = r6_cdf[0]
# bin_edges = r6_cdf[1]
# axs[1].stairs(hist, bin_edges,  linestyle='-', linewidth = 2,  color = '#800080', label = 'round 200',)


# 设置图表属性
font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 22)
axs[1].set_title('RUNOOB hist() TEST', fontproperties = font2)
axs[1].set_xlabel('Value', fontproperties = font2)
axs[1].set_ylabel('Frequency', fontproperties = font2)
axs[1].grid()

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs[1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# x_major_locator=MultipleLocator(0.02)               #把x轴的刻度间隔设置为1，并存在变量里
# axs[1].xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
axs[1].tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
labels = axs[1].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

axs[1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

# axs[1].set_xlim(-0.05, 0.05)  #拉开坐标轴范围显示投影
# 显示图表

out_fig = plt.gcf()
out_fig .savefig('./fig9.svg',   bbox_inches = 'tight')
out_fig .savefig('./fig9.png', dpi = 500,  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.close()































































































































































































































































































































































































































