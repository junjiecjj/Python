#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:13:09 2025

@author: jack
"""
import scipy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns




# 组1：正态分布，较小方差
group1 = 10*np.random.normal(loc=10, scale=5, size=20000)

# 组2：正态分布，较大方差
group2 = 5*np.random.normal(loc=50, scale=15, size=20000)

# 组3：偏态分布，较大方差
group3 = 2*np.random.gamma(shape=2, scale=10, size=20000)


##1
fig, axs = plt.subplots( figsize = (7.5, 6), constrained_layout=True)
sns.kdeplot(group1, fill=True, label='Group 1', color='blue', common_norm=False,)
sns.kdeplot(group2, fill=True, label='Group 2', color='orange', common_norm=False,)
sns.kdeplot(group3, fill=True, label='Group 3', color='green', common_norm=False,)

bw = 2
axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,}
legend1 = axs.legend(loc='upper right', borderaxespad = 0, edgecolor = 'black', prop = font, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs.set_xlabel('Value', fontdict = font, )
axs.set_ylabel('Density', fontdict = font, )
axs.set_title("Density Plot of Values by Group", fontdict = font,  )

axs.tick_params(which = 'major', axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)
axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )

# 显示图形
out_fig = plt.gcf()
plt.show()


# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()
##2
fig, axs = plt.subplots( figsize = (7.5, 6), constrained_layout=True)
sns.histplot(group1, fill=True, label='Group 1', color='blue')
sns.histplot(group2, fill=True, label='Group 2', color='orange')
sns.histplot(group3, fill=True, label='Group 3', color='green')

bw = 2
axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,}
legend1 = axs.legend(loc='upper right', borderaxespad = 0, edgecolor = 'black', prop = font, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs.set_xlabel('Value', fontdict = font, )
axs.set_ylabel('Density', fontdict = font, )
axs.set_title("Density Plot of Values by Group", fontdict = font,  )

axs.tick_params(which = 'major', axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)
axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )

# 显示图形
out_fig = plt.gcf()
plt.show()

##3
fig, axs = plt.subplots( figsize = (7.5, 6), constrained_layout=True)
axs.hist(group1, bins=30, density = True, alpha=0.5, label='Data1: N(10, 5)', color='orange')
axs.hist(group2, bins=30, density = True, alpha=0.5, label='Data1: N(50, 15)', color='blue')
axs.hist(group3, bins=30, density = True, alpha=0.5, label='Data1: N(2, 10)', color='blue')
bw = 2
axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(bw)  ###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(bw)    ###设置上部坐标轴的粗细

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 20,}
legend1 = axs.legend(loc='upper right', borderaxespad = 0, edgecolor = 'black', prop = font, facecolor = 'none',labelspacing = 0.2) ## loc = 'lower left',
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

font = {'family':'Times New Roman', 'weight' : 'normal', 'size': 25,} # 'family':'Times New Roman',
axs.set_xlabel('Value', fontdict = font, )
axs.set_ylabel('Density', fontdict = font, )
axs.set_title("Density Plot of Values by Group", fontdict = font,  )

axs.tick_params(which = 'major', axis='both', direction='in', left = True, right = True, top=True, bottom=True, width=3, length = 5,  labelsize=20, labelfontfamily = 'Times New Roman', pad = 1)
axs.grid(linestyle = (0, (5, 10)), linewidth = 0.5 )

# 显示图形
out_fig = plt.gcf()
plt.show()
