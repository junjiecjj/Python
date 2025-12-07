#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:20:05 2020

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import os


fontt  = {'family':'Times New Roman','style':'normal','size':28,'weight':'bold'}
fontt1 = {'style':'normal','size':17}
fontt2 = {'style':'normal','size':20}

t = [20,40,60,80,100,120,140,160,180,200]
f_nd = [8.21,7.45,5.78,5.47,5.93, 5.62,7.45,7.9,7.45,5.32]
#[7.75,8.05,10.03,9.42,8.97,9.68,10.87,11.7,9.57,6.38]
#f_nd = [8.21,8.05,10.03,9.42,8.97,9.68,10.87,11.7,9.57,11.38]

s_d = [95.37, 95.37, 91.56, 94.85, 88.47, 97.12, 96.30, 96.91, 94.96,95.68]
#[86.22, 85.49, 86.94, 88.38, 86.73, 81.07, 83.12, 84.15, 78.91,81.07]
#s_d = [95.47, 94.49, 93.94, 92.38, 93.73, 91.07, 90.12, 89.15, 89.91,88.07]

fig,axs = plt.subplots(1,1,figsize=(6,4))
ax = axs.twinx()

p1,= axs.plot(t,s_d,'ro-',linewidth=3, label=r'$R_{succ}$')
p2,= ax.plot(t,f_nd, 'b*-',linewidth=3, label=r'$R_{false}$')

axs.yaxis.label.set_color(p1.get_color())
ax.yaxis.label.set_color(p2.get_color())

axs.set_ylabel(r'Sucessful alarm rate(%)',fontdict=fontt2)
ax.set_ylabel(r'False alarm rate(%)',fontdict=fontt2)
axs.set_xlabel(r'${\Delta t}_\mathrm{alarm}$(ms)',fontdict=fontt2)

axs.set_xticks(t)


legend1 =fig.legend(loc='center',prop=fontt1, bbox_to_anchor=(0.75,0.18), bbox_transform=ax.transAxes)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none') # 设置图例legend背景透明

axs.tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=5)
labels = axs.get_xticklabels() + axs.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels]#刻度值字号

ax.tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=5)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(21) for label in labels]#刻度值字号

axs.spines['bottom'].set_linewidth(2)   ###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(2)     ####设置左边坐标轴的粗细
axs.spines['right'].set_linewidth(2)    ###设置右边坐标轴的粗细
axs.spines['top'].set_linewidth(2)      ####设置上部坐标轴的粗细

home        = os.environ['HOME']
picturepath = home+'/Resultpicture/'
plt.tight_layout()
out_fig=plt.gcf()
out_fig.savefig(picturepath+'deltaT.eps',format='eps',dpi=1000,bbox_inches='tight')
plt.show()



