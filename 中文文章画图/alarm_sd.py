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

fontpath = "/usr/share/fonts/truetype/arphic/"
font1 = FontProperties(fname=fontpath+"SimSun.ttf", size = 14)
font2 = FontProperties(fname=fontpath+"SimSun.ttf", size = 20)

fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
fonte1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 24)

t = [20,40,60,80,100,120,140,160,180,200]
f_ndMLP =  [7.75,8.05,10.03,9.42,8.97,6.84,7.45,11.7,9.57,6.38]
#[7.75,8.05,9.03,9.42,10.97,10.68,11.87,12.7,13.57,14.38]#无改
s_dMLP = [86.22, 85.49, 86.94, 88.38, 86.73, 81.07, 83.12, 84.15, 78.91,81.07]

f_ndLSTM = [8.21,7.45,5.78,5.47,5.93, 5.62,7.45,7.9,7.45,5.32]
#f_nd = [8.21,8.05,10.03,9.42,8.97,9.68,10.87,11.7,9.57,11.38]
s_dLSTM = [95.37, 95.37, 91.56, 94.85, 88.47, 97.12, 96.30, 96.91, 94.96,95.68]
#s_d = [95.47, 94.49, 93.94, 92.38, 93.73, 91.07, 90.12, 89.15, 89.91,88.07]

fig,axs = plt.subplots(2,1,figsize=(8,5.8))
fig.subplots_adjust(hspace =0.5)#调节两个子图间的距离

axs[0].plot(t,s_dLSTM,'kD-',linewidth=2, markersize=8,label=r'LSTM')
axs[0].plot(t,s_dMLP,'ko-',linewidth=2,markersize=8, label=r'MLP')

font2 = FontProperties(fname=fontpath+"SimSun.ttf", size = 22)
axs[0].set_ylabel(r'成功预测率(%)',fontproperties=font2)

fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[0].set_xlabel(r'${\Delta t}_\mathrm{alarm}$(ms)',fontproperties=fonte)

axs[0].tick_params(direction='in',axis='both',labelsize=16,width=5)
labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(22) for label in labels]#刻度值字号

font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf",size = 16)
legend1 =axs[0].legend(loc='center',prop=font1, bbox_to_anchor=(0.72,0.62))
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none') # 设置图例legend背景透明

fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[0].set_title('(a)', loc='left',fontproperties=fonte)

axs[0].spines['bottom'].set_linewidth(2)   ###设置底部坐标轴的粗细
axs[0].spines['left'].set_linewidth(2)     ####设置左边坐标轴的粗细
axs[0].spines['right'].set_linewidth(2)    ###设置右边坐标轴的粗细
axs[0].spines['top'].set_linewidth(2)      ####设置上部坐标轴的粗细

axs[0].set_xticks(t)

axs[1].plot(t,f_ndLSTM,'kD-',linewidth=3,markersize=8, label=r'LSTM')
axs[1].plot(t,f_ndMLP,'ko-',linewidth=3,markersize=8, label=r'MLP')

font2 = FontProperties(fname=fontpath+"SimSun.ttf", size = 22)
axs[1].set_ylabel(r'错误预测率(%)',fontproperties=font2)

fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[1].set_xlabel(r'${\Delta t}_\mathrm{alarm}$(ms)',fontproperties=fonte)

axs[1].tick_params(direction='in',axis='both',labelsize=16,width=5)
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(22) for label in labels]#刻度值字号

fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=17)
legend1 =axs[1].legend(loc='center',prop=fonte, bbox_to_anchor=(0.57,0.8 ),)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none') # 设置图例legend背景透明

fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[1].set_title('(b)', loc='left',fontproperties=fonte)


axs[1].spines['bottom'].set_linewidth(2)   ###设置底部坐标轴的粗细
axs[1].spines['left'].set_linewidth(2)     ####设置左边坐标轴的粗细
axs[1].spines['right'].set_linewidth(2)    ###设置右边坐标轴的粗细
axs[1].spines['top'].set_linewidth(2)      ####设置上部坐标轴的粗细
axs[1].set_xticks(t)

home        = os.environ['HOME']
picturepath = home+'/tmp/'
#plt.tight_layout()
out_fig=plt.gcf()
#out_fig.savefig(picturepath+'deltaT.eps',format='eps',dpi=1000,bbox_inches='tight')
out_fig.savefig(picturepath+'fig6.svg',format='svg',bbox_inches='tight')
out_fig.savefig(picturepath+'fig6.pdf',format='pdf',bbox_inches='tight')
plt.show()



