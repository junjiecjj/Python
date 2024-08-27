#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:14:30 2023

@author: jack
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
#内存分析工具
from memory_profiler import profile
import objgraph
import gc

#### 本项目自己编写的库
# sys.path.append("..")
# from  ColorPrint import ColoPrint
# color =  ColoPrint()

fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
color = ['#1E90FF','#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE' ,'#00CED1','#CD5C5C','#7B68EE', '#0000FF', '#FF0000','#808000' ]


Compress       =  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  ]
SNR            =  [ -2, -1, 0,  1,  2 , 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 28, 32, 40]



#===========================================  调整图与图之间的间距 ===============================================


fig, axs = plt.subplots(2, 4, figsize=(32, 16) ,constrained_layout=True) # ,constrained_layout=True
# plt.subplots(constrained_layout=True)的作用是:自适应子图在整个figure上的位置以及调整相邻子图间的间距，使其无重叠。
#=============================================== [0, 0] ======================================================
 # cap = 0.5 * math.log2(1 + 10**(snr/10.0) )
 # filesize = cap * comp * im.size
 # no R_min
 # Raw data: bmp
Res_5_2_exp = np.array([
    [ 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.477, 22.336, 28.426,  ],
    [ 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.701, 20.760, 22.534, 24.103, 25.655, 27.120, 28.532, 33.856, 38.812, 43.499, 51.992,  ],
    [ 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.821, 21.654, 23.982, 26.169, 28.213, 30.174, 32.135, 34.058, 35.926, 37.708, 39.496, 41.249, 48.295, 53.704, 57.784, 60.555,  ],
    [ 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.447, 21.583, 24.409, 27.065, 29.598, 32.086, 34.509, 36.924, 39.219, 41.470, 43.762, 46.053, 48.413, 50.337, 52.096, 57.795, 60.370, 60.584, 60.588,  ],
    [ 18.304, 18.304, 18.304, 18.304, 18.304, 18.304, 18.325, 21.443, 24.712, 27.724, 30.645, 33.556, 36.453, 39.173, 41.931, 44.705, 47.614, 50.083, 52.222, 54.241, 56.109, 57.584, 58.743, 60.556, 60.588, 60.588, 60.588,  ],
    [ 18.304, 18.304, 18.304, 18.304, 18.304, 18.887, 22.989, 26.396, 29.702, 32.998, 36.260, 39.449, 42.601, 45.860, 49.100, 51.678, 54.092, 56.277, 57.933, 59.202, 60.097, 60.462, 60.557, 60.588, 60.588, 60.588, 60.588,  ],
    [ 18.304, 18.304, 18.304, 18.304, 19.112, 23.488, 27.175, 30.748, 34.355, 37.941, 41.429, 45.025, 48.750, 51.678, 54.431, 56.829, 58.503, 59.792, 60.408, 60.551, 60.581, 60.586, 60.588, 60.588, 60.588, 60.588, 60.588,  ],
    [ 18.304, 18.304, 18.304, 18.647, 23.176, 27.120, 30.956, 34.853, 38.675, 42.510, 46.544, 50.303, 53.395, 56.212, 58.280, 59.756, 60.428, 60.562, 60.584, 60.588, 60.588, 60.588, 60.588, 60.588, 60.588, 60.588, 60.588,  ],
    [ 18.304, 18.304, 18.311, 22.202, 26.396, 30.436, 34.555, 38.630, 42.777, 47.154, 51.014, 54.337, 57.170, 59.112, 60.273, 60.545, 60.583, 60.587, 60.588, 60.588, 60.588, 60.588, 60.588, 60.588, 60.588, 60.588, 60.588,  ]])


idx = 0
axs[0,0].plot(Compress, Res_5_2_exp[:, 0], color = color[idx],   linestyle = '-', marker = mark[idx],   markersize=12, label='0(dB)',)
axs[0,0].plot(Compress, Res_5_2_exp[:, 3], color = color[idx+1], linestyle = '-', marker = mark[idx+1], markersize=12, label = '3(dB)',)
axs[0,0].plot(Compress, Res_5_2_exp[:, 10], color = color[idx+2], linestyle = '-', marker = mark[idx+2], markersize=12, label = '10(dB)',)
axs[0,0].plot(Compress, Res_5_2_exp[:, 20], color = color[idx+3], linestyle = '-', marker = mark[idx+3], markersize=12, label='20(dB)',)
# axs[0,0].plot(Compress, Res_5_2_exp[:, 12], color = color[idx+4], linestyle = '-', marker = mark[idx+4], markersize=12, label='30(dB)',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font3  = {'family':'Times New Roman','style':'normal','size':24}
axs[0,0].set_xlabel('k/n', fontproperties = font3)
axs[0,0].set_ylabel(r'PSNR(dB)', fontproperties=font3)
font3  = {'family':'Times New Roman','style':'normal','size':25}
axs[0,0].set_title(r'$cap = \frac{1}{2} \log_{2}(1+10^{snr/10})$', fontproperties=font3, pad = 12)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
legend1 = axs[0,0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[0,0].tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=3, )
# axs[0,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=26, width=3,)
labels = axs[0,0].get_xticklabels() + axs[0,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号



#=====================================================================================




#=====================================================================================
#fig.tight_layout(pad=6, h_pad=4, w_pad=4)

# 调节两个子图间的距离
# plt.subplots_adjust(left=None,bottom=None,right=None,top=0.85,wspace=0.1,hspace=0.1)

# fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('MNIST: JPEG + Capacity', fontproperties=fontt, )


out_fig = plt.gcf()

filepath2 = '/home/jack/公共的/Python/AdversaryAttack/Figures/'
# out_fig.savefig(filepath2+'hh.pdf', format='pdf', dpi=1000, bbox_inches='tight')
#out_fig .savefig(filepath2+'hh.emf',format='emf',dpi=1000, bbox_inches = 'tight')
out_fig .savefig(filepath2+'FashionMnist.eps',format='eps',  bbox_inches = 'tight')
out_fig .savefig(filepath2+'FashionMnist.png',format='png',  bbox_inches = 'tight')
# plt.show()
plt.close()




















































































































