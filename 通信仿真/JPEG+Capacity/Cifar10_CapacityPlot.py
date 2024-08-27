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





#===========================================  调整图与图之间的间距 ===============================================


fig, axs = plt.subplots(3, 4, figsize=(32, 24) ,constrained_layout=True) # ,constrained_layout=True
# plt.subplots(constrained_layout=True)的作用是:自适应子图在整个figure上的位置以及调整相邻子图间的间距，使其无重叠。

#=============================================== [0, 0] ======================================================

 # cap = 0.5 * math.log2(1 + snr )
 # filesize = cap * comp * im.size
 # no R_min
 # Raw data: bmp
Compress       =  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  ]
SNR            =   [ -2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40]


Res_5_2_exp = np.array([
    [ 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321,  ],
    [ 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 26.463, 29.801, 32.130,  ],
    [ 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 16.056, 22.652, 24.970, 26.535, 31.051, 34.049, 35.959, 36.412,  ],
    [ 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 21.026, 24.618, 26.709, 28.163, 29.353, 30.377, 31.318, 32.163, 35.489, 36.412, 36.419, 36.419,  ],
    [ 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 19.474, 24.618, 27.039, 28.689, 30.025, 31.194, 32.243, 33.213, 34.104, 34.880, 35.507, 36.418, 36.419, 36.419, 36.419,  ],
    [ 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 18.735, 24.921, 27.497, 29.265, 30.702, 31.982, 33.138, 34.182, 35.063, 35.743, 36.175, 36.368, 36.413, 36.419, 36.419, 36.419, 36.419,  ],
    [ 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 22.958, 26.675, 28.855, 30.533, 31.982, 33.299, 34.452, 35.394, 36.026, 36.336, 36.411, 36.418, 36.419, 36.419, 36.419, 36.419, 36.419, 36.419,  ],
    [ 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 24.012, 27.409, 29.569, 31.283, 32.799, 34.155, 35.257, 36.008, 36.348, 36.414, 36.419, 36.419, 36.419, 36.419, 36.419, 36.419, 36.419, 36.419, 36.419,  ],
    [ 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 14.321, 24.012, 27.584, 29.841, 31.645, 33.256, 34.646, 35.690, 36.265, 36.408, 36.418, 36.419, 36.419, 36.419, 36.419, 36.419, 36.419, 36.419, 36.419, 36.419, 36.419, ]])



idx = 0
axs[0, 0].plot(Compress, Res_5_2_exp[:, SNR.index(0)], color = color[ idx + 1 ], linestyle = '-',    marker = mark[ idx + 1 ], markersize = 12, label = '0(dB)',)
axs[0, 0].plot(Compress, Res_5_2_exp[:, SNR.index(10)], color = color[ idx + 2 ], linestyle = '-', marker = mark[ idx + 2 ], markersize = 12, label = '10(dB)',)
axs[0, 0].plot(Compress, Res_5_2_exp[:, SNR.index(20)], color = color[ idx + 3 ], linestyle = '-', marker = mark[ idx + 3 ], markersize = 12, label = '20(dB)',)


# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font3  = {'family':'Times New Roman','style':'normal','size':24}
axs[0, 0].set_xlabel('k/n', fontproperties = font3)
axs[0, 0].set_ylabel(r'PSNR(dB)', fontproperties=font3)
font3  = {'family':'Times New Roman','style':'normal','size':25}
axs[0, 0].set_title("(a)", fontproperties=font3, pad = 12, loc='left',)
axs[0, 0].set_title(r'$cap = \frac{1}{2} \log_{2}(1+snr), bmp, im.size$', fontproperties=font3, pad = 12)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
legend1 = axs[0, 0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[0, 0].tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=3, )
labels = axs[0, 0].get_xticklabels() + axs[0, 0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号



#=====================================================================================
#fig.tight_layout(pad=6, h_pad=4, w_pad=4)

# 调节两个子图间的距离
# plt.subplots_adjust(left=None,bottom=None,right=None,top=0.85,wspace=0.1,hspace=0.1)

# fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
fontt  = {'family':'Times New Roman','style':'normal','size':30}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('Cifar10: JPEG + Capacity', fontproperties=fontt, )


out_fig = plt.gcf()

filepath2 = '/home/jack/公共的/Python/AdversaryAttack/JPEG+Capacity/'
out_fig.savefig(filepath2+'Cifar10.pdf', format='pdf', bbox_inches='tight')
#out_fig .savefig(filepath2+'hh.emf',format='emf',dpi=1000, bbox_inches = 'tight')
out_fig .savefig(filepath2+'Cifar10.eps',format='eps',  bbox_inches = 'tight')
out_fig .savefig(filepath2+'Cifar10.png',format='png',  bbox_inches = 'tight')
# plt.show()
plt.close()




















































































































