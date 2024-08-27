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
# cap = 0.5 * math.log2(1 + 10**(snr/10.0))
Res_5_2_exp = np.array([
    [ 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.179, 24.358, 31.450,  ],
    [ 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.440, 22.363, 24.593, 26.404, 28.142, 29.859, 31.580, 38.527, 45.021, 50.507, 60.224,  ],
    [ 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.610, 23.528, 26.269, 28.735, 31.187, 33.669, 36.255, 38.800, 41.282, 43.626, 45.867, 47.934, 55.929, 61.545, 62.848, 62.919,  ],
    [ 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.149, 23.440, 26.742, 29.794, 32.930, 36.188, 39.400, 42.598, 45.527, 48.203, 50.813, 53.467, 56.077, 58.412, 60.315, 62.849, 62.919, 62.919, 62.919,  ],
    [ 19.019, 19.019, 19.019, 19.019, 19.019, 19.019, 19.024, 23.253, 27.078, 30.589, 34.279, 38.132, 41.979, 45.471, 48.743, 51.919, 55.085, 58.089, 60.449, 61.862, 62.586, 62.830, 62.898, 62.919, 62.919, 62.919, 62.919,  ],
    [ 19.019, 19.019, 19.019, 19.019, 19.019, 19.710, 25.122, 28.999, 33.065, 37.394, 41.728, 45.809, 49.517, 53.248, 56.877, 59.918, 61.781, 62.625, 62.859, 62.909, 62.918, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919,  ],
    [ 19.019, 19.019, 19.019, 19.019, 20.043, 25.707, 29.924, 34.418, 39.197, 43.918, 48.151, 52.283, 56.474, 59.918, 61.962, 62.729, 62.888, 62.918, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919,  ],
    [ 19.019, 19.019, 19.019, 19.371, 25.343, 29.859, 34.687, 39.859, 44.859, 49.412, 53.954, 58.362, 61.355, 62.611, 62.876, 62.916, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919,  ],
    [ 19.019, 19.019, 19.019, 24.202, 28.999, 34.006, 39.463, 44.804, 49.720, 54.583, 59.199, 61.910, 62.786, 62.908, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919, 62.919,  ]])


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
out_fig .savefig(filepath2+'Mnist.eps',format='eps',  bbox_inches = 'tight')
out_fig .savefig(filepath2+'Mnist.png',format='png',  bbox_inches = 'tight')
# plt.show()
plt.close()




















































































































