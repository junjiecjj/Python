#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 16:01:17 2025

@author: jack
"""


import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties



# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "SimSun"
plt.rcParams['mathtext.fontset'] = "stix"
plt.rcParams['font.size'] = 14        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 1000      # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 24

# 本项目自己编写的库



fontpath = "/usr/share/fonts/truetype/windows/"
mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D', '_',  ]
color = ['#1E90FF','#FF6347','#00FF00','#0000FF','#4ea142','#FF00FF','#FFA500','#800080','#FF0000','#EE82EE','#00FFFF','#9932CC','#00CED1','#CD5C5C', '#7B68EE','#808000']

compressrate = [0.2, 0.5, 0.9]
snrtrain = [2, 10, 20]
snrtest = np.arange(-5, 36, 1)

r02_2db_dir = "2023-12-01-09:19:20_FLSemantic"
r05_10db_dir = "2023-11-30-21:34:56_FLSemantic"
r09_20db_dir = "2023-11-30-19:35:46_FLSemantic"

home = os.path.expanduser('~')
rootdir = f"{home}/FL_Sem2026/"
central_dir = rootdir + "MNIST_centralized"
fl_noQ_noniid = rootdir + "MNIST_noIID_noQuant_2025-12-08-13:43:21"
fl_4bQ_noniid = rootdir + "MNIST_noIID_4Quant_2025-12-08-13:53:58"
fl_2bQ_noniid = rootdir + "MNIST_noIID_2Quant_2025-12-09-08:48:41"

save_dir = rootdir + 'Figs_plot'

def MNIST_compare( R = 0.5, snrtrain = 10, snrtest = 10, ):
    raw_dir = os.path.join(central_dir, "test_results/raw_image")
    cent_dir = os.path.join(central_dir, f"test_results/Images_compr={R:.1f}_trainSnr={snrtrain}(dB)/testSNR={snrtest}(dB)")
    fl_noquant = os.path.join(fl_noQ_noniid, f"test_results/Images_compr={R:.1f}_trainSnr={snrtrain}(dB)/testSNR={snrtest}(dB)")
    fl_4quant = os.path.join(fl_4bQ_noniid, f"test_results/Images_compr={R:.1f}_trainSnr={snrtrain}(dB)/testSNR={snrtest}(dB)")
    fl_2quant = os.path.join(fl_2bQ_noniid, f"test_results/Images_compr={R:.1f}_trainSnr={snrtrain}(dB)/testSNR={snrtest}(dB)")
    dirs = [raw_dir, cent_dir, fl_noquant, fl_4quant, fl_2quant]

    rows = len(dirs)
    cols = 5
    figsize = (cols*2 , rows*2 + 1)
    fig, axs = plt.subplots(rows, cols, figsize = figsize, constrained_layout = True) #  constrained_layout=True

    for i in range(rows):
        # print(i)
        name = os.listdir(dirs[i])
        if 'raw_grid_images.png' in name:
            name.remove('raw_grid_images.png')
        tmp = f"grid_images_R={R:.1f}_trainSnr={snrtrain}(dB)_testSnr={snrtest}(dB).png"
        if tmp in name:
            name.remove(tmp)
        name = sorted(name, key=lambda x: int(x.split('_')[-2]))
        # print(name)
        font = FontProperties(fname=fontpath+"simsun.ttf", size=35)
        if i == 0:
            lb = "原图"
            axs[i,0].set_ylabel(lb, fontproperties = font)
        elif i == 1:
            lb = "中心式"
            axs[i,0].set_ylabel(lb, fontproperties = font)
        elif i == 2:
            lb = "FL+精确"
            axs[i,0].set_ylabel(lb, fontproperties = font)
        elif i == 3:
            lb = "FL+4bit"
            axs[i,0].set_ylabel(lb, fontproperties = font)
        elif i == 4:
            lb = "FL+2bit"
            axs[i,0].set_ylabel(lb, fontproperties = font)
        for j in range(cols):
            im = imageio.imread(os.path.join(dirs[i], name[j]))
            axs[i, j].imshow(im, cmap = 'Greys', interpolation='none')
            font1 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22, 'color':'blue', }
            real_lab = name[j].split('_')[-1][0]
            axs[i, j].set_title( r"$\mathrm{{label}}:{} \rightarrow {}$".format(real_lab, real_lab),  fontdict = font1, )
            axs[i, j].set_xticks([])  # #不显示x轴刻度值
            axs[i, j].set_yticks([] ) # #不显示y轴刻度值

    supt = r'$\mathrm{{R}}={:.1f},\mathrm{{SNR}}_\mathrm{{train}}={}\mathrm{{(dB)}},\mathrm{{SNR}}_\mathrm{{test}}={}\mathrm{{(dB)}}$'.format(R, snrtrain, snrtest)
    fontt = {'family': 'Times New Roman', 'style': 'normal', 'size': 28,   }
    plt.suptitle(supt, fontproperties=fontt,)
    out_fig = plt.gcf()
    # out_fig.savefig(f"./eps/MNIST_{R:.1f}_trainSnr={snrtrain}(dB)_testSNR={snrtest}(dB).pdf", )
    out_fig.savefig(save_dir + f"/MNIST_{R:.1f}_trainSnr={snrtrain}(dB)_testSNR={snrtest}(dB).pdf")
    plt.show()
    plt.close()
    return

## Fig.14-16
MNIST_compare( R = 0.2, snrtrain = 2, snrtest = -5, )
MNIST_compare( R = 0.2, snrtrain = 2, snrtest = 2, )

MNIST_compare( R = 0.5, snrtrain = 10, snrtest = -5, )
MNIST_compare( R = 0.5, snrtrain = 10, snrtest = 10, )

MNIST_compare( R = 0.9, snrtrain = 20, snrtest = -5, )
MNIST_compare( R = 0.9, snrtrain = 20, snrtest = 20, )



















