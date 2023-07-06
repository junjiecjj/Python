#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 15:36:44 2023

@author: jack


"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator
import matplotlib
from pylab import tick_params
import copy

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'

font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)

fontpath = "/usr/share/fonts/truetype/windows/"

fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"

fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"





#============================================================================================================================
#                                          高斯分布的概率分布直方图
#============================================================================================================================


mean = 1    #均值为0
sigma = 6  #标准差为1，反应数据集中还是分散的值
x = mean + sigma*np.random.randn(100000)
x = np.random.normal(loc=1.0, scale=6.0,  size = (100000,))

fig, axs = plt.subplots(nrows=2, figsize=(9, 6), ) # ,constrained_layout=True

#第二个参数bins越大、则条形bar越窄越密，density=True则画出频率，否则次数
axs[0].hist(x, bins = 100, density = True, histtype='bar',color='yellowgreen', alpha=0.75, label = 'pdf') #normed=True或1 表示频率图
##pdf概率分布图，一万个数落在某个区间内的数有多少个

font1 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font1  = {'family':'Times New Roman','style':'normal','size':17}
axs[0].set_xlabel(r'值', fontproperties=font1)
axs[0].set_ylabel(r'概率', fontproperties=font1)
font1  = {'family':'Times New Roman','style':'normal','size':22}
axs[0].set_title('Pdf', fontproperties=font1)

font1 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
#font1  = {'family':'Times New Roman','style':'normal','size':22}
axs[0].set_title('杰克', loc='left', color='#0000FF', fontproperties=font1)
axs[0].set_title('rose', loc='right', color='#9400D3', fontproperties=font1)
axs[0].grid()

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


# labelsize——标签大小：float 或 str 刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
# width——宽度：刻度线宽度（以磅为单位）。
# 参数axis的值为’x’、‘y’、‘both’，分别代表设置X轴、Y轴以及同时设置，默认值为’both’。
axs[0].tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=3, labelcolor = "red", colors='blue', rotation=25,)
labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


axs[0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

#=======================================================================
#cdf累计概率函数，cumulative累计。比如需要统计小于5的数的概率
axs[1].hist(x, bins = 100, density=True, histtype='bar',facecolor='pink',alpha=0.75,cumulative=True, rwidth=0.8, label = 'cdf')

font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 22)
#font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
#font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

axs[1].set_xlabel(r'值', fontproperties = font2)
axs[1].set_ylabel(r'概率', fontproperties = font2)
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
axs[1].set_title('Cdf', fontproperties = font2)
axs[1].set_title(r'$\mathrm{t}_\mathrm{disr}$=%d' % 1, loc='left', fontproperties = font2)
axs[1].grid()


font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs[1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

x_major_locator=MultipleLocator(5)               #把x轴的刻度间隔设置为1，并存在变量里
axs[1].xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
axs[1].tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


axs[1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.8)

# fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('PDF and CDF', fontproperties=fontt, x=0.5, y=0.99,)

out_fig = plt.gcf()
# out_fig.savefig(filepath2+'hh.pdf', format='pdf', dpi=1000, bbox_inches='tight')

plt.show()



#============================================================================================================================
#                                          使用 hist() 函数绘制多个数据组的直方图
#============================================================================================================================

import matplotlib.pyplot as plt
import numpy as np

# 生成三组随机数据
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(2, 1, 1000)
data3 = np.random.normal(-2, 1, 1000)

# 绘制直方图
plt.hist(data1, bins=30, alpha=0.5, label='Data 1')
plt.hist(data2, bins=30, alpha=0.5, label='Data 2')
plt.hist(data3, bins=30, alpha=0.5, label='Data 3')

# 设置图表属性
plt.title('RUNOOB hist() TEST')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

# 显示图表
plt.show()



#============================================================================================================================
#                                          使用 hist() 函数绘制多个数据组的直方图
#============================================================================================================================





#============================================================================================================================
#                                          使用 hist() 函数绘制多个数据组的直方图
#============================================================================================================================



#============================================================================================================================
#                                          使用 hist() 函数绘制多个数据组的直方图
#============================================================================================================================












#============================================================================================================================
#                                          使用 hist() 函数绘制多个数据组的直方图
#============================================================================================================================









#============================================================================================================================
#                                          使用 hist() 函数绘制多个数据组的直方图
#============================================================================================================================









#============================================================================================================================
#                                          使用 hist() 函数绘制多个数据组的直方图
#============================================================================================================================









#============================================================================================================================
#                                          使用 hist() 函数绘制多个数据组的直方图
#============================================================================================================================









#============================================================================================================================
#                                          使用 hist() 函数绘制多个数据组的直方图
#============================================================================================================================









#============================================================================================================================
#                                          使用 hist() 函数绘制多个数据组的直方图
#============================================================================================================================











































































































































































































































































