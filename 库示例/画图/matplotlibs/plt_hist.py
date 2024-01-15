#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 15:36:44 2023

@author: jack

1、matplotlib.pyplot.hist()
 n,bins,patches=matplotlib.pyplot.hist(x, bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, **kwargs)

参数值：
参数说明：

x：表示要绘制直方图的数据，可以是一个一维数组或列表。
bins：可选参数，表示直方图的箱数。默认为10。
range：可选参数，表示直方图的值域范围，可以是一个二元组或列表。默认为None，即使用数据中的最小值和最大值。
density：可选参数，表示是否将直方图归一化。默认为False，即直方图的高度为每个箱子内的样本数，而不是频率或概率密度。
weights：可选参数，表示每个数据点的权重。默认为None。
cumulative：可选参数，表示是否绘制累积分布图。默认为False。
bottom：可选参数，表示直方图的起始高度。默认为None。
histtype：可选参数，表示直方图的类型，可以是'bar'、'barstacked'、'step'、'stepfilled'等。默认为'bar'。
align：可选参数，表示直方图箱子的对齐方式，可以是'left'、'mid'、'right'。默认为'mid'。
orientation：可选参数，表示直方图的方向，可以是'vertical'、'horizontal'。默认为'vertical'。
rwidth：可选参数，表示每个箱子的宽度。默认为None。
log：可选参数，表示是否在y轴上使用对数刻度。默认为False。
color：可选参数，表示直方图的颜色。
label：可选参数，表示直方图的标签。
stacked：可选参数，表示是否堆叠不同的直方图。默认为False。
**kwargs：可选参数，表示其他绘图参数。
返回值:
n:  直方图向量，是否归一化由参数normed设定。当normed取默认值时，n即为直方图各组内元素的数量（各组频数）
bins: 返回各个bin的区间范围
patches：返回每个bin里面包含的数据，是一个list
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

import torch

# x = np.random.normal(loc=1.0, scale=6.0,  size = (100000,))
x  = torch.normal(2, 1, size=(100000, ))
fig, axs = plt.subplots(nrows=3, figsize=(9, 12), ) # ,constrained_layout=True

#第二个参数bins越大、则条形bar越窄越密，density=True则画出频率，否则次数
re = axs[0].hist(x, bins = 100,  histtype='bar',color='yellowgreen', alpha=0.75, label = 'pdf') #  density = True, 或1 表示频率图
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
# 我们只需更改直方图的图像类型，令histtype=‘step’，就会画出一条曲线来（Figure3，实际上就是将直方柱并在一起，除边界外颜色透明），类似于累积分布曲线。这时，我们就能很好地观察到不同数据分布曲线间的差异。
#cdf累计概率函数，cumulative累计。比如需要统计小于5的数的概率
re1 = axs[1].hist(x, bins = 100, density=True, histtype='bar',color='pink',alpha=0.75, cumulative=True, rwidth=0.8, label = 'cdf')

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

# x_major_locator=MultipleLocator(5)               #把x轴的刻度间隔设置为1，并存在变量里
# axs[1].xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
axs[1].tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


axs[1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细



#=======================================================================
# 我们只需更改直方图的图像类型，令histtype=‘step’，就会画出一条曲线来（Figure3，实际上就是将直方柱并在一起，除边界外颜色透明），类似于累积分布曲线。这时，我们就能很好地观察到不同数据分布曲线间的差异。
#cdf累计概率函数，cumulative累计。比如需要统计小于5的数的概率
# re1 = axs[1].hist(x, bins = 100, density=True, histtype='bar',facecolor='pink',alpha=0.75, cumulative=True, rwidth=0.8, label = 'cdf')

hist, bin_edges = re1[0], re1[1]
## 可以看出： hist == re[0], bin_edges = re[1];
x = [(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]  # 每个柱子的中心坐标
axs[2].bar(x, hist, width=bin_edges[1]-bin_edges[0],  label = 'cdf')  # width表示每个柱子的宽度
font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 22)
#font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
#font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[2].set_xlabel(r'值', fontproperties = font2)
axs[2].set_ylabel(r'概率', fontproperties = font2)
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
axs[2].set_title('Cdf', fontproperties = font2)
axs[2].set_title(r'$\mathrm{t}_\mathrm{disr}$=%d' % 1, loc='left', fontproperties = font2)
axs[2].grid()


font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs[2].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# x_major_locator=MultipleLocator(5)               #把x轴的刻度间隔设置为1，并存在变量里
# axs[1].xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
axs[2].tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


axs[2].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[2].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[2].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[2].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细



plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.8)

# fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('PDF and CDF', fontproperties=fontt, x=0.5, y=0.99,)

out_fig = plt.gcf()
# out_fig.savefig(filepath2+'hh.pdf', format='pdf', dpi=1000, bbox_inches='tight')

# plt.show()
filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()



#============================================================================================================================
#                                          使用 hist() 函数绘制多个数据组的直方图
#============================================================================================================================

import matplotlib.pyplot as plt
import numpy as np



fig, axs = plt.subplots(2,1, figsize=(8, 10), constrained_layout=True)



# 生成三组随机数据
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(2, 1, 1000)
data3 = np.random.normal(-2, 1, 1000)


##===================================================== 1 =================================================
# 绘制直方图
axs[0].hist(data1, bins = 100, density = True, histtype='bar',color='r', alpha=0.75, label='Data 1')
axs[0].hist(data2, bins = 100, density = True, histtype='bar',color='g', alpha=0.75, label='Data 2')
axs[0].hist(data3, bins = 100, density = True, histtype='bar',color='b', alpha=0.75, label='Data 3')

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

x_major_locator=MultipleLocator(2)               #把x轴的刻度间隔设置为1，并存在变量里
axs[0].xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
axs[0].tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

axs[0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

##===================================================== 2 =================================================
# 绘制直方图
axs[1].hist(data1, bins = 100, density=True, histtype='step', facecolor='r', alpha=0.75, cumulative=True,  label='Data 1')
axs[1].hist(data2, bins = 100, density=True, histtype='step', facecolor='g', alpha=0.75, cumulative=True,  label='Data 2')
axs[1].hist(data3, bins = 100, density=True, histtype='step', facecolor='b', alpha=0.75, cumulative=True,  label='Data 3')

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

x_major_locator=MultipleLocator(2)               #把x轴的刻度间隔设置为1，并存在变量里
axs[1].xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
axs[1].tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
labels = axs[1].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

axs[1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


# 显示图表
filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()



#============================================================================================================================
#                                         黑客样式表的贝叶斯方法
#============================================================================================================================





import numpy as np
import matplotlib.pyplot as plt


# Fixing random state for reproducibility
np.random.seed(19680801)

# plt.style.use('bmh')


def plot_beta_hist(ax, a, b):
    ax.hist(np.random.beta(a, b, size=10000),  histtype="stepfilled", bins=25, alpha=0.8, density=True)


fig, axs = plt.subplots(1,1, figsize=(8, 6), constrained_layout=True)

plot_beta_hist(axs, 10, 10)
plot_beta_hist(axs, 4, 12)
plot_beta_hist(axs, 50, 12)
plot_beta_hist(axs, 6, 55)
axs.set_title("'bmh' style sheet")

# 显示图表
filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()




#============================================================================================================================
#                                        具有多个数据集的直方图（hist）函数
#============================================================================================================================



import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

n_bins = 10
x = np.random.randn(1000, 3)

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)

colors = ['red', 'tan', 'lime']
ax0.hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
ax0.legend(prop={'size': 10})
ax0.set_title('bars with legend')

ax1.hist(x, n_bins, density=True, histtype='bar', stacked=True)
ax1.set_title('stacked bar')

ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False)
ax2.set_title('stack step (unfilled)')

# Make a multiple-histogram of data-sets with different length.
x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
ax3.hist(x_multi, n_bins, histtype='bar')
ax3.set_title('different sample sizes')

fig.tight_layout()
plt.show()

# 显示图表
filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()





























































































































































































































































