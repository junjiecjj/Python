#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:53:49 2023

@author: jack

画出给定的离散数据的概率密度函数和累计分布函数。

"""


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
# from matplotlib.pyplot import MultipleLocator

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


# 1、matplotlib.pyplot.hist()
 # n,bins,patches=matplotlib.pyplot.hist(x, bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, **kwargs)

# 参数值：
# 参数说明：

# x：表示要绘制直方图的数据，可以是一个一维数组或列表。
# bins：可选参数，表示直方图的箱数。默认为10。
# range：可选参数，表示直方图的值域范围，可以是一个二元组或列表。默认为None，即使用数据中的最小值和最大值。
# density：可选参数，表示是否将直方图归一化。默认为False，即直方图的高度为每个箱子内的样本数，而不是频率或概率密度。
# weights：可选参数，表示每个数据点的权重。默认为None。
# cumulative：可选参数，表示是否绘制累积分布图。默认为False。
# bottom：可选参数，表示直方图的起始高度。默认为None。
# histtype：可选参数，表示直方图的类型，可以是'bar'、'barstacked'、'step'、'stepfilled'等。默认为'bar'。
# align：可选参数，表示直方图箱子的对齐方式，可以是'left'、'mid'、'right'。默认为'mid'。
# orientation：可选参数，表示直方图的方向，可以是'vertical'、'horizontal'。默认为'vertical'。
# rwidth：可选参数，表示每个箱子的宽度。默认为None。
# log：可选参数，表示是否在y轴上使用对数刻度。默认为False。
# color：可选参数，表示直方图的颜色。
# label：可选参数，表示直方图的标签。
# stacked：可选参数，表示是否堆叠不同的直方图。默认为False。
# **kwargs：可选参数，表示其他绘图参数。
# 返回值:
# n:  直方图向量，是否归一化由参数normed设定。当normed取默认值时，n即为直方图各组内元素的数量（各组频数）
# bins: 返回各个bin的区间范围
# patches：返回每个bin里面包含的数据，是一个list


# 使用ax.hist()函数想要把数据转为密度直方图，但发现直接使用density=true得到的值很奇怪，y轴甚至会大于1，不符合我的预期。
# 查了资料发现density=ture的意思是保证该面积的积分为1，并不是概率和为1，因此我们需要对其进行改进。
# weights = np.ones_like(myarray)/float(len(myarray))
# plt.hist(myarray, weights=weights， bins = 200)
# 但是这时density = True 不能再用了

###======================================== hist pdf =======================================================

fig, axs = plt.subplots(6,1, figsize=(8, 22), constrained_layout=True)
# data = np.random.normal(loc = 2, scale = 1.0, size = 10000000)
data  = torch.normal(2, 1, size=(100000, ))
# data = torch.load("/home/jack/snap/test.pt")
##====================================1: hist pdf ==============================================
# 1、matplotlib.pyplot.hist()
#第二个参数bins越大、则条形bar越窄越密，density=True则画出频率，否则次数; cumulative = True 累积分布图.
weights = np.ones_like(data)/float(len(data))
# re = axs[0].hist(data, bins = 200, density = True,  histtype='bar', color='yellowgreen', alpha=0.75, label = 'pdf', ) #normed=True或1 表示频率图
re = axs[0].hist(data, bins = 200, weights=weights, histtype='step', color='yellowgreen', linewidth = 4,  alpha=0.75, label = 'pdf', ) #normed=True或1 表示频率图
##pdf概率分布图，一万个数落在某个区间内的数有多少个
font1 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font1  = {'family':'Times New Roman','style':'normal','size':17}
axs[0].set_xlabel(r'值', fontproperties=font1)
axs[0].set_ylabel(r'概率', fontproperties=font1)
font1  = {'family':'Times New Roman','style':'normal','size':22}
axs[0].set_title('hist Pdf', fontproperties=font1)

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


##========================= 2: histogram + bar, pdf ===========================
# 2、numpy之histogram（）直方图函数
# hist, bin_edges = np.histogram(a, bins=10,range=None,weights=None,density=False);
# a是待统计数据的数组；
# bins指定统计的区间个数；
# range是一个长度为2的元组，表示统计范围的最小值和最大值，默认值None，表示范围由数据的范围决定
# weights为数组的每个元素指定了权值,histogram()会对区间中数组所对应的权值进行求和
# density 为True时，hist 是每个区间的概率密度；为False，hist是每个区间中元素的个数

# 返回: 默认情况下hist是数据在各个区间上的频率，bin_edges是划分的各个区间的边界，


# hist, bin_edges = np.histogram(data, bins=100, ) # density = True
hist  = re[0]
bin_edges = re[1]
## 可以看出： hist == re[0], bin_edges == re[1];


x = [(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]  # 每个柱子的中心坐标
axs[1].bar(x, hist, width=bin_edges[1]-bin_edges[0], label = 'pdf')  # width表示每个柱子的宽度


##pdf概率分布图，一万个数落在某个区间内的数有多少个
font1 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font1  = {'family':'Times New Roman','style':'normal','size':17}
axs[1].set_xlabel(r'值', fontproperties=font1)
axs[1].set_ylabel(r'概率', fontproperties=font1)
font1  = {'family':'Times New Roman','style':'normal','size':22}
axs[1].set_title('histogram + bar, Pdf', fontproperties=font1)

font1 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
#font1  = {'family':'Times New Roman','style':'normal','size':22}
axs[1].set_title('杰克', loc='left', color='#0000FF', fontproperties=font1)
axs[1].set_title('rose', loc='right', color='#9400D3', fontproperties=font1)
axs[1].grid()

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


# labelsize——标签大小：float 或 str 刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
# width——宽度：刻度线宽度（以磅为单位）。
# 参数axis的值为’x’、‘y’、‘both’，分别代表设置X轴、Y轴以及同时设置，默认值为’both’。
axs[1].tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=3, labelcolor = "red", colors='blue', rotation=25,)
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


axs[1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


##=================================== 3: histogram+stairs pdf ==============================================


# hist, bin_edges = np.histogram(data, bins=100, density = True)
# or
hist  = re[0]
bin_edges = re[1]
axs[2].stairs(hist, bin_edges,  linestyle='-', linewidth = 4,  color = "blue", label = 'pdf',)

##pdf概率分布图，一万个数落在某个区间内的数有多少个
font1 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font1  = {'family':'Times New Roman','style':'normal','size':17}
axs[2].set_xlabel(r'值', fontproperties=font1)
axs[2].set_ylabel(r'概率', fontproperties=font1)
font1  = {'family':'Times New Roman','style':'normal','size':22}
axs[2].set_title('histogram+stairs, PDF', fontproperties=font1)

font1 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
#font1  = {'family':'Times New Roman','style':'normal','size':22}
axs[2].set_title('杰克', loc='left', color='#0000FF', fontproperties=font1)
axs[2].set_title('rose', loc='right', color='#9400D3', fontproperties=font1)
axs[2].grid()

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[2].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


# labelsize——标签大小：float 或 str 刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
# width——宽度：刻度线宽度（以磅为单位）。
# 参数axis的值为’x’、‘y’、‘both’，分别代表设置X轴、Y轴以及同时设置，默认值为’both’。
axs[2].tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=3, labelcolor = "red", colors='blue', rotation=25,)
labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


axs[2].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[2].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[2].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[2].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细



#================================ 4: hist CDF =======================================
# 我们只需更改直方图的图像类型，令histtype=‘step’，就会画出一条曲线来（Figure3，实际上就是将直方柱并在一起，除边界外颜色透明），类似于累积分布曲线。这时，我们就能很好地观察到不同数据分布曲线间的差异。
#cdf累计概率函数，cumulative累计。比如需要统计小于5的数的概率
re1 = axs[3].hist(data, bins = 100, density=True, histtype='step',color='red',alpha=0.75, linewidth = 4,  cumulative=True, rwidth=0.8, label = 'cdf')

font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 22)
#font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
#font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

axs[3].set_xlabel(r'值', fontproperties = font2)
axs[3].set_ylabel(r'概率', fontproperties = font2)
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
axs[3].set_title('hist, CDF', fontproperties = font2)
axs[3].set_title(r'$\mathrm{t}_\mathrm{disr}$=%d' % 1, loc='left', fontproperties = font2)
axs[3].grid()


font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs[3].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# x_major_locator=MultipleLocator(5)               #把x轴的刻度间隔设置为1，并存在变量里
# axs[1].xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
axs[3].tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
labels = axs[3].get_xticklabels() + axs[3].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


axs[3].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[3].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[3].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[3].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细



#================================== 5: hist+ bar CDF =====================================

# re1 = axs[4].hist(data, bins = 100, density=True, histtype='bar',facecolor='pink',alpha=0.75, cumulative=True, rwidth=0.8, label = 'cdf')
hist, bin_edges = re1[0], re1[1]

x = [(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]  # 每个柱子的中心坐标
axs[4].bar(x, hist, width=bin_edges[1]-bin_edges[0], color = 'g',  label = 'cdf')  # width表示每个柱子的宽度
font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 22)
#font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
#font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[4].set_xlabel(r'值', fontproperties = font2)
axs[4].set_ylabel(r'概率', fontproperties = font2)
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
axs[4].set_title('hist+bar Cdf', fontproperties = font2)
axs[4].set_title(r'$\mathrm{t}_\mathrm{disr}$=%d' % 1, loc='left', fontproperties = font2)
axs[4].grid()


font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs[4].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# x_major_locator=MultipleLocator(5)               #把x轴的刻度间隔设置为1，并存在变量里
# axs[1].xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
axs[4].tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
labels = axs[4].get_xticklabels() + axs[4].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


axs[4].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[4].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[4].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[4].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


#================================== 6: hist+ stairs, CDF =====================================
# re1 = axs[3].hist(data, bins = 100, density=True, histtype='bar',facecolor='pink',alpha=0.75, rwidth=0.8, label = 'cdf')
hist, bin_edges = re1[0], re1[1]

axs[5].stairs(hist, bin_edges,  linestyle='-', linewidth = 4,  color = "blue", label = 'cdf',)

font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 22)
#font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
#font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[5].set_xlabel(r'值', fontproperties = font2)
axs[5].set_ylabel(r'概率', fontproperties = font2)
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
axs[5].set_title('hist+stairs Cdf', fontproperties = font2)
axs[5].set_title(r'$\mathrm{t}_\mathrm{disr}$=%d' % 1, loc='left', fontproperties = font2)
axs[5].grid()


font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs[5].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# x_major_locator=MultipleLocator(5)               #把x轴的刻度间隔设置为1，并存在变量里
# axs[1].xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
axs[5].tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
labels = axs[5].get_xticklabels() + axs[5].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


axs[5].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[5].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[5].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[5].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细



out_fig = plt.gcf()
# out_fig.savefig(filepath2+'hh.pdf', format='pdf', dpi=1000, bbox_inches='tight')

# plt.show()
filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()




































































































































































































































































































































































































































