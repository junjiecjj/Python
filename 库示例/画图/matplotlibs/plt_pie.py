#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:46:32 2023

@author: jack

Matplotlib 饼图
饼图（Pie Chart）是一种常用的数据可视化图形，用来展示各类别在总体中所占的比例。

我们可以使用 pyplot 中的 pie() 方法来绘制饼图。

pie() 方法语法格式如下：

matplotlib.pyplot.pie(x, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.6, shadow=False, labeldistance=1.1, startangle=0, radius=1, counterclock=True, wedgeprops=None, textprops=None, center=0, 0, frame=False, rotatelabels=False, *, normalize=None, data=None)[source]
    参数说明：
    x：浮点型数组或列表，用于绘制饼图的数据，表示每个扇形的面积。
    explode：数组，表示各个扇形之间的间隔，默认值为0。
    labels：列表，各个扇形的标签，默认值为 None。
    colors：数组，表示各个扇形的颜色，默认值为 None。
    autopct：设置饼图内各个扇形百分比显示格式，%d%% 整数百分比，%0.1f 一位小数， %0.1f%% 一位小数百分比， %0.2f%% 两位小数百分比。
    labeldistance：标签标记的绘制位置，相对于半径的比例，默认值为 1.1，如 <1则绘制在饼图内侧。
    pctdistance：：类似于 labeldistance，指定 autopct 的位置刻度，默认值为 0.6。
    shadow：：布尔值 True 或 False，设置饼图的阴影，默认为 False，不设置阴影。
    radius：：设置饼图的半径，默认为 1。
    startangle：：用于指定饼图的起始角度，默认为从 x 轴正方向逆时针画起，如设定 =90 则从 y 轴正方向画起。
    counterclock：布尔值，用于指定是否逆时针绘制扇形，默认为 True，即逆时针绘制，False 为顺时针。
    wedgeprops ：字典类型，默认值 None。用于指定扇形的属性，比如边框线颜色、边框线宽度等。例如：wedgeprops={'linewidth':5} 设置 wedge 线宽为5。
    textprops ：字典类型，用于指定文本标签的属性，比如字体大小、字体颜色等，默认值为 None。
    center ：浮点类型的列表，用于指定饼图的中心位置，默认值：(0,0)。
    frame ：布尔类型，用于指定是否绘制饼图的边框，默认值：False。如果是 True，绘制带有表的轴框架。
    rotatelabels ：布尔类型，用于指定是否旋转文本标签，默认为 False。如果为 True，旋转每个 label 到指定的角度。
    data：用于指定数据。如果设置了 data 参数，则可以直接使用数据框中的列作为 x、labels 等参数的值，无需再次传递。

    除此之外，pie() 函数还可以返回三个参数：
    wedges：一个包含扇形对象的列表。
    texts：一个包含文本标签对象的列表。
    autotexts：一个包含自动生成的文本标签对象的列表。

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



# 数据
sizes = [15, 30, 45, 10]

# 饼图的标签
labels = ['A', 'B', 'C', 'D']

# 饼图的颜色
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']

# 突出显示第二个扇形
explode = (0, 0.1, 0, 0)

# 绘制饼图
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)

# 标题
plt.title("RUNOOB Pie Test")

# 显示图形
plt.show()



##===============================================================
def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))+1
    print(pct, absolute)
    return "{:.2f}%\n({:d})".format(pct, absolute)

safe = 12
density = 24
disrupt = 36
other = 10

recipe = [str(safe)+" nondisruptive",
          str(disrupt)+" other dsruptive",
          str(density)+" density limit disruptive",
          str(other)+" other"]
x = [float(x.split()[0]) for x in recipe]


labels = [r'non-disruptive',r'other disruption',r'density limit',r'other'] #定义标签

colors = ['lightskyblue','yellowgreen','red','yellow'] #每块颜色定义
explode = (0.01,0.01,0.1,0.01) #将某一块分割出来，值越大分割出的间隙越大



fig, axs = plt.subplots(1,1, figsize=(8, 10), constrained_layout=True)
patches,text1,text2 = axs.pie(x,
                      explode=explode,
                      labels=labels,
                      colors=colors,
                      radius = 1,     # 设置饼图的半径，默认为 1。
                      labeldistance = 1.1,#图例距圆心半径倍距离
                      autopct= lambda pct: func(pct, x), #数值保留固定小数位
                      shadow = True, #无阴影设置
                      startangle =90, #逆时针起始角度设置
                      pctdistance = 0.9, #数值距圆心半径倍数距离
                      textprops={'fontsize':18}, # 字典类型，用于指定文本标签的属性，比如字体大小、字体颜色等，默认值为 None。
                      wedgeprops={'linewidth':5}, # 字典类型，默认值 None。用于指定扇形的属性，比如边框线颜色、边框线宽度等。
                      center = (0,1) , #浮点类型的列表，用于指定饼图的中心位置，默认值：(0,0)。
                      # frame = True, # 布尔类型，用于指定是否绘制饼图的边框，默认值：False。如果是 True，绘制带有表的轴框架。
                      rotatelabels = 0, #布尔类型，用于指定是否旋转文本标签，默认为 False。如果为 True，旋转每个 label 到指定的角度。
                      )
#patches饼图的返回值，texts1饼图外label的文本，texts2饼图内部文本
# x，y轴刻度设置一致，保证饼图为圆形
axs.axis('equal')
font = {'family': 'Times New Roman', 'style': 'normal', 'size': 21}
axs.legend(prop=font)

# plt.show()
filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()



































































































































































































