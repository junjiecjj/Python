#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:31:52 2023

@author: jack
"""

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)

fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)

fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
fonte1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=24)

fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove Light Nerd Font Complete Mono.otf", size=20)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove SemiLight Nerd Font Complete.otf", size=20)
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove Regular Nerd Font Complete Mono.otf", size=20)

fontt = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
fonttX = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
fonttY = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
fonttitle = {'style': 'normal', 'size': 17}
fontt2 = {'style': 'normal', 'size': 19, 'weight': 'bold'}
fontt3 = {'style': 'normal', 'size': 16, }



#================================================================================================================================
#===========================================  调整图与图之间的间距 3D ===============================================
#================================================================================================================================
t = np.arange(0, 2, 0.1)
s1 = np.sin(2*np.pi*t)
s2 = np.cos(2*np.pi*t)
s3 = np.tan(2*np.pi*t)
t1 = np.arange(-1, 1, 0.1)
s4 = np.arcsin(t1)

fig = plt.figure(figsize=(16, 16) )
# fig, axs = plt.subplots(2, 2, figsize=(16, 16) ) # ,constrained_layout=True
# plt.subplots(constrained_layout=True)的作用是:自适应子图在整个figure上的位置以及调整相邻子图间的间距，使其无重叠。
#=============================================== 0 ======================================================
ax1 = fig.add_subplot(221)
ax1.plot(t, s1, color='b', linestyle='-', label='sin(x)正弦',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font3  = {'family':'Times New Roman','style':'normal','size':22}
ax1.set_xlabel(r'time (s)', fontproperties=font3)
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
ax1.set_ylabel(r'值(sin(x))', fontproperties=font3)
font3  = {'family':'Times New Roman','style':'normal','size':22}
ax1.set_title('sin(x)', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = ax1.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# labelsize——标签大小：float 或 str 刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
# width——宽度：刻度线宽度（以磅为单位）。
# 参数axis的值为’x’、‘y’、‘both’，分别代表设置X轴、Y轴以及同时设置，默认值为’both’。
ax1.tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=13, labelcolor = "red", colors='blue', rotation=25,)
# axs[0,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=26, width=3,)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

ax1.spines['left'].set_linewidth(2.5)  ###设置右边坐标轴的粗细
ax1.spines['left'].set_color('m') ## 设置边框线颜色
ax1.spines['right'].set_linewidth(2.5)  ###设置右边坐标轴的粗细
ax1.spines['right'].set_color('green') ## 设置边框线颜色
ax1.set_xticks([0, 1, 2, 3, 4, 5,6,7,8,9]) # 设置刻度
xlabels = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
ax1.set_xticklabels(xlabels)

#=============================================== 1 ======================================================
ax2 = fig.add_subplot(222)
ax2.plot(t, s2, color='r', linestyle='-', label='cos(x)正弦',)
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=32)
font   = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font2  = {'family':'Times New Roman','style':'normal','size':27, 'color':'#00FF00'}
ax2.set_xlabel(r'time (s)',  fontdict = font2)
ax2.set_ylabel(r'值(cos(x))', fontproperties=font3, fontdict = font2)
ax2.set_title('cos(x)', fontproperties=font3)

ax2.grid(axis='x', color = '#1E90FF', linestyle = '--', linewidth = 0.5, )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = ax2.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


ax2.tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


ax2.set_xlim(-0.2, 2)  #拉开坐标轴范围显示投影
ax2.set_ylim(-1.1, 1.2)  #拉开坐标轴范围显示投影

x_major_locator=MultipleLocator(0.2)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.25)
#把y轴的刻度间隔设置为10，并存在变量里

ax2.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax2.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
#=============================================== 2 ======================================================

ax3 = fig.add_subplot(223, projection='3d')

#生成三维数据
xx = np.arange(-5,5,0.1)
yy = np.arange(-5,5,0.1)
X, Y = np.meshgrid(xx, yy)
Z = np.sin(np.sqrt(X**2+Y**2))

#作图
ax3.plot_surface(X,Y,Z,alpha=0.3, cmap='winter')     #生成表面， alpha 用于控制透明度
ax3.contour(X,Y,Z,zdir='z', offset=-3, cmap="rainbow")  #生成z方向投影，投到x-y平面
ax3.contour(X,Y,Z,zdir='x', offset=-6, cmap="rainbow")  #生成x方向投影，投到y-z平面
ax3.contour(X,Y,Z,zdir='y', offset=6, cmap="rainbow")   #生成y方向投影，投到x-z平面
#ax4.contourf(X,Y,Z,zdir='y', offset=6,cmap="rainbow")   #生成y方向投影填充，投到x-z平面，contourf()函数


font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font2  = {'family':'Times New Roman','style':'normal','size':27, 'color':'#9400D3'}
#设定显示范围
ax3.set_xlabel('X', fontproperties=font3, labelpad = 12.5 )
ax3.set_xlim(-6, 4)  #拉开坐标轴范围显示投影
ax3.set_ylabel('Y', fontproperties=font3, fontdict = font2, labelpad = 12.5)
ax3.set_ylim(-4, 6)
ax3.set_zlabel('Z', fontproperties=font3, fontdict = font2, labelpad = 12.5)
ax3.set_zlim(-3, 3)



ax3.tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=3, labelcolor = "green", colors='blue', rotation=25,)
labels = ax3.get_xticklabels() + ax3.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


# axs[1,0].plot(t, s3, color='g', linestyle='-', label='tan(x)正弦',)
# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
# #font3  = {'family':'Times New Roman','style':'normal','size':22}
# #font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
# axs[1,0].set_xlabel(r'time (s)', fontproperties=font3)
# axs[1,0].set_ylabel(r'值(tan(x))', fontproperties=font3)
# axs[1,0].set_title('tan(x)', fontproperties=font3)

# axs[1,0].grid(  color = '#1E90FF', linestyle = '--', linewidth = 0.5, )

# font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
# font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
# legend1 = axs[1,0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')  # 设置图例legend背景透明


# axs[1,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
# labels = axs[1,0].get_xticklabels() + axs[1,0].get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(24) for label in labels]  # 刻度值字号
#=============================================== 3 ======================================================
ax4 = fig.add_subplot(224)

ax4.plot(t, s4, color='b', linestyle='-', label='arcsin(x)正弦',)
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font2  = {'family':'Times New Roman','style':'normal','size':27, 'color':'#9400D3'}
ax4.set_xlabel(r'time (s)', fontproperties=font3)
ax4.set_ylabel(r'值(arcsin(x))', fontproperties=font3, fontdict = font2)
ax4.set_title('arcsin(x)', fontproperties=font3)

ax4.grid(axis='y', color = '#1E90FF', linestyle = '--', linewidth = 0.5, )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = ax4.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


ax4.tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
labels = ax4.get_xticklabels() + ax4.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号



#=====================================================================================
# 定义数据
x = [1, 2, 3, 4, 5, 6, 7]
y = [1, 3, 4, 2, 5, 8, 6]
#新增区域ax2,嵌套在ax1内
left, bottom, width, height = 0.8, 0.15, 0.15, 0.15
# 获得绘制的句柄
axs =  fig.add_axes([left, bottom, width, height])
axs.plot(x, y, 'b')
#=====================================================================================

#=====================================================================================
# plt.cla() # plt.cla()清除轴 ，即当前图中的当前活动轴。 它使其他轴保持不变。

# plt.clf() # plt.clf()使用其所有轴清除整个当前图形 ，但使窗口保持打开状态，以便可以将其重新用于其他绘图。

# fig.clf() # 清除整个图
#=====================================================================================


#=====================================================================================
#fig.tight_layout(pad=6, h_pad=4, w_pad=4)
# matplotlib.pyplot.tight_layout(*, pad=1.08, h_pad=None, w_pad=None, rect=None)
# pad,h_pad,w_pad分别调整子图和figure边缘，以及子图间的相距高度、宽度。

# 调节两个子图间的距离
# plt.subplots_adjust(left=None,bottom=None,right=None,top=0.85,wspace=0.1,hspace=0.1)
# 有六个可选参数来控制子图布局。值均为0~1之间。其中left、bottom、right、top围成的区域就是子图的区域。 left, right, bottom, top：子图所在区域的边界。 当值大于1.0的时候子图会超出figure的边界从而显示不全；值不大于1.0的时候，子图会自动分布在一个矩形区域（下图灰色部分）。要保证left < right, bottom < top，否则会报错。
# wspace、hspace分别表示子图之间左右、上下的间距。
# wspace, hspace：子图之间的横向间距、纵向间距分别与子图平均宽度、平均高度的比值。实际的默认值由matplotlibrc文件控制的。
plt.subplots_adjust(top=0.92,bottom=0.1, left=0.1, right=0.97, wspace=0.2, hspace=0.3)

fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('cos and sin正弦 and tan ', fontproperties=fontt, x=0.5, y=0.99,)

out_fig = plt.gcf()

filepath2 = '/home/jack/snap/'
# out_fig.savefig(filepath2+'hh.pdf', format='pdf', dpi=1000, bbox_inches='tight')
#out_fig .savefig(filepath2+'hh.emf',format='emf',dpi=1000, bbox_inches = 'tight')
# out_fig .savefig(filepath2+'plotfig.eps',format='eps',dpi=1000, bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()

#=====================================================================================================

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from mpl_toolkits.mplot3d.axes3d import get_test_data
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))

#===============
#  First subplot
#===============
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')

# plot a 3D surface like in the example mplot3d/surface3d_demo
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)
fig.colorbar(surf, shrink=0.5, aspect=10)

#===============
# Second subplot
#===============
# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2, projection='3d')

# plot a 3D wireframe like in the example mplot3d/wire3d_demo
X, Y, Z = get_test_data(0.05)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

out_fig = plt.gcf()

filepath2 = '/home/jack/snap/'
out_fig .savefig(filepath2+'plotfig.eps',format='eps',dpi=1000, bbox_inches = 'tight')
plt.close()

#=====================================================================================================

#=====================================================================================================
#                              all 3d subplot
#=====================================================================================================

t = np.arange(0, 2, 0.1)
s1 = np.sin(2*np.pi*t)
s2 = np.cos(2*np.pi*t)
s3 = np.tan(2*np.pi*t)
t1 = np.arange(-1, 1, 0.1)
s4 = np.arcsin(t1)

fig, axs = plt.subplots(2, 2, figsize=(16, 16) ,subplot_kw={"projection": "3d"}) # ,constrained_layout=True
# plt.subplots(constrained_layout=True)的作用是:自适应子图在整个figure上的位置以及调整相邻子图间的间距，使其无重叠。
#=============================================== 0 ======================================================
axs[0,0].plot(t, s1, color='b', linestyle='-', label='sin(x)正弦',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font3  = {'family':'Times New Roman','style':'normal','size':22}
axs[0,0].set_xlabel(r'time (s)', fontproperties=font3, labelpad = 22.5)
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
axs[0,0].set_ylabel(r'值(sin(x))', fontproperties=font3,  labelpad = 22.5)
axs[0,0].set_zlabel(r'z值(sin(x))', fontproperties=font3,  labelpad = 22.5)
font3  = {'family':'Times New Roman','style':'normal','size':22}
axs[0,0].set_title('sin(x)', fontproperties=font3)

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[0,0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# labelsize——标签大小：float 或 str 刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
# width——宽度：刻度线宽度（以磅为单位）。
# 参数axis的值为’x’、‘y’、‘both’，分别代表设置X轴、Y轴以及同时设置，默认值为’both’。
axs[0,0].tick_params(direction='in', axis='both', top=True, right=True, labelsize=24, width = 8, labelcolor = "red", colors='blue', rotation=25,)
# axs[0,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=26, width=3,)
labels = axs[0,0].get_xticklabels() + axs[0,0].get_yticklabels() + axs[0,0].get_zticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# [label.set_fontsize(24) for label in labels]  # 刻度值字号


xlabels = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
axs[0,0].set_xticklabels(xlabels)
#=============================================== 1 ======================================================
axs[0,1].plot(t, s2, color='r', linestyle='-', label='cos(x)正弦',)
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=32)
font   = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font2  = {'family':'Times New Roman','style':'normal','size':27, 'color':'#00FF00'}
axs[0,1].set_xlabel(r'time (s)',  fontdict = font2, labelpad = 12.5)
axs[0,1].set_ylabel(r'值(cos(x))', fontproperties=font3, fontdict = font2, labelpad = 12.5)
axs[0,1].set_zlabel(r'Z(cos(x))', fontproperties=font3, fontdict = font2, labelpad = 12.5)
axs[0,1].set_title('cos(x)', fontproperties=font3)

axs[0,1].grid(axis='x', color = '#1E90FF', linestyle = '--', linewidth = 0.5, )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[0,1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[0,1].tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
labels = axs[0,1].get_xticklabels() + axs[0,1].get_yticklabels() + axs[0,1].get_zticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

#=============================================== 2 ======================================================

#生成三维数据
xx = np.arange(-5,5,0.1)
yy = np.arange(-5,5,0.1)
X, Y = np.meshgrid(xx, yy)
Z = np.sin(np.sqrt(X**2+Y**2))

#作图
axs[1,0].plot_surface(X,Y,Z,alpha=0.3, cmap='winter')     #生成表面， alpha 用于控制透明度
axs[1,0].contour(X,Y,Z,zdir='z', offset=-3, cmap="rainbow")  #生成z方向投影，投到x-y平面
axs[1,0].contour(X,Y,Z,zdir='x', offset=-6, cmap="rainbow")  #生成x方向投影，投到y-z平面
axs[1,0].contour(X,Y,Z,zdir='y', offset=6, cmap="rainbow")   #生成y方向投影，投到x-z平面
#ax4.contourf(X,Y,Z,zdir='y', offset=6,cmap="rainbow")   #生成y方向投影填充，投到x-z平面，contourf()函数


font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font2  = {'family':'Times New Roman','style':'normal','size':27, 'color':'#9400D3'}
#设定显示范围
axs[1,0].set_xlabel('X', fontproperties=font3, labelpad = 12.5)
axs[1,0].set_xlim(-6, 4)  #拉开坐标轴范围显示投影
axs[1,0].set_ylabel('Y', fontproperties=font3, fontdict = font2, labelpad = 12.5)
axs[1,0].set_ylim(-4, 6)
axs[1,0].set_zlabel('Z', fontproperties=font3, fontdict = font2, labelpad = 12.5)
axs[1,0].set_zlim(-3, 3)



axs[1,0].tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=1, labelcolor = "green", colors='blue', rotation=25,)
labels = axs[1,0].get_xticklabels() + axs[1,0].get_yticklabels() + axs[1,0].get_zticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

#=============================================== 3 ======================================================
#构建xyz
z = np.linspace(0, 1, 100)
x = z * np.sin(20 * z)
y = z * np.cos(20 * z)
c = x + y
axs[1,1].scatter3D(x, y, z, c=c)
# axs[1,1].plot(t, s4, color='b', linestyle='-', label='arcsin(x)正弦',)
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font2  = {'family':'Times New Roman','style':'normal','size':27, 'color':'#9400D3'}
axs[1,1].set_xlabel(r'time (s)', fontproperties=font3, labelpad = 12.5)
axs[1,1].set_ylabel(r'值(arcsin(x))', fontproperties=font3, fontdict = font2, labelpad = 12.5)
axs[1,1].set_zlabel(r'z值(arcsin(x))', fontproperties=font3, fontdict = font2, labelpad = 12.5)
# axs[1,1].set_title('arcsin(x)', fontproperties=font3)
axs[1,1].set_title('3d Scatter plot')
axs[1,1].grid(axis='y', color = '#1E90FF', linestyle = '--', linewidth = 0.5, )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[1,1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[1,1].tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
labels = axs[1,1].get_xticklabels() + axs[1,1].get_yticklabels() + axs[1,1].get_zticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


x_major_locator=MultipleLocator(0.5)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(0.5)
#把y轴的刻度间隔设置为10，并存在变量里

axs[1,1].xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
axs[1,1].yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
#=====================================================================================
#fig.tight_layout(pad=6, h_pad=4, w_pad=4)
# matplotlib.pyplot.tight_layout(*, pad=1.08, h_pad=None, w_pad=None, rect=None)
# pad,h_pad,w_pad分别调整子图和figure边缘，以及子图间的相距高度、宽度。

# 调节两个子图间的距离
# plt.subplots_adjust(left=None,bottom=None,right=None,top=0.85,wspace=0.1,hspace=0.1)
# 有六个可选参数来控制子图布局。值均为0~1之间。其中left、bottom、right、top围成的区域就是子图的区域。 left, right, bottom, top：子图所在区域的边界。 当值大于1.0的时候子图会超出figure的边界从而显示不全；值不大于1.0的时候，子图会自动分布在一个矩形区域（下图灰色部分）。要保证left < right, bottom < top，否则会报错。
# wspace、hspace分别表示子图之间左右、上下的间距。
# wspace, hspace：子图之间的横向间距、纵向间距分别与子图平均宽度、平均高度的比值。实际的默认值由matplotlibrc文件控制的。
plt.subplots_adjust(top=0.92,bottom=0.1, left=0.1, right=0.97, wspace=0.2, hspace=0.3)

fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('cos and sin正弦 and tan ', fontproperties=fontt, x=0.5, y=0.99,)


out_fig = plt.gcf()

filepath2 = '/home/jack/snap/'
# out_fig.savefig(filepath2+'plotfig.eps',  bbox_inches = 'tight')
plt.close()



#=====================================================================================================
