

import matplotlib
# matplotlib.get_backend()
matplotlib.use('TkAgg')
# matplotlib.use('WXagg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
from pylab import tick_params
import copy
import torch


# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"



 

############################################################################################
t = np.arange(0, 2, 0.1)
s1 = np.sin(2*np.pi*t)
s2 = np.cos(2*np.pi*t)
s3 = np.tan(2*np.pi*t)
t1 = np.arange(-1, 1.1, 0.1)
s4 = np.arcsin(t1)


fig, axs = plt.subplots(4, 1, figsize=(10, 16))
############################################## 0 #############################################


axs[0].plot(t, s1, color='b', linestyle='-', label='sin(x)正弦',)
axs[0].plot(t, s3, color='r', linestyle='-', label='tan(x)',)
axs[0].axvline(x=1, ymin=0.4, ymax=0.6, ls='-',
               linewidth=4, color='b', label='tan(x)',)


font1 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
font2  = {'family':'Times New Roman','style':'normal','size':17, 'color':'#00FF00'}
axs[0].set_xlabel(r'time (s)时间', fontproperties=font1, fontdict = font2)
axs[0].set_ylabel(r'值(0-1)', fontproperties=font1, fontdict = font2)
axs[0].set_title('sin and tan 函数', fontproperties=font1)

font1 = FontProperties(fname=fontpath+"simsun.ttf", size=25)
# font1  = {'family':'Times New Roman','style':'normal','size':22}
axs[0].set_title('杰克', loc='left', color='#0000FF', fontproperties=font1)
axs[0].set_title('rose', loc='right', color='#9400D3', fontproperties=font1)
# # 设置 y 就在轴方向显示网格线
axs[0].grid(axis='x', color = '#1E90FF', linestyle = '--', linewidth = 0.5, )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
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
############################################# 1 ###############################################
font  = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)

axs[1].plot(t, s2, color='r', linestyle='--', label=r'$cos(x)$',)
axs[1].axvline(x=1, ls=':', color='b', label='disruption')
axs[1].axhline(y=0.5, ls=':', color='r', label='阈值')
axs[1].annotate(r'$t_{real\_disr}$', xy=(1, -1), xytext=(1.1, -0.7), arrowprops=dict(arrowstyle='->', connectionstyle='arc3'), fontproperties=font)


#font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 22)
font1 = { 'style': 'normal', 'size': 22, 'color':'blue',}
font2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)

axs[1].set_xlabel(r'time (s)', fontproperties=font2, fontdict = font1)
axs[1].set_ylabel(r'值(0-1)', fontproperties=font2, fontdict = font1)
axs[1].set_title('cos(x)', fontproperties=font2)
axs[1].set_title(r'$\mathrm{t}_\mathrm{disr}$=%d' % 1, loc='left', fontproperties=font2)
axs[1].grid()


font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
#font2 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs[1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

x_major_locator=MultipleLocator(1)               #把x轴的刻度间隔设置为1，并存在变量里
axs[1].xaxis.set_major_locator(x_major_locator)  #把x轴的主刻度设置为1的倍数
axs[1].tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


axs[1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细
#################################### 2 #############################################
font  = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)

axs[2].plot(t1, s4, color='blue', linestyle='-', linewidth=1, label='arcsin()')
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[2].set_xlabel(r'time (s)', fontproperties=font3)
axs[2].set_ylabel(r'值(0-1)', fontproperties=font3)
axs[2].set_title('arcsin(x)', fontproperties=font3)
axs[2].axvline(x=0, ymin=0.4, ymax=0.6, ls='-', linewidth=4, color='b',)
axs[2].annotate(r'$t_{disr}$',
                xy=(0, 0), xycoords='data',
                xytext=(0.4, 0.3), textcoords='figure fraction',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3'), fontproperties=font)

axs[2].grid()

font3 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=18)
legend1 = axs[2].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font3,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[2].tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

axs[2].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[2].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[2].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[2].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细
# ##############################################################################################

#################################### 3 #############################################
font  = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)

axs[3].plot(t1, s4, color='blue', linestyle='-', linewidth=1, label='arcsin()')
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[3].set_xlabel(r'time (s)', fontproperties=font3)
axs[3].set_ylabel(r'值(0-1)', fontproperties=font3)
axs[3].set_title('arcsin(x)', fontproperties=font3)
axs[3].axvline(x=0, ymin=0.4, ymax=0.6, ls='-', linewidth=4, color='b',)
axs[3].annotate(r'$t_{disr}$',
                xy=(0, 0), xycoords='data',
                xytext=(0.4, 0.1), textcoords='figure fraction',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3'), fontproperties=font)

axs[3].grid()

font3 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font3 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)
legend1 = axs[3].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font3,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明

# labelsize——标签大小：float 或 str 刻度标签字体大小（以磅为单位）或字符串（例如，“大”）。
# width——宽度：刻度线宽度（以磅为单位）。
axs[3].tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=3,)
labels = axs[3].get_xticklabels() + axs[3].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


axs[3].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[3].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[3].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[3].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

# ##############################################################################################


#fig.subplots_adjust(hspace=0.6)  # 调节两个子图间的距离
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.7)

fontt = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#fontt  = {'family':'Times New Roman','style':'normal','size':22}
#fontt = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
plt.suptitle('cos and sin正弦 and tan ', fontproperties=fontt, x=0.5,y=0.96,)

out_fig = plt.gcf()
out_fig.savefig(filepath2+'hh11.eps', format='eps',dpi=1000, bbox_inches='tight')
#out_fig.savefig(filepath2+'hh.svg', format='svg', dpi=1000, bbox_inches='tight')
out_fig.savefig(filepath2+'hh.pdf', format='pdf', dpi=1000, bbox_inches='tight')
#out_fig.savefig(filepath2+'hh.emf',format='emf',dpi=1000, bbox_inches = 'tight')
#out_fig.savefig(filepath2+'hh.jpg',format='jpg',dpi=1000, bbox_inches = 'tight')
#out_fig.savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()



#===========================================  调整图与图之间的间距 ===============================================

t = np.arange(0, 2, 0.1)
s1 = np.sin(2*np.pi*t)
s2 = np.cos(2*np.pi*t)
s3 = np.tan(2*np.pi*t)
t1 = np.arange(-1, 1, 0.1)
s4 = np.arcsin(t1)

fig, axs = plt.subplots(2, 2, figsize=(16, 16) ) # ,constrained_layout=True
# plt.subplots(constrained_layout=True)的作用是:自适应子图在整个figure上的位置以及调整相邻子图间的间距，使其无重叠。
#=============================================== 0 ======================================================
axs[0,0].plot(t, s1, color='b', linestyle='-', label='sin(x)正弦',)

# font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font3  = {'family':'Times New Roman','style':'normal','size':22}
axs[0,0].set_xlabel(r'time (s)', fontproperties=font3)
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
axs[0,0].set_ylabel(r'值(sin(x))', fontproperties=font3)
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
axs[0,0].tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=3, labelcolor = "red", colors='blue', rotation=25,)
# axs[0,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=26, width=3,)
labels = axs[0,0].get_xticklabels() + axs[0,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号


xlabels = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
axs[0,0].set_xticklabels(xlabels)
#=============================================== 1 ======================================================
axs[0,1].plot(t, s2, color='r', linestyle='-', label='cos(x)正弦',)
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=32)
font   = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font2  = {'family':'Times New Roman','style':'normal','size':27, 'color':'#00FF00'}
axs[0,1].set_xlabel(r'time (s)',  fontdict = font2)
axs[0,1].set_ylabel(r'值(cos(x))', fontproperties=font3, fontdict = font2)
axs[0,1].set_title('cos(x)', fontproperties=font3)

axs[0,1].grid(axis='x', color = '#1E90FF', linestyle = '--', linewidth = 0.5, )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[0,1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[0,1].tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
labels = axs[0,1].get_xticklabels() + axs[0,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

#=============================================== 2 ======================================================
axs[1,0].plot(t, s3, color='g', linestyle='-', label='tan(x)正弦',)
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
axs[1,0].set_xlabel(r'time (s)', fontproperties=font3)
axs[1,0].set_ylabel(r'值(tan(x))', fontproperties=font3)
axs[1,0].set_title('tan(x)', fontproperties=font3)

axs[1,0].grid(  color = '#1E90FF', linestyle = '--', linewidth = 0.5, )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[1,0].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[1,0].tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
labels = axs[1,0].get_xticklabels() + axs[1,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号
#=============================================== 3 ======================================================
axs[1,1].plot(t, s4, color='b', linestyle='-', label='arcsin(x)正弦',)
font3 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
#font3  = {'family':'Times New Roman','style':'normal','size':22}
#font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
font2  = {'family':'Times New Roman','style':'normal','size':27, 'color':'#9400D3'}
axs[1,1].set_xlabel(r'time (s)', fontproperties=font3)
axs[1,1].set_ylabel(r'值(arcsin(x))', fontproperties=font3, fontdict = font2)
axs[1,1].set_title('arcsin(x)', fontproperties=font3)

axs[1,1].grid(axis='y', color = '#1E90FF', linestyle = '--', linewidth = 0.5, )

font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 17}
font2 = FontProperties(fname=fontpath+"simsun.ttf", size=16)
legend1 = axs[1,1].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')  # 设置图例legend背景透明


axs[1,1].tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
labels = axs[1,1].get_xticklabels() + axs[1,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(24) for label in labels]  # 刻度值字号

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
out_fig .savefig(filepath2+'hh22.eps',format='eps',dpi=1000, bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()


