#!/usr/bin/env python
#-*-coding=utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import math
import pylab
import os
from matplotlib.font_manager import FontProperties
from pylab import tick_params

fontpath = "/usr/share/fonts/truetype/windows/"
font = FontProperties(fname=fontpath+"simsun.ttf", size = 18)#fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font1 = FontProperties(fname=fontpath+"simsun.ttf", size = 24)
font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 20)
font3 = FontProperties(fname=fontpath+"simsun.ttf", size = 30)

fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
font4 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 30)
font5 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)

home  = os.environ['HOME']
picturepath = home+'/tmp/'


td = 10
fl = 7
sp = 0.001

x = np.arange(-10,10,0.001)
y = 1/(1+np.exp(-x))


X=np.arange(fl,td+sp,sp)
Y=1/(1+np.exp(-(X-(td-(td-fl)/3))*25))



fig,axs = plt.subplots(2,1,figsize=(6,6))
axs[0].tick_params(direction='in')

axs[0].plot(x,y,label=r'$y=\frac{1}{1+e^{-t}}$',color='black',linewidth=2,)
axs[0].scatter(0,0.5,color='r',linewidth=6)
font1 = FontProperties(fname=fontpath+"simsun.ttf", size = 24)
axs[0].annotate('(0, 0.5)',xy=(0,0.5),xytext=(2.5,0.4),textcoords='data',
       arrowprops=dict(facecolor='black',
              connectionstyle='arc3',width=2),fontproperties = font1)


font1 = FontProperties(fname=fontpath+"simsun.ttf", size = 24)
axs[0].set_xlabel('时间(s)',fontproperties=font1)
font4 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 30)
axs[0].set_ylabel('y',fontproperties=font4)
legend = axs[0].legend(loc='best',prop=font2,edgecolor='black')#shadow=True
frame = legend.get_frame() 
frame.set_alpha(1) 
frame.set_facecolor('none') # 设置图例legend背景透明 
font5 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs[0].set_title('(a)', loc='left',fontproperties=font5)

axs[0].tick_params(labelsize=16,width=4)
labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels]#刻度值字号

axs[0].spines['bottom'].set_linewidth(2); ###设置底部坐标轴的粗细
axs[0].spines['left'].set_linewidth(2);   ####设置左边坐标轴的粗细
axs[0].spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
axs[0].spines['top'].set_linewidth(2);    ####设置上部坐标轴的粗细

###########################################################
axs[1].tick_params(direction='in')
spx = td-(td-fl)/3
spy = 0.5

xx = [fl,spx,td]
#axs[3].axes.set_xticks(XX)
#axs[3].axes.set_xticklabels(XX,)
axs[1].tick_params(direction='in')
axs[1].plot(X,Y,label=r'$y=\frac{1}{1+e^{-(t-t_{n_\mathrm{e}/n_\mathrm{GW}=0.5})\times b}}$',color='black',linewidth=2,)
#axs[1].plot(X,Y,label='输出',color='black',linewidth=2,)
font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 20)
axs[1].annotate('平顶段起始',xy=(fl,0),xytext=(fl+0.5,0.2),textcoords='data',
       arrowprops=dict(facecolor='black',
              connectionstyle='arc3',width=2),fontproperties = font2)
font2 = FontProperties(fname=fontpath+"simsun.ttf", size = 20)
axs[1].annotate('破裂',xy=(td,0),xytext=(td-0.8,0.2),textcoords='data',
       arrowprops=dict(facecolor='black',
              connectionstyle='arc3',width=2),fontproperties = font2)

font = FontProperties(fname=fontpath+"simsun.ttf", size = 18)
axs[1].annotate(r'($t_\mathrm{n_\mathrm{e}/n_\mathrm{GW}=0.5}$,0.5)',xy=(spx,spy),xytext=(spx+0.3,0.6),textcoords='data',
       arrowprops=dict(facecolor='black',
              connectionstyle='arc3',width=2),fontproperties = font)
axs[1].scatter(spx,spy,color='r',linewidth=6)
#axs[1].axvline(x=fl,ls='--',color='green',linewidth=4,)
#axs[1].axvline(x=spx,ls='--',color='b',linewidth=4,)
#axs[1].axvline(x=td,ls='--',color='r',linewidth=4,)
font1 = FontProperties(fname=fontpath+"simsun.ttf", size = 24)
axs[1].set_xlabel('时间(s)',fontproperties=font1)
axs[1].set_ylabel('破裂概率',fontproperties=font1)

axs[1].set_xticks(xx)
axs[1].set_xticklabels([r'$t_\mathrm{flat}$',r'$t_\mathrm{n_\mathrm{e}/n_\mathrm{GW}=0.5}$',r'$t_\mathrm{d}$'],fontsize=8)
#axs[1].grid()
legend1 = axs[1].legend(loc='upper left',prop=font,edgecolor='black')
frame1 = legend1.get_frame() 
frame1.set_alpha(1) 
frame1.set_facecolor('none') # 设置图例legend背景透明 
font5 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
axs[1].set_title('(b)', loc='left',fontproperties = font5)

axs[1].tick_params(labelsize=16,width=4)
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(20) for label in labels]#刻度值字号

axs[1].spines['bottom'].set_linewidth(2)   ###设置底部坐标轴的粗细
axs[1].spines['left'].set_linewidth(2)     ####设置左边坐标轴的粗细
axs[1].spines['right'].set_linewidth(2)    ###设置右边坐标轴的粗细
axs[1].spines['top'].set_linewidth(2)      ####设置上部坐标轴的粗细


fig.subplots_adjust(hspace=0.0)#调节两个子图间的距离
plt.tight_layout()
out_fig=plt.gcf()
out_fig.savefig(picturepath+'fig3.svg',format='svg',dpi=1000,bbox_inches='tight')
out_fig.savefig(picturepath+'fig3.pdf',format='pdf',dpi=1000,bbox_inches='tight')

plt.show()
