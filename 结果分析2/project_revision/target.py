#!/usr/bin/env python
#-*-coding=utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import math
import pylab
import os
from matplotlib.font_manager import FontProperties
from pylab import tick_params

fontpath = "/usr/share/fonts/truetype/msttcorefonts/"
font1    = FontProperties(fname = "/usr/share/fonts/truetype/msttcorefonts/times.ttf", size=26,weight='bold')
fontt  = {'family':'Times New Roman','style':'normal','size':28,'weight':'bold'}
fontt1 = {'style':'normal','size':28}
fontt2 = {'style':'normal','size':28}

home        = os.environ['HOME']
picturepath = home+'/Resultpicture/'


td = 10
fl = 7
sp = 0.001

x = np.arange(-10,10,0.001)
y = 1/(1+np.exp(-x))


X=np.arange(fl,td+sp,sp)
Y=1/(1+np.exp(-(X-(td-(td-fl)/3))*25))



fig,axs = plt.subplots(2,1,figsize=(9,8))
axs[0].tick_params(direction='in')

axs[0].plot(x,y,label=r'$y=\frac{1}{1+e^{-t}}$',color='black',linewidth=2,)
axs[0].scatter(0,0.5,color='r',linewidth=10)
axs[0].annotate('(0, 0.5)',xy=(0,0.5),xytext=(2.5,0.4),textcoords='data',
       arrowprops=dict(facecolor='black',
              connectionstyle='arc3',width=2),fontproperties = font1)
#axs[0].grid()
axs[0].set_xlabel('time(s)',fontsize=38)
axs[0].set_ylabel('y',fontsize=38)
legend = axs[0].legend(loc='best',prop=fontt2,edgecolor='black')#shadow=True
frame = legend.get_frame()
frame.set_alpha(1)
frame.set_facecolor('none') # 设置图例legend背景透明

axs[0].set_title('(a)', loc='left',fontdict=fontt)
axs[0].tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=5)
labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(28) for label in labels]#刻度值字号

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

axs[1].plot(X,Y,label=r'$y=\frac{1}{1+e^{-[t-t_\mathrm{n_\mathrm{e}/n_\mathrm{GW}=k}]\times b}}$',color='black',linewidth=2,)
#axs[1].plot(X,Y,label='Predictor target',color='black',linewidth=2,)
axs[1].annotate('Start of flat-top',xy=(fl,0),xytext=(fl+0.2,0.2),textcoords='data',
       arrowprops=dict(facecolor='black',
              connectionstyle='arc3',width=2),fontproperties = font1)
axs[1].annotate('Disruption',xy=(td,0),xytext=(td-0.8,0.2),textcoords='data',
       arrowprops=dict(facecolor='black',
              connectionstyle='arc3',width=2),fontproperties = font1)
axs[1].annotate(r'($t_\mathrm{n_\mathrm{e}/n_\mathrm{GW}=k}$,0.5)',xy=(spx,spy),xytext=(spx+0.3,0.6),textcoords='data',
       arrowprops=dict(facecolor='black',
              connectionstyle='arc3',width=2),fontproperties = font1)
axs[1].scatter(spx,spy,color='r',linewidth=10)
#axs[1].axvline(x=fl,ls='--',color='green',linewidth=4,)
axs[1].axvline(x=spx,ls='--',color='b',linewidth=4,)
#axs[1].axvline(x=td,ls='--',color='r',linewidth=4,)
axs[1].set_xlabel('time(s)',fontsize=38)
axs[1].set_ylabel('y',fontsize=38)

axs[1].set_xticks(xx)
axs[1].set_xticklabels([r'$t_{flat}$',r'$t_\mathrm{n_\mathrm{e}/n_\mathrm{GW}=k}$',r'$t_d$'],fontsize=32)
#axs[1].grid()
legend1 = axs[1].legend(loc='upper left',prop=fontt2,edgecolor='black')
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none') # 设置图例legend背景透明

axs[1].set_title('(b)', loc='left',fontdict=fontt)

axs[1].tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=5)
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(28) for label in labels]#刻度值字号

axs[1].spines['bottom'].set_linewidth(2)   ###设置底部坐标轴的粗细
axs[1].spines['left'].set_linewidth(2)     ####设置左边坐标轴的粗细
axs[1].spines['right'].set_linewidth(2)    ###设置右边坐标轴的粗细
axs[1].spines['top'].set_linewidth(2)      ####设置上部坐标轴的粗细


fig.subplots_adjust(hspace=0.1)#调节两个子图间的距离
plt.tight_layout()
out_fig=plt.gcf()
#out_fig.savefig(picturepath+'sigmoid.eps',format='eps',dpi=1000,bbox_inches='tight')
plt.show()

