#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:46:06 2019

@author: jack

这是查看某一炮的pcrl01,lmtipref,dfsdev,aminor的文件
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
from matplotlib.font_manager import FontProperties

font = \
FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 12)
font1 = \
FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 10)

file = '/home/jack/数据筛选/'
safe4 = np.load(file+'Bsafe4.npz')['safe']
disrupt4=np.load(file+'Bdisrup4.npz')['disruption']
density4=np.load(file+'BRealdensity4.npz')['Realdensity']

safe5 = np.load(file+'Bsafe5.npz')['safe']
disrupt5 =np.load(file+'Bdisrup5.npz')['disruption']
density5 = np.load(file+'BRealdensity4.npz')['Realdensity']

safe6 = np.load(file+'Bsafe6.npz')['safe']
disrupt6 =np.load(file+'Bdisrup6.npz')['disruption']
density6 = np.load(file+'BRealdensity6.npz')['Realdensity']

safe7 = np.load(file+'Bsafe7.npz')['safe']
disrupt7 =np.load(file+'Bdisrup7.npz')['disruption']
density7 = np.load(file+'BRealdensity7.npz')['Realdensity']

safe8 = np.load(file+'Bsafe8.npz')['safe']
disrupt8 =np.load(file+'Bdisrup8.npz')['disruption']
density8 = np.load(file+'BRealdensity8.npz')['Realdensity']


safe = safe4.shape[0]+safe5.shape[0]+safe6.shape[0]+safe7.shape[0]+safe8.shape[0]
density = density4.shape[0]+density5.shape[0]+density6.shape[0]+density7.shape[0]+density8.shape[0]
disrupt = disrupt4.shape[0]+disrupt5.shape[0]+disrupt6.shape[0]+disrupt7.shape[0]+disrupt8.shape[0]-density
other = 88299-40000 - safe - disrupt - density

fig, axs = plt.subplots(1,1,sharex=True,figsize=(8,5))#figsize=(6,8)labels = [u'大型',u'中型',u'小型',u'微型'] #定义标签

recipe = [str(safe)+" nondisruptive",
          str(disrupt)+" other dsruptive",
          str(density)+" density limit disruptive",
          str(other)+" other"]

data = [float(x.split()[0]) for x in recipe]
ingredients = [x.split()[-1] for x in recipe]


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))+1
    print(pct,absolute)
    return "{:.1f}%\n({:d})".format(pct, absolute)

labels = [r'non-disruptive',r'other disruption',r'density limit',r'other'] #定义标签
sizes = [safe,disrupt-density,density,other] #每块值
colors = ['lightskyblue','yellowgreen','red','yellow'] #每块颜色定义
explode = (0.01,0.01,0.1,0.01) #将某一块分割出来，值越大分割出的间隙越大
patches,text1,text2 = axs.pie(data,
                      explode=explode,
                      labels=labels,
                      colors=colors,
                      labeldistance = 1.1,#图例距圆心半径倍距离
                      autopct=lambda pct: func(pct, data), #数值保留固定小数位
                      shadow = True, #无阴影设置
                      startangle =90, #逆时针起始角度设置
                      pctdistance = 0.6, #数值距圆心半径倍数距离
                      textprops={'fontsize':11}
                      )
#patches饼图的返回值，texts1饼图外label的文本，texts2饼图内部文本
# x，y轴刻度设置一致，保证饼图为圆形
axs.axis('equal')

axs.legend(prop=font1)

out_fig = plt.gcf()
picfile = '/home/jack/result/'
out_fig.savefig(picfile+'static.eps',format='eps',dpi=1000,bbox_inches = 'tight')

plt.show()

