#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Mar 29 12:46:06 2019
@author: jack
这是查看某一炮的pcrl01,lmtipref,dfsdev,aminor的文件
"""

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator


import matplotlib.pyplot as plt
import numpy as np

def func(x):
    return 5+2*x+3*x**2-0.5*x**3

def fund(x):
    return 200-10*x

#np.linspace()取x的值
x=np.linspace(-10,10,50)
y=func(x)
y1=fund(x)

fig,ax=plt.subplots()
plt.plot(x,y,'r-o',lw=1)
plt.plot(x,y1,'g-*',lw=1)
#plt.ylim(ymin=0)  控制y轴现实范围
a,b=-10,3
xf=x[np.where((x>a)&(x<b))]
#plt.fill_between(xf),在xf范围内
# #曲线1：np.zeros(len(xf))与曲线2：func(xf)之间的区域
# #np.zeros()从零开始，np.ones()从1开始，np.ones()*20从20开始 #plt.fill_between(xf, np.zeros(len(xf)), func(xf), color='blue', alpha=.25)
# #两条曲线之间的区域
plt.fill_between(xf,func(xf),fund(xf),color='blue',alpha=0.25)
plt.show()



