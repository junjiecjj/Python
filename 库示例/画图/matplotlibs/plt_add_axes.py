#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:49:47 2023

@author: jack

此程序的功能是：在大图里面画小图
"""

import numpy as np
import matplotlib.pyplot as plt

#新建figure
fig = plt.figure()
# 定义数据
x = [1, 2, 3, 4, 5, 6, 7]
y = [1, 3, 4, 2, 5, 8, 6]

#新建区域ax1
#figure的百分比,从figure 10%的位置开始绘制, 宽高是figure的80%
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
# 获得绘制的句柄
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(x, y, 'r')
ax1.set_title('area1')

#新增区域ax2,嵌套在ax1内
left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
# 获得绘制的句柄
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(x, y, 'b')
ax2.set_title('area2')


filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()


