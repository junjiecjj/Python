#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 23:13:24 2021

@author: jack
"""

# https://www.cxybb.com/article/qq_40074819/105432327

from scipy import signal, special
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker 
from matplotlib.font_manager import FontProperties
fontpath = "/usr/share/fonts/truetype/windows/"
font = FontProperties(fname = fontpath+"simhei.ttf")
plt.rcParams['font.sans-serif']=['SimHei']#设置作图中文显示

T = 1               #基带信号宽度，也就是频率
nb = 100            #定义传输的比特数
delta_T = T/200     #采样间隔
fs = 1/delta_T      #采样频率
fc = 10/T           #载波频率
SNR = 0             #信噪比

t = np.arange(0, nb*T, delta_T)
N = len(t)

# 产生基带信号
data = [1 if x > 0.5 else 0 for x in np.random.randn(1, nb)[0]]  #调用随机函数产生任意在0到1的1*nb的矩阵，大于0.5显示为1，小于0.5显示为0
data0 = []                             #创建一个1*nb/delta_T的零矩阵
for q in range(nb):
    data0 += [data[q]]*int(1/delta_T)  #将基带信号变换成对应波形信号

# 调制信号的产生
data1 = []      #创建一个1*nb/delta_T的零矩阵
datanrz = np.array(data)*2-1              #将基带信号转换成极性码,映射
for q in range(nb):
    data1 += [datanrz[q]]*int(1/delta_T)  #将极性码变成对应的波形信号
    
idata = datanrz[0:(nb-1):2]       #串并转换，将奇偶位分开，间隔为2，i是奇位 q是偶位
qdata = datanrz[1:nb:2]         
ich = []                          #创建一个1*nb/delta_T/2的零矩阵，以便后面存放奇偶位数据
qch = []         
for i in range(int(nb/2)):
    ich += [idata[i]]*int(1/delta_T)    #奇位码元转换为对应的波形信号
    qch += [qdata[i]]*int(1/delta_T)    #偶位码元转换为对应的波形信号

a = []     #余弦函数载波
b = []     #正弦函数载波
for j in range(int(N/2)):
    a.append(np.math.sqrt(2/T)*np.math.cos(2*np.math.pi*fc*t[j]))    #余弦函数载波
    b.append(np.math.sqrt(2/T)*np.math.sin(2*np.math.pi*fc*t[j]))    #正弦函数载波
idata1 = np.array(ich)*np.array(a)          #奇数位数据与余弦函数相乘，得到一路的调制信号
qdata1 = np.array(qch)*np.array(b)          #偶数位数据与余弦函数相乘，得到另一路的调制信号
s = idata1 + qdata1      #将奇偶位数据合并，s即为QPSK调制信号


plt.figure(figsize=(14,12))
plt.subplot(3,1,1)
plt.plot(idata1)
plt.title('同相支路I',fontproperties=font, fontsize=20)
plt.axis([0,500,-3,3])
plt.subplot(3,1,2)
plt.plot(qdata1)
plt.title('正交支路Q',fontproperties=font, fontsize=20)
plt.axis([0,500,-3,3])
plt.subplot(3,1,3)
plt.plot(s)
plt.title('调制信号',fontproperties=font, fontsize=20)
plt.axis([0,500,-3,3])
plt.show()
