#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:25:17 2024

@author: jack

https://blog.csdn.net/m0_46303328/article/details/121092456


"""

import  numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from matplotlib.font_manager import FontProperties


config = {
    "font.family": "serif",  # 使用衬线体
    "font.serif": ["SimSun"],  # 全局默认使用衬线宋体
    "font.size": 28,  # 五号，10.5磅
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
}
plt.rcParams.update(config)


digram = {'11':3, '10':1, '01':-1, '00':-3}#设置数字和幅度的对应关系
spots = {}#放置点

fig, axs = plt.subplots(1,1, figsize=(8, 8), constrained_layout=True)

## x/ylabel
axs.set_xlabel('I', loc='right', labelpad = 0.5)
axs.set_ylabel('Q', loc='top', labelpad = 0.5)  # 设置坐标轴的文字标签


## axis
axs.spines['right'].set_color('none')
axs.spines['top'].set_color('none')  # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
##
axs.xaxis.set_ticks_position('bottom')
axs.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴 指定左边的边为 y 轴
## axis pos
axs.spines['bottom'].set_position(('data', 0))  # 指定 data 设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
axs.spines['left'].set_position(('data', 0))
## axis linewidth
bw = 3
axs.spines['bottom'].set_linewidth(bw) ###设置底部坐标轴的粗细
axs.spines['left'].set_linewidth(bw)   ###设置左边坐标轴的粗细

axs.set_xlim([-5,5,]) ## 设置坐标的数字范围
axs.set_ylim([-5,5,]) ## 设置坐标的数字范围


for i in ['0','1']:
    for j in ['0','1']:
        for k in ['0','1']:
            for p in ['0','1']:
                 Str = ''.join([i,j,k,p])      # 通过循环获得16个4位数的10组合
                 str1 = ''.join([i,j])         # 前两个10组合
                 a = digram[str1]              # 获取前两个10组合对应的幅值
                 # a = int(a)
                 str2 = ''.join([k,p])
                 b = digram[str2]             #获取后两个10组合对应的幅值
                 # b = int(b)
                 complexSpot = complex(a, b)          # 不能写为a+bj，因为编译不通过 生成坐标
                 axs.scatter(a, b, c = 'b', s = 70)   # 绘制点
                 axs.text(a, b+0.3, Str, fontsize  = 25, color = "black", weight = "light", verticalalignment='center', horizontalalignment='center', rotation=0)#绘制10组合
                 tempspot = {Str:complexSpot}#获得数字组合和点
                 spots.update(tempspot)#存入点的集合

out_fig = plt.gcf()
# out_fig.savefig('/home/jack/snap/16QAM.eps',   bbox_inches = 'tight')
plt.show()



import numpy as np
import matplotlib.pyplot as plt

bitsToAmp = {'11': 3, '10': 1, '01': -1, '00': -3}  # 设置数字和幅度的对应关系
spots = {}  # 放置点

for i in ['0', '1']:
    for j in ['0', '1']:
        for k in ['0', '1']:
            for p in ['0', '1']:
                strs = ''.join([i, j, k, p])  # 通过循环获得16个4位数的10组合
                str1 = ''.join([i, j])  # 前两个10组合
                a = bitsToAmp[str1]  # 获取前两个10组合对应的幅值
                a = int(a)

                str2 = ''.join([k, p])
                b = bitsToAmp[str2]
                b = int(b)  # 获取后两个10组合对应的幅值

                complexSpot = complex(a, b)  # 不能写为 a+bj，报错;

                tempSpot = {strs: complexSpot}  # 获得数字组合和点
                spots.update(tempSpot)  # 存入点的集合

fig, axs = plt.subplots(2,1, figsize=(12, 6), constrained_layout=True)
t = np.arange(0, 12.0, 0.5)  # 设置基带信号10的坐标轴，每隔0.5的距离绘制一个基带的二进制信号，一共16个比特

# input
# axs.subplot(2, 1, 1)
y1 = [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1]
axs[0].plot(t, y1, drawstyle = 'steps-post', c = 'b', linewidth = 3)  # 将16个比特每隔0.5绘制到坐标系上
axs[0].set_xlim(-0.3, 12.5)
axs[0].set_ylim(-0.5, 1.5)
axs[0].set_title('16QAM modulation')

# 串并变换
l4 = int(len(y1) / 4)  # 获取比特流的长度除以4，4个比特为一组，则共有多少组
a = np.array(y1)  # 将基带信号转换为numpy格式
y2 = a.reshape(l4, 4)  # 将一维数组转置为二维数组，每一行中有4个比特的数据

# plt.subplot(2, 1, 2)
t = np.arange(0, 12., 0.01)  # 横坐标的数据列表，每个0.01绘制一个点
rectwav = []  # 用来存储纵坐标值的列表
strip = int(t.shape[0]/l4)
# i表示第i个线段，每个线段对应一个二进制的四位组合s0s1s2s3。每个线段的长度为2，是基带信号每个信号长度0.5的四倍
for i in range(l4):
    b = y2[i]  # 取出第i组四位数组合 s0 s1 s2 s3
    str4Bits =  str(b).strip('[').strip(']').replace(' ', '')  # 将列表中的4个比特转换为字符串并且去掉[ ] 和空格
    complexWave = spots[str4Bits]  # 根据四个比特的字符串对应到字典中的复数，得到横坐标和纵坐标的幅度，I Q的幅度值
    xWave = complexWave.real  # 取出横坐标的值
    yWave = complexWave.imag  # 取出纵坐标的值

    # 在t数组中第i段横坐标的点数，此处每个段的波形长度应该是0.5的4倍，也就是2
    t_tmp = t[(i * strip):((i + 1) * strip)]
    xI_tmp = xWave * np.ones(strip)  # 200个横坐标的幅度值
    yQ_tmp = yWave * np.ones(strip)  # 200个纵坐标的幅度值
    # 将幅度分别与两个正交载波相乘求和
    wav_tmp = xI_tmp * np.cos(2 * np.pi * 5 * t_tmp) - yQ_tmp * np.sin(2 * np.pi * 5 * t_tmp)
    rectwav.append(wav_tmp)  # 将调制后的点加到总的波形列表中

# 绘制调制后的波形
axs[1].plot(t, np.array(rectwav).flatten(), c = 'b', linewidth = 3)
axs[1].set_xlim(-0.3, 12.5)
axs[1].set_ylim(-5, 5)

out_fig = plt.gcf()
# out_fig.savefig('/home/jack/snap/16QAM_wave.eps',   bbox_inches = 'tight')
plt.show()






















































































































































































