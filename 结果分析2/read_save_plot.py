#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:15:08 2019
此函数是读取所有泡的数据，前提是准备好了炮号，准备好炮号是代码selectdata.py做的
输入为炮号：密度极限破裂泡和安全泡。
输出为：
1. 每泡的数据，包括诊断信号，诊断信号是经过冲采样的，采样为1KHZ，这样保存时数据量少，便于计算
     外加Ngw，Ngw是需要算出来的，Ngw = Ip/(pi*a^2),如果没有a,则统一用0.45代替
2. 生成一个excel，每行的数据为：
    [
    0. 炮号,shot;
    1. 平顶端开始时刻;
    2. 平顶端结束时刻;
    3. 实验结束时刻/破裂时刻;
    4. 有(1)没有(0)aminor;
    5. 安全炮-1/密度极限破裂炮1
    ]
其中exel表格直接导入，不必计算，本程序的主要任务是读取数据到本地，并剔除
那些有信号(除了aminor)缺失的炮号
yes

@author: jack
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from MDSplus import *
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties
from MDSplus import Connection

font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 14)
font1 = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 12)
filepath  = '/home/jack/density/'
figpath = '/home/jack/picture/'
sample_rate = 0.001
column = 4

A = np.array(pd.read_csv('/home/jack/数据筛选/all.csv'))

def resample(signal):
    f = interp1d(signal[0],signal[1],kind='linear',fill_value="extrapolate")
    t1 = np.around(signal[0,0]/sample_rate)*sample_rate
    t2 = np.around(signal[0,-1]/sample_rate)*sample_rate
    new_time = np.arange(t1,t2,sample_rate)
    new_data = f(new_time)
    New_signal = np.zeros((2,len(new_time)))
    New_signal[0] = new_time
    New_signal[1] = new_data
    return New_signal

def smooth_aminor(T,V):
    if abs(V[0]-0.45)>0.1:
        y[0] = 0.45
    for i in range(len(T)-1):
        if abs(V[i+1]-V[i]) >0.1:
            V[i+1] = V[i]
    return T,V


def generte_aminor(t_s,t_e):
    time = np.arange(t1,t2,sample_rate)
    value = np.array([0.45]*len(time))


def read_save_plot(shot):
    index = np.where(A[:,0]==shot)[0][0]
    coon  = Connection('202.127.204.12')   # connect the MDSplus
    signa = ["\PCRL01","\DFSDEV","\AMINOR","\VP1","\SXR23D",\
             "\PXUV30","\PXUV18","\KMP13T"]

    ylabel = ["pcrl01(MA)",'dfsdev'+r'$(10^{19}m^{-3})$',"aminor(m)","vp1(V)","sxr23d(V)",\
            "pxuv30(V)","pxuv18(V)","kmp13t(V)"]

    Tree = ["pcs_east","pcs_east","efitrt_east","east","east","east","east",\
            "east"]

    dic = ['pcrl01','dfsdev','aminor','vp1','sxr23d','pxuv30','pxuv18','kmp13']

    pcrl01 = 0
    dfsdev = 0
    aminor = 0
    vp1 = 0
    sxr23d = 0
    pxuv30 = 0
    pxu18 = 0
    kmp13 = 0

    fig, axs = plt.subplots(4,2,sharex=True,figsize=(8,10))#figsize=(6,8)
    coon  = Connection('202.127.204.12')
    flag_amin = 1 #是否有aminor信号,1为有,0没有
    for i in range(8):
        try:
            coon.openTree(Tree[i], shot)   # open tree and shot
            value = np.array(coon.get(r'%s'%(signa[i])))   # read data
            #print('%s的单位：'%signame, coon.get('units(\\DFSDEV)'))
            t = np.array(coon.get('dim_of(%s)'%(signa[i]))) # read time data
            coon.closeTree('%s'%(Tree[i]), shot)
        except (TreeFOPENR,TreeNODATA,TdiRECURSIVE):
            if i == 2:
                flag_amin = 0
                print("%d 没有 %s..采用替换.."%(shot,signa[i]))
            else:
                miss_shot.append(shot)
                print("%d 没有 %s..终止.."%(shot,signa[i]))
                return

        if i==2 and flag_amin == 0:
            t = np.arange(A[index,5],A[index,6],sample_rate)
            value = np.array([0.45]*len(t))
            l = len(t)
            Value=np.zeros((2,len(t)))
            Value[0] = t
            Value[1] = value

        else:
            if i==2 and flag_amin == 1:
                t,value = smooth_aminor(t,value)

            l = len(t)
            if t[0] >= 0:
                pass
            else:
                a = np.where(t >= 0)[0][0]
                t = t[a:l-1]
                value = value[a:l-1]
            l = len(t)
            Value=np.zeros((2,len(t)))
            Value[0] = t
            Value[1] = value
            Value = resample(Value)
        Dir = filepath+'%d/'%(shot)
        if os.path.exists(Dir):
            pass
        else:
            os.makedirs(Dir)
        np.save(filepath+'%d/%d_%s.npy'%(shot,shot,dic[i]),Value)
        if signa[i] == "\PCRL01":
            Value[1] = Value[1]/10**6
        axs[i%column, i//column].plot(Value[0], Value[1])
        if i == column-1 or i == 2*column-1:
            axs[i%column,i//column].set_xlabel('time (s)',fontproperties = font1)
        axs[i%column,i//column].set_ylabel('%s'%(ylabel[i]),fontproperties = font1)
        fig.subplots_adjust(hspace=0.3,wspace=0.3)#调节两个子图间的距离
        plt.suptitle('pulse:%d'%shot,fontproperties = font )
        plt.savefig(figpath +'%d.eps'%shot, format='eps',dpi=1000)


miss_shot = []#记录缺少任何一个信号的炮号
read_save_plot(67039)
