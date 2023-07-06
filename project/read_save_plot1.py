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
import os,time
import matplotlib.pyplot as plt
from MDSplus import *
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties
from MDSplus import Connection

def resample(x,y):
    f = interp1d(x,y,kind='linear',fill_value="extrapolate")
    t1 = np.around(x[0]/sample_rate)*sample_rate
    t2 = np.around(x[-1]/sample_rate+1)*sample_rate
    x = np.arange(t1,t2,sample_rate)
    y = f(x)
    signal = np.zeros((2,len(x)))
    signal[0] = x
    signal[1] = y
    return signal


def read_save_plot(shot):
    index = np.where(A[:,0]==shot)[0][0]
    coon  = Connection('202.127.204.12')   # connect the MDSplus
#############################################################
    #print("tt1:", time.ctime())

    try:
        coon.openTree('pcs_east', shot)   # open tree and shot
        pcrl01 = np.array(coon.get(r'\PCRL01'))   # read data
        t1 = np.array(coon.get('dim_of(\PCRL01)')) # read time data
        coon.closeTree('pcs_east', shot)
    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有pcrl01,终止"%shot)
        miss_shot.append(shot)
        return
    l = len(t1)
    if t1[0] >= 0:
        pass
    else:
        a = np.where(t1 >= 0)[0][0]
        t1 = t1[a:l-1]
        pcrl01 = pcrl01[a:l-1]
    Pcrl01 = resample(t1,pcrl01)
    del t1
    del pcrl01
    #print("tt2:", time.ctime())
############################################################
    try:
        coon.openTree('pcs_east', shot)   # open tree and shot
        dfsdev = np.array(coon.get(r'\DFSDEV'))   # read data
        t2 = np.array(coon.get('dim_of(\DFSDEV)')) # read time data
        coon.closeTree('pcs_east', shot)
    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有dfsdev,终止"%shot)
        miss_shot.append(shot)
        return
    l = len(t2)
    if t2[0] >= 0:
        pass
    else:
        a = np.where(t2 >= 0)[0][0]
        t2 = t2[a:l-1]
        dfsdev = dfsdev[a:l-1]
    Dfsdev = resample(t2,dfsdev)
    del t2
    del dfsdev
    #print("tt3:", time.ctime())
############################################################
    try:
        coon.openTree('efitrt_east', shot)   # open tree and shot
        aminor = np.array(coon.get(r'\AMINOR'))   # read data
        t3 = np.array(coon.get('dim_of(\AMINOR)')) # read time data
        coon.closeTree('efitrt_east', shot)
        if t3[0] >= 0:
            pass
        else:
            l = len(t3)
            a = np.where(t3 >= 0)[0][0]
            t3 = t3[a:l-1]
            aminor = aminor[a:l-1]
        Aminor = resample(t3,aminor)
    except (TreeFOPENR,TreeNODATA,TreeNNF,TdiRECURSIVE):
        print("%d 没有aminor,替代"%shot)
        t3 = np.arange(A[index,5],A[index,6],sample_rate)
        aminor = np.array([0.45]*len(t3))
        Aminor = np.zeros((2,len(t3)))
        Aminor[0] = t3
        Aminor[1] = aminor
        #print("tt4:", time.ctime())
    del t3, aminor
#####################################################
    try:
        coon.openTree('east', shot)   # open tree and shot
        vp1 = np.array(coon.get(r'\VP1'))   # read data
        t4 = np.array(coon.get('dim_of(\VP1)')) # read time data
        coon.closeTree('east', shot)
    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有vp1,终止"%shot)
        miss_shot.append(shot)
        return
    l = len(t4)
    if t4[0] >= 0:
        pass
    else:
        a = np.where(t4 >= 0)[0][0]
        t4 = t4[a:l-1]
        vp1 = vp1[a:l-1]
    Vp1 = resample(t4,vp1)
    del t4
    del vp1
    #print("tt5:", time.ctime())
##############################################################
    try:
        coon.openTree('east', shot)   # open tree and shot
        sxr23d = np.array(coon.get(r'\SXR23D'))   # read data
        t5 = np.array(coon.get('dim_of(\SXR23D)')) # read time data
        coon.closeTree('east', shot)
    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有sxr23d,终止"%shot)
        miss_shot.append(shot)
        return
    l = len(t5)
    if t5[0] >= 0:
        pass
    else:
        a = np.where(t5 >= 0)[0][0]
        t5 = t5[a:l-1]
        sxr23d = sxr23d[a:l-1]
    Sxr23d = resample(t5,sxr23d)
    del t5
    del sxr23d
    #print("tt6:", time.ctime())
########################################################
    try:
        coon.openTree('east', shot)   # open tree and shot
        pxuv30 = np.array(coon.get(r'\PXUV30'))   # read data
        t6 = np.array(coon.get('dim_of(\PXUV30)')) # read time data
        coon.closeTree('east', shot)
    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有pxuv30,终止"%shot)
        miss_shot.append(shot)
        return
    l = len(t6)
    if t6[0] >= 0:
        pass
    else:
        a = np.where(t6 >= 0)[0][0]
        t6 = t6[a:l-1]
        pxuv30 = pxuv30[a:l-1]
    Pxuv30 = resample(t6,pxuv30)
    del t6
    del pxuv30
    #print("tt7:", time.ctime())
#################################################################
    try:
        coon.openTree('east', shot)   # open tree and shot
        pxuv18 = np.array(coon.get(r'\PXUV18'))   # read data
        t7 = np.array(coon.get('dim_of(\PXUV18)')) # read time data
        coon.closeTree('east', shot)
    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有pxuv18,终止"%shot)
        miss_shot.append(shot)
        return
    l = len(t7)
    if t7[0] >= 0:
        pass
    else:
        a = np.where(t7 >= 0)[0][0]
        t7 = t7[a:l-1]
        pxuv18 = pxuv18[a:l-1]
    Pxuv18 = resample(t7,pxuv18)
    del t7
    del pxuv18
    #print("tt8:", time.ctime())
###################################################################
    try:
        coon.openTree('east', shot)   # open tree and shot
        kmp13t = np.array(coon.get(r'\KMP13T'))   # read data
        t8 = np.array(coon.get('dim_of(\KMP13T)')) # read time data
        coon.closeTree('east', shot)
    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有kmp13t,终止"%shot)
        miss_shot.append(shot)
        return
    l = len(t8)
    if t8[0] >= 0:
        pass
    else:
        a = np.where(t8 >= 0)[0][0]
        t8 = t8[a:l-1]
        kmp13t = kmp13t[a:l-1]
    Kmp13t = resample(t8,kmp13t)
    del t8
    del kmp13t
    #print("tt9:", time.ctime())
####################################################################
    try:
        coon.openTree('east', shot)   # open tree and shot
        pbrem10 = np.array(coon.get(r'\VBM10'))   # read data
        t9 = np.array(coon.get('dim_of(\VBM10)')) # read time data
        coon.closeTree('east', shot)
    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有VBM10,终止"%shot)
        miss_shot.append(shot)
        return
    l = len(t9)
    if t9[0] >= 0:
        pass
    else:
        a = np.where(t9 >= 0)[0][0]
        t9 = t9[a:l-1]
        pbrem10 = pbrem10[a:l-1]
    Pbrem10 = resample(t9,pbrem10)
    del t9
    del pbrem10
####################################################################
    signa = ["\PCRL01","\DFSDEV","\AMINOR","\VP1","\SXR23D",\
             "\PXUV30","\PXUV18","\KMP13T"]

    ylabel = ["pcrl01(MA)",'dfsdev'+r'$(10^{19}m^{-3})$',"aminor(m)","vp1(V)","sxr23d(V)",\
            "pxuv30(V)","pxuv18(V)","kmp13t(V)"]

    Tree = ["pcs_east","pcs_east","efitrt_east","east","east","east","east",\
            "east"]

    dic = ['pcrl01','dfsdev','aminor','vp1','sxr23d','pxuv30','pxuv18','kmp13']



    fig, axs = plt.subplots(5,2,sharex=True,figsize=(8,8))#figsize=(6,8)


    #np.savez_compressed(filepath+'%d.npz'%(shot),pcrl01=Pcrl01,dfsdev=Dfsdev,\
    #       aminor=Aminor,vp1=Vp1,sxr23d=Sxr23d,pxuv30=Pxuv30,\
    #       pxuv18=Pxuv18,kmp13t=Kmp13t,pbrem10 = Pbrem10)
    #print("tt10:", time.ctime())

    axs[0,0].plot(Pcrl01[0], Pcrl01[1]/10**6)
    axs[0,0].set_ylabel("pcrl01(MA)")

    axs[1,0].plot(Dfsdev[0], Dfsdev[1])
    axs[1,0].set_ylabel('dfsdev'+r'$(10^{19}m^{-3})$')

    axs[2,0].plot(Aminor[0], Aminor[1])
    axs[2,0].set_ylabel("aminor(m)")

    axs[3,0].plot(Vp1[0], Vp1[1])
    axs[3,0].set_ylabel("vp1(V)")

    axs[0,1].plot(Sxr23d[0], Sxr23d[1])
    axs[0,1].set_ylabel("sxr23d(V)")

    axs[1,1].plot(Pxuv30[0], Pxuv30[1])
    axs[1,1].set_ylabel("pxuv30(V)")

    axs[2,1].plot(Pxuv18[0], Pxuv18[1])
    axs[2,1].set_ylabel("pxuv18(V)")

    axs[3,1].plot(Kmp13t[0], Kmp13t[1])
    axs[3,1].set_ylabel("kmp13t(V)")

    axs[4,1].plot(Pbrem10[0], Pbrem10[1])
    axs[4,1].set_ylabel("pbrem(V)")
    
    axs[4,0].set_xlabel('time (s)',fontproperties = font1)
    axs[4,1].set_xlabel('time (s)',fontproperties = font1)

    fig.subplots_adjust(hspace=0.3,wspace=0.3)#调节两个子图间的距离
    if A[index,8] == 0:
        plt.suptitle('nondisruptive pulse:%d'%shot,fontproperties = font )
    elif A[index,8] == 1:
        plt.suptitle('density limit disruptive:%d'%shot,fontproperties = font )
    #plt.savefig(figpath +'%d.eps'%shot, format='eps',dpi=1000)
    plt.show()
    #print("tt11:", time.ctime())
    return


font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 14)
font1 = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 12)
filepath  = '/home/jack/density1/'
if os.path.exists(filepath):
    pass
else:
    os.makedirs(filepath)
figpath = '/home/jack/picture1/'
if os.path.exists(figpath):
    pass
else:
    os.makedirs(figpath)
sample_rate = 0.001
column = 4

A = np.array(pd.read_csv('/home/jack/数据筛选/last2.csv'))

#read_save_plot(85013)

miss_shot = []#记录缺少任何一个信号的炮号
for i in [52434., 61364., 61366., 76459., 77315., 78990., 79265., 80146.,
       81142., 81208., 81411., 82576., 84215., 84353., 85013., 85025.,
       86099.]:
    read_save_plot(i)
'''
miss = np.array(miss_shot)
Miss = pd.DataFrame(miss,columns=['miss_shot'])
Miss.to_csv('/home/jack/数据筛选/miss1.csv',index=False)
'''
