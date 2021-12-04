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
import  MDSplus
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties
from MDSplus import Connection

def resample(x,y):
    f = interp1d(x,y,kind='linear',fill_value="extrapolate")
    t1 = np.around(x[0]/sample_rate)/1000
    t2 = np.around(x[-1]/sample_rate+1)/1000
    x = np.arange(t1,t2,sample_rate)
    y = f(x)
    signal = np.zeros((2,len(x)))
    signal[0] = x
    signal[1] = y
    return signal


def kill_nan(arr):
    a = np.where(np.isnan(arr))
    for i in a[0]:
        arr[i] = arr[i-1]
    return arr

def killanbormal(arr):
    for i in range(1,arr.shape[1]):
        if arr[1,i]<0:
            arr[1,i] = arr[1,i-1]
    return arr


def read_save_plot(shot):
    index = np.where(A[:,0]==shot)[0][0]
    
    print("正在读取%d炮..."%shot)
#############################################################
    #print("tt1:", time.ctime())
    coon  = Connection('202.127.204.12')   # connect the MDSplus
    
    
    try:
        #coon  = Connection('202.127.204.12')   # connect the MDSplus
        coon.openTree('pcs_east', shot)   # open tree and shot
        pcrl01 = np.array(coon.get(r'\PCRL01'))   # read data
        t1 = np.array(coon.get('dim_of(\PCRL01)')) # read time data
        coon.closeTree('pcs_east', shot)
        print("%d的PCRL01读取成功..."%shot)
    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有pcrl01,终止"%shot)
        miss_shot.append(shot)
        return
    #l = len(t1)
    if t1[0] >= 0:
        pass
    else:
        a = np.where(t1 >= 0)[0][0]
        t1 = t1[a:]
        pcrl01 = pcrl01[a:]
        
    Pcrl01 = resample(t1,pcrl01)
    del t1
    del pcrl01
    #print("tt2:", time.ctime())
############################################################
    try:
        #coon  = Connection('202.127.204.12')   # connect the MDSplus
        coon.openTree('pcs_east', shot)   # open tree and shot
        dfsdev = np.array(coon.get(r'\DFSDEV'))   # read data
        t2 = np.array(coon.get('dim_of(\DFSDEV)')) # read time data
        coon.closeTree('pcs_east', shot)
        print("%d的DFSDEV读取成功..."%shot)
    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有dfsdev,终止"%shot)
        miss_shot.append(shot)
        return
    #l = len(t2)
    if t2[0] >= 0:
        pass
    else:
        a = np.where(t2 >= 0)[0][0]
        t2 = t2[a:]
        dfsdev = dfsdev[a:]
        
    Dfsdev = resample(t2,dfsdev)
    del t2
    del dfsdev
    #print("tt3:", time.ctime())
############################################################
    try:
        #coon  = Connection('202.127.204.12')   # connect the MDSplus
        coon.openTree('efitrt_east', shot)   # open tree and shot
        aminor = np.array(coon.get(r'\AMINOR'))   # read data
        t3 = np.array(coon.get('dim_of(\AMINOR)')) # read time data
        coon.closeTree('efitrt_east', shot)
        print("%d的AMINOR读取成功..."%shot)
        if t3[0] >= 0:
            pass
        else:
            #l = len(t3)
            a = np.where(t3 >= 0)[0][0]
            t3 = t3[a:]
            aminor = aminor[a:]
        Aminor = resample(t3,aminor)
    except :#(TreeFOPENR,TreeNODATA,TreeNNF,TdiRECURSIVE):
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
        #coon  = Connection('202.127.204.12')   # connect the MDSplus
        coon.openTree('east_1', shot)   # open tree and shot
        vp1 = np.array(coon.get(r'\VP1'))   # read data
        t4 = np.array(coon.get('dim_of(\VP1)')) # read time data
        coon.closeTree('east_1', shot)
        print("%d的VP1读取成功..."%shot)
    except :#(TreeFOPENR,TreeNODATA,TreeNNF,):
        print("%d 没有vp1,终止"%shot)
        miss_shot.append(shot)
        return
    #l = len(t4)
    if t4[0] >= 0:
        pass
    else:
        a = np.where(t4 >= 0)[0][0]
        t4 = t4[a:]
        vp1 = vp1[a:]
        
    Vp1 = resample(t4,vp1)
    del t4
    del vp1
    #print("tt5:", time.ctime())
##############################################################
    try:
        #coon  = Connection('202.127.204.12')   # connect the MDSplus
        coon.openTree('east_1', shot)   # open tree and shot
        sxr23d = np.array(coon.get(r'\SXR23D'))   # read data
        t5 = np.array(coon.get('dim_of(\SXR23D)')) # read time data
        coon.closeTree('east_1', shot)
        print("%d的SXR23D读取成功..."%shot)
    except :#(TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有sxr23d,终止"%shot)
        miss_shot.append(shot)
        return
    #l = len(t5)
    if t5[0] >= 0:
        pass
    else:
        a = np.where(t5 >= 0)[0][0]
        t5 = t5[a:]
        sxr23d = sxr23d[a:]
        
    Sxr23d = resample(t5,sxr23d)
    del t5
    del sxr23d
    #print("tt6:", time.ctime())
########################################################
    try:
        #coon  = Connection('202.127.204.12')   # connect the MDSplus
        coon.openTree('east_1', shot)   # open tree and shot
        pxuv30 = np.array(coon.get(r'\PXUV30'))   # read data
        t6 = np.array(coon.get('dim_of(\PXUV30)')) # read time data
        coon.closeTree('east_1', shot)
        print("%d的PXUV30读取成功..."%shot)
    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有pxuv30,终止"%shot)
        miss_shot.append(shot)
        return
    #l = len(t6)
    if t6[0] >= 0:
        pass
    else:
        a = np.where(t6 >= 0)[0][0]
        t6 = t6[a:]
        pxuv30 = pxuv30[a:]
        
    Pxuv30 = resample(t6,pxuv30)
    del t6
    del pxuv30
    #print("tt7:", time.ctime())
#################################################################
    try:
        #coon  = Connection('202.127.204.12')   # connect the MDSplus
        coon.openTree('east_1', shot)   # open tree and shot
        pxuv18 = np.array(coon.get(r'\PXUV18'))   # read data
        t7 = np.array(coon.get('dim_of(\PXUV18)')) # read time data
        coon.closeTree('east_1', shot)
        print("%d的PXUV18读取成功..."%shot)
    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有pxuv18,终止"%shot)
        miss_shot.append(shot)
        return
    #l = len(t7)
    if t7[0] >= 0:
        pass
    else:
        a = np.where(t7 >= 0)[0][0]
        t7 = t7[a:]
        pxuv18 = pxuv18[a:]
        
    Pxuv18 = resample(t7,pxuv18)
    del t7
    del pxuv18
    #print("tt8:", time.ctime())
###################################################################
    try:
        #coon  = Connection('202.127.204.12')   # connect the MDSplus
        coon.openTree('east_1', shot)   # open tree and shot
        kmp13t = np.array(coon.get(r'\KMP13T'))   # read data
        t8 = np.array(coon.get('dim_of(\KMP13T)')) # read time data
        coon.closeTree('east_1', shot)
        print("%d的KMP13T读取成功..."%shot)
    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有kmp13t,终止"%shot)
        miss_shot.append(shot)
        return
    #l = len(t8)
    if t8[0] >= 0:
        pass
    else:
        a = np.where(t8 >= 0)[0][0]
        t8 = t8[a:]
        kmp13t = kmp13t[a:]
        
    Kmp13t = resample(t8,kmp13t)
    
    del t8
    del kmp13t
    #print("tt9:", time.ctime())
####################################################################
    try:
        #coon  = Connection('202.127.204.12')   # connect the MDSplus
        coon.openTree('east_1', shot)   # open tree and shot
        pbrem10 = np.array(coon.get(r'\VBM10'))   # read data
        t9 = np.array(coon.get('dim_of(\VBM10)')) # read time data
        coon.closeTree('east_1', shot)
        print("%d的VBM10读取成功..."%shot)
    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有VBM10,终止"%shot)
        miss_shot.append(shot)
        return
    #l = len(t9)
    if t9[0] >= 0:
        pass
    else:
        a = np.where(t9 >= 0)[0][0]
        t9 = t9[a:]
        pbrem10 = pbrem10[a:]
        
    Pbrem10 = resample(t9,pbrem10)
    del t9
    del pbrem10
####################################################################
    #Miss = []
    try:
        #coon  = Connection('202.127.204.12')   # connect the MDSplus
        coon.openTree('efit_east', shot)   # open tree and shot
        q95 = np.array(coon.get(r'\q95'))   # read data
        q95 = kill_nan(q95)
        t10 = np.array(coon.get('dim_of(\q95)')) # read time data
        coon.closeTree('efit_east', shot)
        print("%d的q95读取成功..."%shot)
    except :#(TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有安全因子q95,终止"%shot)
        miss_shot.append(shot)
        return
    #l = len(t10)
    if t10[0] >= 0:
        pass
    else:
        a = np.where(t10 >= 0)[0][0]
        t10 = t10[a:]
        q95 = q95[a:]
        
    Q95 = resample(t10,q95)
    del t10
    del q95

####################################################################
    try:
        #coon  = Connection('202.127.204.12')   # connect the MDSplus
        coon.openTree('efit_east', shot)   # open tree and shot
        li = np.array(coon.get(r'\li'))   # read data
        t11 = np.array(coon.get('dim_of(\li)')) # read time data
        coon.closeTree('efit_east', shot)
        print("%d的li读取成功..."%shot)

    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有内感,终止"%shot)
        miss_shot.append(shot)
        return


    #l = len(t11)
    if t11[0] >= 0:
        pass
    else:
        a = np.where(t11 >= 0)[0][0]
        t11 = t11[a:]
        li = li[a:]
        
    Li = resample(t11,li)
    del t11
    del li
####################################################################
    try:
        #coon  = Connection('202.127.204.12')   # connect the MDSplus
        coon.openTree('efit_east', shot)   # open tree and shot
        betap = np.array(coon.get(r'\BETAP'))   # read data
        t12 = np.array(coon.get('dim_of(\BETAP)')) # read time data
        coon.closeTree('efit_east', shot)
        print("%d的BETAP读取成功..."%shot)
    except (TreeFOPENR,TreeNODATA,TreeNNF,TdiBOMB):
        print("%d 没有极向比压,终止"%shot)
        miss_shot.append(shot)
        return
    #l = len(t12)
    if t12[0] >= 0:
        pass
    else:
        a = np.where(t11 >= 0)[0][0]
        t12 = t12[a:]
        betap = betap[a:]
        
    Betap = resample(t12,betap)
    del t12
    del betap
####################################################################
    try:
        #coon  = Connection('202.127.204.12')   # connect the MDSplus
        coon.openTree('pcs_east', shot)   # open tree and shot
        lmsz = np.array(coon.get(r'\LMSZ'))   # read data
        t13 = np.array(coon.get('dim_of(\LMSZ)')) # read time data
        coon.closeTree('pcs_east', shot)
        print("%d的LMSZ读取成功..."%shot)

    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有极向比压,终止"%shot)
        miss_shot.append(shot)
        return

    #l = len(t13)
    if t13[0] >= 0:
        pass
    else:
        a = np.where(t13 >= 0)[0][0]
        t13 = t13[a:]
        lmsz = lmsz[a:]
        
    Lmsz = resample(t13,lmsz)
    del t13
    del lmsz
####################################################################
    try:
        #coon  = Connection('202.127.204.12')   # connect the MDSplus
        coon.openTree('pcs_east', shot)   # open tree and shot
        ic = np.array(coon.get(r'\IC1'))   # read data
        t14 = np.array(coon.get('dim_of(\IC1)')) # read time data
        coon.closeTree('pcs_east', shot)
        print("%d的IC1读取成功..."%shot)

    except (TreeFOPENR,TreeNODATA,TreeNNF):
        print("%d 没有极向比压,终止"%shot)
        miss_shot.append(shot)
        return
   
    coon.disconnect()

    #l = len(t14)
    if t14[0] >= 0:
        pass
    else:
        a = np.where(t14 >= 0)[0][0]
        t14 = t14[a:]
        ic = ic[a:]
        
    Ic = resample(t14,ic)
    del t14
    del ic

    #np.savez_compressed(filepath+'%d.npz'%(shot),pcrl01=Pcrl01,dfsdev=Dfsdev,\
     #                   aminor=Aminor,vp1=Vp1,sxr23d=Sxr23d,pxuv30=Pxuv30,\
     #                   pxuv18=Pxuv18,kmp13t=Kmp13t,pbrem10 = Pbrem10,lmsz=Lmsz,ic=Ic,\
     #                   q95=Q95,li=Li,betap=Betap)
    #print("tt10:", time.ctime())
    
    fig, axs = plt.subplots(7,2,sharex=True,figsize=(8,10))#figsize=(6,8)

    axs[0,0].plot(Pcrl01[0], Pcrl01[1]/10**6)
    axs[0,0].set_ylabel("pcrl01(MA)")

    axs[1,0].plot(Dfsdev[0], Dfsdev[1])
    axs[1,0].set_ylabel('dfsdev'+r'$(10^{19}m^{-3})$')

    axs[2,0].plot(Aminor[0], Aminor[1])
    axs[2,0].set_ylabel("aminor(m)")

    axs[3,0].plot(Vp1[0], Vp1[1])
    axs[3,0].set_ylabel("vp1(V)")

    axs[4,0].plot(Q95[0], Q95[1])
    axs[4,0].set_ylabel("q95(a.u)")

    axs[5,0].plot(Li[0], Li[1])
    axs[5,0].set_ylabel("Li(a.u)")

    axs[6,0].plot(Lmsz[0], Lmsz[1])
    axs[6,0].set_ylabel("Lmsz(m)")

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

    axs[5,1].plot(Betap[0], Betap[1])
    axs[5,1].set_ylabel("Betap(a.u)")

    axs[6,1].plot(Ic[0], Ic[1])
    axs[6,1].set_ylabel("Ic(A)")

    axs[6,0].set_xlabel('time (s)',fontproperties = font1)
    axs[6,1].set_xlabel('time (s)',fontproperties = font1)

    fig.subplots_adjust(hspace=0.3,wspace=0.3)#调节两个子图间的距离
    if A[index,4] == -1:
        plt.suptitle('nondisruptive pulse:%d'%shot,x=0.5,y=0.93,fontproperties = font )
    elif A[index,4] == 1:
        plt.suptitle('density limit disruptive:%d'%shot,x=0.5,y=0.93,fontproperties = font )
    #plt.savefig(figpath +'%d.eps'%shot, format='eps',dpi=1000)
    plt.show()
    #print("tt11:", time.ctime())
    return


font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 14)
font1 = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 12)

filepath  = '/home/jack/Density/'

if os.path.exists(filepath):
    pass
else:
    os.makedirs(filepath)
figpath = '/home/jack/Picture/'
if os.path.exists(figpath):
    pass
else:
    os.makedirs(figpath)
sample_rate = 0.001
column = 4

A = np.array(pd.read_csv('/home/jack/数据筛选/allsafe_exceedGW.csv'))
B = np.array(pd.read_csv('/home/jack/数据筛选/last8.csv'))[:,0]


miss_shot = []#记录缺少任何一个信号的炮号

#read_save_plot(47327)
for i in [58471.0,
 74719.0,
 76209.0,
 80170.0,
 82491.0,
 82850.0,
 83875.0,
 85315.0,
 85634.0]:
    if i in B:
        pass
    else:
        read_save_plot(i)



'''
miss_shot = [68071.0, 68379.0, 68381.0, 68383.0, 69901.0]
'''

