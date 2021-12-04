#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:46:06 2019

@author: jack

这是在线查看某一炮所有诊断的代码

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


font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 14)
font1 = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 12)

fontt = {'family':'Times New Roman','style':'normal','size':14}
fontt1 = {'style':'normal','weight':'bold','size':16}
fontt2 = {'style':'normal','size':16}

sample_rate = 0.001
column = 4


coon  = Connection('202.127.204.12')   # connect the MDSplus
miss_shot = []#记录缺少任何一个信号的炮号


shot = 43017

#############################################################
#print("tt1:", time.ctime())


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


##############################################################
try:
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    coon.openTree('east_1', shot)   # open tree and shot
    sxr23d = np.array(coon.get(r'\SXR23D'))   # read data
    t3 = np.array(coon.get('dim_of(\SXR23D)')) # read time data
    coon.closeTree('east_1', shot)
    print("%d的SXR23D读取成功..."%shot)
except :#(TreeFOPENR,TreeNODATA,TreeNNF):
    print("%d 没有sxr23d,终止"%shot)
    miss_shot.append(shot)

#l = len(t5)
if t3[0] >= 0:
    pass
else:
    a = np.where(t3 >= 0)[0][0]
    t3 = t3[a:]
    sxr23d = sxr23d[a:]

Sxr23d = resample(t3,sxr23d)
del t3
del sxr23d
#print("tt6:", time.ctime())
########################################################
try:
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    coon.openTree('east_1', shot)   # open tree and shot
    pxuv30 = np.array(coon.get(r'\PXUV30'))   # read data
    t4 = np.array(coon.get('dim_of(\PXUV30)')) # read time data
    coon.closeTree('east_1', shot)
    print("%d的PXUV30读取成功..."%shot)
except (TreeFOPENR,TreeNODATA,TreeNNF):
    print("%d 没有pxuv30,终止"%shot)
    miss_shot.append(shot)

#l = len(t6)
if t4[0] >= 0:
    pass
else:
    a = np.where(t4 >= 0)[0][0]
    t4 = t4[a:]
    pxuv30 = pxuv30[a:]

Pxuv30 = resample(t4,pxuv30)
del t4
del pxuv30
#print("tt7:", time.ctime())
#################################################################
try:
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    coon.openTree('east_1', shot)   # open tree and shot
    pxuv18 = np.array(coon.get(r'\PXUV18'))   # read data
    t5 = np.array(coon.get('dim_of(\PXUV18)')) # read time data
    coon.closeTree('east_1', shot)
    print("%d的PXUV18读取成功..."%shot)
except (TreeFOPENR,TreeNODATA,TreeNNF):
    print("%d 没有pxuv18,终止"%shot)
    miss_shot.append(shot)

#l = len(t7)
if t5[0] >= 0:
    pass
else:
    a = np.where(t5 >= 0)[0][0]
    t5 = t5[a:]
    pxuv18 = pxuv18[a:]

Pxuv18 = resample(t5,pxuv18)
del t5
del pxuv18
#print("tt8:", time.ctime())
###################################################################
try:
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    coon.openTree('east_1', shot)   # open tree and shot
    kmp13t = np.array(coon.get(r'\KMP13T'))   # read data
    t6 = np.array(coon.get('dim_of(\KMP13T)')) # read time data
    coon.closeTree('east_1', shot)
    print("%d的G3读取成功..."%shot)
except (TreeFOPENR,TreeNODATA,TreeNNF):
    print("%d 没有G3,终止"%shot)
    miss_shot.append(shot)

#l = len(t8)
if t6[0] >= 0:
    pass
else:
    a = np.where(t6 >= 0)[0][0]
    t6 = t6[a:]
    kmp13t = kmp13t[a:]

Kmp13t = resample(t6,kmp13t)

del t6
del kmp13t
#print("tt9:", time.ctime())
####################################################################

####################################################################
try:
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    coon.openTree('east_1', shot)   # open tree and shot
    c3 = np.array(coon.get(r'\CIIIL3'))   # read data
    t7 = np.array(coon.get('dim_of(\CIIIL3)')) # read time data
    print('Te的单位：',coon.get('units(\\CIIIL3)'))
    coon.closeTree('east_1', shot)
    print("%d的CIIIL3读取成功..."%shot)
    
except (TreeFOPENR,TreeNODATA,TreeNNF):
    print("%d 没有CIIIL3,终止"%shot)
    miss_shot.append(shot)

#l = len(t13)
if t7[0] >= 0:
    pass
else:
    a = np.where(t7 >= 0)[0][0]
    t7 = t7[a:]
    c3 = c3[a:]

C3 = resample(t7,c3)
del t7
del c3
####################################################################
try:
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    coon.openTree('east_1', shot)   # open tree and shot
    vp = np.array(coon.get(r'\VP1'))   # read data
    t8 = np.array(coon.get('dim_of(\VP1)')) # read time data
    coon.closeTree('east_1', shot)
    print("%d的IC1读取成功..."%shot)

except (TreeFOPENR,TreeNODATA,TreeNNF):
    print("%d 没有vp1,终止"%shot)
    miss_shot.append(shot)


coon.disconnect()

#l = len(t14)
if t8[0] >= 0:
    pass
else:
    a = np.where(t8 >= 0)[0][0]
    t8 = t8[a:]
    vp = vp[a:]

Vp = resample(t8,vp)
del t8
del vp

####################################################################
try:
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    coon.openTree('east_1', shot)   # open tree and shot
    plhi1 = np.array(coon.get(r'\PLHI1'))   # read data
    t9 = np.array(coon.get('dim_of(\PLHI1)')) # read time data
    print('PLHI1的单位：',coon.get('units(\\PLHI1)'))
    coon.closeTree('east_1', shot)
    
    print("%d的PLHI1读取成功..."%shot)

except (TreeFOPENR,TreeNODATA,TreeNNF):
    print("%d 没有极向比压,终止"%shot)
    miss_shot.append(shot)


coon.disconnect()

#l = len(t14)
if t9[0] >= 0:
    pass
else:
    a = np.where(t9 >= 0)[0][0]
    t9 = t9[a:]
    plhi1 = plhi1[a:]

Plhi1 = resample(t9,plhi1)
del t9
del plhi1


####################################################################
try:
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    coon.openTree('east_1', shot)   # open tree and shot
    plhr1 = np.array(coon.get(r'\PLHR1'))   # read data
    t10 = np.array(coon.get('dim_of(\PLHR1)')) # read time data
    print('PLHR1的单位：',coon.get('units(\\PLHR1)'))
    coon.closeTree('east_1', shot)
    print("%d的PLHR1读取成功..."%shot)

except (TreeFOPENR,TreeNODATA,TreeNNF):
    print("%d 没有极向比压,终止"%shot)
    miss_shot.append(shot)


coon.disconnect()

#l = len(t14)
if t10[0] >= 0:
    pass
else:
    a = np.where(t10 >= 0)[0][0]
    t10 = t10[a:]
    plhr1 = plhr1[a:]

Plhr1 = resample(t10,plhr1)
del t10
del plhr1

####################################################################
try:
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    coon.openTree('east_1', shot)   # open tree and shot
    plhi2 = np.array(coon.get(r'\PLHI2'))   # read data
    t11 = np.array(coon.get('dim_of(\PLHI2)')) # read time data
    print('PLHI2的单位：',coon.get('units(\\PLHI2)'))
    coon.closeTree('east_1', shot)
    print("%d的PLHI2读取成功..."%shot)

except (TreeFOPENR,TreeNODATA,TreeNNF):
    print("%d 没有极向比压,终止"%shot)
    miss_shot.append(shot)


coon.disconnect()

#l = len(t14)
if t11[0] >= 0:
    pass
else:
    a = np.where(t11 >= 0)[0][0]
    t11 = t11[a:]
    plhi2 = plhi2[a:]

Plhi2 = resample(t11,plhi2)
del t11
del plhi2


####################################################################
try:
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    coon.openTree('east_1', shot)   # open tree and shot
    plhr2 = np.array(coon.get(r'\PLHR2'))   # read data
    t12 = np.array(coon.get('dim_of(\PLHR2)')) # read time data
    print('PLHR2的单位：',coon.get('units(\\PLHR2)'))
    coon.closeTree('east_1', shot)
    print("%d的PLHR2读取成功..."%shot)

except (TreeFOPENR,TreeNODATA,TreeNNF):
    print("%d 没有极向比压,终止"%shot)
    miss_shot.append(shot)


coon.disconnect()

#l = len(t14)
if t12[0] >= 0:
    pass
else:
    a = np.where(t12 >= 0)[0][0]
    t12 = t12[a:]
    plhr2 = plhr2[a:]

Plhr2 = resample(t12,plhr2)
del t12
del plhr2

#np.savez_compressed(filepath+'%d.npz'%(shot),pcrl01=Pcrl01,dfsdev=Dfsdev,\
 #                   aminor=Aminor,vp1=Vp1,sxr23d=Sxr23d,pxuv30=Pxuv30,\
 #                   pxuv18=Pxuv18,kmp13t=Kmp13t,pbrem10 = Pbrem10,lmsz=Lmsz,ic=Ic,\
 #                   q95=Q95,li=Li,betap=Betap)
#print("tt10:", time.ctime())

fig, axs = plt.subplots(6,2,sharex=True,figsize=(12,16))#figsize=(6,8)

axs[0,0].plot(Pcrl01[0], Pcrl01[1]/10**6,linewidth = 2,color='black')
axs[0,0].set_ylabel(r"$I_{P}$/MA",fontdict=fontt2,)
axs[0,0].set_title('(a)', loc = 'left',fontdict=fontt1)

axs[0,0].tick_params(labelsize=16,width=3)
Labels = axs[0,0].get_xticklabels() + axs[0,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in Labels]

axs[0,0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[0,0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[0,0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[0,0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

axs[1,0].plot(Dfsdev[0], Dfsdev[1],linewidth = 2,color='black')
axs[1,0].set_ylabel(r'$n_{e}/10^{19}m^{-3}$',fontdict=fontt2,)
axs[1,0].set_title('(b)', loc = 'left',fontdict=fontt1)

# 设置坐标刻度值的大小以及刻度值的字体
axs[1,0].tick_params(labelsize=16,width=3)
Labels = axs[1,0].get_xticklabels() + axs[1,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in Labels]

axs[1,0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[1,0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[1,0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[1,0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


axs[2,0].plot(C3[0], C3[1],linewidth = 2,color='black')
axs[2,0].set_ylabel(r"$CIII$/V",fontdict=fontt2,)
axs[2,0].set_title('(c)', loc = 'left',fontdict=fontt1)

axs[2,0].tick_params(labelsize=16,width=3)
Labels = axs[2,0].get_xticklabels() + axs[2,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in Labels]
[label.set_fontsize(18) for label in Labels]#刻度值字号

axs[2,0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[2,0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[2,0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[2,0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


axs[3,0].plot(Sxr23d[0], Sxr23d[1],linewidth = 2,color='black')
axs[3,0].set_ylabel("SXR/V",fontdict=fontt2,)
axs[3,0].set_title('(d)', loc = 'left',fontdict=fontt1)

axs[3,0].tick_params(labelsize=16,width=3)
Labels = axs[3,0].get_xticklabels() + axs[3,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in Labels]
[label.set_fontsize(18) for label in Labels]#刻度值字号

axs[3,0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[3,0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[3,0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[3,0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


axs[4,0].plot(Plhi1[0], Plhi1[1],linewidth = 2,color='black')
axs[4,0].set_ylabel(r"$P_{input1}$/KW",fontdict=fontt2,)
axs[4,0].set_title('(e)', loc = 'left',fontdict=fontt1)

axs[4,0].tick_params(labelsize=16,width=3)
Labels = axs[4,0].get_xticklabels() + axs[4,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in Labels]
[label.set_fontsize(18) for label in Labels]#刻度值字号

axs[4,0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[4,0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[4,0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[4,0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

axs[5,0].plot(Plhr1[0], Plhr1[1],linewidth = 2,color='black')
axs[5,0].set_ylabel(r"$P_{reflect1}$/KW",fontdict=fontt2,)
axs[5,0].set_title('(f)', loc = 'left',fontdict=fontt1)

axs[5,0].tick_params(labelsize=16,width=3)
Labels = axs[5,0].get_xticklabels() + axs[5,0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in Labels]
[label.set_fontsize(18) for label in Labels]#刻度值字号

axs[5,0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[5,0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[5,0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[5,0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细



axs[0,1].plot(Pxuv30[0], Pxuv30[1],linewidth = 2,color='black')
axs[0,1].set_ylabel(r"$XUV_{core}$/V",fontdict=fontt2,)
axs[0,1].set_title('(g)', loc = 'left',fontdict=fontt1)

axs[0,1].tick_params(labelsize=16,width=3)
Labels = axs[0,1].get_xticklabels() + axs[0,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in Labels]
[label.set_fontsize(18) for label in Labels]#刻度值字号

axs[0,1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[0,1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[0,1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[0,1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


axs[1,1].plot(Pxuv18[0], Pxuv18[1],linewidth = 2,color='black')
axs[1,1].set_ylabel(r"$XUV_{edge}$/V",fontdict=fontt2,)
axs[1,1].set_title('(h)', loc = 'left',fontdict=fontt1)
axs[1,1].tick_params(labelsize=16,width=3)
Labels = axs[1,1].get_xticklabels() + axs[1,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in Labels]
[label.set_fontsize(18) for label in Labels]#刻度值字号

axs[1,1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[1,1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[1,1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[1,1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


axs[2,1].plot(Kmp13t[0], Kmp13t[1],linewidth = 2,color='black')
axs[2,1].set_ylabel("Mirnov/V",fontdict=fontt2,)
axs[2,1].set_title('(i)', loc = 'left',fontdict=fontt1)
axs[2,1].tick_params(labelsize=16,width=3)
Labels = axs[2,1].get_xticklabels() + axs[2,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in Labels]
[label.set_fontsize(18) for label in Labels]#刻度值字号

axs[2,1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[2,1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[2,1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[2,1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


axs[3,1].plot(Vp[0], Vp[1],linewidth = 2,color='black')
axs[3,1].set_ylabel(r"$V_{loop}$/V",fontdict=fontt2,)
axs[3,1].set_title('(j)', loc = 'left',fontdict=fontt1)
axs[3,1].tick_params(labelsize=16,width=3)
Labels = axs[3,1].get_xticklabels() + axs[3,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in Labels]
[label.set_fontsize(18) for label in Labels]#刻度值字号

axs[3,1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[3,1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[3,1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[3,1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


axs[4,1].plot(Plhi2[0], Plhi2[1],linewidth = 2,color='black')
axs[4,1].set_ylabel(r"$P_{input2}$/KW",fontdict=fontt2,)
axs[4,1].set_title('(k)', loc = 'left',fontdict=fontt1)

axs[4,1].tick_params(labelsize=16,width=3)
Labels = axs[4,1].get_xticklabels() + axs[4,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in Labels]
[label.set_fontsize(18) for label in Labels]#刻度值字号

axs[4,1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[4,1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[4,1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[4,1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

axs[5,1].plot(Plhr2[0], Plhr2[1],linewidth = 2,color='black')
axs[5,1].set_ylabel(r"$P_{reflect2}$/KW",fontdict=fontt2,)
axs[5,1].set_title('(l)', loc = 'left',fontdict=fontt1)

axs[5,1].tick_params(labelsize=16,width=3)
Labels = axs[5,1].get_xticklabels() + axs[5,1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in Labels]
[label.set_fontsize(18) for label in Labels]#刻度值字号

axs[5,1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
axs[5,1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
axs[5,1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
axs[5,1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

#axs[3,0].set_xlim([0,6])
#axs[3,1].set_xlim([0,6])


axs[5,0].set_xlabel('time (s)', fontdict=fontt1)
axs[5,1].set_xlabel('time (s)', fontdict=fontt1)


fig.subplots_adjust(hspace=0.3,wspace=0.3)#调节两个子图间的距离
plt.suptitle('Pulse:%d'%shot,x=0.5, y=0.95,fontsize=18,fontweight='bold',)
out_fig = plt.gcf()
out_fig.savefig('/home/jack/Resultpicture/' +'%d.eps'%shot, format='eps',dpi=1000)
plt.show()
#print("tt11:", time.ctime())

