#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:46:06 2019

@author: jack

这是查看某一炮的pcrl01,lmtipref,dfsdev,aminor的文件
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
filepath  = '/home/jack/Density1/'
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

A = np.array(pd.read_csv('/home/jack/数据筛选/last5.csv'))


coon  = Connection('202.127.204.12')   # connect the MDSplus
miss_shot = []#记录缺少任何一个信号的炮号


shot = 69491

index = np.where(A[:,0]==shot)[0][0]
#coon  = Connection('202.127.204.12')   # connect the MDSplus
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
    #return
#l = len(t1)
if t1[0] >= 0:
    pass
else:
    a = np.where(t1 >= 0)[0][0]
    t1 = t1[a:]
    pcrl01 = pcrl01[a:]

Pcrl01 = np.zeros((2,len(t1)))
Pcrl01[0:] = t1
Pcrl01[1:] = pcrl01
del t1
del pcrl01

############################################################
try:
    coon.openTree('pcs_east', shot)   # open tree and shot
    dfsdev = np.array(coon.get(r'\DFSDEV'))   # read data
    t2 = np.array(coon.get('dim_of(\DFSDEV)')) # read time data
    coon.closeTree('pcs_east', shot)
except (TreeFOPENR,TreeNODATA,TreeNNF):
    print("%d 没有dfsdev,终止"%shot)
    miss_shot.append(shot)


if t2[0] >= 0:
    pass
else:
    a = np.where(t2 >= 0)[0][0]
    t2 = t2[a:]
    dfsdev = dfsdev[a:]
Dfsdev = np.zeros((2,len(t2)))
Dfsdev[0] = t2
Dfsdev[1] = dfsdev
#Dfsdev = resample(t2,dfsdev)
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
        t3 = t3[a:]
        aminor = aminor[a:]
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
    #return
#l = len(t4)
if t4[0] >= 0:
    pass
else:
    a = np.where(t4 >= 0)[0][0]
    t4 = t4[a:]
    vp1 = vp1[a:]
Vp1 = np.zeros((2,len(t4)))
Vp1[0] = t4
Vp1[1] = vp1
#Vp1 = resample(t4,vp1)
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
    #return
#l = len(t5)
if t5[0] >= 0:
    pass
else:
    a = np.where(t5 >= 0)[0][0]
    t5 = t5[a:]
    sxr23d = sxr23d[a:]
Sxr23d = np.zeros((2,len(t5)))
Sxr23d[0] = t5
Sxr23d[1] = sxr23d
#Sxr23d = resample(t5,sxr23d)
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
    #return
#l = len(t6)
if t6[0] >= 0:
    pass
else:
    a = np.where(t6 >= 0)[0][0]
    t6 = t6[a:]
    pxuv30 = pxuv30[a:]
Pxuv30 = np.zeros((2,len(t6)))
Pxuv30[0] = t6
Pxuv30[1] = pxuv30
#Pxuv30 = resample(t6,pxuv30)
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
    #return
#l = len(t7)
if t7[0] >= 0:
    pass
else:
    a = np.where(t7 >= 0)[0][0]
    t7 = t7[a:]
    pxuv18 = pxuv18[a:]
Pxuv18 = np.zeros((2,len(t7)))
Pxuv18[0] = t7
Pxuv18[1] = pxuv18
#Pxuv18 = resample(t7,pxuv18)
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
    #return
#l = len(t8)
if t8[0] >= 0:
    pass
else:
    a = np.where(t8 >= 0)[0][0]
    t8 = t8[a:]
    kmp13t = kmp13t[a:]
Kmp13t = np.zeros((2,len(t8)))
Kmp13t[0] = t8
Kmp13t[1] = kmp13t
#Kmp13t = resample(t8,kmp13t)
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
    #return
#l = len(t9)
if t9[0] >= 0:
    pass
else:
    a = np.where(t9 >= 0)[0][0]
    t9 = t9[a:]
    pbrem10 = pbrem10[a:]
Pbrem10 = np.zeros((2,len(t9)))
Pbrem10[0] = t9
Pbrem10[1] = pbrem10
#Pbrem10 = resample(t9,pbrem10)
del t9
del pbrem10
####################################################################
#Miss = []
try:
    coon.openTree('efit_east', shot)   # open tree and shot
    q95 = np.array(coon.get(r'\q95'))   # read data
    t10 = np.array(coon.get('dim_of(\q95)')) # read time data
    coon.closeTree('efit_east', shot)

except (TreeFOPENR,TreeNODATA,TreeNNF):
    print("%d 没有安全因子q95,终止"%shot)
    miss_shot.append(shot)
    #return
#l = len(t10)
if t10[0] >= 0:
    pass
else:
    a = np.where(t10 >= 0)[0][0]
    t10 = t10[a:]
    q95 = q95[a:]
Q95 = np.zeros((2,len(t10)))
Q95[0] = t10
Q95[1] = q95
#Q95 = resample(t10,q95)
del t10
del q95

####################################################################
try:
    coon.openTree('efit_east', shot)   # open tree and shot
    li = np.array(coon.get(r'\li'))   # read data
    t11 = np.array(coon.get('dim_of(\li)')) # read time data
    coon.closeTree('efit_east', shot)


except (TreeFOPENR,TreeNODATA,TreeNNF):
    print("%d 没有内感,终止"%shot)
    miss_shot.append(shot)
    #return


#l = len(t11)
if t11[0] >= 0:
    pass
else:
    a = np.where(t11 >= 0)[0][0]
    t11 = t11[a:]
    li = li[a:]
Li = np.zeros((2,len(t11)))
Li[0] = t11
Li[1] = li
#Li = resample(t11,li)
del t11
del li
####################################################################
try:
    coon.openTree('efit_east', shot)   # open tree and shot
    betap = np.array(coon.get(r'\BETAP'))   # read data
    t12 = np.array(coon.get('dim_of(\BETAP)')) # read time data
    coon.closeTree('efit_east', shot)

except (TreeFOPENR,TreeNODATA,TreeNNF,TdiBOMB):
    print("%d 没有极向比压,终止"%shot)
    miss_shot.append(shot)
    #return
#l = len(t12)
if t12[0] >= 0:
    pass
else:
    a = np.where(t11 >= 0)[0][0]
    t12 = t12[a:]
    betap = betap[a:]
Betap = np.zeros((2,len(t12)))
Betap[0] = t12
Betap[1] = betap
#Betap = resample(t12,betap)
del t12
del betap
####################################################################
try:
    coon.openTree('pcs_east', shot)   # open tree and shot
    lmsz = np.array(coon.get(r'\LMSZ'))   # read data
    t13 = np.array(coon.get('dim_of(\LMSZ)')) # read time data
    coon.closeTree('pcs_east', shot)


except (TreeFOPENR,TreeNODATA,TreeNNF):
    print("%d 没有极向比压,终止"%shot)
    miss_shot.append(shot)
    #return

#l = len(t13)
if t13[0] >= 0:
    pass
else:
    a = np.where(t13 >= 0)[0][0]
    t13 = t13[a:]
    lmsz = lmsz[a:]
Lmsz = np.zeros((2,len(t13)))
Lmsz[0] = t13
Lmsz[1] = lmsz
#Lmsz = resample(t13,lmsz)
del t13
del lmsz
####################################################################
try:
    coon.openTree('pcs_east', shot)   # open tree and shot
    ic = np.array(coon.get(r'\IC1'))   # read data
    t14 = np.array(coon.get('dim_of(\IC1)')) # read time data
    coon.closeTree('pcs_east', shot)


except (TreeFOPENR,TreeNODATA,TreeNNF):
    print("%d 没有极向比压,终止"%shot)
    miss_shot.append(shot)

#l = len(t14)
if t14[0] >= 0:
    pass
else:
    a = np.where(t14 >= 0)[0][0]
    t14 = t14[a:]
    ic = ic[a:]
Ic = np.zeros((2,len(t14)))
Ic[0] = t14
Ic[1] = ic
#Ic = resample(t14,ic)
del t14
del ic


#np.savez_compressed(filepath+'%d.npz'%(shot),pcrl01=Pcrl01,dfsdev=Dfsdev,\
#                    aminor=Aminor,vp1=Vp1,sxr23d=Sxr23d,pxuv30=Pxuv30,\
 #                   pxuv18=Pxuv18,kmp13t=Kmp13t,pbrem10 = Pbrem10,lmsz=Lmsz,ic=Ic,\
  #                  q95=Q95,li=Li,betap=Betap)
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
    plt.suptitle('nondisruptive pulse:%d'%shot,x=0.5, y=0.95,fontproperties = font )
elif A[index,4] == 1:
    plt.suptitle('density limit disruptive:%d'%shot,x=0.5,y=0.95,fontproperties = font )
#plt.savefig(figpath +'%d.eps'%shot, format='eps',dpi=1000)
plt.show()
#print("tt11:", time.ctime())





#read_save_plot(47327)
#for i in range(1122,len(A)):
#    read_save_plot(A[i,0])

