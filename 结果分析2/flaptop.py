#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:46:06 2019

@author: jack

此代码是测试利用lmtipref求导找到平顶端时刻可行性，结果是非常靠谱

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MDSplus import *
from MDSplus import Connection
import xlrd
from sklearn.preprocessing import minmax_scale
from matplotlib.font_manager import FontProperties

font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 14)
font1 = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 12)   
#shot = 67039
'''
class secletdata(object):
    def __init__(self):
'''
def data_info():
    sheet = xlrd.open_workbook(r'/home/jack/公共的/excel_data/info.xlsx')
    sheet1 = sheet.sheet_by_name('Sheet1')
    row_num,col_num = sheet1.nrows,sheet1.ncols
    A=[]
    for i in range(row_num):
        A.append(sheet1.row_values(i))
    return np.array(A)

def check(shot):
    index = np.where(A == shot)[0][0]
    fig, axs = plt.subplots(5,1,sharex=True,figsize=(6,8))#figsize=(6,8)
    ###################################################################
    coon  = Connection('202.127.204.12')   # connect the MDSplus
    coon.openTree('pcs_east', shot)   # open tree and shot
    pcrl01 = np.array(coon.get(r'\PCRL01'))   # read data
    pcrl01 = minmax_scale(pcrl01)
    t0 = np.array(coon.get('dim_of(\PCRL01)')) # read time data
    l = len(t0)
    if t0[0] >= 0:
        pass
    else:
        a = np.where(t0 >= 0)[0][0]
        t0 = t0[a:l-1]
        pcrl01 = pcrl01[a:l-1]
        #print('PCRL01的单位：',coon.get('units(\\PCRL01)'))
    
    #######################################################################
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    coon.openTree('pcs_east', shot)   # open tree and shot
    lmtipref = np.array(coon.get(r'\LMTIPREF'))   # read data
    t1 = np.array(coon.get('dim_of(\LMTIPREF)')) # read time data
    l = len(t1)
    if t1[0] >= 0:
        pass
    else:
        a = np.where(t1 >= 0)[0][0]
        t1 = t1[a:l-1]
        lmtipref = lmtipref[a:l-1]
        #print('\LMTIPREF的单位：',coon.get('units(\\LMTIPREF)'))
    
    
    #########################################################################
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    coon.openTree('pcs_east', shot)   # open tree and shot
    dfsdev = np.array(coon.get(r'\DFSDEV'))   # read data
    t2 = np.array(coon.get('dim_of(\DFSDEV)')) # read time data
    l = len(t2)
    if t2[0] >= 0:
       pass
    else:
        a = np.where(t2 >= 0)[0][0]
        t2 = t2[a:l-1]
        dfsdev = dfsdev[a:l-1]
        #print('DFSDEV的单位：',coon.get('units(\\DFSDEV)'))
    
    
    ##############################################################
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    #t3 = np.zeros((3))
    try:
        coon.openTree('efitrt_east', shot)   # open tree and shot
        aminor = np.array(coon.get(r'\AMINOR'))   # read data
        t3 = np.array(coon.get('dim_of(\AMINOR)')) # read time data
        axs[3].plot(t3,aminor)
            #print('AMINOR的单位：',coon.get('units(\\AMINOR)'))
    except TreeFOPENR:
        print("%d 没有aminor"%shot)
        #print(t3)
    except TdiRECURSIVE:
        print("%d aminor timeout"%shot)
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
    diff_ip = np.diff(pcrl01)
    diff_t0 = np.diff(t0)
    dif_ip = diff_ip/diff_t0
    t0_max = t0[np.where(dif_ip == dif_ip.min())[0][0]]
    axs[0].plot(t0,pcrl01,color='b')
    ax = axs[0].twinx()
    ax.plot(t0[:-1],dif_ip,'r--',linewidth=0.8)
    axs[0].axvline( x= t0_max,ls='-',linewidth=3,color='g',)
    t0_f = A[np.where(A==shot)[0][0],1]

    axs[0].axvline( x= t0_f,ls='-',linewidth=2,color='k',)
    ax.set_ylabel(r"$\frac{\mathit{d}pcrl01}{\mathit{d}t}$",fontproperties = font1)
    axs[0].set_ylabel("pcrl01(0-1)",fontproperties = font1)

    B[index,3] = t0_max
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    diff_lm = np.diff(lmtipref)
    diff_t1 = np.diff(t1)
    dif_lmtipref = diff_lm/diff_t1
    try:
        t1_1 = t1[np.where(abs(dif_lmtipref) <= 0.001)[0][0]]
        
        id1 = np.where(abs(dif_lmtipref) <= 0.001)[0][-1]
        if id1 == len(dif_lmtipref)-1 or (id1 < len(dif_lmtipref)-1 and dif_lmtipref[-1]<0):
            t1_2 = t1[id1]
        elif id1 < len(dif_lmtipref)-1 and dif_lmtipref[-1]>0:
            t1_2 = t1[len(dif_lmtipref)]
        B[index,4] = t1_1
        B[index,5] = t1_2
    except (IndexError):
        print("%d没有平顶端"%shot)
        
    axs[1].plot(t1,lmtipref,color='b')
    axs[1].plot(t1[:-1],dif_lmtipref,linestyle='--',color='r')
    axs[1].axvline( x= t1_1,ls='-',linewidth=1,color='g',)
    axs[1].axvline( x= t1_2,ls='-',linewidth=1,color='g',)
    axs[1].set_ylabel("lmtipref(A)",fontproperties = font1)
    B[index,6] = t1[-1]
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
    t2_1 = t2[-1]
    axs[2].plot(t2,dfsdev)
    axs[2].axvline( x= t2_1,ls='--',linewidth=1,color='g',)
    axs[2].set_ylabel("dfsdev"+r'($10^{19}/m^3$)',fontproperties = font1)
    B[index,7] = t2_1
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    axs[3].set_ylabel('aminor(m)',fontproperties = font1)
    
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    axs[4].set_xlabel('time(s)',fontproperties = font1)
    ######################################################################
    plt.suptitle('pulse:%d'%shot,)
    plt.show()
    return

A = data_info()
(m,n) = A.shape
B = np.zeros((m,n+5))
#np.savez_compressed('/home/jack/snap/Bsafe.npz',Bsafe=B)
#B的第一列为炮号，第二列为记录的破裂时间，第三列为标签，第四列为d(pcrl01)/d(t)最大值时刻
#第五列为lmtipref导数为0的第一个点时刻,第六列为lmtipref导数为0的最后一个点时刻,
#第七列为lmtipref最后时刻点,第八列为dfsdev的最后时刻
B[:,0:3] = A
safe_shut = []
dis_shut = []
wiout_flat = []
#check(70382)
for i in A[:493,0]:
    check(i)
#np.savez_compressed('/home/jack/snap/Bdis.npz',Bdis=B)