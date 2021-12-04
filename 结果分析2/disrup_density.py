#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:46:06 2019

@author: jack

此代码功能是利用lmtipref求导区分破裂炮和非破裂炮,输入为炮号,可以是单个,
可以是很多.对输入的泡好判断是否是破裂跑/非破裂炮/密度极限破裂炮,亦或是废炮,
并将炮号的信息保存如下,arry的每行如下:
[
0:炮号,
1:有(1)没有(0)aminor,
2:是(1)否(0)没有平顶端,
3:是(1)否(0)为没用的炮(平顶端太短,如果是,这行数据后面的数不用)
4:安全炮(-1)还是破裂炮(1), 
5:平顶端开始时刻,
6:平顶端结束时刻,
7:破裂炮的破裂时刻/非破裂炮的end时刻,
8:是(1)否(0)为密度极限破裂炮,
9:如果是密度极限,ne/Ngw=0.8的时刻
]

aminor如果有,截取破裂前1s计算GreenWald曲线,如果没有,统一取0.45m
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MDSplus import *
from MDSplus import Connection
import xlrd,math
from scipy.interpolate import interp1d
from sklearn.preprocessing import minmax_scale
from matplotlib.font_manager import FontProperties

font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 14)
font1 = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 12)   
sample_rate = 0.001
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

def data_info():
    sheet = xlrd.open_workbook(r'/home/jack/公共的/excel_data/info.xlsx')
    sheet1 = sheet.sheet_by_name('Sheet1')
    row_num,col_num = sheet1.nrows,sheet1.ncols
    A=[]
    for i in range(row_num):
        A.append(sheet1.row_values(i))
    return np.array(A)

def resample(x,y,t_s,t_e):
    f = interp1d(x,y,kind='linear',fill_value="extrapolate")
    #t2 = math.ceil(signal[0,len(signal[0])-1]/sample_rate)*sample_rate
    X = np.arange(t_s,t_e,sample_rate)
    Y = f(X)
    return Y

def find_aminor(x,y,t_s,t_e):
    s = np.where(x >= t_s)[0][0]-2
    e = np.where(x >= t_e)[0][0]+1
    x = x[s:e]
    y = y[s:e]
    if abs(y[0]-0.45)>0.1:
        y[0] = 0.45
    for i in range(len(x)-1):
        if abs(y[i+1]-y[i]) >0.1:
            y[i+1] = y[i]
    return x,y
    
def check(shot):
    arry = [shot, 1, 0, 0, -1, -1, -1, -1, 0,-1]
    fig, axs = plt.subplots(5,1,sharex=True,figsize=(6,8))#figsize=(6,8)
    
    ############################读取PCRL01#######################################
    try:
        coon  = Connection('202.127.204.12')   # connect the MDSplus
        coon.openTree('pcs_east', shot)   # open tree and shot
        pcrl01 = np.array(coon.get(r'\PCRL01'))   # read data
    
        t0 = np.array(coon.get('dim_of(\PCRL01)')) # read time data
    except (TreeFOPENR,TreeNODATA):
        print("%d 没有pcrl01"%shot)
        return
    l = len(t0)
    if t0[0] >= 0:
        pass
    else:
        a = np.where(t0 >= 0)[0][0]
        t0 = t0[a:l-1]
        pcrl01 = pcrl01[a:l-1]
    pcrl_minmax = minmax_scale(pcrl01)
        #print('PCRL01的单位：',coon.get('units(\\PCRL01)'))
    
    #####################读取LMTIPREF##################################################
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    try:
        coon.openTree('pcs_east', shot)   # open tree and shot
        lmtipref = np.array(coon.get(r'\LMTIPREF'))   # read data
        t1 = np.array(coon.get('dim_of(\LMTIPREF)')) # read time data
    except (TreeFOPENR,TreeNODATA):
        print("%d 没有LMTIPREF"%shot)
    l = len(t1)
    if t1[0] >= 0:
        pass
    else:
        a = np.where(t1 >= 0)[0][0]
        t1 = t1[a:l-1]
        lmtipref = lmtipref[a:l-1]
        #print('\LMTIPREF的单位：',coon.get('units(\\LMTIPREF)'))
    #####################读取DFSDEV####################################################
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    try:
        coon.openTree('pcs_east', shot)   # open tree and shot
        dfsdev = np.array(coon.get(r'\DFSDEV'))   # read data
        t2 = np.array(coon.get('dim_of(\DFSDEV)')) # read time data
    except (TreeFOPENR,TreeNODATA):
        print("%d 没有DFSDEV"%shot)
        return
    l = len(t2)
    if t2[0] >= 0:
       pass
    else:
        a = np.where(t2 >= 0)[0][0]
        t2 = t2[a:l-1]
        dfsdev = dfsdev[a:l-1]
        #print('DFSDEV的单位：',coon.get('units(\\DFSDEV)'))
    
    
    #####################读取AMINOR#########################################
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    #t3 = np.zeros((3))
    try:
        coon.openTree('efitrt_east', shot)   # open tree and shot
        aminor = np.array(coon.get(r'\AMINOR'))   # read data
        t3 = np.array(coon.get('dim_of(\AMINOR)')) # read time data
        axs[4].plot(t3,aminor)
            #print('AMINOR的单位：',coon.get('units(\\AMINOR)'))
    except TreeFOPENR:
        print("%d 没有aminor"%shot)
        arry[1] = 0
    except TdiRECURSIVE:
        arry[1] = 0
        print("%d aminor timeout"%shot)
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
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
    except (IndexError):
        arry[2] = 1
        without_flat.append(arry)
        print("%d没有平顶端"%shot)
        
        return
    if (t1_2-t1_1) < 1:
        print("%d 平顶端太短"%shot)
        arry[3] = 1
        flat_short.append(arry)
        return
    
    if t1[-2]-t1_2 < 0.3:#如果lmtipref导数的最后时刻-平顶端最后时刻<400ms，为破裂炮
        arry[4] = 1
    last_density = t2[-1]
    arry[5] = t1_1+0.2 #平顶端开始时刻
    
    diff_ip = np.diff(pcrl_minmax)
    diff_t0 = np.diff(t0)
    dif_ip = diff_ip/diff_t0
    t0_max = t0[np.where(dif_ip == dif_ip.min())[0][0]]
    
    if arry[4] == -1:#安全炮
        #arry[4] = t1_1+0.1
        arry[6] = min(t1_2,last_density) #平顶端结束
        arry[7] = min(t1_2,last_density)-0.1 #实验结束时刻
    else:#破裂炮
        #arry[4] = t1_1+0.1
        arry[6] = min(t1_2,last_density,t0_max) #平顶端结束
        if t1_2 < t0_max:
            arry[7] = min(t1_2,last_density) #破裂时刻
        else:
            arry[7] = min(t0_max,last_density)-0.02
        #接下来判断是否为密度极限破裂炮
    if arry[1] == 0:#如果没有aminor
        t_end = np.around((min(t0_max,t1_2,last_density,)-0.02)/sample_rate)*sample_rate
    else:
        t_end = np.around((min(t1_2,last_density,t0_max,t3[-1])-0.02)/sample_rate)*sample_rate
    t_sta = t_end-1
    if arry[1] == 1 and t_sta < max(t1_1,t3[0]):
        t_sta = max(t1_1,t3[0])
    Ip = resample(t0,pcrl01,t_sta,t_end)
    Ne = resample(t2,dfsdev,t_sta,t_end)
    if arry[1] == 0:
        amin = np.array([0.45]*len(Ip))
    else:
        amin_t,amin_y = find_aminor(t3.copy(),aminor.copy(),t_sta,t_end)
        amin = resample(amin_t,amin_y,t_sta,t_end)
    Ngw = Ip*(10**-6)*10/(np.pi*amin**2)
    R  = Ne/Ngw
    t_c = np.arange(t_sta,t_end,sample_rate)
    if len(np.where(R>=0.8)[0]) !=0:
        arry[8] = 1
        #找出密度达到Ngw时刻
        t_denlimdis = t_c[np.where(R>=0.8)[0][0]]
        arry[9] = t_denlimdis  
    ######################开始画图#########################
    axs[0].plot(t0,pcrl01,color='b')
    axs[0].set_ylabel("pcrl01",fontproperties = font1)
    

    axs[1].plot(t0,pcrl_minmax,color='b')
    ax = axs[1].twinx()
    ax.plot(t0[:-1],dif_ip,'r--',linewidth=0.8)
    axs[1].axvline( x= t0_max,ls='-',linewidth=3,color='g',)
    ax.set_ylabel(r"$\frac{\mathit{d}pcrl01}{\mathit{d}t}$",fontproperties = font1)
    axs[1].set_ylabel("pcrl01(0-1)",fontproperties = font1)

    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        
    axs[2].plot(t1,lmtipref,color='b')
    axs[2].plot(t1[:-1],dif_lmtipref,linestyle='--',color='r')
    axs[2].axvline( x= t1_1,ls='-',linewidth=1,color='g',)
    axs[2].axvline( x= t1_2,ls='-',linewidth=1,color='g',)
    axs[2].set_ylabel("lmtipref(A)",fontproperties = font1)
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
    t2_1 = t2[-1]
    axs[3].plot(t2,dfsdev)
    axs[3].axvline( x= t2_1,ls='--',linewidth=1,color='g',)
    axs[3].set_ylabel("dfsdev"+r'($10^{19}/m^3$)',fontproperties = font1)

    axs[3].plot(t_c,Ngw,'r-',label='Green wald')
    axs[3].legend()

    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    axs[4].set_ylabel('aminor(m)',fontproperties = font1)
    axs[4].set_xlabel('time(s)',fontproperties = font1)
    ######################################################################
    if arry[4]==-1:
        plt.suptitle('safe pulse:%d'%shot,)
    elif arry[4] == 1 and arry[8] == 0:
        plt.suptitle('disruptive pulse:%d'%shot,)
    elif arry[8] == 1:
        plt.suptitle('density disruptive pulse:%d'%shot,)
    plt.show()
    #print(arry)
    if arry[4] == -1:
        safe_shot.append(arry)
    elif arry[4] ==1:
        dis_shot.append(arry)
    elif arry[8] == 1:
        den_shot.append(arry)
    return


#np.savez_compressed('/home/jack/snap/Bsafe.npz',Bsafe=B)
#B的第一列为炮号，第二列为记录的破裂时间，第三列为标签，第四列为d(pcrl01)/d(t)最大值时刻\
#第五列为lmtipref导数为0的最后一个点时刻,第六列为lmtipref最后时刻点
A  = data_info()
safe_shot = []
dis_shot = []
den_shot = []
flat_short = []
without_flat = []
#check(69086)

for i in A[:492,0]:
    check(i)
safe = np.array(safe_shot)
dis = np.array(dis_shot)
den_shot = np.array(den_shot)
short = np.array(flat_short)
without = np.array(without_flat)