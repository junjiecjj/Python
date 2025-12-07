#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Mar 29 12:46:06 2019
@author: jack

此代码功能是利用lmtipref求导区分破裂炮和非破裂炮,输入为炮号,可以是单个,可以是很多.
对输入的泡号判断是否是破裂跑/非破裂炮,并从安全炮里面选出ne/ngw>=0.8
并将炮号的信息保存如下,arry的每行如下:
[
0:炮号,
1:有(1)没有(0)aminor,
2:有(1)没有(0)平顶端,
3:是(1)不是(0)没用的炮(平顶端太短,如果是,这行数据后面的数不用),
4:安全炮(-1)还是破裂炮(1),
5:平顶端开始时刻,
6:平顶端结束时刻,
7:破裂炮的破裂时刻/非破裂炮的end时刻,
8:是(1)否(0)实际密度超过Grenwald,结合arry[4]可判断是否为密度极限破裂.安全炮也可能超过GW,
9:如果arry[8]=1,ne/Ngw=0.8的时刻,
]

aminor如果有,截取破裂前1s计算GreenWald曲线,如果没有,统一取0.45m.

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



def resample(x,y,t_s,t_e):
    f = interp1d(x,y,kind='linear',fill_value="extrapolate")
    #t2 = math.ceil(signal[0,len(signal[0])-1]/sample_rate)*sample_rate
    X = np.arange(t_s,t_e+0.001,sample_rate)
    Y = f(X)
    return Y



def check(shot):
    arry = [shot, 1, 1, 0, -1, -1, -1, -1, 0, -1]

    ############################读取PCRL01#######################################
    try:
        coon  = Connection('202.127.204.12')   # connect the MDSplus
        coon.openTree('pcs_east', shot)   # open tree and shot
        pcrl01 = np.array(coon.get(r'\PCRL01'))   # read data
        t0 = np.array(coon.get('dim_of(\PCRL01)')) # read time data
        coon.closeTree('pcs_east', shot)
    except (TreeFOPENR,TreeNODATA):
        print("%d 没有pcrl01,终止"%shot)
        return
    l = len(t0)
    if t0[0] >= 0:
        pass
    else:
        try:
            a = np.where(t0 >= 0)[0][0]
            t0 = t0[a:]
            pcrl01 = pcrl01[a:]
        except IndexError:
            print("%d 的pcrl时间没有正值,终止"%shot)
            return
    pcrl_minmax = minmax_scale(pcrl01)
    #print('PCRL01的单位：',coon.get('units(\\PCRL01)'))

    ############################ 读取LMTIPREF #######################################
    #coon  = Connection('202.127.204.12')                  # connect the MDSplus
    try:
        coon.openTree('pcs_east', shot)                    # open tree and shot
        lmtipref = np.array(coon.get(r'\LMTIPREF'))        # read data
        t1 = np.array(coon.get('dim_of(\LMTIPREF)'))       # read time data
        coon.closeTree('pcs_east', shot)
    except (TreeFOPENR,TreeNODATA):
        print("%d 没有LMTIPREF,终止"%shot)
        return
    l = len(t1)
    if t1[0] >= 0:
        pass
    else:
        a = np.where(t1 >= 0)[0][0]
        t1 = t1[a:]
        lmtipref = lmtipref[a:]
        #print('\LMTIPREF的单位：',coon.get('units(\\LMTIPREF)'))

    diff_lm = np.diff(lmtipref)
    diff_t1 = np.diff(t1)
    dif_lmtipref = diff_lm/diff_t1
    try:
        #t1_1平顶端开始时刻
        t1_1 = t1[np.where(abs(dif_lmtipref) <= 0.002)[0][0]]

        id1 = np.where(abs(dif_lmtipref) <= 0.002)[0][-1]
        #t1_2平顶端结束时刻
        if id1 == len(dif_lmtipref)-1 or (id1 < len(dif_lmtipref)-1 and dif_lmtipref[-1]<0):
            t1_2 = t1[id1]
        elif id1 < len(dif_lmtipref)-1 and dif_lmtipref[-1]>0:
            t1_2 = t1[len(dif_lmtipref)]
    except (IndexError):
        arry[2] = 0
        #  without_flat.append(arry)
        print("%d没有平顶端,终止"%shot)
        return
    if (t1_2-t1_1) < 2:
        print("%d 平顶端太短,终止"%shot)
        arry[3] = 1
        #  flat_short.append(arry)
        return

    #####################读取DFSDEV####################################################
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    try:
        coon.openTree('pcs_east', shot)   # open tree and shot
        dfsdev = np.array(coon.get(r'\DFSDEV'))   # read data
        t2 = np.array(coon.get('dim_of(\DFSDEV)')) # read time data
        coon.closeTree('pcs_east', shot)
    except (TreeFOPENR,TreeNODATA,TdiTIMEOUT):
        print("%d 没有DFSDEV,终止"%shot)
        return
    l = len(t2)
    if t2[0] >= 0:
       pass
    else:
        a = np.where(t2 >= 0)[0][0]
        t2 = t2[a:]
        dfsdev = dfsdev[a:]
    if dfsdev.max()>15:
        print("%d密度过大，舍去..."%shot)
        return
        #print('DFSDEV的单位：',coon.get('units(\\DFSDEV)'))

    #####################读取AMINOR#####################################################
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    #t3 = np.zeros((3))

    try:
        coon.openTree('efitrt_east', shot)   # open tree and shot
        aminor = np.array(coon.get(r'\AMINOR'))   # read data
        t3 = np.array(coon.get('dim_of(\AMINOR)')) # read time data
        coon.closeTree('efitrt_east', shot)
        #axs[4].plot(t3,aminor)
            #print('AMINOR的单位：',coon.get('units(\\AMINOR)'))
    except (TdiRECURSIVE,TdiRECURSIVE,TreeFOPENR):
        print("%d 没有aminor"%shot)
        arry[1] = 0

    ####################################################################################

    if t1[-2]-t1_2 < 1: #如果lmtipref导数的最后时刻-平顶端最后时刻<500ms，为破裂炮
        return

    last_density = t2[-1]
    arry[5] = np.ceil((t1_1+0.2)*1000)/1000   #平顶端开始时刻

    diff_ip = np.diff(pcrl_minmax)
    diff_t0 = np.diff(t0)
    dif_ip = diff_ip/diff_t0
    ind_flap = np.where(t0>=t1_1)[0][0]
    t0_max =t0[ind_flap:][np.where(dif_ip[ind_flap:] == dif_ip[ind_flap:].min())[0][0]]

    if arry[4] == -1:#安全炮
        #arry[4] = t1_1+0.1
        arry[6] = np.floor(min(t1_2,last_density)*1000)/1000              # 平顶端结束
        arry[7] = np.floor((min(t1_2,last_density)-0.1)*1000)/1000        # 实验结束时刻

    #接下来判断是否为密度极限破裂炮,对安全炮也判断
    #在t1_2的前1s内Ne是否超过Ngw

    t_end = arry[7]

    t_sta = arry[5]

    Ip = resample(t0,pcrl01,t_sta,t_end)
    Ne = resample(t2,dfsdev,t_sta,t_end)

    Ngw = Ip*(10**-6)*10/(np.pi*(0.45**2))
    R  = Ne/Ngw
    t_c = np.arange(t_sta,t_end+0.001,sample_rate)

    if len(np.where(R>=0.8)[0]) !=0:
        arry[8] = 1

        #找出密度达到Ngw时刻
        arry[9] = t_c[np.where(R>=0.8)[0][0]]


    ######################开始画图#########################
    fig, axs = plt.subplots(5,1,sharex=True,figsize=(6,8))#figsize=(6,8)
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

    axs[3].plot(t2,dfsdev)
    axs[3].axvline( x= t2[-1],ls='--',linewidth=1,color='g',)
    axs[3].set_ylabel("dfsdev"+r'($10^{19}/m^3$)',fontproperties = font1)

    axs[3].plot(t_c,Ngw,'r-',label='Green wald')
    axs[3].legend()

    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if arry[1] == 1:
        axs[4].plot(t3,aminor)
    axs[4].plot(t_c,np.array([0.45]*len(t_c)))
    axs[4].set_ylabel('aminor(m)',fontproperties = font1)
    axs[4].set_xlabel('time(s)',fontproperties = font1)
    ######################################################################
    if arry[4]==-1:
        plt.suptitle('safe pulse:%d'%shot,)
    plt.show()
    #print(arry)
    if arry[4] == -1 and arry[8]==1:
        safe_shot.append(arry)

    return


safe_shot = []#所有的安全炮



allsafe = np.array(pd.read_csv('/home/jack/tmp/allsafe_exceedGW1.csv'))


check(49743)
for i in allsafe[:,0]:
    check(i)




'''
A = np.array(safe_shot)
np.savez_compressed('/home/jack/tmp/safe3.npz',safe3 = A)

for i in range(len(A)):
    for j in [5,6,7,9]:
        A[i,j] = np.around(A[i,j]*1000)/1000


pd_data = pd.DataFrame(A, columns=['shot','W_aminor','W_flat','W_useless',\
                                    'W_disrupt','flat_sta','flat_end',\
                                    'disru_time','W_density','dens_time'])


pd_data.to_csv('/home/jack/tmp/safe.npz',safe=A)

'''
