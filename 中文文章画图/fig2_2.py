#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:46:06 2019
@author: jack
此代码功能是在线读取数据库画出某炮诊断信号的电流、密度、LMTIPREF、小半径，
并作出文章中的图2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MDSplus import *
from MDSplus import Connection
from scipy.interpolate import interp1d
from sklearn.preprocessing import minmax_scale
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
from pylab import tick_params

fontpath = "/usr/share/fonts/truetype/arphic/"
font = FontProperties(fname=fontpath+"SimSun.ttf", size = 24)#fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font1 = FontProperties(fname=fontpath+"SimSun.ttf", size = 18)
font2 = FontProperties(fname=fontpath+"SimSun.ttf", size = 16)
font3 = FontProperties(fname=fontpath+"SimSun.ttf", size = 30)

fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
font4 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 30)
font5 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 18)

sample_rate = 0.001
'''
class secletdata(object):
    def __init__(self):
'''


def resample(x,y,t_s,t_e):
    f = interp1d(x,y,kind='linear',fill_value="extrapolate")
    #t2 = math.ceil(signal[0,len(signal[0])-1]/sample_rate)*sample_rate
    X = np.arange(t_s,t_e,sample_rate)
    Y = f(X)
    return Y

def find_aminor(x,y,t_s,t_e):
    s = np.where(x >= t_s)[0][0]-2
    if s < 0:
        s = 0
    e = np.where(x >= t_e)[0][0]+1
    if e > len(x):
        e = len(x)-1
    x = x[s:e]
    y = y[s:e]
    #print("x.shape:%s, y.shape:%s"%(x.shape,y.shape))
    if abs(y[0]-0.45)>0.1:
        y[0] = 0.45
    for i in range(len(x)-1):
        if abs(y[i+1]-y[i]) >0.1:
            y[i+1] = y[i]
    return x,y

def check(shot):
    arry = [shot, 1, 1, 0, -1, -1, -1, -1, 0,-1]

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
            t0 = t0[a:l-1]
            pcrl01 = pcrl01[a:l-1]
        except IndexError:
            print("%d 的pcrl时间没有正值,终止"%shot)
            return
    #pcrl_minmax = minmax_scale(pcrl01)
    pcrl_minmax = pcrl01/1000000
    #print('PCRL01的单位：',coon.get('units(\\PCRL01)'))

    #####################读取LMTIPREF##################################################
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    try:
        coon.openTree('pcs_east', shot)   # open tree and shot
        lmtipref = np.array(coon.get(r'\LMTIPREF'))   # read data
        t1 = np.array(coon.get('dim_of(\LMTIPREF)')) # read time data
        coon.closeTree('pcs_east', shot)
    except (TreeFOPENR,TreeNODATA):
        print("%d 没有LMTIPREF,终止"%shot)
        return
    l = len(t1)
    if t1[0] >= 0:
        pass
    else:
        a = np.where(t1 >= 0)[0][0]
        t1 = t1[a:l-1]
        lmtipref = lmtipref[a:l-1]
        #print('\LMTIPREF的单位：',coon.get('units(\\LMTIPREF)'))

    diff_lm = np.diff(lmtipref)
    diff_t1 = np.diff(t1)
    dif_lmtipref = diff_lm/diff_t1
    try:
        t1_1 = t1[np.where(abs(dif_lmtipref) <= 0.002)[0][0]]

        id1 = np.where(abs(dif_lmtipref) <= 0.002)[0][-1]
        if id1 == len(dif_lmtipref)-1 or (id1 < len(dif_lmtipref)-1 and dif_lmtipref[-1]<0):
            t1_2 = t1[id1]
        elif id1 < len(dif_lmtipref)-1 and dif_lmtipref[-1]>0:
            t1_2 = t1[len(dif_lmtipref)]
    except (IndexError):
        arry[2] = 0
        without_flat.append(arry)
        print("%d没有平顶端,终止"%shot)
        return
    if (t1_2-t1_1) < 1:
        print("%d 平顶端太短,终止"%shot)
        arry[3] = 1
        flat_short.append(arry)
        return
    #####################读取DFSDEV####################################################
    #coon  = Connection('202.127.204.12')   # connect the MDSplus
    try:
        coon.openTree('pcs_east', shot)   # open tree and shot
        dfsdev = np.array(coon.get(r'\DFSDEV'))   # read data
        t2 = np.array(coon.get('dim_of(\DFSDEV)')) # read time data
        coon.closeTree('pcs_east', shot)
    except (TreeFOPENR,TreeNODATA):
        print("%d 没有DFSDEV,终止"%shot)
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
        coon.closeTree('efitrt_east', shot)
        #axs[4].plot(t3,aminor)
            #print('AMINOR的单位：',coon.get('units(\\AMINOR)'))
    except TreeFOPENR:
        print("%d 没有aminor"%shot)
        arry[1] = 0
    except TdiRECURSIVE:
        arry[1] = 0
        print("%d aminor timeout"%shot)
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    if t1[-2]-t1_2 < 0.3:#如果lmtipref导数的最后时刻-平顶端最后时刻<400ms，为破裂炮
        arry[4] = 1
    last_density = t2[-1]
    arry[5] = np.ceil((t1_1+0.2)/sample_rate)*sample_rate #平顶端开始时刻

    diff_ip = np.diff(pcrl_minmax)
    diff_t0 = np.diff(t0)
    dif_ip = diff_ip/diff_t0
    ind_flap = np.where(t0>=t1_1)[0][0]
    t0_max =t0[ind_flap:][np.where(dif_ip[ind_flap:] == dif_ip[ind_flap:].min())[0][0]]

    if arry[4] == -1:#安全炮
        #arry[4] = t1_1+0.1
        arry[6] = np.floor(min(t1_2,last_density)/sample_rate)*sample_rate #平顶端结束
        arry[7] = np.floor((min(t1_2,last_density)-0.1)/sample_rate)*sample_rate #实验结束时刻
    else:#破裂炮
        #arry[4] = t1_1+0.1
        arry[6] = np.floor(min(t1_2,last_density,t0_max)/sample_rate)*sample_rate #平顶端结束
        if t1_2 < t0_max:
            arry[7] = np.floor((min(t1_2,last_density)-0.01)/sample_rate)*sample_rate #破裂时刻
        else:
            arry[7] = np.floor((min(t0_max,last_density)-0.02)/sample_rate)*sample_rate
    #接下来判断是否为密度极限破裂炮,对安全炮也判断
    #在t1_2的前1s内Ne是否超过Ngw

    if arry[1] == 0:#如果没有aminor
        t_end = np.around((min(t0_max,t1_2,last_density)-0.02)/sample_rate)*sample_rate
    else:#有aminor
        t_end = np.around((min(t1_2,last_density,t0_max,t3[-1])-0.02)/sample_rate)*sample_rate
        #如果aminor前面有,后面没有,则当做没有
        if (t3[-1]-min(t0_max,t1_2,last_density))<-0.5 or t3[0]+0.5>min(t0_max,t1_2,last_density):
            arry[1] = 0
            t_end = np.around((min(t0_max,t1_2,last_density)-0.02)/sample_rate)*sample_rate
    t_sta = arry[5]
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
        if 0.3 <= t1[-2]-t1_2 < 0.4:
            arry[4] = 1
        #找出密度达到Ngw时刻
        t_denlimdis = t_c[np.where(R>=0.8)[0][0]]
        arry[9] = t_denlimdis
    ######################开始画图#########################
    fig, axs = plt.subplots(3,1,sharex=True,figsize=(5,6))#figsize=(6,8)
    fig.subplots_adjust(hspace=0.4)#调节两个子图间的距离


    #把x轴的刻度间隔设置为1，并存在变量里
    x_major_locator=MultipleLocator(2)
    #把y轴的刻度间隔设置为10，并存在变量里
    #y_major_locator=MultipleLocator(1)
    axs[0].tick_params(direction='in')
    p1, = axs[0].plot(t0,pcrl01/10**5,color='b',linewidth=2,)
    ax = axs[0].twinx()
    p2, = ax.plot(t0[:-1],dif_ip,'r--',linewidth=1,)
    #axs[0].axvline( x= t0_max,ls='-',linewidth=3,color='k',)
    ax.set_ylabel(r"电流导数",fontproperties=font1)
    axs[0].set_ylabel(r'电流/$10^{5}$A',fontproperties=font1)
    #axs[0].legend(loc='best',borderaxespad=0,edgecolor='black',prop=font1,shadow=False)

    if arry[4] ==1:
        axs[0].annotate(r'破裂',\
           xy=(t0_max,0.2),xytext=(t0_max-4,0.2),textcoords='data',\
       arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font1)
    else:pass


    axs[0].yaxis.label.set_color(p1.get_color())
    ax.yaxis.label.set_color(p2.get_color())

    #设置刻度的字号
    axs[0].tick_params(axis='y',colors=p1.get_color(),labelsize=16,width=3)

    axs[0].spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
    axs[0].spines['left'].set_linewidth(1.5)    ####设置左边坐标轴的粗细
    axs[0].spines['right'].set_linewidth(1.5)   ###设置右边坐标轴的粗细
    axs[0].spines['top'].set_linewidth(1.5)     ####设置上部坐标轴的粗细

    ax.spines['bottom'].set_linewidth(1.5)      ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1.5)        ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1.5)       ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1.5)         ####设置上部坐标轴的粗细

    ax.xaxis.set_major_locator(x_major_locator)
    #ax.yaxis.set_major_locator(y_major_locator)
    #ax.set_yticks([-1.5, 0, 1])

    ax.tick_params(axis='y', colors=p2.get_color(),labelsize=16,width=3 )
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(18) for label in labels]#刻度值字号

    axs[0].tick_params(labelsize=16,width=3)
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(18) for label in labels]#刻度值字号

    axs[0].set_title('(A)', loc = 'left',fontproperties=font5)
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    axs[1].tick_params(direction='in')
    axs[1].plot(t1,lmtipref,color='b',linewidth=2,label=r'预设电流')
    #axs[1].plot(t1[:-1],dif_lmtipref,linestyle='--',color='r',\
     #  label=r'$\frac{\mathit{d}\mathrm{I_{target}}}{\mathit{d}\mathrm{t}}$',linewidth=2)
    #axs[1].axvline( x= t1_1,ls='--',linewidth=2,color='g',)
    #axs[1].axvline( x= t1_2,ls='-',linewidth=2,color='c',)
    axs[1].axvline( x= t1[-1],ls='--',linewidth=2,color='k',)
    #axs[1].annotate(r'$t_{fs}$',xy=(t1_1,0),xytext=(0.45,0.55),textcoords='figure fraction',\
    #   arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font1)
    #axs[1].annotate(r'$t_{fe}$',xy=(t1_2,0),xytext=(0.4,0.62),textcoords='figure fraction',\
    #   arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font1)
    #axs[1].annotate(r'$t_{e}$',xy=(t1[-1],0.1),xytext=(t1[-1]+1,0.1),textcoords='data',\
    #   arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font1)
    axs[1].set_ylabel(r"$I_{t}$/A",fontproperties=font5)


    legend1 = axs[1].legend(loc='best',borderaxespad=0,edgecolor='black',prop=font1,shadow=False)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none') # 设置图例legend背景透明

    axs[1].set_title('(B)', loc = 'left',fontproperties=font5)

    axs[1].tick_params(labelsize=16,width=3)
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(18) for label in labels]#刻度值字号

    axs[1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
    axs[1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
    axs[1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
    axs[1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    axs[2].tick_params(direction='in')
    axs[2].plot(t2,dfsdev,linewidth=2,label = r'$n_e$',)
    axs[2].axvline( x= t_c[-1],ls='--',linewidth=2,color='k',)
    axs[2].set_ylabel(r'密度/$10^{19} m^{-3}$',fontproperties=font1)

    axs[2].plot(t_c,Ngw,'r-',label=r'$n_{GW}$')


    legend1 = axs[2].legend(loc='best',borderaxespad=0,edgecolor='black',prop=font2,shadow=False)
    frame1 = legend1.get_frame()
    frame1.set_alpha(1)
    frame1.set_facecolor('none') # 设置图例legend背景透明


    axs[2].set_title('(C)', loc = 'left',fontproperties=font5)
    axs[2].tick_params(labelsize=16,width=3)
    labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(22) for label in labels]#刻度值字号


    axs[2].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
    axs[2].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
    axs[2].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
    axs[2].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

    axs[2].set_xlabel('时间/s',fontproperties=font3)
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    ######################################################################
    #axs[0].set_title('方法二')
    if arry[4]==-1:
        plt.suptitle('非破裂炮#%d'%shot,fontproperties=font,fontweight='bold',y=0.98)
    elif arry[4] == 1 and arry[8] == 0:
        plt.suptitle('破裂炮#%d'%shot,fontproperties=font,fontweight='bold',y=0.98)
    elif arry[4] == 1 and arry[8] == 1:
        plt.suptitle('密度极限破裂炮#%d'%shot,fontproperties=font,fontweight='bold',y=0.98)
    
    out_fig = plt.gcf()
    out_fig.savefig(picfile+'%d_1.svg'%shot,format='svg',dpi=1000,bbox_inches = 'tight')
    #out_fig.savefig(picfile+'%d_1.eps'%shot,format='eps',dpi=1000,bbox_inches = 'tight')
    plt.show()
    #print(arry)
    if arry[4] == -1:
        safe_shot.append(arry)
    if arry[4] ==1:
        dis_shot.append(arry)
    if arry[8] == 1:
        green_shot.append(arry)
        if arry[4] ==1:
            density_shot.append(arry)
    return


#np.savez_compressed('/home/jack/snap/Bsafe.npz',Bsafe=B)
#B的第一列为炮号，第二列为记录的破裂时间，第三列为标签，第四列为d(pcrl01)/d(t)最大值时刻\
#第五列为lmtipref导数为0的最后一个点时刻,第六列为lmtipref最后时刻点
picfile = '/home/jack/tmp/'
A  = np.array(pd.read_csv('/home/jack/数据筛选/last7.csv'))
safe_shot = []#所有的安全炮
dis_shot = []#所有的破裂炮
green_shot = []#超过GW极限的,包括安全炮
density_shot = []#破裂炮中超过GW极限的
flat_short = []#平顶端短于1s的
without_flat = []#没有平顶端的


for i in [40144,80096,67039]:
    check(i)

safe = np.array(safe_shot)
disrupt = np.array(dis_shot)
pGreenW = np.array(green_shot)
density = np.array(density_shot)
short = np.array(flat_short)
without = np.array(without_flat)
