#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
此函数是找到所有炮的最大密度时刻和破裂时刻的差值，并作图
"""
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from scipy.interpolate import interp1d
import math
import pandas as pd
from matplotlib.font_manager import FontProperties

def smooth(aminor):
    if abs(aminor[1,0]-0.45)>0.1:
        aminor[1,0] = 0.45
    for i in range(aminor.shape[1]-1):
        if abs(aminor[1,i+1]-aminor[1,i]) > 0.02:
            aminor[1,i+1] = aminor[1,i]
    for i in range(aminor.shape[1]):        
        if abs(aminor[1,i]-0.45) > 0.05:
            aminor[1,i] = 0.45
    return aminor

def get_sta(i):
    #print(i)
    re = []
    pcrl01 = np.load(datapath+'%d.npz'%(A[i,0]))['pcrl01']
    dfsdev = np.load(datapath+'%d.npz'%(A[i,0]))['dfsdev']
    aminor = np.load(datapath+'%d.npz'%(A[i,0]))['aminor']
    try:
        flat_l = int(( A[i,7] - A[i,5])/0.001)
        a3 = np.where(np.around(aminor[0]*1000)==int(A[i,7]*1000))[0][0]
        aminor[:,a3-flat_l:a3+1] = smooth(aminor[:,a3-flat_l:a3+1])
    except IndexError:
        print("%d = shot:%d error ..."%(i,A[i,0]))
        aminor = smooth(aminor)

    t = min(pcrl01[0,-1],dfsdev[0,-1],A[i,7],aminor[0,-1])#-0.013
    if abs(t-A[i,7]) >= 0.5:
        t = min(pcrl01[0,-1],dfsdev[0,-1],A[i,7])#-0.013
        a = 0.45
    else:
        a = aminor[1][np.where(np.around(aminor[0]/sample_rate)*sample_rate == int(t/sample_rate)*sample_rate)[0][0]]
    ne = dfsdev[1][np.where(np.around(dfsdev[0]/sample_rate)*sample_rate == int(t/sample_rate)*sample_rate)[0][0]]
    Ip = pcrl01[1][np.where(np.around(pcrl01[0]/sample_rate)*sample_rate == int(t/sample_rate)*sample_rate)[0][0]]
    #a = aminor[1][np.where(np.around(aminor[0]/sample_rate)*sample_rate == int(t/sample_rate)*sample_rate)[0][0]]
    ngw = Ip*10**-5/(np.pi*a**2)
    R = ne/ngw
    #if 
    re.append(R)
    re.append(ne)
    re.append(ngw)
    re.append(A[i,0])
    re.append(t)
    re.append(Ip)
    re.append(a)
    RES.append(re)
    return

def draw_zhifang(arry):
    fig,axs = plt.subplots(1,1,figsize=(6,4),)
    num  = 50
    n1, bins1, patches1 = axs.hist(arry, bins=num, color='k',alpha=1, normed = 0)
    axs.set_xlabel(r'$n_e/n_{GW}$',fontproperties = font1)
    axs.set_ylabel('number',fontproperties = font)
    plt.savefig(figpath +'zhifang.eps', format='eps',dpi=1000,bbox_inches = 'tight')
    plt.show()
    return 

def draw_ngw(array):
    fig,axs = plt.subplots(1,1,figsize=(5,4),)
    axs.scatter(array[:disru_num,0],array[:disru_num,1],label = 'density limit disruption',c='r')
    axs.scatter(array[disru_num:,0],array[disru_num:,1],label = 'non-disruptive',c='green')
    y = np.arange(0,20,0.1)
    x = 0.8*y
    axs.plot(x,y,'k-',label = r'$\frac{n_e}{n_{GW}} =0.8$')
    axs.set_xlabel(r'$n_e(10^{19}m^{-3})$',fontproperties = font1)
    axs.set_ylabel(r'$n_{GW}(10^{19}m^{-3})$',fontproperties = font1)
    axs.legend(prop = font1)
    plt.savefig(figpath +'negw.eps', format='eps',dpi=1000,bbox_inches = 'tight')
    plt.show()
    axs.grid()
    
sample_rate = 0.001
fontpath = "/usr/share/fonts/truetype/msttcorefonts/"
font = FontProperties(fname=fontpath+"Times_New_Roman.ttf", size = 18)  
font1 = FontProperties(fname=fontpath+"Times_New_Roman.ttf",size = 12)
datapath = "/home/jack/density/"
A = np.array(pd.read_csv("/home/jack/数据筛选/last5.csv"))
disru_num = len(np.where(A[:,4]==1)[0])
figpath = '/home/jack/result/'
    
RES = []



for i in range(len(A)):
    get_sta(i)
Result = np.array(RES)

draw_zhifang(Result[:disru_num,0])
draw_ngw(Result[:,1:3])

weird = []
for i in range(disru_num):
    if Result[i,0]>=1.3:
        weird.append(A[i,0])
weird = np.array(weird)


