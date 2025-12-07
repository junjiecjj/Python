#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:39:08 2019

@author: jack
"""
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from scipy.interpolate import interp1d
import math
import pandas as pd
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import minmax_scale

def smooth(aminor):
    if abs(aminor[1,0]-0.45)>0.1:
        aminor[1,0] = 0.45
    for i in range(aminor.shape[1]-1):
        if abs(aminor[1,i+1]-aminor[1,i]) > 0.1:
            aminor[1,i+1] = aminor[1,i]
    for i in range(aminor.shape[1]):        
        if abs(aminor[1,i]-0.45) > 0.05:
            aminor[1,i] = 0.45
    return aminor


def get_sta(i):
    #print(i)

    pcrl01 = np.load(datapath+'%d.npz'%(A[i,0]))['pcrl01']
    dfsdev = np.load(datapath+'%d.npz'%(A[i,0]))['dfsdev']
    aminor = np.load(datapath+'%d.npz'%(A[i,0]))['aminor']
    aminor1 = smooth(aminor)

    t = min(pcrl01[0,-1],dfsdev[0,-1],A[i,7],aminor[0,-1])#-0.013
    if abs(t-A[i,7]) >= 0.5:
        t = min(pcrl01[0,-1],dfsdev[0,-1],A[i,7])#-0.013
        a = 0.45
    else:
        a = aminor1[1][np.where(np.around(aminor1[0]/sample_rate)*sample_rate == int(t/sample_rate)*sample_rate)[0][0]]
    ne = dfsdev[1][np.where(np.around(dfsdev[0]/sample_rate)*sample_rate == int(t/sample_rate)*sample_rate)[0][0]]
    Ip = pcrl01[1][np.where(np.around(pcrl01[0]/sample_rate)*sample_rate == int(t/sample_rate)*sample_rate)[0][0]]
    #a = aminor[1][np.where(np.around(aminor[0]/sample_rate)*sample_rate == int(t/sample_rate)*sample_rate)[0][0]]
    ngw = Ip*10**-5/(np.pi*a**2)

    #pcrl01[1] = minmax_scale(pcrl01[1])
    diff_ip = np.diff(pcrl01[1])
    diff_t0 = np.diff(pcrl01[0])
    dif_ip = diff_ip/diff_t0
    
    fig, axs = plt.subplots(3,1,sharex=True,figsize=(8,6))#figsize=(6,8)
    axs[0].plot(pcrl01[0],pcrl01[1])
    ax = axs[0].twinx()
    ax.plot(pcrl01[0,:-1],dif_ip,'r')
    axs[0].axvline( x= A[i,7],ls='-',linewidth=3,color='g',)
    
    axs[1].plot(dfsdev[0],dfsdev[1])
    axs[1].axvline( x= A[i,7],ls='-',linewidth=3,color='g',)
    axs[1].axhline(y=ngw,ls='--',color='r',label='threshold')
    axs[1].annotate('ngw=%f'%ngw,xy=(A[i,7],ngw),\
     xytext=(A[i,7]+0.6,ngw+0.5),textcoords='data',\
     arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font) 
    axs[2].plot(aminor[0],aminor[1],'r',linewidth=1,)
    #axs[2].plot(aminor1[0],aminor1[1],'b')
    plt.suptitle('pulse:%d'%A[i,0])
    return

sample_rate = 0.001

font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 12)  
font1 = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 14)
datapath = "/home/jack/density/"
A = np.array(pd.read_csv("/home/jack/数据筛选/last5.csv"))
disru_num = len(np.where(A[:,4]==1)[0])
figpath = '/home/jack/result/'

for i in [843,914]:
    get_sta(i)