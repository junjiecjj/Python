#!/usr/bin/env python3
#!-*-coding=utf-8-*-
###################################################################################
# File Name: getwholedata.py
# Author: 陈俊杰
# mail: 2716705056@qq.com
# Created Time: 2019.12.11
"""
此函数的功能是从本地取每炮的平顶端到破裂时刻的数据，查看每炮各个诊断在这个区间的长度是否一样,
然后找出ne/ngw=0.5,0.6,0.7等依次的时刻.
"""

#####################################################################################

import numpy as np
import matplotlib.pyplot as plt
import xlrd
from scipy.interpolate import interp1d
from os import listdir
import math
from sklearn.preprocessing import minmax_scale
from sklearn.neural_network import MLPRegressor
import logging
import os, time
import multiprocessing
import pandas as pd


class DIS_PRED(object):
    def __init__(self,Resample_rate,Exp_range):
        self.sample_rate = Resample_rate        # 采样率
        self.exp_range = Exp_range              # 训练区间
        self.num = int(self.exp_range/self.sample_rate)
        self.A = np.array(pd.read_csv(infopath+'last7.csv'))  # self.a中的破裂时刻是人工从电子密度和电流信号上观察到的
        self.disr_shot = len(np.where(self.A[:,4]==1)[0])     # 密度极限炮总数
        self.safe_shot = len(np.where(self.A[:,4]==-1)[0])    # 安全炮总数
        self.signal_kind = 13
        self.I = np.zeros((len(self.A),8))



    def split_shot(self):
        all_disr_shot = range(0,self.disr_shot)
        all_safe_shot = range(self.disr_shot, self.disr_shot + self.safe_shot)
        train_D = [i for i in all_disr_shot if i%3==0]
        val_D = [i for i in all_disr_shot if i%3==1]
        test_D = [i for i in all_disr_shot if i%3==2]
        train_nD = [i for i in all_safe_shot if i%3==0]
        val_nD = [i for i in all_safe_shot if i%3==1]
        test_nD = [i for i in all_safe_shot if i%3==2]
        self.train_D = len(train_D)
        self.val_D = len(val_D)
        self.test_D = len(test_D)
        self.train_nD = len(train_nD)
        self.val_nD = len(val_nD)
        self.test_nD = len(test_nD)
        self.all_train_shot = list(sorted(set(train_D).union(set(train_nD))))
        self.all_val_shot = list(sorted(set(val_D).union(set(val_nD))))
        self.all_test_shot = list(sorted(set(test_D).union(set(test_nD))))
        print("训练集炮数:%d"%len(self.all_train_shot))
        print("验证集炮数:%d"%len(self.all_val_shot))
        print("测试集炮数:%d"%len(self.all_test_shot))
        self.all_shot = list(range(self.safe_shot+self.disr_shot))
        return self.all_train_shot,self.all_val_shot,self.all_test_shot


    """
    此函数是对某一炮的某一个信号在指定的时间段[td-1 s, td]上进行取样
    """
    def signal_Range(self,signal,flat_top,disr_time,i):

        try:
            a = np.where(np.around(signal[0]*self.num)==np.around(flat_top*self.num))[0][0]
            b = np.where(np.around(signal[0]*self.num)==np.around(disr_time*self.num))[0][0]
            #print("a = %d, b = %d "%(a,b))
        except IndexError as e:
            print("在处理%d炮时函数signal_Range发生了错误，没有得到b:"%self.a[i,0],e)
        signal = signal[:,a:b+1]
        return signal

    """
    此函数是并行读取某一炮信号的所有诊断信号，同时完成插值重采样,然后取破裂前的一段时间内的样本，但没有归一化
    """
    def get_one_shot(self, i):

        pcrl01  = np.load(datapath+'%d.npz'%(self.A[i,0]))['pcrl01']
        dfsdev  = np.load(datapath+'%d.npz'%(self.A[i,0]))['dfsdev']
        vp1     = np.load(datapath+'%d.npz'%(self.A[i,0]))['vp1']
        sxr23d  = np.load(datapath+'%d.npz'%(self.A[i,0]))['sxr23d']
        pxuv30  = np.load(datapath+'%d.npz'%(self.A[i,0]))['pxuv30']
        pxuv18  = np.load(datapath+'%d.npz'%(self.A[i,0]))['pxuv18']
        kmp13t  = np.load(datapath+'%d.npz'%(self.A[i,0]))['kmp13t']
        pbrem10 = np.load(datapath+'%d.npz'%(self.A[i,0]))['pbrem10']
        lmsz    = np.load(datapath+'%d.npz'%(self.A[i,0]))['lmsz']
        betap   = np.load(datapath+'%d.npz'%(self.A[i,0]))['betap']
        li      = np.load(datapath+'%d.npz'%(self.A[i,0]))['li']
        q95     = np.load(datapath+'%d.npz'%(self.A[i,0]))['q95']
        ic      = np.load(datapath+'%d.npz'%(self.A[i,0]))['ic']

        disr_time = np.floor(min(self.A[i,7],pcrl01[0,-1],\
                                 dfsdev[0,-1],vp1[0,-1],sxr23d[0,-1],\
                                 pxuv30[0,-1],pxuv18[0,-1],kmp13t[0,-1],pbrem10[0,-1],\
                                lmsz[0,-1],betap[0,-1],li[0,-1],\
                                 q95[0,-1],ic[0,-1])/self.sample_rate)/self.num

        flat_top = np.ceil(max(self.A[i,5],pcrl01[0,0],dfsdev[0,0],vp1[0,0],\
                sxr23d[0,0],pxuv30[0,0],pxuv18[0,0],kmp13t[0,0],pbrem10[0,0],\
                                lmsz[0,0],betap[0,0],li[0,0],q95[0,0],ic[0,0])/self.sample_rate)/self.num

        if flat_top >= disr_time:
            print("%d 平顶端时刻大于破裂时刻，冲突\n"%self.a[i,0])

        pcrl01  = self.signal_Range(pcrl01,flat_top, disr_time, i)
        dfsdev  = self.signal_Range(dfsdev, flat_top, disr_time, i)
        vp1     = self.signal_Range(vp1, flat_top, disr_time, i)
        sxr23d  = self.signal_Range(sxr23d, flat_top, disr_time, i)
        pxuv30  = self.signal_Range(pxuv30, flat_top, disr_time, i)
        pxuv18  = self.signal_Range(pxuv18, flat_top, disr_time, i)
        kmp13t  = self.signal_Range(kmp13t, flat_top, disr_time, i)
        pbrem10 = self.signal_Range(pbrem10, flat_top, disr_time, i)
        lmsz    = self.signal_Range(lmsz, flat_top, disr_time, i)
        betap   = self.signal_Range(betap, flat_top, disr_time, i)
        li      = self.signal_Range(li, flat_top, disr_time, i)
        q95     = self.signal_Range(q95, flat_top, disr_time, i)
        ic      = self.signal_Range(ic, flat_top, disr_time, i)

        Ngw     = pcrl01[1]*(10**-5)/(np.pi*(0.45**2))
        R       = dfsdev[1]/Ngw

        a       = [i for i in range(len(Ngw)-1) if (R[i]<=0.3 and R[i+1]>=0.3)]
        b       = [i for i in range(len(Ngw)-1) if (R[i]<=0.4 and R[i+1]>=0.4)]
        c       = [i for i in range(len(Ngw)-1) if (R[i]<=0.5 and R[i+1]>=0.5)]
        d       = [i for i in range(len(Ngw)-1) if (R[i]<=0.6 and R[i+1]>=0.6)]
        e       = [i for i in range(len(Ngw)-1) if (R[i]<=0.7 and R[i+1]>=0.7)]
        f       = [i for i in range(len(Ngw)-1) if (R[i]<=0.8 and R[i+1]>=0.8)]
        g       = [i for i in range(len(Ngw)-1) if (R[i]<=0.9 and R[i+1]>=0.9)]
        h       = [i for i in range(len(Ngw)-1) if (R[i]<=1.0 and R[i+1]>=1.0)]

        if a != []:
            self.I[i,0] = pcrl01[0][a[0]]
        else:
            self.I[i,0] = -1

        if b != []:
            self.I[i,1] = pcrl01[0][b[0]]
        else:
            self.I[i,1] = -1

        if c != []:
            self.I[i,2] = pcrl01[0][c[0]]
        else:
            self.I[i,2] = -1

        if d != []:
            self.I[i,3] = pcrl01[0][d[0]]
        else:
            self.I[i,3] = -1

        if e != []:
            self.I[i,4] = pcrl01[0][e[0]]
        else:
            self.I[i,4] = -1

        if f != []:
            self.I[i,5] = pcrl01[0][f[0]]
        else:
            self.I[i,5] = -1

        if g != []:
            self.I[i,6] = pcrl01[0][g[0]]
        else:
            self.I[i,6] = -1

        if h != []:
            self.I[i,7] = pcrl01[0][h[0]]
        else:
            self.I[i,7] = -1
        print("%d炮完毕..."%self.A[i,0])

        return self.I



home = os.environ['HOME']
infopath = home + '/数据筛选/'
datapath = home + '/Density/'

Dist = DIS_PRED(0.001,1)


#Min, Max, flat_top, disr_time, pcrl01, pcrl011 = Dist.getone(660)


for i in range(Dist.disr_shot):
    Dist.get_one_shot(i)




#resultpath =home+'/Result/result_MLP/result_%d/'%1
#np.savez(resultpath +'DistI.npz',disti = Dist.I)

#DistI = np.load(resultpath +'DistI.npz')['disti']

A = Dist.A[:,:13]
B = np.hstack([A, Dist.I])

for i in range(len(B)):
    for j in [5,6,7,9,10,11,13,14,15,16,17,18,19,20]:
        B[i,j] = np.around(B[i,j]*1000)/1000


pd_data = pd.DataFrame(B, columns=['shot','W_aminor','W_flat','W_useless',\
                                    'W_disrupt','flat_sta','flat_end',\
                                    'disru_time','W_density','dens_time',\
                                    'R_flat_top','R_disrT','flat_len',\
                                   '0.3nGW','0.4nGW','0.5nGW','0.6nGW',\
                                   '0.7nGW','0.8nGW','0.9nGW','1.0nGW'])



pd_data.to_csv('/home/jack/数据筛选/last8.csv',index=False)
