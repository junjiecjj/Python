#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:16:52 2018

@author: jack
"""
import numpy as np
import matplotlib.pyplot as plt
import xlrd
from scipy.interpolate import interp1d
from os import listdir
import math
from sklearn.preprocessing import minmax_scale
from sklearn.neural_network import MLPRegressor
import logging
import pp  #第三方多线程模块
import os, time
import multiprocessing
import pandas as pd
from sklearn.metrics import roc_curve, auc
from multiprocessing import Process,Pool                 #多进程模块
from multiprocessing import Lock
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import GRU
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.callbacks import TensorBoard
import time
import random
'''
In[1]： a=[1,2,3,4,5,6,7,8,9,10]
In[2]:  random.sample(a,3)  #这里的random不是numpy中的random，是单独的random库
Out[2]: [8,2,10]
In[3]:  random.sample(a,4)
Out[3]: [10, 2, 5, 9]
'''

"""
sheet=xlrd.open_workbook('/gpfs/home/junjie/公共的/excel_data/info.xlsx')
sheet1=sheet.sheet_by_name('Sheet1')
"""
#in:  sheet1.row_values(0)
#out: [65001.0, 3.321, 'DFSDEV']
#in:  sheet1.row_values(0)[0]
#out: 65001.0
# row_num,col_num = table.nrows,table.ncols
# table.row(10)[0]   读取第11行第一列的数据

class DIS_PRED(object):
    def __init__(self,Resample_rate,Exp_range):
        self.sample_rate = Resample_rate        # 0.001
        self.exp_range = Exp_range              # 1s
        self.disr_shut = 491              # 491
        self.safe_shut = 680            # 681
        self.batch_size = 32
        self.epochs = 1000
        self.a = self.data_info()               # self.a中的破裂时刻是人工从电子密度和电流信号上观察到的
        self.signal_kind = 12
        #self.a的第一列是炮号，第二列是人工找出的破裂时刻，第三列是标签，1代表破裂炮，-1代表安全炮。
        self.b = np.zeros((self.a.shape))
        #self.b是比较了self.a中的破裂时刻和各个诊断信号最大时间，取最小值以保证不会因为valueError出bug
        self.c = np.zeros((self.a.shape[0],6))
        #self.c是对比实际密度曲线和预测出来的密度曲线，找到其中密度值绝对值最小时的时刻
        #self.a是原始数据，self.b和self.c在实例函数中会被适应性的改变
        self.min_max = np.zeros((self.signal_kind,2))
        self.neural_struct = []
        for i in range(3,13):
            for j in range(3,13):
                self.neural_struct.append((i,j))

    @staticmethod
    def data_info():
        sheet = xlrd.open_workbook(r'/gpfs/home/junjie/公共的/excel_data/info.xlsx')
        sheet1 = sheet.sheet_by_name('Sheet1')
        row_num,col_num = sheet1.nrows,sheet1.ncols
        A=[]
        for i in range(row_num):
            A.append(sheet1.row_values(i))
        return np.array(A)

    """此函数是先读取某一炮的某一个诊断信号，然后插值采样"""
    def get_and_resample(self,A,i,Signal):
        signal = np.loadtxt('/gpfs/home/junjie/data/all/%d/%d_%s.txt' % (A[i,0],A[i,0],Signal))

        f = interp1d(signal[1],signal[0],kind='linear',fill_value="extrapolate")
        t2 = math.ceil(signal[1,len(signal[1])-1]/self.sample_rate)*self.sample_rate
        new_time = np.arange(0,t2,self.sample_rate)
        new_data = f(new_time)
        New_signal = np.zeros((2,len(new_time)))
        New_signal[0] = new_time
        New_signal[1] = new_data
        return New_signal

    """此函数通过比较第A[i,0]炮破裂炮的破裂时间和各信号的时间序列的最后一个值的最小值来确定破裂时刻"""
    def find_Dis_tm(self,A,pcr,dfsdev,lmsz,kmp,ic,sxr,pxu30,pxu18,vp,betap,li,q,i):
        b = []
        disr_time = A[i,1]; b.append(disr_time)
        pcr_D = pcr[0,len(pcr[0,:])-1]; b.append(pcr_D)
        dfsdev_D = dfsdev[0,len(dfsdev[0,:])-1]; b.append(dfsdev_D)
        lmsz_D = lmsz[0,len(lmsz[0,:])-1]; b.append(lmsz_D)
        kmp_D = kmp[0,len(kmp[0,:])-1]; b.append(kmp_D)
        ic_D = ic[0,len(ic[0,:])-1]; b.append(ic_D)
        sxr_D = sxr[0,len(sxr[0,:])-1]; b.append(sxr_D)
        pxu30_D = pxu30[0,len(pxu30[0,:])-1]; b.append(pxu30_D)
        pxu18_D = pxu18[0,len(pxu18[0,:])-1]; b.append(pxu18_D)
        vp_D = vp[0,len(vp[0,:])-1]; b.append(vp_D)
        betap_D = betap[0,len(betap[0,:])-1]; b.append(betap_D)
        li_D = li[0,len(li[0,:])-1]; b.append(li_D)
        q_D = q[0,len(q[0,:])-1]; b.append(q_D)
        disr_time = min(b)
        Index = b.index(min(b))
        Dict = {0:'DT',1:'pcr',2:'dfsdev',3:'lmsz',4:'kmp',5:'ic',6:'sxr',7:'pxu30',8:'pxu18',9:'vp',10:'betap',11:'li',12:'q'}
        print("the min disr_time of shut %d is :%s,and disr time is %f."% (i,Dict[Index],disr_time))
        #self.b[i,0] = A[i,0]
        #self.b[i,1] = min(b)
        #self.b[i,2] = A[i,2]
        return disr_time

    """此函数是对某一炮的某一个信号在指定的时间段上进行取样"""
    def signal_Range(self,signal,disr_time,i):
        #b = np.where(np.around(signal[0],decimals=3)==disr_time)[0][0]
        #b = np.where(abs(signal[0]-disr_time)==abs(signal[0]-disr_time).min())[0][0]
        #b = np.where(np.trunc(signal[0]*1000)==int(disr_time*1000))[0][0]
        num = int(self.exp_range/self.sample_rate)
        try:
            b = np.where(np.around(signal[0]*num)==int(disr_time*num))[0][0]
        except IndexError as e:
            print("在处理%d炮时函数signal_Range发生了错误，没有得到b:"%i,e)
        a = b - int(self.exp_range/self.sample_rate)
        new_Signal = signal[:,a+1:b+1]
        return new_Signal

    """此函数是并行读取某一炮信号的所有诊断信号，同时完成插值重采样,然后取破裂前的一段时间内的样本，但没有
    归一化"""
    def get_one_shut(self,i):
        A = self.a
        pcr = self.get_and_resample(A,i,'PCRL')
        dfsdev = self.get_and_resample(A,i,'DFSDEV')
        lmsz = self.get_and_resample(A,i,'LMSZ')
        kmp = self.get_and_resample(A,i,'KMP13T')
        ic = self.get_and_resample(A,i,'IC1')
        sxr = self.get_and_resample(A,i,'SXR23D')
        pxu30 = self.get_and_resample(A,i,'PXUV30')
        pxu18 = self.get_and_resample(A,i,'PXUV18')
        vp = self.get_and_resample(A,i,'VP1')
        betap = self.get_and_resample(A,i,'BETAP')
        li = self.get_and_resample(A,i,'LI')
        q = self.get_and_resample(A,i,'q')

        disr_time = self.find_Dis_tm(A,pcr,dfsdev,lmsz,kmp,ic,sxr,pxu30,pxu18,vp,betap,li,q,i)

        pcr = self.signal_Range(pcr,disr_time,i)
        dfsdev = self.signal_Range(dfsdev,disr_time,i)
        lmsz = self.signal_Range(lmsz,disr_time,i)
        kmp = self.signal_Range(kmp,disr_time,i)
        ic = self.signal_Range(ic,disr_time,i)
        sxr = self.signal_Range(sxr,disr_time,i)
        pxu30 = self.signal_Range(pxu30,disr_time,i)
        pxu18 = self.signal_Range(pxu18,disr_time,i)
        vp = self.signal_Range(vp,disr_time,i)
        betap = self.signal_Range(betap,disr_time,i)
        li = self.signal_Range(li,disr_time,i)
        q = self.signal_Range(q,disr_time,i)

        one_shut_train = np.zeros((14,int(self.exp_range/self.sample_rate)))
        #new_pcr, new_dfsdev, new_sxr, new_li = self.one_shut_autonormal(pcr2,dfsdev2,sxr2,li2)
        try:
            one_shut_train[0,:] = pcr[1,:]
        except ValueError as e:
            print("在处理%d炮时函数one_train_witho_nor出现错误，pcr出错:"%i,e)
            print("pcr is:")
            print(pcr)
            print("pcr的形状为: (%d,%d).\n"% pcr.shape)
            print("dfsdev的形状为: (%d,%d).\n"% dfsdev.shape)
            print("lmsz的形状为: (%d,%d).\n"% lmsz.shape)
            print("kmp的形状为: (%d,%d).\n"% kmp.shape)
            print("ic的形状为: (%d,%d).\n"% ic.shape)
            print("sxr的形状为: (%d,%d).\n"% sxr.shape)
            print("pxu30的形状为: (%d,%d).\n"% pxu30.shape)
            print("pxu18的形状为: (%d,%d).\n"% pxu18.shape)
            print("vp的形状为: (%d,%d).\n"% vp.shape)
            print("betap的形状为: (%d,%d).\n"% betap.shape)
            print("li的形状为: (%d,%d).\n"% li.shape)
            print("q的形状为: (%d,%d).\n"% q.shape)

        one_shut_train[1,:] = dfsdev[1,:]
        one_shut_train[2,:] = lmsz[1,:]
        one_shut_train[3,:] = kmp[1,:]
        one_shut_train[4,:] = ic[1,:]
        one_shut_train[5,:] = sxr[1,:]
        one_shut_train[6,:] = pxu30[1,:]
        one_shut_train[7,:] = pxu18[1,:]
        one_shut_train[8,:] = vp[1,:]
        one_shut_train[9,:] = betap[1,:]
        one_shut_train[10,:] = li[1,:]
        one_shut_train[11,:] = q[1,:]
        one_shut_train[13,:] = dfsdev[0,:]
        if self.a[i,2]==1:
            one_shut_train[12,:] = 1/(1+np.exp(-(one_shut_train[13,:]-(disr_time-0.3))*20))
        elif self.a[i,2]==-1:
            one_shut_train[12,:] = 0
        print("%d炮信号在差值取样但没归一化后的形状为:(%d,%d)"%(i,one_shut_train.shape[0],one_shut_train.shape[1]))
        return one_shut_train,disr_time

    def split_shut1(self):
        np.random.seed(1)
        all_disr_shut = range(0,self.disr_shut)
        all_safe_shut = range(self.disr_shut,self.disr_shut+self.safe_shut)

        train_Dindex = random.sample(all_disr_shut,200)
        self.train_D = len(train_Dindex)
        train_Dindex.sort()
        left_D_shut = list(sorted(set(all_disr_shut).difference(set(train_Dindex))))
        val_Dindex = random.sample(left_D_shut,100)
        self.val_D = len(val_Dindex)
        val_Dindex.sort()
        test_Dindex = list(sorted(set(left_D_shut).difference(set(val_Dindex))))
        self.test_D = len(test_Dindex)
        test_Dindex.sort()

        train_Sindex = random.sample(all_safe_shut,200)
        self.train_S = len(train_Sindex)
        train_Sindex.sort()
        left_S_shut = list(sorted(set(all_safe_shut).difference(set(train_Sindex))))
        val_Sindex = random.sample(left_S_shut,200)
        self.val_S = len(val_Sindex)
        val_Sindex.sort()
        test_Sindex = list(sorted(set(left_S_shut).difference(set(val_Sindex))))
        self.test_S = len(test_Sindex)
        test_Sindex.sort()

        all_train_shut = list(sorted(set(train_Dindex).union(set(train_Sindex))))
        all_val_shut = list(sorted(set(val_Dindex).union(set(val_Sindex))))
        all_test_shut = list(sorted(set(test_Dindex).union(set(test_Sindex))))
        np.savez_compressed(filepath+'shut_split20.npz',\
                            train_shut=all_train_shut,test_shut=all_test_shut,val_shut=all_val_shut)
        self.all_train_shut = all_train_shut
        self.all_val_shut = all_val_shut
        self.all_test_shut = all_test_shut
        self.all_shut = list(range(self.safe_shut+self.disr_shut))
        return all_train_shut,all_val_shut,all_test_shut

    def split_shut(self):
        all_disr_shut = range(0,self.disr_shut)
        all_safe_shut = range(self.disr_shut,self.disr_shut+self.safe_shut)
        train_D = [i for i in all_disr_shut if i%3==0]
        val_D = [i for i in all_disr_shut if i%3==1]
        test_D = [i for i in all_disr_shut if i%3==2]
        train_nD = [i for i in all_safe_shut if i%3==0]
        val_nD = [i for i in all_safe_shut if i%3==1]
        test_nD = [i for i in all_safe_shut if i%3==2]
        self.train_D = len(train_D)
        self.val_D = len(val_D)
        self.test_D = len(test_D)
        self.train_nD = len(train_nD)
        self.val_nD = len(val_nD)
        self.test_nD = len(test_nD)
        self.all_train_shut = list(sorted(set(train_D).union(set(train_nD))))
        self.all_val_shut = list(sorted(set(val_D).union(set(val_nD))))
        self.all_test_shut = list(sorted(set(test_D).union(set(test_nD))))
        self.all_shut = list(range(self.safe_shut+self.disr_shut))
        return self.all_train_shut,self.all_val_shut,self.all_test_shut
    '''
    此函数的功能是产生训练集，并且返回归一化后的训练集,此函数内部改变了类的属性self.min_max,当且仅
    当此函数被调用后self.min_max的值才改变
    '''
    def get_trainData(self):
        res_l = []
        arry_l =[]
        pool = Pool(multiprocessing.cpu_count())

        for i in self.all_train_shut:
            res = pool.apply_async(self.get_one_shut,(i,))
            res_l.append(res)
        pool.close()
        pool.join()

        for res in res_l:
            arry_l.append(res.get())

        for i,arry in zip(self.all_train_shut,arry_l):
            try:
                self.b[i,0] = self.a[i,0]
            except TypeError as e:
                print("在训练集中处理将self.a[i,0]赋值给self.b[i,0]时出错:",e)
                print("i 为 %d"%i)
                print("self.a[%d,0] 和 i 分别为:%d ,%d ."%(i,self.a[i,0],i))
                print("arry 为:",arry)
            self.b[i,1] = arry[1]
            self.b[i,2] = self.a[i,2]
            if i==self.all_train_shut[0]:
                train_mat = arry[0]
            else:
                train_mat = np.hstack([train_mat,arry[0]]) #np.hstack为水平拼接数组函数
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)
        #train_mat[-2,:] = (train_mat[-2,:]-self.min_max[1,0])/(self.min_max[1,1]-self.min_max[1,0])
        np.savez_compressed(filepath+'self_b_train20.npz',self_b_train=self.b)
        np.savez_compressed(filepath+'train20.npz',train_set=train_mat,train_shut=self.all_train_shut)
        return train_mat

    def get_testData(self,shut_list,name):
        #all_test_shut = range(0,(self.disr_shut+self.safe_shut))
        #all_test_shut = list(sorted(set(range(self.safe_shut+self.disr_shut)).difference(set(all_train_shut))))
        res_l = []
        arry_l =[]
        pool = Pool(multiprocessing.cpu_count())
        for i in shut_list:
            res = pool.apply_async(self.get_one_shut,(i,))
            res_l.append(res)
        pool.close()
        pool.join()

        for res in res_l:
            arry_l.append(res.get())

        for i,arry in zip(shut_list,arry_l):
            self.b[i,0] = self.a[i,0]
            self.b[i,1] = arry[1]
            self.b[i,2] = self.a[i,2]
            if i==shut_list[0]:
                test_mat = arry[0]
            else:
                test_mat = np.hstack([test_mat,arry[0]])
        test_mat[:-2,:] = test_mat[:-2,:]-np.tile(self.min_max[:,0].reshape(-1,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)
        test_mat[:-2,:] = test_mat[:-2,:]/np.tile(Range,(1,test_mat.shape[1]))
        #test_mat[-2,:] = (test_mat[-2,:]-self.min_max[1,0])/(self.min_max[1,1]-self.min_max[1,0])
        if name=='validation':
            np.savez_compressed(filepath+'self_b_val20.npz',self_b_val=self.b)
            np.savez_compressed(filepath+'val20.npz',val_set=test_mat,val_shut=shut_list)
        elif name=='testdata':
            np.savez_compressed(filepath+'self_b_test20.npz',self_b_test=self.b)
            np.savez_compressed(filepath+'test20.npz',test_set=test_mat,test_shut=shut_list)
        elif name=='alldata':
            np.savez_compressed(filepath+'self_b_all20.npz',self_b_all=self.b)
            np.savez_compressed(filepath+'all20.npz',all_set=test_mat,all_shut=shut_list)
        return test_mat
