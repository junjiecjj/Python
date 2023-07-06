#!/usr/bin/env python
#-*-coding=utf-8-*-
'''
此版本用的是keras模块，用的是多对一且数据生成方式是0-10,1-11,2-12，这样可以产生很多数据，且标签的产生也做了改变
'''

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
from multiprocessing import Process,Pool #多进程模块
from multiprocessing import Lock
from keras.models import Sequential
from keras.layers import Dense
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
        self.num = int(self.exp_range/self.sample_rate)
        self.disr_shut = 491              # 491
        self.safe_shut = 680              # 681
        self.batch_size = 128
        self.epochs = 100
        self.time_step = 100
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
        a = b - int(self.exp_range/self.sample_rate + self.time_step)
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

        one_shut_train = np.zeros((14,int(self.exp_range/self.sample_rate+self.time_step)))
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
        np.savez_compressed(filepath+'shut_split21.npz',\
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
        np.savez_compressed(filepath+'self_b_train21.npz',self_b_train=self.b)
        np.savez_compressed(filepath+'Train21.npz',train_set=train_mat,train_shut=self.all_train_shut)
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
            np.savez_compressed(filepath+'self_b_val21.npz',self_b_val=self.b)
            np.savez_compressed(filepath+'val21.npz',val_set=test_mat,val_shut=shut_list)
        elif name=='testdata':
            np.savez_compressed(filepath+'self_b_test21.npz',self_b_test=self.b)
            np.savez_compressed(filepath+'test21.npz',test_set=test_mat,test_shut=shut_list)
        elif name=='alldata':
            np.savez_compressed(filepath+'self_b_all21.npz',self_b_all=self.b)
            np.savez_compressed(filepath+'all21.npz',all_set=test_mat,all_shut=shut_list)
        return test_mat

    '''此函数是找到预测曲线与阈值的差值的最小值的时间以及此最小值，且默认采样频率为１KHZ'''
    def find_pred_Dm(self,thresh,time,pred_signal):
        #f = interp1d(time,pred_signal,kind='linear',fill_value="extrapolate")
        #t2 = math.ceil(pred_signal[len(pred_signal[1])-1]/0.001)*0.001
        #new_time = np.arange(time[0],time[-1],0.001)
        #new_data = f(new_time)
        num = int(self.exp_range/self.sample_rate)
        Dist = pred_signal-thresh
        ind = [i for i in range(num-1) if ((Dist[i]<=0 and Dist[i+1]>=0))]
        if ind==[]:
            min_dis = abs(Dist).min()
            pred_Dt = time[np.where(abs(Dist)==min_dis)][0]
            return pred_Dt,min_dis+1e-16
        else:
            min_dis = 0
            pred_Dt = time[ind[0]]
            return pred_Dt,min_dis

    def build_model(self,optimizer='rmsprop',init='random_normal'):
        model = Sequential()
        model.add(LSTM(units=20,input_shape=(self.time_step,self.signal_kind),\
                       dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
        model.add(LSTM(units=20,dropout=0.2,recurrent_dropout=0.2))

        model.add(Dense(units=1,activation='sigmoid'))
        model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        print(model.summary())
        return model

    def change_selfc(self,name,thresh,time,pred_result):
        if name=='test':
            List = self.all_test_shut
        elif name=='val':
            List = self.all_val_shut
        elif name=='train':
            List = self.all_train_shut
        elif name=='all':
            List = self.all_shut
        else:pass
        for i,j in enumerate(List):
            pred_Dt,min_Dis = self.find_pred_Dm(thresh,time[i],pred_result[i])
            self.c[j,0] = self.b[j,0]
            self.c[j,1] = self.b[j,1]
            self.c[j,2] = self.b[j,2]
            self.c[j,3] = pred_Dt
            self.c[j,4] = min_Dis
        for i in List:
            if self.c[i,2]==-1:
                if self.c[i,4]!=0:
                    self.c[i,5]=-1
                else:
                    self.c[i,5]=0
            elif self.c[i,2]==1:
                if self.c[i,4]==0 and 0.01<=(self.c[i,1]-self.c[i,3])<=0.1:
                    self.c[i,5]=1
                elif self.c[i,4]==0 and (self.c[i,1]-self.c[i,3])<0.01:
                    self.c[i,5]=2
                elif self.c[i,4]==0 and 0.01<(self.c[i,1]-self.c[i,3])>0.1:
                    self.c[i,5]=3
                elif self.c[i,4]!=0:
                    self.c[i,5]=4
                else:pass
            else:pass
        return

    def cal_Num(self,name):
        Num = []
        Rate = []
        if name=='test':
            List = self.all_test_shut
        elif name=='val':
            List = self.all_val_shut
        elif name=='train':
            List = self.all_train_shut
        elif name=='all':
            List = self.all_shut
        else:pass
        succ_nD = len([x for x in List if self.c[x,5]==-1])
        fal_nD = len([x for x in List if self.c[x,5]==0])
        succ_D = len([x for x in List if self.c[x,5]==1])
        late_D = len([x for x in List if self.c[x,5]==2])
        pre_D = len([x for x in List if self.c[x,5]==3])
        fal_D = len([x for x in List if self.c[x,5]==4])
        Num.append(succ_nD)
        Num.append(fal_nD)
        Num.append(succ_D)
        Num.append(late_D)
        Num.append(pre_D)
        Num.append(fal_D)
        if name=='train':
            SnD = succ_nD/self.train_nD
            FnD = fal_nD/self.train_nD
            SD = succ_D/self.train_D
            LD = late_D/self.train_D
            PD = pre_D/self.train_D
            FD = fal_D/self.train_D
        elif name=='val':
            SnD = succ_nD/self.val_nD
            FnD = fal_nD/self.val_nD
            SD = succ_D/self.val_D
            LD = late_D/self.val_D
            PD = pre_D/self.val_D
            FD = fal_D/self.val_D
        elif name=='test':
            SnD = succ_nD/self.test_nD
            FnD = fal_nD/self.test_nD
            SD = succ_D/self.test_D
            LD = late_D/self.test_D
            PD = pre_D/self.test_D
            FD = fal_D/self.test_D
        elif name=='all':
            SnD = succ_nD/self.safe_shut
            FnD = fal_nD/self.safe_shut
            SD = succ_D/self.disr_shut
            LD = late_D/self.disr_shut
            PD = pre_D/self.disr_shut
            FD = fal_D/self.disr_shut
        else:pass
        Rate.append(SnD);Rate.append(FnD)
        Rate.append(SD);Rate.append(LD)
        Rate.append(PD);Rate.append(FD)
        return Num,Rate

    def find_best_thresh(self,all_test_shut,time,pred_result):
        num = int(self.exp_range/self.sample_rate)
        thresh_num = len(np.arange(0,1,0.01))
        pred_Rate = np.zeros((thresh_num,7))
        pred_Res = np.zeros((thresh_num,7))
        max_area = 0
        roc_AUC = []
        for k,thresh in enumerate(np.arange(0,1,0.01)):
            self.change_selfc('test',thresh,time,pred_result)
            ################################
            #统计在不同阈值下测试集预测结果
            pred_res,pred_rate = self.cal_Num('test')
            pred_res.append(thresh)
            pred_rate.append(thresh)
            pred_Rate[k,:] = pred_rate
            pred_Res[k,:] = pred_res
            arry = self.c[all_test_shut].copy() #注意，这里一定要用self.a.copy(),不然arry改变，self.c也变了
            for i in range(len(arry)):
                if arry[i,5]==0:
                    arry[i,5]=1
                elif arry[i,5]==2 or arry[i,5]==3:
                    arry[i,5]=1
                elif arry[i,5]==4:
                    arry[i,5]=-1
                else:pass
            fpr,tpr,threshold = roc_curve(arry[:,2],arry[:,5])
            roc_auc = auc(fpr,tpr)
            roc_AUC.append(roc_auc)
            if roc_auc > max_area:
                np.savez_compressed(filepath+'selfc21_test_findBthresh%d.npz'%k,selfc=self.c)
        np.savez_compressed(filepath+'pred_rate_res_fB.npz',\
                            pred_res=pred_Res,pred_rate=pred_Rate,AUC=roc_AUC)
        Index = roc_AUC.index(max(roc_AUC))
        print('AUC最大值为%f,在阈值为%f下取得.'%(max(roc_AUC),np.arange(0,1,0.01)[Index]))
        return np.arange(0,1,0.01)[Index]

    def generate_data(self,data,shut_list,name):
        data1 = np.empty((data.shape[0],data.shape[1],data.shape[2]-1))
        data1[:,:,:-1] = data[:,:,:-2]
        data1[:,:,-1] = data[:,:,-1]
        data_X = []
        data_y = []
        for i,j in enumerate(shut_list):
            if self.a[j,2]==-1:
                y = np.zeros(self.num)
            else:
                y = 1/(1+np.exp(-(data1[i][:,-1][-self.num:]-(self.b[j,1]-0.3))*20))
            data_y.append(y)
            for k in range(len(data1[i])-self.time_step):
                x = data1[i][k:k+self.time_step,:-1]
                data_X.append(x)
        data_X = np.array(data_X)
        data_y = np.array(data_y).reshape(-1,1)
        #print('%s data_X shape is %s.'%(name,str(data_X.shape)))
        #print('%s data_y shape is %s.'%(name,str(data_y.shape)))
        return data_X,data_y

    def fit_and_predict(self,all_data):
        start = time.ctime()
        print("训练和测试开始于%s"%start)
        Start = time.time()
        num = int(self.exp_range/self.sample_rate)  #采样点个数
        #将输入转换成[样本个数，时间步长，特征]
        all_data = all_data.T.reshape(self.safe_shut+self.disr_shut,num+self.time_step,self.signal_kind+2)
        X ,y = self.generate_data(all_data,self.all_shut,'all_data')
        print('X shape is %s,y shape is %s.'%(str(X.shape),str(y.shape)))
        np.savez_compressed(filepath+'all_xy21.npz',data=all_data)
        #训练集
        train_mat = all_data[self.all_train_shut]
        X_train ,y_train = self.generate_data(train_mat,self.all_train_shut,'train')
        print('X_train shape is %s,y_train shape is %s.'%(str(X_train.shape),str(y_train.shape)))
        np.savez_compressed(filepath+'train_xy21.npz',data=train_mat)
        #验证集
        val_mat = all_data[self.all_val_shut]
        X_val,y_val = self.generate_data(val_mat,self.all_val_shut,'validation')
        print('X_val shape is %s,y_val shape is %s.'%(str(X_val.shape),str(y_val.shape)))
        np.savez_compressed(filepath+'val_xy21.npz',data=val_mat)
        #测试集
        test_mat = all_data[self.all_test_shut]
        X_test,y_test = self.generate_data(test_mat,self.all_test_shut,'test')
        print('X_test shape is %s,y_test shape is %s.'%(str(X_test.shape),str(y_test.shape)))
        np.savez_compressed(filepath+'test_xy21.npz',data=test_mat)

        model = self.build_model()
        #保存模型结构
        model_json = model.to_json()
        with open(filepath+'my_model.json','w') as file:
            file.write(model_json)

        if os.path.exists(filepath2+'my_logs'):
            pass
        else:
            os.makedirs(filepath2+'my_logs')

        #设置回调函数分别用于降低学习率、中断训练、保存最优模型、可视化框架
        callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5),\
                         EarlyStopping(monitor='val_loss',patience=20),\
                         ModelCheckpoint(filepath=filepath+'best.weight.h5',monitor='val_loss',verbose=1,save_best_only=True),\
                         TensorBoard(log_dir=filepath2+'/my_logs',histogram_freq=1)]
        #在训练集上训练模型
        History = model.fit(x=X_train,y=y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,y_val),verbose=2,\
                            callbacks=callback_list)
        print(History.history.keys())
        history_dict = pd.DataFrame(History.history)
        history_dict.to_csv(filepath+'history_dict21.csv')
        #在测试集上进行预测
        print("+++++++++开始在测试集上测试+++++++++++")
        result0 = model.predict(X_test,batch_size=32,verbose=1)     #result0.shape=(all_test_shut*1000,1)
        result1 = result0.reshape(-1,num)                            #result1.shape=(all_test_shut,1000)
        np.savez_compressed(filepath+'pred_test_result21.npz',pred_result=result0,pred_result1=result1)
        test_time = test_mat[:,-self.num:,-1]
        best_thresh = self.find_best_thresh(self.all_test_shut,test_time,result1)
        #在所有数据上测试
        print("+++++++开始在所有数据集上测试+++++++++++")
        result2 = model.predict(X,batch_size=32,verbose=1)
        result3 = result2.reshape(-1,num)
        np.savez_compressed(filepath+'pred_all_result21.npz',pred_result2=result2,pred_result3=result3)
        all_time = all_data[:,-self.num:,-1]
        self.change_selfc('all',best_thresh,all_time,result3)
        np.savez(filepath+'selfc21_all.npz',selfc=self.c)
        train_Res,train_Rate = self.cal_Num('train')
        val_Res,val_Rate = self.cal_Num('val')
        test_Res,test_Rate = self.cal_Num('test')
        all_Res,all_Rate = self.cal_Num('all')
        print("result on train data:\n",'train_Res:',train_Res,'\ntrain_Rate:',train_Rate)
        print("result on val data:\n",'val_Res:',val_Res,'\nval_Rate:',val_Rate)
        print("result on test data:\n",'test_Res:',test_Res,'\ntest_Rate:',test_Rate)
        print("result on all data:\n",'all_Res:',all_Res,'\nall_Rate:',all_Rate)
        end = time.ctime()
        print("训练和测试结束于%s."%end)
        End = time.time()
        print('训练和测试花费%f小时。'%((End-Start)/3600))
        return

def main():
    if os.path.exists(filepath):
        pass
    else:
        os.makedirs(filepath)
    start = time.time()
    dis_pred = DIS_PRED(0.001,1)
    all_train_shut,all_val_shut,all_test_shut = dis_pred.split_shut()
    train_mat = dis_pred.get_trainData()
    end1 = time.time()
    all_data = dis_pred.get_testData(dis_pred.all_shut,'alldata')
    end2 = time.time()
    dis_pred.fit_and_predict(all_data)
    end = time.time()
    print("get and preprocess train data consum :%f hours.\n" % ((end1-start)/3600))
    print("get and preprocess all data consum :%f hours.\n" % ((end2-end1)/3600))
    print("data get and pre-process consum time: %f hours.\n" % ((end2-start)/3600))
    print("Total calcuate time is %f hours.\n"%((end-end2)/3600))
    print("Total run time is %f hours" % ((end-start)/3600))
    return
numb = 3
filepath = '/gpfs/home/junjie/音乐/data/data21/data21_%d/'%numb
filepath2 = '/gpfs/home/junjie/电影/data21_%d/'%numb
if __name__=="__main__":
    main()
