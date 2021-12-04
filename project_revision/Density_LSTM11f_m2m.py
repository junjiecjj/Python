#!/usr/bin/env python3
#!-*-coding=utf-8-*-

#########################################################################
# File Name: getwholedata.py
# Author: 陈俊杰
# mail: 2716705056@qq.com
# Created Time: 2019.12.11

"""
此函数的功能是从破裂前1s开始训练，同时也从破裂前1s开始验证，从平顶端测试；LSTM的many-to-many模型
用的是13种诊断
"""
#########################################################################

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
from sklearn.metrics import roc_curve, auc
from multiprocessing import Process,Pool                 #多进程模块
from multiprocessing import Lock
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers.noise import AlphaDropout
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import LeakyReLU
from keras.constraints import maxnorm
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.models import load_model
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

########################################################################


class DIS_PRED(object):
    def __init__(self,Resample_rate,Exp_range):
        self.sample_rate = Resample_rate        # 采样率 0.001
        self.exp_range = Exp_range               # 1s
        self.num = int(self.exp_range/self.sample_rate)    # = 1000
        self.A = np.array(pd.read_csv(infopath+'last7.csv'))  # self.A中的破裂时刻是人工从电子密度和电流信号上观察到的
        self.disr_shot = len(np.where(self.A[:,4]==1)[0])       # 密度极限炮总数
        self.safe_shot = len(np.where(self.A[:,4]==-1)[0])      # 安全炮总数

        self.all_test_data = {}  #以字典的形式存储所有的训练集{67039:array}
        self.all_data = {}       #以字典的形式存储所有的训练集{67039：array} 其中array.shape=15 x flao_len

        self.cpu_use = 8
        self.batch_size = 256
        self.signal_kind = 13
        self.epochs = 1000
        self.time_step = 100

        #self.b:第一列为炮号，第二列为是否为破裂炮，第三列为flat_top时刻，第四列为disr时刻
        #第五列为每炮flat_top到disr时间段样本数
        self.b = np.zeros(( self.A.shape[0], 5 ))
        #self.c:第一列为炮号,第二列为实际破裂时刻，第三列为实际破裂时刻，第四列为预测出来的破裂时刻
        #第五列为预测出来的破裂时刻时阈值和实际输出的距离，第六列为预测标签
        self.c = np.zeros(( self.A.shape[0], 6 ))
        #self.a是原始数据，self.b和self.c在实例函数中会被适应性的改变
        #Max_Min每行第一个是炮号，之后是每个诊断的最大最小值
        #self.Max_Min =np.zeros((len(self.a), self.signal_kind*2+1))
        self.min_max = np.zeros(( self.signal_kind, 2 ))


    def split_shot(self):
        all_disr_shot = range(0, self.disr_shot)
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
        return


    """此函数是对某一炮的某一个信号在指定的时间段[td-1 s, td]上进行取样"""
    def signal_Range(self,signal,flat_top,disr_time,i):

        #b = np.where(np.around(signal[0],decimals=3)==disr_time)[0][0]
        #b = np.where(abs(signal[0]-disr_time)==abs(signal[0]-disr_time).min())[0][0]
        #b = np.where(np.trunc(signal[0]*1000)==int(disr_time*1000))[0][0]
        try:
            a = np.where(np.around(signal[0]*self.num)==np.around(flat_top*self.num))[0][0]
            b = np.where(np.around(signal[0]*self.num)==np.around(disr_time*self.num))[0][0]
        except IndexError as e:
            print("在处理%d炮时函数signal_Range发生了错误，没有得到b:"%self.A[i,0],e)
        signal = signal[:,a:b+1]
        return signal

    """此函数是并行读取某一炮所有诊断信号从平顶端到破裂,但没有归一化"""
    def get_one_shot(self, i, name):

        pcrl01 = np.load(datapath+'%d.npz'%(self.A[i,0]))['pcrl01']
        dfsdev = np.load(datapath+'%d.npz'%(self.A[i,0]))['dfsdev']
        vp1 = np.load(datapath+'%d.npz'%(self.A[i,0]))['vp1']
        sxr23d = np.load(datapath+'%d.npz'%(self.A[i,0]))['sxr23d']
        pxuv30 = np.load(datapath+'%d.npz'%(self.A[i,0]))['pxuv30']
        pxuv18 = np.load(datapath+'%d.npz'%(self.A[i,0]))['pxuv18']
        kmp13t = np.load(datapath+'%d.npz'%(self.A[i,0]))['kmp13t']
        pbrem10 = np.load(datapath+'%d.npz'%(self.A[i,0]))['pbrem10']
        lmsz = np.load(datapath+'%d.npz'%(self.A[i,0]))['lmsz']
        betap = np.load(datapath+'%d.npz'%(self.A[i,0]))['betap']
        li = np.load(datapath+'%d.npz'%(self.A[i,0]))['li']
        q95 = np.load(datapath+'%d.npz'%(self.A[i,0]))['q95']
        ic = np.load(datapath+'%d.npz'%(self.A[i,0]))['ic']

        disr_time = np.floor(min(self.A[i,7],pcrl01[0,-1],dfsdev[0,-1],vp1[0,-1],\
                                 sxr23d[0,-1],pxuv30[0,-1],pxuv18[0,-1],\
                                 kmp13t[0,-1],pbrem10[0,-1],lmsz[0,-1],betap[0,-1],\
                                 li[0,-1],q95[0,-1],ic[0,-1])/self.sample_rate)/self.num

        if name == 'flat2disr':
            flat_top = np.ceil(max(self.A[i,5],pcrl01[0,0],dfsdev[0,0],vp1[0,0],\
                               sxr23d[0,0],pxuv30[0,0],pxuv18[0,0],kmp13t[0,0],pbrem10[0,0],\
                               lmsz[0,0],betap[0,0],li[0,0],\
                               q95[0,0],ic[0,0])/self.sample_rate)/self.num

            if (np.around(disr_time*self.num) - np.around(flat_top*self.num) < self.num):
                #flat_top = np.around(disr_time*self.num - self.num)/self.num
                flat_top = (np.around(disr_time*self.num) - (self.num - 1))/self.num
            else:
                Int = int(np.around((disr_time-flat_top)*self.num)//self.time_step)
                flat_top = (np.around(disr_time*self.num) - (Int*self.time_step - 1))/self.num
        elif name == '1s':
            flat_top = (np.around(disr_time*self.num) - (self.num - 1))/self.num
        else:
            pass
        if flat_top>=disr_time:
            print("%d 平顶端时刻大于破裂时刻，冲突\n"%self.A[i,0])


        pcrl01 = self.signal_Range(pcrl01,flat_top, disr_time, i)
        dfsdev = self.signal_Range(dfsdev, flat_top, disr_time, i)
        vp1 = self.signal_Range(vp1, flat_top, disr_time, i)
        sxr23d = self.signal_Range(sxr23d, flat_top, disr_time, i)
        pxuv30 = self.signal_Range(pxuv30, flat_top, disr_time, i)
        pxuv18 = self.signal_Range(pxuv18, flat_top, disr_time, i)
        kmp13t = self.signal_Range(kmp13t, flat_top, disr_time, i)
        pbrem10 = self.signal_Range(pbrem10, flat_top, disr_time, i)
        lmsz = self.signal_Range(lmsz, flat_top, disr_time, i)
        betap = self.signal_Range(betap, flat_top, disr_time, i)
        li = self.signal_Range(li, flat_top, disr_time, i)
        q95 = self.signal_Range(q95, flat_top, disr_time, i)
        ic = self.signal_Range(ic, flat_top, disr_time, i)


        #print("%d 处理结束..."%self.A[i,0])

        one_shot_train = np.zeros((self.signal_kind+2, len(pcrl01[0])))
        try:
            one_shot_train[0,:] = pcrl01[1,:]
        except ValueError as e:
            print("在处理%d炮时函数one_train_witho_nor出现错误，pcrl01出错:"%self.A[i,0],e)
            print("pcrl01 is:")
            print(pcrl01)
            print("pcr的形状为: (%d,%d).\n"% pcrl01.shape)
            print("dfsdev的形状为: (%d,%d).\n"% dfsdev.shape)
            print("vp的形状为: (%d,%d).\n"% vp1.shape)
            print("sxr的形状为: (%d,%d).\n"% sxr23d.shape)
            print("pxu30的形状为: (%d,%d).\n"% pxuv30.shape)
            print("pxu18的形状为: (%d,%d).\n"% pxuv18.shape)
            print("kmp的形状为: (%d,%d).\n"% kmp13t.shape)
        one_shot_train[1,:] = dfsdev[1,:]
        one_shot_train[2,:] = vp1[1,:]
        one_shot_train[3,:] = sxr23d[1,:]
        one_shot_train[4,:] =  pxuv30[1,:]
        one_shot_train[5,:] = pxuv18[1,:]
        one_shot_train[6,:] = kmp13t[1,:]
        one_shot_train[7,:] = pbrem10[1,:]
        one_shot_train[8,:] = lmsz[1,:]
        one_shot_train[9,:] = betap[1,:]
        one_shot_train[10,:] = li[1,:]
        one_shot_train[11,:] = q95[1,:]
        one_shot_train[12,:] = ic[1,:]

        one_shot_train[-1,:] = dfsdev[0,:]
        if self.A[i,4]==1:
            one_shot_train[-2,:] = 1/(1+np.exp(-(one_shot_train[-1,:]-(disr_time-0.5))*25))
        elif self.A[i,4]==-1:
            one_shot_train[-2,:] = 0

        #print("%d炮信号的形状为:%s"%(self.A[i,0],str(one_shot_train.shape)))
        return one_shot_train, flat_top, disr_time, len(pcrl01[0])


    '''
    此函数的功能是产生训练集，并且返回归一化后的训练集,此函数内部改变了类的属性self.min_max,当且仅
    当此函数被调用后self.min_max的值才改变
    '''
    def get_trainData(self,way):
        print("正在得到训练集.....")
        res_l = []
        arry_l =[]
        pool = Pool(self.cpu_use)

        for i in self.all_train_shot:
            try:
                res = pool.apply_async(self.get_one_shot,(i,way))
                res_l.append(res)
            except NameError:
                print("%d 炮出错"% self.A[i,0])
        pool.close()
        pool.join()

        try:
            for res in res_l:
                arry_l.append(res.get())
        except NameError:
            print("炮出错..." )

        for i,arry in zip(self.all_train_shot, arry_l):
            #  try:
            #      self.b[i,0] = self.A[i,0]  #炮号
            #  except TypeError as e:
            #      print("在训练集中处理将self.A[i,0]赋值给self.b[i,0]时出错:",e)
            #      print("炮号信息 self.A[%d,0] 为:%d ." % (i,self.A[i,0]))
            #      print("arry 为:",arry)
            #  self.b[i,1] = self.A[i,4]  #是破裂炮(+1)还是安全炮(-1)
            #  self.b[i,2] = arry[1]      #flat_top时刻
            #  self.b[i,3] = arry[2]      #disr_time时刻
            #  self.b[i,4] = arry[3]      #flat_top到disr_time长度

            if i==self.all_train_shot[0]:
                train_mat = arry[0]
            else:
                train_mat = np.hstack([train_mat,arry[0]]) #np.hstack为水平拼接数组函数

        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max[0,1] = 1
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)

        print("train_mat.shape = %s"%str(train_mat.shape))
        #np.savez_compressed(resultpath+'train_mat_minmax.npz',train_data = train_mat, minmax = self.min_max)
        np.savez_compressed(resultpath+'minmax.npz', minmax = self.min_max)
        print("得到训练集完毕...... \n")
        return train_mat

    #得到验证集数据
    def get_valData(self,way):

        print("正在得到验证集.....")
        res_l = []
        arry_l =[]
        pool = Pool(self.cpu_use)
        for i in self.all_val_shot:
            res = pool.apply_async(self.get_one_shot,(i,way,))
            res_l.append(res)
        pool.close()
        pool.join()

        for res in res_l:
            arry_l.append(res.get())

        for i, arry in zip(self.all_val_shot,arry_l):
            if i == self.all_val_shot[0]:
                val_mat = arry[0]
            else:
                val_mat = np.hstack([val_mat,arry[0]])

        val_mat[:-2,:] = (val_mat[:-2,:]-self.min_max[:,0].reshape(-1,1))/(self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)


        print("val_mat.shape = %s ." % str(val_mat.shape))
        #np.savez_compressed(resultpath+'val_data.npz',val_data=val_mat)
        print("所有验证集得到完毕.....\n")
        return val_mat


    #得到所有的数据
    def get_testData(self):
        print("正在得到所有数据集.....")
        res_l = []
        arry_l =[]
        pool = Pool(self.cpu_use)
        for i in self.all_shot:
            res = pool.apply_async(self.get_one_shot,(i,'flat2disr',))
            res_l.append(res)
        pool.close()
        pool.join()

        for res in res_l:
            arry_l.append(res.get())

        for i,arry in zip(self.all_shot,arry_l):

            self.b[i,0] = self.A[i,0]  #炮号
            self.b[i,1] = self.A[i,4]  #是破裂炮(+1)还是安全炮(-1)
            self.b[i,2] = arry[1]      #flat_top时刻
            self.b[i,3] = arry[2]      #disr_time时刻
            self.b[i,4] = arry[3]      #flat_top到disr_time长度

            self.c[i,0] = self.b[i,0]          # self.c第一列为炮号
            self.c[i,1] = self.b[i,1]          # self.c第二例为是否为DL破裂
            self.c[i,2] = self.b[i,3]          # self.c第三列为实际破裂时刻

            #self.all_data[self.A[i,0]] = arry[0] #以字典的形式存储所有的数据

            if i == self.all_shot[0]:
                test_mat = arry[0]
            else:
                test_mat = np.hstack([test_mat,arry[0]])

        test_mat[:-2,:] = (test_mat[:-2,:]-self.min_max[:,0].reshape(-1,1))/(self.min_max[:,1]-self.min_max[:,0]).reshape(-1,1)


        print("test_mat.shape = %s ." % str(test_mat.shape))
        #np.savez_compressed(resultpath+'allData_selfb.npz',all_data=test_mat, selfb=self.b)
        np.savez_compressed(resultpath+'selfb.npz', selfb=self.b)
        print("所有数据集得到完毕..... \n")
        return test_mat


    #此函数是找到预测曲线与阈值的差值的最小值的时间以及此最小值，且默认采样频率为１KHZ,loose
    def find_pred_Dm1(self, thresh, rtime, pred_signal):
        Dist = pred_signal - thresh
        ind = [i for i in range(len(rtime)) if ((Dist[i]<=0 and Dist[i+1]>=0))]
        if ind==[]:
            min_dis = abs(Dist).min()
            pred_Dt = rtime[np.where(abs(Dist)==min_dis)][0]
            return pred_Dt,min_dis+1e-16
        else:
            min_dis = 0
            pred_Dt = rtime[ind[0]]
            return pred_Dt,min_dis


    #此函数是严格的判断是否在td-1s时超过阈值的函数,tense
    def find_pred_Dm(self, way, thresh, rtime, pred_signal):
        Dist = pred_signal - thresh
        if way=='tense' and Dist[0] >= 0:
            min_dis = -1
            pred_Dt = -1
        else:
            ind = [i for i in range(len(rtime)-1) if (Dist[i]<=0 and Dist[i+1]>=0)]
            if ind == []:
                min_dis  = abs(Dist).min() + 1e-16
                pred_Dt = rtime[np.where(abs(Dist)==abs(Dist).min())][0]
            else:
                min_dis = 0
                pred_Dt = rtime[ind[0]]
        return pred_Dt, min_dis


    '''stateful=False的LSTM模型，many-to-many'''
    def build_model(self):
        model = Sequential()
        model.add(LSTM(units=100,input_shape=(self.time_step,self.signal_kind),\
                       dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
        model.add(LSTM(units=100,dropout=0.2,\
                       recurrent_dropout=0.2,return_sequences=True))
        model.add(TimeDistributed(Dense(units=1,activation='sigmoid')))

        model.compile(optimizer='nadam',loss='mse',metrics=['mae'])
        print(model.summary())
        return model


    def change_selfc(self, name, thresh, time_dict, pred_result_dict):
        if name=='train':
            List = self.all_train_shot
        elif name=='val':
            List = self.all_val_shot
        elif name=='test':
            List = self.all_test_shot
        elif name=='all':
            List = self.all_shot
        else:
            pass

        for j in List:
            pred_Dt,min_Dis = self.find_pred_Dm('tense',thresh, time_dict[j], pred_result_dict[j])
            #self.c[j,0] = self.b[j,0]  #self.c第一列为炮号
            #self.c[j,1] = self.b[j,1]   #self.c第二例为是否为DL破裂
            #self.c[j,2] = self.b[j,3]    #self.c第三列为实际破裂时刻
            self.c[j,3] = pred_Dt         #预测出来的破裂时刻
            self.c[j,4] = min_Dis        #预测出来的破裂时刻的最小距离

        for i in List:
            if self.c[i,1]==-1:
                if self.c[i,4] > 0:
                    self.c[i,5]=-1  #安全炮,预测正确
                else:
                    self.c[i,5]=0   #安全炮,预测错误
            elif self.c[i,1]==1:
                if self.c[i,4]==0 and 0.02<=(self.c[i,2]-self.c[i,3])<=0.5: #破裂炮,预测正确
                    self.c[i,5]=1
                elif self.c[i,4]==0 and (self.c[i,2]-self.c[i,3])>0.5: #破裂炮,预测提前
                    self.c[i,5]=2
                elif self.c[i,4]==0 and (self.c[i,2]-self.c[i,3])<0.02: #破裂炮,预测滞后
                    self.c[i,5]=3
                elif self.c[i,4]!=0: #破裂炮,预测错误
                    self.c[i,5]=4
                else:pass
            else:
                pass
        return

    def cal_Num(self,name):
        Num = []
        Rate = []
        if name == 'train':
            List = self.all_train_shot
        elif name == 'val':
            List = self.all_val_shot
        elif name == 'test':
            List = self.all_test_shot
        elif name == 'all':
            List = self.all_shot
        else:
            pass

        succ_nD = len([x for x in List if self.c[x,5]==-1]); Num.append(succ_nD)
        fal_nD = len([x for x in List if self.c[x,5]==0])  ; Num.append(fal_nD)
        succ_D = len([x for x in List if self.c[x,5]==1])  ; Num.append(succ_D)
        pre_D = len([x for x in List if self.c[x,5]==2])  ; Num.append(pre_D)
        late_D = len([x for x in List if self.c[x,5]==3])   ; Num.append(late_D)
        fal_D = len([x for x in List if self.c[x,5]==4])   ; Num.append(fal_D)

        if name=='train':
            SnD = succ_nD/self.train_nD
            FnD = fal_nD/self.train_nD
            SD = succ_D/self.train_D
            PD = pre_D/self.train_D
            LD = late_D/self.train_D
            FD = fal_D/self.train_D
        elif name=='val':
            SnD = succ_nD/self.val_nD
            FnD = fal_nD/self.val_nD
            SD = succ_D/self.val_D
            PD = pre_D/self.val_D
            LD = late_D/self.val_D
            FD = fal_D/self.val_D
        elif name=='test':
            SnD = succ_nD/self.test_nD
            FnD = fal_nD/self.test_nD
            SD = succ_D/self.test_D
            PD = pre_D/self.test_D
            LD = late_D/self.test_D
            FD = fal_D/self.test_D
        elif name=='all':
            SnD = succ_nD/self.safe_shot
            FnD = fal_nD/self.safe_shot
            SD = succ_D/self.disr_shot
            PD = pre_D/self.disr_shot
            LD = late_D/self.disr_shot
            FD = fal_D/self.disr_shot
        else:
            pass

        Rate.append(SnD);Rate.append(FnD)
        Rate.append(SD);Rate.append(PD)
        Rate.append(LD);Rate.append(FD)

        Rate = list(map(lambda x: float('%.4f'%x), Rate))

        return Num,Rate


    def change_selfc_cal(self, name, thresh, time_all, pred_result):
        if name=='train':
            List = self.all_train_shot
        elif name=='val':
            List = self.all_val_shot
        elif name=='test':
            List = self.all_test_shot
        elif name=='all':
            List = self.all_shot
        else:
            pass

        for i, j in enumerate(List):
            pred_Dt,min_Dis = self.find_pred_Dm('tense',thresh, time_all[i], pred_result[i])
            #self.c[j,0] = self.b[j,0]     #self.c第一列为炮号
            #self.c[j,1] = self.b[j,1]     #self.c第二例为是否为DL破裂
            #self.c[j,2] = self.b[j,3]     #self.c第三列为实际破裂时刻
            self.c[j,3] = pred_Dt          #预测出来的破裂时刻
            self.c[j,4] = min_Dis          #预测出来的破裂时刻的最小距离

        for i in List:
            if self.c[i,1]==-1:
                if self.c[i,4] > 0:
                    self.c[i,5]=-1  #安全炮,预测正确
                else:
                    self.c[i,5]=0   #安全炮,预测错误
            elif self.c[i,1]==1:
                if self.c[i,4]==0 and 0.02<=(self.c[i,2]-self.c[i,3])<=0.5: #破裂炮,预测正确
                    self.c[i,5]=1
                elif self.c[i,4]==0 and (self.c[i,2]-self.c[i,3])>0.5: #破裂炮,预测提前
                    self.c[i,5]=2
                elif self.c[i,4]==0 and (self.c[i,2]-self.c[i,3])<0.02: #破裂炮,预测滞后
                    self.c[i,5]=3
                elif self.c[i,4]!=0: #破裂炮,预测错误
                    self.c[i,5]=4
                else:pass
            else:
                pass

        Num = []
        Rate = []
        succ_nD = len([x for x in List if self.c[x,5] == -1]); Num.append(succ_nD)
        fal_nD = len([x for x in List if self.c[x,5]==0])    ; Num.append(fal_nD)
        succ_D = len([x for x in List if self.c[x,5]==1])    ; Num.append(succ_D)
        pre_D = len([x for x in List if self.c[x,5]==2])     ; Num.append(pre_D)
        late_D = len([x for x in List if self.c[x,5]==3])    ; Num.append(late_D)
        fal_D = len([x for x in List if self.c[x,5]==4])     ; Num.append(fal_D)

        if name=='train':
            SnD = succ_nD/self.train_nD
            FnD = fal_nD/self.train_nD
            SD = succ_D/self.train_D
            PD = pre_D/self.train_D
            LD = late_D/self.train_D
            FD = fal_D/self.train_D
        elif name=='val':
            SnD = succ_nD/self.val_nD
            FnD = fal_nD/self.val_nD
            SD = succ_D/self.val_D
            PD = pre_D/self.val_D
            LD = late_D/self.val_D
            FD = fal_D/self.val_D
        elif name=='test':
            SnD = succ_nD/self.test_nD
            FnD = fal_nD/self.test_nD
            SD = succ_D/self.test_D
            PD = pre_D/self.test_D
            LD = late_D/self.test_D
            FD = fal_D/self.test_D
        elif name=='all':
            SnD = succ_nD/self.safe_shot
            FnD = fal_nD/self.safe_shot
            SD = succ_D/self.disr_shot
            PD = pre_D/self.disr_shot
            LD = late_D/self.disr_shot
            FD = fal_D/self.disr_shot
        else:
            pass

        Rate.append(SnD);Rate.append(FnD)
        Rate.append(SD);Rate.append(PD)
        Rate.append(LD);Rate.append(FD)


        arry = self.c[self.all_val_shot]
        for i in range(len(arry)):
            if arry[i,5]==0:
                arry[i,5]=1
            elif arry[i,5] == 2:
                arry[i,5] = 1
            elif arry[i,5] == 3 or arry[i,5] == 4:
                arry[i,5] = -1
            else:
                pass
        fpr,tpr,threshold = roc_curve(arry[:,1],arry[:,5])
        roc_auc = auc(fpr,tpr)

        Num.append(thresh)
        Num.append(roc_auc)

        Rate.append(thresh)
        Rate.append(roc_auc)

        Rate = list(map(lambda x: float('%.4f'%x), Rate))

        return Num,Rate


    def find_best_thresh(self,rtime,pred_result):
        t1 = time.time()
        print("寻找最佳阈值.....")
        thresh_num = len(np.arange(0, 1, 0.01))
        pred_Rate = np.zeros((thresh_num,8))
        pred_Res = np.zeros((thresh_num,8))

        roc_AUC = []

        res_l = []
        arry_l =[]

        pool = Pool(self.cpu_use)
        for k,thresh in enumerate(np.arange(0, 1, 0.01)):
            res = pool.apply_async(self.change_selfc_cal,('val',thresh,rtime,pred_result,))
            res_l.append(res)
        pool.close()
        pool.join()

        for res in res_l:
            arry_l.append(res.get())

        for i,arry in enumerate(arry_l):
            pred_Res[i] = arry[0]
            pred_Rate[i] = arry[1]

        print('pred_Res = ',pred_Res,'\n')

        np.savez_compressed(resultpath+'pred_rate_res_fB.npz',\
                            pred_res=pred_Res, pred_rate=pred_Rate)
        Index = np.where(pred_Res[:,-1] == pred_Res[:,-1].max())

        max_thresh = np.arange(0,1,0.01)[Index]
        t2 = time.time()

        print('AUC最大值为%f,'%(max(pred_Res[:,-1])),'最佳阈值为',max_thresh)
        print("寻找最佳阈值耗时 %f 小时" % ((t2-t1)/3600))
        return np.arange(0,1,0.01)[Index[0][0]]



    def find_best_thresh1(self,rtime,pred_result):
        print("寻找最佳阈值.....")
        thresh_num = len(np.arange(0,1,0.01))
        pred_Rate = np.zeros((thresh_num,8))
        pred_Res = np.zeros((thresh_num,8))
        max_area = 0
        roc_AUC = []
        for k,thresh in enumerate(np.arange(0,1,0.01)):
            self.change_selfc('val',thresh,rtime,pred_result)
            #统计在不同阈值下验证集预测结果
            pred_res,pred_rate = self.cal_Num('val')
            arry = self.c[self.all_val_shot]
            for i in range(len(arry)):
                if arry[i,5]==0:
                    arry[i,5]=1
                elif arry[i,5] == 2:
                    arry[i,5] = 1
                elif arry[i,5] == 3 or arry[i,5] == 4:
                    arry[i,5] = -1
                else:
                    pass
            fpr,tpr,threshold = roc_curve(arry[:,1],arry[:,5])
            roc_auc = auc(fpr,tpr)

            roc_AUC.append(roc_auc)

            pred_res.append(thresh)
            pred_res.append(roc_auc)

            pred_rate.append(thresh)
            pred_rate.append(roc_auc)

            pred_Rate[k,:] = pred_rate
            pred_Res[k,:] = pred_res

        np.savez_compressed(resultpath+'pred_rate_res_fB.npz',\
                            pred_res=pred_Res, pred_rate=pred_Rate)
        Index = roc_AUC.index(max(roc_AUC))
        print('AUC最大值为%f,在阈值为%f下取得.'%(max(roc_AUC),np.arange(0,1,0.01)[Index]))
        return np.arange(0,1,0.01)[Index]


    #统计出在所有炮上被正确预测的平均提前时间
    def cal_aver(self):
        s_nd = []
        f_nd = []
        s_d = []
        l_d = []
        p_d = []
        f_d = []
        for i in range(len(self.A)):
            if self.c[i,5]==-1:
                s_nd.append(i)
            elif self.c[i,5]==0:
                f_nd.append(i)
            elif self.c[i,5]==1:
                s_d.append(i)
            elif self.c[i,5]==2:
                p_d.append(i)
            elif self.c[i,5]==3:
                l_d.append(i)
            elif self.c[i,5]==4:
                f_d.append(i)
            else:
                pass
        a = s_d.copy()
        a.extend(p_d)
        a.sort()
        chzhi = self.c[a,2]-self.c[a,3]
        print("在所有被正确预测的破裂炮上的平均提前 = %fs"%np.average(chzhi))
        return


    def Transform(self, arr, name):
        if name == 'val':
            lisT = self.all_val_shot
            S = self.b[self.all_val_shot, -1]  #取出验证集中每炮的平顶端长度
        elif name == 'all':
            lisT = self.all_shot
            S = self.b[:,-1]                   #取出所有炮的平顶端长度

        Dict = {}
        #A = np.zeros((len(S), 2),dtype = np.int32)
        #lisT[i] = j
        for i,j in enumerate(lisT):
            a = int(sum(S[:i]))
            b = int(sum(S[:i+1]))
            Dict[j] = arr[a:b]
        print("转换完成....")
        return Dict


    def fit_and_predict(self, train_data, all_data):
        start = time.ctime()
        print("训练和测试开始于%s"%start)
        #Start = time.time()

        #训练集
        X_train = train_data.T[:,:-2]    # shape = (训练集炮数x1000，13)
        Y_train = train_data.T[:,-2]     # shape  = (训练集炮数x1000，)
        print('1: X_train.shape = %s, Y_train.shape = %s.'%(str(X_train.shape),str(Y_train.shape)))

        X_train = X_train.reshape(-1, self.time_step, self.signal_kind)   #shape = (训练炮中所有炮长度的和/time_step，time_step，13)
        Y_train = Y_train.reshape(-1, self.time_step, 1)                  #shape = (训练炮中所有炮长度的和/time_step，time_step，1)
        print('2: X_train.shape = %s, Y_train.shape = %s.'%(str(X_train.shape),str(Y_train.shape)))
        #np.savez_compressed(resultpath+'XY_train.npz',x_train=X_train,y_train=Y_train)
        del train_data  #释放内存

        #验证集
        val_data = self.get_valData('1s')
        print("val_mat.shape = %s."%str(val_data.shape))

        X_val = val_data.T[:,:-2]      # shape = (验证集炮数x1000，13)
        Y_val = val_data.T[:,-2]       # shape  = (验证集炮数x1000，)
        val_time = val_data.T[:,-1].reshape(-1,self.num)
        print('1:  X_val.shape = %s, Y_val.shape = %s, val_time.shape = %s.'\
              %(str(X_val.shape),str(Y_val.shape), str(val_time.shape) ))

        X_val = X_val.reshape(-1, self.time_step, self.signal_kind)
        Y_val = Y_val.reshape(-1, self.time_step, 1)
        print('2:  X_val.shape = %s, Y_val.shape = %s.'%(str(X_val.shape),str(Y_val.shape)))
        #np.savez_compressed(resultpath+'XY_val.npz',x_val=X_val,y_val=Y_val)
        del val_data  #释放内存

        #创建模型
        model = self.build_model()

        #保存模型结构,但是没有权值
        model_json = model.to_json()
        with open(resultpath+'my_model.json','w') as File:
            File.write(model_json)

        #if os.path.exists(resultpath+'my_logs'):
         #   pass
        #else:
         #   os.makedirs(resultpath+'my_logs')

        #设置回调函数分别用于降低学习率、中断训练、保存最优模型(不仅是结构，而且保存权值)、可视化框架
        callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.6,patience=5),\
                         EarlyStopping(monitor='val_loss',patience=2),\
                         ModelCheckpoint(filepath=resultpath+'mymodel.h5', \
                                         monitor='val_loss',verbose=1,save_best_only=True),]
        #TensorBoard(log_dir=resultpath+'/my_logs/',histogram_freq=1)
        #在训练集上训练模型
        History = model.fit(x=X_train,y=Y_train,epochs=self.epochs,\
                            batch_size=self.batch_size,validation_data=(X_val,Y_val),verbose=1,\
                            callbacks=callback_list)

        print(History.history.keys())
        history_dict = pd.DataFrame(History.history)
        history_dict.to_csv(resultpath+'history_dict.csv')

        #删除模型,释放内存
        del model

        del X_train, Y_train
        #加载最优模型
        '''
        #方法一：
        with open(resultpath+'my_model.json','r') as file:
            model_json1 = file.read()
        new_model = model_from_json(model_json1)
        new_model.load_weights(resultpath+'best.weight.h5')
        new_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['mse'])
        '''

        #方法二
        best_model = load_model(resultpath+'mymodel.h5')

        #在验证集上进行预测
        print("+++++++++开始在验证集上测试+++++++++++")
        result0 = best_model.predict(X_val,batch_size=self.batch_size,verbose=1) #result0.shape=(验证集炮数x1000,1)
        result1 = result0.reshape(-1, self.num)      #result1.shape=(验证集炮数,1000)
        print("result0.shape = %s \nresult1.shape = %s "%(str(result0.shape), str(result1.shape)) )
        #np.savez_compressed(resultpath+'val_result_time.npz', pred_result1=result1,t_val=val_time)
        best_thresh = self.find_best_thresh(val_time,result1)

        #在所有数据上测试

        X = all_data.T[:,:-2]
        Y = all_data.T[:,-2]
        all_time = all_data.T[:,-1]
        print('1: X.shape = %s, Y.shape =  %s, all_time.shape = %s.'\
              %(str(X.shape),str(Y.shape), str(all_time.shape) ))

        X = X.reshape(-1, self.time_step, self.signal_kind)
        Y = Y.reshape(-1, self.time_step, 1)
        print('2: X.shape = %s, Y.shape = %s.'%(str(X.shape),str(Y.shape)))
        #np.savez_compressed(resultpath+'XY_all.npz',x_all=X,y_all=Y)

        all_time = self.Transform(all_data.T[:,-1],'all')
        all_Y = self.Transform(all_data.T[:,-2],'all')
        del all_data  #释放内存

        print("+++++++开始在所有数据集上测试+++++++++++")
        result2 = best_model.predict(X,batch_size=self.batch_size,verbose=1)
        result3 = result2.reshape(-1,)
        print("result2.shape = %s \n result3.shape = %s "%(str(result2.shape), str(result3.shape)) )
        result3 = self.Transform(result3,'all')
        self.change_selfc('all', best_thresh, all_time, result3)
        np.savez_compressed(resultpath+'pred_allY_time_selfc.npz',pred_result3=result3,\
                            allY=all_Y,alltime=all_time,selfc=self.c)
        '''
        selfc_all.npz中selfc第一列为炮号，第二列为实际破裂时刻(对于安全炮是采样最后时刻)，
        第三列是实际标签，第四列和第五列分别是模型预测值和阈值差值的最小距离以及最小距离出现的时刻，
        第六列是模型预测的标签
        '''

        train_Res,train_Rate = self.cal_Num('train')
        val_Res,val_Rate = self.cal_Num('val')
        test_Res,test_Rate = self.cal_Num('test')
        all_Res,all_Rate = self.cal_Num('all')

        self.cal_aver()
        print("在训练集上的结果:\n",'train_Res:',train_Res,'\n','train_Rate:\n',train_Rate,'\n')
        print("在验证集上的结果:\n",'val_Res:',val_Res,'\n','val_Rate:\n',val_Rate,'\n')
        print("在测试集上的结果:\n",'test_Res:',test_Res,'\n','test_Rate:\n',test_Rate,'\n')
        print("在所有数据上的结果:\n",'all_Res:',all_Res,'\n','all_Rate:\n',all_Rate,'\n')

        end = time.ctime()
        print("训练和测试结束于%s."%end)
        #End = time.time()
        #print('训练和测试花费%f小时。'%((End-Start)/3600))
        return


numb = 3
home = os.environ['HOME']
datapath = home+'/Density/'
infopath = home+'/数据筛选/'
resultpath =home+'/Result/result_LSTM_m2m/result_%d/'%numb


def main():
    if os.path.exists(resultpath):
        pass
    else:
        os.makedirs(resultpath)

    start = time.time()

    dis_pred = DIS_PRED(0.001,1)
    #if home == '/home/users/junjie':
    #    dis_pred.cpu_use = 32

    dis_pred.split_shot()

    train_mat = dis_pred.get_trainData('1s')
    end1 = time.time()

    time.sleep(2)
    all_data = dis_pred.get_testData()
    end2 = time.time()

    time.sleep(2)
    dis_pred.fit_and_predict(train_mat, all_data)
    end = time.time()

    print("读取和预处理训练数据时间:%f 小时.\n" % ((end1-start)/3600))
    print("读取和预处理所有数据时间:%f 小时.\n" % ((end2-end1)/3600))
    print("读取和预处理总时间: %f 小时.\n" % ((end2-start)/3600))
    print("总的训练预测时间是 %f 小时.\n"%((end-end2)/3600))
    print("程序耗时 %f 小时" % ((end-start)/3600))
    return


if __name__ == "__main__":
    main()
