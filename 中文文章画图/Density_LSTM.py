#!/usr/bin/env python
#-*-coding=utf-8-*-
'''
此版本用的是keras模块，用的是多对一且数据生成方式是0-10,1-11,2-12，这样可以产生很多数据，
且标签的产生也做了改变
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
from keras.layers import GRU
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
        self.a = np.array(pd.read_csv(infopath+'last5.csv'))  # self.a中的破裂时刻是人工从电子密度和电流信号上观察到的
        self.disr_shot = len(np.where(self.a[:,4]==1)[0])                   # 密度极限炮总数
        self.safe_shot = len(np.where(self.a[:,4]==-1)[0])
        self.cpu_use = 8
        self.batch_size = 128
        self.epochs = 1000
        self.time_step = 20
        self.signal_kind = 8
        #self.a的第一列是炮号，第二列是人工找出的破裂时刻，第三列是标签，1代表破裂炮，-1代表安全炮。
        self.b = np.zeros((self.a.shape[0],3))
        #self.b是比较了self.a中的破裂时刻和各个诊断信号最大时间，取最小值以保证不会因为valueError出bug
        self.c = np.zeros((self.a.shape[0],6))
        #self.c是对比实际密度曲线和预测出来的密度曲线，找到其中密度值绝对值最小时的时刻
        #self.a是原始数据，self.b和self.c在实例函数中会被适应性的改变
        self.min_max = np.zeros((self.signal_kind,2))
        self.neural_struct = []
        for i in range(3,13):
            for j in range(3,13):
                self.neural_struct.append((i,j))

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

    """此函数是对某一炮的某一个信号在指定的时间段上进行取样"""
    def signal_Range(self,signal,disr_time,i):
        #b = np.where(np.around(signal[0],decimals=3)==disr_time)[0][0]
        #b = np.where(abs(signal[0]-disr_time)==abs(signal[0]-disr_time).min())[0][0]
        #b = np.where(np.trunc(signal[0]*1000)==int(disr_time*1000))[0][0]

        try:
            b = np.where(np.around(signal[0]*self.num)==int(disr_time*self.num))[0][0]
        except IndexError as e:
            print("在处理%d炮时函数signal_Range发生了错误，没有得到b:"%i,e)
        a = b - int(self.num + self.time_step)
        new_Signal = signal[:,a+1:b+1]
        return new_Signal


    """此函数是并行读取某一炮信号的所有诊断信号，同时完成插值重采样,然后取破裂前的一段时间内的样本，但没有
    归一化"""
    def get_one_shot(self,i):

        pcrl01 = np.load(datapath+'%d.npz'%(self.a[i,0]))['pcrl01']
        dfsdev = np.load(datapath+'%d.npz'%(self.a[i,0]))['dfsdev']
        vp1 = np.load(datapath+'%d.npz'%(self.a[i,0]))['vp1']
        sxr23d = np.load(datapath+'%d.npz'%(self.a[i,0]))['sxr23d']
        pxuv30 = np.load(datapath+'%d.npz'%(self.a[i,0]))['pxuv30']
        pxuv18 = np.load(datapath+'%d.npz'%(self.a[i,0]))['pxuv18']
        kmp13t = np.load(datapath+'%d.npz'%(self.a[i,0]))['kmp13t']
        pbrem10 = np.load(datapath+'%d.npz'%(self.a[i,0]))['pbrem10']

        disr_time = np.floor(min(self.a[i,7],pcrl01[0,-1],dfsdev[0,-1],vp1[0,-1],\
        sxr23d[0,-1],pxuv30[0,-1],pxuv18[0,-1],kmp13t[0,-1],pbrem10[0,-1])/self.sample_rate)*self.sample_rate


        pcrl01 = self.signal_Range(pcrl01,disr_time,i)
        dfsdev = self.signal_Range(dfsdev,disr_time,i)
        vp1 = self.signal_Range(vp1,disr_time,i)
        sxr23d = self.signal_Range(sxr23d,disr_time,i)
        pxuv30 = self.signal_Range(pxuv30,disr_time,i)
        pxuv18 = self.signal_Range(pxuv18,disr_time,i)
        kmp13t = self.signal_Range(kmp13t,disr_time,i)
        pbrem10 = self.signal_Range(pbrem10,disr_time,i)

        one_shot_train = np.zeros((self.signal_kind+2, self.num+self.time_step))
        #new_pcr, new_dfsdev, new_sxr, new_li = self.one_shot_autonormal(pcr2,dfsdev2,sxr2,li2)
        try:
            one_shot_train[0,:] = pcrl01[1,:]
        except ValueError as e:
            print("在处理%d炮时函数one_train_witho_nor出现错误，pcrl01出错:"%self.a[i,0],e)
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

        one_shot_train[-1,:] = dfsdev[0,:]

        if self.a[i,4]==1:
            one_shot_train[-2,:] = 1/( 1+np.exp( -( one_shot_train[-1,:]-(disr_time-0.2) )*25 ) )
        elif self.a[i,4]==-1:
            one_shot_train[-2,:] = 0
        print("%d炮信号的形状为:%s"%(self.a[i,0],str(one_shot_train.shape)))
        return one_shot_train, disr_time

    '''
    此函数的功能是产生训练集，并且返回归一化后的训练集,此函数内部改变了类的属性self.min_max,当且仅
    当此函数被调用后self.min_max的值才改变
    '''
    def get_trainData(self):
        res_l = []
        arry_l =[]
        pool = Pool(self.cpu_use)

        for i in self.all_train_shot:
            res = pool.apply_async(self.get_one_shot,(i,))
            res_l.append(res)
        pool.close()
        pool.join()

        for res in res_l:
            arry_l.append(res.get())

        for i,arry in zip(self.all_train_shot,arry_l):
            try:
                self.b[i,0] = self.a[i,0]
            except TypeError as e:
                print("在训练集中处理将self.a[i,0]赋值给self.b[i,0]时出错:",e)
                print("i 为 %d"%i)
                print("self.a[%d,0] 和 i 分别为:%d ,%d ."%(i,self.a[i,0],i))
                print("arry 为:",arry)
            self.b[i,1] = arry[1]
            self.b[i,2] = self.a[i,4]
            if i==self.all_train_shot[0]:
                train_mat = arry[0]
            else:
                train_mat = np.hstack([train_mat,arry[0]]) #np.hstack为水平拼接数组函数
        self.min_max[:,0] = train_mat[:-2,:].min(axis=1)
        self.min_max[:,1] = train_mat[:-2,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:-2,:] = minmax_scale(train_mat[:-2,:],axis=1)
        print("训练集形状:%s"%str(train_mat.shape))
        np.savez_compressed(resultpath+'minmax.npz',minmax = self.min_max)
        #return train_mat
        return

    def get_testData(self,shot_list,name):
        #all_test_shot = range(0,(self.disr_shot+self.safe_shot))
        #all_test_shot = list(sorted(set(range(self.safe_shot+self.disr_shot)).difference(set(all_train_shot))))
        res_l = []
        arry_l = []
        pool = Pool(multiprocessing.cpu_count())
        for i in shot_list:
            res = pool.apply_async(self.get_one_shot,(i,))
            res_l.append(res)
        pool.close()
        pool.join()

        for res in res_l:
            arry_l.append(res.get())

        for i,arry in zip(shot_list,arry_l):
            self.b[i,0] = self.a[i,0]
            self.b[i,1] = arry[1]
            self.b[i,2] = self.a[i,4]
            if i==shot_list[0]:
                test_mat = arry[0]
            else:
                test_mat = np.hstack([test_mat,arry[0]])

        test_mat[:-2,:] = (test_mat[:-2,:] - self.min_max[:,0].reshape(-1,1))/(self.min_max[:,1] - self.min_max[:,0]).reshape(-1,1)
        print("test_mat.shape:%s"%str(test_mat.shape))
        np.save(resultpath+'selfb',self.b)
        return test_mat

    '''此函数是找到预测曲线与阈值的差值的最小值的时间以及此最小值，且默认采样频率为１KHZ'''
    def find_pred_Dm1(self,thresh,time,pred_signal):
        #f = interp1d(time,pred_signal,kind='linear',fill_value="extrapolate")
        #t2 = math.ceil(pred_signal[len(pred_signal[1])-1]/0.001)*0.001
        #new_time = np.arange(time[0],time[-1],0.001)
        #new_data = f(new_time)

        Dist = pred_signal-thresh
        ind = [i for i in range(self.num-1) if ((Dist[i]<=0 and Dist[i+1]>=0))]
        if ind==[]:
            min_dis = abs(Dist).min()
            pred_Dt = time[np.where(abs(Dist)==min_dis)][0]
            return pred_Dt,min_dis+1e-16
        else:
            min_dis = 0
            pred_Dt = time[ind[0]]
            return pred_Dt,min_dis

    def build_model(self):
        model = Sequential()
        model.add(LSTM(units=100,input_shape=(self.time_step,self.signal_kind),\
                       dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
        model.add(LSTM(units=100,dropout=0.2,recurrent_dropout=0.2,\
                       return_sequences=True))
        model.add(LSTM(units=100,dropout=0.2,recurrent_dropout=0.2))
        model.add(Dense(units=1,activation='sigmoid'))
        model.compile(optimizer='nadam',loss='mse',metrics=['mae'])
        print(model.summary())
        return model

    def find_pred_Dm(self,thresh,time,pred_signal):
        Dist = pred_signal - thresh
        if Dist[0] >= 0:
            min_dis = -1
            pred_Dt = -1
        else:
            ind = [i for i in range(self.num-1) if (Dist[i]<=0 and Dist[i+1]>=0)]
            if ind == []:
                min_dis  = abs(Dist).min()+1e-16
                pred_Dt = time[np.where(abs(Dist)==abs(Dist).min())][0]
            else:
                min_dis = 0
                pred_Dt = time[ind[0]]
        return pred_Dt, min_dis

    def change_selfc(self,name,thresh,time,pred_result):
        if name=='test':
            List = self.all_test_shot
        elif name=='val':
            List = self.all_val_shot
        elif name=='train':
            List = self.all_train_shot
        elif name=='all':
            List = self.all_shot
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
                if self.c[i,4]>0:
                    self.c[i,5]=-1 #安全炮,预测正确
                else:
                    self.c[i,5]=0 #安全炮,预测错误
            elif self.c[i,2]==1:
                if self.c[i,4]==0 and 0.01<=(self.c[i,1]-self.c[i,3])<=0.4:  #破裂炮,预测正确
                    self.c[i,5]=1
                elif self.c[i,4]==0 and (self.c[i,1]-self.c[i,3])<0.01:   #破裂炮,预测滞后
                    self.c[i,5]=2
                elif self.c[i,4]==0 and (self.c[i,1]-self.c[i,3])>0.4:  #破裂炮,预测提前
                    self.c[i,5]=3
                elif self.c[i,4]!=0:  #破裂炮,预测错误
                    self.c[i,5]=4
                else:pass
            else:pass
        return

    def cal_Num(self,name):
        Num = []
        Rate = []
        if name=='test':
            List = self.all_test_shot
        elif name=='val':
            List = self.all_val_shot
        elif name=='train':
            List = self.all_train_shot
        elif name=='all':
            List = self.all_shot
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
            SnD = succ_nD/self.safe_shot
            FnD = fal_nD/self.safe_shot
            SD = succ_D/self.disr_shot
            LD = late_D/self.disr_shot
            PD = pre_D/self.disr_shot
            FD = fal_D/self.disr_shot
        else:pass
        Rate.append(SnD);Rate.append(FnD)
        Rate.append(SD);Rate.append(LD)
        Rate.append(PD);Rate.append(FD)
        return Num,Rate

    def find_best_thresh(self,val_shot,time,pred_result):
        thresh_num = len(np.arange(0,1,0.01))
        pred_Rate = np.zeros((thresh_num,7))
        pred_Res = np.zeros((thresh_num,7))
        max_area = 0
        roc_AUC = []
        for k,thresh in enumerate(np.arange(0,1,0.01)):
            self.change_selfc('val',thresh,time,pred_result)
            #统计在不同阈值下测试集预测结果
            pred_res,pred_rate = self.cal_Num('val')
            pred_res.append(thresh)
            pred_rate.append(thresh)
            pred_Rate[k,:] = pred_rate
            pred_Res[k,:] = pred_res
            arry = self.c[val_shot].copy() #注意，这里一定要用self.a.copy(),不然arry改变，self.c也变了
            for i in range(len(arry)):
                if arry[i,5]==0:
                    arry[i,5] = 1
                elif arry[i,5] == 3:
                    arry[i,5] = 1
                elif arry[i,5] == 2 or arry[i,5] == 4:
                    arry[i,5] = -1
                else:pass
            fpr,tpr,threshold = roc_curve(arry[:,2],arry[:,5])
            roc_auc = auc(fpr,tpr)
            roc_AUC.append(roc_auc)
        np.savez_compressed(resultpath+'pred_rate_res_fB.npz',\
                            pred_res=pred_Res,pred_rate=pred_Rate,AUC=roc_AUC)
        Index = roc_AUC.index(max(roc_AUC))
        print('AUC最大值为%f,在阈值为%f下取得.'%(max(roc_AUC),np.arange(0,1,0.01)[Index]))
        return np.arange(0,1,0.01)[Index]

    def cal_aver(self):
        s_nd = []
        f_nd = []
        s_d = []
        l_d = []
        p_d = []
        f_d = []
        for i in range(len(self.a)):
            if self.c[i,5]==-1:
                s_nd.append(i)
            elif self.c[i,5]==0:
                f_nd.append(i)
            elif self.c[i,5]==1:
                s_d.append(i)
            elif self.c[i,5]==2:
                l_d.append(i)
            elif self.c[i,5]==3:
                p_d.append(i)
            elif self.c[i,5]==4:
                f_d.append(i)
            else:pass
        a = s_d.copy()
        a.extend(p_d)
        a.sort()
        chzhi = self.c[a,1]-self.c[a,3]
        print("平均提前 = %fs"%np.average(chzhi))
        return

    def generate_data(self,data,shot_list,name):
        data1 = np.empty((data.shape[0],data.shape[1],data.shape[2]-1))
        data1[:,:,:-1] = data[:,:,:-2]
        data1[:,:,-1] = data[:,:,-1]
        data_X = []
        data_y = []
        for i,j in enumerate(shot_list):
            if self.a[j,4] == -1:
                y = np.zeros(self.num)
            else:
                y = 1/(1+np.exp(-(data1[i][:,-1][-self.num:]-(self.b[j,1]-0.2))*25))
            data_y.append(y)
            for k in range(self.num):
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

        #将输入转换成[样本个数，时间步长，特征]
        all_data = all_data.T.reshape(self.safe_shot+self.disr_shot,self.num+self.time_step,self.signal_kind+2)
        X, y = self.generate_data(all_data,self.all_shot,'all_data')
        #print('X 形状为 ：%s,y 形状为 ： %s.'%(str(X.shape),str(y.shape)))
        np.savez_compressed(resultpath+'all_xy.npz',data=all_data)
        np.savez_compressed(resultpath+'allXY.npz',X=X,Y=y)
        #训练集
        train_mat = all_data[self.all_train_shot]
        X_train, y_train = self.generate_data(train_mat,self.all_train_shot,'train')
        #print('X_train 形状为 ：%s,y_train形状为 ： %s.'%(str(X_train.shape),str(y_train.shape)))
        #np.savez_compressed(resultpath+'train_xy.npz',data=train_mat)

        #验证集
        val_mat = all_data[self.all_val_shot]
        X_val, y_val = self.generate_data(val_mat,self.all_val_shot,'validation')
        #print('X_val 形状为 ：%s,y_val 形状为 ：%s.'%(str(X_val.shape),str(y_val.shape)))
        #np.savez_compressed(resultpath+'val_xy.npz',data=val_mat)

        #测试集
        #test_mat = all_data[self.all_test_shot]
        #X_test, y_test = self.generate_data(test_mat,self.all_test_shot,'test')
        #print('X_test shape is %s,y_test shape is %s.'%(str(X_test.shape),str(y_test.shape)))
        #np.savez_compressed(resultpath+'test_xy.npz',data=test_mat)

        model = self.build_model()
        #保存模型结构
        model_json = model.to_json()
        with open(resultpath+'my_model.json','w') as File:
            File.write(model_json)

        if os.path.exists(resultpath+'my_logs'):
            pass
        else:
            os.makedirs(resultpath+'my_logs')

        #设置回调函数分别用于降低学习率、中断训练、保存最优模型、可视化框架
        callback_list = [ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=4),\
                         EarlyStopping(monitor='val_loss',patience=10),\
                         ModelCheckpoint(filepath=resultpath+'best.weight.h5',\
                                         monitor='val_loss',verbose=1,save_best_only=True)]
                         #TensorBoard(log_dir=resultpath'/my_logs',histogram_freq=1)
        #在训练集上训练模型
        History = model.fit(x=X_train,y=y_train,epochs=20,\
                            batch_size=512, validation_data=(X_val,y_val),verbose=1,\
                            callbacks=callback_list)
        #释放内存
        #del X_train,y_train

        print(History.history.keys())
        history_dict = pd.DataFrame(History.history)
        history_dict.to_csv(resultpath+'history_dict.csv')

        #在验证集上进行预测

        #加载最优模型
        #best_model = load_model(resultpath+'best.weight.h5')
        print('X 形状为 ：%s,y 形状为 ： %s.'%(str(X.shape),str(y.shape)))
        print('X_train 形状为 ：%s,y_train形状为 ： %s.'%(str(X_train.shape),str(y_train.shape)))
        print('X_val 形状为 ：%s,y_val 形状为 ：%s.'%(str(X_val.shape),str(y_val.shape)))

        print("+++++++++开始在验证集上测试+++++++++++")
        result0 = model.predict(X_val,batch_size=512, verbose=1)     #result0.shape=(all_test_shot*1000,1)
        print("result0  形状为:%s"%str(result0.shape))
        result1 = result0.reshape(-1,self.num)             #result1.shape=(all_test_shot,1000)
        print("result1  形状为:%s"%str(result1.shape))
        np.savez_compressed(resultpath+'pred_val_result.npz',pred_result1=result1)
        val_time = val_mat[:,-self.num:,-1]
        print("val_time  形状为:%s"%str(val_time.shape))
        best_thresh = self.find_best_thresh(self.all_val_shot,val_time,result1)

        #del X_val,y_val
        #在所有数据上测试
        print("+++++++开始在所有数据集上测试+++++++++++")

        result2 = model.predict(X,batch_size=512,verbose=1)
        result3 = result2.reshape(-1,self.num)

        np.savez_compressed(resultpath+'pred_all_result.npz',pred_result3=result3)
        all_time = all_data[:,-self.num:,-1]
        self.change_selfc('all',best_thresh,all_time,result3)
        np.savez(resultpath+'selfc_all.npz',selfc=self.c)

        train_Res,train_Rate = self.cal_Num('train')
        val_Res,val_Rate = self.cal_Num('val')
        test_Res,test_Rate = self.cal_Num('test')
        all_Res,all_Rate = self.cal_Num('all')
        self.cal_aver()
        print("在训练集上的结果:\n",'train_Res:',train_Res,'\n train_Rate:\n',train_Rate,'\n')
        print("在验证集上的结果:\n",'val_Res:',val_Res,'\n val_Rate:\n',val_Rate,'\n')
        print("在测试集上的结果:\n",'test_Res:',test_Res,'\n test_Rate:\n',test_Rate,'\n')
        print("在所有数据上的结果:\n",'all_Res:',all_Res,'\n all_Rate:\n',all_Rate,'\n')
        end = time.ctime()
        print("训练和测试结束于%s."%end)
        End = time.time()
        print('训练和测试花费%f小时。'%((End-Start)/3600))
        return

def main():
    if os.path.exists(resultpath):
        pass
    else:
        os.makedirs(resultpath)
    start = time.time()
    dis_pred = DIS_PRED(0.001,1)
    if home == '/home/users/junjie':
        dis_pred.cpu_use = 16
    all_train_shot,all_val_shot,all_test_shot = dis_pred.split_shot()

    dis_pred.get_trainData()
    print("所有的训练数据得到！！！！！！")

    end1 = time.time()
    time.sleep(2)
    all_data = dis_pred.get_testData(dis_pred.all_shot,'alldata')
    print("所有的数据得到！！！！！！")
    print("所有数据集形状:%s"%str(all_data.shape))
    end2 = time.time()
    dis_pred.fit_and_predict(all_data)
    end = time.time()
    print("读取和预处理训练数据时间:%f 小时.\n" % ((end1-start)/3600))
    print("总的计算时间是 %f 小时.\n"%((end-end2)/3600))
    print("读取和预处理所有数据时间:%f 小时.\n" % ((end2-end1)/3600))
    print("读取和预处理总时间: %f 小时.\n" % ((end2-start)/3600))
    print("程序耗时 %f 小时" % ((end-start)/3600))
    return


numb = 9
home = os.environ['HOME']
datapath = home+'/density/'
infopath = home+'/数据筛选/'
resultpath =home+'/result/result_LSTM/result_%d/'%numb


if __name__=="__main__":
    main()

