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
import math
from multiprocessing import Process,Pool  
import multiprocessing

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
        #self.a的第一列是炮号，第二列是人工找出的破裂时刻，第三列是标签，1代表破裂炮，-1代表安全炮。
        self.b = np.zeros((self.a.shape[0],5))

    @staticmethod
    def data_info():
        sheet = xlrd.open_workbook(r'/home/jack/公共的/excel_data/info.xlsx')
        sheet1 = sheet.sheet_by_name('Sheet1')
        row_num,col_num = sheet1.nrows,sheet1.ncols
        A=[]
        for i in range(row_num):
            A.append(sheet1.row_values(i))
        return np.array(A)

    """此函数是先读取某一炮的某一个诊断信号，然后插值采样"""
    def get_and_resample(self,A,i,Signal):
        signal = np.loadtxt('/home/jack/data/%d/%d_%s.txt' % (A[i,0],A[i,0],Signal))

        f = interp1d(signal[1],signal[0],kind='linear',fill_value="extrapolate")
        t2 = math.ceil(signal[1,len(signal[1])-1]/self.sample_rate)*self.sample_rate
        new_time = np.arange(0,t2,self.sample_rate)
        new_data = f(new_time)
        New_signal = np.zeros((2,len(new_time)))
        New_signal[0] = new_time
        New_signal[1] = new_data
        return New_signal
    
        """此函数通过比较第A[i,0]炮破裂炮的破裂时间和各信号的时间序列的最后一个值的最小值来确定破裂时刻"""
    def find_Dis_tm(self,A,dfsdev,i):
        b = []
        disr_time = A[i,1]; b.append(disr_time)
        dfsdev_D = dfsdev[0,len(dfsdev[0,:])-1]; b.append(dfsdev_D)
        disr_time = min(b)
        Index = b.index(min(b))
        Dict = {0:'DT',1:'dfsdev'}
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

    """此函数是并行读取某一炮信号的所有诊断信号，同时完成插值重采样,
    然后取破裂前的一段时间内的样本，但没有归一化"""
    def get_one_shut(self,i):
        A = self.a
        dfsdev = self.get_and_resample(A,i,'DFSDEV')
        disr_time = self.find_Dis_tm(A,dfsdev,i)
        new_dens = self.signal_Range(dfsdev,disr_time,i)
        max_Dtime = new_dens[0][np.where(new_dens[1]==new_dens[1].max())][0]
        return disr_time,max_Dtime

    def get_testData(self,shut_list):
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
            self.b[i,1] = self.a[i,1]
            self.b[i,2] = self.a[i,2]
            self.b[i,3] = arry[0]
            self.b[i,4] = arry[1]
        #np.savez_compressed('/gpfs/home/junjie/音乐/data/statist.npz',selfb=self.b)
        return self.b


get = DIS_PRED(0.001,1)
statis=get.get_testData(list(range(491)))
a = statis[:,3]-statis[:,4]
res = plt.hist(a, bins=50, color='steelblue', normed=True )
