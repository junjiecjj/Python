# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import xlrd
from scipy.interpolate import interp1d
from os import listdir
from sklearn.preprocessing import MinMaxScaler,minmax_scale
from sklearn.neural_network import MLPClassifier,MLPRegressor

all_file=listdir(r'D:\disruption-code\数据\密度极限')
m=len(all_file)
sheet=xlrd.open_workbook('D:\disruption-code\excel_of_data/密度极限.xlsx')
table=sheet.sheet_by_name('Sheet1')

def getkey(dict,arg):
    for k,v in dict.items():
        if v==arg:
            return k
        else:
            pass
#此函数是对某一种信号进行重采样
def one_signal_resample(signal,sample_rate=0.001):
    f=interp1d(signal[1],signal[0],kind='linear',fill_value="extrapolate")
    new_time=np.arange(0,signal[1,len(signal[1])-1],sample_rate)
    new_data=f(new_time)
    new_signal=np.zeros((2,len(new_time)))
    new_signal[0]=new_time
    new_signal[1]=new_data
    return new_signal

#此函数是对某一炮的所有信号进行重采样
def one_shut_resample(pcr,dfsdev,sxr,li,samp_fre=0.001):
    new_pcr=one_signal_resample(pcr,sample_rate=samp_fre)
    new_dfsdev=one_signal_resample(dfsdev,sample_rate=samp_fre)
    new_sxr=one_signal_resample(sxr,sample_rate=samp_fre)
    new_li=one_signal_resample(li,sample_rate=samp_fre)
    return new_pcr, new_dfsdev, new_sxr, new_li

def find_Dis_tm(pcr,dfsdev,sxr,li,i):
    a=[]
    disr_time=table.row_values(i)[1]; a.append(disr_time)
    pcr_D=pcr[1,len(pcr[1,:])-1]; a.append(pcr_D)
    dfsdev_D=dfsdev[1,len(dfsdev[1,:])-1]; a.append(dfsdev_D)
    sxr_D=sxr[1,len(sxr[1,:])-1]; a.append(sxr_D)
    li_D=li[1,len(li[1,:])-1]; a.append(li_D)
    disr_time=min(a)
    return disr_time

#此函数是对某一炮的某一个信号在指定的时间段上进行取样,i为excel的第i行
def signal_Range(signal,disr_time):
    b=int(np.where(signal==disr_time)[1])
    a=int(np.where(signal==disr_time-1)[1])
    new_signal=signal[:,a:b]
    return new_signal

def shut_Range(pcr,dfsdev,sxr,li,i):
    disr_time=find_Dis_tm(pcr,dfsdev,sxr,li,i)
    new_pcr=signal_Range(pcr,disr_time)
    new_dfsdev=signal_Range(pcr,disr_time)
    new_sxr=signal_Range(sxr,disr_time)
    new_li=signal_Range(li,disr_time)
    return new_pcr, new_dfsdev, new_sxr, new_li

'''这里的minmax_scale函数很奇特，当输入的minmax_scale(a)中的a为二维数组时，默认以
对每列进行归一化，相当于minmax_(a,axis=0);想要对每一行归一化时需要minmax_(a,axis=1)；
当a为一位数组时，比如a=array([1,2,3,4]),这时minmax_(a)对每一行求归一化，相当于minmax_(a,axis=0)；
反之，minmax_(a,axis=1)是对每一列求归一化'''
#此函数是对某一炮的某一个信号进行归一化
def one_signal_normal(signal):
    new_signal=np.zeros(signal.shape)
    signal_data=minmax_scale(signal[0])
    new_signal[0]=signal[1]
    new_signal[1]=signal_data
    return new_signal

#此函数是对某一炮的所有信号进行归一化
def one_shut_autonormal(pcr,dfsdev,sxr,li):
    new_pcr=one_signal_normal(pcr)
    new_dfsdev=one_signal_normal(dfsdev)
    new_sxr=one_signal_normal(sxr)
    new_li=one_signal_normal(li)
    return new_pcr, new_dfsdev, new_sxr, new_li

def one_trainset(pcr,dfsdev,sxr,li,i):
    pcr1,dfsdev1,sxr1,li1=one_shut_resample(pcr,dfsdev,sxr,li,samp_fre=0.001)
    pcr2,dfsdev2,sxr2,li2=shut_Range(pcr1,dfsdev1,sxr1,li1,i)
    shut_train=np.zeros((5,dfsdev.shape[1]))
    new_pcr, new_dfsdev, new_sxr, new_li=one_shut_autonormal(pcr2,dfsdev2,sxr2,li2)
    shut_train[0,:]=new_pcr
    shut_train[1,:]=new_dfsdev
    shut_train[2,:]=new_sxr
    shut_train[3,:]=new_li
    shut_train[4,:]=new_dfsdev[1,:].max()
    return shut_train

def all_resample():
    #in:  sheet1.row_values(0)
    #out: [65001.0, 3.321, 'DFSDEV']
    #in:  sheet1.row_values(0)[0]
    #out: 65001.0
    # row_num,col_num=table.nrows,table.ncols
    # table.row(10)[0]   读取第11行第一列的数据
    for i in range():
        try:
            pcr=np.loadtxt(r'D:\disruption-code\数据\密度极限\%d\%d_PCR.txt' % (table.row_values(i)[0],table.row_values(i)[0]))
            dfsdev=np.loadtxt(r'D:\disruption-code\数据\密度极限\%d\%d_DFSDEV.txt' % (table.row_values(i)[0],table.row_values(i)[0]))
            li=np.loadtxt(r'D:\disruption-code\数据\密度极限\%d\%d_LI.txt' % (table.row_values(i)[0],table.row_values(i)[0]))
            sxr=np.loadtxt(r'D:\disruption-code\数据\密度极限\%d\%d_SXR.txt' % (table.row_values(i)[0],table.row_values(i)[0]))
            new_pcr,new_dfsdev, new_sxr,new_li=one_shut_resample(pcr,dfsdev,sxr,li)
        except FileNotFoundError as e:
            continue
        #data_Af_resample=one_resample(pcr,dfsdev,li,sxr)
    return new_pcr,new_dfsdev, new_sxr,new_li

all_resample()
