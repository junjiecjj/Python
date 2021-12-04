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
from multiprocessing import Process,Pool
import multiprocessing
#from matplotlib.font_manager import FontProperties
#font = FontProperties(fname = "/usr/share/fonts/truetype/arphic/SimSun.ttf", size=13) 
#font1 = FontProperties(fname = "/usr/share/fonts/truetype/arphic/SimSun.ttf", size=16)
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
        self.disr_shut = 491                    # 491
        self.safe_shut = 680                    # 681
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
    

def Draw(name):
    if(name == "disruption"):
        arry = statis[list(range(491))]
        lab = '破裂炮总数:491'
        file1 = '/home/jack/snap/ahead_disruption.eps'
        file2 = '/home/jack/snap/ahead_disruption.svg'
    elif(name == "safe"):
        arry = statis[list(range(491,1171))]
        lab = '安全炮总数:680'
        file1 = '/home/jack/snap/ahead_safe.eps'
        file2 = '/home/jack/snap/ahead_safe.svg'
    chazhi= arry[:,3]-arry[:,4]
    fig,axs = plt.subplots(1,1,figsize=(6,4),)
    num  = 50
    n, bins, patches = axs.hist(chazhi, bins=num, color='k',alpha=0.5, normed = 0)
    a = []
    for i in range(num):
        a.append((bins[i]+bins[i+1])/2)
    plt.plot(a,n,color = "k")
    #plt.legend(prop=font,loc='best',borderaxespad=0,edgecolor='black',)
    plt.xlabel('时间差值/s',fontproperties = font)
    plt.ylabel('炮数',fontproperties = font)
    plt.tight_layout()
    plt.savefig(file1,format='eps',dpi=1000)
    plt.savefig(file2,format='svg',dpi=1000)
    return

def DRaw():
    arry1 = statis[list(range(491))]
    lab1 = '破裂炮总数:491'
    arry2 = statis[list(range(491,1171))]
    lab2 = '安全炮总数:680'

    chazhi1 = arry1[:,3]-arry1[:,4]
    chazhi2 = arry2[:,3]-arry2[:,4]
    
    fig,axs = plt.subplots(2,1,figsize=(6,6),)
    num  = 50
    n1, bins1, patches1 = axs[0].hist(chazhi1, bins=num, color='k',alpha=0.5, normed = 0)
    a1 = []
    for i in range(num):
        a1.append((bins1[i]+bins1[i+1])/2)
    axs[0].plot(a1,n1,color = "k" ,label = 'disruptive pulses')
    #plt.legend(prop=font,loc='best',borderaxespad=0,edgecolor='black',)
    axs[0].set_xlabel('time difference(s)')
    axs[0].set_ylabel('number of pulse')
    axs[0].set_title('(a)',loc = 'left')
    axs[0].legend(loc = 'best')
    n2, bins2, patches2 = axs[1].hist(chazhi2
                             , bins=num, color='k',alpha=0.5, normed = 0)
    a2 = []
    for i in range(num):
        a2.append((bins2[i]+bins2[i+1])/2)
    axs[1].plot(a2,n2,color = "k",label = 'safe pulse',)
    #plt.legend(prop=font,loc='best',borderaxespad=0,edgecolor='black',)
    axs[1].set_xlabel('time difference(s)')
    axs[1].set_ylabel('pluse number')
    axs[1].set_title('(b)',loc = 'left')
    axs[1].legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(file + 'ahead_time.eps',format='eps',dpi=1000)
    plt.savefig(file + 'ahead_time.svg',format='svg',dpi=1000)
    return

file = '/home/jack/snap/picture/englishpicture/'
get = DIS_PRED(0.001,1)
statis=get.get_testData(list(range(1171)))
DRaw()
