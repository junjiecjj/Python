#!/usr/bin/env python
#-*-coding=utf-8-*-
'''
此版本是在第clust8的基础上把训练测试部分变成了并行，前面读取、插值、重采样、归一化等预处理阶段的代码也并行化，
这是主要耗时的地方。
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
from multiprocessing import Process,Pool #多进程模块
from multiprocessing import Lock
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
    def __init__(self,Resample_rate,Exp_range,dis_shut,saf_shut,Train_shut,Test_shut_D,Test_shut_nD):
        self.sample_rate = Resample_rate        # 0.001
        self.exp_range = Exp_range              # 1s
        self.disr_shut = dis_shut               # 491
        self.safe_shut = saf_shut               # 681
        self.train_shut = Train_shut            # 200
        self.test_shut_D = Test_shut_D          # 491
        self.test_shut_nD = Test_shut_nD        # 681
        self.a = self.data_info()               # self.a中的破裂时刻是人工从电子密度和电流信号上观察到的
        #self.a的第一列是炮号，第二列是人工找出的破裂时刻，第三列是标签，1代表破裂炮，-1代表安全炮。
        self.b = np.zeros((self.a.shape))
        #self.b是比较了self.a中的破裂时刻和各个诊断信号最大时间，取最小值以保证不会因为valueError出bug
        self.c = np.zeros((self.a.shape[0],5))
        #self.c是对比实际密度曲线和预测出来的密度曲线，找到其中密度值绝对值最小时的时刻
        #self.a是原始数据，self.b和self.c在实例函数中会被适应性的改变
        self.min_max = np.zeros((9,2))
        self.neural_struct = []
        for i in range(3,13):
            for j in range(3,13):
                self.neural_struct.append((i,j))

    @staticmethod
    def data_info():
        sheet=xlrd.open_workbook(r'/home/jack/公共的/excel_data/info.xlsx')
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
    def find_Dis_tm(self,A,pcr,dfsdev,lmsz,kmp,ic,sxr,pxu30,pxu18,vp,i):
        b = []
        disr_time = A[i,1]; b.append(disr_time)
        pcr_D = pcr[0,len(pcr[0,:])-1]; b.append(pcr_D) ; del pcr
        dfsdev_D = dfsdev[0,len(dfsdev[0,:])-1]; b.append(dfsdev_D) ; del dfsdev
        lmsz_D = lmsz[0,len(lmsz[0,:])-1]; b.append(lmsz_D) ; del lmsz
        kmp_D = kmp[0,len(kmp[0,:])-1]; b.append(kmp_D) ; del kmp
        ic_D = ic[0,len(ic[0,:])-1]; b.append(ic_D) ; del ic
        sxr_D = sxr[0,len(sxr[0,:])-1]; b.append(sxr_D) ; del sxr
        pxu30_D = pxu30[0,len(pxu30[0,:])-1]; b.append(pxu30_D) ; del pxu30
        pxu18_D = pxu18[0,len(pxu18[0,:])-1]; b.append(pxu18_D) ; del pxu18
        vp_D = vp[0,len(vp[0,:])-1]; b.append(vp_D) ; del vp
        #betap_D = betap[0,len(betap[0,:])-1]; b.append(betap_D) ; del betap
        #li_D = li[0,len(li[0,:])-1]; b.append(li_D) ; del li
        #q_D = q[0,len(q[0,:])-1]; b.append(q_D) ; del q
        disr_time = min(b)
        Index = b.index(min(b))
        Dict = {0:'DT',1:'pcr',2:'dfsdev',3:'lmsz',4:'kmp',5:'ic',6:'sxr',7:'pxu30',8:'pxu18',9:'vp'}
        print("the min disr_time is :%s."% Dict[Index])
        self.b[i,0] = A[i,0]
        self.b[i,1] = disr_time
        self.b[i,2] = A[i,2]
        return disr_time

    """此函数是对某一炮的某一个信号在指定的时间段上进行取样"""
    def signal_Range(self,signal,disr_time):
        b = int(np.where(abs(signal[0]-disr_time)<=self.sample_rate)[0][0])
        a = b - int(self.exp_range/self.sample_rate)
        new_Signal = signal[:,a:b]
        return new_Signal

    """此函数是并行读取某一炮信号的所有诊断信号，同时完成插值重采样,然后取破裂前的一段时间内的样本"""
    def get_one_shut(self,i):
        A = self.a
        pool = Pool(9)
        res_l = []
        arry_l = []
        Signal_name = ['PCRL','DFSDEV','LMSZ','KMP13T','IC1','SXR23D','PXUV30','PXUV18','VP1','BETAP']
        for k in range(9):
            res = pool.apply_async(self.get_and_resample,args=(A,i,Signal_name[k],))
            res_l.append(res)
            #arry_l.append(res.get())
        pool.close()
        pool.join()

        for res in res_l:
            arry_l.append(res.get())
        pcr,dfsdev,lmsz,kmp,ic,sxr,pxu30,pxu18,vp = arry_l

        disr_time = self.find_Dis_tm(A,pcr,dfsdev,lmsz,kmp,ic,sxr,pxu30,pxu18,vp,i)
        pool = Pool(9)
        res_l = []
        arry_l = []
        sig_name = [pcr,dfsdev,lmsz,kmp,ic,sxr,pxu30,pxu18,vp]
        for k in range(9):
            res = pool.apply_async(self.signal_Range,(sig_name[k],disr_time,))
            #print(res.get())
            res_l.append(res)
            #arry_l.append(res.get())
        pool.close()
        pool.join()
        for res in res_l:
            arry_l.append(res.get())
            print(res.get(),'\n')
        #print(arry_l)
        pcr,dfsdev,lmsz,kmp,ic,sxr,pxu30,pxu18,vp = arry_l
        return pcr,dfsdev,lmsz,kmp,ic,sxr,pxu30,pxu18,vp

    '''
    此函数返回的是各信号经过插值重采样、截取、但没有归一化的训练集,one_shut_train的前12行分别是信号值，第13行是标签，也就是最大电子密度，
    第15行是时间
    '''
    def one_train_witho_nor(self,i):
        pcr,dfsdev,lmsz,kmp,ic,sxr,pxu30,pxu18,vp = self.get_one_shut(i)
        one_shut_train = np.zeros((11,dfsdev.shape[1]))
        #new_pcr, new_dfsdev, new_sxr, new_li = self.one_shut_autonormal(pcr2,dfsdev2,sxr2,li2)
        one_shut_train[0,:] = pcr[1,:]
        one_shut_train[1,:] = dfsdev[1,:]
        one_shut_train[2,:] = lmsz[1,:]
        one_shut_train[3,:] = kmp[1,:]
        one_shut_train[4,:] = ic[1,:]
        one_shut_train[5,:] = sxr[1,:]
        one_shut_train[6,:] = pxu30[1,:]
        one_shut_train[7,:] = pxu18[1,:]
        one_shut_train[8,:] = vp[1,:]
        #one_shut_train[9,:] = betap[1,:]
        #one_shut_train[10,:] = li[1,:]
        #one_shut_train[10,:] = q[1,:]
        one_shut_train[9,:] = dfsdev[1,:].max()
        one_shut_train[10,:] = dfsdev[0,:]
        return one_shut_train

    '''
    此函数的功能是产生训练集，并且返回归一化后的训练集,此函数内部改变了类的属性self.min_max,当且仅
    当此函数被调用后self.min_max的值才改变
    '''
    def all_trainset(self):
        #all_train_shut = random.sample(range(self.disr_shut),self.train_shut)
        #all_train_shut = sorted(all_train_shut)
        all_train_shut = range(10)
        for i,j in enumerate(all_train_shut):
            one_train_shut = self.one_train_witho_nor(j)
            if i==0:
                train_mat = one_train_shut
            else:
                train_mat = np.hstack([train_mat, one_train_shut]) #np.hstack为水平拼接数组函数

        self.min_max[:,0] = train_mat[:9,:].min(axis=1)
        self.min_max[:,1] = train_mat[:9,:].max(axis=1)
        #self.min_max = min_max_Ran
        train_mat[:9,:] = minmax_scale(train_mat[:9,:],axis=1)
        train_mat[9,:] = (train_mat[9,:]-self.min_max[1,0])/(self.min_max[1,1]-self.min_max[1,0])
        #np.savez_compressed('/gpfs/home/junjie/公共的/py文件/data/train.npz',train_set=train_mat,train_shut=all_train_shut)
        return train_mat,all_train_shut

    def all_testset(self):
        all_test_shut = range(0,(self.disr_shut+self.safe_shut))
        for i,j in enumerate(all_test_shut):
            one_test_shut = self.one_train_witho_nor(j)
            if i==0:
                test_mat = one_test_shut
            else:
                test_mat = np.hstack([test_mat, one_test_shut])
        test_mat[:12,:] = test_mat[:12,:]-np.tile(self.min_max[:12,0].reshape(12,1),(1,test_mat.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(12,1)
        test_mat[:12,:] = test_mat[:12,:]/np.tile(Range,(1,test_mat.shape[1]))
        test_mat[12,:] = (test_mat[12,:]-self.min_max[1,0])/(self.min_max[1,1]-self.min_max[1,0])
        np.savez_compressed('/gpfs/home/junjie/公共的/py文件/data/test.npz',test_set=test_mat,test_shut=all_test_shut)
        return test_mat, all_test_shut

    def One_Testset(self,test_shut):
        pcr,dfsdev,lmsz,kmp,ic,sxr,pxu30,pxu18,vp,betap,li,q = self.get_one_shut(test_shut)
        one_test_shut = self.one_train_witho_nor(pcr,dfsdev,lmsz,kmp,ic,sxr,pxu30,pxu18,vp,betap,li,q)
        one_test_shut[:12,:] = one_test_shut[:12,:]-np.tile(self.min_max[:,0].reshape(12,1),(1,one_test_shut.shape[1]))
        Range = (self.min_max[:,1]-self.min_max[:,0]).reshape(12,1)
        one_test_shut[:12,:] = one_test_shut[:12,:]/np.tile(Range,(1,one_test_shut.shape[1]))
        one_test_shut[12,:] = (one_test_shut[12,:]-self.min_max[1,0])/(self.min_max[1,1]-self.min_max[1,0])
        return one_test_shut


    #@staticmethod
    def plot_one_shut(self ,time_y_real ,y_predict ,test_shut_num):
        Disr_time = self.b[test_shut_num,1]
        pred_D_T = self.c[test_shut_num,1]
        fig,ax = plt.subplot()
        l1, = ax.plot(time_y_real[1,:],time_y_real[0,:],color='k',linestyle='-')
        l2, = ax.plot(time_y_real[1,:],y_predict.T,color='r',linestyle='-')
        ax.axvline(x=Disr_time, ls=':', color='#0000FF')
        ax.annotate(r'$t_{real\_disr}$',xy=(Disr_time,0),xytext=(Disr_time+0.1,0.2),\
        arrowprops = dict(arrowstyle='->',connectionstyle='arc3',color='#0000FF'))

        ax.axvline(x=pred_D_T, ls=':', color='#9400D3')
        ax.annotate(r'$t_{pred\_disr}$',xy=(pred_D_T,0),xytext=(pred_D_T+0.1,0.2),\
        arrowprops = dict(arrowstyle='->',connectionstyle='arc3',color='#9400D3'))

        ax.legend((l1,l2), ('real density','predict density limit'), loc='best')
        ax.set_xlabel('time(s)')
        ax.set_ylabel('electronic density')
        ax.set_title(r'$t_{real\_disr}$=%.3f'%Disr_time,loc='left',color='#0000FF')
        ax.set_title(r'$t_{pred\_disr}$=%.3f'%pred_D_T,loc='right',color='#9400D3')
        fig.suptitle('Shut:%d' % test_shut_num)
        plt.show()
        return

    #@staticmethod
    def fun(self,k,trainset,test_mat,all_test_shut,pred_array):
        Start = time.time()
        start = time.ctime()
        #name = multiprocessing.current_process().name
        print("process %s start at %s .\n" % (name,time.ctime()))
        trainset = trainset.T
        X_train = trainset[:,:-3]
        y_train = trainset[:,-3]
        clf = MLPRegressor(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=self.neural_struct[k])
        clf.fit(X_train,y_train)
        #cengindex = 0
        #for wi in clf.coefs_:
        #    cengindex += 1  # 表示第几层神经网络。
        #    print('第%d层网络层:' % cengindex)
        #    print('权重矩阵维度:',wi.shape)
        #    print('系数矩阵:\n',wi)
        #print('网络节点信息: %s' % clf.coefs_)
        #return clf

    #def test_all(self,clf,test_mat,all_test_shut):

        m,n = test_mat.shape
        Y_pred = np.zeros((1,n))
        for i,j in enumerate(all_test_shut):
            one_test_shut = test_mat[:,1000*i:1000*(i+1)]
            x_test = one_test_shut[:-3,:]
            y_predict = clf.predict(x_test.T)
            Dist = abs(y_predict - x_test[1,:]) #y_predict是行数组，且为array([........])形式
            pred_Dis_T = one_test_shut[14,:][np.where(Dist==Dist.min())[0][0]]
            if i==1:
                Y_pred = y_predict
            else:
                Y_pred = np.hstack([Y_pred,y_predict])
            self.c[j,0] = self.b[j,0]
            self.c[j,1] = self.b[j,1]
            self.c[j,2] = self.b[j,2]
            self.c[j,3] = pred_Dis_T
            self.c[j,4] = Dist.min()
        pred_array[k,:] = Y_pred
        #return

    #def cal_MR_SR(self,lock,all_test_shut,k):
        for i in all_test_shut:
            if self.c[i,2]==-1:
                if self.c[i,4]>=0.001:
                    self.c[i,5] = -1
                else:
                    self.c[i,5] = 0
            elif self.c[i,2]==1:
                if self.c[i,4]<=0.001 and 0.01<=(self.c[i,2]-self.c[i,3])<=1:
                    self.c[i,5] = 1
                elif self.c[i,4]<=0.001 and (self.c[i,2]-self.c[i,3])>0.01:
                    self.c[i,5] = 2
                elif self.c[i,4]<=0.001 and (self.c[i,2]-self.c[i,3])>1:
                    self.c[i,5] = 3
                elif self.c[i,4]>0.001:
                    self.c[i,5] = 4
                else:
                    pass
            else: 
                pass

        succ_D = 0
        late_D = 0
        pre_D = 0
        fal_D = 0
        succ_nD = 0
        fal_nD = 0
        for i in all_test_shut:
            if selc.c[i,5]== -1:
                succ_nD+=1
            elif self.c[i,5]==0:
                fal_nD+=1
            elif self.c[i,5]==1:
                succ_D+=1
            elif self.c[i,5]==2:
                late_D+=1
            elif self.c[i,5]==3:
                pre_D+=1
            elif self.c[i,5]==4:
                fal_D+=1
        succ_rate = (succ_D+succ_nD)/(self.test_shut_D+self.test_shut_nD)
        false_rate = (fal_D+fal_nD)/(self.test_shut_D+self.test_shut_nD)
        late_rate = late_D/self.test_shut_D
        pre_rate = pre_D/self.test_shut_D

        end = time.ctime()
        End = time.time()
        #lock.acquire()
        with open('/gpfs/home/junjie/公共的/py文件/result/D_clust9.txt','a') as f:
            f.write('***************分割线*******************\n')
            f.write('process %s start at %s,finish at %s,counsum %f hours.\n'%(name,start,end,(End-Start)/3600))
            f.write('the result under the structure:(%d,%d)is:\n'%self.neural_struct[k])
            f.write('the succ rate is       : %f\n' % succ_rate)
            f.write('the false rate is      : %f\n' % false_rate)
            f.write('the late rate is       : %f\n' % late_rate)
            f.write('the premature rate is  : %f\n' % pre_rate)
        #print('The predict result is:\n ')
        #print('suss_rate: %.3f\n' % suss_rate)
        #print('miss_rate: %.3f\n' % miss_rate)
        #print('late_rate: %.3f\n' % late_rate)
        #print('false_rate: %.3f\n'% false_rate)
        #lock.release()
        print("process %s end at %s .\n" % (name,time.ctime()))
        return self.c

print("t1 is :%s" % time.ctime())

dis_pred2 = DIS_PRED(0.001,1,491,680,200,491,680)
print("t2 is :%s" % time.ctime())
'''
pcr2_1,dfsdev2_1,lmsz2_1,kmp2_1,ic2_1,sxr2_1,pxu302_1,pxu182_1,vp2_1,betap2_1,li2_1,q2_1=dis_pred2.get_one_shut(0)
print("get data and resample t3 is :%s" % time.ctime())
disr_time2=dis_pred2.find_Dis_tm(pcr2_1,dfsdev2_1,lmsz2_1,kmp2_1,ic2_1,sxr2_1,pxu302_1,pxu182_1,vp2_1,betap2_1,li2_1,q2_1,0)
print("find disr t4 is :%s" % time.ctime())
pcr2_2,dfsdev2_2,lmsz2_2,kmp2_2,ic2_2,sxr2_2,pxu302_2,pxu182_2,vp2_2,betap2_2,li2_2,q2_2=dis_pred2.shut_Range(pcr2_1,dfsdev2_1,lmsz2_1,kmp2_1,ic2_1,sxr2_1,pxu302_1,pxu182_1,vp2_1,betap2_1,li2_1,q2_1,0)
print("shut range t5 is :%s" % time.ctime())

#one_shut_train2=dis_pred2.one_train_witho_nor(0)
print("one_train_witho_nor t is :%s" % time.ctime())

train_mat2,all_train_shut2=dis_pred2.all_trainset()
print("all_trainset t is :%s" % time.ctime())
'''
pcr,dfsdev,lmsz,kmp,ic,sxr,pxu30,pxu18,vp=dis_pred2.get_one_shut(0)

