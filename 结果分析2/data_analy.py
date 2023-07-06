#!/usr/bin/env python3
#!-*-coding=utf-8-*-
#########################################################################
# File Name: data_analy.py
# Author: chenjunjie
# mail: 2716705056@qq.com
# Created Time: 2018.09.04
#########################################################################
"""
此程序是把在某个模型下预测出的(491+680)1171炮密度极限与实际密度值求差，
然后求出每炮的差值的最小值，做出类似于《disruption_predict》中华科的图8.
"""
import numpy as np
import matplotlib.pyplot as plt
import xlrd

#shut1 = int(input("which shut you want to shart draw:\n"))
#shut2 = int(input("which shut you want to end draw:\n"))

class Draw_Re(object):
    def __init__(self):
        self.a = self.data_info()
        self.disr_num = 491
        self.safe_num = 680
        self.c1 = np.load('/home/jack/音乐/data/data5/Selfc/selfc16_7_3.npz')['selfc']
        #self.c2 = np.load('/home/jack/音乐/data/data5/Selfc/selfc16_7_12.npz')['selfc']
        #self.c3 = np.load('/home/jack/音乐/data/data5/Selfc/selfc16_12_12.npz')['selfc']
        self.test_shut = np.load('/home/jack/音乐/data/data5/test16.npz')['test_shut']
        self.b = np.load('/home/jack/音乐/data/data5/self_b_test16.npz')['self_b_test']
        self.neural_struct = []
        for i in range(3,13):
            for j in range(3,13):
                self.neural_struct.append((i,j))

    @staticmethod
    def data_info():
        sheet = xlrd.open_workbook(r'/home/jack/公共的/excel_data/info.xlsx')
        sheet1 = sheet.sheet_by_name('Sheet1')
        row_num,col_num = sheet1.nrows,sheet1.ncols
        A = []
        for i in range(row_num):
            A.append(sheet1.row_values(i))
        return np.array(A)

    #这里的i是测试炮序列的索引
    def Draw(self,test_mat,pred_mat,all_test_shut,k,best_st):
        i = list(all_test_shut).index(k)
        self.c1 = np.load('/home/jack/音乐/data/data5/Selfc/selfc16_%d_%d.npz'%self.neural_struct[best_st])['selfc']
        real_shut = self.a[k,0]
        t = test_mat[-1,1000*i:1000*(i+1)] #t为时间
        y1 = pred_mat[best_st,1000*i:1000*(i+1)]  #y1,y2,y3分别为神经网络结构为3-3,7-12,12-12下的预测结果
        #y2 = pred_mat[49,1000*i:1000*(i+1)]; y.append(y2)
        #y3 = pred_mat[99,1000*i:1000*(i+1)]; y.append(y3)
        y_real = test_mat[1,1000*i:1000*(i+1)]   # y_real为真实的密度值
        y_max = test_mat[-2,1000*i:1000*(i+1)]   # y_max为每炮最大密度值
        dis_index = np.where(np.trunc(t*1000)==int(self.c1[k,1]*1000))[0][0]
        Y = y_real[dis_index]  #Y为真实密度曲线在实际破裂时刻的值
        #selfc = []
        #selfc.append(self.c1[k,3])
        #selfc为st_index神经网络模型下预测出来的破裂时刻
        #selfc.append(self.c2[i,3])
        #selfc.append(self.c3[i,3])
        #Y_l = []
        dis_index1 = np.where(np.trunc(t*1000)==int(self.c1[k,3]*1000))[0][0]
        #dis_index2 = np.where(np.trunc(t*1000)==int(self.c2[i,3]*1000))[0][0]
        #dis_index3 = np.where(np.trunc(t*1000)==int(self.c3[i,3]*1000))[0][0]

        Y1 = y1[dis_index1]
        #Y_l.append(Y1)
        #Y1为st_index神经结构下预测出来的破裂时刻对应的密度预测值
        #Y2 = y2[dis_index2]; Y_l.append(Y2)
        #Y3 = y3[dis_index3]; Y_l.append(Y3)

        #struct = [self.neural_struct[st_index],(7,12),(12,12)]
        fig, axs = plt.subplots(1,1,sharex=True,figsize=(8,6))
        #for j in range(1):
        #j=0
        axs.plot(t,y_real,color='k',linestyle='-',linewidth=1,label='real_density')
        axs.plot(t,y_max,color='r',linestyle='--',linewidth=1,label='desire NN output')
        axs.plot(t,y1,color='b',linestyle='--',linewidth=1,label='predict density')
        axs.axvline(x=self.c1[k,1],ls=':',color='firebrick',label='real disr_time')   #真实破裂时刻处的竖线
        axs.axvline(x=self.c1[k,3],ls=':',color='g',label='pred_disr time')            #预测出的破裂时刻处的竖线
        axs.legend(loc='best',shadow=True)
        axs.annotate(r'$t_{real\_disr}$',xy=(self.b[i,1],Y),xycoords='data',xytext=(-30,-30),textcoords='offset points',\
           arrowprops=dict(facecolor='fuchsia',arrowstyle='->',connectionstyle='arc3'))
        axs.annotate(r'$t_{pred\_disr}$',xy=(self.c1[k,3],Y1),xycoords='data',xytext=(-30,-30),textcoords='offset points',\
           arrowprops=dict(facecolor='fuchsia',arrowstyle='->',connectionstyle='arc3'))
        axs.set_xlabel('time(s)')
        axs.set_ylabel('normal density')
        axs.set_title('result under (%d,%d)'%self.neural_struct[best_st])
        axs.set_title(r'$t_{pred\_disr}$=%.3f'%self.c1[k,3],loc='left')
        axs.set_title(r'$t_{real\_disr}$=%.3f'%self.b[k,1],loc='right')
        #fig.subplots_adjust(hspace=0.8)
        plt.suptitle("predict result of shut %d under (%d,%d) struct"%\
                     (real_shut,self.neural_struct[best_st][0],self.neural_struct[best_st][1]))
        plt.savefig('/home/jack/音乐/data/data5/picture/%d.jpg'%real_shut,format='jpg',dpi=1500)
        plt.show()

    def find_min(self,pred_mat,test_mat,st_index):
        min_dist = []
        for i in range(self.a.shape[0]):
            sub = pred_mat[st_index,1000*i:1000*(i+1)]-test_mat[1,1000*i:1000*(i+1)]
            min_dist.append(sub.min())
        fig = plt.figure(figsize=(8,6))
        ax = fig.subplots()
        ax.set_xlabel('shut index',fontsize=20)
        ax.set_ylabel(r'$\Delta{n_{e}}^{alarm}$',fontsize=20)
        ax.scatter(range(0,491),min_dist[:491],c='r',label='disruption',marker='^')
        #ax.set_ylim(-0.3,0.25)
        ax.scatter(range(491,1171),min_dist[491:1171],c='g',label='non-disruptive',marker='*',s=50)
        ax.legend(loc='best')
        plt.suptitle(r'$\Delta{n_{e}}^{alarm}$ of all test shut',fontsize=20)
        plt.savefig('/home/jack/音乐/data/data5/picture/result.jpg',format='jpg',dpi=1500)

#def main():
pred_mat = np.load('/home/jack/音乐/data/data5/predict_y16.npz')['predicty']
#pred_mat是100个神经网络下的预测结果，我们这里选取三个来画图，0,49,99
test_mat = np.load('/home/jack/音乐/data/data5/test16.npz')['test_set']
all_test_shut = np.load('/home/jack/音乐/data/data5/test16.npz')['test_shut']
pred_resu = np.load('/home/jack/音乐/data/data5/predict_res16.npz')['predict_res']
pred_rate = np.load('/home/jack/音乐/data/data5/predict_rate16.npz')['predict_rate']
best_st = list(pred_rate[:,2]).index(pred_rate[:,2].min())
test = Draw_Re()
test.c1 = np.load('/home/jack/音乐/data/data5/Selfc/selfc16_%d_%d.npz'%test.neural_struct[best_st])['selfc']
print("最好的模型是第%d个，结构为:(%d,%d)"%(best_st,test.neural_struct[best_st][0],test.neural_struct[best_st][1]))
print("最好的预测结果为:[%d,%d,%d,%d,%d,%d,%d]"%tuple(pred_resu[best_st]))
print("最好的预测概率值为:[%d,%f,%f,%f,%f]"%tuple(pred_rate[best_st]))
    #test.c1 = np.load('/home/jack/音乐/data/data5/Selfc/selfc16_%d_%d.npz'%test.neural_struct[best_st])['selfc']
    #test_mat是测试集
res_snd = [x for x in all_test_shut if test.c1[x,5]==-1]
res_fnd = [x for x in all_test_shut if test.c1[x,5]==0]
res_sd = [x for x in all_test_shut if test.c1[x,5]==1]
res_ld = [x for x in all_test_shut if test.c1[x,5]==2]
res_pd = [x for x in all_test_shut if test.c1[x,5]==3]
res_fd = [x for x in all_test_shut if test.c1[x,5]==4]


for k in res_ld:
    test.Draw(test_mat,pred_mat,all_test_shut,k,best_st)
test.find_min(pred_mat,test_mat,best_st)
#all_test_shut = np.load('/home/jack/音乐/data/data5/test16.npz')['test_shut']
#if __name__=='__main__':
#    main()
'''
def main():
    pred_mat = np.load('/home/jack/音乐/data/data3/predict13.npz')['predict']
    #pred_mat是100个神经网络下的预测结果，我们这里选取三个来画图，0,49,99
    test_mat = np.load('/home/jack/音乐/data/data3/test13.npz')['test_set']
    #test_mat是测试集
    draw_resu = Draw_Re()
    for i in range(454,455):
        draw_resu.Draw(test_mat,pred_mat,i)

    draw_resu.find_min(pred_mat,test_mat)
if __name__=='__main__':
    main()

'''
