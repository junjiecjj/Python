#!/usr/bin/env python3
#!-*-coding=utf-8-*-
#########################################################################
# File Name: viewdata.py
# Author: chenjunjie
# mail: 2716705056@qq.com
# Created Time: 2018.09.02
#########################################################################
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing
import xlrd

class ViewData(object):
    def __init__(self,sign_kind):
        self.signal_kind = sign_kind
        self.a = self.data_info()               # self.a中的破裂时刻是人工从电子密度和电流信号上观察到的

    @staticmethod
    def data_info():
        sheet=xlrd.open_workbook(r'/home/jack/公共的/excel_data/info.xlsx')
        sheet1 = sheet.sheet_by_name('Sheet1')
        row_num,col_num = sheet1.nrows,sheet1.ncols
        A=[]
        for i in range(row_num):
            A.append(sheet1.row_values(i))
        return np.array(A)

    def see(self,i):
        signal = ['PCRL','DFSDEV','LMSZ','KMP13T','IC1','SXR23D','PXUV30','PXUV18','VP1','BETAP','LI','q']
        res_l= []
        arry_l = []
        pool = Pool(multiprocessing.cpu_count())
        for j in range(self.signal_kind):
            filel = '/home/jack/data/%d/%d_%s.txt'%(self.a[i,0],self.a[i,0],signal[j])
            res = pool.apply_async(np.loadtxt,(filel,))
            res_l.append(res)
        pool.close()
        pool.join()

        for res in res_l:
            arry_l.append(res.get())

        fig,axs = plt.subplots(6,2,sharex=True,figsize=(10,10))

        for j in range(6):
            for k in range(2):
                axs[j,k].plot(arry_l[2*j+k][1,:],arry_l[2*j+k][0,:],linewidth=1)
                axs[j,k].set_xlabel('time(s)')
                axs[j,k].set_ylabel('%s'%signal[2*j+k])

        fig.subplots_adjust(hspace=0.8,wspace=0.4)#调节两个子图间的距离
        plt.suptitle('shut:%d'%self.a[i,0])
        plt.savefig('/home/jack/图片/破裂图片/%d.jpg'%self.a[i,0],format='jpg',dpi=1000)
        plt.show()


def main():
    test = ViewData(12)
    test.see(1117)
    #for i in range(test.a.shape[0]):
     #   test.see(i)

if __name__=='__main__':
    main()
