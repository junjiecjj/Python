#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
此函数是画出某炮的四种信号图，并在最大密度处画出一条竖直虚线.
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import xlrd
import os
from matplotlib.font_manager import FontProperties
font = FontProperties( size=10) #fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font1 = FontProperties(size=16)

class ViewData(object):
    def __init__(self):
        self.signal_kind = 4
        self.a = self.data_info()               # self.a中的破裂时刻是人工从电子密度和电流信号上观察到的

    @staticmethod
    def data_info():
        sheet=xlrd.open_workbook(r'/home/jack/公共的/excel_data/info.xlsx')
        sheet1 = sheet.sheet_by_name('Sheet1')
        row_num, col_num = sheet1.nrows,sheet1.ncols
        A=[]
        for i in range(row_num):
            A.append(sheet1.row_values(i))
        return np.array(A)

    def see(self,i):
        signal = ['PCRL','DFSDEV','SXR23D','VP1']
        labels = ['plasma current'+r'($(10^{5}A)$)','density'+r'($(10^{19}m^{-3})$)','soft X ray(V)','loop voltage(V)']
        title = ['a)','b)','c)','d)']
        res_l= []
        arry_l = []
        pool = Pool(4)
        for j in range(self.signal_kind):
            filel = '/home/jack/data/%d/%d_%s.txt'%(self.a[i,0],self.a[i,0],signal[j])
            res = pool.apply_async(np.loadtxt,(filel,))
            res_l.append(res)
        pool.close()
        pool.join()

        for res in res_l:
            arry_l.append(res.get())
        density = arry_l[1]
        a = np.where(density[0]==density[0].max())[0][0]
        print(a)
        inde = density[1][np.where(density[0]==density[0].max())][0]
        fig,axs = plt.subplots(4,1,sharex=True,figsize=(6,8))

        for j in range(4):
            if j == 0:
                arry_l[j][0,:] = arry_l[j][0,:]/100000
            axs[j].plot(arry_l[j][1,:],arry_l[j][0,:],linewidth=1)
            if self.a[i,2] == -1:
                Label = 'end of experiment'
            else:
                Label = 'disruption'
            axs[j].axvline(x=8,ls='--',color='r',label = Label,)#,label = '破裂'
            axs[j].set_xlabel('time(s)',fontproperties = font)
            axs[j].set_ylabel('%s'%labels[j],fontproperties = font)
            axs[j].set_title(title[j], loc = 'left')
        #axs[0].legend(prop=font,loc='upper right',borderaxespad=0,edgecolor='black',)
        axs[0].legend(prop=font,loc='best',borderaxespad=0,edgecolor='black',)
        fig.subplots_adjust(hspace=0.3,wspace=0.1)#调节两个子图间的距离
        if self.a[i,2]==1:
            plt.suptitle('density limit disruption,pulse:%d'%self.a[i,0],fontproperties = font1)
        else:
            plt.suptitle('safe pulse:%d'%self.a[i,0],fontproperties = font1)
        fig.subplots_adjust(hspace=0.4)#调节两个子图间的距离
        plt.savefig(filepath2+'%d.svg'%self.a[i,0],format='svg',dpi=1000)
        plt.savefig(filepath2+'%d.eps'%self.a[i,0],format='eps',dpi=1000)
        plt.show()


def main():
    test = ViewData( )
    test.see(948)#95,948
    test.see(95)
    #for i in range(899,test.a.shape[0]):
    #    test.see(i)
     
filepath2 = '/home/jack/snap/picture/englishpicture/'
if os.path.exists(filepath2):
    pass
else:
    os.makedirs(filepath2)

#A = data_info()

if __name__=='__main__':
    main()
