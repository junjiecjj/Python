#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
此函数是利用本地(已经下载到本地)数据画出某炮的所有信号图，并在破裂时刻处画出一条竖直虚线.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
from matplotlib.font_manager import FontProperties

home = os.environ['HOME']
datapath = home + '/density/'
infopath = home + '/数据筛选/'
resultpath = home + '/result/'
fontpath = "/usr/share/fonts/truetype/msttcorefonts/"
font = FontProperties(fname=fontpath+"Times_New_Roman.ttf", size = 12)#fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font1 = FontProperties(fname=fontpath+"Times_New_Roman.ttf", size = 16)

class ViewData(object):
    def __init__(self):
        self.signal_kind = 8
        self.num = 1000
        self.A = np.array(pd.read_csv(infopath+'last5.csv'))             # self.a中的破裂时刻是人工从电子密度和电流信号上观察到的
        
    def readd(self,I,signal):
        data = np.load(datapath+'%d.npz'%(self.A[I,0]))[signal]
        return data
    
    def smooth(self,amino):
        if abs(amino[1,0]-0.45)>0.1:
            amino[1,0] = 0.45
        for i in range(amino.shape[1]-1):
            if abs(amino[1,i+1]-amino[1,i]) > 0.02:
                amino[1,i+1] = amino[1,i]
        for i in range(amino.shape[1]):
            if abs(amino[1,i]-0.45) > 0.05:
                amino[1,i] = amino[1,i-1]
        return amino
    def see(self,i):
        signal = ['pcrl01','dfsdev','vp1','sxr23d','pxuv30','pxuv18','kmp13t','pbrem10']
        labels = [r'$I_{P}(10^{5}A)$',r'$n_{e}(10^{19}m^{-3})$',r'$V_{p}$(V)','SXR(V)',\
                  r'$p_{xuv1}$(V)',r'$p_{xuv2}$(V)',r'$B_{\theta}$(V)',r'$p_{brem}$(V)']
        title = ['a)','b)','c)','d)','e)','f)','g)','h)']
        res_l= []
        arry_l = []
        pool = Pool(4)
        for j in range(self.signal_kind):
            res = pool.apply_async(self.readd,(i,signal[j],))
            res_l.append(res)
        pool.close()
        pool.join()

        for res in res_l:
            arry_l.append(res.get())
        density = arry_l[1]
        a = np.where(density[1]==density[1].max())[0][0]
        print(a)
        inde = density[1][np.where(density[1]==density[1].max())][0]
        fig,axs = plt.subplots(self.signal_kind,1,sharex=True,figsize=(6,12))

        for j in range(self.signal_kind):
            if j == 0:
                arry_l[j][1,:] = arry_l[j][1,:]/100000
            axs[j].plot(arry_l[j][0,:],arry_l[j][1,:],linewidth=1)
            if self.A[i,4] == 1 and j == 0:
                Label = 'disruption'
                axs[j].axvline(x=self.A[i,7],ls='--',color='r',label = Label,)#,label = '破裂'
            else:
                 axs[j].axvline(x=self.A[i,7],ls='--',color='r')
            if j==7:
                 axs[j].set_xlabel('time(s)',fontproperties = font1)
            axs[j].set_ylabel('%s'%labels[j],fontproperties = font)
            axs[j].set_title(title[j], loc = 'left',fontproperties = font1)
            axs[j].tick_params(labelsize=16)
            Labels = axs[j].get_xticklabels() + axs[j].get_yticklabels()
            [label.set_fontname('Times New Roman') for label in Labels]
        #axs[0].legend(prop=font,loc='upper right',borderaxespad=0,edgecolor='black',)
        
        if self.A[i,4] == 1:
            Ip = np.load(datapath+'%d.npz'%(self.A[i,0]))['pcrl01']
            ne = np.load(datapath+'%d.npz'%(self.A[i,0]))['dfsdev']
            aminor = np.load(datapath+'%d.npz'%(self.A[i,0]))['aminor']
            flat_l = int((self.A[i,7] - self.A[i,5])/0.001)
            print("flat_l:",flat_l)
            a1 = np.where(np.around(Ip[0]*self.num)==int(self.A[i,7]*self.num))[0][0]
            Ip1 = Ip[:,a1-flat_l:a1+1]
            #a2 = np.where(np.around(ne[0]*self.num)==int(self.A[i,7]*self.num))[0][0]
            #ne1 = ne[:,a1-999:a2+1]
            a3 = np.where(np.around(aminor[0]*self.num)==int(self.A[i,7]*self.num))[0][0]
            aminor1 = aminor[:,a3-flat_l:a3+1]
            aminor1 = self.smooth(aminor1)
            Ngw = Ip1[1]*(10**-6)*10/(np.pi*aminor1[1]**2)
            axs[1].plot(aminor1[0],Ngw,'lime',label = 'GreenWald limit')
        if self.A[i,4] == 1:
             axs[0].legend(prop=font,loc='best',borderaxespad=0,edgecolor='black',)
             axs[1].legend(prop=font,loc='lower right',borderaxespad=0,edgecolor='black',)
        fig.subplots_adjust(hspace=0.3,wspace=0.1)#调节两个子图间的距离
        if self.A[i,4] == 1:
            plt.suptitle('density limit disruptive pulse:%d'%self.A[i,0],fontproperties = font1,x=0.5,y = 0.92)
        else:
            plt.suptitle('nondisruptive pulse:%d'%self.A[i,0],fontproperties = font1,x=0.5,y = 0.92)
        fig.subplots_adjust(hspace=0.4)#调节两个子图间的距离
        #plt.savefig(filepath2+'%d.svg'%self.a[i,0],format='svg',dpi=1000)
        #plt.savefig(resultpath +'%d.eps'%self.A[i,0],format='eps',dpi=1000,bbox_inches = 'tight')
        plt.show()


def main():
    test = ViewData( )
    test.see(1011)#340,1430
    test.see(1566)
    #for i in range(899,test.a.shape[0]):
    #    test.see(i)
     



#A = data_info()

if __name__=='__main__':
    main()
