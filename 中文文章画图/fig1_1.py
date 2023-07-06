#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
此函数是利用本地(已经下载到本地)数据画出某炮的所有信号图，并在破裂时刻处画出一条竖直虚线.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import tick_params
import os
from matplotlib.patches import ConnectionPatch
from matplotlib.font_manager import FontProperties

home = os.environ['HOME']
datapath = home + '/Density/'
infopath = home + '/数据筛选/'
resultpath = home + '/Resultpicture/'
fontpath = "/usr/share/fonts/truetype/msttcorefonts/"
font = FontProperties(fname=fontpath+"Times_New_Roman.ttf", size = 10)#fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font1 = FontProperties(fname=fontpath+"Times_New_Roman.ttf", size = 14)
font2 = FontProperties(fname=fontpath+"Times_New_Roman.ttf", size = 24)

fontt = {'family':'Times New Roman','style':'normal','size':20}
fontt1 = {'style':'normal','weight':'bold','size':20}
fontt2 = {'style':'normal','size':20}
fontt3 = {'style':'normal','weight':'bold','size':24}

#'family':'Times New Roman',

class ViewData(object):
    def __init__(self):
        self.signal_kind = 4
        self.num = 1000
        self.A = np.array(pd.read_csv(infopath+'last7.csv'))             # self.a中的破裂时刻是人工从电子密度和电流信号上观察到的

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
        signal = ['pcrl01','dfsdev','sxr23d','kmp13t']
        labels = [r'$I_{P}(10^{5}$A)',r'$n_{e}(10^{19}m^{-3})$','SXR(V)',\
                  r'$B_{\theta}$(V)']
        label = [r'$I_{P}$',r'$n_{e}$',r'SXR',\
                  r'$B_{\theta}$']
        title = ['a)','b)','c)','d)']

        Ip = np.load(datapath+'%d.npz'%(self.A[i,0]))['pcrl01']
        ne = np.load(datapath+'%d.npz'%(self.A[i,0]))['dfsdev']
        aminor = np.load(datapath+'%d.npz'%(self.A[i,0]))['aminor']
        sxr = np.load(datapath+'%d.npz'%(self.A[i,0]))['sxr23d']
        Bt = np.load(datapath+'%d.npz'%(self.A[i,0]))['kmp13t']
        flat_l = int((self.A[i,7] - self.A[i,5])/0.001)
        print("flat_l:",flat_l)

        a1 = np.where(np.around(Ip[0]*self.num)==np.around(self.A[i,7]*self.num))[0][0]
        Ip1 = Ip[:,a1-flat_l:a1+1]

        a2 = np.where(np.around(ne[0]*self.num)==np.around(self.A[i,7]*self.num))[0][0]
        ne1 = ne[:,a2-flat_l:a2+1]

        a3 = np.where(np.around(aminor[0]*self.num)==np.around(self.A[i,7]*self.num))[0][0]
        aminor1 = aminor[:,a3-flat_l:a3+1]
        aminor1 = self.smooth(aminor1)

        Ngw = Ip1[1]*(10**-6)*10/(np.pi * aminor1[1]**2)
        R = ne1[1]/Ngw
        print(aminor1.shape, Ngw.shape)
        aa = [i for i in range(len(Ngw)-1) if (R[i]<=0.8 and R[i+1]>=0.8)]
        #print("aa=",aa)
        X = aminor1[0][aa[0]]
        #print("x = ",x)


        fig,axs = plt.subplots(self.signal_kind,2,sharex='col',figsize=(14,10),)#sharex=True,

        tick_params(direction='in')

        axs[0,0].tick_params(direction='in')
        axs[0,0].plot(Ip[0],Ip[1]/100000,linewidth=2,label=label[0])
        axs[0,0].axvline(x=self.A[i,7],ls='--',color='k',label='disruption',linewidth=2,)
        axs[0,0].axvline(x=X,ls='--',color='r',linewidth=2,)
        axs[0,0].set_ylabel('%s'%labels[0],fontdict=fontt2,)
        axs[0,0].set_title(title[0],loc='left',fontdict=fontt1)
        #axs[0].text(0.5,0.8,r'$I_P$',fontdict=fontt1)
        Ip_max = max(Ip[1]/100000)
        Ip_min = min(Ip[1]/100000)
        axs[0,0].set_ylim(Ip_min-0.5,Ip_max+0.5)
        legend1 = axs[0,0].legend(prop=fontt,loc='lower left',borderaxespad=0,edgecolor='black')
        frame1 = legend1.get_frame() 
        frame1.set_alpha(1) 
        frame1.set_facecolor('none') # 设置图例legend背景透明

        axs[0,0].tick_params(labelsize=16,width=3)
        Labels = axs[0,0].get_xticklabels() + axs[0,0].get_yticklabels()
        [label.set_fontname('Times New Roman') for label in Labels]
        [label.set_fontsize(24) for label in Labels]#刻度值字号
        
        axs[0,0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
        axs[0,0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
        axs[0,0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
        axs[0,0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

        x0 = X-0.2
        x1 = self.A[i,7]+0.2
        y0 = 3.7
        y1 = 4.3#Ip[1][aa[0]]+1
        a = np.where(np.around(Ip[0]*1000)==np.around(x0*1000))[0][0]
        b = np.where(np.around(Ip[0]*1000)==np.around(x1*1000))[0][0]
        y0 = Ip[1][a:b+1].min()/100000
        y1 = Ip[1][a:b+1].max()/100000
        sx = [x0,x1,x1,x0,x0]
        sy = [y0,y0,y1,y1,y0]
        axs[0,0].plot(sx,sy,"purple",linewidth=2,)

        axs[0,1].tick_params(direction='in')
        axs[0,1].axvline(x=self.A[i,7]+0.026,ls='--',color='k',label='disruption',linewidth=2,)
        axs[0,1].axvline(x=X,ls='--',color='r',linewidth=2,)
        axs[0,1].plot(Ip[0],Ip[1]/100000,linewidth=2,label=label[0])
        #axs[0,1].set_ylabel('%s'%labels[0],fontdict=fontt2,)
        axs[0,1].set_title('(e)',loc='left',fontdict=fontt1)


        axs[0,1].tick_params(labelsize=16,width=3)
        Labels = axs[0,1].get_xticklabels() + axs[0,1].get_yticklabels()
        [label.set_fontname('Times New Roman') for label in Labels]
        [label.set_fontsize(24) for label in Labels]#刻度值字号
        
        axs[0,1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
        axs[0,1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
        axs[0,1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
        axs[0,1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


        axs[0,1].set_xlim([X-0.2, self.A[i,7]+0.2])

        xy=(self.A[i,7],y1)
        xy2 = (x0,y1)
        con = ConnectionPatch(xyA=xy,xyB=xy2,coordsA="data",coordsB="data",arrowstyle='->',\
                axesA=axs[0,0],axesB=axs[0,1])
        axs[0,0].add_artist(con)

        xy=(self.A[i,7],y0)
        xy2 = (x0,y0)
        con = ConnectionPatch(xyA=xy,xyB=xy2,coordsA="data",coordsB="data",arrowstyle='->',\
                axesA=axs[0,0],axesB=axs[0,1])
        axs[0,0].add_artist(con)

        axs[1,0].tick_params(direction='in')
        axs[1,0].plot(ne[0], ne[1], linewidth = 2, label = label[1])
        axs[1,0].axvline(x=self.A[i,7], ls='--', color='k',linewidth = 2,)
        axs[1,0].axvline(x=X,ls='--',color='r',label=r'$\frac{n_e}{n_{GW}}$=0.8',linewidth = 2,)
        axs[1,0].set_ylabel('%s'%labels[1], fontdict=fontt2)
        axs[1,0].set_title(title[1], loc = 'left',fontdict=fontt1)
        ne_max = max(ne[1])
        ne_min = min(ne[1])
        axs[1,0].set_ylim(ne_min-0.7,ne_max+0.7)

        axs[1,0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
        axs[1,0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
        axs[1,0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
        axs[1,0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

        # 设置坐标刻度值的大小以及刻度值的字体
        axs[1,0].tick_params(labelsize=20,width=3)
        Labels = axs[1,0].get_xticklabels() + axs[1,0].get_yticklabels()
        [label.set_fontname('Times New Roman') for label in Labels]
        [label.set_fontsize(24) for label in Labels]#刻度值字号
        
        x0 = X-0.2
        x1 = self.A[i,7]+0.2

        a = np.where(np.around(ne[0]*1000)==np.around(x0*1000))[0][0]
        b = np.where(np.around(ne[0]*1000)==np.around(self.A[i,7]*1000))[0][0]
        y0 = ne[1][a:b+1].min()
        y1 = ne[1][a:b+1].max()
        sx = [x0,x1,x1,x0,x0]
        sy = [y0,y0,y1,y1,y0]
        axs[1,0].plot(sx,sy,"purple",linewidth=2,)

        axs[1,1].tick_params(direction='in')
        axs[1,1].plot(ne[0], ne[1], linewidth=2,label=label[1])
        axs[1,1].axvline(x=self.A[i,7]+0.026,ls='--',color='k',label='disruption',linewidth=2,)
        axs[1,1].axvline(x=X,ls='--',color='r',linewidth=2,)
        #axs[1,1].set_ylabel('%s'%labels[1],fontdict=fontt2,)
        axs[1,1].set_title('(f)',loc='left',fontdict=fontt1)

        axs[1,1].tick_params(labelsize=16,width=3)
        Labels = axs[1,1].get_xticklabels() + axs[1,1].get_yticklabels()
        [label.set_fontname('Times New Roman') for label in Labels]
        [label.set_fontsize(24) for label in Labels]#刻度值字号
        
        axs[1,1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
        axs[1,1].spines['left'].set_linewidth(1.5);  ####设置左边坐标轴的粗细
        axs[1,1].spines['right'].set_linewidth(1.5); ###设置右边坐标轴的粗细
        axs[1,1].spines['top'].set_linewidth(1.5);   ####设置上部坐标轴的粗细

        axs[1,1].set_xlim([X-0.2, self.A[i,7]+0.2])
        axs[1,1].set_ylim([y0-0.2,y1+0.2])

        xy=(self.A[i,7],y1)
        xy2 = (x0,y1)
        con = ConnectionPatch(xyA=xy,xyB=xy2,coordsA="data",coordsB="data",arrowstyle='->',\
                axesA=axs[1,0],axesB=axs[1,1])
        axs[1,0].add_artist(con)

        xy=(self.A[i,7],y0)
        xy2 = (x0,y0)
        con = ConnectionPatch(xyA=xy,xyB=xy2,coordsA="data",coordsB="data",arrowstyle='->',\
                axesA=axs[1,0],axesB=axs[1,1])
        axs[1,0].add_artist(con)


        axs[2,0].tick_params(direction='in')
        axs[2,0].plot(sxr[0],sxr[1], linewidth = 2,)
        axs[2,0].axvline(x=X,ls='--',color='r',linewidth = 2,)
        axs[2,0].axvline(x=self.A[i,7],ls='--',color='k',linewidth = 2,)
        axs[2,0].set_ylabel('%s'%labels[2], fontdict=fontt2)
        axs[2,0].set_title('(c)', loc = 'left',fontdict=fontt1)
        #axs[2,0].legend(prop=fontt,loc='best',borderaxespad=0,edgecolor='black')

        axs[2,0].tick_params(labelsize=16,width=3)
        Labels = axs[2,0].get_xticklabels() + axs[2,0].get_yticklabels()
        [label.set_fontname('Times New Roman') for label in Labels]
        [label.set_fontsize(24) for label in Labels]#刻度值字号
        
        axs[2,0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
        axs[2,0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
        axs[2,0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
        axs[2,0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


        x0 = X-0.2
        x1 = self.A[i,7]+0.2
        a = np.where(np.around(sxr[0]*1000)==np.around(x0*1000))[0][0]
        b = np.where(np.around(sxr[0]*1000)==np.around(x1*1000))[0][0]
        y0 = sxr[1][a:b+1].min()
        y1 = sxr[1][a:b+1].max()
        #y0 = 0.1
        #y1 = 0.26#Ip[1][aa[0]]+1
        sx = [x0,x1,x1,x0,x0]
        sy = [y0,y0,y1,y1,y0]
        axs[2,0].plot(sx,sy,"purple",linewidth=2,)

        axs[2,1].tick_params(direction='in')
        axs[2,1].plot(sxr[0],sxr[1],linewidth=2,label=label[0])
        axs[2,1].axvline(x=self.A[i,7]+0.026,ls='--',color='k',label='disruption',linewidth=2,)
        axs[2,1].axvline(x=X,ls='--',color='r',linewidth=2,)
        #axs[2,1].set_ylabel('%s'%labels[2],fontdict=fontt2,)
        axs[2,1].set_title('(g)',loc='left',fontdict=fontt1)


        axs[2,1].tick_params(labelsize=16,width=3)
        Labels = axs[2,1].get_xticklabels() + axs[2,1].get_yticklabels()
        [label.set_fontname('Times New Roman') for label in Labels]
        [label.set_fontsize(24) for label in Labels]#刻度值字号
        
        axs[2,1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
        axs[2,1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
        axs[2,1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
        axs[2,1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


        axs[2,1].set_xlim([X-0.2, self.A[i,7]+0.2])

        xy=(self.A[i,7],y1)
        xy2 = (x0,y1)
        con = ConnectionPatch(xyA=xy,xyB=xy2,coordsA="data",coordsB="data",arrowstyle='->',\
                axesA=axs[2,0],axesB=axs[2,1])
        axs[2,0].add_artist(con)

        xy=(self.A[i,7],y0)
        xy2 = (x0,y0)
        con = ConnectionPatch(xyA=xy,xyB=xy2,coordsA="data",coordsB="data",arrowstyle='->',\
                axesA=axs[2,0],axesB=axs[2,1])
        axs[2,0].add_artist(con)


        axs[3,0].tick_params(direction='in')
        axs[3,0].plot(Bt[0], Bt[1], label=label[3],linewidth = 2,)
        axs[3,0].axvline(x=X,ls='--',color='r',linewidth = 2,)
        axs[3,0].axvline(x=self.A[i,7],ls='--',color='k',linewidth = 2,)
        axs[3,0].set_ylabel('%s'%labels[3],fontdict=fontt2)
        axs[3,0].set_xlabel('time(s)',fontdict=fontt3)
        axs[3,0].set_title(title[3], loc = 'left',fontdict=fontt1)
        axs[3,0].legend(prop=fontt,loc='lower left',borderaxespad=0,edgecolor='black')


        axs[3,0].tick_params(labelsize=20,width=3)#刻度宽度
        Labels = axs[3,0].get_xticklabels() + axs[3,0].get_yticklabels()
        [label.set_fontname('Times New Roman') for label in Labels]#刻度值字体
        [label.set_fontsize(24) for label in Labels]#刻度值字号

        axs[3,0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
        axs[3,0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
        axs[3,0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
        axs[3,0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


        x0 = X-0.2
        x1 = self.A[i,7]+0.2

        a = np.where(np.around(Bt[0]*1000)==np.around(x0*1000))[0][0]
        b = np.where(np.around(Bt[0]*1000)==np.around(x1*1000))[0][0]
        y0 = Bt[1][a:b+1].min()
        y1 = Bt[1][a:b+1].max()
        sx = [x0,x1,x1,x0,x0]
        sy = [y0,y0,y1,y1,y0]
        axs[3,0].plot(sx,sy,"purple",linewidth=2,)


        axs[3,1].tick_params(direction='in')
        axs[3,1].plot(Bt[0], Bt[1], linewidth=2,label=label[1])
        axs[3,1].axvline(x=self.A[i,7]+0.026,ls='--',color='k',label='disruption',linewidth=2,)
        axs[3,1].axvline(x=X,ls='--',color='r',linewidth=2,)
        #axs[3,1].set_ylabel('%s'%labels[3],fontdict=fontt2,)
        axs[3,1].set_title('(h)',loc='left',fontdict=fontt1)

        axs[3,1].tick_params(labelsize=16,width=3)
        Labels = axs[3,1].get_xticklabels() + axs[3,1].get_yticklabels()
        [label.set_fontname('Times New Roman') for label in Labels]
        [label.set_fontsize(24) for label in Labels]#刻度值字号
        
        axs[3,1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
        axs[3,1].spines['left'].set_linewidth(1.5);  ####设置左边坐标轴的粗细
        axs[3,1].spines['right'].set_linewidth(1.5); ###设置右边坐标轴的粗细
        axs[3,1].spines['top'].set_linewidth(1.5);   ####设置上部坐标轴的粗细


        axs[3,1].set_xlim([X-0.2, self.A[i,7]+0.2])
        axs[3,1].set_ylim([y0-0.2,y1+0.2])
        axs[3,1].set_xlabel('time(s)',fontdict=fontt3)

        xy=(self.A[i,7],y1)
        xy2 = (x0,y1)
        con = ConnectionPatch(xyA=xy,xyB=xy2,coordsA="data",coordsB="data",arrowstyle='->',\
                axesA=axs[3,0],axesB=axs[3,1])
        axs[3,0].add_artist(con)

        xy=(self.A[i,7],y0)
        xy2 = (x0,y0)
        con = ConnectionPatch(xyA=xy,xyB=xy2,coordsA="data",coordsB="data",arrowstyle='->',\
                axesA=axs[3,0],axesB=axs[3,1])
        axs[3,0].add_artist(con)


        axs[1,0].plot(aminor1[0],Ngw,'red',label = 'GreenWald limit',linewidth = 2,)
        #axs[1,0].legend(prop=fontt,loc='lower left',borderaxespad=0,edgecolor='black')
        legend1 = axs[1,0].legend(prop=fontt,loc='lower left',borderaxespad=0,edgecolor='black')
        frame1 = legend1.get_frame()
        frame1.set_alpha(1)
        frame1.set_facecolor('none') # 设置图例legend背景透明

        fig.subplots_adjust(hspace=0.3,wspace=0.2)#调节两个子图间的距离
        if self.A[i,4] == 1:
            plt.suptitle('Density limit disruptive pulse:%d'%self.A[i,0],fontsize=22,fontweight='bold',x=0.5,y = 0.96)
        else:
            plt.suptitle('Nondisruptive pulse:%d'%self.A[i,0],fontsize=22,fontweight='bold',x=0.5,y = 0.96)

        #plt.savefig(filepath2+'%d.svg'%self.a[i,0],format='svg',dpi=1000)
        plt.savefig(resultpath +'fig1.eps'%self.A[i,0],format='eps',dpi=1000,bbox_inches = 'tight')
        plt.show()


def main():
    test = ViewData()
    #for i in [340,956,957,958,959,960,961,962,963,964,965]:
    #    test.see(i)#340,1430
    test.see(338)
    #for i in range(899,test.a.shape[0]):
    #    test.see(i)




#A = data_info()

if __name__=='__main__':
    main()
