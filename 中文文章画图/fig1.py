
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
from matplotlib.font_manager import FontProperties

home = os.environ['HOME']
datapath = home + '/Density/'
infopath = home + '/数据筛选/'
resultpath = home + '/tmp/'

fontpath = "/usr/share/fonts/truetype/arphic/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"

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
        labels = [r'电流($10^{5}$A)',r'密度($10^{19}m^{-3}$)','软X射线(V)',\
                  r'磁扰动(V)']
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

        fig,axs = plt.subplots(self.signal_kind,1,sharex=True,figsize=(7,8))#sharex=True,
        
        
        axs[0].plot(Ip[0],Ip[1]/100000,c='k',linestyle='-',linewidth=2)
        axs[0].tick_params(direction='in')
        axs[0].axvline(x=self.A[i,7],ls='--',color='k',label='破裂',linewidth=2,)
        axs[0].axvline(x=X,ls=':',color='k',linewidth=2,)

        #axs[0].text(0.5,0.8,r'$I_P$',fontdict=fontt1)
        Ip_max = max(Ip[1]/100000)
        Ip_min = min(Ip[1]/100000)
        axs[0].set_ylim(Ip_min-0.5,Ip_max+0.5)
        
        font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf",size = 20)
        axs[0].set_ylabel(r'$I_\mathrm{P}(10^{5}$A)',fontproperties=font1,)
        fontt1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)
        axs[0].set_title('(a)',loc='left',fontproperties=fontt1)
        
        font2 = FontProperties(fname=fontpath+"SimSun.ttf",size = 22)
        legend1 = axs[0].legend(prop=font2,bbox_to_anchor=(0.5,0.56),borderaxespad=0,edgecolor='black')
        frame1 = legend1.get_frame() 
        frame1.set_alpha(1) 
        frame1.set_facecolor('none') # 设置图例legend背景透明

        axs[0].tick_params(labelsize=16,width=3)
        Labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
        [label.set_fontname('Times New Roman') for label in Labels]
        [label.set_fontsize(24) for label in Labels]#刻度值字号
        
        axs[0].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
        axs[0].spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
        axs[0].spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
        axs[0].spines['top'].set_linewidth(2);####设置上部坐标轴的粗细

        axs[1].tick_params(direction='in')
        axs[1].plot(ne[0], ne[1], linewidth=1,color='k',linestyle='-',label=r'$n_\mathrm{e}$')
        axs[1].axvline(x=self.A[i,7], ls='--', color='k',linewidth = 2,)
        fontt1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=19)
        axs[1].axvline(x=X,ls=':',color='k',linewidth=2,)
        axs[1].annotate(r'$n_\mathrm{e}/n_\mathrm{GW}$=0.8',xy=(X,2),xytext=(X-4,0),\
           textcoords='data',arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),\
           fontproperties=fontt1)


        ne_max = max(ne[1])
        ne_min = min(ne[1])
        axs[1].set_ylim(ne_min-0.7,ne_max+0.7)

        # 设置坐标刻度值的大小以及刻度值的字体
        axs[1].tick_params(labelsize=16,width=3)
        Labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
        [label.set_fontname('Times New Roman') for label in Labels]
        [label.set_fontsize(24) for label in Labels]#刻度值字号
        
       
        axs[1].plot(aminor1[0],Ngw,color='k',linestyle='-.',label = r'$n_{GW}$',linewidth = 2,)
        fontt2 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 14)

        font1 = FontProperties(fname=fontpath+"SimSun.ttf",size=19)
        axs[1].set_ylabel(r'($10^{19}m^{-3}$)', fontproperties=font1)
        fontt1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf",size=22)
        axs[1].set_title('(b)', loc = 'left',fontproperties=fontt1)

        axs[1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
        axs[1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
        axs[1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
        axs[1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细
        
        fontt1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=16)
        legend1 = axs[1].legend(prop=fontt1,bbox_to_anchor=(0.23,0.36),borderaxespad=0,edgecolor='black')
        frame1 = legend1.get_frame() 
        frame1.set_alpha(1) 
        frame1.set_facecolor('none') # 设置图例legend背景透明

        axs[2].tick_params(direction='in')
        axs[2].plot(sxr[0],sxr[1],linewidth = 2,color='k')
        axs[2].axvline(x=X,ls=':',color='k',linewidth = 2,)
        axs[2].axvline(x=self.A[i,7],ls='--',color='k',linewidth = 2,)

        axs[2].tick_params(labelsize=16,width=3)
        Labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
        [label.set_fontname('Times New Roman') for label in Labels]
        [label.set_fontsize(24) for label in Labels]#刻度值字号
        
        axs[2].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
        axs[2].spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
        axs[2].spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
        axs[2].spines['top'].set_linewidth(2);####设置上部坐标轴的粗细

        font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf",size = 18)
        axs[2].set_ylabel(r'SXR(V)', fontproperties=font1)
        fontt1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        axs[2].set_title('(c)', loc = 'left',fontproperties=fontt1)
        #axs[2].legend(prop=fontt,loc='best',borderaxespad=0,edgecolor='black')


        axs[3].tick_params(direction='in')
        axs[3].plot(Bt[0], Bt[1], linewidth = 2,color='k')
        axs[3].axvline(x=X,ls=':',color='k',linewidth = 2,)
        axs[3].axvline(x=self.A[i,7],ls='--',color='k',linewidth = 2,)
        font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf",size = 18)
        axs[3].set_ylabel(r'$B_\mathrm{\theta}$(V)',fontproperties=font1)
        fontt1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
        axs[3].set_title('(d)', loc = 'left',fontproperties=fontt1)
        font3 = FontProperties(fname=fontpath+"SimSun.ttf", size = 28)
        axs[3].set_xlabel('时间(s)',fontproperties=font3)
        #axs[3].legend(prop=fontt,loc='best',borderaxespad=0,edgecolor='black')

        
        axs[3].tick_params(labelsize=16,width=3)#刻度宽度
        Labels = axs[3].get_xticklabels() + axs[3].get_yticklabels()
        [label.set_fontname('Times New Roman') for label in Labels]#刻度值字体
        [label.set_fontsize(24) for label in Labels]#刻度值字号
        
        axs[3].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
        axs[3].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
        axs[3].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
        axs[3].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细
        
        plt.xlim([-0.5,9])

        font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf",size=24)
        fig.subplots_adjust(hspace=0.4,wspace=0.1)#调节两个子图间的距离
        if self.A[i,4] == 1:
            plt.suptitle('#%d'%self.A[i,0],fontproperties=font3,fontsize=23,fontweight='bold',x=0.5,y = 0.96)
        else:
            plt.suptitle('Nondisruptive pulse:%d'%self.A[i,0],fontsize=23,fontweight='bold',x=0.5,y = 0.96)
        out_fig = plt.gcf()
        out_fig.savefig(resultpath+'fig1.svg',format='svg',bbox_inches = 'tight')
        out_fig.savefig(resultpath +'fig1.pdf',format='pdf',bbox_inches = 'tight')
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
