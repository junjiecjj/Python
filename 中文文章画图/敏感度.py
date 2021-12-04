#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 18:15:02 2019
@author: jack
"""
import pandas as pd
from   pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import os
from pylab import tick_params
from   matplotlib.font_manager import FontProperties

fontpath = "/usr/share/fonts/truetype/arphic/"
font = FontProperties(fname=fontpath+"SimSun.ttf", size = 26)#
fontt2 = {'style':'normal','weight':'bold','size':28}
home = os.environ['HOME']
picture = home+'/tmp/'

resultpath =home+'/Result/result_MLP/'
Ip = pd.read_csv(resultpath + 'result_%d/history_dict.csv'%9).mean()
ne = pd.read_csv(resultpath + 'result_%d/history_dict.csv'%10).mean()
vp = pd.read_csv(resultpath + 'result_%d/history_dict.csv'%11).mean()
sxr = pd.read_csv(resultpath + 'result_%d/history_dict.csv'%12).mean()
pxuv30 = pd.read_csv(resultpath + 'result_%d/history_dict.csv'%13).mean()
pxuv18 = pd.read_csv(resultpath + 'result_%d/history_dict.csv'%14).mean()
kmp13t = pd.read_csv(resultpath + 'result_%d/history_dict.csv'%15).mean()
pbrem10 = pd.read_csv(resultpath + 'result_%d/history_dict.csv'%16).mean()
lmsz = pd.read_csv(resultpath + 'result_%d/history_dict.csv'%17).mean()
betap = pd.read_csv(resultpath + 'result_%d/history_dict.csv'%18).mean()
li = pd.read_csv(resultpath + 'result_%d/history_dict.csv'%19).mean()
q95 = pd.read_csv(resultpath + 'result_%d/history_dict.csv'%20).mean()
ic = pd.read_csv(resultpath + 'result_%d/history_dict.csv'%21).mean()


R = pd.concat([Ip,ne,vp,sxr,pxuv30,pxuv18,kmp13t,pbrem10,li,betap,lmsz,q95,ic],axis=1)
R.columns=['ip','ne','vp','sxr','pxuv30','pxuv18','kmp13t','pbrem10','li','betap','lmsz','q95','ic']
R = R.drop(['Unnamed: 0'])

def mingan(R,way):
    fig,axs = plt.subplots(1,1,figsize=(7,5),)
    tick_params(direction='in')
    name_list = [r'$I_P$',r'$n_e$',r'$V_{loop}$',r'$SXR$',r'$P_{core}$',r'$P_{edge}$',r'$B_{\theta}$',r'$P_{brem}$',\
                 r'$l_i$',r'$\beta_{P}$',r'$D_\mathrm{V}$',r'$q_{95}$',r'IC']

    axs.barh(range(len(name_list)), R.loc[way]/R.loc[way].max(), color='k',tick_label=name_list)
    #axs.set_xlabel('归一化敏感度',fontproperties=font)
    font = FontProperties(fname=fontpath+"SimSun.ttf", size = 24)#
    axs.set_title('归一化敏感度',fontproperties=font)
    axs.tick_params(labelsize=16,width=5)
    labels = axs.get_xticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(22) for label in labels]#刻度值字号
    
    
    labels = axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(18) for label in labels]#刻度值字号

    #axs.tick_params(labelsize=16,width=4)
    axs.spines['bottom'].set_linewidth(3);###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(3);####设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(3);###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(3);####设置上部坐标轴的粗细
    
    axs.set_xlim(0,1)
    
    plt.tight_layout()
    out_fig=plt.gcf()
    out_fig.savefig(picture+'fig4.pdf',format='pdf',dpi=1000, bbox_inches = 'tight')
    out_fig.savefig(picture+'fig4.svg',format='svg',dpi=1000, bbox_inches = 'tight')
    return

mingan(R,'loss')
mingan(R,'val_loss')
'''

R.index = ['Unnamed: 0','val_loss','val_mse','loss','mse','lr','Max_auc','best_trh','s_nd','f_nd','s_d','l_d','p_d','f_d']
R = R.drop(['Unnamed: 0','lr','best_trh'])
R.loc['R_f_d'] = R.iloc[-1]+R.iloc[-3]
R = R.drop(['s_nd','f_nd','s_d','l_d','p_d','f_d'])



def mingan(a):
    fig,axs = plt.subplots(6,1,figsize=(6,16),)
    ylabel = ['val_loss','val_mse','loss','mse','Max_auc','R_f_d']
    name_list = [r'$I_p$',r'$n_e$',r'vp',r'sxr',r'$P_{XUV30}$',r'$P_{XUV18}$',r'$B_{\theta}$',r'$P_{brem}$']
    for i in range(6):
        axs[i].bar(range(len(name_list)), (a.iloc[i]-a.iloc[i].min())/(a.iloc[i].max()-a.iloc[i].min()),color='b',tick_label=name_list)
        axs[i].set_ylabel(ylabel[i])
    fig.subplots_adjust(hspace =1.1)#调节两个子图间的距离
    return
    
    
def mingan1(a):
    fig,axs = plt.subplots(2,2,figsize=(10,6),)
    ylabel = ['val_BCE','val_MSE','train_BCE','train_MSE','Max_auc','R_f_d']
    name_list = [r'$I_p$',r'$n_e$',r'vp',r'sxr',r'$P_{XUV30}$',r'$P_{XUV18}$',r'$B_{\theta}$',r'$P_{brem}$']
    for i in range(4):
        axs[i%2,i//2].bar(range(len(name_list)), a.iloc[i]/a.iloc[i].max(),color='b',tick_label=name_list)
        axs[i%2,i//2].set_ylabel(ylabel[i])
    
    fig.subplots_adjust(hspace =0.3)#调节两个子图间的距离
    plt.suptitle("Normalized Impact factor",fontproperties = font)
    plt.savefig(picture+'leave.eps',format='eps',dpi=1000, bbox_inches = 'tight')
    plt.show()
    return

def mingan2(a,i,name):
    fig,axs = plt.subplots(1,1,figsize=(8,6),)
    ylabel = ['val_loss','val_mse','loss','mse','Max_auc','R_f_d']
    name_list = [r'$I_p$',r'$n_e$',r'vp',r'sxr',r'$P_{XUV30}$',r'$P_{XUV18}$',r'$B_{\theta}$',r'$P_{brem}$']
    axs.bar(range(len(name_list)), \
            a.iloc[i]/a.iloc[i].max(),color='b',capsize=19)#tick_label=name_list,
    axs.set_ylabel('normalized MSE',fontproperties = font)
    plt.xticks(list(range(8)),(r'$I_p$',r'$n_e$',r'vp',r'sxr',\
               r'$P_{XUV30}$',r'$P_{XUV18}$',r'$B_{\theta}$',r'$P_{brem}$'),\
            fontproperties = font1)
    fig.subplots_adjust(hspace =0.5)#调节两个子图间的距
    #plt.savefig(picture+'leave_%s.eps'%name,format='eps',dpi=1000, bbox_inches = 'tight')
    plt.show()
    return

fontpath = "/usr/share/fonts/truetype/msttcorefonts/"
font = FontProperties(fname = fontpath+"Times_New_Roman.ttf", size = 16)
font1 = FontProperties(fname = fontpath+"Times_New_Roman.ttf", size = 14) #fname = "/usr/share/fonts/truetype/arphic/SimSun.ttf", 
font2 = FontProperties(fname = fontpath+"Times_New_Roman.ttf", size = 11)

mingan(R)
mingan1(R)
mingan2(R,1,'valmse')

'''



