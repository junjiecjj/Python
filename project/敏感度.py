#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 18:15:02 2019

@author: jack
"""
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.font_manager import FontProperties


home = os.environ['HOME']
picture = home+'/result/'
#resultpath =home+'/result/result_%d/'%num
hist_Ip = pd.read_csv(home+'/result/result_MLP/result_%d/'%7 +'history_dict.csv').max()
hist_ne = pd.read_csv(home+'/result/result_MLP/result_%d/'%8 +'history_dict.csv').max()
hist_vp = pd.read_csv(home+'/result/result_MLP/result_%d/'%9 +'history_dict.csv').min()
hist_sxr = pd.read_csv(home+'/result/result_MLP/result_%d/'%10 +'history_dict.csv').min()
hist_pxuv30 = pd.read_csv(home+'/result/result_MLP/result_%d/'%11 +'history_dict.csv').min()
hist_pxuv18 = pd.read_csv(home+'/resultresult_MLP//result_%d/'%12 +'history_dict.csv').min()
hist_kmp13t = pd.read_csv(home+'/result/result_MLP/result_%d/'%13 +'history_dict.csv').min()
hist_pbrem10 = pd.read_csv(home+'/result/result_MLP/result_%d/'%14 +'history_dict.csv').min()

r_Ip = Series([0.793489,0.310000,0.956845,0.0431547,0.3285714,0.02653061,0.28979,0.3551],\
              index = ['Max_auc','best_thresh','s_nd','f_nd','s_d','l_d','p_d','f_d'])

r_ne = Series([0.777,0.220,0.9553571 , 0.0446428 , 0.362244 , 0.020408 , 0.25, 0.36734693 ],\
              index = ['Max_auc','best_thresh','s_nd','f_nd','s_d','l_d','p_d','f_d'])

r_vp = Series([0.836057,0.210,0.97023 , 0.0297619 , 0.455102 , 0.01836734 , 0.2275510 , 0.2989795 ],\
              index = ['Max_auc','best_thresh','s_nd','f_nd','s_d','l_d','p_d','f_d'])

r_sxr = Series([0.8177,0.2900,0.973214 , 0.026785 , 0.4244897 , 0.02551020 , 0.214285 , 0.3357142 ],\
              index = ['Max_auc','best_thresh','s_nd','f_nd','s_d','l_d','p_d','f_d'])

r_pxuv30 = Series([0.826760,0.330, 0.970238, 0.02976190, 0.45510204, 0.02857142, 0.207142, 0.3091836],\
              index = ['Max_auc','best_thresh','s_nd','f_nd','s_d','l_d','p_d','f_d'])

r_pxuv18 = Series([0.815476,0.2200,0.97172619 , 0.02827380 , 0.43163265 , 0.0224489 , 0.2316326 , 0.31428571 ],\
              index = ['Max_auc','best_thresh','s_nd','f_nd','s_d','l_d','p_d','f_d'])

r_kmp13t = Series([0.81688,0.2700,0.9672619 , 0.03273809 , 0.42959183 , 0.0244897959 , 0.2234693 , 0.322448 ],\
              index = ['Max_auc','best_thresh','s_nd','f_nd','s_d','l_d','p_d','f_d'])

r_pbrem10 = Series([0.816179,0.2500,0.973214 , 0.0267857 , 0.4428571 , 0.02040816 , 0.2214285 , 0.31530 ],\
              index = ['Max_auc','best_thresh','s_nd','f_nd','s_d','l_d','p_d','f_d'])

Ip = hist_Ip.append(r_Ip)
ne = hist_ne.append(r_ne)
vp = hist_vp.append(r_vp)
sxr = hist_sxr.append(r_sxr)
pxuv30 = hist_pxuv30.append(r_pxuv30)
pxuv18 = hist_pxuv18.append(r_pxuv30)
kmp13t = hist_kmp13t.append(r_kmp13t)
pbrem10 = hist_pbrem10.append(r_pbrem10)






R = pd.concat([Ip,ne,vp,sxr,pxuv30,pxuv18,kmp13t,pbrem10],axis=1)
R1 = R.copy()
R.columns=['ip','ne','vp','sxr','pxuv30','pxuv18','kmp13t','pbrem10']
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