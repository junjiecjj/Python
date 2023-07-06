#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
此函数是分析D_clust20,D_clust19的结果的函数，分别对不同阈值下的rate画图，以及对训练过程的loss,val_loss
等画图，最后还画出预测值和阈值以及实际密度和标签的图
'''
import numpy as np
from matplotlib.font_manager import * 
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('qt4agg') 
import pandas as pd
from sklearn.preprocessing import minmax_scale
import os
import xlrd
from matplotlib.font_manager import FontProperties
font = FontProperties(fname = "/usr/share/fonts/truetype/arphic/ukai.ttc", size=14) 

def split_shut():
    all_disr_shut = range(0,491)
    all_safe_shut = range(491,491+680)
    train_D = [i for i in all_disr_shut if i%3==0]
    val_D = [i for i in all_disr_shut if i%3==1]
    test_D = [i for i in all_disr_shut if i%3==2]
    train_nD = [i for i in all_safe_shut if i%3==0]
    val_nD = [i for i in all_safe_shut if i%3==1]
    test_nD = [i for i in all_safe_shut if i%3==2]

    all_train_shut = list(sorted(set(train_D).union(set(train_nD))))
    all_val_shut = list(sorted(set(val_D).union(set(val_nD))))
    all_test_shut = list(sorted(set(test_D).union(set(test_nD))))
    return all_train_shut,all_val_shut,all_test_shut
#*************************开始rate,res的分析***************************************
def analy_res_rate():
    index3 = np.arange(0,1,0.01)[np.where(roc_AUC==roc_AUC.max())[0][0]]
    max_auc = roc_AUC[np.where(roc_AUC==roc_AUC.max())[0][0]]
    fig,axs = plt.subplots(3,1,figsize=(4,6))
    axs[0].plot(np.arange(0,1,0.01),pred_res[:,0],'r',label='succ_nd')
    axs[0].plot(np.arange(0,1,0.01),pred_res[:,1],'b',label='fal_nd')
    axs[0].plot(np.arange(0,1,0.01),pred_res[:,2],'seagreen',label='succ_d')
    axs[0].plot(np.arange(0,1,0.01),pred_res[:,3],'cyan',label='late_d')
    axs[0].plot(np.arange(0,1,0.01),pred_res[:,4],'yellow',label='prem_d')
    axs[0].plot(np.arange(0,1,0.01),pred_res[:,5],'black',label='fal_d')
    axs[0].axvline(x=index3,ls='--',color='fuchsia',label='max_auc')
    axs[0].legend(loc='best')
    axs[0].set_ylabel('res',fontsize=15)

    axs[1].plot(np.arange(0,1,0.01),pred_rate[:,0],'r',label='SnD')
    axs[1].plot(np.arange(0,1,0.01),pred_rate[:,1],'b',label='FnD')
    axs[1].plot(np.arange(0,1,0.01),pred_rate[:,2],'seagreen',label='SD')
    axs[1].plot(np.arange(0,1,0.01),pred_rate[:,3],'cyan',label='LD')
    axs[1].plot(np.arange(0,1,0.01),pred_rate[:,4],'yellow',label='PD')
    axs[1].plot(np.arange(0,1,0.01),pred_rate[:,5],'black',label='FD')
    axs[1].axvline(x=index3,ls='--',color='fuchsia',label='max_auc')
    axs[1].legend(loc='best')
    axs[1].set_ylabel('rate',fontsize=15)

    axs[2].plot(np.arange(0,1,0.01),roc_AUC,'b',label='roc_AUC')
    axs[2].axvline(x=index3,ls='--',color='fuchsia',label='max_suc')
    axs[2].annotate(r'$max\_auc$:(%.2f,%.3f)'%(index3,max_auc),xy=(index3,max_auc),\
       xytext=(index3+0.08,max_auc-0.08),\
       arrowprops=dict(facecolor='fuchsia',arrowstyle='->',connectionstyle='arc3'))
    axs[2].legend(loc='best')
    axs[2].set_xlabel('threshold',fontsize=15)
    axs[2].set_ylabel('auc',fontsize=15)
    plt.savefig(filepath2+'rate_res%d_%d.jpg'%(num1,num),format='jpg',dpi=1000)
    return
#*************************结束rate,res的分析***************************************

#*************************开始loss,mae，lr的分析***************************************
def analy_loss():
    print('\n')
    print('hist keys:',hist.keys())
    index1 = hist['val_loss'][hist['val_loss']==hist['val_loss'].min()].index[0]
    min_Vloss = hist['val_loss'].min()
    fig,axs = plt.subplots(3,1,figsize=(8,10),)
    axs[0].plot(hist['Unnamed: 0'],hist['loss'],'r',label='train_loss')
    axs[0].plot(hist['Unnamed: 0'],hist['val_loss'],'b',label='val_loss')
    axs[0].axvline(x=index1,ls='--',color='fuchsia',label='min_val_loss')
    axs[0].annotate(r'$min\_vl$',xy=(index1,min_Vloss),xytext=(index1+2,min_Vloss+0.01),\
       arrowprops=dict(facecolor='fuchsia',arrowstyle='->',connectionstyle='arc3'))
    axs[0].legend(loc='best')
    axs[0].set_ylabel('loss',fontsize=15)

    index2 = hist['val_mean_absolute_error'][hist['val_mean_absolute_error']==hist['val_mean_absolute_error'].min()].index[0]
    min_Vmae = hist['val_mean_absolute_error'].min()
    axs[1].plot(hist['Unnamed: 0'],hist['mean_absolute_error'],'r',label='train_mean_absolute_error')
    axs[1].plot(hist['Unnamed: 0'],hist['val_mean_absolute_error'],'b',label='val_mae')
    axs[1].axvline(x=index2,ls='--',color='fuchsia',label='min_val_mae')
    axs[1].annotate(r'$min\_vmae$',xy=(index2,min_Vmae),xytext=(index2+2,min_Vmae+0.01),\
       arrowprops=dict(facecolor='fuchsia',arrowstyle='->',connectionstyle='arc3'))
    axs[1].legend(loc='best')
    axs[1].set_ylabel('mean_absolute_error',fontsize=15)

    axs[2].plot(hist['Unnamed: 0'],hist['lr'],'lime',label='lr')
    axs[2].legend(loc='best')
    axs[2].set_xlabel('epochs',fontsize=15)
    axs[2].set_ylabel('learning_rate',fontsize=15) 
    plt.savefig(filepath2+'thresh%d_%d.jpg'%(num1,num),format='jpg',dpi=1000)
    return
#*************************结束loss,mae，lr的分析***************************************

#*************************开始预测结果的分析******************************************
def Draw(shut):
    if all_selfc[shut,5]==-1:
        kind = 'succ_ND'
    elif all_selfc[shut,5]==0:
        kind = 'false_ND'
    elif all_selfc[shut,5]==1:
        kind = 'succ_D'
    elif all_selfc[shut,5]==2:
        kind = 'late_D'
    elif all_selfc[shut,5]==3:
        kind = 'prem_D'
    elif all_selfc[shut,5]==4:
        kind = 'false_D'

    index3 = np.arange(0,1,0.01)[np.where(roc_AUC==roc_AUC.max())[0][0]]
    fig,axs = plt.subplots(2,1,figsize=(6,4),)
    t1 = time[shut][np.where(density[shut]==density[shut].min())][0]
    t2 = time[shut][np.where(density[shut]==density[shut].max())][0]
    axs[0].plot(time[shut],real_lab[shut],'black',label='real_label')
    axs[0].plot(time[shut],pred_resu[shut],'fuchsia',label='pred_resu')

    axs[0].set_ylabel('disruption_propably',fontsize=10)
    #axs[0].set_title('shut:%d,%s'%(all_selfc[shut,0],kind))
    #axs[0].set_title(r'$t_{pred\_disr}$=%.3f'%all_selfc[shut,3],loc='left')
    #axs[0].set_title(r'$t_{real\_disr}$=%.3f'%all_selfc[shut,1],loc='right')
    axs[0].set_title('(a)')
    axs[0].set_ylim(-0.1,1.1)
    axs[0].axvline(x=all_selfc[shut,1],ls='--',color='lime',label='real_disr')
    axs[0].axvline(x=all_selfc[shut,3],ls='--',color='cyan',label='pred_disr')
    axs[0].axhline(y=index3,ls='--',color='red',label='thresh')
    axs[0].legend(loc='best')
    axs[1].plot(time[shut],minmax_scale(density[shut]),'seagreen',label='norm_density')
    axs[1].plot(time[shut],real_lab[shut],'black',label='real_label')
    axs[1].axvline(x=t1,ls='--',color='b',label='min_density')
    axs[1].axvline(x=t2,ls='--',color='r',label='max_density')
    axs[1].set_xlabel('time',fontsize=10)
    axs[1].set_ylabel('norm_density',fontsize=10)
    axs[1].set_title('(b)')
    axs[1].legend(loc='best')
    fig.subplots_adjust(hspace=0.4)#调节两个子图间的距离
    plt.suptitle(r'$t_{pred\_disr}$=%.3f   shut:%d,%s   $t_{real\_disr}$=%.3f'%\
                 (all_selfc[shut,3],all_selfc[shut,0],kind,all_selfc[shut,1]))
    plt.savefig(filepath2+'%d.jpg'%(all_selfc[shut,0]),format='jpg',dpi=1000)
    plt.show()
    return

def Stacic():
    fig,axs = plt.subplots(1,1,figsize=(16,6),)
    colors = ['b','r','fuchsia','cyan','lime','black']
    mark = ['*','<','o','s','+','|']
    label = ['sNd','fNd','sD','lD','pD','fD']
    for i,c,m,l in zip(range(-1,5),colors,mark,label):
        x = np.where(all_selfc[:,5]==i)[0]
        y = [i]*len(x)
        axs.scatter(x,y,color=c,marker=m,label=l)
    axs.legend(loc='best')
    axs.scatter(range(1171),all_selfc[:,5],\
                c=all_selfc[:,5])
    axs.legend(loc='best')
    plt.show()
    return

def shut_index():
    res_snd = []
    res_fnd = []
    res_sd = []
    res_ld = []
    res_pd = []
    res_fd = []
    for i in all_shut:
        if all_selfc[i,5]==-1:
            res_snd.append(i)
        elif all_selfc[i,5]==0:
            res_fnd.append(i)
        elif all_selfc[i,5]==1:
            res_sd.append(i)
        elif all_selfc[i,5]==2:
            res_ld.append(i)
        elif all_selfc[i,5]==3:
            res_pd.append(i)
        elif all_selfc[i,5]==4:
            res_fd.append(i)
        else:pass
    return res_snd,res_fnd,res_sd,res_ld,res_pd,res_fd

def draw_chazhi():
    a = res_sd.copy()
    a.extend(res_pd)
    a.sort()
    chzhi = all_selfc[a,1]-all_selfc[a,3]
    #matplotlib.rcParams['font.family']= 'STSong'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig,axs = plt.subplots(1,1,figsize=(6,4),)
    axs.hist(chzhi, bins=50, color='steelblue', normed=True )
    plt.xlabel("提前时间(s)",fontproperties = font,)
    plt.ylabel("炮数",fontproperties = font)
    plt.savefig(filepath2+'aheadtime.jpg',format='jpg',dpi=1000)
    print("averge ahead time = %f s"%np.average(chzhi))

    return chzhi

def find_jiaoji(name):
    if name == "train":
        Set = all_train_shut
    if name == "val":
        Set = all_val_shut
    if name == "test":
        Set = all_test_shut
    a = res_sd.copy()
    a.extend(res_pd)
    a.sort()
    Set_snd = list(set(res_snd).intersection(set(Set)))
    Set_snd.sort()

    Set_fnd = list(set(res_fnd).intersection(set(Set)))
    Set_fnd.sort()

    Set_sd = list(set(a).intersection(set(Set)))
    Set_sd.sort()

    Set_ld = list(set(res_ld).intersection(set(Set)))
    Set_ld.sort()

    Set_fd = list(set(res_fd).intersection(set(Set)))
    Set_fd.sort()
    return Set_snd ,Set_fnd, Set_sd, Set_ld, Set_fd

def turn_shut(List):
    sheet = xlrd.open_workbook(r'/home/jack/公共的/excel_data/info.xlsx')
    sheet1 = sheet.sheet_by_name('Sheet1')
    row_num,col_num = sheet1.nrows,sheet1.ncols
    A=[]
    for i in range(row_num):
        A.append(sheet1.row_values(i))
    a = []
    for i in List:
        a.append(A[i][0])
    return a


num1 = 20
num = 40

filepath1 ='/home/jack/音乐/data/data%d/data%d_%d/'%(num1,num1,num)
filepath2 = '/home/jack/图片/data%d/data%d_%d/'%(num1,num1,num)
if os.path.exists(filepath2):
    pass
else:
    os.makedirs(filepath2)

pred_res = np.load(filepath1+'pred_rate_res_fB.npz')['pred_res']
pred_rate = np.load(filepath1+'pred_rate_res_fB.npz')['pred_rate']
roc_AUC = np.load(filepath1+'pred_rate_res_fB.npz')['AUC']
hist = pd.read_csv(filepath1+'history_dict%d.csv'%num1)

all_selfc = np.load(filepath1+'selfc%d_all.npz'%num1)['selfc']
pred_resu = np.load(filepath1+'pred_all_result%d.npz'%num1)['pred_result3']
real_lab = np.load(filepath1+'all_xy%d.npz'%num1)['data'][:,:,-2]
time = np.load(filepath1+'all_xy%d.npz'%num1)['data'][:,:,-1]
all_shut = np.load(filepath1+'all%d.npz'%num1)['all_shut']

density = np.load(filepath1+'all_xy%d.npz'%num1)['data'][:,:,1]

res_snd,res_fnd,res_sd,res_ld,res_pd,res_fd = shut_index()
all_train_shut,all_val_shut,all_test_shut = split_shut()

Set_snd,Set_fnd,Set_sd,Set_ld,Set_fd = find_jiaoji("test")

#b = turn_shut(res_fnd)

chzhi = draw_chazhi()
#Draw(322)
#analy_res_rate()
#analy_loss()

#Draw(407)
#for i in Set_fd:
#    Draw(i)

#*************************结束预测结果的分析******************************************

