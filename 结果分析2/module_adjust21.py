#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
此函数是分析D_clust21的结果的函数，分别对不同阈值下的rate画图，以及对训练过程的loss,val_loss
等画图，最后还画出预测值和阈值以及实际密度和标签的图
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import minmax_scale
import os
import xlrd
from matplotlib.font_manager import FontProperties
font = FontProperties(fname = "/usr/share/fonts/truetype/arphic/SimSun.ttf", size=14) 
font1 = FontProperties(fname = "/usr/share/fonts/truetype/arphic/SimSun.ttf", size=8) 


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
    fig,axs = plt.subplots(2,1,figsize=(6,6))

    axs[0].plot(np.arange(0,1,0.01),pred_rate[:,0],'r',label='安全炮的正确预测率')
    axs[0].plot(np.arange(0,1,0.01),pred_rate[:,1],'b',label='安全炮的错误预测率')
    S_disru = pred_rate[:,2] + pred_rate[:,4]
    axs[0].plot(np.arange(0,1,0.01),S_disru,'seagreen',label='破裂炮的正确预测率')
    axs[0].plot(np.arange(0,1,0.01),pred_rate[:,3],'cyan',label='破裂炮的滞后预测率')
    #axs[1].plot(np.arange(0,1,0.01),pred_rate[:,4],'yellow',label='PD')
    axs[0].plot(np.arange(0,1,0.01),pred_rate[:,5],'black',label='破裂炮的错误预测率')
    axs[0].axvline(x=index3,ls='--',color='fuchsia',)
    axs[0].legend(prop=font1,bbox_to_anchor=(1.01, 1), borderaxespad=0,edgecolor='black',)
    axs[0].set_ylabel('比重',fontproperties = font)
    axs[0].set_xlabel('阈值',fontproperties = font)
    axs[0].set_title('(a)')

    axs[1].plot(np.arange(0,1,0.01),roc_AUC,'b',label='AUC值')
    axs[1].axvline(x=index3,ls='--',color='fuchsia',label='最佳阈值')
    axs[1].annotate(r'最大'+r'$AUC$:(%.2f,%.3f)'%(index3,max_auc),xy=(index3,max_auc),\
       xytext=(index3+0.1,max_auc-0.2),arrowprops=dict(facecolor='fuchsia',arrowstyle='->',
              connectionstyle='arc3'),fontproperties = font)
    axs[1].legend(prop=font1,bbox_to_anchor=(1.259, 1), borderaxespad=0,edgecolor='black',)
    axs[1].set_xlabel('阈值',fontproperties = font)
    axs[1].set_ylabel('AUC值',fontproperties = font)
    axs[1].set_title('(a)')
    plt.tight_layout()
    plt.savefig(filepath2+'rate_res%d_%d.eps'%(num1,num),format='eps',dpi=1000)
    plt.savefig(filepath2+'rate_res%d_%d.svg'%(num1,num),format='svg',dpi=1000)
    plt.show()
    return
#*************************结束rate,res的分析***************************************

#*************************开始loss,mae，lr的分析***************************************
def analy_loss():
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
        kind = '正确预测的安全炮'
    elif all_selfc[shut,5]==0:
        kind = '错误预测的安全炮'
    elif all_selfc[shut,5]==1:
        kind = '正确预测的破裂炮'
    elif all_selfc[shut,5]==2:
        kind = '滞后预测的破裂炮'
    elif all_selfc[shut,5]==3:
        kind = '正确预测的破裂炮'#'过早预测的破裂炮'
    elif all_selfc[shut,5]==4:
        kind = '错误预测的破裂炮'

    index3 = np.arange(0,1,0.01)[np.where(roc_AUC==roc_AUC.max())[0][0]]
    fig,axs = plt.subplots(2,1,figsize=(6.1,5),)
    a = time[shut,-1000:]
    b = density[shut,-1000:]
    t1 = a[np.where(b==b.min())][0]
    t2 = a[np.where(b==b.max())][0]

    if  all_selfc[shut,2] == 1:
        real_lab = 1/(1+np.exp(-(a-(t1+t2)/2)*(10/(t2-t1)))) 
    elif all_selfc[shut,2] == -1:
        real_lab = np.zeros(1000)
    axs[0].plot(a,real_lab,'k',label='真实标签')
    axs[0].plot(a,pred_resu[shut],'fuchsia',label='模型预测值')

    axs[0].set_ylabel('破裂概率',fontproperties = font)
    axs[0].set_xlabel('时间/s',fontproperties = font)
    axs[0].set_title(r'$t_{pred}$=%.3fs'%all_selfc[shut,3],loc='left')
    axs[0].set_title(r'$t_{d}$=%.3fs'%all_selfc[shut,1],loc='right')
    axs[0].set_title('(a)')
    axs[0].set_ylim(-0.1,1.1)
    if all_selfc[shut,2] == -1:
        axs[0].axvline(x=all_selfc[shut,1],ls='--',color='b',label='实验结束时刻')
    elif all_selfc[shut,2] == 1:
        axs[0].axvline(x=all_selfc[shut,1],ls='--',color='b',label='实际破裂时刻')
    if all_selfc[shut,4] == 0:
        axs[0].axvline(x=all_selfc[shut,3],ls='--',color='g',label='预测出的破裂时刻')
    elif all_selfc[shut,4] != 0:
        pass
    axs[0].axhline(y=index3,ls='--',color='red',label='阈值')
    axs[0].legend(prop=font1,bbox_to_anchor=(1.01, 1), borderaxespad=0,edgecolor='black',)
    
    axs[1].plot(a,minmax_scale(b),'seagreen',label='归一化密度')
    axs[1].plot(a,real_lab,'black',label='真实标签')
    axs[1].axvline(x=t1,ls='--',color='b',label='最小密度')
    axs[1].axvline(x=t2,ls='--',color='r',label='最大密度')
    axs[1].set_xlabel('时间/s',fontproperties = font)
    axs[1].set_ylabel('归一化密度',fontproperties = font)
    axs[1].set_title('(b)')
    axs[1].legend(prop=font1,bbox_to_anchor=(1.01, 1), borderaxespad=0,edgecolor='black',)
    
    fig.subplots_adjust(hspace =1.1)#调节两个子图间的距离
    fig.suptitle('炮号:%d,%s'%(all_selfc[shut,0],kind), x = 0.5, y = 1.01, fontproperties = font)
    plt.tight_layout()
    plt.savefig(filepath2+'%d.eps'%(all_selfc[shut,0]),format='eps',dpi=1000,bbox_inches = 'tight')
    plt.savefig(filepath2+'%d.svg'%(all_selfc[shut,0]),format='svg',dpi=1000,bbox_inches = 'tight')
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
    axs.hist(chzhi, bins=50, color='black', normed=True )
    plt.xlabel("提前时间(s)",fontproperties = font)
    plt.ylabel("炮数",fontproperties = font)
    plt.savefig(filepath2+'aheadtime.eps',format='eps',dpi=1000)
    plt.savefig(filepath2+'aheadtime.svg',format='svg',dpi=1000)
    print("平均提前 = %f s"%np.average(chzhi))

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


num1 = 21
num = 6

filepath1 ='/home/jack/音乐/data/data%d/data%d_%d/'%(num1,num1,num)
filepath2 = '/home/jack/snap/data%d/data%d_%d/'%(num1,num1,num)
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


chzhi = draw_chazhi()
#analy_loss()
analy_res_rate()


Draw(962)
Draw(524)
Draw(407)
Draw(86)
Draw(221)

#for i in Set_sd:
#    Draw(i)

#*************************结束预测结果的分析******************************************

