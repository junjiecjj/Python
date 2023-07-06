#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
'''
此函数是分析D_clust20,D_clust19的结果的函数，分别对不同阈值下的rate画图，以及对训练过程的loss,val_loss
等画图，最后还画出预测值和阈值以及实际密度和标签的图
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import minmax_scale
import os
import xlrd
from matplotlib.font_manager import FontProperties


def split_shut():
    all_disr_shut = range(0,disru_num)
    all_safe_shut = range(disru_num,len(A))
    train_D = [i for i in all_disr_shut if i%3==0]
    val_D = [i for i in all_disr_shut if i%3==1]
    test_D = [i for i in all_disr_shut if i%3==2]
    train_nD = [i for i in all_safe_shut if i%3==0]
    val_nD = [i for i in all_safe_shut if i%3==1]
    test_nD = [i for i in all_safe_shut if i%3==2]
    #print("train_D : ",len(train_D))
    #print("train_nD :" ,len(train_nD))
    #print("val_D:", len(val_D))
    #print("val_nD:", len(val_nD))
    #print("test_D:",len(test_D))
    #print("test_nD:",len(test_nD))
    all_train_shut = list(sorted(set(train_D).union(set(train_nD))))
    all_val_shut = list(sorted(set(val_D).union(set(val_nD))))
    all_test_shut = list(sorted(set(test_D).union(set(test_nD))))
    All = list(range(2025))
    return all_train_shut, all_val_shut, all_test_shut,All



#*************************开始rate,res的分析***************************************
def analy_res_rate():
    index = np.arange(0,1,0.01)[np.where(roc_AUC==roc_AUC.max())[0][0]]
    max_auc = roc_AUC[np.where(roc_AUC==roc_AUC.max())[0][0]]
    fig,axs = plt.subplots(2,1,figsize=(6,6))
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

    axs[0].plot(np.arange(0,1,0.01),pred_rate[:,0],'r',label=r'$s_{nd}$')
    axs[0].plot(np.arange(0,1,0.01),pred_rate[:,1],'b',label=r'$f_{nd}$')
    S_disru = pred_rate[:,2] + pred_rate[:,4]
    axs[0].plot(np.arange(0,1,0.01),S_disru,'seagreen',label=r'$s_{d}$')
    axs[0].plot(np.arange(0,1,0.01),pred_rate[:,3],'cyan',label=r'$l_{d}$')
    #axs[1].plot(np.arange(0,1,0.01),pred_rate[:,4],'yellow',label='PD')
    axs[0].plot(np.arange(0,1,0.01),pred_rate[:,5],'black',label=r'$f_{d}$')
    axs[0].axvline(x=index,ls='--',color='fuchsia',)
    axs[0].legend( loc = 'best',borderaxespad=0, edgecolor='black', prop=font2,shadow=False)#bbox_to_anchor=(1.01, 1),
    axs[0].set_ylabel('ratio', fontproperties = font1)
    axs[0].set_xlabel('threshold', fontproperties = font1)
    axs[0].set_title('(a)',loc = 'left', fontproperties = font1)
    axs[0].tick_params(labelsize=16)
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


    axs[1].plot(np.arange(0,1,0.01),roc_AUC,'b',label='AUC')
    axs[1].axvline(x=index,ls='--',color='fuchsia',label='Best threshold')
    axs[1].annotate(r'Maximum AUC:(%.2f,%.3f)'%(index,max_auc),xy=(index,max_auc),\
       xytext=(0.3,0.3),textcoords='figure fraction',arrowprops=dict(facecolor='fuchsia',arrowstyle='->',
              connectionstyle='arc3'),fontproperties = font)
    axs[1].legend( loc = 'best', borderaxespad=0,edgecolor='black',prop=font2,)#bbox_to_anchor=(1.3, 1),
    axs[1].set_xlabel('threshold',fontproperties = font1)
    axs[1].set_ylabel('AUC',fontproperties = font1)
    axs[1].set_title('(b)',loc = 'left',fontproperties = font1)
    axs[1].tick_params(labelsize=16)
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.tight_layout()
    out_fig=plt.gcf()
    #out_fig.savefig(picture+'rate_res.eps',format='eps',dpi=1000,bbox_inches = 'tight')
    #out_fig.savefig(picture+'rate_res.svg',format='svg',dpi=1000,bbox_inches = 'tight')
    plt.show()
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
    axs[1].plot(hist['Unnamed: 0'],hist['mean_absolute_error'],'r',label='train_mse')
    axs[1].plot(hist['Unnamed: 0'],hist['val_mean_absolute_error'],'b',label='val_mse')
    axs[1].axvline(x=index2,ls='--',color='fuchsia',label='min_val_mse')
    axs[1].annotate(r'$min\_vmae$',xy=(index2,min_Vmae),xytext=(index2+2,min_Vmae+0.01),\
       arrowprops=dict(facecolor='fuchsia',arrowstyle='->',connectionstyle='arc3'))
    axs[1].legend(loc='best')
    axs[1].set_ylabel('mean_absolute_error',fontsize=15)

    axs[2].plot(hist['Unnamed: 0'],hist['lr'],'lime',label='lr')
    axs[2].legend(loc='best')
    axs[2].set_xlabel('epochs',fontsize=15)
    axs[2].set_ylabel('learning_rate',fontsize=15)
    #plt.savefig(picture+'thresh.jpg',format='jpg',dpi=1000)
    return
#*************************结束loss,mae，lr的分析***************************************


def Draw1(shut):
    if all_selfc[shut,5]==-1:
        kind = 'Correctly predicted non-disruptive pulse'
    elif all_selfc[shut,5]==0:
        kind = 'The mispredicted non-disruptive pulse'
    elif all_selfc[shut,5]==1:
        kind = 'Correctly predicted disruptive pulse'
    elif all_selfc[shut,5]==2:
        kind = 'late predicted disruptive pulse'
    elif all_selfc[shut,5]==3:
        kind = 'Correctly predicted disruptive pulse'#'过早预测的破裂炮'
    elif all_selfc[shut,5]==4:
        kind = 'The mispredicted disruptive pulse'

    index = np.arange(0,1,0.01)[np.where(roc_AUC==roc_AUC.max())[0][0]]
    fig,axs = plt.subplots(2,1,figsize=(7.5,5),)
    t1 = time[shut][np.where(density[shut]==density[shut].min())][0]
    t2 = time[shut][np.where(density[shut]==density[shut].max())][0]

    if all_selfc[shut,2] == -1:
        axs[0].plot(time[shut],real_lab[shut],'k',label='real target')
    else:
        axs[0].plot(time[shut],real_lab[shut],'k',label='target 1')
    axs[0].plot(time[shut],pred_resu[shut],'fuchsia',label='the outputs of model 1')
    axs[0].set_ylabel('disruption probability',fontproperties = font1)
    axs[0].set_xlabel('time(s)',fontproperties = font1)
    #axs[0].set_title('shut:%d,%s'%(all_selfc[shut,0],kind))
    if all_selfc[shut,4] == 0:
        axs[0].set_title(r'$t_{pred}$=%.3fs'%all_selfc[shut,3],loc='left',fontproperties = font1)
        top = (index+0.2)/1.2
        down = (index)/1.2
        axs[0].axvline(x=all_selfc[shut,3],ymin=down,ymax = top,ls='-',color='g',linewidth=3,)#label=r'$t_{pred}$'
        axs[0].annotate(r'$t_{pred}$',xy=(all_selfc[shut,3],index),\
           xytext=(all_selfc[shut,3]-0.2,index+0.15),textcoords='data',\
                  arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font)
    else:
        pass
    axs[0].set_title(r'$t_{d}$=%.3fs'%all_selfc[shut,1],loc='right',fontproperties = font1)
    axs[0].set_title('(a)',fontproperties = font1)
    axs[0].set_ylim(-0.1,1.1)
    axs[0].axvline(x=all_selfc[shut,1],ls='--',color='b',label=r'$t_{d}$')


    axs[0].axhline(y=index,ls='--',color='r',label='threshold')
    axs[0].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0,edgecolor='black',prop=font2,)
    axs[0].tick_params(labelsize=14)
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]



    axs[1].plot(time[shut],minmax_scale(density[shut]),'seagreen',label='normalized density')
    if all_selfc[shut,2] == -1:
        axs[1].plot(time[shut],real_lab[shut],'k',label='real target')
    else:
        axs[1].plot(time[shut],real_lab[shut],'k',label='target 1')
    #axs[1].axvline(x=t1,ls='--',color='b',label=r'$t_{min}$')

    #axs[1].axvline(x=t2,ls='--',color='r',label=r'$t_{max}$')
    axs[1].set_xlabel('time(s)',fontproperties = font1)
    axs[1].set_ylabel('normalized density',fontproperties = font1)
    axs[1].set_title('(b)',fontproperties = font)
    axs[1].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0,edgecolor='black',prop=font2,)
    axs[1].tick_params(labelsize=14)
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


    fig.subplots_adjust(hspace =1.1)#调节两个子图间的距离
    fig.suptitle('pulse :%d,%s'%(all_selfc[shut,0],kind), x = 0.5, y = 1.01, fontproperties = font)
    plt.tight_layout()
    #plt.savefig(picture+'%d.eps'%(all_selfc1[shut,0]),format='eps',dpi=1000, bbox_inches = 'tight')
    #plt.savefig(picture+'%d.svg'%(all_selfc1[shut,0]),format='svg',dpi=1000, bbox_inches = 'tight')
    plt.show()
    return


def shut_index():

    s_nd = []
    f_nd = []
    s_d = []
    l_d = []
    p_d = []
    f_d = []
    for i in range(len(A)):
        if all_selfc[i,5]==-1:
            s_nd.append(i)
        elif all_selfc[i,5]==0:
            f_nd.append(i)
        elif all_selfc[i,5]==1:
            s_d.append(i)
        elif all_selfc[i,5]==2:
            l_d.append(i)
        elif all_selfc[i,5]==3:
            p_d.append(i)
        elif all_selfc[i,5]==4:
            f_d.append(i)
        else:pass
    return s_nd, f_nd, s_d, l_d, p_d, f_d

def draw_chazhi():
    res_snd,res_fnd,res_sd,res_ld,res_pd,res_fd = shut_index()

    a = res_sd.copy()
    a.extend(res_pd)
    a.sort()
    chzhi = all_selfc[a,1]-all_selfc[a,3]
    fig,axs = plt.subplots(1,1,figsize=(6,4),)
    axs.tick_params(labelsize=16)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    axs.hist(chzhi, bins=50, facecolor='black',alpha=0.75 )
    plt.xlabel("time difference(s)",fontproperties = font)
    plt.ylabel("number of pulse",fontproperties = font)
    #plt.savefig(picture+'aheadtime.eps',format='eps',dpi=1000)
    #plt.savefig(picture+'aheadtime.svg',format='svg',dpi=1000)
    print("平均提前 = %fs"%np.average(chzhi))
    return chzhi

def find_jiaoji(name):
    if name == "train":
        Set = all_train_shut
    if name == "val":
        Set = all_val_shut
    if name == "test":
        Set = all_test_shut
    a = s_d.copy()
    a.extend(p_d)
    a.sort()
    Set_snd = list(set(s_nd).intersection(set(Set)))
    Set_snd.sort()

    Set_fnd = list(set(f_nd).intersection(set(Set)))
    Set_fnd.sort()

    Set_sd = list(set(a).intersection(set(Set)))
    Set_sd.sort()

    Set_ld = list(set(l_d).intersection(set(Set)))
    Set_ld.sort()

    Set_fd = list(set(f_d).intersection(set(Set)))
    Set_fd.sort()
    return Set_snd ,Set_fnd, Set_sd, Set_ld, Set_fd

def turn_shut(List):
    sheet = xlrd.open_workbook(r'/home/jack/公共的/excel_data/info.xlsx')
    sheet1 = sheet.sheet_by_name('Sheet1')
    row_num = sheet1.nrows
    A=[]
    for i in range(row_num):
        A.append(sheet1.row_values(i))
    a = []
    for i in List:
        a.append(A[i][0])
    return a


num = 3
home = os.environ['HOME']

infopath = home+'/数据筛选/'
resultpath =home+'/result/result_LSTM/result_%d/'%num
picture = home+'/resultpicture/LSTM/result_%d/'%num

A = np.array(pd.read_csv(infopath+'last5.csv'))   
fontpath = "/usr/share/fonts/truetype/msttcorefonts/"
font = FontProperties(fname = fontpath+"Times_New_Roman.ttf", size = 14)
font1 = FontProperties(fname = fontpath+"Times_New_Roman.ttf", size = 14) #fname = "/usr/share/fonts/truetype/arphic/SimSun.ttf", 
font2 = FontProperties(fname = fontpath+"Times_New_Roman.ttf", size = 11)

disru_num = len(np.where(A[:,4]==1)[0])
if os.path.exists(picture):
    pass
else:
    os.makedirs(picture)


#######################################################################################3
pred_res = np.load(resultpath +'pred_rate_res_fB.npz')['pred_res']
pred_rate = np.load(resultpath +'pred_rate_res_fB.npz')['pred_rate']
roc_AUC = np.load(resultpath +'pred_rate_res_fB.npz')['AUC']
hist = pd.read_csv(resultpath +'history_dict.csv')

all_selfc = np.load(resultpath +'selfc_all.npz')['selfc']
pred_resu = np.load(resultpath +'pred_all_result.npz')['pred_result3']
real_lab = np.load(resultpath +'all_xy.npz')['data'][:,-1000:,-2]
time = np.load(resultpath +'all_xy.npz')['data'][:,-1000:,-1]
density = np.load(resultpath +'all_xy.npz')['data'][:,-1000:,1]
##############################################################################################

##############################################################################################
s_nd, f_nd, s_d, l_d, p_d, f_d = shut_index()
all_train_shut,all_val_shut,all_test_shut,All = split_shut()


#b = turn_shut(res_fnd)

chzhi = draw_chazhi()
analy_res_rate()
analy_loss()

'''
Draw1(962)
Draw1(524)
Draw1(407)
Draw1(86)
Draw1(221)

'''
for i in f_nd:
    Draw1(i)

#*************************结束预测结果的分析******************************************

