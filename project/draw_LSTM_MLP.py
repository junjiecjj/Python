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



def smooth(aminor):
    if abs(aminor[1,0]-0.45)>0.1:
        aminor[1,0] = 0.45
    for i in range(aminor.shape[1]-1):
        if abs(aminor[1,i+1]-aminor[1,i]) > 0.02:
            aminor[1,i+1] = aminor[1,i]
    for i in range(aminor.shape[1]):        
        if abs(aminor[1,i]-0.45) > 0.05:
            aminor[1,i] = 0.45
    return aminor

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
def analy_res_rate(name):
    if name=='LSTM':
        roc_AUC = roc_AUC1
        pred_rate = pred_rate1
    elif name == 'MLP':
        roc_AUC = roc_AUC2
        pred_rate = pred_rate2
        
    index = np.arange(0,1,0.01)[np.where(roc_AUC==roc_AUC.max())[0][0]]
    max_auc = roc_AUC[np.where(roc_AUC==roc_AUC.max())[0][0]]
    fig,axs = plt.subplots(2,1,figsize=(6,6))


    axs[0].plot(np.arange(0,1,0.01),pred_rate[:,0],'r',label=r'$s_{nd}$')
    axs[0].plot(np.arange(0,1,0.01),pred_rate[:,1],'b',label=r'$f_{nd}$')
    S_disru = pred_rate[:,2] + pred_rate[:,4]
    axs[0].plot(np.arange(0,1,0.01),S_disru,'seagreen',label=r'$s_{d}$')
    axs[0].plot(np.arange(0,1,0.01),pred_rate[:,3],'cyan',label=r'$l_{d}$')
    #axs[1].plot(np.arange(0,1,0.01),pred_rate[:,4],'yellow',label='PD')
    axs[0].plot(np.arange(0,1,0.01),pred_rate[:,5],'black',label=r'$f_{d}$')
    axs[0].axvline(x=index,ls='--',color='fuchsia',)
    axs[0].legend( loc = 'best',borderaxespad=0, edgecolor='black', prop=font1,shadow=False)#bbox_to_anchor=(1.01, 1),
    axs[0].set_ylabel('rate', fontproperties = font3)
    axs[0].set_xlabel('threshold', fontproperties = font2)
    axs[0].set_title('(A)',loc = 'left', fontproperties = font2)
    axs[0].tick_params(labelsize=16)
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


    axs[1].plot(np.arange(0,1,0.01),roc_AUC,'b',label='AUC')
    axs[1].axvline(x=index,ls='--',color='fuchsia',label='Best threshold')
    axs[1].annotate(r'Max AUC:(%.2f,%.3f)'%(index,max_auc),xy=(index,max_auc),\
       xytext=(0.3,0.3),textcoords='figure fraction',arrowprops=dict(facecolor='fuchsia',arrowstyle='->',
              connectionstyle='arc3'),fontproperties = font1)
    axs[1].legend( loc = 'best', borderaxespad=0,edgecolor='black',prop=font1,)#bbox_to_anchor=(1.3, 1),
    axs[1].set_xlabel('threshold',fontproperties = font2)
    axs[1].set_ylabel('AUC',fontproperties = font2)
    axs[1].set_title('(B)',loc = 'left',fontproperties = font2)
    axs[1].tick_params(labelsize=16)
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.tight_layout()
    out_fig=plt.gcf()
    #out_fig.savefig(picturepath+'AUC_%s.eps'%name,format='eps',dpi=1000,bbox_inches='tight')

    plt.show()
    return
#*************************结束rate,res的分析***************************************

#*************************开始loss,mae，lr的分析***************************************
def analy_loss(name):
    if name=='LSTM':
        hist = hist1
    elif name == 'MLP':
        hist = hist2
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


def Draw(shut,name):
    if name=='LSTM':
        all_selfc = all_selfc1
        roc_AUC = roc_AUC1
        density = density1
        time = time1
        real_lab = real_lab1
        pred_resu = pred_resu1
    elif name == 'MLP':
        all_selfc = all_selfc2
        roc_AUC = roc_AUC2
        density = density2
        time = time2
        real_lab = real_lab2
        pred_resu = pred_resu2

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
    
    axs[0].plot(time[shut],real_lab[shut],'k',label='real target')
    if name == "LSTM":
        axs[0].plot(time[shut],pred_resu[shut],'fuchsia',label='the outputs of LSTM')
    else:
        axs[0].plot(time[shut],pred_resu[shut],'fuchsia',label='the outputs of MLP')

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

def Draw_hybrid(shot):
    #######################  LSTM  #######################################
    index1 = np.arange(0,1,0.01)[np.where(roc_AUC1==roc_AUC1.max())[0][0]]
    
    fig,axs = plt.subplots(4,1,figsize=(7,8),sharex=True)
    axs[0].plot(time1[shot],real_lab1[shot],'k',label='real target')
    axs[0].plot(time1[shot],pred_resu1[shot],'fuchsia',label='the outputs of LSTM')
    axs[0].axvline(x=all_selfc1[shot,1],ls='--',color='b',label=r'$t_{d}$',)
    axs[0].axhline(y=index1,ls='--',color='r',label='threshold')
    if all_selfc1[shot,4] == 0:
        axs[0].set_title(r'$t_{pred}$=%.3fs'%all_selfc1[shot,3],loc='center',fontproperties = font2)
        top = (index1+0.2)/1.2
        down = (index1)/1.2
        axs[0].axvline(x=all_selfc1[shot,3],ymin=down,ymax = top,ls='-',color='g',linewidth=3,)#label=r'$t_{pred}$'
        axs[0].annotate(r'$t_{pred}$',xy=(all_selfc1[shot,3],index1),\
           xytext=(all_selfc1[shot,3]-0.2,index1+0.15),textcoords='data',\
                  arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font1)
    axs[0].annotate('Threshold=%.2f'%index1,xy=(time1[shot][0],index1),xytext=(time1[shot][0]+0.05,index1+0.3),\
       textcoords='data',\
       arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font1)
    axs[0].set_title('(A)',fontproperties = font2,loc = 'left')
    axs[0].set_ylim(-0.1,1.1)
    axs[0].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0,edgecolor='black',prop=font,)
    axs[0].set_ylabel('LSTM Outputs',fontproperties = font2)
    axs[0].set_xlabel('time(s)',fontproperties = font3)
    axs[0].set_title(r'$t_{d}$=%.3fs'%all_selfc1[shot,1],loc='right',fontproperties = font2)
    
    axs[0].tick_params(labelsize=14)
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


    ##############################  MLP   ####################################
    index2 = np.arange(0,1,0.01)[np.where(roc_AUC2==roc_AUC2.max())[0][0]]
    axs[1].plot(time2[shot],real_lab2[shot],'k',label='real target')
    axs[1].plot(time2[shot],pred_resu2[shot],'fuchsia',label='the outputs of MLP')
    axs[1].axvline(x=all_selfc2[shot,1],ls='--',color='b',label=r'$t_{d}$')
    axs[1].axhline(y=index2,ls='--',color='r',label='threshold')
    if all_selfc2[shot,4] == 0:
        axs[1].set_title(r'$t_{pred}$=%.3fs'%all_selfc2[shot,3],loc='center',fontproperties = font2)
        top = (index2+0.2)/1.2
        down = (index2)/1.2
        axs[1].axvline(x=all_selfc2[shot,3],ymin=down,ymax = top,ls='-',color='g',linewidth=3,)#label=r'$t_{pred}$'
        axs[1].annotate(r'$t_{pred}$',xy=(all_selfc2[shot,3],index2),\
           xytext=(all_selfc2[shot,3]-0.2,index2+0.15),textcoords='data',\
                  arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font1)
    axs[1].annotate('Threshold=%.2f'%index2,xy=(time2[shot][0],index2),xytext=(time2[shot][0]+0.05,index2+0.3),\
       textcoords='data',\
       arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font1)
    axs[1].set_title('(B)',fontproperties = font2,loc = 'left')
    axs[1].set_ylim(-0.1,1.1)
    axs[1].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0,edgecolor='black',prop=font,)
    axs[1].set_ylabel('MLP Outputs',fontproperties = font2)
    axs[1].set_xlabel('time(s)',fontproperties = font3)
    axs[1].set_title(r'$t_{d}$=%.3fs'%all_selfc2[shot,1],loc='right',fontproperties = font2)
    
    axs[1].tick_params(labelsize=14)
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    
    '''
    ###############################################################
    axs[2].plot(time1[shot],minmax_scale(density1[shot]),'seagreen',label='normalized density')
    axs[2].plot(time1[shot],real_lab1[shot],'k',label='real target')

    #axs[1].axvline(x=t1,ls='--',color='b',label=r'$t_{min}$')

    #axs[1].axvline(x=t2,ls='--',color='r',label=r'$t_{max}$')
    axs[2].set_xlabel('time(s)',fontproperties = font1)
    axs[2].set_ylabel('normalized density',fontproperties = font1)
    axs[2].set_title('(C)',fontproperties = font2,loc = 'left')
    axs[2].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0,edgecolor='black',prop=font,)
    axs[2].tick_params(labelsize=14)
    labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    '''
    ############################## 画电流 ##############################
    Ip = np.load(datapath+'%d.npz'%(A[shot,0]))['pcrl01']
    i1 = np.where((np.around(Ip[0]/0.001)*0.001)== np.around(time1[shot,0]/0.001)*0.001)[0][0]
    i2 = np.where((np.around(Ip[0]/0.001)*0.001)== np.around(time1[shot,-1]/0.001)*0.001)[0][0]
    axs[2].plot(Ip[0,i1:i2+250],Ip[1,i1:i2+250]/(10**5))

    #axs[2].axvline(x=all_selfc1[shot,1],ls='--',color='b',label=r'$t_{d}$')
    #axs[2].set_title(r'$t_{d}$=%.3fs'%all_selfc1[shot,1],loc='right',fontproperties = font1)
    axs[2].set_ylabel(r'$I_{P}(10^{5}A)$',fontproperties = font2)
    axs[2].set_xlabel('time(s)',fontproperties = font3)
    axs[2].set_title('(C)',fontproperties = font2,loc = 'left')
    ############################  画密度  ##############################
    
    Ne = np.load(datapath+'%d.npz'%(A[shot,0]))['dfsdev']
    i3 = np.where((np.around(Ne[0]/0.001)*0.001)== np.around(time1[shot,0]/0.001)*0.001)[0][0]
    i4 = np.where((np.around(Ne[0]/0.001)*0.001)== np.around(time1[shot,-1]/0.001)*0.001)[0][0]
    axs[3].plot(Ne[0,i3:i4+250],Ne[1,i3:i4+250])

    #axs[3].axvline(x=all_selfc1[shot,1],ls='--',color='b',label=r'$t_{d}$')
    
    ############################  画出GW极限  ################################
    aminor = np.load(datapath+'%d.npz'%(A[shot,0]))['aminor']
    if aminor[0,-1]-time1[shot,-1]<0 or aminor[0,0]-time1[shot,0]>0:
        a = Ip[:,i1:i2].copy()
        a[1,:] = np.array([0.45]*(i2-i1))
        print("没有aminor")
    else:
        i5 = np.where((np.around(aminor[0]/0.001)*0.001)== np.around(time1[shot,0]/0.001)*0.001)[0][0]
        i6 = np.where((np.around(aminor[0]/0.001)*0.001)== np.around(time1[shot,-1]/0.001)*0.001)[0][0]
        a = aminor[:,i5:i6]
    a = smooth(a)
    #i7 = np.where((np.around(Ne[0]/0.001)*0.001)== np.around(time1[shot,0]/0.001)*0.001)[0][0]
    #i8 = np.where((np.around(Ne[0]/0.001)*0.001)== np.around(time1[shot,-1]/0.001)*0.001)[0][0]
    
    ngw = Ip[1,i1:i2]*10**-5/(np.pi*a[1]**2)
    ne = Ne[:,i3:i4]
    R = ne[1]/ngw[1]
    i9 = [i for i in range(len(R)-1) if (R[i]<=0.95 and R[i+1]>=0.95)]
    #print(i9)
    MIN = min(Ne[1,i3:i4+250].min(),ngw.min())-0.1
    MAX = max(Ne[1,i3:i4+250].max(),ngw.max()) + 0.1
    #print(MIN,MAX)

    if i9 != []:
        #axs[3].set_title(r'$t_{pred}$=%.3fs'%all_selfc2[shot,3],loc='center',fontproperties = font1)
        index3 = ne[1,i9[0]]
        index4 = ngw[i9[0]]
        #print(index3,index4)
        top1 = (index4-MIN)/(MAX-MIN)+0.1
        down1 = (index3-MIN)/(MAX-MIN)-0.1
        #print(top1,down1)
        axs[3].axvline(x=ne[0,i9[0]],ymin=down1,ymax =top1,ls='-',color='g',linewidth=3,)#label=r'$t_{pred}$'
        #r'$t_s$=%.3f'%(ne[0,i9[0]]),
        axs[3].annotate(r'$t_s$=%.3fs'%(ne[0,i9[0]]),xy=(ne[0,i9[0]],index3),\
           xytext=(0.4,0.1),textcoords='figure fraction',\
           arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font1)        
    axs[3].plot(a[0],ngw,color = 'r',label='GreenWald limit')
    axs[3].legend(loc='best',prop=font,)

    axs[3].set_ylim(MIN,MAX)
    #axs[3].set_title(r'$t_{d}$=%.3fs'%all_selfc1[shot,1],loc='right',fontproperties = font1)
    axs[3].set_ylabel(r'$n_{e}(10^{19}m^{-3})$',fontproperties = font2)
    axs[3].set_xlabel('time(s)',fontproperties = font3)
    axs[3].set_title('(D)',fontproperties = font2,loc = 'left')


    #################################################################################
    
    plt.subplots_adjust(hspace = 0.2)#调节两个子图间的距离
    plt.suptitle('pulse :%d'%(all_selfc1[shot,0]), x = 0.5, y = 1.02, fontproperties = font3)
    plt.tight_layout()
    #plt.savefig(picturepath+'%d.eps'%(all_selfc1[shot,0]),format='eps',dpi=1000,bbox_inches='tight')
    #plt.savefig(picture+'%d.svg'%(all_selfc1[shut,0]),format='svg',dpi=1000, bbox_inches = 'tight')
    plt.show()   
    return

def shut_index(name):
    
    if name == 'LSTM':
        all_selfc = all_selfc1
    elif name == 'MLP':
        all_selfc = all_selfc2

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

def draw_chazhi(name):
    if name == 'LSTM':
        all_selfc = all_selfc1
    elif name == 'MLP':
        all_selfc = all_selfc2

    res_snd,res_fnd,res_sd,res_ld,res_pd,res_fd = shut_index(name)

    a = res_sd.copy()
    a.extend(res_pd)
    a.sort()
    chzhi = all_selfc[a,1]-all_selfc[a,3]
    fig,axs = plt.subplots(1,1,figsize=(6,4),)
    axs.tick_params(labelsize=16)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    axs.hist(chzhi, bins=50, facecolor='black',alpha=0.75 )
    plt.xlabel("time difference(s)",fontproperties = font3)
    plt.ylabel("number of pulse",fontproperties = font3)
    #plt.savefig(picturepath+'average_%s.eps'%name,format='eps',dpi=1000,bbox_inches='tight')
    #plt.savefig(picture+'aheadtime.svg',format='svg',dpi=1000)
    print("%s的平均提前 = %f s"%(name,np.average(chzhi)))
    return chzhi

def find_jiaoji(name):
    if name == "train":
        Set = all_train_shut
    if name == "val":
        Set = all_val_shut
    if name == "test":
        Set = all_test_shut

    if name == 'LSTM':
        s_nd = s_nd1
        f_nd = f_nd1
        s_d = s_d1
        p_d = p_d1
        l_d = l_d1
        f_d = f_d1
    elif name == 'MLP':
        s_nd = s_nd2
        f_nd = f_nd2
        s_d = s_d2
        p_d = p_d2
        l_d = l_d2
        f_d = f_d2

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

#################################  公用的  ##############################################

home = os.environ['HOME']
infopath = home+'/数据筛选/'
A = np.array(pd.read_csv(infopath+'last5.csv'))
datapath = home + '/density/'
picturepath = home+'/resultpicture/'

fontpath = "/usr/share/fonts/truetype/msttcorefonts/"
font = FontProperties(fname = "/usr/share/fonts/truetype/msttcorefonts/times.ttf", size = 11,weight='bold')
font1 = FontProperties(fname ="/usr/share/fonts/truetype/msttcorefonts/times.ttf", size = 12,weight='bold')
font2 = FontProperties(fname = "/usr/share/fonts/truetype/msttcorefonts/times.ttf", size = 14)
font3 = FontProperties(fname = "/usr/share/fonts/truetype/msttcorefonts/times.ttf", size = 16)

disru_num = len(np.where(A[:,4]==1)[0])
all_train_shut,all_val_shut,all_test_shut,All = split_shut()

########################### LSTM ########################################
num1 = 3
resultpath1 =home+'/result/result_LSTM/result_%d/'%num1
picture1 = home+'/resultpicture/LSTM/result_%d/'%num1

if os.path.exists(picture1):
    pass
else:
    os.makedirs(picture1)


#######################################################################################
pred_res1 = np.load(resultpath1 +'pred_rate_res_fB.npz')['pred_res']
pred_rate1 = np.load(resultpath1 +'pred_rate_res_fB.npz')['pred_rate']
roc_AUC1 = np.load(resultpath1 +'pred_rate_res_fB.npz')['AUC']
hist1 = pd.read_csv(resultpath1 +'history_dict.csv')

all_selfc1 = np.load(resultpath1 +'selfc_all.npz')['selfc']
pred_resu1 = np.load(resultpath1 +'pred_all_result.npz')['pred_result3']
real_lab1 = np.load(resultpath1 +'all_xy.npz')['data'][:,-1000:,-2]
time1 = np.load(resultpath1 +'all_xy.npz')['data'][:,-1000:,-1]
density1 = np.load(resultpath1 +'all_xy.npz')['data'][:,-1000:,1]


s_nd1, f_nd1, s_d1, l_d1, p_d1, f_d1 = shut_index('LSTM')
S_d1 = s_d1.copy()
S_d1.extend(p_d1)
S_d1.sort()

########################### MLP ########################################
num2 = 47
resultpath2 =home+'/result/result_MLP/result_%d/'%num2
picture2 = home+'/resultpicture/MLP/result_%d/'%num2

if os.path.exists(picture2):
    pass
else:
    os.makedirs(picture2)


#######################################################################################
pred_res2 = np.load(resultpath2 +'pred_rate_res_fB.npz')['pred_res']
pred_rate2 = np.load(resultpath2 +'pred_rate_res_fB.npz')['pred_rate']
roc_AUC2 = np.load(resultpath2 +'pred_rate_res_fB.npz')['AUC']
hist2 = pd.read_csv(resultpath2 +'history_dict.csv')

all_selfc2 = np.load(resultpath2 +'selfc_all.npz')['selfc']
pred_resu2 = np.load(resultpath2 +'pred_all_result.npz')['pred_result3']
real_lab2 = np.load(resultpath2 +'all_xy.npz')['data'][:,:,-2]
time2 = np.load(resultpath2 +'all_xy.npz')['data'][:,:,-1]
density2 = np.load(resultpath2 +'all_xy.npz')['data'][:,:,1]



s_nd2, f_nd2, s_d2, l_d2, p_d2, f_d2 = shut_index('MLP')
S_d2 = s_d2.copy()
S_d2.extend(p_d2)
S_d2.sort()
#######################################################################################


'''
chzhi = draw_chazhi('MLP')
analy_res_rate('MLP')
analy_loss('MLP')

chzhi = draw_chazhi('LSTM')
analy_res_rate('LSTM')
analy_loss('LSTM')

LSTM和MLP都对[46985,47066,51011,53926,55592,56482,56731,56865,56867,57021,\
            57367,57493,61366,62271,64493,64859,64886,66765,66904,66972,67039,\
            70389,70522,71026,71537,72731,73448,73571,74357,77147,77169,77956,\
            78406,79468,79550,80431,80442,83354,83815,86660,87666]

[2,6,65,69,90,127,141,148,149,167,195,209,250,266,313,315,316,333,\
334,339,340,378,381,391,434,452,484,495,530,598,599,611,624,675,678,\
737,738,807,820,921, 955]

LSTM和MLP都错[47251,74132,74742,78223,56181,74100,83355,83356]
                [  9, 523, 539, 620, 109, 515, 808, 809]


LSTM对MLP错[45273,47055,71504,71501,79306,80379]
            [  0,   4, 428, 427, 662, 718]

LSTM对MLP迟[47327] [12]

Draw1(962)
Draw1(524)
Draw1(407)
Draw1(86)
Draw1(221)


for i in [983,1590]:
    Draw_hybrid(i)


Draw_hybrid(340)
#Draw(0,'LSTM')


del pred_res1,pred_rate1,roc_AUC1,hist1,all_selfc1,pred_resu1,real_lab1,time1,density1
del pred_res2,pred_rate2,roc_AUC2,hist2,all_selfc2,pred_resu2,real_lab2,time2,density2

print("delete")
'''


#Draw_hybrid(1318)
chzhi = draw_chazhi('LSTM')
chzhi = draw_chazhi('MLP')