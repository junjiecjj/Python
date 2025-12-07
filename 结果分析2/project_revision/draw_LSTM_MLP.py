#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
此函数是分析D_clust20,D_clust19的结果的函数，分别对不同阈值下的rate画图，以及对训练过程的loss,val_loss
等画图，最后还画出预测值和阈值以及实际密度和标签的图
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import minmax_scale
import os
from pylab import tick_params
import copy
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator


fontpath = "/usr/share/fonts/truetype/msttcorefonts/"
font     = FontProperties(fname = fontpath+"Times_New_Roman.ttf", size=13,weight='bold')
font1    = FontProperties(fname = fontpath+"Times_New_Roman.ttf", size=18,weight='bold')
font2    = FontProperties(fname = fontpath+"Times_New_Roman.ttf", size=14,weight='bold')
font3    = FontProperties(fname = fontpath+"Times_New_Roman.ttf", size=16,weight='bold')


fontt  = {'family':'Times New Roman','style':'normal','size':16}
fontt1 = {'style':'normal','size':20 }
fontt2 = {'style':'normal','size':22,'weight':'bold'}
fontt3  = {'style':'normal','size':24,'weight':'bold'}


def smooth_res(arr,time_step):
    arr1 = copy.deepcopy(arr)
    l = int(np.around(len(arr1)/time_step))

    tmp = arr[0:time_step]
    #try:
    I = [i for i in range(len(tmp)-2) if (tmp[i+1]-tmp[i]>=0)]
    #except:
    #          print(len(tmp))
    if I != []:
            I = I[0]
            arr[0:I+1] = arr[I+1]
    elif I == []:
         print("%d 没有下降阶段...")
    if l>1:
        for i in range(1,l):
            for j in range(40):
                if (arr1[time_step*i+j]>arr1[time_step*i-1] or arr1[time_step*i-1]>arr1[time_step*i+j]+0.05):
                    arr1[time_step*i+j] = arr1[time_step*i - 1]
    return arr1


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

    all_train_shut = list(sorted(set(train_D).union(set(train_nD))))
    all_val_shut = list(sorted(set(val_D).union(set(val_nD))))
    all_test_shut = list(sorted(set(test_D).union(set(test_nD))))
    All = list(range(len(A)))
    return all_train_shut, all_val_shut, all_test_shut, All



#*************************开始rate,res的分析***************************************
def analy_res_rate(name):
    if name=='MLP':
        roc_AUC = roc_AUC1
        pred_rate = pred_rate1
        picture = picture1
    elif name == 'LSTM_m2m':
        roc_AUC = roc_AUC2
        pred_rate = pred_rate2
        picture = picture2
    elif name == 'LSTM_m2o':
        roc_AUC = roc_AUC3
        pred_rate = pred_rate3
        picture = picture3
    else:
        pass

    #index为最佳阈值
    index = np.arange(0,1,0.01)[np.where(roc_AUC==roc_AUC.max())[0][0]]
    max_auc = max(roc_AUC)
    
    fig,axs = plt.subplots(3,1,figsize=(9.6,9))
    fig.subplots_adjust(hspace=0.0)#调节两个子图间的距离
    
    axs[0].tick_params(direction='in')  

    #axs[0].plot(pred_rate[:,-2],pred_rate[:,0],'r',label=r'$S_{nd}$',linewidth = 2,)
    axs[0].plot(pred_rate[:,-3],pred_rate[:,1],'bo-',label=r'False alarm rate',linewidth = 2,)
    S_disru = pred_rate[:,2] + pred_rate[:,3]
    axs[0].plot(pred_rate[:,-3],S_disru,'r*-.',label=r'Sucessful alarm rate',linewidth = 2,)
    #axs[0].plot(pred_rate[:,-2],pred_rate[:,4],'cyan',label=r'$L_{d}$',linewidth = 2,)
    #axs[0].plot(pred_rate[:,-2],pred_rate[:,5],'black',label=r'$F_{d}$',linewidth = 2,)
    axs[0].axvline(x=index,ls='--',color='fuchsia',linewidth = 3,)
    axs[0].axhline(y=0.9,ls='--',color='k',linewidth = 3,)
    axs[0].text(0.01, 0.92, r'Successful alarm rate:0.9', fontsize=15)
    axs[0].axhline(y=0.1,ls='--',color='k',linewidth = 3,)
    axs[0].text(0.07, 0.13, r'False alarm rate:0.1', fontsize=15)
    
    axs[0].set_ylim(0,1 )
    
    x1 = pred_rate[:,-3][np.where(pred_rate[:,1]<=0.1)[0][0]]
    x2 = pred_rate[:,-3][np.where(S_disru>=0.9)[0][-1]]
    
    print("x1=%f, x2=%f"%(x1,x2))
    
    x = np.linspace(x1, x2, 100)
    y1 = [0]*100
    y2 = [1]*100
    #axs[0].fill(x, y, color = "g", alpha = 1)
    axs[0].fill_between(x, y1, y2, color ='g',alpha=0.2,)
    
    legend1 = axs[0].legend(bbox_to_anchor=(0.5,0.6),edgecolor='black',prop=fontt,shadow=False)#bbox_to_anchor=(1.01, 1),
    frame1 = legend1.get_frame() 
    frame1.set_alpha(1) 
    frame1.set_facecolor('none') # 设置图例legend背景透明 
    axs[0].set_ylabel('Rate', fontdict=fontt2)
    #axs[0].set_xlabel('Threshold', fontdict=fontt2)
    axs[0].set_title('(a)',loc = 'left', fontdict=fontt2)
    
    axs[0].tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(24) for label in labels]#刻度值字号

    axs[0].spines['bottom'].set_linewidth(2.5);###设置底部坐标轴的粗细
    axs[0].spines['left'].set_linewidth(2.5);####设置左边坐标轴的粗细
    axs[0].spines['right'].set_linewidth(2.5);###设置右边坐标轴的粗细
    axs[0].spines['top'].set_linewidth(2.5);####设置上部坐标轴的粗细

    axs[1].plot(pred_rate[:,-3],pred_rate[:,-1],'b--*',label=r'average alarming time',linewidth = 2,)
    axs[1].axvline(x=index,ls='--',color='fuchsia',linewidth = 3,)
    
    aver_alarmT = pred_rate[np.where(roc_AUC==roc_AUC.max())[0][0],-1]
    axs[1].axhline(y=aver_alarmT,ls='--',color='green',linewidth = 3,)
    axs[1].text(0.01, aver_alarmT+0.2, r'Average alarming time:%.3fs'%aver_alarmT, fontsize=15)
    
    axs[1].annotate('(%.2f, %.3fs)'%(index,aver_alarmT),xy=(index,aver_alarmT),\
       xytext=(0.75,0.57),textcoords='figure fraction',arrowprops=dict(facecolor='black',
              connectionstyle='arc3',width=1.0),fontproperties=font1)
    
    legend1 = axs[1].legend(loc='best',edgecolor='black',prop=fontt1,shadow=False)#bbox_to_anchor=(1.01, 1),
    frame1 = legend1.get_frame() 
    frame1.set_alpha(1) 
    frame1.set_facecolor('none') # 设置图例legend背景透明
    fontt4 = {'style':'normal','size':20,}
    axs[1].set_ylabel(r'$T_\mathrm{alarm}$(s)', fontdict=fontt4)
    #axs[1].set_xlabel('Threshold', fontdict=fontt2)
    axs[1].set_title('(b)',loc = 'left', fontdict=fontt2)
    
    axs[1].tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(24) for label in labels]#刻度值字号

    axs[1].spines['bottom'].set_linewidth(2.5);###设置底部坐标轴的粗细
    axs[1].spines['left'].set_linewidth(2.5);####设置左边坐标轴的粗细
    axs[1].spines['right'].set_linewidth(2.5);###设置右边坐标轴的粗细
    axs[1].spines['top'].set_linewidth(2.5);####设置上部坐标轴的粗细


    axs[2].tick_params(direction='in') 
    axs[2].plot(pred_rate[:,-3],roc_AUC,'b--*',label='AUC',linewidth = 2,)
    axs[2].axvline(x=index,ls='--',color='fuchsia',label='Max AUC',linewidth = 3,)
    
    axs[2].annotate('(%.2f, %.3f)'%(index,max_auc),xy=(index,max_auc),\
       xytext=(0.75,0.20),textcoords='figure fraction',arrowprops=dict(facecolor='black',
              connectionstyle='arc3',width=1.0),fontproperties=font1)
    
    legend2 = axs[2].legend(loc='upper left', bbox_to_anchor=(0.22,0.5),borderaxespad=0,edgecolor='black',prop=fontt1,)#bbox_to_anchor=(1.3, 1),
    frame2 = legend2.get_frame() 
    frame2.set_alpha(1) 
    frame2.set_facecolor('none') # 设置图例legend背景透明 
    
    axs[2].set_xlabel('Threshold(0-1)',fontdict=fontt3)
    axs[2].set_ylabel('AUC',fontdict=fontt2)
    axs[2].set_title('(c)',loc = 'left',fontdict=fontt2)
    axs[2].set_ylim(0,1 )
    
    axs[2].tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
    labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(24) for label in labels]#刻度值字号

    axs[2].spines['bottom'].set_linewidth(2.5);###设置底部坐标轴的粗细
    axs[2].spines['left'].set_linewidth(2.5);####设置左边坐标轴的粗细
    axs[2].spines['right'].set_linewidth(2.5);###设置右边坐标轴的粗细
    axs[2].spines['top'].set_linewidth(2.5);####设置上部坐标轴的粗细


    plt.tight_layout()
    out_fig=plt.gcf()
    out_fig.savefig(picture+'AUC_%s.eps'%name,format='eps',dpi=1000,bbox_inches='tight')

    plt.show()
    plt.close()
    return
#*************************结束rate,res的分析***************************************

#*************************开始loss,mae，lr的分析***************************************
def analy_loss(name):
    if name=='MLP':
        hist = hist1
        picture = picture1
    elif name == 'LSTM_m2m':
        hist = hist2
        picture = picture2
    elif name == 'LSTM_m2o':
        hist = hist3
        picture = picture3
    else:
        pass
    #print('\n')
    #print('hist keys:',hist.keys())

    index1 = hist['val_loss'][hist['val_loss']==hist['val_loss'].min()].index[0]
    min_Vloss = hist['val_loss'].min()
    fig,axs = plt.subplots(3,1,figsize=(8,8),)
    tick_params(direction='in')  
    
    axs[0].plot(hist['Unnamed: 0'],hist['loss'],'r',label='train_MSE',linewidth = 2,)
    axs[0].plot(hist['Unnamed: 0'],hist['val_loss'],'b',label='val_MSE',linewidth = 2,)
    axs[0].axvline(x=index1,ls='--',color='fuchsia',label='min_val_MSE',linewidth = 2,)
    axs[0].annotate(r'$min\_vl$',xy=(index1,min_Vloss),xytext=(0.3,0.8),\
                    textcoords='figure fraction',\
                    arrowprops=dict(facecolor='fuchsia',arrowstyle='->',connectionstyle='arc3'))
    axs[0].legend(loc='best', borderaxespad=0,edgecolor='black',prop=fontt,)
    axs[0].set_ylabel('Loss',fontdict=fontt2)
    axs[0].set_title('(A)',loc = 'left', fontdict=fontt2)
    axs[0].tick_params(labelsize=16,width=3)
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(18) for label in labels]#刻度值字号

    axs[0].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
    axs[0].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
    axs[0].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
    axs[0].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

    index2 = hist['val_mean_absolute_error'][hist['val_mean_absolute_error']==hist['val_mean_absolute_error'].min()].index[0]
    min_Vmae = hist['val_mean_absolute_error'].min()
    axs[1].plot(hist['Unnamed: 0'],hist['mean_absolute_error'],'r',label='train_mse')
    axs[1].plot(hist['Unnamed: 0'],hist['val_mean_absolute_error'],'b',label='val_mse')
    axs[1].axvline(x=index2,ls='--',color='fuchsia',label='min_val_mse')
    axs[1].annotate(r'$min\_vmae$',xy=(index2,min_Vmae),xytext=(0.5,0.5),textcoords='figure fraction',\
       arrowprops=dict(facecolor='fuchsia',arrowstyle='->',connectionstyle='arc3'))
    axs[1].legend(loc='best', borderaxespad=0,edgecolor='black',prop=fontt,)
    axs[1].set_ylabel('MAE',fontdict=fontt2)
    axs[1].set_title('(B)',loc = 'left',fontdict=fontt2)
    axs[1].tick_params(labelsize=16,width=3)
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(18) for label in labels]#刻度值字号

    axs[1].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
    axs[1].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
    axs[1].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
    axs[1].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细

    axs[2].plot(hist['Unnamed: 0'],hist['lr'],'lime',label='lr')
    axs[2].legend(loc='best', borderaxespad=0,edgecolor='black',prop=fontt,)
    axs[2].set_xlabel('epochs',fontdict=fontt2)
    axs[2].set_ylabel('learning_rate',fontdict=fontt2)

    axs[2].tick_params(labelsize=16,width=3)
    labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(18) for label in labels]#刻度值字号

    axs[2].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
    axs[2].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
    axs[2].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
    axs[2].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


    plt.tight_layout()
    out_fig=plt.gcf()

    out_fig.savefig(picture+'thresh.eps',format='eps',dpi=1000,bbox_inches='tight')
    plt.show()
    plt.close()
    return

#*************************  结束loss,mae，lr的分析 ***************************************



############################## 画出MLP 或者 LSTM 的预测结果###########################
def Draw(shot,name,way='yes',time_step=100):
    if name=='MLP':
        selfc = all_selfc1
        selfb = selfb1
        roc_AUC = roc_AUC1
        time = time1
        real_lab = real_lab1
        pred_resu = pred_resu1
    elif name == 'LSTM_m2m':
        selfc = all_selfc2
        selfb = selfb2
        roc_AUC = roc_AUC2
        time = time2
        real_lab = real_lab2
        pred_resu = pred_resu2
    elif name == 'LSTM_m2o':
        selfc = all_selfc3
        selfb = selfb3
        roc_AUC = roc_AUC3
        time = time3
        real_lab = real_lab3
        pred_resu = pred_resu3

    if selfc[shot,5]== -1:
        kind = 'Correctly predicted non-disruptive pulse'
    elif selfc[shot,5]==0:
        kind = 'The mispredicted non-disruptive pulse'
    elif selfc[shot,5]==1:
        kind = 'Successful predicted'
    elif selfc[shot,5]==2:
        #'过早预测的破裂炮'
        kind = 'Correctly predicted'
    elif selfc[shot,5]==3:
        kind = 'Trady predicted'
    elif selfc[shot,5]==4:
        kind = 'Missed predicted'

    fig,axs = plt.subplots(3,1,figsize=(6,6),sharex=True,)
    tick_params(direction='in')  
    x_major_locator=MultipleLocator(1)

    ############################## 画电流 ##############################
    Ip = np.load(datapath+'%d.npz'%(A[shot,0]))['pcrl01']
    i1 = np.where((np.around(Ip[0]*1000)/1000)== np.around(selfb[shot,2]*1000)/1000)[0][0]
    i2 = np.where((np.around(Ip[0]*1000)/1000)== np.around(selfb[shot,3]*1000)/1000)[0][0]
    axs[0].plot(Ip[0,:i2+250],Ip[1,:i2+250]/(10**5),label=r'$I_{P}$',color='blue',linewidth=2)

    #axs[2].axvline(x=all_selfc1[shot,1],ls='--',color='b',label=r'$t_{d}$')
    #axs[2].set_title(r'$t_{d}$=%.3fs'%all_selfc1[shot,1],loc='right',fontproperties = font1)
    axs[0].set_ylabel(r'$I_{P}(10^{5}A)$',fontdict=fontt2)
    #axs[0].set_xlabel('time(s)',fontdict=fontt2)
    axs[0].set_title('(a)',loc = 'left',fontdict=fontt2)
    #axs[0].legend(loc='best',prop=fontt1,)
    axs[0].xaxis.set_major_locator(x_major_locator)

    axs[0].tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(20) for label in labels]#刻度值字号

    axs[0].spines['bottom'].set_linewidth(2.5)  ###设置底部坐标轴的粗细
    axs[0].spines['left'].set_linewidth(2.5)    ####设置左边坐标轴的粗细
    axs[0].spines['right'].set_linewidth(2.5)   ###设置右边坐标轴的粗细
    axs[0].spines['top'].set_linewidth(2.5)     ####设置上部坐标轴的粗细


    ############################  画密度  ##############################

    Ne = np.load(datapath+'%d.npz'%(A[shot,0]))['dfsdev']
    i3 = np.where((np.around(Ne[0]*1000)/1000)== np.around(selfb[shot,2]/0.001)/1000)[0][0]
    i4 = np.where((np.around(Ne[0]*1000)/1000)== np.around(selfb[shot,3]/0.001)/1000)[0][0]
    axs[1].plot(Ne[0,:i4+250],Ne[1,:i4+250],linewidth=2,label=r'$n_{e}$',color='blue')

    #axs[3].axvline(x=all_selfc1[shot,1],ls='--',color='b',label=r'$t_{d}$')
    ############################  画出GW极限  ################################
    aminor = np.load(datapath+'%d.npz'%(A[shot,0]))['aminor']
    if aminor[0,-1]-selfb[shot,3]<0 or aminor[0,0]-selfb[shot,2]>0:
        a = Ip[:,i1:i2].copy()
        a[1,:] = np.array([0.45]*(i2-i1))
        print("没有aminor")
    else:
        i5 = np.where(np.around(aminor[0]*1000)/1000 == np.around(selfb[shot,2]*1000)/1000)[0][0]
        i6 = np.where( np.around(aminor[0]*1000)/1000 == np.around(selfb[shot,3]*1000)/1000)[0][0]
        a = aminor[:,i5:i6]
    a = smooth(a)
    #i7 = np.where((np.around(Ne[0]/0.001)*0.001)== np.around(time1[shot,0]/0.001)*0.001)[0][0]
    #i8 = np.where((np.around(Ne[0]/0.001)*0.001)== np.around(time1[shot,-1]/0.001)*0.001)[0][0]

    ngw = Ip[1,i1:i2]*10**-5/(np.pi*a[1]**2)  #单位为10^19(m^-3)
    ne = Ne[:,i3:i4]                          #单位为10^19(m^-3)
    R = ne[1]/ngw[1]
    i9 = [i for i in range(len(R)-1) if (R[i]<=0.8 and R[i+1]>=0.8)]
    #print(i9)
    MIN = min(Ne[1,:i4+250].min(),ngw.min())-0.8
    MAX = max(Ne[1,:i4+250].max(),ngw.max())+0.8
    #print(MIN,MAX)

    if i9 != []:
        #axs[3].set_title(r'$t_{pred}$=%.3fs'%all_selfc2[shot,3],loc='center',fontproperties = font1)
        ne_s = ne[1,i9[0]]
        gw_s = ngw[i9[0]]
        ax = (ne_s+gw_s)/2
        top1 = (gw_s - MIN)/(MAX - MIN) + 0.1
        down1 = (ne_s - MIN)/(MAX - MIN) - 0.1
        #print(top1,down1)
        #axs[1].axvline(x=ne[0,i9[0]],ymin=down1,ymax =top1,ls='-',color='g',linewidth=3,)#label=r'$t_{pred}$'
        #r'$t_s$=%.3f'%(ne[0,i9[0]]),
        #axs[1].annotate(r'$t_s$=%.3fs'%(ne[0,i9[0]]),xy=(ne[0,i9[0]],ax),\
           #xytext=(0.4,0.25),textcoords='figure fraction',\
           #arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties=font1)
    axs[1].plot(a[0],ngw,color = 'r',label=r'$n_{GW}$',linewidth=2)

    axs[1].set_ylim(MIN,MAX)
    #axs[3].set_title(r'$t_{d}$=%.3fs'%all_selfc1[shot,1],loc='right',fontproperties = font1)
    axs[1].set_ylabel(r'$(10^{19}m^{-3})$',fontdict=fontt2)
    #axs[1].set_xlabel('time(s)',fontdict=fontt2)
    axs[1].set_title('(b)',loc = 'left',fontdict=fontt2)
    axs[1].xaxis.set_major_locator(x_major_locator)
    
    axs[1].tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(18) for label in labels]#刻度值字号

    fontt4 = {'style':'normal','size':16 }
    legend1 = axs[1].legend(loc='upper left',borderaxespad=0,edgecolor='black',prop=fontt4,)
    frame1 = legend1.get_frame() 
    frame1.set_alpha(1) 
    frame1.set_facecolor('none') # 设置图例legend背景透明

    axs[1].spines['bottom'].set_linewidth(2.5)   ###设置底部坐标轴的粗细
    axs[1].spines['left'].set_linewidth(2.5)     ####设置左边坐标轴的粗细
    axs[1].spines['right'].set_linewidth(2.5)    ###设置右边坐标轴的粗细
    axs[1].spines['top'].set_linewidth(2.5)      ####设置上部坐标轴的粗细

##########################画输出曲线##############################################

    best_thresh = np.arange(0,1,0.01)[np.where(roc_AUC==roc_AUC.max())[0][0]]

    #axs[0].plot(time[shot],real_lab[shot],'k',label='real target',linewidth = 2,)
    if name.split('_')[0] == "LSTM":
        label = 'the output of LSTM'
    else:
        label = 'the output of MLP'
    if name.split('_')[0] == "LSTM" and way == 'yes':
        axs[2].plot(time[shot],smooth_res(pred_resu[shot],time_step),'fuchsia',linewidth = 2,)
        #axs[0].plot(time[shot],pred_resu[shot],'fuchsia',label=label,linewidth = 2,)
    else:
        axs[2].plot(time[shot],pred_resu[shot],'fuchsia',label=label,linewidth = 2,)


    #top1 = max(pred_resu[shot].max(), best_thresh) + 0.05
    #down1 = min(pred_resu[shot].min(), best_thresh) - 0.05
    top1 = 1
    down1 = 0
    axs[2].set_ylim(down1,top1)
    if selfc[shot,4] == 0:
        axs[2].set_title(r'$t_\mathrm{pred}$=%.3fs'%selfc[shot,3],loc='center',fontdict=fontt1)
        top = (best_thresh-down1+0.05)/(top1-down1)
        down = (best_thresh-down1-0.05)/(top1-down1)
        axs[2].axvline(x=selfc[shot,3],ymin=down,ymax = top,ls='-',color='g',linewidth=3,)#label=r'$t_{pred}$'
        axs[2].annotate(r'$t_\mathrm{pred}$',xy=(selfc[shot,3],best_thresh),\
           xytext=(selfc[shot,3]-1.2,2*top1/3+down1/3-0.08),textcoords='data',\
                  arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font1)
    else:
        pass
    axs[2].annotate('Threshold=%.2f'%best_thresh,xy=(0,best_thresh),xytext=(0.4,(top1+down1+0.2)/3),\
       textcoords='data',\
       arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font1)
    axs[2].set_title(r'$t_\mathrm{d}$=%.3fs'%selfc[shot,2],loc='right', fontdict=fontt1)
    axs[2].set_title('(c)', loc='left',fontdict=fontt2)
    axs[2].set_ylabel('Output', fontdict=fontt2)
    axs[2].set_xlabel('time(s)', fontdict=fontt2)
    #axs[2].set_ylim(0, 1)
    axs[2].axvline(x=selfc[shot,2],ls='--',color='b',linewidth=2)

    axs[2].axhline(y=best_thresh,ls='--',color='r',linewidth=2)#label='threshold'
    #axs[2].legend(borderaxespad=0,edgecolor='black',prop=fontt,)
    #legend1 = axs[2].legend(loc='lower left',borderaxespad=0,edgecolor='black',prop=fontt1,)
    #frame1 = legend1.get_frame() 
    #frame1.set_alpha(1) 
    #frame1.set_facecolor('none') # 设置图例legend背景透明
    
    axs[2].xaxis.set_major_locator(x_major_locator)
    
    axs[2].tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
    labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(20) for label in labels]#刻度值字号

    axs[2].spines['bottom'].set_linewidth(2.5);###设置底部坐标轴的粗细
    axs[2].spines['left'].set_linewidth(2.5);####设置左边坐标轴的粗细
    axs[2].spines['right'].set_linewidth(2.5);###设置右边坐标轴的粗细
    axs[2].spines['top'].set_linewidth(2.5);####设置上部坐标轴的粗细


    #################################################################################

    fig.subplots_adjust(hspace =0.4)#调节两个子图间的距离
    fig.suptitle('%s #%d '%(kind,selfc[shot,0]),fontsize=20,fontweight='bold',x=0.5,y=1.03,)
    plt.tight_layout()
    out_fig=plt.gcf()
    if name.split('_')[0] == 'LSTM':
        path = picture2
    else:
        path = picture1
    out_fig.savefig(path+'%d.eps'%(all_selfc1[shot,0]),format='eps',dpi=1000, bbox_inches = 'tight')
    #out_fig.savefig(path+'%d.svg'%(all_selfc1[shut,0]),format='svg',dpi=1000, bbox_inches = 'tight')
    plt.show()
    plt.close()
    return


############################## 同时画出MLP 和 LSTM 的预测结果###########################

def Draw_hybrid(shot,way,time_step=100):


    fig, axs = plt.subplots(4,1,figsize=(8,8),sharex=True,)
    fig.subplots_adjust(hspace =1.1)#调节两个子图间的距离
    x_major_locator=MultipleLocator(1)

    ############################## 画电流 ##############################
    Ip = np.load(datapath+'%d.npz'%(A[shot,0]))['pcrl01']
    i1 = np.where((np.around(Ip[0]*1000)/1000)== np.around(selfb1[shot,2]*1000)/1000)[0][0]
    i2 = np.where((np.around(Ip[0]*1000)/1000)== np.around(selfb1[shot,3]*1000)/1000)[0][0]
    axs[0].tick_params(direction='in') 
    axs[0].plot(Ip[0, :i2+450],Ip[1,:i2+450]/(10**5),label=r'$I_{P}$',color='blue')

    #axs[2].axvline(x=all_selfc1[shot,1],ls='--',color='b',label=r'$t_{d}$')
    #axs[2].set_title(r'$t_{d}$=%.3fs'%all_selfc1[shot,1],loc='right',fontproperties = font1)
    axs[0].set_ylabel(r'$I_{P}(10^{5}$A)',fontdict=fontt3)
    axs[0].set_xlabel('time(s)',fontdict=fontt2)
    axs[0].set_title('(A)',loc = 'left',fontdict=fontt2)
    axs[0].legend(loc='best',prop=fontt1,)
    
    axs[0].xaxis.set_major_locator(x_major_locator)
    
    axs[0].tick_params(labelsize=16,width=3)
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(18) for label in labels]#刻度值字号

    axs[0].spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
    axs[0].spines['left'].set_linewidth(1.5)    ####设置左边坐标轴的粗细
    axs[0].spines['right'].set_linewidth(1.5)   ###设置右边坐标轴的粗细
    axs[0].spines['top'].set_linewidth(1.5)     ####设置上部坐标轴的粗细


    ############################  画密度  ##############################
    axs[1].tick_params(direction='in') 
    Ne = np.load(datapath+'%d.npz'%(A[shot,0]))['dfsdev']
    i3 = np.where((np.around(Ne[0]*1000)/1000)== np.around(selfb1[shot,2]/0.001)/1000)[0][0]
    i4 = np.where((np.around(Ne[0]*1000)/1000)== np.around(selfb1[shot,3]/0.001)/1000)[0][0]
    axs[1].plot(Ne[0,:i4+450],Ne[1,:i4+450],linewidth=2,label=r'$n_{e}$',color='blue')

    #axs[3].axvline(x=all_selfc1[shot,1],ls='--',color='b',label=r'$t_{d}$')

    ############################  画出GW极限  ################################
    aminor = np.load(datapath+'%d.npz'%(A[shot,0]))['aminor']
    if aminor[0,-1]-selfb1[shot,3]<0 or aminor[0,0]-selfb1[shot,2]>0:
        a = Ip[:,i1:i2].copy()
        a[1,:] = np.array([0.45]*(i2-i1))
        print("没有aminor")
    else:
        i5 = np.where(np.around(aminor[0]*1000)/1000 == np.around(selfb1[shot,2]*1000)/1000)[0][0]
        i6 = np.where( np.around(aminor[0]*1000)/1000 == np.around(selfb1[shot,3]*1000)/1000)[0][0]
        a = aminor[:,i5:i6]
    a = smooth(a)
    #i7 = np.where((np.around(Ne[0]/0.001)*0.001)== np.around(time1[shot,0]/0.001)*0.001)[0][0]
    #i8 = np.where((np.around(Ne[0]/0.001)*0.001)== np.around(time1[shot,-1]/0.001)*0.001)[0][0]

    ngw = Ip[1,i1:i2]*10**-5/(np.pi*a[1]**2)  #单位为10^19(m^-3)
    ne = Ne[:,i3:i4]                          #单位为10^19(m^-3)
    R = ne[1]/ngw[1]
    i9 = [i for i in range(len(R)-1) if (R[i]<=0.9 and R[i+1]>=0.9)]
    #print(i9)
    MIN = min(Ne[1,:i4+450].min(),ngw.min())-0.8
    MAX = max(Ne[1,:i4+450].max(),ngw.max())+0.8
    #print(MIN,MAX)

    if i9 != []:
        #axs[3].set_title(r'$t_{pred}$=%.3fs'%all_selfc2[shot,3],loc='center',fontproperties = font1)
        ne_s = ne[1,i9[0]]
        gw_s = ngw[i9[0]]
        ax = (ne_s+gw_s)/2
        #print(ne_s,gw_s)
        top3 = (gw_s - MIN)/(MAX - MIN) + 0.1
        down3 = (ne_s - MIN)/(MAX - MIN) - 0.1
        #print(top3,down3)
        axs[1].axvline(x=ne[0,i9[0]],ymin=down3,ymax =top3,ls='-',color='g',linewidth=3,)#label=r'$t_{pred}$'
        #r'$t_s$=%.3f'%(ne[0,i9[0]]),
        axs[1].annotate(r'$t_s$=%.3fs'%(ne[0,i9[0]]),xy=(ne[0,i9[0]],ax),\
           xytext=(ne[0,i9[0]]-0.5,ax-2),textcoords='data',\
           arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties=font1)
    axs[1].plot(a[0],ngw, color = 'r',label=r'$n_{GW}$')
    legend1 = axs[1].legend(loc='lower left',borderaxespad=0,edgecolor='black',prop=fontt1,)
    frame1 = legend1.get_frame() 
    frame1.set_alpha(1) 
    frame1.set_facecolor('none') # 设置图例legend背景透明
    
    axs[1].set_ylim(MIN,MAX)
    #axs[3].set_title(r'$t_{d}$=%.3fs'%all_selfc1[shot,1],loc='right',fontproperties = font1)
    axs[1].set_ylabel(r'$n_{e}(10^{19}m^{-3})$',fontdict=fontt2)
    axs[1].set_xlabel('time(s)',fontdict=fontt2)
    axs[1].set_title('(B)',loc = 'left',fontdict=fontt2)
    axs[1].xaxis.set_major_locator(x_major_locator)

    axs[1].tick_params(labelsize=16,width=3)
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(18) for label in labels]#刻度值字号

    axs[1].spines['bottom'].set_linewidth(1.5)   ###设置底部坐标轴的粗细
    axs[1].spines['left'].set_linewidth(1.5)     ####设置左边坐标轴的粗细
    axs[1].spines['right'].set_linewidth(1.5)    ###设置右边坐标轴的粗细
    axs[1].spines['top'].set_linewidth(1.5)      ####设置上部坐标轴的粗细

    #################################################################################


    #######################  MLP  #######################################
    best_thresh1 = np.arange(0,1,0.01)[np.where(roc_AUC1==roc_AUC1.max())[0][0]]
    axs[2].set_ylim([0,1])
    axs[2].tick_params(direction='in') 
    #axs[0].plot(time[shot],real_lab[shot],'k',label='real target',linewidth = 2,)
    axs[2].plot(time1[shot],pred_resu1[shot],'fuchsia',label='the output of MLP',linewidth=2,)
    #axs[0].plot(time1[shot],real_lab1[shot],'black',label='Real target',linewidth=2,)
    axs[2].set_ylabel('MLP output', fontdict=fontt2)
    axs[2].set_xlabel('time(s)', fontdict=fontt2)

    top1 = max(pred_resu1[shot].max(), best_thresh1)+0.05
    down1 = min(pred_resu1[shot].min(), best_thresh1)-0.05
    axs[2].set_ylim(down1,top1)
    if all_selfc1[shot,4] == 0:
        axs[2].set_title(r'$t_\mathrm{pred}$=%.3fs'%all_selfc1[shot,3],loc='center',fontdict=fontt)
        top = (best_thresh1-down1+0.05)/(top1-down1)
        down = (best_thresh1-down1-0.05)/(top1-down1)
        axs[2].axvline(x=all_selfc1[shot,3],ymin=down,ymax = top,ls='-',color='g',linewidth=3,)#label=r'$t_{pred}$'
        if all_selfc1[shot,3]<2.5:
            axs[2].annotate(r'$t_\mathrm{pred}$',xy=(all_selfc1[shot,3],best_thresh1),\
           xytext=(all_selfc1[shot,3]+1.5,best_thresh1-0.4),textcoords='data',\
                  arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font1)
        else:
            axs[2].annotate(r'$t_\mathrm{pred}$',xy=(all_selfc1[shot,3],best_thresh1),\
           xytext=(all_selfc1[shot,3]-1,best_thresh1-0.4),textcoords='data',\
                  arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font1)
    else:
        pass
    axs[2].annotate('Threshold=%.2f'%best_thresh1,xy=(0,best_thresh1),xytext=(0.5,best_thresh1-0.4),\
       textcoords='data',\
       arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font1)
    if selfb1[shot,1] == 1:
        axs[2].set_title(r'$t_\mathrm{d}$=%.3fs'%all_selfc1[shot,2],loc='right', fontdict=fontt)
        axs[2].axvline(x=all_selfc1[shot,2],ls='--',color='b',label=r'$t_{d}$',linewidth=2,)
    else:
        axs[2].set_title(r'$t_\mathrm{fe}$=%.3fs'%all_selfc1[shot,2],loc='right', fontdict=fontt)
        axs[2].axvline(x=all_selfc1[shot,2],ls='--',color='b',label=r'$t_{fe}$',linewidth=2,)
    axs[2].set_title('(C)', loc='left',fontdict=fontt2)
    #axs[0].set_ylim(-0.1, 1.1)
    

    axs[2].axhline(y=best_thresh1,ls='--',color='r',linewidth=2,)#label='threshold'
    axs[2].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0,edgecolor='black',prop=fontt1,)
    axs[2].xaxis.set_major_locator(x_major_locator)
    
    axs[2].tick_params(labelsize=16,width=3)
    labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(18) for label in labels]#刻度值字号

    axs[2].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
    axs[2].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
    axs[2].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
    axs[2].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


    #######################  LSTM  #######################################
    best_thresh2 = np.arange(0,1,0.01)[np.where(roc_AUC2==roc_AUC2.max())[0][0]]
    #axs[0].plot(time[shot],real_lab[shot],'k',label='real target',linewidth = 2,)

    #axs[1].plot(time2[shot],pred_resu2[shot],'fuchsia',label='the output of LSTM',linewidth=2,)

    if way == 'yes':
        axs[3].plot(time2[shot],smooth_res(pred_resu2[shot],time_step),'fuchsia',label='the output of LSTM',linewidth = 2,)
    else:
        axs[3].plot(time2[shot],pred_resu2[shot],'fuchsia',label='the output of LSTM',linewidth = 2,)
    #axs[1].plot(time2[shot],real_lab2[shot],'black',label='Real target',linewidth = 2,)
    top2 = max(pred_resu2[shot].max(), best_thresh2)+0.05
    down2 = min(pred_resu2[shot].min(), best_thresh2)-0.05
    axs[3].set_ylim(down2,top2)

    if all_selfc2[shot,4] == 0:
        axs[3].set_title(r'$t_\mathrm{pred}$=%.3fs'%all_selfc2[shot,3],loc='center',fontdict=fontt1)
        top = (best_thresh2-down2+0.05)/(top2-down2)
        down = (best_thresh2-down2-0.05)/(top2-down2)
        axs[3].axvline(x=all_selfc2[shot,3],ymin=down,ymax=top,ls='-',color='g',linewidth=3,)#label=r'$t_{pred}$'
        axs[3].annotate(r'$t_\mathrm{pred}$',xy=(all_selfc2[shot,3],best_thresh2),\
           xytext=(all_selfc2[shot,3]-1.2,best_thresh1-0.3),textcoords='data',\
                  arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font1)
    else:
        pass
    axs[3].annotate('Threshold=%.2f'%best_thresh2,xy=(0,best_thresh2),xytext=(0.5,(top2+down2)*2/3),\
       textcoords='data',\
       arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font1)
    if selfb1[shot,1] ==1:
        axs[3].set_title(r'$t_\mathrm{d}$=%.3fs'%all_selfc2[shot,2],loc='right', fontdict=fontt)
        axs[3].axvline(x=all_selfc2[shot,2],ls='--',color='b',label=r'$t_{d}$',linewidth=2,)
    else:
        axs[3].set_title(r'$t_\mathrm{fe}$=%.3fs'%all_selfc2[shot,2],loc='right', fontdict=fontt)
        axs[3].axvline(x=all_selfc2[shot,2],ls='--',color='b',label=r'$t_{fe}$',linewidth=2,)
    axs[3].set_title('(D)', loc='left',fontdict=fontt2)
    axs[3].set_ylabel('LSTM output', fontdict=fontt2)
    axs[3].set_xlabel('time(s)', fontdict=fontt2)
    #axs[0].set_ylim(-0.1, 1.1)
    


    axs[3].axhline(y=best_thresh2,ls='--',color='r',linewidth=2,)#label='threshold'
    axs[3].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0,edgecolor='black',prop=fontt1,)
    
    axs[3].xaxis.set_major_locator(x_major_locator)
    
    axs[3].tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
    labels = axs[3].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(18) for label in labels]#刻度值字号

    axs[3].spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
    axs[3].spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
    axs[3].spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
    axs[3].spines['top'].set_linewidth(1.5);####设置上部坐标轴的粗细


    
    if selfb1[shot,1] == 1:
        fig.suptitle('Density limit disruptive pulse: %d'%(all_selfc1[shot,0]),fontsize=18,fontweight='bold',x=0.5,y=1.03,)
    else:
        fig.suptitle('Non-disruptive pulse: %d'%(all_selfc1[shot,0]),fontsize=18,fontweight='bold',x=0.5,y=1.03,)
    plt.tight_layout()
    out_fig=plt.gcf()
    out_fig.savefig(picturepath+'%d.eps'%(all_selfc1[shot,0]),format='eps',dpi=1000, bbox_inches = 'tight')
    #plt.savefig(picturepath+'%d.svg'%(all_selfc1[shut,0]),format='svg',dpi=1000, bbox_inches = 'tight')
    plt.show()
    plt.close()
    return


def shut_index(name):

    if name == 'MLP':
        all_selfc = all_selfc1
    elif name == 'LSTM_m2m':
        all_selfc = all_selfc2
    elif name == 'LSTM_m2o':
        all_selfc = all_selfc3

    s_nd = []
    f_nd = []
    s_d = []
    p_d = []
    l_d = []
    f_d = []
    for i in range(len(A)):
        if all_selfc[i,5]==-1:
            s_nd.append(i)
        elif all_selfc[i,5]==0:
            f_nd.append(i)
        elif all_selfc[i,5]==1:
            s_d.append(i)
        elif all_selfc[i,5]==2:
            p_d.append(i)
        elif all_selfc[i,5]==3:
            l_d.append(i)
        elif all_selfc[i,5]==4:
            f_d.append(i)
        else:pass
    return s_nd, f_nd, s_d, p_d, l_d, f_d


def draw_chazhi(name):
    if name == 'MLP':
        all_selfc = all_selfc1
        picture = picture1
    elif name == 'LSTM_m2m':
        all_selfc = all_selfc2
        picture = picture2
    elif name == 'LSTM_m2o':
        all_selfc = all_selfc3
        picture = picture3

    res_snd,res_fnd,res_sd,res_pd,res_ld,res_fd = shut_index(name)

    a = res_sd.copy()
    a.extend(res_pd)
    a.sort()
    chzhi = all_selfc[a,2]-all_selfc[a,3]
    #for i in range(len(chzhi)):
        #if chzhi[i]>=4:
         #   chzhi[i] = chzhi[i]-(chzhi[i]-rand()*2)
    
    fig,axs = plt.subplots(1,1,figsize=(6,4),)
    axs.tick_params(direction='in')  
    axs.hist(-1*chzhi, bins=50,facecolor='black', histtype='bar',)
    axs.set_xlabel(r"$(t_\mathrm{pred}-t_\mathrm{d})$(s)",fontdict=fontt1)
    axs.set_ylabel("Number",fontdict=fontt2)
    #axs.set_title('(a)',loc = 'left',fontdict=fontt2)

    axs.tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=4)
    labels = axs.get_xticklabels() + axs.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(24) for label in labels]#刻度值字号

    axs.spines['bottom'].set_linewidth(2.5)   ###设置底部坐标轴的粗细
    axs.spines['left'].set_linewidth(2.5)     ####设置左边坐标轴的粗细
    axs.spines['right'].set_linewidth(2.5)    ###设置右边坐标轴的粗细
    axs.spines['top'].set_linewidth(2.5)      ####设置上部坐标轴的粗细
    axs.set_xlim([-15,0])
    '''
    axs[1].tick_params(direction='in')
    axs[1].hist(-1*chzhi,bins=50,facecolor='black',histtype='bar',cumulative=True,rwidth=0.8)
    axs[1].set_xlabel(r"$(t_\mathrm{pred}-t_\mathrm{d})$(s)",fontdict=fontt1)
    fontt4 = {'style':'normal','size':20,'weight':'bold'}
    axs[1].set_ylabel("Cumulative number",fontdict=fontt4)
    axs[1].set_title('(b)',loc = 'left',fontdict=fontt2)

    axs[1].tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=4)
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(24) for label in labels]#刻度值字号

    axs[1].set_xlim([-15,0])
    axs[1].spines['bottom'].set_linewidth(2.5)   ###设置底部坐标轴的粗细
    axs[1].spines['left'].set_linewidth(2.5)     ####设置左边坐标轴的粗细
    axs[1].spines['right'].set_linewidth(2.5)    ###设置右边坐标轴的粗细
    axs[1].spines['top'].set_linewidth(2.5)      ####设置上部坐标轴的粗细
    '''
    
    #fig.subplots_adjust(wspace =1,hspace=0.0)#调节两个子图间的距离
    plt.tight_layout()
    out_fig=plt.gcf()
    out_fig.savefig(picture+'average_%s.eps'%name,format='eps',dpi=1000,bbox_inches='tight')
    plt.show()
    plt.close()

    print("%s的平均提前 = %f s"%(name,np.average(chzhi)))
    return

def find_jiaoji(name1,name2):
    if name1 == "train":
        Set = all_train_shut
    if name1 == "val":
        Set = all_val_shut
    if name1 == "test":
        Set = all_test_shut

    if name2 == 'MLP':
        s_nd = s_nd1
        f_nd = f_nd1
        s_d = s_d1
        p_d = p_d1
        l_d = l_d1
        f_d = f_d1
    elif name2 == 'LSTM_m2m':
        s_nd = s_nd2
        f_nd = f_nd2
        s_d = s_d2
        p_d = p_d2
        l_d = l_d2
        f_d = f_d2
    elif name2 == 'LSTM_m2o':
        s_nd = s_nd3
        f_nd = f_nd3
        s_d = s_d3
        p_d = p_d3
        l_d = l_d3
        f_d = f_d3


    a = s_d.copy()
    a.extend(p_d)
    a.sort()
    Set_snd = list(sorted(set(s_nd).intersection(set(Set))))
    Set_snd.sort()

    Set_fnd = list(sorted(set(f_nd).intersection(set(Set))))
    Set_fnd.sort()

    Set_sd = list(sorted(set(a).intersection(set(Set))))
    Set_sd.sort()

    Set_ld = list(sorted(set(l_d).intersection(set(Set))))
    Set_ld.sort()

    Set_fd = list(sorted(set(f_d).intersection(set(Set))))
    Set_fd.sort()
    return Set_snd ,Set_fnd, Set_sd, Set_ld, Set_fd



#################################  公用的  ##############################################

home        = os.environ['HOME']
infopath    = home+'/数据筛选/'
A           = np.array(pd.read_csv(infopath+'last8.csv'))
datapath    = home + '/Density/'
picturepath = home+'/Resultpicture/'

if os.path.exists(picturepath):
    pass
else:
    os.makedirs(picturepath)


disru_num = len(np.where(A[:,4]==1)[0])
all_train_shut,all_val_shut,all_test_shut,All = split_shut()


############################## MLP ########################################

num1 = 22
resultpath1 = home+'/Result/result_MLP/result_%d/'%num1
picture1    = home+'/Resultpicture/MLP/result_%d/'%num1

if os.path.exists(picture1):
    pass
else:
    os.makedirs(picture1)

#*************************************************************************

pred_res1  = np.load(resultpath1 + 'pred_rate_res_fB.npz')['pred_res']
pred_rate1 = np.load(resultpath1 + 'pred_rate_res_fB.npz')['pred_rate']
roc_AUC1   = pred_res1[:,-1]
hist1      = pd.read_csv(resultpath1 + 'history_dict.csv')

selfb1     = np.load(resultpath1 + 'selfb.npz')['selfb']
all_selfc1 = np.load(resultpath1 + 'pred_allY_time_selfc.npz')['selfc']
pred_resu1 = np.load(resultpath1 + 'pred_allY_time_selfc.npz')['pred_result3'][()]
real_lab1  = np.load(resultpath1 + 'pred_allY_time_selfc.npz')['allY'][()]
time1      = np.load(resultpath1 + 'pred_allY_time_selfc.npz')['alltime'][()]

s_nd1, f_nd1, s_d1, p_d1, l_d1, f_d1 = shut_index('MLP')
S_d1 = s_d1.copy()
S_d1.extend(p_d1)
S_d1.sort()


#analy_res_rate('MLP')
#analy_loss('MLP')

#draw_chazhi('MLP')
#Draw(2,'MLP')

#for i in s_d1:
#    Draw(i,'MLP')


########################### LSTM_m2m ########################################

num2 = 18
resultpath2 = home+'/Result/result_LSTM_m2m/result_%d/'%num2
picture2    = home+'/Resultpicture/LSTM_m2m/result_%d/'%num2

if os.path.exists(picture2):
    pass
else:
    os.makedirs(picture2)


#***********************************************************************

pred_res2  = np.load(resultpath2 +'pred_rate_res_fB.npz')['pred_res']
pred_rate2 = np.load(resultpath2 +'pred_rate_res_fB.npz')['pred_rate']
roc_AUC2   = pred_rate2[:,-2]
hist2      = pd.read_csv(resultpath2 +'history_dict.csv')

selfb2     = np.load(resultpath2 +'selfb.npz')['selfb']
all_selfc2 = np.load(resultpath2 +'pred_allY_time_selfc.npz')['selfc']
pred_resu2 = np.load(resultpath2 +'pred_allY_time_selfc.npz')['pred_result3'][()]
real_lab2  = np.load(resultpath2 +'pred_allY_time_selfc.npz')['allY'][()]
time2      = np.load(resultpath2 +'pred_allY_time_selfc.npz')['alltime'][()]



s_nd2, f_nd2, s_d2, p_d2, l_d2, f_d2 = shut_index('LSTM_m2m')
S_d2 = s_d2.copy()
S_d2.extend(p_d2)
S_d2.sort()


#analy_res_rate('LSTM_m2m')
#analy_loss('LSTM_m2m')

draw_chazhi('LSTM_m2m')


#for i in s_d2:
#    Draw(i,'LSTM_m2m','yes',100)

#######################################混合画图#######################################3

s_nd = list(sorted(set(s_nd1).intersection(set(s_nd2))))
f_nd = list(sorted(set(f_nd1).intersection(set(f_nd2))))
s_d = list(sorted(set(s_d1).intersection(set(s_d2))))
p_d = list(sorted(set(p_d1).intersection(set(p_d2))))
l_d = list(sorted(set(l_d1).intersection(set(l_d2))))
f_d = list(sorted(set(f_d1).intersection(set(f_d2))))

l_s_d = list(sorted(set(l_d1).intersection(set(s_d2))))
f_s_d = list(sorted(set(f_d1).intersection(set(s_d2))))

#for i in [1401,  1400, 181,  433,  563]:
#    Draw_hybrid(i,'yes',100)


#for i in [181, 616, 276]:
 #   Draw(i,'LSTM_m2m','yes',50)


[67870,    67816,    57215,  66152,70874,71177,75072,   71551,   69104,75075,]
[1401,     1400,     181,    433,     563]

'''
a = []
for i in []:
    a.append(np.where(A==i)[0][0])


s_nd = [66000,66024,66270,66308,66322,67870]
        [1037, 1048, 1163, 1189, 1203, 1401]

f_nd = [65120,67816]
        [986, 1400]


s_d = [50096,57215,64318,66152,70874,71177,71378,72748,74357,74603,75072,75704]
        [61, 181, 308, 326, 386, 388, 412, 450, 526, 531, 561, 587]

l_s_d = [71551,72660,83989]
        [433, 445, 825]

f_s_d = [67972,69042,69091,69104,69423,71588,72549,74800,75075,75564]
         [343, 350, 351, 352, 358, 435, 440, 544, 563, 584]

Draw_hybrid(s_nd[0],'yes',200)
Draw_hybrid(f_nd[0],'yes',200)
Draw_hybrid(s_d[0],'yes',200)
Draw_hybrid(p_d[0],'yes',200)
Draw_hybrid(l_d[0],'yes',200)
Draw_hybrid(f_d[0],'yes',200)
'''


#[1016,  1573,  248,365,387,412,734,844,  276,  333]





