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
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 14)  
font1 = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 14) #fname = "/usr/share/fonts/truetype/arphic/SimSun.ttf", 
font2 = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size = 11)
#fname = "/usr/share/fonts/truetype/arphic/SimSun.ttf", 
#fname="/usr/share/fonts/truetype/msttcorefonts/times.ttf",

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
def analy_res_rate(name):
    if name == "26":
        file = file1
    elif name == "42":
        file = file2
    index3 = np.arange(0,1,0.01)[np.where(roc_AUC1==roc_AUC1.max())[0][0]]
    max_auc = roc_AUC1[np.where(roc_AUC1==roc_AUC1.max())[0][0]]
    fig,axs = plt.subplots(2,1,figsize=(6,6))
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

    axs[0].plot(np.arange(0,1,0.01),pred_rate1[:,0],'r',label=r'$s_{nd}$')
    axs[0].plot(np.arange(0,1,0.01),pred_rate1[:,1],'b',label=r'$f_{nd}$')
    S_disru = pred_rate1[:,2] + pred_rate1[:,4]
    axs[0].plot(np.arange(0,1,0.01),S_disru,'seagreen',label=r'$s_{d}$')
    axs[0].plot(np.arange(0,1,0.01),pred_rate1[:,3],'cyan',label=r'$l_{d}$')
    #axs[1].plot(np.arange(0,1,0.01),pred_rate[:,4],'yellow',label='PD')
    axs[0].plot(np.arange(0,1,0.01),pred_rate1[:,5],'black',label=r'$f_{d}$')
    axs[0].axvline(x=index3,ls='--',color='fuchsia',)
    axs[0].legend( loc = 'best',borderaxespad=0, edgecolor='black', prop=font2,shadow=False)#bbox_to_anchor=(1.01, 1), 
    axs[0].set_ylabel('ratio', fontproperties = font1)
    axs[0].set_xlabel('threshold', fontproperties = font1)
    axs[0].set_title('(a)',loc = 'left', fontproperties = font1)
    axs[0].tick_params(labelsize=16)
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    
    
    axs[1].plot(np.arange(0,1,0.01),roc_AUC1,'b',label='AUC')
    axs[1].axvline(x=index3,ls='--',color='fuchsia',label='Best threshold')
    axs[1].annotate(r'Maximum AUC:(%.2f,%.3f)'%(index3,max_auc),xy=(index3,max_auc),\
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
    out_fig.savefig(file+'rate_res%d_%d.eps'%(num1,num),format='eps',dpi=1000,bbox_inches = 'tight')
    out_fig.savefig(file+'rate_res%d_%d.svg'%(num1,num),format='svg',dpi=1000,bbox_inches = 'tight')
    plt.show()
    return
#*************************结束rate,res的分析***************************************

#*************************开始loss,mae，lr的分析***************************************
def analy_loss():
    print('\n')
    print('hist keys:',hist1.keys())
    index1 = hist1['val_loss'][hist1['val_loss']==hist1['val_loss'].min()].index[0]
    min_Vloss = hist1['val_loss'].min()
    fig,axs = plt.subplots(3,1,figsize=(8,10),)
    axs[0].plot(hist1['Unnamed: 0'],hist1['loss'],'r',label='train_loss')
    axs[0].plot(hist1['Unnamed: 0'],hist1['val_loss'],'b',label='val_loss')
    axs[0].axvline(x=index1,ls='--',color='fuchsia',label='min_val_loss')
    axs[0].annotate(r'$min\_vl$',xy=(index1,min_Vloss),xytext=(index1+2,min_Vloss+0.01),\
       arrowprops=dict(facecolor='fuchsia',arrowstyle='->',connectionstyle='arc3'))
    axs[0].legend(loc='best')
    axs[0].set_ylabel('loss',fontsize=15)

    index2 = hist1['val_mean_absolute_error'][hist1['val_mean_absolute_error']==hist1['val_mean_absolute_error'].min()].index[0]
    min_Vmae = hist1['val_mean_absolute_error'].min()
    axs[1].plot(hist1['Unnamed: 0'],hist1['mean_absolute_error'],'r',label='train_mean_absolute_error')
    axs[1].plot(hist1['Unnamed: 0'],hist1['val_mean_absolute_error'],'b',label='val_mae')
    axs[1].axvline(x=index2,ls='--',color='fuchsia',label='min_val_mae')
    axs[1].annotate(r'$min\_vmae$',xy=(index2,min_Vmae),xytext=(index2+2,min_Vmae+0.01),\
       arrowprops=dict(facecolor='fuchsia',arrowstyle='->',connectionstyle='arc3'))
    axs[1].legend(loc='best')
    axs[1].set_ylabel('mean_absolute_error',fontsize=15)

    axs[2].plot(hist1['Unnamed: 0'],hist1['lr'],'lime',label='lr')
    axs[2].legend(loc='best')
    axs[2].set_xlabel('epochs',fontsize=15)
    axs[2].set_ylabel('learning_rate',fontsize=15) 
    plt.savefig(filepath2+'thresh%d_%d.jpg'%(num1,num),format='jpg',dpi=1000)
    return
#*************************结束loss,mae，lr的分析***************************************

#*************************开始预测结果的分析******************************************
def Draw12(shut,name):
    if name == "26":
        all_selfc = all_selfc1
    elif name == "42":
        all_selfc = all_selfc2

    fig,axs = plt.subplots(3,1,figsize=(7.5,8),)
    index3_1 = np.arange(0,1,0.01)[np.where(roc_AUC1==roc_AUC1.max())[0][0]]
    t1_1 = time1[shut][np.where(density1[shut]==density1[shut].min())][0]
    t2_1 = time1[shut][np.where(density1[shut]==density1[shut].max())][0]
    
    index3_2 = np.arange(0,1,0.01)[np.where(roc_AUC2==roc_AUC2.max())[0][0]]
    t1_2 = time1[shut][np.where(density2[shut]==density2[shut].min())][0]
    t2_2 = time1[shut][np.where(density2[shut]==density2[shut].max())][0]
    
 ################################################################ 

    axs[0].plot(time1[shut],real_lab1[shut],'k',label='target 1')
    axs[0].plot(time1[shut],pred_resu1[shut],'fuchsia',label='the outputs of model 1')
    axs[0].set_ylabel('disruption probability',fontproperties = font1)
    axs[0].set_xlabel('time(s)',fontproperties = font1)
    #axs[0].set_title('shut:%d,%s'%(all_selfc[shut,0],kind))
    if all_selfc1[shut,4] == 0:
        axs[0].set_title(r'$t_{pred}$=%.3fs'%all_selfc1[shut,3],loc='left',fontproperties = font1)
        top = (index3_1+0.2)/1.2
        down = (index3_1)/1.2
        axs[0].axvline(x=all_selfc1[shut,3],ymin=down,ymax = top,ls='-',color='g',linewidth=3,)#label=r'$t_{pred}$'
        axs[0].annotate(r'$t_{pred}$',xy=(all_selfc1[shut,3],(index3_1)),\
           xytext=(all_selfc1[shut,3]-0.2,index3_1+0.15),textcoords='data',\
                  arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font)
    else:
        pass
    axs[0].set_title(r'$t_{d}$=%.3fs'%all_selfc1[shut,1],loc='right',fontproperties = font1)
    axs[0].set_title('(a)',fontproperties = font1)
    axs[0].set_ylim(-0.1,1.1)
    if all_selfc1[shut,2] == -1:
        axs[0].axvline(x=all_selfc1[shut,1],ls='--',color='b',label='experiment terminated moment')
    elif all_selfc1[shut,2] == 1:
        axs[0].axvline(x=all_selfc1[shut,1],ls='--',color='b',label= r'$t_{d}$')

    axs[0].axhline(y=index3_1,ls='--',color='r',)#label='threshold'
    axs[0].annotate('Threshold',xy=(time1[shut][0],index3_1),xytext=(0.1,0.84),textcoords='figure fraction',\
                  arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font)
    axs[0].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0,edgecolor='black',prop=font2,)
    #axs[0].grid()
    axs[0].tick_params(labelsize=14)
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
######################################################################
    axs[1].plot(time2[shut],real_lab2[shut],'k',label='target 2')
    axs[1].plot(time2[shut],pred_resu2[shut],'fuchsia',label='the outputs of model 2')
    axs[1].set_ylabel('disruption probability',fontproperties = font1)
    axs[1].set_xlabel('time(s)',fontproperties = font1)
    #axs[0].set_title('shut:%d,%s'%(all_selfc[shut,0],kind))
    if all_selfc2[shut,4] == 0:
        axs[1].set_title(r'$t_{pred}$=%.3fs'%all_selfc2[shut,3],loc='left',fontproperties = font1)
        top = (index3_2+0.2)/1.2
        down = (index3_2)/1.2
        axs[1].axvline(x=all_selfc2[shut,3],ymin=down,ymax = top,ls='-',color='g',linewidth=3,)#label=r'$t_{pred}$'
        axs[1].annotate(r'$t_{pred}$',xy=(all_selfc2[shut,3],index3_2),\
           xytext=(all_selfc2[shut,3]+0.15,index3_2+0.15),textcoords='data',\
                  arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font)
    else:
        pass
    axs[1].set_title(r'$t_{d}$=%.3fs'%all_selfc2[shut,1],loc='right',fontproperties = font1)
    axs[1].set_title('(b)',fontproperties = font1)
    axs[1].set_ylim(-0.1,1.1)
    if all_selfc2[shut,2] == -1:
        axs[1].axvline(x=all_selfc2[shut,1],ls='--',color='b',label='experiment terminated moment')
    elif all_selfc2[shut,2] == 1:
        axs[1].axvline(x=all_selfc2[shut,1],ls='--',color='b',label=r'$t_{d}$')

    axs[1].axhline(y=index3_2,ls='--',color='r',)#label='threshold'
    axs[1].annotate('Threshold',xy=(time2[shut][0],index3_2),xytext=(0.1,0.54),textcoords='figure fraction',\
                  arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font)
    axs[1].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0, edgecolor='black',prop=font2,)
    #axs[1].grid()
    axs[1].tick_params(labelsize=14)
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
#################################################################################
    
    axs[2].plot(time1[shut],minmax_scale(density1[shut]),'seagreen',label='normalized density')
    axs[2].plot(time1[shut],real_lab1[shut],'black',label='target 1')
    axs[2].plot(time2[shut],real_lab2[shut],'m',label='target 2')
    axs[2].axvline(x=t1_1,ls='--',color='b',label=r'$t_{min}$')
    axs[2].axvline(x=t2_1,ls='--',color='r',label=r'$t_{max}$')
    axs[2].set_xlabel('time(s)',fontproperties = font1)
    axs[2].set_ylabel('normalized density',fontproperties = font1)
    axs[2].set_title('(c)',fontproperties = font)
    axs[2].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0,edgecolor='black',prop=font2,)
    #axs[2].grid()
    axs[2].tick_params(labelsize=16)
    labels = axs[2].get_xticklabels() + axs[2].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    
    fig.subplots_adjust(hspace =1.2)#调节两个子图间的距离
    fig.suptitle('pulse :%d'%(all_selfc1[shut,0]), x = 0.5, y = 1.01, fontproperties = font1)
    plt.tight_layout()
    plt.savefig(file3+'%d.eps'%(all_selfc[shut,0]),format='eps',dpi=1000,bbox_inches = 'tight')
    plt.savefig(file3+'%d.svg'%(all_selfc[shut,0]),format='svg',dpi=1000,bbox_inches = 'tight')
    plt.show()
    return

def Draw1(shut):
    if all_selfc1[shut,5]==-1:
        kind = 'Correctly predicted non-disruptive pulse'
    elif all_selfc1[shut,5]==0:
        kind = 'The mispredicted non-disruptive pulse'
    elif all_selfc1[shut,5]==1:
        kind = 'Correctly predicted disruptive pulse'
    elif all_selfc1[shut,5]==2:
        kind = 'late predicted disruptive pulse'
    elif all_selfc1[shut,5]==3:
        kind = 'Correctly predicted disruptive pulse'#'过早预测的破裂炮'
    elif all_selfc1[shut,5]==4:
        kind = 'The mispredicted disruptive pulse'

    index3 = np.arange(0,1,0.01)[np.where(roc_AUC1==roc_AUC1.max())[0][0]]
    fig,axs = plt.subplots(2,1,figsize=(7.5,5),)
    t1 = time1[shut][np.where(density1[shut]==density1[shut].min())][0]
    t2 = time1[shut][np.where(density1[shut]==density1[shut].max())][0]
    
    if all_selfc1[shut,2] == -1:
        axs[0].plot(time1[shut],real_lab1[shut],'k',label='real target')
    else:
        axs[0].plot(time1[shut],real_lab1[shut],'k',label='target 1')
    axs[0].plot(time1[shut],pred_resu1[shut],'fuchsia',label='the outputs of model 1')
    axs[0].set_ylabel('disruption probability',fontproperties = font1)
    axs[0].set_xlabel('time(s)',fontproperties = font1)
    #axs[0].set_title('shut:%d,%s'%(all_selfc[shut,0],kind))
    if all_selfc1[shut,4] == 0:
        axs[0].set_title(r'$t_{pred}$=%.3fs'%all_selfc1[shut,3],loc='left',fontproperties = font1)
        top = (index3+0.2)/1.2
        down = (index3)/1.2
        axs[0].axvline(x=all_selfc1[shut,3],ymin=down,ymax = top,ls='-',color='g',linewidth=3,)#label=r'$t_{pred}$'
        axs[0].annotate(r'$t_{pred}$',xy=(all_selfc1[shut,3],index3),\
           xytext=(all_selfc1[shut,3]-0.2,index3+0.15),textcoords='data',\
                  arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font)
    else:
        pass
    axs[0].set_title(r'$t_{d}$=%.3fs'%all_selfc1[shut,1],loc='right',fontproperties = font1)
    axs[0].set_title('(a)',fontproperties = font1)
    axs[0].set_ylim(-0.1,1.1)
    if all_selfc1[shut,2] == -1:
        axs[0].axvline(x=all_selfc1[shut,1],ls='--',color='b',label=r'$t_{d}$')
    elif all_selfc1[shut,2] == 1:
        axs[0].axvline(x=all_selfc1[shut,1],ls='--',color='b',label=r'$t_{d}$')

    axs[0].axhline(y=index3,ls='--',color='r',label='threshold')
    axs[0].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0,edgecolor='black',prop=font2,)
    axs[0].tick_params(labelsize=14)
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    
    
    
    axs[1].plot(time1[shut],minmax_scale(density1[shut]),'seagreen',label='normalized density')
    if all_selfc1[shut,2] == -1:
        axs[1].plot(time1[shut],real_lab1[shut],'k',label='real target')
    else:
        axs[1].plot(time1[shut],real_lab1[shut],'k',label='target 1')
    axs[1].axvline(x=t1,ls='--',color='b',label=r'$t_{min}$')
    axs[1].axvline(x=t2,ls='--',color='r',label=r'$t_{max}$')
    axs[1].set_xlabel('time(s)',fontproperties = font1)
    axs[1].set_ylabel('normalized density',fontproperties = font1)
    axs[1].set_title('(b)',fontproperties = font)
    axs[1].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0,edgecolor='black',prop=font2,)
    axs[1].tick_params(labelsize=14)
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    
    
    fig.subplots_adjust(hspace =1.1)#调节两个子图间的距离
    fig.suptitle('pulse :%d,%s'%(all_selfc1[shut,0],kind), x = 0.5, y = 1.01, fontproperties = font)
    plt.tight_layout()
    plt.savefig(file1+'%d.eps'%(all_selfc1[shut,0]),format='eps',dpi=1000, bbox_inches = 'tight')
    plt.savefig(file1+'%d.svg'%(all_selfc1[shut,0]),format='svg',dpi=1000, bbox_inches = 'tight')
    plt.show()
    return

def Draw2(shut):
    if all_selfc1[shut,5]==-1:
        kind = 'Correctly predicted non-disruptive pulse'
    elif all_selfc1[shut,5]==0:
        kind = 'The mispredicted non-disruptive pulse'
    elif all_selfc1[shut,5]==1:
        kind = 'Correctly predicted disruptive pulse'
    elif all_selfc1[shut,5]==2:
        kind = 'late predicted disruptive pulse'
    elif all_selfc1[shut,5]==3:
        kind = 'Correctly predicted disruptive pulse'#'过早预测的破裂炮'
    elif all_selfc1[shut,5]==4:
        kind = 'The mispredicted disruptive pulse'

    index3 = np.arange(0,1,0.01)[np.where(roc_AUC2==roc_AUC2.max())[0][0]]
    fig,axs = plt.subplots(2,1,figsize=(7.5, 5),)
    t1 = time2[shut][np.where(density2[shut]==density2[shut].min())][0]
    t2 = time2[shut][np.where(density2[shut]==density2[shut].max())][0]
    
    
    if all_selfc1[shut,2] == -1:
        axs[0].plot(time2[shut],real_lab2[shut],'k',label='real target')
    else:
        axs[0].plot(time2[shut],real_lab2[shut],'k',label='target 2')
    axs[0].plot(time2[shut],pred_resu2[shut],'fuchsia',label='the outputs of model 2')
    axs[0].set_ylabel('disruption probability',fontproperties = font1)
    axs[0].set_xlabel('time(s)',fontproperties = font1)
    #axs[0].set_title('shut:%d,%s'%(all_selfc[shut,0],kind))
    if all_selfc2[shut,4] == 0:
        axs[0].set_title(r'$t_{pred}$=%.3fs'%all_selfc2[shut,3],loc='left',fontproperties = font1)
        top = (index3+0.2)/1.2
        down = (index3)/1.2
        axs[0].axvline(x=all_selfc2[shut,3],ymin=down,ymax = top,ls='-',color='g',linewidth=3,)#label=r'$t_{pred}$'
        axs[0].annotate(r'$t_{pred}$',xy=(all_selfc2[shut,3],index3),\
           xytext=(all_selfc2[shut,3]-0.2,index3+0.15),textcoords='data',\
                  arrowprops=dict(arrowstyle='->',connectionstyle='arc3'),fontproperties = font)
    else:
        pass
    axs[0].set_title(r'$t_{d}$=%.3fs'%all_selfc2[shut,1],loc='right',fontproperties = font1)
    axs[0].set_title('(a)',fontproperties = font1)
    axs[0].set_ylim(-0.1,1.1)
    if all_selfc2[shut,2] == -1:
        axs[0].axvline(x=all_selfc2[shut,1],ls='--',color='b',label='experiment terminated moment')
    elif all_selfc2[shut,2] == 1:
        axs[0].axvline(x=all_selfc2[shut,1],ls='--',color='b',label='real moment of disruption')

    axs[0].axhline(y=index3,ls='--',color='r',label='threshold')
    axs[0].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0,edgecolor='black',prop=font2,)
    axs[0].tick_params(labelsize=16)
    labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


    axs[1].plot(time2[shut],minmax_scale(density2[shut]),'seagreen',label='normalized density')
    if all_selfc1[shut,2] == -1:
        axs[1].plot(time2[shut],real_lab2[shut],'k',label='real target')
    else:
        axs[1].plot(time2[shut],real_lab2[shut],'k',label='target 2')
    axs[1].axvline(x=t1,ls='--',color='b',label='The moment of minimum density')
    axs[1].axvline(x=t2,ls='--',color='r',label='The moment of maximum density')
    axs[1].set_xlabel('time(s)',fontproperties = font1)
    axs[1].set_ylabel('normalized density',fontproperties = font1)
    axs[1].set_title('(b)',fontproperties = font)
    axs[1].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0,edgecolor='black',prop=font2,)
    axs[1].tick_params(labelsize=16)
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    
    
    fig.subplots_adjust(hspace =1.1)#调节两个子图间的距离
    fig.suptitle('pulse :%d,%s'%(all_selfc2[shut,0],kind), x = 0.5, y = 1.01, fontproperties = font1)
    plt.tight_layout()
    plt.savefig(file2+'%d.eps'%(all_selfc1[shut,0]),format='eps',dpi=1000,bbox_inches = 'tight')
    plt.savefig(file2+'%d.svg'%(all_selfc1[shut,0]),format='svg',dpi=1000,bbox_inches = 'tight')
    plt.show()
    return


def shut_index(name):
    if name == "26":
        all_selfc = all_selfc1
    elif name == "42":
        all_selfc = all_selfc2

    res_snd = []
    res_fnd = []
    res_sd = []
    res_ld = []
    res_pd = []
    res_fd = []
    for i in all_shut1:
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

def draw_chazhi(name):
    if name == "26":
        all_selfc = all_selfc1
        file = file1
        res_snd,res_fnd,res_sd,res_ld,res_pd,res_fd = shut_index("26")
    elif name == "42":
        all_selfc = all_selfc2
        file = file2
        res_snd,res_fnd,res_sd,res_ld,res_pd,res_fd = shut_index("42")
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
    plt.savefig(file+'aheadtime.eps',format='eps',dpi=1000)
    plt.savefig(file+'aheadtime.svg',format='svg',dpi=1000)
    print("平均提前 = %fs"%np.average(chzhi))
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
    row_num = sheet1.nrows
    A=[]
    for i in range(row_num):
        A.append(sheet1.row_values(i))
    a = []
    for i in List:
        a.append(A[i][0])
    return a


num = 20
num1 = 26    # 26 42
num2 = 42
filepath1 ='/home/jack/音乐/data/data%d/data%d_%d/'%(num,num,num1)
filepath2 ='/home/jack/音乐/data/data%d/data%d_%d/'%(num,num,num2)
file1 = '/home/jack/snap/picture/englishpicture/data%d/data%d_%d/'%(num,num,num1)
file2 = '/home/jack/snap/picture/englishpicture/data%d/data%d_%d/'%(num,num,num2)
file3 = '/home/jack/snap/picture/englishpicture/data%d/'%(num)

if os.path.exists(file1):
    pass
else:
    os.makedirs(file1)
if os.path.exists(file2):
    pass
else:
    os.makedirs(file2)
    
#######################################################################################3
pred_res1 = np.load(filepath1+'pred_rate_res_fB.npz')['pred_res']
pred_rate1 = np.load(filepath1+'pred_rate_res_fB.npz')['pred_rate']
roc_AUC1 = np.load(filepath1+'pred_rate_res_fB.npz')['AUC']
hist1 = pd.read_csv(filepath1+'history_dict%d.csv'%num)

all_selfc1 = np.load(filepath1+'selfc%d_all.npz'%num)['selfc']
pred_resu1 = np.load(filepath1+'pred_all_result%d.npz'%num)['pred_result3']
real_lab1 = np.load(filepath1+'all_xy%d.npz'%num)['data'][:,:,-2]
time1 = np.load(filepath1+'all_xy%d.npz'%num)['data'][:,:,-1]
all_shut1 = np.load(filepath1+'all%d.npz'%num)['all_shut']

density1 = np.load(filepath1+'all_xy%d.npz'%num)['data'][:,:,1]
##############################################################################################
pred_res2 = np.load(filepath2+'pred_rate_res_fB.npz')['pred_res']
pred_rate2 = np.load(filepath2+'pred_rate_res_fB.npz')['pred_rate']
roc_AUC2 = np.load(filepath2+'pred_rate_res_fB.npz')['AUC']
hist2 = pd.read_csv(filepath2+'history_dict%d.csv'%num)

all_selfc2 = np.load(filepath2+'selfc%d_all.npz'%num)['selfc']
pred_resu2 = np.load(filepath2+'pred_all_result%d.npz'%num)['pred_result3']
real_lab2 = np.load(filepath2+'all_xy%d.npz'%num)['data'][:,:,-2]
time2 = np.load(filepath2+'all_xy%d.npz'%num)['data'][:,:,-1]
all_shut2 = np.load(filepath2+'all%d.npz'%num)['all_shut']

density2 = np.load(filepath2+'all_xy%d.npz'%num)['data'][:,:,1]
##############################################################################################
res_snd,res_fnd,res_sd,res_ld,res_pd,res_fd = shut_index("26")
all_train_shut,all_val_shut,all_test_shut = split_shut()

Set_snd,Set_fnd,Set_sd,Set_ld,Set_fd = find_jiaoji("test")

#b = turn_shut(res_fnd)

chzhi = draw_chazhi("42")
chzhi = draw_chazhi("26")
analy_res_rate("26")

#analy_loss()
     
Draw1(962)
Draw1(524)
Draw1(407)
Draw1(86)
Draw1(221)

Draw2(962)
Draw2(524)
Draw2(407)
Draw2(86)
Draw2(221)

Draw12(407,"26")
Draw12(86,"26")
Draw12(221,"26")

#for i in Set_sd:
#    Draw(i)

#*************************结束预测结果的分析******************************************

