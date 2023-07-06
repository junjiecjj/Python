#!/usr/bin/env  python3
#!-*-coding=utf-8-*-
#########################################################################
# File Name: select_alreadhaveSafe.py
# Author: 陈俊杰
# mail: 2716705056@qq.com
# Created Time: 2020.01.15
'''
此程序的功能是：
'''
#########################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, time

'''
file = '/home/jack/tmp/'
safe1 = np.load(file + 'safe1.npz')['safe1']
safe2 = np.load(file + 'safe2.npz')['safe2']
safe3 = np.load(file + 'safe3.npz')['safe3']
safe4 = np.load(file + 'safe4.npz')['safe4']

allsafe = np.vstack([safe1,safe2,safe3,safe4])


for i in range(len(allsafe)):
    for j in [5,6,7,9]:
        allsafe[i,j] = np.around(allsafe[i,j]*1000)/1000


pd_data = pd.DataFrame(allsafe, columns=['shot','W_aminor','W_flat','W_useless',\
                                    'W_disrupt','flat_sta','flat_end',\
                                    'disru_time','W_density','dens_time'])
pd_data.to_csv('/home/jack/tmp/allsafe1.csv',index=False)




A = np.array(pd.read_csv('/home/jack/数据筛选/last8.csv'))
allsafe = np.array(pd.read_csv('/home/jack/数据筛选/allsafe_exceedGW.csv'))

#allsafe[:,4] = -2

B = np.zeros((831,21))
B[:,:10] = allsafe[:]


C = np.vstack([A[:972],B])

for i in range(len(C)):
    for j in [5,6,7,9,10,11,13,14,15,16,17,18,19,20]:
        C[i,j] = np.around(C[i,j]*1000)/1000


pd_data = pd.DataFrame(C, columns=['shot','W_aminor','W_flat','W_useless',\
                                    'W_disrupt','flat_sta','flat_end',\
                                    'disru_time','W_density','dens_time',\
                                    'R_flat_top','R_disrT','flat_len',\
                                   '0.3nGW','0.4nGW','0.5nGW','0.6nGW',\
                                   '0.7nGW','0.8nGW','0.9nGW','1.0nGW'])


pd_data.to_csv('/home/jack/数据筛选/last10.csv',index=False)

'''
A = np.array(pd.read_csv('/home/jack/数据筛选/last8.csv'))
allsafe = np.array(pd.read_csv('/home/jack/数据筛选/allsafe_exceedGW.csv'))

#allsafe[:,4] = -2
C = np.zeros((allsafe.shape[0], A.shape[1]))
C[:,:10] = allsafe


D = np.vstack([A[:972],C])

for i in range(len(D)):
    for j in [5,6,7,9,10,11,13,14,15,16,17,18,19,20]:
        D[i,j] = np.around(D[i,j]*1000)/1000

pd_data = pd.DataFrame(D, columns=['shot','W_aminor','W_flat','W_useless',\
                                    'W_disrupt','flat_sta','flat_end',\
                                    'disru_time','W_density','dens_time',\
                                    'R_flat_top','R_disrT','flat_len',\
                                   '0.3nGW','0.4nGW','0.5nGW','0.6nGW',\
                                   '0.7nGW','0.8nGW','0.9nGW','1.0nGW'])

pd_data.to_csv('/home/jack/数据筛选/last10.csv',index=False)





















