#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:51:57 2019

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


file ='/home/jack/density/'
shot = 67039
pcrl = np.load(file +'%d.npz'%(shot))['pcrl01']
dfsdev = np.load(file +'%d.npz'%shot)['dfsdev']
aminor = np.load(file +'%d.npz'%shot)['aminor']
vp = np.load(file +'%d.npz'%shot)['vp1']
sxr = np.load(file +'%d.npz'%shot)['sxr23d']
pxu30 = np.load(file +'%d.npz'%shot)['pxuv30']
pxu18 = np.load(file +'%d.npz'%shot)['pxuv18']
kmp = np.load(file +'%d.npz'%shot)['kmp13t']
#pddata = pd.DataFrame(a)
#pddata.to_csv(file+'a.csv',index=False)
A = np.array(pd.read_csv('/home/jack/数据筛选/all.csv'))