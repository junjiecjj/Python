#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:46:06 2019

@author: jack

这是查看某一炮的pcrl01,lmtipref,dfsdev,aminor的文件
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math


file = '/home/jack/snap/'
#safe = np.load(file+'Bsafe8.npz')['safe']

#disrupt=np.load(file+'Bdisrup8.npz')['disruption']

#pGreenW = np.load(file+'Bgreen8.npz')['gw']

#density=np.load(file+'Bdensity8.npz')['density']

real_density = np.load(file+'BRealdensity5.npz')['Realdensity']

#short=np.load(file+'Bshort8.npz')['short']

#without=np.load(file+'Bwithout8.npz')['without']

'''
Del = [80025,80176,80314,80324,80376,80377,80378,80382,80477,80497,80512,80599,80601,\
       80602,80603,80604,80607,80611,80645,80651,80652,80658,80663,80669,80670,80671,\
       80672,80697,80856,80863,80929,81035,81039,81040,81064,81070,81073,81089,81090,\
       81091,81097,81138,81139,81162,81164,81207,81280,81281,81297,81358,81447,81449,\
       81473,81492,81501,81515,81549,81550,81568,81579,81587,81673,81674,\
       81675,81683,82911,82930,82989,83219,83324,83350,83352,83353,83377,83411,83661,\
       83676,83678,83689,83744,83754,83805,83826,83843,83912,83981,84027,84078,84511,\
       84519,84528,84547,84617,84695,84844,84916,84959,84960,85061,85104,85173,85179,\
       85181,85253,85292,85294,85295,85296,85297,85332,85333,85334,85341,85342,85352,\
       85362,85365,85389,85460,85478,85500,85503,85504,85509,85511,85551,85552,85556,\
       85557,85558,85559,85560,85561,85562,85563,85564,85565,85566,85568,85569,85570,\
       85572,85573,85574,85575,85576,85577,85578,85677,85680,85682,85683,85743,85754,\
       85777,86031,86295,86393,86445,86446,86447,86450,86451,86456,86525,86527,86535,\
       86538,86549,86655,86656,86713,86841,87152,87283,87499,87531,87661,87704,87705,\
       87707,88147,88284,80046,80056,80161,80297,80375,80472,80474,80479,80598,80903,\
       81054,81287,82680,83847,83874,83980,84081,84651,84768,85281,85555,85689,86528,\
       86605,87493,87494,87500,88142]


Den = density.copy()
for i in Del:
    print("%d.."%i)
    j = np.where(Den==i)[0][0]
    Den = np.delete(Den,j,axis=0)

np.savez_compressed('/home/jack/snap/BRealdensity8.npz',Realdensity=Den)
'''

pd_data = pd.DataFrame(real_density,columns=['shot','W_aminor','W_flat','W_useless',\
                                    'W_disrupt','flat_sta','flat_end',\
                                    'disru_time','W_density','dens_time'])
pd_data.to_csv('/home/jack/数据筛选/density5.csv',index=False)
