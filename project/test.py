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
from matplotlib.font_manager import FontProperties

A = np.array(pd.read_csv("/home/jack/数据筛选/last4.csv"))

Del = [47148., 47258., 47559., 47600., 47704., 47739., 47740., 47746.,
       47865., 47897., 48203., 48441., 48460., 48463., 48492., 48565.,
       48568., 48656., 48890., 49584., 49748., 50157., 50170., 54987.,
       55391., 55394., 55561., 55579., 55642., 56180., 56222., 56349.,
       56358., 56479., 56704., 56717., 56728., 56734., 56740., 56820.,
       56863., 56919., 56940., 56968., 56971., 57046., 57161., 57176.,
       57226., 57247., 57262., 57381., 57424., 57500., 58946., 60465.,
       60496., 60667., 60930., 60953., 61039., 61133., 61139., 61240.,
       61291., 61320., 61445., 61519., 61535., 61538., 61539., 62087.,
       62129., 62420., 62787., 62848., 62959., 63238., 63463., 64316.,
       64393., 64605., 64717., 64820., 64929., 65835., 66041., 66091.,
       66416., 66906., 67008., 67012., 67016., 67523., 67869., 68943.,
       69236., 69260., 69417., 69634., 69689., 69959., 70119., 70922.,
       70925., 70928., 71011., 71198., 71213., 71269., 71329., 71395.,
       71462., 71500., 71507., 71592., 72612., 72683., 72693., 72767.,
       73158., 73439., 73463., 73576., 74085., 74089., 74113., 74131.,
       74205., 74610., 74746., 74825., 75058., 75464., 75615., 75690.,
       77706., 77754., 77755., 77806., 77810., 77813., 78038., 78642.,
       78650., 78690., 78709., 78711., 78979., 79365., 79535., 79605.,
       79616., 79792., 80023., 80042., 80150., 80338., 80481., 80485.,
       80614., 80806., 81209., 81210., 81250., 81268., 81364., 81367.,
       82892., 82918., 83828., 83829., 83939., 83946., 84199., 84211.,
       84213., 84215., 84330., 84339., 84347., 84452., 84505., 84506.,
       84515., 84520., 84691., 84982., 85042., 85140., 85174., 85354.,
       85384., 85436., 85502., 85641., 85781., 86441., 86443., 87491.,
       88018.]
B = A.copy()
for i in Del:
    print("%d.."%i)
    j = np.where(B==i)[0][0]
    B = np.delete(B,j,axis=0)


pd_data = pd.DataFrame(B,columns=['shot','W_aminor','W_flat','W_useless',\
                                    'W_disrupt','flat_sta','flat_end',\
                                    'disru_time','W_density','dens_time','R_disrupT'])
pd_data.to_csv('/home/jack/数据筛选/last5.csv',index=False)
