#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:46:06 2019

@author: jack

此代码是测试利用lmtipref求导找到平顶端时刻可行性，结果是非靠谱
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#B的第0列为炮号，第1列为记录的破裂时间，第2列为标签，第3列为d(pcrl01)/d(t)最大值时刻
#第4列为lmtipref导数为0的第一个点时刻,第5列为lmtipref导数为0的最后一个点时刻,
#第6列为lmtipref最后时刻点,第7列为dfsdev的最后时刻
B = np.load('/home/jack/snap/Bdis.npz')['Bdis']
fig, axs=plt.subplots(11,1,figsize=(6,14))
c1 = B[:491,1]-B[:491,3]#-0.027422389301594893
c2 = B[:491,1]-B[:491,5]#-0.0667368494874678
c3 = B[:491,1]-B[:491,6]#-0.09622727711158598
c4 = B[:491,1]-B[:491,7]#-0.09472666611741132
c5 = B[:491,3]-B[:491,5]#-0.03931446018587291
c6 = B[:491,3]-B[:491,6]#-0.06880488780999108
c7 = B[:491,3]-B[:491,7]#-0.06730427681581641
c8 = B[:491,5]-B[:491,4]#4.0275871588740895
c9 = B[:491,6]-B[:491,5]#0.029490427624118166
c10 = B[:491,5]-B[:491,7]#-0.027989816629943504
c11 = B[:491,6]-B[:491,7]#0.0015006109941746615
for i,c in zip(range(1,12),[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11]):
    print("c%d 平均值为:%f"%(i,np.average(c)))

axs[0].plot(np.arange(491),c1,c = 'r' ,label = 'c1')
axs[0].legend()
axs[1].plot(np.arange(491),c2,c = 'r' ,label = 'c2')
axs[1].legend()
axs[2].plot(np.arange(491),c3,c = 'r' ,label = 'c3')
axs[2].legend()
axs[3].plot(np.arange(491),c4,c = 'r' ,label = 'c4')
axs[3].legend()
axs[4].plot(np.arange(491),c5,c = 'r' ,label = 'c5')
axs[4].legend()
axs[5].plot(np.arange(491),c6,c = 'r' ,label = 'c6')
axs[5].legend()
axs[6].plot(np.arange(491),c7,c = 'r' ,label = 'c7')
axs[6].legend()
axs[7].plot(np.arange(491),c8,c = 'r' ,label = 'c8')
axs[7].legend()
axs[8].plot(np.arange(491),c9,c = 'r' ,label = 'c9')
axs[8].legend()
axs[9].plot(np.arange(491),c10,c = 'r' ,label = 'c10')
axs[9].legend()
axs[10].plot(np.arange(491),c11,c = 'r' ,label = 'c11')
axs[10].legend()

D = np.load('/home/jack/snap/Bsafe.npz')['safe']
fig, axs=plt.subplots(11,1,figsize=(6,14))
d1 = D[491:,1]-D[491:,3]#-1.4274396818915263
d2 = D[491:,1]-D[491:,5]#-0.11212468521237737
d3 = D[491:,1]-D[491:,6]#-1.735148357585225
d4 = D[491:,1]-D[491:,7]#-1.7419764458032796
d5 = D[491:,3]-D[491:,5]#1.3153149966791489
d6 = D[491:,3]-D[491:,6]#-0.30770867569369853
d7 = D[491:,3]-D[491:,7]#-0.31453676391175334
d8 = D[491:,5]-D[491:,4]#6.602014689213868
d9 = D[491:,6]-D[491:,5]#1.6230236723728473
d10 = D[491:,5]-D[491:,7]#-1.6298517605909022
d11 = D[491:,6]-D[491:,7]#-0.006828088218054852


axs[0].plot(np.arange(680),d1,c = 'r' ,label = 'd1')
axs[0].legend()
axs[1].plot(np.arange(680),d2,c = 'r' ,label = 'd2')
axs[1].legend()
axs[2].plot(np.arange(680),d3,c = 'r' ,label = 'd3')
axs[2].legend()
axs[3].plot(np.arange(680),d4,c = 'r' ,label = 'd4')
axs[3].legend()
axs[4].plot(np.arange(680),d5,c = 'r' ,label = 'd5')
axs[4].legend()
axs[5].plot(np.arange(680),d6,c = 'r' ,label = 'd6')
axs[5].legend()
axs[6].plot(np.arange(680),d7,c = 'r' ,label = 'd7')
axs[6].legend()
axs[7].plot(np.arange(680),d8,c = 'r' ,label = 'd8')
axs[7].legend()
axs[8].plot(np.arange(680),d9,c = 'r' ,label = 'd9')
axs[8].legend()
axs[9].plot(np.arange(680),d10,c = 'r' ,label = 'd10')
axs[9].legend()
axs[10].plot(np.arange(680),d11,c = 'r' ,label = 'd11')
axs[10].legend()