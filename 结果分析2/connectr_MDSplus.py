# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 15:37:46 2018

@author: bhguo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MDSplus import *
from MDSplus import Connection

signa = ["\PCRL01","\DFSDEV","\AMINOR","\KMP13T","\IC1","\SXR23D",\
            "\PXUV30","\VP1","\PXUV18","\BETAP","\LI","\q95"]
Dict = ["pcrl01","dfsdev","aminor","kmp13t","ic1","sxr23d",\
            "pxuv30","vp1","pxuv18","betap","LI","q95"]
Tree = ["pcs_east","pcs_east","efitrt_east","east","pcs_east","east",\
            "east","east","east","efit_east","efit_east","efit_east"]
'''
indeX = [65023,65035,65078,65079,65080,65101,65141,65147,65148,65150,65151,65153,65154,65158,65161,65662,65968,66248,66346]
def read_save(shut):
    coon  = Connection('202.127.204.12')   # connect the MDSplus
    coon.openTree("pcs_east", shut)   # open tree and shot                 
    density = coon.get(r'\DFSDEV')   # read data
    print('密度的单位：',coon.get('units(\\DFSDEV)'))
    t = coon.get('dim_of(\DFSDEV)') # read time data
    coon.closeTree('pcs_east', shut)
    data=np.zeros((2,len(t)))
    data[0]=np.array(density)
    data[1]=np.array(t)
    
    plt.plot(data[1],data[0])
    #np.savetxt('/home/jack/data/%d/%d_DFSDEV.txt'%(shut,shut),data)
    return
read_save(51056)

for shut in indeX:
    read_save(shut)
''' 
shot = 67039
fig, axs = plt.subplots(9,2,sharex=True,figsize=(10,8))#figsize=(6,8)


coon  = Connection('202.127.204.12')   # connect the MDSplus
coon.openTree('pcs_east', shot)   # open tree and shot                 
pcrl01 = coon.get(r'\PCRL01')   # read data
t0 = coon.get('dim_of(\PCRL01)') # read time data
print('PCRL01的单位：',coon.get('units(\\PCRL01)'))
axs[0,0].plot(t0,pcrl01)

coon.openTree('east', shot)   # open tree and shot                 
ipm = coon.get(r'\ipm')   # read data
t1 = coon.get('dim_of(\ipm)') # read time data
print('\ipm的单位：',coon.get('units(\\ipm)'))
axs[1,0].plot(t1,ipm)

coon.openTree('pcs_east', shot)   # open tree and shot                 
lmtipref = coon.get(r'\LMTIPREF')   # read data
t2 = coon.get('dim_of(\LMTIPREF)') # read time data
print('\LMTIPREF的单位：',coon.get('units(\\LMTIPREF)'))
axs[2,0].plot(t2,lmtipref)

coon.openTree('pcs_east', shot)   # open tree and shot                 
dfsdev = coon.get(r'\DFSDEV')   # read data
t3 = coon.get('dim_of(\DFSDEV)') # read time data
print('DFSDEV的单位：',coon.get('units(\\DFSDEV)'))
axs[3,0].plot(t3,dfsdev)


coon.openTree('efitrt_east', shot)   # open tree and shot                 
aminor = coon.get(r'\AMINOR')   # read data
t4 = coon.get('dim_of(\AMINOR)') # read time data
print('AMINOR的单位：',coon.get('units(\\AMINOR)'))
axs[4,0].plot(t4,aminor)

coon.openTree('east', shot)   # open tree and shot                 
aminor = coon.get(r'\VP1')   # read data
t5 = coon.get('dim_of(\VP1)') # read time data
print('\VP1的单位：',coon.get('units(\\VP1)'))
axs[5,0].plot(t5,aminor)


coon.openTree('east', shot)   # open tree and shot                 
sxr23d = coon.get(r'\SXR23D')   # read data
t6 = coon.get('dim_of(\SXR23D)') # read time data
print('SXR23D的单位：',coon.get('units(\\SXR23D)'))
axs[6,0].plot(t6,sxr23d)


coon.openTree('east', shot)   # open tree and shot                 
sxr18v = coon.get(r'\SXR18V')   # read data
t7 = coon.get('dim_of(\SXR18V)') # read time data
print('SXR18V的单位：',coon.get('units(\\SXR18V)'))
axs[7,0].plot(t7,sxr18v)


coon.openTree('east', shot)   # open tree and shot                 
pxuv30 = coon.get(r'\PXUV30')   # read data
t8 = coon.get('dim_of(\PXUV30)') # read time data
print('\PXUV30的单位：',coon.get('units(\\PXUV30)'))
axs[0,1].plot(t8,pxuv30)


coon.openTree('east', shot)   # open tree and shot                 
pxuv18 = coon.get(r'\PXUV18')   # read data
t9 = coon.get('dim_of(\PXUV18)') # read time data
print('\PXUV18的单位：',coon.get('units(\\PXUV18)'))
axs[1,1].plot(t9,pxuv18)


coon.openTree('east', shot)   # open tree and shot                 
kmp13 = coon.get(r'\KMP13T')   # read data
t10 = coon.get('dim_of(\KMP13T)') # read time data
print('KMP13T的单位：',coon.get('units(\\KMP13T)'))
axs[2,1].plot(t10,kmp13)


coon.openTree('east', shot)   # open tree and shot                 
ciiil3 = coon.get(r'\CIIIL3')   # read data
t11 = coon.get('dim_of(\CIIIL3)') # read time data
print('\CIIIL3的单位：',coon.get('units(\\CIIIL3)'))
axs[3,1].plot(t11,ciiil3)


coon.openTree('efitrt_east', shot)   # open tree and shot                 
kmp13 = coon.get(r'\q95')   # read data
t13 = coon.get('dim_of(\q95)') # read time data
print('\q95的单位：',coon.get('units(\\q95)'))
axs[4,1].plot(t13,kmp13)


coon.openTree('efitrt_east', shot)   # open tree and shot                 
sxr23d = coon.get(r'\BETAP')   # read data
t14 = coon.get('dim_of(\BETAP)') # read time data
print('\BETAP的单位：',coon.get('units(\\BETAP)'))
axs[5,1].plot(t14,sxr23d)


coon.openTree('efitrt_east', shot)   # open tree and shot                 
kmp13 = coon.get(r'\LI')   # read data
t15 = coon.get('dim_of(\LI)') # read time data
print('\LI的单位：',coon.get('units(\\LI)'))
axs[6,1].plot(t15,kmp13)


coon  = Connection('202.127.204.12')   # connect the MDSplus
coon.openTree('efitrt_east', shot)   # open tree and shot                 
area = coon.get(r'\AREA')   # read data
t16 = coon.get('dim_of(\AREA)') # read time data
print('\AREA的单位：',coon.get('units(\\AREA)'))
axs[7,1].plot(t16,area)

coon  = Connection('202.127.204.12')   # connect the MDSplus
coon.openTree('east', shot)   # open tree and shot                 
pbem = coon.get(r'\VBM10')   # read data
t17 = coon.get('dim_of(\VBM10)') # read time data
print('\Pbrem的单位：',coon.get('units(\\VBM10)'))
axs[8,1].plot(t17,pbem )



'''
ip = pd.DataFrame(list(ip))
t = pd.DataFrame(list(t))
ip = pd.DataFrame(ip, dtype=np.float) # 把数据类型变成float
ip.columns = ['ip']
t = pd.DataFrame(t, dtype=np.float)
t.columns = ['t']
Data = t.join(ip) # 合拼两个DataFrame

t_ip_max = Data[Data.ip == max(Data['ip'])]['t']
t_80_20 = Data[(Data.ip <= (0.8 * max(Data['ip']))) & (Data.ip >= (0.2 * max(Data['ip']))) & (Data.t > t_ip_max.data[0])]['t']
t_80 = t_80_20.data[0]
t_20 = t_80_20.data[-1]
t_CQ = (t_20 - t_80) / 0.6 * 1000
'''