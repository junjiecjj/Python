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

indeX = [65023,65035,65078,65079,65080,65101,65141,65147,65148,65150,65151,65153,65154,65158,65161,65662,65968,66248,66346]
def read_save(shut):
    coon  = Connection('202.127.204.12')   # connect the MDSplus
    coon.openTree('pcs_east', shut)   # open tree and shot                 
    density = coon.get(r'\DFSDEV2')   # read data
    t = coon.get('dim_of(\DFSDEV2)') # read time data
    data=np.zeros((2,len(t)))
    data[0]=np.array(density)
    data[1]=np.array(t)
    
    plt.plot(data[1],data[0])
    #np.savetxt('/home/jack/data/%d/%d_DFSDEV.txt'%(shut,shut),data)
    return
read_save(69592)
''' 
for shut in indeX:
    read_save(shut)
 
shot = 65821
coon  = Connection('202.127.204.12')   # connect the MDSplus
coon.openTree('pcs_east', shot)   # open tree and shot                 
ip = coon.get(r'\DFSDEV')   # read data
t = coon.get('dim_of(\DFSDEV)') # read time data
plt.plot(t,ip)


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