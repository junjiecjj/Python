# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 15:37:46 2018

@author: bhguo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from MDSplus import *
from MDSplus import Connection

shot = 75000
coon  = Connection('202.127.204.12')   # connect the MDSplus
coon.openTree('east', shot)   # open tree and shot                 
ip = coon.get(r'\IPG')   # read data
t = coon.get('dim_of(\IPG)') # read time data

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