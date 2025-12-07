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

a = np.arange(1024).reshape(4,256)
b = np.arange(6).reshape(3,2)*1.1
c=np.arange(24).reshape(4,6)
d = [1,2,3,4,5.5,6,7.7,8,9.12]

C = {}
C[1] = a
C[2] = b
#C[3] = c
#C[4] = d


np.savez_compressed('./test.npz',C=C, c=c, d=d)
CC = np.load('./test.npz')['C'][()]
cc = np.load('./test.npz')['c']
dd = np.load('./test.npz')['d']



"""
In[5]: CC
Out[5]:
array({1: array([[0, 1, 2],
       [3, 4, 5]]), 2: array([[0. , 1.1],
       [2.2, 3.3],
       [4.4, 5.5]])}, dtype=object)

IN[2]: CC[()]
Out[2]:
{1: array([[0, 1, 2],
        [3, 4, 5]]), 2: array([[0. , 1.1],
        [2.2, 3.3],
        [4.4, 5.5]])}



"""

################################ 2 ########################################
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

A = np.array(pd.read_csv('/home/jack/数据筛选/last6.csv'))

datapath = '/home/jack/Density/'
miss_l = []
for i in range(len(A)):
    a = np.load(datapath+'%d.npz'%(A[i,0]))
    if len(a.files)!=14:
        miss_l.append(A[i,0])

"""

################################# 2 ##################################

############################## 3#########################
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def change_arr(arr):
    print(arr)
    print(2,id(arr),'\n')

    arr[0] = 1
    print(arr)
    print(3,id(arr),'\n')

    arr = arr[:,0:3]
    print(arr)
    print(4,id(arr),'\n')
    return


def change_dict(dic):
    print(dic)
    print(2,id(dic),'\n')

    dic[1] = 'aa'
    print(dic)
    print(3,id(dic),'\n')

    return



A = np.arange(12).reshape(2,6)
print(A)
print(1,id(A),'\n')

change_arr(A)

print(A)
print(5,id(A),'\n')

print("######################")

D = {1:'a',2:'b',3:'c'}
print(D)
print(1,id(D),'\n')

change_dict(D)

print(D)
print(4,id(D),'\n')
'''
############################## 3#########################
################### 4 ############################

'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DIS(object):
    def __init__(self):
        self.a = np.arange(24).reshape(6,4)
        self.b = np.zeros((6,4))
        self.c = np.ones((6,4))
    def change(self):
        arr = self.a[[1,3,5]]
        print('arr  1 :\n',arr,'\n')
        print('self.a  2 :\n',hh.a,'\n')
        '''
        self.b[[1,2,3]] = arr
        print('self.b  1 :\n',self.b,'\n')
        print('self.a  3 :\n',hh.a,'\n')

        self.b[1] = 1
        print('self.b  2 :\n',self.b,'\n')
        print('arr  3 :\n',arr,'\n')
        print('self.a  4 :\n',hh.a,'\n')
        '''
        arr[0] = 11
        print('self.b  3 :\n',self.b,'\n')
        print('arr  4 :\n',arr,'\n')
        print('self.a  5 :\n',hh.a,'\n')
        return

hh = DIS()
print('self.a  1 :\n',hh.a,'\n')

print("#################")
hh.change()
print('self.a  6 :\n',hh.a,'\n')

print('self.b  4 :\n',hh.b,'\n')
'''
################### 4 ############################











