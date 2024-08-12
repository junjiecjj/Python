#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:21:38 2023

@author: jack
"""


#============================================================================================================
#                         写入字典
#============================================================================================================


#  写入pickle文件 pickle.dump（obj,file,protocal） 序列化对象，并将结果数据流写入到文件对象中。参数protocol是序列化模式，默认值为0，表示以文本的形式序列化。protocol的值还可以是1或2，表示以二进制的形式序列化。
import pickle
d = dict(name='Bob', age=21, score=99)
with open('val.pickle', 'wb') as f:
    pickle.dump(d, f)


# 读取pickle文件  pickle.load(file) 反序列化对象。将文件中的数据解析为一个Python对象。
# 方法一
import pickle
files = open('val.pickle','rb')  # 以二进制读模式（rb）打开pkl文件
data = pickle.load(files)  # 读取存储的pickle文件
print(type(data))   # 查看数据类型
for i, (k, v) in enumerate(data.items()):   # 读取字典中前十个键值对
    if i in range(0, 10):
        print(k, v)


#============================================================================================================
#                         写入 numpy
#============================================================================================================


import pickle
import numpy as np


nup = np.random.randint(low = 0, high = 100, size = (4, 5))*1.1
with open('numpy.pickle', 'wb') as f:
    pickle.dump(nup, f)


import pickle
files = open('numpy.pickle','rb')  # 以二进制读模式（rb）打开pkl文件
nup1 = pickle.load(files)  # 读取存储的pickle文件
print(type(nup1))   # 查看数据类型

print(f"nup = \n{nup}")



#============================================================================================================
#                         写入 numpy 字典
#============================================================================================================


import pickle
import numpy as np


nup = np.random.randint(low = 0, high = 100, size = (4, 5))*1.1
nupp = np.random.randint(low = 0, high = 100, size = (4, 5))


A = {'a':nup1, 'b':nupp }

with open('numpy.pickle', 'wb') as f:
    pickle.dump(A, f)


import pickle
files = open('numpy.pickle','rb')  # 以二进制读模式（rb）打开pkl文件
nup1 = pickle.load(files)  # 读取存储的pickle文件
print(type(nup1))   # 查看数据类型

print(f"nup1 = \n{nup1}")



#============================================================================================================
#                         写入 tensor
#============================================================================================================
import pickle
import numpy as np
import torch

ts = torch.randint(low = 0, high = 100, size = (4, 5))*1.2
with open('tensor.pickle', 'wb') as f:
    pickle.dump(ts, f)

files = open('tensor.pickle','rb')  # 以二进制读模式（rb）打开pkl文件
ts1 = pickle.load(files)  # 读取存储的pickle文件
print(type(ts1))   # 查看数据类型
print(f"ts1 = \n{ts1}")

#============================================================================================================
#                         写入 tensor + numpy 字典
#============================================================================================================
import pickle
import numpy as np
import torch

ts = torch.randint(low = 0, high = 100, size = (4, 5))*1.2
nupp = np.random.randint(low = 0, high = 100, size = (4, 5))
Ats = {'a':ts, 'b':nupp }

with open('tensor.pickle', 'wb') as f:
    pickle.dump(Ats, f)

files = open('tensor.pickle','rb')  # 以二进制读模式（rb）打开pkl文件
ts1 = pickle.load(files)  # 读取存储的pickle文件
print(type(ts1))   # 查看数据类型
print(f"ts1 = \n{ts1}")


















