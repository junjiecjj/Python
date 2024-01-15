#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:20:43 2022

https://blog.51cto.com/u_15365984/4762422

@author: jack
"""

import random
import numpy as np
import torch
#我这里考虑输入是 ​​torch.Tensor​​的一个 只包含0 ，1 元素的张量
#对于输入是numpy或者0，1字符串的方法就更简单了，总之都先要将输入处理成为 0，1字符串，例如​​“1010”​​
#首先构造一个输入：

a = [1 for i in range(16)]
b = [0 for i in range(16)]
a.extend(b)
random.shuffle(a)
a = np.array(a)
a = torch.Tensor(a)


#将 a 处理成为 0，1字符串：
a = str(a.numpy().tolist())[1:-1].replace('.0','').replace(',','').replace(' ','')


#构造添加CRC码的方法:
def add_crc(wm):
    a = bytes(full_wm, encoding='utf-8')
    print(f"{len(a)}:{a}")
    a = binascii.crc32(a)
    a = bin(a)
    a = str(a)[2:]
    padding = 32-len(a)
    for i in range(padding):
        a = '0'+a
    print(f"{len(a)}:{a}")
    crc = torch.Tensor([int(i) for i in a])
    return torch.cat([wm,crc],dim=0)

#构造CRC校验的方法
def verify_crc(wm):
    #32位CRC校验
    full_wm = str(wm.numpy().tolist())[1:-1].replace('.0','').replace(',','').replace(' ','')
    wm = full_wm[:-32]
    crc = full_wm[-32:]
    # a = int(wm,2) #转换为一个数字
    # a = bin(a)
    a = bytes(wm, encoding='utf-8')
    a = binascii.crc32(a)
    if a == int(crc,2):
        return True
    else:
        return False



a = add_crc(a)
print(a)
results = verify_crc(a)
print(results)

a = add_crc(a)
print(a)
a[12:15] = 0. # 加入扰动
results = verify_crc(a)
print(results)
