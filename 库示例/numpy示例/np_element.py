#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:04:03 2023

@author: jack
"""


import numpy as np





G = 122
w = np.random.randn(12).reshape(3,4)*0.1
a = (w*G + 1)/2
b = np.where(a < 0, 0, a)
p = np.where(b > 1, 1, b)


c = np.ones((1000, 100))*0.5
f1 = np.frompyfunc(lambda x : int(np.random.binomial(1, x, 1)[0]), 1, 1)
d = f1(c)
print(d.mean())


f1 = np.frompyfunc(lambda x : x if x<5 else x-1, 1, 1)


p = min(1, max(0, (w*G+1)/2  ))







##==================================================================================================================

import torch


G = 122
w = torch.randn(3,4) * 0.1
a = (w*G + 1)/2
b = torch.where(a < 0, 0, a)
p = torch.where(b > 1, 1, b)

p = torch.ones((1000,100))*0.2
d = torch.bernoulli(p)
print(d.mean())
e = torch.where(d < 1, -1, d)
print(e.mean())



## B比特的 stochastic rounding (SR), np
def SR_np(param):
    p = param - np.floor(param)
    # print(p)
    param = np.floor(param) + torch.bernoulli(torch.tensor(p)).numpy()
    return param



def Quantization1bits_NP_int(params,  B = 8, rounding = "nr"):
    G =  2**(B - 1)
    p = (params * G + 1)/2
    p = np.clip(p, a_min = 0, a_max = 1, )
    Int = torch.bernoulli(torch.tensor(p)).numpy().astype('int8')
    return Int


def deQuantization1bits_NP_int(bin_recv,  B = 8,  ):
    G =  2**(B - 1)
    param_recv = np.where(bin_recv < 1, -1, bin_recv).astype('float32')/G
    return param_recv




a = np.random.randn(10,)*0.001


B = 8
G =  2**(B - 1)
p = (a*G + 1)/2
# p1 = np.where(p < 0, 0, p)
# p2 = np.where(p1 > 1, 1, p1)
p = np.clip(p, a_min = 0, a_max = 1, )
Int = torch.bernoulli(torch.tensor(p)).numpy().astype('int32')
# param = np.where(param < 1, -1, param)/G

binary_send = np.zeros(Int.size, dtype = np.int8 )
for idx, num in enumerate(Int):
    binary_send[idx] = int(np.binary_repr(num, width = 1))


##==de dequan

binary_recv = binary_send.copy()

param_recv = np.where(binary_recv < 1, -1, binary_recv).astype('float32')/G


a1 = Quantization1bits_NP_int(a)

a2 = deQuantization1bits_NP_int(a1)







































































































































































































































































































































































































































































































































