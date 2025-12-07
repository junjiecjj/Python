#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:51:57 2019

@author: jack
"""

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

print("###################################\n")

D = {1:'a',2:'b',3:'c'}
print(D)
print(1,id(D),'\n')

change_dict(D)

print(D)
print(4,id(D),'\n')






