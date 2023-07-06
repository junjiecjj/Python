#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:39:54 2022

@author: jack

使用 Numba 让 Python 计算得更快：两行代码，提速 13 倍


"""

from numba import njit




def monotonically_increasing(a):
    b = a.copy()
    max_value = 0
    for i in range(len(b)):
        if a[i] > max_value:
            max_value = b[i]
        b[i] = max_value
    return b
        

@njit  
def monotonically_increasing_numba(a):
    b = a.copy()
    max_value = 0
    for i in range(len(b)):
        if a[i] > max_value:
            max_value = b[i]
        b[i] = max_value
    return b
        


a = [1, 2, 1, 3, 3, 5, 4, 6]
b = monotonically_increasing(a)


b1 = monotonically_increasing_numba(a)

