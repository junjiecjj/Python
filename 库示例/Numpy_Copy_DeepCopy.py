#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:54:54 2022

@author: jack
"""
import numpy as np
import copy


print("============================================== 8 =========================================================")
#1.赋值/引用
a = np.array([1,2,3,4])
b = a
print("a = \n{},\nb = \n{} ".format(a,b, ))

a[2] = 0
print("a = \n{},\nb = \n{} ".format(a,b, ))


#2.数组切片（也就是视图/引用）
print("============================================== 9 =========================================================")
c = np.array([1,2,3,4])
d = c[:]
print("c = \n{},\nd = \n{} ".format(c,d ))

d[0] = 0
print("c = \n{},\nd = \n{} ".format(c,d ))

c[1] = 1
print("c = \n{},\nd = \n{} ".format(c,d ))


c = np.arange(6).reshape(2,3)
d = c[:]  #c,d一起变
print("c = \n{},\nd = \n{} ".format(c,d ))

d[0,1] = 32
print("c = \n{},\nd = \n{} ".format(c,d ))

c[1,1] = 54
print("c = \n{},\nd = \n{} ".format(c,d ))


#3.copy()【类似列表中的deepcopy()】
print("============================================== 10 =========================================================")
import numpy as np

a = np.array([1,2,3,4])
b = a.copy()
print("a = \n{},\nb = \n{} ".format(a,b, ))

a[0] = 32
print("a = \n{},\nb = \n{} ".format(a,b, ))

b[0] = 0
print("a = \n{},\nb = \n{} ".format(a,b, ))

c = np.arange(6).reshape(2,3)
d = c.copy() #c,d无关联
print("c = \n{},\nd = \n{} ".format(c,d ))

d[0,1] = 32
print("c = \n{},\nd = \n{} ".format(c,d ))

c[1,1] = 54
print("c = \n{},\nd = \n{} ".format(c,d ))






