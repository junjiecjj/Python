#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 20:19:45 2023

@author: jack
"""

import numpy as np




T=np.dtype([("name",np.str_,40),("num",np.int32),("price",np.float64)])
                                   #T等价于 np.int32,np.int64,np.float32
print(T["name"])#-------------------------><U40
print(T["num"])#-------------------------->int32
print(T["price"])#------------------------>float64
print(type(T))#---------------------------><class 'numpy.dtype'>

products=np.array([("DVD", 42, 3.14),("Butter", 13, 2.72)],dtype=T)



#运用for in对数组进行遍历
for i in range(products.size):    #products.size表示products里的元素个数，相当于len(products)
    # print(products[i])----------------->('DVD', 42, 3.14)  ('Butter', 13, 2.72)
    for j in range(len(products[i])):     #products[i].size=1，j只取0
        print(products[i][j])#------------>DVD   Butter









import numpy as np

# int8，int16，int32，int64 可替换为等价的字符串 'i1'，'i2'，'i4'，以及其他。
student = np.dtype([('name','S20'),  ('age',  'i1'),  ('marks',  'f4')])
a = np.array([('abc',  21,  50),('xyz',  18,  75)], dtype = student)
print(a)
# [(b'abc', 21, 50.) (b'xyz', 18, 75.)]
# 每个内建类型都有一个唯一定义它的字符代码：
# 'b'：布尔值
# 'i'：符号整数
# 'u'：无符号整数
# 'f'：浮点
# 'c'：复数浮点
# 'm'：时间间隔
# 'M'：日期时间
# 'O'：Python 对象
# 'S', 'a'：字节串
# 'U'：Unicode
# 'V'：原始数据(void)



#NumPy提供了那些数据类型
#int8 int16 int32 int64 float32(单精度)、float64或float（双精度）
#bool

a = np.array([['a',1,2],[3,4,5],[6,7,8]])
print(a)
print("******************************")

#定义表头，用来表示数组中的数据类型
t = np.dtype([('name', np.str_,20),('age', np.int8),('salary', np.float32)])

b = np.array([('a',1,2),(3,4,5),(6,7,8)],dtype=t)
print(b)
print("******************************")

items = np.array([('Bill',30,12345),('Mary',24,8000)])
print(items)
print("********************************")
items1 = np.array([('Bill',30,12345),('Mary',24,8000)],dtype=t)

# [['a' '1' '2']
#  ['3' '4' '5']
#  ['6' '7' '8']]
# ******************************
# [('a', 1, 2.) ('3', 4, 5.) ('6', 7, 8.)]
# ******************************
# [['Bill' '30' '12345']
#  ['Mary' '24' '8000']]
# ********************************





























