#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:54:54 2022

@author: jack


本程序主要演示了list和numpy的深浅拷贝时，数值和shape的关联问题。

"""
import numpy as np
import copy


print("==============================================  一维数组 =========================================================")
#1.赋值/引用
print("============================================== #1.赋值/引用 =========================================================")
import numpy as np
a = np.array([1,2,3,4])
b = a   # a,b 值关联
print("a = \n{},\nb = \n{} ".format(a,b, ))

b[2] = 0
print("a = \n{},\nb = \n{} ".format(a,b, ))

a[1] = 44
print("a = \n{},\nb = \n{} ".format(a,b, ))

# a,b 的shape 同步变化
b.shape = 2,2
print("a = \n{},\nb = \n{} ".format(a,b, ))


#2.数组切片（也就是视图/引用）
print("============================================== #2.数组切片（也就是视图/引用） =========================================================")
import numpy as np
a = np.array([1,2,3,4])
b = a[:] # a,b 值关联
print("a = \n{},\nb = \n{} ".format(a,b, ))

b[0] = 77
print("a = \n{},\nb = \n{} ".format(a,b, ))

a[1] = 88
print("a = \n{},\nb = \n{} ".format(a,b, ))

# a,b 的shape 无关
b.shape = 2,2
print("a = \n{},\nb = \n{} ".format(a,b, ))

print("============================================== 3.copy() =========================================================")
import numpy as np
import copy
a = np.array([1,2,3,4])
b = copy.copy(a)  # c,d值无关
print("a = \n{},\nb = \n{} ".format(a,b, ))

b[0] = 0
print("a = \n{},\nb = \n{} ".format(a,b, ))

a[1] = 33
print("a = \n{},\nb = \n{} ".format(a,b, ))

# a,b 的shape 无关
b.shape = 2,2
print("a = \n{},\nb = \n{} ".format(a,b, ))

print("============================================== 4. deepcopy() =========================================================")
import copy
import numpy as np
a = np.array([1,2,3,4])
b = copy.deepcopy(a)# c,d无关
print("a = \n{},\nb = \n{} ".format(a,b, ))

b[0] = 0
print("a = \n{},\nb = \n{} ".format(a,b, ))

a[1] = 33
print("a = \n{},\nb = \n{} ".format(a,b, ))

# a,b 的shape 无关
b.shape = 2,2
print("a = \n{},\nb = \n{} ".format(a,b, ))

print("============================================== 5.reshape() =========================================================")


import copy
import numpy as np
a = np.arange(6)
b = a.reshape(2,3)   # a,b 值关联

print("a = \n{},\nb = \n{} ".format(a,b ))

a[0] = 98
print("a = \n{},\nb = \n{} ".format(a,b ))

b[1,1] = 43
print("a = \n{},\nb = \n{} ".format(a,b ))
# a,b 的shape 无关
b.shape = 3,2
print("a = \n{},\nb = \n{} ".format(a,b ))

import copy
import numpy as np
a = np.arange(6) 
b = a.copy().reshape(2,3)   # a,b  无关联

print("a = \n{},\nb = \n{} ".format(a,b ))

a[0] = 98
print("a = \n{},\nb = \n{} ".format(a,b ))

b[1,1] = 43
print("a = \n{},\nb = \n{} ".format(a,b ))

# a,b 的shape 无关
b.shape = 3,2
print("a = \n{},\nb = \n{} ".format(a,b ))

print("============================================== 6 .view() =========================================================")
import copy
import numpy as np
a = np.arange(6)
b = a.view()   # a,b 值关联

print("a = \n{},\nb = \n{} ".format(a,b ))

a[0] = 98
print("a = \n{},\nb = \n{} ".format(a,b ))

b[1] = 43
print("a = \n{},\nb = \n{} ".format(a,b ))

# a,b 的shape 无关
b.shape = 3,2
print("a = \n{},\nb = \n{} ".format(a,b ))

import copy
import numpy as np
a = np.arange(6)
b = a.view().reshape(2,3)   # a,b 关联

print("a = \n{},\nb = \n{} ".format(a,b ))

a[0] = 98
print("a = \n{},\nb = \n{} ".format(a,b ))

b[1,1] = 43
print("a = \n{},\nb = \n{} ".format(a,b ))


# a,b 的shape 无关
b.shape = 3,2
print("a = \n{},\nb = \n{} ".format(a,b ))

import copy
import numpy as np
a = np.arange(6)
b = a.reshape(2,3).view()  # a,b 值关联

print("a = \n{},\nb = \n{} ".format(a,b ))

a[0] = 98
print("a = \n{},\nb = \n{} ".format(a,b ))

b[1,1] = 43
print("a = \n{},\nb = \n{} ".format(a,b ))

# a,b 的shape 无关
b.shape = 3,2
print("a = \n{},\nb = \n{} ".format(a,b ))

print("==============================================  二维数组 =========================================================")
#1.赋值/引用
print("============================================== #1.赋值/引用 =========================================================")
import copy
import numpy as np
a = np.arange(6).reshape(2,3)
b = a   #c,d值关联
print("a = \n{},\nb = \n{} ".format(a,b ))

b[0,1] = 32
print("a = \n{},\nb = \n{} ".format(a,b ))

a[1,1] = 54
print("a = \n{},\nb = \n{} ".format(a,b ))

# a,b 的shape 同步变化
b.shape = 3,2
print("a = \n{},\nb = \n{} ".format(a,b ))


print("============================================== 2.切片 =========================================================")
import copy
import numpy as np
a = np.arange(6).reshape(2,3)
b = a[:]  #c,d值关联
print("a = \n{},\nb = \n{} ".format(a,b ))

b[0,1] = 32
print("a = \n{},\nb = \n{} ".format(a,b ))

a[1,1] = 54
print("a = \n{},\nb = \n{} ".format(a,b ))

# a,b 的shape 无关
b.shape = 3,2
print("a = \n{},\nb = \n{} ".format(a,b ))


import copy
import numpy as np
a = np.arange(6).reshape(2,3)
b = a[:,:]    #c,d值关联
print("a = \n{},\nb = \n{} ".format(a,b ))
print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))


b[0,1] = 32
print("a = \n{},\nb = \n{} ".format(a,b ))
print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))

a[1,1] = 54
print("a = \n{},\nb = \n{} ".format(a,b ))
print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))

# a,b 的shape 无关
b.shape = 3,2
print("a = \n{},\nb = \n{} ".format(a,b ))

print("============================================== 3.copy() =========================================================")
#3.copy()【类似列表中的deepcopy()】
import copy
import numpy as np
a = np.arange(6).reshape(2,3)
b = copy.copy(a)   #c,d值没关系
print("a = \n{},\nb = \n{} ".format(a,b ))

b[0,1] = 32
print("a = \n{},\nb = \n{} ".format(a,b ))

a[1,1] = 54
print("a = \n{},\nb = \n{} ".format(a,b ))

# a,b 的shape 无关
b.shape = 3,2
print("a = \n{},\nb = \n{} ".format(a,b ))



import numpy as np
import copy
a = np.arange(6).reshape(2,3)
b = a.copy()  #a,b没关系
print("a = \n{},\nb = \n{} ".format(a,b, ))
print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))

a[0] = 32
print("a = \n{},\nb = \n{} ".format(a,b, ))
print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))


b[0] = 0
print("a = \n{},\nb = \n{} ".format(a,b, ))
print("id(a) = \n{},\nid(b) = \n{}, ".format(id(a),id(b), ))

# a,b 的shape 无关
b.shape = 3,2
print("a = \n{},\nb = \n{} ".format(a,b ))


print("============================================== 3. deepcopy() =========================================================")
import copy
import numpy as np
a = np.arange(6).reshape(2,3)
b = copy.deepcopy(a)     #c,d值无关联
print("a = \n{},\nb = \n{} ".format(a,b ))

b[0,1] = 32
print("a = \n{},\nb = \n{} ".format(a,b ))

a[1,1] = 54
print("a = \n{},\nb = \n{} ".format(a,b ))


# a,b 的shape 无关
b.shape = 3,2
print("a = \n{},\nb = \n{} ".format(a,b ))

print("============================================== 4.reshape() =========================================================")


import copy
import numpy as np
a = np.arange(12).reshape(3,4)
b = a.reshape(2,6)   # a,b 关联

print("a = \n{},\nb = \n{} ".format(a,b ))

a[0,0] = 98
print("a = \n{},\nb = \n{} ".format(a,b ))

b[1,1] = 43
print("a = \n{},\nb = \n{} ".format(a,b ))

# a,b 的shape 无关
b.shape = 4,3
print("a = \n{},\nb = \n{} ".format(a,b ))

# 下面两个例子很奇怪
import copy
import numpy as np
a = np.arange(6).reshape(3,2)
b = a.T.reshape(2,3)   # a,b 值关联

print("a = \n{},\nb = \n{} ".format(a,b ))

a[0] = 98
print("a = \n{},\nb = \n{} ".format(a,b ))

b[1,1] = 43
print("a = \n{},\nb = \n{} ".format(a,b ))


# a,b 的shape 无关
#b.shape = 1,6    #error
print("a = \n{},\nb = \n{} ".format(a,b ))

import copy
import numpy as np
a = np.arange(12).reshape(3,4)
b = a.T.reshape(2,6)   #c,d值无关联

print("a = \n{},\nb = \n{} ".format(a,b ))

a[0] = 98
print("a = \n{},\nb = \n{} ".format(a,b ))

b[1,1] = 43
print("a = \n{},\nb = \n{} ".format(a,b ))

# a,b 的shape 无关
b.shape = 4,3
print("a = \n{},\nb = \n{} ".format(a,b ))


import copy
import numpy as np
a = np.arange(12).reshape(3,4)
b = a.copy().reshape(2,6)   #c,d值无关联

print("a = \n{},\nb = \n{} ".format(a,b ))

a[0] = 98
print("a = \n{},\nb = \n{} ".format(a,b ))

b[1,1] = 43
print("a = \n{},\nb = \n{} ".format(a,b ))

# a,b 的shape 无关
b.shape = 4,3
print("a = \n{},\nb = \n{} ".format(a,b ))

import copy
import numpy as np
a = np.arange(12).reshape(3,4)
b = a.T.copy().reshape(2,6)   #c,d值无关联

print("a = \n{},\nb = \n{} ".format(a,b ))

a[0] = 98
print("a = \n{},\nb = \n{} ".format(a,b ))

b[1,1] = 43
print("a = \n{},\nb = \n{} ".format(a,b ))

# a,b 的shape 无关
b.shape = 4,3
print("a = \n{},\nb = \n{} ".format(a,b ))

print("============================================== 5.  view() =========================================================")

import copy
import numpy as np
a = np.arange(6).reshape(2,3)
b = a.view()   # a,b 值关联

print("a = \n{},\nb = \n{} ".format(a,b ))

#修改数据会影响到原始数组：
b[0,0] = 77
print("a = \n{},\nb = \n{} ".format(a,b ))


a[0,1] = 45
print("a = \n{},\nb = \n{} ".format(a,b ))

# a,b 的shape 无关
b.shape = 3,2
print("a = \n{},\nb = \n{} ".format(a,b ))

import copy
import numpy as np
a = np.arange(6).reshape(2,3)
b = a.T.view()   # a,b 值关联

print("a = \n{},\nb = \n{} ".format(a,b ))

a[0,0] = 98
print("a = \n{},\nb = \n{} ".format(a,b ))

b[1,1] = 43
print("a = \n{},\nb = \n{} ".format(a,b ))


# a,b 的shape 无关
b.shape = 3,2
print("a = \n{},\nb = \n{} ".format(a,b ))

import copy
import numpy as np
a = np.arange(6).reshape(2,3)
b = a.copy().view()   # a,b 值无关联

print("a = \n{},\nb = \n{} ".format(a,b ))

a[0,0] = 98
print("a = \n{},\nb = \n{} ".format(a,b ))

b[1,1] = 43
print("a = \n{},\nb = \n{} ".format(a,b ))

# a,b 的shape 无关
b.shape = 3,2
print("a = \n{},\nb = \n{} ".format(a,b ))

import copy
import numpy as np
a = np.arange(6).reshape(2,3)
b = a.T.copy().view()   # a,b 值无关联

print("a = \n{},\nb = \n{} ".format(a,b ))

a[0,0] = 98
print("a = \n{},\nb = \n{} ".format(a,b ))

b[1,1] = 43
print("a = \n{},\nb = \n{} ".format(a,b ))

# a,b 的shape 无关
b.shape = 3,2
print("a = \n{},\nb = \n{} ".format(a,b ))

















