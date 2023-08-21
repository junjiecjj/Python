#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 22:06:28 2023

@author: jack



本文档是记录 Python中的 位运算 和 逻辑运算；
位运算包括 ：bitwise_and、bitwise_or、invert、left_shift、right_shift 以及  "&"、 "~"、 "|" 和 "^"
逻辑运算包括： logical_and()，logical_or(), logical_not(), logical_xor()

当然， 当数据为0-1比特时， 这两者是等价的。
"""


"""
对于二元域上的矩阵运算，矩阵内的乘法就是模2相乘，矩阵内的加法就是异或操作。


#  https://blog.csdn.net/qq_41800366/article/details/88076180
Numpy中也有逻辑判断的函数，即 logical_and()，logical_or(), logical_not(), logical_xor() 这几个函数的大致用法相同，功能上有较大的区别，下面一一详解:
注意，logical_and()，logical_or(), logical_not(), logical_xor()输出的都是0/1 True/False，即逻辑判断的结果；与后面的位运算函数： bitwise_and、bitwise_or、invert、left_shift、right_shift 以及  "&"、 "~"、 "|" 和 "^" 等操作符不同。
bitwise_and、bitwise_or、invert、left_shift、right_shift 以及  "&"、 "~"、 "|" 和 "^" 等操作符是对数据进行按位运算，返回后的值不一定是0/1。

(一) numpy. logical_and()
    numpy.logical_and(x1, x2)
    返回X1和X2与逻辑后的布尔值。
    输入数组。 x1和x2必须具有相同的形状
    返回：
    y：ndarray或bool
    布尔结果，与x1和x2的相应元素上的逻辑AND运算，结果和x1和x2形状相同。 如果x1和x2都是标量，则也返回标量。

    [ logical_and 相当于二进制乘法 ]

(二) numpy.logical_or(x1, x2)
    返回X1和X2或逻辑后的布尔值。

    主要参数：
    x1，x2：array_like
    输入数组。 x1和x2必须具有相同的形状。
    随着版本的变化，函数的参数也在更新，更多详情点击 查看。
    返回：
    y：ndarray或bool
    布尔结果，与x1和x2的相应元素上的逻辑OR运算，结果和x1和x2形状相同。 如果x1和x2都是标量，则也返回标量。


(三) numpy.logical_not(x)
    返回X非逻辑后的布尔值。

    主要参数：
    x：array_like
    输入数组。
    随着版本的变化，函数的参数也在更新，更多详情点击 查看。
    返回：
    y：ndarray或bool
    布尔结果，与x的相应元素上的逻辑NOT运算，结果和x形状相同。 如果x是标量，则也返回标量。

(四) numpy.logical_xor(x1，x2)
    返回X1和X2异或逻辑后的布尔值。

    主要参数：
    x1，x2：array_like
    输入数组。 x1和x2必须具有相同的形状。
    随着版本的变化，函数的参数也在更新，更多详情点击 查看。
    返回：
    y：ndarray或bool
    布尔结果，与x1和x2的相应元素上的逻辑XOR运算，结果和x1和x2形状相同。 如果x1和x2都是标量，则也返回标量。

    [ logical_xor 相当于二进制加法 ]

"""

import numpy  as  np

def square(A):
    A_2 = np.zeros((A.shape[0], A.shape[1]), dtype=int)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_2[i][j] = np.logical_xor.reduce(np.logical_and(A[i], A[:, j]))
    return A_2

def Multipul(A, B):  # 二元域上的矩阵乘法
    A_B = np.zeros((A.shape[0], B.shape[1]), dtype=int)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            A_B[i][j] = np.logical_xor.reduce(np.logical_and(A[i], B[:, j]))
    return A_B


codechk = 10
codedim = 10
codelen = 20

# uu = np.ones(codedim, dtype = np.int8)
uu = np.random.randint(low = 0, high = 2, size = (codedim,), dtype = np.int8)
cc = np.random.randint(low = 0, high = 2, size = (codedim,), dtype = np.int8)









"""

https://www.runoob.com/numpy/numpy-binary-operators.html



NumPy 位运算
NumPy "bitwise_" 开头的函数是位运算函数。

NumPy 位运算包括以下几个函数：

函数	                描述
bitwise_and	      对数组元素执行位与操作
bitwise_or	      对数组元素执行位或操作
bitwise_xor	      对数组元素执行位异或操作
invert	          按位取反
left_shift	      向左移动二进制表示的位
right_shift	      向右移动二进制表示的位
注：也可以使用:
    与:"&"、 取反/非:"~"、 或:"|" 和 异或:"^" 等操作符进行计算。

(一)
    与： bitwise_and， 二进制乘法
    bitwise_and() 函数对数组中整数的二进制形式执行位与运算。

    位与操作运算规律如下：

    A	B	AND
    1	1	1
    1	0	0
    0	1	0
    0	0	0

    或： bitwise_or
    bitwise_or()函数对数组中整数的二进制形式执行位或运算。
    位或操作运算规律如下：
    A	B	OR
    1	1	1
    1	0	1
    0	1	1
    0	0	0

    异或：bitwise_xor，二进制加法
    bitwise_xor()函数对数组中整数的二进制形式执行位或运算。
    位或操作运算规律如下：
    A	B	XOR
    1	1	0
    1	0	1
    0	1	1
    0	0	0


(二)  invert
    invert() 函数对数组中整数进行位取反运算，即 0 变成 1，1 变成 0。

    对于有符号整数，取该二进制数的补码，然后 +1。二进制数，最高位为0表示正数，最高位为 1 表示负数。

    看看 ~1 的计算步骤：

    将1(这里叫：原码)转二进制 ＝ 00000001
    按位取反 ＝ 11111110
    发现符号位(即最高位)为1(表示负数)，将除符号位之外的其他数字取反 ＝ 10000001
    末位加1取其补码 ＝ 10000010
    转换回十进制 ＝ -2

(三)  left_shift
    left_shift() 函数将数组元素的二进制形式向左移动到指定位置，右侧附加相等数量的 0。


(四)  right_shift
    right_shift() 函数将数组元素的二进制形式向右移动到指定位置，左侧附加相等数量的 0。


"""



#======================================================
import numpy as np

print ('13 和 17 的二进制形式：')
a,b = 13,17
print (bin(a), bin(b))
print ('\n')
# 13 和 17 的二进制形式：
# 0b1101 0b10001

print (np.binary_repr(13, width = 8))
print (np.binary_repr(17, width = 8))
# 00001101
# 00010001


print ('13 和 17 的位与：')
print (np.bitwise_and(13, 17))
print ( 13 & 17)
print (np.binary_repr(13 & 17, width = 8))
# 13 和 17 的 与：
# 1
# 1
# 00000001

print ('13 和 17 的位或：')
print (np.bitwise_or(13, 17))
print ( 13 | 17)
print (np.binary_repr(13 | 17, width = 8))
# 13 和 17 的 或：
# 29
# 29
# 00011101

print ('13 和 17 的位异或：')
print (np.bitwise_xor(13, 17))
print ( 13 ^ 17)
print (np.binary_repr(13 ^ 17, width = 8))
# 13 和 17 的异或：
# 28
# 28
# 00011100

a = np.random.randint(low = 0, high = 2, size = (10, ))
b = np.random.randint(low = 0, high = 2, size = (10, ))
print(a)
print(b)
# [1 0 1 0 0 0 0 1 0 1]
# [0 0 1 1 0 1 1 0 1 1]


print ('a 和 b 的位与：')
print (np.bitwise_and(a, b))
print ( a & b)
# a 和 b 的位与：
# [0 0 1 0 0 0 0 0 0 1]
# [0 0 1 0 0 0 0 0 0 1]


print ('a 和 17 的位或：')
print (np.bitwise_or(a, b))
print ( a | b)
# a 和 17 的位或：
# [1 0 1 1 0 1 1 1 1 1]
# [1 0 1 1 0 1 1 1 1 1]

print ('a 和 17 的位或：')
print (np.bitwise_xor(a, b))
print ( a ^ b)
# a 和 17 的位或：
# [1 0 0 1 0 1 1 1 1 0]
# [1 0 0 1 0 1 1 1 1 0]


# (二) 位反转
import numpy as np

print ('13 的位反转，其中 ndarray 的 dtype 是 uint8：')
print (np.invert(np.array([13], dtype = np.uint8)))
print ('\n')
# 13 的位反转，其中 ndarray 的 dtype 是 uint8：
# [242]

# 比较 13 和 242 的二进制表示，我们发现了位的反转

print ('13 的二进制表示：')
print (np.binary_repr(13, width = 8))
print ('\n')
# 13 的二进制表示：
# 00001101

print ('242 的二进制表示：')
print (np.binary_repr(242, width = 8))
# 242 的二进制表示：
# 11110010



print ('13 的位反转，其中 ndarray 的 dtype 是 int8：')
print (np.invert(np.array([13], dtype = np.int8)))
print ('\n')
# 比较 13 和 -14 的二进制表示，我们发现了位的反转

print ('13 的二进制表示：')
print (np.binary_repr(13, width = 8))
print ('\n')
# 13 的二进制表示：
# 00001101

print ('-14 的二进制表示：')
print (np.binary_repr(-14, width = 8))
# 242 的二进制表示：
# 11110010



# (三) 左移
import numpy as np

print ('将 10 左移两位：')
print (np.left_shift(10, 2))
print ('\n')

print ('10 的二进制表示：')
print (np.binary_repr(10, width = 8))
print ('\n')
# 10 的二进制表示：
# 00001010

print ('40 的二进制表示：')
print (np.binary_repr(40, width = 8))
#  '00001010' 中的两位移动到了左边，并在右边添加了两个 0。
# 40 的二进制表示：
# 00101000

# (四) 右移
import numpy as np

print ('将 40 右移两位：')
print (np.right_shift(40,2))
print ('\n')

print ('40 的二进制表示：')
print (np.binary_repr(40, width = 8))
print ('\n')
# 40 的二进制表示：
# 00101000

print ('10 的二进制表示：')
print (np.binary_repr(10, width = 8))
#  '00001010' 中的两位移动到了右边，并在左边添加了两个 0。
# 10 的二进制表示：
# 00001010



































































































































































































































































































































