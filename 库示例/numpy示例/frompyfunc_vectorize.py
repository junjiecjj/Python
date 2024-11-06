#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 14:44:54 2024

@author: jack
"""


import numpy as np

#%%>>>>>>>>>>>>>>>>>>>>>>>>>>> np.frompyfunc >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def test1(x):
    return x*np.array([0.0, 1.0])

fpf_test1 = np.frompyfunc(test1, 1, 1)
vec_test1 = np.vectorize(test1)
x = np.linspace(0,1,3)

print(fpf_test1(x))
# [array([0., 0.]) array([0. , 0.5]) array([0., 1.])]

# print(vec_test1(x)) # error


# https://blog.csdn.net/S_o_l_o_n/article/details/103032020
import numpy as np

arr = np.array([[1,2,3], [2,3,4]])
f1 = np.frompyfunc(lambda x:x+1 if x<5 else x-1, 1, 1)
f2 = np.frompyfunc(lambda x:(x+1, x-1), 1, 2)
f3 = np.frompyfunc(lambda x,y:x+y, 2, 1)
f4 = np.frompyfunc(lambda x,y:(x+y, x-y), 2, 2)

print(f1(arr))
# output:
# array([[2, 3, 4],
#        [3, 4, 5]], dtype=object)

a = f2(arr)
print(a)
# output:
# (array([[2, 3, 4],
#         [3, 4, 5]], dtype=object),
#  array([[0, 1, 2],
#         [1, 2, 3]], dtype=object))

print(f3(arr, arr))
# output:
# array([[2, 4, 6],
#        [4, 6, 8]], dtype=object)

print(f4(arr, arr))
# output:
# (array([[2, 4, 6],
#         [4, 6, 8]], dtype=object),
#  array([[0, 0, 0],
#         [0, 0, 0]], dtype=object))

ar=np.array([1,2,3])
print(f4(arr, ar))
# output:
# (array([[2, 4, 6],
#         [3, 5, 7]], dtype=object),
#  array([[0, 0, 0],
#         [1, 1, 1]], dtype=object))



# https://zhuanlan.zhihu.com/p/613005189
import numpy as np
import matplotlib.pyplot as plt

def y(t):
    if t <= -1 / 2:
        r = -3 * t + 1
    elif t >= 2:
        r = 3 * t - 1
    else:
        r = 3 + t
    return r

x = np.linspace(-5, 6, 1000)
triangle_ufunc1 = np.frompyfunc(y,1,1)
y = triangle_ufunc1(x)

plt.plot(x, y)
plt.show()

# https://geek-docs.com/numpy/python-numpy-logic-functions/numpy-frompyfunc-in-python.html#google_vignette
import numpy as np

# create an array of numbers
a = np.array([34, 67, 89, 15, 33, 27])

# python str function as ufunc
string_generator = np.frompyfunc(str, 1, 1)

print("Original array: ", a)
print("After conversion to string: ", string_generator(a))

import numpy as np
# create an array of numbers
a = np.array([345, 122, 454, 232, 334, 56, 66])

# user-defined function to check
# whether a no. is palindrome or not
def fun(x):
    s = str(x)
    return s[::-1]== s

# 'check_palindrome' as universal function
check_palindrome = np.frompyfunc(fun, 1, 1)
print("Original array: ", a)
print("Checking of number as palindrome: ", check_palindrome(a))

#%%>>>>>>>>>>>>>>>>>>>>>>>>>>> np.vectorize >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def myfunc(a, b):
    "Return a-b if a>b, otherwise return a+b"
    if a > b:
        return a - b
    else:
        return a + b

vfunc = np.vectorize(myfunc)

print(vfunc([1, 2, 3, 4], 2))
# Out[87]: array([3, 4, 1, 2])

# https://blog.csdn.net/pgsld2333/article/details/121176599
def add(a,b):
	print("a:", a, "b:", b)
	return a + b

add([1,2,3],[4,5,6])
# Out[139]: [1, 2, 3, 4, 5, 6]

add_vectorized_func1 = np.vectorize(add)
add_vectorized_func1([1,2,3],[4,5,6])
# a: 1 b: 4
# a: 1 b: 4
# a: 2 b: 5
# a: 3 b: 6
# Out[2]: array([5, 7, 9])

add_vectorized_func2 = np.vectorize(add, signature="(),(n)->(n)")
add_vectorized_func2([1,2,3], np.array([4,5,6])) # 这里用np.array来保证数和数组的加法可以实现


add_vectorized_func2 = np.vectorize(add, signature="(),(n)->(k)")
add_vectorized_func2([1,2,3], np.array([4,5,6])) #


a = np.array([[1,1,1],
            [2,2,2],
            [3,3,3]])
b = np.array([[4,4,4],
             [5,5,5],
             [6,6,6]])
add_vectorized_func3 = np.vectorize(add, signature="(n),(m,n)->(m,n)")
add_vectorized_func3( a, b ) # 这里用np.array来保证数组和数组的加法可以实现




###################################
A = np.random.randn(4,5)
BG = 8
G =  2**BG
p = (A * G + 1)/2
p = np.clip(p, a_min = 0, a_max = 1, )
Int =  np.random.binomial(1, p).astype(np.int8)

def quantizer(params,  BG = 8,):
    G =  2**BG
    p = (params * G + 1)/2
    p = np.clip(p, a_min = 0, a_max = 1, )
    Int =  np.random.binomial(1, p).astype(np.int8)
    # Int = f1(p).astype(np.int8)
    return Int

B = quantizer(A)

































