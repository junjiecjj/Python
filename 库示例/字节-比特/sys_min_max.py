#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:38:09 2023

@author: jack
"""

import numpy as np


import sys

max_int = sys.maxsize
print(max_int)

print(np.iinfo(np.int8).min)
print(np.iinfo(np.int8).max)
# -128
# 127

print(np.iinfo(np.uint8).min)
print(np.iinfo(np.uint8).max)
# 0
# 255

print(np.iinfo(np.int16).min)
print(np.iinfo(np.int16).max)
# -32768
# 32767

print(np.iinfo(np.uint16).min)
print(np.iinfo(np.uint16).max)
# 0
# 65535

print(np.iinfo(np.int32).min)
print(np.iinfo(np.int32).max)
# -2147483648
# 2147483647

print(np.iinfo(np.uint32).min)
print(np.iinfo(np.uint32).max)
# 0
# 4294967295


print(np.finfo(np.float32).min)
print(np.finfo(np.float32).max)
# -3.4028235e+38
# 3.4028235e+38


max_float = float("inf")  # 无限大 比所有数大
min_float = float("-inf")  #无限小 比所有数小
print(type(max_float))
print(type(min_float))


# （1）就大小而言：
import sys
print(float('inf')>sys.maxsize)

# （2）就占用空间而言：
import sys
print(sys.getsizeof(float('inf')))
print(sys.getsizeof(sys.maxsize))































































































































































































































































































