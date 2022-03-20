#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 19:19:04 2022

@author: jack
"""

import numpy as np


#从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
print("np.random.uniform(1,2,(2,3)) = \n {}".format(np.random.uniform(1,2,(2,3))))

print("np.random.uniform(1,2,5) = \n {}".format(np.random.uniform(1,2,5)))

# numpy.random.randint(low, high=None, size=None, dtype='l')，产生随机整数；
print("numpy.random.randint(5) = \n{}".format(np.random.randint(5)))

print("numpy.random.randint(5,size=(2,3)) = \n{}".format(np.random.randint(5,size=(2,3))))

# numpy.random.rand(d0, d1, ..., dn)，产生d0 - d1 - ... - dn形状的在[0,1)上均匀分布的float型数。
print("numpy.random.rand(2,3) = \n{}".format(np.random.rand(2,3)))


# randn: 原型：numpy.random.randn（d0,d1,...,dn),产生d0 - d1 - ... - dn形状的标准正态分布的float型数。
print("numpy.random.randn(3,4) = \n{}".format(np.random.randn(3,4)))






















