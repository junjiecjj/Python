#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:01:29 2024

@author: jack
https://blog.csdn.net/forest_LL/article/details/129507243
https://blog.csdn.net/qq_45889056/article/details/128032969

https://zhuanlan.zhihu.com/p/135396870

https://zhuanlan.zhihu.com/p/480389473

https://blog.csdn.net/qfikh/article/details/103994319

"""
import numpy as np



A = np.array([[1,2],[0,0],[0,0]])
U,S,VH = np.linalg.svd(A)



print("U = \n",U)
print("S = \n",S)
print("V^H = \n",VH)















































































