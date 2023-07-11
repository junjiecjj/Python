#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:48:00 2023

@author: jack

在处理机器学习问题中，我们常常会碰到范数，这里我将通过code给大家讲解一下范数。
范式代码：np.linalg.norm(x, ord=None, axis=None)
其中：
    (1) linalg=linear（线性）+algebra（代数），norm表示范数
    (2) x代表矩阵，ord为范数类型
    (3) axis为处理类型：当 axis=1 时表示按行向量处理，求多个行向量的范数。当 axis=0 时表示按列向量处理，求多个列向量的范数。当axis=None表示矩阵范数




向量范数:
    1-范数： 即向量元素绝对值之和，matlab调用函数norm(x, 1) 。
    2-范数： Euclid范数（欧几里得范数，常用计算向量长度），即向量元素绝对值的平方和再开方，matlab调用函数norm(x, 2)。
    ∞-范数： 即所有向量元素绝对值中的最大值，matlab调用函数norm(x, inf)。
    -∞-范数： 即所有向量元素绝对值中的最小值，matlab调用函数norm(x, -inf)。
    p-范数： 即向量元素绝对值的p次方和的1/p次幂，matlab调用函数norm(x, p)。

矩阵范数:
    1-范数： 列和范数，即所有矩阵列向量绝对值之和的最大值，matlab调用函数norm(A, 1)。
    2-范数： 谱范数，即A'A矩阵的最大特征值的开平方。matlab调用函数norm(x, 2)。
    ∞-范数： 行和范数，即所有矩阵行向量绝对值之和的最大值，matlab调用函数norm(A, inf)。
    F-范数： Frobenius范数，即矩阵元素绝对值的平方和再开平方，matlab调用函数norm(A, ’fro‘)。


np.linalg.norm(X):
    X为向量时，默认求向量2范数，即求向量元素绝对值的平方和再开方；
    X为矩阵是，默认求的是F范数。矩阵的F范数即：矩阵的各个元素平方之和再开平方根，它通常也叫做矩阵的L2范数，它的有点在它是一个凸函数，可以求导求解，易于计算。


=====  ============================  ==========================
ord    norm for matrices             norm for vectors
=====  ============================  ==========================
None   Frobenius norm                2-norm
'fro'  Frobenius norm                --
'nuc'  nuclear norm                  --
inf    max(sum(abs(x), axis=1))      max(abs(x))
-inf   min(sum(abs(x), axis=1))      min(abs(x))
0      --                            sum(x != 0)
1      max(sum(abs(x), axis=0))      as below
-1     min(sum(abs(x), axis=0))      as below
2      2-norm (largest sing. value)  as below
-2     smallest singular value       as below
other  --                            sum(abs(x)**ord)**(1./ord)
=====  ============================  ==========================

The Frobenius norm is given by [1]_:

    :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

The nuclear norm is the sum of the singular values.

Both the Frobenius and nuclear norm orders are only defined for
matrices and raise a ValueError when ``x.ndim != 2``.

"""




#=================================================================================================================
#                                                      矩阵范数
#=================================================================================================================

import numpy as np

A = np.array([[3, -4],[-6, 2]])
print(A)



# , keepdims = True  注意 keepdims = True 时，输出的是一个二维数组。
ret_1 = np.linalg.norm(A, ord = 1, axis = None, keepdims = True)                          #  列范数，即所有矩阵列向量绝对值之和的最大值，matlab调用函数norm(A, 1)。
print(ret_1)
ret_1 = np.linalg.norm(A, ord = -1, axis = None, keepdims = True)                         #  列范数，即所有矩阵列向量绝对值之和的最小值，matlab调用函数norm(A, 1)。
print(ret_1)


ret_inf = np.linalg.norm(A, ord = np.Inf, axis = None, keepdims = True)                   #  行范数，即所有矩阵行向量绝对值之和的最大值，matlab调用函数norm(A, inf)。
print(ret_inf)
ret_inf = np.linalg.norm(A, ord = -np.Inf, axis = None, keepdims = True)                  #  行范数，即所有矩阵行向量绝对值之和的最小值，matlab调用函数norm(A, inf)。
print(ret_inf)


ret_all = np.linalg.norm(A, ord = 'fro', axis = None, keepdims = True)                    # Frobenius范数，即矩阵元素绝对值的平方和再开平方，matlab调用函数norm(A, ’fro‘)。
print(ret_all)
## X为矩阵是，默认求的是F范数。矩阵的F范数即：矩阵的各个元素平方之和再开平方根，它通常也叫做矩阵的L2范数，它的有点在它是一个凸函数，可以求导求解，易于计算。
ret_all = np.linalg.norm(A, keepdims = True)
print(ret_all)

ret_2 = np.linalg.norm(A, ord = 2, axis = None, keepdims = True)                          #  谱范数，即A'A矩阵的最大特征值的开平方。matlab调用函数norm(x, 2)。
print(ret_2)
ret_2 = np.linalg.norm(A, ord = -2, axis = None, keepdims = True)                         #  谱范数，即A'A矩阵的最小特征值的开平方。matlab调用函数norm(x, 2)。
print(ret_2)

# ret_3 = np.linalg.norm(A, ord = 3, axis = None)                                                   # ret_3 返回的是  ValueError: Invalid norm order for matrices.
# print(ret_3)
# ret_3 = np.linalg.norm(A, ord = -3, axis = None)                                                  # ret_3 返回的是  ValueError: Invalid norm order for matrices.
# print(ret_3)

ret_nuc = np.linalg.norm(A, ord = 'nuc', axis = None)                    # ret_nuc 返回的是 核范数的值; 核范数是矩阵奇异值的和，用于约束矩阵的低秩，
print(ret_nuc)




#=================================================================================================================
#                                                      向量范数
#=================================================================================================================

## axis为处理类型：当axis=1时表示按行向量处理，求多个行向量的范数。当axis=0时表示按列向量处理，求多个列向量的范数。当axis=None表示矩阵范数

import numpy as np

A = np.array([[3, -4],[-6, 2]])
print(A)


# ret_all = np.linalg.norm(A, ord = 'fro', axis = 1)                                    # ValueError: Invalid norm order 'fro' for vectors
# print(ret_all)

# ret_all = np.linalg.norm(A, ord = 'fro', axis = 0)                                    # ValueError: Invalid norm order 'fro' for vectors
# print(ret_all)


ret_0 = np.linalg.norm(A, ord = 0,  axis = 1,keepdims = True )                          # ret_0 返回的是 0 范数，表示向量中非零元素的个数。
print(ret_0)
ret_0 = np.linalg.norm(A, ord = 0,  axis = 0, keepdims = True )                          # ret_0 返回的是 0 范数，表示向量中非零元素的个数。
print(ret_0)

ret_1 = np.linalg.norm(A, ord = 1, axis = 1, keepdims = True)                               # ret_1 返回的是 1 范数, 向量元素绝对值之和，matlab调用函数norm(x, 1) 。
print(ret_1)
ret_1 = np.linalg.norm(A, ord = 1, axis = 0, keepdims = True)                               # ret_1 返回的是 1 范数, 向量元素绝对值之和，matlab调用函数norm(x, 1) 。
print(ret_1)
ret_1 = np.linalg.norm(A, ord = -1, axis = 1, keepdims = True)                              # ret_1 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_1)
ret_1 = np.linalg.norm(A, ord = -1, axis = 0, keepdims = True)                              # ret_1 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_1)

## X为向量时，默认求向量2范数，即求向量元素绝对值的平方和再开方；
# ret_all = np.linalg.norm(A,         axis = 1, keepdims = True)
# print(ret_all)
ret_2 = np.linalg.norm(A, ord = 2,  axis = 1, keepdims = True)                               # ret_2 返回的是 2 范数的值; Euclid范数（欧几里得范数，常用计算向量长度），即向量元素绝对值的平方和再开方，matlab调用函数norm(x, 2)。
print(ret_2)
ret_2 = np.linalg.norm(A, ord = 2,  axis = 0, keepdims = True)                               # ret_2 返回的是 2 范数的值; Euclid范数（欧几里得范数，常用计算向量长度），即向量元素绝对值的平方和再开方，matlab调用函数norm(x, 2)。
print(ret_2)
ret_2 = np.linalg.norm(A, ord = -2, axis = 1, keepdims = True)                               # ret_2 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_2)
ret_2 = np.linalg.norm(A, ord = -2, axis = 0, keepdims = True)                               # ret_2 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_2)

## 向量的p范数：
ret_3 = np.linalg.norm(A, ord = 3, axis = 1, keepdims = True)                               # ret_2 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_3)
ret_3 = np.linalg.norm(A, ord = 3, axis = 0, keepdims = True)                               # ret_2 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_3)
ret_3 = np.linalg.norm(A, ord = -3, axis = 1, keepdims = True)                               # ret_2 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_3)
ret_3 = np.linalg.norm(A, ord = -3, axis = 0, keepdims = True)                               # ret_2 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_3)

ret_inf = np.linalg.norm(A, ord = np.Inf, axis = 1, keepdims = True)                         # 当 axis=1 时表示按行向量处理，求多个行向量的范数； ret_all返回的是 inf 范数的值; 即所有向量元素绝对值中的最大值，matlab调用函数norm(x, inf)。
print(ret_inf)
ret_inf = np.linalg.norm(A, ord = np.Inf, axis = 0, keepdims = True)                         # 当 axis=0 时表示按列向量处理，求多个列向量的范数； ret_all返回的是 inf 范数的值; 即所有向量元素绝对值中的最大值，matlab调用函数norm(x, inf)。
print(ret_inf)
ret_inf = np.linalg.norm(A, ord = -np.Inf, axis = 1, keepdims = True)                         # 当 axis=1 时表示按行向量处理，求多个行向量的范数； ret_all返回的是 inf 范数的值; 即所有向量元素绝对值中的最小值，matlab调用函数norm(x, inf)。
print(ret_inf)
ret_inf = np.linalg.norm(A, ord = -np.Inf, axis = 0, keepdims = True)                         # 当 axis=0 时表示按列向量处理，求多个列向量的范数； ret_all返回的是 inf 范数的值; 即所有向量元素绝对值中的最小值，matlab调用函数norm(x, inf)。
print(ret_inf)




#=================================================================================================================
#                                                      向量范数
#=================================================================================================================


import numpy as np

A = np.array([3, -4, -6, 2, 0])
print(A)

ret_0 = np.linalg.norm(A, ord = 0,   )                             # ret_0 返回的是 0 范数，表示向量中非零元素的个数。
print(ret_0)

ret_1 = np.linalg.norm(A, ord = 1, )                               # ret_1 返回的是 1 范数, 向量元素绝对值之和，matlab调用函数norm(x, 1) 。
print(ret_1)

ret_1 = np.linalg.norm(A, ord = -1, )                               # ret_1 返回的是sum(abs(x)**ord)**(1./ord)
print(ret_1)

## X为向量时，默认求向量2范数，即求向量元素绝对值的平方和再开方；
ret_all = np.linalg.norm(A,    )
print(ret_all)
ret_2 = np.linalg.norm(A, ord = 2,  )                               # ret_2 返回的是 2 范数的值; Euclid范数（欧几里得范数，常用计算向量长度），即向量元素绝对值的平方和再开方，matlab调用函数norm(x, 2)。
print(ret_2)
ret_2 = np.linalg.norm(A, ord = -2,  )                               # ret_2 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_2)

## 向量的p范数：
ret_3 = np.linalg.norm(A, ord = 3,  )                               # ret_3 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_3)

ret_4 = np.linalg.norm(A, ord = 4,  )                               # ret_3 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_4)

ret_inf = np.linalg.norm(A, ord = np.Inf, )                         # 当 axis=1 时表示按行向量处理，求多个行向量的范数； ret_all返回的是 inf 范数的值; 即所有向量元素绝对值中的最大值，matlab调用函数norm(x, inf)。
print(ret_inf)
ret_inf = np.linalg.norm(A, ord = -np.Inf, )                         # 当 axis=1 时表示按行向量处理，求多个行向量的范数； ret_all返回的是 inf 范数的值; 即所有向量元素绝对值中的最小值，matlab调用函数norm(x, inf)。
print(ret_inf)





#=================================================================================================================
#                                                      矩阵范数 torch
#=================================================================================================================

"""
torch.linalg.norm(A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None) → Tensor  函数的用法:

ord                      norm for matrices                   norm for vectors
--------------------------------------------------------------------------------
None (default)           Frobenius norm                    2-norm (see below)
‘fro’                    Frobenius norm                     – not supported –
‘nuc’                     nuclear norm                      – not supported –
inf                  max(sum(abs(x), dim=1))                 max(abs(x))
-inf                  min(sum(abs(x), dim=1))                min(abs(x))
0                      – not supported –                      sum(x != 0)
1                      max(sum(abs(x), dim=0))                 as below
-1                    min(sum(abs(x), dim=0))                    as below
2                     largest singular value                    as below
-2                     smallest singular value                  as below
other int or float     – not supported –         sum(abs(x)^{ord})^{(1 / ord)}

where inf refers to float(‘inf’), NumPy’s inf object, or any equivalent object.


"""

import torch


A = torch.tensor([[3., -4.],[-6., 2.]])
print(A)

# , keepdims = True  注意 keepdims = True 时，输出的是一个二维数组。
ret_1 = torch.linalg.norm(A, ord = 1, dim = None, keepdim = True)                          #  列范数，即所有矩阵列向量绝对值之和的最大值，matlab调用函数norm(A, 1)。
print(ret_1)
ret_1 = torch.linalg.norm(A, ord = -1, dim = None, keepdim = True)                         #  列范数，即所有矩阵列向量绝对值之和的最小值，matlab调用函数norm(A, 1)。
print(ret_1)


ret_inf = torch.linalg.norm(A, ord = np.Inf, dim = None, keepdim = True)                   #  行范数，即所有矩阵行向量绝对值之和的最大值，matlab调用函数norm(A, inf)。
print(ret_inf)
ret_inf = torch.linalg.norm(A, ord = -np.Inf, dim = None, keepdim = True)                  #  行范数，即所有矩阵行向量绝对值之和的最小值，matlab调用函数norm(A, inf)。
print(ret_inf)


ret_all = torch.linalg.norm(A, ord = 'fro', dim = None, keepdim = True)                    # Frobenius范数，即矩阵元素绝对值的平方和再开平方，matlab调用函数norm(A, ’fro‘)。
print(ret_all)
## X为矩阵是，默认求的是F范数。矩阵的F范数即：矩阵的各个元素平方之和再开平方根，它通常也叫做矩阵的L2范数，它的有点在它是一个凸函数，可以求导求解，易于计算。
ret_all = torch.linalg.norm(A, keepdim = True)
print(ret_all)

ret_2 = torch.linalg.norm(A, ord = 2, dim = None, keepdim = True)                          #  谱范数，即A'A矩阵的最大特征值的开平方。matlab调用函数norm(x, 2)。
print(ret_2)
ret_2 = torch.linalg.norm(A, ord = -2, dim = None, keepdim = True)                         #  谱范数，即A'A矩阵的最小特征值的开平方。matlab调用函数norm(x, 2)。
print(ret_2)

# ret_3 = np.linalg.norm(A, ord = 3, axis = None)                                                # ret_3 返回的是  ValueError: Invalid norm order for matrices.
# print(ret_3)
# ret_3 = np.linalg.norm(A, ord = -3, axis = None)                                                # ret_3 返回的是  ValueError: Invalid norm order for matrices.
# print(ret_3)

ret_nuc = torch.linalg.norm(A, ord = 'nuc', dim = None)                    # ret_nuc 返回的是 核范数的值; 核范数是矩阵奇异值的和，用于约束矩阵的低秩，
print(ret_nuc)


#=================================================================================================================
#                                                      向量范数 torch
#=================================================================================================================


import torch

A = torch.tensor([3.0, -4.0, -6.0, 2.0,  0])
print(A)

ret_0 = torch.linalg.norm(A, ord = 0,  )                               # ret_0 返回的是 0 范数，表示向量中非零元素的个数。
print(ret_0)

ret_1 = torch.linalg.norm(A, ord = 1,  )                               # ret_1 返回的是 1 范数, 向量元素绝对值之和，matlab调用函数norm(x, 1) 。
print(ret_1)

ret_1 = torch.linalg.norm(A, ord = -1,  )                               # ret_1 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_1)


## X为向量时，默认求向量2范数，即求向量元素绝对值的平方和再开方；
ret_all = torch.linalg.norm(A,    )
print(ret_all)
ret_2 = torch.linalg.norm(A, ord = 2,  )                               # ret_2 返回的是 2 范数的值; Euclid范数（欧几里得范数，常用计算向量长度），即向量元素绝对值的平方和再开方，matlab调用函数norm(x, 2)。
print(ret_2)
ret_2 = torch.linalg.norm(A, ord = -2,  )                               # ret_2 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_2)

## 向量的p范数：
ret_3 = torch.linalg.norm(A, ord = 3,  )                               # ret_3 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_3)

ret_4 = torch.linalg.norm(A, ord = 4,  )                               # ret_3 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_4)

ret_inf = torch.linalg.norm(A, ord = float('inf'), )                         # 当 axis=1 时表示按行向量处理，求多个行向量的范数； ret_all返回的是 inf 范数的值; 即所有向量元素绝对值中的最大值，matlab调用函数norm(x, inf)。
print(ret_inf)
ret_inf = torch.linalg.norm(A, ord = -float('inf'), )                         # 当 axis=1 时表示按行向量处理，求多个行向量的范数； ret_all返回的是 inf 范数的值; 即所有向量元素绝对值中的最小值，matlab调用函数norm(x, inf)。
print(ret_inf)


#=================================================================================================================
#                                                      向量范数 torch
#=================================================================================================================
import torch

A = torch.tensor([[3., -4.],[-6., 2.]])
print(A)


ret_0 = torch.linalg.norm(A, ord = 0,  dim = 1, keepdim = True )                          # ret_0 返回的是 0 范数，表示向量中非零元素的个数。
print(ret_0)
ret_0 = torch.linalg.norm(A, ord = 0,  dim = 0, keepdim = True )                          # ret_0 返回的是 0 范数，表示向量中非零元素的个数。
print(ret_0)

ret_1 = torch.linalg.norm(A, ord = 1, dim = 1, keepdim = True)                               # ret_1 返回的是 1 范数, 向量元素绝对值之和，matlab调用函数norm(x, 1) 。
print(ret_1)
ret_1 = torch.linalg.norm(A, ord = 1, dim = 0, keepdim = True)                               # ret_1 返回的是 1 范数, 向量元素绝对值之和，matlab调用函数norm(x, 1) 。
print(ret_1)
ret_1 = torch.linalg.norm(A, ord = -1, dim = 1, keepdim = True)                              # ret_1 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_1)
ret_1 = torch.linalg.norm(A, ord = -1, dim = 0, keepdim = True)                              # ret_1 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_1)

## X为向量时，默认求向量2范数，即求向量元素绝对值的平方和再开方；
# ret_all = np.linalg.norm(A,         axis = 1, keepdims = True)
# print(ret_all)
ret_2 = torch.linalg.norm(A, ord = 2,  dim = 1, keepdim = True)                               # ret_2 返回的是 2 范数的值; Euclid范数（欧几里得范数，常用计算向量长度），即向量元素绝对值的平方和再开方，matlab调用函数norm(x, 2)。
print(ret_2)
ret_2 = torch.linalg.norm(A, ord = 2,  dim = 0, keepdim = True)                               # ret_2 返回的是 2 范数的值; Euclid范数（欧几里得范数，常用计算向量长度），即向量元素绝对值的平方和再开方，matlab调用函数norm(x, 2)。
print(ret_2)
ret_2 = torch.linalg.norm(A, ord = -2, dim = 1, keepdim = True)                               # ret_2 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_2)
ret_2 = torch.linalg.norm(A, ord = -2, dim = 0, keepdim = True)                               # ret_2 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_2)

## 向量的p范数：
ret_3 = torch.linalg.norm(A, ord = 3, dim = 1, keepdim = True)                               # ret_2 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_3)
ret_3 = torch.linalg.norm(A, ord = 3, dim = 0, keepdim = True)                               # ret_2 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_3)
ret_3 = torch.linalg.norm(A, ord = -3, dim = 1, keepdim = True)                               # ret_2 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_3)
ret_3 = torch.linalg.norm(A, ord = -3, dim = 0, keepdim = True)                               # ret_2 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_3)

ret_inf = torch.linalg.norm(A, ord = np.Inf, dim = 1, keepdim = True)                         # 当 axis=1 时表示按行向量处理，求多个行向量的范数； ret_all返回的是 inf 范数的值; 即所有向量元素绝对值中的最大值，matlab调用函数norm(x, inf)。
print(ret_inf)
ret_inf = torch.linalg.norm(A, ord = float('inf'), dim = 0, keepdim = True)                         # 当 axis=0 时表示按列向量处理，求多个列向量的范数； ret_all返回的是 inf 范数的值; 即所有向量元素绝对值中的最大值，matlab调用函数norm(x, inf)。
print(ret_inf)
ret_inf = torch.linalg.norm(A, ord = -np.Inf, dim = 1, keepdim = True)                         # 当 axis=1 时表示按行向量处理，求多个行向量的范数； ret_all返回的是 inf 范数的值; 即所有向量元素绝对值中的最小值，matlab调用函数norm(x, inf)。
print(ret_inf)
ret_inf = torch.linalg.norm(A, ord = -float('inf'), dim = 0, keepdim = True)                         # 当 axis=0 时表示按列向量处理，求多个列向量的范数； ret_all返回的是 inf 范数的值; 即所有向量元素绝对值中的最小值，matlab调用函数norm(x, inf)。
print(ret_inf)



#=================================================================================================================
#                                                      矩阵范数 torch
#=================================================================================================================

"""
torch.linalg.matrix_norm(A, ord='fro', dim=(- 2, - 1), keepdim=False, *, dtype=None, out=None) → Tensor

=======================================================
ord                           matrix norm
=======================================================
‘fro’ (default)               Frobenius norm
‘nuc’                         nuclear norm
inf                           max(sum(abs(x), dim=1))
-inf                          min(sum(abs(x), dim=1))
1                             max(sum(abs(x), dim=0))
-1                            min(sum(abs(x), dim=0))
2                             largest singular value
-2                            smallest singular value

"""

import torch


A = torch.tensor([[3., -4.],[-6., 2.]])
print(A)

# , keepdims = True  注意 keepdims = True 时，输出的是一个二维数组。
ret_1 = torch.linalg.matrix_norm(A, ord = 1,   keepdim = True)                          #  列范数，即所有矩阵列向量绝对值之和的最大值，matlab调用函数norm(A, 1)。
print(ret_1)
ret_1 = torch.linalg.matrix_norm(A, ord = -1,   keepdim = True)                         #  列范数，即所有矩阵列向量绝对值之和的最小值，matlab调用函数norm(A, 1)。
print(ret_1)


ret_inf = torch.linalg.matrix_norm(A, ord = np.Inf, keepdim = True)                   #  行范数，即所有矩阵行向量绝对值之和的最大值，matlab调用函数norm(A, inf)。
print(ret_inf)
ret_inf = torch.linalg.matrix_norm(A, ord = -np.Inf, keepdim = True)                  #  行范数，即所有矩阵行向量绝对值之和的最小值，matlab调用函数norm(A, inf)。
print(ret_inf)


ret_all = torch.linalg.matrix_norm(A, ord = 'fro',  keepdim = True)                    # Frobenius范数，即矩阵元素绝对值的平方和再开平方，matlab调用函数norm(A, ’fro‘)。
print(ret_all)
## X为矩阵是，默认求的是F范数。矩阵的F范数即：矩阵的各个元素平方之和再开平方根，它通常也叫做矩阵的L2范数，它的有点在它是一个凸函数，可以求导求解，易于计算。
ret_all = torch.linalg.matrix_norm(A, keepdim = True)
print(ret_all)

ret_2 = torch.linalg.matrix_norm(A, ord = 2, keepdim = True)                          #  谱范数，即A'A矩阵的最大特征值的开平方。matlab调用函数norm(x, 2)。
print(ret_2)
ret_2 = torch.linalg.matrix_norm(A, ord = -2,  keepdim = True)                         #  谱范数，即A'A矩阵的最小特征值的开平方。matlab调用函数norm(x, 2)。
print(ret_2)

# ret_3 = np.linalg.norm(A, ord = 3, axis = None)                                                # ret_3 返回的是  ValueError: Invalid norm order for matrices.
# print(ret_3)
# ret_3 = np.linalg.norm(A, ord = -3, axis = None)                                                # ret_3 返回的是  ValueError: Invalid norm order for matrices.
# print(ret_3)

ret_nuc = torch.linalg.matrix_norm(A, ord = 'nuc', )                    # ret_nuc 返回的是 核范数的值; 核范数是矩阵奇异值的和，用于约束矩阵的低秩，
print(ret_nuc)






#=================================================================================================================
#                                                      向量范数 torch
#=================================================================================================================
"""
torch.linalg.vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None, out=None) → Tensor

=======================================================
ord                              vector norm
=======================================================
2 (default)                      2-norm (see below)
inf                              max(abs(x))
-inf                             min(abs(x))
0                                sum(x != 0)
other int or float              sum(abs(x)^{ord})^{(1 / ord)}
"""

import torch

A = torch.tensor([3.0, -4.0, -6.0, 2.0,  0])
print(A)



ret_0 = torch.linalg.vector_norm(A, ord = 0,  )                               # ret_0 返回的是 0 范数，表示向量中非零元素的个数。
print(ret_0)

ret_1 = torch.linalg.vector_norm(A, ord = 1,  )                               # ret_1 返回的是 1 范数, 向量元素绝对值之和，matlab调用函数norm(x, 1) 。
print(ret_1)

ret_1 = torch.linalg.vector_norm(A, ord = -1,  )                               # ret_1 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_1)


## X为向量时，默认求向量2范数，即求向量元素绝对值的平方和再开方；
ret_all = torch.linalg.vector_norm(A,    )
print(ret_all)
ret_2 = torch.linalg.vector_norm(A, ord = 2,  )                               # ret_2 返回的是 2 范数的值; Euclid范数（欧几里得范数，常用计算向量长度），即向量元素绝对值的平方和再开方，matlab调用函数norm(x, 2)。
print(ret_2)
ret_2 = torch.linalg.vector_norm(A, ord = -2,  )                               # ret_2 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_2)

## 向量的p范数：
ret_3 = torch.linalg.vector_norm(A, ord = 3,  )                               # ret_3 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_3)

ret_4 = torch.linalg.vector_norm(A, ord = 4,  )                               # ret_3 返回的是 sum(abs(x)**ord)**(1./ord)
print(ret_4)

ret_inf = torch.linalg.vector_norm(A, ord = float('inf'), )                         # 当 axis=1 时表示按行向量处理，求多个行向量的范数； ret_all返回的是 inf 范数的值; 即所有向量元素绝对值中的最大值，matlab调用函数norm(x, inf)。
print(ret_inf)
ret_inf = torch.linalg.vector_norm(A, ord = -float('inf'), )                         # 当 axis=1 时表示按行向量处理，求多个行向量的范数； ret_all返回的是 inf 范数的值; 即所有向量元素绝对值中的最小值，matlab调用函数norm(x, inf)。
print(ret_inf)











































































































































































































