#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:50:35 2023

@author: jack

此文档是记录torch中快速的找到数组中满足某些条件的元素的位置，并改变它的值；

"""

#======================================================================================
##     torch.nonzero()和torch.index_select()
#======================================================================================

import torch
# 1. torch.nonzero()的定义
label = torch.tensor([[1, 0, 0], [1, 0, 1]])
print(label.nonzero())
# tensor([[0, 0],
#         [1, 0],
#         [1, 2]])

#2 torch.nonzero()用来筛选张量中符合某个条件的元素，得到这些元素的索引

a = torch.tensor([9.3, 4.2, 8.5, 2.7, 5.9])
b = torch.nonzero(a > 6, as_tuple=False)
print(b)
# tensor([[0],
#         [2]])





# 3. torch.index_select()的定义和示例
    # torch.index_select(input, dim, index) 函数返回的是沿着输入张量的指定维度的指定索引号进行索引的张量子集，函数参数有：

    # input(Tensor) - 需要进行索引操作的输入张量；
    # dim(int) - 需要对输入张量进行索引的维度；
    # index(LongTensor) - 包含索引号的 1D 张量；
# 一维例子：
a = torch.tensor([9.3, 4.2, 8.5, 2.7, 5.9])
idx = torch.tensor([1, 3, 4])

b = torch.index_select(a, dim = 0, index = idx)
print(b)
# tensor([4.2000, 2.7000, 5.9000])

# 二维例子：
a = torch.tensor([[9.3, 4.2, 8.5], [2.7, 5.9, 8.7]])
idx = torch.tensor([0, 2])

b = torch.index_select(a, dim = 1, index = idx)
print(b)
# tensor([[9.3000, 8.5000],
#         [2.7000, 8.7000]])


# 4. torch.nonzero()和torch.index_select()结合使用
# 结合使用torch.nonzero()和torch.index_select()，可以选出符合某种条件的元素。下面的例子是从一维张量a中选出大于6的元素：

a = torch.tensor([9.3, 4.2, 8.5, 2.7, 5.9])
b = torch.nonzero(a > 6, as_tuple=False)
c = torch.index_select(a, dim = 0, index = b.squeeze())
print(c)
# tensor([9.3000, 8.5000])


# 5. torch.where使用
a = torch.randn(3,4) #.to('cuda')
print(a)
# tensor([[ 1.5248,  0.4148, -0.5237, -0.8694],
#         [ 0.6598, -3.0501, -1.2047, -0.9820],
#         [-0.3626,  0.8303, -0.1453, -1.3892]])
print(torch.where(a > 0))
# (tensor([0, 0, 1, 2]), tensor([0, 1, 0, 1]))


a[torch.where(a > 0)] = a[torch.where(a > 0)].mean()
a[torch.where(a < 0)] = a[torch.where(a < 0)].mean()
print(a)
# tensor([[ 0.8574,  0.8574, -1.0659, -1.0659],
#         [ 0.8574, -1.0659, -1.0659, -1.0659],
#         [-1.0659,  0.8574, -1.0659, -1.0659]])



















































































































