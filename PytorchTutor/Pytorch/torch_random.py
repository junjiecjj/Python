#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 20:40:34 2023

@author: jack

该文件是总结torch生成指定分布，和以指定概率生成数组的总结
"""

import torch


##==================== 以指定概率生成伯努利分布的数组 ===================================================
p = torch.empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
a = torch.bernoulli(p)
print(a)
# tensor([[0., 1., 0.],
#         [1., 1., 1.],
#         [0., 0., 0.]])

p = torch.ones(3, 3) # probability of drawing "1" is 1
a = torch.bernoulli(p)
print(a)
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])

p = torch.zeros(3, 3) # probability of drawing "1" is 0
a =  torch.bernoulli(p)
print(a)
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]])


torch.distributions.bernoulli.Bernoulli(
    probs=None,
    logits=None,
    validate_args=None)





























