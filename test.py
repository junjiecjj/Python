#!/usr/bin/env python
# -*- coding: utf-8 -*-



import torch
from torch.nn import functional as f
 
import torch

x = torch.Tensor([[1], [2], [3]])
x0 = x.size(0)  # 取x第一维的尺寸，x0 = 3
x1 = x.expand(-1, 2)
x2 = x.expand(3, 2)


