#!/usr/bin/env python
# -*- coding: utf-8 -*-



import torch
from torch.nn import functional as f
 
x = torch.arange(0, 1*3*15*15).float()
x = x.view(1,3,15,15)
# print(x)
x1 = f.unfold(x, kernel_size=3, dilation=1, stride=1)
print(f"x1.shape = {x1.shape}")
# x1.shape = torch.Size([1, 27, 169])

B, C_kh_kw, L = x1.size()
x1 = x1.permute(0, 2, 1)
print(f"x1.shape = {x1.shape}")
# x1.shape = torch.Size([1, 169, 27])


x1 = x1.view(B, L, -1, 3, 3)
print(f"x1.shape = {x1.shape}")
# x1.shape = torch.Size([1, 169, 3, 3, 3])
#print(x1)


#==============================================================================

x = torch.arange(0, 1*3*5*5).float()
x = x.view(1,3,5,5)
# print(x)
x1 = f.unfold(x, kernel_size=2, )
print(f"x1.shape = {x1.shape}")
# x1.shape = torch.Size([1, 12, 16])

B, C_kh_kw, L = x1.size()
x2 = x1.permute(0, 2, 1)
print(f"x2.shape = {x2.shape}")
# x2.shape = torch.Size([1, 16, 12])


x3 = x2.view(B, L, -1, 2, 2)
print(f"x3.shape = {x3.shape}")
# x3.shape = torch.Size([1, 16, 3, 2, 2])
#print(x1)



#==============================================================================

x = torch.arange(0, 1*3*5*5).float()
x = x.view(1,3,5,5)
x1= torch.nn.functional.unfold(x,2, )
print(f"x1.shape = {x1.shape}")
# x1.shape = torch.Size([1, 12, 16])

x2 = x1.transpose(1,2)
print(f"x2.shape = {x2.shape}")
# x2.shape = torch.Size([1, 16, 12])


x3 = x2.transpose(0,1)
print(f"x3.shape = {x3.shape}")
# x3.shape = torch.Size([16, 1, 12])








