#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:27:40 2022

@author: jack
1. 相同点
都是返回转置后矩阵。
都可以操作高纬矩阵，permute在高维的功能性更强。

2.不同点
a.合法性不同
torch.transpose(x)合法， x.transpose()合法。
tensor.permute(x)不合法，x.permute()合法。


b. 操作dim不同：
transpose()只能一次操作两个维度；permute()可以一次操作多维数据，且必须传入所有维度数，因为permute()的参数是int*。


c. transpose()中的dim没有数的大小区分；permute()中的dim有数的大小区分

d.  4.关于连续contiguous()


调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一毛一样。
（这一段看文字你肯定不理解，你也可以不用理解，有空我会画图补上）

只需要记住了，每次在使用view()之前，该tensor只要使用了transpose()和permute()这两个函数一定要contiguous().

"""

import torch 
import numpy as np

# 创造二维数据x，dim=0时候2，dim=1时候3
x = torch.randn(2,3)     #  'x.shape  →  [2,3]'
# 创造三维数据y，dim=0时候2，dim=1时候3，dim=2时候4
y = torch.randn(2,3,4)  # 'y.shape  →  [2,3,4]'


# 对于transpose
x.transpose(0,1)   #  'shape→[3,2] '  
x.transpose(1,0)   #  'shape→[3,2] '  
y.transpose(0,1)   #  'shape→[3,2,4]' 
#y.transpose(0,2,1)  #'error，操作不了多维'

# 对于permute()
x.permute(0,1)    # 'shape→[2,3]'
x.permute(1,0)    # 'shape→[3,2], 注意返回的shape不同于x.transpose(1,0) '
#y.permute(0,1)    # "error 没有传入所有维度数"
y.permute(1,0,2)  #'shape→[3,2,4]'



# 对于transpose，不区分dim大小
x1 = x.transpose(0,1)   #'shape→[3,2] '  
x2 = x.transpose(1,0)   #'也变换了，shape→[3,2] '  
print(torch.equal(x1,x2))
#' True ，value和shape都一样'

# 对于permute()
x1 = x.permute(0,1)    # '不同transpose，shape→[2,3] '  
x2 = x.permute(1,0)     #'shape→[3,2] '  
print(torch.equal(x1,x2))
#'False，和transpose不同'

y1 = y.permute(0,1,2)     #'保持不变，shape→[2,3,4] '  
y2 = y.permute(1,0,2)     #'shape→[3,2,4] '  
y3 = y.permute(1,2,0)     #'shape→[3,4,2] '  



x = torch.rand(3,4)
x = x.transpose(0,1)
print(x.is_contiguous()) # 是否连续
#'False'
# 再view会发现报错
# x.view(3,4)
'''报错
RuntimeError: invalid argument 2: view size is not compatible with input tensor's....
'''

# 但是下面这样是不会报错。
x = x.contiguous()
x.view(3,4)




x = torch.rand(3,4)
x = x.permute(1,0) # 等价x = x.transpose(0,1)
x.reshape(3,4)
'''这就不报错了
说明x.reshape(3,4) 这个操作
等于x = x.contiguous().view()
尽管如此，但是torch文档中还是不推荐使用reshape
理由是除非为了获取完全不同但是数据相同的克隆体
'''
 


























