#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 20:36:40 2022

@author: jack
总之，两者都是用来重塑tensor的shape的。view只适合对满足连续性条件（contiguous）的tensor进行操作，
而reshape同时还可以对不满足连续性条件的tensor进行操作，具有更好的鲁棒性。view能干的reshape都能干，如果view不能干就可以用reshape来处理。



相同点：都是可以改变tensor的形状
不同点：
1   .view()方法只能改变连续的(contiguous)张量，否则需要先调用.contiguous()方法；而.reshape()方法不受此限制；如果对 tensor 调用过 transpose, 
permute等操作的话会使该 tensor 在内存中变得不再连续。

2    .view()方法返回的张量与原张量共享基础数据(存储器，注意不是共享内存地址)；.reshape()方法返回的可能是原张量的copy，也可能不是，这个我们不知道。


对“视图(view)”字眼的理解
视图是数据的一个别称或引用，通过该别称或引用亦便可访问、操作原有数据，但原有数据不会产生拷贝。如果我们对视图进行修改，它会影响到原始数据，物理内存在同一位置，这样避免了重新创建张量的高内存开销。由上面介绍的PyTorch的张量存储方式可以理解为：对张量的大部分操作就是视图操作！

与之对应的概念就是副本。副本是一个数据的完整的拷贝，如果我们对副本进行修改，它不会影响到原始数据，物理内存不在同一位置。
有关视图与副本，在NumPy中也有着重要的应用。


"""



#======================================================================================
#   
#   https://blog.csdn.net/Flag_ing/article/details/109129752
#======================================================================================



import torch
a = torch.arange(5)  # 初始化张量 a 为 [0, 1, 2, 3, 4]
b = a[2:]            # 截取张量a的部分值并赋值给b，b其实只是改变了a对数据的索引方式
print('a:', a)
print('b:', b)
print('ptr of storage of a:', a.storage().data_ptr())  # 打印a的存储区地址
print('ptr of storage of b:', b.storage().data_ptr())  # 打印b的存储区地址,可以发现两者是共用存储区
 
print('==================================================================')
 
b[1] = 0    # 修改b中索引为1，即a中索引为3的数据为0
print('a:', a)
print('b:', b)
print('ptr of storage of a:', a.storage().data_ptr())  # 打印a的存储区地址,可以发现a的相应位置的值也跟着改变，说明两者是共用存储区
print('ptr of storage of b:', b.storage().data_ptr())  # 打印b的存储区地址
 
 
'''   运行结果 

a: tensor([0, 1, 2, 3, 4])
b: tensor([2, 3, 4])
ptr of storage of a: 94824007650304
ptr of storage of b: 94824007650304
==================================================================
a: tensor([0, 1, 2, 0, 4])
b: tensor([2, 0, 4])
ptr of storage of a: 94824007650304
ptr of storage of b: 94824007650304

'''

import torch
a = torch.arange(6).reshape(2, 3)  # 初始化张量 a
b = torch.arange(6).view(3, 2)     # 初始化张量 b
print('a:', a)
print('stride of a:', a.stride())  # 打印a的stride
print('b:', b)
print('stride of b:', b.stride())  # 打印b的stride

'''   运行结果   

a: tensor([[0, 1, 2],
        [3, 4, 5]])
stride of a: (3, 1)
b: tensor([[0, 1],
        [2, 3],
        [4, 5]])
stride of b: (2, 1)


'''


import torch
a = torch.arange(9).reshape(3, 3)  # 初始化张量a
print('struct of a:\n', a)
print('size   of a:', a.size())    # 查看a的shape
print('stride of a:', a.stride())  # 查看a的stride
 
'''   运行结果

struct of a:
 tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
size   of a: torch.Size([3, 3])
stride of a: (3, 1)

'''


import torch
a = torch.arange(9).reshape(3, 3)     # 初始化张量a
b = a.permute(1, 0)  # 对a进行转置
print('struct of b:\n', b)
print('size   of b:', b.size())    # 查看b的shape
print('stride of b:', b.stride())  # 查看b的stride
 
'''   运行结果

struct of b:
 tensor([[0, 3, 6],
        [1, 4, 7],
        [2, 5, 8]])
size   of b: torch.Size([3, 3])
stride of b: (1, 3)

'''



import torch
a = torch.arange(9).reshape(3, 3)             # 初始化张量a
print('ptr of storage of a: ', a.storage().data_ptr())  # 查看a的storage区的地址
print('storage of a: \n', a.storage())        # 查看a的storage区的数据存放形式
b = a.permute(1, 0)                           # 转置
print('ptr of storage of b: ', b.storage().data_ptr())  # 查看b的storage区的地址
print('storage of b: \n', b.storage())        # 查看b的storage区的数据存放形式
 
'''   运行结果   
ptr of storage of a:  94824009018240
storage of a: 
  0
 1
 2
 3
 4
 5
 6
 7
 8
[torch.LongStorage of size 9]
ptr of storage of b:  94824009018240
storage of b: 
  0
 1
 2
 3
 4
 5
 6
 7
 8
[torch.LongStorage of size 9]


'''


import torch
a = torch.arange(9).reshape(3, 3)             # 初始化张量a
print(a.view(9))
print('============================================')
b = a.permute(1, 0)  # 转置
print(b.view(9))

""" 运行结果 
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
============================================
Traceback (most recent call last):

  File "/tmp/ipykernel_19557/2053947720.py", line 6, in <module>
    print(b.view(9))

RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

"""


import torch
a = torch.arange(9).reshape(3, 3)      # 初始化张量a
print('storage of a:\n', a.storage())  # 查看a的stride
print('+++++++++++++++++++++++++++++++++++++++++++++++++')
b = a.permute(1, 0).contiguous()       # 转置,并转换为符合连续性条件的tensor
print('size    of b:', b.size())       # 查看b的shape
print('stride  of b:', b.stride())     # 查看b的stride
print('viewd      b:\n', b.view(9))    # 对b进行view操作，并打印结果
print('+++++++++++++++++++++++++++++++++++++++++++++++++')
print('storage of a:\n', a.storage())  # 查看a的存储空间
print('storage of b:\n', b.storage())  # 查看b的存储空间
print('+++++++++++++++++++++++++++++++++++++++++++++++++')
print('ptr of a:\n', a.storage().data_ptr())  # 查看a的存储空间地址
print('ptr of b:\n', b.storage().data_ptr())  # 查看b的存储空间地址

'''   运行结果   
storage of a:
  0
 1
 2
 3
 4
 5
 6
 7
 8
[torch.LongStorage of size 9]
+++++++++++++++++++++++++++++++++++++++++++++++++
size    of b: torch.Size([3, 3])
stride  of b: (3, 1)
viewd      b:
 tensor([0, 3, 6, 1, 4, 7, 2, 5, 8])
+++++++++++++++++++++++++++++++++++++++++++++++++
storage of a:
  0
 1
 2
 3
 4
 5
 6
 7
 8
[torch.LongStorage of size 9]
storage of b:
  0
 3
 6
 1
 4
 7
 2
 5
 8
[torch.LongStorage of size 9]
+++++++++++++++++++++++++++++++++++++++++++++++++
ptr of a:
 94824007728448
ptr of b:
 94823966505280


由上述结果可以看出，张量a与b已经是两个存在于不同存储区的张量了。也印证了contiguous()方法开辟了一个新的存储区给b，并改变了b原始存储区数据的存放顺序。对应文章开头提到的浅拷贝，这种开辟一个新的内存区的方式其实就是深拷贝。
'''


"""

torch的view()与reshape()方法都可以用来重塑tensor的shape，区别就是使用的条件不一样。view()方法只适用于满足连续性条件的tensor，并且该操作不会开辟新的内存空间，
只是产生了对原存储空间的一个新别称和引用，返回值是视图。而reshape()方法的返回值既可以是视图，也可以是副本，当满足连续性条件时返回view，
否则返回副本[ 此时等价于先调用contiguous()方法在使用view() ]。因此当不确能否使用view时，可以使用reshape。如果只是想简单地重塑一个tensor的shape，
那么就是用reshape，但是如果需要考虑内存的开销而且要确保重塑后的tensor与之前的tensor共享存储空间，那就使用view()。


"""



import torch 
import numpy as np

#======================================================================================
#   https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
#======================================================================================
x = torch.randn(4, 4)
print("x.size() = \n{}".format(x.size()))
#torch.Size([4, 4])


y = x.view(16)
print("y.size() = \n{}".format(y.size()))
#torch.Size([16])
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print("z.size() = \n{}".format(z.size()))
#torch.Size([2, 8])

a = torch.randn(1, 2, 3, 4)
print("a.size() = \n{}".format(a.size()))
#torch.Size([1, 2, 3, 4])
b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
print("b.size() = \n{}".format(b.size()))
#torch.Size([1, 3, 2, 4])
c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
print("c.size() = \n{}".format(c.size()))
#torch.Size([1, 3, 2, 4])
print("torch.equal(b, c) = \n{}".format(torch.equal(b, c)))
#False

x = torch.randn(4, 4)
y = x.view(torch.int32)
y[0, 0] = 1000000000


#======================================================================================
#   https://funian788.github.io/post/pytorch-view-reshape/
#======================================================================================
'''
torch.view()通过共享内存地址的方式使用原tensor的基础数据，通过改变数据读取方式来返回一个具有新shape的新tensor；
只能使用torch.Tensor.view()方式调用；在使用时要求新shape与原shape的尺寸兼容，即函数只能应用于内存中连续存储的tensor，
使用transpose、permute等函数改变tensor在内存内连续性后需使用contiguous()方法返回拷贝后的值再调用该函数。
'''
a = torch.arange(24).view(1,2,3,4)
b = a.view(1,3,2,4)     # b.shape: 1 * 3 * 2 * 4  
c = a.transpose(1,2)    # c.shape: 1 * 3 * 2 * 4
# d = c.view(2, 12)     # raise error because of the uncontinuous data.
d = c.contiguous().view(2, 12)  #contiguous()后，已经是a的一个独立拷贝了，这时候改变a，d不会变
print("a = \n{}, \nb = \n{}, \nc= \n{},\nd=\n{}".format(a,b,c,d))  

'''
a = 
tensor([[[[ 0,  1,  2,  3],
          [ 4,  5,  6,  7],
          [ 8,  9, 10, 11]],

         [[12, 13, 14, 15],
          [16, 17, 18, 19],
          [20, 21, 22, 23]]]]), 
b = 
tensor([[[[ 0,  1,  2,  3],
          [ 4,  5,  6,  7]],

         [[ 8,  9, 10, 11],
          [12, 13, 14, 15]],

         [[16, 17, 18, 19],
          [20, 21, 22, 23]]]]), 
c= 
tensor([[[[ 0,  1,  2,  3],
          [12, 13, 14, 15]],

         [[ 4,  5,  6,  7],
          [16, 17, 18, 19]],

         [[ 8,  9, 10, 11],
          [20, 21, 22, 23]]]]),
d=
tensor([[ 0,  1,  2,  3, 12, 13, 14, 15,  4,  5,  6,  7],
        [16, 17, 18, 19,  8,  9, 10, 11, 20, 21, 22, 23]])
'''
print(id(b) == id(c))           # False
print(id(b.data) == id(c.data)) # True

a[0, 0, :, :] = 100
print("a = \n{}, \nb = \n{}, \nc= \n{},\nd=\n{}".format(a,b,c,d)) # 'b' and 'c' will also change its data.
"""
a = 
tensor([[[[100, 100, 100, 100],
          [100, 100, 100, 100],
          [100, 100, 100, 100]],

         [[ 12,  13,  14,  15],
          [ 16,  17,  18,  19],
          [ 20,  21,  22,  23]]]]), 
b = 
tensor([[[[100, 100, 100, 100],
          [100, 100, 100, 100]],

         [[100, 100, 100, 100],
          [ 12,  13,  14,  15]],

         [[ 16,  17,  18,  19],
          [ 20,  21,  22,  23]]]]), 
c= 
tensor([[[[100, 100, 100, 100],
          [ 12,  13,  14,  15]],

         [[100, 100, 100, 100],
          [ 16,  17,  18,  19]],

         [[100, 100, 100, 100],
          [ 20,  21,  22,  23]]]]),
d=
tensor([[ 0,  1,  2,  3, 12, 13, 14, 15,  4,  5,  6,  7],
        [16, 17, 18, 19,  8,  9, 10, 11, 20, 21, 22, 23]])
"""
b[0, 0, :, :] = 31
print("a = \n{}, \nb = \n{}, \nc= \n{},\nd=\n{}".format(a,b,c,d)) # 'b' and 'c' will also change its data.
"""
a = 
tensor([[[[ 31,  31,  31,  31],
          [ 31,  31,  31,  31],
          [100, 100, 100, 100]],

         [[ 12,  13,  14,  15],
          [ 16,  17,  18,  19],
          [ 20,  21,  22,  23]]]]), 
b = 
tensor([[[[ 31,  31,  31,  31],
          [ 31,  31,  31,  31]],

         [[100, 100, 100, 100],
          [ 12,  13,  14,  15]],

         [[ 16,  17,  18,  19],
          [ 20,  21,  22,  23]]]]), 
c= 
tensor([[[[ 31,  31,  31,  31],
          [ 12,  13,  14,  15]],

         [[ 31,  31,  31,  31],
          [ 16,  17,  18,  19]],

         [[100, 100, 100, 100],
          [ 20,  21,  22,  23]]]]),
d=
tensor([[ 0,  1,  2,  3, 12, 13, 14, 15,  4,  5,  6,  7],
        [16, 17, 18, 19,  8,  9, 10, 11, 20, 21, 22, 23]])
"""

a = torch.zeros(3, 2)
b = a.reshape(6)
c = a.t().reshape(6) #a.t()后已经是a的拷贝了，这时候改变a，c不变
print("a = \n{}, \nb = \n{}, \nc= \n{}".format(a,b,c,))  
"""
a = 
tensor([[0., 0.],
        [0., 0.],
        [0., 0.]]), 
b = 
tensor([0., 0., 0., 0., 0., 0.]), 
c= 
tensor([0., 0., 0., 0., 0., 0.])
"""
a.fill_(1)
print("a = \n{}, \nb = \n{}, \nc= \n{}".format(a,b,c,))  
"""
a = 
tensor([[1., 1.],
        [1., 1.],
        [1., 1.]]), 
b = 
tensor([1., 1., 1., 1., 1., 1.]), 
c= 
tensor([0., 0., 0., 0., 0., 0.])
"""



#=========================================
# Pytorch与TensorFlow对比
#=========================================
a = torch.zeros(6,5)
for i in range(6):
    a[i,:] = i
a1=a.T

b = a.view(2,5,3)
c = a.reshape(2,5,3)


d = a.reshape(2,3,5)
e = a.reshape(2,3,5).transpose(1,2)


print("a = \n{},\na1 = \n{}, \nb = \n{}, \nc= \n{},\nd=\n{},\ne = \n{}".format(a,a1,b,c,d,e))  



#=========================================
# https://www.cnblogs.com/sddai/p/14403333.html
#=========================================
import torch 
import numpy as np

a = torch.randint(0, 10, (3, 4))
b = a.view(2, 6)
c = a.reshape(2, 6)
print("a = \n{}, \nb = \n{}, \nc= \n{}".format(a,b,c,))  
"""
a = 
tensor([[3, 6, 9, 9],
        [5, 5, 0, 3],
        [2, 2, 1, 2]]), 
b = 
tensor([[3, 6, 9, 9, 5, 5],
        [0, 3, 2, 2, 1, 2]]), 
c= 
tensor([[3, 6, 9, 9, 5, 5],
        [0, 3, 2, 2, 1, 2]])
"""

# 非严格意义上讲，id可以认为是对象的内存地址
print(id(a)==id(b), id(a)==id(c), id(b)==id(c))
"""
前提：python的变量和数据是保存在不同的内存空间中的，PyTorch中的Tensor的存储也是类似的机制，tensor相当于python变量，
保存了tensor的形状(size)、步长(stride)、数据类型(type)等信息(或其引用)，当然也保存了对其对应的存储器Storage的引用，
存储器Storage就是对数据data的封装。
viewed对象和reshaped对象都存储在与原始对象不同的地址内存中，但是它们共享存储器Storage，也就意味着它们共享基础数据。
"""
print(id(a.storage())==id(b.storage()),
      id(a.storage())==id(c.storage()),
      id(b.storage())==id(c.storage()))
"""
Out:
False False False
True True True
"""
 
a[0]=0
print("a = \n{}, \nb = \n{}, \nc= \n{}".format(a,b,c,))  
"""
a = 
tensor([[0, 0, 0, 0],
        [5, 5, 0, 3],
        [2, 2, 1, 2]]), 
b = 
tensor([[0, 0, 0, 0, 5, 5],
        [0, 3, 2, 2, 1, 2]]), 
c= 
tensor([[0, 0, 0, 0, 5, 5],
        [0, 3, 2, 2, 1, 2]])
"""
 
c[0]=32
print("a = \n{}, \nb = \n{}, \nc= \n{}".format(a,b,c,))  
"""
a = 
tensor([[32, 32, 32, 32],
        [32, 32,  1,  2],
        [ 3,  3,  2,  5]]), 
b = 
tensor([[32, 32, 32, 32, 32, 32],
        [ 1,  2,  3,  3,  2,  5]]), 
c= 
tensor([[32, 32, 32, 32, 32, 32],
        [ 1,  2,  3,  3,  2,  5]])
"""


"""
torch.Tensor.resize_()
torch.Tensor.resize_() 方法的功能跟.reshape() / .view()方法的功能一样，也是将原张量元素(按顺序)重组为新的shape。

当resize前后的shape兼容时，返回原张量的视图(view)；当目标大小(resize后的总元素数)大于当前大小(resize前的总元素数)时，基础存储器的大小将改变(即增大)，
以适应新的元素数，任何新的内存(新元素值)都是未初始化的；当目标大小(resize后的总元素数)小于当前大小(resize前的总元素数)时，
基础存储器的大小保持不变，返回目标大小的元素重组后的张量，未使用的元素仍然保存在存储器中，如果再次resize回原来的大小，这些元素将会被重新使用。

(这里说的shape兼容的意思是：resize前后的shape包含的总元素数是一致的，即resize前后的shape的所有维度的乘积是相同的。如resize前，shape为(1, 2 ,3)，
 那resize之后的张量的总元素数需要是1*2*3，故目标shape可以是(2, 3)， 可以是(3, 2, 1)，可以是(2, 1, 3)等尺寸。)

–> 文字说明有点干燥，看点例子感受一下：
"""

a = torch.arange(12).view(3, 4)
b = a.resize_(4, 3)
c = a.resize_(3, 3)
d = a.resize_(3, 5)

print("a = \n{}, \nb = \n{}, \nc= \n{},\nd= \n{}".format(a,b,c,d))  
"""
a = 
tensor([[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11,  0,  0,  0]]), 
b = 
tensor([[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11,  0,  0,  0]]), 
c= 
tensor([[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11,  0,  0,  0]]),
d= 
tensor([[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11,  0,  0,  0]])
"""


#=========================================
#  https://www.daimajiaoliu.com/daima/60eb2b2a899ec09
#=========================================

"""
view()只可以由torch.Tensor.view()来调用
view()和reshape()在效果上是一样的，区别是view（）只能操作contiguous的tensor，且view后的tensor和原tensor共享存储，reshape（）对于是否contiuous的tensor都可以操作。
"""

a = np.arange(24)
b = a.reshape(4,3,2)
print("a = \n{}, \nb = \n{}, ".format(a,b, ))  


"""
将输入数据input的第dim0维和dim1维进行交换 
torch.transpose(input, dim0, dim1) -> Tensor
"""
a = torch.randn(2, 3)
b = torch.transpose(a, 0, 1)
print("a = \n{}, \nb = \n{}, ".format(a,b, ))  



"""
torch.flatten()的输入是tensor
torch.flatten(input, start_dim=0, end_dim=-1) → Tensor
其作用是将输入tensor的第start_dim维到end_dim维之间的数据“拉平”成一维tensor，
"""


a = torch.tensor([[[1, 2],[3, 4]],[[5, 6],[7, 8]]])
b = torch.flatten(a)
c = torch.flatten(a, start_dim=1)
print("a = \n{}, \nb = \n{}, \nc= \n{}, ".format(a,b,c, ))  







"""
这里以torch.Tensor下的reshape，view，resize_来举例

一、先来说一说reshape和view之间的区别
相同点：都是可以改变tensor的形状

不同点：

.view()方法只能改变连续的(contiguous)张量，否则需要先调用.contiguous()方法；而.reshape()方法不受此限制；如果对 tensor 调用过 transpose, permute等操作的话会使该 tensor 在内存中变得不再连续。

.view()方法返回的张量与原张量共享基础数据(存储器，注意不是共享内存地址)；.reshape()方法返回的可能是原张量的copy，也可能不是，这个我们不知道。

"""


a = torch.randint(0, 10, (3, 4))
b = a.permute(1,0)
c = a.transpose(0,1)
print("a = \n{}, \nb = \n{}, \nc= \n{}, ".format(a,b,c, ))  
'''
a = 
tensor([[7, 2, 6, 1],
        [6, 6, 9, 6],
        [5, 8, 9, 5]]), 
b = 
tensor([[7, 6, 5],
        [2, 6, 8],
        [6, 9, 9],
        [1, 6, 5]]), 
'''
print("a.is_contiguous() = %s \nb.is_contiguous() = %s \nc.is_contiguous() = %s"%(a.is_contiguous(),b.is_contiguous(),c.is_contiguous()))


#非连续的情况下使用.view和.reshape:
#d=c.view(3,4)     #error

d=c.reshape(3,4)
print("d = \n{}".format(d))
#可以看出在非连续的情况下，是不能使用 view 的。
#现在我们使用.contiguous()将tensor变成连续的，然后再使用.view和.reshape:

e=c.contiguous()
print("e.is_contiguous() = %s"%(e.is_contiguous())) #输出：True

f=e.view(1,12)

g=e.reshape(2,6)
print("e.shape = {}, f.shape = {}, g.shape = {} ".format(e.shape,f.shape,g.shape))

'''

'''





"""
二、再来说一说reshape/view和resize_之间的区别
它们之间的区别就比较明显，前者在改变形状的时候，总的数据个数不能变，而后者在改变形状的时候是可以只截取一部分数据的。看下面的例子：
"""
x = torch.tensor([[1, 2], [3, 4], [5, 6]])
x.resize_(2, 2)
print(x)

'''
输出:tensor([[1, 2],
             [3, 4]])
'''
x = torch.tensor([[1, 2], [3, 4], [5, 6]])
#x=x.reshape(2, 2)  #  error
print(x)







# https://blog.csdn.net/weixin_43002433/article/details/104325931

import copy
import torch
import numpy as np

# 创建原始tensor/ndarray对象
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]]
t0 = torch.tensor(data)  # Out: tensor([[1, 2, 3],
        				 #              [4, 5, 6],
        				 # 				[7, 8, 9],
        				 #				[0, 0, 0]])


# view
t1 = t0.view(3, 4)  # Out: tensor[[1, 2, 3, 4], 
					#            [5, 6, 7, 8], 
					# 			 [9, 0, 0, 0]]  

print(t1.shape,)  # Out: torch.Size([3, 4]) (3, 4)
print(id(t1)==id(t0))  # False False
print(id(t1[0])==id(t0[0]))  # True True 
print(id(t1[0][0])==id(t0[0][0]))  # True/False True (注意这里第一个我用了/，因为这里每次运行的结果可能是不一样的，原因不明，可能跟数据存储有关)

# copy.copy
t2 = copy.copy(t0)  # Out: tensor([[1, 2, 3],
        			#             [4, 5, 6],
        			# 			  [7, 8, 9],
        			#			  [0, 0, 0]])

print(id(t2)==id(t0))  # False False
print(id(t2[0])==id(t0[0]))  # True True
print(id(t2[0][0])==id(t0[0][0]))  # True/False True (注意这里第一个我用了/，因为这里每次运行的结果可能是不一样的，原因不明，可能跟数据存储有关)

# 改变原始对象的元素
t0[-1] = 999
print(t0[-1], t1[-1], t2[-1])  # Out: tensor([999, 999, 999]) tensor([  9, 999, 999, 999]) tensor([999, 999, 999])


t0[0][0] = 666
print(t0[0][0], t1[0][0], t2[0][0])  # Out: tensor(666) tensor(666) tensor(666)


t2[1] = 0
print(t0[1], t1[1], t2[1])  # Out: tensor([0, 0, 0]) tensor([5, 6, 7, 8]) tensor([0, 0, 0])



import copy
import torch
import numpy as np
a = torch.arange(12).reshape(4,3) +1 
# view
b = a.view(3, 4)    #
# copy
c = copy.copy(a)

#返回一个和源张量同shape、dtype和device的张量，与源张量不共享数据内存，但提供梯度的回溯。
d = a.clone()  

#detach的机制则与clone完全不同，即返回一个和源张量同shape、dtype和device的张量，与源张量共享数据内存，但不提供梯度计算，即requires_grad=False，因此脱离计算图。
e = a.detach()

# clone提供了非数据共享的梯度追溯功能，而detach又“舍弃”了梯度功能，因此clone和detach意味着着只做简单的数据复制，既不数据共享，也不对梯度共享，从此两个张量无关联。
# 置于是先clone还是先detach，其返回值一样，一般采用tensor.clone().detach()。
f = a.clone().detach()


print("\n\na = \n{},\nb=\n{},\nc =\n{},\nd =\n{},\ne =\n{},\nf =\n{}".format(a,b,c,d,e,f))

a[0,0] = 77
print("\n\na = \n{},\nb=\n{},\nc =\n{},\nd =\n{},\ne =\n{},\nf =\n{}".format(a,b,c,d,e,f))



b[1,0] = 33
print("\n\na = \n{},\nb=\n{},\nc =\n{},\nd =\n{},\ne =\n{},\nf =\n{}".format(a,b,c,d,e,f))



c[2,0] = 54
print("\n\na = \n{},\nb=\n{},\nc =\n{},\nd =\n{},\ne =\n{},\nf =\n{}".format(a,b,c,d,e,f))


d[3,0] = 67
print("\n\na = \n{},\nb=\n{},\nc =\n{},\nd =\n{},\ne =\n{},\nf =\n{}".format(a,b,c,d,e,f))


e[1,1] = 321
print("\n\na = \n{},\nb=\n{},\nc =\n{},\nd =\n{},\ne =\n{},\nf =\n{}".format(a,b,c,d,e,f))

f[0,-1] = 98
print("\n\na = \n{},\nb=\n{},\nc =\n{},\nd =\n{},\ne =\n{},\nf =\n{}".format(a,b,c,d,e,f))




































































































































































































































































































































































































































































































































