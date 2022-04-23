#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:06:47 2022

@author: jack
"""

import torch 
import numpy as np







# https://aitechtogether.com/article/6956.html
#======================================================================================
#  1.tensor 初始化
#======================================================================================

# 直接数据
data=[[1,2],[3,4]]
x_data=torch.tensor(data)


# numpy 数组
np_array=np.array(data)
x_np=torch.from_numpy(np_array)


# 从另一个tensor
x_ones=torch.ones_like(x_data)#保留shape,datatype
print(f'ones tensor:\n{x_ones}\n')
x_rands=torch.rand_like(x_data,dtype=torch.float)#保留shape
print(f'random tensor:\n{x_rands}\n')




shape=(2,3,)
rand_tensor=torch.rand(shape)
ones_tensor=torch.ones(shape)
zeros_tensor=torch.zeros(shape)
print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)

rand_tensor=torch.randint(3,20,(4,5))
print(rand_tensor)


#比如有个张量a，那么a.normal_()就表示用标准正态分布填充a，是in_place操作，如下图所示：
a = torch.ones([2,3])
print(f"a = {a}")

a.normal_(mean=0, std=0.01)
print(f"a = {a}")


#比如有个张量b，那么b.fill_(0)就表示用0填充b，是in_place操作，如下图所示：
a = torch.rand([2,3])
print(f"a = {a}")

a.fill_(0)
print(f"a = {a}")

#======================================================================================
#    2.tensor 性质
#======================================================================================

#shape,datatype,device(存储位置）

tensor=torch.rand(3,4)
print(tensor.shape,'\n',tensor.dtype,'\n',tensor.device)


#======================================================================================
#   3.tensor 运算
#======================================================================================

# 转置、索引、切片、数学、线性代数、随机采样

# 索引和切片
tensor=torch.ones(4,4)
tensor[:,1]=0
print(tensor)


# 连接
t1=torch.cat([tensor,tensor,tensor],dim=1)
print("t1 = \n{}".format(t1))




# 数字乘
print("tensor.mul(tensor) = \n{}".format(tensor.mul(tensor) ))
print(" tensor*tensor = \n{}".format( tensor*tensor ))



# 矩阵乘
print("tensor.matmul(tensor.T) = \n{}".format(tensor.matmul(tensor.T)))
print("tensor@tensor.T = \n{}".format(tensor@tensor.T))



# 就地操作_
print(tensor)
tensor.add_(4)
print(tensor)


#======================================================================================
#   4.bridge numpy
#======================================================================================


# tensor-->numpy
t=torch.ones(5)
print(f'Tensor:{t}')
n=t.numpy()
print(f'numpy:{n}')

# tensor变化会在numpy中反应
t.add_(1)
print(t)
print(n)



# numpy-->tensor
n=np.ones(5)
t=torch.from_numpy(n)
np.add(n,1,out=n)
print(t)
print(n)



#======================================================================================
#   differentiation in autogradnumpy
#======================================================================================
import torch
#requires_grad=True:every operation on them should be tracked.
a=torch.tensor([2.,3.],requires_grad=True)
b=torch.tensor([6.,4.],requires_grad=True)
 
#a,b是NN参数，Q是误差
Q=3*a**3-b**2
 
external_grad=torch.tensor([1,1])
#Q.backward:计算Q对a,b的gradients并储存在tensor.grad中
Q.backward(gradient=external_grad)
print(a.grad)
print(b.grad)
 







# https://blog.csdn.net/luschka/article/details/114750073
import torch
tensor = torch.Tensor([[2,3],[4,5],[6,7]]) #生成tensor
E = torch.from_numpy #使用numpy生成tensor
torch_e = torch.from_numpy
print(format(tensor)) #3*2
print(tensor.reshape(2,3)) #重新排列元素
print(tensor + 1)#逐元素运算
t_int8 = torch.tensor([1, 2], dtype = torch.int8)
t1 = torch.empty(2) #未初始化
t2 = torch.zeros(2, 2) #所有元素值为0
t3 = torch.ones(2, 2, 2) # 所有元素值为1
t4 = torch.full((2,2,2,2), 3.) #所有元素值为3
t5 = torch.ones_like(t2) #ones可以被替换为zeros、empty、full使用


T = torch.tensor([[1,2,3],[4,5,6]])

print(format(T))

print(T.size())


print(T.dim())


print(T.numel())

print(format(T.dtype))

#======================================================================================
# # 等差数列
#======================================================================================

print("torch.arange(0, 4, step = 1) = \n{}".format(torch.arange(0, 4, step = 1))) #从0到3，步长是1
#print("torch.range(0, 3, step = 1) = \n{}".format(torch.range(0, 3, step = 1)))
print("torch.linspace(0,3, steps = 4) = \n{}".format(torch.linspace(0,3, steps = 4))) #表示只有4个元素

#======================================================================================
# 等比数列
#======================================================================================

torch.logspace(0, 3, steps = 4) #生成张量为1，10， 100， 1000


#torch.rand() & torch.rand_like()
#生成标准均匀分布的随机变量。标准均匀分布在[0, 1)上的概率都相同
print(torch.rand(2, 3))
print(torch.rand_like(torch.ones(2, 3)))

#======================================================================================
#   各种分布
#======================================================================================
# torch.normal()
#生成正态分布的随机变量，不同之处在于它可以指定正态分布的均值和方差

mean = torch.tensor([0., 1.])
std = torch.tensor([3., 2.])
print("torch.normal(mean, std) = \n{}".format(torch.normal(mean, std)))
#输出
#tensor([-1.7072,  3.5845])

tc = torch.arange(12)
print('tc = {}'.format(tc))
t322 = tc.reshape(3,2,2) #张量大小设置为322或者也可以是（4，3）
print('t322 = {}'.format(t322))



t = torch.arange(24).reshape(2, 1, 3, 1, 4) #大小为(2,1,3,1,4)
t.squeeze() #大小=(2, 3, 4)
t = torch.arange(24).reshape(2, 3, 4)

 
#use permute to order tensor in a new way which is defined by dims
t = torch.arange(24).reshape(1, 2, 3, 4)
t.permute(dims = [2, 0, 1, 3]) #大小[1, 2, 3, 4]
#transpose the tensor
t12 = torch.tensor([[5., -9.], ])
t21 = t12.transpose(0, 1)

 
t = torch.arange(24).reshape(2, 3, 4)
index = torch.tensor([1, 2])
t.index_select(1, index)
#选取大小为(2, 2, 4)


#选取部分张量，张量的扩展和拼接
t = torch.arange(12)
print(t[3])#第三个
print(t[-5])#倒数第五个
print(t[3:6])#从第3到第6个
print(t[:6])#前6个
print(t[3:])#从第三个往后
print(t[-5:])#倒数后五个
print(t[3:6:2])#三和六两个元素
print(t[3::2])#从三开始步长为2
#repeat
t12 = torch.tensor([[5., -9.], ])
print('t12 = {}'.format(t12))
t34 = t12.repeat(3, 2)#把t12作为一个元素重复三行两列
print('t34 = {}'.format(t34))
#cat
tp = torch.arange(12).reshape(3, 4)
tn = -tp
tc0 = torch.cat([tp, tn], 0)#参数为0则表示竖直拼接
print('tc0 = {}'.format(tc0))
tc1 = torch.cat([tp, tp, tn, -tn], 1)#参数1则表示横向拼接
print('tc1 = {}'.format(tc1))
#stack 输入的张量大小需要完全相同
tp = torch.arange(12).reshape(3, 4)
tn = -tp
tc0 = torch.stack([tp, tn], 0)#参数为0则表示竖直拼接
print('tc0 = {}'.format(tc0))
tc1 = torch.stack([tp, tp, tn, -tn], 1)#参数1则表示横向拼接
print('tc1 = {}'.format(tc1))




#======================================================================================
# 有理运算和广播语义
#======================================================================================

tl = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
tr = torch.tensor([[7., 8., 9.], [10., 11., 12.]])
print(tl+tr)
print(tl-tr)
print(tl*tr)
print(tl/tr)
print(tl**tr)
print(tl**(1/tr))
print(tl+5)
print(-6*tr)
print(torch.ones(2, 3, 4)+torch.ones(4))

print(t1.reciprocal())#求倒数，数据类型不能为long
print(t1.sqrt())#求平方根


#======================================================================================
#  统计函数 
#======================================================================================

t = torch.arange(5)*0.3
print(t.var())#方差
print(t.prod())#各元素乘积
print(t.max())#最大值
print(t.median())#中位数
print(t.kthvalue())#第2大值
t1 = torch.arange(12).reshape(2, 2, 3)

#各元素的方差，添加dim参数后就是对某一个维度进行统计
print(t1.prod(dim = 1))
print(t1.var(dim = 1))



# https://blog.csdn.net/qq_36810398/article/details/104845401
import numpy as np
import torch

x = [[[1,2,3,4],[5,6,7,8],[9,10,11,12]],
[[13,14,15,16],[17,18,19,20],[21,22,23,24]]]
x = torch.tensor(x).float()
#
print("shape of x:")  ##[2,3,4]                                                                      
print(x.shape)                                                                                   
#
print("shape of x.mean(axis=0,keepdim=True):")          #[1, 3, 4]
print(x.mean(axis=0,keepdim=True).shape)                       
#
print("shape of x.mean(axis=0,keepdim=False):")         #[3, 4]
print(x.mean(axis=0,keepdim=False).shape)                     
#
print("shape of x.mean(axis=1,keepdim=True):")          #[2, 1, 4]
print(x.mean(axis=1,keepdim=True).shape)                      
#
print("shape of x.mean(axis=1,keepdim=False):")         #[2, 4]
print(x.mean(axis=1,keepdim=False).shape)                    
#
print("shape of x.mean(axis=2,keepdim=True):")          #[2, 3, 1]
print(x.mean(axis=2,keepdim=True).shape)                     
#
print("shape of x.mean(axis=2,keepdim=False):")         #[2, 3]
print(x.mean(axis=2,keepdim=False).shape)                  




#基本类型
#Tensor的基本数据类型有五种： 
#- 32位浮点型：torch.FloatTensor。 (默认) 
#- 64位整型：torch.LongTensor。 
#- 32位整型：torch.IntTensor。 
#- 16位整型：torch.ShortTensor。 
#- 64位浮点型：torch.DoubleTensor。

#除以上数字类型外，还有 byte和chart型
tensor = torch.tensor([3.1433223]) 
print(f"tensor = {tensor}\n")
print(f"tensor.dtype = {tensor.dtype}\n")
# tensor = tensor([3.1433])
# tensor.dtype = torch.float32

double = tensor.double()
print(f"double = {double}\n")
print(f"double.dtype = {double.dtype}\n")
# double = tensor([3.1433], dtype=torch.float64)
# double.dtype = torch.float64

do = torch.DoubleTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
print(f"do = {do}\n")
print(f"do.dtype = {do.dtype}\n")
# do.dtype = torch.float64

flo = tensor.float()
print(f"flo = {flo}\n")
print(f"flo.dtype = {flo.dtype}\n")
# flo = tensor([3.1433])
# flo.dtype = torch.float32

fl = torch.FloatTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
print(f"fl = {fl}\n")
print(f"fl.dtype = {fl.dtype}\n")
# fl.dtype = torch.float32


half=tensor.half()
print(f"half = {half}\n")
print(f"half.dtype = {half.dtype}\n")
# half = tensor([3.1426], dtype=torch.float16)
# half.dtype = torch.float16

ha = torch.HalfTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
print(f"ha = {ha}\n")
print(f"ha.dtype = {ha.dtype}\n")
#ha.dtype = torch.float16

long=tensor.long()
print(f"long = {long}\n")
print(f"long.dtype = {long.dtype}\n")
# long = tensor([3])
# long.dtype = torch.int64

Lo = torch.LongTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
print(f"Lo = {Lo}\n")
print(f"Lo.dtype = {Lo.dtype}\n")
#Lo.dtype = torch.int64

int_t=tensor.int()
print(f"int_t = {int_t}\n")
print(f"int_t.dtype = {int_t.dtype}\n")
# int_t = tensor([3], dtype=torch.int32)
# int_t.dtype = torch.int32

In = torch.IntTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
print(f"In = {In}\n")
print(f"In.dtype = {In.dtype}\n")
# In.dtype = torch.int32


short = tensor.short()
print(f"short = {short}\n")
print(f"short.dtype = {short.dtype}\n")
#  = tensor([3], dtype=torch.int16)
# short.dtype = torch.int16

sh = torch.ShortTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
print(f"sh = {sh}\n")
print(f"sh.dtype = {sh.dtype}\n")
# sh.dtype = torch.int16

ch = tensor.char()
print(f"ch = {ch}\n")
print(f"ch.dtype = {ch.dtype}\n")
#  ch = tensor([3], dtype=torch.int8)
# ch.dtype = torch.int8

ch = torch.CharTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
print(f"ch = {ch}\n")
print(f"ch.dtype = {ch.dtype}\n")
# ch.dtype = torch.int8

bt = tensor.byte()
print(f"bt = {bt}\n")
print(f"bt.dtype = {bt.dtype}\n")
# bt = tensor([3], dtype=torch.uint8)
# bt.dtype = torch.uint8

bt = torch.ByteTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
print(f"bt = {bt}\n")
print(f"bt.dtype = {bt.dtype}\n")
# bt.dtype = torch.uint8


#================================================= data type chage=================================================
data = torch.ones(2, 2)
print(data.dtype)
#result: torch.float32
# 可能在操作过程中指定其他数据类型--这里就按照ones--对应int64类型
data = data.type(torch.float64)  # 要接收类型已经改变的tensor数据，否则data本身是不会直接改变数据类型的
print(data.dtype)
#result: torch.float64


data = data.type(torch.float32)  # 要接收类型已经改变的tensor数据，否则data本身是不会直接改变数据类型的
print(data.dtype)
#result: torch.float32


data = data.type(torch.float)  # 要接收类型已经改变的tensor数据，否则data本身是不会直接改变数据类型的
print(data.dtype)
#result: torch.float32

data = data.type(torch.float16)  # 要接收类型已经改变的tensor数据，否则data本身是不会直接改变数据类型的
print(data.dtype)
#result: torch.float16

data = data.type(torch.int64)  # 要接收类型已经改变的tensor数据，否则data本身是不会直接改变数据类型的
print(data.dtype)
#result: torch.int64

data = data.type(torch.int32)  # 要接收类型已经改变的tensor数据，否则data本身是不会直接改变数据类型的
print(data.dtype)
#result: torch.int32

data = data.type(torch.int16)  # 要接收类型已经改变的tensor数据，否则data本身是不会直接改变数据类型的
print(data.dtype)
# torch.int16

data = data.type(torch.int8)  # 要接收类型已经改变的tensor数据，否则data本身是不会直接改变数据类型的
print(data.dtype)
# torch.int8

data = data.type(torch.double)  # 要接收类型已经改变的tensor数据，否则data本身是不会直接改变数据类型的
print(data.dtype)
#result: torch.float64

data = data.type(torch.uchar)  # 要接收类型已经改变的tensor数据，否则data本身是不会直接改变数据类型的
print(data.dtype)
#result: torch.float64

data = data.type(torch.bool)  # 要接收类型已经改变的tensor数据，否则data本身是不会直接改变数据类型的
print(data.dtype)
# torch.bool

data = data.type(torch.uint8)  # 要接收类型已经改变的tensor数据，否则data本身是不会直接改变数据类型的
print(data.dtype)
# torch.uint8


data = torch.ones(2, 2)
data_float = torch.randn(2, 2,dtype=torch.float64)  # 这里的数据类型为torch.float64
print(data.dtype)
print(data_float.dtype)
# torch.float32
# torch.float64
# 可能在操作过程中指定其他数据类型--这里就按照ones--对应int64类型
data = data.type_as(data_float )
print(data.dtype)
#   torch.float64




"""
scatter(dim, index, src)的三个参数为：

（1）dim:沿着哪个维度进行索引

（2）index: 用来scatter的元素索引

（3）src: 用来scatter的源元素，可以使一个标量也可以是一个张量

官方给的例子为三维情况下的例子：


y = y.scatter(dim,index,src)
 
#则结果为：
y[ index[i][j][k]  ] [j][k] = src[i][j][k] # if dim == 0
y[i] [ index[i][j][k] ] [k] = src[i][j][k] # if dim == 1
y[i][j] [ index[i][j][k] ]  = src[i][j][k] # if dim == 2

如果是二维的例子，则应该对应下面的情况：

y = y.scatter(dim,index,src)
 
#则：
y [ index[i][j] ] [j] = src[i][j] #if dim==0
y[i] [ index[i][j] ]  = src[i][j] #if dim==1




"""




import torch
 
x = torch.randn(2,4)
print(x)
y = torch.zeros(3,4)
y = y.scatter_(0,torch.LongTensor([[2,1,2,2],[0,2,1,1]]),x)
print(y)


#那么这个函数有什么作用呢？其实可以利用这个功能将pytorch 中mini batch中的返回的label（特指[ 1,0,4,9 ]，即size为[4]这样的label）转为one-hot类型的label,举例子如下：
import torch
 
mini_batch = 4
out_planes = 6
out_put = torch.rand(mini_batch, out_planes)
softmax = torch.nn.Softmax(dim=1)
out_put = softmax(out_put)
 
print(out_put)
label = torch.tensor([1,3,3,5])
one_hot_label = torch.zeros(mini_batch, out_planes).scatter_(1,label.unsqueeze(1),1)
print(one_hot_label)


x = torch.rand(2, 5)
print(x)


torch.zeros(3, 5).scatter_(0, torch.LongTensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), x)

z = torch.zeros(2, 4).scatter_(1, torch.LongTensor([[2], [3]]), 1.23)
print(z)





#pytorch: gather函数，index_fill函数
a=torch.arange(0,16).view(4,4)

index=torch.LongTensor([[0,1,2,3]])

torch.gather(a, 0, index)


index=torch.LongTensor([[0],[1],[1],[2]])

a.gather(1,index)



#index_fill(dim,index,val)按照指定的维度轴dim 根据index去对应位置，将原tensor用参数val值填充，这里强调一下，index必须是1D tensor，index去指定轴上索引数据时候会广播，与上面gather的index去索引不同(一一对应查)
a=torch.arange(0,16).view(4,4)
print(a)

index=torch.LongTensor([[0],[1],[1],[2]])
index.squeeze()

a.index_fill(0,index.squeeze(),100)


import torch
a = torch.randn(4, 3)
print(a)
# tensor([[-1.7189,  0.9798, -0.0428],
#         [ 0.7184, -0.2824, -1.0289],
#         [ 1.2858,  0.8423, -1.0473],
#         [-0.0269, -0.9876, -2.3126]])

index = torch.tensor([0, 2])
b=a.index_fill(1, index, 9)#要填充1维
print(b)
# tensor([[ 9.0000,  0.9798,  9.0000],
#         [ 9.0000, -0.2824,  9.0000],
#         [ 9.0000,  0.8423,  9.0000],
#         [ 9.0000, -0.9876,  9.0000]])

c=a.index_fill(0, index, 9)#要填充0维
print(c)
# tensor([[ 9.0000,  9.0000,  9.0000],
#         [ 0.7184, -0.2824, -1.0289],
#         [ 9.0000,  9.0000,  9.0000],
#         [-0.0269, -0.9876, -2.3126]])


#torch.nonezero()的作用就是找到tensor中所有不为0的索引。（要注意返回值的size）
import torch
a=torch.randint(-1,2,(10,),dtype=torch.int)
print(a)
print(a.size())
print(torch.nonzero(a))
print(torch.nonzero(a).size())
print(torch.nonzero(a).squeeze())

import torch
a=torch.randint(-1,2,(3,4),dtype=torch.int)
print(a)
print(a.size())
print(torch.nonzero(a))
print(torch.nonzero(a).size())
print(torch.nonzero(a).squeeze())



#================================================= nn.LayerNorm =================================================

import torch
import torch.nn as nn
import numpy as np
a = np.array([[1, 20, 3, 4],
               [5, 6, 7, 8,],
               [9, 10, 11, 12]], dtype=np.double)
b = torch.from_numpy(a).type(torch.FloatTensor)
a1 = np.zeros_like(a)


#只对最后 1 个维度进行标准化
layer_norm = nn.LayerNorm(4, eps=1e-6) # 最后一个维度大小为4，因此normalized_shape是4
c = layer_norm(b)
print(c)

#怎么验证对不对呢？我们可以使用 np 对数组 a 手动计算下标准化看看：
mean_a = np.mean(a, axis=1)  # 计算最后一个维度的均值 = [7. 6.5 10.5]
var_a = np.var(a, axis=1)    # 计算最后一个维度的方差 = [57.5 1.25 1.25]
# 对最后一个维度做标准化 减均值后除以标准差
a1[0, :] = (a[0, :] - mean_a[0]) / np.sqrt(var_a[0])
a1[1, :] = (a[1, :] - mean_a[1]) / np.sqrt(var_a[1])
a1[2, :] = (a[2, :] - mean_a[2]) / np.sqrt(var_a[2])
print(a1)


#举例-对最后 D 个维度进行标准化

layer_norm = nn.LayerNorm([3, 4], eps=1e-6)
c = layer_norm(b)
print(c)


#怎么做验证呢？也让 np 在所有数据上做标准化：

mean_a = np.mean(a)  # 计算所有数据的均值，返回标量
var_a = np.var(a)    # 计算所有数据的方差，返回标量
a = (a - mean_a) / np.sqrt(var_a)  # 对整体做标准化
print(a)


# NLP Example
batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)
print(f"layer_norm(embedding) = \n{layer_norm(embedding)}")



# Image Example
N, C, H, W = 20, 5, 10, 10
input = torch.randn(N, C, H, W)
layer_norm = nn.LayerNorm([C, H, W])
print(f"layer_norm(input) = \n{layer_norm(input)}")



#================================================= rearrange=================================================

import numpy as np
import torch
from einops import rearrange, repeat

a = np.arange(2*3*4*5).reshape(2,3,4,5)
print(f"a.shape = {a.shape}") #images.shape = (2, 3, 4, 5)

print(f"rearrange(a, 'b h w c -> b h w c').shape = {rearrange(a, 'b h w c -> b h w c').shape}")  #  (2, 3, 4, 5)

#沿height维进行concat
print(f"rearrange(a, 'b h w c -> (b h) w c').shape = {rearrange(a, 'b h w c -> (b h) w c').shape}")  # (6, 4, 5)



# 沿width维进行concat
print(f"rearrange(a, 'b h w c -> h (b w) c').shape = {rearrange(a, 'b h w c -> h (b w) c').shape}")  # (3, 8, 5)




# 转换维度的次序，比如将通道维度放在height和weight前边
print(f"rearrange(a, 'b h w c -> b c h w').shape = {rearrange(a, 'b h w c -> b c h w').shape}")


#放缩宽和高，通道数
# 这里(h h1) (w w1)就相当于h与w变为原来的1/h1,1/w1倍
# split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2  ：(128, 15, 20, 3)
print(rearrange(a, 'b h (w w1) c -> (b  w1) h w c',  w1=2).shape)



#
#将单通道灰度图，按照通道层扩增
import numpy as np
from einops import rearrange, repeat, reduce

# a grayscale image (of shape height x width)
image = np.random.randn(2, 3)
# change it to RGB format by repeating in each channel：(30, 40, 3)
print(repeat(image, 'h w -> h w c', c=4).shape)




#扩增height，变为原来的2倍
print(repeat(image, 'h w -> (repeat h) w', repeat=2).shape)



#扩增weight，变为原来的3倍
print(repeat(image, 'h w -> h (repeat w)', repeat=3).shape)



#把每一个pixel扩充4倍
print(repeat(image, 'h w -> h (repeat w)', repeat=3).shape)


#把每一个pixel扩充4倍
print(repeat(image, 'h w -> (h h2) (w w2)', h2=2, w2=2).shape)



#先下采样，然后上采样
downsampled = reduce(image, '(h h2) (w w2) -> h w', 'mean', h2=2, w2=2)
print(repeat(downsampled, 'h w -> (h h2) (w w2)', h2=2, w2=2).shape)

#减少一维
import numpy as np
from einops import rearrange, reduce

x = np.random.randn(100, 32, 64)
# perform max-reduction on the first axis:(32, 64)
print(reduce(x, 't b c -> b c', 'max').shape)



# 和上面的操作一样，只不过，更易读
# same as previous, but with clearer axes meaning:(32, 64)
print(reduce(x, 'time batch channel -> batch channel', 'max').shape)



#模拟最大池化功能
x = np.random.randn(10, 20, 30, 40)
# 2d max-pooling with kernel size = 2 * 2 for image processing:(10, 20, 15, 20)
y1 = reduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h2=2, w2=2)
print(y1.shape)



#全局平均池化
print(reduce(x, 'b c h w -> b c', 'mean').shape)




a = np.arange(2*3*4).reshape(2,3,4)
print(f"a.shape = {a.shape}") #images.shape = (2, 3, 4, 5)

print(f"rearrange(a, 'h w c -> w h c').shape = {rearrange(a, 'h w c -> w h c').shape}")  #  (2, 3, 4, 5)




a = np.arange(2*3*8).reshape(2,3,8)
print(f"a.shape = {a.shape}") #images.shape = (2, 3, 4, 5)

print(f"rearrange(a, 'b n (h d) -> b h n d', h = 2).shape = {rearrange(a, 'b n (h d) -> b h n d', h = 2).shape}")  #  (2, 3, 4, 5)





































































































































































































































































































































































































































































