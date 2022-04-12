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





#========================================= mask() =============================================
import torch
import math



q = torch.Tensor([np.random.random(10),np.random.random(10),np.random.random(10), np.random.random(10), np.zeros((10,1)), np.zeros((10,1))])
k = torch.Tensor([np.random.random(10),np.random.random(10),np.random.random(10), np.random.random(10), np.zeros((10,1)), np.zeros((10,1))])
scores = torch.matmul(q, k.transpose(0,1)) / math.sqrt(10)
mask = torch.Tensor([1,1,1,1,0,0])
mask1 = mask.unsqueeze(1)
scores1 = scores.masked_fill(mask1==0, -np.inf)



mas = torch.from_numpy( np.triu(np.ones((6,6)), k=1),).byte()
scores2 = scores.masked_fill(mas==0, -np.inf)





"""
5. masked_fill_(mask, value)方法

其中mask是张量，元素是布尔值， value是要填充的值。该方法会在mask中为True的位置上填充value值。mask和value的形状要么是相同的， 要么是可以进行广播的， 否则会报错。
"""
#========================================= mask() =============================================
# https://codeantenna.com/a/SqCLQ4AQNN
import torch
a=torch.tensor([[[5,5,5,5], [6,6,6,6], [7,7,7,7]], [[1,1,1,1],[2,2,2,2],[3,3,3,3]]])
print(f"a = {a}")
"""
tensor([[[5, 5, 5, 5],
         [6, 6, 6, 6],
         [7, 7, 7, 7]],

        [[1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3]]])
"""
print(a.size())
#torch.Size([2, 3, 4])
print("#############################################3")
#############################################3
mask = torch.ByteTensor([[[1],[1],[0]],[[0],[1],[1]]])
print(f"mask.size() = {mask.size()}")
#torch.Size([2, 3, 1])
b = a.masked_fill(mask, value=torch.tensor(-1e9))
print(f"b1 = {b}")





#可以看到a和mask的shape对应分别是 2 3 4 对应 2 3 1  ，可以看到mask为中的第一个1，使得a的第一行全部被mask掉了，那么我把mask的shape改成2 3 4 ，是不是可以指定位置mask掉呢
mask1 = torch.ByteTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
b = a.masked_fill(mask1, value=torch.tensor(-1e9))


#的确可以，好的，如果shape相同，那就是对应位置被mask掉，
#那么现在，我把mask的shape改成1，3，4 a保持为 2 ，3 ，4 会不会对于a的最外层的两个维度进行一样的mask呢？

mask = torch.ByteTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]]])
b1 = a.masked_fill(mask, value=torch.tensor(-1e9))
print(f"b = {b1}")
"""
tensor([[[-1000000000, -1000000000,           5,           5],
         [-1000000000,           6,           6,           6],
         [          7,           7,           7,           7]],

        [[-1000000000, -1000000000,           1,           1],
         [-1000000000,           2,           2,           2],
         [          3,           3,           3,           3]]])
"""
print(f"a.shape() = {a.shape()}")

#的确是这样的，最外层的两个维度进行了相同的mask
#那么再改一改，mask改成1，1，4，这样是不是行a都会被相同的mask掉
mask = torch.ByteTensor([[[1,1,0,0]]])
b = a.masked_fill(mask, value=torch.tensor(-1e9))
print(f"b = {b}")
"""
tensor([[[-1000000000, -1000000000,           5,           5],
         [-1000000000, -1000000000,           6,           6],
         [-1000000000, -1000000000,           7,           7]],

        [[-1000000000, -1000000000,           1,           1],
         [-1000000000, -1000000000,           2,           2],
         [-1000000000, -1000000000,           3,           3]]])
"""




import torch

mask = torch.randint(0, 2, (3, 1)).bool()
target = torch.randn(3, 2)
print(target)
# tensor([[-0.4297,  0.6459],
#         [ 1.2334, -1.5065],
#         [ 0.1295,  0.2587]])

print(mask)
# tensor([[False],
#         [False],
#         [ True]])

# 注意mask和target是可以广播的
target.masked_fill_(mask, -100)
print(target)
# tensor([[-100.0000, -100.0000],
#         [ 1.2334, -1.5065],
#         [-1.0000, -1.0000]])
# 如果执行target.masked_fill(mask, -1)， 是非in_place操作， 那么target本身的值不会改变



import torch
import torch.nn as nn
a = torch.randint(0, 255, (2, 3, 3))
print(f"a = {a}")
mask = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).bool()
print(f"mask = {mask}")
a.masked_fill_(~mask, 0)
print(f"a = {a}")


# 当然， 你也可以自己手动mask
a = torch.tensor([[1,2,3], [2,1,0]])
model = nn.Embedding(num_embeddings = 10, embedding_dim = 6)
b = model(a)
mask = (a!=0).float().unsqueeze(-1)
result = b * mask









































































































































































































































































































































































































































































































































































































































