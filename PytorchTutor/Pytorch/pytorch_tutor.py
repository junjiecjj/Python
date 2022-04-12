# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 23:13:12 2022

@author: 陈俊杰
"""

from __future__ import print_function
import torch
import numpy as np



#=============================================================================
# 张量的创建和属性
#=============================================================================
# 直接从数据初始化¶
data = [[1, 2], [3,4]]
x_data = torch.tensor(data)
print(x_data)
# tensor([[1, 2],
#        [3, 4]])

#从Numpy的array初始化
np_array = np.array(data)
print('np_nrray:\n', np_array)
x_np = torch.from_numpy(np_array)
print(f"x_np = {x_np}\n")

#创建一个 5x3 矩阵，但是未初始化:
x = torch.empty(5, 3)

print(f"x = {x}\n")

# 使用[0,1]均匀分布随机初始化二维数组
x = torch.rand(5, 3)
print(f"x = {x}\n")

# 可以使用与numpy相同的shape属性查看
print(f"x.shape = {x.shape}\n")
# 也可以使用size()函数，返回的结果都是相同的
print(f"x.size() = {x.size()}\n")

#多维的张量：
y=torch.rand(2,3,4,5)
print(f"y.size() = {y.size()}\n")
print(f"y = {y}\n")



#创建一个随机初始化的矩阵:
x = torch.rand(5, 3)
print(x)


#从另外一个张量（Tensors）获取,注意：除非明确覆盖，否则新张量保留参数张量的属性（形状、数据类型）
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"全1的张量: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"随机张量: \n {x_rand} \n")

#shape是张量维度的元组。在下面的函数中，它决定了输出张量的维度。
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"随机张量: \n {rand_tensor} \n")
print(f"全1张量: \n {ones_tensor} \n")
print(f"零张量: \n {zeros_tensor}")



#张量（Tensors）的属性
#张量属性描述了它们的形状、数据类型和存储它们的设备（比如CPU还是GPU）。
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
#Shape of tensor: torch.Size([3, 4])
#Datatype of tensor: torch.float32
#Device tensor is stored on: cpu



if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")



# numpy标准操作：索引和切片
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)  
  
#连接张量
#您可以torch.cat用来连接沿给定维度的一系列张量。另请参阅torch.stack，另一个加入 op 的张量，与torch.cat。
t1 = torch.cat([tensor, tensor, tensor], dim=1)  # dim=0 代表行，dim=1代表列
print(f"t1: \n {t1}")


# 乘法张量
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

print(f'tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n')
print(f'tensor @ tensor.T) \n {tensor @ tensor.T}')


a = torch.arange(96).reshape(2,4,12)

b = a.view(2, -1, 2, 6)

X = a.view(2, -1, 2, 6).transpose(1,2)

C = X.transpose(-2, -1)

B = torch.matmul(X, C)

D = torch.matmul(B, X)

print(f"a = \n{a}\nb = \n{b}\nX = \n{X}\nC = \n{C}\nB = \n{B}\nD = \n{D}")


#本地化操作
#具有_后缀的操作是本地化操作。例如：x.copy_(y), x.t_(), 将改变x。
#注意：就地操作节省了一些内存，但在计算导数时可能会出现问题，因为会立即丢失历史记录。因此，不鼓励使用它们。
print(tensor, '\n')
tensor.add_(5)
print(tensor)

#使用Numpy桥接（可选看）
#CPU 和 NumPy 数组上的张量可以共享它们的底层内存位置，改变一个将改变另一个。
# 张量（Tensors）到Numpy
t = torch.zeros((3,2))
print(f't: \n {t}')
n = t.numpy()
print(f'n: \n {n}')

t.add_(5)
print(f't: \n {t}')
print(f'n: \n {n}')


#Numpy到张量（Tensors）
n = np.zeros((3,4))
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: \n {t}")
print(f"n: \n {n}")


#创建一个 0 填充的矩阵，数据类型为 long:
x = torch.zeros(5, 3, dtype=torch.long)
print(x)


#创建 tensor 并使用现有数据初始化:
x = torch.tensor([5.5, 3])
print(x)


#根据现有的张量创建张量。 这些方法将重用输入张量的属性，例如， dtype，除非设置新的值进行覆盖
x = x.new_ones(5, 3, dtype=torch.double)      # new_* 方法来创建对象
print(x)

x = torch.randn_like(x, dtype=torch.float)    # 覆盖 dtype!
print(x)  
#获取 size
print(x.size())

#加法 1:
y = torch.rand(5, 3)
print(x + y)


#加法 2
print(torch.add(x, y))

#替换
# adds x to y
y.add_(x)
print(y)


#任何 以_ 结尾的操作都会用结果替换原变量. 例如: x.copy_(y), x.t_(), 都会改变 x.
#你可以使用与 NumPy 索引方式相同的操作来进行对张量的操作
print(x[:, 1])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  #  size -1 从其他维度推断
print(x.size(), y.size(), z.size())
# torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])



#如果你有只有一个元素的张量，使用.item() 来得到 Python 数据类型的数值
x = torch.randn(1)
print(x)
print(x.item())
#tensor([-0.2368])
#-0.23680149018764496




#Torch Tensor 与 NumPy 数组共享底层内存地址，修改一个会导致另一个的变化。
#将一个 Torch Tensor 转换为 NumPy 数组
a = torch.ones(5)
print(a)
#tensor([1., 1., 1., 1., 1.])
b = a.numpy()
print(b)
#[1. 1. 1. 1. 1.]
#观察 numpy 数组的值是如何改变的。

a.add_(1)
print(a)
print(b)
#tensor([2., 2., 2., 2., 2.])
#[2. 2. 2. 2. 2.]



#NumPy Array 转化成 Torch Tensor
#使用 from_numpy 自动转化
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


#使用.to 方法 可以将 Tensor 移动到任何设备中
# is_available 函数判断是否有cuda可以使用
# ``torch.device``将张量移动到指定的设备中
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA 设备对象
    y = torch.ones_like(x, device=device)  # 直接从GPU创建张量
    x = x.to(device)                       # 或者直接使用``.to("cuda")``将张量移动到cuda中
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` 也会对变量的类型做更改
#tensor([0.7632], device='cuda:0')
#tensor([0.7632], dtype=torch.float64)


#其中要特别注意的就是标量，我们先生成一个标量：
#我们直接使用现有数字生成
scalar =torch.tensor(3.1433223)
print(f"scalar = {scalar}\n" )
#打印标量的大小
print(f"scalar.size() = {scalar.size()}\n")

#对于标量，我们可以直接使用 .item() 从中取出其对应的python对象的数值
print(f"scalar.item()  ={scalar.item()}\n")

tensor = torch.tensor([3.1433223]) 
print(f"tensor =  {tensor}\n")
print(f"tensor.size() = {tensor.size()}\n")

#特别的：如果张量中只有一个元素的tensor也可以调用tensor.item方法
print(f"tensor.item() = {tensor.item()}\n")


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
print(f"sh = {do}\n")
print(f"ch.dtype = {do.dtype}\n")


flo = tensor.float()
print(f"flo = {flo}\n")
print(f"flo.dtype = {flo.dtype}\n")
# flo = tensor([3.1433])
# flo.dtype = torch.float32

fl = torch.FloatTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
print(f"sh = {fl}\n")
print(f"ch.dtype = {fl.dtype}\n")

half=tensor.half()
print(f"half = {half}\n")
print(f"half.dtype = {half.dtype}\n")
# half = tensor([3.1426], dtype=torch.float16)
# half.dtype = torch.float16

ha = torch.HalfTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
print(f"sh = {ha}\n")
print(f"ch.dtype = {ha.dtype}\n")

long=tensor.long()
print(f"long = {long}\n")
print(f"long.dtype = {long.dtype}\n")
# long = tensor([3])
# long.dtype = torch.int64

Lo = torch.LongTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
print(f"sh = {Lo}\n")
print(f"ch.dtype = {Lo.dtype}\n")


int_t=tensor.int()
print(f"int_t = {int_t}\n")
print(f"int_t.dtype = {int_t.dtype}\n")
# int_t = tensor([3], dtype=torch.int32)
# int_t.dtype = torch.int32

In = torch.IntTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
print(f"sh = {In}\n")
print(f"ch.dtype = {In.dtype}\n")

short = tensor.short()
print(f"short = {short}\n")
print(f"short.dtype = {short.dtype}\n")
#  = tensor([3], dtype=torch.int16)
# short.dtype = torch.int16

sh = torch.ShortTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
print(f"sh = {sh}\n")
print(f"ch.dtype = {sh.dtype}\n")

ch = tensor.char()
print(f"ch = {ch}\n")
print(f"ch.dtype = {ch.dtype}\n")
#  ch = tensor([3], dtype=torch.int8)
# ch.dtype = torch.int8

ch = torch.CharTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
print(f"ch = {ch}\n")
print(f"ch.dtype = {ch.dtype}\n")

bt = tensor.byte()
print(f"bt = {bt}\n")
print(f"bt.dtype = {bt.dtype}\n")
# bt = tensor([3], dtype=torch.uint8)
# bt.dtype = torch.uint8

bt = torch.ByteTensor([[[1,1,0,0],[1,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[1,1,1,1]]])
print(f"bt = {bt}\n")
print(f"bt.dtype = {bt.dtype}\n")




x = torch.randn(3, 3)
print(x)
#tensor([[ 0.6922, -0.4824,  0.8594],
#        [ 0.4509, -0.8155, -0.0368],
#        [ 1.3533,  0.5545, -0.0509]])

# 沿着行取最大值
max_value, max_idx = torch.max(x, dim=1)
print(max_value, max_idx)
#tensor([0.8594, 0.4509, 1.3533]) tensor([2, 0, 0])

# 每行 x 求和
sum_x = torch.sum(x, dim=1)
print(sum_x)
#tensor([ 1.0692, -0.4014,  1.8568])
y=torch.randn(3, 3)
z = x + y
print(z)
#tensor([[-0.3821, -2.6932, -1.3884],
#        [ 0.7468, -0.7697, -0.0883],
#        [ 0.7688, -1.3485,  0.7517]])
#正如官方60分钟教程中所说，以_为结尾的，均会改变调用值

# add 完成后x的值改变了
x.add_(y)
print(x)
#tensor([[-0.3821, -2.6932, -1.3884],
#        [ 0.7468, -0.7697, -0.0883],
#        [ 0.7688, -1.3485,  0.7517]])


#=============================================================================
# Autograd: 自动求导机制
#=============================================================================


#----------------------------------------------- 分割线 ----------------------------------------------
import torch

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
#x->y->z->out
print(f"x = {x}\n")
print(f"y = {y}\n")
print(f"z = {z}\n")
print(f"out = {out}\n")
print(f"out.grad_fn = {out.grad_fn}\n")

"""
x = tensor([[1., 1.],
        [1., 1.]], requires_grad=True)

y = tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)

z = tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>)

out = 27.0

out.grad_fn = <MeanBackward0 object at 0x7f1366312940>



这里，我们可以将x假想成神经网络的输入，y是神经网络的隐藏层，z是神经网络的输出，最后的out是损失函数；或者说我们建立了一个简单的计算图，数据从x流向out。可以看到，只要x设置了requires_grad=True，那么计算图后续的节点用grad_fn记录了计算图中各步的传播过程。
现在，我们从out开始进行反向传播：
"""
#反向传播 因为 out是一个纯量（scalar），out.backward() 等于out.backward(torch.tensor(1))。
out.backward(torch.tensor(1))

print(f"x.grad = {x.grad}\n")

"""
x.grad = tensor([[9., 9.],
        [9., 9.]])


拓展到深度学习，从输入开始，每层都有大量参数W和b，这些参数也是Tensor结构。给Tensor设置了requires_grad=True后，PyTorch会跟踪Tensor之后的所有计算，经过.backward()后，PyTorch自动帮我们计算损失函数对于这些参数的梯度，梯度存储在了.grad属性里，PyTorch会按照梯度下降法更新参数。

在PyTorch中，.backward()方法默认只会对计算图中的叶子节点求导。在上面的例子里，x就是叶子节点，y和z都是中间变量，他们的.grad属性都是None。而且，PyTorch目前只支持浮点数的求导。

另外，PyTorch的自动求导一般只是标量对向量/矩阵求导。在深度学习中，最后的损失函数一般是一个标量值，是样本数据经过前向传播得到的损失值的和，而输入数据是一个向量或矩阵。在刚才的例子中，y是一个矩阵，.mean()对y求导，得到的是标量。
"""


#----------------------------------------------- 分割线 ----------------------------------------------

import torch


x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
#x->y->z->out
print(f"x = {x}\n")
print(f"y = {y}\n")
print(f"z = {z}\n")
print(f"out = {out}\n")
print(f"out.grad_fn = {out.grad_fn}\n")

"""
x = tensor([[1., 1.],
        [1., 1.]], requires_grad=True)

y = tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)

z = tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>)

out = 27.0

out.grad_fn = <MeanBackward0 object at 0x7fa344125850>
"""

gradients = torch.tensor([[0, 1],[2,3]], dtype=torch.float)
#反向传播 因为 out是一个纯量（scalar），out.backward() 等于out.backward(torch.tensor(1))。
z.backward(gradients)

print(f"x.grad = {x.grad}\n")
# x.grad = tensor([[ 0., 18.],
#                 [36., 54.]])

#----------------------------------------------- 分割线 ----------------------------------------------
import torch

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
#x->y->z->out
print(f"x = {x}\n")
print(f"y = {y}\n")
print(f"z = {z}\n")
print(f"out = {out}\n")
print(f"out.grad_fn = {out.grad_fn}\n")

"""
x = tensor([[1., 1.],
        [1., 1.]], requires_grad=True)

y = tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)

z = tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>)

out = 27.0

out.grad_fn = <MeanBackward0 object at 0x7fa3440aab50>
"""

gradients = torch.tensor([[0, 1],[2,3]], dtype=torch.float)
#反向传播 因为 out是一个纯量（scalar），out.backward() 等于out.backward(torch.tensor(1))。
out.backward(torch.tensor(2))

print(f"x.grad = {x.grad}\n")

#----------------------------------------------- 分割线 ----------------------------------------------
import torch
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(f"a.requires_grad = {a.requires_grad}\n")
a.requires_grad_(True)
print(f"a.requires_grad = {a.requires_grad}\n")
b = (a * a).sum()
print(f"b.grad_fn = {b.grad_fn}\n")


"""
a.requires_grad = False

a.requires_grad = True

b.grad_fn = <SumBackward0 object at 0x7f136630a070>
"""



import torch
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(f"a.requires_grad = {a.requires_grad}\n")

b = (a * a).sum()
print(f"b.grad_fn = {b.grad_fn}\n")


"""
a.requires_grad = False

b.grad_fn = None
"""

#----------------------------------------------- 分割线 ----------------------------------------------




import torch
x = torch.randn(3, requires_grad=True)

y = x * 2

#   data.norm()首先，它对张量y每个元素进行平方，然后对它们求和，最后取平方根。 这些操作计算就是所谓的L2或欧几里德范数 。
while y.data.norm() < 1000: 
    y = y * 2

print(f"t = {y}\n")


"""
如果需要计算导数，你可以在Tensor上调用.backward()。 如果Tensor是一个标量（即它包含一个元素数据）则不需要为backward()指定任何参数， 
但是如果它有更多的元素，你需要指定一个gradient 参数来匹配张量的形状。

"""


#在这个情形中，y不再是个标量。torch.autograd无法直接计算出完整的雅可比行列，但是如果我们只想要vector-Jacobian product，只需将向量作为参数传入backward：

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)

print(f"x.grad = {x.grad}\n")


#如果.requires_grad=True但是你又不希望进行autograd的计算， 那么可以将变量包裹在 with torch.no_grad()中:
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)



#----------------------------------------------- 分割线 ----------------------------------------------
import torch
from torch.autograd import Variable

x = torch.Tensor([[1.,2.,3.],[4.,5.,6.]])  #grad_fn是None
x = Variable(x, requires_grad=True)
y = x + 2
z = y*y*3
out = z.mean()
#x->y->z->out
print(f"x = {x}\n")
print(f"y = {y}\n")
print(f"z = {z}\n")
print(f"out = {out}\n")

"""
x = tensor([[1., 2., 3.],
        [4., 5., 6.]], requires_grad=True)

y = tensor([[3., 4., 5.],
        [6., 7., 8.]], grad_fn=<AddBackward0>)

z = tensor([[ 27.,  48.,  75.],
        [108., 147., 192.]], grad_fn=<MulBackward0>)

out = 99.5


这里，我们可以将x假想成神经网络的输入，y是神经网络的隐藏层，z是神经网络的输出，
最后的out是损失函数；或者说我们建立了一个简单的计算图，数据从x流向out。
可以看到，只要x设置了requires_grad=True，那么计算图后续的节点用grad_fn记录了计算图中各步的传播过程。
"""


out.backward()
print(f"x.grad = {x.grad}\n")
print(f"y.grad = {y.grad}\n")
print(f"z.grad = {z.grad}\n")
#结果:
"""
x.grad = tensor([[3., 4., 5.],
        [6., 7., 8.]])

y.grad = None

z.grad = None

拓展到深度学习，从输入开始，每层都有大量参数W和b，这些参数也是Tensor结构。给Tensor设置了requires_grad=True后，PyTorch会跟踪Tensor之后的所有计算，经过.backward()后，PyTorch自动帮我们计算损失函数对于这些参数的梯度，梯度存储在了.grad属性里，PyTorch会按照梯度下降法更新参数。

在PyTorch中，.backward()方法默认只会对计算图中的叶子节点求导。在上面的例子里，x就是叶子节点，y和z都是中间变量，他们的.grad属性都是None。而且，PyTorch目前只支持浮点数的求导。

另外，PyTorch的自动求导一般只是标量对向量/矩阵求导。在深度学习中，最后的损失函数一般是一个标量值，是样本数据经过前向传播得到的损失值的和，而输入数据是一个向量或矩阵。在刚才的例子中，y是一个矩阵，.mean()对y求导，得到的是标量。

"""

#----------------------------------------------- 分割线 ----------------------------------------------

import torch
from torch.autograd import Variable

x = torch.Tensor([[1.,2.,3.],[4.,5.,6.]])  #grad_fn是None
x = Variable(x, requires_grad=True)
y = x + 2
z = y*y*3
out = z.mean()
#x->y->z->out
print(f"x = {x}\n")
print(f"y = {y}\n")
print(f"z = {z}\n")
print(f"out = {out}\n")
#反向传播 因为 out是一个纯量（scalar），out.backward() 等于out.backward(torch.tensor(1))。
out.backward()

print(f"x.grad = {x.grad}\n")


#如果是z关于x求导就必须指定gradient参数：
import torch
from torch.autograd import Variable

x = torch.Tensor([[1.,2.,3.],[4.,5.,6.]])  #grad_fn是None
x = Variable(x, requires_grad=True)
y = x + 2
z = y*y*3
out = z.mean()
#x->y->z->out
print(f"x = {x}\n")
print(f"y = {y}\n")
print(f"z = {z}\n")
print(f"out = {out}\n")

#如果是z关于x求导就必须指定gradient参数：

gradients = torch.Tensor([[2.,1.,1.],[3.,1.,1.]])

z.backward(gradient=gradients)
#若z不是一个标量，那么就先构造一个标量的值：L = torch.sum(z*gradient)，再关于L对各个leaf Variable计算梯度
#对x关于L求梯度
print(f"x.grad = \n{x.grad}\n")

#结果:
#tensor([[36., 24., 30.],
#        [36., 42., 48.]])

#错误情况
# z.backward()
# print(x.grad) 
#报错:RuntimeError: grad can be implicitly created only for scalar outputs只能为标量创建隐式变量
  

#----------------------------------------------- 分割线 ----------------------------------------------
"""
https://lulaoshi.info/machine-learning/neural-network/pytorch-tensor-autograd
下面是一个使用PyTorch训练神经网络的例子。在这个例子中，我们随机初始化了输入x和输出y，分别作为模型的特征和要拟合的目标值。这个模型有两层，第一层是输入层，第二层为隐藏层，模型的前向传播如下所示：

H=ReLU(W[1]X)
Y=W[2]H
"""


import torch

dtype = torch.float
device = torch.device("cpu") # 使用CPU
# device = torch.device("cuda:0") # 如果使用GPU，请打开注释

# N: batch size
# D_in: 输入维度
# H: 隐藏层
# D_out: 输出维度 
N, D_in, H, D_out = 64, 1000, 100, 10

# 初始化随机数x, y
# x, y用来模拟机器学习的输入和输出
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 初始化模型的参数w1和w2
# 均设置为 requires_grad=True
# PyTorch会跟踪w1和w2上的计算，帮我们自动求导
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 前向传播过程：
    # h1 = relu(x * w1)
    # y = h1 * w2
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # 计算损失函数loss
    # loss是误差的平方和
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # 反向传播过程：
    # PyTorch会对设置了requires_grad=True的Tensor自动求导，本例中是w1和w2
    # 执行完backward()后，w1.grad 和 w2.grad 里存储着对于loss的梯度
    loss.backward()

    # 根据梯度，更新参数w1和w2
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 将 w1.grad 和 w2.grad 中的梯度设为零
        # PyTorch的backward()方法计算梯度会默认将本次计算的梯度与.grad中已有的梯度加和
        # 必须在下次反向传播前先将.grad中的梯度清零
        w1.grad.zero_()
        w2.grad.zero_()




#=======================================================================================
# pytorch nn.Embedding的用法和理解
#=======================================================================================
"""
https://www.jianshu.com/p/63e7acc5e890

torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None,
 max_norm=None,  norm_type=2.0,   scale_grad_by_freq=False, 
 sparse=False,  _weight=None)
其为一个简单的存储固定大小的词典的嵌入向量的查找表，意思就是说，给一个编号，嵌入层就能返回这个编号对应的嵌入向量，嵌入向量反映了各个编号代表的符号之间的语义关系。

输入为一个编号列表，输出为对应的符号嵌入向量列表。

参数解释
num_embeddings (python:int) – 词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0-4999）
embedding_dim (python:int) – 嵌入向量的维度，即用多少维来表示一个符号。
padding_idx (python:int, optional) – 填充id，比如，输入长度为100，但是每次的句子长度并不一样，后面就需要用统一的数字填充，而这里就是指定这个数字，这样，网络在遇到填充id时，就不会计算其与其它符号的相关性。（初始化为0）
max_norm (python:float, optional) – 最大范数，如果嵌入向量的范数超过了这个界限，就要进行再归一化。
norm_type (python:float, optional) – 指定利用什么范数计算，并用于对比max_norm，默认为2范数。
scale_grad_by_freq (boolean, optional) – 根据单词在mini-batch中出现的频率，对梯度进行放缩。默认为False.
sparse (bool, optional) – 若为True,则与权重矩阵相关的梯度转变为稀疏张量。

"""

batch = [['i', 'am', 'a', 'boy', '.'], ['i', 'am', 'very', 'luck', '.'], ['how', 'are', 'you', '?']]
#可见，每个句子的长度，即每个内层list的元素数为：5,5,4。这个长度也要记录。
lens = [5,5,4]

batch = [[3,6,5,6,7],[6,4,7,9,5],[4,5,8,7]]

#同时，每个句子结尾要加EOS，假设EOS在词典中的index是1。
batch = [[3,6,5,6,7,1],[6,4,7,9,5,1],[4,5,8,7,1]]


#那么长度要更新：
lens = [6,6,5]

#很显然，这个mini-batch中的句子长度不一致！所以为了规整的处理，对长度不足的句子，进行填充。填充PAD假设序号是2，填充之后为：
batch = [[3,6,5,6,7,1],[6,4,7,9,5,1],[4,5,8,7,1,2]]


#batch还要转成LongTensor：

batch=torch.LongTensor(batch)


#建立词向量层
embed = torch.nn.Embedding(num_embeddings=20,embedding_dim=6)


#好了，现在使用建立了的embedding直接通过batch取词向量了，如：
embed_batch = embed(batch)


print(f"embed_batch = \n{embed_batch}")






#=======================================================================================
import numpy as np
import torch
import torch.nn as nn

# Input size表示这批有2个句子，每个句子由4个单词构成
Input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
print(f"Input = \n{Input}")

# 构造一个(假装)vocab size=10，每个vocab用3-d向量表示的table
embedding = nn.Embedding(10, 3)
# 可以看做每行是一个词汇的向量表示！
print(f"embedding.weight = \n{embedding.weight}")

Out = embedding(Input)
print(f"Out = \n{Out}")

#a=embedding(input)是去embedding.weight中取对应index的词向量！
#看a的第一行，input处index=1，对应取出weight中index=1的那一行。其实就是按index取词向量！
embedding = nn.Embedding(10, 6)
Out = embedding(Input)
print(f"Out = \n{Out},\n Out.shape = {Out.shape}")

#==============================================================================
# https://zhuanlan.zhihu.com/p/272844969
embedding = nn.Embedding(5, 3)  # 定义一个具有5个单词，维度为3的查询矩阵
print(embedding.weight)  # 展示该矩阵的具体内容
test = torch.LongTensor([[0, 2, 0, 1],
                         [1, 3, 4, 4]])  # 该test矩阵用于被embed，其size为[2, 4]
# 其中的第一行为[0, 2, 0, 1]，表示获取查询矩阵中ID为0, 2, 0, 1的查询向量
# 可以在之后的test输出中与embed的输出进行比较
test = embedding(test)
print(test.size())  # 输出embed后test的size，为[2, 4, 3]，增加
# 的3，是因为查询向量的维度为3
print(test)  # 输出embed后的test的内容












































































































































































































































































































































































































































































































































































































































































































































































































































































