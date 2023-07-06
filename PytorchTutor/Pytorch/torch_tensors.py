#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:06:47 2022

@author: jack

(一)
detach(), torch.no_grad() 和 model.eval() 的区别和联系:

1. model.eval()
    eval主要是用来影响网络中的dropout层和batchnorm层的行为。在dropout层保留所有的神经网络单元，batchnorm层使用在训练阶段学习得到的mean和var值。另外eval不会影响网络参数的梯度的计算，只不过不回传更新参数而已。所以eval模式要比with torch.no_grad更费时间和显存。

    model.eval()
            使用model.eval()切换到测试模式，不会更新模型的k，b参数
            通知dropout层和batchnorm层在train和val中间进行切换
            在train模式，dropout层会按照设定的参数p设置保留激活单元的概率（保留概率=p，比如keep_prob=0.8），batchnorm层会继续计算数据的mean和var并进行更新
            在val模式下，dropout层会让所有的激活单元都通过，而batchnorm层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值
            model.eval()不会影响各层的gradient计算行为，即gradient计算和存储与training模式一样，只是不进行反向传播(backprobagation)

2. model.train()
     这个就是训练模式，是网络的默认模式。在这个模式下，dropout层会按照设置好的失活概率进行失活，batchnorm会继续计算数据的均值和方差等参数并在每个batch size之间不断更新。

3. with troch.no_grad()
     with torch.no_grad会影响网络的自动求导机制，也就是网络前向传播后不会进行求导和进行反向传播。另外他不会影响dropout层和batchnorm层。

4. torch.set_grad_enabled（mode）
    在使用的时候是设置一个上下文环境，也就是说只要设置了torch.set_grad_enabled(False)那么接下来所有的tensor运算产生的新的节点都是不可求导的，这个相当于一个全局的环境，即使是多个循环或者是在函数内设置的调用，只要torch.set_grad_enabled(False)出现，
    则不管是在下一个循环里还是在主函数中，都不再求导，除非单独设置一个孤立节点，并把他的requires_grad设置成true。
    与with troch.no_grad 相似，会将在这个with包裹下的所有的计算出的 新的变量 的required_grad 置为false。但原有的变量required_grad 不会改变。这实际上也就是影响了网络的自动求导机制。与with torch.no_grad() 相似，不过接受一个bool类型的值。

    detach() 和 torch.no_grad() 都可以实现相同的效果，只是前者会麻烦一点，对每一个变量都要加上，而后者就不用管了:
        - detach() 会返回一个新的Tensor对象，不会在反向传播中出现，是相当于复制了一个变量，将它原本requires_grad=True变为了requires_grad=False
        - torch.no_grad() 通常是在推断(inference)的时候，用来禁止梯度计算，仅进行前向传播。在训练过程中，就像画了个圈，来，在我这个圈里面跑一下，都不需要计算梯度，就正向传播一下。



model.eval()与with torch.no_grad()
    共同点：
        在PyTorch中进行validation时，使用这两者均可切换到测试模式。
        如用于通知dropout层和batchnorm层在train和val模式间切换。
        在train模式下，dropout网络层会按照设定的参数p设置保留激活单元的概率（保留概率=p); batchnorm层会继续计算数据的mean和var等参数并更新。
        在val模式下，dropout层会让所有的激活单元都通过，而batchnorm层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值。

    不同点：
        model.eval()会影响各层的gradient计算行为，即gradient计算和存储与training模式一样，只是不进行反传。
        with torch.zero_grad()则停止autograd模块的工作，也就是停止gradient计算，以起到加速和节省显存的作用，从而节省了GPU算力和显存，但是并不会影响dropout和batchnorm层的行为。
        也就是说，如果不在意显存大小和计算时间的话，仅使用model.eval()已足够得到正确的validation的结果；而with torch.zero_grad()则是更进一步加速和节省gpu空间（因为不用计算和存储gradient），从而可以更快计算，也可以跑更大的batch来测试。

model.eval()与torch.no_grad()可以同时用，更加节省cpu的算力.


(二)
pytorch提供了clone、detach、copy_和new_tensor等多种张量的复制操作，尤其前两者在深度学习的网络架构中经常被使用，本文旨在对比这些操作的差别。

1. clone
    返回一个和源张量同shape、dtype和device的张量，与源张量不共享数据内存，但提供梯度的回溯。
    clone后的返回值是个中间variable，因此支持梯度的回溯。因此，clone操作在一定程度上可以视为是一个identity-mapping函数。
    clone作为一个中间variable，会将梯度传给源张量进行叠加。
    但若源张量的require_grad=False，而clone后的张量require_grad=True，显然此时不存在张量回溯现象，clone后的张量可以求导。
    综上论述，clone操作在不共享数据内存的同时支持梯度回溯，所以常用在神经网络中某个单元需要重复使用的场景下。

2. detach
    detach的机制则与clone完全不同，即返回一个和源张量同shape、dtype和device的张量，与源张量共享数据内存，但不提供梯度计算，即requires_grad=False，因此脱离计算图。
    detach后的张量，即使重新定义requires_grad=True，也与源张量的梯度没有关系。
    综上论述，detach操作在共享数据内存的脱离计算图，所以常用在神经网络中仅要利用张量数值，而不需要追踪导数的场景下。

3. clone和detach联合使用
    clone提供了非数据共享的梯度追溯功能，而detach又“舍弃”了梯度功能，因此clone和detach联合使用意味着着只做简单的数据复制，既不数据共享，也不对梯度共享，从此两个张量无关联。
    置于是先clone还是先detach，其返回值一样，一般采用tensor.clone().detach()。

4. new_tensor
    new_tensor可以将源张量中的数据复制到目标张量(数据不共享)，同时提供了更细致的device、dtype和requires_grad属性控制：
    其默认参数下的操作等同于.clone().detach()，而requires_grad=True时的效果相当于.clone().detach()requires_grad_(True)。上面两种情况都推荐使用后者。

5. copy_
    copy_同样将源张量中的数据复制到目标张量(数据不共享)，其device、dtype和requires_grad一般都保留目标张量的设定，仅仅进行数据复制，同时其支持broadcast操作。

    .clone()是深拷贝，开辟新的存储地址而不是引用来保存旧的tensor，在梯度会传的时候clone()充当中间变量，会将梯度传给源张量进行叠加，但是本身不保存其grad，值为None。
    .detach是浅拷贝，新的tensor会脱离计算图，不会牵扯梯度计算。

.Tensor和.tensor是深拷贝，在内存中创建一个额外的数据副本，不共享内存，所以不受数组改变的影响。.from_numpy和as_tensor是浅拷贝，在内存中共享数据。


"""

import torch
import numpy as np

import torch
import numpy as np


#=======================================================================================================================================================
#                                                                           张量的创建和属性
#=======================================================================================================================================================
# 直接从数据初始化
data = np.arange(6).reshape(2, 3)
tensor = torch.tensor(data)
print(f"data = \n{data}")
print(f"tensor = \n{tensor}")

Tensor = torch.Tensor(data)
print(f"Tensor = \n{Tensor}")

as_tensor = torch.as_tensor(data)
print(f"as_tensor = \n{as_tensor}")

from_numpy = torch.from_numpy(data)
print(f"from_numpy = {from_numpy}\n")


data[0, 0] = 34
print(f"data = \n{data}")
print(f"tensor = \n{tensor}")
print(f"Tensor = \n{Tensor}")
print(f"as_tensor = \n{as_tensor}")
print(f"from_numpy = {from_numpy}\n")

tensor[0, 1] = 98
print(f"data = \n{data}")
print(f"tensor = \n{tensor}")
print(f"Tensor = \n{Tensor}")
print(f"as_tensor = \n{as_tensor}")
print(f"from_numpy = {from_numpy}\n")

Tensor[0, 2] = 45
print(f"data = \n{data}")
print(f"tensor = \n{tensor}")
print(f"Tensor = \n{Tensor}")
print(f"as_tensor = \n{as_tensor}")
print(f"from_numpy = {from_numpy}\n")

as_tensor[1, 0] = 467
print(f"data = \n{data}")
print(f"tensor = \n{tensor}")
print(f"Tensor = \n{Tensor}")
print(f"as_tensor = \n{as_tensor}")
print(f"from_numpy = {from_numpy}\n")

from_numpy[1, 1] = 565
print(f"data = \n{data}")
print(f"tensor = \n{tensor}")
print(f"Tensor = \n{Tensor}")
print(f"as_tensor = \n{as_tensor}")
print(f"from_numpy = {from_numpy}\n")


# 从以上可以看出, tensor 和 Tensor 是深拷贝, 而 from_numpy 和 as_tensor 是浅拷贝, data, from_numpy 和 as_tensor同步变化.
# .Tensor和.tensor是深拷贝，在内存中创建一个额外的数据副本，不共享内存，所以不受数组改变的影响。.from_numpy和as_tensor是浅拷贝，在内存中共享数据。


#创建一个 5x3 矩阵，但是未初始化:
x = torch.empty(5, 3)

print(f"x = {x}\n")

##=================================================================
# 使用[0,1]均匀分布随机初始化二维数组
##=================================================================

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

a = torch.arange(96).reshape(2,4,12)
print(f"a = \n{a}\na[:,:2] = \n{a[:,:2]}\n")
print(f"a = \n{a}\na[:,:2,:] = \n{a[:,:2,:]}\n")

a = torch.arange(24).reshape(2,3,4)
b = torch.arange(12).reshape(1,3,4)
c = a+b
print(f"a = \n{a}")
print(f"b = \n{b}")
print(f"c = \n{c}")


##=================================================================
#   标准正态分布(均值为0，方差为 1，即高斯白噪声)
##=================================================================

x = torch.randn(5, 3)
print(f"x = {x}\n")

# 可以使用与numpy相同的shape属性查看
print(f"x.shape = {x.shape}\n")
# 也可以使用size()函数，返回的结果都是相同的
print(f"x.size() = {x.size()}\n")

##=================================================================
#   给定参数n，返回一个从[0, n -1) 的随机整数排列。
##=================================================================

# 给定参数n，返回一个从[0, n -1) 的随机整数排列。
x = torch.randperm(10)
print(f"x = {x}\n")

# 可以使用与numpy相同的shape属性查看
print(f"x.shape = {x.shape}\n")
# 也可以使用size()函数，返回的结果都是相同的
print(f"x.size() = {x.size()}\n")



##=================================================================
#    线性间距向量
##=================================================================

x = torch.linspace(2, 9, steps=5)
print(f"x = {x}\n")

# 可以使用与numpy相同的shape属性查看
print(f"x.shape = {x.shape}\n")
# 也可以使用size()函数，返回的结果都是相同的
print(f"x.size() = {x.size()}\n")

##=================================================================
#     离散正态分布
##=================================================================

x = torch.normal(mean=0.5, std=torch.arange(1., 6.))
print(f"x = {x}\n")

# 可以使用与numpy相同的shape属性查看
print(f"x.shape = {x.shape}\n")
# 也可以使用size()函数，返回的结果都是相同的
print(f"x.size() = {x.size()}\n")


x = torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
print(f"x = {x}\n")

# 可以使用与numpy相同的shape属性查看
print(f"x.shape = {x.shape}\n")
# 也可以使用size()函数，返回的结果都是相同的
print(f"x.size() = {x.size()}\n")


x = torch.normal(mean = 2, std = 3, size=(1, 4))
print(f"x = {x}\n")

# 可以使用与numpy相同的shape属性查看
print(f"x.shape = {x.shape}\n")
# 也可以使用size()函数，返回的结果都是相同的
print(f"x.size() = {x.size()}\n")


##=================================================================
#     torch.Tensor.uniform_()
##=================================================================

a = torch.Tensor(2,3).uniform_(5,6)
b = torch.zeros(2,3).uniform_(5,6)
x = torch.ones(2,3).uniform_(5,6)

print(f"x = {x}\n")

# 可以使用与numpy相同的shape属性查看
print(f"x.shape = {x.shape}\n")
# 也可以使用size()函数，返回的结果都是相同的
print(f"x.size() = {x.size()}\n")


# torch.rand和torch.Tensor.uniform_ : 两个都能取0-1之间的均匀分布，但是问题在于rand取不到1，uniform_可以取到1。





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

#======================================================================================
#   取整/取余 运算
#======================================================================================

x = torch.randint(low = 0, high = 20, size = (2,3))*2.313132

# 向下取整
x.floor()

# 向上取整
x.ceil()

# 四舍五入
x.round()

# 剪裁, 只取整数部分
x.trunc()

# 只取小数部分
x.frac()





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

a = torch.arange(8).reshape(2,4)
b = torch.arange(8).reshape(2,4) + 1

print(f"a = \n{a},\nb = \n{b}")
print(f"a.mul(b) \n {a.mul(b)} \n")

print(f"a*b \n {a*b} \n")



print(f'tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n')
print(f'tensor @ tensor.T) \n {tensor @ tensor.T}')

a = torch.arange(8).reshape(2,4)
b = torch.arange(8).reshape(2,4) + 1

print(f"a = \n{a},\nb = \n{b}")
print(f"a.matmul(b) \n {a.matmul(b.T)} \n")

print(f"a@b \n {a@b.T} \n")

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





arr1 = np.array([1,2,3], dtype=np.float32)
arr2 = np.array([4,5,6])
print(arr1.dtype)
print("nunpy中array的默认数据类型为：", arr2.dtype)

tensor = torch.tensor(arr2)
Tensor = torch.Tensor(arr2)
as_tensor = torch.as_tensor(arr2)
from_numpy = torch.from_numpy(arr2)

print(tensor.dtype, "|",Tensor.dtype, "|",as_tensor.dtype, "|",from_numpy.dtype)
arr2[0] = 10
print(tensor, Tensor, as_tensor, from_numpy)
'''
结果为：
float32
numpy中array的默认数据类型为： int64
torch.int64 | torch.float32 | torch.int64 | torch.int64
tensor([4, 5, 6]) tensor([4., 5., 6.]) tensor([10,  5,  6]) tensor([10,  5,  6])

上述中可以看出来：

numpy中array默认的数据格式是int64类型，而torch中tensor默认的数据格式是float32类型。
as_tensor和from_numpy是浅拷贝，而tensor和Tensor则是属于深拷贝，浅拷贝是直接共享内存内存空间的，这样效率更高，而深拷贝是直接创建一个新的副本。

'''


# numpy-->tensor
n=np.ones(5)
t=torch.tensor(n)
np.add(n,1,out=n)
print(t)
print(n)


n=np.ones(5)
t=torch.Tensor(n)
np.add(n,1,out=n)
print(t)
print(n)


n=np.ones(5)
t=torch.as_tensor(n)
np.add(n,1,out=n)
print(t)
print(n)


n=np.ones(5)
t=torch.from_numpy(n)
np.add(n,1,out=n)
print(t)
print(n)



# 二、torch中的tensor转化为numpy数组：
import torch
import numpy as np
a = torch.ones(5)
b = a.numpy()
b[0] = 2

print(a)
print(b)
'''
tensor([2., 1., 1., 1., 1.])
[2. 1. 1. 1. 1.]
'''
# 从上述的结果看来这个numpy() 方法将tensor转numpy的array也是内存共享的。


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
print(x.mean(axis=0, keepdim=True).shape)
#
print("shape of x.mean(axis=0,keepdim=False):")         #[3, 4]
print(x.mean(axis=0, keepdim=False).shape)
#
print("shape of x.mean(axis=1,keepdim=True):")          #[2, 1, 4]
print(x.mean(axis=1, keepdim=True).shape)
#
print("shape of x.mean(axis=1,keepdim=False):")         #[2, 4]
print(x.mean(axis=1, keepdim=False).shape)
#
print("shape of x.mean(axis=2,keepdim=True):")          #[2, 3, 1]
print(x.mean(axis=2, keepdim=True).shape)
#
print("shape of x.mean(axis=2,keepdim=False):")         #[2, 3]
print(x.mean(axis=2, keepdim=False).shape)


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

half = tensor.half()
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





































































































































































































































































































































































































































































