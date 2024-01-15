# -*- coding: utf-8 -*-


"""
detach(), torch.no_grad() 和 model.eval() 的区别和联系:


detach() 和 torch.no_grad() 都可以实现相同的效果，只是前者会麻烦一点，对每一个变量都要加上，而后者就不用管了:
        - detach() 会返回一个新的Tensor对象，不会在反向传播中出现，是相当于复制了一个变量，将它原本requires_grad=True变为了requires_grad=False
        - torch.no_grad() 通常是在推断(inference)的时候，用来禁止梯度计算，仅进行前向传播。在训练过程中，就像画了个圈，来，在我这个圈里面跑一下，都不需要计算梯度，就正向传播一下。


而model.eval()和torch.no_grad()两者都用，因为两者有着不同的作用：
        - torch.no_grad()：在autograd中禁用梯度跟踪
        - model.eval()：更改所调用模块的forward()行为。例如，它禁用dropout，并使用测试时bn norm

model.eval()
        使用model.eval()切换到测试模式，不会更新模型的k，b参数
        通知dropout层和batchnorm层在train和val中间进行切换
        在train模式，dropout层会按照设定的参数p设置保留激活单元的概率（保留概率=p，比如keep_prob=0.8），batchnorm层会继续计算数据的mean和var并进行更新
        在val模式下，dropout层会让所有的激活单元都通过，而batchnorm层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值
        model.eval()不会影响各层的gradient计算行为，即gradient计算和存储与training模式一样，只是不进行反向传播(backprobagation)


model.eval()与torch.no_grad()可以同时用，更加节省cpu的算力

"""

from __future__ import print_function
import torch
import numpy as np



#=============================================================================
# Autograd: 自动求导机制
#=============================================================================

# requires_grad: 如果需要为张量计算梯度，则为True，否则为False。我们使用pytorch创建tensor时，可以指定requires_grad为True（默认为False），

# grad_fn： grad_fn用来记录变量是怎么来的，方便计算梯度，y = x*3,grad_fn记录了y由x计算的过程。

# grad：当执行完了backward()之后，通过x.grad查看x的梯度值。

# 像x这种直接创建的称为叶子节点，叶子节点对应的grad_fn是None。

# 如果一个张量地requires_grad=True,那么在调用backward()方法时反向传播计算梯度,我们
#     会为这个张量计算梯度,但是计算完梯度之后这个梯度并不一定会一直保存在属性grad中.只有对于
#     requires_grad=True的叶子张量,我们才会将梯度一直保存在该叶子张量的grad属性中,对于非叶子节点,
#     即中间节点的张量,我们在计算完梯度之后为了更高效地利用内存,我们会将梯度grad的内存释放掉.)

#----------------------------------------------- requires_grad ----------------------------------------------
# 测试一些什么都不做，查看计算的梯度
import torch

x = torch.tensor([1.0, 2.0])
y1 = x ** 2
y2 = y1 * 2
y3 = y1 + y2

print(y1, y1.requires_grad)
print(y2, y2.requires_grad)
print(y3, y3.requires_grad)

# 为什么backward里面需要加一个torch.ones(y3.shape)？
# 这是另外一个需要讨论的问题了可以在留言区一起讨论
y3.backward(torch.ones(y3.shape))  # y1.backward() y2.backward()
print(x.grad)

# tensor([1., 4.]) False
# tensor([2., 8.]) False
# tensor([ 3., 12.]) False

# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn


#----------------------------------------------- 分割线 ----------------------------------------------
# 测试一些什么都不做，查看计算的梯度
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y1 = x ** 2
y2 = y1 * 2
y3 = y1 + y2

print(y1, y1.requires_grad)
print(y2, y2.requires_grad)
print(y3, y3.requires_grad)

y3.backward(torch.ones(y3.shape))  # y1.backward() y2.backward()
print(x.grad)

# tensor([1., 4.], grad_fn=<PowBackward0>) True
# tensor([2., 8.], grad_fn=<MulBackward0>) True
# tensor([ 3., 12.], grad_fn=<AddBackward0>) True
# tensor([ 6., 12.])

"""
1. 当grad_fn设置为Fasle或者默认时：计算梯度会出现如下错误
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
因为并没有追踪到任何计算历史，所以就不存在梯度的计算了

2. 因此在最开始定义x张量的时候，就应当设置好是否计算追踪历史计算记录
"""


#----------------------------------------------- detach(): ----------------------------------------------
# 设置好requires_grad的值为True
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y1 = x ** 2
y2 = y1.detach() * 2     # 注意这里在计算y2的时候对y1进行了detach()
y3 = y1 + y2

print(y1, y1.requires_grad)
print(y2, y2.requires_grad)
print(y3, y3.requires_grad)


y3.backward(torch.ones(y3.shape))  # y1.backward() y2.backward()
print(x.grad)

# tensor([1., 4.], grad_fn=<PowBackward0>) True
# tensor([2., 8.]) False
# tensor([ 3., 12.], grad_fn=<AddBackward0>) True
# tensor([2., 4.])


#----------------------------------------------- detach(): ----------------------------------------------
# 当使用detach()分离tensor但是没有更改这个tensor时，并不会影响backward():

import torch

a = torch.tensor([1, 2, 3.], requires_grad=True)
print(a.grad)
out = a.sigmoid()
print(out)

#添加detach(),c的requires_grad为False
c = out.detach()
print(c)
print(out.grad)
print(c.grad)

#这时候没有对c进行更改，所以并不会影响backward()
out.sum().backward()
print(a.grad)

'''返回：
None
tensor([0.7311, 0.8808, 0.9526], grad_fn=<SigmoidBackward>)
tensor([0.7311, 0.8808, 0.9526])
tensor([0.1966, 0.1050, 0.0452])
'''


#----------------------------------------------- detach(): ----------------------------------------------
# 当使用detach()分离tensor，然后用这个分离出来的tensor去求导数，会影响backward()，会出现错误
import torch

a = torch.tensor([1, 2, 3.], requires_grad=True)
print(a.grad)
out = a.sigmoid()
print(out)

#添加detach(),c的requires_grad为False
c = out.detach()
print(c)

#使用新生成的Variable进行反向传播
c.sum().backward()
print(a.grad)

'''返回：
None
tensor([0.7311, 0.8808, 0.9526], grad_fn=<SigmoidBackward>)
tensor([0.7311, 0.8808, 0.9526])
Traceback (most recent call last):
  File "test.py", line 13, in <module>
    c.sum().backward()
  File "/anaconda3/envs/deeplearning/lib/python3.6/site-packages/torch/tensor.py", line 102, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
  File "/anaconda3/envs/deeplearning/lib/python3.6/site-packages/torch/autograd/__init__.py", line 90, in backward
    allow_unreachable=True)  # allow_unreachable flag
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
'''

#----------------------------------------------- detach(): ----------------------------------------------
# 当使用detach()分离tensor并且更改这个tensor时，即使再对原来的out求导数，会影响backward()，会出现错误


import torch

a = torch.tensor([1, 2, 3.], requires_grad=True)
print(a.grad)
out = a.sigmoid()
print(out)

#添加detach(),c的requires_grad为False
c = out.detach()
print(c)
c.zero_() #使用in place函数对其进行修改

#会发现c的修改同时会影响out的值
print(c)
print(out)

#这时候对c进行更改，所以会影响backward()，这时候就不能进行backward()，会报错
out.sum().backward()
print(a.grad)

'''返回：
None
tensor([0.7311, 0.8808, 0.9526], grad_fn=<SigmoidBackward>)
tensor([0.7311, 0.8808, 0.9526])
tensor([0., 0., 0.])
tensor([0., 0., 0.], grad_fn=<SigmoidBackward>)
Traceback (most recent call last):
  File "test.py", line 16, in <module>
    out.sum().backward()
  File "/anaconda3/envs/deeplearning/lib/python3.6/site-packages/torch/tensor.py", line 102, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
  File "/anaconda3/envs/deeplearning/lib/python3.6/site-packages/torch/autograd/__init__.py", line 90, in backward
    allow_unreachable=True)  # allow_unreachable flag
RuntimeError: one of the variables needed for gradient computation has been modified
by an inplace operation
'''


"""
对比一下使用detach()前后的梯度值tensor([ 6., 12.])和tensor([2., 4.])
（1）tensor([ 6., 12.])

y3 = y2 + y1,根据 y2 = y1*2, 而y1 = x ** 2
所以y3 = 3x**2,    y3对xi的偏导则为6xi
针对x = [1, 2]
所以，对应的梯度（偏导）则为：[6, 12]

（2）tensor([ 2., 4.])

y3 = y2 + y1,因为y2是根据y1.detach()得到的；
根据定义，所以计算梯度的时候不考虑y2,但是实际计算y3的值还是按原公式
因此计算梯度时。y3 = y1 + (y2不考虑)，所以y3 = x ** 2
y3对xi的偏导则为2xi
针对x = [1, 2]
所以，对应的梯度（偏导）则为：[2, 4]

当我们在计算到某一步时，不需要在记录某一个张量的时，就可以使用detach()将其从追踪记录当中分离出来，这样一来该张量对应计算产生的梯度就不会被考虑了。


"""

#----------------------------------------------- with torch.no_grad() ----------------------------------------------
# 设置好requires_grad的值为True
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y1 = x ** 2

with torch.no_grad():  # 这里使用了no_grad()包裹不需要被追踪的计算过程
    y2 = y1 * 2

y3 = y1 + y2

print(y1, y1.requires_grad)
print(y2, y2.requires_grad)
print(y3, y3.requires_grad)

y3.backward(torch.ones(y3.shape))  # y1.backward() y2.backward()
print(x.grad)

"""
可想而知，实际上torch.no_grad()功能和detach()方法作用是一致的。
有差区别？
detach()是考虑将单个张量从追踪记录当中脱离出来；
而torch.no_grad()是一个warper，可以将多个计算步骤的张量计算脱离出去，本质上没啥区别。
"""

"""
requires_grad：在最开始创建Tensor时候可以设置的属性，用于表明是否追踪当前Tensor的计算操作。后面也可以通过requires_grad_（）方法设置该参数，但是只有叶子节点才可以设置该参数。
detach()方法：则是用于将某一个Tensor从计算图中分离出来。返回的是一个内存共享的Tensor，一变都变。
torch.no_grad()：对所有包裹的计算操作进行分离。
但是torch.no_grad()将会使用更少的内存，因为从包裹的开始，就表明不需要计算梯度了，因此就不需要保存中间结果。3
.data则是以前Pytorch中Variable的一个属性，返回的是一个共享内存的Tensor，一变都变，只是现在很少使用了。
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
x = torch.randn(10, 5, requires_grad = True)
y = torch.randn(10, 5, requires_grad = True)
z = torch.randn(10, 5, requires_grad = True)
print(f"x.requires_grad = {x.requires_grad}")
print(f"x.grad_fn = {x.grad_fn}\n")
with torch.no_grad():
    w = x + y + z
    out = w.sum()
    #out.backward()
    print(f"x.requires_grad = {x.requires_grad}")
    print(f"x.grad = {x.grad}")
    print(f"x.grad_fn = {x.grad_fn}\n")

    print(f"w.requires_grad = {w.requires_grad}")
    print(f"w.grad = {w.grad}")
    print(f"w.grad_fn = {w.grad_fn}\n")

    print(f"out.requires_grad = {out.requires_grad}")
    print(f"out.grad = {out.grad}")
    print(f"out.grad_fn = {out.grad_fn}\n")

print(f"x.requires_grad = {x.requires_grad}")
print(f"w.requires_grad = {w.requires_grad}")

#----------------------------------------------- 分割线 ----------------------------------------------
x = torch.randn(10, 5, requires_grad = True)
y = torch.randn(10, 5, requires_grad = True)
z = torch.randn(10, 5, requires_grad = True)
print(f"x.requires_grad = {x.requires_grad}")
print(f"x.grad_fn = {x.grad_fn}\n")

w = x + y + z
out = w.sum()
#out.backward()
print(f"x.requires_grad = {x.requires_grad}")
print(f"x.grad = {x.grad}")
print(f"x.grad_fn = {x.grad_fn}\n")

print(f"w.requires_grad = {w.requires_grad}")
print(f"w.grad = {w.grad}")
print(f"w.grad_fn = {w.grad_fn}\n")

print(f"out.requires_grad = {out.requires_grad}")
print(f"out.grad = {out.grad}")
print(f"out.grad_fn = {out.grad_fn}\n")

print(f"x.requires_grad = {x.requires_grad}")
print(f"w.requires_grad = {w.requires_grad}")


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

torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None,  norm_type=2.0,   scale_grad_by_freq=False, sparse=False, _weight=None)
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
print(f"batch.shape = {batch.shape}")
#batch.shape = torch.Size([3, 6])

#建立词向量层
embed = torch.nn.Embedding(num_embeddings=20,embedding_dim=8)


#好了，现在使用建立了的embedding直接通过batch取词向量了，如：
embed_batch = embed(batch)
print(f"embed_batch.shape = {embed_batch.shape}")
#embed_batch.shape = torch.Size([3, 6, 8])
print(f"embed_batch = \n{embed_batch}")






#=======================================================================================
import numpy as np
import torch
import torch.nn as nn
# 2D
# Input size表示这批有2个句子，每个句子由4个单词构成
Input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
print(f"Input = \n{Input}")

# 构造一个(假装)vocab size=10，每个vocab用3-d向量表示的table
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)
# 可以看做每行是一个词汇的向量表示！
print(f"embedding.weight = \n{embedding.weight}")

Out = embedding(Input)
print(f"Out = \n{Out}")

#a=embedding(input)是去embedding.weight中取对应index的词向量！
#看a的第一行，input处index=1，对应取出weight中index=1的那一行。其实就是按index取词向量！
embedding = nn.Embedding(num_embeddings=10, embedding_dim=6)
Out = embedding(Input)
print(f"Out = \n{Out},\n Out.shape = {Out.shape}")


embedding = nn.Embedding(num_embeddings=10, embedding_dim=12)
Out = embedding(Input)
print(f"Out = \n{Out},\n Out.shape = {Out.shape}")


embedding = nn.Embedding(num_embeddings=5, embedding_dim=6)   # num_embeddings必须大于input的最大元素值
Out = embedding(Input)
print(f"Out = \n{Out},\n Out.shape = {Out.shape}")
# IndexError: index out of range in self


# 3D
Input = torch.LongTensor([[[1,2,4,5],[4,3,2,9],[10,12,18,11]],[[1,2,4,5],[4,3,2,9],[22,21,18,20]]])
print(f"Input = \n{Input}")
print(f" Input.shape = {Input.shape}")
embedding = nn.Embedding(num_embeddings=30, embedding_dim=6)   # num_embeddings必须大于input的最大元素值
Out = embedding(Input)
print(f"Out = \n{Out},\n Out.shape = {Out.shape}")


# 1D
Input = torch.LongTensor([1,2,3,4])
print(f"Input = \n{Input}")
print(f" Input.shape = {Input.shape}")
embedding = nn.Embedding(num_embeddings=30, embedding_dim=6)   # num_embeddings必须大于input的最大元素值
Out = embedding(Input)
print(f"Out = \n{Out},\n Out.shape = {Out.shape}")

# 2D
Input = torch.LongTensor([1,2,3,4]).view(-1,1)
print(f"Input = \n{Input}")
print(f" Input.shape = {Input.shape}")
embedding = nn.Embedding(num_embeddings=30, embedding_dim=6)   # num_embeddings必须大于input的最大元素值
Out = embedding(Input)
print(f"Out = \n{Out},\n Out.shape = {Out.shape}")



Input = torch.LongTensor([0.1,2.3,0.5,5.6]).view(-1,1)   # input元素必须为整数，否则Embedding会将其转化为整数
print(f"Input = \n{Input}")
print(f" Input.shape = {Input.shape}")
embedding = nn.Embedding(num_embeddings=30, embedding_dim=6)   # num_embeddings必须大于input的最大元素值
Out = embedding(Input)
print(f"Out = \n{Out},\n Out.shape = {Out.shape}")
"""
Input =
tensor([[0],
        [2],
        [0],
        [5]])
 Input.shape = torch.Size([4, 1])
Out =
tensor([[[ 1.9019, -0.8279, -0.4124,  0.4178, -0.0228,  0.5323]],

        [[-0.0520,  0.6144, -1.8603, -0.5358,  1.9237,  2.0032]],

        [[ 1.9019, -0.8279, -0.4124,  0.4178, -0.0228,  0.5323]],

        [[ 0.9151, -0.9837,  0.3220,  0.8575, -0.3587,  0.1849]]],
       grad_fn=<EmbeddingBackward0>),
 Out.shape = torch.Size([4, 1, 6])
 """

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





#==================================Python中::（双冒号）的用法============================================
print(f"list(range(10)[::2]) = {list(list(range(10)[::2]))}")

print(f"range(100)[5:18:2] = {list(range(100)[5:18:2])}")

s = range(20)

print(f"s[::3] = {list(s[::3])}")


print(f"s[2::3] = {list(s[2::3])}")


print(f"s[:10:3] = {list(s[:10:3])}")


print(f"'123123123'[::3] = {'123123123'[::3]}")


# a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍。所以你看到一个倒序的东东。

print(f"s[::-1] = {list(s[::-1])}")






#======================================== attention =============================================
a = torch.arange(96).reshape(2,4,12)

b = a.view(2, -1, 2, 6)

X = a.view(2, -1, 2, 6).transpose(1,2)

C = X.transpose(-2, -1)

B = torch.matmul(X, C)

D = torch.matmul(B, X)

print(f"a = a.shape = {a.shape} \n{a}\nb = b.shape = {b.shape} \n{b} \nX = X.shape = {X.shape} \n{X}\n\
C = C.shape = {C.shape}\n {C}\n B = B.shape = {B.shape} \n{B}\n D = D.shape = {D.shape} \n{D}\n")

"""
X =
tensor([[[[ 0,  1,  2,  3,  4,  5],
          [12, 13, 14, 15, 16, 17],
          [24, 25, 26, 27, 28, 29],
          [36, 37, 38, 39, 40, 41]],

         [[ 6,  7,  8,  9, 10, 11],
          [18, 19, 20, 21, 22, 23],
          [30, 31, 32, 33, 34, 35],
          [42, 43, 44, 45, 46, 47]]],


        [[[48, 49, 50, 51, 52, 53],
          [60, 61, 62, 63, 64, 65],
          [72, 73, 74, 75, 76, 77],
          [84, 85, 86, 87, 88, 89]],

         [[54, 55, 56, 57, 58, 59],
          [66, 67, 68, 69, 70, 71],
          [78, 79, 80, 81, 82, 83],
          [90, 91, 92, 93, 94, 95]]]])
C =
tensor([[[[ 0, 12, 24, 36],
          [ 1, 13, 25, 37],
          [ 2, 14, 26, 38],
          [ 3, 15, 27, 39],
          [ 4, 16, 28, 40],
          [ 5, 17, 29, 41]],

         [[ 6, 18, 30, 42],
          [ 7, 19, 31, 43],
          [ 8, 20, 32, 44],
          [ 9, 21, 33, 45],
          [10, 22, 34, 46],
          [11, 23, 35, 47]]],


        [[[48, 60, 72, 84],
          [49, 61, 73, 85],
          [50, 62, 74, 86],
          [51, 63, 75, 87],
          [52, 64, 76, 88],
          [53, 65, 77, 89]],

         [[54, 66, 78, 90],
          [55, 67, 79, 91],
          [56, 68, 80, 92],
          [57, 69, 81, 93],
          [58, 70, 82, 94],
          [59, 71, 83, 95]]]])
B =
tensor([[[[   55,   235,   415,   595],
          [  235,  1279,  2323,  3367],
          [  415,  2323,  4231,  6139],
          [  595,  3367,  6139,  8911]],

         [[  451,  1063,  1675,  2287],
          [ 1063,  2539,  4015,  5491],
          [ 1675,  4015,  6355,  8695],
          [ 2287,  5491,  8695, 11899]]],


        [[[15319, 18955, 22591, 26227],
          [18955, 23455, 27955, 32455],
          [22591, 27955, 33319, 38683],
          [26227, 32455, 38683, 44911]],

         [[19171, 23239, 27307, 31375],
          [23239, 28171, 33103, 38035],
          [27307, 33103, 38899, 44695],
          [31375, 38035, 44695, 51355]]]])
D =
tensor([[[[   34200,    35500,    36800,    38100,    39400,    40700],
          [  192312,   199516,   206720,   213924,   221128,   228332],
          [  350424,   363532,   376640,   389748,   402856,   415964],
          [  508536,   527548,   546560,   565572,   584584,   603596]],

         [[  168144,   173620,   179096,   184572,   190048,   195524],
          [  403152,   416260,   429368,   442476,   455584,   468692],
          [  638160,   658900,   679640,   700380,   721120,   741860],
          [  873168,   901540,   929912,   958284,   986656,  1015028]]],


        [[[ 5702232,  5785324,  5868416,  5951508,  6034600,  6117692],
          [ 7056120,  7158940,  7261760,  7364580,  7467400,  7570220],
          [ 8410008,  8532556,  8655104,  8777652,  8900200,  9022748],
          [ 9763896,  9906172, 10048448, 10190724, 10333000, 10475276]],

         [[ 7522704,  7623796,  7724888,  7825980,  7927072,  8028164],
          [ 9119376,  9241924,  9364472,  9487020,  9609568,  9732116],
          [10716048, 10860052, 11004056, 11148060, 11292064, 11436068],
          [12312720, 12478180, 12643640, 12809100, 12974560, 13140020]]]])
.shape =
torch.Size([2, 4, 12])
b.shape =
torch.Size([2, 4, 2, 6])
X.shape =
torch.Size([2, 2, 4, 6])
C.shape =
torch.Size([2, 2, 6, 4])
B.shape =
torch.Size([2, 2, 4, 4])
D.shape =
torch.Size([2, 2, 4, 6])
"""


#=====================================attention QKV================================================
a = torch.arange(60).reshape(3, 5, 4)  # batch, target_len, feats
b = torch.arange(72).reshape(3, 6, 4)  # batch, seq_len, feats
c = torch.arange(144).reshape(3, 6, 8)  # batch, seq_len, val_feats

d = torch.matmul(a, b.transpose(-2, -1))

e = torch.matmul(d, c)
print(f"a = \n{a}\nb = \n{b}\nc = \n{c}\nd = \n{d}\ne = \n{e}\n")

print(f"a.shape = \n{a.shape}\nb.shape = \n{b.shape}\nc.shape = \n{c.shape}\nd.shape = \n{d.shape}\ne.shape = \n{e.shape}\n")






#=========================================展示mask的过程 mask() =============================================
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
#=========================================展示mask的过程 mask() =============================================
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
#         [ -100.0000, -100.0000],
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



# 当然， 你也可以自己手动mask
# attn_output = self.mha(x, x, x, mask)
# x.shape=(128,31,d_model=128), mask.shape = torch.Size([128, 1, 31])
# Q*K = (128,8,31,31)  mask = (128,1,1,31)
a = torch.arange(48).reshape(2,2,3,4)
masK = torch.randint(0,2,(2,1,1,4))
b = a+masK*(-1e9)


# attn_output = self.mha(x, x, x, mask)
# x.shape = torch.Size(128, 30,128), tgt_mask.shape =  torch.Size([128, 30, 30])
# Q*K = (128,8,30,30)  mask = (128,1,30,30)
a = torch.arange(3*2*4*4).reshape(3,2,4,4)
masK = torch.randint(0,2,(3,1,4,4))
b = a+masK*(-1e9)



a = torch.arange(3*2*4*4).reshape(3,2,4,4)
masK = torch.randint(0,2,(3,1,1,4))
b = a+masK*(-1e9)



# attn_output = self.src_mha(x, memory, memory, src_mask)  # q, k, v
#  x.shape = torch.Size(128, 30,128),src_mask.shape = torch.Size([128, 1, 31])
# Q*K = (128,8,30,31)  mask = (128,1,1,31)
a = torch.arange(3*2*3*4).reshape(3,2,3,4)
masK = torch.randint(0,2,(3,1,1,4))
b = a+masK*(-1e9)





#==========================  展示mask叠加的过程============================================

seq_len = 5   # 30
batchsize = 3  #128
pad = 0



tgt = torch.randint(1, 10, (batchsize, seq_len))
tgt[-1,-4:] = 0
tgt[-2,-3:] = 0
tgt[-3,-1:] = 0

tgt_mask = (tgt != pad).unsqueeze(-2).type(torch.uint8)
print(f"tgt_mask.shape = {tgt_mask.shape}")
print(f"tgt_mask = \n{tgt_mask}")


attn_shape = (1, seq_len, seq_len)
sub_mask = torch.from_numpy(np.triu(np.ones(attn_shape), k=1) .astype('uint8'))==0
sub_mask = sub_mask.type(torch.uint8)
print(f"sub_mask = \n{sub_mask}")


tgt_mask = tgt_mask & sub_mask
print(f"tgt_mask.shape = {tgt_mask.shape}")
print(f"tgt_mask = {tgt_mask}")



#============================================================================
def subsequent_mask(size):
    "Mask out subsequent positions."
    ##生成向后遮掩的掩码张量，参数size是掩码张量最后两个维度的大小，它最后两维形成一个方阵
    attn_shape = (1, size, size)
    #然后使用np.ones方法向这个形状中添加1元素，形成上三角阵(k=1)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)


trg = torch.randint(1, 10, (batchsize, seq_len))
trg[-1,-4:] = 0
trg[-2,-3:] = 0
trg[-3,-1:] = 0
trg_mask = (trg == pad ).unsqueeze(-2).type(torch.FloatTensor) # [batch, 1, seq_len]  torch.Size([128, 1, 30])
print(f"trg_mask = \n{trg_mask}")


look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)#  torch.Size([1, 30, 30])
print(f"look_ahead_mask = \n{look_ahead_mask}")

# 将 pad 产生的 mask，和序列一次预测下一个单词产生的 mask 结合起来
combined_mask = torch.max(trg_mask, look_ahead_mask)    # torch.Size([128, 30, 30])
print(f"combined_mask = \n{combined_mask}")




a = torch.arange(24).reshape(2,3,4)
# tensor([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]],

#         [[12, 13, 14, 15],
#          [16, 17, 18, 19],
#          [20, 21, 22, 23]]])


print(f"a[:,-1] = {a[:,-1]}")
# tensor([[ 8,  9, 10, 11],
#         [20, 21, 22, 23]])


print(f"a[:,-1,:-1] = {a[:,-1,:-1]}")
# tensor([[ 8,  9, 10],
#         [20, 21, 22]])


print(f"a[:,-1,-1] = {a[:,-1,-1]}")
# tensor([11, 23])
