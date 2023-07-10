#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 22:19:41 2022

@author: jack
"""

"""
torch.optim.SGD
随机梯度下降算法，带有动量（momentum）的算法作为一个可选参数可以进行设置，样例如下：

#lr参数为学习率，对于SGD来说一般选择0.1 0.01.0.001，如何设置会在后面实战的章节中详细说明
##如果设置了momentum，就是带有动量的SGD，可以不设置
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
torch.optim.RMSprop
除了以上的带有动量Momentum梯度下降法外，RMSprop（root mean square prop）也是一种可以加快梯度下降的算法，利用RMSprop算法，可以减小某些维度梯度更新波动较大的情况，使其梯度下降的速度变得更快

#我们的课程基本不会使用到RMSprop所以这里只给一个实例
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
torch.optim.Adam
Adam 优化算法的基本思想就是将 Momentum 和 RMSprop 结合起来形成的一种适用于不同深度学习结构的优化算法

# 这里的lr，betas，还有eps都是用默认值即可，所以Adam是一个使用起来最简单的优化方法
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

"""




"""


### nn.L1Loss:

输入x和目标y之间差的绝对值，要求 x 和 y 的维度要一样（可以是向量或者矩阵），得到的 loss 维度也是对应一样的

$ loss(x,y)=1/n\sum|x_i-y_i| $

### nn.NLLLoss:

用于多分类的负对数似然损失函数

$ loss(x, class) = -x[class]$

NLLLoss中如果传递了weights参数，会对损失进行加权，公式就变成了

$ loss(x, class) = -weights[class] * x[class] $



### nn.MSELoss:

均方损失函数 ，输入x和目标y之间均方差

$ loss(x,y)=1/n\sum(x_i-y_i)^2 $

### nn.CrossEntropyLoss:

多分类用的交叉熵损失函数，LogSoftMax和NLLLoss集成到一个类中，会调用nn.NLLLoss函数，我们可以理解为CrossEntropyLoss()=log_softmax() + NLLLoss()

$loss(x,class)=−log(\frac{exp(x[class])}{∑_j exp(x[j])}) =−x[class]+log(∑_j exp(x[j]))$

因为使用了NLLLoss，所以也可以传入weight参数，这时loss的计算公式变为：

$ loss(x, class) = weights[class] * (-x[class] + log(\sum_j exp(x[j]))) $

所以一般多分类的情况会使用这个损失函数

# output是网络的输出，size=[batch_size, class]
#如网络的batch size为128，数据分为10类，则size=[128, 10]

# target是数据的真实标签，是标量，size=[batch_size]
#如网络的batch size为128，则size=[128]

crossentropyloss=nn.CrossEntropyLoss()
crossentropyloss_output=crossentropyloss(output,target)



### nn.BCELoss:

计算 x 与 y 之间的二进制交叉熵。

$ loss(o,t)=-\frac{1}{n}\sum_i(t[i] *log(o[i])+(1-t[i])* log(1-o[i])) $

与NLLLoss类似，也可以添加权重参数：

$ loss(o,t)=-\frac{1}{n}\sum_iweights[i] *(t[i]* log(o[i])+(1-t[i])* log(1-o[i])) $

用的时候需要在该层前面加上 Sigmoid 函数。

"""


"""
通常会在遍历epochs的过程中依次用到optimizer.zero_grad(),loss.backward()和optimizer.step()三个函数
总得来说，这三个函数的作用是先将梯度归零（optimizer.zero_grad()），然后反向传播计算得到每个参数的梯度值（loss.backward()），最后通过梯度下降执行一步参数更新（optimizer.step()）
一、optimizer.zero_grad()：
optimizer.zero_grad()函数会遍历模型的所有参数，通过p.grad.detach_()方法截断反向传播的梯度流，再通过p.grad.zero_()函数将每个参数的梯度值设为0，即上一次的梯度记录被清空。

因为训练的过程通常使用mini-batch方法，所以如果不将梯度清零的话，梯度会与上一个batch的数据相关，因此该函数要写在反向传播和梯度下降之前。


二、loss.backward()：
PyTorch的反向传播(即tensor.backward())是通过autograd包来实现的，autograd包会根据tensor进行过的数学运算来自动计算其对应的梯度。

具体来说，torch.tensor是autograd包的基础类，如果你设置tensor的requires_grads为True，就会开始跟踪这个tensor上面的所有运算，如果你做完运算后使用tensor.backward()，所有的梯度就会自动运算，tensor的梯度将会累加到它的.grad属性里面去。

更具体地说，损失函数loss是由模型的所有权重w经过一系列运算得到的，若某个w的requires_grads为True，则w的所有上层参数（后面层的权重w）的.grad_fn属性中就保存了对应的运算，然后在使用loss.backward()后，会一层层的反向传播计算每个w的梯度值，并保存到该w的.grad属性中。

如果没有进行tensor.backward()的话，梯度值将会是None，因此loss.backward()要写在optimizer.step()之前。

三、optimizer.step()：
step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值。因为梯度下降是基于梯度的，所以在执行optimizer.step()函数前应先执行loss.backward()函数来计算梯度。

注意：optimizer只负责通过梯度下降进行优化，而不负责产生梯度，梯度是tensor.backward()方法产生的。



loss.backward()的作用
我们都知道，loss.backward()函数的作用是根据loss来计算网络参数的梯度，其对应的输入默认为网络的叶子节点，即数据集内的数据，


optimizer.step()的作用
优化器的作用就是针对计算得到的参数梯度对网络参数进行更新，所以要想使得优化器起作用，主要需要两个东西：

优化器需要知道当前的网络模型的参数空间
优化器需要知道反向传播的梯度信息（即backward计算得到的信息）


"""

#====================================================
print("=="*60)
print("   nn.L1Loss  ")
print("=="*60)
import torch
import torch.nn as nn

input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)

loss = nn.L1Loss(reduction='none')
output = loss(input, target)
print(f"output = {output}")

loss = nn.L1Loss(reduction='mean')
output = loss(input, target)
print(f"output = {output}")

loss = nn.L1Loss(reduction='sum')
output = loss(input, target)
print(f"output = {output}")


# https://zhuanlan.zhihu.com/p/98785902
# https://blog.csdn.net/weixin_38314865/article/details/104311969
# https://blog.csdn.net/geter_CS/article/details/84857220


import torch
import torch.nn as nn
import math

# 注意这里的标签值class，并不参与直接计算，而是作为一个索引,索引对象为实际类别
# 举个栗子，我们一共有三种类别，批量大小为1（为了好计算），那么输入size为（1,3），具体值为torch.Tensor([[-0.7715, -0.6205,-0.2562]])。标签值为target = torch.tensor([0])，
# 这里标签值为0，表示属于第0类。loss计算如下：




print("=="*60)
print("   nn.NLLLoss()  ")
print("=="*60)
#我们在看看是否等价nn.logSoftmax()和nn.NLLLoss()的整合：
m = nn.LogSoftmax()
loss = nn.NLLLoss()
inputx =torch.Tensor([[-0.7715, -0.6205,-0.2562]])
target = torch.tensor([0])
print(f"inputx 1 = {inputx}")

inputx=m(inputx)
print(f"inputx 2 = {inputx}")

output2 = loss(inputx, target) #target 和 input千万别反了,会报错
print(f"output2 = {output2}")

# 注意，使用nn.CrossEntropyLoss()时，不需要现将输出经过softmax层，否则计算的损失会有误，
# 即直接将网络输出用来计算损失即可

#====================================================
print("=="*60)
print("   nn.NLLLoss()  ")
print("=="*60)
#我们在看看是否等价nn.logSoftmax()和nn.NLLLoss()的整合：
m = nn.LogSoftmax()
loss = nn.NLLLoss(reduction='none')
inputx = torch.Tensor([[-0.7715, -0.6205,-0.2562],[-1.7715, -0.6305,-0.2562]])
target = torch.tensor([0,1])
print(f"inputx 1 = {inputx}")

inputx=m(inputx)
print(f"inputx 2 = {inputx}")

output2 = loss(inputx, target)#target 和 input千万别反了,会报错
print(f"output2 = {output2}")




#====================================================
print("=="*60)
print("   nn.NLLLoss()  ")
print("=="*60)
#我们在看看是否等价nn.logSoftmax()和nn.NLLLoss()的整合：
m = nn.LogSoftmax()
loss = nn.NLLLoss(reduction='mean')
inputx = torch.Tensor([[-0.7715, -0.6205,-0.2562],[-1.7715, -0.6305,-0.2562]])
target = torch.tensor([0,1])
print(f"inputx 1 = {inputx}")

inputx=m(inputx)
print(f"inputx 2 = {inputx}")

output2 = loss(inputx, target) #target 和 input千万别反了,会报错
print(f"output2 = {output2}")



#====================================================
print("=="*60)
print("   nn.NLLLoss()  ")
print("=="*60)
#我们在看看是否等价nn.logSoftmax()和nn.NLLLoss()的整合：
m = nn.LogSoftmax()
loss = nn.NLLLoss(reduction='sum')
inputx = torch.Tensor([[-0.7715, -0.6205,-0.2562],[-1.7715, -0.6305,-0.2562]])
target = torch.tensor([0,1])
print(f"inputx 1 = {inputx}")

inputx=m(inputx)
print(f"inputx 2 = {inputx}")

output2 = loss(inputx, target)#target 和 input千万别反了,会报错
print(f"output2 = {output2}")






#=====================================================================
#    https://zhuanlan.zhihu.com/p/98785902
#=====================================================================
import torch
import torch.nn as nn
x_input=torch.randn(3,3)#随机生成输入
print('x_input:\n',x_input)
y_target=torch.tensor([1,2,0])#设置输出具体值
print('y_target\n',y_target)

#计算输入softmax，此时可以看到每一行加到一起结果都是1
softmax_func=nn.Softmax(dim=1)
soft_output=softmax_func(x_input)
print('soft_output:\n',soft_output)

#在softmax的基础上取log
log_output=torch.log(soft_output)
print('log_output:\n',log_output)

#对比softmax与log的结合与nn.LogSoftmaxloss(负对数似然损失)的输出结果，发现两者是一致的。
logsoftmax_func=nn.LogSoftmax(dim=1)
logsoftmax_output=logsoftmax_func(x_input)
print('logsoftmax_output:\n',logsoftmax_output)

#pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
nllloss_func=nn.NLLLoss()
nlloss_output=nllloss_func(logsoftmax_output,y_target)#target 和 input千万别反了,会报错
print('nlloss_output:\n',nlloss_output)


#直接使用pytorch中的loss_func=nn.CrossEntropyLoss()看与经过NLLLoss的计算是不是一样
crossentropyloss=nn.CrossEntropyLoss()
crossentropyloss_output=crossentropyloss(x_input,y_target)#target 和 input千万别反了,会报错
print('crossentropyloss_output:\n',crossentropyloss_output)


#直接使用pytorch中的loss_func=nn.CrossEntropyLoss()看与经过NLLLoss的计算是不是一样
crossentropyloss=nn.CrossEntropyLoss(reduction='mean')
crossentropyloss_output=crossentropyloss(x_input,y_target)#target 和 input千万别反了,会报错
print('crossentropyloss_output:\n',crossentropyloss_output)


#直接使用pytorch中的loss_func=nn.CrossEntropyLoss()看与经过NLLLoss的计算是不是一样
crossentropyloss=nn.CrossEntropyLoss(reduction='none')
crossentropyloss_output=crossentropyloss(x_input,y_target)#target 和 input千万别反了,会报错
print('crossentropyloss_output:\n',crossentropyloss_output)


#直接使用pytorch中的loss_func=nn.CrossEntropyLoss()看与经过NLLLoss的计算是不是一样
crossentropyloss=nn.CrossEntropyLoss(reduction='sum')
crossentropyloss_output=crossentropyloss(x_input,y_target)#target 和 input千万别反了,会报错
print('crossentropyloss_output:\n',crossentropyloss_output)


#通过上面的结果可以看出，直接使用pytorch中的loss_func=nn.CrossEntropyLoss()计算得到的结果与softmax-log-NLLLoss计算得到的结果是一致的。


"""
https://www.codeleading.com/article/37034424506/

Pytorch中的交叉熵损失函数nn.CrossEntropyLoss之中包含两个非常有用的参数：

weight：用来平衡样本之间的不平衡；
reduction：用来指定损失结果返回的是mean、sum还是none。
事实的使用方法是：

使用reduction='mean’的同时指定weights的时候，不会产生任何错误，而且reduction='mean’也是默认的参数值；
使用reduction='none’的同时指定weights的时候，会导致产生的结果有错误，这是因为当我们指定reduction='none’的时候，该函数不会对结果使用weight进行规范化。

"""
x = torch.tensor([[-2.2105,  0.0971,  0.3266, -0.0403, -0.7734],
        [-0.9611,  0.9855, -0.7131,  1.9102, -0.0249],
        [ 1.0615, -0.8349,  0.8135, -0.8576, -0.3983],
        [ 0.0135, -1.0360, -0.4179,  1.3075,  2.4753],
        [-1.7417, -0.8576, -2.5673,  0.7397,  1.4467],
        [-1.8565,  0.5760, -1.6394, -1.3055, -1.0683],
        [ 1.7444, -0.4352,  0.5160,  1.1678,  0.4746],
        [ 0.2409,  0.0209,  0.4686, -0.1821,  1.1949],
        [ 0.7831, -2.3620,  0.5771, -0.4640, -1.4672],
        [-0.0283,  0.8300, -1.4431, -1.1761,  0.4875]])
y= torch.tensor([3, 1, 0, 4, 1, 0, 1, 2, 0, 2])   #这里的5不能>=5,必须<=x.size(-1)
weights = torch.tensor([1., 2., 3., 4., 5.])
print(f"x = \n{x}\ny = {y}")

criterion_good = nn.CrossEntropyLoss(weight=weights)
loss_good = criterion_good(x, y)
print(f"loss_good = {loss_good}")


criterion = nn.CrossEntropyLoss(weight=weights, reduction='none')
loss = criterion(x, y)  #target 和 input千万别反了,会报错
print(f"loss = {loss}")

loss = loss.sum() / weights[y].sum()
print(f"loss = {loss}")


criterion = nn.CrossEntropyLoss(weight=weights, reduction='mean')
loss = criterion(x, y)  #target 和 input千万别反了,会报错
print(f"loss = {loss}")

criterion = nn.CrossEntropyLoss(weight=weights, reduction='sum')
loss = criterion(x, y)  #target 和 input千万别反了,会报错
print(f"loss = {loss}")

"""
x = torch.randn(10, 5)
y= torch.randint(0, 5, (10,))   #这里的5不能>=5,必须<=x.size(-1)
weights = torch.tensor([1., 2., 3., 4., 5.])
print(f"x = \n{x}\ny = {y}")
x =
tensor([[-2.2105,  0.0971,  0.3266, -0.0403, -0.7734],
        [-0.9611,  0.9855, -0.7131,  1.9102, -0.0249],
        [ 1.0615, -0.8349,  0.8135, -0.8576, -0.3983],
        [ 0.0135, -1.0360, -0.4179,  1.3075,  2.4753],
        [-1.7417, -0.8576, -2.5673,  0.7397,  1.4467],
        [-1.8565,  0.5760, -1.6394, -1.3055, -1.0683],
        [ 1.7444, -0.4352,  0.5160,  1.1678,  0.4746],
        [ 0.2409,  0.0209,  0.4686, -0.1821,  1.1949],
        [ 0.7831, -2.3620,  0.5771, -0.4640, -1.4672],
        [-0.0283,  0.8300, -1.4431, -1.1761,  0.4875]])
y = tensor([3, 1, 0, 4, 1, 0, 1, 2, 0, 2])


criterion_good = nn.CrossEntropyLoss(weight=weights)
loss_good = criterion_good(x, y)
print(f"loss_good = {loss_good}")
loss_good = 1.7052634954452515

criterion = nn.CrossEntropyLoss(weight=weights, reduction='none')
loss = criterion(x, y)  #target 和 input千万别反了,会报错
print(f"loss = {loss}")
loss = tensor([5.7261, 2.8753, 0.8370, 1.9656, 5.6130, 2.8659, 5.9798, 4.8431, 0.8107,
        9.4097])


loss = loss.sum() / weights[y].sum()
print(f"loss = {loss}")
loss = 1.705263614654541

criterion = nn.CrossEntropyLoss(weight=weights, reduction='mean')
loss = criterion(x, y)  #target 和 input千万别反了,会报错
print(f"loss = {loss}")
loss = 1.7052634954452515

criterion = nn.CrossEntropyLoss(weight=weights, reduction='sum')
loss = criterion(x, y)  #target 和 input千万别反了,会报错
print(f"loss = {loss}")
loss = 40.92632293701172

(0.0403 + np.log( np.exp(-2.2105) +np.exp(0.0971) +np.exp(0.3266) +np.exp(-0.0403) +np.exp(-0.7734) ))*4
Out[154]: 5.726134558306921


a = [1,2,3,4,5]

b = [3, 1, 0, 4, 1, 0, 1, 2, 0, 2]


40.92632293701172/sum([a[i] for i in b])
Out[163]: 1.7052634557088215

"""

print("=="*60)
print("   nn.CrossEntropyLoss() ")
print("=="*60)

entroy=nn.CrossEntropyLoss()
input=torch.Tensor([[-0.7715, -0.6205,-0.2562]])
target = torch.tensor([0])  #这里最大为2,不能超过input.size(-1)-1 = 2
output = entroy(input, target)  #target 和 input千万别反了,会报错
print(f"output1 = {output}")
#根据公式计算
# −x[0]+log(exp(x[0])+exp(x[1])+exp(x[2]))
# = 0.7715 + log( exp( − 0.7715 ) + exp( − 0.6205 ) + exp( − 0.2562 ) )
# =0.7715+log(exp(−0.7715)+exp(−0.6205)+exp(−0.2562)=1.3447266007601868


#=======================1=============================

y_hat=torch.Tensor([[-0.7715, -0.6205,-0.2562],[-1.7715, -0.6305,-0.2562]])
y = torch.tensor([0,1])

loss = nn.CrossEntropyLoss(size_average=False, reduce=False, reduction='sum')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")



loss = nn.CrossEntropyLoss(size_average=False, reduce=False, reduction='mean')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")


loss = nn.CrossEntropyLoss(size_average=False, reduce=False, reduction='none')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")

# output = tensor([1.3447, 1.0201])
# output = tensor([1.3447, 1.0201])
# output = tensor([1.3447, 1.0201])

#========================= 2 ===========================
loss = nn.CrossEntropyLoss(size_average=True, reduce=False, reduction='sum')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")

loss = nn.CrossEntropyLoss(size_average=True, reduce=False, reduction='mean')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")

loss = nn.CrossEntropyLoss(size_average=True, reduce=False, reduction='none')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")

# output = tensor([1.3447, 1.0201])
# output = tensor([1.3447, 1.0201])
# output = tensor([1.3447, 1.0201])

#=========================== 3 =========================
loss = nn.CrossEntropyLoss(size_average=False, reduce=True, reduction='sum')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")


loss = nn.CrossEntropyLoss(size_average=False, reduce=True, reduction='mean')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")


loss = nn.CrossEntropyLoss(size_average=False, reduce=True, reduction='none')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")

# output = 2.3648266792297363
# output = 2.3648266792297363
# output = 2.3648266792297363

#=========================== 4 =========================
loss = nn.CrossEntropyLoss(size_average=True, reduce=True, reduction='sum')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")



loss = nn.CrossEntropyLoss(size_average=True, reduce=True, reduction='mean')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")

loss = nn.CrossEntropyLoss(size_average=True, reduce=True, reduction='none')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")

# output = 1.1824133396148682
# output = 1.1824133396148682
# output = 1.1824133396148682

#=========================== 5 =========================
loss = nn.CrossEntropyLoss( reduction='sum')

output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")


loss = nn.CrossEntropyLoss( reduction='mean')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")


loss = nn.CrossEntropyLoss( reduction='none')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")

# output = 2.3648266792297363
# output = 1.1824133396148682
# output = tensor([1.3447, 1.0201])
#=========================== 6  =========================
y_hat = torch.randn(3, 5, requires_grad = True)
y =  torch.randn(3, 5).softmax(dim = 1)
print(f"y_hat = {y_hat}, \ny = {y}")

loss = nn.CrossEntropyLoss( reduction='sum')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")


loss = nn.CrossEntropyLoss( reduction = 'mean')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")


loss = nn.CrossEntropyLoss( reduction='none')
output = loss(y_hat, y)#target 和 input千万别反了,会报错
print(f"output = {output}")

#========================== weights ==========================
input=torch.Tensor([[-0.7715, -0.6205,-0.2562],[-1.7715, -0.6305,-0.2562]])
target = torch.tensor([0, 1])
weights = torch.tensor([4, 2., 8.])
entroy=nn.CrossEntropyLoss(weight=weights,reduction='none')

output = entroy(input, target)  #target 和 input千万别反了,会报错
print(f"output1 = {output}")
#根据公式计算
# −x[0]+log(exp(x[0])+exp(x[1])+exp(x[2]))
# = 0.7715 + log( exp( − 0.7715 ) + exp( − 0.6205 ) + exp( − 0.2562 ) = 1.3447266007601868
# =0.7715+log(exp(−0.7715)+exp(−0.6205)+exp(−0.2562)=1.3447266007601868




print("=="*60)
print(" torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction=‘mean’) ")
print("=="*60)
"""
PyTorch对应函数为：
torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction=‘mean’)
计算目标值和预测值之间的二进制交叉熵损失函数。

有四个可选参数：weight、size_average、reduce、reduction

(1) weight必须和target的shape一致，默认为none。定义BCELoss的时候指定即可。
(2) 默认情况下 nn.BCELoss()，reduce = True，size_average = True。
(3) 如果reduce为False，size_average不起作用，返回向量形式的loss。
(4) 如果reduce为True，size_average为True，返回loss的均值，即loss.mean()。
(5) 如果reduce为True，size_average为False，返回loss的和，即loss.sum()。
(6) 如果reduction = ‘none’，直接返回向量形式的 loss。
(7) 如果reduction = ‘sum’，返回loss之和。
(8) 如果reduction = 'elementwise_mean'，返回loss的平均值。
(9) 如果reduction = ''mean，返回loss的平均值


放在torch.nn.BCELoss()中的参数一定要经过sigmoid激活
"""


# https://blog.csdn.net/qq_29631521/article/details/104907401
import torch
import torch.nn as nn

m = nn.Sigmoid()
# m = nn.Softmax()
y_hat1 = torch.randn(3, requires_grad=True)
y = torch.empty(3).random_(2)
y_hat = m(y_hat1)
print(f"y = {y}\ny_hat = {y_hat} \n\n")

#======================================================================================
loss = nn.BCELoss(size_average=False, reduce=False, reduction='sum')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output} ")


loss = nn.BCELoss(size_average=False, reduce=False, reduction='mean')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output} ")

loss = nn.BCELoss(size_average=False, reduce=False, reduction='none')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output}\n\n")

#======================================================================================
loss = nn.BCELoss(size_average=True, reduce=False, reduction='sum')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output} ")


loss = nn.BCELoss(size_average=True, reduce=False, reduction='mean')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output} ")

loss = nn.BCELoss(size_average=True, reduce=False, reduction='none')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output}\n\n")


#======================================================================================
loss = nn.BCELoss(size_average=False, reduce=True, reduction='sum')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output} ")


loss = nn.BCELoss(size_average=False, reduce=True, reduction='mean')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output} ")

loss = nn.BCELoss(size_average=False, reduce=True, reduction='none')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output}\n\n")

#======================================================================================
loss = nn.BCELoss(size_average=True, reduce=True, reduction='sum')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output} ")


loss = nn.BCELoss(size_average=True, reduce=True, reduction='mean')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output} ")

loss = nn.BCELoss(size_average=True, reduce=True, reduction='none')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output}\n\n")

#======================================================================================
loss = nn.BCELoss(  reduction='sum')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output} ")


loss = nn.BCELoss(  reduction='mean')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output} ")

loss = nn.BCELoss( reduction='none')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output}\n\n")

loss = nn.BCELoss( )
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output}\n\n")


#=======================================================================================
import torch
import torch.nn as nn

m = nn.Sigmoid()
y_hat1 = torch.randn(3, requires_grad=True)
y = torch.randn(3, requires_grad=True)
y = m(y)
y_hat = m(y_hat1)
print(f"y = {y}\ny_hat = {y_hat} \n\n")

#======================================================================================
loss = nn.BCELoss( reduction='sum')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output} ")


loss = nn.BCELoss( reduction='mean')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output} ")

loss = nn.BCELoss( reduction='none')
output = loss(y_hat, y)   #target 和 lossinput千万别反了，不会报错，但是结果没意义
# 放在loss(lossinput, target)中的参数lossinput, target一定要经过sigmoid激活
print(f"output = {output}\n\n")




import torch
import torch.nn as nn

m = nn.Sigmoid()
weights=torch.randn(3)

loss = nn.BCELoss(weight=weights,size_average=False, reduce=False)
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
lossinput = m(input)
output = loss(lossinput, target)  #target 和 lossinput千万别反了，不会报错，但是结果没意义

print("输入值:")
print(lossinput)
print("输出的目标值:")
print(target)
print("权重值")
print(weights)
print("计算loss的结果:")
print(output)
"""
输入值:
tensor([0.5537, 0.9069, 0.1573], grad_fn=<SigmoidBackward0>)
输出的目标值:
tensor([0., 1., 0.])
权重值
tensor([-1.2380,  0.7973, -0.0140])
计算loss的结果:
tensor([-0.9987,  0.0779, -0.0024], grad_fn=<BinaryCrossEntropyBackward0>)
"""






# https://www.jianshu.com/p/0062d04a2782
"""
该类主要用来创建衡量目标和输出之间的二进制交叉熵的标准。
用法如下：
torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')

参数：

weight，表示对loss中每个元素的加权权值；
reduction, 指定输出的格式，包括'none'，'mean'，'sum'；
其它两个参数一般不推荐使用；
形状：

input，(N, *)；
target，(N, *)；
output，标量，当reduction为'none'时，为(N, *)。

计算
当reduction参数为'none'，即对计算结果不进行处理时，loss可以表示为，

l(x,y) = L = {l1,l2,...,lN}


其中，
ln = -w_n [yn*log(xn) + (1-yn)*log(1-xn)]

这里N为batch size。如果reduction参数不为'none'时，


l(x,y) = mean(L)   if reduction = 'mean'
l(x,y) = sum(L)   if reduction = 'sum'

这里需要注意的是target的值必须为0或1，虽然不是0或1不会报错，但是实际使用中只有0和1才有意义。


"""

inpuT = torch.tensor([[-0.2383, 0.4086, 0.0835],
                     [-1.2237, 2.3024, -0.1782],
                     [0.6650, -0.3253, -0.6224]])

target = torch.tensor([[0, 1, 0],
                     [1, 0, 0],
                     [1, 1, 1]])*1.0



m = torch.nn.Sigmoid()
input1 = m(inpuT)
print(f"input1 = {input1}")
"""
input1 = tensor([[0.4407, 0.6008, 0.5209],
        [0.2273, 0.9091, 0.4556],
        [0.6604, 0.4194, 0.3492]])
"""


loss1 = torch.nn.BCELoss()
output1 = loss1(input1,target) #target 和 input千万别反了,，不会报错，但是结果没意义
print(f"output1 = {output1}")
# output1 = 0.9610683917999268

loss1 = torch.nn.BCELoss(reduction='mean')
output1 = loss1(input1,target)  #target 和 input千万别反了,，不会报错，但是结果没意义
print(f"output1 = {output1}")
# output1 = 0.9610683917999268

loss1 = torch.nn.BCELoss(reduction='none')
output1 = loss1(input1,target)   #target 和 input千万别反了,，不会报错，但是结果没意义
print(f"output1 = {output1}")
"""
output1 = tensor([[0.5811, 0.5096, 0.7358],
        [1.4815, 2.3977, 0.6080],
        [0.4149, 0.8690, 1.0520]])
"""


loss1 = torch.nn.BCELoss(reduction='sum')
output1 = loss1(input1,target)  #target 和 input千万别反了,，不会报错，但是结果没意义
print(f"output1 = {output1}")
#output1 = 8.649615287780762


"""
2. BCEWithLogitsLoss
这个loss类将sigmoid操作和与BCELoss集合到了一个类。
用法如下：
torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)

参数：

weight (Tensor)，针对每个loss元素的加权权值；
reduction (string), 指定输出的格式，包括'none'，'mean'，'sum'；
pos_weight (Tensor)，正样例的权重，必须为长度为类别数量的向量，主要可用来处理类别不均衡情形。
其它两个参数一般不推荐使用；
形状：

input，(N, *)；
target，(N, *)；
output，标量，当reduction为'none'时，为(N, *)。
计算:计算过程与BCELoss类似，除了增加一个sigmoid层，
ln = -wn*[yn * log\sigma(xn) + (1-yn)*(1-\sigma(xn))]

"""
inpuT = torch.tensor([[-0.2383, 0.4086, 0.0835],
                     [-1.2237, 2.3024, -0.1782],
                     [0.6650, -0.3253, -0.6224]])

target = torch.tensor([[0, 1, 0],
                     [1, 0, 0],
                     [1, 1, 1]])*1.0
loss2 = torch.nn.BCEWithLogitsLoss()
output2 = loss2(inpuT,target) #target 和 input千万别反了,，不会报错，但是结果没意义
print(f"output2 = {output2}")




inpuT = torch.randn(3,10,10)

target = torch.randint(0,2,size=(3,10,10))*1.0

loss2 = torch.nn.BCEWithLogitsLoss(reduction='none')
output2 = loss2(inpuT,target) #target 和 input千万别反了,，不会报错，但是结果没意义
print(f"output2.shape = {output2.shape}")



# nn.MSELoss
"""
https://www.cnblogs.com/amazingter/p/14044236.html


PyTorch中MSELoss的使用
参数
torch.nn.MSELoss(size_average=None, reduce=None, reduction: str = 'mean')

size_average和reduce在当前版本的pytorch已经不建议使用了，只设置reduction就行了。

reduction的可选参数有：'none' 、'mean' 、'sum'

reduction='none'：求所有对应位置的差的平方，返回的仍然是一个和原来形状一样的矩阵。

reduction='mean'：求所有对应位置差的平方的均值，返回的是一个标量。

reduction='sum'：求所有对应位置差的平方的和，返回的是一个标量。


"""
import torch
x = torch.Tensor([[1, 2, 3],
                      [2, 1, 3],
                      [3, 1, 2]])

y = torch.Tensor([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])



#如果reduction='none'：

criterion1 = nn.MSELoss(reduction='none')
loss1 = criterion1(x, y)  #x,y顺序不影响结果，因为是均方根
print(loss1)
"""
tensor([[0., 4., 9.],
        [4., 0., 9.],
        [9., 1., 1.]])
"""



#如果reduction='mean'：
criterion2 = nn.MSELoss(reduction='mean')
loss2 = criterion2(x, y) #x,y顺序不影响结果，因为是均方根
print(loss2)

#tensor(4.1111)


#如果reduction='sum'：
criterion3 = nn.MSELoss(reduction='sum')
loss3 = criterion3(x, y) #x,y顺序不影响结果，因为是均方根
print(loss3)

#tensor(37.)


#https://blog.csdn.net/Will_Ye/article/details/104994504
import torch
import torch.nn.functional as F
input = torch.randn(2,2,3)
print(input)


m = nn.Softmax()
print(m(input))

#当dim=0时， 是对每一维度相同位置的数值进行softmax运算
m = nn.Softmax(dim=0)
print(m(input))
#当dim=-3时， 是对每一维度相同位置的数值进行softmax运算
m = nn.Softmax(dim=-3)
print(m(input))

#当dim=1时， 是对某一维度的列进行softmax运算：
m = nn.Softmax(dim=1)
print(m(input))
#当dim=-2时， 是对某一维度的列进行softmax运算：
m = nn.Softmax(dim=-2)
print(m(input))

#当dim=2时， 是对某一维度的行进行softmax运算：
m = nn.Softmax(dim=2)
print(m(input))


#要注意的是当dim=-1时， 是对某一维度的行进行softmax运算：
m = nn.Softmax(dim=-1)
print(m(input))













































































































































































































































































































































