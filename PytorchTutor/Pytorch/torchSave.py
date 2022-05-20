#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 21:28:59 2022

@author: jack


https://blog.csdn.net/qq_38962621/article/details/111148636

https://blog.csdn.net/weixin_38145317/article/details/111152491



"""


import torch



#PyTorch中的tensor可以保存成 .pt 或者 .pth 格式的文件，使用torch.save()方法保存张量，使用torch.load()来读取张量：

x = torch.rand(4,5)
path = "/home/jack/公共的/MLData/TrashFile/test.pt"
torch.save(x, path)
y = torch.load(path)
print(y)





#save和load方法也适用于其他数据类型，比如list、tuple、dict等：
a = {'a':torch.rand(2,2), 'b':torch.rand(3,4)}
torch.save(a, path)
print(f"a = \n{a}")
b = torch.load(path)
print(f"b = \n{b}")



#PyTorch中，使用 torch.save 保存的不仅有其中的数据，还包括一些它的信息，包括它与其它数据（可能存在）的关系，这一点是很有趣的。
x = torch.arange(20)
y = x[:5]

torch.save([x,y], path)
x_, y_ = torch.load(path)

y_ *= 100

print(x_)


"""
比如在上边的例子中，我们看到y是x的一个前五位的切片，当我们同时保存x和y后，它们的切片关系也被保存了下来，再将他们加载出来，
它们之间依然保留着这个关系，因此可以看到，我们将加载出来的 y_ 乘以100后，x_ 也跟着变化了。
如果不想保留他们的关系，其实也很简单，再保存y之前使用 clone 方法保存一个只有数据的“克隆体”，这样就能只保存数据而不保留关系：
"""


x = torch.arange(20)
y = x[:5]
torch.save([x,y.clone()], path)
x_, y_ = torch.load(path)
y_ *= 100
print(x_)

"""
当我们只保存y而不同时保存x会怎样呢？这样的话确实可以避免如上的情况，即不会再在读取数据后保留他们的关系，但是实际上有一个不容易被看到的影响存在，那就是保存的数据所占用的空间会和其“父亲”级别的数据一样大：

"""

x = torch.arange(1000)
y = x[:5]

path1 = "/home/jack/公共的/MLData/TrashFile/test1.pt"
path2 = "/home/jack/公共的/MLData/TrashFile/test2.pt"

torch.save(y, path1)
torch.save(y.clone(), path2)

y1_ = torch.load(path1)
y2_ = torch.load(path2)

print(y1_.storage().size())
print(y2_.storage().size())


# 1000
# 5


"""

保存与加载模型
保存与加载state_dict
这是一种较为推荐的保存方法，即只保存模型的参数，保存的模型文件会较小，而且比较灵活。但是当加载时，需要先实例化一个模型，然后通过加载将参数赋给这个模型的实例，也就是说加载的时候也需要直到模型的结构。

保存：
torch.save(model.state_dict(), PATH)

加载：
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()

比较重要的点是：

保存模型时调用 state_dict() 获取模型的参数，而不保存结构
加载模型时需要预先实例化一个对应的结构
加载模型使用 load_state_dict 方法，其参数不是文件路径，而是 torch.load(PATH)
如果加载出来的模型用于验证，不要忘了使用 model.eval() 方法，它会丢弃 dropout、normalization 等层，因为这些层不能在inference的时候使用，否则得到的推断结果不一致。

"""

































