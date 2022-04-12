#!/usr/bin/env python3.6
#-*-coding=utf-8-*-

# from ipyvolume import p3


# fig = p3.figure()
# p3.style.use('dark')

# s = p3.quiver(*ds_stream.data,size=6)
# p3.animate_glyphs(s,interval = 200)
# p3.show()



import torch
import torch.utils.data as Data
 

"""

https://zhuanlan.zhihu.com/p/86763381

1. torch.unsqueeze 详解
torch.unsqueeze(input, dim, out=None)

作用：扩展维度
返回一个新的张量，对输入的既定位置插入维度 1

注意： 返回张量与输入张量共享内存，所以改变其中一个的内容会改变另一个。
如果dim为负，则将会被转化dim+input.dim()+1
参数:
tensor (Tensor) – 输入张量
dim (int) – 插入维度的索引
out (Tensor, optional) – 结果张量
"""
import torch

x = torch.Tensor([1, 2, 3, 4])  # torch.Tensor是默认的tensor类型（torch.FlaotTensor）的简称。

print('-' * 50)
print(x)  # tensor([1., 2., 3., 4.])
print(x.size())  # torch.Size([4])
print(x.dim())  # 1
print(x.numpy())  # [1. 2. 3. 4.]

print('-' * 50)
print(torch.unsqueeze(x, 0))  # tensor([[1., 2., 3., 4.]])
print(torch.unsqueeze(x, 0).size())  # torch.Size([1, 4])
print(torch.unsqueeze(x, 0).dim())  # 2
print(torch.unsqueeze(x, 0).numpy())  # [[1. 2. 3. 4.]]

print('-' * 50)
print(torch.unsqueeze(x, 1))
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.]])
print(torch.unsqueeze(x, 1).size())  # torch.Size([4, 1])
print(torch.unsqueeze(x, 1).dim())  # 2

print('-' * 50)
print(torch.unsqueeze(x, -1))
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.]])
print(torch.unsqueeze(x, -1).size())  # torch.Size([4, 1])
print(torch.unsqueeze(x, -1).dim())  # 2

print('-' * 50)
print(torch.unsqueeze(x, -2))  # tensor([[1., 2., 3., 4.]])
print(torch.unsqueeze(x, -2).size())  # torch.Size([1, 4])
print(torch.unsqueeze(x, -2).dim())  # 2

# 边界测试
# 说明：A dim value within the range [-input.dim() - 1, input.dim() + 1) （左闭右开）can be used.
# print('-' * 50)
# print(torch.unsqueeze(x, -3))
# IndexError: Dimension out of range (expected to be in range of [-2, 1], but got -3)

# print('-' * 50)
# print(torch.unsqueeze(x, 2))
# IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)

# 为何取值范围要如此设计呢？
# 原因：方便操作
# 0(-2)-行扩展
# 1(-1)-列扩展
# 正向：我们在0，1位置上扩展
# 逆向：我们在-2，-1位置上扩展
# 维度扩展：1维->2维，2维->3维，...，n维->n+1维
# 维度降低：n维->n-1维，n-1维->n-2维，...，2维->1维

# 以 1维->2维 为例，

# 从【正向】的角度思考：

# torch.Size([4])
# 最初的 tensor([1., 2., 3., 4.]) 是 1维，我们想让它扩展成 2维，那么，可以有两种扩展方式：

# 一种是：扩展成 1行4列 ，即 tensor([[1., 2., 3., 4.]])
# 针对第一种，扩展成 [1, 4]的形式，那么，在 dim=0 的位置上添加 1

# 另一种是：扩展成 4行1列，即
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.]])
# 针对第二种，扩展成 [4, 1]的形式，那么，在dim=1的位置上添加 1

# 从【逆向】的角度思考：
# 原则：一般情况下， "-1" 是代表的是【最后一个元素】
# 在上述的原则下，
# 扩展成[1, 4]的形式，就变成了，在 dim=-2 的的位置上添加 1
# 扩展成[4, 1]的形式，就变成了，在 dim=-1 的的位置上添加 1


"""
2. unsqueeze_和 unsqueeze 的区别
unsqueeze_ 和 unsqueeze 实现一样的功能,区别在于 unsqueeze_ 是 in_place 操作,
即 unsqueeze 不会对使用 unsqueeze 的 tensor 进行改变,想要获取 unsqueeze 后的值必须赋予个新值, 
unsqueeze_ 则会对自己改变。
"""
print("-" * 50)
a = torch.Tensor([1, 2, 3, 4])
print(a)
# tensor([1., 2., 3., 4.])

b = torch.unsqueeze(a, 1)
print(b)
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.]])

print(a)
# tensor([1., 2., 3., 4.])


print("-" * 50)
a = torch.Tensor([1, 2, 3, 4])
print(a)
# tensor([1., 2., 3., 4.])

print(a.unsqueeze_(1))
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.]])

print(a)
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.]])


"""
3. torch.squeeze 详解
作用：降维
torch.squeeze(input, dim=None, out=None)

将输入张量形状中的1 去除并返回。 如果输入是形如(A×1×B×1×C×1×D)，那么输出形状就为： (A×B×C×D)

当给定dim时，那么挤压操作只在给定维度上。例如，输入形状为: (A×1×B), squeeze(input, 0) 将会保持张量不变，只有用 squeeze(input, 1)，形状会变成 (A×B)。

注意： 返回张量与输入张量共享内存，所以改变其中一个的内容会改变另一个。
参数:
input (Tensor) – 输入张量
dim (int, optional) – 如果给定，则input只会在给定维度挤压
out (Tensor, optional) – 输出张量
为何只去掉 1 呢？
多维张量本质上就是一个变换，如果维度是 1 ，那么，1 仅仅起到扩充维度的作用，而没有其他用途，因而，在进行降维操作时，为了加快计算，是可以去掉这些 1 的维度。
"""
print("*" * 50)

m = torch.zeros(2, 1, 2, 1, 2)
print(m.size())  # torch.Size([2, 1, 2, 1, 2])

n = torch.squeeze(m)
print(n.size())  # torch.Size([2, 2, 2])

n = torch.squeeze(m, 0)  # 当给定dim时，那么挤压操作只在给定维度上
print(n.size())  # torch.Size([2, 1, 2, 1, 2])

n = torch.squeeze(m, 1)
print(n.size())  # torch.Size([2, 2, 1, 2])

n = torch.squeeze(m, 2)
print(n.size())  # torch.Size([2, 1, 2, 1, 2])

n = torch.squeeze(m, 3)
print(n.size())  # torch.Size([2, 1, 2, 2])

print("@" * 50)
p = torch.zeros(2, 1, 1)
print(p)
# tensor([[[0.]],
#         [[0.]]])
print(p.numpy())
# [[[0.]]
#  [[0.]]]

print(p.size())
# torch.Size([2, 1, 1])

q = torch.squeeze(p)
print(q)
# tensor([0., 0.])

print(q.numpy())
# [0. 0.]

print(q.size())
# torch.Size([2])


print(torch.zeros(3, 2).numpy())
# [[0. 0.]
#  [0. 0.]
#  [0. 0.]]



#===========================================================================================
x = torch.Tensor([[0,1,2],[3,4,5]])  # torch.Tensor是默认的tensor类型（torch.FlaotTensor）的简称。

print('-' * 50)
print(x)   
print(x.size())  # torch.Size([2,3])
print(x.dim())   # 2
print(x.numpy())   

print('-' * 50)
print(torch.unsqueeze(x, 0))            
print(torch.unsqueeze(x, 0).size())     # torch.Size([1, 2, 3])
print(torch.unsqueeze(x, 0).dim())     # 3
print(torch.unsqueeze(x, 0).numpy())    

print('-' * 50)
print(torch.unsqueeze(x, -3))            
print(torch.unsqueeze(x, -3).size())     # torch.Size([1, 2, 3])
print(torch.unsqueeze(x, -3).dim())     # 3
print(torch.unsqueeze(x, -3).numpy())    


print('-' * 50)
print(torch.unsqueeze(x, 1))
print(torch.unsqueeze(x, 1).size())   #torch.Size([2, 1, 3])
print(torch.unsqueeze(x, 1).dim())    # 3
print(torch.unsqueeze(x, 1).numpy())   

print('-' * 50)
print(torch.unsqueeze(x, -2))            # -2+2+1 = 1
print(torch.unsqueeze(x, -2).size())    #torch.Size([2, 1, 3])
print(torch.unsqueeze(x, -2).dim())     # 2
print(torch.unsqueeze(x, -2).numpy())   


print('-' * 50)
print(torch.unsqueeze(x, 2))
print(torch.unsqueeze(x, 2))   
print(torch.unsqueeze(x, 2).size())     #  torch.Size([2, 3, 1])
print(torch.unsqueeze(x, 2).dim())      # 3
print(torch.unsqueeze(x, 2).numpy())   



print('-' * 50)
print(torch.unsqueeze(x, -1))
print(torch.unsqueeze(x, -1))   
print(torch.unsqueeze(x, -1).size())     #  torch.Size([2, 3, 1])
print(torch.unsqueeze(x, -1).dim())      # 3
print(torch.unsqueeze(x, -1).numpy())   




x = torch.zeros(3,2,4,1,2,1)# dimension of 3*2*4*1*2
print(x.size())             # torch.Size([3, 2, 4, 1, 2, 1])
print(x.shape)
 
y = torch.squeeze(x)        # Returns a tensor with all the dimensions of input of size 1 removed.
print(y.size())             # torch.Size([3, 2, 4, 2])
print(y.shape)
 
z = torch.unsqueeze(y,dim=0)# Add a dimension of 1 in the 0th position
print(z.size())             # torch.Size([1, 3, 2, 4, 2])
print(z.shape)
 
z = torch.unsqueeze(y,dim=1)# Add a dimension of 1 in the 1st position
print(z.size())             # torch.Size([3, 1, 2, 4, 2])
print(z.shape)
 
z = torch.unsqueeze(y,dim=2)# Add a dimension of 1 in the 2nd position
print(z.size())             # torch.Size([3, 2, 1, 4, 2])
print(z.shape)
 
y = torch.squeeze(x,dim=0)  # remove the 0th position of 1 (no 1)
print('dim=0', y.size())    # torch.Size([3, 2, 4, 1, 2, 1])
print('dim=0', y.shape)
 
y = torch.squeeze(x, dim=1)  # remove the 1st position of 1 (no 1)
print('dim=1', y.size())     # torch.Size([3, 2, 4, 1, 2, 1])
print('dim=1', y.shape)
 
y = torch.squeeze(x, dim=2)  # remove the 2nd position of 1 (no 1)
print('dim=2', y.size())     # torch.Size([3, 2, 4, 1, 2, 1])
print('dim=2', y.shape)
 
y = torch.squeeze(x, dim=3)  # remove the 3rd position of 1 (yes)
print('dim=3', y.size())     # torch.Size([3, 2, 4, 2])
print('dim=3', y.shape)
 
y = torch.squeeze(x, dim=4)  # remove the 4th position of 1 (no 1)
print('dim=4', y.size())     # torch.Size([3, 2, 4, 1, 2, 1])
print('dim=4', y.shape)
 
y = torch.squeeze(x, dim=5)  # remove the 5th position of 1 (yes)
print('dim=5', y.size())     # torch.Size([3, 2, 4, 1, 2])
print('dim=5', y.shape)
 
y = torch.squeeze(x, dim=6)  # RuntimeError: Dimension out of range (expected to be in range of [-6, 5], but got 6)
print('dim=6', y.size())
print('dim=6', y.shape)



a=torch.rand(2,3,1)       
print(torch.unsqueeze(a,2).size())#torch.Size([2, 3, 1, 1]) 
print(a.size())         #torch.Size([2, 3, 1])
print(a.squeeze().size())    #torch.Size([2, 3]) 
print(a.squeeze(0).size())   #torch.Size([2, 3, 1])
  
print(a.squeeze(-1).size())   #torch.Size([2, 3])
print(a.size())         #torch.Size([2, 3, 1])
print(a.squeeze(-2).size())   #torch.Size([2, 3, 1])
print(a.squeeze(-3).size())   #torch.Size([2, 3, 1])
print(a.squeeze(1).size())   #torch.Size([2, 3, 1])
print(a.squeeze(2).size())   #torch.Size([2, 3])
print(a.squeeze(3).size())   #RuntimeError: Dimension out of range (expected to be in range of [-3, 2], but got 3)
  
print(a.unsqueeze().size())   #TypeError: unsqueeze() missing 1 required positional arguments: "dim"
print(a.unsqueeze(-3).size())  #torch.Size([2, 1, 3, 1])
print(a.unsqueeze(-2).size())  #torch.Size([2, 3, 1, 1])
print(a.unsqueeze(-1).size())  #torch.Size([2, 3, 1, 1])
print(a.unsqueeze(0).size())  #torch.Size([1, 2, 3, 1])
print(a.unsqueeze(1).size())  #torch.Size([2, 1, 3, 1])
print(a.unsqueeze(2).size())  #torch.Size([2, 3, 1, 1])
print(a.unsqueeze(3).size())  #torch.Size([2, 3, 1, 1])
print(torch.unsqueeze(a,3))
b=torch.rand(2,1,3,1)
print(b.squeeze().size())    #torch.Size([2, 3])



a=torch.randn(768)
print(a.shape) # torch.Size([768])
a=a.unsqueeze(0)
print(a.shape) #torch.Size([1, 768])
a = a.unsqueeze(2)
print(a.shape) #torch.Size([1, 768, 1])



# 也可以直接使用链式编程：
a=torch.randn(768)
print(a.shape) # torch.Size([768])
a=a.unsqueeze(1).unsqueeze(0)
print(a.shape) #torch.Size([1, 768, 1])


a=torch.randn(2,1,768)
print(a)
print(a.shape) #torch.Size([2, 1, 768])
a=a.squeeze()
print(a)
print(a.shape) #torch.Size([2, 768])

#注意的是：squeeze()只能压缩维度为1的维；其他大小的维不起作用。
a=torch.randn(2,768)
print(a.shape) #torch.Size([2, 768])
a=a.squeeze()
print(a.shape) #torch.Size([2, 768])




"""
repeat(*sizes)
沿着指定的维度，对原来的tensor进行数据复制。这个函数和expand()还是有点区别的。expand()只能对维度为1的维进行扩大，而repeat()对所有的维度可以随意操作。
"""
a=torch.randn(2,1,768)
print(a)
print(a.shape) #torch.Size([2, 1, 768])
b=a.repeat(1,2,1)
print(b)
print(b.shape) #torch.Size([2, 2, 768])
c=a.repeat(3,3,3)
print(c)
print(c.shape) #torch.Size([6, 3, 2304])







"""
view()
tensor.view()这个函数有点类似reshape的功能，简单的理解就是：先把一个tensor转换成一个一维的tensor，然后再组合成指定维度的tensor。例如：

"""
word_embedding=torch.randn(16,3,768)
print(word_embedding.shape)
new_word_embedding=word_embedding.view(8,6,768)
print(new_word_embedding.shape)

# 当然这里指定的维度的乘积一定要和原来的tensor的维度乘积相等，不然会报错的。16*3*768=8*6*768
# 另外当我们需要改变一个tensor的维度的时候，知道关键的维度，有不想手动的去计算其他的维度值，就可以使用view(-1)，pytorch就会自动帮你计算出来。 
word_embedding=torch.randn(16,3,768)
print(word_embedding.shape)
new_word_embedding=word_embedding.view(-1)
print(new_word_embedding.shape)
new_word_embedding=word_embedding.view(1,-1)
print(new_word_embedding.shape)
new_word_embedding=word_embedding.view(-1,768)
print(new_word_embedding.shape)





"""
cat(seq,dim,out=None)，表示把两个或者多个tensor拼接起来。

其中 seq表示要连接的两个序列，以元组的形式给出，例如:seq=(a,b), a,b 为两个可以连接的序列

dim 表示以哪个维度连接，dim=0, 横向连接 dim=1,纵向连接

"""

a=torch.randn(4,3)
b=torch.randn(4,3)
  
c=torch.cat((a,b),dim=0)#横向拼接，增加行 torch.Size([8, 3])
print(c.shape)
d=torch.cat((a,b),dim=1)#纵向拼接，增加列 torch.Size([4, 6])
print(d.shape)

#还有一种写法：cat(list,dim,out=None)，其中list中的元素为tensor。
tensors=[]
for i in range(10):
  tensors.append(torch.randn(4,3))
a=torch.cat(tensors,dim=0) #torch.Size([40, 3])
print(a.shape)
b=torch.cat(tensors,dim=1) #torch.Size([4, 30])
print(b.shape)




# torch.max用法
"""
torch.max(input, dim, keepdim=False, out=None) 
输出是： (Tensor, LongTensor)
返回第dim的最大值

torch.max(a,0) ：返回第0维，即每一列中最大值的那个元素，且返回索引（返回最大元素在这一列的行索引）

torch.max(a,1)  返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）


torch.max()[0]， 只返回最大值的每个数

troch.max()[1]， 只返回最大值的每个索引

torch.max()[1].data 只返回variable中的数据部分（去掉Variable containing:）

torch.max()[1].data.numpy() 把数据转化成numpy ndarry

torch.max()[1].data.numpy().squeeze() 把数据条目中维度为1 的删除掉

torch.max(tensor1,tensor2) element-wise 比较tensor1 和tensor2 中的元素，返回较大的那个值
"""




x = torch.rand(4,4)
print('x:\n',x)
print('torch.max(x,1):\n',torch.max(x,1))
print('torch.max(x,0):\n',torch.max(x,0))
print('torch.max(x,1)[0]:\n',torch.max(x,1)[0])
print('torch.max(x,1)[1]:\n',torch.max(x,1)[1])
print('torch.max(x,1)[1].data:\n',torch.max(x,1)[1].data)
print('torch.max(x,1)[1].numpy():\n',torch.max(x,1)[1].numpy())
print('torch.max(x,1)[1].data.numpy():\n',torch.max(x,1)[1].data.numpy())

"""
*注：在有的地方我们会看到torch.max(a, 1).data.numpy()的写法，这是因为在早期的pytorch的版本中，variable变量和tenosr是不一样的数据格式，variable可以进行反向传播，tensor不可以，需要将variable转变成tensor再转变成numpy。现在的版本已经将variable和tenosr合并，所以只用torch.max(a,1).numpy()就可以了。

"""


print('torch.max(x,1)[1].numpy().squeeze():\n',torch.max(x,1)[1].numpy().squeeze())
print('torch.max(x,1)[1].data.numpy().squeeze():\n',torch.max(x,1)[1].data.numpy().squeeze())

print('torch.max(x,1)[0].data:\n',torch.max(x,1)[0].data)

print('torch.max(x,1)[0].numpy():\n',torch.max(x,1)[0].numpy())
print('torch.max(x,1)[0].data.numpy():\n',torch.max(x,1)[0].data.numpy())

print('torch.max(x,1)[0].numpy().squeeze():\n',torch.max(x,1)[0].numpy().squeeze())
print('torch.max(x,1)[0].data.numpy().squeeze():\n',torch.max(x,1)[0].data.numpy().squeeze())





import torch
a = torch.randn(1, 3)
print(f"torch.max(a) = {torch.max(a)}")



import torch
a = torch.tensor([[1, 5, 62, 54], [2, 6, 2, 6], [2, 65, 2, 6]])
print('a:', a,
      '\n\ntorch.max(a):', torch.max(a),
      '\n\ntorch.max(a, 0):', torch.max(a, 0),
      '\n\ntorch.max(a, 0)[0]:', torch.max(a, 0)[0],
      '\n\ntorch.max(a, 0)[1]:', torch.max(a, 0)[1],
      '\n\ntorch.max(a, 0)[1].data:', torch.max(a, 0)[1].data,
      '\n\ntorch.max(a, 0)[1].data.numpy():', torch.max(a, 0)[1].data.numpy(),
      '\n\ntorch.max(a, 0)[1].numpy():', torch.max(a, 0)[1].numpy(),
      '\n\ntorch.max(a, 1):', torch.max(a, 1))



import torch
a = torch.randn(4)
b = torch.randn(4)
c = torch.max(a, b)
print(f"a = {a}\nb = {b}\ntorch.max(a,b) = {torch.max(a,b)}")


import torch
a = torch.arange(12).reshape(3,2,2)
b = (torch.arange(12)*1.1).reshape(3,2,2)
c= torch.max(a, b)
print(f"a = \n{a}\nb = \n{b}\ntorch.max(a,b) = \n{torch.max(a,b)}")


import torch
a = torch.arange(12).reshape(6,1,2)
b = torch.arange(4).reshape(1,2,2)
c= torch.max(a, b)
print(f"a = \n{a}\nb = \n{b}\ntorch.max(a,b) = \n{torch.max(a,b)}")






import torch
a = torch.rand(4, 1, 3)
b = torch.rand(1, 3, 3)
c= torch.max(a, b)
print(f"a = \n{a}\nb = \n{b}\ntorch.max(a,b) = \n{torch.max(a,b)}")




























































































































































































































































































































































