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














































































































































































































































































































































































