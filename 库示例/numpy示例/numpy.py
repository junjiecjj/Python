#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 17:22:52 2022

@author: jack

https://www.runoob.com/numpy/numpy-dtype.html


"""

import numpy as np

print(np.linspace(0.1, 1, 10, endpoint=True))
# np.linspace(start=0.1, stop=1, num=10)
# start：起始数值
# stop：结束数值
# num：观测数值，示例为生成10个观测数值

# 3）三个参数时，第一个参数为起点，第二个参数为终点，第三个参数为步长。其中步长支持小数
print(np.arange(0.1, 1.1, 0.1))

"""总结：
  linspace 侧重于num, 即要产生多少个元素，不在乎增量
  arange 侧重点在于增量，不管产生多少个数
"""

# 等比数列
# numpy.logspace(start=1, stop=1000, num=50, endpoint=True, base=10.0)
# start： array_like
# base ** start：序列的起始值10
# stop： array_like
# base ** stop：序列的最终值10000，除非端点是假的。在这种情况下，num + 1值在log-space 中的间隔上隔开，其中除了最后一个(长度序列数) 被退回。
# endpoint： 布尔值，可选，如果为真，则停止是最后一个样本。否则，不包括在内。默认为真。
# base： 默认值为 10.0。
print(np.logspace(1, 4, 4, endpoint=True, base=2))


# 求二维矩阵某一维的最大值, 分类时常用
x = np.random.randint(low = 0, high = 100, size = (12,4))
y = np.random.randint(low = 0, high = 4, size = (12,))

label1  = np.argmax(x, axis = 1 )
label2 = x.argmax(axis = 1)
label3 = x.max(axis = 1)

print(f"y = \n  {y}")
print(f"label1 = \n  {label1}")
print(f"label2 = \n  {label2}")
print(f"label3 = \n  {label3}")

num = (label1 == y).sum()
print(f"num = \n  {num}")


org = np.arange(20)
print(f"org = {org}")

np.random.shuffle(org)
print(f"org = {org}")

order = np.random.permutation(20)
print(f"order = {order}")

choice = np.random.choice(range(20), 4, replace=False)
print(f"choice = {choice}")

#========================================================================

import torch
y = torch.randint(low = 0, high = 4, size = (12,))
x = torch.randn(size = (12, 4))
label1  = np.argmax(x, axis = 1 )
label2 = x.argmax(axis = 1)
label3 = x.max(axis = 1)

print(f"y = \n  {y}")
print(f"label1 = \n  {label1}")
print(f"label2 = \n  {label2}")
print(f"label3 = \n  {label3}")

num = (x.argmax(axis=1) == y).sum().item()
print(f"num = \n  {num}")

#========================================================================
# 格式：np.argsort(a)
# 注意：返回的是元素值从小到大排序后的索引值的数组

x = np.array([1,4,3,-1,6,9])
print(f"x.argsort() = {x.argsort()}")

# 取数组x的最小值可以写成:
print(f"{x[x.argsort()[0]]}")
# 或者用argmin()函数
print(f"{x[x.argmin()]}")

# 数组x的最大值，写成：
print(x[x.argsort()[-1]])  # -1代表从后往前反向的索引)
print(x[x.argmax()])


#  输出排序后的数组
print(x[x.argsort()])
# 或
print(x[np.argsort(x)])


# (二维数组)
# 沿着行向下(每列)的元素进行排序
# 沿着列向右(每行)的元素进行排序

#=====================================================================================
# NumPy 数据类型
#=====================================================================================


import numpy as np
# int8, int16, int32, int64 四种数据类型可以使用字符串 'i1', 'i2','i4','i8' 代替
dt = np.dtype('i4')
print(dt)

import numpy as np
# 字节顺序标注
dt = np.dtype('<i4')
print(dt)


# 首先创建结构化数据类型
import numpy as np
dt = np.dtype([('age',np.int8)])
print(dt)



# 将数据类型应用于 ndarray 对象
import numpy as np
dt = np.dtype([('age', np.int8)])
a = np.array([(10,), (20,), (30,)], dtype = dt)
print(a)


# 类型字段名可以用于存取实际的 age 列
import numpy as np
dt = np.dtype([('age',np.int8)])
a = np.array([(10,), (20,), (30,)], dtype = dt)
print(a['age'])

#=====================================================================================
# NumPy 高级索引
#=====================================================================================

import numpy as np

a = np.array([[1,2,3], [4,5,6],[7,8,9]])
b = a[1:3, 1:3]
c = a[1:3,[1,2]]
d = a[...,1:]
print(b)
print(c)
print(d)



x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])
print ('我们的数组是：')
print (x)
print ('\n')
# 现在我们会打印出大于 5 的元素
print  ('大于 5 的元素是：')
print (x[x >  5])

import numpy as np

a = np.array([np.nan,  1,2,np.nan,3,4,5])
print (a[~np.isnan(a)])




x=np.arange(32).reshape((8,4))
print (x[[4,2,1,7]])


x=np.arange(32).reshape((8,4))
print (x[[-4,-2,-1,-7]])


x=np.arange(32).reshape((8,4))
print (x[np.ix_([1,5,7,2],[0,3,1,2])])




#=====================================================================================
#  NumPy 广播(Broadcast)
#=====================================================================================

import numpy as np

a = np.array([1,2,3,4])
b = np.array([10,20,30,40])
c = a * b
print (c)


a = np.array([[ 0, 0, 0],
           [10,10,10],
           [20,20,20],
           [30,30,30]])
b = np.array([1,2,3])
print(a + b)



a = np.array([[ 0, 0, 0],
           [10,10,10],
           [20,20,20],
           [30,30,30]])
b = np.array([1,2,3])
bb = np.tile(b, (4, 1))  # 重复 b 的各个维度
print(a + bb)


#=====================================================================================
#  Numpy 数组操作
#=====================================================================================

"""
Numpy 中包含了一些函数用于处理数组，大概可分为以下几类：

修改数组形状
翻转数组
修改数组维度
连接数组
分割数组
数组元素的添加与删除
修改数组形状
函数	描述
reshape	不改变数据的条件下修改形状
flat	数组元素迭代器
flatten	返回一份数组拷贝，对拷贝所做的修改不会影响原始数组
ravel	返回展开数组


"""

a = np.arange(8)
print ('原始数组：')
print (a)
print ('\n')

b = a.reshape(4,2)
print ('修改后的数组：')
print (b)



a = np.arange(9).reshape(3,3)
print ('原始数组：')
for row in a:
    print (row)

#对数组中每个元素都进行处理，可以使用flat属性，该属性是一个数组元素迭代器：
print ('迭代后的数组：')
for element in a.flat:
    print (element)


# numpy.ndarray.flatten 返回一份数组拷贝，对拷贝所做的修改不会影响原始数组，格式如下：
# ndarray.flatten(order='C')
# order：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'K' -- 元素在内存中的出现顺序。

a = np.arange(8).reshape(2,4)

print ('原数组：')
print (a)
print ('\n')
# 默认按行

print ('展开的数组：')
print (a.flatten())
print ('\n')

print ('以 F 风格顺序展开的数组：')
print (a.flatten(order = 'F'))


"""
numpy.ravel() 展平的数组元素，顺序通常是"C风格"，返回的是数组视图（view，有点类似 C/C++引用reference的意味），修改会影响原始数组。
该函数接收两个参数：
numpy.ravel(a, order='C')
参数说明：
order：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'K' -- 元素在内存中的出现顺序。
"""
a = np.arange(8).reshape(2,4)

print ('原数组：')
print (a)
print ('\n')

print ('调用 ravel 函数之后：')
print (a.ravel())
print ('\n')

print ('以 C 风格顺序调用 ravel 函数之后：')
print (a.ravel(order = 'C'))

print ('以 A 风格顺序调用 ravel 函数之后：')
print (a.ravel(order = 'A'))

print ('以 K 风格顺序调用 ravel 函数之后：')
print (a.ravel(order = 'K'))


print ('以 F 风格顺序调用 ravel 函数之后：')
print (a.ravel(order = 'F'))





"""
翻转数组
函数	描述
transpose	对换数组的维度
ndarray.T	和 self.transpose() 相同
rollaxis	向后滚动指定的轴
swapaxes	对换数组的两个轴
numpy.transpose
numpy.transpose 函数用于对换数组的维度，格式如下：

numpy.transpose(arr, axes)
参数说明:
arr：要操作的数组
axes：整数列表，对应维度，通常所有维度都会对换。

numpy.rollaxis
numpy.rollaxis 函数向后滚动特定的轴到一个特定位置，格式如下：
numpy.rollaxis(arr, axis, start)
参数说明：
arr：数组
axis：要向后滚动的轴，其它轴的相对位置不会改变
start：默认为零，表示完整的滚动。会滚动到特定位置。

numpy.swapaxes
numpy.swapaxes 函数用于交换数组的两个轴，格式如下：

numpy.swapaxes(arr, axis1, axis2)
arr：输入的数组
axis1：对应第一个轴的整数
axis2：对应第二个轴的整数

"""

a = np.arange(12).reshape(3,4)

print ('原数组：')
print (a )
print ('\n')

print ('对换数组：')
print (np.transpose(a))



a = np.arange(12).reshape(3,4)
print ('原数组：')
print (a)
print ('\n')

print ('转置数组：')
print (a.T)


# 创建了三维的 ndarray
a = np.arange(8).reshape(2,2,2)

print ('原数组：')
print (a)
print ('获取数组中一个值：')
print(np.where(a==6))
print(a[1,1,0])  # 为 6
print ('\n')


# 将轴 2 滚动到轴 0（宽度到深度）

print ('调用 rollaxis 函数：')
b = np.rollaxis(a,2,0)
print (b)
# 查看元素 a[1,1,0]，即 6 的坐标，变成 [0, 1, 1]
# 最后一个 0 移动到最前面
print(np.where(b==6))
print ('\n')

# 将轴 2 滚动到轴 1：（宽度到高度）

print ('调用 rollaxis 函数：')
c = np.rollaxis(a,2,1)
print (c)
# 查看元素 a[1,1,0]，即 6 的坐标，变成 [1, 0, 1]
# 最后的 0 和 它前面的 1 对换位置
print(np.where(c==6))
print ('\n')



# 创建了三维的 ndarray
a = np.arange(8).reshape(2,2,2)

print ('原数组：')
print (a)
print ('\n')
# 现在交换轴 0（深度方向）到轴 2（宽度方向）

print ('调用 swapaxes 函数后的数组：')
print (np.swapaxes(a, 2, 0))




"""
连接数组
函数	描述
concatenate	连接沿现有轴的数组序列
stack	沿着新的轴加入一系列数组。
hstack	水平堆叠序列中的数组（列方向）
vstack	竖直堆叠序列中的数组（行方向）

numpy.concatenate
numpy.concatenate 函数用于沿指定轴连接相同形状的两个或多个数组，格式如下：
numpy.concatenate((a1, a2, ...), axis)
参数说明：
a1, a2, ...：相同类型的数组
axis：沿着它连接数组的轴，默认为 0

numpy.stack
numpy.stack 函数用于沿新轴连接数组序列，格式如下：
numpy.stack(arrays, axis)
参数说明：
arrays相同形状的数组序列
axis：返回数组中的轴，输入数组沿着它来堆叠


numpy.hstack
numpy.hstack 是 numpy.stack 函数的变体，它通过水平堆叠来生成数组。


numpy.vstack
numpy.vstack 是 numpy.stack 函数的变体，它通过垂直堆叠来生成数组。


"""

a = np.array([[1,2],[3,4]])

print ('第一个数组：')
print (a)
print ('\n')
b = np.array([[5,6],[7,8]])

print ('第二个数组：')
print (b)
print ('\n')
# 两个数组的维度相同

print ('沿轴 0 连接两个数组：')
print (np.concatenate((a,b)))
print ('\n')

print ('沿轴 1 连接两个数组：')
print (np.concatenate((a,b),axis = 1))



a = np.array([[1,2],[3,4]])

print ('第一个数组：')
print (a)
print ('\n')
b = np.array([[5,6],[7,8]])

print ('第二个数组：')
print (b)
print ('\n')

print ('沿轴 0 堆叠两个数组：')
print (np.stack((a,b),0))
print ('\n')

print ('沿轴 1 堆叠两个数组：')
print (np.stack((a,b),1))



a = np.array([[1,2],[3,4]])

print ('第一个数组：')
print (a)
print ('\n')
b = np.array([[5,6],[7,8]])

print ('第二个数组：')
print (b)
print ('\n')

print ('水平堆叠：')
c = np.hstack((a,b))
print (c)
print ('\n')


a = np.array([[1,2],[3,4]])

print ('第一个数组：')
print (a)
print ('\n')
b = np.array([[5,6],[7,8]])

print ('第二个数组：')
print (b)
print ('\n')

print ('竖直堆叠：')
c = np.vstack((a,b))
print (c)



"""
分割数组
函数	数组及操作
split	将一个数组分割为多个子数组
hsplit	将一个数组水平分割为多个子数组（按列）
vsplit	将一个数组垂直分割为多个子数组（按行）
numpy.split
numpy.split 函数沿特定的轴将数组分割为子数组，格式如下：

numpy.split(ary, indices_or_sections, axis)
参数说明：
ary：被分割的数组
indices_or_sections：如果是一个整数，就用该数平均切分，如果是一个数组，为沿轴切分的位置（左开右闭）
axis：设置沿着哪个方向进行切分，默认为 0，横向切分，即水平方向。为 1 时，纵向切分，即竖直方向。

numpy.hsplit
numpy.hsplit 函数用于水平分割数组，通过指定要返回的相同形状的数组数量来拆分原数组。

numpy.vsplit
numpy.vsplit 沿着垂直轴分割，其分割方式与hsplit用法相同。


"""

a = np.arange(9)

print ('第一个数组：')
print (a)
print ('\n')

print ('将数组分为三个大小相等的子数组：')
b = np.split(a,3)
print (b)
print ('\n')

print ('将数组在一维数组中表明的位置分割：')
b = np.split(a,[4,7])
print (b)




a = np.arange(16).reshape(4, 4)
print('第一个数组：')
print(a)
print('\n')
print('默认分割（0轴）：')
b = np.split(a,2)
print(b)
print('\n')

print('沿水平方向分割：')
c = np.split(a,2,1)
print(c)
print('\n')

print('沿水平方向分割：')
d= np.hsplit(a,2)
print(d)



harr = np.floor(10 * np.random.random((2, 6)))
print ('原array：')
print(harr)

print ('拆分后：')
print(np.hsplit(harr, 3))


a = np.arange(16).reshape(4,4)

print ('第一个数组：')
print (a)
print ('\n')

print ('竖直分割：')
b = np.vsplit(a,2)
print (b)





"""
数组元素的添加与删除
函数	元素及描述
resize	返回指定形状的新数组
append	将值添加到数组末尾
insert	沿指定轴将值插入到指定下标之前
delete	删掉某个轴的子数组，并返回删除后的新数组
unique	查找数组内的唯一元素


numpy.resize
numpy.resize 函数返回指定大小的新数组。
如果新数组大小大于原始大小，则包含原始数组中的元素的副本。
numpy.resize(arr, shape)
参数说明：
arr：要修改大小的数组
shape：返回数组的新形状



numpy.append
numpy.append 函数在数组的末尾添加值。 追加操作会分配整个数组，并把原来的数组复制到新数组中。 此外，输入数组的维度必须匹配否则将生成ValueError。
append 函数返回的始终是一个一维数组。
numpy.append(arr, values, axis=None)
参数说明：
arr：输入数组
values：要向arr添加的值，需要和arr形状相同（除了要添加的轴）
axis：默认为 None。当axis无定义时，是横向加成，返回总是为一维数组！当axis有定义的时候，分别为0和1的时候。当axis有定义的时候，分别为0和1的时候（列数要相同）。当axis为1时，数组是加在右边（行数要相同）。



numpy.insert
numpy.insert 函数在给定索引之前，沿给定轴在输入数组中插入值。
如果值的类型转换为要插入，则它与输入数组不同。 插入没有原地的，函数会返回一个新数组。 此外，如果未提供轴，则输入数组会被展开。
numpy.insert(arr, obj, values, axis)
参数说明：
arr：输入数组
obj：在其之前插入值的索引
values：要插入的值
axis：沿着它插入的轴，如果未提供，则输入数组会被展开

numpy.delete
numpy.delete 函数返回从输入数组中删除指定子数组的新数组。 与 insert() 函数的情况一样，如果未提供轴参数，则输入数组将展开。
Numpy.delete(arr, obj, axis)
参数说明：
arr：输入数组
obj：可以被切片，整数或者整数数组，表明要从输入数组删除的子数组
axis：沿着它删除给定子数组的轴，如果未提供，则输入数组会被展开



numpy.unique
numpy.unique 函数用于去除数组中的重复元素。
numpy.unique(arr, return_index, return_inverse, return_counts)
arr：输入数组，如果不是一维数组则会展开
return_index：如果为true，返回新列表元素在旧列表中的位置（下标），并以列表形式储
return_inverse：如果为true，返回旧列表元素在新列表中的位置（下标），并以列表形式储
return_counts：如果为true，返回去重数组中的元素在原数组中的出现次数





"""


a = np.array([[1,2,3],[4,5,6]])

print ('第一个数组：')
print (a)
print ('\n')

print ('第一个数组的形状：')
print (a.shape)
print ('\n')
b = np.resize(a, (3,2))

print ('第二个数组：')
print (b)
print ('\n')

print ('第二个数组的形状：')
print (b.shape)
print ('\n')
# 要注意 a 的第一行在 b 中重复出现，因为尺寸变大了

print ('修改第二个数组的大小：')
b = np.resize(a,(3,3))
print (b)


a = np.array([[1,2,3],[4,5,6]])

print ('第一个数组：')
print (a)
print ('\n')

print ('向数组添加元素：')
print (np.append(a, [7,8,9]))
print ('\n')

print ('沿轴 0 添加元素：')
print (np.append(a, [[7,8,9]],axis = 0))
print ('\n')

print ('沿轴 1 添加元素：')
print (np.append(a, [[5,5,5],[7,8,9]],axis = 1))



a = np.array([[1,2],[3,4],[5,6]])

print ('第一个数组：')
print (a)
print ('\n')

print ('未传递 Axis 参数。 在删除之前输入数组会被展开。')
print (np.insert(a,3,[11,12]))
print ('\n')
print ('传递了 Axis 参数。 会广播值数组来配输入数组。')

print ('沿轴 0 广播：')
print (np.insert(a,1,[11],axis = 0))
print ('\n')

print ('沿轴 1 广播：')
print (np.insert(a,1,11,axis = 1))





a = np.arange(12).reshape(3,4)

print ('第一个数组：')
print (a)
print ('\n')

print ('未传递 Axis 参数。 在插入之前输入数组会被展开。')
print (np.delete(a,5))
print ('\n')

print ('删除第二列：')
print (np.delete(a,1,axis = 1))
print ('\n')

print ('包含从数组中删除的替代值的切片：')
a = np.array([1,2,3,4,5,6,7,8,9,10])
print (np.delete(a, np.s_[::2]))





a = np.array([5,2,6,2,7,5,6,8,2,9])

print ('第一个数组：')
print (a)
print ('\n')

print ('第一个数组的去重值：')
u = np.unique(a)
print (u)
print ('\n')

print ('去重数组的索引数组：')
u,indices = np.unique(a, return_index = True)
print (indices)
print ('\n')

print ('我们可以看到每个和原数组下标对应的数值：')
print (a)
print ('\n')

print ('去重数组的下标：')
u,indices = np.unique(a,return_inverse = True)
print (u)
print ('\n')

print ('下标为：')
print (indices)
print ('\n')

print ('使用下标重构原数组：')
print (u[indices])
print ('\n')

print ('返回去重元素的重复数量：')
u,indices = np.unique(a,return_counts = True)
print (u)
print (indices)





#=====================================================================================
#    NumPy 位运算
#=====================================================================================

"""

NumPy "bitwise_" 开头的函数是位运算函数。

NumPy 位运算包括以下几个函数：

函数	描述
bitwise_and	对数组元素执行位与操作
bitwise_or	对数组元素执行位或操作
invert	按位取反
left_shift	向左移动二进制表示的位
right_shift	向右移动二进制表示的位
注：也可以使用 "&"、 "~"、 "|" 和 "^" 等操作符进行计算。

bitwise_and
bitwise_and() 函数对数组中整数的二进制形式执行位与运算。

bitwise_or
bitwise_or()函数对数组中整数的二进制形式执行位或运算。

invert
invert() 函数对数组中整数进行位取反运算，即 0 变成 1，1 变成 0。

对于有符号整数，取该二进制数的补码，然后 +1。二进制数，最高位为0表示正数，最高位为 1 表示负数。

看看 ~1 的计算步骤：

将1(这里叫：原码)转二进制 ＝ 00000001
按位取反 ＝ 11111110
发现符号位(即最高位)为1(表示负数)，将除符号位之外的其他数字取反 ＝ 10000001
末位加1取其补码 ＝ 10000010
转换回十进制 ＝ -2
表达式
二进制值（2 的补数）

十进制值
5	00000000 00000000 00000000 0000010	5
~5	11111111 11111111 11111111 11111010	-6

left_shift
left_shift() 函数将数组元素的二进制形式向左移动到指定位置，右侧附加相等数量的 0。

right_shift
right_shift() 函数将数组元素的二进制形式向右移动到指定位置，左侧附加相等数量的 0。



"""



print ('13 和 17 的二进制形式：')
a,b = 13,17
print (bin(a), bin(b))
print ('\n')

print ('13 和 17 的位与：')
print (np.bitwise_and(13, 17))


a,b = 13,17
print ('13 和 17 的二进制形式：')
print (bin(a), bin(b))

print ('13 和 17 的位或：')
print (np.bitwise_or(13, 17))



print ('13 的位反转，其中 ndarray 的 dtype 是 uint8：')
print (np.invert(np.array([13], dtype = np.uint8)))
print ('\n')
# 比较 13 和 242 的二进制表示，我们发现了位的反转

print ('13 的二进制表示：')
print (np.binary_repr(13, width = 8))
print ('\n')

print ('242 的二进制表示：')
print (np.binary_repr(242, width = 8))



import numpy as np

print ('将 10 左移两位：')
print (np.left_shift(10,2))
print ('\n')

print ('10 的二进制表示：')
print (np.binary_repr(10, width = 8))
print ('\n')

print ('40 的二进制表示：')
print (np.binary_repr(40, width = 8))
#  '00001010' 中的两位移动到了左边，并在右边添加了两个 0。




print ('将 40 右移两位：')
print (np.right_shift(40,2))
print ('\n')

print ('40 的二进制表示：')
print (np.binary_repr(40, width = 8))
print ('\n')

print ('10 的二进制表示：')
print (np.binary_repr(10, width = 8))
#  '00001010' 中的两位移动到了右边，并在左边添加了两个 0。







#=====================================================================================
#    NumPy 位运算
#=====================================================================================
"""
NumPy 字符串函数
以下函数用于对 dtype 为 numpy.string_ 或 numpy.unicode_ 的数组执行向量化字符串操作。 它们基于 Python 内置库中的标准字符串函数。

这些函数在字符数组类（numpy.char）中定义。

函数	描述
add()	对两个数组的逐个字符串元素进行连接
multiply()	返回按元素多重连接后的字符串
center()	居中字符串
capitalize()	将字符串第一个字母转换为大写
title()	将字符串的每个单词的第一个字母转换为大写
lower()	数组元素转换为小写
upper()	数组元素转换为大写
split()	指定分隔符对字符串进行分割，并返回数组列表
splitlines()	返回元素中的行列表，以换行符分割
strip()	移除元素开头或者结尾处的特定字符
join()	通过指定分隔符来连接数组中的元素
replace()	使用新字符串替换字符串中的所有子字符串
decode()	数组元素依次调用str.decode
encode()	数组元素依次调用str.encode



numpy.char.add()
numpy.char.add() 函数依次对两个数组的元素进行字符串连接。


numpy.char.multiply()
numpy.char.multiply() 函数执行多重连接。



numpy.char.capitalize()
numpy.char.capitalize() 函数将字符串的第一个字母转换为大写：




numpy.char.center()
numpy.char.center() 函数用于将字符串居中，并使用指定字符在左侧和右侧进行填充。



numpy.char.title()
numpy.char.title() 函数将字符串的每个单词的第一个字母转换为大写：

numpy.char.lower()
numpy.char.lower() 函数对数组的每个元素转换为小写。它对每个元素调用 str.lower。

numpy.char.upper()
numpy.char.upper() 函数对数组的每个元素转换为大写。它对每个元素调用 str.upper。


numpy.char.split()
numpy.char.split() 通过指定分隔符对字符串进行分割，并返回数组。默认情况下，分隔符为空格。

numpy.char.splitlines()
numpy.char.splitlines() 函数以换行符作为分隔符来分割字符串，并返回数组。


\n，\r，\r\n 都可用作换行符。

numpy.char.strip()
numpy.char.strip() 函数用于移除开头或结尾处的特定字符。


numpy.char.join()
numpy.char.join() 函数通过指定分隔符来连接数组中的元素或字符串


numpy.char.replace()
numpy.char.replace() 函数使用新字符串替换字符串中的所有子字符串。


numpy.char.encode()
numpy.char.encode() 函数对数组中的每个元素调用 str.encode 函数。 默认编码是 utf-8，可以使用标准 Python 库中的编解码器。

numpy.char.decode()
numpy.char.decode() 函数对编码的元素进行 str.decode() 解码。

"""

print ('连接两个字符串：')
print (np.char.add(['hello'],[' xyz']))
print ('\n')

print ('连接示例：')
print (np.char.add(['hello', 'hi'],[' abc', ' xyz']))




print (np.char.multiply('Runoob ',3))


# np.char.center(str , width,fillchar) ：
# str: 字符串，width: 长度，fillchar: 填充字符
print (np.char.center('Runoob', 20,fillchar = '*'))



print (np.char.capitalize('runoob'))


print (np.char.title('i like runoob'))


#操作数组
print (np.char.lower(['RUNOOB','GOOGLE']))

# 操作字符串
print (np.char.lower('RUNOOB'))

#操作数组
print (np.char.upper(['runoob','google']))

# 操作字符串
print (np.char.upper('runoob'))


# 分隔符默认为空格
print (np.char.split ('i like runoob?'))
# 分隔符为 .
print (np.char.split ('www.runoob.com', sep = '.'))


# 换行符 \n
print (np.char.splitlines('i\nlike runoob?'))
print (np.char.splitlines('i\rlike runoob?'))





# 移除字符串头尾的 a 字符
print (np.char.strip('ashok arunooba','a'))

# 移除数组元素头尾的 a 字符
print (np.char.strip(['arunooba','admin','java'],'a'))


# 操作字符串
print (np.char.join(':','runoob'))

# 指定多个分隔符操作数组元素
print (np.char.join([':','-'],['runoob','google']))


print (np.char.replace ('i like runoob', 'oo', 'cc'))



a = np.char.encode('runoob', 'cp500')
print (a)


a = np.char.encode('runoob', 'cp500')
print (a)
print (np.char.decode(a,'cp500'))



#=====================================================================================
#    NumPy 数学函数
#=====================================================================================
"""
NumPy 包含大量的各种数学运算的函数，包括三角函数，算术运算的函数，复数处理函数等。

三角函数
NumPy 提供了标准的三角函数：sin()、cos()、tan()。

arcsin，arccos，和 arctan 函数返回给定角度的 sin，cos 和 tan 的反三角函数。

这些函数的结果可以通过 numpy.degrees() 函数将弧度转换为角度。


舍入函数
numpy.around() 函数返回指定数字的四舍五入值。
numpy.around(a,decimals)
参数说明：
a: 数组
decimals: 舍入的小数位数。 默认值为0。 如果为负，整数将四舍五入到小数点左侧的位置


numpy.floor()
numpy.floor() 返回小于或者等于指定表达式的最大整数，即向下取整。


numpy.ceil()
numpy.ceil() 返回大于或者等于指定表达式的最小整数，即向上取整。







"""

a = np.array([0,30,45,60,90])
print ('不同角度的正弦值：')
# 通过乘 pi/180 转化为弧度
print (np.sin(a*np.pi/180))
print ('\n')
print ('数组中角度的余弦值：')
print (np.cos(a*np.pi/180))
print ('\n')
print ('数组中角度的正切值：')
print (np.tan(a*np.pi/180))


a = np.array([0,30,45,60,90])
print ('含有正弦值的数组：')
sin = np.sin(a*np.pi/180)
print (sin)
print ('\n')
print ('计算角度的反正弦，返回值以弧度为单位：')
inv = np.arcsin(sin)
print (inv)
print ('\n')
print ('通过转化为角度制来检查结果：')
print (np.degrees(inv))
print ('\n')
print ('arccos 和 arctan 函数行为类似：')
cos = np.cos(a*np.pi/180)
print (cos)
print ('\n')
print ('反余弦：')
inv = np.arccos(cos)
print (inv)
print ('\n')
print ('角度制单位：')
print (np.degrees(inv))
print ('\n')
print ('tan 函数：')
tan = np.tan(a*np.pi/180)
print (tan)
print ('\n')
print ('反正切：')
inv = np.arctan(tan)
print (inv)
print ('\n')
print ('角度制单位：')
print (np.degrees(inv))


import numpy as np

a = np.array([1.0,5.55,  123,  0.567,  25.532])
print  ('原数组：')
print (a)
print ('\n')
print ('舍入后：')
print (np.around(a))
print (np.around(a, decimals =  1))
print (np.around(a, decimals =  -1))


import numpy as np

a = np.array([-1.7,  1.5,  -0.2,  0.6,  10])
print ('提供的数组：')
print (a)
print ('\n')
print ('修改后的数组：')
print (np.floor(a))


import numpy as np

a = np.array([-1.7,  1.5,  -0.2,  0.6,  10])
print  ('提供的数组：')
print (a)
print ('\n')
print ('修改后的数组：')
print (np.ceil(a))


#=====================================================================================
#   NumPy 算术函数
#=====================================================================================

"""
NumPy 算术函数包含简单的加减乘除: add()，subtract()，multiply() 和 divide()。

需要注意的是数组必须具有相同的形状或符合数组广播规则。

numpy.reciprocal()
numpy.reciprocal() 函数返回参数逐元素的倒数。如 1/4 倒数为 4/1。

numpy.power()
numpy.power() 函数将第一个输入数组中的元素作为底数，计算它与第二个输入数组中相应元素的幂。



numpy.mod()
numpy.mod() 计算输入数组中相应元素的相除后的余数。 函数 numpy.remainder() 也产生相同的结果。




"""


import numpy as np

a = np.arange(9, dtype = np.float_).reshape(3,3)
print ('第一个数组：')
print (a)
print ('\n')
print ('第二个数组：')
b = np.array([10,10,10])
print (b)
print ('\n')
print ('两个数组相加：')
print (np.add(a,b))
print ('\n')
print ('两个数组相减：')
print (np.subtract(a,b))
print ('\n')
print ('两个数组相乘：')
print (np.multiply(a,b))
print ('\n')
print ('两个数组相除：')
print (np.divide(a,b))


import numpy as np

a = np.array([0.25,  1.33,  1,  100])
print ('我们的数组是：')
print (a)
print ('\n')
print ('调用 reciprocal 函数：')
print (np.reciprocal(a))




import numpy as np

a = np.array([10,100,1000])
print ('我们的数组是；')
print (a)
print ('\n')
print ('调用 power 函数：')
print (np.power(a,2))
print ('\n')
print ('第二个数组：')
b = np.array([1,2,3])
print (b)
print ('\n')
print ('再次调用 power 函数：')
print (np.power(a,b))





import numpy as np

a = np.array([10,20,30])
b = np.array([3,5,7])
print ('第一个数组：')
print (a)
print ('\n')
print ('第二个数组：')
print (b)
print ('\n')
print ('调用 mod() 函数：')
print (np.mod(a,b))
print ('\n')
print ('调用 remainder() 函数：')
print (np.remainder(a,b))





#=====================================================================================
#    NumPy 统计函数
#=====================================================================================
"""
NumPy 提供了很多统计函数，用于从数组中查找最小元素，最大元素，百分位标准差和方差等。 函数说明如下：

numpy.amin() 和 numpy.amax()
numpy.amin() 用于计算数组中的元素沿指定轴的最小值。

numpy.amax() 用于计算数组中的元素沿指定轴的最大值。


numpy.ptp()
numpy.ptp()函数计算数组中元素最大值与最小值的差（最大值 - 最小值）。




numpy.percentile()
百分位数是统计中使用的度量，表示小于这个值的观察值的百分比。 函数numpy.percentile()接受以下参数。

numpy.percentile(a, q, axis)
参数说明：
a: 输入数组
q: 要计算的百分位数，在 0 ~ 100 之间
axis: 沿着它计算百分位数的轴
首先明确百分位数：
第 p 个百分位数是这样一个值，它使得至少有 p% 的数据项小于或等于这个值，且至少有 (100-p)% 的数据项大于或等于这个值。
举个例子：高等院校的入学考试成绩经常以百分位数的形式报告。比如，假设某个考生在入学考试中的语文部分的原始分数为 54 分。相对于参加同一考试的其他学生来说，他的成绩如何并不容易知道。但是如果原始分数54分恰好对应的是第70百分位数，我们就能知道大约70%的学生的考分比他低，而约30%的学生考分比他高。
这里的 p = 70。



numpy.median()
numpy.median() 函数用于计算数组 a 中元素的中位数（中值）

numpy.mean()
numpy.mean() 函数返回数组中元素的算术平均值。 如果提供了轴，则沿其计算。
算术平均值是沿轴的元素的总和除以元素的数量。


numpy.average()
numpy.average() 函数根据在另一个数组中给出的各自的权重计算数组中元素的加权平均值。
该函数可以接受一个轴参数。 如果没有指定轴，则数组会被展开。
加权平均值即将各数值乘以相应的权数，然后加总求和得到总体值，再除以总的单位数。
考虑数组[1,2,3,4]和相应的权重[4,3,2,1]，通过将相应元素的乘积相加，并将和除以权重的和，来计算加权平均值。

标准差
标准差是一组数据平均值分散程度的一种度量。
标准差是方差的算术平方根。
标准差公式如下：
std = sqrt(mean((x - x.mean())**2))
如果数组是 [1，2，3，4]，则其平均值为 2.5。 因此，差的平方是 [2.25,0.25,0.25,2.25]，并且再求其平均值的平方根除以 4，即 sqrt(5/4) ，结果为 1.1180339887498949。

方差
统计中的方差（样本方差）是每个样本值与全体样本值的平均数之差的平方值的平均数，即 mean((x - x.mean())** 2)。

换句话说，标准差是方差的平方根。

"""

import numpy as np

a = np.array([[3,7,5],[8,4,3],[2,4,9]])
print ('我们的数组是：')
print (a)
print ('\n')
print ('调用 amin() 函数：')
print (np.amin(a,1))
print ('\n')
print ('再次调用 amin() 函数：')
print (np.amin(a,0))
print ('\n')
print ('调用 amax() 函数：')
print (np.amax(a))
print ('\n')
print ('再次调用 amax() 函数：')
print (np.amax(a, axis =  0))


import numpy as np

a = np.array([[3,7,5],[8,4,3],[2,4,9]])
print ('我们的数组是：')
print (a)
print ('\n')
print ('调用 ptp() 函数：')
print (np.ptp(a))
print ('\n')
print ('沿轴 1 调用 ptp() 函数：')
print (np.ptp(a, axis =  1))
print ('\n')
print ('沿轴 0 调用 ptp() 函数：')
print (np.ptp(a, axis =  0))




import numpy as np

a = np.array([[10, 7, 4], [3, 2, 1]])
print ('我们的数组是：')
print (a)

print ('调用 percentile() 函数：')
# 50% 的分位数，就是 a 里排序之后的中位数
print (np.percentile(a, 50))

# axis 为 0，在纵列上求
print (np.percentile(a, 50, axis=0))

# axis 为 1，在横行上求
print (np.percentile(a, 50, axis=1))

# 保持维度不变
print (np.percentile(a, 50, axis=1, keepdims=True))



import numpy as np

a = np.array([[30,65,70],[80,95,10],[50,90,60]])
print ('我们的数组是：')
print (a)
print ('\n')
print ('调用 median() 函数：')
print (np.median(a))
print ('\n')
print ('沿轴 0 调用 median() 函数：')
print (np.median(a, axis =  0))
print ('\n')
print ('沿轴 1 调用 median() 函数：')
print (np.median(a, axis =  1))


import numpy as np

a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print ('我们的数组是：')
print (a)
print ('\n')
print ('调用 mean() 函数：')
print (np.mean(a))
print ('\n')
print ('沿轴 0 调用 mean() 函数：')
print (np.mean(a, axis =  0))
print ('\n')
print ('沿轴 1 调用 mean() 函数：')
print (np.mean(a, axis =  1))




import numpy as np

a = np.array([1,2,3,4])
print ('我们的数组是：')
print (a)
print ('\n')
print ('调用 average() 函数：')
print (np.average(a))
print ('\n')
# 不指定权重时相当于 mean 函数
wts = np.array([4,3,2,1])
print ('再次调用 average() 函数：')
print (np.average(a,weights = wts))
print ('\n')
# 如果 returned 参数设为 true，则返回权重的和
print ('权重的和：')
print (np.average([1,2,3,  4],weights =  [4,3,2,1], returned =  True))

import numpy as np

a = np.arange(6).reshape(3,2)
print ('我们的数组是：')
print (a)
print ('\n')
print ('修改后的数组：')
wt = np.array([3,5])
print (np.average(a, axis =  1, weights = wt))
print ('\n')
print ('修改后的数组：')
print (np.average(a, axis =  1, weights = wt, returned =  True))


import numpy as np

print (np.std([1,2,3,4]))



import numpy as np

print (np.var([1,2,3,4]))




#=====================================================================================
#    NumPy 排序、条件刷选函数
#=====================================================================================

"""

NumPy 提供了多种排序的方法。 这些排序函数实现不同的排序算法，每个排序算法的特征在于执行速度，最坏情况性能，所需的工作空间和算法的稳定性。 下表显示了三种排序算法的比较。

种类	速度	最坏情况	工作空间	稳定性
'quicksort'（快速排序）	1	O(n^2)	0	否
'mergesort'（归并排序）	2	O(n*log(n))	~n/2	是
'heapsort'（堆排序）	3	O(n*log(n))	0	否
numpy.sort()
numpy.sort() 函数返回输入数组的排序副本。函数格式如下：

numpy.sort(a, axis, kind, order)
参数说明：

a: 要排序的数组
axis: 沿着它排序数组的轴，如果没有数组会被展开，沿着最后的轴排序， axis=0 按列排序，axis=1 按行排序
kind: 默认为'quicksort'（快速排序）
order: 如果数组包含字段，则是要排序的字段


numpy.argsort()
numpy.argsort() 函数返回的是数组值从小到大的索引值。

numpy.lexsort()
numpy.lexsort() 用于对多个序列进行排序。把它想象成对电子表格进行排序，每一列代表一个序列，排序时优先照顾靠后的列。
这里举一个应用场景：小升初考试，重点班录取学生按照总成绩录取。在总成绩相同时，数学成绩高的优先录取，在总成绩和数学成绩都相同时，按照英语成绩录取…… 这里，总成绩排在电子表格的最后一列，数学成绩在倒数第二列，英语成绩在倒数第三列。

msort、sort_complex、partition、argpartition
函数	描述
msort(a)	数组按第一个轴排序，返回排序后的数组副本。np.msort(a) 相等于 np.sort(a, axis=0)。
sort_complex(a)	对复数按照先实部后虚部的顺序进行排序。
partition(a, kth[, axis, kind, order])	指定一个数，对数组进行分区
argpartition(a, kth[, axis, kind, order])	可以通过关键字 kind 指定算法沿着指定轴对数组进行分区

numpy.argmax() 和 numpy.argmin()
numpy.argmax() 和 numpy.argmin()函数分别沿给定轴返回最大和最小元素的索引。

numpy.nonzero()
numpy.nonzero() 函数返回输入数组中非零元素的索引。

numpy.where()
numpy.where() 函数返回输入数组中满足给定条件的元素的索引。


numpy.extract()
numpy.extract() 函数根据某个条件从数组中抽取元素，返回满条件的元素。




"""

import numpy as np

a = np.array([[3,7],[9,1]])
print ('我们的数组是：')
print (a)
print ('\n')
print ('调用 sort() 函数：')
print (np.sort(a))
print ('\n')
print ('按列排序：')
print (np.sort(a, axis =  0))
print ('\n')
# 在 sort 函数中排序字段
dt = np.dtype([('name',  'S10'),('age',  int)])
a = np.array([("raju",21),("anil",25),("ravi",  17),  ("amar",27)], dtype = dt)
print ('我们的数组是：')
print (a)
print ('\n')
print ('按 name 排序：')
print (np.sort(a, order =  'name'))


import numpy as np

x = np.array([3,  1,  2])
print ('我们的数组是：')
print (x)
print ('\n')
print ('对 x 调用 argsort() 函数：')
y = np.argsort(x)
print (y)
print ('\n')
print ('以排序后的顺序重构原数组：')
print (x[y])
print ('\n')
print ('使用循环重构原数组：')
for i in y:
    print (x[i], end=" ")


import numpy as np

nm =  ('raju','anil','ravi','amar')
dv =  ('f.y.',  's.y.',  's.y.',  'f.y.')
ind = np.lexsort((dv,nm))
print ('调用 lexsort() 函数：')
print (ind)
print ('\n')
print ('使用这个索引来获取排序后的数据：')
print ([nm[i]  +  ", "  + dv[i]  for i in ind])

import numpy as np
np.sort_complex([5, 3, 6, 2, 1])
np.sort_complex([1 + 2j, 2 - 1j, 3 - 2j, 3 - 3j, 3 + 5j])


a = np.array([3, 4, 2, 1])

np.partition(a, 3)  # 将数组 a 中所有元素（包括重复元素）从小到大排列，3 表示的是排序数组索引为 3 的数字，比该数字小的排在该数字前面，比该数字大的排在该数字的后面
np.partition(a, (1, 3)) # 小于 1 的在前面，大于 3 的在后面，1和3之间的在中间


#找到数组的第 3 小（index=2）的值和第 2 大（index=-2）的值
arr = np.array([46, 57, 23, 39, 1, 10, 0, 120])
arr[np.argpartition(arr, 2)[2]]
arr[np.argpartition(arr, -2)[-2]]

#同时找到第 3 和第 4 小的值。注意这里，用 [2,3] 同时将第 3 和第 4 小的排序好，然后可以分别通过下标 [2] 和 [3] 取得。
arr[np.argpartition(arr, [2,3])[2]]
arr[np.argpartition(arr, [2,3])[3]]


import numpy as np

a = np.array([[30,40,70],[80,20,10],[50,90,60]])
print  ('我们的数组是：')
print (a)
print ('\n')
print ('调用 argmax() 函数：')
print (np.argmax(a))
print ('\n')
print ('展开数组：')
print (a.flatten())
print ('\n')
print ('沿轴 0 的最大值索引：')
maxindex = np.argmax(a, axis =  0)
print (maxindex)
print ('\n')
print ('沿轴 1 的最大值索引：')
maxindex = np.argmax(a, axis =  1)
print (maxindex)
print ('\n')
print ('调用 argmin() 函数：')
minindex = np.argmin(a)
print (minindex)
print ('\n')
print ('展开数组中的最小值：')
print (a.flatten()[minindex])
print ('\n')
print ('沿轴 0 的最小值索引：')
minindex = np.argmin(a, axis =  0)
print (minindex)
print ('\n')
print ('沿轴 1 的最小值索引：')
minindex = np.argmin(a, axis =  1)
print (minindex)



import numpy as np

a = np.array([[30,40,0],[0,20,10],[50,0,60]])
print ('我们的数组是：')
print (a)
print ('\n')
print ('调用 nonzero() 函数：')
print (np.nonzero (a))



import numpy as np

x = np.arange(9.).reshape(3,  3)
print ('我们的数组是：')
print (x)
print ( '大于 3 的元素的索引：')
y = np.where(x >  3)
print (y)
print ('使用这些索引来获取满足条件的元素：')
print (x[y])



import numpy as np

x = np.arange(9.).reshape(3,  3)
print ('我们的数组是：')
print (x)
# 定义条件, 选择偶数元素
condition = np.mod(x,2)  ==  0
print ('按元素的条件值：')
print (condition)
print ('使用条件提取元素：')
print (np.extract(condition, x))


#=====================================================================================
#      NumPy 副本和视图
#=====================================================================================

"""

副本是一个数据的完整的拷贝，如果我们对副本进行修改，它不会影响到原始数据，物理内存不在同一位置。

视图是数据的一个别称或引用，通过该别称或引用亦便可访问、操作原有数据，但原有数据不会产生拷贝。如果我们对视图进行修改，它会影响到原始数据，物理内存在同一位置。

视图一般发生在：

1、numpy 的切片操作返回原数据的视图。
2、调用 ndarray 的 view() 函数产生一个视图。
副本一般发生在：

Python 序列的切片操作，调用deepCopy()函数。
调用 ndarray 的 copy() 函数产生一个副本。
无复制
简单的赋值不会创建数组对象的副本。 相反，它使用原始数组的相同id()来访问它。 id()返回 Python 对象的通用标识符，类似于 C 中的指针。

此外，一个数组的任何变化都反映在另一个数组上。 例如，一个数组的形状改变也会改变另一个数组的形状。

视图或浅拷贝
ndarray.view() 方会创建一个新的数组对象，该方法创建的新数组的维数变化不会改变原始数据的维数。




副本或深拷贝
ndarray.copy() 函数创建一个副本。 对副本数据进行修改，不会影响到原始数据，它们物理内存不在同一位置。
"""

import numpy as np

a = np.arange(6)
print ('我们的数组是：')
print (a)
print ('调用 id() 函数：')
print (id(a))
print ('a 赋值给 b：')
b = a
print (b)
print ('b 拥有相同 id()：')
print (id(b))
print ("b 是 a 吗？ {}".format(b is a))
print(id(a.storage())==id(b.storage()))
print ('修改 b 的形状：')
b.shape =  3,2
print (b)
print ('a 的形状也修改了：')
print (a)
# 改变b，a也会改变
b[0,1] = 33
print ('a 的形状：')
print (a)
print ('b 的形状：')
print (b)



import numpy as np
#ndarray.view() 方会创建一个新的数组对象，该方法创建的新数组的维数变化不会改变原始数据的维数。
# 最开始 a 是个 3X2 的数组
a = np.arange(6).reshape(3,2)
print ('数组 a：')
print (a)
print ('创建 a 的视图：')
b = a.view()
print (b)
print ('两个数组的 id() 不同：')
print ('a 的 id()：')
print (id(a))
print ('b 的 id()：' )
print (id(b))
print ("b 是 a 吗？ {}".format(b is a))
# 修改 b 的形状，并不会修改 a
b.shape =  2,3
print ('b 的形状：')
print (b)
print ('a 的形状：')
print (a)
# 改变b，a也会改变
b[0,1] = 33
print ('a 的形状：')
print (a)
print ('b 的形状：')
print (b)


#使用切片创建视图修改数据会影响到原始数组：
import numpy as np

arr = np.arange(12)
print ('我们的数组：')
print (arr)
print ('创建切片：')
a=arr[3:]
b=arr[3:]
a[1]=123
b[2]=234
print(arr)
print(id(a),id(b),id(arr[3:]))
#变量 a,b 都是 arr 的一部分视图，对视图的修改会直接反映到原数据中。但是我们观察 a,b 的 id，他们是不同的，也就是说，视图虽然指向原数据，但是他们和赋值引用还是有区别的。


import numpy as np

a = np.array([[10,10],  [2,3],  [4,5]])
print ('数组 a：')
print (a)
print ('创建 a 的深层副本：')
b = a.copy()
print ('数组 b：')
print (b)
print ("b 是 a 吗？ {}".format(b is a))
# b 与 a 不共享任何内容
print ('我们能够写入 b 来写入 a 吗？')

print ('修改 b 的内容：')
b[0,0]  =  100
print ('修改后的数组 b：')
print (b)
print ('a 保持不变：')
print (a)



import numpy as np

a = np.array([[10,10],  [2,3],  [4,5]])
print ('数组 a：')
print (a)
print ('创建 b：')
b = a.reshape((2,3))
print ('数组 b：')
print (b)
# b 与 a 不共享任何内容
print ('我们能够写入 b 来写入 a 吗？')
print (b is a)
print ('修改 b 的内容：')
b[0,0]  =  100
print ('修改后的数组 b：')
print (b)
print ('a 保持不变：')
print (a)

import numpy as np

a = np.arange(12)
print ('数组 a：')
print (a)
print ('创建 b = a.resize(3,4)：')
b = a.resize(3,4)
print ('数组 b：')
print (b)
# b 与 a 不共享任何内容
print ('我们能够写入 b 来写入 a 吗？')
print (b is a)
print ('修改 b 的内容：')
b[0,0]  =  100
print ('修改后的数组 b：')
print (b)
print ('a 保持不变：')
print (a)




#=====================================================================================
#      NumPy 矩阵库(Matrix)
#=====================================================================================
"""
matlib.empty()
matlib.empty() 函数返回一个新的矩阵，语法格式为：

numpy.matlib.empty(shape, dtype, order)
参数说明：

shape: 定义新矩阵形状的整数或整数元组
Dtype: 可选，数据类型
order: C（行序优先） 或者 F（列序优先）


numpy.matlib.zeros()
numpy.matlib.zeros() 函数创建一个以 0 填充的矩阵。


numpy.matlib.ones()
numpy.matlib.ones()函数创建一个以 1 填充的矩阵。


numpy.matlib.eye()
numpy.matlib.eye() 函数返回一个矩阵，对角线元素为 1，其他位置为零。
numpy.matlib.eye(n, M,k, dtype)
参数说明：
n: 返回矩阵的行数
M: 返回矩阵的列数，默认为 n
k: 对角线的索引
dtype: 数据类型


numpy.matlib.identity()
numpy.matlib.identity() 函数返回给定大小的单位矩阵。
单位矩阵是个方阵，从左上角到右下角的对角线（称为主对角线）上的元素均为 1，除此以外全都为 0。

numpy.matlib.rand()
numpy.matlib.rand() 函数创建一个给定大小的矩阵，数据是随机填充的。



"""

import numpy as np

a = np.arange(12).reshape(3,4)

print ('原数组：')
print (a)
print ('\n')

print ('转置数组：')
print (a.T)


import numpy.matlib
import numpy as np

print (np.matlib.empty((2,2)))
# 填充为随机数据


import numpy.matlib
import numpy as np

print (np.matlib.zeros((2,2)))




import numpy.matlib
import numpy as np

print (np.matlib.ones((2,2)))



import numpy.matlib
import numpy as np

print (np.matlib.eye(n =  3, M =  4, k =  0, dtype =  float))






import numpy.matlib
import numpy as np

# 大小为 5，类型位浮点型
print (np.matlib.identity(5, dtype =  float))




import numpy.matlib
import numpy as np

print (np.matlib.rand(3,3))


import numpy.matlib
import numpy as np

i = np.matrix('1,2;3,4')
print (i)



import numpy.matlib
import numpy as np

j = np.asarray(i)
print (j)


import numpy.matlib
import numpy as np

k = np.asmatrix(j)
print (k)







#=====================================================================================
#      NumPy 线性代数
#=====================================================================================
"""
NumPy 提供了线性代数函数库 linalg，该库包含了线性代数所需的所有功能，可以看看下面的说明：

函数	描述
dot	两个数组的点积，即元素对应相乘。
vdot	两个向量的点积
inner	两个数组的内积
matmul	两个数组的矩阵积
determinant	数组的行列式
solve	求解线性矩阵方程
inv	计算矩阵的乘法逆矩阵
numpy.dot()
numpy.dot() 对于两个一维的数组，计算的是这两个数组对应下标元素的乘积和(数学上称之为内积)；对于二维数组，计算的是两个数组的矩阵乘积；对于多维数组，它的通用计算公式如下，即结果数组中的每个元素都是：数组a的最后一维上的所有元素与数组b的倒数第二位上的所有元素的乘积和： dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])。

numpy.dot(a, b, out=None)
参数说明：
a : ndarray 数组
b : ndarray 数组
out : ndarray, 可选，用来保存dot()的计算结果


numpy.vdot()
numpy.vdot() 函数是两个向量的点积。 如果第一个参数是复数，那么它的共轭复数会用于计算。 如果参数是多维数组，它会被展开。

numpy.inner()
numpy.inner() 函数返回一维数组的向量内积。对于更高的维度，它返回最后一个轴上的和的乘积。



numpy.matmul
numpy.matmul 函数返回两个数组的矩阵乘积。 虽然它返回二维数组的正常乘积，但如果任一参数的维数大于2，则将其视为存在于最后两个索引的矩阵的栈，并进行相应广播。
另一方面，如果任一参数是一维数组，则通过在其维度上附加 1 来将其提升为矩阵，并在乘法之后被去除。
对于二维数组，它就是矩阵乘法：




numpy.linalg.det()
numpy.linalg.det() 函数计算输入矩阵的行列式。
行列式在线性代数中是非常有用的值。 它从方阵的对角元素计算。 对于 2×2 矩阵，它是左上和右下元素的乘积与其他两个的乘积的差。
换句话说，对于矩阵[[a，b]，[c，d]]，行列式计算为 ad-bc。 较大的方阵被认为是 2×2 矩阵的组合。


numpy.linalg.solve()
numpy.linalg.solve() 函数给出了矩阵形式的线性方程的解。

考虑以下线性方程：
x + y + z = 6
2y + 5z = -4
2x + 5y - z = 27


numpy.linalg.inv()
numpy.linalg.inv() 函数计算矩阵的乘法逆矩阵。

逆矩阵（inverse matrix）：设A是数域上的一个n阶矩阵，若在相同数域上存在另一个n阶矩阵B，使得： AB=BA=E ，则我们称B是A的逆矩阵，而A则被称为可逆矩阵。注：E为单位矩阵。




"""



#向量是一维矩阵，两个向量进行内积运算时，需要保证两个向量包含的元素个数是相同的。
#例：

import numpy as np

x = np.array([1,2,3,4,5,6,7])
y = np.array([2,3,4,5,6,7,8])
result = np.dot(x, y)
print(f"result = {result}")


#2、矩阵乘法运算
#两个矩阵（x,y）如果可以进行乘法运算，需要满足以下条件：
#x为mxn阶矩阵，y为nxp阶矩阵，则相乘的结果result为mxp阶矩阵。
#例：

import numpy as np

x = np.array([[1,2,3],[4,5,6]])
y = np.array([[2,3],[4,5],[6,7]])
result = np.dot(x, y)
print(result)
print("x阶数：" + str(x.shape))
print("y阶数：" + str(y.shape))
print("result阶数：" + str(result.shape))


#3、矩阵与向量乘法
#矩阵x为mxn阶，向量y为n阶向量，则矩阵x和向量y可以进行乘法运算，结果为m阶向量。进行运算时，会首先将后面一项进行自动转置操作，之后再进行乘法运算。

import numpy as np

x = np.array([[1,2,3],[4,5,6]])
y = np.array([1,2,3])
result = np.dot(x, y)
print(result)
print("x阶数：" + str(x.shape))
print("y阶数：" + str(y.shape))
print("result阶数：" + str(result.shape))


import numpy as np

x = np.array([[1, 2, 3],
   [3, 4, 4],
   [0, 1, 1]])
y = np.array([1, 2, 3])
result1 = np.dot(x, y) # 1×1 + 2×2 + 3×3 = 14（result1的第一个元素）
result2 = np.dot(y, x) # 1×1 + 2×3 + 3×0 = 7 （result2的第一个元素）

print("result1 = " + str(result1))
print("result2 = " + str(result2))


import numpy.matlib
import numpy as np

a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])
print(np.dot(a,b))



import numpy as np

a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])

# vdot 将数组展开计算内积
print (np.vdot(a,b))




import numpy as np

print (np.inner(np.array([1,2,3]),np.array([0,1,0])))
# 等价于 1*0+2*1+3*0



import numpy as np
a = np.array([[1,2], [3,4]])

print ('数组 a：')
print (a)
b = np.array([[11, 12], [13, 14]])

print ('数组 b：')
print (b)

print ('内积：')
print (np.inner(a,b))


import numpy.matlib
import numpy as np

a = [[1,0],[0,1]]
b = [[4,1],[2,2]]
print (np.matmul(a,b))



import numpy.matlib
import numpy as np

a = [[1,0],[0,1]]
b = [1,2]
print (np.matmul(a,b))
print (np.matmul(b,a))






import numpy.matlib
import numpy as np

a = np.arange(8).reshape(2,2,2)
b = np.arange(4).reshape(2,2)
print (np.matmul(a,b))




import numpy as np
a = np.array([[1,2], [3,4]])
print (np.linalg.det(a))



import numpy as np

b = np.array([[6,1,1], [4, -2, 5], [2,8,7]])
print (b)
print (np.linalg.det(b))
print (6*(-2*7 - 5*8) - 1*(4*7 - 5*2) + 1*(4*8 - -2*2))




import numpy as np

x = np.array([[1,2],[3,4]])
y = np.linalg.inv(x)
print (x)
print (y)
print (np.dot(x,y))



import numpy as np

a = np.array([[1,1,1],[0,2,5],[2,5,-1]])

print ('数组 a：')
print (a)
ainv = np.linalg.inv(a)

print ('a 的逆：')
print (ainv)

print ('矩阵 b：')
b = np.array([[6],[-4],[27]])
print (b)

print ('计算：A^(-1)B：')
x = np.linalg.solve(a,b)
print (x)
# 这就是线性方向 x = 5, y = 3, z = -2 的解


x = np.dot(ainv,b)
print (x)

"""
矩阵的秩定义
假设向量组A的最大无关组为：
A0 = { a 1 , a 2 , ⋯   , a r }
A0 的向量个数r称为向量组A的秩，记做rank(A)，有时也记作r(A)。

只含零向量的向量组没有最大无关组，规定它的秩为0。
"""

import numpy as np
from numpy.linalg import matrix_rank
a1 = np.array([[1,1],[2,3]])
a2 = np.array([0.0])
a3 = np.array([[1,1,1],[1,2,0],[2,3,1]])

rank1 = matrix_rank(a1)
rank2 = matrix_rank(a2)
rank3 = matrix_rank(a3)

print(a1)
print(rank1)
print(a2)
print(rank2)
print(a3)
print(rank3)




#=====================================================================================
#    NumPy Matplotlib
#=====================================================================================
"""

作为线性图的替代，可以通过向 plot() 函数添加格式字符串来显示离散值。 可以使用以下格式化字符。

字符	描述
'-'	实线样式
'--'	短横线样式
'-.'	点划线样式
':'	虚线样式
'.'	点标记
','	像素标记
'o'	圆标记
'v'	倒三角标记
'^'	正三角标记
'&lt;'	左三角标记
'&gt;'	右三角标记
'1'	下箭头标记
'2'	上箭头标记
'3'	左箭头标记
'4'	右箭头标记
's'	正方形标记
'p'	五边形标记
'*'	星形标记
'h'	六边形标记 1
'H'	六边形标记 2
'+'	加号标记
'x'	X 标记
'D'	菱形标记
'd'	窄菱形标记
'&#124;'	竖直线标记
'_'	水平线标记
以下是颜色的缩写：

字符	颜色
'b'	蓝色
'g'	绿色
'r'	红色
'c'	青色
'm'	品红色
'y'	黄色
'k'	黑色
'w'	白色
要显示圆来代表点，而不是上面示例中的线，请使用 ob 作为 plot() 函数中的格式字符串。


subplot()
subplot() 函数允许你在同一图中绘制不同的东西。

以下实例绘制正弦和余弦值:



bar()
pyplot 子模块提供 bar() 函数来生成条形图。

以下实例生成两组 x 和 y 数组的条形图。

"""

import numpy as np
from matplotlib import pyplot as plt

x = np.arange(1,11)
y =  2  * x +  5
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x,y)
plt.show()


import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# fname 为 你下载的字体库路径，注意 SourceHanSansSC-Bold.otf 字体的路径
zhfont1 = matplotlib.font_manager.FontProperties(fname="SourceHanSansSC-Bold.otf")

x = np.arange(1,11)
y =  2  * x +  5
plt.title("菜鸟教程 - 测试", fontproperties=zhfont1)

# fontproperties 设置中文显示，fontsize 设置字体大小
plt.xlabel("x 轴", fontproperties=zhfont1)
plt.ylabel("y 轴", fontproperties=zhfont1)
plt.plot(x,y)
plt.show()



import numpy as np
from matplotlib import pyplot as plt

x = np.arange(1,11)
y =  2  * x +  5
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x,y,"ob")
plt.show()



import numpy as np
import matplotlib.pyplot as plt
# 计算正弦曲线上点的 x 和 y 坐标
x = np.arange(0,  3  * np.pi,  0.1)
y = np.sin(x)
plt.title("sine wave form")
# 使用 matplotlib 来绘制点
plt.plot(x, y)
plt.show()




import numpy as np
import matplotlib.pyplot as plt
# 计算正弦和余弦曲线上的点的 x 和 y 坐标
x = np.arange(0,  3  * np.pi,  0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
# 建立 subplot 网格，高为 2，宽为 1
# 激活第一个 subplot
plt.subplot(2,  1,  1)
# 绘制第一个图像
plt.plot(x, y_sin)
plt.title('Sine')
# 将第二个 subplot 激活，并绘制第二个图像
plt.subplot(2,  1,  2)
plt.plot(x, y_cos)
plt.title('Cosine')
# 展示图像
plt.show()



from matplotlib import pyplot as plt
x =  [5,8,10]
y =  [12,16,6]
x2 =  [6,9,11]
y2 =  [6,15,7]
plt.bar(x, y, align =  'center')
plt.bar(x2, y2, color =  'g', align =  'center')
plt.title('Bar graph')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()



import numpy as np

a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
np.histogram(a,bins =  [0,20,40,60,80,100])
hist,bins = np.histogram(a,bins =  [0,20,40,60,80,100])
print (hist)
print (bins)



from matplotlib import pyplot as plt
import numpy as np

a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
plt.hist(a, bins =  [0,20,40,60,80,100])
plt.title("histogram")
plt.show()



#=====================================================================================
#   NumPy  矩阵
#=====================================================================================


data = np.arange(25).reshape(5,5)

#k=0表示正常的上三角矩阵
upper_triangle = np.triu(data, 0)
"""
[[1 2 3 4 5]
 [0 5 6 7 8]
 [0 0 7 8 9]
 [0 0 0 7 8]
 [0 0 0 0 5]]
"""

#k=-1表示对角线的位置下移1个对角线
upper_triangle = np.triu(data, -1)
"""
[[1 2 3 4 5]
 [4 5 6 7 8]
 [0 7 7 8 9]
 [0 0 6 7 8]
 [0 0 0 4 5]]
"""

#k=1表示对角线的位置上移1个对角线
upper_triangle = np.triu(data, 1)
"""
[[0 2 3 4 5]
 [0 0 6 7 8]
 [0 0 0 8 9]
 [0 0 0 0 8]
 [0 0 0 0 0]]
"""


lower_triangle = np.tril(data, 0)
"""
[[1 0 0 0 0]
 [4 5 0 0 0]
 [6 7 7 0 0]
 [4 5 6 7 0]
 [1 2 3 4 5]]
"""
lower_triangle = np.tril(data, -1)
"""
[[0 0 0 0 0]
 [4 0 0 0 0]
 [6 7 0 0 0]
 [4 5 6 0 0]
 [1 2 3 4 0]]
"""
lower_triangle = np.tril(data, 1)
"""
[[1 2 0 0 0]
 [4 5 6 0 0]
 [6 7 7 8 0]
 [4 5 6 7 8]
 [1 2 3 4 5]]
"""







































































































































































































































































































































































































































































































































































































































































































































































































































































