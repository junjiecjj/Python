#!/usr/bin/env python3
#!-*-coding=utf-8-*-
#########################################################################
# File Name: numpy.py
# Author: 陈俊杰
# Created Time: 2021年12月12日 星期日 15时52分24秒

# mail: 2716705056@qq.com
#此程序的功能是：

#########################################################################
import pandas as pd
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import os, time





"""
#=========================================================
#numpy    array
#============================================================
"""
List = np.arange(0, 1, 0.1)


A = np.array([[1,2,3],[4,5,6]])
B = np.array([[1,2],[3,4],[5,6]])

M1 = np.ones((3, 2))
print("M1.shape = \n",M1.shape)

M2 = np.zeros((2, 3))
print("M2.shape = \n",M2.shape)

M3 = np.eye(3)#3维单位矩阵
print("M3.shape = \n",M3.shape)

y = np.array([4, 5, 6])
print("y.shape = \n",y.shape)

y1 = np.array([[4, 5, 6]])
print("y1.shape = \n",y1.shape)

M4 = np.diag(y)#以y为主对角线创建矩阵
print("M4.shape = \n",M4.shape)

M5 = np.random.randint(0, 10, (4,3))
print("M5.shape = \n",M5.shape)

a =  np.arange(7)

# 每间隔2个取一个数
print(a[ : 7: 2])


b = np.arange(24).reshape(4,6)

print(b,'\n\n',b.T)

print(b.reshape(6,4))


#函数resize（）的作用跟reshape（）类似，但是会改变所作用的数组，相当于有inplace=True的效果
b.resize(6,4)

print(b)


"""
ravel()和flatten()，将多维数组转换成一维数组，如下：
两者的区别在于返回拷贝（copy）还是返回视图（view），flatten()返回一份拷贝，需要分配新的内存空间，对拷贝所做的修改不会影响原始矩阵，而ravel()返回的是视图（view），会影响原始矩阵。
"""

b = np.arange(12).reshape(3,4)
print("b = \n{}".format(b ))

print("b.ravel() = \n{}".format(b.ravel()))
print("b.flatten() = \n{}".format(b.flatten()))

c = b.flatten()
d  = b.ravel()
print("b = \n{},\nc= \n{},\nd= \n{}".format(b,c,d ))

c[1]=121
print("b = \n{},\nc= \n{},\nd= \n{}".format(b,c,d ))

d[2]=899
print("b = \n{},\nc= \n{},\nd= \n{}".format(b,c,d ))

b = np.array([[ 0,  1, 20,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11]])

c = b*2

print("np.hstack((b,c)) = \n{}".format(np.hstack((b,c))))#水平叠加

print("np.vstack((b,c)) = \n{}".format(np.vstack((b,c))))#垂直叠加


print("np.concatenate((b,c),axis=1) = \n{}".format(np.concatenate((b,c),axis=1)))  # axis=1时，沿水平方向叠加

print("np.concatenate((b,c),axis=0) = \n{}".format(np.concatenate((b,c),axis=0)))  # axis=0时，沿垂直方向叠加



#跟数组的叠加类似，数组的拆分可以分为横向拆分、纵向拆分以及深度拆分。涉及的函数为 hsplit()、vsplit()、dsplit() 以及split()
##沿横向轴拆分（axis=1）
print("np.hsplit(b, 2) = \n{}".format(np.hsplit(b, 2)))
print("np.split(b,2, axis=1) = \n{}".format(np.split(b,2, axis=1)))  #沿横向轴拆分（axis=1）


#沿纵向轴拆分（axis=0）
print("np.vsplit(b, 2) = \n{}".format(np.vsplit(b, 2)))
print("np.split(b,2,axis=0) = \n{}".format(np.split(b,2,axis=0)))  #沿横向轴拆分（axis=1）


"""
常用的函数如下：

请注意函数在使用时需要指定axis轴的方向，若不指定，默认统计整个数组。

np.sum()，返回求和
np.mean()，返回均值
np.max()，返回最大值
np.min()，返回最小值
np.ptp()，数组沿指定轴返回最大值减去最小值，即（max-min）
np.std()，返回标准偏差（standard deviation）
np.var()，返回方差（variance）
np.cumsum()，返回累加值
np.cumprod()，返回累乘积值

"""
print("np.max(b) =\n{}".format(np.max(b)))


# 沿axis=1轴方向统计
np.max(b,axis=1)

# 沿axis=0轴方向统计
np.max(b,axis=0)

#np.ptp()，返回整个数组的最大值减去最小值，如下：
# 沿axis=0轴方向
np.ptp(b, axis=0)

# 沿axis=1轴方向
np.ptp(b, axis=1)

#np.cumsum()，沿指定轴方向进行累加
b.resize(4,3)
np.cumsum(b, axis=1)

np.cumsum(b, axis=0)


#np.cumprod()，沿指定轴方向进行累乘积

np.cumprod(b,axis=1)

np.cumprod(b,axis=0)



#==========================================================
#np.array 直接创建，接受list或tuple参数
#==========================================================
print(np.array([1,2,3]))
print("\n")
print(np.array([[1,2,3],[4,5,6]]))
print("\n")
print(np.array((1,2,3)))

#==========================================================
# np.arange 创建等差矩阵或者向量
#==========================================================
print("#========== np.arange 创建等差矩阵或者向量 =======================\n")
print(np.arange(1, 10, 2)) 	#参数为：(起始点，终点，步长)；注意！ 不包括终点
print("\n")
print(np.arange(20))  		# 参数为一个值，直接定义。 0-19，不包涵20.步长默认为一
print("\n")
#第二种用法可以联系python的range记忆，功能和拼写类似。
for i in range(10):
	print(i)

## ========================================================
#np.zeros 创建全部为零的矩阵或者向量
#==========================================================
print("#========== np.zeros 创建全部为零的矩阵或者向量 =======================\n")
#用tuple表示形状
print(np.zeros((3,3)))

## ========================================================
#np.ones创建全部为一的矩阵或者向量
#==========================================================
print("#========== #np.ones创建全部为一的矩阵或者向量 =======================\n")
print(np.ones((3,3)))
print("\n")
# 利用dtype属性直接创建相应类型的矩阵
print(np.zeros((3,4), dtype = bool))
print("\n")
print(np.ones((3,4), dtype = bool))
print("\n")

#==========================================================
#np.random.random 产生0-1 之间的随机数（不包含1）
#==========================================================
print("#========== np.random.random 产生0-1 之间的随机数（不包含1） =======================\n")
#参数为一个整形：生成包含3个0-1之间的数的向量
print(np.random.random(3))
print("\n")
#参数为tuple
print(np.random.random((2,3)))   #产生特定形状的0-1之间的矩阵,这里参数用list也可以但是不推荐。


#==========================================================
# np.random.rand 功能同random.random 但是参数有差别
#==========================================================
print("#========== np.random.rand 功能同random.random 但是参数有差别 =======================\n")
#rank 直接接受int型参数表示shape , random接受list或tuple
print(np.random.rand(3,5))
print("\n")
#print(np.random.rand([3,5])) # 报错
#print(np.random.rand((3,5))) # 报错
#print(np.random.random(3,4)) # 报错


#==========================================================
# np.linspace 创建等差向量或矩阵
#==========================================================
print("#========== np.linspace 创建等差向量或矩阵 =======================\n")
#参数：(起始点，终点，分割份数) 注意！包括终点
print(np.linspace(0, 10, 5))  # => [ 0. ,  2.5,  5. ,  7.5, 10. ]
#注意！与arange的区别， arange定义步长，linspace定义等差元素的个数。
print("\n")

#==========================================================
# np.logspace 创建等差指数向量或矩阵
#==========================================================
print("#========== np.logspace 创建等差指数向量或矩阵 =======================\n")
#logspace(start, stop, num, endpoint=True, base=10.0, dtype=None, axis=0,) 创建base为底的等距元素个数的指数矩阵， 前三个参数与linspace()效果一致。
# 创建以 2 为底，1-10产生等差数列包含10个值，那么这个数列就是1到10，这十个值将会成为2的指数生成向量：
print(np.logspace(1, 10, 10, base = 2))
print("\n")
# 1-10产生等差数列包含3个值： 1 5.5 10 ，这三个值将会成为2的指数生成向量。
print(np.logspace(1, 10, 3, base = 2))
print("\n")
# 1-10产生的等差数列包含四个值： 1 4 7 10，这四个值将会成为2的指数生成向量。
print(np.logspace(1, 10, 4, base = 2))
print("\n")

#==========================================================
# np.diag 获取或创建对角线向量
#==========================================================
# diag(v,k=0) 创建对角线数组 v 只能是一维或二维矩阵
# 如果v是一维返回以v为对角线的二维数组
# 如果v是二维返回k位置的对角线数组
# v是一维
print("#========== np.diag 获取或创建对角线向量 =======================\n")
print(np.diag([1,2,3,4]))
print("\n")
print(np.diag((1,2,3,4)))
print("\n")
# 对角线偏移 k>0 向右偏反之向左偏，偏移之后会保证[1,2,3,4]完整
print(np.diag([1,2,3,4], k = 1)) # 主对角线偏移 1 的矩阵
print("\n")
print(np.diag([1,2,3,4], k = -1)) # 主对角线偏移 1 的矩
print("\n")

# v是二维
a = np.arange(1, 10, 1).reshape(3,3)
print(a)
print("\n")
print(np.diag(a)) 			#返回主对角线向量
print("\n")
print(np.diag(a, k = 1))	#返回右偏移1的对角线向量
print("\n")
# v 是三维 报错
a = np.arange(1, 19, 1).reshape(3,3,2)
print(a)
#print(np.diag(a))  #ValueError: Input must be 1- or 2-d
print("\n")





"""#=========================================================
#numpy    Matrix
#============================================================"""

print("#========== numpy    Matrix =======================\n")
# https://www.92python.com/view/93.html
"""

NumPy 提供了两个基本的对象：一个多维数组（ndarray）对象和一个通用函数（ufunc）对象，其他对象都是在它们的基础上构建的。在 NumPy 中还包含两种基本的数据类型，即数组和矩阵。

NumPy 中的矩阵对象为 matrix，它包含有矩阵的数据处理、矩阵计算、转置、可逆性等功能。matrix 是 ndarray 的子类，矩阵对象是继承自 NumPy 数组对象的二维数组对象，因此，矩阵会含有数组的所有数据属性和方法。

但是，矩阵与数组还有一些重要的区别。

1) 矩阵对象可以使用一个 MATLAB 风格的字符串来创建，也就是一个以空格分隔列，以分号分隔行的字符串。

2) 矩阵是维数为2的特殊数组。矩阵的维数是固定的，即便是加减乘除各种运算，矩阵的维数不会发生变化，而数组在运算时其维数会发生变化。总之，矩阵的维数永远是二维的。

3) 矩阵与数组之间的转化，数组转矩阵用 numpy.asmatrix 或者 numpy.matrix，矩阵转数组用 numpy.asarray 或者 matrix 的 A 属性。

4) 数组中的很多通用函数（ufunc）运算都是元素级的，即函数是针对数组中的每个元素进行处理的，而矩阵是根据矩阵的定义进行整体处理的。

5) 矩阵默认的__array_priority__是 10.0，因而数组与矩阵对象混合的运算总是返回矩阵。

6) 矩阵有几个特有的属性使计算更加容易，这些属性如下：
.T：返回自身的转置；
.H：返回自身的共轭转置；
.I：返回自身的逆矩阵；
.A：返回自身数据的二维数组的一个视图（没有做任何的拷贝）。
np.linalg.matrix_rank(H)  矩阵的秩
a.diagonal()/np.diadiagonal(A) #返回矩阵的对角线元素，也可以通过offset参数在主角线的上下偏移，获取偏移后的对角线元素。a.diagonal(offset=1)返回array([1.10])
a.trace()/np.trace(A) #返回迹



矩阵对象（matrix）也可以使用其他的 matrix 对象、字符串或者其他的可以转换为一个 ndarray 的参数来构造。下面将介绍矩阵的创建、计算及操作。
1. 矩阵的创建
在 NumPy 中，使用 mat()、matrix() 以及 bmat() 函数创建矩阵的方法如下。
1) 使用字符串创建矩阵
在 mat() 函数中输入一个 MATLAB 风格的字符串，该字符串以空格分隔列，以分号分隔行。例如，numpy.mat('1 2 3;4 5 6;7 8 9')，可创建一个 3 行 3 列矩阵，矩阵中元素为整数。
2) 使用嵌套序列创建矩阵
在 mat() 函数中输入嵌套序列，例如，numpy.mat([[2,4,6,8],[1.0,3,5,7.0]])，可创建一个 2 行 4 列的矩阵，矩阵中的元素为浮点数。
3) 使用一个数组创建矩阵
在 mat() 函数中输入数组，例如 numpy.mat(numpy.arange(9).reshape(3,3))，可创建一个 3 行 3 列的矩阵，矩阵中的元素为整数。
4) 使用 matrix() 函数创建矩阵
matrix() 函数可以将字符串、嵌套序列、数组和 matrix 转换成矩阵，其函数格式如下：
matrix(data,dtype=None,copy=True)


其中，data 是指输入的用于转换为矩阵的数据。如果 dtype 是 None，那么数据类型将由 data 的内容来决定。如果 copy 为 True，则会复制 data 中的数据，否则会使用原来的数据缓冲。如果没有找到数据的缓冲区，当然也会进行数据复制。这说明用 matrix 函数创建矩阵，如果修改矩阵的值是不会改变输入 matrix 函数的 data 数据值。
注意：用 mat() 函数创建矩阵时，若输入 matrix 或 ndarray 对象，则不会为它们创建副本。也就是说用 mat() 函数创建矩阵时，如果修改矩阵的值，同时会改变输入 mat() 函数的 matrix 或 ndarray 的值。

5) 使用 bmat() 函数创建矩阵
如果想将小矩阵组合成大矩阵，在 NumPy 中，可以使用 bmat 分块（Block Matrix）矩阵函数实现，其函数格式如下：
bmat(obj,ldict=None,gdict=None)

其中，obj 为 matrix；ldict 和 gdict 为 None。

"""



x = np.matrix([[1, 3, 5], [2, 4, 6]])
print(x)
print("\n")
y = np.matrix("1,2,3;4,5,6;7,8,9")
print(y)

# https://www.92python.com/view/93.html
#使用字符串创建矩阵
str1 = '1 2 3;4 5 6;7 8 9'
a = np.mat(str1)
a[1, 1] = 11                           #修改a矩阵1行1列值，不会影响str
print ('用字符串创建矩阵:',a,str)      #观察str输出没有变化
print("\n")
#使用嵌套序列创建矩阵
b = np.mat( [[2,4,6,8],[1.0,3,5,7.0]])
print ('用嵌套序列创建矩阵:',b)
print("\n")
#使用一个数组创建矩阵
arr = np.arange(9).reshape(3,3)     #创建3行3列数组
c = np.mat(arr)
c[1, 1] = 55                         #修改c矩阵1行1列值，arr值也跟着变化
print('数组创建矩阵:',c,arr)
print("\n")
#使用matrix函数创建矩阵
c = np.matrix(arr,dtype=np.float)   #用arr数组创建矩阵
c[1, 1] = 66                         #修改c矩阵1行1列值，arr值不变化
d = np.matrix([[2,4,6,8],[1.0,3,5,7.0]],dtype=np.int64)  #用序列创建矩阵
e = np.matrix('1 2 3;4 5 6;7 8 9',dtype=np.str_)         #用字符串创建矩阵
f = np.matrix(a,dtype=np.str_)                           #用matrix对象创建矩阵
print('用matrix函数创建矩阵',c,arr, d, e,f)
print("\n")
#使用bmat函数创建矩阵
a=np.mat('3 3 3;4 4 4')
b=np.mat('1 1 1;2 2 2')
print ('用bmat函数创建矩阵:',np.bmat('a b; b a'))
print("\n")


#矩阵计算
ma = np.mat( [[6,4,2,8],[2.0,1,5,7.0]])        #创建2行4列矩阵
mb = np.mat(np.arange(9).reshape(3,3))         #创建3行3列矩阵
mc = np.mat(np.arange(8).reshape(2,4))         #创建2行4列矩阵
#print('矩阵相加：',ma+mb)            #不同行列数矩阵不能相加，会报错
print('矩阵相加：',ma+mc)             #相同行列数矩阵能相加

print("\n")
print('矩阵相减：',ma-mc)
print("\n")
print('矩阵相除：',mc/ma)
print("\n")
#矩阵相乘*，执行计算矩阵的矢量积操作
print('矩阵相乘：',ma,mc.T,ma*mc.T)    #mc.T为转置操作
print("\n")
#使用函数multiply, 执行计算矩阵数量积（点乘）操作
# Numpy中的两种矩阵乘法：Numpy.dot(a,b)和运算符 ‘@’； 两种点乘：Numpy.multiply(a,b)和*；
print('矩阵点乘:',np.multiply(ma,mc))
print("\n")
#矩阵操作
arr = np.array([[2,4,6,8],[1.0,3,5,7.0]])    #创建数组

print('取矩阵ma第1行值',ma[0])                 #取矩阵第1行值
print("\n")
print('取矩阵ma第1行值',ma[0,:])                 #取矩阵第1行值
print("\n")
print("ma[1,1]  = ",ma[1,1] )             #取矩阵第2行第2个数据
print("arr[1][1]  = ",arr[1][1])            #数组取值，注意矩阵不能用ma[1][1]取值,会发生错误
print("ma.shape = ",ma.shape)             #获得ma矩阵的行列数
print("ma.shape[0] = ",ma.shape[0])           #获得ma矩阵的行数
print("ma.shape[1] = ",ma.shape[1])           #获得ma矩阵的列数
#索引取值
ma[0,:]              #取得第1行的所有元素
print(ma[0,:])
print("\n")
ma[0,1:2]            #第1行第2个元素，注意左闭右开
print(ma[0,1:2])
print("\n")
ma.sort()            #对每1行进行排序,会改变ma的原值；

print(ma)
print("\n")
#将Python的列表转换成NumPy的矩阵
list=[1,2,3,5,6]
np.mat(list)         #列表转换成NumPy的矩阵

print(np.mat(list))
print("\n")


# multiple() 函数用于两个矩阵的逐元素乘法，示例如下：
import numpy as np
array1=np.array([[1,2,3],[4,5,6],[7,8,9]],ndmin=3)
array2=np.array([[9,8,7],[6,5,4],[3,2,1]],ndmin=3)
result=np.multiply(array1,array2)
print(result)

# matmul() 用于计算两个数组的矩阵乘积。示例如下：
import numpy as np
array1=np.array([[1,2,3],[4,5,6],[7,8,9]],ndmin=3)
array2=np.array([[9,8,7],[6,5,4],[3,2,1]],ndmin=3)
result=np.matmul(array1,array2)
print(result)


# dot() 函数用于计算两个矩阵的点积。如下所示：
import numpy as np
array1=np.array([[1,2,3],[4,5,6],[7,8,9]],ndmin=3)
array2=np.array([[9,8,7],[6,5,4],[3,2,1]],ndmin=3)
result=np.dot(array1,array2)
print(result)

# 使用mat() 创建矩阵以 ; 隔开每一行，句号隔开行中的列。
e = np.mat("1 2+3j;3 4+5j")
print("原矩阵：\n", e)
# 共轭矩阵
print("共轭矩阵：\n", e.conjugate())
# 共轭转置
print("共轭转置: \n", e.T.conjugate())


# https://blog.csdn.net/qq_42379006/article/details/80563560?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-4.fixedcolumn&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-4.fixedcolumn

'''--------------------numpy矩阵的创建-----------------------'''
import numpy as np
#numpy矩阵创建的方法

# 1.直接使用分号隔开的字符串
mat1 = np.mat("1 2 3;4 5 6;7 8 9")
print(mat1)  #<class 'numpy.matrixlib.defmatrix.matrix'>
print("\n")
print(type(mat1))

# 2.使用numpy数组创建矩阵
arr = np.arange(1,10).reshape(3,3)
# print(type(arr))  #<class 'numpy.ndarray'>
# print(arr)
mat2 = np.mat(arr)
print(mat2)
print("\n")
# print(type(mat2))  #<class 'numpy.matrixlib.defmatrix.matrix'>

# 3.从已有的矩阵中通过bmat函数复合创建矩阵
A = np.eye(2)
B = A*2

mat3 = np.bmat("A B;B A;A B")
print(mat3)
print("\n")
print(mat3.shape)  #(6, 4)
print("\n")
'''--------------------numpy随机矩阵的创建-----------------------'''
import numpy as np


#创建一个服从正太分布的随机矩阵
normal = np.random.normal(size=100)
print(normal)
#打印直方图
# plt.hist(normal)
# plt.show()
print("\n")
#创建一个服从beta分布的随机样本
beta = np.random.beta(a=.5,b=.5,size=100)
# plt.hist(beta)
# plt.show()

#产生均匀分布的随机数
rand = np.random.rand(3,2,4)  #三层两行四列
print(rand)
print("\n")
#产生4x4随机正太分布样本
normal2 = np.random.normal(size=(100,100))
print("normal2-100x100:\n",normal2)
plt.hist(normal2)
plt.show()
print("\n")
#产生在某个范围内的随机整数矩阵
numbers = np.mat(np.random.randint(1,50,[5,5]))
# print(numbers)
print("\n")
#产生0-1之间随机浮点数
floatNum = np.mat(np.random.random(10))  #产生10个0-1的浮点数
print(floatNum)
print("\n")
#在某个范围内的所有数中随机抽取一个
num = np.random.choice(10)  #[0,10)内随机选一个数
print(num)
print("\n")
num1 = np.random.choice(5,10)  #[0,5)内随机选10个数
print(num1)
print("\n")

'''--------------------numpy矩阵的运算add/multiply/*-----------------------'''
import numpy as np

mat1 = np.mat(np.array([2,6,5]))
mat2 = np.mat(np.array([1,4,7]))

#矩阵的加法
addResult = np.add(mat1,mat2)
print(type(addResult))  #<class 'numpy.matrixlib.defmatrix.matrix'>
print("\n")
print(addResult)  #[[ 3 10 12]]
print("\n")
#数组的乘法
multiResult = np.multiply(mat1,mat2) # 元素相乘，mat1和mat2同样的维度
print(multiResult)  #[[ 2 24 35]]
print("\n")
#矩阵相乘 mxp矩阵乘以pxn矩阵，相乘后得mxn矩阵
mat3 = np.mat(np.arange(6).reshape(2,3))
mat4 = np.mat(np.arange(6).reshape(3,2))
print("矩阵相乘mat3*mat4=\n",mat3*mat4)
print("\n")


'''-------------numpy矩阵的运算divide/floor_divide/mod/remainder/fmod-------------'''
import numpy as np

a = np.mat(np.array([4,5,8]))
b = np.mat(np.array([2,3,5]))

#数组的除法
result = np.divide(a,b)
print("数组相除：",result)
print("直接数组相除：",a/b)
#数组除法并将结果向下取整
result2 = np.floor_divide(a,b)
print("相除向下取整：",result2)
print("相除向下取整：",a//b)

# 矩阵取模运算/求余数
result3 = np.remainder(a,b)
print("remainder结果：",result3)

# 取模运算/求余数
result4 = np.mod(a,b)
print("mod结果：",result4)

# 取模运算/求余数
print("取模运算:",a%b)
result5 = np.fmod(a,b)  #所得余数的正负由被除数决定，与除数的正负无关
print("fmod结果：",result5)


#矩阵相除,python2.7和3的结果还不一致，python3里true_divide与divide结果一样
floatResult = np.true_divide(a,b)
print("true_divide:",floatResult)


'''----------------numpy一元通用函数-----------------'''
import numpy as np

mat = np.mat(np.array([-4,3,0,12]))
print("abs:",np.abs(mat))      #[[ 4  3  0 12]]
print("fabs:",np.fabs(mat))   #[[ 4.  3.  0. 12.]]

#获取mat矩阵中各元素的正负号
sign = np.sign(mat)
print("sign:",sign)     #[[-1  1  0  1]]

#将数组中元素的小数和整数部分抽取出来
arr = np.mat(np.array([[1.2,3.14],
               [-2.5,6.8]]))
arr1,arr2 = np.modf(arr)
print("小数部分：",arr1)    #[[ 0.2   0.14]  [-0.5   0.8 ]]
print("整数部分：",arr2)    #[[ 1.  3.]  [-2.  6.]]

#求平方根
C = np.mat(np.arange(1,6))
print("平方根：",np.sqrt(C))

#求平方
D = np.mat(np.arange(-3,2))
print("平方：",np.square(D))

#取整
E = np.mat(np.array([[2.3,4.6],[1.2,1.9]]))
print("ceil:",np.ceil(E))
print("floor:",np.floor(E))
print("rint:",np.rint(E))

'''----------------numpy创建通用函数-----------------'''
import numpy as np

a = np.arange(4).reshape(2,2)
print("a",a)  #[[0 1] [2 3]]

like_a = np.zeros_like(a)
print("like_a-1",like_a)  #[[0 0] [0 0]]

like_a.flat = 42
print("like_a-2:",like_a)  #[[42 42]  [42 42]]

#创建一个numpy通用函数
def like(ndarray):
    result = np.zeros_like(ndarray)
    result = 42   #result.flat = 42
    return result

#调用numpy创建通用函数的方法
myfunc = np.frompyfunc(like,1,1)  #1 个输入， 1 个输出
test = myfunc(np.arange(9).reshape(3,3))
print("test:",test)  #[[42 42 42] [42 42 42] [42 42 42]]

'''-------------------二元通用函数-------------------'''
import numpy as np
#power
mat = np.mat(np.array([1,2,3,4,5]))
I = np.mat(np.array([2,3,2,1,3]))
print("mat power I:",np.power(mat,I))   #power: [[  1   8   9   4 125]]
print("I power mat:",np.power(I,mat))    #power: [[  2   9   8   1 243]]

#获取两个数组对应元素的最大值和最小值，返回到至一个新的数组中
d2 = np.array([[22,15],
              [5.4,6.6]])
d3 = np.array([[15,28],
               [7.9,4.0]])
maximum = np.maximum(d2,d3)
print("maxinum:",maximum)   #[[22.  28. ],[ 7.9  6.6]]
minimum = np.minimum(d2,d3)
print("minimum:",minimum)   #[[15.  15. ],[ 5.4  4. ]]

#数组比较函数
result = np.greater(d2,d3)
print("greater:",result)   #[[ True False], [False  True]]

result1 = np.greater_equal(d2,d3)
print("greater_equal:",result1)  #[[ True False], [False  True]]

#python转换为bool为False:0 None ""  {}  []  ()   False
bool1 = np.array([[1,0],[3,5]])
bool2 = np.array([[0,1],[12,0]])
bool3 = np.logical_and(bool1,bool2)
print("逻辑与：",bool3)   #[[False False] [ True False]]
bool4 = np.logical_xor(bool1,bool2)
print("逻辑异或：",bool4)   #[[ True  True] [False  True]]
bool5 = np.logical_or(bool1,bool2)
print("逻辑或：",bool5)   # [[ True  True] [ True  True]]


'''-------------------np.add模块下的通用函数-------------------'''


#np.add模块下面的通用函数
a = np.arange(9)
print("a=",a)

#reduce求和
print("reduce(a)=",np.add.reduce(a))
print("sum(a)=",np.sum(a))

#accumulate 依次将数组元素相加的每一步结果保存到一个新数组
print("accumulate(a)=",np.add.accumulate(a))

#reduceat,根据给定区间分段进行reduce求和操作
print("reduceat=",np.add.reduceat(a,[0,5,2,7]))  #reduceat= [10  5 20 15]
#上面求的应该是[0,5)之间的和，因为5>2,就去5，再求[2,7)之间和，再求[7:)后面的求和

#outer，将第一个数组中额每个元素分别和第二个数组中的所有元素求和
print("outer=",np.add.outer(np.arange(1,4),a))


'''-------------------矩阵求逆np.linalg.inv()-------------------'''


A = np.mat(np.array([[0,1,2],
                     [1,0,3],
                     [4,-3,8]]))
print("A:\n",A)

#求A矩阵的逆矩阵
A_ = np.linalg.inv(A)
print("A的逆矩阵：\n",A_)

#验证A*A_的结果是否是一个单位矩阵
print("A*A_:\n",A*A_)



'''----------------矩阵求解方程组np.linalg.solve()-------------------'''


'''
x − 2y + z = 0
2y - 8z = 8
−4x + 5y + 9z = −9
'''
#numpy求解三元一次方程组
A = np.mat("1 -2 1;0 2 -8;-4 5 9")
print("方程组的系数：\n",A)
b = np.array([0,8,-9])   #不能用矩阵表示
print("方程组的常数：\n",b)
#调用numpy的solve方法求解
C = np.linalg.solve(A,b)
print("x={},y={},z={}".format(C[0],C[1],C[2]))


'''-------------------矩阵求特征值和特征向量np.linalg.eigvals()/np.linalg.eig()-------------------'''


vector = np.mat("3 -2;1 0")
print(vector)
#求向量的特征值
eigenvalues = np.linalg.eigvals(vector)
print("向量的特征值：",eigenvalues)
#同时求向量的特征值和特征向量
eigenvalues,eigvector = np.linalg.eig(vector)
print("特征值：",eigenvalues)
print("特征向量：",eigvector)


"""

1. np.linalg.eig
Compute the eigenvalues and right eigenvectors of a square array

特点：

（1）可以计算任意矩阵的特征值和特征向量。

（2）特征值没有按大小排序，需自行根据实部排序。

（3）对于相同特征值的情况，得到的特征向量不相互正交，甚至不线性无关。

2. np.linalg.eigh
Return the eigenvalues and eigenvectors of a complex Hermitian (conjugate symmetric) or a real symmetric matrix.

特点：

（1）计算厄密矩阵或实对称矩阵的特征值和特征向量，即特征值一定为实数。

（2）特征值已按大小排序，从小到大。

（3）对于相同特征值的情况，得到的特征向量相互正交。

3. 结论
在已知是厄密矩阵（包括实对称矩阵）的情况下，最好使用np.linalg.eigh，可以避免相同特征值对应的特征向量线性相关和不正交的情况，同时特征值也经过了排序。


"""


from scipy import linalg
import numpy as np

# 特征值特征函数
#=================================================================================================
#======================= H为实对称矩阵 ================================
#=================================================================================================
H = np.array([[1,-2,-4],[-2,4,-2],[-4,-2,1]])

#================== 使用 np.linalg.eig =========================
value, P = np.linalg.eig(H)
print(f"特征值：\n{value}\n特征向量：\n{P}")

A = np.linalg.inv(P)@H@P
# A = np.dot(np.dot(np.linalg.inv(P),H),P)
A = np.around(A,decimals=2)
print(f"相似对角阵 = \n{A}")

PP = P.T@P
print(f"特征向量组成的矩阵是否正交阵 = \n{PP}")




orthP = linalg.orth(P)
A = np.linalg.inv(P)@H@P
# A = np.dot(np.dot(np.linalg.inv(P),H),P)
A = np.around(A,decimals=2)
print(f"特征向量正交化后的相似对角阵 = \n{A}")
print(f"特征向量正交化后的特征矩阵：\n{np.array(orthP,dtype=float)}")
print(f"特征向量正交化后的特征矩阵是否是正交矩阵：\n{np.dot(orthP,orthP.T)}")

PP = orthP@orthP.T
print(f"特征向量正交化后的矩阵是否正交阵 = \n{PP}")


print(f"验证是否正交化")
print(f"{sum(orthP[0,:]*orthP[1,:])}")
print(f"{sum(orthP[0,:]*orthP[2,:])}")
print(f"{sum(orthP[1,:]*orthP[2,:])}")

print(f"{sum(orthP[:,0]*orthP[:,1])}")
print(f"{sum(orthP[:,0]*orthP[:,2])}")
print(f"{sum(orthP[:,1]*orthP[:,2])}")

#================== 使用 np.linalg.eigh =========================
H = np.array([[1,-2,-4],[-2,4,-2],[-4,-2,1]])
value, P = np.linalg.eigh(H)
print(f"特征值：\n{value}\n特征向量：\n{P}")

A = np.linalg.inv(P)@H@P
# A = np.dot(np.dot(np.linalg.inv(P),H),P)
A = np.around(A,decimals=2)
print(f"相似对角阵 = \n{A}")

PP = P.T@P
PP = np.around(PP,decimals=2)
print(f"特征向量组成的矩阵是否正交阵 = \n{PP}")


# 以下不必了，因为np.linalg.eigh 对于相同特征值的情况，得到的特征向量相互正交。
# orthP = linalg.orth(P)
# print(f"正交化后：\n{np.array(linalg.orth(orthP),dtype=float)}")

# print(f"特征矩阵是否正交矩阵：\n{np.dot(orthP,orthP.T)}")


#============================= 使用 Matrix  =========================================
from sympy import Matrix

H = Matrix([[1,-2,-4],[-2,4,-2],[-4,-2,1]])
print("\n\n计算矩阵的特征值，代数重数，特征向量")
print(H.eigenvects())



print("计算矩阵的对角化矩阵")
P,D = H.diagonalize()
P = np.array(P)
D = np.array(D)
print(f"可逆矩阵P = \n{P}")
print(f"对角化矩阵 = \n{D}")

H = np.array(H*1.0)
A = np.linalg.inv(P)@H@P
# A = np.dot(np.dot(np.linalg.inv(P),H),P)
A = np.around(A,decimals=2)
print(f"相似对角阵 = \n{A}")

PP = P.T@P
print(f"特征向量组成的矩阵是否正交阵 = \n{PP}")


#=========================================================================
M = Matrix([[3,-2,4,-2],[5,3,-3,-2],[5,-2,2,-2],[5,-2,-3,3]])

print("\n\n计算矩阵的特征向量")
print(M.eigenvects())



M = Matrix([[3,-2,4,-2],[5,3,-3,-2],[5,-2,2,-2],[5,-2,-3,3]])

print("计算矩阵的对角化矩阵")
P,D = M.diagonalize()
print(f"可逆矩阵P = \n{np.array(P)}")
print(f"对角化矩阵 = \n{np.array(D)}")


#==========================================================================
"""
https://www.guanjihuan.com/archives/17789


This code is supported by the website: https://www.guanjihuan.com
The newest version of this code is on the web page: https://www.guanjihuan.com/archives/17789
"""

import numpy as np

def hamiltonian(width=2, length=2):
    h00 = np.zeros((width*length, width*length))
    for i0 in range(length):
        for j0 in range(width-1):
            h00[i0*width+j0, i0*width+j0+1] = 1
            h00[i0*width+j0+1, i0*width+j0] = 1
    for i0 in range(length-1):
        for j0 in range(width):
            h00[i0*width+j0, (i0+1)*width+j0] = 1
            h00[(i0+1)*width+j0, i0*width+j0] = 1
    return h00

print('矩阵：\n', hamiltonian(), '\n')

eigenvalue, eigenvector = np.linalg.eig(hamiltonian())
print('eig求解特征值：', eigenvalue)
print('eig求解特征向量：\n',eigenvector)
print('判断特征向量是否正交：\n', np.dot(eigenvector.transpose(), eigenvector))

print()

eigenvalue, eigenvector = np.linalg.eigh(hamiltonian())
print('eigh求解特征值：', eigenvalue)
print('eigh求解特征向量：\n',eigenvector)
print('判断特征向量是否正交：\n', np.dot(eigenvector.transpose(), eigenvector))

#==========================================================================





#=============================================================================
#=========  Jordan标准型 ========================
#=============================================================================
# https://www.jianshu.com/p/5c0fa92a0c02
import numpy as np
from sympy import Matrix
import sympy
import pprint

A = np.array([[3,1,0,0],[-4,-1,0,0],[6,2,0,-1],[-2,0,1,2]])
a = Matrix(A)
P, Ja = a.jordan_form()
pprint.pprint(Ja)


B = np.array([[-1,-4,-8],[1,3,8],[0,0,-1]])
b = Matrix(B)
P,Jb = b.jordan_form()
pprint.pprint(Jb)




'''------------------- 进行 Cholesky分解 -------------------'''
# A能够作Cholesky分解的充要条件是A是对称正定矩阵
# https://blog.csdn.net/qq_38689263/article/details/117223903

#yangbocsu 2021.05.24  科技园7栋
import numpy as np
from scipy import linalg
A = np.array([[9, 6, 3],
              [6, 13, 11],
              [3, 11, 35]])
L = linalg.cholesky(A, lower=True) # 默认计算 upper， 所以指定 lower = True
U = linalg.cholesky(A)

print("L = \n",L)
print("\nU = \n",U)

# 验证  np.allclose()
print(f"np.dot(L, L.T) = \n{np.dot(L, L.T)}")


# https://zhuanlan.zhihu.com/p/112091443
import numpy as np
from scipy import linalg

a = np.array([[4, 12, -16],
              [12, 37, -43],
              [-16, -43, 98]])

L = linalg.cholesky(a, lower=True) # 默认计算 upper， 所以指定 lower = True

# array([[ 2.,  0.,  0.],
#       [ 6.,  1.,  0.],
#       [-8.,  5.,  3.]])

print(f"np.dot(L, L.T) = \n{np.dot(L, L.T)}") # 验证计算
'''-------------------矩阵奇异值分解np.linalg.svd()-------------------'''
import numpy as np

#1. SVD分解
A= [[1,1,3,6,1],[5,1,8,4,2],[7,9,2,1,2]]
A=np.array(A)
U,s,VT = np.linalg.svd(A)
# 为节省空间，svd输出s只有奇异值的向量
print('奇异值：',s)
# 根据奇异值向量s，生成奇异值矩阵
Sigma = np.zeros(np.shape(A))
Sigma[:len(s),:len(s)] = np.diag(s)

print("U：\n",U)
print('Sigma：\n',Sigma)
print('VT：\n',VT)

#2.SVD重构
B = U.dot(Sigma.dot(VT))
print('重构后的矩阵B：\n', B)

print('原矩阵与重构矩阵是否相同？',np.allclose(A,B))

# 3. SVD矩阵压缩（降维）
for k in range(3,0,-1):  # 3,2,1
    # U的k列，VT的k行
    D = U[:,:k].dot(Sigma[:k,:k].dot(VT[:k,:]))
    print('k=',k,"压缩后的矩阵：\n",np.round(D,1))  # round取整数


ATA = A.T@A

V , P = np.linalg.eigh(ATA)

print(f"A = \n{A}\n特征值 = \n{V}\n特征向量 = \n{P}\n")


U1,s1,VT1 = np.linalg.svd(ATA)
# 为节省空间，svd输出s只有奇异值的向量
print('奇异值：',s1)
# 根据奇异值向量s，生成奇异值矩阵
Sigma1 = np.zeros(np.shape(ATA))
Sigma1[:len(s1),:len(s1)] = np.diag(s1)

print("U1：\n",U1)
print('Sigma1：\n',Sigma1)
print('VT1：\n',VT1)

#2.SVD重构
B1 = U1.dot(Sigma1.dot(VT1))
print('重构后的矩阵B：\n', B1)

print('原矩阵与重构矩阵是否相同？',np.allclose(ATA,B1))

'''-------------------  最小二乘解 -------------------'''
import numpy as np
import scipy.linalg

A = np.arange(12).reshape(3, 4)
b = np.array([1, 2, 3])
print("A:", A)
print("b:", b)

# 求解
x1 = np.linalg.lstsq(A, b)[0]
x2 = scipy.linalg.lstsq(A, b)[0]
print("numpy:", x1)
print("scipy:", x2)

# 验算
b_hat_1 = A.dot(x1)
b_hat_2 = A.dot(x2)
print("numpy reconstruct:", b_hat_1)
print("scipy reconstruct:", b_hat_2)


'''----------------矩阵的秩 np.linalg.matrix_rank(H)-------------------'''
A = np.mat(np.array([[0,1,2],
                     [1,0,3],
                     [4,-3,8]]))
print(f"矩阵A的秩={np.linalg.matrix_rank(A)}")


'''-------------------矩阵的行列式np.linalg.det()-------------------'''


vector = np.mat("3 4;5 6")

print(vector)
#求矩阵的行列式
value = np.linalg.det(vector)
print("矩阵的行列式的值：",value)

'''-------------------数组的排序函数np.argsort()/sort()-------------------'''
#二维数组排序
d2 = np.array([[12,8,35],
               [22,45,9]])
d2.sort(axis=0)  #按列排序
print("当axis=0时排序：",d2)
d2.sort(axis=1)  #按行排序
print("当axis=1时排序：",d2)

# 2.返回排序后数组元素的索引
argsort = np.argsort(arr)
print("argsort:",argsort)
#将多维数组按列排序后返回列索引
print(np.argsort(d2,axis=0))

#将多维数组按行降序排序后返回行索引
print(np.argsort(-d2,axis=1))

'''-------------------搜索函数np.searchsorted()-------------------'''

a = np.array([[12,56,-34],
              [18,200,6]])
b = np.array([34,22,34,67,11,90,20])
#获取数组中元素最大值的下标，如果是多维数组，则展平后输出最大值的下标
argmax = np.argmax(a)
print(argmax)  #4  将二维数组展平后输出最大值的索引
print(np.argmax(b))  #5

#根据条件在数组中搜索非零元素，并分组返回对应的下标
print("argwhere:\n",np.argwhere(b>20))   #[[0][1][2][3][5]]
print("argwhere:\n",np.argwhere(a>40))  # [[0 1] [1 1]]

#searchsort为需要添加到数组中的元素寻找合适的下标，
#使得原数组的排序顺序不变
sorted =np.arange(5)
print("添加新元素之前：",sorted)  #[0 1 2 3 4]
indices = np.searchsorted(sorted,[-2,7])
print("寻找到的插入位置下标：",indices)  #[0 5]
#根据寻找到的下标，将元素添加到数组中，返回新的数组
result = np.insert(sorted,indices,[-2,7])
print("新数组：",result)  #[-2  0  1  2  3  4  7]

'''-----------------专有函数np.nonzero/np.extract----------------'''


#提取数组中的非零元素索引np.nonzero()
arr = np.array([[0,1,2],
                [0,3,4],
                [0,5,6]])
rows,cols = np.nonzero(arr)
print("数组中的非零元素的行索引:", rows)
print("数组中的非零元素的列索引:", cols)
#利用深度组合将行列索引合并
indices = np.dstack((rows, cols))
print("数组中的非零元素的索引:", indices)
#----------------------------------------


a = np.arange(10)
#生成一个抽取数组元素花式索引
condition = a%2 ==0 #赋值运算符，算术运算符，逻辑运算符优先级
print("花式索引", condition)  #[ True False  True False  True False  True False  True False]

#np.extract()根据给定条件提取数组元素
even = np.extract(condition, a)
print("数组中的偶数项：", even)   #[0 2 4 6 8]

#用数组的compress方法提取元素
even2 = a.compress(condition)
print("even2:", even2)     #[0 2 4 6 8]

#用np.take函数结合np.where实现提取偶数项
indices = np.where(a%2 ==0)
print("偶数项元素的索引：", indices)  #(array([0, 2, 4, 6, 8], dtype=int64),)
print("even3:", np.take(a,indices))   #[[0 2 4 6 8]]






