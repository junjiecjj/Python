# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 23:01:26 2022

@author: 陈俊杰

"""


print("-"*70)
"""
random.random
random.random () 用于生成一个 0 到 1 的随机浮点数: 0 <= n < 1.0

random.uniform
random.uniform 的函数原型为：random.uniform (a, b)，用于生成一个指定范围内的随机符点数，两个参数其中一个是上限，一个是下限。如果 a > b，则生成的随机数 n: a <= n <= b。如果 a <b， 则 b <= n <= a。

random.randint
random.randint () 的函数原型为：random.randint (a, b)，用于生成一个指定范围内的整数。其中参数 a 是下限，参数 b 是上限，生成的随机数 n: a <= n <= b

random.randrange
random.randrange 的函数原型为：random.randrange ([start], stop [, step])，从指定范围内，按指定基数递增的集合中 
获取一个随机数。如：random.randrange (10, 100, 2)，结果相当于从 [10, 12, 14, 16, ... 96, 98] 序列中获取一个随机数。random.randrange (10, 100, 2) 在结果上与 random.choice (range (10, 100, 2) 等效。

                            
random.choice
random.choice 从序列中获取一个随机元素。其函数原型为：random.choice (sequence)。参数 sequence 表示一个有序类型。
这里要说明 一下：sequence 在 python 不是一种特定的类型，而是泛指一系列的类型。list, tuple, 字符串都属于 sequence。
                            

                            
random.shuffle
random.shuffle 的函数原型为：random.shuffle (x [, random])，用于将一个列表中的元素打乱。如:
                            
random.sample
random.sample 的函数原型为：random.sample (sequence, k)，从指定序列中随机获取指定长度的片断。sample 函数不会修改原有序列。
                   
"""


print("-"*70)
print("random 模块")
print("-"*70)

import random


a = random.random()
b = random.random()
print(a,b)  
#0.14950126763787908 0.18635283756700527


import random
a = random.uniform(5,10)
b = random.uniform(10,5)
print(a,b)
#6.9049288850125485 9.36059520278101



import random
a = random.randint(2,4)
b = random.randint(1,5)
c  = random.randint(10,80)
print(a,b,c)
#2 3 34
#随机选取 0 到 100 间的偶数：
print("random.randrange(0, 101, 2) = {}".format(random.randrange(0, 101, 2)))
#random.randrange(0, 101, 2) = 10

import random
a = random.randrange(1,10,2)
b = random.randrange(1,10,2)
c = random.randrange(50)
print(a,b)
# 2 3
print(c)
# 48



import random
list_1 = ['python','java','c']
str_1 = "i love python"
tuple_1 = (1,2,'kai')
print(random.choice(list_1))  #java
print(random.choice(str_1))   #v
print(random.choice(tuple_1)) #2

a = random.choice(['jiandao','shitou','bu'])
print (a) #jiandao


import random
list_1 = ['python','java','c','c++']
random.shuffle(list_1)
print(list_1)
# ['c', 'python', 'java', 'c++']



import random
list_1 = ['one','two','three','four']
slice1 = random.sample(list_1,2)
print(list_1)  # ['one', 'two', 'three', 'four']
print(slice1)   # ['two', 'three']


print("random.sample('qwertyuiop',3) = {}".format(random.sample('qwertyuiop',3)))
# random.sample('qwertyuiop',3) = ['t', 'e', 'p']



"""
normalvariate() 是内置的方法 random 模块。它用于返回具有正态分布的随机浮点数。

用法： random.normalvariate(mu, sigma)
参数：
mu：平均
sigma：标准偏差

返回：随机正态分布浮点数
"""
import random

# 生成呈正态分布的随机数
# print("normalvariate: ", random.normalvariate(0, 1))
# 产生一组满足正太分布的随机数
walk = []
for _ in range(1000):
    walk.append(random.normalvariate(0, 1))

# 画成直方图
import matplotlib.pyplot as plt  # 导入模块

plt.hist(walk, bins=30)  # bins直方图的柱数
plt.show()
#print(walk)


nums = []  
mu = 100
sigma = 50
    
for i in range(10000):  
    temp = random.normalvariate(mu, sigma)  
    nums.append(temp)  
        
# plotting a graph  
plt.hist(nums, bins = 200)  
plt.show()



"""
gauss() 是内置的方法 random 模块。它用于返回具有高斯分布的随机浮点数。

用法： random.gauss(mu, sigma)
参数：
mu：平均
sigma：标准偏差

返回：随机高斯分布浮点数
"""
# store the random numbers in a list  
nums = []  
mu = 100
sigma = 50
    
for i in range(10000):  
    temp = random.gauss(mu, sigma)  
    nums.append(temp)  
        
# plotting a graph  
plt.hist(nums, bins = 200)  
plt.show()


print("-"*70)
print("numpy  模块")
print("-"*70)

import numpy as np

# numpy.random.rand(d0, d1, ..., dn)，产生d0 - d1 - ... - dn形状的在[0,1)上均匀分布的float型数。
print("numpy.random.rand(2,3) = \n{}".format(np.random.rand(2,3)))


#从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
print("np.random.uniform(1,2,(2,3)) = \n {}".format(np.random.uniform(1,2,(2,3))))

print("np.random.uniform(1,2,5) = \n {}".format(np.random.uniform(1,2,5)))

#从以low为下限，high为上限的随机均匀分布数中随机取样，生成size维度的均匀分布数组，浮点型。
#包括下限，不包括上限，不指定low和high时，默认值为low =0，high =1；如果high = None，则取[0,low)区间。
print("np.random.uniform(low=2,high=8,size=(3,6))  = \n {}".format(np.random.uniform(low=2,high=8,size=(3,6)) ))




# numpy.random.randint(low, high=None, size=None, dtype='l')，产生随机整数；
print("numpy.random.randint(5, size = (3,4)) = \n{}".format(np.random.randint(5,  size = (3,4))))

print("numpy.random.randint(5,size=(2,3)) = \n{}".format(np.random.randint(5,size=(2,3))))

print("np.random.randint(low=2,high=8,size=(3,6))   = \n {}".format(np.random.randint(low=2,high=8,size=(3,6))  ))
#从以low为下限，high为上限的随机均匀分布数中随机取样，生成size维度的均匀分布数组，整型。
#包括下限，不包括上限，不指定low和high时，如果high = None，则取[0,low)区间。



print("-"*70)
#生成随机正态分布数。
# loc：float类型，表示此正态分布的均值（对应整个分布中心）
# scale：float类型，表示此正态分布的标准差（对应于分布的密度，scale越大越矮胖，数据越分散；scale越小越瘦高，数据越集中）
# size：输出的shape，size=(k,m,n) 表示输出k维，m行，n列的数，默认为None，只输出一个值，size=100，表示输出100个值
print("np.random.normal(loc =0.0 , scale= 1.0,size = (5,4)) = \n {}".format(np.random.normal(loc =0.0 , scale= 1.0,size = (5,4))))


# randn: 原型：numpy.random.randn（d0,d1,...,dn),产生d0 - d1 - ... - dn形状的标准正态分布的float型数。
print("numpy.random.randn(3,4) = \n{}".format(np.random.randn(3,4)))



#返回指定形状的标准正态分布数组
print("np.random.standard_normal(size = (5,6)) =\n {}".format(np.random.standard_normal(size = (5,6))))


import numpy as np
import matplotlib.pyplot as plt#导入模块

plt.hist(np.random.normal(loc=0.0, scale=1.0, size=1000), bins=30)#bins直方图的柱数
plt.show()



print("-"*70)
print("numpy 模块计算均值、方差、标准差")
print("-"*70)
import numpy as  np

print("---------------------------- 均值 -------------------------------------------\n")
a = np.array([1,2,3,4,5,6,7,8,9])
print("np.mean(a) 均值 = \n{}".format(np.mean(a)))

#除了 np.mean 函数，还有 np.average 函数也可以用来计算 mean，不一样的地方时，np.average 函数可以带一个 weights 参数：
print("np.average(a) 均值 = \n{}".format(np.average(a)))

print("np.average(a, weights=(1,1,1,1,1,1,1,6,1)) = {}".format(np.average(a, weights=(1,1,1,1,1,1,1,6,1))))


a = np.array([[ 0, 1, 2, 3, 4],[ 5, 6, 7, 8, 9],[10, 11, 12, 13, 14],[15, 16, 17, 18, 19]])

print("a =\n {}\n\n".format(a))

#求每列的均值
print("np.mean(a, axis=0) = {}".format(np.mean(a, axis=0)))

#求每行的均值
print("np.mean(a, axis=1) = {}".format(np.mean(a, axis=1)))

#求所有的均值
print("np.mean(a) = {}".format(np.mean(a)))


print("---------------------------- 方差 -------------------------------------------\n")
#计算方差时，可以利用 numpy 中的 var 函数，默认是总体方差（计算时除以样本数 N），若需要得到样本方差（计算时除以 N - 1），需要跟参数 ddo f= 1，例如
#求每列的总体方差
print("np.var(a, axis=0) = {}".format(np.var(a, axis=0)))

#求每行的总体方差
print("np.var(a, axis=1) = {}".format(np.var(a, axis=1)))

#求所有的总体方差
print("np.var(a) = {}".format(np.var(a)))

#np.var 函数计算方差。注意 ddof 参数，默认情况下，np.var 函数计算方差时，是除以 n=len (a)，此时 ddof=0。我们都知道用样本方差来估计总体方差的计算公式是除以 n-1，此时 ddof=1。
#求每列的样本方差
print("np.var(a, axis=0) = {}".format(np.var(a, axis=0, ddof=1)))

#求每行的样本方差
print("np.var(a, axis=1) = {}".format(np.var(a, axis=1, ddof=1)))

#求所有的样本方差
print("np.var(a) = {}".format(np.var(a, ddof=1)))

print("----------------------------标准差-------------------------------------------\n")
#计算标准差时，可以利用 numpy 中的 std 函数，使用方法与 var 函数很像，默认是总体标准差，若需要得到样本标准差，需要跟参数 ddof =1，
#求每列的总体标准差
print("np.var(a, axis=0) = {}".format(np.std(a, axis=0, )))

#求每行的总体标准差
print("np.var(a, axis=1) = {}".format(np.std(a, axis=1, )))

# 计算矩阵所有元素的总体标准差
print("np.var(a) = {}".format(np.std(a, )))



#求每列的样本标准差
print("np.var(a, axis=0) = {}".format(np.std(a, axis=0, ddof=1)))

#求每行的样本标准差
print("np.var(a, axis=1) = {}".format(np.std(a, axis=1, ddof=1)))

# 计算矩阵所有元素的样本标准差
print("np.var(a) = {}".format(np.std(a, ddof=1)))

print("----------------------------以正太分布为例计算均值、标准差、方差-------------------------------------------\n")
import numpy as np
#a = np.random.randn(2,100000000)
a = np.random.normal(loc =2 , scale= 3,size = (2,10000000))

print("a每一行的均值 = {}\n".format(a.mean(axis = 1)))  #a每一行的均值 = [1.99966296 1.99928042]
print("a每一行的均值 = {}\n".format(np.mean(a,axis=1)))  #

print("a每一列的均值 = {}\n".format(a.mean(axis = 0)))  #a每一列的均值 = [7.11342338 0.2291015  5.80927679 ... 4.08929181 2.20833559 4.99018901]
print("a每一列的均值 = {}\n".format(np.mean(a,axis=0)))  #a每一列的均值 = [7.11342338 0.2291015  5.80927679 ... 4.08929181 2.20833559 4.99018901]

print("a所有元素的均值 = {}\n".format(a.mean()))  #a所有元素的均值 = 1.9994716916165884
print("a所有元素的均值 = {}\n".format(np.mean(a)))  #a所有元素的均值 = 1.9994716916165884



print("a每一行的总体标准差 = {}\n".format(a.std(axis = 1)))  #a每一行的总体标准差 = [2.9996407  2.99993355]
print("a每一行的总体标准差 = {}\n".format(np.std(a, axis=1, )))  #a每一行的总体标准差 = [2.9996407  2.99993355]

print("a每一列的总体标准差 = {}\n".format(a.std(axis = 0)))  #
print("a每一列的总体标准差 = {}\n".format(np.std(a, axis=0, )))  #

print("a所有元素的总体标准差 = {}\n".format(a.std()))  #a所有元素的总体标准差 = 2.999787135559725
print("a所有元素的总体标准差 = {}\n".format(np.std(a)))  #a所有元素的总体标准差 = 2.999787135559725


print("a每一行的样本标准差 = {}\n".format(a.std(axis = 1, ddof=1)))  #a每一行的样本标准差 = [2.99964085 2.9999337 ]
print("a每一行的样本标准差 = {}\n".format(np.std(a, axis=1,  ddof=1)))  #a每一行的样本标准差 = [2.99964085 2.9999337 ]

print("a每一列的样本标准差 = {}\n".format(a.std(axis = 0, ddof=1)))  #
print("a每一列的样本标准差 = {}\n".format(np.std(a, axis=0,  ddof=1)))  #

print("a所有元素的样本标准差 = {}\n".format(a.std( ddof=1)))  #a所有元素的样本标准差 = 2.9997872105544063
print("a所有元素的样本标准差 = {}\n".format(np.std(a, ddof=1)))  #a所有元素的样本标准差 = 2.9997872105544063



print("a每一行的总体方差 = {}\n".format(a.var(axis = 1)))  #a每一行的总体方差 = [8.99784433 8.99960131]
print("a每一行的总体方差 = {}\n".format(np.var(a, axis=1, )))  # a每一行的总体方差 = [8.99784433 8.99960131]

print("a每一列的总体方差 = {}\n".format(a.var(axis = 0)))  #
print("a每一列的总体方差 = {}\n".format(np.var(a, axis=0, )))  #

print("a所有元素的总体方差 = {}\n".format(a.var()))  #a所有元素的总体方差 = 8.998722858669622
print("a所有元素的总体方差 = {}\n".format(np.var(a)))  #a所有元素的总体方差 = 8.998722858669622


print("a每一行的样本方差 = {}\n".format(a.var(axis = 1, ddof=1)))  #a每一行的样本方差 = [8.99784523 8.99960221]
print("a每一行的样本方差 = {}\n".format(np.var(a, axis=1,  ddof=1)))  #a每一行的样本方差 = [8.99784523 8.99960221]

print("a每一列的样本方差 = {}\n".format(a.var(axis = 0, ddof=1)))  #
print("a每一列的样本方差 = {}\n".format(np.var(a, axis=0,  ddof=1)))  #

print("a所有元素的样本方差 = {}\n".format(a.var( ddof=1)))  #a所有元素的样本方差 = 8.998723308605786
print("a所有元素的样本方差 = {}\n".format(np.var(a, ddof=1)))  #a所有元素的样本方差 = 8.998723308605786

n=2
b = a * n

print(" b 所有元素的均值 = {}\n".format(b.mean()))  # b 所有元素的均值 = 3.998943383233177
print(" b 所有元素的均值 = {}\n".format(np.mean(b)))  # b 所有元素的均值 = 3.998943383233177

print(" b 所有元素的总体标准差 = {}\n".format(b.std()))  # b 所有元素的总体标准差 = 5.99957427111945
print(" b 所有元素的总体标准差 = {}\n".format(np.std(b)))  # b 所有元素的总体标准差 = 5.99957427111945

print(" b 所有元素的总体方差 = {}\n".format(b.var()))  # b 所有元素的总体方差 = 35.994891434678486
print(" b 所有元素的总体方差 = {}\n".format(np.var(b)))  # b 所有元素的总体方差 = 35.994891434678486






print("-"*70)
print("产生数列")
print("-"*70)
# range(start,stop,step)
# start： 默认值为0，包括start，可省略
# stop：  终止值，不包括stop，不可省略
# step:   步长，默认值为1，不能是浮点型，可省略


#等价于range(0,10,1)   可以省略起始值和步长。
print("range(10)  = \n{} ".format(list(range(10))))



range(2,10,2)   #以2为起始，10为终止，步长为2的序列，[2,10)
#在python3中返回的是一个可迭代对象，通过list(range(2,10,2)) 可以列表形式输出，而在python2中则之间以列表形式输出




#numpy中np.arange()函数
#np.arange(start,stop,step)函数：创建一个数组ndarray
#start： 默认值为0，包括start，可省略
#stop：  终止值，不包括stop，不可省略
#step:   步长，默认值为1，可以是浮点型，可省略
 
 
print("np.arange(10,20,2)  = \n{}".format(np.arange(10,20,2) )) #以10为起始，20为终止，步长为2的numpy数组[10,20)
#array([10, 12, 14, 16, 18])
 
print("np.arange(1,5,0.5) = \n{}".format(np.arange(1,5,0.5)))  #可以指定步长为浮点型，生成浮点型数组。
#array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5])


"""
start
start 参数数值范围的起始点。如果设置为 0，则结果的第一个数为 0. 该参数必须提供。

stop
stop 参数数值范围的终止点。通常其为结果的最后一个值，但如果修改 endpoint = False, 则结果中不包括该值 (后面示例会说明)。

num (可选)
num 参数控制结果中共有多少个元素。如果 num=5，则输出数组个数为 5. 该参数可选，缺省为 50.

endpoint (可选)
endpoint 参数决定终止值 (stop 参数指定) 是否被包含在结果数组中。如果 endpoint = True, 结果中包括终止值，反之不包括。缺省为 True。

dtype (可选)
和其他的 NumPy 一样，np.linspace 中的 dtype 参数决定输出数组的数据类型。如果不指定，python 基于其他参数值推断数据类型。如果需要可以显示指定，参数值为 NumPy 和 Python 支持的任意数据类型。

"""
# 通过定义均匀间隔创建数值序列。其实，需要指定间隔起始点、终止端，以及指定分隔值总数（包括起始点和终止点）；最终函数返回间隔类均匀分布的数值序列。请看示例：
#  从 0 到 100，间隔为 10 的数值序列
np.linspace(start = 0, stop = 100, num = 11)


#前文提到，endpoint 参数决定终止值是否被包含在结果数组中。缺省为 True，即包括在结果中，反之不包括，请看示例：
np.linspace(start = 1, stop = 5, num = 4, endpoint = False)

#默认 linspace 根据其他参数类型推断数据类型，很多时候，输出结果为 float 类型。如果需要指定数据类型，可以通过 dtype 设置。该参数很直接，除了 linspace 其他函数也一样，如：np.array,np.arange 等。示例：
np.linspace(start = 0, stop = 100, num = 5, dtype = int)


print("-"*70)
print("scipy模块")
print("-"*70)

# Python 生成均值为 2 ，标准差为 3 的一维正态分布样本 500
import numpy as np
import scipy.stats as st 
import matplotlib.pyplot as plt
 
s=np.random.normal(2, 3, 500)
s_fit = np.linspace(s.min(), s.max())
plt.plot(s_fit, st.norm(2, 3).pdf(s_fit), lw=2, c='r')
plt.show()

from scipy import stats
a=stats.norm.rvs(0,2,size=500)




import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib import style
style.use('fivethirtyeight')
mu_params = [-1, 0, 1]
sd_params = [0.5, 1, 1.5]
x = np.linspace(-7, 7, 100)
f, ax = plt.subplots(len(mu_params), len(sd_params), sharex=True, sharey=True, figsize=(12,8))
for i in range(3):
    for j in range(3):
        mu = mu_params[i]
        sd = sd_params[j]
        y = stats.norm(mu, sd).pdf(x)
        ax[i, j].plot(x, y)
        ax[i, j].plot(0,0, label='mu={:3.2f}\nsigma={:3.2f}'.format(mu,sd), alpha=0)
        ax[i, j].legend(fontsize=10)
ax[2,1].set_xlabel('x', fontsize=16)
ax[1,0].set_ylabel('pdf(x)', fontsize=16)
plt.suptitle('Gaussian PDF', fontsize=16)
plt.tight_layout()
plt.show()






import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
from pylab import *
 
mu, sigma = 5, 0.7
lower, upper = mu - 2 * sigma, mu + 2 * sigma  # 截断在[μ-2σ, μ+2σ]
X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
N = stats.norm(loc=mu, scale=sigma)
 
 
figure(1)
subplot(2,1,1)
plt.hist(X.rvs(10000), bins=30)   # 截断正态分布的直方图
subplot(2,1,2)
plt.hist(N.rvs(10000),   bins=30)   # 常规正态分布的直方图
plt.show()


























































































































































































































































































































































































































































































































