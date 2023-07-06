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



#==============================================================

# seed()改变随机数生成器的种子，在调用其他随机模块函数之前调用此函数
# seed()没有参数时，每次生成的随机数是不一样的，seed()有参数时是一样的，不同的参数生成的随机数不一样
a = random.random()
b = random.random()
print(a,b)


random.seed(10)

a = random.random()
b = random.random()
print(a,b)


# 随机数不一样
random.seed()
print('随机数1：',random.random())
random.seed()
print('随机数2：',random.random())

# 随机数一样
random.seed(1)
print('随机数3：',random.random())
random.seed(1)
print('随机数4：',random.random())
random.seed(2)
print('随机数5：',random.random())

random.seed(10)
for i in range(6):
    print(random.random())





print("-"*70)
print("ramdon模块  瑞利分布概率分布")
print("-"*70)

print("当一个随机二维向量的两个分量呈独立的、有着相同的方差、均值为0的正态分布时，这个向量的模呈瑞利分布。例如，当随机复数的实部和虚部独立同分布于0均值，同方差的正态分布时，该复数的绝对值服从瑞利分布。\
      \n瑞利分布的概率函数为：")
from IPython.display import Latex
Latex(r'$ f(x;\sigma)=\frac{x}{\sigma ^2} e^{-\frac{x^2}{2\sigma^2}}  $')




"""
https://blog.csdn.net/share727186630/article/details/107403761

https://baike.baidu.com/item/%E7%91%9E%E5%88%A9%E5%88%86%E5%B8%83/10284554

验证由两个均值为0，方差相同的独立同分布的正太分布分别为实部和虚部的复数向量分布符合瑞丽分布
"""
import numpy as np
mean, std = 0, 3
real = np.random.normal(loc =mean , scale= std, size = 100000000)
img = np.random.normal(loc =mean , scale= std, size = 100000000)

print(" real 所有元素的均值 = {}\n".format(np.mean(real)))  # real所有元素的均值 = 1.9994716916165884
print(" img 所有元素的均值 = {}\n".format(np.mean(img)))    # img所有元素的均值 = 1.9994716916165884

print(" real 所有元素的总体方差 = {}\n".format(real.var()))  #a所有元素的总体方差 = 8.998722858669622
print(" img 所有元素的总体方差 = {}\n".format(np.var(img)))  #a所有元素的总体方差 = 8.998722858669622


H = real  + img*1j
#H = real/np.sqrt(2) + img*1j/np.sqrt(2)

H_mod = np.abs(H)



print("H_mod mean = {}, var = {}\n".format( np.sqrt(np.pi/2)*std, (4-np.pi)*std**2/2 ))
# H_mod mean = 3.7599424119465006, var = 3.862833058845931
print(" H_mod 所有元素的均值 = {}\n".format(np.mean(H_mod)))    # H_mod 所有元素的均值 = 3.7598373748179172
print(" H_mod 所有元素的总体方差 = {}\n".format(np.var(H_mod))) # H_mod 所有元素的总体方差 = 3.863108295676793


















































































































































































































































































































































































































































