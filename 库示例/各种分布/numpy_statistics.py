#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 19:19:04 2022

@author: jack

https://numpy.org/doc/stable/reference/random/legacy.html

https://numpy.org/doc/1.14/reference/generated/numpy.random.RandomState.html

"""

import numpy as np
import random


#===========================================================================================================================
#                                             "均匀分布"
#===========================================================================================================================
"""
np.random.uniform(low, high ,size)  作用于从一个均匀分布的区域中随机采样。

其形成的均匀分布区域为[low, high)  注意定义域是左闭右开，即包含low，不包含high.
    low：采样区域的下界，float类型或者int类型或者数组类型或者迭代类型，默认值为0
    high：采样区域的上界，float类型或者int类型或者数组类型或者迭代类型，默认值为1
    size：输出样本的数目(int类型或者tuple类型或者迭代类型)
    返回对象：ndarray类型，形状和size中的数值一样

"""
#从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
print("np.random.uniform(1,2, (2,3) ) = \n {}".format(np.random.uniform(1,2, (2,3))))

# 产生 [1,2)上均匀分布的浮点数
print("np.random.uniform(1,2,5) = \n {}".format(np.random.uniform(1,2, 5)))



#从以low为下限，high为上限的随机均匀分布数中随机取样，生成size维度的均匀分布数组，浮点型。
#包括下限，不包括上限，不指定low和high时，默认值为low =0，high =1；如果high = None，则取[0,low)区间。
print("np.random.uniform(low=2,high=8,size=(3,6))  = \n {}".format(np.random.uniform(low = 2, high = 8, size = (3,6)) ))

#===========================================================================================================================
#                                             "[0,1)上均匀分布的float型数"
#===========================================================================================================================
import numpy as np

# numpy.random.rand(d0, d1, ..., dn)，产生d0 - d1 - ... - dn形状的在[0,1)上均匀分布的float型数。
print("numpy.random.rand(2,3) = \n{}".format(np.random.rand(2,3)))



#===========================================================================================================================
#                                             "随机整数"
#===========================================================================================================================

"""
random.randint(low, high=None, size=None, dtype=int)
从一个均匀分布中随机采样，生成一个整数或N维整数数组，


函数的作用是，返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)。
如果没有写参数high的值，则返回[0,low)的值。
参数如下：
    low: int, 生成的数值最低要大于等于low。 （hign = None时，生成的数值要在[0, low)区间内）
    high: int (可选) 如果使用这个值，则生成的数值在[low, high)区间。
    size: int or tuple of ints(可选) 输出随机数的尺寸，比如size = (m * n* k)则输出同规模即m * n* k个随机数。默认是None的，仅仅返回满足要求的单一随机数。
    dtype: dtype(可选)：
    想要输出的格式。如int64、int等等

"""


#  产生0到5之间的随机整数；
print("numpy.random.randint(5) = \n{}".format(np.random.randint(5)))

# 产生0到5之间的shape为 (2,3 ) 的随机整数；
print("numpy.random.randint(5,size=(2,3)) = \n{}".format(np.random.randint(5, size = (2,3))))




# 产生[-2, 8)之间的形状为 (3,6)的整数
print("np.random.randint(low=2,high=8,size=(3,6))   = \n {}".format(np.random.randint(low = -2, high = 8, size=(3,6))  ))
#从以low为下限，high为上限的随机均匀分布数中随机取样，生成size维度的均匀分布数组，整型。
#包括下限，不包括上限，不指定low和high时，如果high = None，则取[0,low)区间。

#===========================================================================================================================
#                                             "[0,1)上 标准正态分布的float型数"
#===========================================================================================================================


# randn: 原型：numpy.random.randn（d0,d1,...,dn),产生d0 - d1 - ... - dn形状的标准正态分布的float型数。
print("numpy.random.randn(3,4) = \n{}".format(np.random.randn(3,4)))


#返回指定形状的标准正态分布数组
print("np.random.standard_normal(size = (5,6)) =\n {}".format(np.random.standard_normal(size = (5,6))))



#===========================================================================================================================
#                                             " 指定的高斯分布"
#===========================================================================================================================


print("numpy.random.normal(loc=0.0, scale=1.0, 3,4) = \n{}".format(np.random.normal(loc=0.0, scale=1.0,  size = (3,4))))
# np.random.normal(loc=0.0, scale=1.0, size=None) 的作用是生成高斯分布的概率密度随机数:
# loc(float)：此概率分布的均值（对应着整个分布的中心centre
# scale(float)：此概率分布的标准差（对应于分布的宽度，scale越大，图形越矮胖；scale越小，图形越瘦高）
# size(int or tuple of ints)：输出的shape，默认为None，只输出一个值
# np.random.normal(loc=0, scale=1, size)就表示标准正太分布（μ=0, σ=1）。


#生成随机正态分布数。
# loc：float类型，表示此正态分布的均值（对应整个分布中心）
# scale：float类型，表示此正态分布的标准差（对应于分布的密度，scale越大越矮胖，数据越分散；scale越小越瘦高，数据越集中）
# size：输出的shape，size=(k,m,n) 表示输出k维，m行，n列的数，默认为None，只输出一个值，size=100，表示输出100个值
print("np.random.normal(loc =0.0 , scale= 1.0,size = (5,4)) = \n {}".format(np.random.normal(loc = 0.0 , scale= 1.0, size = (5,4))))


import numpy as np
import matplotlib.pyplot as plt#导入模块

plt.hist(np.random.normal(loc = 0.0, scale = 1.0, size = 100000), bins = 30)#bins直方图的柱数
plt.show()



plt.hist(np.random.normal(loc = 0.0, scale = 1.0, size = (100000,1)), bins = 30)#bins直方图的柱数
plt.show()


#===========================================================================================================================
#                                              np.random.choice
#===========================================================================================================================

# 以数组形式
import numpy as np
import pandas as pd

"""
在Python中我们可以通过Numpy包的random模块中的choice()函数来生成服从待定的概率质量函数的随机数。
choice()函数：

choice(a, size=None, replace=True, p=None)
参数a: 随机变量可能的取值序列。
参数size: 我们要生成随机数数组的大小。
参数replace: 决定了生成随机数时是否是有放回的。
参数p：为一个与x等长的向量，指定了每种结果出现的可能性。
"""
from collections import  Counter

L = 1000000
RandomNumber = np.random.choice([1,2,3,4,5],size = L, replace=True,p=[0.1,0.1,0.3,0.3,0.2])
pd.Series(RandomNumber).value_counts() # 计算频数分布value_counts()函数
pd.Series(RandomNumber).value_counts()/L  #计算概率分布

data_count2 = Counter(RandomNumber)
# 返回的是一个dict字典，可以用data_count2.keys()和data_count2.values()取相应的key值和value值





#===========================================================================================================================
#                                              二项分布
#===========================================================================================================================
"""
二项分布是由伯努利提出的概念，指的是重复n次（注意：这里的n和binomial()函数参数n不是一个意思）独立的伯努利试验，如果事件X服从二项式分布，则可以表示为X~B(n,p)，则期望E(X)=np，方差D(X)=np(1-p)。
简单来讲就是在每次试验中只有两种可能的结果（例如：抛一枚硬币，不是正面就是反面，而掷六面体色子就不是二项式分布），而且两种结果发生与否互相对立，并且相互独立，与其它各次试验结果无关，事件发生与否的概率在每一次独立试验中都保持不变。

n为实验总次数,k是成功的次数,p是成功概率

    P(X=k)=C_n^kp^k(1-p)^{n-k}

numpy.random.RandomState.binomial(n, p, size=None)
表示对一个二项分布进行采样（size表示采样的次数，draw samples from a binomial distribution.），参数中的n,p分别对应于公式中的n,p，函数的返回值表示n中成功（success）的次数。

"""


n, p = 2, .5
sum(np.random.binomial(n, p, size=20000)==2)/20000.
# 0.24605        # 和我们的精确概率值相接近

# 其中一个为反面
sum(np.random.binomial(n, p, size=20000)==1)/20000.
# 0.5075

# 两个都是反面
n, p = 2, .5
sum(np.random.binomial(n, p, size=20000)==0)/20000.
# 0.257


n, p = 2, .5
sum(np.random.binomial(n, p, size=20000)==10)/20000.
# 0.0


# 一次抛5枚硬币，每枚硬币正面朝上概率为0.5，做10次试验，求每次试验发生正面朝上的硬币个数：
test = np.random.binomial(5, 0.5, 10)
print(test)


#================================================

def plot_binomial(n,p):
    '''绘制二项分布的概率质量函数'''
    sample = np.random.binomial(n, p, size=10000)  # 产生10000个符合二项分布的随机数
    bins = np.arange(n+2)
    plt.hist(sample, bins=bins, align='left', density=True, rwidth=0.1)  # 绘制直方图
    #设置标题和坐标
    plt.title('Binomial PMF with n={}, p={}'.format(n,p))
    plt.xlabel('number of successes')
    plt.ylabel('probability')

plot_binomial(10, 0.5)



fig = plt.figure(figsize=(12,4.5)) #设置画布大小
p1 = fig.add_subplot(121)  # 添加第一个子图
plot_binomial(10, 0.2)
p2 = fig.add_subplot(122)  # 添加第二个子图
plot_binomial(10, 0.8)


#===========================================================================================================================
#                                              泊松分布
#===========================================================================================================================

# 泊松分布用于描述单位时间内随机事件发生次数的概率分布，它也是离散分布，其概率质量函数为：

# 比如你在等公交车，假设这些公交车的到来是独立且随机的（当然这不是现实），前后车之间没有关系，那么在1小时中到来的公交车数量就符合泊松分布。同样使用统计模拟的方法绘制该泊松分布，这里假设每小时平均来6辆车（即上述公式中lambda=6）。

lamb = 6
sample = np.random.poisson(lamb, size=10000)  # 生成10000个符合泊松分布的随机数
bins = np.arange(20)
plt.hist(sample, bins=bins, align='left', rwidth=0.1, density=True) # 绘制直方图
# 设置标题和坐标轴
plt.title('Poisson PMF (lambda=6)')
plt.xlabel('number of arrivals')
plt.ylabel('probability')
plt.show()



"""
 正态分布和泊松分布的区别
正态分布是连续的，而泊松是离散的。

但我们可以看到，对于一个足够大的泊松分布，类似于二项分布，它会变得类似于具有一定std开发和均值的正态分布。



"""
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.normal(loc=50, scale=7, size=1000), hist=False, label='normal')
sns.distplot(random.poisson(lam=50, size=1000), hist=False, label='poisson')

plt.show()



"""
泊松分布和二项分布的区别
差异非常细微，因为二项式分布用于离散试验，而泊松分布用于连续试验。

但是对于非常大的n和接近零的p，二项式分布几乎与泊松分布相同，因此n * p几乎等于lam。


"""
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.binomial(n=1000, p=0.01, size=1000), hist=False, label='binomial')
sns.distplot(random.poisson(lam=10, size=1000), hist=False, label='poisson')

plt.show()

#===========================================================================================================================
#                                             numpy 模块   gamma分布
#===========================================================================================================================
shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)
s = np.random.gamma(shape, scale, 1000)



#===========================================================================================================================
#                                             numpy 模块   几何分布
#===========================================================================================================================


z = np.random.geometric(p=0.35, size=10000)


#===========================================================================================================================
#                                              逻辑斯谛分布
#===========================================================================================================================

# 在逻辑回归，神经网络等机器学习中广泛使用。

# 它具有三个参数：
# loc-平均值，即峰值所在的位置。 默认值0。
# scale-标准偏差，分布的平坦度。 默认值1。
# size-返回数组的形状。
# 例如：
# 从均值1和stddev 2.0的逻辑分布中抽取2x3个样本：

from numpy import random

x = random.logistic(loc=1, scale=2, size=(2, 3))

print(x)


# 逻辑斯谛分布可视化
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.logistic(size=1000), hist=False)

plt.show()



"""
3、逻辑斯谛分布和正态分布的区别
两种分布几乎相同，但逻辑分布的尾部区域更大。 即。 它表示发生事件的可能性远非均值。

对于较高的尺度值(标准差)，正态分布和逻辑斯谛分布除了峰值外几乎相同。
"""
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.normal(scale=2, size=1000), hist=False, label='normal')
sns.distplot(random.logistic(size=1000), hist=False, label='logistic')

plt.show()


#===========================================================================================================================
#                                              帕累托分布
#===========================================================================================================================


# 它有两个参数：
# a-形状参数。
# size-返回数组的形状。
# 例如：
# 绘制一个样本以进行大小为2x3的形状为2的pareto分发：

from numpy import random

x = random.pareto(a=2, size=(2, 3))

print(x)


# 2、帕累托分布的可视化
# 例如：

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.pareto(a=2, size=1000), kde=False)

plt.show()


#===========================================================================================================================
#                                              卡方分布
#===========================================================================================================================



# 1、卡方分布
# 使用卡方分布作为验证假设的基础。
# 它有两个参数：
# df- (degree of freedom).
# size-返回数组的形状。
# 例如：
# 画出一个样本，用于卡方分布，自由度为2，大小为2x3：

import numpy as np

x = np.random.chisquare(df=2, size=(2, 3))

print(x)


# 2、卡方分布的可视化
# 例如：

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(np.random.chisquare(df=1, size=1000), hist=False)

plt.show()

#===========================================================================================================================
#                                              #指数分布
#===========================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tau = 10
lam = 1 / tau
sample = np.random.exponential(tau, size=10000)  # 产生10000个满足指数分布的随机数
plt.hist(sample, bins=80, alpha=0.7, density=True) #绘制直方图
plt.margins(0.02)

# 根据公式绘制指数分布的概率密度函数lam = 1 / tau
x = np.arange(0,80,0.1)
y = lam * np.exp(- lam * x)
plt.plot(x,y,color='orange', lw=3)

#设置标题和坐标轴
plt.title('Exponential distribution, 1/lambda=10')
plt.xlabel('time')
plt.ylabel('PDF')
plt.show()


#===========================================================================================================================
#                                              # 正态分布
#===========================================================================================================================


def norm_pdf(x,mu,sigma):
    '''正态分布概率密度函数'''
    pdf = np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

mu = 0    # 均值为0
sigma = 1 # 标准差为1

# 用统计模拟绘制正态分布的直方图
sample = np.random.normal(mu, sigma, size=10000)
plt.hist(sample, bins=100, alpha=0.7, density=True)

# 根据正态分布的公式绘制PDF曲线
x = np.arange(-5, 5, 0.01)
y = norm_pdf(x, mu, sigma)
plt.plot(x,y, color='orange', lw=3)
plt.show()



#===========================================================================================================================
#                                              # 瑞利分布
#===========================================================================================================================
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.RandomState.rayleigh.html#numpy.random.RandomState.rayleigh


#绘制一个样本，用于瑞利分布，比例为2，大小为2x3：
from numpy import random
x = random.rayleigh(scale=2, size=(2, 3))
print(x)

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.rayleigh(scale=1, size=10000), hist=False)
plt.show()



print("当一个随机二维向量的两个分量呈独立的、有着相同的方差、均值为0的正态分布时，这个向量的模呈瑞利分布。例如，当随机复数的实部和虚部独立同分布于0均值，同方差的正态分布时，该复数的绝对值服从瑞利分布。\
    \n瑞利分布的概率函数为：")
from IPython.display import Latex
Latex(r'$ f(x;\sigma)=\frac{x}{\sigma ^2} e^{-\frac{x^2}{2\sigma^2}}  $')
Latex(r'$ f(x;\sigma)=\frac{x}{scale ^2} e^{-\frac{x^2}{2 * scale^2}}  $')



from matplotlib.pyplot import hist

a = np.random.rayleigh(3, size = 100000)

a = np.random.rayleigh(3, size = (4, 5))


scale = 2
s = np.random.rayleigh(scale, 1000000)

#===========================================================================================================================
#                                              numpy 模块计算均值、方差、标准差
#===========================================================================================================================


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
print("np.std(a, axis=0) = {}".format(np.std(a, axis=0, )))

#求每行的总体标准差
print("np.std(a, axis=1) = {}".format(np.std(a, axis=1, )))

# 计算矩阵所有元素的总体标准差
print("np.std(a) = {}".format(np.std(a, )))



#求每列的样本标准差
print("np.std(a, axis=0) = {}".format(np.std(a, axis=0, ddof=1)))

#求每行的样本标准差
print("np.std(a, axis=1) = {}".format(np.std(a, axis=1, ddof=1)))

# 计算矩阵所有元素的样本标准差
print("np.std(a) = {}".format(np.std(a, ddof=1)))

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


#验证对正太分布乘以一个常数加上一个常数对均值和方差的影响。
n=2
b = a * n + 5

print(" b 所有元素的均值 = {}\n".format(b.mean()))  # b 所有元素的均值 = 9.001396174270969
print(" b 所有元素的均值 = {}\n".format(np.mean(b)))  # b 所有元素的均值 = 9.001396174270969

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
np.linspace(start,stop,num=50, endpoint=True, retstep=False, dtype=None, axis=0,)

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
print("  numpy.random.multivariate_normal()函数解析  ")
print("-"*70)
import numpy as np
mean = (1, 2, 3)
cov = [[1, 0, 0], [0, 1, 0],[0, 0, 1]]
x = np.random.multivariate_normal(mean, cov, (100, 100))
print(x.shape)





#==============================================================

"""
seed值设为某一定值，则np.random下随机数生成函数生成的随机数永远是不变的。更清晰的说，即当你把设置为seed(0)，则你每次运行代码第一次用np.random.rand()产生的随机数永远是0.5488135039273248；第二次用np.random.rand()产生的随机数永远是0.7151893663724195
"""
print("numpy.random.rand(2,3) = \n{}".format(np.random.rand(2,3)))


np.random.seed(10)
print("numpy.random.rand(2,3) = \n{}".format(np.random.rand(2,3)))


np.random.seed(10)
for i in range(6):
    print(np.random.rand())


np.random.seed(10)
for i in range(3):
    print(np.random.rand())
#==============================================================





